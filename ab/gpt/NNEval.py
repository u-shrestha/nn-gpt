import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ab.nn.util.Util import release_memory, uuid4
from ab.gpt.util.Util import read_py_file_as_string
from ab.gpt.util.Const import epoch_dir, new_nn_file, nngpt_dir, synth_dir, hp_file, NN_TRAIN_EPOCHS
from ab.gpt.util.Util import verify_nn_code, copy_to_lemur
from ab.gpt.util.CycleResults import generate_cycle_results, collect_cycle_metrics, save_cycle_results
from ab.gpt.util import nneval_worker_pool as NNEvalWorkerPool


TASK = "img-classification"
DATASET = "cifar-10"
METRIC = "acc"

LR = 0.01
BATCH = 64
DROPOUT = 0.2
MOMENTUM = 0.9
TRANSFORM = "norm_256_flip"

SAVE_TO_DB = True
NN_NAME_PREFIX = None
NN_ALTER_EPOCHS = None
ONLY_EPOCH = None
EPOCH_LIMIT_MINUTES = None
CUSTOM_SYNTH_DIR = None
CYCLE = None


def _resolve_use_all_visible_gpus(use_all_visible_gpus: Optional[bool]) -> bool:
    if use_all_visible_gpus is not None:
        return bool(use_all_visible_gpus)
    raw = os.getenv("NNGPT_NNEVAL_USE_ALL_VISIBLE_GPUS")
    if raw is None or raw == "":
        return True
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _extract_accuracy_from_eval_payload(payload: Dict[str, Any]) -> Optional[float]:
    eval_results = payload.get("eval_results")
    if not isinstance(eval_results, dict):
        return None
    accuracy = eval_results.get("accuracy", eval_results.get("acc"))
    if accuracy is None:
        epochs_data = eval_results.get("epochs", [])
        if epochs_data:
            first_epoch = epochs_data[0] or {}
            if isinstance(first_epoch, dict):
                accuracy = first_epoch.get("accuracy", first_epoch.get("acc"))
    if accuracy is None:
        return None
    try:
        return float(accuracy)
    except (TypeError, ValueError):
        return None


def _load_existing_success_result(model_dir_path: Path) -> Optional[Dict[str, Any]]:
    eval_info_path = model_dir_path / "eval_info.json"
    if not eval_info_path.exists():
        return None
    try:
        payload = json.loads(eval_info_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    accuracy = _extract_accuracy_from_eval_payload(payload)
    if accuracy is None:
        return None
    return {
        "model_id": model_dir_path.name,
        "success": True,
        "accuracy": float(accuracy),
        "skipped": True,
        "code_file": str(model_dir_path / new_nn_file),
    }


def _write_success_outputs(spec: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    model_dir_path = Path(spec["model_dir"])
    accuracy = float(result["accuracy"])
    eval_info_data = {
        "eval_args": result.get("eval_args", {}),
        "eval_results": {
            "checksum": result.get("checksum"),
            "accuracy": accuracy,
            "full_result": result.get("full_result", ""),
        },
        "cli_args": {
            "task": spec["task"],
            "dataset": spec["dataset"],
            "metric": spec["metric"],
            "lr": spec["prm"].get("lr"),
            "batch": spec["prm"].get("batch"),
            "dropout": spec["prm"].get("dropout"),
            "momentum": spec["prm"].get("momentum"),
            "transform": spec["prm"].get("transform"),
        },
    }
    (model_dir_path / "1.json").write_text(
        json.dumps([{"epoch": int(spec["prm"].get("epoch", 1)), "accuracy": accuracy}], indent=2),
        encoding="utf-8",
    )
    (model_dir_path / "eval_info.json").write_text(
        json.dumps(eval_info_data, indent=4, default=str),
        encoding="utf-8",
    )
    error_path = model_dir_path / "error.txt"
    if error_path.exists():
        error_path.unlink()
    verification_failure_path = model_dir_path / "eval_verification_failed.txt"
    if verification_failure_path.exists():
        verification_failure_path.unlink()

    nn_name = uuid4(read_py_file_as_string(spec["code_file"]))
    lemur_prefix = spec.get("lemur_prefix")
    if lemur_prefix:
        nn_name = str(lemur_prefix) + "-" + nn_name
    copy_to_lemur(
        model_dir_path,
        nn_name,
        spec["task"],
        spec["dataset"],
        spec["metric"],
    )
    return {
        "model_id": spec["model_id"],
        "success": True,
        "accuracy": accuracy,
        "skipped": False,
        "code_file": spec["code_file"],
    }


def _write_failure_outputs(spec: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    model_dir_path = Path(spec["model_dir"])
    error_text = str(result.get("error", "Unknown evaluation error"))
    traceback_text = str(result.get("traceback", ""))
    (model_dir_path / "error.txt").write_text(
        f"{error_text}\n\n{traceback_text}".strip() + "\n",
        encoding="utf-8",
    )
    return {
        "model_id": spec["model_id"],
        "success": False,
        "error": error_text,
        "is_oom": bool(result.get("is_oom", False)),
        "skipped": False,
    }


def _build_eval_request(
    *,
    model_id: str,
    model_dir_path: Path,
    code_file_path: Path,
    task: str,
    dataset: str,
    metric: str,
    prm: Dict[str, Any],
    save_to_db: bool,
    prefix_for_db: Optional[str],
    epoch_limit_minutes: Optional[int],
    lemur_prefix: Optional[str],
) -> Dict[str, Any]:
    return {
        "model_id": str(model_id),
        "model_dir": str(model_dir_path),
        "code_file": str(code_file_path),
        "task": str(task),
        "dataset": str(dataset),
        "metric": str(metric),
        "prm": dict(prm),
        "save_to_db": bool(save_to_db),
        "prefix": prefix_for_db,
        "save_path": str(model_dir_path),
        "epoch_limit_minutes": epoch_limit_minutes,
        "lemur_prefix": lemur_prefix,
        "use_ast_validation": None,
    }


def _collect_epoch_requests(
    *,
    models_base_dir: Path,
    nn_name_prefix: Optional[str],
    nn_train_epochs: int,
    save_to_db: bool,
    task: str,
    dataset: str,
    metric: str,
    lr: float,
    batch: int,
    dropout: float,
    momentum: float,
    transform: str,
    epoch_limit_minutes: Optional[int],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    requests: List[Dict[str, Any]] = []
    immediate_results: List[Dict[str, Any]] = []
    base_nngpt_path = nngpt_dir

    for model_id in sorted(os.listdir(models_base_dir)):
        model_dir_path = models_base_dir / model_id
        if not model_dir_path.is_dir():
            continue

        code_file_path = model_dir_path / new_nn_file
        df_file_path = model_dir_path / "dataframe.df"

        if not code_file_path.exists():
            print(f"Code file {new_nn_file} not found in {model_dir_path}. Skipping.")
            continue

        existing_result = _load_existing_success_result(model_dir_path)
        if existing_result is not None:
            print(
                f"  [SKIP] {model_dir_path.relative_to(base_nngpt_path)} already evaluated "
                f"({existing_result['accuracy'] * 100:.2f}%)"
            )
            immediate_results.append(existing_result)
            continue

        print(f"\n--- Evaluating Model: {model_dir_path.relative_to(base_nngpt_path)} ---")
        if not verify_nn_code(model_dir_path, code_file_path):
            print(f"Code verification failed for {code_file_path}. Skipping evaluation.")
            (model_dir_path / "eval_verification_failed.txt").write_text(
                "Initial code verification failed.\n",
                encoding="utf-8",
            )
            immediate_results.append(
                {
                    "model_id": model_id,
                    "success": False,
                    "error": "Initial code verification failed.",
                    "is_oom": False,
                    "skipped": False,
                }
            )
            continue

        resolved_task = task
        resolved_dataset = dataset
        resolved_metric = metric
        prm = None
        hp_path = model_dir_path / hp_file
        if hp_path.exists():
            try:
                prm = json.loads(hp_path.read_text(encoding="utf-8"))
                print(f"Training model {model_id} with LLM recommended prm {prm}")
            except Exception as exc:
                print(f"Error loading LLM recommended training params from {hp_path}: {exc}.")
        if not prm:
            prm = {
                "lr": lr,
                "batch": batch,
                "dropout": dropout,
                "momentum": momentum,
                "transform": transform,
            }
            print(f"Training model {model_id} with command-line/default training params {prm}")

        prefix_for_db = nn_name_prefix
        orig_pref = None
        if df_file_path.exists():
            try:
                origdf = pd.read_pickle(df_file_path)
                resolved_task = origdf.get("task", resolved_task)
                resolved_dataset = origdf.get("dataset", resolved_dataset)
                resolved_metric = origdf.get("metric", resolved_metric)
                orig_pref = origdf["nn"].split("-")[0]
                original_prm_from_df = origdf.get("prm")
                if isinstance(original_prm_from_df, dict):
                    prm.update(original_prm_from_df)
                prefix_for_db = nn_name_prefix or (
                    origdf.get("nn", "unknown").split("-")[0]
                    if "nn" in origdf
                    else prefix_for_db
                )
                print(
                    f"  Loaded metadata from dataframe.df: "
                    f"task={resolved_task}, dataset={resolved_dataset}, metric={resolved_metric}"
                )
            except Exception as exc:
                print(
                    f"  Error loading dataframe.df from {df_file_path}: {exc}. "
                    "Using command-line/default parameters."
                )
        else:
            print("  No dataframe.df found. Using command-line/default evaluation parameters.")

        prm["epoch"] = int(nn_train_epochs)
        if prm.get("transform") is None or not isinstance(prm.get("transform"), str):
            prm["transform"] = transform if transform else TRANSFORM

        print(f"  Final parameters for Eval: {prm}")
        print(
            f"  Task: {resolved_task}, Dataset: {resolved_dataset}, "
            f"Metric: {resolved_metric}, Prefix: {prefix_for_db}"
        )

        requests.append(
            _build_eval_request(
                model_id=model_id,
                model_dir_path=model_dir_path,
                code_file_path=code_file_path,
                task=resolved_task,
                dataset=resolved_dataset,
                metric=resolved_metric,
                prm=prm,
                save_to_db=save_to_db,
                prefix_for_db=prefix_for_db,
                epoch_limit_minutes=epoch_limit_minutes,
                lemur_prefix=nn_name_prefix or orig_pref,
            )
        )

    return requests, immediate_results


def main(
    nn_name_prefix=NN_NAME_PREFIX,
    nn_train_epochs=NN_TRAIN_EPOCHS,
    only_epoch=ONLY_EPOCH,
    save_to_db=SAVE_TO_DB,
    nn_alter_epochs=NN_ALTER_EPOCHS,
    task=TASK,
    dataset=DATASET,
    metric=METRIC,
    lr=LR,
    batch=BATCH,
    dropout=DROPOUT,
    momentum=MOMENTUM,
    transform=TRANSFORM,
    epoch_limit_minutes=EPOCH_LIMIT_MINUTES,
    custom_synth_dir=CUSTOM_SYNTH_DIR,
    cycle=CYCLE,
    use_all_visible_gpus: Optional[bool] = None,
):
    base_nngpt_path = nngpt_dir
    if nn_alter_epochs is None:
        if epoch_dir().is_dir():
            nn_alter_epochs = len(os.listdir(epoch_dir()))
        else:
            print(f"Directory {epoch_dir()} doesn't exist", file=sys.stderr)
            nn_alter_epochs = 0

    run_summary = {"epochs": []}
    resolved_use_all_visible_gpus = _resolve_use_all_visible_gpus(use_all_visible_gpus)

    try:
        if nn_alter_epochs:
            epoch_indices = [only_epoch] if only_epoch is not None else list(range(nn_alter_epochs))
            for i in epoch_indices:
                current_cycle = cycle if cycle is not None else i
                current_epoch = i
                cycle_start_time = time.time()
                current_alter_epoch_path = epoch_dir(i)
                models_base_dir = Path(custom_synth_dir) if custom_synth_dir else synth_dir(current_alter_epoch_path)

                if not models_base_dir.exists():
                    print(f"Directory {models_base_dir} for NNAlter epoch {i} not found. Skipping.")
                    continue

                print(f"\n--- Scanning NNAlter Epoch Directory: {current_alter_epoch_path} ---")
                print(f"--- Synthesized Models Directory: {models_base_dir} ---")

                requests, epoch_results = _collect_epoch_requests(
                    models_base_dir=models_base_dir,
                    nn_name_prefix=nn_name_prefix,
                    nn_train_epochs=nn_train_epochs,
                    save_to_db=save_to_db,
                    task=task,
                    dataset=dataset,
                    metric=metric,
                    lr=lr,
                    batch=batch,
                    dropout=dropout,
                    momentum=momentum,
                    transform=transform,
                    epoch_limit_minutes=epoch_limit_minutes,
                )

                if requests:
                    NNEvalWorkerPool.prewarm_nneval_workers(
                        use_all_visible_gpus=resolved_use_all_visible_gpus,
                        timeout_seconds=60.0,
                    )
                    entries = [{"payload": request} for request in requests]
                    worker_results = NNEvalWorkerPool.evaluate_model_entries(
                        entries,
                        use_all_visible_gpus=resolved_use_all_visible_gpus,
                    )
                    for request, worker_result in zip(requests, worker_results):
                        model_id = request["model_id"]
                        if worker_result.get("success"):
                            print(f"  Evaluation results for {model_id}: {worker_result}")
                            epoch_results.append(_write_success_outputs(request, worker_result))
                        else:
                            print(f"  Error evaluating model {model_id}: {worker_result.get('error')}")
                            epoch_results.append(_write_failure_outputs(request, worker_result))
                        release_memory()

                cycle_end_time = time.time()
                cycle_time_minutes = (cycle_end_time - cycle_start_time) / 60.0
                eval_results_list, model_dirs_list, successful_models, failed_models = collect_cycle_metrics(
                    models_base_dir,
                    current_alter_epoch_path,
                )
                cycle_results = generate_cycle_results(
                    cycle=current_cycle,
                    models_base_dir=models_base_dir,
                    eval_results_list=eval_results_list,
                    model_dirs_list=model_dirs_list,
                    successful_models=successful_models,
                    failed_models=failed_models,
                    cycle_time_minutes=cycle_time_minutes,
                    current_alter_epoch_path=current_alter_epoch_path,
                )
                cycle_results_path = base_nngpt_path / "cycle_results.json"
                if cycle_results_path.exists():
                    backup_path = base_nngpt_path / f"cycle_results_{i-1}.json"
                    shutil.copy2(cycle_results_path, backup_path)
                    print(f"Backup saved -> {backup_path}")
                save_cycle_results(cycle_results, cycle_results_path)
                print(
                    f"\n--- Cycle {current_cycle} (Epoch {current_epoch}) results saved to: "
                    f"{cycle_results_path} ---"
                )
                print(f"  Found {len(eval_results_list)} successful evaluations from eval_info.json files")
                print(f"  Found {len(failed_models)} failed models")
                run_summary["epochs"].append(
                    {
                        "epoch": current_epoch,
                        "cycle": current_cycle,
                        "models_base_dir": str(models_base_dir),
                        "cycle_results_path": str(cycle_results_path),
                        "model_results": sorted(epoch_results, key=lambda item: str(item.get("model_id", ""))),
                        "successful_models": len(successful_models),
                        "failed_models": len(failed_models),
                    }
                )
    finally:
        NNEvalWorkerPool.shutdown_nneval_workers()

    return run_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Neural Networks generated by NNAlter.py.")
    parser.add_argument(
        "-ae",
        "--nn_alter_epochs",
        type=int,
        default=NN_ALTER_EPOCHS,
        help="Number of epochs NNAlter.py was run for.",
    )
    parser.add_argument(
        "-oe",
        "--only_epoch",
        type=int,
        default=ONLY_EPOCH,
        help="Run NNAlter.py for the specified epoch only.",
    )
    parser.add_argument(
        "-te",
        "--nn_train_epochs",
        type=int,
        default=NN_TRAIN_EPOCHS,
        help=f"Number of epochs to train each altered NN during evaluation (default: {NN_TRAIN_EPOCHS}).",
    )
    parser.add_argument("--task", type=str, default=TASK, help=f"Default task for NNEval (default: {TASK}).")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET,
        help=f"Default dataset for NNEval (default: {DATASET}).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=METRIC,
        help=f"Default metric for NNEval (default: {METRIC}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help=f"Learning rate for NNEval if not in dataframe.df's prm (default: {LR}).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH,
        help=f"Batch size for NNEval if not in dataframe.df's prm (default: {BATCH}).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DROPOUT,
        help=f"Dropout rate for NNEval if not in dataframe.df's prm (default: {DROPOUT}).",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=MOMENTUM,
        help=f"Momentum for NNEval if not in dataframe.df's prm (default: {MOMENTUM}).",
    )
    parser.add_argument(
        "--transform",
        type=str,
        default=TRANSFORM,
        help=f"Default transform for NNEval if not in dataframe.df's prm (default: {TRANSFORM}).",
    )
    parser.add_argument(
        "--save_to_db",
        action=argparse.BooleanOptionalAction,
        default=SAVE_TO_DB,
        help="Whether to save evaluation results to the database.",
    )
    parser.add_argument(
        "--nn_name_prefix",
        type=str,
        default=NN_NAME_PREFIX,
        help=f"Default neural network name prefix (default: {NN_NAME_PREFIX}).",
    )
    parser.add_argument(
        "--custom_synth_dir",
        dest="custom_synth_dir",
        type=str,
        default=CUSTOM_SYNTH_DIR,
        help="Custom directory containing generated models",
    )
    parser.add_argument(
        "--epoch_limit_minutes",
        type=int,
        default=EPOCH_LIMIT_MINUTES,
        help="Max minutes allowed per epoch (default: specified in NN Dataset).",
    )
    parser.add_argument(
        "--cycle",
        type=int,
        default=CYCLE,
        help="Cycle number (finetuning iteration, separate from epoch).",
    )

    args = parser.parse_args()
    print("Starting evaluation of altered NNs...")
    print(f"NNAlter run epochs to scan: {args.nn_alter_epochs}")
    print(f"Each altered NN will be trained for: {args.nn_train_epochs} epochs for evaluation.")
    print(f"Base task: {args.task}, Base dataset: {args.dataset}, Base metric: {args.metric}")
    print("Base Hyperparameters for NNEval (before df override):")
    print(
        f"  LR: {args.lr}, Batch Size: {args.batch_size}, Dropout: {args.dropout}, "
        f"Momentum: {args.momentum}, Transform: {args.transform}"
    )
    print(f"Save to DB: {args.save_to_db}")
    print(f"Prefix for the names of generated neural network: {args.nn_name_prefix}")

    main(
        nn_name_prefix=args.nn_name_prefix,
        nn_train_epochs=args.nn_train_epochs,
        only_epoch=args.only_epoch,
        save_to_db=args.save_to_db,
        nn_alter_epochs=args.nn_alter_epochs,
        task=args.task,
        dataset=args.dataset,
        metric=args.metric,
        lr=args.lr,
        batch=args.batch_size,
        dropout=args.dropout,
        momentum=args.momentum,
        transform=args.transform,
        epoch_limit_minutes=args.epoch_limit_minutes,
        custom_synth_dir=args.custom_synth_dir,
        cycle=args.cycle,
    )
