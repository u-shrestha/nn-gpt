import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import ab.nn.api as api
from ab.gpt.util.Const import trans_dir, out_dir
from ab.gpt.util.Util import read_py_file_as_string

RESULT_DIR = trans_dir / 'augment/gen_result'
CONFIG_DIR = trans_dir / 'augment/config'
CONFIG_FILE = "gen_epoch_A0.json"
RESNET_FILE = trans_dir / 'augment/config/Resnet.py'

TASK = "img-classification"
DATASET = "imagenette"
METRIC = "acc"
MODEL_FILTER = "ResNet"
NUM_EPOCHS = 30      
SEED = 42

DEFAULTS = {
    'lr': 0.01,
    'batch': 64,
    'dropout': 0.2,
    'momentum': 0.9,
    'transform': "norm_256"
}

INFRA_RETRIES = 2
INFRA_MARKERS = ("out of memory", "cuda", "cudnn", "nccl",
                 "database", "disk", "connection", "no such file")


def set_seed(seed: int):
    """Best-effort reproducibility (Train.py seeds nothing internally)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def classify_error(exc: Exception) -> str:
    """'infra_error' must never become a 0.0 label; 'code_error' may."""
    if isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError, OSError)):
        return "infra_error"
    msg = str(exc).lower()
    return "infra_error" if any(m in msg for m in INFRA_MARKERS) else "code_error"


def get_best_model_from_db():
    df = api.data(only_best_accuracy=True, task=TASK, dataset=DATASET,
                  metric=METRIC, nn=MODEL_FILTER, max_rows=1)
    if df.empty:
        raise ValueError("No matching models found in database")
    best = df.iloc[0].to_dict()
    print(f"Found best model: {best['nn']} with accuracy: {best['accuracy']}")
    print("  [NOTE] DB nn_code may be the STALE, unpatched model — "
          "cosine runs should use --nn_file with the patched ResNet.")
    return best


def get_model_from_file(nn_file):
    """Load nn_code from a LOCAL .py file (the cosine-patched ResNet).
    read_py_file_as_string EXECUTES the module on read — trusted files only."""
    nn_code = read_py_file_as_string(str(nn_file))
    if not nn_code:
        raise ValueError(f"Could not load model code from {nn_file} "
                         f"(read_py_file_as_string returned None — check "
                         f"syntax/imports in that file)")
    name = Path(nn_file).stem
    print(f"Loaded model '{name}' from file: {nn_file} ({len(nn_code)} chars)")
    return {'nn': name, 'nn_code': nn_code}


def load_configs(path: Path):
    """Yield (config_num, config_id, schedule_list, source) for both JSON formats.
    'source' distinguishes how the config was produced: 'llm_gen' from sched_gen()
    (the fine-tuning loop); anything untagged defaults to 'config_gen'
    (ConfigGen.py / PermuteGen.py output)."""
    with open(path) as f:
        raw = json.load(f)
    for key, val in raw.items():
        if isinstance(val, dict) and "augment_configs" in val:   # new format
            yield int(key), val.get("config_id", key), val["augment_configs"], val.get("source", "config_gen")
        else:                                                    # old flat format
            yield int(key), key, val, "config_gen"


def load_training_summary(cand_tag: str):
    """Copy Train.py's shared training_summary.json into a per-candidate
    file before the next check_nn call overwrites it; return its contents.
    NOTE: race-prone under concurrent runs sharing one out_dir — keep one
    evaluation per job workspace."""
    shared = Path(out_dir) / 'training_summary.json'
    if not shared.exists():
        return None
    try:
        with open(shared) as f:
            summary = json.load(f)
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(shared, RESULT_DIR / f'{cand_tag}_training_summary.json')
        return summary
    except Exception as e:
        print(f"  [WARN] Could not read/copy training summary: {e}")
        return None


def extract_history(summary):
    """SINGLE definition (the previous file had two; the 5-value version
    shadowed the 4-value one and crashed the 4-value call site)."""
    history, best_epoch, best_acc, gpu_type, lr_schedule = [], None, None, None, None
    if summary:
        c = summary.get('learning_curves', {})
        eps = c.get('epochs', [])
        for i, ep in enumerate(eps):
            history.append({
                "epoch": ep,
                "train_accuracy": c.get('train_accuracy', [None] * len(eps))[i],
                "test_accuracy": c.get('test_accuracy', [None] * len(eps))[i],
                "train_loss": c.get('train_loss', [None] * len(eps))[i],
                "test_loss": c.get('test_loss', [None] * len(eps))[i],
                "lr": c.get('lr', [None] * len(eps))[i],
            })
        ts = summary.get('training_summary', {})
        best_epoch, best_acc = ts.get('best_epoch'), ts.get('best_accuracy')
        gpu_type = summary.get('system_info', {}).get('gpu_type')
        lrs = [h['lr'] for h in history if h['lr'] is not None]
        if len(lrs) >= 2:  # cosine-verification tripwire
            lr_schedule = "decaying" if lrs[-1] < lrs[0] * 0.99 else "constant"
    return history, best_epoch, best_acc, gpu_type, lr_schedule


def run_eval(start_from=0, seed=SEED, config_file=CONFIG_FILE,
             nn_file=RESNET_FILE, num_epochs=NUM_EPOCHS):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    best_model = get_model_from_file(nn_file) if nn_file else get_best_model_from_db()
    model_source = f"file:{nn_file}" if nn_file else "lemur_db"

    configs = list(load_configs(Path(CONFIG_DIR) / config_file))
    total = len(configs)
    print(f"Starting from config #{start_from} / {total} "
          f"(fidelity={num_epochs} ep, seed={seed})")

    for config_num, config_id, augment, source in configs:
        if config_num < start_from:
            continue

        cand_tag = f"e{num_epochs}_{DATASET}_{MODEL_FILTER}_cfg{config_id}_seed{seed}"
        json_path = RESULT_DIR / f"{cand_tag}.json"
        if json_path.exists():                      # resumable job arrays
            print(f"  Config {config_num}: result exists, skipping")
            continue

        set_seed(seed)
        prm = DEFAULTS.copy()
        prm['augment'] = augment
        prm['epoch'] = num_epochs   
        prm['seed'] = seed          

        result, err, validity = None, None, "ok"
        t0 = time.monotonic()
        for attempt in range(INFRA_RETRIES + 1):
            try:
                result = api.check_nn(
                    nn_code=best_model['nn_code'], task=TASK, dataset=DATASET,
                    metric=METRIC, prm=prm, save_to_db=False,
                    prefix="augment_data_collect", save_path=RESULT_DIR)
                break
            except Exception as e:
                err, validity = e, classify_error(e)
                if validity != "infra_error":
                    break
                print(f"  Config {config_num}: infra error "
                      f"(attempt {attempt + 1}): {e}")
                time.sleep(15 * (attempt + 1))
        gpu_seconds = time.monotonic() - t0

        base = {
            "dataset": DATASET, "architecture": best_model['nn'],
            "model_source": model_source,
            "config_num": config_num, "config_id": config_id, "source": source,
            "augment": augment, "seed": seed, "fidelity": num_epochs,
            "gpu_seconds": round(gpu_seconds, 1),
            "batch": prm['batch'], "lr": prm['lr'],
            "momentum": prm['momentum'], "transform": prm['transform'],
        }

        if result is None:
            if validity == "infra_error":
                # UNLABELED: an infra failure is not a candidate property.
                base.update({"validity_class": "infra_error", "error": str(err)})
                out = RESULT_DIR / f"INFRA_{cand_tag}.json"
            else:
                base.update({"accuracy": 0.0, "validity_class": "code_error",
                             "error": str(err)})
                out = json_path
            with open(out, 'w') as f:
                json.dump(base, f, indent=2)
            print(f"  Config {config_num}: {validity}: {err}")
            continue

        _, accuracy, time_metric, code_score = result
        history, best_epoch, best_acc, gpu_type, lr_schedule = \
            extract_history(load_training_summary(cand_tag))
        if not history:
            print(f"  [WARN] No training_summary.json; "
                  f"only final accuracy available for config {config_id}")
        if num_epochs > 1 and lr_schedule == "constant":
            print("  [WARN] lr curve is CONSTANT — cosine schedule not active; "
                  "the model that reached check_nn is not the patched ResNet")

        base.update({
            "accuracy": float(accuracy), "validity_class": "ok",
            "best_epoch": best_epoch, "best_accuracy": best_acc,
            "final_accuracy": history[-1]["test_accuracy"] if history else float(accuracy),
            "lr_schedule": lr_schedule,
            "epochs_completed": len(history), "time_metric": time_metric,
            "code_score": code_score, "gpu_type": gpu_type,
            "history": history,
        })
        with open(json_path, 'w') as f:
            json.dump(base, f, indent=2)
        print(f"  Config {config_num}/{total} | Acc={accuracy} "
              f"| lr={lr_schedule} | {gpu_seconds:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--start_from', type=int, default=0,
                        help="Config number to start from (default: 1)")
    parser.add_argument('--seed', type=int, default=SEED,
                        help="Training seed; pass different values from a "
                             "job array for the multi-seed label protocol")
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
                        help="training epochs = label fidelity; also sets "
                             "T_max of the cosine scheduler in the patched "
                             "model (1 = proxy, constant-lr by design)")
    parser.add_argument('--config_file', type=str, default=CONFIG_FILE)
    parser.add_argument('--nn_file', type=str, default=str(RESNET_FILE),
                        help="path to the PATCHED model .py; pass an empty "
                             "string to fall back to the LEMUR DB (stale code)")
    args = parser.parse_args()
    run_eval(start_from=args.start_from, seed=args.seed,
             config_file=args.config_file,
             nn_file=args.nn_file or None,
             num_epochs=args.num_epochs)