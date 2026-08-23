"""
TPE/Optuna baseline arm for augmentation-schedule search.

Searches the ScheduleValidate-defined space (METHODS, alpha ranges) using a
define-by-run encoding: for each of up to MAX_SEGMENTS slots, Optuna picks a
method, an alpha (if the method needs one), and a raw fraction weight. The
raw fractions need not sum to 1; validate_schedule() renormalizes them, which
is exactly what turns Optuna's independent per-slot floats into a legal
schedule without a dedicated simplex sampler.

Shares its evaluation harness with AugEval.py rather than reimplementing it:
same model loading, same infra-error retry/classification, same
training_summary extraction, same RESULT_DIR and file-naming convention
(`e{epochs}_{dataset}_{model}_cfg{config_id}_seed{seed}.json`). Results carry
an extra "search": "tpe" field, which is the only thing that distinguishes a
TPE-found config from one in the ConfigGen corpus once it lands in
gen_result/ — everything downstream (stat_utils.py) reads the two the same
way.

Caching: before training anything, check whether a result file for that
exact (config_id, fidelity, seed) already exists in RESULT_DIR. If so, reuse
its accuracy instead of retraining. This applies regardless of whether the
existing file came from the brute-force corpus, the noise-floor reruns, or
an earlier TPE trial, since they all share one naming convention.

Resumability: the Optuna study itself is persisted to a SQLite database
(augment/tpe_studies.db) keyed by --study. Passing the same --study name
again with load_if_exists=True continues the same study; --n_trials is the
number of ADDITIONAL trials to run this invocation, not a total to reach.

Run:
    python -m ab.gpt.brute.trans.augment.TpeSearch --n_trials 60 -e 15 --seed 42
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import optuna

from ab.gpt.util.Const import trans_dir
from ab.gpt.brute.trans.augment.ScheduleValidate import (
    validate_schedule, ScheduleError, METHODS,
)
from ab.gpt.brute.trans.augment.AugEval import (
    RESULT_DIR, RESNET_FILE, TASK, DATASET, METRIC, MODEL_FILTER,
    DEFAULTS, INFRA_RETRIES,
    set_seed, classify_error, get_model_from_file,
    load_training_summary, extract_history,
)
import ab.nn.api as api

STUDY_DB = trans_dir / "augment/tpe_studies.db"
MAX_SEGMENTS = 4
AUG_TYPES = list(METHODS.keys())  # single-sourced from ScheduleValidate


def sample_schedule(trial: "optuna.Trial", max_segments: int = MAX_SEGMENTS):
    """Define-by-run sample of a raw (pre-validation) schedule."""
    num_segments = trial.suggest_int("num_segments", 1, max_segments)
    raw = []
    for i in range(num_segments):
        method = trial.suggest_categorical(f"method_{i}", AUG_TYPES)
        if method == "none":
            alpha = None
        else:
            lo, hi, _ = METHODS[method]
            alpha = trial.suggest_float(f"alpha_{i}", lo, hi)
        # Raw weight, not yet a fraction — validate_schedule renormalizes
        # the whole segment list to sum to exactly 1.0.
        frac = trial.suggest_float(f"frac_{i}", 0.01, 1.0)
        raw.append([method, alpha, frac])
    return raw


def config_id_for(canonical) -> str:
    """Same hash convention as sched_gen() in Tune.py, so TPE-found configs
    dedup correctly against LLM-generated ones and the brute-force corpus."""
    return hashlib.md5(json.dumps(canonical).encode()).hexdigest()[:12]


def evaluate_config(canonical, config_id: str, seed: int, epochs: int, best_model: dict) -> float:
    cand_tag = f"e{epochs}_{DATASET}_{MODEL_FILTER}_cfg{config_id}_seed{seed}"
    json_path = RESULT_DIR / f"{cand_tag}.json"

    if json_path.exists():
        with open(json_path) as f:
            cached = json.load(f)
        acc = cached.get("accuracy")
        if acc is not None:
            print(f"  [CACHE] {config_id} @ e{epochs} seed{seed} -> acc={acc:.4f}")
            return float(acc)

    set_seed(seed)
    prm = DEFAULTS.copy()
    prm["epoch"] = epochs
    prm["seed"] = seed
    prm["augment"] = canonical

    result, err, validity = None, None, "ok"
    t0 = time.monotonic()
    for attempt in range(INFRA_RETRIES + 1):
        try:
            result = api.check_nn(
                nn_code=best_model["nn_code"], task=TASK, dataset=DATASET,
                metric=METRIC, prm=prm, save_to_db=False,
                prefix="tpe_search", save_path=RESULT_DIR)
            break
        except Exception as e:
            err, validity = e, classify_error(e)
            if validity != "infra_error":
                break
            print(f"  [INFRA] {config_id}: retrying (attempt {attempt + 1}): {e}")
            time.sleep(15 * (attempt + 1))
    gpu_seconds = time.monotonic() - t0

    base = {
        "dataset": DATASET, "architecture": best_model["nn"],
        "model_source": f"file:{RESNET_FILE}",
        "config_id": config_id, "augment": canonical, "seed": seed,
        "fidelity": epochs, "search": "tpe",
        "gpu_seconds": round(gpu_seconds, 1),
        "batch": prm["batch"], "lr": prm["lr"],
        "momentum": prm["momentum"], "transform": prm["transform"],
    }

    if result is None:
        if validity == "infra_error":
            # Never labeled 0.0 — an infra failure is not a candidate property.
            base.update({"validity_class": "infra_error", "error": str(err)})
            with open(RESULT_DIR / f"INFRA_{cand_tag}.json", "w") as f:
                json.dump(base, f, indent=2)
            raise optuna.TrialPruned(f"infra error: {err}")
        base.update({"accuracy": 0.0, "validity_class": "code_error", "error": str(err)})
        with open(json_path, "w") as f:
            json.dump(base, f, indent=2)
        print(f"  [{config_id}] code_error: {err}")
        return 0.0

    _, accuracy, time_metric, code_score = result
    # Unpack defensively: extract_history's arity has drifted between
    # checkouts before (5 values on the cluster's current AugEval.py,
    # the 5th being lr_schedule — a cosine-schedule sanity tripwire).
    # Take what's offered in the documented order; don't assume exactly 4.
    extracted = extract_history(load_training_summary(cand_tag))
    history = extracted[0] if len(extracted) > 0 else []
    best_epoch = extracted[1] if len(extracted) > 1 else None
    best_acc = extracted[2] if len(extracted) > 2 else None
    gpu_type = extracted[3] if len(extracted) > 3 else None
    lr_schedule = extracted[4] if len(extracted) > 4 else None
    if lr_schedule == "constant" and epochs > 1:
        print(f"  [WARN] {config_id}: lr curve is CONSTANT — cosine schedule "
              f"not active; check that the patched ResNet was actually loaded")

    base.update({
        "accuracy": float(accuracy), "validity_class": "ok",
        "best_epoch": best_epoch, "best_accuracy": best_acc,
        "final_accuracy": history[-1]["test_accuracy"] if history else float(accuracy),
        "lr_schedule": lr_schedule,
        "epochs_completed": len(history), "time_metric": time_metric,
        "code_score": code_score, "gpu_type": gpu_type, "history": history,
    })
    with open(json_path, "w") as f:
        json.dump(base, f, indent=2)
    print(f"  [{config_id}] acc={accuracy:.4f}  lr={lr_schedule}  ({gpu_seconds:.0f}s)")
    return float(accuracy)


def objective(trial: "optuna.Trial", best_model: dict, epochs: int, seed: int) -> float:
    raw = sample_schedule(trial)
    try:
        canonical, report = validate_schedule(raw, policy="renormalize")
    except ScheduleError as e:
        raise optuna.TrialPruned(str(e))
    if canonical is None:
        raise optuna.TrialPruned(report.get("reason", "invalid schedule"))

    config_id = config_id_for(canonical)
    trial.set_user_attr("config_id", config_id)
    trial.set_user_attr("canonical", canonical)
    return evaluate_config(canonical, config_id, seed, epochs, best_model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=100,
                         help="additional trials to run this invocation")
    parser.add_argument("-e", "--epochs", type=int, default=30,
                         help="training epochs per trial (fidelity); match "
                              "AUG_EVAL_EPOCHS in Tune.py for a fair "
                              "comparison against the LLM generation loop")
    parser.add_argument("--seed", type=int, default=42,
                         help="training seed, and the TPESampler's own seed")
    parser.add_argument("--study", type=str, default="tpe_imagenette_resnet",
                         help="Optuna study name; storage is "
                              "augment/tpe_studies.db")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    best_model = get_model_from_file(RESNET_FILE)

    storage = f"sqlite:///{STUDY_DB}"
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study, storage=storage, load_if_exists=True,
        direction="maximize", sampler=sampler,
    )
    print(f"Study '{args.study}': {len(study.trials)} trial(s) already recorded, "
          f"running {args.n_trials} more (fidelity={args.epochs}, seed={args.seed})")

    study.optimize(
        lambda trial: objective(trial, best_model, args.epochs, args.seed),
        n_trials=args.n_trials,
    )

    print(f"\nBest so far: acc={study.best_value:.4f}")
    print(f"  config_id={study.best_trial.user_attrs.get('config_id')}")
    print(f"  schedule={study.best_trial.user_attrs.get('canonical')}")


if __name__ == "__main__":
    main()
