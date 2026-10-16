"""Statistical correctness of the per-epoch report (ab/gpt/util/data/EpochStats.py).

Both cases here are regressions of bugs that shipped: a Wilson interval that
divided the wrong term by n, and a baseline that averaged the seed accuracy of
every generation attempt instead of only the ones the LLM modified successfully.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    cs = importlib.util.spec_from_file_location(
        "ab.gpt.util.data.CycleResults", ROOT / "ab/gpt/util/data/CycleResults.py")
    cr = importlib.util.module_from_spec(cs)
    cs.loader.exec_module(cr)
    sys.modules.setdefault("ab.gpt.util.data.CycleResults", cr)
    es = importlib.util.spec_from_file_location(
        "_epoch_stats_under_test", ROOT / "ab/gpt/util/data/EpochStats.py")
    mod = importlib.util.module_from_spec(es)
    es.loader.exec_module(mod)
    return mod


ES = _load()


@pytest.mark.parametrize("k, n, lo, hi", [
    (0, 10, 0.0000, 0.2775),
    (2, 10, 0.0566, 0.5098),
    (4, 10, 0.1682, 0.6873),
    (10, 10, 0.7225, 1.0000),
])
def test_wilson_interval_matches_reference(k, n, lo, hi):
    _, got_lo, got_hi = ES.wilson_ci(k, n)
    assert got_lo == pytest.approx(lo, abs=1e-4)
    assert got_hi == pytest.approx(hi, abs=1e-4)


def test_wilson_interval_is_degenerate_only_for_no_trials():
    assert ES.wilson_ci(0, 0) == (0.0, 0.0, 0.0)
    # zero successes out of n must still bound the rate above zero
    assert ES.wilson_ci(0, 10)[2] > 0.0


def test_mean_ci_uses_t_not_z_for_small_samples():
    vals = [0.5895, 0.41, 0.33, 0.25]
    out = ES.mean_ci(vals)
    assert out["n"] == 4
    half_t = out["ci_high"] - out["mean"]
    half_z = 1.96 * out["std"] / 2.0          # sqrt(n) == 2
    assert half_t > half_z                     # t(3)=3.182 is wider than 1.96


def _make_epoch(tmp_path, valid, failed_seeds):
    d = tmp_path / "A0" / "synth_nn"
    d.mkdir(parents=True)
    pd = pytest.importorskip("pandas")
    for i, (seed, acc) in valid.items():
        md = d / f"B{i}"
        md.mkdir()
        pd.Series({"nn": f"seed-{i}", "accuracy": seed}).to_pickle(md / "dataframe.df")
        (md / "eval_info.json").write_text(json.dumps({"eval_results": {"accuracy": acc}}))
    for i, seed in failed_seeds.items():
        md = d / f"B{i}"
        md.mkdir()
        pd.Series({"nn": f"seed-{i}", "accuracy": seed}).to_pickle(md / "dataframe.df")
        (md / "error.txt").write_text("compile error")
    return d


def test_baseline_covers_only_successfully_modified_models(tmp_path):
    # Seeds of the failing models are deliberately far from the succeeding ones,
    # so averaging all of them gives a visibly different answer.
    valid = {0: (0.30, 0.5895), 1: (0.32, 0.41), 2: (0.28, 0.33), 3: (0.34, 0.25)}
    failed = {i: 0.90 for i in range(4, 10)}
    stats = ES.collect_epoch_stats(_make_epoch(tmp_path, valid, failed), 0)

    assert stats["models_generated"] == 10
    assert stats["models_valid"] == 4
    assert stats["baseline_accuracy_n"] == 4          # not 10
    assert stats["baseline_accuracy_mean"] == pytest.approx(0.31)
    assert stats["baseline_models"] == ["seed-0", "seed-1", "seed-2", "seed-3"]


def test_improvement_is_paired_per_model(tmp_path):
    valid = {0: (0.30, 0.5895), 1: (0.32, 0.41), 2: (0.28, 0.33), 3: (0.34, 0.25)}
    stats = ES.collect_epoch_stats(_make_epoch(tmp_path, valid, {}), 0)
    expected = sum(a - s for s, a in valid.values()) / len(valid)
    assert stats["paired_n"] == 4
    assert stats["mean_improvement"] == pytest.approx(expected)
    assert stats["improvement_ci_low"] < stats["mean_improvement"] < stats["improvement_ci_high"]


def test_epoch_with_no_valid_models_reports_nulls_not_zeros(tmp_path):
    stats = ES.collect_epoch_stats(_make_epoch(tmp_path, {}, {i: 0.5 for i in range(10)}), 2)
    assert stats["models_valid"] == 0
    assert stats["accuracy_mean"] is None
    assert stats["accuracy_max"] is None
    assert stats["mean_improvement"] is None
    assert stats["valid_rate"] == 0.0
    assert stats["valid_rate_ci_high"] > 0.0


def test_tracker_replaces_rather_than_duplicates_a_rerun_epoch(tmp_path):
    path = tmp_path / "epoch_tracker.json"
    ES.append_epoch_stats(path, {"epoch": 0, "accuracy_max": 0.1})
    ES.append_epoch_stats(path, {"epoch": 1, "accuracy_max": 0.2})
    rows = ES.append_epoch_stats(path, {"epoch": 0, "accuracy_max": 0.9})
    assert [r["epoch"] for r in rows] == [0, 1]
    assert rows[0]["accuracy_max"] == 0.9


# --- finalisation of the per-epoch results file ---------------------------

def _stats(epoch, valid=2, total=10, best=0.45):
    return {"epoch": epoch, "models_valid": valid, "models_generated": total,
            "accuracy_max": best, "accuracy_mean": best, "valid_rate": valid / total}


def test_finalize_adopts_a_generic_file_left_by_the_last_cycle(tmp_path):
    # The run that "kept the name cycle_results.json": NNEval wrote only the
    # generic file, so the epoch has no record of its own.
    generic = tmp_path / "cycle_results.json"
    generic.write_text(json.dumps(
        {"cycle": 3, "evaluation": {"best_accuracy": 0.45}, "cycle_time_minutes": 12.0}))

    ES.finalize_epoch_results(tmp_path, 3, _stats(3))

    per_epoch = tmp_path / "cycle_results_3.json"
    assert per_epoch.is_file()
    payload = json.loads(per_epoch.read_text())
    assert payload["cycle"] == 3
    assert payload["cycle_time_minutes"] == 12.0      # original content preserved
    assert payload["statistics"]["models_valid"] == 2  # statistics merged in


def test_finalize_does_not_adopt_a_generic_file_from_another_epoch(tmp_path):
    (tmp_path / "cycle_results.json").write_text(json.dumps(
        {"cycle": 2, "evaluation": {"best_accuracy": 0.99}, "cycle_time_minutes": 5.0}))

    ES.finalize_epoch_results(tmp_path, 3, _stats(3, best=0.45))

    payload = json.loads((tmp_path / "cycle_results_3.json").read_text())
    assert payload["cycle"] == 3
    assert payload["evaluation"]["best_accuracy"] == 0.45   # rebuilt, not epoch 2's 0.99
    assert payload.get("cycle_time_minutes") is None


def test_finalize_writes_a_record_when_evaluation_never_ran(tmp_path):
    ES.finalize_epoch_results(tmp_path, 4, _stats(4, valid=0, total=10, best=None))
    payload = json.loads((tmp_path / "cycle_results_4.json").read_text())
    assert payload["cycle"] == 4
    assert payload["success"] is False
    assert payload["generation"]["total_generated"] == 10
    assert payload["evaluation"]["best_accuracy"] is None
    # the generic name is refreshed too, so both agree
    assert json.loads((tmp_path / "cycle_results.json").read_text())["cycle"] == 4


def test_update_epoch_report_finalises_and_tracks_a_collapsed_epoch(tmp_path):
    pd = pytest.importorskip("pandas")
    d = tmp_path / "A2" / "synth_nn"
    d.mkdir(parents=True)
    for i in range(10):                      # all generations failed
        md = d / f"B{i}"
        md.mkdir()
        pd.Series({"nn": f"seed-{i}", "accuracy": 0.5}).to_pickle(md / "dataframe.df")
        (md / "error.txt").write_text("compile error")

    ES.update_epoch_report(d, 2, tmp_path)

    rows = json.loads((tmp_path / "epoch_tracker.json").read_text())
    assert [r["epoch"] for r in rows] == [2]
    assert rows[0]["models_valid"] == 0 and rows[0]["models_generated"] == 10
    assert (tmp_path / "cycle_results_2.json").is_file()


# --- clearing the previous run's artifacts --------------------------------

def _seed_run_dir(tmp_path, epochs=range(20)):
    (tmp_path / "cycle_results.json").write_text("{}")
    (tmp_path / "epoch_tracker.json").write_text("[]")
    (tmp_path / "epoch_statistics.png").write_bytes(b"\x89PNG")
    for e in epochs:
        (tmp_path / f"cycle_results_{e}.json").write_text("{}")
    # files that belong to the model, not to the statistics, must survive
    (tmp_path / "run_config.json").write_text("{}")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"w")
    keep = tmp_path / "checkpoint"
    keep.mkdir()
    (keep / "adapter_config.json").write_text("{}")


def test_reset_removes_every_statistics_file_of_a_previous_run(tmp_path):
    _seed_run_dir(tmp_path)
    removed = ES.reset_run_artifacts(tmp_path)

    assert len(removed) == 23            # 20 per-epoch + 3 named
    assert not list(tmp_path.glob("cycle_results*.json"))
    assert not (tmp_path / "epoch_tracker.json").exists()
    assert not (tmp_path / "epoch_statistics.png").exists()


def test_reset_leaves_model_and_config_files_alone(tmp_path):
    _seed_run_dir(tmp_path, epochs=range(3))
    ES.reset_run_artifacts(tmp_path)

    assert (tmp_path / "run_config.json").is_file()
    assert (tmp_path / "adapter_model.safetensors").is_file()
    assert (tmp_path / "checkpoint" / "adapter_config.json").is_file()


def test_reset_is_a_noop_on_a_clean_or_missing_directory(tmp_path):
    assert ES.reset_run_artifacts(tmp_path) == []
    assert ES.reset_run_artifacts(tmp_path / "does_not_exist") == []


def test_without_reset_a_short_run_inherits_a_longer_run_s_tail(tmp_path):
    # 20-epoch run, then a 4-epoch run writing over epochs 0..3 only.
    path = tmp_path / "epoch_tracker.json"
    for e in range(20):
        ES.append_epoch_stats(path, {"epoch": e, "accuracy_max": 0.1})
    for e in range(4):
        rows = ES.append_epoch_stats(path, {"epoch": e, "accuracy_max": 0.9})
    assert len(rows) == 20                      # epochs 4..19 are stale leftovers

    ES.reset_run_artifacts(tmp_path)
    for e in range(4):
        rows = ES.append_epoch_stats(path, {"epoch": e, "accuracy_max": 0.9})
    assert [r["epoch"] for r in rows] == [0, 1, 2, 3]
