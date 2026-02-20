from __future__ import annotations

import os
import sys
import time
import statistics
from typing import Iterable, Sequence

import ab.nn.api as lemur
from ab.nn.api import JoinConf
import functools

def measure_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result

    return wrapper

#Fixed Benchmarks
TASK = os.getenv("LEMUR_TASK", "img-classification")
DATASET = os.getenv("LEMUR_DATASET", "cifar-10")
METRIC = os.getenv("LEMUR_METRIC", "acc")

#Tuning knobs
REPEATS = int(os.getenv("LEMUR_BENCH_REPEATS", "3"))          # how many timed runs (median taken)
WARMUP = int(os.getenv("LEMUR_BENCH_WARMUP", "1"))           # warmup runs (not counted)
LEGACY_ROWS = int(os.getenv("LEMUR_LEGACY_ROWS", "20000"))   # legacy perf smoke max_rows
VAR_N_ROWS = int(os.getenv("LEMUR_VARN_ROWS", "500"))        # var-N perf smoke max_rows
VAR_N = int(os.getenv("LEMUR_VARN", "10"))                   # N for variable-N query

# thresholds (seconds)
THRESH_LEGACY = float(os.getenv("LEMUR_THRESH_LEGACY", "60"))
THRESH_VARN = float(os.getenv("LEMUR_THRESH_VARN", "30"))
#THRESH_SCHEMA = float(os.getenv("LEMUR_THRESH_SCHEMA", "9999"))  # usually don't gate schema test by time


def assert_columns(df, required: Sequence[str], label: str = ""):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise AssertionError(f"{label} missing columns: {missing}. Have={list(df.columns)}")


def bench(fn, *, repeats: int = REPEATS, warmup: int = WARMUP) -> tuple[float, list[float]]:
    # Warmup (build caches, load modules, first tmp_data creation, etc.)
    for _ in range(max(0, warmup)):
        fn()

    times = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)

    return statistics.median(times), times


def print_bench(name: str, median_s: float, runs: list[float]):
    runs_str = ", ".join(f"{t:.2f}" for t in runs)
    print(f"[BENCH] {name}: median={median_s:.2f}s runs=[{runs_str}] (task={TASK}, dataset={DATASET}, metric={METRIC})")


# ---------------------- Tests ----------------------
@measure_time
def test_legacy_pairwise_schema():
    """
    Legacy behavior: 2-model join, wide schema.
    MUST stay unchanged.
    """
    conf = JoinConf(
        num_joint_nns=2,
        same_columns=("epoch", "metric", "dataset", "task"),
        diff_columns=("nn",),
        enhance_nn=True,
        similarity_mode="none",
    )

    df = lemur.data(
        sql=conf,
        include_nn_stats=True,
        # IMPORTANT: these must be here so tmp_data is scoped!
        task=TASK,
        dataset=DATASET,
        metric=METRIC,
        max_rows=200,   # keep schema test small
    )

    required = [
        "nn", "epoch", "nn_code", "accuracy", "prm_id", "prm", "transform_code",
        "nn_2", "epoch_2", "nn_code_2", "accuracy_2", "prm_id_2", "prm_2", "transform_code_2"
    ]
    assert_columns(df, required, label="legacy schema")

    print("[PASS] legacy pairwise schema")
    print(df[["nn", "accuracy", "nn_2", "accuracy_2"]].head(5))

@measure_time
def test_sql_variable_n_correctness():
    """
    NEW: SQL-only variable-N model selection (tall schema).
    One row per model, no joins.
    """
    N = 5
    conf = JoinConf(
        num_joint_nns=N,
        similarity_mode="none",
    )

    df = lemur.data(
        sql=conf,
        include_nn_stats=False,
        task=TASK,
        dataset=DATASET,
        metric=METRIC,
        max_rows=200,
    )

    if len(df) != N:
        raise AssertionError(f"Expected {N} rows, got {len(df)}")

    required = ["nn", "accuracy", "prm_id"]
    assert_columns(df, required, label="var-N correctness")

    if not df["nn"].is_unique:
        raise AssertionError("Duplicate nn entries detected")

    print("[PASS] SQL variable-N selection correctness")
    print(df[["nn", "accuracy"]].head(10))

@measure_time
def test_otf_anchor_band_correctness():
    """
    OTF anchor band: returns rows with anchor_jaccard.
    """
    conf = JoinConf(
        num_joint_nns=5,
        similarity_mode="anchor_band_otf",
        anchor_nn=os.getenv("LEMUR_ANCHOR_NN", "rl-back-init-0b0d0b7728c47eb6cc9fe040ebb9d239"),
        similarity_band=os.getenv("LEMUR_SIM_BAND", "high"),
    )

    df = lemur.data(
        sql=conf,
        include_nn_stats=False,
        task=TASK,
        dataset=DATASET,
        metric=METRIC,
        max_rows=200,
    )

    if len(df) <= 0:
        raise AssertionError("Expected >0 rows for OTF anchor band")
    assert_columns(df, ["nn", "accuracy", "anchor_jaccard"], label="OTF anchor band")

    print("[PASS] OTF anchor band correctness")
    print(df[["nn", "accuracy", "anchor_jaccard"]].head(10))

@measure_time
def test_legacy_performance_smoke():
    """
    Performance smoke test for legacy path.
    """
    conf = JoinConf(
        num_joint_nns=2,
        diff_columns=("nn",),
        similarity_mode="none",
    )

    def run():
        lemur.data(
            sql=conf,
            include_nn_stats=False,
            task=TASK,
            dataset=DATASET,
            metric=METRIC,
            max_rows=LEGACY_ROWS,
        )

    median_s, runs = bench(run)
    print_bench("legacy_perf", median_s, runs)

    if median_s > THRESH_LEGACY:
        raise AssertionError(f"Legacy query unexpectedly slow: median {median_s:.2f}s > {THRESH_LEGACY:.2f}s")

@measure_time
def test_sql_variable_n_performance_smoke():
    """
    Performance smoke test for SQL-only variable-N path.
    """
    conf = JoinConf(
        num_joint_nns=VAR_N,
        similarity_mode="none",
    )

    def run():
        df = lemur.data(
            sql=conf,
            include_nn_stats=False,
            task=TASK,
            dataset=DATASET,
            metric=METRIC,
            max_rows=VAR_N_ROWS,
        )
        # Hard correctness guarantees
        if len(df) != VAR_N:
            raise AssertionError(f"Expected {VAR_N} rows, got {len(df)}")
        if not df["nn"].is_unique:
            raise AssertionError("Duplicate nn entries detected")

    median_s, runs = bench(run)
    print_bench("sql_variable_n_perf", median_s, runs)

    if median_s > THRESH_VARN:
        raise AssertionError(f"SQL variable-N query unexpectedly slow: median {median_s:.2f}s > {THRESH_VARN:.2f}s")


def main() -> int:
    print("LEMUR / NNGPT integration tests")
    print(f"[SCOPE] task={TASK} dataset={DATASET} metric={METRIC}")
    print(f"[BENCH CFG] repeats={REPEATS} warmup={WARMUP} legacy_rows={LEGACY_ROWS} varN={VAR_N} varN_rows={VAR_N_ROWS}")
    print(f"[THRESH] legacy<{THRESH_LEGACY}s varN<{THRESH_VARN}s")

    # Schema/correctness first
    t0 = time.perf_counter()
    test_legacy_pairwise_schema()
    test_sql_variable_n_correctness()
    test_otf_anchor_band_correctness()
    schema_dt = time.perf_counter() - t0
    print(f"[INFO] correctness suite took {schema_dt:.2f}s")
    #if schema_dt > THRESH_SCHEMA:
        #raise AssertionError("Correctness suite unexpectedly slow (usually indicates huge tmp_data scope)")

    # Perf smoke after correctness
    test_legacy_performance_smoke()
    test_sql_variable_n_performance_smoke()

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
