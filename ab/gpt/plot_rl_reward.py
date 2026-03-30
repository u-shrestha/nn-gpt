#!/usr/bin/env python3
"""Plot cycle-level SFT RL reward diagnostics from generation_samples.jsonl and group_progress.jsonl."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - scipy is optional at runtime
    scipy_stats = None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return None


def _as_percent(values: Iterable[float]) -> list[float]:
    return [float(value) * 100.0 for value in values]


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _median(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def _best(values: list[float]) -> float:
    return float(np.max(values)) if values else float("nan")


def _ci_half_width(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    array = np.asarray(values, dtype=float)
    sem = float(np.std(array, ddof=1) / math.sqrt(len(array)))
    if scipy_stats is not None:
        return float(sem * scipy_stats.t.ppf(0.975, len(array) - 1))
    return float(1.96 * sem)


def _series_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "best": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "ci": float("nan"),
        }
    return {
        "best": _best(values),
        "mean": _mean(values),
        "median": _median(values),
        "ci": _ci_half_width(values),
    }


def _load_generation_samples(log_dir: Path) -> dict[int, list[dict[str, Any]]]:
    sample_path = log_dir / "generation_samples.jsonl"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing generation_samples.jsonl under {log_dir}")

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with sample_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            api_result = payload.get("api_result") or {}
            group_id = api_result.get("reward_group_id")
            if group_id is None:
                continue
            try:
                grouped[int(group_id)].append(api_result)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Line {line_no}: invalid reward_group_id {group_id!r}") from exc
    return dict(grouped)


def _load_group_progress(log_dir: Path) -> dict[int, dict[str, Any]]:
    progress_path = log_dir / "group_progress.jsonl"
    if not progress_path.exists():
        return {}

    progress_by_group: dict[int, dict[str, Any]] = {}
    with progress_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            group_id = payload.get("group_id")
            if group_id is None:
                raise ValueError(f"Line {line_no}: group_progress row missing group_id")
            progress_by_group[int(group_id)] = payload
    return progress_by_group


def _metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _coerce_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _rate(rows: list[dict[str, Any]], *, predicate) -> float:
    if not rows:
        return float("nan")
    hits = sum(1 for row in rows if predicate(row))
    return float(hits) / float(len(rows))


def _trainable_rate(rows: list[dict[str, Any]]) -> float:
    return _rate(
        rows,
        predicate=lambda row: all(
            _coerce_bool(row.get(field)) is True
            for field in ("built_ok", "forward_shape_ok", "backward_ok", "loss_drop_ok")
        ),
    )


def _positive_reward_rate(rows: list[dict[str, Any]]) -> float:
    return _rate(rows, predicate=lambda row: (_coerce_float(row.get("reward")) or 0.0) > 0.0)


def _timeout_rate(rows: list[dict[str, Any]]) -> float:
    return _rate(rows, predicate=lambda row: _coerce_bool(row.get("timed_out")) is True)


def _plot_accuracy_panel(
    ax,
    cycles: list[int],
    grouped_samples: dict[int, list[dict[str, Any]]],
    progress_by_group: dict[int, dict[str, Any]],
    *,
    metric_key: str,
    progress_key: Optional[str],
    title: str,
) -> None:
    best_values: list[float] = []
    mean_values: list[float] = []
    median_values: list[float] = []
    ci_values: list[float] = []
    closed_means: list[float] = []

    for cycle in cycles:
        stats = _series_stats(_metric_values(grouped_samples[cycle], metric_key))
        best_values.append(stats["best"])
        mean_values.append(stats["mean"])
        median_values.append(stats["median"])
        ci_values.append(stats["ci"])
        if progress_key is None or cycle not in progress_by_group:
            closed_means.append(float("nan"))
        else:
            closed_means.append(_coerce_float(progress_by_group[cycle].get(progress_key)) or float("nan"))

    mean_pct = np.asarray(_as_percent(mean_values), dtype=float)
    median_pct = np.asarray(_as_percent(median_values), dtype=float)
    best_pct = np.asarray(_as_percent(best_values), dtype=float)
    ci_pct = np.asarray(_as_percent([0.0 if math.isnan(v) else v for v in ci_values]), dtype=float)
    closed_mean_pct = np.asarray(_as_percent(closed_means), dtype=float)

    ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.errorbar(
        cycles,
        mean_pct,
        yerr=ci_pct,
        fmt="none",
        ecolor="#FF9800",
        elinewidth=1.6,
        capsize=4,
        alpha=0.65,
        zorder=1,
        label="95% CI",
    )
    ax.scatter(cycles, best_pct, marker="D", color="#E65100", s=56, edgecolors="white", linewidth=0.8, label="Best", zorder=5)
    ax.scatter(cycles, median_pct, marker="_", color="#D32F2F", s=180, linewidths=2.4, label="Median", zorder=6)
    ax.scatter(cycles, mean_pct, marker="o", color="#F57C00", s=36, edgecolors="black", linewidth=0.4, label="Mean", zorder=4)
    ax.plot(cycles, mean_pct, color="#F57C00", linewidth=1.4, alpha=0.75, zorder=2)
    if not np.all(np.isnan(closed_mean_pct)):
        ax.plot(cycles, closed_mean_pct, color="#1565C0", linewidth=1.8, linestyle="--", label="Closed Mean", zorder=3)
    ax.set_title(title)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)


def _plot_rate_panel(ax, cycles: list[int], grouped_samples: dict[int, list[dict[str, Any]]]) -> None:
    trainable = _as_percent([_trainable_rate(grouped_samples[cycle]) for cycle in cycles])
    positive = _as_percent([_positive_reward_rate(grouped_samples[cycle]) for cycle in cycles])
    timed_out = _as_percent([_timeout_rate(grouped_samples[cycle]) for cycle in cycles])

    ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.plot(cycles, trainable, marker="o", color="#1565C0", linewidth=2.0, label="Trainable Rate")
    ax.plot(cycles, positive, marker="s", color="#2E7D32", linewidth=2.0, label="Positive Reward Rate")
    ax.plot(cycles, timed_out, marker="^", color="#C62828", linewidth=2.0, label="Timeout Rate")
    ax.set_title("Cycle Stability")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="best", fontsize=8)


def _plot_component_panel(ax, cycles: list[int], grouped_samples: dict[int, list[dict[str, Any]]]) -> None:
    component_keys = [
        ("r_dense", "#1565C0"),
        ("r_prev_group", "#00897B"),
        ("r_best_group", "#5E35B1"),
        ("r_goal_best", "#EF6C00"),
        ("r_goal_match", "#7CB342"),
        ("r_trainset_novelty", "#6D4C41"),
        ("r_generalization", "#C62828"),
        ("r_repeat_family", "#546E7A"),
    ]
    ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
    for key, color in component_keys:
        values = [_mean(_metric_values(grouped_samples[cycle], key)) for cycle in cycles]
        ax.plot(cycles, values, marker="o", linewidth=1.8, markersize=4, label=key, color=color)
    ax.set_title("Reward Components")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Mean Component")
    ax.legend(loc="best", fontsize=7, ncol=2)


def _highlight_warmup(ax, cycles: list[int], grouped_samples: dict[int, list[dict[str, Any]]]) -> None:
    if not cycles:
        return
    warmup_cycles = [
        cycle
        for cycle in cycles
        if any(_coerce_bool(row.get("group_warmup")) is True for row in grouped_samples.get(cycle, []))
    ]
    for cycle in warmup_cycles:
        ax.axvspan(cycle - 0.4, cycle + 0.4, color="#FFF3E0", alpha=0.55, zorder=0)


def _print_summary(cycles: list[int], grouped_samples: dict[int, list[dict[str, Any]]], progress_by_group: dict[int, dict[str, Any]]) -> None:
    if not cycles:
        print("No cycle data found.")
        return
    last_cycle = cycles[-1]
    last_rows = grouped_samples[last_cycle]
    frozen_test = _metric_values(last_rows, "frozen_test_acc")
    reward_values = _metric_values(last_rows, "reward")
    progress = progress_by_group.get(last_cycle, {})
    print(
        "Last cycle summary:",
        json.dumps(
            {
                "cycle": last_cycle,
                "samples": len(last_rows),
                "frozen_test_best": _best(frozen_test) if frozen_test else None,
                "frozen_test_mean": _mean(frozen_test) if frozen_test else None,
                "reward_mean": _mean(reward_values) if reward_values else None,
                "closed_mean_reward_target_acc": _coerce_float(progress.get("closed_mean_reward_target_acc")),
                "closed_mean_train_acc": _coerce_float(progress.get("closed_mean_train_acc")),
                "closed_mean_test_acc": _coerce_float(progress.get("closed_mean_test_acc")),
            },
            ensure_ascii=False,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True, help="Directory containing generation_samples.jsonl")
    parser.add_argument("--output", default="cycle_reward_analysis.png", help="Output image path")
    args = parser.parse_args()

    log_dir = Path(args.log_dir).expanduser().resolve()
    grouped_samples = _load_generation_samples(log_dir)
    progress_by_group = _load_group_progress(log_dir)
    cycles = sorted(grouped_samples.keys())
    if not cycles:
        raise RuntimeError(f"No reward_group_id samples found under {log_dir}")

    fig, axes = plt.subplots(3, 2, figsize=(15, 13), dpi=140)
    axes = axes.flatten()

    _plot_accuracy_panel(
        axes[0],
        cycles,
        grouped_samples,
        progress_by_group,
        metric_key="frozen_test_acc",
        progress_key="closed_mean_test_acc",
        title="Frozen Test Accuracy by Cycle",
    )
    _plot_accuracy_panel(
        axes[1],
        cycles,
        grouped_samples,
        progress_by_group,
        metric_key="frozen_train_acc",
        progress_key="closed_mean_train_acc",
        title="Frozen Train Accuracy by Cycle",
    )
    _plot_accuracy_panel(
        axes[2],
        cycles,
        grouped_samples,
        progress_by_group,
        metric_key="unfrozen_test_acc",
        progress_key="closed_mean_unfrozen_test_acc",
        title="Unfrozen Test Accuracy by Cycle",
    )
    _plot_accuracy_panel(
        axes[3],
        cycles,
        grouped_samples,
        progress_by_group,
        metric_key="unfrozen_train_acc",
        progress_key="closed_mean_unfrozen_train_acc",
        title="Unfrozen Train Accuracy by Cycle",
    )
    _plot_rate_panel(axes[4], cycles, grouped_samples)
    _plot_component_panel(axes[5], cycles, grouped_samples)

    for ax in axes:
        _highlight_warmup(ax, cycles, grouped_samples)
        ax.set_xticks(cycles)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(loc="best", fontsize=8)

    fig.suptitle("RL Reward Cycle Analysis", fontsize=16, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_path = Path(args.output).expanduser().resolve()
    fig.savefig(output_path)
    print(f"Saved plot to {output_path}")
    _print_summary(cycles, grouped_samples, progress_by_group)


if __name__ == "__main__":
    main()
