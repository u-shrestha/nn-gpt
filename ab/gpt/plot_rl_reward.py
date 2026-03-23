#!/usr/bin/env python3
"""Plot a single-run RL reward diagnostics dashboard from generation_samples.jsonl."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Sequence


@dataclass
class RewardLogData:
    run_name: str
    sample_index: list[int]
    reward: list[float]
    built_ok: list[float]
    forward_shape_ok: list[float]
    backward_ok: list[float]
    loss_drop_ok: list[float]
    train_acc: list[float]
    accuracy_baseline: list[float]
    train_acc_gain: list[float]
    train_acc_improved: list[float]
    loss_start: list[float]
    loss_end: list[float]
    loss_drop: list[float]
    val_metric: list[float]
    reward_positive: list[float]
    novel_vs_trainset_family: list[float]
    novel_vs_trainset_graph: list[float]
    r_trainset_novelty: list[float]
    xml_tag_exact: list[float]
    dual_backbone_ok: list[float]
    format_violation: list[float]

    @property
    def count(self) -> int:
        return len(self.reward)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("window must be a positive integer")
    return parsed


def _is_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _require_mapping(container: dict[str, Any], key: str, line_no: int) -> dict[str, Any]:
    if key not in container:
        raise ValueError(f"Line {line_no}: missing required mapping field '{key}'")
    value = container[key]
    if not isinstance(value, dict):
        raise ValueError(f"Line {line_no}: field '{key}' must be a JSON object")
    return value


def _optional_mapping(container: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = container.get(key)
    return value if isinstance(value, dict) else None


def _require_numeric(container: dict[str, Any], key: str, line_no: int) -> float:
    if key not in container:
        raise ValueError(f"Line {line_no}: missing required numeric field '{key}'")
    value = container[key]
    if not _is_number(value):
        raise ValueError(f"Line {line_no}: field '{key}' must be numeric")
    return float(value)


def _require_optional_numeric(container: dict[str, Any], key: str, line_no: int) -> float:
    if key not in container:
        raise ValueError(f"Line {line_no}: missing required field '{key}'")
    value = container[key]
    if value is None:
        return float("nan")
    if not _is_number(value):
        raise ValueError(f"Line {line_no}: field '{key}' must be numeric or null")
    return float(value)


def _optional_numeric(container: dict[str, Any] | None, key: str, *, default: float, line_no: int) -> float:
    if container is None or key not in container:
        return default
    value = container[key]
    if value is None:
        return float("nan")
    if not _is_number(value):
        raise ValueError(f"Line {line_no}: field '{key}' must be numeric or null")
    return float(value)


def _require_bool(container: dict[str, Any], key: str, line_no: int) -> float:
    if key not in container:
        raise ValueError(f"Line {line_no}: missing required boolean field '{key}'")
    value = container[key]
    if not isinstance(value, bool):
        raise ValueError(f"Line {line_no}: field '{key}' must be boolean")
    return 1.0 if value else 0.0


def _optional_bool(container: dict[str, Any] | None, key: str, *, default: float, line_no: int) -> float:
    if container is None or key not in container:
        return default
    value = container[key]
    if not isinstance(value, bool):
        raise ValueError(f"Line {line_no}: field '{key}' must be boolean")
    return 1.0 if value else 0.0


def rolling_nanmean(values: Sequence[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")

    result: list[float] = []
    for idx in range(len(values)):
        window_slice = values[max(0, idx - window + 1): idx + 1]
        valid = [value for value in window_slice if not math.isnan(value)]
        result.append(sum(valid) / len(valid) if valid else float("nan"))
    return result


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _nanmean(values: Sequence[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return sum(valid) / len(valid) if valid else float("nan")


def _format_metric(value: float, *, percent: bool = False) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.2%}" if percent else f"{value:.4f}"


def load_reward_log(log_dir: Path) -> RewardLogData:
    log_dir = Path(log_dir)
    log_file = log_dir / "generation_samples.jsonl"
    if not log_file.exists():
        raise FileNotFoundError(f"Required log file not found: {log_file}")

    reward: list[float] = []
    built_ok: list[float] = []
    forward_shape_ok: list[float] = []
    backward_ok: list[float] = []
    loss_drop_ok: list[float] = []
    train_acc: list[float] = []
    accuracy_baseline: list[float] = []
    train_acc_gain: list[float] = []
    train_acc_improved: list[float] = []
    loss_start: list[float] = []
    loss_end: list[float] = []
    loss_drop: list[float] = []
    val_metric: list[float] = []
    reward_positive: list[float] = []
    novel_vs_trainset_family: list[float] = []
    novel_vs_trainset_graph: list[float] = []
    r_trainset_novelty: list[float] = []
    xml_tag_exact: list[float] = []
    dual_backbone_ok: list[float] = []
    format_violation: list[float] = []

    with log_file.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                raise ValueError(f"Line {line_no}: blank lines are not allowed")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_no}: top-level JSON value must be an object")

            reward_value = _require_numeric(record, "reward", line_no)
            api_result = _require_mapping(record, "api_result", line_no)
            open_discovery = _optional_mapping(api_result, "open_discovery")
            raw_extraction = _optional_mapping(api_result, "raw_extraction")

            reward.append(reward_value)
            built_ok.append(_optional_bool(api_result, "built_ok", default=0.0, line_no=line_no))
            forward_shape_ok.append(_optional_bool(api_result, "forward_shape_ok", default=0.0, line_no=line_no))
            backward_ok.append(_optional_bool(api_result, "backward_ok", default=0.0, line_no=line_no))
            loss_drop_ok.append(_optional_bool(api_result, "loss_drop_ok", default=0.0, line_no=line_no))
            train_acc.append(_optional_numeric(api_result, "train_acc", default=float("nan"), line_no=line_no))
            accuracy_baseline.append(
                _optional_numeric(api_result, "accuracy_baseline", default=float("nan"), line_no=line_no)
            )
            train_acc_gain.append(_optional_numeric(api_result, "train_acc_gain", default=float("nan"), line_no=line_no))
            train_acc_improved.append(_optional_bool(api_result, "train_acc_improved", default=0.0, line_no=line_no))
            loss_start.append(_optional_numeric(api_result, "loss_start", default=float("nan"), line_no=line_no))
            loss_end.append(_optional_numeric(api_result, "loss_end", default=float("nan"), line_no=line_no))
            loss_drop.append(_optional_numeric(api_result, "loss_drop", default=float("nan"), line_no=line_no))
            val_metric.append(_optional_numeric(api_result, "val_metric", default=float("nan"), line_no=line_no))
            reward_positive.append(1.0 if reward_value > 0.0 else 0.0)

            novel_vs_trainset_family.append(
                _optional_bool(open_discovery, "novel_vs_trainset_family", default=0.0, line_no=line_no)
            )
            novel_vs_trainset_graph.append(
                _optional_bool(open_discovery, "novel_vs_trainset_graph", default=0.0, line_no=line_no)
            )
            r_trainset_novelty.append(
                _optional_numeric(open_discovery, "r_trainset_novelty", default=0.0, line_no=line_no)
            )

            xml_tag_exact.append(_optional_bool(raw_extraction, "xml_tag_exact", default=float("nan"), line_no=line_no))
            dual_backbone_ok.append(
                _optional_bool(raw_extraction, "dual_backbone_ok", default=float("nan"), line_no=line_no)
            )
            class_count = _optional_numeric(raw_extraction, "class_count", default=float("nan"), line_no=line_no)
            import_count = _optional_numeric(raw_extraction, "import_count", default=float("nan"), line_no=line_no)
            bad_signature_count = _optional_numeric(
                raw_extraction, "bad_signature_count", default=float("nan"), line_no=line_no
            )
            if any(math.isnan(value) for value in (class_count, import_count, bad_signature_count)):
                format_violation.append(float("nan"))
            else:
                format_violation.append(1.0 if (class_count > 0 or import_count > 0 or bad_signature_count > 0) else 0.0)

    if not reward:
        raise ValueError(f"No samples found in {log_file}")

    return RewardLogData(
        run_name=log_dir.name or str(log_dir),
        sample_index=list(range(1, len(reward) + 1)),
        reward=reward,
        built_ok=built_ok,
        forward_shape_ok=forward_shape_ok,
        backward_ok=backward_ok,
        loss_drop_ok=loss_drop_ok,
        train_acc=train_acc,
        accuracy_baseline=accuracy_baseline,
        train_acc_gain=train_acc_gain,
        train_acc_improved=train_acc_improved,
        loss_start=loss_start,
        loss_end=loss_end,
        loss_drop=loss_drop,
        val_metric=val_metric,
        reward_positive=reward_positive,
        novel_vs_trainset_family=novel_vs_trainset_family,
        novel_vs_trainset_graph=novel_vs_trainset_graph,
        r_trainset_novelty=r_trainset_novelty,
        xml_tag_exact=xml_tag_exact,
        dual_backbone_ok=dual_backbone_ok,
        format_violation=format_violation,
    )


def compute_summary(data: RewardLogData, window: int) -> dict[str, float]:
    reward_rolling = rolling_nanmean(data.reward, window)
    return {
        "count": float(data.count),
        "reward_min": min(data.reward),
        "reward_max": max(data.reward),
        "reward_mean": _mean(data.reward),
        "reward_final_rolling_mean": reward_rolling[-1],
        "positive_reward_rate": _mean(data.reward_positive),
        "built_ok_rate": _mean(data.built_ok),
        "forward_shape_ok_rate": _mean(data.forward_shape_ok),
        "backward_ok_rate": _mean(data.backward_ok),
        "loss_drop_ok_rate": _mean(data.loss_drop_ok),
        "train_acc_mean": _nanmean(data.train_acc),
        "accuracy_baseline_mean": _nanmean(data.accuracy_baseline),
        "train_acc_gain_mean": _nanmean(data.train_acc_gain),
        "train_acc_gain_max": max(value for value in data.train_acc_gain if not math.isnan(value))
        if any(not math.isnan(value) for value in data.train_acc_gain)
        else float("nan"),
        "train_acc_improved_rate": _mean(data.train_acc_improved),
        "novel_family_rate": _mean(data.novel_vs_trainset_family),
        "novel_graph_rate": _mean(data.novel_vs_trainset_graph),
        "xml_tag_exact_rate": _nanmean(data.xml_tag_exact),
        "dual_backbone_ok_rate": _nanmean(data.dual_backbone_ok),
    }


def _style_axis(ax: Any, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Sample Index", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(fontsize=8, framealpha=0.9)


def plot_dashboard(data: RewardLogData, output_path: Path, window: int, title: str | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reward_rolling = rolling_nanmean(data.reward, window)
    built_rate = rolling_nanmean(data.built_ok, window)
    shape_rate = rolling_nanmean(data.forward_shape_ok, window)
    backward_rate = rolling_nanmean(data.backward_ok, window)
    loss_drop_rate = rolling_nanmean(data.loss_drop_ok, window)

    train_acc_rolling = rolling_nanmean(data.train_acc, window)
    accuracy_baseline_rolling = rolling_nanmean(data.accuracy_baseline, window)
    train_acc_gain_rolling = rolling_nanmean(data.train_acc_gain, window)
    train_acc_improved_rate = rolling_nanmean(data.train_acc_improved, window)

    val_metric_rolling = rolling_nanmean(data.val_metric, window)
    positive_reward_rate = rolling_nanmean(data.reward_positive, window)

    novel_family_rate = rolling_nanmean(data.novel_vs_trainset_family, window)
    novel_graph_rate = rolling_nanmean(data.novel_vs_trainset_graph, window)
    trainset_novelty_rolling = rolling_nanmean(data.r_trainset_novelty, window)

    xml_exact_rate = rolling_nanmean(data.xml_tag_exact, window)
    dual_backbone_rate = rolling_nanmean(data.dual_backbone_ok, window)
    violation_rate = rolling_nanmean(data.format_violation, window)

    figure, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=160)
    axes = axes.flatten() if hasattr(axes, "flatten") else axes

    axes[0].plot(data.sample_index, data.reward, color="#B0BEC5", linewidth=1.1, alpha=0.55, label="Raw reward")
    axes[0].plot(data.sample_index, reward_rolling, color="#1565C0", linewidth=2.2, label=f"Rolling mean ({window})")
    axes[0].axhline(0.0, color="#424242", linestyle="--", linewidth=1.0, label="Zero reward")
    _style_axis(axes[0], "Reward Trend", "Reward")

    axes[1].plot(data.sample_index, built_rate, color="#2E7D32", linewidth=2.0, label="built_ok")
    axes[1].plot(data.sample_index, shape_rate, color="#00897B", linewidth=2.0, label="forward_shape_ok")
    axes[1].plot(data.sample_index, backward_rate, color="#6A1B9A", linewidth=2.0, label="backward_ok")
    axes[1].plot(data.sample_index, loss_drop_rate, color="#EF6C00", linewidth=2.0, label="loss_drop_ok")
    axes[1].set_ylim(0.0, 1.0)
    _style_axis(axes[1], "Trainability Gates", "Rolling Pass Rate")

    axes[2].plot(data.sample_index, train_acc_rolling, color="#1565C0", linewidth=2.0, label="train_acc")
    axes[2].plot(
        data.sample_index,
        accuracy_baseline_rolling,
        color="#6D4C41",
        linewidth=2.0,
        label="accuracy_baseline",
    )
    axes[2].plot(data.sample_index, train_acc_gain_rolling, color="#00897B", linewidth=2.0, label="train_acc_gain")
    axes[2].plot(
        data.sample_index,
        train_acc_improved_rate,
        color="#EF6C00",
        linewidth=2.0,
        label="train_acc_improved",
    )
    _style_axis(axes[2], "Train Accuracy & Gain", "Accuracy / Gain / Rate")

    axes[3].plot(data.sample_index, val_metric_rolling, color="#7B1FA2", linewidth=2.0, label="val_metric")
    axes[3].plot(data.sample_index, positive_reward_rate, color="#F9A825", linewidth=2.0, label="reward > 0")
    axes[3].set_ylim(0.0, 1.0)
    _style_axis(axes[3], "Validation & Success Rate", "Value / Rate")

    axes[4].plot(data.sample_index, novel_family_rate, color="#1B5E20", linewidth=2.0, label="novel family")
    axes[4].plot(data.sample_index, novel_graph_rate, color="#0D47A1", linewidth=2.0, label="novel graph")
    axes[4].plot(data.sample_index, trainset_novelty_rolling, color="#F4511E", linewidth=2.0, label="r_trainset_novelty")
    axes[4].set_ylim(0.0, 1.0)
    _style_axis(axes[4], "Novelty Trend", "Rate / Reward")

    axes[5].plot(data.sample_index, xml_exact_rate, color="#004D40", linewidth=2.0, label="xml_tag_exact")
    axes[5].plot(data.sample_index, dual_backbone_rate, color="#37474F", linewidth=2.0, label="dual_backbone_ok")
    axes[5].plot(data.sample_index, violation_rate, color="#D84315", linewidth=2.0, label="format violation")
    axes[5].set_ylim(0.0, 1.0)
    _style_axis(axes[5], "Format / Constraint Quality", "Rolling Rate")

    display_title = title or data.run_name
    figure.suptitle(f"{display_title} Reward Dashboard (n={data.count})", fontsize=15, fontweight="bold")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def print_summary(summary: dict[str, float], run_name: str) -> None:
    print(f"Run: {run_name}")
    print(f"Samples: {int(summary['count'])}")
    print(
        "Reward: "
        f"min={_format_metric(summary['reward_min'])}, max={_format_metric(summary['reward_max'])}, "
        f"mean={_format_metric(summary['reward_mean'])}, "
        f"final_rolling_mean={_format_metric(summary['reward_final_rolling_mean'])}"
    )
    print(f"Positive reward rate: {_format_metric(summary['positive_reward_rate'], percent=True)}")
    print(
        "Gate pass rates: "
        f"built={_format_metric(summary['built_ok_rate'], percent=True)}, "
        f"shape={_format_metric(summary['forward_shape_ok_rate'], percent=True)}, "
        f"backward={_format_metric(summary['backward_ok_rate'], percent=True)}, "
        f"loss_drop={_format_metric(summary['loss_drop_ok_rate'], percent=True)}"
    )
    print(
        "Train accuracy: "
        f"mean={_format_metric(summary['train_acc_mean'], percent=True)}, "
        f"baseline_mean={_format_metric(summary['accuracy_baseline_mean'], percent=True)}, "
        f"gain_mean={_format_metric(summary['train_acc_gain_mean'], percent=True)}, "
        f"gain_max={_format_metric(summary['train_acc_gain_max'], percent=True)}, "
        f"improved_rate={_format_metric(summary['train_acc_improved_rate'], percent=True)}"
    )
    print(
        "Novelty rates: "
        f"family={_format_metric(summary['novel_family_rate'], percent=True)}, "
        f"graph={_format_metric(summary['novel_graph_rate'], percent=True)}"
    )
    print(
        "Format rates: "
        f"xml_exact={_format_metric(summary['xml_tag_exact_rate'], percent=True)}, "
        f"dual_backbone_ok={_format_metric(summary['dual_backbone_ok_rate'], percent=True)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a single-run RL reward diagnostics dashboard")
    parser.add_argument("--log-dir", required=True, help="Directory containing generation_samples.jsonl")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument("--title", default=None, help="Optional chart title override")
    parser.add_argument("--window", type=_positive_int, default=32, help="Rolling window size")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    log_dir = Path(args.log_dir)
    data = load_reward_log(log_dir)
    output_path = Path(args.output) if args.output else log_dir / "reward_dashboard.png"

    plot_dashboard(data, output_path=output_path, window=args.window, title=args.title)
    summary = compute_summary(data, window=args.window)
    print_summary(summary, run_name=data.run_name)
    print(f"Saved dashboard to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
