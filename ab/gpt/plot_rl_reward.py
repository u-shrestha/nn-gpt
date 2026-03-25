#!/usr/bin/env python3
"""Plot a single-run SFT RL reward dashboard from generation_samples.jsonl and training_progress.log."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Sequence


PROGRESS_PATTERN = re.compile(
    r"progress:\s*"
    r"(?P<generation_total>\d+)\s+generation，"
    r"(?P<success_count>\d+)\s+success，success rate\s+"
    r"(?P<success_rate>\d+(?:\.\d+)?)%\s+"
    r"warmup_trainable_count=(?P<warmup_trainable_count>\d+)\s+"
    r"warmup_positive_count=(?P<warmup_positive_count>\d+)\s+"
    r"timeout_count=(?P<timeout_count>\d+)\s+"
    r"improved_count=(?P<improved_count>\d+)"
)
_MISSING = object()


@dataclass
class RewardLogData:
    run_name: str
    sample_index: list[int]
    reward: list[float]
    reward_positive: list[float]
    built_ok: list[float]
    forward_shape_ok: list[float]
    backward_ok: list[float]
    loss_drop_ok: list[float]
    timed_out: list[float]
    train_acc: list[float]
    seed_accuracy_baseline: list[float]
    seed_train_acc_gap: list[float]
    group_baseline_train_acc: list[float]
    group_train_acc_gain: list[float]
    group_train_acc_improved: list[float]
    reward_batch_index: list[int]
    reward_group_id: list[int]
    group_warmup: list[float]
    loss_start: list[float]
    loss_end: list[float]
    loss_drop: list[float]
    val_metric: list[float]
    estimated_total_seconds: list[float]
    eval_limit_seconds: list[float]
    warmup_dense_reward: list[float]
    anti_collapse_trainable_ok: list[float]
    xml_tag_exact: list[float]
    dual_backbone_ok: list[float]
    format_violation: list[float]
    backbone_model_names: list[list[str]]
    progress_generation_total: list[int]
    progress_success_count: list[float]
    progress_success_rate: list[float]
    progress_warmup_trainable_count: list[float]
    progress_warmup_positive_count: list[float]
    progress_timeout_count: list[float]
    progress_improved_count: list[float]

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


def _optional_mapping(container: dict[str, Any], key: str, line_no: int) -> dict[str, Any] | None:
    if key not in container or container[key] is None:
        return None
    value = container[key]
    if not isinstance(value, dict):
        raise ValueError(f"Line {line_no}: field '{key}' must be a JSON object when present")
    return value


def _require_numeric(container: dict[str, Any], key: str, line_no: int) -> float:
    if key not in container:
        raise ValueError(f"Line {line_no}: missing required numeric field '{key}'")
    value = container[key]
    if not _is_number(value):
        raise ValueError(f"Line {line_no}: field '{key}' must be numeric")
    return float(value)


def _require_int(container: dict[str, Any], key: str, line_no: int) -> int:
    value = _require_numeric(container, key, line_no)
    if not float(value).is_integer():
        raise ValueError(f"Line {line_no}: field '{key}' must be an integer")
    return int(value)


def _require_optional_numeric(container: dict[str, Any], key: str, line_no: int) -> float:
    if key not in container:
        raise ValueError(f"Line {line_no}: missing required field '{key}'")
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


def _coerce_bool(value: Any, field_name: str, line_no: int) -> float:
    if not isinstance(value, bool):
        raise ValueError(f"Line {line_no}: field '{field_name}' must be boolean")
    return 1.0 if value else 0.0


def _bool_from_candidates(container: dict[str, Any], candidate_keys: Sequence[str], line_no: int) -> float | None:
    for candidate in candidate_keys:
        if candidate not in container or container[candidate] is None:
            continue
        return _coerce_bool(container[candidate], candidate, line_no)
    return None


def _has_numeric_candidate(container: dict[str, Any], candidate_keys: Sequence[str], line_no: int) -> bool:
    for candidate in candidate_keys:
        if candidate not in container or container[candidate] is None:
            continue
        value = container[candidate]
        if not _is_number(value):
            raise ValueError(f"Line {line_no}: field '{candidate}' must be numeric or null when present")
        return True
    return False


def _require_bool_alias(
    container: dict[str, Any],
    key: str,
    aliases: Sequence[str],
    line_no: int,
    *,
    default: object = _MISSING,
) -> float:
    candidate_keys = (key, *aliases)
    for candidate in candidate_keys:
        if candidate not in container:
            continue
        return _coerce_bool(container[candidate], candidate, line_no)
    if default is not _MISSING:
        return 1.0 if bool(default) else 0.0
    alias_text = ", ".join(repr(alias) for alias in aliases)
    raise ValueError(f"Line {line_no}: missing required boolean field '{key}' (accepted aliases: {alias_text})")


def _optional_numeric(container: dict[str, Any] | None, key: str, line_no: int) -> float:
    if container is None or key not in container or container[key] is None:
        return float("nan")
    value = container[key]
    if not _is_number(value):
        raise ValueError(f"Line {line_no}: field '{key}' must be numeric or null when present")
    return float(value)


def _optional_numeric_alias(container: dict[str, Any], key: str, aliases: Sequence[str], line_no: int) -> float:
    for candidate in (key, *aliases):
        if candidate not in container or container[candidate] is None:
            continue
        value = container[candidate]
        if not _is_number(value):
            raise ValueError(f"Line {line_no}: field '{candidate}' must be numeric or null when present")
        return float(value)
    return float("nan")


def _optional_bool(container: dict[str, Any] | None, key: str, line_no: int) -> float:
    if container is None or key not in container or container[key] is None:
        return float("nan")
    value = container[key]
    if not isinstance(value, bool):
        raise ValueError(f"Line {line_no}: field '{key}' must be boolean when present")
    return 1.0 if value else 0.0


def _optional_int(container: dict[str, Any], key: str, line_no: int, *, default: int = 0) -> int:
    if key not in container or container[key] is None:
        return default
    value = container[key]
    if not _is_number(value):
        raise ValueError(f"Line {line_no}: field '{key}' must be an integer when present")
    numeric = float(value)
    if not numeric.is_integer():
        raise ValueError(f"Line {line_no}: field '{key}' must be an integer")
    return int(numeric)


def _optional_string_list(container: dict[str, Any], key: str, line_no: int) -> list[str]:
    if key not in container or container[key] is None:
        return []
    value = container[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Line {line_no}: field '{key}' must be a list of strings when present")
    return list(value)


def _compat_loss_drop_ok(api_result: dict[str, Any], line_no: int) -> float:
    if "loss_drop_ok" in api_result:
        return _coerce_bool(api_result["loss_drop_ok"], "loss_drop_ok", line_no)

    loss_start = _optional_numeric_alias(api_result, "loss_start", (), line_no)
    loss_end = _optional_numeric_alias(api_result, "loss_end", (), line_no)
    loss_drop = _optional_numeric_alias(api_result, "loss_drop", (), line_no)

    if not math.isnan(loss_start) and not math.isnan(loss_end):
        rel_drop_ok = loss_start > 0.0 and (loss_end <= loss_start * 0.98)
        return 1.0 if (loss_end < (loss_start - 1e-3) or rel_drop_ok) else 0.0
    if not math.isnan(loss_drop):
        return 1.0 if loss_drop > 1e-3 else 0.0
    return 0.0


def _compat_built_ok(api_result: dict[str, Any], line_no: int) -> float:
    explicit = _bool_from_candidates(api_result, ("built_ok",), line_no)
    if explicit is not None:
        return explicit
    if _bool_from_candidates(api_result, ("forward_shape_ok", "forward_ok", "trained_step_ok", "backward_ok"), line_no) == 1.0:
        return 1.0
    if _has_numeric_candidate(api_result, ("loss_start", "loss_end", "loss_drop", "train_acc", "val_metric", "latency_ms", "params_m"), line_no):
        return 1.0
    return 0.0


def _compat_forward_shape_ok(api_result: dict[str, Any], line_no: int) -> float:
    explicit = _bool_from_candidates(api_result, ("forward_shape_ok", "forward_ok"), line_no)
    if explicit is not None:
        return explicit
    if _bool_from_candidates(api_result, ("trained_step_ok", "backward_ok"), line_no) == 1.0:
        return 1.0
    if _has_numeric_candidate(api_result, ("loss_start", "loss_end", "loss_drop", "train_acc", "val_metric"), line_no):
        return 1.0
    built_ok = _bool_from_candidates(api_result, ("built_ok",), line_no)
    if built_ok == 0.0:
        return 0.0
    return 0.0


def _compat_backward_ok(api_result: dict[str, Any], line_no: int) -> float:
    explicit = _bool_from_candidates(api_result, ("backward_ok", "trained_step_ok"), line_no)
    if explicit is not None:
        return explicit
    if _has_numeric_candidate(api_result, ("loss_start", "loss_end", "loss_drop", "train_acc", "val_metric"), line_no):
        return 1.0
    return 0.0


def _compat_group_improved(api_result: dict[str, Any], line_no: int) -> float:
    if "group_train_acc_improved" in api_result:
        return _coerce_bool(api_result["group_train_acc_improved"], "group_train_acc_improved", line_no)
    group_gain = _optional_numeric_alias(api_result, "group_train_acc_gain", (), line_no)
    if math.isnan(group_gain):
        return 0.0
    return 1.0 if group_gain > 0.0 else 0.0


def _compat_group_warmup(api_result: dict[str, Any], line_no: int, reward_group_id_value: int) -> float:
    explicit = _bool_from_candidates(api_result, ("group_warmup",), line_no)
    if explicit is not None:
        return explicit
    return 1.0 if reward_group_id_value == 0 else 0.0


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


def _format_count(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}"


def _last_or_nan(values: Sequence[float] | Sequence[int]) -> float:
    if not values:
        return float("nan")
    return float(values[-1])


def _load_training_progress(log_dir: Path) -> dict[str, list[float] | list[int]]:
    progress_file = log_dir / "training_progress.log"
    series = {
        "generation_total": [],
        "success_count": [],
        "success_rate": [],
        "warmup_trainable_count": [],
        "warmup_positive_count": [],
        "timeout_count": [],
        "improved_count": [],
    }
    if not progress_file.exists():
        return series

    with progress_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            match = PROGRESS_PATTERN.search(raw_line)
            if not match:
                continue
            series["generation_total"].append(int(match.group("generation_total")))
            series["success_count"].append(float(match.group("success_count")))
            series["success_rate"].append(float(match.group("success_rate")) / 100.0)
            series["warmup_trainable_count"].append(float(match.group("warmup_trainable_count")))
            series["warmup_positive_count"].append(float(match.group("warmup_positive_count")))
            series["timeout_count"].append(float(match.group("timeout_count")))
            series["improved_count"].append(float(match.group("improved_count")))
    return series


def load_reward_log(log_dir: Path) -> RewardLogData:
    log_dir = Path(log_dir)
    log_file = log_dir / "generation_samples.jsonl"
    if not log_file.exists():
        raise FileNotFoundError(f"Required log file not found: {log_file}")

    reward: list[float] = []
    reward_positive: list[float] = []
    built_ok: list[float] = []
    forward_shape_ok: list[float] = []
    backward_ok: list[float] = []
    loss_drop_ok: list[float] = []
    timed_out: list[float] = []
    train_acc: list[float] = []
    seed_accuracy_baseline: list[float] = []
    seed_train_acc_gap: list[float] = []
    group_baseline_train_acc: list[float] = []
    group_train_acc_gain: list[float] = []
    group_train_acc_improved: list[float] = []
    reward_batch_index: list[int] = []
    reward_group_id: list[int] = []
    group_warmup: list[float] = []
    loss_start: list[float] = []
    loss_end: list[float] = []
    loss_drop: list[float] = []
    val_metric: list[float] = []
    estimated_total_seconds: list[float] = []
    eval_limit_seconds: list[float] = []
    warmup_dense_reward: list[float] = []
    anti_collapse_trainable_ok: list[float] = []
    xml_tag_exact: list[float] = []
    dual_backbone_ok: list[float] = []
    format_violation: list[float] = []
    backbone_model_names: list[list[str]] = []

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
            raw_extraction = _optional_mapping(api_result, "raw_extraction", line_no)
            anti_collapse = _optional_mapping(api_result, "anti_collapse", line_no)
            reward_batch_index_value = _optional_int(api_result, "reward_batch_index", line_no, default=0)
            reward_group_id_value = _optional_int(api_result, "reward_group_id", line_no, default=0)

            reward.append(reward_value)
            reward_positive.append(1.0 if reward_value > 0.0 else 0.0)
            built_ok.append(_compat_built_ok(api_result, line_no))
            forward_shape_ok.append(_compat_forward_shape_ok(api_result, line_no))
            backward_ok.append(_compat_backward_ok(api_result, line_no))
            loss_drop_ok.append(_compat_loss_drop_ok(api_result, line_no))
            timed_out.append(_require_bool_alias(api_result, "timed_out", (), line_no, default=False))
            train_acc.append(_optional_numeric_alias(api_result, "train_acc", (), line_no))
            seed_accuracy_baseline.append(_optional_numeric_alias(api_result, "seed_accuracy_baseline", ("accuracy_baseline",), line_no))
            seed_train_acc_gap.append(_optional_numeric_alias(api_result, "seed_train_acc_gap", ("train_acc_gain",), line_no))
            group_baseline_train_acc.append(_optional_numeric_alias(api_result, "group_baseline_train_acc", (), line_no))
            group_train_acc_gain.append(_optional_numeric_alias(api_result, "group_train_acc_gain", (), line_no))
            group_train_acc_improved.append(_compat_group_improved(api_result, line_no))
            reward_batch_index.append(reward_batch_index_value)
            reward_group_id.append(reward_group_id_value)
            group_warmup.append(_compat_group_warmup(api_result, line_no, reward_group_id_value))
            loss_start.append(_optional_numeric_alias(api_result, "loss_start", (), line_no))
            loss_end.append(_optional_numeric_alias(api_result, "loss_end", (), line_no))
            loss_drop.append(_optional_numeric_alias(api_result, "loss_drop", (), line_no))
            val_metric.append(_optional_numeric_alias(api_result, "val_metric", (), line_no))
            estimated_total_seconds.append(_optional_numeric_alias(api_result, "estimated_total_seconds", (), line_no))
            eval_limit_seconds.append(_optional_numeric_alias(api_result, "eval_limit_seconds", (), line_no))
            warmup_dense_reward.append(_optional_numeric_alias(api_result, "warmup_dense_reward", (), line_no))
            anti_collapse_trainable_ok.append(_optional_bool(anti_collapse, "trainable_ok", line_no))
            xml_tag_exact.append(_optional_bool(raw_extraction, "xml_tag_exact", line_no))
            dual_backbone_ok.append(_optional_bool(raw_extraction, "dual_backbone_ok", line_no))
            backbone_model_names.append(_optional_string_list(api_result, "backbone_model_names", line_no))

            class_count = _optional_numeric(raw_extraction, "class_count", line_no)
            import_count = _optional_numeric(raw_extraction, "import_count", line_no)
            bad_signature_count = _optional_numeric(raw_extraction, "bad_signature_count", line_no)
            if any(math.isnan(value) for value in (class_count, import_count, bad_signature_count)):
                format_violation.append(float("nan"))
            else:
                format_violation.append(
                    1.0 if (class_count > 0 or import_count > 0 or bad_signature_count > 0) else 0.0
                )

    if not reward:
        raise ValueError(f"No samples found in {log_file}")

    progress = _load_training_progress(log_dir)

    return RewardLogData(
        run_name=log_dir.name or str(log_dir),
        sample_index=list(range(1, len(reward) + 1)),
        reward=reward,
        reward_positive=reward_positive,
        built_ok=built_ok,
        forward_shape_ok=forward_shape_ok,
        backward_ok=backward_ok,
        loss_drop_ok=loss_drop_ok,
        timed_out=timed_out,
        train_acc=train_acc,
        seed_accuracy_baseline=seed_accuracy_baseline,
        seed_train_acc_gap=seed_train_acc_gap,
        group_baseline_train_acc=group_baseline_train_acc,
        group_train_acc_gain=group_train_acc_gain,
        group_train_acc_improved=group_train_acc_improved,
        reward_batch_index=reward_batch_index,
        reward_group_id=reward_group_id,
        group_warmup=group_warmup,
        loss_start=loss_start,
        loss_end=loss_end,
        loss_drop=loss_drop,
        val_metric=val_metric,
        estimated_total_seconds=estimated_total_seconds,
        eval_limit_seconds=eval_limit_seconds,
        warmup_dense_reward=warmup_dense_reward,
        anti_collapse_trainable_ok=anti_collapse_trainable_ok,
        xml_tag_exact=xml_tag_exact,
        dual_backbone_ok=dual_backbone_ok,
        format_violation=format_violation,
        backbone_model_names=backbone_model_names,
        progress_generation_total=list(progress["generation_total"]),
        progress_success_count=list(progress["success_count"]),
        progress_success_rate=list(progress["success_rate"]),
        progress_warmup_trainable_count=list(progress["warmup_trainable_count"]),
        progress_warmup_positive_count=list(progress["warmup_positive_count"]),
        progress_timeout_count=list(progress["timeout_count"]),
        progress_improved_count=list(progress["improved_count"]),
    )


def compute_summary(data: RewardLogData, window: int) -> dict[str, float]:
    reward_rolling = rolling_nanmean(data.reward, window)
    warmup_mask = [flag > 0.5 for flag in data.group_warmup]
    warmup_count = sum(1 for flag in warmup_mask if flag)
    warmup_trainable_count = sum(
        1
        for idx, is_warmup in enumerate(warmup_mask)
        if is_warmup
        and data.built_ok[idx] > 0.5
        and data.forward_shape_ok[idx] > 0.5
        and data.backward_ok[idx] > 0.5
        and data.loss_drop_ok[idx] > 0.5
    )
    warmup_positive_count = sum(
        1
        for idx, is_warmup in enumerate(warmup_mask)
        if is_warmup and data.reward_positive[idx] > 0.5
    )
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
        "timed_out_rate": _mean(data.timed_out),
        "train_acc_mean": _nanmean(data.train_acc),
        "seed_accuracy_baseline_mean": _nanmean(data.seed_accuracy_baseline),
        "seed_train_acc_gap_mean": _nanmean(data.seed_train_acc_gap),
        "group_baseline_train_acc_mean": _nanmean(data.group_baseline_train_acc),
        "group_train_acc_gain_mean": _nanmean(data.group_train_acc_gain),
        "group_train_acc_gain_max": max(value for value in data.group_train_acc_gain if not math.isnan(value))
        if any(not math.isnan(value) for value in data.group_train_acc_gain)
        else float("nan"),
        "group_train_acc_improved_rate": _mean(data.group_train_acc_improved),
        "warmup_dense_reward_mean": _nanmean(data.warmup_dense_reward),
        "estimated_total_seconds_mean": _nanmean(data.estimated_total_seconds),
        "warmup_trainable_rate": (warmup_trainable_count / warmup_count) if warmup_count > 0 else float("nan"),
        "warmup_positive_rate": (warmup_positive_count / warmup_trainable_count) if warmup_trainable_count > 0 else float("nan"),
        "anti_collapse_trainable_ok_rate": _nanmean(data.anti_collapse_trainable_ok),
        "xml_tag_exact_rate": _nanmean(data.xml_tag_exact),
        "dual_backbone_ok_rate": _nanmean(data.dual_backbone_ok),
        "format_violation_rate": _nanmean(data.format_violation),
        "progress_latest_total": _last_or_nan(data.progress_generation_total),
        "progress_latest_success_count": _last_or_nan(data.progress_success_count),
        "progress_latest_success_rate": _last_or_nan(data.progress_success_rate),
        "progress_latest_warmup_trainable_count": _last_or_nan(data.progress_warmup_trainable_count),
        "progress_latest_warmup_positive_count": _last_or_nan(data.progress_warmup_positive_count),
        "progress_latest_timeout_count": _last_or_nan(data.progress_timeout_count),
        "progress_latest_improved_count": _last_or_nan(data.progress_improved_count),
    }


def _style_axis(ax: Any, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Sample Index", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(fontsize=8, framealpha=0.9)


def _style_progress_axis(ax: Any, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Generation Count", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(fontsize=8, framealpha=0.9)


def _mark_group_regions(axes: Sequence[Any], data: RewardLogData) -> None:
    if not data.sample_index:
        return

    warmup_indices = [idx for idx, flag in zip(data.sample_index, data.group_warmup) if flag > 0.5]
    if warmup_indices:
        warmup_start = warmup_indices[0] - 0.5
        warmup_end = warmup_indices[-1] + 0.5
        for ax in axes:
            if hasattr(ax, "axvspan"):
                ax.axvspan(warmup_start, warmup_end, color="#FFF8E1", alpha=0.35)

    for sample_idx, prev_group, next_group in zip(
        data.sample_index[1:],
        data.reward_group_id[:-1],
        data.reward_group_id[1:],
    ):
        if prev_group == next_group:
            continue
        boundary = sample_idx - 0.5
        for ax in axes:
            if hasattr(ax, "axvline"):
                ax.axvline(boundary, color="#9E9E9E", linestyle=":", linewidth=0.9, alpha=0.8)


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
    timeout_rate = rolling_nanmean(data.timed_out, window)

    train_acc_rolling = rolling_nanmean(data.train_acc, window)
    group_baseline_rolling = rolling_nanmean(data.group_baseline_train_acc, window)
    group_gain_rolling = rolling_nanmean(data.group_train_acc_gain, window)
    group_improved_rate = rolling_nanmean(data.group_train_acc_improved, window)

    seed_accuracy_rolling = rolling_nanmean(data.seed_accuracy_baseline, window)
    seed_gap_rolling = rolling_nanmean(data.seed_train_acc_gap, window)
    warmup_dense_rolling = rolling_nanmean(data.warmup_dense_reward, window)
    estimated_total_rolling = rolling_nanmean(data.estimated_total_seconds, window)

    anti_collapse_rate = rolling_nanmean(data.anti_collapse_trainable_ok, window)
    xml_exact_rate = rolling_nanmean(data.xml_tag_exact, window)
    dual_backbone_rate = rolling_nanmean(data.dual_backbone_ok, window)
    violation_rate = rolling_nanmean(data.format_violation, window)

    figure, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=160)
    axes = axes.flatten() if hasattr(axes, "flatten") else axes
    _mark_group_regions(axes[:5], data)

    axes[0].plot(data.sample_index, data.reward, color="#B0BEC5", linewidth=1.1, alpha=0.55, label="raw reward")
    axes[0].plot(data.sample_index, reward_rolling, color="#1565C0", linewidth=2.2, label=f"rolling mean ({window})")
    axes[0].axhline(0.0, color="#424242", linestyle="--", linewidth=1.0, label="zero reward")
    _style_axis(axes[0], "Reward Trend", "Reward")

    axes[1].plot(data.sample_index, built_rate, color="#2E7D32", linewidth=2.0, label="built_ok")
    axes[1].plot(data.sample_index, shape_rate, color="#00897B", linewidth=2.0, label="forward_shape_ok")
    axes[1].plot(data.sample_index, backward_rate, color="#6A1B9A", linewidth=2.0, label="backward_ok")
    axes[1].plot(data.sample_index, loss_drop_rate, color="#EF6C00", linewidth=2.0, label="loss_drop_ok")
    axes[1].plot(data.sample_index, timeout_rate, color="#C62828", linewidth=2.0, label="timed_out")
    axes[1].set_ylim(0.0, 1.0)
    _style_axis(axes[1], "Trainability Gates", "Rolling Pass Rate")

    axes[2].plot(data.sample_index, train_acc_rolling, color="#1565C0", linewidth=2.0, label="train_acc")
    axes[2].plot(data.sample_index, group_baseline_rolling, color="#6D4C41", linewidth=2.0, label="group_baseline_train_acc")
    axes[2].plot(data.sample_index, group_gain_rolling, color="#00897B", linewidth=2.0, label="group_train_acc_gain")
    axes[2].plot(data.sample_index, group_improved_rate, color="#EF6C00", linewidth=2.0, label="group_train_acc_improved")
    _style_axis(axes[2], "Train Accuracy & Group Gain", "Accuracy / Gain / Rate")

    axes[3].plot(data.sample_index, seed_accuracy_rolling, color="#5D4037", linewidth=2.0, label="seed_accuracy_baseline")
    axes[3].plot(data.sample_index, seed_gap_rolling, color="#00838F", linewidth=2.0, label="seed_train_acc_gap")
    axes[3].plot(data.sample_index, warmup_dense_rolling, color="#F9A825", linewidth=2.0, label="warmup_dense_reward")
    axes[3].plot(data.sample_index, estimated_total_rolling, color="#8E24AA", linewidth=2.0, label="estimated_total_seconds")
    _style_axis(axes[3], "Warmup / Seed / Budget", "Value / Seconds")

    axes[4].plot(data.sample_index, anti_collapse_rate, color="#283593", linewidth=2.0, label="anti_collapse.trainable_ok")
    axes[4].plot(data.sample_index, xml_exact_rate, color="#004D40", linewidth=2.0, label="xml_tag_exact")
    axes[4].plot(data.sample_index, dual_backbone_rate, color="#37474F", linewidth=2.0, label="dual_backbone_ok")
    axes[4].plot(data.sample_index, violation_rate, color="#D84315", linewidth=2.0, label="format_violation")
    axes[4].set_ylim(0.0, 1.0)
    _style_axis(axes[4], "Quality / Constraint", "Rolling Rate")

    if data.progress_generation_total:
        axes[5].plot(
            data.progress_generation_total,
            data.progress_warmup_trainable_count,
            color="#1565C0",
            linewidth=2.0,
            label="warmup_trainable_count",
        )
        axes[5].plot(
            data.progress_generation_total,
            data.progress_warmup_positive_count,
            color="#F9A825",
            linewidth=2.0,
            label="warmup_positive_count",
        )
        axes[5].plot(
            data.progress_generation_total,
            data.progress_timeout_count,
            color="#C62828",
            linewidth=2.0,
            label="timeout_count",
        )
        axes[5].plot(
            data.progress_generation_total,
            data.progress_improved_count,
            color="#2E7D32",
            linewidth=2.0,
            label="improved_count",
        )
    else:
        axes[5].plot([], [], color="#78909C", linewidth=2.0, label="training_progress.log unavailable")
    _style_progress_axis(axes[5], "Training Progress", "Count")

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
        "Trainability: "
        f"built={_format_metric(summary['built_ok_rate'], percent=True)}, "
        f"shape={_format_metric(summary['forward_shape_ok_rate'], percent=True)}, "
        f"backward={_format_metric(summary['backward_ok_rate'], percent=True)}, "
        f"loss_drop={_format_metric(summary['loss_drop_ok_rate'], percent=True)}, "
        f"timed_out={_format_metric(summary['timed_out_rate'], percent=True)}"
    )
    print(
        "Train accuracy: "
        f"train_mean={_format_metric(summary['train_acc_mean'], percent=True)}, "
        f"group_baseline_mean={_format_metric(summary['group_baseline_train_acc_mean'], percent=True)}, "
        f"group_gain_mean={_format_metric(summary['group_train_acc_gain_mean'], percent=True)}, "
        f"group_gain_max={_format_metric(summary['group_train_acc_gain_max'], percent=True)}, "
        f"group_improved_rate={_format_metric(summary['group_train_acc_improved_rate'], percent=True)}"
    )
    print(
        "Warmup / budget: "
        f"seed_accuracy_mean={_format_metric(summary['seed_accuracy_baseline_mean'], percent=True)}, "
        f"seed_gap_mean={_format_metric(summary['seed_train_acc_gap_mean'], percent=True)}, "
        f"warmup_dense_mean={_format_metric(summary['warmup_dense_reward_mean'])}, "
        f"warmup_trainable_rate={_format_metric(summary['warmup_trainable_rate'], percent=True)}, "
        f"warmup_positive_rate={_format_metric(summary['warmup_positive_rate'], percent=True)}, "
        f"estimated_total_seconds_mean={_format_metric(summary['estimated_total_seconds_mean'])}"
    )
    print(
        "Quality: "
        f"anti_collapse_trainable_ok={_format_metric(summary['anti_collapse_trainable_ok_rate'], percent=True)}, "
        f"xml_exact={_format_metric(summary['xml_tag_exact_rate'], percent=True)}, "
        f"dual_backbone_ok={_format_metric(summary['dual_backbone_ok_rate'], percent=True)}, "
        f"format_violation={_format_metric(summary['format_violation_rate'], percent=True)}"
    )
    print(
        "Training progress: "
        f"total={_format_count(summary['progress_latest_total'])}, "
        f"success={_format_count(summary['progress_latest_success_count'])}, "
        f"success_rate={_format_metric(summary['progress_latest_success_rate'], percent=True)}, "
        f"warmup_trainable={_format_count(summary['progress_latest_warmup_trainable_count'])}, "
        f"warmup_positive={_format_count(summary['progress_latest_warmup_positive_count'])}, "
        f"timeout={_format_count(summary['progress_latest_timeout_count'])}, "
        f"improved={_format_count(summary['progress_latest_improved_count'])}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a single-run SFT RL reward diagnostics dashboard")
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
