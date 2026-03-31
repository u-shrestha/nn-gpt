import ast
import csv
import subprocess
import sys
import threading
import warnings


_RL_FILTERED_LOG_PATTERNS = (
    "Skipping import of cpp extensions due to incompatible torch version",
    "github.com/pytorch/ao/issues/2919",
)


class _RLFilteredStream:
    def __init__(self, wrapped) -> None:
        self._wrapped = wrapped
        self._buffer = ""

    def write(self, text):
        text = str(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._write_line(line + "\n")
        return len(text)

    def flush(self):
        if self._buffer:
            self._write_line(self._buffer)
            self._buffer = ""
        return self._wrapped.flush()

    def _write_line(self, text: str) -> None:
        normalized = " ".join(text.split())
        if all(pattern in normalized for pattern in _RL_FILTERED_LOG_PATTERNS):
            return
        self._wrapped.write(text)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def _install_rl_runtime_noise_filters() -> None:
    if getattr(_install_rl_runtime_noise_filters, "_installed", False):
        return
    warnings.filterwarnings(
        "ignore",
        message=r".*Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers\. Use `HF_HOME` instead\..*",
        category=FutureWarning,
    )
    sys.stdout = _RLFilteredStream(sys.stdout)
    sys.stderr = _RLFilteredStream(sys.stderr)
    _install_rl_runtime_noise_filters._installed = True


_install_rl_runtime_noise_filters()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig
from datasets import Dataset
import ab.gpt.util.SFTUtil as SFTUtil
from ab.gpt.util.ArchDiscovery import (
    ensure_pattern_name,
    extract_graph_info,
    normalize_pattern_name,
)
from ab.gpt.util.Util import extract_str
from ab.gpt.util.Const import conf_train_dir, conf_test_dir, epoch_dir, new_nn_file, synth_dir, new_out_file
from ab.nn.util.Util import create_file
from ab.gpt.util.Reward import (
    PersistentEvalWorkerError,
    evaluate_code_and_reward,
    evaluate_code_and_reward_batch,
    get_eval_worker_diagnostics,
    shutdown_eval_worker,
)
import ab.nn.api as api

import os
import re
import textwrap
import shutil
import json
from pathlib import Path
from dataclasses import dataclass, asdict

from ab.gpt.util.simple_logger import SimpleCodeLogger
from typing import Tuple, Any, List, Dict, Optional, Set
from collections import Counter

# Open-architecture archives are keyed by canonical graph structure, not prompt labels.
graph_archive_counts = Counter()
family_archive_counts = Counter()
family_hash_archive_counts = Counter()
family_metric_best: Dict[str, float] = {}
motif_name_counts = Counter()
saved_graph_counts = Counter()
saved_family_hash_counts = Counter()
goal_graph_archive_counts: Dict[str, Counter] = {}
goal_family_hash_archive_counts: Dict[str, Counter] = {}
saved_goal_family_hash_counts: Dict[str, Counter] = {}
train_graph_hashes: Set[str] = set()
train_family_hashes: Set[str] = set()
train_descriptor_keys: Set[str] = set()
train_reference_stats: Dict[str, int] = {}

# ===== Configuration Options =====
base_model = "ABrain/NNGPT-Backbone-deepseek-coder-6.7b-instruct" # 使用新的 Backbone 模型
LOAD_EXISTING_MODEL = False  # Model is already merged
SAVED_MODEL_PATH = "rl_backbone_model" 
B_index = 0
GROUP_BATCH_SIZE = 20
GROUP_IMPROVEMENT_DELTA = 0.003
BEST_GROUP_REFRESH_DELTA = 0.0015
GOAL_REFRESH_DELTA = 0.0015
NON_IMPROVING_REWARD_CAP = 0.04
BATCH_ELITE_SOFT_BONUSES = (0.10, 0.07, 0.05, 0.03, 0.02)
BATCH_ELITE_IMPROVING_BONUSES = (0.18, 0.13, 0.09, 0.06, 0.04)
STRUCTURE_MACRO_BONUS = 0.04
STRUCTURE_MULTI_STAGE_BONUS = 0.03
STRUCTURE_MOTIF_BONUS = 0.02
STRUCTURE_BATCH_DIVERSITY_BONUS = 0.03
STRUCTURE_NON_DOMINANT_FAMILY_BONUS = 0.02
STRUCTURE_ARCHIVE_RARITY_STRONG_BONUS = 0.03
STRUCTURE_ARCHIVE_RARITY_MEDIUM_BONUS = 0.02
STRUCTURE_ARCHIVE_RARITY_LIGHT_BONUS = 0.01
REPEAT_FAMILY_PENALTY = -0.10
PLAIN_FUSE_PENALTY = -0.10
NO_PROGRESS_PENALTY = -0.06
GOAL_REFRESH_BONUS = 0.30
GOAL_MATCH_REWARD_SCALE = 0.12
TRAINSET_NOVEL_FAMILY_BONUS = 0.04
TRAINSET_NOVEL_GRAPH_BONUS = 0.02
GENERALIZATION_GAP_TOLERANCE = 0.02
GENERALIZATION_PENALTY_SCALE = 2.0
GENERALIZATION_PENALTY_CAP = -0.20
REWARD_TARGET_METRIC = "frozen_test_acc"
FEEDBACK_GRAPH_EXPR_MAX_CHARS = 160
FEEDBACK_SUMMARY_MAX_CHARS = 240
FEEDBACK_SUMMARY_LIMIT = 2
reward_batch_index = 0
current_group_id = 0
current_group_reward_target_sum = 0.0
current_group_reward_target_count = 0
current_group_frozen_train_acc_sum = 0.0
current_group_frozen_train_acc_count = 0
current_group_frozen_test_acc_sum = 0.0
current_group_frozen_test_acc_count = 0
current_group_unfrozen_train_acc_sum = 0.0
current_group_unfrozen_train_acc_count = 0
current_group_unfrozen_test_acc_sum = 0.0
current_group_unfrozen_test_acc_count = 0
prev_closed_group_mean_reward_target_acc: Optional[float] = None
best_closed_group_mean_reward_target_acc: Optional[float] = None
prev_closed_group_train_acc_mean: Optional[float] = None
best_closed_group_mean_train_acc: Optional[float] = None
prev_closed_group_mean_test_acc: Optional[float] = None
best_closed_group_mean_test_acc: Optional[float] = None
best_closed_group_id: Optional[int] = None
best_reward_target_by_goal: Dict[str, float] = {}
dominant_family_hash: Optional[str] = None
dominant_family_share: float = 0.0
prev_group_feedback: List["GroupFeedbackSummary"] = []
best_group_feedback: List["GroupFeedbackSummary"] = []
current_group_top_feedback: List["GroupFeedbackSummary"] = []
current_group_goal_best_candidates: Dict[str, float] = {}
# ==================================


class NullCodeLogger:
    def log_to_file(self, message: str) -> None:
        return

    def log_generation(self, prompt: str, completion: str, reward: float, api_result: Any = None) -> None:
        return

    def save_log(self) -> None:
        return


code_logger: Any = NullCodeLogger()

SHALLOW_COLLAPSE_FAMILIES = {
    "ParallelTriple_Shallow",
    "DualBackboneFuse_Shallow",
    "TripleBackboneFuse_Shallow",
}


@dataclass
class GroupFeedbackSummary:
    goal_key: str
    pattern_name: str
    graph_expr_short: str
    reward_target_value: float
    frozen_train_acc: float
    frozen_test_acc: float
    unfrozen_train_acc: Optional[float]
    unfrozen_test_acc: Optional[float]
    backbone_model_names: List[str]
    stats_short: str
    summary: str
    family_hash: str
    signature: str
    reward_group_id: int


def _counter_payload(counter: Counter) -> Dict[str, int]:
    return {str(key): int(value) for key, value in counter.items()}


def _nested_counter_payload(mapping: Dict[str, Counter]) -> Dict[str, Dict[str, int]]:
    return {
        str(key): _counter_payload(counter)
        for key, counter in mapping.items()
    }


def _restore_counter(counter: Counter, payload: Optional[Dict[str, int]]) -> None:
    counter.clear()
    if payload:
        counter.update({str(key): int(value) for key, value in payload.items()})


def _restore_nested_counters(target: Dict[str, Counter], payload: Optional[Dict[str, Dict[str, int]]]) -> None:
    target.clear()
    for key, counter_payload in (payload or {}).items():
        counter = Counter()
        counter.update({str(inner_key): int(value) for inner_key, value in (counter_payload or {}).items()})
        target[str(key)] = counter


def _feedback_summaries_from_payload(items: Optional[List[Dict[str, Any]]]) -> List["GroupFeedbackSummary"]:
    return [GroupFeedbackSummary(**dict(item)) for item in (items or [])]


def capture_reward_runtime_state() -> Dict[str, Any]:
    return {
        "B_index": B_index,
        "reward_batch_index": reward_batch_index,
        "current_group_id": current_group_id,
        "current_group_reward_target_sum": current_group_reward_target_sum,
        "current_group_reward_target_count": current_group_reward_target_count,
        "current_group_frozen_train_acc_sum": current_group_frozen_train_acc_sum,
        "current_group_frozen_train_acc_count": current_group_frozen_train_acc_count,
        "current_group_frozen_test_acc_sum": current_group_frozen_test_acc_sum,
        "current_group_frozen_test_acc_count": current_group_frozen_test_acc_count,
        "current_group_unfrozen_train_acc_sum": current_group_unfrozen_train_acc_sum,
        "current_group_unfrozen_train_acc_count": current_group_unfrozen_train_acc_count,
        "current_group_unfrozen_test_acc_sum": current_group_unfrozen_test_acc_sum,
        "current_group_unfrozen_test_acc_count": current_group_unfrozen_test_acc_count,
        "prev_closed_group_mean_reward_target_acc": prev_closed_group_mean_reward_target_acc,
        "best_closed_group_mean_reward_target_acc": best_closed_group_mean_reward_target_acc,
        "prev_closed_group_train_acc_mean": prev_closed_group_train_acc_mean,
        "best_closed_group_mean_train_acc": best_closed_group_mean_train_acc,
        "prev_closed_group_mean_test_acc": prev_closed_group_mean_test_acc,
        "best_closed_group_mean_test_acc": best_closed_group_mean_test_acc,
        "best_closed_group_id": best_closed_group_id,
        "best_reward_target_by_goal": {
            str(key): float(value)
            for key, value in best_reward_target_by_goal.items()
        },
        "dominant_family_hash": dominant_family_hash,
        "dominant_family_share": dominant_family_share,
        "graph_archive_counts": _counter_payload(graph_archive_counts),
        "family_archive_counts": _counter_payload(family_archive_counts),
        "family_hash_archive_counts": _counter_payload(family_hash_archive_counts),
        "family_metric_best": {
            str(key): float(value)
            for key, value in family_metric_best.items()
        },
        "motif_name_counts": _counter_payload(motif_name_counts),
        "saved_graph_counts": _counter_payload(saved_graph_counts),
        "saved_family_hash_counts": _counter_payload(saved_family_hash_counts),
        "goal_graph_archive_counts": _nested_counter_payload(goal_graph_archive_counts),
        "goal_family_hash_archive_counts": _nested_counter_payload(goal_family_hash_archive_counts),
        "saved_goal_family_hash_counts": _nested_counter_payload(saved_goal_family_hash_counts),
        "prev_group_feedback": _feedback_summary_payload(prev_group_feedback),
        "best_group_feedback": _feedback_summary_payload(best_group_feedback),
        "current_group_top_feedback": _current_group_top_feedback_payload(),
        "current_group_goal_best_candidates": {
            str(key): float(value)
            for key, value in current_group_goal_best_candidates.items()
        },
    }


def restore_reward_runtime_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    scalar_defaults = {
        "B_index": 0,
        "reward_batch_index": 0,
        "current_group_id": 0,
        "current_group_reward_target_sum": 0.0,
        "current_group_reward_target_count": 0,
        "current_group_frozen_train_acc_sum": 0.0,
        "current_group_frozen_train_acc_count": 0,
        "current_group_frozen_test_acc_sum": 0.0,
        "current_group_frozen_test_acc_count": 0,
        "current_group_unfrozen_train_acc_sum": 0.0,
        "current_group_unfrozen_train_acc_count": 0,
        "current_group_unfrozen_test_acc_sum": 0.0,
        "current_group_unfrozen_test_acc_count": 0,
        "prev_closed_group_mean_reward_target_acc": None,
        "best_closed_group_mean_reward_target_acc": None,
        "prev_closed_group_train_acc_mean": None,
        "best_closed_group_mean_train_acc": None,
        "prev_closed_group_mean_test_acc": None,
        "best_closed_group_mean_test_acc": None,
        "best_closed_group_id": None,
        "dominant_family_hash": None,
        "dominant_family_share": 0.0,
    }
    for name, default_value in scalar_defaults.items():
        globals()[name] = state.get(name, default_value)

    _restore_counter(graph_archive_counts, state.get("graph_archive_counts"))
    _restore_counter(family_archive_counts, state.get("family_archive_counts"))
    _restore_counter(family_hash_archive_counts, state.get("family_hash_archive_counts"))
    family_metric_best.clear()
    family_metric_best.update(
        {
            str(key): float(value)
            for key, value in (state.get("family_metric_best") or {}).items()
        }
    )
    _restore_counter(motif_name_counts, state.get("motif_name_counts"))
    _restore_counter(saved_graph_counts, state.get("saved_graph_counts"))
    _restore_counter(saved_family_hash_counts, state.get("saved_family_hash_counts"))
    _restore_nested_counters(goal_graph_archive_counts, state.get("goal_graph_archive_counts"))
    _restore_nested_counters(goal_family_hash_archive_counts, state.get("goal_family_hash_archive_counts"))
    _restore_nested_counters(saved_goal_family_hash_counts, state.get("saved_goal_family_hash_counts"))

    best_reward_target_by_goal.clear()
    best_reward_target_by_goal.update(
        {
            str(key): float(value)
            for key, value in (state.get("best_reward_target_by_goal") or {}).items()
        }
    )

    prev_group_feedback[:] = _feedback_summaries_from_payload(state.get("prev_group_feedback"))
    best_group_feedback[:] = _feedback_summaries_from_payload(state.get("best_group_feedback"))
    current_group_top_feedback[:] = _feedback_summaries_from_payload(state.get("current_group_top_feedback"))
    current_group_goal_best_candidates.clear()
    current_group_goal_best_candidates.update(
        {
            str(key): float(value)
            for key, value in (state.get("current_group_goal_best_candidates") or {}).items()
        }
    )


def _distributed_initialized() -> bool:
    return bool(torch.distributed.is_available() and torch.distributed.is_initialized())


def _distributed_world_size() -> int:
    if _distributed_initialized():
        return int(torch.distributed.get_world_size())
    return max(1, env_int("WORLD_SIZE", 1))


def _distributed_rank() -> int:
    if _distributed_initialized():
        return int(torch.distributed.get_rank())
    return env_int("RANK", 0)


def is_main_process() -> bool:
    return _distributed_rank() == 0


def _all_gather_object(payload: Any) -> List[Any]:
    if not _distributed_initialized() or _distributed_world_size() <= 1:
        return [payload]
    gathered: List[Any] = [None] * _distributed_world_size()
    torch.distributed.all_gather_object(gathered, payload)
    return gathered


def _broadcast_object(payload: Any, *, src: int = 0) -> Any:
    if not _distributed_initialized() or _distributed_world_size() <= 1:
        return payload
    objects = [payload if _distributed_rank() == src else None]
    torch.distributed.broadcast_object_list(objects, src=src)
    return objects[0]


def has_structural_motif(graph_info) -> bool:
    return bool(graph_info and (graph_info.project_calls or graph_info.stem_calls or graph_info.fractal_calls))


def is_multi_stage_architecture(graph_info) -> bool:
    return bool(graph_info and (graph_info.depth >= 5 or graph_info.merges >= 2 or graph_info.fractal_calls >= 2))


def passes_macro_structure_gate(graph_info) -> bool:
    if not graph_info or not graph_info.parse_ok or graph_info.is_plain_parallel_triple:
        return False
    if graph_info.project_calls or graph_info.stem_calls:
        return True
    return is_multi_stage_architecture(graph_info)


def is_shallow_one_shot_fuse(graph_info) -> bool:
    return bool(
        graph_info
        and graph_info.parse_ok
        and not graph_info.is_plain_parallel_triple
        and graph_info.fuse_calls >= 1
        and graph_info.merges <= 1
        and graph_info.depth <= 4
        and graph_info.project_calls == 0
        and graph_info.stem_calls == 0
        and graph_info.fractal_calls <= 1
        and graph_info.backbone_calls >= 1
    )


def family_save_cap(graph_info) -> int:
    return 4


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def best_mixed_precision() -> Dict[str, Any]:
    bf16_requested = os.getenv("NNGPT_RL_USE_BF16", "").strip().lower() in {"1", "true", "yes", "on"}
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_bf16 = bool(bf16_requested and bf16_ok)
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
    return {
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "torch_dtype": torch_dtype,
        "label": "bf16" if use_bf16 else "fp16",
    }


class DTypeSafeLinearWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    @property
    def weight(self):
        return getattr(self.module, "weight", None)

    @property
    def bias(self):
        return getattr(self.module, "bias", None)

    def forward(self, inputs, *args, **kwargs):
        weight = getattr(self.module, "weight", None)
        if weight is not None and hasattr(inputs, "dtype") and inputs.dtype != weight.dtype:
            inputs = inputs.to(weight.dtype)
        return self.module(inputs, *args, **kwargs)


def align_generation_head_dtype(model, torch_dtype: torch.dtype) -> None:
    aligned_modules = []
    wrapped_modules = []
    visited_models = set()
    visited_modules = set()
    wrapper_cache: Dict[int, DTypeSafeLinearWrapper] = {}

    def _cast_module(module, label: str) -> None:
        if module is None or not hasattr(module, "weight"):
            return
        if isinstance(module, DTypeSafeLinearWrapper):
            module = module.module
        module_id = id(module)
        if module_id in visited_modules:
            return
        visited_modules.add(module_id)
        weight = getattr(module, "weight", None)
        if weight is None:
            return
        before_dtype = weight.dtype
        if before_dtype == torch_dtype:
            return
        module.to(dtype=torch_dtype)
        aligned_modules.append(f"{label}:{before_dtype}->{torch_dtype}")

    def _ensure_wrapper(module, label: str):
        if module is None or not hasattr(module, "weight"):
            return module
        if isinstance(module, DTypeSafeLinearWrapper):
            return module
        module_id = id(module)
        wrapped = wrapper_cache.get(module_id)
        if wrapped is None:
            wrapped = DTypeSafeLinearWrapper(module)
            wrapper_cache[module_id] = wrapped
            wrapped_modules.append(label)
        return wrapped

    def _walk_model_tree(current_model, prefix: str) -> None:
        if current_model is None:
            return
        model_id = id(current_model)
        if model_id in visited_models:
            return
        visited_models.add(model_id)

        _cast_module(getattr(current_model, "lm_head", None), f"{prefix}.lm_head")
        try:
            _cast_module(current_model.get_output_embeddings(), f"{prefix}.output_embeddings")
        except Exception:
            pass

        head_module = getattr(current_model, "lm_head", None)
        wrapped_head = _ensure_wrapper(head_module, f"{prefix}.lm_head")
        if wrapped_head is not head_module:
            try:
                setattr(current_model, "lm_head", wrapped_head)
            except Exception:
                pass

        try:
            output_module = current_model.get_output_embeddings()
        except Exception:
            output_module = None
        wrapped_output = _ensure_wrapper(output_module, f"{prefix}.output_embeddings")
        if wrapped_output is not output_module and hasattr(current_model, "set_output_embeddings"):
            try:
                current_model.set_output_embeddings(wrapped_output)
            except Exception:
                pass

        for attr_name in ("base_model", "model", "module"):
            nested_model = getattr(current_model, attr_name, None)
            if nested_model is not None and nested_model is not current_model:
                _walk_model_tree(nested_model, f"{prefix}.{attr_name}")

    _walk_model_tree(model, "model")

    config = getattr(model, "config", None)
    if config is not None:
        try:
            config.torch_dtype = torch_dtype
        except Exception:
            pass

    if aligned_modules:
        print(f"[RL] Output dtype alignment: {', '.join(aligned_modules)}")
    if wrapped_modules:
        print(f"[RL] Output dtype safety wrappers: {', '.join(wrapped_modules)}")


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _optional_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _result_reward_target_value(res: Dict[str, Any]) -> Optional[float]:
    reward_target_value = _optional_float(res.get("reward_target_value"))
    if reward_target_value is not None:
        return reward_target_value
    return _optional_float(res.get("frozen_test_acc", res.get("val_metric")))


def _increment_optional_metric(sum_name: str, count_name: str, value: Optional[float]) -> None:
    if value is None:
        return
    globals()[sum_name] += float(value)
    globals()[count_name] += 1


def _mean_from_accumulator(sum_value: float, count_value: int) -> Optional[float]:
    if count_value <= 0:
        return None
    return float(sum_value) / float(count_value)


def _truncate_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _feedback_stats_short(open_discovery: Dict[str, Any]) -> str:
    structure_progress = float(open_discovery.get("r_structure_group", 0.0) or 0.0) + float(
        open_discovery.get("r_structure_archive", 0.0) or 0.0
    )
    return (
        f"depth:{int(open_discovery.get('depth', 0))} "
        f"merges:{int(open_discovery.get('merges', 0))} "
        f"stem:{int(open_discovery.get('stem_calls', 0))} "
        f"project:{int(open_discovery.get('project_calls', 0))} "
        f"fuse:{int(open_discovery.get('fuse_calls', 0))} "
        f"struct:{structure_progress:.2f}"
    )


def _group_feedback_paths() -> Tuple[Path, Path, Path]:
    log_dir = Path(run_log_dir())
    log_dir.mkdir(parents=True, exist_ok=True)
    return (
        log_dir / "group_progress.jsonl",
        log_dir / "group_feedback_samples.jsonl",
        log_dir / "best_group_feedback.json",
    )


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _current_group_top_feedback_payload() -> List[Dict[str, Any]]:
    return [asdict(item) for item in current_group_top_feedback[:FEEDBACK_SUMMARY_LIMIT]]


def _feedback_summary_payload(items: List[GroupFeedbackSummary]) -> List[Dict[str, Any]]:
    return [asdict(item) for item in items[:FEEDBACK_SUMMARY_LIMIT]]


def _build_group_feedback_summary(
    *,
    goal_key: str,
    res: Dict[str, Any],
    graph_info,
    reward_group_id: int,
) -> GroupFeedbackSummary:
    graph_expr_short = _truncate_text(str(res.get("graph_expr") or ""), FEEDBACK_GRAPH_EXPR_MAX_CHARS)
    pattern_name = str(res.get("pattern_name") or res.get("suggested_pattern_name") or "unknown")
    reward_target_value = float(_result_reward_target_value(res) or 0.0)
    frozen_train_acc = float(_optional_float(res.get("frozen_train_acc", res.get("train_acc"))) or 0.0)
    frozen_test_acc = float(_optional_float(res.get("frozen_test_acc", res.get("test_acc", res.get("val_metric")))) or 0.0)
    unfrozen_train_acc = _optional_float(res.get("unfrozen_train_acc"))
    unfrozen_test_acc = _optional_float(res.get("unfrozen_test_acc"))
    backbone_names = list(res.get("backbone_model_names") or [])
    open_discovery = dict(res.get("open_discovery") or {})
    stats_short = _feedback_stats_short(open_discovery)
    summary = (
        f"pattern={pattern_name}; "
        f"target={reward_target_value:.4f}; "
        f"frozen_train={frozen_train_acc:.4f}; "
        f"frozen_test={frozen_test_acc:.4f}; "
        f"backbones=[{', '.join(backbone_names)}]; "
        f"graph={graph_expr_short}; "
        f"stats={stats_short}"
    )
    summary = _truncate_text(summary, FEEDBACK_SUMMARY_MAX_CHARS)
    return GroupFeedbackSummary(
        goal_key=goal_key,
        pattern_name=pattern_name,
        graph_expr_short=graph_expr_short,
        reward_target_value=reward_target_value,
        frozen_train_acc=frozen_train_acc,
        frozen_test_acc=frozen_test_acc,
        unfrozen_train_acc=unfrozen_train_acc,
        unfrozen_test_acc=unfrozen_test_acc,
        backbone_model_names=backbone_names,
        stats_short=stats_short,
        summary=summary,
        family_hash=str(getattr(graph_info, "family_hash", "") or res.get("family_hash") or ""),
        signature=str(res.get("signature") or ""),
        reward_group_id=reward_group_id,
    )


def _update_top_feedback(summary: GroupFeedbackSummary) -> None:
    current_group_top_feedback.append(summary)
    current_group_top_feedback.sort(key=lambda item: item.reward_target_value, reverse=True)
    del current_group_top_feedback[FEEDBACK_SUMMARY_LIMIT:]


def _record_current_group_trainable_sample(goal_key: str, res: Dict[str, Any], graph_info) -> None:
    reward_target_value = _result_reward_target_value(res)
    if reward_target_value is None:
        return
    current_best = current_group_goal_best_candidates.get(goal_key)
    if current_best is None or float(reward_target_value) > current_best:
        current_group_goal_best_candidates[goal_key] = float(reward_target_value)
    summary = _build_group_feedback_summary(
        goal_key=goal_key,
        res=res,
        graph_info=graph_info,
        reward_group_id=current_group_id,
    )
    _update_top_feedback(summary)


def _reset_current_group_feedback_state() -> None:
    current_group_top_feedback.clear()
    current_group_goal_best_candidates.clear()


def get_prompt_feedback_state() -> Dict[str, Any]:
    return {
        "prev_closed_group_mean_reward_target_acc": prev_closed_group_mean_reward_target_acc,
        "best_closed_group_mean_reward_target_acc": best_closed_group_mean_reward_target_acc,
        "prev_closed_group_mean_train_acc": prev_closed_group_train_acc_mean,
        "best_closed_group_mean_train_acc": best_closed_group_mean_train_acc,
        "prev_closed_group_mean_test_acc": prev_closed_group_mean_test_acc,
        "best_closed_group_mean_test_acc": best_closed_group_mean_test_acc,
        "best_closed_group_id": best_closed_group_id,
        "dominant_family_hash": dominant_family_hash,
        "dominant_family_share": dominant_family_share,
        "prev_group_feedback": _feedback_summary_payload(prev_group_feedback),
        "best_group_feedback": _feedback_summary_payload(best_group_feedback),
    }


def _format_optional_metric(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _format_target_metric(base_value: Optional[float], delta: float) -> str:
    if base_value is None:
        return "n/a"
    return f"{float(base_value) + float(delta):.4f}"


def render_prompt_feedback_text(*, feedback_char_budget: int = 1200) -> str:
    state = get_prompt_feedback_state()
    header_lines = [
        f"- Reward Target Metric: {REWARD_TARGET_METRIC}",
        f"- Previous Closed Group Mean Target Acc: {_format_optional_metric(state['prev_closed_group_mean_reward_target_acc'])}",
        f"- Current Best Closed Group Mean Target Acc: {_format_optional_metric(state['best_closed_group_mean_reward_target_acc'])}",
        f"- Previous Closed Group Mean Frozen Train Acc: {_format_optional_metric(state['prev_closed_group_mean_train_acc'])}",
        f"- Previous Closed Group Mean Frozen Test Acc: {_format_optional_metric(state['prev_closed_group_mean_test_acc'])}",
        f"- Meaningful Reward Target: >= {_format_target_metric(state['prev_closed_group_mean_reward_target_acc'], GROUP_IMPROVEMENT_DELTA)}",
        f"- Stretch Target To Refresh Best: >= {_format_target_metric(state['best_closed_group_mean_reward_target_acc'], BEST_GROUP_REFRESH_DELTA)}",
        f"- Target Rule: beat previous closed group mean {REWARD_TARGET_METRIC} by at least {GROUP_IMPROVEMENT_DELTA:.4f}",
        "- Rule: prioritize higher frozen test accuracy, not just easier train accuracy",
        "- Rule: overfit patterns with large frozen-train minus frozen-test gaps are penalized",
        "- Rule: dominant-family reuse or plain classifier-only fuse below target is penalized",
        "- Rule: mutate strong motifs locally with stem/project/bridge/fuse improvements instead of resubmitting them",
        (
            "- Current Dominant Family To Avoid When Not Improving: "
            f"{state['dominant_family_hash'] or 'n/a'} "
            f"(share={float(state['dominant_family_share'] or 0.0):.2%})"
        ),
    ]

    prev_lines = [
        f"  - {item['summary']}"
        for item in state.get("prev_group_feedback", [])[:FEEDBACK_SUMMARY_LIMIT]
    ]
    best_lines = [
        f"  - {item['summary']}"
        for item in state.get("best_group_feedback", [])[:FEEDBACK_SUMMARY_LIMIT]
    ]

    def _compose_lines(current_prev_lines: List[str], current_best_lines: List[str]) -> str:
        lines = list(header_lines)
        if current_prev_lines:
            lines.append("- Previous Group Strong Examples:")
            lines.extend(current_prev_lines)
        else:
            lines.append("- Previous Group Strong Examples: none yet")

        if current_best_lines:
            lines.append("- Current Best Group Strong Examples:")
            lines.extend(current_best_lines)
        else:
            lines.append("- Current Best Group Strong Examples: none yet")
        return "\n".join(lines)

    text = _compose_lines(prev_lines, best_lines)
    if len(text) <= feedback_char_budget:
        return text

    if len(best_lines) >= 2:
        best_lines = best_lines[:1]
    text = _compose_lines(prev_lines, best_lines)
    if len(text) <= feedback_char_budget:
        return text

    if len(prev_lines) >= 2:
        prev_lines = prev_lines[:1]
        text = _compose_lines(prev_lines, best_lines)
    return _truncate_text(text, feedback_char_budget)


def reset_reward_runtime_state() -> None:
    global B_index
    global reward_batch_index
    global current_group_id
    global current_group_reward_target_sum
    global current_group_reward_target_count
    global current_group_frozen_train_acc_sum
    global current_group_frozen_train_acc_count
    global current_group_frozen_test_acc_sum
    global current_group_frozen_test_acc_count
    global current_group_unfrozen_train_acc_sum
    global current_group_unfrozen_train_acc_count
    global current_group_unfrozen_test_acc_sum
    global current_group_unfrozen_test_acc_count
    global prev_closed_group_mean_reward_target_acc
    global best_closed_group_mean_reward_target_acc
    global prev_closed_group_train_acc_mean
    global best_closed_group_mean_train_acc
    global prev_closed_group_mean_test_acc
    global best_closed_group_mean_test_acc
    global best_closed_group_id
    global dominant_family_hash
    global dominant_family_share

    graph_archive_counts.clear()
    family_archive_counts.clear()
    family_hash_archive_counts.clear()
    family_metric_best.clear()
    motif_name_counts.clear()
    saved_graph_counts.clear()
    saved_family_hash_counts.clear()
    goal_graph_archive_counts.clear()
    goal_family_hash_archive_counts.clear()
    saved_goal_family_hash_counts.clear()

    B_index = 0
    reward_batch_index = 0
    current_group_id = 0
    current_group_reward_target_sum = 0.0
    current_group_reward_target_count = 0
    current_group_frozen_train_acc_sum = 0.0
    current_group_frozen_train_acc_count = 0
    current_group_frozen_test_acc_sum = 0.0
    current_group_frozen_test_acc_count = 0
    current_group_unfrozen_train_acc_sum = 0.0
    current_group_unfrozen_train_acc_count = 0
    current_group_unfrozen_test_acc_sum = 0.0
    current_group_unfrozen_test_acc_count = 0
    prev_closed_group_mean_reward_target_acc = None
    best_closed_group_mean_reward_target_acc = None
    prev_closed_group_train_acc_mean = None
    best_closed_group_mean_train_acc = None
    prev_closed_group_mean_test_acc = None
    best_closed_group_mean_test_acc = None
    best_closed_group_id = None
    best_reward_target_by_goal.clear()
    dominant_family_hash = None
    dominant_family_share = 0.0
    prev_group_feedback.clear()
    best_group_feedback.clear()
    _reset_current_group_feedback_state()


def current_reward_group_context() -> Dict[str, Any]:
    return {
        "reward_batch_index": reward_batch_index + 1,
        "reward_group_id": current_group_id,
        "group_warmup": current_group_id == 0,
        "group_baseline_train_acc": prev_closed_group_train_acc_mean,
        "group_baseline_reward_target_acc": prev_closed_group_mean_reward_target_acc,
        "group_baseline_test_acc": prev_closed_group_mean_test_acc,
        "best_closed_group_mean_train_acc": best_closed_group_mean_train_acc,
        "best_closed_group_mean_reward_target_acc": best_closed_group_mean_reward_target_acc,
        "best_closed_group_mean_test_acc": best_closed_group_mean_test_acc,
        "best_closed_group_id": best_closed_group_id,
        "dominant_family_hash": dominant_family_hash,
        "dominant_family_share": dominant_family_share,
    }


def _read_process_rss_gib() -> Optional[float]:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / (1024.0 * 1024.0)
                    break
    except OSError:
        return None
    return None


def _cuda_memory_gib() -> Tuple[Optional[float], Optional[float]]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    try:
        allocated = torch.cuda.memory_allocated() / float(1024 ** 3)
        reserved = torch.cuda.memory_reserved() / float(1024 ** 3)
        return allocated, reserved
    except RuntimeError:
        return None, None


def _format_mem_value(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _visible_cuda_device_tokens() -> List[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return []
    raw = raw.strip()
    if raw in {"", "-1"}:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _resolved_train_gpu_index() -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    visible_gpu_count = int(torch.cuda.device_count())
    if visible_gpu_count <= 0:
        return None
    try:
        current_device = int(torch.cuda.current_device())
        if 0 <= current_device < visible_gpu_count:
            return current_device
    except Exception:
        pass
    raw_local_rank = env_int("LOCAL_RANK", 0)
    if visible_gpu_count == 1:
        return 0
    if 0 <= raw_local_rank < visible_gpu_count:
        return raw_local_rank
    return 0


def _visible_cuda_memory_snapshots(*, include_all_visible_gpus: bool) -> List[Dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    visible_gpu_count = int(torch.cuda.device_count())
    if visible_gpu_count <= 0:
        return []
    device_tokens = _visible_cuda_device_tokens()
    train_gpu_index = _resolved_train_gpu_index()
    if include_all_visible_gpus:
        device_indices = list(range(visible_gpu_count))
    elif train_gpu_index is not None:
        device_indices = [int(train_gpu_index)]
    else:
        device_indices = [0]

    snapshots: List[Dict[str, Any]] = []
    for device_index in device_indices:
        total_gib = None
        free_gib = None
        used_gib = None
        allocated_gib = None
        reserved_gib = None
        device_name = ""
        try:
            props = torch.cuda.get_device_properties(device_index)
            total_gib = float(props.total_memory) / float(1024 ** 3)
            device_name = str(getattr(props, "name", "") or "")
        except Exception:
            pass
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            free_gib = float(free_bytes) / float(1024 ** 3)
            used_gib = float(total_bytes - free_bytes) / float(1024 ** 3)
            if total_gib is None:
                total_gib = float(total_bytes) / float(1024 ** 3)
        except Exception:
            pass
        try:
            allocated_gib = float(torch.cuda.memory_allocated(device_index)) / float(1024 ** 3)
        except Exception:
            allocated_gib = None
        try:
            reserved_gib = float(torch.cuda.memory_reserved(device_index)) / float(1024 ** 3)
        except Exception:
            reserved_gib = None

        other_used_gib = None
        if used_gib is not None and allocated_gib is not None:
            other_used_gib = max(0.0, float(used_gib) - float(allocated_gib))

        device_token = (
            device_tokens[device_index]
            if 0 <= device_index < len(device_tokens)
            else str(int(device_index))
        )
        snapshots.append(
            {
                "device_index": int(device_index),
                "device_token": str(device_token),
                "device_name": device_name,
                "total_gib": total_gib,
                "free_gib": free_gib,
                "used_gib": used_gib,
                "allocated_gib": allocated_gib,
                "reserved_gib": reserved_gib,
                "other_used_gib": other_used_gib,
                "is_train_gpu": bool(train_gpu_index is not None and int(train_gpu_index) == int(device_index)),
            }
        )
    return snapshots


def _current_cuda_allocator_snapshot() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    try:
        current_device = int(torch.cuda.current_device())
    except Exception:
        current_device = _resolved_train_gpu_index()
    if current_device is None:
        return {}
    try:
        stats = torch.cuda.memory_stats(current_device)
    except Exception:
        return {}
    return {
        "current_device": int(current_device),
        "active_gib": float(stats.get("active_bytes.all.current", 0.0)) / float(1024 ** 3),
        "reserved_gib": float(stats.get("reserved_bytes.all.current", 0.0)) / float(1024 ** 3),
        "inactive_split_gib": float(stats.get("inactive_split_bytes.all.current", 0.0)) / float(1024 ** 3),
        "num_ooms": int(stats.get("num_ooms", 0)),
        "num_alloc_retries": int(stats.get("num_alloc_retries", 0)),
    }


def _query_nvidia_smi_csv(query_kind: str, columns: List[str]) -> List[List[str]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    command = [
        executable,
        f"--query-{query_kind}={','.join(columns)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return []
    if completed.returncode != 0 or not completed.stdout.strip():
        return []
    reader = csv.reader(completed.stdout.splitlines())
    return [[cell.strip() for cell in row] for row in reader if row]


def _log_nvidia_smi_snapshot(stage: str) -> None:
    gpu_rows = _query_nvidia_smi_csv(
        "gpu",
        ["index", "uuid", "name", "memory.total", "memory.used", "memory.free"],
    )
    if not gpu_rows:
        print(f"[OOM nvidia-smi] stage={stage} unavailable=True")
        return

    visible_tokens = set(_visible_cuda_device_tokens())
    filter_visible = bool(visible_tokens)
    gpu_rows_by_uuid: Dict[str, Dict[str, str]] = {}
    for row in gpu_rows:
        if len(row) < 6:
            continue
        gpu_index, gpu_uuid, gpu_name, total_mib, used_mib, free_mib = row[:6]
        if filter_visible and gpu_index not in visible_tokens and gpu_uuid not in visible_tokens:
            continue
        gpu_rows_by_uuid[gpu_uuid] = {
            "index": gpu_index,
            "uuid": gpu_uuid,
            "name": gpu_name,
            "total_mib": total_mib,
            "used_mib": used_mib,
            "free_mib": free_mib,
        }
        print(
            "[OOM nvidia-smi GPU] "
            f"stage={stage} "
            f"gpu={gpu_index} "
            f"name={gpu_name!r} "
            f"used_mib={used_mib} "
            f"free_mib={free_mib} "
            f"total_mib={total_mib}"
        )

    process_rows = _query_nvidia_smi_csv(
        "compute-apps",
        ["gpu_uuid", "pid", "process_name", "used_memory"],
    )
    process_snapshots: List[Dict[str, Any]] = []
    for row in process_rows:
        if len(row) < 4:
            continue
        gpu_uuid, pid, process_name, used_mib = row[:4]
        gpu_info = gpu_rows_by_uuid.get(gpu_uuid)
        if gpu_info is None:
            continue
        try:
            used_mib_value = int(float(used_mib))
        except (TypeError, ValueError):
            used_mib_value = 0
        process_snapshots.append(
            {
                "gpu": gpu_info["index"],
                "pid": pid,
                "process_name": process_name,
                "used_mib": used_mib_value,
            }
        )
    process_snapshots.sort(key=lambda item: int(item["used_mib"]), reverse=True)
    for snapshot in process_snapshots[:24]:
        print(
            "[OOM nvidia-smi Proc] "
            f"stage={stage} "
            f"gpu={snapshot['gpu']} "
            f"pid={snapshot['pid']} "
            f"used_mib={snapshot['used_mib']} "
            f"process={snapshot['process_name']!r}"
        )


def is_cuda_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    normalized = " ".join(str(exc).split()).lower()
    return "out of memory" in normalized and "cuda" in normalized


def log_cuda_oom_diagnostics(
    stage: str,
    exc: BaseException,
    *,
    group_context: Optional[Dict[str, Any]] = None,
) -> None:
    print(f"[OOM] stage={stage} error={type(exc).__name__}: {exc}")
    log_memory_snapshot(stage, group_context=group_context, include_all_visible_gpus=True)
    allocator_snapshot = _current_cuda_allocator_snapshot()
    if allocator_snapshot:
        print(
            "[OOM Allocator] "
            f"stage={stage} "
            f"current_device={allocator_snapshot['current_device']} "
            f"active_gib={_format_mem_value(allocator_snapshot['active_gib'])} "
            f"reserved_gib={_format_mem_value(allocator_snapshot['reserved_gib'])} "
            f"inactive_split_gib={_format_mem_value(allocator_snapshot['inactive_split_gib'])} "
            f"num_ooms={allocator_snapshot['num_ooms']} "
            f"num_alloc_retries={allocator_snapshot['num_alloc_retries']}"
        )
    _log_nvidia_smi_snapshot(stage)


class _CudaMemoryMonitor:
    def __init__(self, stage_prefix: str) -> None:
        self._stage_prefix = str(stage_prefix)
        self._enabled = bool(torch.cuda.is_available()) and env_int("NNGPT_CUDA_MEMORY_MONITOR", 1) > 0
        self._interval_seconds = max(1.0, env_float("NNGPT_CUDA_MEMORY_MONITOR_INTERVAL_SECONDS", 30.0))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> Optional["_CudaMemoryMonitor"]:
        if not self._enabled:
            return None
        print(
            "[Memory Monitor] "
            f"stage_prefix={self._stage_prefix} "
            f"interval_seconds={self._interval_seconds:.1f}"
        )
        self._thread = threading.Thread(
            target=self._run,
            name=f"nngpt-cuda-memory-monitor-{self._stage_prefix}",
            daemon=True,
        )
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            try:
                log_memory_snapshot(f"{self._stage_prefix}:tick")
            except Exception as exc:
                print(
                    "[Memory Monitor] "
                    f"stage_prefix={self._stage_prefix} "
                    f"error={type(exc).__name__}: {exc}"
                )

    def close(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=max(2.0, self._interval_seconds + 1.0))
        self._thread = None


def start_cuda_memory_monitor(stage_prefix: str) -> Optional[_CudaMemoryMonitor]:
    monitor = _CudaMemoryMonitor(stage_prefix)
    return monitor.start()


def log_memory_snapshot(
    stage: str,
    *,
    group_context: Optional[Dict[str, Any]] = None,
    include_all_visible_gpus: Optional[bool] = None,
) -> None:
    effective_group_context = group_context or current_reward_group_context()
    cuda_allocated_gib, cuda_reserved_gib = _cuda_memory_gib()
    worker_info = get_eval_worker_diagnostics()
    worker_pid = worker_info.get("worker_pids", [worker_info.get("pid")]) if worker_info else None
    rank = _distributed_rank()
    local_rank = env_int("LOCAL_RANK", 0)
    world_size = _distributed_world_size()
    train_gpu = _resolved_train_gpu_index()
    if include_all_visible_gpus is None:
        include_all_visible_gpus = bool(world_size <= 1 or is_main_process())
    visible_cuda_snapshots = _visible_cuda_memory_snapshots(
        include_all_visible_gpus=bool(include_all_visible_gpus)
    )
    print(
        "[Memory] "
        f"stage={stage} "
        f"pid={os.getpid()} "
        f"rank={rank} "
        f"local_rank={local_rank} "
        f"world_size={world_size} "
        f"train_gpu={train_gpu} "
        f"reward_batch_index={effective_group_context.get('reward_batch_index')} "
        f"reward_group_id={effective_group_context.get('reward_group_id')} "
        f"rss_gib={_format_mem_value(_read_process_rss_gib())} "
        f"cuda_allocated_gib={_format_mem_value(cuda_allocated_gib)} "
        f"cuda_reserved_gib={_format_mem_value(cuda_reserved_gib)} "
        f"worker_pid={worker_pid}"
    )
    for snapshot in visible_cuda_snapshots:
        train_gpu_marker = "*" if snapshot.get("is_train_gpu") else ""
        print(
            "[Memory GPU] "
            f"stage={stage} "
            f"gpu={snapshot['device_index']}{train_gpu_marker} "
            f"token={snapshot['device_token']} "
            f"name={snapshot['device_name']!r} "
            f"free_gib={_format_mem_value(snapshot['free_gib'])} "
            f"used_gib={_format_mem_value(snapshot['used_gib'])} "
            f"total_gib={_format_mem_value(snapshot['total_gib'])} "
            f"proc_allocated_gib={_format_mem_value(snapshot['allocated_gib'])} "
            f"proc_reserved_gib={_format_mem_value(snapshot['reserved_gib'])} "
            f"other_used_gib={_format_mem_value(snapshot['other_used_gib'])}"
        )
    if worker_info:
        print(
            "[Memory Workers] "
            f"stage={stage} "
            f"mode={worker_info.get('mode')} "
            f"pool_size={worker_info.get('pool_size')} "
            f"shared_train_gpu={worker_info.get('shared_train_gpu')} "
            f"reward_gpu_indices={worker_info.get('reward_gpu_indices')} "
            f"per_gpu_worker_counts={worker_info.get('per_gpu_worker_counts')} "
            f"worker_pids={worker_info.get('worker_pids')} "
            f"total_worker_rss_gib={_format_mem_value(worker_info.get('total_rss_gib'))}"
        )


def update_current_group_metrics(results: List[Dict[str, Any]]) -> None:
    for res in results:
        _increment_optional_metric(
            "current_group_reward_target_sum",
            "current_group_reward_target_count",
            _result_reward_target_value(res),
        )
        _increment_optional_metric(
            "current_group_frozen_train_acc_sum",
            "current_group_frozen_train_acc_count",
            _optional_float(res.get("frozen_train_acc", res.get("train_acc"))),
        )
        _increment_optional_metric(
            "current_group_frozen_test_acc_sum",
            "current_group_frozen_test_acc_count",
            _optional_float(res.get("frozen_test_acc", res.get("test_acc", res.get("val_metric")))),
        )
        _increment_optional_metric(
            "current_group_unfrozen_train_acc_sum",
            "current_group_unfrozen_train_acc_count",
            _optional_float(res.get("unfrozen_train_acc")),
        )
        _increment_optional_metric(
            "current_group_unfrozen_test_acc_sum",
            "current_group_unfrozen_test_acc_count",
            _optional_float(res.get("unfrozen_test_acc")),
        )


def close_reward_group_if_needed() -> Optional[Dict[str, Any]]:
    global reward_batch_index
    global current_group_id
    global current_group_reward_target_sum
    global current_group_reward_target_count
    global current_group_frozen_train_acc_sum
    global current_group_frozen_train_acc_count
    global current_group_frozen_test_acc_sum
    global current_group_frozen_test_acc_count
    global current_group_unfrozen_train_acc_sum
    global current_group_unfrozen_train_acc_count
    global current_group_unfrozen_test_acc_sum
    global current_group_unfrozen_test_acc_count
    global prev_closed_group_mean_reward_target_acc
    global best_closed_group_mean_reward_target_acc
    global prev_closed_group_train_acc_mean
    global best_closed_group_mean_train_acc
    global prev_closed_group_mean_test_acc
    global best_closed_group_mean_test_acc
    global best_closed_group_id
    global dominant_family_hash
    global dominant_family_share

    reward_batch_index += 1
    if reward_batch_index % GROUP_BATCH_SIZE != 0:
        return None

    previous_closed_reward_target_mean = prev_closed_group_mean_reward_target_acc
    previous_best_reward_target_mean = best_closed_group_mean_reward_target_acc
    previous_closed_train_mean = prev_closed_group_train_acc_mean
    previous_closed_test_mean = prev_closed_group_mean_test_acc
    closed_mean_reward_target = _mean_from_accumulator(
        current_group_reward_target_sum,
        current_group_reward_target_count,
    )
    closed_mean_train = _mean_from_accumulator(
        current_group_frozen_train_acc_sum,
        current_group_frozen_train_acc_count,
    )
    closed_mean_test = _mean_from_accumulator(
        current_group_frozen_test_acc_sum,
        current_group_frozen_test_acc_count,
    )
    closed_mean_unfrozen_train = _mean_from_accumulator(
        current_group_unfrozen_train_acc_sum,
        current_group_unfrozen_train_acc_count,
    )
    closed_mean_unfrozen_test = _mean_from_accumulator(
        current_group_unfrozen_test_acc_sum,
        current_group_unfrozen_test_acc_count,
    )
    prev_closed_group_mean_reward_target_acc = closed_mean_reward_target
    prev_closed_group_train_acc_mean = closed_mean_train
    prev_closed_group_mean_test_acc = closed_mean_test

    if best_closed_group_mean_reward_target_acc is None or (
        closed_mean_reward_target is not None and closed_mean_reward_target > best_closed_group_mean_reward_target_acc
    ):
        if closed_mean_reward_target is not None:
            best_closed_group_mean_reward_target_acc = closed_mean_reward_target
            best_closed_group_id = current_group_id
            best_group_feedback[:] = list(current_group_top_feedback[:FEEDBACK_SUMMARY_LIMIT])

    for goal_key, candidate_best in current_group_goal_best_candidates.items():
        best_reward_target_by_goal[goal_key] = max(
            float(candidate_best),
            float(best_reward_target_by_goal.get(goal_key, float("-inf"))),
        )

    if best_closed_group_mean_train_acc is None or (
        closed_mean_train is not None and closed_mean_train > best_closed_group_mean_train_acc
    ):
        if closed_mean_train is not None:
            best_closed_group_mean_train_acc = closed_mean_train

    if best_closed_group_mean_test_acc is None or (
        closed_mean_test is not None and closed_mean_test > best_closed_group_mean_test_acc
    ):
        if closed_mean_test is not None:
            best_closed_group_mean_test_acc = closed_mean_test

    total_valid = sum(family_hash_archive_counts.values())
    if total_valid > 0:
        dominant_family_hash, dominant_count = family_hash_archive_counts.most_common(1)[0]
        dominant_family_share = dominant_count / total_valid
    else:
        dominant_family_hash = None
        dominant_family_share = 0.0

    prev_group_feedback[:] = list(current_group_top_feedback[:FEEDBACK_SUMMARY_LIMIT])

    progress_path, feedback_path, best_feedback_path = _group_feedback_paths()
    worker_info = get_eval_worker_diagnostics()
    group_progress_payload = {
        "group_id": current_group_id,
        "group_warmup": current_group_id == 0,
        "reward_batch_index": reward_batch_index,
        "closed_mean_reward_target_acc": closed_mean_reward_target,
        "prev_closed_group_mean_reward_target_acc": previous_closed_reward_target_mean,
        "best_closed_group_mean_reward_target_acc": best_closed_group_mean_reward_target_acc,
        "closed_mean_train_acc": closed_mean_train,
        "closed_mean_test_acc": closed_mean_test,
        "closed_mean_unfrozen_train_acc": closed_mean_unfrozen_train,
        "closed_mean_unfrozen_test_acc": closed_mean_unfrozen_test,
        "prev_closed_group_mean_train_acc": previous_closed_train_mean,
        "best_closed_group_mean_train_acc": best_closed_group_mean_train_acc,
        "prev_closed_group_mean_test_acc": previous_closed_test_mean,
        "best_closed_group_mean_test_acc": best_closed_group_mean_test_acc,
        "best_closed_group_id": best_closed_group_id,
        "improvement_vs_prev": None if closed_mean_reward_target is None or previous_closed_reward_target_mean is None else float(closed_mean_reward_target - previous_closed_reward_target_mean),
        "improvement_vs_best": None if closed_mean_reward_target is None or previous_best_reward_target_mean is None else float(closed_mean_reward_target - previous_best_reward_target_mean),
        "dominant_family_hash": dominant_family_hash,
        "dominant_family_share": dominant_family_share,
        "trainable_samples": current_group_reward_target_count,
        "main_process_rss_gib": _read_process_rss_gib(),
        "worker_rss_gib": worker_info.get("total_rss_gib", worker_info.get("rss_gib")) if worker_info else None,
        "prev_group_feedback": _feedback_summary_payload(prev_group_feedback),
        "best_group_feedback": _feedback_summary_payload(best_group_feedback),
    }
    _append_jsonl(progress_path, group_progress_payload)

    for summary in prev_group_feedback:
        sample_payload = {
            "group_id": current_group_id,
            "group_warmup": current_group_id == 0,
            "summary": asdict(summary),
            "closed_mean_reward_target_acc": closed_mean_reward_target,
            "closed_mean_train_acc": closed_mean_train,
            "closed_mean_test_acc": closed_mean_test,
        }
        _append_jsonl(feedback_path, sample_payload)

    if best_closed_group_id == current_group_id and best_closed_group_mean_reward_target_acc is not None:
        _write_json(
            best_feedback_path,
            {
                "group_id": best_closed_group_id,
                "best_closed_group_mean_reward_target_acc": best_closed_group_mean_reward_target_acc,
                "best_closed_group_mean_train_acc": best_closed_group_mean_train_acc,
                "best_closed_group_mean_test_acc": best_closed_group_mean_test_acc,
                "feedback": _feedback_summary_payload(best_group_feedback),
            },
        )

    target_text = "n/a" if closed_mean_reward_target is None else f"{closed_mean_reward_target:.4f}"
    prev_target_text = "n/a" if previous_closed_reward_target_mean is None else f"{previous_closed_reward_target_mean:.4f}"
    best_target_text = "n/a" if best_closed_group_mean_reward_target_acc is None else f"{best_closed_group_mean_reward_target_acc:.4f}"
    train_text = "n/a" if closed_mean_train is None else f"{closed_mean_train:.4f}"
    test_text = "n/a" if closed_mean_test is None else f"{closed_mean_test:.4f}"
    print(
        f"[Reward Group] Closed group {current_group_id} after {GROUP_BATCH_SIZE} reward batches: "
        f"mean_target_acc={target_text}, prev_target={prev_target_text}, best_target={best_target_text}, "
        f"mean_frozen_train_acc={train_text}, mean_frozen_test_acc={test_text}, "
        f"trainable_samples={current_group_reward_target_count}, dominant_family={dominant_family_hash or 'n/a'} "
        f"({dominant_family_share:.2%})"
    )
    current_group_id += 1
    current_group_reward_target_sum = 0.0
    current_group_reward_target_count = 0
    current_group_frozen_train_acc_sum = 0.0
    current_group_frozen_train_acc_count = 0
    current_group_frozen_test_acc_sum = 0.0
    current_group_frozen_test_acc_count = 0
    current_group_unfrozen_train_acc_sum = 0.0
    current_group_unfrozen_train_acc_count = 0
    current_group_unfrozen_test_acc_sum = 0.0
    current_group_unfrozen_test_acc_count = 0
    _reset_current_group_feedback_state()
    return group_progress_payload


def _coerce_accuracy_baseline(value: Any, *, context: str) -> float:
    if value is None:
        raise ValueError(f"{context}: missing required sample accuracy baseline")
    if isinstance(value, bool):
        raise ValueError(f"{context}: accuracy baseline must be numeric, got bool")
    try:
        baseline = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: accuracy baseline must be numeric, got {value!r}") from exc
    if baseline != baseline or baseline in {float("inf"), float("-inf")}:
        raise ValueError(f"{context}: accuracy baseline must be finite, got {value!r}")
    return baseline


def require_sample_accuracy_baselines(kwargs: Dict[str, Any], expected_count: int) -> List[float]:
    if "accuracy" not in kwargs:
        raise ValueError("compute_reward requires kwargs['accuracy'] for every sample")
    raw_values = kwargs["accuracy"]
    if len(raw_values) != expected_count:
        raise ValueError(
            f"compute_reward expected {expected_count} accuracy baselines, got {len(raw_values)}"
        )
    return [
        _coerce_accuracy_baseline(value, context=f"completion[{idx}]")
        for idx, value in enumerate(raw_values)
    ]


def run_epoch_dir(*args):
    root_override = os.getenv("NNGPT_RL_EPOCH_ROOT")
    if root_override:
        e_dir = Path(root_override)
        for d in args:
            e_dir = e_dir / f"A{d}"
        return e_dir
    return epoch_dir(*args)


def run_log_dir() -> str:
    return os.getenv("NNGPT_RL_LOG_DIR", "rl_output")


def run_model_out() -> str:
    return os.getenv("NNGPT_RL_MODEL_OUT", SAVED_MODEL_PATH)


def _iter_text_candidates(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: List[str] = []
        for item in value.values():
            out.extend(_iter_text_candidates(item))
        return out
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            out.extend(_iter_text_candidates(item))
        return out
    return []


def _score_seed_source_candidate(field_name: str, text: str) -> int:
    lowered_field = (field_name or "").lower()
    lowered_text = text.lower()
    score = 0
    if "<init>" in lowered_text and "<forward>" in lowered_text:
        score += 100
    if "class net" in lowered_text and "def forward" in lowered_text:
        score += 80
    if "def __init__" in lowered_text and "def forward" in lowered_text:
        score += 60
    if any(token in lowered_field for token in ("completion", "response", "output", "assistant", "xml")):
        score += 25
    if any(token in lowered_field for token in ("code", "nn", "model", "content", "text")):
        score += 10
    if len(text) > 200:
        score += 5
    return score


def _extract_method_from_module_text(source_text: str, class_name: str, method_name: str) -> str:
    try:
        tree = ast.parse(source_text)
    except Exception:
        return ""

    lines = source_text.splitlines()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    if item.end_lineno is None:
                        return ""
                    snippet = "\n".join(lines[item.lineno - 1:item.end_lineno])
                    return textwrap.dedent(snippet).strip()
    return ""


def _extract_seed_init_forward_from_text(text: str) -> Tuple[str, str]:
    candidate = clean_block(text)
    if not candidate:
        return "", ""

    _, init_code, forward_code = extract_completion_blocks(candidate)
    if init_code and forward_code:
        return init_code, forward_code

    stripped = candidate.replace("```python", "").replace("```", "").strip()
    init_code = _extract_method_from_module_text(stripped, "Net", "__init__")
    forward_code = _extract_method_from_module_text(stripped, "Net", "forward")
    return init_code, forward_code


def _extract_seed_candidates_from_row(row: Any) -> List[str]:
    row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    ranked: List[Tuple[int, str]] = []
    seen: Set[str] = set()

    for key, value in row_dict.items():
        for text in _iter_text_candidates(value):
            stripped = text.strip()
            if not stripped or stripped in seen:
                continue
            score = _score_seed_source_candidate(str(key), stripped)
            if score <= 0:
                continue
            ranked.append((score, stripped))
            seen.add(stripped)

    ranked.sort(key=lambda item: (-item[0], -len(item[1])))
    return [text for _, text in ranked]


def bootstrap_trainset_reference_library(data) -> None:
    train_graph_hashes.clear()
    train_family_hashes.clear()
    train_descriptor_keys.clear()

    stats = {
        "rows_seen": 0,
        "rows_parsed": 0,
        "rows_skipped": 0,
        "candidate_texts": 0,
    }

    for _, row in data.iterrows():
        stats["rows_seen"] += 1
        candidates = _extract_seed_candidates_from_row(row)
        stats["candidate_texts"] += len(candidates)
        parsed_ok = False

        for candidate in candidates:
            init_code, forward_code = _extract_seed_init_forward_from_text(candidate)
            if not init_code or not forward_code:
                continue
            graph_info = extract_graph_info(
                init_code,
                forward_code,
                legacy_patterns=SFTUtil.legacy_patterns,
            )
            if not graph_info.parse_ok:
                continue
            train_graph_hashes.add(graph_info.graph_hash)
            train_family_hashes.add(graph_info.family_hash)
            train_descriptor_keys.add(graph_info.descriptor_key)
            parsed_ok = True
            break

        if parsed_ok:
            stats["rows_parsed"] += 1
        else:
            stats["rows_skipped"] += 1

    train_reference_stats.clear()
    train_reference_stats.update(stats)
    print(
        "[Trainset Reference] "
        f"rows={stats['rows_seen']}, parsed={stats['rows_parsed']}, skipped={stats['rows_skipped']}, "
        f"graph_hashes={len(train_graph_hashes)}, family_hashes={len(train_family_hashes)}, "
        f"descriptor_keys={len(train_descriptor_keys)}"
    )


def extract_prompt_goal_tags(prompt_text: str) -> List[str]:
    if not prompt_text:
        return []
    match = re.search(r"(?:Discovery|Optimization) Target Tags:\s*([A-Za-z0-9_, \-]+)", prompt_text)
    if not match:
        return []
    return [tag.strip() for tag in match.group(1).split(",") if tag.strip()]


def prompt_goal_satisfied(graph_info, tag: str) -> bool:
    if not graph_info or not graph_info.parse_ok:
        return False
    if tag == "stem":
        return graph_info.stem_calls > 0
    if tag == "project":
        return graph_info.project_calls > 0
    if tag == "multi_stage":
        return is_multi_stage_architecture(graph_info)
    if tag == "fractal_deep":
        return graph_info.fractal_calls >= 2 or (graph_info.fractal_calls >= 1 and graph_info.depth >= 5)
    if tag == "branch_reuse":
        return graph_info.merges >= 2 or (graph_info.project_calls > 0 and graph_info.fuse_calls >= 2)
    if tag == "single_backbone":
        return graph_info.backbone_calls == 1
    if tag == "wide_fuse":
        return graph_info.max_fan_in >= 3 and graph_info.fuse_calls >= 1
    return False


def primary_goal_key(prompt_goal_tags: List[str]) -> str:
    return "__".join(prompt_goal_tags or ["open"])


def goal_family_save_cap(graph_info) -> int:
    return 2


def get_goal_counter(store: Dict[str, Counter], goal_key: str) -> Counter:
    if goal_key not in store:
        store[goal_key] = Counter()
    return store[goal_key]


def clean_block(text: str) -> str:
    """Remove common LLM artifacts like markdown code blocks."""
    if not text: return ""
    text = text.strip()
    # Remove ```python ... ```
    text = re.sub(r'^```python\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()

def extract_completion_blocks(completion: str) -> Tuple[str, str, str]:
    """Extract the three XML code blocks and normalize their formatting."""
    block_code = clean_block(extract_str(completion, '<block>', '</block>'))
    init_code = clean_block(extract_str(completion, '<init>', '</init>'))
    forward_code = clean_block(extract_str(completion, '<forward>', '</forward>'))
    return block_code, init_code, forward_code


def render_completion_xml(block_code: str, init_code: str, forward_code: str) -> str:
    return "\n".join(
        [
            "<block>",
            textwrap.dedent(block_code).strip(),
            "</block>",
            "<init>",
            textwrap.dedent(init_code).strip(),
            "</init>",
            "<forward>",
            textwrap.dedent(forward_code).strip(),
            "</forward>",
        ]
    )


def _extract_backbone_model_names(init_code: str) -> List[str]:
    matches: Dict[str, str] = {}
    patterns = (
        r"self\.(backbone_[ab])\s*=\s*TorchVision\(\s*model\s*=\s*['\"]([^'\"]+)['\"]",
        r"self\.(backbone_[ab])\s*=\s*TorchVision\(\s*['\"]([^'\"]+)['\"]",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, init_code or ""):
            matches.setdefault(match.group(1), match.group(2))
    return [matches[name] for name in ("backbone_a", "backbone_b") if name in matches]


def reconstruct_code(
    completion: str,
    *,
    pattern_name_override: str = "",
) -> str:
    """Rebuild a runnable Python module from the XML blocks."""
    block_code, init_code, forward_code = extract_completion_blocks(completion)
    if not block_code or not init_code or not forward_code:
        return ""

    if pattern_name_override:
        init_code = ensure_pattern_name(init_code, pattern_name_override)

    code = SFTUtil.open_discovery_skeleton_code
    sig_block = "def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):"
    code = code.replace(sig_block, textwrap.dedent(block_code))

    sig_init = "    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:"
    code = code.replace(sig_init, textwrap.indent(textwrap.dedent(init_code), "    "))

    sig_forward = "    def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:"
    code = code.replace(sig_forward, textwrap.indent(textwrap.dedent(forward_code), "    "))
    return code


def _compute_build_partial_reward(res: Dict[str, Any]) -> float:
    error_str = str(res.get('error', ''))
    build_partial = 0.0
    if 'SyntaxError' in error_str:
        build_partial = -0.3
    elif 'NameError' in error_str or 'ImportError' in error_str:
        build_partial = -0.2
    elif 'TypeError' in error_str:
        build_partial = -0.1
    elif 'RuntimeError' in error_str and 'shape' in error_str.lower():
        build_partial = 0.05
    elif error_str:
        build_partial = -0.15
    return build_partial


def _compute_warmup_dense_reward(test_acc: Optional[float]) -> Optional[float]:
    if test_acc is None:
        return None
    return max(0.05, min(0.30, 0.08 + 0.55 * float(test_acc)))


def _goal_tag_match_stats(graph_info, prompt_goal_tags: Optional[List[str]]) -> Tuple[int, int, float]:
    tags = list(prompt_goal_tags or [])
    if not tags:
        return 0, 0, 0.0
    hit_count = sum(1 for tag in tags if prompt_goal_satisfied(graph_info, tag))
    total_count = len(tags)
    hit_rate = float(hit_count) / float(total_count) if total_count > 0 else 0.0
    return hit_count, total_count, hit_rate


def _discovery_failure_result(
    reward: float,
    error: str,
    *,
    seed_accuracy_baseline: float,
    backbone_model_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "reward": reward,
        "built_ok": False,
        "forward_ok": False,
        "forward_shape_ok": False,
        "trained_step_ok": False,
        "backward_ok": False,
        "loss_start": None,
        "loss_end": None,
        "loss_drop": None,
        "loss_drop_ok": False,
        "test_acc": None,
        "train_acc": None,
        "frozen_train_acc": None,
        "frozen_test_acc": None,
        "unfrozen_train_acc": None,
        "unfrozen_test_acc": None,
        "frozen_eval": None,
        "unfrozen_eval": None,
        "seed_accuracy_baseline": seed_accuracy_baseline,
        "seed_train_acc_gap": None,
        "seed_train_acc_improved": False,
        "accuracy_baseline": seed_accuracy_baseline,
        "train_acc_gain": None,
        "train_acc_improved": False,
        "group_baseline_train_acc": None,
        "group_train_acc_gain": None,
        "group_train_acc_improved": False,
        "group_baseline_reward_target_acc": None,
        "group_reward_target_gain": None,
        "group_reward_target_improved": False,
        "reward_batch_index": None,
        "reward_group_id": None,
        "group_warmup": False,
        "val_metric": None,
        "latency_ms": None,
        "params_m": None,
        "timed_out": False,
        "estimated_total_seconds": None,
        "eval_limit_seconds": None,
        "warmup_dense_reward": None,
        "backbone_model_names": list(backbone_model_names or []),
        "reward_target_metric": REWARD_TARGET_METRIC,
        "reward_target_value": None,
        "best_closed_group_mean_reward_target_acc": best_closed_group_mean_reward_target_acc,
        "best_closed_group_mean_train_acc": best_closed_group_mean_train_acc,
        "best_closed_group_mean_test_acc": best_closed_group_mean_test_acc,
        "best_reward_target_for_goal": None,
        "r_dense": 0.0,
        "r_prev_group": 0.0,
        "r_best_group": 0.0,
        "r_goal_best": 0.0,
        "r_goal_match": 0.0,
        "r_trainset_novelty": 0.0,
        "r_generalization": 0.0,
        "r_structure_group": 0.0,
        "r_structure_archive": 0.0,
        "r_batch_elite": 0.0,
        "r_repeat_family": 0.0,
        "r_plain_fuse_penalty": 0.0,
        "r_no_progress_penalty": 0.0,
        "batch_elite_rank": None,
        "batch_elite_tier": "none",
        "batch_elite_threshold_passed": False,
        "goal_tag_hit_count": 0,
        "goal_tag_total_count": 0,
        "goal_tag_hit_rate": 0.0,
        "prev_target_reward_target_acc": None,
        "best_target_reward_target_acc": None,
        "open_discovery": {
            "r_primary": 0.0,
            "r_tiebreak": 0.0,
            "r_trainset_novelty": 0.0,
            "r_dense": 0.0,
            "r_prev_group": 0.0,
            "r_best_group": 0.0,
            "r_goal_best": 0.0,
            "r_goal_match": 0.0,
            "r_generalization": 0.0,
            "r_structure_group": 0.0,
            "r_structure_archive": 0.0,
            "r_batch_elite": 0.0,
            "r_repeat_family": 0.0,
            "r_plain_fuse_penalty": 0.0,
            "r_no_progress_penalty": 0.0,
            "batch_elite_rank": None,
            "batch_elite_tier": "none",
            "batch_elite_threshold_passed": False,
            "novel_vs_trainset_family": False,
            "novel_vs_trainset_graph": False,
            "archive_snapshot_family_freq": 0,
            "batch_same_family_count": 0,
            "reward_target_metric": REWARD_TARGET_METRIC,
            "reward_target_value": None,
            "goal_tag_hit_count": 0,
            "goal_tag_total_count": 0,
            "goal_tag_hit_rate": 0.0,
        },
        "error": error,
    }


def _is_trainable_candidate(res: Dict[str, Any], graph_info) -> bool:
    return bool(
        graph_info
        and graph_info.parse_ok
        and res.get("built_ok")
        and res.get("forward_shape_ok")
        and res.get("backward_ok")
        and res.get("loss_drop_ok")
    )


def _apply_trainability_clamp(res: Dict[str, Any], reward_value: float, graph_info) -> float:
    parse_ok = bool(graph_info and graph_info.parse_ok)
    if not parse_ok:
        return min(reward_value, -0.25)
    if not res.get("built_ok"):
        build_partial = float(res.get("r_build_partial", 0.0))
        return min(reward_value, -0.8 + build_partial)
    if not res.get("forward_shape_ok"):
        return min(reward_value, -0.50)
    if not res.get("backward_ok"):
        return min(reward_value, -0.10)
    if not res.get("loss_drop_ok"):
        return min(reward_value, 0.0)
    return reward_value


def _archive_rarity_bonus(archive_snapshot_family_freq: int) -> float:
    if archive_snapshot_family_freq <= 0:
        return STRUCTURE_ARCHIVE_RARITY_STRONG_BONUS
    if archive_snapshot_family_freq == 1:
        return STRUCTURE_ARCHIVE_RARITY_MEDIUM_BONUS
    if archive_snapshot_family_freq <= 3:
        return STRUCTURE_ARCHIVE_RARITY_LIGHT_BONUS
    return 0.0


def _structure_progress_components(
    graph_info,
    *,
    batch_same_family_count: int,
    archive_snapshot_family_freq: int,
    novel_vs_trainset_family: bool,
    novel_vs_trainset_graph: bool,
    shallow_one_shot: bool,
) -> Tuple[float, float]:
    if not graph_info or not graph_info.parse_ok:
        return 0.0, 0.0

    r_structure_group = 0.0
    if passes_macro_structure_gate(graph_info):
        r_structure_group += STRUCTURE_MACRO_BONUS
    if is_multi_stage_architecture(graph_info):
        r_structure_group += STRUCTURE_MULTI_STAGE_BONUS
    if has_structural_motif(graph_info):
        r_structure_group += STRUCTURE_MOTIF_BONUS
    if batch_same_family_count <= 1:
        r_structure_group += STRUCTURE_BATCH_DIVERSITY_BONUS
    elif batch_same_family_count == 2:
        r_structure_group += STRUCTURE_BATCH_DIVERSITY_BONUS * 0.5
    if (
        dominant_family_hash
        and graph_info.family_hash != dominant_family_hash
        and float(dominant_family_share or 0.0) >= 0.20
    ):
        r_structure_group += STRUCTURE_NON_DOMINANT_FAMILY_BONUS
    if shallow_one_shot:
        r_structure_group = max(0.0, r_structure_group - 0.02)

    r_structure_archive = 0.0
    if novel_vs_trainset_family:
        r_structure_archive += TRAINSET_NOVEL_FAMILY_BONUS
    elif novel_vs_trainset_graph:
        r_structure_archive += TRAINSET_NOVEL_GRAPH_BONUS
    r_structure_archive += _archive_rarity_bonus(archive_snapshot_family_freq)

    return _clip(r_structure_group, 0.0, 0.14), _clip(r_structure_archive, 0.0, 0.08)


def _recompute_discovery_reward(
    res: Dict[str, Any],
    graph_info,
) -> Tuple[float, float, float]:
    r_primary = (
        float(res.get("r_dense", 0.0) or 0.0)
        + float(res.get("r_prev_group", 0.0) or 0.0)
        + float(res.get("r_best_group", 0.0) or 0.0)
        + float(res.get("r_goal_best", 0.0) or 0.0)
        + float(res.get("r_generalization", 0.0) or 0.0)
        + float(res.get("r_structure_group", 0.0) or 0.0)
        + float(res.get("r_structure_archive", 0.0) or 0.0)
        + float(res.get("r_batch_elite", 0.0) or 0.0)
        + float(res.get("r_repeat_family", 0.0) or 0.0)
        + float(res.get("r_plain_fuse_penalty", 0.0) or 0.0)
        + float(res.get("r_no_progress_penalty", 0.0) or 0.0)
    )
    r_tiebreak = float(res.get("r_goal_match", 0.0) or 0.0)
    total_reward = _clip(r_primary + r_tiebreak, -2.0, 2.0)
    total_reward = _apply_trainability_clamp(res, total_reward, graph_info)
    return total_reward, r_primary, r_tiebreak


def _attach_group_context(
    res: Dict[str, Any],
    *,
    seed_accuracy_baseline: float,
    group_context: Dict[str, Any],
) -> Dict[str, Any]:
    frozen_train_acc = _optional_float(res.get("frozen_train_acc", res.get("train_acc")))
    frozen_test_acc = _optional_float(res.get("frozen_test_acc", res.get("test_acc", res.get("val_metric"))))
    res.setdefault("test_acc", frozen_test_acc)
    res.setdefault("frozen_train_acc", frozen_train_acc)
    res.setdefault("frozen_test_acc", frozen_test_acc)
    res.setdefault("unfrozen_train_acc", None)
    res.setdefault("unfrozen_test_acc", None)
    res.setdefault("frozen_eval", None)
    res.setdefault("unfrozen_eval", None)
    res.setdefault("seed_accuracy_baseline", seed_accuracy_baseline)
    res.setdefault("seed_train_acc_gap", None)
    res.setdefault("seed_train_acc_improved", False)
    res.setdefault("accuracy_baseline", seed_accuracy_baseline)
    res.setdefault("train_acc_gain", None)
    res.setdefault("train_acc_improved", False)
    res.setdefault("group_baseline_train_acc", group_context["group_baseline_train_acc"])
    res.setdefault("group_train_acc_gain", None)
    res.setdefault("group_train_acc_improved", False)
    res.setdefault("reward_target_metric", REWARD_TARGET_METRIC)
    res.setdefault("reward_target_value", _result_reward_target_value(res))
    res.setdefault("group_baseline_reward_target_acc", group_context["group_baseline_reward_target_acc"])
    res.setdefault("group_reward_target_gain", None)
    res.setdefault("group_reward_target_improved", False)
    res.setdefault("reward_batch_index", group_context["reward_batch_index"])
    res.setdefault("reward_group_id", group_context["reward_group_id"])
    res.setdefault("group_warmup", group_context["group_warmup"])
    res.setdefault("timed_out", False)
    res.setdefault("estimated_total_seconds", None)
    res.setdefault("eval_limit_seconds", None)
    res.setdefault("warmup_dense_reward", None)
    res.setdefault("backbone_model_names", [])
    res.setdefault("best_closed_group_mean_reward_target_acc", group_context["best_closed_group_mean_reward_target_acc"])
    res.setdefault("best_closed_group_mean_train_acc", group_context["best_closed_group_mean_train_acc"])
    res.setdefault("best_closed_group_mean_test_acc", group_context["best_closed_group_mean_test_acc"])
    res.setdefault("best_reward_target_for_goal", None)
    res.setdefault("r_dense", 0.0)
    res.setdefault("r_prev_group", 0.0)
    res.setdefault("r_best_group", 0.0)
    res.setdefault("r_goal_best", 0.0)
    res.setdefault("r_goal_match", 0.0)
    res.setdefault("r_trainset_novelty", 0.0)
    res.setdefault("r_generalization", 0.0)
    res.setdefault("r_structure_group", 0.0)
    res.setdefault("r_structure_archive", 0.0)
    res.setdefault("r_batch_elite", 0.0)
    res.setdefault("r_repeat_family", 0.0)
    res.setdefault("r_plain_fuse_penalty", 0.0)
    res.setdefault("r_no_progress_penalty", 0.0)
    res.setdefault("batch_elite_rank", None)
    res.setdefault("batch_elite_tier", "none")
    res.setdefault("batch_elite_threshold_passed", False)
    res.setdefault("goal_tag_hit_count", 0)
    res.setdefault("goal_tag_total_count", 0)
    res.setdefault("goal_tag_hit_rate", 0.0)
    res.setdefault("prev_target_reward_target_acc", None)
    res.setdefault("best_target_reward_target_acc", None)
    res.setdefault("prev_target_train_acc", None)
    res.setdefault("best_target_train_acc", None)

    open_discovery = res.setdefault("open_discovery", {})
    open_discovery.setdefault("r_primary", 0.0)
    open_discovery.setdefault("r_tiebreak", 0.0)
    open_discovery.setdefault("r_trainset_novelty", res.get("r_trainset_novelty", 0.0))
    open_discovery.setdefault("r_dense", res.get("r_dense", 0.0))
    open_discovery.setdefault("r_prev_group", res.get("r_prev_group", 0.0))
    open_discovery.setdefault("r_best_group", res.get("r_best_group", 0.0))
    open_discovery.setdefault("r_goal_best", res.get("r_goal_best", 0.0))
    open_discovery.setdefault("r_goal_match", res.get("r_goal_match", 0.0))
    open_discovery.setdefault("r_generalization", res.get("r_generalization", 0.0))
    open_discovery.setdefault("r_structure_group", res.get("r_structure_group", 0.0))
    open_discovery.setdefault("r_structure_archive", res.get("r_structure_archive", 0.0))
    open_discovery.setdefault("r_batch_elite", res.get("r_batch_elite", 0.0))
    open_discovery.setdefault("r_repeat_family", res.get("r_repeat_family", 0.0))
    open_discovery.setdefault("r_plain_fuse_penalty", res.get("r_plain_fuse_penalty", 0.0))
    open_discovery.setdefault("r_no_progress_penalty", res.get("r_no_progress_penalty", 0.0))
    open_discovery.setdefault("batch_elite_rank", res.get("batch_elite_rank"))
    open_discovery.setdefault("batch_elite_tier", res.get("batch_elite_tier", "none"))
    open_discovery.setdefault("batch_elite_threshold_passed", res.get("batch_elite_threshold_passed", False))
    open_discovery.setdefault("novel_vs_trainset_family", False)
    open_discovery.setdefault("novel_vs_trainset_graph", False)
    open_discovery.setdefault("archive_snapshot_family_freq", 0)
    open_discovery.setdefault("batch_same_family_count", 0)
    open_discovery.setdefault("group_baseline_train_acc", group_context["group_baseline_train_acc"])
    open_discovery.setdefault("group_baseline_reward_target_acc", group_context["group_baseline_reward_target_acc"])
    open_discovery.setdefault("best_closed_group_mean_train_acc", group_context["best_closed_group_mean_train_acc"])
    open_discovery.setdefault("best_closed_group_mean_reward_target_acc", group_context["best_closed_group_mean_reward_target_acc"])
    open_discovery.setdefault("best_closed_group_mean_test_acc", group_context["best_closed_group_mean_test_acc"])
    open_discovery.setdefault("prev_target_train_acc", res.get("prev_target_train_acc"))
    open_discovery.setdefault("best_target_train_acc", res.get("best_target_train_acc"))
    open_discovery.setdefault("group_train_acc_gain", res.get("group_train_acc_gain"))
    open_discovery.setdefault("group_train_acc_improved", res.get("group_train_acc_improved", False))
    open_discovery.setdefault("reward_target_metric", res.get("reward_target_metric", REWARD_TARGET_METRIC))
    open_discovery.setdefault("reward_target_value", res.get("reward_target_value"))
    open_discovery.setdefault("group_reward_target_gain", res.get("group_reward_target_gain"))
    open_discovery.setdefault("group_reward_target_improved", res.get("group_reward_target_improved", False))
    open_discovery.setdefault("goal_tag_hit_count", res.get("goal_tag_hit_count", 0))
    open_discovery.setdefault("goal_tag_total_count", res.get("goal_tag_total_count", 0))
    open_discovery.setdefault("goal_tag_hit_rate", res.get("goal_tag_hit_rate", 0.0))
    open_discovery.setdefault("prev_target_reward_target_acc", res.get("prev_target_reward_target_acc"))
    open_discovery.setdefault("best_target_reward_target_acc", res.get("best_target_reward_target_acc"))
    open_discovery.setdefault("reward_batch_index", group_context["reward_batch_index"])
    open_discovery.setdefault("reward_group_id", group_context["reward_group_id"])
    open_discovery.setdefault("group_warmup", group_context["group_warmup"])
    return res


def base_discovery_reward_fn(
    completion: str,
    *,
    seed_accuracy_baseline: float,
    precomputed_eval_result: Optional[Dict[str, Any]] = None,
    graph_info=None,
    batch_graph_hashes: List[str] = None,
    batch_family_hashes: List[str] = None,
    prompt_goal_tags: List[str] = None,
    archive_snapshot_family_counts: Optional[Dict[str, int]] = None,
    group_baseline_train_acc: Optional[float] = None,
    group_baseline_reward_target_acc: Optional[float] = None,
    reward_batch_index: Optional[int] = None,
    reward_group_id: Optional[int] = None,
    group_warmup: bool = False,
    completion_index: Optional[int] = None,
    batch_last_item: bool = False,
) -> Dict[str, Any]:
    block_code, init_code, forward_code = extract_completion_blocks(completion)
    backbone_model_names = _extract_backbone_model_names(init_code)
    if not block_code or not init_code or not forward_code:
        return _discovery_failure_result(
            -2.0,
            "Reconstruction failed (tags missing?)",
            seed_accuracy_baseline=seed_accuracy_baseline,
            backbone_model_names=backbone_model_names,
        )

    if "self.pattern" in forward_code:
        return _discovery_failure_result(
            -5.0,
            "CHEAT DETECTED: Accessed self.pattern inside forward block",
            seed_accuracy_baseline=seed_accuracy_baseline,
            backbone_model_names=backbone_model_names,
        )

    graph_info = graph_info or extract_graph_info(
        init_code,
        forward_code,
        legacy_patterns=SFTUtil.legacy_patterns,
    )
    effective_pattern_name = (
        graph_info.pattern_name if graph_info.has_custom_pattern_name else graph_info.suggested_pattern_name
    )
    pattern_override = graph_info.suggested_pattern_name if not graph_info.has_custom_pattern_name else ""

    final_code = reconstruct_code(completion, pattern_name_override=pattern_override)
    if not final_code:
        return _discovery_failure_result(
            -2.0,
            "Code reconstruction failed",
            seed_accuracy_baseline=seed_accuracy_baseline,
            backbone_model_names=backbone_model_names,
        )

    if precomputed_eval_result is not None:
        res = dict(precomputed_eval_result)
    else:
        res = evaluate_code_and_reward(
            final_code,
            in_shape=(1, 3, 224, 224),
            out_shape=(10,),
            prm={'lr': 0.01, 'batch': 64, 'dropout': 0.3, 'momentum': 0.9,
                 'transform': 'norm_256_flip', 'epoch': 1},
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed_accuracy_baseline=seed_accuracy_baseline,
            reward_batch_index=reward_batch_index,
            completion_index=completion_index,
            batch_last_item=batch_last_item,
        )

    if not res.get('built_ok'):
        res['r_build_partial'] = _compute_build_partial_reward(res)
    res.setdefault("backbone_model_names", backbone_model_names)

    shallow_one_shot = is_shallow_one_shot_fuse(graph_info)
    batch_same_family_count = batch_family_hashes.count(graph_info.family_hash) if batch_family_hashes and graph_info.parse_ok else 0
    archive_snapshot_family_freq = int((archive_snapshot_family_counts or {}).get(graph_info.family_hash, 0)) if graph_info.parse_ok else 0

    novel_vs_trainset_family = False
    novel_vs_trainset_graph = False
    frozen_train_acc = _optional_float(res.get("frozen_train_acc", res.get("train_acc")))
    frozen_test_acc = _optional_float(res.get("frozen_test_acc", res.get("test_acc", res.get("val_metric"))))
    unfrozen_train_acc = _optional_float(res.get("unfrozen_train_acc"))
    unfrozen_test_acc = _optional_float(res.get("unfrozen_test_acc"))
    train_acc = frozen_train_acc
    test_acc = frozen_test_acc
    reward_target_value = frozen_test_acc
    goal_key = primary_goal_key(prompt_goal_tags or [])
    best_reward_target_for_goal = best_reward_target_by_goal.get(goal_key)
    group_train_acc_gain = None
    group_train_acc_improved = False
    group_reward_target_gain = None
    group_reward_target_improved = False
    r_primary = 0.0
    r_tiebreak = 0.0
    r_dense = 0.0
    r_prev_group = 0.0
    r_best_group = 0.0
    r_goal_best = 0.0
    r_goal_match = 0.0
    r_trainset_novelty = 0.0
    r_generalization = 0.0
    r_structure_group = 0.0
    r_structure_archive = 0.0
    r_batch_elite = 0.0
    r_repeat_family = 0.0
    r_plain_fuse_penalty = 0.0
    r_no_progress_penalty = 0.0
    trainable_candidate = _is_trainable_candidate(res, graph_info)
    goal_tag_hit_count, goal_tag_total_count, goal_tag_hit_rate = _goal_tag_match_stats(graph_info, prompt_goal_tags)
    effective_group_baseline_reward_target_acc = (
        group_baseline_reward_target_acc
        if group_baseline_reward_target_acc is not None
        else prev_closed_group_mean_reward_target_acc
    )
    prev_target_train_acc = None
    best_target_train_acc = None
    prev_target_reward_target_acc = None
    best_target_reward_target_acc = None
    beat_prev_target = False
    beat_best_target = False

    if (train_acc is not None) and (group_baseline_train_acc is not None) and (not group_warmup):
        group_train_acc_gain = float(train_acc - group_baseline_train_acc)
        group_train_acc_improved = bool(group_train_acc_gain >= GROUP_IMPROVEMENT_DELTA)
    if (reward_target_value is not None) and (effective_group_baseline_reward_target_acc is not None) and (not group_warmup):
        group_reward_target_gain = float(reward_target_value - effective_group_baseline_reward_target_acc)
        group_reward_target_improved = bool(group_reward_target_gain >= GROUP_IMPROVEMENT_DELTA)

    if trainable_candidate and reward_target_value is not None:
        train_acc_value = float(train_acc or 0.0)
        reward_target_float = float(reward_target_value)
        r_dense = _clip(
            0.03 + 0.20 * reward_target_float + 0.04 * max(0.0, train_acc_value - 0.50),
            0.02,
            0.22,
        )
        if (not group_warmup) and (group_baseline_train_acc is not None):
            prev_target_train_acc = float(group_baseline_train_acc) + GROUP_IMPROVEMENT_DELTA
        if (not group_warmup) and (best_closed_group_mean_train_acc is not None):
            best_target_train_acc = float(best_closed_group_mean_train_acc) + BEST_GROUP_REFRESH_DELTA
        if (not group_warmup) and (effective_group_baseline_reward_target_acc is not None):
            prev_target_reward_target_acc = float(effective_group_baseline_reward_target_acc) + GROUP_IMPROVEMENT_DELTA
            beat_prev_target = reward_target_float >= prev_target_reward_target_acc
            r_prev_group = _clip(
                10.0 * (reward_target_float - prev_target_reward_target_acc),
                -1.8,
                1.8,
            )
        if (not group_warmup) and (best_closed_group_mean_reward_target_acc is not None):
            best_target_reward_target_acc = float(best_closed_group_mean_reward_target_acc) + BEST_GROUP_REFRESH_DELTA
            beat_best_target = reward_target_float >= best_target_reward_target_acc
            r_best_group = _clip(
                12.0 * (reward_target_float - best_target_reward_target_acc),
                -1.2,
                1.2,
            )
        if (
            (not group_warmup)
            and (best_reward_target_for_goal is not None)
            and reward_target_float >= float(best_reward_target_for_goal) + GOAL_REFRESH_DELTA
        ):
            r_goal_best = GOAL_REFRESH_BONUS
        r_goal_match = GOAL_MATCH_REWARD_SCALE * goal_tag_hit_rate
        if (
            (not group_warmup)
            and prev_target_reward_target_acc is not None
            and not beat_prev_target
        ):
            r_no_progress_penalty = NO_PROGRESS_PENALTY
        if (
            (not group_warmup)
            and dominant_family_hash
            and graph_info.parse_ok
            and graph_info.family_hash == dominant_family_hash
            and (
                best_target_reward_target_acc is None
                or not beat_best_target
            )
        ):
            r_repeat_family = REPEAT_FAMILY_PENALTY
        if (
            (not group_warmup)
            and graph_info.is_plain_parallel_triple
            and (
                prev_target_reward_target_acc is None
                or not beat_prev_target
            )
        ):
            r_plain_fuse_penalty = PLAIN_FUSE_PENALTY

        novel_vs_trainset_family = graph_info.family_hash not in train_family_hashes
        novel_vs_trainset_graph = graph_info.graph_hash not in train_graph_hashes
        if novel_vs_trainset_family:
            r_trainset_novelty = TRAINSET_NOVEL_FAMILY_BONUS
        elif novel_vs_trainset_graph:
            r_trainset_novelty = TRAINSET_NOVEL_GRAPH_BONUS
        r_structure_group, r_structure_archive = _structure_progress_components(
            graph_info,
            batch_same_family_count=batch_same_family_count,
            archive_snapshot_family_freq=archive_snapshot_family_freq,
            novel_vs_trainset_family=novel_vs_trainset_family,
            novel_vs_trainset_graph=novel_vs_trainset_graph,
            shallow_one_shot=shallow_one_shot,
        )
        if (frozen_train_acc is not None) and (frozen_test_acc is not None):
            overfit_gap = max(0.0, float(frozen_train_acc) - float(frozen_test_acc) - GENERALIZATION_GAP_TOLERANCE)
            r_generalization = _clip(
                -GENERALIZATION_PENALTY_SCALE * overfit_gap,
                GENERALIZATION_PENALTY_CAP,
                0.0,
            )

    r_primary = (
        r_dense
        + r_prev_group
        + r_best_group
        + r_goal_best
        + r_generalization
        + r_structure_group
        + r_structure_archive
        + r_batch_elite
        + r_repeat_family
        + r_plain_fuse_penalty
        + r_no_progress_penalty
    )
    r_tiebreak = r_goal_match

    warmup_dense_reward = None
    if group_warmup and trainable_candidate:
        warmup_dense_reward = _compute_warmup_dense_reward(reward_target_value)
        total_reward = float(warmup_dense_reward or 0.0)
    else:
        total_reward = _clip(r_primary + r_tiebreak, -2.0, 2.0)
        if trainable_candidate and prev_target_reward_target_acc is not None and not beat_prev_target:
            total_reward = min(total_reward, NON_IMPROVING_REWARD_CAP)
    total_reward = _apply_trainability_clamp(res, total_reward, graph_info)

    res['reward'] = total_reward
    res['test_acc'] = test_acc
    res['train_acc'] = train_acc
    res['frozen_train_acc'] = frozen_train_acc
    res['frozen_test_acc'] = frozen_test_acc
    res['unfrozen_train_acc'] = unfrozen_train_acc
    res['unfrozen_test_acc'] = unfrozen_test_acc
    res['val_metric'] = frozen_test_acc
    res['seed_accuracy_baseline'] = seed_accuracy_baseline
    res['group_baseline_train_acc'] = group_baseline_train_acc
    res['group_train_acc_gain'] = group_train_acc_gain
    res['group_train_acc_improved'] = group_train_acc_improved
    res['reward_target_metric'] = REWARD_TARGET_METRIC
    res['reward_target_value'] = reward_target_value
    res['group_baseline_reward_target_acc'] = effective_group_baseline_reward_target_acc
    res['group_reward_target_gain'] = group_reward_target_gain
    res['group_reward_target_improved'] = group_reward_target_improved
    res['reward_batch_index'] = reward_batch_index
    res['reward_group_id'] = reward_group_id
    res['group_warmup'] = group_warmup
    res['warmup_dense_reward'] = warmup_dense_reward
    res['best_closed_group_mean_reward_target_acc'] = best_closed_group_mean_reward_target_acc
    res['best_closed_group_mean_train_acc'] = best_closed_group_mean_train_acc
    res['best_closed_group_mean_test_acc'] = best_closed_group_mean_test_acc
    res['best_reward_target_for_goal'] = best_reward_target_for_goal
    res['r_dense'] = r_dense
    res['r_prev_group'] = r_prev_group
    res['r_best_group'] = r_best_group
    res['r_goal_best'] = r_goal_best
    res['r_goal_match'] = r_goal_match
    res['r_trainset_novelty'] = r_trainset_novelty
    res['r_generalization'] = r_generalization
    res['r_structure_group'] = r_structure_group
    res['r_structure_archive'] = r_structure_archive
    res['r_batch_elite'] = r_batch_elite
    res['r_repeat_family'] = r_repeat_family
    res['r_plain_fuse_penalty'] = r_plain_fuse_penalty
    res['r_no_progress_penalty'] = r_no_progress_penalty
    res['batch_elite_rank'] = None
    res['batch_elite_tier'] = "none"
    res['batch_elite_threshold_passed'] = False
    res['goal_tag_hit_count'] = goal_tag_hit_count
    res['goal_tag_total_count'] = goal_tag_total_count
    res['goal_tag_hit_rate'] = goal_tag_hit_rate
    res['prev_target_reward_target_acc'] = prev_target_reward_target_acc
    res['best_target_reward_target_acc'] = best_target_reward_target_acc
    res['prev_target_train_acc'] = prev_target_train_acc
    res['best_target_train_acc'] = best_target_train_acc
    res['signature'] = f"{normalize_pattern_name(effective_pattern_name)}_{graph_info.graph_hash[:6]}"
    res['graph_hash'] = graph_info.graph_hash
    res['family_id'] = graph_info.family_id
    res['family_expr'] = graph_info.family_expr
    res['family_hash'] = graph_info.family_hash
    res['descriptor_key'] = graph_info.descriptor_key
    res['graph_expr'] = graph_info.graph_expr
    res['pattern_name'] = effective_pattern_name
    res['suggested_pattern_name'] = graph_info.suggested_pattern_name
    res['open_discovery'] = {
        'r_primary': r_primary,
        'r_tiebreak': r_tiebreak,
        'r_trainset_novelty': r_trainset_novelty,
        'r_dense': r_dense,
        'r_prev_group': r_prev_group,
        'r_best_group': r_best_group,
        'r_goal_best': r_goal_best,
        'r_goal_match': r_goal_match,
        'r_generalization': r_generalization,
        'r_structure_group': r_structure_group,
        'r_structure_archive': r_structure_archive,
        'r_batch_elite': r_batch_elite,
        'r_repeat_family': r_repeat_family,
        'r_plain_fuse_penalty': r_plain_fuse_penalty,
        'r_no_progress_penalty': r_no_progress_penalty,
        'batch_elite_rank': None,
        'batch_elite_tier': "none",
        'batch_elite_threshold_passed': False,
        'group_baseline_train_acc': group_baseline_train_acc,
        'group_baseline_reward_target_acc': effective_group_baseline_reward_target_acc,
        'reward_target_metric': REWARD_TARGET_METRIC,
        'reward_target_value': reward_target_value,
        'best_closed_group_mean_reward_target_acc': best_closed_group_mean_reward_target_acc,
        'best_closed_group_mean_train_acc': best_closed_group_mean_train_acc,
        'best_closed_group_mean_test_acc': best_closed_group_mean_test_acc,
        'best_reward_target_for_goal': best_reward_target_for_goal,
        'goal_tag_hit_count': goal_tag_hit_count,
        'goal_tag_total_count': goal_tag_total_count,
        'goal_tag_hit_rate': goal_tag_hit_rate,
        'prev_target_reward_target_acc': prev_target_reward_target_acc,
        'best_target_reward_target_acc': best_target_reward_target_acc,
        'prev_target_train_acc': prev_target_train_acc,
        'best_target_train_acc': best_target_train_acc,
        'group_train_acc_gain': group_train_acc_gain,
        'group_train_acc_improved': group_train_acc_improved,
        'group_reward_target_gain': group_reward_target_gain,
        'group_reward_target_improved': group_reward_target_improved,
        'reward_batch_index': reward_batch_index,
        'reward_group_id': reward_group_id,
        'group_warmup': group_warmup,
        'prompt_goal_tags': list(prompt_goal_tags or []),
        'macro_structure_ok': passes_macro_structure_gate(graph_info),
        'is_multi_stage_architecture': is_multi_stage_architecture(graph_info),
        'is_shallow_one_shot_fuse': shallow_one_shot,
        'family_id': graph_info.family_id,
        'family_hash': graph_info.family_hash,
        'descriptor_key': graph_info.descriptor_key,
        'depth': graph_info.depth,
        'merges': graph_info.merges,
        'max_fan_in': graph_info.max_fan_in,
        'backbone_calls': graph_info.backbone_calls,
        'fractal_calls': graph_info.fractal_calls,
        'stem_calls': graph_info.stem_calls,
        'project_calls': graph_info.project_calls,
        'fuse_calls': graph_info.fuse_calls,
        'is_plain_parallel_triple': graph_info.is_plain_parallel_triple,
        'is_legacy_pattern_name': graph_info.is_legacy_pattern_name,
        'parse_ok': graph_info.parse_ok,
        'novel_vs_trainset_family': novel_vs_trainset_family,
        'novel_vs_trainset_graph': novel_vs_trainset_graph,
        'archive_snapshot_family_freq': archive_snapshot_family_freq,
        'batch_same_family_count': batch_same_family_count,
    }
    return res


def reward_fn(
    completion: str,
    *,
    seed_accuracy_baseline: float,
    precomputed_eval_result: Optional[Dict[str, Any]] = None,
    graph_info=None,
    batch_graph_hashes: List[str] = None,
    batch_family_hashes: List[str] = None,
    prompt_goal_tags: List[str] = None,
    archive_snapshot_family_counts: Optional[Dict[str, int]] = None,
    group_baseline_train_acc: Optional[float] = None,
    group_baseline_reward_target_acc: Optional[float] = None,
    reward_batch_index: Optional[int] = None,
    reward_group_id: Optional[int] = None,
    group_warmup: bool = False,
    completion_index: Optional[int] = None,
    batch_last_item: bool = False,
) -> Dict[str, Any]:
    """Reward open-ended motif discovery while keeping the existing XML output ABI."""
    return base_discovery_reward_fn(
        completion,
        seed_accuracy_baseline=seed_accuracy_baseline,
        precomputed_eval_result=precomputed_eval_result,
        graph_info=graph_info,
        batch_graph_hashes=batch_graph_hashes,
        batch_family_hashes=batch_family_hashes,
        prompt_goal_tags=prompt_goal_tags,
        archive_snapshot_family_counts=archive_snapshot_family_counts,
        group_baseline_train_acc=group_baseline_train_acc,
        group_baseline_reward_target_acc=group_baseline_reward_target_acc,
        reward_batch_index=reward_batch_index,
        reward_group_id=reward_group_id,
        group_warmup=group_warmup,
        completion_index=completion_index,
        batch_last_item=batch_last_item,
    )


def _apply_batch_elite_bonuses(scored_results: List[Dict[str, Any]], group_context: Dict[str, Any]) -> None:
    if group_context["group_warmup"]:
        return

    eligible: List[Tuple[float, Dict[str, Any]]] = []
    threshold = None
    if group_context["group_baseline_reward_target_acc"] is not None:
        threshold = float(group_context["group_baseline_reward_target_acc"]) + GROUP_IMPROVEMENT_DELTA

    for item in scored_results:
        res = item["result"]
        graph_info = item["graph_info"]
        reward_target_value = _result_reward_target_value(res)
        if not _is_trainable_candidate(res, graph_info):
            continue
        if reward_target_value is None:
            continue
        eligible.append((float(reward_target_value), item))

    eligible.sort(key=lambda pair: pair[0], reverse=True)
    elite_summaries: List[str] = []
    max_elites = min(len(BATCH_ELITE_SOFT_BONUSES), len(BATCH_ELITE_IMPROVING_BONUSES))
    for rank, (reward_target_float, item) in enumerate(eligible[:max_elites]):
        threshold_passed = threshold is not None and reward_target_float >= threshold
        tier = "improving" if threshold_passed else "soft"
        bonus = (
            BATCH_ELITE_IMPROVING_BONUSES[rank]
            if threshold_passed
            else BATCH_ELITE_SOFT_BONUSES[rank]
        )
        res = item["result"]
        graph_info = item["graph_info"]
        if float(res.get("r_no_progress_penalty", 0.0) or 0.0) < 0.0:
            res["r_no_progress_penalty"] = 0.0
        res["r_batch_elite"] = bonus
        res["batch_elite_rank"] = rank + 1
        res["batch_elite_tier"] = tier
        res["batch_elite_threshold_passed"] = threshold_passed
        total_reward, r_primary, r_tiebreak = _recompute_discovery_reward(res, graph_info)
        res["reward"] = total_reward
        open_discovery = res.setdefault("open_discovery", {})
        open_discovery["r_batch_elite"] = bonus
        open_discovery["r_primary"] = r_primary
        open_discovery["r_tiebreak"] = r_tiebreak
        open_discovery["batch_elite_rank"] = rank + 1
        open_discovery["batch_elite_tier"] = tier
        open_discovery["batch_elite_threshold_passed"] = threshold_passed
        item["score"] = float(total_reward)
        elite_summaries.append(
            f"#{rank + 1} target={reward_target_float:.4f} tier={tier} bonus={bonus:.3f} "
            f"struct={float(res.get('r_structure_group', 0.0) or 0.0) + float(res.get('r_structure_archive', 0.0) or 0.0):.3f}"
        )
    if elite_summaries:
        code_logger.log_to_file(
            f"[Reward Batch Elite] reward_batch_index={group_context['reward_batch_index']} "
            + "; ".join(elite_summaries)
        )

def _reward_failure_result(
    *,
    error: str,
    seed_accuracy_baseline: float,
    group_context: Dict[str, Any],
) -> Dict[str, Any]:
    return _attach_group_context(
        {
            "reward": -1.0,
            "built_ok": False,
            "forward_ok": False,
            "forward_shape_ok": False,
            "trained_step_ok": False,
            "backward_ok": False,
            "loss_start": None,
            "loss_end": None,
            "loss_drop": None,
            "loss_drop_ok": False,
            "train_acc": None,
            "val_metric": None,
            "latency_ms": None,
            "params_m": None,
            "error": error,
        },
        seed_accuracy_baseline=seed_accuracy_baseline,
        group_context=group_context,
    )


def _prepare_local_reward_entries(
    prompts,
    completions,
    *,
    seed_accuracy_baselines: List[float],
    group_context: Dict[str, Any],
    precompute_eval: bool = True,
) -> List[Dict[str, Any]]:
    runtime_rank = _distributed_rank()
    batch_graph_infos: List[Any] = []
    batch_prompt_goal_tags = [extract_prompt_goal_tags(prompt) for prompt in prompts]

    for i, completion in enumerate(completions):
        _, init_code, forward_code = extract_completion_blocks(completion)
        if init_code and forward_code:
            graph_info = extract_graph_info(
                init_code,
                forward_code,
                legacy_patterns=SFTUtil.legacy_patterns,
            )
        else:
            graph_info = None
        batch_graph_infos.append(graph_info)

    local_entries: List[Dict[str, Any]] = []
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        graph_info = batch_graph_infos[i]
        local_entries.append(
            {
                "rank": runtime_rank,
                "local_index": i,
                "prompt": prompt,
                "completion": completion,
                "graph_info": graph_info,
                "prompt_goal_tags": batch_prompt_goal_tags[i],
                "goal_key": primary_goal_key(batch_prompt_goal_tags[i]),
                "seed_accuracy_baseline": seed_accuracy_baselines[i],
                "precomputed_eval_result": None,
            }
        )
    if precompute_eval:
        _precompute_eval_results(local_entries, group_context=group_context)
    return local_entries


def _build_batched_eval_specs(
    entries: List[Dict[str, Any]],
    *,
    group_context: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    eval_cfg_builder = getattr(evaluate_code_and_reward, "_nngpt_eval_cfg_builder", None)
    batched_eval_entries: List[Dict[str, Any]] = []
    batched_eval_specs: List[Dict[str, Any]] = []

    for entry in entries:
        if entry.get("precomputed_eval_result") is not None:
            continue

        completion = str(entry.get("completion", ""))
        graph_info = entry.get("graph_info")
        block_code, init_code, forward_code = extract_completion_blocks(completion)
        if not block_code or not init_code or not forward_code:
            continue
        if "self.pattern" in forward_code or graph_info is None:
            continue

        pattern_override = graph_info.suggested_pattern_name if not graph_info.has_custom_pattern_name else ""
        final_code = reconstruct_code(completion, pattern_name_override=pattern_override)
        if not final_code:
            continue

        spec = {
            "code": final_code,
            "in_shape": (1, 3, 224, 224),
            "out_shape": (10,),
            "prm": {
                "lr": 0.01,
                "batch": 64,
                "dropout": 0.3,
                "momentum": 0.9,
                "transform": "norm_256_flip",
                "epoch": 1,
            },
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "seed_accuracy_baseline": entry["seed_accuracy_baseline"],
            "reward_batch_index": group_context["reward_batch_index"],
            "completion_index": int(entry["local_index"]),
            "batch_last_item": False,
        }
        if callable(eval_cfg_builder):
            spec["cfg"] = eval_cfg_builder(
                in_shape=(1, 3, 224, 224),
                out_shape=(10,),
                prm=spec["prm"],
                cfg=None,
            )

        batched_eval_entries.append(entry)
        batched_eval_specs.append(spec)

    if batched_eval_specs:
        batched_eval_specs[-1]["batch_last_item"] = True

    return batched_eval_entries, batched_eval_specs


def _precompute_eval_results(
    entries: List[Dict[str, Any]],
    *,
    group_context: Dict[str, Any],
) -> None:
    batched_eval_entries, batched_eval_specs = _build_batched_eval_specs(
        entries,
        group_context=group_context,
    )
    if not batched_eval_specs:
        return
    log_memory_snapshot(
        "reward/precompute_eval:start",
        group_context=group_context,
        include_all_visible_gpus=True,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    batched_eval_results = evaluate_code_and_reward_batch(batched_eval_specs)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_memory_snapshot(
        "reward/precompute_eval:end",
        group_context=group_context,
        include_all_visible_gpus=True,
    )
    for entry, eval_result in zip(batched_eval_entries, batched_eval_results):
        entry["precomputed_eval_result"] = eval_result


def _format_reward_trace_context(context: Optional[Dict[str, Any]]) -> str:
    if not isinstance(context, dict) or not context:
        return ""
    preferred_keys = (
        "freeze_backbones",
        "formal_eval_backend",
        "formal_eval_duration_seconds",
        "trainer_device",
        "trainer_in_shape",
        "dataset_out_shape",
        "forward_output_shape",
        "params_m",
        "batch",
        "epoch",
        "epoch_limit_minutes",
        "transform",
        "num_workers",
        "estimated_training_time_minutes",
        "reported_accuracy",
        "reported_duration_seconds",
    )
    parts = []
    for key in preferred_keys:
        if key in context and context[key] is not None:
            parts.append(f"{key}={context[key]!r}")
    code_trace = context.get("code_trace")
    if isinstance(code_trace, dict):
        for key in ("references_input_spec", "assigns_input_spec", "references_pattern_attr", "line_count"):
            if key in code_trace and code_trace[key] is not None:
                parts.append(f"code_trace.{key}={code_trace[key]!r}")
    return ", ".join(parts)


def _log_reward_failure_trace(entry: Dict[str, Any], res: Dict[str, Any]) -> None:
    graph_info = entry.get("graph_info")
    pattern_name = getattr(graph_info, "suggested_pattern_name", None) if graph_info is not None else None
    branches = [("root", res)]
    frozen_eval = res.get("frozen_eval")
    unfrozen_eval = res.get("unfrozen_eval")
    if isinstance(frozen_eval, dict):
        branches.append(("frozen", frozen_eval))
    if isinstance(unfrozen_eval, dict):
        branches.append(("unfrozen", unfrozen_eval))

    seen = set()
    for branch_name, payload in branches:
        error = payload.get("error")
        if not error:
            continue
        stage = payload.get("error_stage")
        hint = payload.get("error_hint")
        context = payload.get("error_context")
        dedupe_key = (branch_name, str(error), str(stage), str(hint), repr(context))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        trace_message = (
            f"[Reward Failure Trace] rank={entry['rank']} "
            f"batch_index={entry['local_index']} "
            f"branch={branch_name} "
            f"pattern={pattern_name!r} "
            f"stage={stage or 'unknown'} "
            f"error={error!r}"
        )
        if hint:
            trace_message += f" hint={hint!r}"
        formatted_context = _format_reward_trace_context(context)
        if formatted_context:
            trace_message += f" context=({formatted_context})"
        code_logger.log_to_file(trace_message)


def _score_reward_entries(
    entries: List[Dict[str, Any]],
    *,
    group_context: Dict[str, Any],
    archive_snapshot_family_counts: Dict[str, int],
) -> List[Dict[str, Any]]:
    batch_graph_hashes = [
        entry["graph_info"].graph_hash if entry.get("graph_info") and entry["graph_info"].parse_ok else "incomplete"
        for entry in entries
    ]
    batch_family_hashes = [
        entry["graph_info"].family_hash if entry.get("graph_info") and entry["graph_info"].parse_ok else "incomplete"
        for entry in entries
    ]
    scored_results: List[Dict[str, Any]] = []

    for position, entry in enumerate(entries):
        index = int(entry["local_index"])
        code_logger.log_to_file("=" * 50)
        try:
            res = reward_fn(
                entry["completion"],
                seed_accuracy_baseline=entry["seed_accuracy_baseline"],
                precomputed_eval_result=entry.get("precomputed_eval_result"),
                graph_info=entry.get("graph_info"),
                batch_graph_hashes=batch_graph_hashes,
                batch_family_hashes=batch_family_hashes,
                prompt_goal_tags=entry.get("prompt_goal_tags"),
                archive_snapshot_family_counts=archive_snapshot_family_counts,
                group_baseline_train_acc=group_context["group_baseline_train_acc"],
                group_baseline_reward_target_acc=group_context["group_baseline_reward_target_acc"],
                reward_batch_index=group_context["reward_batch_index"],
                reward_group_id=group_context["reward_group_id"],
                group_warmup=group_context["group_warmup"],
                completion_index=index,
                batch_last_item=position == (len(entries) - 1),
            )
            res = _attach_group_context(
                res,
                seed_accuracy_baseline=entry["seed_accuracy_baseline"],
                group_context=group_context,
            )
            dispatch_parts = []
            if res.get("worker_slot") is not None:
                dispatch_parts.append(f"worker_slot={res.get('worker_slot')}")
            if res.get("assigned_gpu") is not None:
                dispatch_parts.append(f"assigned_gpu={res.get('assigned_gpu')}")
            if res.get("worker_device") is not None:
                dispatch_parts.append(f"worker_device={res.get('worker_device')}")
            if dispatch_parts:
                code_logger.log_to_file(
                    f"[Reward Dispatch] rank={entry['rank']} batch_index={index}, " + ", ".join(dispatch_parts)
                )
            _log_reward_failure_trace(entry, res)
            score = float(res.get("reward", -2.0))
        except PersistentEvalWorkerError:
            raise
        except Exception as exc:
            code_logger.log_to_file(f"Reward calculation failed at rank={entry['rank']} index={index}: {exc}")
            res = _reward_failure_result(
                error=str(exc),
                seed_accuracy_baseline=entry["seed_accuracy_baseline"],
                group_context=group_context,
            )
            score = -1.0
        scored_results.append(
            {
                **entry,
                "result": res,
                "score": score,
            }
        )

    _apply_batch_elite_bonuses(scored_results, group_context)
    for item in scored_results:
        item["score"] = float(item["result"].get("reward", item.get("score", -1.0)))
    return scored_results


def _finalize_scored_results(scored_results: List[Dict[str, Any]]) -> None:
    global B_index

    current_batch_results: List[Dict[str, Any]] = []
    for item in scored_results:
        index = int(item["local_index"])
        prompt = item["prompt"]
        completion = item["completion"]
        graph_info = item["graph_info"]
        goal_key = item["goal_key"]
        res = item["result"]
        score = float(item["score"])
        sig = res.get("signature", "unknown")

        is_trainable = _is_trainable_candidate(res, graph_info)
        if is_trainable:
            current_batch_results.append(res)
            _record_current_group_trainable_sample(goal_key, res, graph_info)
            graph_archive_counts[graph_info.graph_hash] += 1
            family_archive_counts[graph_info.family_id] += 1
            family_hash_archive_counts[graph_info.family_hash] += 1
            motif_name_counts[res.get("pattern_name", graph_info.suggested_pattern_name)] += 1
            get_goal_counter(goal_graph_archive_counts, goal_key)[graph_info.graph_hash] += 1
            get_goal_counter(goal_family_hash_archive_counts, goal_key)[graph_info.family_hash] += 1
            current_best = family_metric_best.get(graph_info.family_hash, float("-inf"))
            gain_value = res.get("group_reward_target_gain")
            family_metric_best[graph_info.family_hash] = max(
                current_best,
                float(gain_value if gain_value is not None else float("-inf")),
            )

        code_logger.log_to_file(
            f"Rank {item['rank']} batch index {index}, Motif: {res.get('pattern_name')}, Signature: {sig}, Result: {res}"
        )

        should_save = (
            bool(graph_info)
            and graph_info.parse_ok
            and res.get("built_ok")
            and res.get("forward_shape_ok")
            and res.get("backward_ok")
            and res.get("loss_drop_ok")
            and not res.get("group_warmup")
            and float(res.get("group_reward_target_gain") or 0.0) >= GROUP_IMPROVEMENT_DELTA
            and saved_graph_counts[graph_info.graph_hash] == 0
            and saved_family_hash_counts[graph_info.family_hash] < family_save_cap(graph_info)
            and get_goal_counter(saved_goal_family_hash_counts, goal_key)[graph_info.family_hash] < goal_family_save_cap(graph_info)
        )

        if should_save:
            pattern_override = "" if graph_info.has_custom_pattern_name else res.get("suggested_pattern_name", "")
            block_code, init_code, forward_code = extract_completion_blocks(completion)
            if pattern_override:
                init_code = ensure_pattern_name(init_code, pattern_override)
            final_code = reconstruct_code(completion, pattern_name_override=pattern_override)
            normalized_completion = render_completion_xml(block_code, init_code, forward_code)
            out_path = run_epoch_dir(0)
            model_dir = synth_dir(out_path) / f"B{B_index}"
            model_dir.mkdir(exist_ok=True, parents=True)

            code_file = model_dir / new_nn_file
            with open(code_file, "w") as handle:
                handle.write(final_code)

            create_file(model_dir, new_out_file, normalized_completion)
            code_logger.log_to_file(f"[INFO] Saved successful code to B{B_index} (Signature: {sig})")
            saved_graph_counts[graph_info.graph_hash] += 1
            saved_family_hash_counts[graph_info.family_hash] += 1
            get_goal_counter(saved_goal_family_hash_counts, goal_key)[graph_info.family_hash] += 1
            B_index += 1

        code_logger.log_generation(prompt, completion, score, res)

    update_current_group_metrics(current_batch_results)
    group_close_result = close_reward_group_if_needed()
    if group_close_result is not None:
        code_logger.log_to_file(f"[Reward Group] {group_close_result}")
        log_memory_snapshot("reward_group:closed")


def _print_discovery_metrics() -> None:
    total_valid = sum(family_hash_archive_counts.values())
    unique_count = len(graph_archive_counts)
    unique_families = len(family_archive_counts)
    unique_skeletons = len(family_hash_archive_counts)

    if total_valid > 0:
        most_common_count = family_hash_archive_counts.most_common(1)[0][1]
        dominant_share = most_common_count / total_valid
        import math
        entropy = -sum(
            (count / total_valid) * math.log2(count / total_valid)
            for count in family_hash_archive_counts.values()
            if count > 0
        )
    else:
        dominant_share = 0.0
        entropy = 0.0

    print(
        f"\n[Discovery Metrics] Unique Graphs: {unique_count}, "
        f"Families: {unique_families}, Skeletons: {unique_skeletons}, Dominant Family Share: {dominant_share:.2%}, Entropy: {entropy:.2f}"
    )
    print(f"[Graph Archive] Top 5 Exact Graphs: {dict(graph_archive_counts.most_common(5))}")
    print(f"[Family Archive] Top 5 Family IDs: {dict(family_archive_counts.most_common(5))}")
    print(f"[Family Archive] Top 5 Skeletons: {dict(family_hash_archive_counts.most_common(5))}")
    print(f"[Motif Names] Top 5: {dict(motif_name_counts.most_common(5))}")
    goal_summary = {
        goal_key: len(counter)
        for goal_key, counter in goal_family_hash_archive_counts.items()
    }
    print(f"[Goal Skeleton Coverage] {goal_summary}")


def compute_reward(prompts, completions, **kwargs):
    import ab.gpt.TuneRLRaw as TuneRLRaw

    TuneRLRaw.clear_extraction_meta_cache()
    seed_accuracy_baselines = require_sample_accuracy_baselines(kwargs, len(completions))
    group_context = current_reward_group_context()
    log_memory_snapshot("compute_reward:start", group_context=group_context)

    try:
        expected_world_size = max(1, env_int("WORLD_SIZE", 1))
        distributed_mode = _distributed_initialized() and _distributed_world_size() > 1
        if expected_world_size > 1 and not distributed_mode:
            raise RuntimeError(
                "compute_reward expected an initialized torch.distributed process group "
                f"for WORLD_SIZE={expected_world_size}, but it is not initialized"
            )

        local_entries = _prepare_local_reward_entries(
            prompts,
            completions,
            seed_accuracy_baselines=seed_accuracy_baselines,
            group_context=group_context,
            precompute_eval=not distributed_mode,
        )
        archive_snapshot_family_counts = dict(family_hash_archive_counts)

        if not distributed_mode:
            scored_results = _score_reward_entries(
                local_entries,
                group_context=group_context,
                archive_snapshot_family_counts=archive_snapshot_family_counts,
            )
            rewards = [-1.0] * len(completions)
            for item in scored_results:
                rewards[int(item["local_index"])] = float(item["score"])
            _finalize_scored_results(scored_results)
            _print_discovery_metrics()
            return rewards

        gathered_entries = _all_gather_object(local_entries)
        rank = _distributed_rank()

        if is_main_process():
            global_entries: List[Dict[str, Any]] = []
            for rank_entries in gathered_entries:
                global_entries.extend(list(rank_entries or []))
            _precompute_eval_results(global_entries, group_context=group_context)
            scored_results = _score_reward_entries(
                global_entries,
                group_context=group_context,
                archive_snapshot_family_counts=archive_snapshot_family_counts,
            )
            _finalize_scored_results(scored_results)
            _print_discovery_metrics()

            rewards_by_rank: Dict[int, List[float]] = {
                world_rank: [-1.0] * len(gathered_entries[world_rank])
                for world_rank in range(len(gathered_entries))
            }
            for item in scored_results:
                rewards_by_rank[int(item["rank"])][int(item["local_index"])] = float(item["score"])

            broadcast_payload = {
                "rewards_by_rank": rewards_by_rank,
                "reward_state": capture_reward_runtime_state(),
            }
        else:
            broadcast_payload = None

        synced_payload = _broadcast_object(broadcast_payload, src=0)
        restore_reward_runtime_state(synced_payload.get("reward_state"))
        return list(synced_payload["rewards_by_rank"].get(rank, [-1.0] * len(completions)))
    finally:
        TuneRLRaw.clear_extraction_meta_cache()
        log_memory_snapshot(
            "compute_reward:end",
            group_context=current_reward_group_context(),
        )

PROMPT_TEMPLATE = SFTUtil.open_discovery_prompt_template

def load_rl_dataset(tokenizer):
    """Load seed tasks for open-ended architecture discovery."""
    data = api.data(task='img-classification', nn_prefixes=("rl-bb-test1",))
    if data.empty:
        print("No 'rl-bb-test1' data found, falling back to all img-classification")
        data = api.data(only_best_accuracy=True, task='img-classification', dataset='cifar-10')

    print(f"Loaded {len(data)} examples for RL")
    bootstrap_trainset_reference_library(data)

    prompts = []
    legacy_patterns = ", ".join(SFTUtil.legacy_patterns)
    goal_profiles = SFTUtil.open_discovery_goal_profiles

    for _, row in data.iterrows():
        accuracy = _coerce_accuracy_baseline(row.get('accuracy'), context="seed row accuracy")
        for profile in goal_profiles:
            user_prompt = PROMPT_TEMPLATE.format(
                accuracy=accuracy,
                skeleton_code=SFTUtil.open_discovery_skeleton_code,
                available_backbones=", ".join(SFTUtil.available_backbones),
                legacy_patterns=legacy_patterns,
                goal_name=profile["name"],
                target_tags=", ".join(profile["tags"]),
                design_brief=profile["brief"],
                module_hints=", ".join(profile["module_hints"]),
            )

            messages = [{"role": "user", "content": user_prompt}]
            prompt_str = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )

            prompts.append({
                "prompt": prompt_str,
                "accuracy": accuracy,
                "goal_name": profile["name"],
                "target_tags": ", ".join(profile["tags"]),
            })

    rl_dataset = Dataset.from_list(prompts)
    return rl_dataset.shuffle(seed=42)

def main():
    torch.cuda.empty_cache()  
    reset_reward_runtime_state()
    precision = best_mixed_precision()

    print(f"Using RL base model: {base_model}")
    print(f"[RL] Mixed precision: {precision['label']} (torch_dtype={precision['torch_dtype']})")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load RL dataset (limit for training speed)
    rl_dataset = load_rl_dataset(tokenizer)
    dataset_limit = env_int("NNGPT_RL_DATASET_LIMIT", 500)
    if len(rl_dataset) > dataset_limit:
        rl_dataset = rl_dataset.select(range(dataset_limit))

    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=precision["torch_dtype"],
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load model (merged SFT) with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=precision["torch_dtype"],
    )

    if LOAD_EXISTING_MODEL and os.path.exists(SAVED_MODEL_PATH):
        print(f"Loading extra SFT adapter from {SAVED_MODEL_PATH}...")
        model = PeftModel.from_pretrained(model, SAVED_MODEL_PATH)
        model = model.merge_and_unload()

    model = prepare_model_for_kbit_training(model)
    align_generation_head_dtype(model, precision["torch_dtype"])

    # Apply LoRA specifically for RL phase
    peft_config = LoraConfig(
        r=16, # Optimized further for memory (was 32)
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    align_generation_head_dtype(model, precision["torch_dtype"])

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads() 

    model.print_trainable_parameters()

    grpo_config = GRPOConfig(
        temperature=env_float("NNGPT_RL_TEMPERATURE", 1.0),  # Lowered from 1.3 to reduce gibberish while maintaining diversity
        learning_rate=env_float("NNGPT_RL_LR", 5e-5),
        max_completion_length=env_int("NNGPT_RL_MAX_COMPLETION_LENGTH", 1024), # Optimized to fit valid code and reduce trailing trash
        per_device_train_batch_size=1,
        gradient_accumulation_steps=env_int("NNGPT_RL_GRAD_ACCUM", 16),
        lr_scheduler_type="cosine",
        num_train_epochs=env_int("NNGPT_RL_NUM_EPOCHS", 5), # Increased from 1 to 5 to allow extensive exploration across curriculum phases
        remove_unused_columns=False,
        logging_steps=1,
        output_dir=os.getenv("NNGPT_RL_TRAINER_OUT", "./grpo_backbone_outputs"),
        eval_strategy="no",
        bf16=precision["bf16"],
        fp16=precision["fp16"],
        gradient_checkpointing=True,
        num_generations=env_int("NNGPT_RL_NUM_GENERATIONS", 8),
    )

    trainer = GRPOTrainer(
        model=model,
        train_dataset=rl_dataset,
        reward_funcs=compute_reward, 
        args=grpo_config,
    )

    print("Starting GRPO training for Backbone Search...")
    memory_monitor = start_cuda_memory_monitor("rl/trainer")
    try:
        log_memory_snapshot("rl/before_trainer_train")
        trainer.train()
    except Exception as exc:
        if is_cuda_oom_error(exc):
            log_cuda_oom_diagnostics("rl/trainer.train", exc)
        raise
    finally:
        if memory_monitor is not None:
            memory_monitor.close()
        shutdown_eval_worker()

    model_out = run_model_out()
    print(f"Saving model to {model_out}...")
    model.save_pretrained(model_out)
    print("Model saved successfully!")

    return model

if __name__ == "__main__":
    from ab.gpt.util.simple_logger import SimpleCodeLogger
    from ab.gpt.util.Reward import evaluate_code_and_reward
    from typing import Dict

    # Ensure directories exist
    log_dir = run_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    code_logger = SimpleCodeLogger(log_dir)

    # 清空旧模型目录
    print(f"Cleaning existing models in {run_epoch_dir()}...")
    shutil.rmtree(run_epoch_dir(), ignore_errors=True)

    main()
