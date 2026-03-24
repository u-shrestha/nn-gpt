import ast
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
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
    get_eval_worker_diagnostics,
    shutdown_eval_worker,
)
import ab.nn.api as api

import os
import re
import textwrap
import shutil
from pathlib import Path

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
reward_batch_index = 0
current_group_id = 0
current_group_train_acc_sum = 0.0
current_group_train_acc_count = 0
prev_closed_group_train_acc_mean: Optional[float] = None
# ==================================

SHALLOW_COLLAPSE_FAMILIES = {
    "ParallelTriple_Shallow",
    "DualBackboneFuse_Shallow",
    "TripleBackboneFuse_Shallow",
}


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


def reset_reward_runtime_state() -> None:
    global B_index
    global reward_batch_index
    global current_group_id
    global current_group_train_acc_sum
    global current_group_train_acc_count
    global prev_closed_group_train_acc_mean

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
    current_group_train_acc_sum = 0.0
    current_group_train_acc_count = 0
    prev_closed_group_train_acc_mean = None


def current_reward_group_context() -> Dict[str, Any]:
    return {
        "reward_batch_index": reward_batch_index + 1,
        "reward_group_id": current_group_id,
        "group_warmup": current_group_id == 0,
        "group_baseline_train_acc": prev_closed_group_train_acc_mean,
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


def log_memory_snapshot(stage: str, *, group_context: Optional[Dict[str, Any]] = None) -> None:
    effective_group_context = group_context or current_reward_group_context()
    cuda_allocated_gib, cuda_reserved_gib = _cuda_memory_gib()
    worker_info = get_eval_worker_diagnostics()
    worker_pid = worker_info["pid"] if worker_info else None
    print(
        "[Memory] "
        f"stage={stage} "
        f"pid={os.getpid()} "
        f"reward_batch_index={effective_group_context.get('reward_batch_index')} "
        f"reward_group_id={effective_group_context.get('reward_group_id')} "
        f"rss_gib={_format_mem_value(_read_process_rss_gib())} "
        f"cuda_allocated_gib={_format_mem_value(cuda_allocated_gib)} "
        f"cuda_reserved_gib={_format_mem_value(cuda_reserved_gib)} "
        f"worker_pid={worker_pid}"
    )


def update_current_group_train_acc(train_acc_values: List[float]) -> None:
    global current_group_train_acc_sum
    global current_group_train_acc_count

    for train_acc_value in train_acc_values:
        current_group_train_acc_sum += float(train_acc_value)
        current_group_train_acc_count += 1


def close_reward_group_if_needed() -> None:
    global reward_batch_index
    global current_group_id
    global current_group_train_acc_sum
    global current_group_train_acc_count
    global prev_closed_group_train_acc_mean

    reward_batch_index += 1
    if reward_batch_index % GROUP_BATCH_SIZE != 0:
        return
    if current_group_train_acc_count <= 0:
        raise RuntimeError(f"Reward group {current_group_id} closed without any trainable samples")

    prev_closed_group_train_acc_mean = current_group_train_acc_sum / current_group_train_acc_count
    print(
        f"[Reward Group] Closed group {current_group_id} after {GROUP_BATCH_SIZE} reward batches: "
        f"mean_train_acc={prev_closed_group_train_acc_mean:.4f}, "
        f"trainable_samples={current_group_train_acc_count}"
    )
    current_group_id += 1
    current_group_train_acc_sum = 0.0
    current_group_train_acc_count = 0


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


def _compute_warmup_dense_reward(train_acc: Optional[float]) -> Optional[float]:
    if train_acc is None:
        return None
    return max(0.05, min(0.30, 0.10 + 0.50 * float(train_acc)))


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
        "train_acc": None,
        "seed_accuracy_baseline": seed_accuracy_baseline,
        "seed_train_acc_gap": None,
        "seed_train_acc_improved": False,
        "accuracy_baseline": seed_accuracy_baseline,
        "train_acc_gain": None,
        "train_acc_improved": False,
        "group_baseline_train_acc": None,
        "group_train_acc_gain": None,
        "group_train_acc_improved": False,
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
        "open_discovery": {
            "r_primary": 0.0,
            "r_tiebreak": 0.0,
            "r_trainset_novelty": 0.0,
            "novel_vs_trainset_family": False,
            "novel_vs_trainset_graph": False,
            "archive_snapshot_family_freq": 0,
            "batch_same_family_count": 0,
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


def _attach_group_context(
    res: Dict[str, Any],
    *,
    seed_accuracy_baseline: float,
    group_context: Dict[str, Any],
) -> Dict[str, Any]:
    res.setdefault("seed_accuracy_baseline", seed_accuracy_baseline)
    res.setdefault("seed_train_acc_gap", None)
    res.setdefault("seed_train_acc_improved", False)
    res.setdefault("accuracy_baseline", seed_accuracy_baseline)
    res.setdefault("train_acc_gain", None)
    res.setdefault("train_acc_improved", False)
    res.setdefault("group_baseline_train_acc", group_context["group_baseline_train_acc"])
    res.setdefault("group_train_acc_gain", None)
    res.setdefault("group_train_acc_improved", False)
    res.setdefault("reward_batch_index", group_context["reward_batch_index"])
    res.setdefault("reward_group_id", group_context["reward_group_id"])
    res.setdefault("group_warmup", group_context["group_warmup"])
    res.setdefault("timed_out", False)
    res.setdefault("estimated_total_seconds", None)
    res.setdefault("eval_limit_seconds", None)
    res.setdefault("warmup_dense_reward", None)
    res.setdefault("backbone_model_names", [])

    open_discovery = res.setdefault("open_discovery", {})
    open_discovery.setdefault("r_primary", 0.0)
    open_discovery.setdefault("r_tiebreak", 0.0)
    open_discovery.setdefault("r_trainset_novelty", 0.0)
    open_discovery.setdefault("novel_vs_trainset_family", False)
    open_discovery.setdefault("novel_vs_trainset_graph", False)
    open_discovery.setdefault("archive_snapshot_family_freq", 0)
    open_discovery.setdefault("batch_same_family_count", 0)
    open_discovery.setdefault("group_baseline_train_acc", group_context["group_baseline_train_acc"])
    open_discovery.setdefault("group_train_acc_gain", res.get("group_train_acc_gain"))
    open_discovery.setdefault("group_train_acc_improved", res.get("group_train_acc_improved", False))
    open_discovery.setdefault("reward_batch_index", group_context["reward_batch_index"])
    open_discovery.setdefault("reward_group_id", group_context["reward_group_id"])
    open_discovery.setdefault("group_warmup", group_context["group_warmup"])
    return res


def base_discovery_reward_fn(
    completion: str,
    *,
    seed_accuracy_baseline: float,
    graph_info=None,
    batch_graph_hashes: List[str] = None,
    batch_family_hashes: List[str] = None,
    prompt_goal_tags: List[str] = None,
    archive_snapshot_family_counts: Optional[Dict[str, int]] = None,
    group_baseline_train_acc: Optional[float] = None,
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

    res = evaluate_code_and_reward(
        final_code,
        in_shape=(1, 3, 224, 224),
        out_shape=(10,),
        prm={'lr': 0.01, 'batch': 16, 'dropout': 0.3, 'momentum': 0.9,
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
    train_acc = res.get("train_acc")
    group_train_acc_gain = None
    group_train_acc_improved = False
    r_primary = 0.0
    if (train_acc is not None) and (group_baseline_train_acc is not None) and (not group_warmup):
        group_train_acc_gain = float(train_acc - group_baseline_train_acc)
        group_train_acc_improved = bool(group_train_acc_gain > 0.0)
        r_primary = max(-2.0, min(2.0, 4.0 * group_train_acc_gain))

    r_tiebreak = 0.0
    if _is_trainable_candidate(res, graph_info) and bool(group_train_acc_improved):
        novel_vs_trainset_family = graph_info.family_hash not in train_family_hashes
        novel_vs_trainset_graph = graph_info.graph_hash not in train_graph_hashes

        if novel_vs_trainset_family:
            r_tiebreak = 0.02
        elif novel_vs_trainset_graph:
            r_tiebreak = 0.01

    warmup_dense_reward = None
    if group_warmup and _is_trainable_candidate(res, graph_info):
        warmup_dense_reward = _compute_warmup_dense_reward(res.get("train_acc"))
        total_reward = float(warmup_dense_reward or 0.0)
    else:
        total_reward = max(-2.0, min(2.0, r_primary + r_tiebreak))
    total_reward = _apply_trainability_clamp(res, total_reward, graph_info)

    res['reward'] = total_reward
    res['seed_accuracy_baseline'] = seed_accuracy_baseline
    res['group_baseline_train_acc'] = group_baseline_train_acc
    res['group_train_acc_gain'] = group_train_acc_gain
    res['group_train_acc_improved'] = group_train_acc_improved
    res['reward_batch_index'] = reward_batch_index
    res['reward_group_id'] = reward_group_id
    res['group_warmup'] = group_warmup
    res['warmup_dense_reward'] = warmup_dense_reward
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
        'r_trainset_novelty': r_tiebreak,
        'group_baseline_train_acc': group_baseline_train_acc,
        'group_train_acc_gain': group_train_acc_gain,
        'group_train_acc_improved': group_train_acc_improved,
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
    graph_info=None,
    batch_graph_hashes: List[str] = None,
    batch_family_hashes: List[str] = None,
    prompt_goal_tags: List[str] = None,
    archive_snapshot_family_counts: Optional[Dict[str, int]] = None,
    group_baseline_train_acc: Optional[float] = None,
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
        graph_info=graph_info,
        batch_graph_hashes=batch_graph_hashes,
        batch_family_hashes=batch_family_hashes,
        prompt_goal_tags=prompt_goal_tags,
        archive_snapshot_family_counts=archive_snapshot_family_counts,
        group_baseline_train_acc=group_baseline_train_acc,
        reward_batch_index=reward_batch_index,
        reward_group_id=reward_group_id,
        group_warmup=group_warmup,
        completion_index=completion_index,
        batch_last_item=batch_last_item,
    )

def compute_reward(prompts, completions, **kwargs):
    import ab.gpt.TuneRLRaw as TuneRLRaw

    global B_index
    rewards = []
    TuneRLRaw.clear_extraction_meta_cache()
    seed_accuracy_baselines = require_sample_accuracy_baselines(kwargs, len(completions))
    group_context = current_reward_group_context()
    log_memory_snapshot("compute_reward:start", group_context=group_context)

    try:
        batch_graph_infos = []
        for completion in completions:
            _, init_code, forward_code = extract_completion_blocks(completion)
            if init_code and forward_code:
                batch_graph_infos.append(
                    extract_graph_info(
                        init_code,
                        forward_code,
                        legacy_patterns=SFTUtil.legacy_patterns,
                    )
                )
            else:
                batch_graph_infos.append(None)

        batch_graph_hashes = [
            info.graph_hash if info and info.parse_ok else "incomplete"
            for info in batch_graph_infos
        ]
        batch_family_hashes = [
            info.family_hash if info and info.parse_ok else "incomplete"
            for info in batch_graph_infos
        ]
        batch_prompt_goal_tags = [extract_prompt_goal_tags(prompt) for prompt in prompts]
        archive_snapshot_family_counts = dict(family_hash_archive_counts)
        scored_results = []

        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            code_logger.log_to_file("=" * 50)
            torch.cuda.empty_cache()

            try:
                graph_info = batch_graph_infos[i]
                goal_key = primary_goal_key(batch_prompt_goal_tags[i])
                res = reward_fn(
                    completion,
                    seed_accuracy_baseline=seed_accuracy_baselines[i],
                    graph_info=graph_info,
                    batch_graph_hashes=batch_graph_hashes,
                    batch_family_hashes=batch_family_hashes,
                    prompt_goal_tags=batch_prompt_goal_tags[i],
                    archive_snapshot_family_counts=archive_snapshot_family_counts,
                    group_baseline_train_acc=group_context["group_baseline_train_acc"],
                    reward_batch_index=group_context["reward_batch_index"],
                    reward_group_id=group_context["reward_group_id"],
                    group_warmup=group_context["group_warmup"],
                    completion_index=i,
                    batch_last_item=i == (len(completions) - 1),
                )
                res = _attach_group_context(
                    res,
                    seed_accuracy_baseline=seed_accuracy_baselines[i],
                    group_context=group_context,
                )
                score = res.get('reward', -2.0)
                rewards.append(score)
                scored_results.append(
                    {
                        "index": i,
                        "prompt": prompt,
                        "completion": completion,
                        "graph_info": graph_info,
                        "goal_key": goal_key,
                        "result": res,
                        "score": score,
                    }
                )
            except PersistentEvalWorkerError:
                raise
            except Exception as e:
                code_logger.log_to_file(f"Reward calculation failed at index {i}: {e}")
                rewards.append(-1.0)
                failure_result = _attach_group_context(
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
                        "error": str(e),
                    },
                    seed_accuracy_baseline=seed_accuracy_baselines[i],
                    group_context=group_context,
                )
                scored_results.append(
                    {
                        "index": i,
                        "prompt": prompt,
                        "completion": completion,
                        "graph_info": batch_graph_infos[i],
                        "goal_key": primary_goal_key(batch_prompt_goal_tags[i]),
                        "result": failure_result,
                        "score": -1.0,
                    }
                )

        current_batch_train_accs: List[float] = []
        for item in scored_results:
            i = item["index"]
            prompt = item["prompt"]
            completion = item["completion"]
            graph_info = item["graph_info"]
            goal_key = item["goal_key"]
            res = item["result"]
            score = item["score"]
            sig = res.get('signature', 'unknown')

            is_trainable = _is_trainable_candidate(res, graph_info)
            if is_trainable:
                train_acc_value = res.get("train_acc")
                if train_acc_value is not None:
                    current_batch_train_accs.append(float(train_acc_value))
                graph_archive_counts[graph_info.graph_hash] += 1
                family_archive_counts[graph_info.family_id] += 1
                family_hash_archive_counts[graph_info.family_hash] += 1
                motif_name_counts[res.get('pattern_name', graph_info.suggested_pattern_name)] += 1
                get_goal_counter(goal_graph_archive_counts, goal_key)[graph_info.graph_hash] += 1
                get_goal_counter(goal_family_hash_archive_counts, goal_key)[graph_info.family_hash] += 1
                current_best = family_metric_best.get(graph_info.family_hash, float("-inf"))
                gain_value = res.get("group_train_acc_gain")
                family_metric_best[graph_info.family_hash] = max(
                    current_best,
                    float(gain_value if gain_value is not None else float("-inf")),
                )

            code_logger.log_to_file(
                f"Batch index {i}, Motif: {res.get('pattern_name')}, Signature: {sig}, Result: {res}"
            )

            should_save = (
                bool(graph_info)
                and graph_info.parse_ok
                and res.get('built_ok')
                and res.get('forward_shape_ok')
                and res.get('backward_ok')
                and res.get('loss_drop_ok')
                and not res.get("group_warmup")
                and float(res.get("group_train_acc_gain") or 0.0) > 0.0
                and saved_graph_counts[graph_info.graph_hash] == 0
                and saved_family_hash_counts[graph_info.family_hash] < family_save_cap(graph_info)
                and get_goal_counter(saved_goal_family_hash_counts, goal_key)[graph_info.family_hash] < goal_family_save_cap(graph_info)
            )

            if should_save:
                pattern_override = "" if graph_info.has_custom_pattern_name else res.get('suggested_pattern_name', '')
                block_code, init_code, forward_code = extract_completion_blocks(completion)
                if pattern_override:
                    init_code = ensure_pattern_name(init_code, pattern_override)
                final_code = reconstruct_code(completion, pattern_name_override=pattern_override)
                normalized_completion = render_completion_xml(block_code, init_code, forward_code)
                out_path = run_epoch_dir(0)
                model_dir = synth_dir(out_path) / f"B{B_index}"
                model_dir.mkdir(exist_ok=True, parents=True)

                code_file = model_dir / new_nn_file
                with open(code_file, 'w') as f:
                    f.write(final_code)

                create_file(model_dir, new_out_file, normalized_completion)
                code_logger.log_to_file(f"[INFO] Saved successful code to B{B_index} (Signature: {sig})")
                saved_graph_counts[graph_info.graph_hash] += 1
                saved_family_hash_counts[graph_info.family_hash] += 1
                get_goal_counter(saved_goal_family_hash_counts, goal_key)[graph_info.family_hash] += 1
                B_index += 1

            code_logger.log_generation(prompt, completion, score, res)

        update_current_group_train_acc(current_batch_train_accs)
        close_reward_group_if_needed()

        # 计算开放式架构多样性指标
        total_valid = sum(family_hash_archive_counts.values())
        unique_count = len(graph_archive_counts)
        unique_families = len(family_archive_counts)
        unique_skeletons = len(family_hash_archive_counts)
        
        if total_valid > 0:
            most_common_count = family_hash_archive_counts.most_common(1)[0][1]
            dominant_share = most_common_count / total_valid

            import math
            entropy = -sum((count/total_valid) * math.log2(count/total_valid) for count in family_hash_archive_counts.values() if count > 0)
        else:
            dominant_share = 0
            entropy = 0

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
        return rewards
    finally:
        shutdown_eval_worker()
        TuneRLRaw.clear_extraction_meta_cache()
        log_memory_snapshot(
            "compute_reward:end",
            group_context=group_context,
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

    print(f"Using RL base model: {base_model}")
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
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load model (merged SFT) with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )

    if LOAD_EXISTING_MODEL and os.path.exists(SAVED_MODEL_PATH):
        print(f"Loading extra SFT adapter from {SAVED_MODEL_PATH}...")
        model = PeftModel.from_pretrained(model, SAVED_MODEL_PATH)
        model = model.merge_and_unload()

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
        bf16=True,
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
    try:
        trainer.train()
    finally:
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
