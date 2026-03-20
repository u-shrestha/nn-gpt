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
from ab.gpt.util.Reward import evaluate_code_and_reward
import ab.nn.api as api

import os
import re
import textwrap
import shutil
from pathlib import Path

from ab.gpt.util.simple_logger import SimpleCodeLogger
from typing import Tuple, Any, List, Dict
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

# ===== Configuration Options =====
base_model = "ABrain/NNGPT-Backbone-deepseek-coder-6.7b-instruct" # 使用新的 Backbone 模型
LOAD_EXISTING_MODEL = False  # Model is already merged
SAVED_MODEL_PATH = "rl_backbone_model" 
B_index = 0
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
    return 4 if passes_macro_structure_gate(graph_info) else 1


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


def extract_prompt_goal_tags(prompt_text: str) -> List[str]:
    if not prompt_text:
        return []
    match = re.search(r"Discovery Target Tags:\s*([A-Za-z0-9_, \-]+)", prompt_text)
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
    if not graph_info:
        return 1
    if graph_info.stem_calls and graph_info.project_calls:
        return 3
    if is_multi_stage_architecture(graph_info):
        return 2
    return 1


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


def reward_fn(
    completion: str,
    *,
    graph_info=None,
    batch_graph_hashes: List[str] = None,
    batch_family_hashes: List[str] = None,
    prompt_goal_tags: List[str] = None,
) -> Dict[str, Any]:
    """Reward open-ended motif discovery while keeping the existing XML output ABI."""
    block_code, init_code, forward_code = extract_completion_blocks(completion)
    if not block_code or not init_code or not forward_code:
        return {"reward": -2.0, "built_ok": False, "error": "Reconstruction failed (tags missing?)"}

    if "self.pattern" in forward_code:
        return {"reward": -5.0, "built_ok": False, "error": "CHEAT DETECTED: Accessed self.pattern inside forward block"}

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
        return {"reward": -2.0, "built_ok": False, "error": "Code reconstruction failed"}

    res = evaluate_code_and_reward(
        final_code,
        in_shape=(1, 3, 224, 224),
        out_shape=(10,),
        prm={'lr': 0.01, 'batch': 16, 'dropout': 0.3, 'momentum': 0.9,
             'transform': 'norm_256_flip', 'epoch': 1},
        device="cuda" if torch.cuda.is_available() else "cpu",
        val_metric_baseline=0.05,
    )

    # --- Layered build-failure partial reward ---
    # Provide gradient signal between "code doesn't parse" and "almost builds"
    if not res.get('built_ok'):
        error_str = str(res.get('error', ''))
        build_partial = 0.0
        if 'SyntaxError' in error_str:
            build_partial = -0.3  # Code has syntax errors
        elif 'NameError' in error_str or 'ImportError' in error_str:
            build_partial = -0.2  # References undefined names
        elif 'TypeError' in error_str:
            build_partial = -0.1  # Wrong argument types (close to building)
        elif 'RuntimeError' in error_str and 'shape' in error_str.lower():
            build_partial = 0.05  # Shape mismatch — very close!
        elif 'RuntimeError' in error_str:
            build_partial = 0.0
        elif error_str:  # Some other error but code was exec'd
            build_partial = -0.15
        # else: no error info, default 0.0
        res['r_build_partial'] = build_partial

    macro_structure_ok = passes_macro_structure_gate(graph_info)
    shallow_one_shot = is_shallow_one_shot_fuse(graph_info)
    r_structure = 0.0
    if not graph_info.parse_ok:
        r_structure -= 0.5
    if graph_info.is_legacy_pattern_name:
        r_structure -= 0.20
    elif graph_info.has_custom_pattern_name:
        r_structure += 0.08
    else:
        r_structure -= 0.02

    if graph_info.is_plain_parallel_triple:
        r_structure -= 1.00
    elif shallow_one_shot:
        r_structure -= 0.85

    if graph_info.family_id == "TripleBackboneFuse_Shallow":
        r_structure -= 0.80
    elif graph_info.family_id == "DualBackboneFuse_Shallow":
        r_structure -= 0.55

    if graph_info.project_calls:
        r_structure += 0.35
    if graph_info.stem_calls:
        r_structure += 0.45
    if graph_info.fractal_calls >= 2:
        r_structure += 0.30
    elif graph_info.fractal_calls == 1:
        r_structure += 0.15
    if macro_structure_ok and has_structural_motif(graph_info) and graph_info.backbone_calls >= 2 and graph_info.fuse_calls:
        r_structure += 0.15

    if graph_info.backbone_calls == 0:
        r_structure -= 0.50
    elif macro_structure_ok and graph_info.backbone_calls >= 2 and has_structural_motif(graph_info):
        r_structure += 0.30
    elif shallow_one_shot and graph_info.backbone_calls >= 2:
        r_structure -= 0.15

    if not res.get('forward_ok'):
        r_structure -= 0.35
    elif not res.get('trained_step_ok'):
        r_structure -= 0.10

    r_batch = 0.0
    if batch_graph_hashes:
        same_count = batch_graph_hashes.count(graph_info.graph_hash)
        if same_count > 1:
            r_batch -= 0.35 * (same_count - 1)
    if batch_family_hashes:
        same_family_count = batch_family_hashes.count(graph_info.family_hash)
        if same_family_count > 1:
            r_batch -= 0.40 * (same_family_count - 1)

    r_novel_family = 0.0
    r_novel_skeleton = 0.0
    r_novel_exact = 0.0
    r_name_novel = 0.0
    r_local_comp = 0.0
    r_prompt_align = 0.0
    r_goal_novel = 0.0
    prompt_goal_hits: Dict[str, bool] = {}
    goal_key = primary_goal_key(prompt_goal_tags)
    goal_graph_counts = get_goal_counter(goal_graph_archive_counts, goal_key)
    goal_family_counts = get_goal_counter(goal_family_hash_archive_counts, goal_key)

    if res.get('forward_ok') and res.get('trained_step_ok') and graph_info.parse_ok:
        for tag in prompt_goal_tags or []:
            ok = prompt_goal_satisfied(graph_info, tag)
            prompt_goal_hits[tag] = ok
            if macro_structure_ok and ok:
                r_prompt_align += 0.28
            elif ok:
                r_prompt_align += 0.06
            else:
                r_prompt_align -= 0.10 if not macro_structure_ok else 0.08
        if prompt_goal_tags and macro_structure_ok and all(prompt_goal_hits.values()):
            r_prompt_align += 0.20
        elif prompt_goal_tags and not any(prompt_goal_hits.values()):
            r_prompt_align -= 0.12

    if (
        res.get('forward_ok')
        and res.get('trained_step_ok')
        and graph_info.parse_ok
        and macro_structure_ok
    ):

        goal_family_freq = goal_family_counts.get(graph_info.family_hash, 0)
        if goal_family_freq == 0:
            r_goal_novel += 0.55
        elif goal_family_freq < 2:
            r_goal_novel += 0.10
        else:
            r_goal_novel -= 0.30

        goal_graph_freq = goal_graph_counts.get(graph_info.graph_hash, 0)
        if goal_graph_freq == 0:
            r_goal_novel += 0.10
        elif goal_graph_freq >= 2:
            r_goal_novel -= 0.10

        family_freq = family_archive_counts.get(graph_info.family_id, 0)
        if family_freq == 0:
            r_novel_family += 0.85
        elif family_freq < 3:
            r_novel_family += 0.15
        else:
            r_novel_family -= 0.65

        family_hash_freq = family_hash_archive_counts.get(graph_info.family_hash, 0)
        if family_hash_freq == 0:
            r_novel_skeleton += 0.55
        elif family_hash_freq < 2:
            r_novel_skeleton += 0.10
        else:
            r_novel_skeleton -= 0.45

        exact_freq = graph_archive_counts.get(graph_info.graph_hash, 0)
        if exact_freq == 0:
            r_novel_exact += 0.15
        elif exact_freq < 2:
            r_novel_exact += 0.05
        else:
            r_novel_exact -= 0.20

        if graph_info.has_custom_pattern_name:
            name_freq = motif_name_counts.get(effective_pattern_name, 0)
            if name_freq == 0:
                r_name_novel += 0.12
            elif name_freq < 3:
                r_name_novel += 0.03
            else:
                r_name_novel -= 0.05

        best_metric = family_metric_best.get(graph_info.family_hash, float("-inf"))
        val_metric = float(res.get('val_metric') or 0.0)
        if val_metric > best_metric + 1e-9:
            r_local_comp += 0.25

    total_reward = (
        res.get('reward', 0.0)
        + r_structure
        + r_batch
        + r_novel_family
        + r_novel_skeleton
        + r_novel_exact
        + r_name_novel
        + r_local_comp
        + r_prompt_align
        + r_goal_novel
    )
    # Treat feasibility as a hard constraint: invalid models should never outrank trainable ones.
    if not graph_info.parse_ok:
        total_reward = min(total_reward, -0.25)
    if not res.get('built_ok'):
        # Use layered clamp: allow better build-failures to score higher than worse ones
        build_partial = float(res.get('r_build_partial', 0.0))
        build_clamp = -0.8 + build_partial  # Range: [-1.1, -0.75] based on error type
        total_reward = min(total_reward, build_clamp)
    elif not res.get('forward_ok'):
        total_reward = min(total_reward, -0.50)
    elif not res.get('trained_step_ok'):
        total_reward = min(total_reward, 0.0)
    elif not macro_structure_ok:
        total_reward = min(total_reward, 0.0)

    res['reward'] = total_reward
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
        'r_structure': r_structure,
        'r_batch': r_batch,
        'r_novel_family': r_novel_family,
        'r_novel_skeleton': r_novel_skeleton,
        'r_novel_exact': r_novel_exact,
        'r_name_novel': r_name_novel,
        'r_local_comp': r_local_comp,
        'r_prompt_align': r_prompt_align,
        'r_goal_novel': r_goal_novel,
        'prompt_goal_hits': prompt_goal_hits,
        'prompt_goal_tags': list(prompt_goal_tags or []),
        'goal_key': goal_key,
        'macro_structure_ok': macro_structure_ok,
        'is_multi_stage_architecture': is_multi_stage_architecture(graph_info),
        'is_shallow_one_shot_fuse': shallow_one_shot,
        'family_id': graph_info.family_id,
        'family_hash': graph_info.family_hash,
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
    }
    return res

def compute_reward(prompts, completions, **kwargs):
    global B_index
    rewards = []

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

    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        code_logger.log_to_file("="*50)
        torch.cuda.empty_cache()

        try:
            graph_info = batch_graph_infos[i]
            goal_key = primary_goal_key(batch_prompt_goal_tags[i])
            res = reward_fn(
                completion,
                graph_info=graph_info,
                batch_graph_hashes=batch_graph_hashes,
                batch_family_hashes=batch_family_hashes,
                prompt_goal_tags=batch_prompt_goal_tags[i],
            )
            score = res.get('reward', -2.0)
            sig = res.get('signature', 'unknown')

            if (
                graph_info
                and graph_info.parse_ok
                and res.get('forward_ok')
                and res.get('trained_step_ok')
                and passes_macro_structure_gate(graph_info)
            ):
                graph_archive_counts[graph_info.graph_hash] += 1
                family_archive_counts[graph_info.family_id] += 1
                family_hash_archive_counts[graph_info.family_hash] += 1
                motif_name_counts[res.get('pattern_name', graph_info.suggested_pattern_name)] += 1
                get_goal_counter(goal_graph_archive_counts, goal_key)[graph_info.graph_hash] += 1
                get_goal_counter(goal_family_hash_archive_counts, goal_key)[graph_info.family_hash] += 1
                current_best = family_metric_best.get(graph_info.family_hash, float("-inf"))
                family_metric_best[graph_info.family_hash] = max(current_best, float(res.get('val_metric') or 0.0))

            code_logger.log_to_file(
                f"Batch index {i}, Motif: {res.get('pattern_name')}, Signature: {sig}, Result: {res}"
            )

            # Save successful models (B_index)
            should_save = (
                bool(graph_info)
                and graph_info.parse_ok
                and passes_macro_structure_gate(graph_info)
                and res.get('built_ok')
                and res.get('forward_ok')
                and res.get('trained_step_ok')
                and score > 0
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
            rewards.append(score)

        except Exception as e:
            code_logger.log_to_file(f"Reward calculation failed at index {i}: {e}")
            rewards.append(-1.0)

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

PROMPT_TEMPLATE = SFTUtil.open_discovery_prompt_template

def load_rl_dataset(tokenizer):
    """Load seed tasks for open-ended architecture discovery."""
    data = api.data(task='img-classification', nn_prefixes=("rl-bb-test1",))
    if data.empty:
        print("No 'rl-bb-test1' data found, falling back to all img-classification")
        data = api.data(only_best_accuracy=True, task='img-classification', dataset='cifar-10')

    print(f"Loaded {len(data)} examples for RL")

    prompts = []
    legacy_patterns = ", ".join(SFTUtil.legacy_patterns)
    goal_profiles = SFTUtil.open_discovery_goal_profiles

    for _, row in data.iterrows():
        accuracy = row.get('accuracy', 0.8)
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
    trainer.train()

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
