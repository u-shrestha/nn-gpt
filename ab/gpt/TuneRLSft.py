"""
SFT-based RL training entry point with anti-mode-collapse mechanisms.

Reuses TuneRLRaw's XML parsing and extraction logic, but adds:
- Higher temperature (1.3) to break SFT prior
- Cumulative generation frequency tracker (ALL generations, not just successful builds)
- Log-scale anti-collapse penalty for dominant family_hash structures
- Novel structure exploration bonus even when built_ok=False
"""
import os
import math
import shutil
from typing import Dict, List
from collections import Counter

# ── SFT-specific configuration ──────────────────────────────────────────
SFT_BASE_MODEL = "ABrain/NNGPT-Backbone-deepseek-coder-6.7b-instruct"
SFT_MODEL_OUT = "rl_backbone_model_sft"
SFT_LOG_DIR = "rl_output/sft"
SFT_EPOCH_ROOT = "out/nngpt/llm/epoch_sft"
SFT_TRAINER_OUT = "grpo_backbone_outputs/sft"
SFT_TEMPERATURE = "1.3"          # Higher than raw (0.9) to break SFT prior
SFT_NUM_GENERATIONS = "8"
SFT_GRAD_ACCUM = "8"
SFT_MAX_COMPLETION_LENGTH = "1536"

# Set env vars BEFORE importing TuneRL/TuneRLRaw (they read env at import time)
os.environ["TRANSFORMERS_CACHE"] = "out/llm"
os.environ["NNGPT_RL_LOG_DIR"] = SFT_LOG_DIR
os.environ["NNGPT_RL_EPOCH_ROOT"] = SFT_EPOCH_ROOT
os.environ["NNGPT_RL_TRAINER_OUT"] = SFT_TRAINER_OUT
os.environ["NNGPT_RL_MODEL_OUT"] = SFT_MODEL_OUT
os.environ["NNGPT_RL_MAX_COMPLETION_LENGTH"] = SFT_MAX_COMPLETION_LENGTH
os.environ["NNGPT_RL_TEMPERATURE"] = SFT_TEMPERATURE
os.environ["NNGPT_RL_NUM_GENERATIONS"] = SFT_NUM_GENERATIONS
os.environ["NNGPT_RL_GRAD_ACCUM"] = SFT_GRAD_ACCUM

import ab.gpt.TuneRL as TuneRL
import ab.gpt.TuneRLRaw as TuneRLRaw
import ab.gpt.util.SFTUtil as SFTUtil

# ── Anti-collapse state ─────────────────────────────────────────────────
# Tracks ALL generations (not just successful builds) to detect mode collapse.
global_family_gen_counts: Counter = Counter()
# Tracks how many total generations we've seen for diversity baseline.
global_total_gen_count: int = 0

# Collapse detection thresholds
COLLAPSE_FREQ_THRESHOLD = 5   # After this many same-family generations, start penalizing
COLLAPSE_PENALTY_SCALE = 0.5  # Controls how fast the penalty grows
NOVEL_EXPLORE_BONUS = 0.3     # Bonus for generating a never-seen-before family_hash


def sft_reward_fn(
    completion: str,
    *,
    graph_info=None,
    batch_graph_hashes: List[str] = None,
    batch_family_hashes: List[str] = None,
    prompt_goal_tags: List[str] = None,
):
    """
    Wraps TuneRLRaw.raw_reward_fn with additional anti-mode-collapse mechanisms.
    """
    global global_total_gen_count

    # Delegate to raw_reward_fn for all existing logic
    # (layered build-failure, tiered format clamps, diversity scaling, etc.)
    res = TuneRLRaw.raw_reward_fn(
        completion,
        graph_info=graph_info,
        batch_graph_hashes=batch_graph_hashes,
        batch_family_hashes=batch_family_hashes,
        prompt_goal_tags=prompt_goal_tags,
    )

    # Extract meta and graph info for anti-collapse logic
    meta = TuneRLRaw.extract_completion_meta(completion)
    built_ok = res.get("built_ok", False)
    family_hash = res.get("family_hash", "")

    # If no family_hash from reward_fn, try to compute one from graph_info
    if not family_hash and graph_info and hasattr(graph_info, "family_hash"):
        family_hash = graph_info.family_hash

    anti_collapse_delta = 0.0

    if family_hash:
        # ── Track ALL generations globally ──
        global_family_gen_counts[family_hash] += 1
        global_total_gen_count += 1
        freq = global_family_gen_counts[family_hash]

        # ── Anti-collapse penalty ──
        # Once a family_hash has been generated more than COLLAPSE_FREQ_THRESHOLD
        # times (across all batches), penalize it with logarithmically growing cost.
        # This pushes the model AWAY from dominant structures over time.
        if freq > COLLAPSE_FREQ_THRESHOLD:
            anti_collapse_penalty = -COLLAPSE_PENALTY_SCALE * math.log2(
                freq / COLLAPSE_FREQ_THRESHOLD
            )
            anti_collapse_delta += anti_collapse_penalty

        # ── Novel structure exploration bonus ──
        # Reward the model for trying a BRAND NEW family_hash, even if it
        # doesn't build. This is critical for breaking out of SFT collapse.
        if freq == 1:
            anti_collapse_delta += NOVEL_EXPLORE_BONUS
            if not built_ok:
                # Extra encouragement: novel structure that doesn't build yet
                # still gets a small bump to incentivize exploration
                anti_collapse_delta += 0.1

    # ── Batch uniformity penalty ──
    # If the entire batch is the same family_hash, that's extreme collapse.
    if batch_family_hashes:
        unique_families = len(set(h for h in batch_family_hashes if h and h != "incomplete"))
        total_valid = len([h for h in batch_family_hashes if h and h != "incomplete"])
        if total_valid >= 4 and unique_families <= 1:
            # Entire batch collapsed to one structure
            anti_collapse_delta -= 1.5
        elif total_valid >= 4 and unique_families <= 2:
            anti_collapse_delta -= 0.5

    res["reward"] = float(res.get("reward", -2.0)) + anti_collapse_delta
    res["anti_collapse"] = {
        "family_hash_freq": global_family_gen_counts.get(family_hash, 0),
        "total_gen_count": global_total_gen_count,
        "unique_families_seen": len(global_family_gen_counts),
        "anti_collapse_delta": anti_collapse_delta,
    }
    return res


# Clean SFT prompt: explicitly require ALL tags and remove any mention of pre-filling.
SFT_DISCOVERY_PROMPT_TEMPLATE = """
You are writing one novel image-classification architecture.

Return EXACTLY three XML blocks: `<block>...</block>`, `<init>...</init>`, and `<forward>...</forward>`.
- Do not write markdown, explanations, or prose

Discovery Track
- Track Name: {goal_name}
- Discovery Target Tags: {target_tags}
- Design Brief: {design_brief}

Available runtime helpers already exist:
- `TorchVision(model=..., in_channels=...)`
- `FractalBlock(in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob)`
- `adaptive_pool_flatten(x)`
- `self.infer_dimensions_dynamically(out_shape[0])`

Backbone rules are mandatory.
- Use EXACTLY two backbones named `self.backbone_a` and `self.backbone_b`
- Initialize both with `TorchVision(model=..., in_channels=...)`
- Both backbones must be used in `forward`
- You may add stem / project / bridge / fractal / fuse modules around them, but never replace them

Choose the two backbone models from:
[{available_backbones}]

Hard rules
1. `self.pattern` must be a NEW motif name, not one of: {legacy_patterns}
2. In `__init__`, set `self.device = device`, `self.use_amp = torch.cuda.is_available()`, and `self._input_spec = tuple(in_shape[1:])`
3. Call `self.infer_dimensions_dynamically(out_shape[0])`
4. `forward` must return classifier logits
5. Use `adaptive_pool_flatten(...)` before concatenation or classifier input
6. Do not use `if self.pattern` in `forward`
7. Avoid the plain one-shot Parallel_Triple topology
8. Use the target tags with actual graph structure, not only names
9. Do not define any new classes or helper methods besides the 3 required defs
10. Keep `forward` as straight-line assignments plus a final return
11. If unsure, still output the three XML blocks with best-effort valid Python
12. `self.backbone_a` and `self.backbone_b` must both appear in `__init__` and `forward`
13. Prefer modules named like `stem`, `project_a`, `project_b`, `bridge`, `fractal`, `fuse`
14. Prefer `TorchVision`, plain CNN blocks, or `FractalBlock`; avoid invented helper class names
15. Do not emit `import ...` lines or `class ...` definitions in the completion
16. Never define `DropConv3x3Block` or any wrapper class; write only the required defs
17. Do not omit either backbone, and do not add a third backbone

Helpful module names:
{module_hints}

Implement exactly these signatures:
<block>
{block_signature}
    ...
</block>
<init>
{init_signature}
    ...
</init>
<forward>
{forward_signature}
    ...
</forward>
"""


def load_rl_dataset_sft(tokenizer) -> TuneRL.Dataset:
    """Load RL dataset without pre-filled assistant prefix."""
    from datasets import Dataset
    import ab.gpt.util.SFTUtil as SFTUtil
    
    # We essentially copy TuneRLRaw.load_rl_dataset_raw but remove the assistant prefix
    goal_profiles = SFTUtil.open_discovery_goal_profiles
    available_backbones = SFTUtil.available_backbones
    skeleton_code = SFTUtil.open_discovery_skeleton_code
    
    from ab.gpt.TuneRLRaw import BLOCK_SIGNATURE, INIT_SIGNATURE, FORWARD_SIGNATURE

    legacy_patterns = ", ".join(TuneRLRaw.PLACEHOLDER_PATTERN_NAMES)
    accuracy = 0.8  # placeholder
    
    prompts = []
    # Similar logic to TuneRLRaw.load_rl_dataset_raw
    for _ in range(1): # Single pass through profiles
        for profile in goal_profiles:
            module_hints = (
                "self.backbone_a",
                "self.backbone_b",
                *profile["module_hints"],
            )
            user_prompt = SFT_DISCOVERY_PROMPT_TEMPLATE.format(
                accuracy=accuracy,
                skeleton_code=skeleton_code,
                available_backbones=", ".join(available_backbones),
                legacy_patterns=legacy_patterns,
                goal_name=profile["name"],
                target_tags=", ".join(profile["tags"]),
                design_brief=profile["brief"],
                module_hints=", ".join(module_hints),
                block_signature=BLOCK_SIGNATURE,
                init_signature=INIT_SIGNATURE,
                forward_signature=FORWARD_SIGNATURE,
            )

            # EMPTY assistant prefix - model must generate <block> tag itself
            messages = [
                {"role": "user", "content": user_prompt},
            ]
            prompt_str = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            prompts.append(
                {
                    "prompt": prompt_str,
                    "accuracy": accuracy,
                    "goal_name": profile["name"],
                    "target_tags": ", ".join(profile["tags"]),
                }
            )

    return Dataset.from_list(prompts).shuffle(seed=42)


def patch_sft_runtime() -> None:
    """Patch TuneRL to use SFT model and anti-collapse reward function."""
    TuneRL.base_model = SFT_BASE_MODEL
    TuneRL.SAVED_MODEL_PATH = SFT_MODEL_OUT
    TuneRL.PROMPT_TEMPLATE = SFT_DISCOVERY_PROMPT_TEMPLATE
    TuneRL.extract_completion_blocks = TuneRLRaw.extract_completion_blocks_tolerant
    TuneRL.reward_fn = sft_reward_fn  # Our anti-collapse wrapper
    TuneRL.load_rl_dataset = load_rl_dataset_sft  # CUSTOM LOADER WITHOUT PREFIX
    
    # Also disable prefix in TuneRLRaw just in case anything else uses it
    TuneRLRaw.RAW_ASSISTANT_PREFIX = ""


def bootstrap_sft_runtime() -> None:
    """Initialize logging and output directories for SFT RL."""
    TuneRLRaw.EXTRACTION_META_CACHE.clear()
    global_family_gen_counts.clear()
    global global_total_gen_count
    global_total_gen_count = 0

    log_dir = TuneRL.run_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    TuneRL.code_logger = TuneRLRaw.RawCodeLogger(log_dir)

    print(f"Cleaning existing models in {TuneRL.run_epoch_dir()}...")
    shutil.rmtree(TuneRL.run_epoch_dir(), ignore_errors=True)


def main() -> None:
    TuneRLRaw.configure_raw_defaults()

    # Override the raw defaults with SFT-specific values
    os.environ["NNGPT_RL_LOG_DIR"] = SFT_LOG_DIR
    os.environ["NNGPT_RL_EPOCH_ROOT"] = SFT_EPOCH_ROOT
    os.environ["NNGPT_RL_TRAINER_OUT"] = SFT_TRAINER_OUT
    os.environ["NNGPT_RL_MODEL_OUT"] = SFT_MODEL_OUT
    os.environ["NNGPT_RL_TEMPERATURE"] = SFT_TEMPERATURE

    patch_sft_runtime()
    bootstrap_sft_runtime()

    print(f"[SFT RL] Base model: {SFT_BASE_MODEL}")
    print(f"[SFT RL] Temperature: {SFT_TEMPERATURE}")
    print(f"[SFT RL] Anti-collapse: threshold={COLLAPSE_FREQ_THRESHOLD}, "
          f"scale={COLLAPSE_PENALTY_SCALE}, explore_bonus={NOVEL_EXPLORE_BONUS}")

    TuneRL.main()


if __name__ == "__main__":
    main()
