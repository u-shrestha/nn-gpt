import math
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


# ── SFT runtime configuration ─────────────────────────────────────────────
SFT_BASE_MODEL_ID = "ABrain/NNGPT-Backbone-deepseek-coder-6.7b-instruct"
SFT_INIT_ADAPTER = ""
SFT_LOAD_INITIAL_ADAPTER = False
SFT_SAVE_RL_MODEL = False
SFT_MODEL_OUT = "rl_backbone_model_sft"
SFT_LOG_DIR = "rl_output/sft"
SFT_EPOCH_ROOT = "out/nngpt/llm/epoch_sft"
SFT_TRAINER_OUT = "grpo_backbone_outputs/sft"
SFT_TEMPERATURE = 1.1
SFT_NUM_GENERATIONS = 8
SFT_GRAD_ACCUM = 8
SFT_MAX_COMPLETION_LENGTH = 1536
SFT_DATASET_LIMIT = 500
SFT_LR = 5e-5
SFT_NUM_EPOCHS = 5
SFT_LORA_R = 16
SFT_LORA_ALPHA = 32
SFT_LORA_DROPOUT = 0.05

# CIFAR-10 quick evaluation proxy for RL reward.
SFT_EVAL_IMAGE_SIZE = 256
SFT_EVAL_BATCH_SIZE = 8
SFT_EVAL_TRAIN_SUBSET = 64
SFT_EVAL_VAL_SUBSET = 128
SFT_EVAL_VAL_BATCHES = 2
SFT_EVAL_DATA_ROOT = "data_v2"
SFT_EVAL_DOWNLOAD = True
SFT_VAL_METRIC_BASELINE = 0.10

# Local desktop cache roots.
# Keep this block on the local machine if you want Hugging Face downloads/cache on
# the mounted disk. On the server, if the model is already placed under
# `out/llm/ABrain/NNGPT-Backbone-deepseek-coder-6.7b-instruct`, you can comment
# out this whole block and the script will still load from `out/llm` first.
SFT_HF_HOME = "/media/xi/Data/hf-cache"
SFT_HF_HUB_CACHE = "/media/xi/Data/hf-cache/hub"
SFT_TRANSFORMERS_CACHE = "/media/xi/Data/hf-cache/transformers"

os.environ["HF_HOME"] = SFT_HF_HOME
os.environ["HF_HUB_CACHE"] = SFT_HF_HUB_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = SFT_HF_HUB_CACHE
os.environ["TRANSFORMERS_CACHE"] = SFT_TRANSFORMERS_CACHE

import ab.gpt.TuneRL as TuneRL
import ab.gpt.TuneRLRaw as TuneRLRaw
import ab.gpt.util.Reward as RewardUtil
import ab.gpt.util.SFTUtil as SFTUtil


# ── Diversity state ───────────────────────────────────────────────────────
goal_family_gen_counts: Dict[str, Counter] = {}
global_total_gen_count: int = 0

COLLAPSE_FREQ_THRESHOLD = 5
COLLAPSE_PENALTY_SCALE = 0.5
NOVEL_EXPLORE_BONUS = 0.3


def _repo_model_dir(model_id: str) -> Path:
    return Path("out/llm") / model_id


def _repo_tokenizer_dir(model_id: str) -> Path:
    return Path("out/tokenizer") / model_id


def _has_model_files(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    if not (model_dir / "config.json").exists():
        return False
    return any(
        (model_dir / filename).exists()
        for filename in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )


def _has_tokenizer_files(tokenizer_dir: Path) -> bool:
    if not tokenizer_dir.is_dir():
        return False
    return any(
        (tokenizer_dir / filename).exists()
        for filename in (
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "vocab.json",
        )
    )


def resolve_sft_model_sources() -> tuple[str, str, str]:
    repo_model_dir = _repo_model_dir(SFT_BASE_MODEL_ID)
    repo_tokenizer_dir = _repo_tokenizer_dir(SFT_BASE_MODEL_ID)

    if _has_model_files(repo_model_dir):
        if _has_tokenizer_files(repo_model_dir):
            return str(repo_model_dir), str(repo_model_dir), "out/llm"
        if _has_tokenizer_files(repo_tokenizer_dir):
            return str(repo_model_dir), str(repo_tokenizer_dir), "out/llm+out/tokenizer"
        return str(repo_model_dir), SFT_BASE_MODEL_ID, "out/llm+model-id-tokenizer"

    if _has_tokenizer_files(repo_tokenizer_dir):
        return SFT_BASE_MODEL_ID, str(repo_tokenizer_dir), "model-id+out/tokenizer"

    return SFT_BASE_MODEL_ID, SFT_BASE_MODEL_ID, "model-id"


def get_goal_generation_counter(goal_key: str) -> Counter:
    if goal_key not in goal_family_gen_counts:
        goal_family_gen_counts[goal_key] = Counter()
    return goal_family_gen_counts[goal_key]


def _build_cifar10_eval_loaders(batch_size: int):
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010),
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((SFT_EVAL_IMAGE_SIZE, SFT_EVAL_IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((SFT_EVAL_IMAGE_SIZE, SFT_EVAL_IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=SFT_EVAL_DATA_ROOT,
        train=True,
        download=SFT_EVAL_DOWNLOAD,
        transform=train_transform,
    )
    val_dataset = datasets.CIFAR10(
        root=SFT_EVAL_DATA_ROOT,
        train=False,
        download=SFT_EVAL_DOWNLOAD,
        transform=val_transform,
    )

    if 0 < SFT_EVAL_TRAIN_SUBSET < len(train_dataset):
        train_dataset = Subset(train_dataset, range(SFT_EVAL_TRAIN_SUBSET))
    if 0 < SFT_EVAL_VAL_SUBSET < len(val_dataset):
        val_dataset = Subset(val_dataset, range(SFT_EVAL_VAL_SUBSET))

    train_batch = max(1, min(batch_size, len(train_dataset)))
    val_batch = max(1, min(batch_size, len(val_dataset)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


def _cifar_eval_error(error: Exception) -> Dict[str, Any]:
    error_type = type(error).__name__
    error_msg = f"{error_type}: {error}"
    return {
        "reward": 0.0,
        "components": {
            "reward": 0.0,
            "r_build": 0.0,
            "r_forward": 0.0,
            "r_trainstep": 0.0,
            "r_metric": 0.0,
            "r_eff": 0.0,
            "r_critic": 0.0,
            "r_kl": 0.0,
        },
        "val_metric": None,
        "built_ok": False,
        "forward_ok": False,
        "trained_step_ok": False,
        "latency_ms": None,
        "params_m": None,
        "error": error_msg,
    }


def _evaluate_code_and_reward_cifar_direct(
    code: str,
    *,
    in_shape=(1, 3, 256, 256),
    out_shape=(10,),
    prm=None,
    device: str = "cpu",
    val_metric_baseline: float | None = None,
    cfg=None,
) -> Dict[str, Any]:
    if prm is None:
        prm = {"lr": 1e-2, "momentum": 0.9, "dropout": 0.3}
    defaults = {"lr": 1e-2, "momentum": 0.9, "batch": SFT_EVAL_BATCH_SIZE, "epoch": 1}
    prm = {**defaults, **prm}

    if cfg is None:
        cfg = RewardUtil.EvalConfig(
            device=device,
            input_shape=in_shape,
            n_classes=int(out_shape[0]),
            train_steps=1,
            max_val_batches=SFT_EVAL_VAL_BATCHES,
            measure_latency=True,
            kl_div=None,
            critic_fn=None,
            weights=None,
        )

    effective_baseline = float(val_metric_baseline if val_metric_baseline is not None else SFT_VAL_METRIC_BASELINE)
    effective_baseline = max(effective_baseline, SFT_VAL_METRIC_BASELINE)

    try:
        builder = RewardUtil.build_fn_from_code(code, in_shape, out_shape, prm, device)
    except Exception as exc:
        return _cifar_eval_error(exc)

    try:
        eval_batch = int(prm.get("batch") or SFT_EVAL_BATCH_SIZE)
        eval_batch = max(1, min(eval_batch, SFT_EVAL_BATCH_SIZE))
        train_loader, val_loader = _build_cifar10_eval_loaders(eval_batch)
        res = RewardUtil.evaluate_and_reward(
            build_fn=builder,
            train_loader=train_loader,
            val_loader=val_loader,
            val_metric_baseline=effective_baseline,
            cfg=cfg,
        )
        if res["reward"] == 0.0:
            res["reward"] = -1.0
        return res
    except Exception as exc:
        return _cifar_eval_error(exc)


def _eval_cifar_subprocess_worker(queue, code, in_shape, out_shape, prm, device, val_metric_baseline, cfg):
    try:
        result = _evaluate_code_and_reward_cifar_direct(
            code,
            in_shape=in_shape,
            out_shape=out_shape,
            prm=prm,
            device=device,
            val_metric_baseline=val_metric_baseline,
            cfg=cfg,
        )
        queue.put(result)
    except Exception as exc:
        queue.put({"error": str(exc), "reward": -1.0})


def evaluate_code_and_reward_cifar(
    code: str,
    *,
    in_shape=(1, 3, 256, 256),
    out_shape=(10,),
    prm=None,
    device: str = "cpu",
    val_metric_baseline=None,
    cfg=None,
) -> Dict[str, Any]:
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_eval_cifar_subprocess_worker,
        args=(queue, code, in_shape, out_shape, prm, device, val_metric_baseline, cfg),
    )

    try:
        process.start()
        res = queue.get(timeout=300)
        process.join()
        return res
    except Exception as exc:
        print(f"[Reward Subprocess] Critical Error or Timeout: {exc}")
        if process.is_alive():
            process.terminate()
            process.join()
        return {
            "reward": -1.0,
            "components": {
                "reward": -1.0,
                "r_build": 0.0,
                "r_forward": 0.0,
                "r_trainstep": 0.0,
                "r_metric": 0.0,
                "r_eff": 0.0,
                "r_critic": 0.0,
                "r_kl": 0.0,
            },
            "val_metric": None,
            "built_ok": False,
            "forward_ok": False,
            "trained_step_ok": False,
            "latency_ms": None,
            "params_m": None,
            "error": str(exc),
        }


def _is_trainable_architecture(res: Dict[str, Any], graph_info) -> bool:
    discovery_meta = res.get("open_discovery", {})
    parse_ok = bool(getattr(graph_info, "parse_ok", False) or discovery_meta.get("parse_ok", False))
    macro_structure_ok = bool(discovery_meta.get("macro_structure_ok", False))
    return bool(
        parse_ok
        and macro_structure_ok
        and res.get("built_ok")
        and res.get("forward_ok")
        and res.get("trained_step_ok")
    )


def _reapply_trainability_clamp(res: Dict[str, Any], reward_value: float, graph_info) -> float:
    discovery_meta = res.get("open_discovery", {})
    parse_ok = bool(getattr(graph_info, "parse_ok", False) or discovery_meta.get("parse_ok", False))
    macro_structure_ok = bool(discovery_meta.get("macro_structure_ok", False))

    if not parse_ok:
        reward_value = min(reward_value, -0.25)
    if not res.get("built_ok"):
        build_partial = float(res.get("r_build_partial", 0.0))
        reward_value = min(reward_value, -0.8 + build_partial)
    elif not res.get("forward_ok"):
        reward_value = min(reward_value, -0.50)
    elif not res.get("trained_step_ok"):
        reward_value = min(reward_value, 0.0)
    elif not macro_structure_ok:
        reward_value = min(reward_value, 0.0)
    return reward_value


def sft_reward_fn(
    completion: str,
    *,
    graph_info=None,
    batch_graph_hashes: List[str] = None,
    batch_family_hashes: List[str] = None,
    prompt_goal_tags: List[str] = None,
):
    """
    Keep TuneRLRaw's XML-format pressure, but only reward diversity for viable
    CIFAR-10-ready architectures.
    """
    global global_total_gen_count

    res = TuneRLRaw.raw_reward_fn(
        completion,
        graph_info=graph_info,
        batch_graph_hashes=batch_graph_hashes,
        batch_family_hashes=batch_family_hashes,
        prompt_goal_tags=prompt_goal_tags,
    )

    family_hash = res.get("family_hash", "")
    if not family_hash and graph_info and hasattr(graph_info, "family_hash"):
        family_hash = graph_info.family_hash

    goal_key = TuneRL.primary_goal_key(prompt_goal_tags)
    trainable_ok = _is_trainable_architecture(res, graph_info)
    anti_collapse_delta = 0.0
    goal_family_freq = 0
    goal_unique_count = 0

    if family_hash and trainable_ok:
        goal_counter = get_goal_generation_counter(goal_key)
        goal_counter[family_hash] += 1
        global_total_gen_count += 1
        goal_family_freq = goal_counter[family_hash]
        goal_unique_count = len(goal_counter)

        if goal_family_freq > COLLAPSE_FREQ_THRESHOLD:
            anti_collapse_delta -= COLLAPSE_PENALTY_SCALE * math.log2(
                goal_family_freq / COLLAPSE_FREQ_THRESHOLD
            )

        if goal_family_freq == 1:
            anti_collapse_delta += NOVEL_EXPLORE_BONUS

        if batch_family_hashes:
            valid_batch_families = [h for h in batch_family_hashes if h and h != "incomplete"]
            unique_families = len(set(valid_batch_families))
            total_valid = len(valid_batch_families)
            if total_valid >= 4 and unique_families <= 1:
                anti_collapse_delta -= 1.5
            elif total_valid >= 4 and unique_families <= 2:
                anti_collapse_delta -= 0.5

    total_reward = float(res.get("reward", -2.0)) + anti_collapse_delta
    res["reward"] = _reapply_trainability_clamp(res, total_reward, graph_info)
    res["anti_collapse"] = {
        "goal_key": goal_key,
        "goal_family_hash_freq": goal_family_freq,
        "goal_unique_families_seen": goal_unique_count,
        "total_viable_generations": global_total_gen_count,
        "trainable_ok": trainable_ok,
        "anti_collapse_delta": anti_collapse_delta,
    }
    return res


SFT_DISCOVERY_PROMPT_TEMPLATE = """
You are writing one novel image-classification architecture.

Return EXACTLY three XML blocks and nothing else.
- The first non-whitespace token must be `<block>`
- The last non-whitespace token must be `</forward>`
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
- Use two DIFFERENT backbone model names
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
18. Do not simply pool the two backbones once and concatenate them only at the classifier input

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
    """Load SFT-aligned RL prompts without a pre-filled assistant prefix."""
    from datasets import Dataset
    from ab.gpt.TuneRLRaw import BLOCK_SIGNATURE, FORWARD_SIGNATURE, INIT_SIGNATURE

    data = TuneRL.api.data(task="img-classification", nn_prefixes=("rl-bb-test1",))
    if data.empty:
        print("No 'rl-bb-test1' data found, falling back to all img-classification")
        data = TuneRL.api.data(only_best_accuracy=True, task="img-classification", dataset="cifar-10")

    print(f"Loaded {len(data)} examples for SFT RL")

    prompts = []
    legacy_patterns = ", ".join(SFTUtil.legacy_patterns)

    for _, row in data.iterrows():
        accuracy = row.get("accuracy", 0.8)
        for profile in SFTUtil.open_discovery_goal_profiles:
            module_hints = (
                "self.backbone_a",
                "self.backbone_b",
                *profile["module_hints"],
            )
            user_prompt = SFT_DISCOVERY_PROMPT_TEMPLATE.format(
                accuracy=accuracy,
                skeleton_code=SFTUtil.open_discovery_skeleton_code,
                available_backbones=", ".join(SFTUtil.available_backbones),
                legacy_patterns=legacy_patterns,
                goal_name=profile["name"],
                target_tags=", ".join(profile["tags"]),
                design_brief=profile["brief"],
                module_hints=", ".join(module_hints),
                block_signature=BLOCK_SIGNATURE,
                init_signature=INIT_SIGNATURE,
                forward_signature=FORWARD_SIGNATURE,
            )

            messages = [{"role": "user", "content": user_prompt}]
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


def sft_run_epoch_dir(*args) -> Path:
    epoch_dir = Path(SFT_EPOCH_ROOT)
    for value in args:
        epoch_dir = epoch_dir / f"A{value}"
    return epoch_dir


def run_sft_training():
    import torch
    from transformers import BitsAndBytesConfig

    torch.cuda.empty_cache()

    print(f"Using RL base model: {TuneRL.base_model}")
    tokenizer_source = getattr(TuneRL, "tokenizer_source", TuneRL.base_model)
    if tokenizer_source != TuneRL.base_model:
        print(f"Using RL tokenizer: {tokenizer_source}")
    tokenizer = TuneRL.AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rl_dataset = TuneRL.load_rl_dataset(tokenizer)
    if len(rl_dataset) > SFT_DATASET_LIMIT:
        rl_dataset = rl_dataset.select(range(SFT_DATASET_LIMIT))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = TuneRL.AutoModelForCausalLM.from_pretrained(
        TuneRL.base_model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )

    if SFT_LOAD_INITIAL_ADAPTER:
        if not SFT_INIT_ADAPTER:
            raise ValueError("SFT_INIT_ADAPTER is empty, but SFT_LOAD_INITIAL_ADAPTER is True.")
        if not os.path.exists(SFT_INIT_ADAPTER):
            raise FileNotFoundError(f"Initial adapter not found: {SFT_INIT_ADAPTER}")
        print(f"Loading initial SFT adapter from {SFT_INIT_ADAPTER}...")
        model = TuneRL.PeftModel.from_pretrained(model, SFT_INIT_ADAPTER)
        model = model.merge_and_unload()

    peft_config = TuneRL.LoraConfig(
        r=SFT_LORA_R,
        lora_alpha=SFT_LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=SFT_LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = TuneRL.get_peft_model(model, peft_config)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    grpo_config = TuneRL.GRPOConfig(
        temperature=SFT_TEMPERATURE,
        learning_rate=SFT_LR,
        max_completion_length=SFT_MAX_COMPLETION_LENGTH,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=SFT_GRAD_ACCUM,
        lr_scheduler_type="cosine",
        num_train_epochs=SFT_NUM_EPOCHS,
        remove_unused_columns=False,
        logging_steps=1,
        output_dir=SFT_TRAINER_OUT,
        eval_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        num_generations=SFT_NUM_GENERATIONS,
    )

    trainer = TuneRL.GRPOTrainer(
        model=model,
        train_dataset=rl_dataset,
        reward_funcs=TuneRL.compute_reward,
        args=grpo_config,
    )

    print("Starting GRPO training for Backbone Search...")
    trainer.train()

    if SFT_SAVE_RL_MODEL:
        print(f"Saving model to {SFT_MODEL_OUT}...")
        model.save_pretrained(SFT_MODEL_OUT)
        print("Model saved successfully!")
    else:
        print("[SFT RL] Skipping RL adapter save. Next run will start from the same initial model.")

    return model


def patch_sft_runtime() -> tuple[str, str, str]:
    """Patch TuneRL to use the SFT runtime and CIFAR-aware reward."""
    model_source, tokenizer_source, source_mode = resolve_sft_model_sources()
    TuneRL.base_model = model_source
    TuneRL.tokenizer_source = tokenizer_source
    TuneRL.LOAD_EXISTING_MODEL = SFT_LOAD_INITIAL_ADAPTER
    TuneRL.SAVED_MODEL_PATH = SFT_INIT_ADAPTER if SFT_LOAD_INITIAL_ADAPTER else ""
    TuneRL.PROMPT_TEMPLATE = SFT_DISCOVERY_PROMPT_TEMPLATE
    TuneRL.extract_completion_blocks = TuneRLRaw.extract_completion_blocks_tolerant
    TuneRL.evaluate_code_and_reward = evaluate_code_and_reward_cifar
    TuneRL.reward_fn = sft_reward_fn
    TuneRL.load_rl_dataset = load_rl_dataset_sft
    TuneRL.run_log_dir = lambda: SFT_LOG_DIR
    TuneRL.run_model_out = lambda: SFT_MODEL_OUT
    TuneRL.run_epoch_dir = sft_run_epoch_dir
    TuneRLRaw.RAW_ASSISTANT_PREFIX = ""
    return model_source, tokenizer_source, source_mode


def bootstrap_sft_runtime() -> None:
    """Initialize logging and reset diversity counters."""
    TuneRLRaw.EXTRACTION_META_CACHE.clear()
    goal_family_gen_counts.clear()

    global global_total_gen_count
    global_total_gen_count = 0

    log_dir = TuneRL.run_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    TuneRL.code_logger = TuneRLRaw.RawCodeLogger(log_dir)

    print(f"Cleaning existing models in {TuneRL.run_epoch_dir()}...")
    shutil.rmtree(TuneRL.run_epoch_dir(), ignore_errors=True)


def main() -> None:
    model_source, tokenizer_source, source_mode = patch_sft_runtime()
    bootstrap_sft_runtime()

    print(f"[SFT RL] Base model id: {SFT_BASE_MODEL_ID}")
    print(f"[SFT RL] Preferred local model dir: {_repo_model_dir(SFT_BASE_MODEL_ID)}")
    print(f"[SFT RL] Base model source ({source_mode}): {model_source}")
    if tokenizer_source != model_source:
        print(f"[SFT RL] Tokenizer source: {tokenizer_source}")
    print(f"[SFT RL] Load init adapter: {SFT_LOAD_INITIAL_ADAPTER}")
    if SFT_LOAD_INITIAL_ADAPTER:
        print(f"[SFT RL] Init adapter path: {SFT_INIT_ADAPTER}")
    print(f"[SFT RL] Temperature: {SFT_TEMPERATURE}")
    print(
        f"[SFT RL] CIFAR-10 eval: resize={SFT_EVAL_IMAGE_SIZE}, batch<={SFT_EVAL_BATCH_SIZE}, "
        f"train_subset={SFT_EVAL_TRAIN_SUBSET}, val_subset={SFT_EVAL_VAL_SUBSET}, "
        f"val_batches={SFT_EVAL_VAL_BATCHES}, baseline={SFT_VAL_METRIC_BASELINE:.2f}"
    )
    print(
        f"[SFT RL] Anti-collapse: threshold={COLLAPSE_FREQ_THRESHOLD}, "
        f"scale={COLLAPSE_PENALTY_SCALE}, explore_bonus={NOVEL_EXPLORE_BONUS}"
    )
    print(f"[SFT RL] Save RL adapter: {SFT_SAVE_RL_MODEL}")

    run_sft_training()


if __name__ == "__main__":
    main()
