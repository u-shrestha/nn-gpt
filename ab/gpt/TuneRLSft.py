import os
import shutil
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
SFT_EVAL_BATCH_SIZE = 32
SFT_EVAL_TRAIN_SUBSET = 256
SFT_EVAL_VAL_SUBSET = 128
SFT_EVAL_TRAIN_STEPS = 8
SFT_EVAL_VAL_BATCHES = 2
SFT_EVAL_DATA_ROOT = "data_v2"
SFT_EVAL_DOWNLOAD = True
SFT_VAL_METRIC_BASELINE = 0.10

# Local desktop cache roots.
# Keep this block on the local machine if you want Hugging Face downloads/cache on
# the mounted disk. On the server, if the model is already placed under
# `out/llm/ABrain/NNGPT-Backbone-deepseek-coder-6.7b-instruct`, you can comment
# out this whole block and the script will still load from `out/llm` first.
# SFT_HF_HOME = "/media/xi/Data/hf-cache"
# SFT_HF_HUB_CACHE = "/media/xi/Data/hf-cache/hub"
# SFT_TRANSFORMERS_CACHE = "/media/xi/Data/hf-cache/transformers"

# os.environ["HF_HOME"] = SFT_HF_HOME
# os.environ["HF_HUB_CACHE"] = SFT_HF_HUB_CACHE
# os.environ["HUGGINGFACE_HUB_CACHE"] = SFT_HF_HUB_CACHE
# os.environ["TRANSFORMERS_CACHE"] = SFT_TRANSFORMERS_CACHE

import ab.gpt.TuneRL as TuneRL
import ab.gpt.TuneRLRaw as TuneRLRaw
import ab.gpt.util.Reward as RewardUtil
import ab.gpt.util.SFTUtil as SFTUtil


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


def _cifar_eval_error(error: Exception, *, accuracy_baseline: float | None = None) -> Dict[str, Any]:
    error_type = type(error).__name__
    error_msg = f"{error_type}: {error}"
    return {
        "reward": 0.0,
        "components": {
            "reward": 0.0,
            "r_build": 0.0,
            "r_forward_shape": 0.0,
            "r_backward": 0.0,
            "r_loss_drop": 0.0,
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
        "forward_shape_ok": False,
        "trained_step_ok": False,
        "backward_ok": False,
        "loss_start": None,
        "loss_end": None,
        "loss_drop": None,
        "loss_drop_ok": False,
        "train_acc": None,
        "accuracy_baseline": accuracy_baseline,
        "train_acc_gain": None,
        "train_acc_improved": False,
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
    accuracy_baseline: float | None = None,
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
            train_steps=SFT_EVAL_TRAIN_STEPS,
            max_val_batches=SFT_EVAL_VAL_BATCHES,
            measure_latency=True,
            kl_div=None,
            critic_fn=None,
            weights=None,
        )

    try:
        builder = RewardUtil.build_fn_from_code(code, in_shape, out_shape, prm, device)
    except Exception as exc:
        return _cifar_eval_error(exc, accuracy_baseline=accuracy_baseline)

    try:
        eval_batch = int(prm.get("batch") or SFT_EVAL_BATCH_SIZE)
        eval_batch = max(1, min(eval_batch, SFT_EVAL_BATCH_SIZE))
        train_loader, val_loader = _build_cifar10_eval_loaders(eval_batch)
        res = RewardUtil.evaluate_and_reward(
            build_fn=builder,
            train_loader=train_loader,
            val_loader=val_loader,
            val_metric_baseline=val_metric_baseline,
            accuracy_baseline=accuracy_baseline,
            cfg=cfg,
        )
        if res["reward"] == 0.0:
            res["reward"] = -1.0
        return res
    except Exception as exc:
        return _cifar_eval_error(exc, accuracy_baseline=accuracy_baseline)


def _eval_cifar_subprocess_worker(send_conn, code, in_shape, out_shape, prm, device, val_metric_baseline, accuracy_baseline, cfg):
    try:
        result = _evaluate_code_and_reward_cifar_direct(
            code,
            in_shape=in_shape,
            out_shape=out_shape,
            prm=prm,
            device=device,
            val_metric_baseline=val_metric_baseline,
            accuracy_baseline=accuracy_baseline,
            cfg=cfg,
        )
        send_conn.send(result)
    except Exception as exc:
        send_conn.send(_cifar_eval_error(exc, accuracy_baseline=accuracy_baseline))
    finally:
        send_conn.close()


def evaluate_code_and_reward_cifar(
    code: str,
    *,
    in_shape=(1, 3, 256, 256),
    out_shape=(10,),
    prm=None,
    device: str = "cpu",
    val_metric_baseline=None,
    accuracy_baseline=None,
    cfg=None,
) -> Dict[str, Any]:
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_eval_cifar_subprocess_worker,
        args=(send_conn, code, in_shape, out_shape, prm, device, val_metric_baseline, accuracy_baseline, cfg),
    )

    try:
        process.start()
        send_conn.close()  # Parent must close the write end so read end receives EOF properly
        if recv_conn.poll(300):
            res = recv_conn.recv()
        else:
            raise TimeoutError("Evaluation timed out after 300 seconds.")
        process.join(5)
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
                "r_forward_shape": 0.0,
                "r_backward": 0.0,
                "r_loss_drop": 0.0,
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
            "forward_shape_ok": False,
            "trained_step_ok": False,
            "backward_ok": False,
            "loss_start": None,
            "loss_end": None,
            "loss_drop": None,
            "loss_drop_ok": False,
            "train_acc": None,
            "accuracy_baseline": accuracy_baseline,
            "train_acc_gain": None,
            "train_acc_improved": False,
            "latency_ms": None,
            "params_m": None,
            "error": str(exc),
        }
    finally:
        recv_conn.close()


def _is_trainable_architecture(res: Dict[str, Any], graph_info) -> bool:
    discovery_meta = res.get("open_discovery", {})
    parse_ok = bool(getattr(graph_info, "parse_ok", False) or discovery_meta.get("parse_ok", False))
    return bool(
        parse_ok
        and res.get("built_ok")
        and res.get("forward_shape_ok")
        and res.get("backward_ok")
        and res.get("loss_drop_ok")
    )


def _reapply_trainability_clamp(res: Dict[str, Any], reward_value: float, graph_info) -> float:
    discovery_meta = res.get("open_discovery", {})
    parse_ok = bool(getattr(graph_info, "parse_ok", False) or discovery_meta.get("parse_ok", False))

    if not parse_ok:
        reward_value = min(reward_value, -0.25)
    if not res.get("built_ok"):
        build_partial = float(res.get("r_build_partial", 0.0))
        reward_value = min(reward_value, -0.8 + build_partial)
    elif not res.get("forward_shape_ok"):
        reward_value = min(reward_value, -0.50)
    elif not res.get("backward_ok"):
        reward_value = min(reward_value, -0.10)
    elif not res.get("loss_drop_ok"):
        reward_value = min(reward_value, 0.0)
    return reward_value


def sft_reward_fn(
    completion: str,
    *,
    accuracy_baseline: float,
    graph_info=None,
    batch_graph_hashes: List[str] = None,
    batch_family_hashes: List[str] = None,
    prompt_goal_tags: List[str] = None,
    archive_snapshot_family_counts: Dict[str, int] = None,
):
    res = TuneRLRaw.raw_reward_fn(
        completion,
        accuracy_baseline=accuracy_baseline,
        graph_info=graph_info,
        batch_graph_hashes=batch_graph_hashes,
        batch_family_hashes=batch_family_hashes,
        prompt_goal_tags=prompt_goal_tags,
        archive_snapshot_family_counts=archive_snapshot_family_counts,
    )
    res["reward"] = _reapply_trainability_clamp(res, float(res.get("reward", -2.0)), graph_info)
    res["anti_collapse"] = {
        "goal_key": TuneRL.primary_goal_key(prompt_goal_tags),
        "trainable_ok": _is_trainable_architecture(res, graph_info),
        "anti_collapse_delta": 0.0,
    }
    return res


SFT_DISCOVERY_PROMPT_TEMPLATE = SFTUtil.open_discovery_rl_prompt_template


def load_rl_dataset_sft(tokenizer) -> TuneRL.Dataset:
    """Load SFT-aligned RL prompts without a pre-filled assistant prefix."""
    from datasets import Dataset
    from ab.gpt.TuneRLRaw import BLOCK_SIGNATURE, FORWARD_SIGNATURE, INIT_SIGNATURE

    data = TuneRL.api.data(task="img-classification", nn_prefixes=("rl-bb-test1",))
    if data.empty:
        print("No 'rl-bb-test1' data found, falling back to all img-classification")
        data = TuneRL.api.data(only_best_accuracy=True, task="img-classification", dataset="cifar-10")

    print(f"Loaded {len(data)} examples for SFT RL")
    TuneRL.bootstrap_trainset_reference_library(data)

    prompts = []
    legacy_patterns = ", ".join(SFTUtil.legacy_patterns)

    for _, row in data.iterrows():
        accuracy = TuneRL._coerce_accuracy_baseline(row.get("accuracy"), context="seed row accuracy")
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
    """Initialize logging and reset extraction cache."""
    TuneRLRaw.EXTRACTION_META_CACHE.clear()

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
        f"train_steps={SFT_EVAL_TRAIN_STEPS}, val_batches={SFT_EVAL_VAL_BATCHES}, "
        f"baseline={SFT_VAL_METRIC_BASELINE:.2f}"
    )
    print(f"[SFT RL] Save RL adapter: {SFT_SAVE_RL_MODEL}")

    run_sft_training()


if __name__ == "__main__":
    main()
