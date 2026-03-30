import os
import multiprocessing as mp
import shutil
import random
from pathlib import Path
from typing import Any, Dict, List
from torch.utils.data import Dataset as TorchDataset


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
SFT_FEEDBACK_CHAR_BUDGET = 1200
SFT_LR = 5e-5
SFT_NUM_EPOCHS = 5
SFT_LORA_R = 16
SFT_LORA_ALPHA = 32
SFT_LORA_DROPOUT = 0.05

# CIFAR-10 reward evaluation proxy.
SFT_EVAL_IMAGE_SIZE = 256
SFT_EVAL_BATCH_SIZE = 32
SFT_EVAL_TRAIN_SUBSET = 256
SFT_EVAL_VAL_SUBSET = 128
SFT_EVAL_TRAIN_EPOCHS = 1
SFT_EVAL_VAL_BATCHES = 2
SFT_EVAL_FULL_TEST_ACC = True
SFT_EVAL_RUN_UNFROZEN = True
SFT_EVAL_LIMIT_SECONDS = 900
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
from ab.gpt.util.torchvision_prewarm import torchvision_prewarm_main


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


def _cifar_eval_error(error: Exception, *, seed_accuracy_baseline: float | None = None) -> Dict[str, Any]:
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
        "test_acc": None,
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
        "latency_ms": None,
        "params_m": None,
        "timed_out": False,
        "estimated_total_seconds": None,
        "eval_limit_seconds": None,
        "warmup_dense_reward": None,
        "backbone_model_names": [],
        "frozen_train_acc": None,
        "frozen_test_acc": None,
        "unfrozen_train_acc": None,
        "unfrozen_test_acc": None,
        "frozen_eval": None,
        "unfrozen_eval": None,
        "reward_target_metric": "frozen_test_acc",
        "reward_target_value": None,
        "error": error_msg,
    }


def _torch_hub_checkpoints_dir() -> Path:
    import torch

    return Path(torch.hub.get_dir()) / "checkpoints"


def _print_runtime_cache_roots() -> None:
    print(f"[SFT RL] HF_HOME={os.environ.get('HF_HOME', '')!r}")
    print(f"[SFT RL] TORCH_HOME={os.environ.get('TORCH_HOME', '')!r}")
    print(f"[SFT RL] Torch hub checkpoints dir: {_torch_hub_checkpoints_dir()}")


def _run_torchvision_prewarm(backbone_names: List[str]) -> None:
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    process = ctx.Process(
        target=torchvision_prewarm_main,
        args=(child_conn, list(backbone_names)),
    )

    TuneRL.log_memory_snapshot("sft/prewarm_before_spawn")
    process.start()
    child_conn.close()
    print(f"[TorchVision Prewarm] Starting CPU prewarm for {len(backbone_names)} backbones")

    try:
        while True:
            if not parent_conn.poll(600):
                raise RuntimeError("TorchVision prewarm timed out after 600 seconds")
            message = parent_conn.recv()
            if not isinstance(message, dict):
                raise RuntimeError(f"Unexpected TorchVision prewarm message: {message!r}")

            cmd = message.get("cmd")
            if cmd == "prewarm_ready":
                print(
                    "[TorchVision Prewarm] Ready "
                    f"pid={message['pid']} "
                    f"rss_gib={message['rss_gib']:.2f} "
                    f"torch_home={message['torch_home']!r} "
                    f"checkpoints_dir={message['checkpoints_dir']}"
                )
                continue

            if cmd == "prewarm_progress":
                print(
                    "[TorchVision Prewarm] "
                    f"backbone={message['backbone']} "
                    f"cache_hit={message['cache_hit']} "
                    f"completed={message['completed']} "
                    f"failed={message['failed']} "
                    f"rss_gib={message['rss_gib']:.2f} "
                    f"checkpoints_dir={message['checkpoints_dir']}"
                )
                continue

            if cmd == "prewarm_error":
                raise RuntimeError(
                    "TorchVision prewarm failed for "
                    f"{message['backbone']}: {message['error']}"
                )

            if cmd == "prewarm_done":
                print(
                    "[TorchVision Prewarm] Done "
                    f"completed={message['completed']} "
                    f"failed={message['failed']} "
                    f"rss_gib={message['rss_gib']:.2f} "
                    f"checkpoints_dir={message['checkpoints_dir']}"
                )
                break

            if cmd == "prewarm_fatal":
                raise RuntimeError(f"TorchVision prewarm fatal error: {message['error']}")

            raise RuntimeError(f"Unknown TorchVision prewarm message: {message!r}")

        process.join(timeout=30)
        if process.is_alive():
            raise RuntimeError("TorchVision prewarm process did not exit cleanly")
        if process.exitcode != 0:
            raise RuntimeError(f"TorchVision prewarm process exited with code {process.exitcode}")
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    TuneRL.log_memory_snapshot("sft/prewarm_after_done")


def evaluate_code_and_reward_cifar(
    code: str,
    *,
    in_shape=(1, 3, 256, 256),
    out_shape=(10,),
    prm=None,
    device: str = "cpu",
    val_metric_baseline=None,
    seed_accuracy_baseline=None,
    cfg=None,
    reward_batch_index: int | None = None,
    completion_index: int | None = None,
    batch_last_item: bool = False,
) -> Dict[str, Any]:
    try:
        import torch

        eval_device = "cuda" if torch.cuda.is_available() else "cpu"
        if prm is None:
            prm = {"lr": 1e-2, "momentum": 0.9, "dropout": 0.3}
        defaults = {"lr": 1e-2, "momentum": 0.9, "batch": SFT_EVAL_BATCH_SIZE, "epoch": 1}
        prm = {**defaults, **prm}

        if cfg is None:
            cfg = RewardUtil.EvalConfig(
                device=eval_device,
                input_shape=in_shape,
                n_classes=int(out_shape[0]),
                train_epochs=int(prm.get("epoch", SFT_EVAL_TRAIN_EPOCHS) or SFT_EVAL_TRAIN_EPOCHS),
                max_val_batches=SFT_EVAL_VAL_BATCHES,
                default_batch_size=SFT_EVAL_BATCH_SIZE,
                train_subset_size=SFT_EVAL_TRAIN_SUBSET,
                val_subset_size=SFT_EVAL_VAL_SUBSET,
                data_root=SFT_EVAL_DATA_ROOT,
                download=SFT_EVAL_DOWNLOAD,
                measure_latency=True,
                kl_div=None,
                critic_fn=None,
                weights=None,
                eval_limit_seconds=SFT_EVAL_LIMIT_SECONDS,
                run_unfrozen_backbone_eval=SFT_EVAL_RUN_UNFROZEN,
                full_test_acc=SFT_EVAL_FULL_TEST_ACC,
                reward_target_metric="frozen_test_acc",
            )
        else:
            cfg = RewardUtil.EvalConfig(
                device=eval_device,
                input_shape=cfg.input_shape,
                n_classes=cfg.n_classes,
                train_epochs=int(prm.get("epoch", getattr(cfg, "train_epochs", SFT_EVAL_TRAIN_EPOCHS)) or SFT_EVAL_TRAIN_EPOCHS),
                train_steps=cfg.train_steps,
                max_val_batches=cfg.max_val_batches,
                default_batch_size=cfg.default_batch_size,
                train_subset_size=cfg.train_subset_size,
                val_subset_size=cfg.val_subset_size,
                data_root=cfg.data_root,
                download=cfg.download,
                measure_latency=cfg.measure_latency,
                kl_div=cfg.kl_div,
                critic_fn=cfg.critic_fn,
                weights=cfg.weights,
                eval_limit_seconds=cfg.eval_limit_seconds,
                budget_probe_batches=cfg.budget_probe_batches,
                run_unfrozen_backbone_eval=cfg.run_unfrozen_backbone_eval,
                full_test_acc=cfg.full_test_acc,
                reward_target_metric=cfg.reward_target_metric,
            )

        return RewardUtil.evaluate_code_and_reward(
            code,
            in_shape=in_shape,
            out_shape=out_shape,
            prm=prm,
            device=eval_device,
            val_metric_baseline=val_metric_baseline,
            seed_accuracy_baseline=seed_accuracy_baseline,
            cfg=cfg,
            reward_batch_index=reward_batch_index,
            completion_index=completion_index,
            batch_last_item=batch_last_item,
        )
    except Exception as exc:
        return _cifar_eval_error(exc, seed_accuracy_baseline=seed_accuracy_baseline)


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
    seed_accuracy_baseline: float,
    precomputed_eval_result: Dict[str, Any] | None = None,
    graph_info=None,
    batch_graph_hashes: List[str] = None,
    batch_family_hashes: List[str] = None,
    prompt_goal_tags: List[str] = None,
    archive_snapshot_family_counts: Dict[str, int] = None,
    group_baseline_train_acc: float | None = None,
    group_baseline_reward_target_acc: float | None = None,
    reward_batch_index: int | None = None,
    reward_group_id: int | None = None,
    group_warmup: bool = False,
    completion_index: int | None = None,
    batch_last_item: bool = False,
):
    res = TuneRLRaw.raw_reward_fn(
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
    res["reward"] = _reapply_trainability_clamp(res, float(res.get("reward", -2.0)), graph_info)
    res["anti_collapse"] = {
        "goal_key": TuneRL.primary_goal_key(prompt_goal_tags),
        "trainable_ok": _is_trainable_architecture(res, graph_info),
        "anti_collapse_delta": 0.0,
    }
    return res


SFT_DISCOVERY_PROMPT_TEMPLATE = SFTUtil.open_discovery_rl_prompt_template


class DynamicSFTPromptDataset(TorchDataset):
    column_names = ["prompt", "accuracy", "goal_name", "target_tags", "goal_profile_id"]

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        tokenizer,
        *,
        block_signature: str,
        init_signature: str,
        forward_signature: str,
    ) -> None:
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.block_signature = block_signature
        self.init_signature = init_signature
        self.forward_signature = forward_signature

    def __len__(self) -> int:
        return len(self.rows)

    def select(self, indices) -> "DynamicSFTPromptDataset":
        if hasattr(indices, "tolist"):
            indices = indices.tolist()
        return DynamicSFTPromptDataset(
            [self.rows[int(index)] for index in indices],
            self.tokenizer,
            block_signature=self.block_signature,
            init_signature=self.init_signature,
            forward_signature=self.forward_signature,
        )

    def _render_prompt(self, row: Dict[str, Any]) -> str:
        profile = SFTUtil.open_discovery_goal_profiles[int(row["goal_profile_id"])]
        module_hints = (
            "self.backbone_a",
            "self.backbone_b",
            *profile["module_hints"],
        )
        user_prompt = SFT_DISCOVERY_PROMPT_TEMPLATE.format(
            accuracy=row["accuracy"],
            skeleton_code=SFTUtil.open_discovery_skeleton_code,
            available_backbones=", ".join(SFTUtil.available_backbones),
            legacy_patterns=", ".join(SFTUtil.legacy_patterns),
            goal_name=profile["name"],
            target_tags=", ".join(profile["tags"]),
            design_brief=profile["brief"],
            module_hints=", ".join(module_hints),
            block_signature=self.block_signature,
            init_signature=self.init_signature,
            forward_signature=self.forward_signature,
        )
        feedback_text = TuneRL.render_prompt_feedback_text(
            feedback_char_budget=SFT_FEEDBACK_CHAR_BUDGET,
        )
        feedback_section = "\n\n### Current Optimization Feedback\n" + feedback_text.strip() + "\n"
        marker = "### Output Requirement (STRICT)"
        if marker in user_prompt:
            user_prompt = user_prompt.replace(marker, feedback_section + "\n" + marker, 1)
        else:
            user_prompt = user_prompt + feedback_section
        messages = [{"role": "user", "content": user_prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[int(index)]
        return {
            "prompt": self._render_prompt(row),
            "accuracy": row["accuracy"],
            "goal_name": row["goal_name"],
            "target_tags": row["target_tags"],
            "goal_profile_id": row["goal_profile_id"],
        }


def load_rl_dataset_sft(tokenizer) -> TuneRL.Dataset:
    """Load SFT-aligned RL prompts while rendering feedback lazily at access time."""
    from ab.gpt.TuneRLRaw import BLOCK_SIGNATURE, FORWARD_SIGNATURE, INIT_SIGNATURE

    data = TuneRL.api.data(task="img-classification", nn_prefixes=("rl-bb-test1",))
    if data.empty:
        print("No 'rl-bb-test1' data found, falling back to all img-classification")
        data = TuneRL.api.data(only_best_accuracy=True, task="img-classification", dataset="cifar-10")

    print(f"Loaded {len(data)} examples for SFT RL")
    TuneRL.bootstrap_trainset_reference_library(data)

    rows: List[Dict[str, Any]] = []

    for _, row in data.iterrows():
        accuracy = TuneRL._coerce_accuracy_baseline(row.get("accuracy"), context="seed row accuracy")
        for profile_id, profile in enumerate(SFTUtil.open_discovery_goal_profiles):
            rows.append(
                {
                    "accuracy": accuracy,
                    "goal_name": profile["name"],
                    "target_tags": ", ".join(profile["tags"]),
                    "goal_profile_id": profile_id,
                }
            )

    random.Random(42).shuffle(rows)
    if len(rows) > SFT_DATASET_LIMIT:
        rows = rows[:SFT_DATASET_LIMIT]
    return DynamicSFTPromptDataset(
        rows,
        tokenizer,
        block_signature=BLOCK_SIGNATURE,
        init_signature=INIT_SIGNATURE,
        forward_signature=FORWARD_SIGNATURE,
    )


def sft_run_epoch_dir(*args) -> Path:
    epoch_dir = Path(SFT_EPOCH_ROOT)
    for value in args:
        epoch_dir = epoch_dir / f"A{value}"
    return epoch_dir


def run_sft_training():
    import torch
    from transformers import BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("SFT RL requires CUDA for GRPO training, but no CUDA device is available")
    runtime = RewardUtil.get_distributed_runtime_info()
    visible_cuda_devices = int(runtime["visible_gpu_count"])
    if visible_cuda_devices < 1:
        raise RuntimeError(
            f"SFT RL requires at least one visible CUDA device, got {visible_cuda_devices}"
        )
    local_rank = int(runtime["local_rank"])
    raw_local_rank = int(runtime["raw_local_rank"])
    rank = int(runtime["rank"])
    world_size = int(runtime["world_size"])
    if world_size > 1 and visible_cuda_devices > 1 and raw_local_rank != local_rank:
        raise RuntimeError(
            "SFT RL detected an inconsistent distributed CUDA mapping: "
            f"rank={rank}, raw_local_rank={raw_local_rank}, resolved_local_rank={local_rank}, "
            f"visible_cuda_devices={visible_cuda_devices}"
        )
    if not (0 <= local_rank < visible_cuda_devices):
        raise RuntimeError(
            "SFT RL resolved an invalid local CUDA rank: "
            f"local_rank={local_rank}, raw_local_rank={raw_local_rank}, visible_cuda_devices={visible_cuda_devices}"
        )
    torch.cuda.set_device(local_rank)
    train_device = f"cuda:{local_rank}"

    torch.cuda.empty_cache()
    TuneRL.reset_reward_runtime_state()
    precision = TuneRL.best_mixed_precision()

    print(f"Using RL base model: {TuneRL.base_model}")
    print(
        "[SFT RL] Distributed Runtime: "
        f"rank={rank} local_rank={local_rank} raw_local_rank={raw_local_rank} world_size={world_size}"
    )
    print(f"[SFT RL] Fixed training device: {train_device}")
    print(f"[SFT RL] Visible CUDA devices: {visible_cuda_devices}")
    print(f"[SFT RL] Mixed precision: {precision['label']} (torch_dtype={precision['torch_dtype']})")
    reward_worker_plan = RewardUtil.get_reward_worker_plan()
    print(
        "[SFT RL] Reward Worker Plan: "
        f"mode={reward_worker_plan['mode']} "
        f"rank={reward_worker_plan['rank']} "
        f"local_rank={reward_worker_plan['local_rank']} "
        f"world_size={reward_worker_plan['world_size']} "
        f"train_gpu={reward_worker_plan['train_gpu']} "
        f"reward_gpu_indices={reward_worker_plan['reward_gpu_indices']} "
        f"reward_gpu_tokens={reward_worker_plan.get('reward_gpu_tokens', [])}"
    )
    _print_runtime_cache_roots()
    tokenizer_source = getattr(TuneRL, "tokenizer_source", TuneRL.base_model)
    if tokenizer_source != TuneRL.base_model:
        print(f"Using RL tokenizer: {tokenizer_source}")
    tokenizer = TuneRL.AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    TuneRL.log_memory_snapshot("sft/tokenizer_loaded")

    rl_dataset = TuneRL.load_rl_dataset(tokenizer)
    if len(rl_dataset) > SFT_DATASET_LIMIT:
        rl_dataset = rl_dataset.select(range(SFT_DATASET_LIMIT))
    _run_torchvision_prewarm(list(SFTUtil.available_backbones))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=precision["torch_dtype"],
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = TuneRL.AutoModelForCausalLM.from_pretrained(
        TuneRL.base_model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map={"": train_device},
        dtype=precision["torch_dtype"],
    )
    TuneRL.log_memory_snapshot("sft/base_model_loaded")

    if SFT_LOAD_INITIAL_ADAPTER:
        if not SFT_INIT_ADAPTER:
            raise ValueError("SFT_INIT_ADAPTER is empty, but SFT_LOAD_INITIAL_ADAPTER is True.")
        if not os.path.exists(SFT_INIT_ADAPTER):
            raise FileNotFoundError(f"Initial adapter not found: {SFT_INIT_ADAPTER}")
        print(f"Loading initial SFT adapter from {SFT_INIT_ADAPTER}...")
        model = TuneRL.PeftModel.from_pretrained(model, SFT_INIT_ADAPTER)
        model = model.merge_and_unload()

    model = TuneRL.prepare_model_for_kbit_training(model)
    TuneRL.align_generation_head_dtype(model, precision["torch_dtype"])

    peft_config = TuneRL.LoraConfig(
        r=SFT_LORA_R,
        lora_alpha=SFT_LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=SFT_LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = TuneRL.get_peft_model(model, peft_config)
    TuneRL.align_generation_head_dtype(model, precision["torch_dtype"])

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    TuneRL.log_memory_snapshot("sft/lora_wrapped")

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
        bf16=precision["bf16"],
        fp16=precision["fp16"],
        gradient_checkpointing=True,
        num_generations=SFT_NUM_GENERATIONS,
    )

    trainer = TuneRL.GRPOTrainer(
        model=model,
        train_dataset=rl_dataset,
        reward_funcs=TuneRL.compute_reward,
        args=grpo_config,
    )
    TuneRL.log_memory_snapshot("sft/grpo_trainer_initialized")

    print("Starting GRPO training for Backbone Search...")
    try:
        TuneRL.log_memory_snapshot("sft/before_trainer_train")
        trainer.train()
    finally:
        RewardUtil.shutdown_eval_worker()

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
    TuneRLRaw.clear_extraction_meta_cache()
    RewardUtil.shutdown_eval_worker()

    log_dir = TuneRL.run_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    stale_files = (
        "generation_samples.jsonl",
        "group_progress.jsonl",
        "group_feedback_samples.jsonl",
        "best_group_feedback.json",
    )
    for filename in stale_files:
        path = Path(log_dir) / filename
        if path.exists():
            print(f"Removing stale runtime log: {path}")
            path.unlink()
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
        f"train_subset={SFT_EVAL_TRAIN_SUBSET}, "
        f"{'test_subset=full' if SFT_EVAL_FULL_TEST_ACC else f'val_subset={SFT_EVAL_VAL_SUBSET}'}, "
        f"train_epochs={SFT_EVAL_TRAIN_EPOCHS}, val_batches={SFT_EVAL_VAL_BATCHES}, "
        f"full_test_acc={SFT_EVAL_FULL_TEST_ACC}, run_unfrozen={SFT_EVAL_RUN_UNFROZEN}, "
        f"eval_limit_seconds={SFT_EVAL_LIMIT_SECONDS}, "
        f"baseline={SFT_VAL_METRIC_BASELINE:.2f}"
    )
    print(f"[SFT RL] Save RL adapter: {SFT_SAVE_RL_MODEL}")

    run_sft_training()


if __name__ == "__main__":
    main()
