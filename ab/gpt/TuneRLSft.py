import os
import shutil
import random
import inspect
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
from torch.utils.data import Dataset as TorchDataset
from ab.gpt.util.Const import conf_dir


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
SFT_DEEPSPEED_DEFAULT_CONFIG = str(conf_dir / "DeepSpeedSftGrpo.json")

# CIFAR-10 reward evaluation via nn-dataset / NNEval-aligned formal acc.
SFT_EVAL_IMAGE_SIZE = 256
SFT_EVAL_BATCH_SIZE = 64
SFT_EVAL_TRAIN_SUBSET = 256
SFT_EVAL_VAL_SUBSET = 128
SFT_EVAL_TRAIN_EPOCHS = 1
SFT_EVAL_VAL_BATCHES = 2
SFT_EVAL_FULL_TEST_ACC = True
SFT_EVAL_RUN_UNFROZEN = True
SFT_EVAL_LIMIT_SECONDS = 900
SFT_EVAL_FORMAL_EPOCH_LIMIT_MINUTES = 30
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


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return int(default)
    return int(raw)


def _runtime_is_main_process(runtime: Dict[str, Any]) -> bool:
    return int(runtime.get("rank", 0)) == 0


def _suggested_sft_worker_count(runtime: Dict[str, Any]) -> int:
    return max(1, int(runtime.get("visible_gpu_count", 0) or 0))


def _validate_sft_visible_worker_count(runtime: Dict[str, Any]) -> None:
    world_size = int(runtime.get("world_size", 1))
    suggested_worker_count = _suggested_sft_worker_count(runtime)
    if world_size <= suggested_worker_count:
        return
    raise RuntimeError(
        "SFT RL worker count exceeds visible training GPUs: "
        f"world_size={world_size}, "
        f"visible_cuda_devices={int(runtime.get('visible_gpu_count', 0))}, "
        f"suggested_nproc_per_node={suggested_worker_count}. "
        "Launch torchrun with as many workers as visible training GPUs."
    )


def _resolve_sft_local_rank(runtime: Dict[str, Any]) -> Dict[str, Any]:
    visible_cuda_devices = int(runtime.get("visible_gpu_count", 0))
    raw_local_rank = int(runtime.get("raw_local_rank", 0))
    rank = int(runtime.get("rank", 0))
    world_size = int(runtime.get("world_size", 1))

    if visible_cuda_devices < 1:
        raise RuntimeError(
            f"SFT RL requires at least one visible CUDA device, got {visible_cuda_devices}"
        )

    local_rank = 0
    resolution = "single_visible_gpu"

    if visible_cuda_devices == 1:
        local_rank = 0
    elif 0 <= raw_local_rank < visible_cuda_devices:
        local_rank = raw_local_rank
        resolution = "direct"
    else:
        raise RuntimeError(
            "SFT RL detected an invalid distributed CUDA mapping: "
            f"rank={rank}, raw_local_rank={raw_local_rank}, world_size={world_size}, "
            f"visible_cuda_devices={visible_cuda_devices}, "
            f"suggested_nproc_per_node={_suggested_sft_worker_count(runtime)}. "
            "Launch torchrun with as many workers as visible training GPUs."
        )

    if not (0 <= local_rank < visible_cuda_devices):
        raise RuntimeError(
            "SFT RL resolved an invalid local CUDA rank: "
            f"local_rank={local_rank}, raw_local_rank={raw_local_rank}, visible_cuda_devices={visible_cuda_devices}"
        )

    return {
        "rank": rank,
        "world_size": world_size,
        "raw_local_rank": raw_local_rank,
        "local_rank": local_rank,
        "visible_cuda_devices": visible_cuda_devices,
        "resolution": resolution,
    }


def _maybe_relaunch_sft_with_visible_gpu_workers() -> None:
    if os.getenv("NNGPT_SFT_AUTO_TORCHRUN_DONE", "") == "1":
        return
    if os.getenv("WORLD_SIZE") not in (None, "", "1"):
        return
    if os.getenv("LOCAL_RANK") not in (None, ""):
        return

    import torch

    if not torch.cuda.is_available():
        return
    visible_cuda_devices = int(torch.cuda.device_count())
    if visible_cuda_devices <= 1:
        return

    os.environ["NNGPT_SFT_AUTO_TORCHRUN_DONE"] = "1"
    print(
        "[SFT RL] Relaunching with visible-GPU worker count: "
        f"nproc_per_node={visible_cuda_devices}"
    )
    os.execvpe(
        sys.executable,
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={visible_cuda_devices}",
            "-m",
            "ab.gpt.TuneRLSft",
        ],
        os.environ,
    )


def _maybe_init_single_process_deepspeed_group(
    *,
    use_deepspeed: bool,
    world_size: int,
    visible_cuda_devices: int,
    local_rank: int,
) -> None:
    import torch

    if not use_deepspeed:
        return
    if int(world_size) > 1:
        return
    if not torch.distributed.is_available():
        return
    if torch.distributed.is_initialized():
        return

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    init_dir = Path(tempfile.mkdtemp(prefix="nngpt_sft_pg_"))
    init_file = (init_dir / "store").resolve()
    init_file.touch()
    init_method = init_file.as_uri()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    torch.distributed.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=1,
        rank=0,
    )
    print(
        "[SFT RL] Initialized single-process distributed group for DeepSpeed: "
        f"backend={backend} local_rank={local_rank} visible_cuda_devices={visible_cuda_devices}"
    )


def resolve_sft_runtime_settings(runtime: Dict[str, Any]) -> Dict[str, int]:
    return {
        "dataset_limit": _env_int(
            "NNGPT_SFT_DATASET_LIMIT",
            SFT_DATASET_LIMIT,
        ),
        "grad_accum": _env_int("NNGPT_SFT_GRAD_ACCUM", SFT_GRAD_ACCUM),
        "max_completion_length": _env_int(
            "NNGPT_SFT_MAX_COMPLETION_LENGTH",
            SFT_MAX_COMPLETION_LENGTH,
        ),
        "num_generations": _env_int(
            "NNGPT_SFT_NUM_GENERATIONS",
            SFT_NUM_GENERATIONS,
        ),
    }


def _resolve_sft_deepspeed_enabled(runtime: Dict[str, Any]) -> bool:
    raw = os.getenv("NNGPT_SFT_USE_DEEPSPEED")
    if raw is None or raw == "":
        return int(runtime.get("world_size", 1)) > 1
    return _env_flag("NNGPT_SFT_USE_DEEPSPEED", False)


def _resolve_sft_deepspeed_config_path() -> str:
    raw_path = os.getenv("NNGPT_SFT_DEEPSPEED_CONFIG", SFT_DEEPSPEED_DEFAULT_CONFIG)
    config_path = Path(raw_path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"SFT DeepSpeed config not found: {config_path}")
    return str(config_path)


def _maybe_init_hf_deepspeed_config(config_path: str) -> Any:
    last_error: Exception | None = None
    for module_name in ("transformers.integrations", "transformers.deepspeed"):
        try:
            module = __import__(module_name, fromlist=["HfDeepSpeedConfig"])
            config_cls = getattr(module, "HfDeepSpeedConfig", None)
            if config_cls is not None:
                return config_cls(config_path)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        "DeepSpeed ZeRO-3 requested for SFT GRPO, but HfDeepSpeedConfig could not be imported"
    ) from last_error


def _bootstrap_run_token(runtime: Dict[str, Any] | None = None) -> str:
    runtime = runtime or {}
    for value in (
        os.getenv("TORCHELASTIC_RUN_ID"),
        os.getenv("MASTER_PORT"),
        os.getenv("SLURM_JOB_ID"),
    ):
        if value:
            return str(value)
    return f"rank0_world{int(runtime.get('world_size', 1))}"


def _bootstrap_sentinel_path(log_dir: str, runtime: Dict[str, Any] | None = None) -> Path:
    return Path(log_dir) / f".sft_bootstrap_complete.{_bootstrap_run_token(runtime)}"


def _wait_for_bootstrap_sentinel(
    sentinel_path: Path,
    *,
    timeout_seconds: float = 600.0,
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if sentinel_path.exists():
            return
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for rank0 bootstrap sentinel: {sentinel_path}")


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
                formal_nn_eval=True,
                formal_task="img-classification",
                formal_dataset="cifar-10",
                formal_metric="acc",
                formal_epoch_limit_minutes=SFT_EVAL_FORMAL_EPOCH_LIMIT_MINUTES,
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
                formal_nn_eval=getattr(cfg, "formal_nn_eval", True),
                formal_task=getattr(cfg, "formal_task", "img-classification"),
                formal_dataset=getattr(cfg, "formal_dataset", "cifar-10"),
                formal_metric=getattr(cfg, "formal_metric", "acc"),
                formal_epoch_limit_minutes=getattr(
                    cfg,
                    "formal_epoch_limit_minutes",
                    SFT_EVAL_FORMAL_EPOCH_LIMIT_MINUTES,
                ),
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


def build_sft_reward_eval_cfg(
    *,
    in_shape=(1, 3, 256, 256),
    out_shape=(10,),
    prm=None,
    cfg=None,
):
    import torch

    eval_device = "cuda" if torch.cuda.is_available() else "cpu"
    if prm is None:
        prm = {"lr": 1e-2, "momentum": 0.9, "dropout": 0.3}
    defaults = {"lr": 1e-2, "momentum": 0.9, "batch": SFT_EVAL_BATCH_SIZE, "epoch": 1}
    prm = {**defaults, **prm}

    if cfg is None:
        return RewardUtil.EvalConfig(
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
            formal_nn_eval=True,
            formal_task="img-classification",
            formal_dataset="cifar-10",
            formal_metric="acc",
            formal_epoch_limit_minutes=SFT_EVAL_FORMAL_EPOCH_LIMIT_MINUTES,
        )

    return RewardUtil.EvalConfig(
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
        formal_nn_eval=getattr(cfg, "formal_nn_eval", True),
        formal_task=getattr(cfg, "formal_task", "img-classification"),
        formal_dataset=getattr(cfg, "formal_dataset", "cifar-10"),
        formal_metric=getattr(cfg, "formal_metric", "acc"),
        formal_epoch_limit_minutes=getattr(
            cfg,
            "formal_epoch_limit_minutes",
            SFT_EVAL_FORMAL_EPOCH_LIMIT_MINUTES,
        ),
    )


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

    runtime_settings = resolve_sft_runtime_settings(RewardUtil.get_distributed_runtime_info())
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
    if len(rows) > runtime_settings["dataset_limit"]:
        rows = rows[:runtime_settings["dataset_limit"]]
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


def _build_sft_grpo_config(
    *,
    precision: Dict[str, Any],
    use_deepspeed: bool,
    deepspeed_config_path: str | None,
    runtime_settings: Dict[str, int],
) -> Any:
    config_signature = inspect.signature(TuneRL.GRPOConfig.__init__)
    config_kwargs: Dict[str, Any] = {
        "temperature": SFT_TEMPERATURE,
        "learning_rate": SFT_LR,
        "max_completion_length": runtime_settings["max_completion_length"],
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": runtime_settings["grad_accum"],
        "lr_scheduler_type": "cosine",
        "num_train_epochs": SFT_NUM_EPOCHS,
        "remove_unused_columns": False,
        "logging_steps": 1,
        "output_dir": SFT_TRAINER_OUT,
        "eval_strategy": "no",
        "bf16": precision["bf16"],
        "fp16": precision["fp16"],
        "gradient_checkpointing": True,
        "num_generations": runtime_settings["num_generations"],
    }
    if use_deepspeed:
        if "deepspeed" not in config_signature.parameters:
            raise RuntimeError("Installed GRPOConfig does not support the `deepspeed` argument")
        config_kwargs["deepspeed"] = deepspeed_config_path
        if "ds3_gather_for_generation" in config_signature.parameters:
            config_kwargs["ds3_gather_for_generation"] = False
    return TuneRL.GRPOConfig(**config_kwargs)


def run_sft_training():
    import torch
    from transformers import BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("SFT RL requires CUDA for GRPO training, but no CUDA device is available")
    runtime = RewardUtil.get_distributed_runtime_info()
    runtime_settings = resolve_sft_runtime_settings(runtime)
    _validate_sft_visible_worker_count(runtime)
    local_rank_resolution = _resolve_sft_local_rank(runtime)
    visible_cuda_devices = int(local_rank_resolution["visible_cuda_devices"])
    local_rank = int(local_rank_resolution["local_rank"])
    raw_local_rank = int(local_rank_resolution["raw_local_rank"])
    rank = int(local_rank_resolution["rank"])
    world_size = int(local_rank_resolution["world_size"])
    use_deepspeed = _resolve_sft_deepspeed_enabled(runtime)
    deepspeed_config_path = _resolve_sft_deepspeed_config_path() if use_deepspeed else None
    os.environ["NNGPT_SFT_USE_DEEPSPEED"] = "1" if use_deepspeed else "0"
    if deepspeed_config_path is not None:
        os.environ["NNGPT_SFT_DEEPSPEED_CONFIG"] = deepspeed_config_path
    torch.cuda.set_device(local_rank)
    train_device = f"cuda:{local_rank}"
    _maybe_init_single_process_deepspeed_group(
        use_deepspeed=use_deepspeed,
        world_size=world_size,
        visible_cuda_devices=visible_cuda_devices,
        local_rank=local_rank,
    )

    torch.cuda.empty_cache()
    TuneRL.reset_reward_runtime_state()
    precision = TuneRL.best_mixed_precision()
    grpo_config = _build_sft_grpo_config(
        precision=precision,
        use_deepspeed=use_deepspeed,
        deepspeed_config_path=deepspeed_config_path,
        runtime_settings=runtime_settings,
    )
    hf_deepspeed_config = _maybe_init_hf_deepspeed_config(deepspeed_config_path) if use_deepspeed else None

    print(f"Using RL base model: {TuneRL.base_model}")
    print(
        "[SFT RL] Distributed Runtime: "
        f"rank={rank} local_rank={local_rank} raw_local_rank={raw_local_rank} world_size={world_size}"
    )
    if use_deepspeed and world_size <= 1:
        print(
            "[SFT RL] DeepSpeed single-process fallback active: "
            f"visible_cuda_devices={visible_cuda_devices} local_rank={local_rank}"
        )
    print(f"[SFT RL] DeepSpeed Enabled: {use_deepspeed}")
    if deepspeed_config_path is not None:
        print(f"[SFT RL] DeepSpeed Config: {deepspeed_config_path}")
    print(f"[SFT RL] Fixed training device: {train_device}")
    print(f"[SFT RL] Visible CUDA devices: {visible_cuda_devices}")
    print(f"[SFT RL] Mixed precision: {precision['label']} (torch_dtype={precision['torch_dtype']})")
    print(
        "[SFT RL] Runtime limits: "
        f"dataset_limit={runtime_settings['dataset_limit']} "
        f"max_completion_length={runtime_settings['max_completion_length']} "
        f"grad_accum={runtime_settings['grad_accum']} "
        f"num_generations={runtime_settings['num_generations']}"
    )
    reward_worker_plan = RewardUtil.get_reward_worker_plan()
    print(
        "[SFT RL] Reward Worker Plan: "
        f"mode={reward_worker_plan['mode']} "
        f"reward_workers_per_gpu={reward_worker_plan.get('workers_per_gpu', 1)} "
        f"per_gpu_worker_counts={reward_worker_plan.get('per_gpu_worker_counts', [])} "
        f"rank={reward_worker_plan['rank']} "
        f"local_rank={reward_worker_plan['local_rank']} "
        f"world_size={reward_worker_plan['world_size']} "
        f"train_gpu={reward_worker_plan['train_gpu']} "
        f"reward_gpu_indices={reward_worker_plan['reward_gpu_indices']} "
        f"reward_gpu_tokens={reward_worker_plan.get('reward_gpu_tokens', [])} "
        f"reason={reward_worker_plan.get('reason', '')!r}"
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
    if len(rl_dataset) > runtime_settings["dataset_limit"]:
        rl_dataset = rl_dataset.select(range(runtime_settings["dataset_limit"]))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=precision["torch_dtype"],
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model_load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "quantization_config": bnb_config,
        "dtype": precision["torch_dtype"],
    }
    if not use_deepspeed:
        model_load_kwargs["device_map"] = {"": train_device}
    model = TuneRL.AutoModelForCausalLM.from_pretrained(
        TuneRL.base_model,
        **model_load_kwargs,
    )
    _ = hf_deepspeed_config
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

    trainer = TuneRL.GRPOTrainer(
        model=model,
        train_dataset=rl_dataset,
        reward_funcs=TuneRL.compute_reward,
        args=grpo_config,
    )
    TuneRL.log_memory_snapshot("sft/grpo_trainer_initialized")

    print("Starting GRPO training for Backbone Search...")
    memory_monitor = TuneRL.start_cuda_memory_monitor("sft/trainer")
    try:
        TuneRL.log_memory_snapshot("sft/before_trainer_train")
        trainer.train()
    except Exception as exc:
        if TuneRL.is_cuda_oom_error(exc):
            TuneRL.log_cuda_oom_diagnostics("sft/trainer.train", exc)
        raise
    finally:
        if memory_monitor is not None:
            memory_monitor.close()
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
    setattr(TuneRL.evaluate_code_and_reward, "_nngpt_eval_cfg_builder", build_sft_reward_eval_cfg)
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
    runtime = RewardUtil.get_distributed_runtime_info()
    is_main = _runtime_is_main_process(runtime)

    log_dir = TuneRL.run_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    sentinel_path = _bootstrap_sentinel_path(log_dir, runtime)
    trainer_out_dir = Path(SFT_TRAINER_OUT)
    stale_files = (
        "generation_samples.jsonl",
        "group_progress.jsonl",
        "group_feedback_samples.jsonl",
        "best_group_feedback.json",
    )
    if is_main:
        if sentinel_path.exists():
            sentinel_path.unlink()
        for filename in stale_files:
            path = Path(log_dir) / filename
            if path.exists():
                print(f"Removing stale runtime log: {path}")
                path.unlink()
        print(f"Cleaning existing models in {TuneRL.run_epoch_dir()}...")
        shutil.rmtree(TuneRL.run_epoch_dir(), ignore_errors=True)
        print(f"Cleaning existing trainer outputs in {trainer_out_dir}...")
        shutil.rmtree(trainer_out_dir, ignore_errors=True)
        TuneRL.code_logger = TuneRLRaw.RawCodeLogger(log_dir)
        sentinel_path.write_text(str(os.getpid()), encoding="utf-8")
        return

    TuneRL.code_logger = TuneRL.NullCodeLogger()
    _wait_for_bootstrap_sentinel(sentinel_path)


def main() -> None:
    _maybe_relaunch_sft_with_visible_gpu_workers()
    _validate_sft_visible_worker_count(RewardUtil.get_distributed_runtime_info())
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
        f"[SFT RL] CIFAR-10 eval: backend=nn-dataset-formal, resize={SFT_EVAL_IMAGE_SIZE}, batch<={SFT_EVAL_BATCH_SIZE}, "
        f"train_set=full, test_set=full, "
        f"train_epochs={SFT_EVAL_TRAIN_EPOCHS}, run_unfrozen={SFT_EVAL_RUN_UNFROZEN}, "
        f"formal_epoch_limit_minutes={SFT_EVAL_FORMAL_EPOCH_LIMIT_MINUTES}, "
        f"worker_eval_limit_seconds={SFT_EVAL_LIMIT_SECONDS}, "
        f"baseline={SFT_VAL_METRIC_BASELINE:.2f}"
    )
    print(f"[SFT RL] Save RL adapter: {SFT_SAVE_RL_MODEL}")

    run_sft_training()


if __name__ == "__main__":
    main()
