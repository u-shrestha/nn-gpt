from typing import Optional, Dict, Tuple, Callable, Any
from dataclasses import dataclass, asdict, replace
import atexit
import time
import gc
import multiprocessing as mp
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset


class PersistentEvalWorkerError(RuntimeError):
    pass


class _PersistentEvalWorkerSession:
    def __init__(self) -> None:
        from ab.gpt.util.reward_worker_bootstrap import reward_worker_main

        self.ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = self.ctx.Pipe()
        self._process = self.ctx.Process(
            target=reward_worker_main,
            args=(child_conn,),
        )
        self._process.start()
        child_conn.close()
        self._worker_info = self._wait_for_ready(timeout=30.0)
        print(
            "[Reward Worker] Ready "
            f"pid={self._worker_info['pid']}, "
            f"cuda_visible_devices={self._worker_info['cuda_visible_devices']!r}, "
            f"cuda_available={self._worker_info['cuda_available']}, "
            f"cuda_device_count={self._worker_info['cuda_device_count']}"
        )

    def request(self, payload: Dict[str, Any], *, timeout: float) -> Dict[str, Any]:
        if not self._process.is_alive():
            raise PersistentEvalWorkerError("Persistent eval worker exited before handling a request")
        try:
            self._parent_conn.send(payload)
        except Exception as exc:
            raise PersistentEvalWorkerError(f"Persistent eval worker send failed: {exc}") from exc

        if not self._parent_conn.poll(timeout):
            self.close(force=True)
            raise PersistentEvalWorkerError(f"Persistent eval worker timed out after {timeout:.0f} seconds")

        try:
            response = self._parent_conn.recv()
        except EOFError as exc:
            self.close(force=True)
            raise PersistentEvalWorkerError("Persistent eval worker closed its pipe unexpectedly") from exc
        except Exception as exc:
            self.close(force=True)
            raise PersistentEvalWorkerError(f"Persistent eval worker recv failed: {exc}") from exc

        if not isinstance(response, dict):
            self.close(force=True)
            raise PersistentEvalWorkerError("Persistent eval worker returned a non-dict response")
        return response

    def _wait_for_ready(self, *, timeout: float) -> Dict[str, Any]:
        if not self._process.is_alive():
            raise PersistentEvalWorkerError("Persistent eval worker exited during startup")
        if not self._parent_conn.poll(timeout):
            self.close(force=True)
            raise PersistentEvalWorkerError(
                f"Persistent eval worker did not send a startup handshake within {timeout:.0f} seconds"
            )
        try:
            response = self._parent_conn.recv()
        except EOFError as exc:
            self.close(force=True)
            raise PersistentEvalWorkerError("Persistent eval worker closed its pipe during startup") from exc
        except Exception as exc:
            self.close(force=True)
            raise PersistentEvalWorkerError(f"Persistent eval worker startup recv failed: {exc}") from exc

        if not isinstance(response, dict):
            self.close(force=True)
            raise PersistentEvalWorkerError("Persistent eval worker startup handshake was not a dict")
        if response.get("cmd") == "worker_init_error":
            self.close(force=True)
            raise PersistentEvalWorkerError(
                f"Persistent eval worker failed to initialize: {response.get('error', 'unknown error')}"
            )
        if response.get("cmd") != "worker_ready":
            self.close(force=True)
            raise PersistentEvalWorkerError(
                f"Persistent eval worker returned unexpected startup message: {response!r}"
            )

        required_keys = {"pid", "cuda_visible_devices", "cuda_available", "cuda_device_count"}
        missing = sorted(required_keys.difference(response))
        if missing:
            self.close(force=True)
            raise PersistentEvalWorkerError(
                f"Persistent eval worker startup handshake missing fields: {', '.join(missing)}"
            )
        if bool(response["cuda_available"]) or int(response["cuda_device_count"]) != 0:
            self.close(force=True)
            raise PersistentEvalWorkerError(
                "Persistent eval worker must be CPU-only, but reported CUDA visibility "
                f"(available={response['cuda_available']}, count={response['cuda_device_count']})"
            )
        return {
            "pid": int(response["pid"]),
            "cuda_visible_devices": str(response["cuda_visible_devices"]),
            "cuda_available": bool(response["cuda_available"]),
            "cuda_device_count": int(response["cuda_device_count"]),
        }

    def diagnostics(self) -> Dict[str, Any]:
        return {**self._worker_info, "alive": self._process.is_alive()}

    def close(self, *, force: bool = False) -> None:
        try:
            if not force and self._process.is_alive():
                self._parent_conn.send({"cmd": "shutdown"})
        except Exception:
            pass

        if self._process.is_alive():
            if force:
                self._process.terminate()
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5)

        try:
            self._parent_conn.close()
        except Exception:
            pass


_PERSISTENT_EVAL_WORKER: Optional[_PersistentEvalWorkerSession] = None

def compute_cv_reward_simple(
    *,
    built_ok: bool,
    forward_shape_ok: Optional[bool] = None,
    backward_ok: Optional[bool] = None,
    loss_drop_ok: bool = False,
    val_metric: Optional[float],
    val_metric_baseline: Optional[float] = None,
    latency_ms: Optional[float] = None,
    params_m: Optional[float] = None,
    flops_g: Optional[float] = None,
    critic_score: Optional[float] = None,
    kl_div: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    forward_ok: Optional[bool] = None,
    trained_step_ok: Optional[bool] = None,
) -> Dict[str, float]:
    """
    Simple scalar reward for CV model generation.
    Returns a dict with 'reward' and component terms.
    """
    w = {
        "build": 0.2,          # bonus if model builds
        "forward_shape": 0.2,  # bonus if logits shape is correct
        "backward": 0.3,       # bonus if backward + optimizer step succeed
        "loss_drop": 0.6,      # bonus if short-run training shows learning signal
        "metric_gain": 0.0,    # validation stays logging-only in RL reward
        "eff_latency": 0.0,   # optional: -latency_ms * coeff
        "eff_params":  0.0,   # optional: -params_m  * coeff
        "eff_flops":   0.0,   # optional: -flops_g   * coeff
        "critic": 0.5,        # optional: critic score in [0,1]
        "kl": 0.02,           # optional: KL penalty coeff
        "clip_lo": -2.0,
        "clip_hi":  2.0,
    }
    if weights:
        w.update(weights)

    shape_ok = bool(forward_shape_ok if forward_shape_ok is not None else forward_ok)
    train_ok = bool(backward_ok if backward_ok is not None else trained_step_ok)

    r_build = w["build"] if built_ok else 0.0
    r_forward_shape = w["forward_shape"] if shape_ok else 0.0
    r_backward = w["backward"] if train_ok else 0.0
    r_loss_drop = w["loss_drop"] if loss_drop_ok else 0.0

    r_metric = 0.0
    if (val_metric is not None) and (val_metric_baseline is not None):
        r_metric = w["metric_gain"] * (val_metric - val_metric_baseline)

    r_eff = 0.0
    if latency_ms is not None and w["eff_latency"] != 0.0:
        r_eff += - w["eff_latency"] * float(latency_ms)
    if params_m is not None and w["eff_params"] != 0.0:
        r_eff += - w["eff_params"]  * float(params_m)
    if flops_g is not None and w["eff_flops"] != 0.0:
        r_eff += - w["eff_flops"]   * float(flops_g)

    r_critic = 0.0
    if critic_score is not None:
        critic_score = max(0.0, min(1.0, float(critic_score)))
        r_critic = w["critic"] * critic_score

    r_kl = 0.0
    if kl_div is not None and kl_div > 0.0:
        r_kl = - w["kl"] * float(kl_div)

    reward = r_build + r_forward_shape + r_backward + r_loss_drop + r_metric + r_eff + r_critic + r_kl
    reward = max(w["clip_lo"], min(w["clip_hi"], reward))

    return {
        "reward": reward,
        "r_build": r_build,
        "r_forward_shape": r_forward_shape,
        "r_backward": r_backward,
        "r_loss_drop": r_loss_drop,
        # Legacy aliases kept so older logging / callers do not break.
        "r_forward": r_forward_shape,
        "r_trainstep": r_backward,
        "r_metric": r_metric,
        "r_eff": r_eff,
        "r_critic": r_critic,
        "r_kl": r_kl,
    }

def _count_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def _freeze_dual_backbones(model: nn.Module) -> None:
    for backbone_name in ("backbone_a", "backbone_b"):
        backbone = getattr(model, backbone_name, None)
        if backbone is None:
            continue
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()


def _trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


@torch.no_grad()
def _quick_forward(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int],
    device: str = "cpu",
    n_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Try a single forward pass. Returns shape + latency diagnostics.
    """
    model.eval()
    x = torch.randn(*input_shape, device=device)
    try:
        t0 = time.time()
        y = model(x)
        torch.cuda.synchronize() if device.startswith("cuda") and torch.cuda.is_available() else None
        dt = (time.time() - t0) * 1000.0
        output_shape = tuple(y.shape) if hasattr(y, "shape") else None
        shape_ok = bool(
            isinstance(y, torch.Tensor)
            and y.dim() == 2
            and y.shape[0] == input_shape[0]
            and (n_classes is None or y.shape[1] == n_classes)
        )
        return {
            "forward_ok": True,
            "forward_shape_ok": shape_ok,
            "latency_ms": dt,
            "output_shape": output_shape,
        }
    except Exception as exc:
        return {
            "forward_ok": False,
            "forward_shape_ok": False,
            "latency_ms": None,
            "output_shape": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _toy_loader(
    n: int = 128,
    input_shape: Tuple[int, int, int, int] = (2, 3, 32, 32),
    n_classes: int = 10,
    device: str = "cpu",
    batch_size: int = 16
) -> DataLoader:
    """
    Fallback DataLoader with random data for quick sanity train/val.
    """
    N = max(n, batch_size)
    x = torch.randn((N,) + input_shape[1:], device=device)
    y = torch.randint(0, n_classes, (N,), device=device)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def _train_steps(
    model: nn.Module,
    train_loader: DataLoader,
    steps: int = 8,
    device: str = "cpu",
    n_classes: int = 10,
) -> Dict[str, Any]:
    """
    Mini-batch training steps. Returns training diagnostics.
    """
    model.train()
    _freeze_dual_backbones(model)
    loss_start = None
    loss_end = None
    steps_completed = 0
    try:
        trainable_params = _trainable_parameters(model)
        if not trainable_params:
            return {
                "backward_ok": False,
                "trained_step_ok": False,
                "loss_start": None,
                "loss_end": None,
                "loss_drop": None,
                "loss_drop_ok": False,
                "steps_completed": 0,
                "error": "RuntimeError: no trainable parameters remain after freezing backbones",
            }
        opt = torch.optim.SGD(trainable_params, lr=1e-3, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            if not isinstance(logits, torch.Tensor) or logits.dim() != 2:
                return {
                    "backward_ok": False,
                    "trained_step_ok": False,
                    "loss_start": None,
                    "loss_end": None,
                    "loss_drop": None,
                    "loss_drop_ok": False,
                    "steps_completed": steps_completed,
                    "error": f"RuntimeError: logits must be (N, C), got {tuple(logits.shape) if hasattr(logits, 'shape') else type(logits)}",
                }
            if logits.shape[0] != y.shape[0] or logits.shape[1] != n_classes:
                return {
                    "backward_ok": False,
                    "trained_step_ok": False,
                    "loss_start": None,
                    "loss_end": None,
                    "loss_drop": None,
                    "loss_drop_ok": False,
                    "steps_completed": steps_completed,
                    "error": f"RuntimeError: logits shape {tuple(logits.shape)} incompatible with labels {tuple(y.shape)} / classes {n_classes}",
                }
            loss = criterion(logits, y)
            if torch.isnan(loss) or torch.isinf(loss):
                return {
                    "backward_ok": False,
                    "trained_step_ok": False,
                    "loss_start": None,
                    "loss_end": None,
                    "loss_drop": None,
                    "loss_drop_ok": False,
                    "steps_completed": steps_completed,
                    "error": "RuntimeError: loss is NaN or Inf",
                }
            loss_value = float(loss.detach().item())
            if loss_start is None:
                loss_start = loss_value
            loss.backward()
            opt.step()
            loss_end = loss_value
            steps_completed += 1
            if steps_completed >= steps:
                break
        if steps_completed == 0 or loss_start is None or loss_end is None:
            return {
                "backward_ok": False,
                "trained_step_ok": False,
                "loss_start": loss_start,
                "loss_end": loss_end,
                "loss_drop": None,
                "loss_drop_ok": False,
                "steps_completed": steps_completed,
                "error": "RuntimeError: no training steps completed",
            }
        loss_drop = float(loss_start - loss_end)
        rel_drop_ok = loss_start > 0.0 and (loss_end <= loss_start * 0.98)
        loss_drop_ok = bool(loss_end < (loss_start - 1e-3) or rel_drop_ok)
        return {
            "backward_ok": True,
            "trained_step_ok": True,
            "loss_start": loss_start,
            "loss_end": loss_end,
            "loss_drop": loss_drop,
            "loss_drop_ok": loss_drop_ok,
            "steps_completed": steps_completed,
        }
    except Exception as exc:
        return {
            "backward_ok": False,
            "trained_step_ok": False,
            "loss_start": loss_start,
            "loss_end": loss_end,
            "loss_drop": None if loss_start is None or loss_end is None else float(loss_start - loss_end),
            "loss_drop_ok": False,
            "steps_completed": steps_completed,
            "error": f"{type(exc).__name__}: {exc}",
        }


@torch.no_grad()
def _quick_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    device: str = "cpu",
    max_batches: int = 10,
) -> float:
    """
    Evaluate accuracy in [0,1] on the provided loader without interpolation or fallback.
    """
    model.eval()
    correct = 0
    total = 0
    batches_seen = 0

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        if logits.dim() != 2:
            raise RuntimeError(f"logits must be (N,C), got {tuple(logits.shape)}")
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        batches_seen += 1
        if batches_seen >= max_batches:
            break

    return (correct / total) if total > 0 else 0.0


def _build_cifar10_loaders(cfg: "EvalConfig") -> Tuple[DataLoader, DataLoader]:
    from torchvision import datasets, transforms

    height = int(cfg.input_shape[2])
    width = int(cfg.input_shape[3])
    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010),
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=cfg.data_root,
        train=True,
        download=cfg.download,
        transform=train_transform,
    )
    val_dataset = datasets.CIFAR10(
        root=cfg.data_root,
        train=False,
        download=cfg.download,
        transform=val_transform,
    )

    if 0 < cfg.train_subset_size < len(train_dataset):
        train_dataset = Subset(train_dataset, range(cfg.train_subset_size))
    if 0 < cfg.val_subset_size < len(val_dataset):
        val_dataset = Subset(val_dataset, range(cfg.val_subset_size))

    train_batch = max(1, min(cfg.default_batch_size, len(train_dataset)))
    val_batch = max(1, min(cfg.default_batch_size, len(val_dataset)))

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


# ------------------------------------
# Public API: evaluate & produce reward
# ------------------------------------

@dataclass
class EvalConfig:
    device: str = "cpu"
    input_shape: Tuple[int, int, int, int] = (2, 3, 32, 32)  # (B,C,H,W)
    n_classes: int = 10
    train_steps: int = 8
    max_val_batches: int = 2
    default_batch_size: int = 32
    train_subset_size: int = 256
    val_subset_size: int = 128
    data_root: str = "data_v2"
    download: bool = True
    # Optional efficiency logging
    measure_latency: bool = True
    # Optional PPO/critic
    kl_div: Optional[float] = None
    critic_fn: Optional[Callable[[nn.Module, Dict[str, Any]], float]] = None
    # Optional reward weights
    weights: Optional[Dict[str, float]] = None


def _empty_eval_result(
    *,
    reward: float = 0.0,
    error: Optional[str] = None,
    seed_accuracy_baseline: Optional[float] = None,
) -> Dict[str, Any]:
    components = {
        "reward": reward,
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
    }
    result = {
        "reward": reward,
        "components": components,
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
    }
    if error:
        result["error"] = error
    return result


def evaluate_and_reward(
    *,
    # Either pass a ready model OR a builder that returns nn.Module
    model: Optional[nn.Module] = None,
    build_fn: Optional[Callable[[], nn.Module]] = None,

    # Optional loaders; if None, CIFAR-10 proxy loaders will be built
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,

    # Validation accuracy stays logging-only. Seed accuracy is logged but no longer drives RL reward.
    val_metric_baseline: Optional[float] = None,
    seed_accuracy_baseline: Optional[float] = None,

    # Config & weights
    cfg: EvalConfig = EvalConfig(),
) -> Dict[str, Any]:
    """
    End-to-end:
      1) Build model (if builder provided)
      2) Forward sanity
      3) 1..K train steps
      4) Quick validation (accuracy)
      5) (optional) critic score / KL
      6) Compute scalar reward
    Returns:
      {
        'reward': float,
        'components': {...},
        'val_metric': float or None,
        'built_ok': bool,
        'forward_ok': bool,
        'forward_shape_ok': bool,
        'trained_step_ok': bool,
        'backward_ok': bool,
        'loss_start': float | None,
        'loss_end': float | None,
        'loss_drop': float | None,
        'loss_drop_ok': bool,
        'latency_ms': float or None,
        'params_m': float,
    }
    """
    device = cfg.device
    built_ok = False
    forward_ok = False
    forward_shape_ok = False
    trained_step_ok = False
    backward_ok = False
    latency_ms = None
    params_m = None
    val_metric = None
    critic_score = None
    loss_start = None
    loss_end = None
    loss_drop = None
    loss_drop_ok = False
    train_acc = None
    seed_train_acc_gap = None
    seed_train_acc_improved = False

    # 1) Build or use provided model
    mdl = model
    local_train_loader = None
    local_val_loader = None

    try:
        if mdl is None and build_fn is not None:
            try:
                mdl = build_fn()
            except Exception:
                mdl = None
        if mdl is not None and isinstance(mdl, nn.Module):
            try:
                mdl.to(device)
                _freeze_dual_backbones(mdl)
                params_m = _count_params_m(mdl)
                built_ok = True
            except Exception:
                built_ok = False
                mdl = None

        # If build failed, compute minimal reward and return
        if not built_ok or mdl is None:
            components = compute_cv_reward_simple(
                built_ok=False,
                forward_shape_ok=False,
                backward_ok=False,
                loss_drop_ok=False,
                val_metric=None,
                val_metric_baseline=val_metric_baseline,
                latency_ms=None,
                params_m=None,
                flops_g=None,
                critic_score=None,
                kl_div=cfg.kl_div,
                weights=cfg.weights
            )
            return {
                "reward": components["reward"],
                "components": components,
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
            }

        # 2) Forward sanity (and optional latency)
        forward_result = _quick_forward(
            mdl,
            cfg.input_shape,
            device=device,
            n_classes=cfg.n_classes,
        )
        forward_ok = bool(forward_result["forward_ok"])
        forward_shape_ok = bool(forward_result["forward_shape_ok"])
        if cfg.measure_latency:
            latency_ms = forward_result["latency_ms"]

        # 3) Mini training steps
        if train_loader is None or val_loader is None:
            local_train_loader, local_val_loader = _build_cifar10_loaders(cfg)
        used_train_loader = train_loader if train_loader is not None else local_train_loader
        train_result = _train_steps(
            mdl,
            used_train_loader,
            steps=cfg.train_steps,
            device=device,
            n_classes=cfg.n_classes,
        )
        backward_ok = bool(train_result["backward_ok"])
        trained_step_ok = bool(train_result["trained_step_ok"])
        loss_start = train_result["loss_start"]
        loss_end = train_result["loss_end"]
        loss_drop = train_result["loss_drop"]
        loss_drop_ok = bool(train_result["loss_drop_ok"])

        # 4) Quick validation (accuracy)
        used_val_loader = val_loader if val_loader is not None else local_val_loader
        train_metric_batches = max(1, min(cfg.train_steps, int(train_result["steps_completed"] or 0) or cfg.train_steps))
        train_acc = _quick_accuracy(
            mdl,
            used_train_loader,
            device=device,
            max_batches=train_metric_batches,
        )
        val_metric = _quick_accuracy(
            mdl,
            used_val_loader,
            device=device,
            max_batches=cfg.max_val_batches,
        )
        if seed_accuracy_baseline is not None:
            seed_train_acc_gap = float(train_acc - seed_accuracy_baseline)
            seed_train_acc_improved = bool(seed_train_acc_gap > 0.0)

        # 5) Optional critic score
        if cfg.critic_fn is not None:
            try:
                critic_score = float(cfg.critic_fn(mdl, {
                    "built_ok": built_ok,
                    "forward_ok": forward_ok,
                    "forward_shape_ok": forward_shape_ok,
                    "trained_step_ok": trained_step_ok,
                    "backward_ok": backward_ok,
                    "loss_start": loss_start,
                    "loss_end": loss_end,
                    "loss_drop": loss_drop,
                    "loss_drop_ok": loss_drop_ok,
                    "val_metric": val_metric,
                    "params_m": params_m,
                    "latency_ms": latency_ms
                }))
                critic_score = max(0.0, min(1.0, critic_score))
            except Exception:
                critic_score = None

        # 6) Compute reward
        components = compute_cv_reward_simple(
            built_ok=built_ok,
            forward_shape_ok=forward_shape_ok,
            backward_ok=backward_ok,
            loss_drop_ok=loss_drop_ok,
            val_metric=val_metric,
            val_metric_baseline=val_metric_baseline,
            latency_ms=latency_ms,
            params_m=params_m,
            flops_g=None,                 # flops left as None (optional to integrate)
            critic_score=critic_score,
            kl_div=cfg.kl_div,
            weights=cfg.weights
        )

        return {
            "reward": components["reward"],
            "components": components,
            "val_metric": val_metric,
            "built_ok": built_ok,
            "forward_ok": forward_ok,
            "forward_shape_ok": forward_shape_ok,
            "trained_step_ok": trained_step_ok,
            "backward_ok": backward_ok,
            "loss_start": loss_start,
            "loss_end": loss_end,
            "loss_drop": loss_drop,
            "loss_drop_ok": loss_drop_ok,
            "train_acc": train_acc,
            "seed_accuracy_baseline": seed_accuracy_baseline,
            "seed_train_acc_gap": seed_train_acc_gap,
            "seed_train_acc_improved": seed_train_acc_improved,
            "accuracy_baseline": seed_accuracy_baseline,
            "train_acc_gain": seed_train_acc_gap,
            "train_acc_improved": seed_train_acc_improved,
            "group_baseline_train_acc": None,
            "group_train_acc_gain": None,
            "group_train_acc_improved": False,
            "reward_batch_index": None,
            "reward_group_id": None,
            "group_warmup": False,
            "latency_ms": latency_ms,
            "params_m": params_m,
        }

    finally:
        # Clean up memory explicitly and GUARANTEED
        if mdl is not None:
            try:
                mdl.to("cpu")
            except Exception:
                pass
            del mdl
        if local_train_loader is not None:
            del local_train_loader
        if local_val_loader is not None:
            del local_val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def _safe_exec_code(code: str) -> Dict[str, Any]:
    """
    Execute user-provided model code with torch/nn in scope and return the namespace.
    Expect class Net(in_shape, out_shape, prm, device) to exist.
    """
    ns: Dict[str, Any] = {"__builtins__": __builtins__, "torch": torch, "nn": nn}
    exec(code, ns, ns)
    return ns


def build_fn_from_code(
    code: str,
    in_shape: Tuple[int, int, int, int],
    out_shape: Tuple[int, ...],
    prm: Dict[str, Any],
    device_str: str = "cpu",
) -> Callable[[], nn.Module]:
    """
    Returns a builder function that instantiates Net(in_shape, out_shape, prm, device).
    """
    ns = _safe_exec_code(code)
    Net = ns.get("Net", None)
    if not isinstance(Net, type):
        raise RuntimeError("Net class not found or invalid in provided code.")
    device = torch.device(device_str)

    def _builder() -> nn.Module:
        return Net(in_shape, out_shape, prm, device)

    return _builder


def _serialize_eval_cfg(cfg: EvalConfig) -> Dict[str, Any]:
    return asdict(cfg)


def _deserialize_eval_cfg(cfg_payload: Optional[Dict[str, Any]]) -> EvalConfig:
    if cfg_payload is None:
        return EvalConfig()
    return EvalConfig(**cfg_payload)


def _loader_cache_key(cfg: EvalConfig) -> Tuple[Any, ...]:
    return (
        tuple(cfg.input_shape),
        cfg.default_batch_size,
        cfg.train_subset_size,
        cfg.val_subset_size,
        cfg.data_root,
        cfg.download,
    )


def _get_or_create_eval_worker() -> _PersistentEvalWorkerSession:
    global _PERSISTENT_EVAL_WORKER
    if _PERSISTENT_EVAL_WORKER is None:
        _PERSISTENT_EVAL_WORKER = _PersistentEvalWorkerSession()
    return _PERSISTENT_EVAL_WORKER


def shutdown_eval_worker() -> None:
    global _PERSISTENT_EVAL_WORKER
    if _PERSISTENT_EVAL_WORKER is None:
        return
    _PERSISTENT_EVAL_WORKER.close()
    _PERSISTENT_EVAL_WORKER = None


def get_eval_worker_diagnostics() -> Optional[Dict[str, Any]]:
    if _PERSISTENT_EVAL_WORKER is None:
        return None
    return _PERSISTENT_EVAL_WORKER.diagnostics()


def _persistent_eval_worker_entry(conn) -> None:
    try:
        conn.send(
            {
                "cmd": "worker_ready",
                "pid": os.getpid(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
            }
        )
        _persistent_eval_worker_loop(conn)
    except Exception as exc:
        try:
            conn.send(
                {
                    "cmd": "worker_init_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        except Exception:
            pass
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass


def evaluate_code_and_reward(
    code: str,
    *,
    in_shape: Tuple[int, int, int, int] = (8, 3, 32, 32),
    out_shape: Tuple[int, ...] = (10,),
    prm: Dict[str, Any] = None,
    device: str = "cpu",
    val_metric_baseline: Optional[float] = None,
    seed_accuracy_baseline: Optional[float] = None,
    cfg: Optional[EvalConfig] = None,
) -> Dict[str, Any]:
    worker = _get_or_create_eval_worker()
    worker_device = "cpu"
    effective_cfg = replace(
        cfg,
        device=worker_device,
    ) if cfg is not None else EvalConfig(
        device=worker_device,
        input_shape=in_shape,
        n_classes=int(out_shape[0]),
        train_steps=8,
        max_val_batches=2,
        measure_latency=True,
        kl_div=None,
        critic_fn=None,
        weights=None,
    )
    payload = {
        "cmd": "evaluate",
        "code": code,
        "in_shape": tuple(in_shape),
        "out_shape": tuple(out_shape),
        "prm": prm,
        "device": worker_device,
        "val_metric_baseline": val_metric_baseline,
        "seed_accuracy_baseline": seed_accuracy_baseline,
        "cfg": _serialize_eval_cfg(effective_cfg),
    }
    return worker.request(payload, timeout=300)


def _persistent_eval_worker_loop(conn) -> None:
    loader_cache: Dict[Tuple[Any, ...], Tuple[DataLoader, DataLoader]] = {}
    try:
        while True:
            try:
                request = conn.recv()
            except EOFError:
                break

            if not isinstance(request, dict):
                conn.send(_empty_eval_result(error="Persistent eval worker received a non-dict request"))
                continue

            if request.get("cmd") == "shutdown":
                break
            if request.get("cmd") != "evaluate":
                conn.send(_empty_eval_result(error=f"Persistent eval worker received unknown command: {request.get('cmd')!r}"))
                continue

            cfg = _deserialize_eval_cfg(request.get("cfg"))
            loader_key = _loader_cache_key(cfg)
            if loader_key not in loader_cache:
                loader_cache[loader_key] = _build_cifar10_loaders(cfg)
            train_loader, val_loader = loader_cache[loader_key]

            try:
                result = _evaluate_code_and_reward_direct(
                    request["code"],
                    in_shape=tuple(request["in_shape"]),
                    out_shape=tuple(request["out_shape"]),
                    prm=request.get("prm"),
                    device=request.get("device", "cpu"),
                    val_metric_baseline=request.get("val_metric_baseline"),
                    seed_accuracy_baseline=request.get("seed_accuracy_baseline"),
                    cfg=cfg,
                    train_loader=train_loader,
                    val_loader=val_loader,
                )
            except Exception as exc:
                result = _empty_eval_result(
                    reward=-1.0,
                    error=f"{type(exc).__name__}: {exc}",
                    seed_accuracy_baseline=request.get("seed_accuracy_baseline"),
                )
                result["components"]["reward"] = -1.0

            conn.send(result)
    finally:
        try:
            conn.close()
        except Exception:
            pass

def _evaluate_code_and_reward_direct(
    code: str,
    *,
    in_shape: Tuple[int, int, int, int] = (8, 3, 32, 32),
    out_shape: Tuple[int, ...] = (10,),
    prm: Dict[str, Any] = None,
    device: str = "cpu",
    val_metric_baseline: Optional[float] = None,
    seed_accuracy_baseline: Optional[float] = None,
    cfg: Optional[EvalConfig] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
) -> Dict[str, Any]:
    """
    The original evaluation logic (renamed).
    """
    if prm is None:
        prm = {"lr": 1e-2, "momentum": 0.9}
    defaults = {"lr": 1e-2, "momentum": 0.9, "batch": 32, "epoch": 1, "transform": None}
    prm = {**defaults, **prm} 
    if cfg is None:
        cfg = EvalConfig(
            device=device,
            input_shape=in_shape,
            n_classes=int(out_shape[0]),
            train_steps=8,
            max_val_batches=2,
            measure_latency=True,
            kl_div=None,
            critic_fn=None,
            weights=None,
        )

    try:
        builder = build_fn_from_code(code, in_shape, out_shape, prm, device)
    except Exception as e:
        # Pass through error type so reward_fn can assign layered partial rewards
        error_type = type(e).__name__
        error_msg = f"{error_type}: {e}"
        return _empty_eval_result(error=error_msg, seed_accuracy_baseline=seed_accuracy_baseline)
    try: 
        res = evaluate_and_reward(
            build_fn=builder,
            train_loader=train_loader,
            val_loader=val_loader,
            val_metric_baseline=val_metric_baseline,
            seed_accuracy_baseline=seed_accuracy_baseline,
            cfg=cfg,
        )
        return res
    except Exception as e:
        error_type = type(e).__name__
        error_msg = f"{error_type}: {e}"
        return _empty_eval_result(error=error_msg, seed_accuracy_baseline=seed_accuracy_baseline)


atexit.register(shutdown_eval_worker)

if __name__ == "__main__":
    demo_code = r'''
import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.device = device
        
                           
        self.conv1 = nn.Conv2d(in_shape[0], prm['conv1_filters'], kernel_size=prm['conv1_kernel'], stride=prm['conv1_stride'])
        self.bn1 = nn.BatchNorm2d(prm['conv1_filters']) if prm['use_batchnorm'] else None
        self.conv2 = nn.Conv2d(prm['conv1_filters'], prm['conv2_filters'], kernel_size=prm['conv2_kernel'])
        self.bn2 = nn.BatchNorm2d(prm['conv2_filters']) if prm['use_batchnorm'] else None
        self.conv3 = nn.Conv2d(prm['conv2_filters'], prm['conv3_filters'], kernel_size=prm['conv3_kernel'])
        self.bn3 = nn.BatchNorm2d(prm['conv3_filters']) if prm['use_batchnorm'] else None
        self.conv4 = nn.Conv2d(prm['conv3_filters'], prm['conv4_filters'], kernel_size=prm['conv4_kernel'])
        self.bn4 = nn.BatchNorm2d(prm['conv4_filters']) if prm['use_batchnorm'] else None
        self.conv5 = nn.Conv2d(prm['conv4_filters'], prm['conv5_filters'], kernel_size=prm['conv5_kernel'])
        self.bn5 = nn.BatchNorm2d(prm['conv5_filters']) if prm['use_batchnorm'] else None
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(prm['fc1_neurons'], prm['fc2_neurons'])
        self.fc2 = nn.Linear(prm['fc2_neurons'], out_shape)
        
                       
        self.dropout = nn.Dropout(p=prm['dropout'])
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x'''
    res = evaluate_code_and_reward(
        demo_code,
        in_shape=(1, 3, 224, 224),
        out_shape=(10,),
        prm={"lr": 1e-2, "momentum": 0.9, "dropout": 0.5},
        device="cpu",
        val_metric_baseline=0.10,
    )
    print(res)
