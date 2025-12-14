from typing import Optional, Dict, Tuple, Callable, Any
from dataclasses import dataclass
import time
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms
import os

# --- Reuse logic from ab.gpt.util.Reward.py ---

def _cifar10_loader(
    is_train: bool = True,
    device: str = "cpu",
    batch_size: int = 128,
    root: str = "./dataset",
    quick_debug_size: Optional[int] = None
) -> DataLoader:
    """
    Real CIFAR-10 Data Loader.
    Autodownloads to ./dataset/cifar-10-batches-py
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    os.makedirs(root, exist_ok=True)
    
    dataset = torchvision.datasets.CIFAR10(
        root=root, 
        train=is_train,
        download=True,
        transform=transform
    )
    
    # For quick RL feedback loops, we might want a smaller subset
    # instead of the full 50k images every time, unless running full training.
    if quick_debug_size and quick_debug_size < len(dataset):
        indices = torch.randperm(len(dataset))[:quick_debug_size]
        dataset = Subset(dataset, indices)
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=0)

def _toy_loader(
    n: int = 128,
    input_shape: Tuple[int, int, int, int] = (2, 3, 32, 32),
    n_classes: int = 10,
    device: str = "cpu",
    batch_size: int = 16
) -> DataLoader:
    # Use CIFAR-10 as default instead of random noise
    # Map input_shape to quick_debug_size logic if needed, but mostly ignored
    return _cifar10_loader(
        is_train=True, 
        device=device, 
        batch_size=batch_size,
        quick_debug_size=512 # Default to small subset for fast feedback
    )

def compute_cv_reward_simple(
    *,
    built_ok: bool,
    forward_ok: bool,
    trained_step_ok: bool,
    val_metric: Optional[float],
    val_metric_baseline: Optional[float] = None,
    latency_ms: Optional[float] = None,
    params_m: Optional[float] = None,
    flops_g: Optional[float] = None,
    critic_score: Optional[float] = None,
    kl_div: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Simple scalar reward for CV model generation.
    Returns a dict with 'reward' and component terms.
    """
    w = {
        "build": 0.2,         # bonus if model builds
        "forward": 0.2,       # bonus if forward works
        "trainstep": 0.2,     # bonus if 1 mini-batch trains
        "metric_gain": 1.0,   # (val - baseline)
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

    r_build    = w["build"]   if built_ok else 0.0
    r_forward  = w["forward"] if forward_ok else 0.0
    r_train    = w["trainstep"] if trained_step_ok else 0.0

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

    reward = r_build + r_forward + r_train + r_metric + r_eff + r_critic + r_kl
    reward = max(w["clip_lo"], min(w["clip_hi"], reward))

    return {
        "reward": reward,
        "r_build": r_build,
        "r_forward": r_forward,
        "r_trainstep": r_train,
        "r_metric": r_metric,
        "r_eff": r_eff,
        "r_critic": r_critic,
        "r_kl": r_kl,
    }

def _count_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def _quick_forward(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int],
    device: str = "cpu"
) -> Tuple[bool, Optional[float]]:
    """
    Try a single forward pass. Returns (ok, latency_ms).
    """
    model.eval()
    x = torch.randn(*input_shape, device=device)
    try:
        t0 = time.time()
        y = model(x)
        if hasattr(y, "shape"):
             _ = tuple(y.shape)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.time() - t0) * 1000.0
        return True, dt
    except Exception:
        return False, None


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
    steps: int = 1,
    device: str = "cpu"
) -> bool:
    """
    Mini-batch training steps. Returns ok flag.
    """
    model.train()
    try:
        opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        k = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            # FRACTAL SPEED HACK: Use shallowest path for training if available
            if hasattr(model, 'forward_shallowest'):
                logits = model.forward_shallowest(x)
            else:
                logits = model(x)
                
            loss = criterion(logits, y)
            if torch.isnan(loss) or torch.isinf(loss):
                return False
            loss.backward()
            opt.step()
            k += 1
            if k >= steps:
                break
        return True
    except Exception:
        return False


@torch.no_grad()
def _quick_validate_acc(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = "cpu",
    max_batches: int = 10
) -> float:
    """
    Evaluation. Returns accuracy in [0,1].
    """
    try:
        model.eval()
        correct, total, bs = 0, 0, 0
        for x, y in val_loader:
            x = x.to(device); y = y.to(device)
            # FRACTAL SPEED HACK: Use shallowest path if available
            if hasattr(model, 'forward_shallowest'):
                logits = model.forward_shallowest(x)
            else:
                logits = model(x)
                                   
            if logits.dim() != 2:
                # raise RuntimeError(f"logits must be (N,C), got {tuple(logits.shape)}")
                return 0.0
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
            bs += 1
            if bs >= max_batches:
                break
        return (correct / total) if total > 0 else 0.0
    except Exception:
        return 0.0


# ------------------------------------
# Public API: evaluate & produce reward
# ------------------------------------

@dataclass
class EvalConfig:
    device: str = "cpu"
    input_shape: Tuple[int, int, int, int] = (2, 3, 32, 32)  # (B,C,H,W)
    n_classes: int = 10
    train_steps: int = 1
    max_val_batches: int = 10
    measure_latency: bool = True
    kl_div: Optional[float] = None
    critic_fn: Optional[Callable[[nn.Module, Dict[str, Any]], float]] = None
    weights: Optional[Dict[str, float]] = None


def evaluate_and_reward(
    *,
    model: Optional[nn.Module] = None,
    build_fn: Optional[Callable[[], nn.Module]] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    val_metric_baseline: Optional[float] = None,
    cfg: EvalConfig = EvalConfig(),
) -> Dict[str, Any]:
    """
    Evaluates model and returns comprehensive reward details.
    """
    device = cfg.device
    built_ok = False
    forward_ok = False
    trained_step_ok = False
    latency_ms = None
    params_m = None
    val_metric = None
    critic_score = None

    # 1) Build or use provided model
    mdl = model
    if mdl is None and build_fn is not None:
        try:
            mdl = build_fn()
        except Exception as e:
            print(f"DEBUG: build_fn failed: {e}")
            mdl = None
    if mdl is not None and isinstance(mdl, nn.Module):
        try:
            mdl.to(device)
            params_m = _count_params_m(mdl)
            built_ok = True
        except Exception as e:
            print(f"DEBUG: mdl setup failed: {e}")
            built_ok = False
            mdl = None

    # If build failed, compute minimal reward
    if not built_ok or mdl is None:
        components = compute_cv_reward_simple(
            built_ok=False,
            forward_ok=False,
            trained_step_ok=False,
            val_metric=None,
            val_metric_baseline=val_metric_baseline,
            kl_div=cfg.kl_div,
            weights=cfg.weights
        )
        return {
            "reward": components["reward"],
            "components": components,
            "val_metric": None,
            "built_ok": False,
            "forward_ok": False,
            "trained_step_ok": False,
            "latency_ms": None,
            "params_m": None,
        }

    # 2) Forward sanity
    if cfg.measure_latency:
        forward_ok, latency_ms = _quick_forward(mdl, cfg.input_shape, device=device)
    else:
        forward_ok, _ = _quick_forward(mdl, cfg.input_shape, device=device)

    # 3) Mini training steps
    if train_loader is None:
        # Use simple loader if none provided
        train_loader = _toy_loader(
            n=128,
            input_shape=cfg.input_shape,
            n_classes=cfg.n_classes,
            device=device,
            batch_size=max(2, cfg.input_shape[0])
        )
    trained_step_ok = _train_steps(mdl, train_loader, steps=cfg.train_steps, device=device)

    # 4. Quick validation (accuracy)
    if val_loader is None:
        val_loader = _cifar10_loader(
            is_train=False,
            device=device,
            batch_size=128,
            quick_debug_size=512 # Fast validation on 500 images
        )
    val_metric = _quick_validate_acc(mdl, val_loader, device=device, max_batches=cfg.max_val_batches)

    # 5) Compute reward
    components = compute_cv_reward_simple(
        built_ok=built_ok,
        forward_ok=forward_ok,
        trained_step_ok=trained_step_ok,
        val_metric=val_metric,
        val_metric_baseline=val_metric_baseline,
        latency_ms=latency_ms,
        params_m=params_m,
        kl_div=cfg.kl_div,
        weights=cfg.weights
    )

    return {
        "reward": components["reward"],
        "components": components,
        "val_metric": val_metric,
        "built_ok": built_ok,
        "forward_ok": forward_ok,
        "trained_step_ok": trained_step_ok,
        "latency_ms": latency_ms,
        "params_m": params_m,
    }


def _safe_exec_code(code: str) -> Dict[str, Any]:
    """
    Execute user-provided model code with torch/nn in scope and return the namespace.
    """
    ns: Dict[str, Any] = {"__builtins__": __builtins__, "torch": torch, "nn": nn, "F": F}
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


def evaluate_code_and_reward(
    code: str,
    *,
    in_shape: Tuple[int, int, int, int] = (8, 3, 32, 32),
    out_shape: Tuple[int, ...] = (10,),
    prm: Dict[str, Any] = None,
    device: str = "cpu",
    val_metric_baseline: Optional[float] = None,
    cfg: Optional[EvalConfig] = None,
    log_file: Optional[str] = "dataset/mutation_log.jsonl", # NEW: Log File
    prompt_used: Optional[str] = None # NEW: Prompt used
) -> Dict[str, Any]:
    """
    High-level entry: evaluate code and calculate reward.
    New: Supports logging for fine-tuning dataset.
    """
    if prm is None:
        prm = {"lr": 1e-2, "momentum": 0.9, "dropout": 0.5}
    else:
        # Ensure defaults exist if user provided partial prm
        defaults = {"lr": 1e-2, "momentum": 0.9, "dropout": 0.5}
        for k, v in defaults.items():
            if k not in prm:
                prm[k] = v
        
    if cfg is None:
        cfg = EvalConfig(
            device=device,
            input_shape=in_shape,
            n_classes=int(out_shape[0]),
            train_steps=1,
            max_val_batches=10, 
            measure_latency=True,
            weights={"metric_gain": 2.0} # Upweight accuracy gain
        )

    try:
        builder = build_fn_from_code(code, in_shape, out_shape, prm, device)
    except Exception as e:
        # Build failed
        return {
            "reward": -1.0, 
            "val_metric": 0.0,
            "error": str(e)
        }

    try: 
        res = evaluate_and_reward(
            build_fn=builder,
            val_metric_baseline=val_metric_baseline,
            cfg=cfg,
        )
        
        # --- Fine-Tuning Logging ---
        if log_file and res["built_ok"] and (prompt_used is not None):
            # Only log valid models that perform at least somewhat basically
            # We want to capture the link: Prompt -> Code -> Reward
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            log_entry = {
                "prompt": prompt_used,
                "code": code,
                "reward": res["reward"],
                "accuracy": res["val_metric"],
                "baseline_acc": val_metric_baseline or 0.0,
                "params": res.get("params_m"),
                "timestamp": time.time()
            }
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        return res
        
    except Exception as e:
        return {
            "reward": 0.0,
            "val_metric": 0.0,
             "error": str(e)
        }
