#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


NN_GPT_ROOT = Path(__file__).resolve().parent
RL_ROOT = NN_GPT_ROOT.parent
NN_DATASET_ROOT = RL_ROOT / "nn-dataset"

if not NN_DATASET_ROOT.exists():
    raise FileNotFoundError(f"Expected sibling repo at {NN_DATASET_ROOT}")

if str(NN_DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(NN_DATASET_ROOT))
if str(NN_GPT_ROOT) not in sys.path:
    sys.path.insert(0, str(NN_GPT_ROOT))

import torch

import ab.nn.api as nn_api
from ab.gpt import TuneRL


TASK = "img-classification"
DATASET = "cifar-10"
METRIC = "acc"
EPOCH_LIMIT_MINUTES = 40.0
INPUT_SHAPE = (1, 3, 256, 256)
OUTPUT_SHAPE = (10,)
BACKBONE_A = "mobilenet_v3_small"
BACKBONE_B = "shufflenet_v2_x0_5"
BACKBONE_WEIGHTS = "DEFAULT"
BASE_PRM: dict[str, Any] = {
    "lr": 0.01,
    "momentum": 0.9,
    "dropout": 0.2,
    "batch": 64,
    "epoch": 1,
    "transform": "norm_256_flip",
    "num_workers": 4,
}


@dataclass
class ProbeSummary:
    label: str
    freeze_backbones: bool
    model_name: str | None
    elapsed_seconds: float
    accuracy: float | None
    accuracy_to_time: float | None
    code_score: float | None
    total_params: int
    trainable_params: int
    backbone_a_trainable_params: int
    backbone_b_trainable_params: int
    head_trainable_params: int


def build_completion_xml() -> str:
    block_code = """
def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if dropout_prob > 0.0:
        layers.append(nn.Dropout2d(p=float(dropout_prob)))
    return nn.Sequential(*layers)
""".strip()

    init_code = f"""
def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
    super().__init__()
    self.pattern = "FreezeVsUnfreezeProbe"
    self.device = device
    self.use_amp = str(device).startswith("cuda")
    self.freeze_backbones = bool(prm.get("freeze_backbones", True))
    self._input_spec = tuple(in_shape[1:])
    in_channels = int(in_shape[1])
    self.backbone_a = TorchVision(model="{BACKBONE_A}", weights={BACKBONE_WEIGHTS!r}, in_channels=in_channels)
    self.backbone_b = TorchVision(model="{BACKBONE_B}", weights={BACKBONE_WEIGHTS!r}, in_channels=in_channels)
    self.fuse_dropout = nn.Dropout(p=float(prm.get("dropout", 0.2)))
    self.classifier = nn.Identity()
    self.infer_dimensions_dynamically(int(out_shape[0]))
""".strip()

    forward_code = """
def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
    x = self._norm4d(x)
    feat_a = adaptive_pool_flatten(self.backbone_a(x)).flatten(1)
    feat_b = adaptive_pool_flatten(self.backbone_b(x)).flatten(1)
    fused = torch.cat([feat_a, feat_b], dim=1)
    if is_probing:
        return fused
    return self.classifier(self.fuse_dropout(fused))
""".strip()

    return TuneRL.render_completion_xml(block_code, init_code, forward_code)


def reconstruct_probe_code() -> str:
    completion = build_completion_xml()
    code = TuneRL.reconstruct_code(completion, pattern_name_override="FreezeVsUnfreezeProbe")
    if not code.strip():
        raise RuntimeError("Failed to reconstruct code from XML completion")
    return code


def load_net_class(code: str):
    scope: dict[str, Any] = {}
    exec(code, scope, scope)
    net_cls = scope.get("Net")
    if net_cls is None:
        raise RuntimeError("Reconstructed code did not define Net")
    return net_cls


def count_params(params) -> int:
    return sum(int(param.numel()) for param in params)


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def inspect_trainable_params(code: str, freeze_backbones: bool) -> dict[str, int]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prm = dict(BASE_PRM)
    prm["freeze_backbones"] = bool(freeze_backbones)
    net_cls = load_net_class(code)
    model = net_cls(INPUT_SHAPE, OUTPUT_SHAPE, prm, device)
    model.train_setup(prm)
    total_params = count_params(model.parameters())
    trainable_params = count_params(param for param in model.parameters() if param.requires_grad)
    backbone_a_trainable_params = count_params(
        param for param in model.backbone_a.parameters() if param.requires_grad
    )
    backbone_b_trainable_params = count_params(
        param for param in model.backbone_b.parameters() if param.requires_grad
    )
    head_trainable_params = max(
        0,
        trainable_params - backbone_a_trainable_params - backbone_b_trainable_params,
    )
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "backbone_a_trainable_params": backbone_a_trainable_params,
        "backbone_b_trainable_params": backbone_b_trainable_params,
        "head_trainable_params": head_trainable_params,
    }


def run_checknn_case(code: str, freeze_backbones: bool, label: str) -> ProbeSummary:
    prm = dict(BASE_PRM)
    prm["freeze_backbones"] = bool(freeze_backbones)
    param_stats = inspect_trainable_params(code, freeze_backbones)
    started_at = time.time()
    model_name, accuracy, accuracy_to_time, code_score = nn_api.check_nn(
        code,
        TASK,
        DATASET,
        METRIC,
        prm,
        False,
        None,
        None,
        False,
        EPOCH_LIMIT_MINUTES,
    )
    elapsed_seconds = max(0.0, time.time() - started_at)
    return ProbeSummary(
        label=label,
        freeze_backbones=bool(freeze_backbones),
        model_name=model_name,
        elapsed_seconds=elapsed_seconds,
        accuracy=maybe_float(accuracy),
        accuracy_to_time=maybe_float(accuracy_to_time),
        code_score=maybe_float(code_score),
        **param_stats,
    )


def main() -> None:
    code = reconstruct_probe_code()
    code_hash = hashlib.sha1(code.encode("utf-8")).hexdigest()[:12]
    print(
        json.dumps(
            {
                "task": TASK,
                "dataset": DATASET,
                "metric": METRIC,
                "epoch_limit_minutes": EPOCH_LIMIT_MINUTES,
                "input_shape": INPUT_SHAPE,
                "output_shape": OUTPUT_SHAPE,
                "backbone_a": BACKBONE_A,
                "backbone_b": BACKBONE_B,
                "backbone_weights": BACKBONE_WEIGHTS,
                "code_sha1_12": code_hash,
                "base_prm": BASE_PRM,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    results = [
        run_checknn_case(code, True, "freeze"),
        run_checknn_case(code, False, "unfreeze"),
    ]
    print(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True), flush=True)

    freeze_result, unfreeze_result = results
    accuracy_delta = None
    if freeze_result.accuracy is not None and unfreeze_result.accuracy is not None:
        accuracy_delta = unfreeze_result.accuracy - freeze_result.accuracy
    print(
        json.dumps(
            {
                "accuracy_delta_unfreeze_minus_freeze": accuracy_delta,
                "trainable_param_delta_unfreeze_minus_freeze": (
                    unfreeze_result.trainable_params - freeze_result.trainable_params
                ),
                "same_code_hash": True,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
