from typing import Optional, Dict, Tuple, Callable, Any
from dataclasses import dataclass, asdict, replace
import atexit
import time
import gc
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import importlib
import os
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset


_NN_DATASET_IMPORT_READY = False


class PersistentEvalWorkerError(RuntimeError):
    pass


class EvalTimeException(RuntimeError):
    def __init__(
        self,
        *,
        estimated_total_seconds: float,
        eval_limit_seconds: float,
        phase: str,
        partial: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.estimated_total_seconds = float(estimated_total_seconds)
        self.eval_limit_seconds = float(eval_limit_seconds)
        self.phase = str(phase)
        self.partial = dict(partial or {})
        super().__init__(
            f"Estimated evaluation time {self.estimated_total_seconds:.1f}s exceeds "
            f"limit {self.eval_limit_seconds:.1f}s during {self.phase}"
        )


def _safe_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _safe_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _visible_cuda_device_tokens() -> Optional[list[str]]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    raw = raw.strip()
    if raw in {"", "-1"}:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def get_distributed_runtime_info() -> Dict[str, Any]:
    visible_gpu_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    world_size = max(1, _safe_int_env("WORLD_SIZE", 1))
    rank = _safe_int_env("RANK", 0)
    raw_local_rank = _safe_int_env("LOCAL_RANK", 0)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            world_size = max(world_size, int(torch.distributed.get_world_size()))
        except Exception:
            pass
        try:
            rank = int(torch.distributed.get_rank())
        except Exception:
            pass

    if visible_gpu_count <= 0:
        local_rank = 0
    elif visible_gpu_count == 1:
        local_rank = 0
    elif 0 <= raw_local_rank < visible_gpu_count:
        local_rank = raw_local_rank
    else:
        local_rank = 0

    visible_gpu_tokens = _visible_cuda_device_tokens()
    train_gpu = local_rank if visible_gpu_count > 0 else None
    train_gpu_token = None
    if train_gpu is not None:
        if visible_gpu_tokens and train_gpu < len(visible_gpu_tokens):
            train_gpu_token = visible_gpu_tokens[train_gpu]
        else:
            train_gpu_token = str(int(train_gpu))

    return {
        "distributed": world_size > 1,
        "world_size": world_size,
        "rank": rank,
        "raw_local_rank": raw_local_rank,
        "local_rank": local_rank,
        "visible_gpu_count": visible_gpu_count,
        "visible_gpu_tokens": list(visible_gpu_tokens or []),
        "train_gpu": train_gpu,
        "train_gpu_token": train_gpu_token,
    }


class EvalTimeBudget:
    def __init__(self, cfg: "EvalConfig") -> None:
        self.eval_limit_seconds = float(cfg.eval_limit_seconds)
        self.budget_probe_batches = max(1, int(cfg.budget_probe_batches))
        self.start_time = time.time()
        self.completed_units = 0
        self.observed_train_batches = 0
        self.estimated_total_seconds: Optional[float] = None
        self.total_units = 1

    def set_expected_work(
        self,
        *,
        train_batches: int,
        train_accuracy_batches: int,
        val_accuracy_batches: int,
    ) -> None:
        self.total_units = max(
            1,
            2
            + max(0, int(train_batches))
            + max(1, int(train_accuracy_batches))
            + max(1, int(val_accuracy_batches)),
        )

    def check(self, phase: str) -> None:
        self._maybe_raise_elapsed(phase)
        self._maybe_raise_estimate(phase)

    def mark_build_complete(self) -> None:
        self.completed_units += 1
        self.check("build")

    def mark_forward_complete(self) -> None:
        self.completed_units += 1
        self.check("forward")

    def mark_train_batch_start(self) -> None:
        self.check("train")

    def mark_train_batch_end(self) -> None:
        self.observed_train_batches += 1
        self.completed_units += 1
        self.check("train")

    def mark_accuracy_batch_start(self, phase: str) -> None:
        self.check(phase)

    def mark_accuracy_batch_end(self, phase: str) -> None:
        self.completed_units += 1
        self.check(phase)

    def _estimate_total_seconds(self) -> float:
        elapsed = max(1e-3, time.time() - self.start_time)
        completed_units = max(1, self.completed_units)
        estimated_total_seconds = elapsed * self.total_units / completed_units
        self.estimated_total_seconds = estimated_total_seconds
        return estimated_total_seconds

    def _maybe_raise_elapsed(self, phase: str) -> None:
        elapsed = max(1e-3, time.time() - self.start_time)
        if elapsed <= self.eval_limit_seconds:
            return
        estimated_total_seconds = max(elapsed, self._estimate_total_seconds())
        raise EvalTimeException(
            estimated_total_seconds=estimated_total_seconds,
            eval_limit_seconds=self.eval_limit_seconds,
            phase=phase,
        )

    def _maybe_raise_estimate(self, phase: str) -> None:
        if self.observed_train_batches < self.budget_probe_batches:
            return
        estimated_total_seconds = self._estimate_total_seconds()
        if estimated_total_seconds > self.eval_limit_seconds:
            raise EvalTimeException(
                estimated_total_seconds=estimated_total_seconds,
                eval_limit_seconds=self.eval_limit_seconds,
                phase=phase,
            )


def get_reward_worker_plan() -> Dict[str, Any]:
    runtime = get_distributed_runtime_info()
    visible_gpu_count = int(runtime["visible_gpu_count"])
    deepspeed_enabled = _safe_bool_env("NNGPT_SFT_USE_DEEPSPEED", False)
    if visible_gpu_count <= 0:
        return {
            "mode": "cpu_fallback",
            "visible_gpu_count": 0,
            "train_gpu": None,
            "reward_gpu_indices": [],
            "reward_gpu_tokens": [],
            "shared_train_gpu": False,
            "pool_size": 1,
            "workers_per_gpu": 1,
            "deepspeed_enabled": deepspeed_enabled,
            "distributed": bool(runtime["distributed"]),
            "rank": int(runtime["rank"]),
            "local_rank": int(runtime["local_rank"]),
            "world_size": int(runtime["world_size"]),
        }
    if runtime["distributed"]:
        train_gpu = runtime["train_gpu"]
        workers_per_gpu = 2 if deepspeed_enabled else 1
        reward_gpu_indices = [int(train_gpu)] * workers_per_gpu if train_gpu is not None else []
        reward_gpu_tokens = (
            [str(runtime["train_gpu_token"])] * workers_per_gpu
            if runtime["train_gpu_token"] is not None
            else []
        )
        return {
            "mode": "distributed_local_gpu_dual" if workers_per_gpu > 1 else "distributed_local_gpu",
            "visible_gpu_count": visible_gpu_count,
            "train_gpu": train_gpu,
            "reward_gpu_indices": reward_gpu_indices,
            "reward_gpu_tokens": reward_gpu_tokens,
            "shared_train_gpu": True,
            "pool_size": max(1, len(reward_gpu_indices)),
            "workers_per_gpu": workers_per_gpu,
            "deepspeed_enabled": deepspeed_enabled,
            "distributed": True,
            "rank": int(runtime["rank"]),
            "local_rank": int(runtime["local_rank"]),
            "world_size": int(runtime["world_size"]),
        }
    if visible_gpu_count == 1:
        return {
            "mode": "single_gpu_shared",
            "visible_gpu_count": 1,
            "train_gpu": 0,
            "reward_gpu_indices": [0],
            "reward_gpu_tokens": [str(runtime["train_gpu_token"] or "0")],
            "shared_train_gpu": True,
            "pool_size": 1,
            "workers_per_gpu": 1,
            "deepspeed_enabled": deepspeed_enabled,
            "distributed": False,
            "rank": int(runtime["rank"]),
            "local_rank": int(runtime["local_rank"]),
            "world_size": int(runtime["world_size"]),
        }
    reward_gpu_indices = list(range(visible_gpu_count))
    reward_gpu_tokens = list(runtime["visible_gpu_tokens"] or [str(index) for index in reward_gpu_indices])
    return {
        "mode": "multi_gpu_pool",
        "visible_gpu_count": visible_gpu_count,
        "train_gpu": 0,
        "reward_gpu_indices": reward_gpu_indices,
        "reward_gpu_tokens": reward_gpu_tokens,
        "shared_train_gpu": True,
        "pool_size": len(reward_gpu_indices),
        "workers_per_gpu": 1,
        "deepspeed_enabled": deepspeed_enabled,
        "distributed": False,
        "rank": int(runtime["rank"]),
        "local_rank": int(runtime["local_rank"]),
        "world_size": int(runtime["world_size"]),
    }


class _PersistentEvalWorkerSession:
    def __init__(
        self,
        *,
        assigned_gpu: Optional[int],
        assigned_cuda_visible_device: Optional[str],
        worker_slot: int,
    ) -> None:
        from ab.gpt.util.reward_worker_bootstrap import reward_worker_main

        self._assigned_gpu = assigned_gpu
        self._assigned_cuda_visible_device = (
            None if assigned_cuda_visible_device is None else str(assigned_cuda_visible_device)
        )
        self._worker_slot = int(worker_slot)
        self.ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = self.ctx.Pipe()
        self._process = self.ctx.Process(
            target=reward_worker_main,
            args=(child_conn, assigned_gpu, self._assigned_cuda_visible_device),
        )
        self._process.start()
        child_conn.close()
        self._worker_info = self._wait_for_ready(timeout=30.0)
        print(
            "[Reward Worker] Ready "
            f"slot={self._worker_slot} "
            f"pid={self._worker_info['pid']}, "
            f"assigned_gpu={self._worker_info['assigned_gpu']}, "
            f"assigned_cuda_visible_device={self._worker_info['assigned_cuda_visible_device']!r}, "
            f"worker_device={self._worker_info['worker_device']}, "
            f"cuda_visible_devices={self._worker_info['cuda_visible_devices']!r}, "
            f"cuda_available={self._worker_info['cuda_available']}, "
            f"cuda_device_count={self._worker_info['cuda_device_count']}, "
            f"rss_gib={self._worker_info['rss_gib']:.2f}, "
            f"torch_home={self._worker_info['torch_home']!r}"
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

        required_keys = {
            "pid",
            "assigned_gpu",
            "assigned_cuda_visible_device",
            "worker_device",
            "cuda_visible_devices",
            "cuda_available",
            "cuda_device_count",
            "rss_gib",
            "torch_home",
        }
        missing = sorted(required_keys.difference(response))
        if missing:
            self.close(force=True)
            raise PersistentEvalWorkerError(
                f"Persistent eval worker startup handshake missing fields: {', '.join(missing)}"
            )
        cuda_visible_devices = str(response["cuda_visible_devices"]).strip()
        cuda_device_count = int(response["cuda_device_count"])
        assigned_gpu = response["assigned_gpu"]
        assigned_cuda_visible_device = response["assigned_cuda_visible_device"]
        worker_device = str(response["worker_device"])
        if self._assigned_gpu is None:
            if worker_device != "cpu":
                self.close(force=True)
                raise PersistentEvalWorkerError(
                    f"CPU reward worker returned unexpected device {worker_device!r}"
                )
            if cuda_visible_devices not in {"", "-1"} or cuda_device_count != 0:
                self.close(force=True)
                raise PersistentEvalWorkerError(
                    "CPU reward worker unexpectedly reported CUDA visibility "
                    f"(CUDA_VISIBLE_DEVICES={response['cuda_visible_devices']!r}, count={response['cuda_device_count']})"
                )
        else:
            if assigned_gpu != self._assigned_gpu:
                self.close(force=True)
                raise PersistentEvalWorkerError(
                    f"Reward worker slot {self._worker_slot} bound unexpected GPU {assigned_gpu!r}, expected {self._assigned_gpu!r}"
                )
            if (
                self._assigned_cuda_visible_device is not None
                and cuda_visible_devices != self._assigned_cuda_visible_device
            ):
                self.close(force=True)
                raise PersistentEvalWorkerError(
                    "Reward worker slot "
                    f"{self._worker_slot} bound unexpected CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}, "
                    f"expected {self._assigned_cuda_visible_device!r}"
                )
            if worker_device != "cuda:0":
                self.close(force=True)
                raise PersistentEvalWorkerError(
                    f"GPU reward worker returned unexpected local device {worker_device!r}"
                )
            if cuda_device_count != 1 or not bool(response["cuda_available"]):
                self.close(force=True)
                raise PersistentEvalWorkerError(
                    "GPU reward worker must expose exactly one visible CUDA device, but reported "
                    f"(available={response['cuda_available']}, count={response['cuda_device_count']})"
                )
        return {
            "pid": int(response["pid"]),
            "assigned_gpu": assigned_gpu,
            "assigned_cuda_visible_device": (
                None if assigned_cuda_visible_device is None else str(assigned_cuda_visible_device)
            ),
            "worker_device": worker_device,
            "cuda_visible_devices": str(response["cuda_visible_devices"]),
            "cuda_available": bool(response["cuda_available"]),
            "cuda_device_count": int(response["cuda_device_count"]),
            "rss_gib": float(response["rss_gib"]),
            "torch_home": str(response["torch_home"]),
            "slot": self._worker_slot,
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


class _EvalWorkerPool:
    def __init__(self) -> None:
        self._plan = get_reward_worker_plan()
        self._sessions: list[_PersistentEvalWorkerSession] = []
        try:
            if self._plan["reward_gpu_indices"]:
                reward_gpu_tokens = list(self._plan.get("reward_gpu_tokens") or [])
                for slot, assigned_gpu in enumerate(self._plan["reward_gpu_indices"]):
                    assigned_cuda_visible_device = (
                        reward_gpu_tokens[slot]
                        if slot < len(reward_gpu_tokens)
                        else str(int(assigned_gpu))
                    )
                    self._sessions.append(
                        _PersistentEvalWorkerSession(
                            assigned_gpu=int(assigned_gpu),
                            assigned_cuda_visible_device=assigned_cuda_visible_device,
                            worker_slot=slot,
                        )
                    )
            else:
                self._sessions.append(
                    _PersistentEvalWorkerSession(
                        assigned_gpu=None,
                        assigned_cuda_visible_device=None,
                        worker_slot=0,
                    )
                )
        except Exception:
            for session in self._sessions:
                session.close(force=True)
            self._sessions = [
                _PersistentEvalWorkerSession(
                    assigned_gpu=None,
                    assigned_cuda_visible_device=None,
                    worker_slot=0,
                )
            ]
            self._plan = {
                **self._plan,
                "mode": "cpu_fallback",
                "reward_gpu_indices": [],
                "reward_gpu_tokens": [],
                "shared_train_gpu": False,
                "pool_size": 1,
                "workers_per_gpu": 1,
            }
        print(
            "[Reward Worker Pool] "
            f"mode={self._plan['mode']} "
            f"visible_gpu_count={self._plan['visible_gpu_count']} "
            f"deepspeed_enabled={self._plan.get('deepspeed_enabled', False)} "
            f"rank={self._plan['rank']} "
            f"local_rank={self._plan['local_rank']} "
            f"world_size={self._plan['world_size']} "
            f"train_gpu={self._plan['train_gpu']} "
            f"workers_per_gpu={self._plan.get('workers_per_gpu', 1)} "
            f"reward_gpu_indices={self._plan['reward_gpu_indices']} "
            f"reward_gpu_tokens={self._plan.get('reward_gpu_tokens', [])}"
        )

    def worker_count(self) -> int:
        return len(self._sessions)

    def primary_device(self) -> str:
        if not self._sessions:
            return "cpu"
        return str(self._sessions[0].diagnostics().get("worker_device", "cpu"))

    def diagnostics(self) -> Dict[str, Any]:
        workers = [session.diagnostics() for session in self._sessions]
        total_rss_gib = sum(float(worker.get("rss_gib") or 0.0) for worker in workers)
        primary = workers[0] if workers else {}
        return {
            **primary,
            "mode": self._plan["mode"],
            "visible_gpu_count": self._plan["visible_gpu_count"],
            "distributed": bool(self._plan["distributed"]),
            "rank": self._plan["rank"],
            "local_rank": self._plan["local_rank"],
            "world_size": self._plan["world_size"],
            "train_gpu": self._plan["train_gpu"],
            "reward_gpu_indices": list(self._plan["reward_gpu_indices"]),
            "reward_gpu_tokens": list(self._plan.get("reward_gpu_tokens") or []),
            "shared_train_gpu": bool(self._plan["shared_train_gpu"]),
            "pool_size": len(workers),
            "workers_per_gpu": int(self._plan.get("workers_per_gpu", 1)),
            "deepspeed_enabled": bool(self._plan.get("deepspeed_enabled", False)),
            "total_rss_gib": total_rss_gib,
            "workers": workers,
            "worker_pids": [worker["pid"] for worker in workers],
        }

    def _timeout_result_for_entry(
        self,
        entry: Dict[str, Any],
        *,
        error: str,
        estimated_total_seconds: float,
        assigned_gpu: Optional[int],
        worker_device: str,
        worker_slot: int,
    ) -> Dict[str, Any]:
        result = _timeout_eval_result(
            error=error,
            estimated_total_seconds=estimated_total_seconds,
            eval_limit_seconds=entry["effective_cfg"].eval_limit_seconds,
            seed_accuracy_baseline=entry.get("seed_accuracy_baseline"),
            backbone_model_names=entry.get("backbone_model_names"),
        )
        result["assigned_gpu"] = assigned_gpu
        result["worker_device"] = worker_device
        result["worker_slot"] = worker_slot
        return result

    def _replace_session(self, slot: int) -> _PersistentEvalWorkerSession:
        old_session = self._sessions[slot]
        old_info = old_session.diagnostics()
        assigned_gpu = old_info.get("assigned_gpu")
        assigned_cuda_visible_device = old_info.get("assigned_cuda_visible_device")
        old_session.close(force=True)
        try:
            new_session = _PersistentEvalWorkerSession(
                assigned_gpu=assigned_gpu if assigned_gpu is not None else None,
                assigned_cuda_visible_device=assigned_cuda_visible_device,
                worker_slot=slot,
            )
        except Exception:
            new_session = _PersistentEvalWorkerSession(
                assigned_gpu=None,
                assigned_cuda_visible_device=None,
                worker_slot=slot,
            )
        self._sessions[slot] = new_session
        return new_session

    def _request_entry(self, slot: int, entry: Dict[str, Any], *, timeout: float) -> Dict[str, Any]:
        session = self._sessions[slot]
        session_info = session.diagnostics()
        assigned_gpu = session_info.get("assigned_gpu")
        worker_device = str(session_info.get("worker_device", "cpu"))
        worker_slot = int(session_info.get("slot", slot))
        entry["payload"]["worker_slot"] = worker_slot
        try:
            result = session.request(entry["payload"], timeout=timeout)
        except PersistentEvalWorkerError as exc:
            self._replace_session(slot)
            return self._timeout_result_for_entry(
                entry,
                error=f"{type(exc).__name__}: {exc}",
                estimated_total_seconds=float(timeout),
                assigned_gpu=assigned_gpu,
                worker_device=worker_device,
                worker_slot=worker_slot,
            )
        result["assigned_gpu"] = assigned_gpu
        result["worker_device"] = worker_device
        result["worker_slot"] = worker_slot
        return result

    def request_entry(self, entry: Dict[str, Any], *, timeout: float) -> Dict[str, Any]:
        entry["payload"]["worker_batch_first_item"] = True
        entry["payload"]["worker_batch_last_item"] = True
        return self._request_entry(0, entry, timeout=timeout)

    def map_entries(self, entries: list[Dict[str, Any]], *, timeout: float) -> list[Dict[str, Any]]:
        if not entries:
            return []
        if self.worker_count() <= 1:
            return [self.request_entry(entry, timeout=timeout) for entry in entries]

        indexed_results: list[Optional[Dict[str, Any]]] = [None] * len(entries)
        assignments: list[list[tuple[int, Dict[str, Any]]]] = [[] for _ in self._sessions]
        for index, entry in enumerate(entries):
            slot = index % self.worker_count()
            assignments[slot].append((index, entry))

        def _process_worker_tasks(slot: int, tasks: list[tuple[int, Dict[str, Any]]]) -> list[tuple[int, Dict[str, Any]]]:
            results: list[tuple[int, Dict[str, Any]]] = []
            for task_index, (index, entry) in enumerate(tasks):
                entry["payload"]["worker_batch_first_item"] = task_index == 0
                entry["payload"]["worker_batch_last_item"] = task_index == (len(tasks) - 1)
                results.append((index, self._request_entry(slot, entry, timeout=timeout)))
            return results

        non_empty_assignments = [(slot, tasks) for slot, tasks in enumerate(assignments) if tasks]
        with ThreadPoolExecutor(max_workers=len(non_empty_assignments)) as executor:
            futures = [
                executor.submit(_process_worker_tasks, slot, tasks)
                for slot, tasks in non_empty_assignments
            ]
            for future in futures:
                for index, result in future.result():
                    indexed_results[index] = result

        return [
            result
            if result is not None
            else self._timeout_result_for_entry(
                entries[index],
                error="PersistentEvalWorkerError: missing result from worker pool dispatch",
                estimated_total_seconds=timeout,
                assigned_gpu=None,
                worker_device="cpu",
                worker_slot=-1,
            )
            for index, result in enumerate(indexed_results)
        ]

    def close(self) -> None:
        for session in self._sessions:
            try:
                session.close()
            except Exception:
                pass
        self._sessions.clear()


_EVAL_WORKER_POOL: Optional[_EvalWorkerPool] = None


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


def _format_mem_value(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _extract_backbone_model_names_from_code(code: str) -> list[str]:
    if not code:
        return []
    matches: Dict[str, str] = {}
    patterns = (
        r"self\.(backbone_[ab])\s*=\s*TorchVision\(\s*model\s*=\s*['\"]([^'\"]+)['\"]",
        r"self\.(backbone_[ab])\s*=\s*TorchVision\(\s*['\"]([^'\"]+)['\"]",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, code):
            matches.setdefault(match.group(1), match.group(2))
    return [
        matches[attr_name]
        for attr_name in ("backbone_a", "backbone_b")
        if attr_name in matches
    ]


def _base_eval_result(
    *,
    seed_accuracy_baseline: Optional[float] = None,
    eval_limit_seconds: Optional[int] = None,
    backbone_model_names: Optional[list[str]] = None,
) -> Dict[str, Any]:
    return {
        "val_metric": None,
        "test_acc": None,
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
        "eval_limit_seconds": eval_limit_seconds,
        "warmup_dense_reward": None,
        "backbone_model_names": list(backbone_model_names or []),
        "frozen_train_acc": None,
        "frozen_test_acc": None,
        "unfrozen_train_acc": None,
        "unfrozen_test_acc": None,
        "frozen_eval": None,
        "unfrozen_eval": None,
        "reward_target_metric": "frozen_test_acc",
        "reward_target_value": None,
    }


def _nested_eval_payload(
    result: Optional[Dict[str, Any]],
    *,
    eval_mode: str,
    backbone_frozen: bool,
) -> Optional[Dict[str, Any]]:
    if result is None:
        return None
    return {
        "eval_mode": eval_mode,
        "backbone_frozen": backbone_frozen,
        "reward": result.get("reward"),
        "components": dict(result.get("components") or {}),
        "built_ok": result.get("built_ok"),
        "forward_ok": result.get("forward_ok"),
        "forward_shape_ok": result.get("forward_shape_ok"),
        "trained_step_ok": result.get("trained_step_ok"),
        "backward_ok": result.get("backward_ok"),
        "loss_start": result.get("loss_start"),
        "loss_end": result.get("loss_end"),
        "loss_drop": result.get("loss_drop"),
        "loss_drop_ok": result.get("loss_drop_ok"),
        "train_acc": result.get("train_acc"),
        "test_acc": result.get("test_acc", result.get("val_metric")),
        "val_metric": result.get("val_metric"),
        "latency_ms": result.get("latency_ms"),
        "params_m": result.get("params_m"),
        "timed_out": result.get("timed_out", False),
        "estimated_total_seconds": result.get("estimated_total_seconds"),
        "eval_limit_seconds": result.get("eval_limit_seconds"),
        "backbone_model_names": list(result.get("backbone_model_names") or []),
        "seed_accuracy_baseline": result.get("seed_accuracy_baseline"),
        "seed_train_acc_gap": result.get("seed_train_acc_gap"),
        "seed_train_acc_improved": result.get("seed_train_acc_improved"),
        "error": result.get("error"),
    }


def _merge_dual_eval_results(
    *,
    frozen_result: Dict[str, Any],
    unfrozen_result: Optional[Dict[str, Any]],
    cfg: "EvalConfig",
) -> Dict[str, Any]:
    merged = dict(frozen_result)
    frozen_train_acc = frozen_result.get("train_acc")
    frozen_test_acc = frozen_result.get("test_acc", frozen_result.get("val_metric"))
    unfrozen_train_acc = None if unfrozen_result is None else unfrozen_result.get("train_acc")
    unfrozen_test_acc = None if unfrozen_result is None else unfrozen_result.get("test_acc", unfrozen_result.get("val_metric"))
    reward_target_metric = str(getattr(cfg, "reward_target_metric", "frozen_test_acc") or "frozen_test_acc")

    if reward_target_metric == "frozen_test_acc":
        reward_target_value = frozen_test_acc
    elif reward_target_metric == "frozen_train_acc":
        reward_target_value = frozen_train_acc
    elif reward_target_metric == "unfrozen_test_acc":
        reward_target_value = unfrozen_test_acc
    elif reward_target_metric == "unfrozen_train_acc":
        reward_target_value = unfrozen_train_acc
    else:
        reward_target_value = frozen_test_acc

    merged.update(
        {
            "test_acc": frozen_test_acc,
            "train_acc": frozen_train_acc,
            "val_metric": frozen_test_acc,
            "frozen_train_acc": frozen_train_acc,
            "frozen_test_acc": frozen_test_acc,
            "unfrozen_train_acc": unfrozen_train_acc,
            "unfrozen_test_acc": unfrozen_test_acc,
            "frozen_eval": _nested_eval_payload(
                frozen_result,
                eval_mode="frozen",
                backbone_frozen=True,
            ),
            "unfrozen_eval": _nested_eval_payload(
                unfrozen_result,
                eval_mode="unfrozen",
                backbone_frozen=False,
            ),
            "reward_target_metric": reward_target_metric,
            "reward_target_value": reward_target_value,
        }
    )
    return merged


def _request_timeout_seconds(cfg: "EvalConfig") -> float:
    eval_runs = 2 if bool(getattr(cfg, "run_unfrozen_backbone_eval", False)) else 1
    base_timeout = float(max(1, int(getattr(cfg, "eval_limit_seconds", 270))))
    return max(360.0, (base_timeout * eval_runs) + 120.0)

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


def _set_dual_backbones_trainable(model: nn.Module, *, freeze_backbones: bool) -> None:
    for backbone_name in ("backbone_a", "backbone_b"):
        backbone = getattr(model, backbone_name, None)
        if backbone is None:
            continue
        for param in backbone.parameters():
            param.requires_grad = not freeze_backbones
        if freeze_backbones:
            backbone.eval()
        else:
            backbone.train()


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
    *,
    epochs: int = 1,
    max_steps: Optional[int] = None,
    device: str = "cpu",
    n_classes: int = 10,
    time_budget: Optional[EvalTimeBudget] = None,
    freeze_backbones: bool = True,
) -> Dict[str, Any]:
    """
    Train for full epochs over the provided loader, optionally capped by max_steps.
    Returns training diagnostics.
    """
    model.train()
    _set_dual_backbones_trainable(model, freeze_backbones=freeze_backbones)
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
        effective_epochs = max(1, int(epochs))
        effective_max_steps = None if max_steps is None or int(max_steps) <= 0 else int(max_steps)
        stop_early = False
        for _epoch_index in range(effective_epochs):
            for x, y in train_loader:
                if time_budget is not None:
                    try:
                        time_budget.mark_train_batch_start()
                    except EvalTimeException as exc:
                        exc.partial.update(
                            {
                                "backward_ok": False,
                                "trained_step_ok": False,
                                "loss_start": loss_start,
                                "loss_end": loss_end,
                                "loss_drop": None if loss_start is None or loss_end is None else float(loss_start - loss_end),
                                "loss_drop_ok": False,
                                "steps_completed": steps_completed,
                            }
                        )
                        raise
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
                if time_budget is not None:
                    try:
                        time_budget.mark_train_batch_end()
                    except EvalTimeException as exc:
                        exc.partial.update(
                            {
                                "backward_ok": False,
                                "trained_step_ok": False,
                                "loss_start": loss_start,
                                "loss_end": loss_end,
                                "loss_drop": None if loss_start is None or loss_end is None else float(loss_start - loss_end),
                                "loss_drop_ok": False,
                                "steps_completed": steps_completed,
                            }
                        )
                        raise
                if effective_max_steps is not None and steps_completed >= effective_max_steps:
                    stop_early = True
                    break
            if stop_early:
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
    except EvalTimeException:
        raise
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
    max_batches: Optional[int] = 10,
    time_budget: Optional[EvalTimeBudget] = None,
    phase: str = "accuracy",
) -> float:
    """
    Evaluate accuracy in [0,1] on the provided loader without interpolation or fallback.
    """
    model.eval()
    correct = 0
    total = 0
    batches_seen = 0

    for x, y in data_loader:
        if time_budget is not None:
            try:
                time_budget.mark_accuracy_batch_start(phase)
            except EvalTimeException as exc:
                exc.partial.update(
                    {
                        "batches_seen": batches_seen,
                        "partial_accuracy": (correct / total) if total > 0 else None,
                    }
                )
                raise
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        if logits.dim() != 2:
            raise RuntimeError(f"logits must be (N,C), got {tuple(logits.shape)}")
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        batches_seen += 1
        if time_budget is not None:
            try:
                time_budget.mark_accuracy_batch_end(phase)
            except EvalTimeException as exc:
                exc.partial.update(
                    {
                        "batches_seen": batches_seen,
                        "partial_accuracy": (correct / total) if total > 0 else None,
                    }
                )
                raise
        if max_batches is not None and int(max_batches) > 0 and batches_seen >= int(max_batches):
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
    if (not bool(getattr(cfg, "full_test_acc", False))) and 0 < cfg.val_subset_size < len(val_dataset):
        val_dataset = Subset(val_dataset, range(cfg.val_subset_size))

    train_batch = max(1, min(cfg.default_batch_size, len(train_dataset)))
    val_batch = max(1, min(cfg.default_batch_size, len(val_dataset)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch,
        shuffle=True,
        num_workers=0,
        pin_memory=str(cfg.device).startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch,
        shuffle=False,
        num_workers=0,
        pin_memory=str(cfg.device).startswith("cuda"),
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
    train_epochs: int = 1
    train_steps: Optional[int] = None
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
    eval_limit_seconds: int = 270
    budget_probe_batches: int = 2
    run_unfrozen_backbone_eval: bool = False
    full_test_acc: bool = False
    reward_target_metric: str = "frozen_test_acc"
    formal_nn_eval: bool = False
    formal_task: str = "img-classification"
    formal_dataset: str = "cifar-10"
    formal_metric: str = "acc"


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)


def _normalize_optional_steps(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _build_effective_eval_cfg(
    *,
    cfg: Optional["EvalConfig"],
    prm: Optional[Dict[str, Any]],
    device: str,
    in_shape: Tuple[int, int, int, int],
    out_shape: Tuple[int, ...],
) -> "EvalConfig":
    prm_payload = dict(prm or {})
    base_cfg = cfg if cfg is not None else EvalConfig(
        device=device,
        input_shape=in_shape,
        n_classes=int(out_shape[0]),
        measure_latency=True,
        kl_div=None,
        critic_fn=None,
        weights=None,
    )
    return replace(
        base_cfg,
        device=device,
        input_shape=tuple(base_cfg.input_shape if cfg is not None else in_shape),
        n_classes=int(base_cfg.n_classes if cfg is not None else out_shape[0]),
        train_epochs=_coerce_positive_int(
            prm_payload.get("epoch"),
            getattr(base_cfg, "train_epochs", 1),
        ),
        train_steps=_normalize_optional_steps(getattr(base_cfg, "train_steps", None)),
        default_batch_size=_coerce_positive_int(
            prm_payload.get("batch"),
            getattr(base_cfg, "default_batch_size", 32),
        ),
    )


def _is_formal_nn_eval_enabled(cfg: Optional["EvalConfig"]) -> bool:
    return bool(cfg is not None and getattr(cfg, "formal_nn_eval", False))


def _ensure_nn_dataset_importable() -> None:
    global _NN_DATASET_IMPORT_READY
    if _NN_DATASET_IMPORT_READY:
        return
    importlib.invalidate_caches()
    importlib.import_module("ab.nn.api")
    _NN_DATASET_IMPORT_READY = True


def _formal_first_batch_loss(trainer: Any, prm: Dict[str, Any]) -> Optional[float]:
    try:
        trainer.model.train_setup(prm)
        trainer.model.eval()
        batch = next(iter(trainer.train_loader))
        inputs, labels = batch
        inputs = inputs.to(trainer.device)
        labels = labels.to(trainer.device)
        outputs = trainer.model(inputs)
        criterion = getattr(trainer.model, "criterion", None)
        if criterion is None:
            criterion = nn.CrossEntropyLoss().to(trainer.device)
        loss = criterion(outputs, labels)
        if not torch.isfinite(loss):
            return None
        return float(loss.detach().item())
    except Exception:
        return None


def _formal_eval_with_nn_dataset(
    code: str,
    *,
    prm: Dict[str, Any],
    cfg: "EvalConfig",
    freeze_backbones: bool,
    seed_accuracy_baseline: Optional[float],
    backbone_model_names: list[str],
) -> Dict[str, Any]:
    try:
        _ensure_nn_dataset_importable()
    except Exception as exc:
        raise RuntimeError(
            "Formal nn-dataset evaluation requested, but `ab.nn.api` is not importable "
            "from the current Python environment."
        ) from exc

    from ab.nn.util.Const import ab_root_path, out  # type: ignore
    from ab.nn.util.Loader import load_dataset  # type: ignore
    from ab.nn.util.Train import Train  # type: ignore
    from ab.nn.util.Util import create_file, release_memory, uuid4  # type: ignore

    model_name = f"rl-eval-{uuid4([code, os.getpid(), time.time_ns(), freeze_backbones])}"
    tmp_module = ".".join((out, "nn", "tmp"))
    tmp_module_name = f"{tmp_module}.{model_name}"
    tmp_dir = ab_root_path / tmp_module.replace(".", "/")
    create_file(tmp_dir, "__init__.py")
    temp_file_path = tmp_dir / f"{model_name}.py"

    trainer = None
    train_set = None
    test_set = None
    try:
        temp_file_path.write_text(code, encoding="utf-8")
        safe_prm = dict(prm)
        safe_prm["freeze_backbones"] = bool(freeze_backbones)
        out_shape, _minimum_accuracy, train_set, test_set = load_dataset(
            str(getattr(cfg, "formal_task", "img-classification")),
            str(getattr(cfg, "formal_dataset", "cifar-10")),
            str(safe_prm["transform"]),
        )
        num_workers = int(safe_prm.get("num_workers", 1))
        trainer = Train(
            config=(
                str(getattr(cfg, "formal_task", "img-classification")),
                str(getattr(cfg, "formal_dataset", "cifar-10")),
                str(getattr(cfg, "formal_metric", "acc")),
                model_name,
            ),
            out_shape=out_shape,
            minimum_accuracy=-1.0,
            batch=int(safe_prm["batch"]),
            nn_module=tmp_module_name,
            task=str(getattr(cfg, "formal_task", "img-classification")),
            train_dataset=train_set,
            test_dataset=test_set,
            metric=str(getattr(cfg, "formal_metric", "acc")),
            num_workers=num_workers,
            prm=safe_prm,
            save_to_db=False,
            is_code=True,
        )
        params_m = _count_params_m(trainer.model)
        forward_result = _quick_forward(
            trainer.model,
            tuple(trainer.in_shape),
            device=str(trainer.device),
            n_classes=int(out_shape[0]),
        )
        initial_loss = _formal_first_batch_loss(trainer, safe_prm)
        accuracy, _accuracy_to_time, duration_ns = trainer.train_n_eval(
            int(safe_prm.get("epoch", 1) or 1),
            max(1.0 / 60.0, float(cfg.eval_limit_seconds) / 60.0),
            False,
            False,
            train_set,
            save_path=None,
        )
        epoch_metrics = trainer.epoch_history[-1] if trainer.epoch_history else None
        train_acc = float(epoch_metrics.train_accuracy) if epoch_metrics is not None else None
        test_acc = float(epoch_metrics.test_accuracy) if epoch_metrics is not None else float(accuracy)
        loss_end = float(epoch_metrics.train_loss) if epoch_metrics is not None else None
        loss_drop = None if initial_loss is None or loss_end is None else float(initial_loss - loss_end)
        rel_drop_ok = bool(
            initial_loss is not None
            and loss_end is not None
            and initial_loss > 0.0
            and loss_end <= initial_loss * 0.98
        )
        loss_drop_ok = bool(
            initial_loss is not None
            and loss_end is not None
            and (loss_end < (initial_loss - 1e-3) or rel_drop_ok)
        )
        components = compute_cv_reward_simple(
            built_ok=True,
            forward_shape_ok=bool(forward_result.get("forward_shape_ok")),
            backward_ok=True,
            loss_drop_ok=loss_drop_ok,
            val_metric=test_acc,
            val_metric_baseline=None,
            latency_ms=forward_result.get("latency_ms"),
            params_m=params_m,
            flops_g=None,
            critic_score=None,
            kl_div=cfg.kl_div,
            weights=cfg.weights,
        )
        return {
            "reward": components["reward"],
            "components": components,
            **{
                **_base_eval_result(
                    seed_accuracy_baseline=seed_accuracy_baseline,
                    eval_limit_seconds=cfg.eval_limit_seconds,
                    backbone_model_names=backbone_model_names,
                ),
                "test_acc": test_acc,
                "val_metric": test_acc,
                "built_ok": True,
                "forward_ok": bool(forward_result.get("forward_ok")),
                "forward_shape_ok": bool(forward_result.get("forward_shape_ok")),
                "trained_step_ok": True,
                "backward_ok": True,
                "loss_start": initial_loss,
                "loss_end": loss_end,
                "loss_drop": loss_drop,
                "loss_drop_ok": loss_drop_ok,
                "train_acc": train_acc,
                "seed_train_acc_gap": None if train_acc is None or seed_accuracy_baseline is None else float(train_acc - seed_accuracy_baseline),
                "seed_train_acc_improved": bool(
                    train_acc is not None
                    and seed_accuracy_baseline is not None
                    and float(train_acc - seed_accuracy_baseline) > 0.0
                ),
                "train_acc_gain": None if train_acc is None or seed_accuracy_baseline is None else float(train_acc - seed_accuracy_baseline),
                "train_acc_improved": bool(
                    train_acc is not None
                    and seed_accuracy_baseline is not None
                    and float(train_acc - seed_accuracy_baseline) > 0.0
                ),
                "latency_ms": forward_result.get("latency_ms"),
                "params_m": params_m,
                "estimated_total_seconds": float(duration_ns) / 1e9,
            },
        }
    finally:
        try:
            if temp_file_path.exists():
                temp_file_path.unlink()
        except Exception:
            pass
        sys.modules.pop(tmp_module_name, None)
        gc.collect()
        if trainer is not None:
            try:
                del trainer.model
            except Exception:
                pass
        try:
            del train_set
        except Exception:
            pass
        try:
            del test_set
        except Exception:
            pass
        try:
            release_memory()
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _empty_eval_result(
    *,
    reward: float = 0.0,
    error: Optional[str] = None,
    seed_accuracy_baseline: Optional[float] = None,
    eval_limit_seconds: Optional[int] = None,
    backbone_model_names: Optional[list[str]] = None,
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
        **_base_eval_result(
            seed_accuracy_baseline=seed_accuracy_baseline,
            eval_limit_seconds=eval_limit_seconds,
            backbone_model_names=backbone_model_names,
        ),
    }
    if error:
        result["error"] = error
    return result


def _timeout_eval_result(
    *,
    error: str,
    estimated_total_seconds: Optional[float],
    eval_limit_seconds: Optional[int],
    seed_accuracy_baseline: Optional[float] = None,
    backbone_model_names: Optional[list[str]] = None,
    built_ok: bool = False,
    forward_ok: bool = False,
    forward_shape_ok: bool = False,
    trained_step_ok: bool = False,
    backward_ok: bool = False,
    loss_start: Optional[float] = None,
    loss_end: Optional[float] = None,
    loss_drop: Optional[float] = None,
    loss_drop_ok: bool = False,
    latency_ms: Optional[float] = None,
    params_m: Optional[float] = None,
    kl_div: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    components = compute_cv_reward_simple(
        built_ok=built_ok,
        forward_shape_ok=forward_shape_ok,
        backward_ok=backward_ok,
        loss_drop_ok=loss_drop_ok,
        val_metric=None,
        val_metric_baseline=None,
        latency_ms=latency_ms,
        params_m=params_m,
        flops_g=None,
        critic_score=None,
        kl_div=kl_div,
        weights=weights,
    )
    return {
        "reward": components["reward"],
        "components": components,
        **{
            **_base_eval_result(
                seed_accuracy_baseline=seed_accuracy_baseline,
                eval_limit_seconds=eval_limit_seconds,
                backbone_model_names=backbone_model_names,
            ),
            "built_ok": built_ok,
            "forward_ok": forward_ok,
            "forward_shape_ok": forward_shape_ok,
            "trained_step_ok": trained_step_ok,
            "backward_ok": backward_ok,
            "loss_start": loss_start,
            "loss_end": loss_end,
            "loss_drop": loss_drop,
            "loss_drop_ok": loss_drop_ok,
            "latency_ms": latency_ms,
            "params_m": params_m,
            "timed_out": True,
            "estimated_total_seconds": estimated_total_seconds,
            "error": error,
        },
    }


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
    backbone_model_names: Optional[list[str]] = None,
    freeze_backbones: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end:
      1) Build model (if builder provided)
      2) Forward sanity
      3) Train full epoch(s)
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
    time_budget = EvalTimeBudget(cfg)

    # 1) Build or use provided model
    mdl = model
    local_train_loader = None
    local_val_loader = None

    try:
        effective_train_epochs = _coerce_positive_int(getattr(cfg, "train_epochs", 1), 1)
        effective_train_step_cap = _normalize_optional_steps(getattr(cfg, "train_steps", None))
        if mdl is None and build_fn is not None:
            try:
                mdl = build_fn()
            except Exception:
                mdl = None
        if mdl is not None and isinstance(mdl, nn.Module):
            try:
                mdl.to(device)
                params_m = _count_params_m(mdl)
                built_ok = True
                time_budget.mark_build_complete()
            except Exception:
                built_ok = False
                mdl = None
        else:
            time_budget.check("build")

        # If build failed, compute minimal reward and return
        if not built_ok or mdl is None:
            time_budget.check("build")
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
                **_base_eval_result(
                    seed_accuracy_baseline=seed_accuracy_baseline,
                    eval_limit_seconds=cfg.eval_limit_seconds,
                    backbone_model_names=backbone_model_names,
                ),
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
        if forward_ok:
            time_budget.mark_forward_complete()
        else:
            time_budget.check("forward")

        # 3) Train full epoch(s) on the reward subset.
        if train_loader is None or val_loader is None:
            local_train_loader, local_val_loader = _build_cifar10_loaders(cfg)
        used_train_loader = train_loader if train_loader is not None else local_train_loader
        used_val_loader = val_loader if val_loader is not None else local_val_loader
        expected_train_batches = max(1, len(used_train_loader) * effective_train_epochs)
        if effective_train_step_cap is not None:
            expected_train_batches = min(expected_train_batches, effective_train_step_cap)
        effective_val_batches = (
            len(used_val_loader)
            if bool(getattr(cfg, "full_test_acc", False))
            else max(1, min(cfg.max_val_batches, len(used_val_loader)))
        )
        time_budget.set_expected_work(
            train_batches=expected_train_batches,
            train_accuracy_batches=max(1, len(used_train_loader)),
            val_accuracy_batches=effective_val_batches,
        )
        try:
            train_result = _train_steps(
                mdl,
                used_train_loader,
                epochs=effective_train_epochs,
                max_steps=effective_train_step_cap,
                device=device,
                n_classes=cfg.n_classes,
                time_budget=time_budget,
                freeze_backbones=freeze_backbones,
            )
            backward_ok = bool(train_result["backward_ok"])
            trained_step_ok = bool(train_result["trained_step_ok"])
            loss_start = train_result["loss_start"]
            loss_end = train_result["loss_end"]
            loss_drop = train_result["loss_drop"]
            loss_drop_ok = bool(train_result["loss_drop_ok"])

            # 4) Quick validation (accuracy)
            train_metric_batches = max(1, len(used_train_loader))
            train_acc = _quick_accuracy(
                mdl,
                used_train_loader,
                device=device,
                max_batches=train_metric_batches,
                time_budget=time_budget,
                phase="train_accuracy",
            )
            val_metric = _quick_accuracy(
                mdl,
                used_val_loader,
                device=device,
                max_batches=None if bool(getattr(cfg, "full_test_acc", False)) else cfg.max_val_batches,
                time_budget=time_budget,
                phase="val_accuracy",
            )
            if seed_accuracy_baseline is not None:
                seed_train_acc_gap = float(train_acc - seed_accuracy_baseline)
                seed_train_acc_improved = bool(seed_train_acc_gap > 0.0)
        except EvalTimeException as exc:
            partial = exc.partial or {}
            loss_start = partial.get("loss_start", loss_start)
            loss_end = partial.get("loss_end", loss_end)
            loss_drop = partial.get("loss_drop", loss_drop)
            backward_ok = bool(partial.get("backward_ok", False))
            trained_step_ok = bool(partial.get("trained_step_ok", False))
            return _timeout_eval_result(
                error=f"{type(exc).__name__}: {exc}",
                estimated_total_seconds=exc.estimated_total_seconds,
                eval_limit_seconds=cfg.eval_limit_seconds,
                seed_accuracy_baseline=seed_accuracy_baseline,
                backbone_model_names=backbone_model_names,
                built_ok=built_ok,
                forward_ok=forward_ok,
                forward_shape_ok=forward_shape_ok,
                trained_step_ok=trained_step_ok,
                backward_ok=backward_ok,
                loss_start=loss_start,
                loss_end=loss_end,
                loss_drop=loss_drop,
                loss_drop_ok=False,
                latency_ms=latency_ms,
                params_m=params_m,
                kl_div=cfg.kl_div,
                weights=cfg.weights,
            )

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
            **{
                **_base_eval_result(
                    seed_accuracy_baseline=seed_accuracy_baseline,
                    eval_limit_seconds=cfg.eval_limit_seconds,
                    backbone_model_names=backbone_model_names,
                ),
                "test_acc": val_metric,
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
                "seed_train_acc_gap": seed_train_acc_gap,
                "seed_train_acc_improved": seed_train_acc_improved,
                "train_acc_gain": seed_train_acc_gap,
                "train_acc_improved": seed_train_acc_improved,
                "latency_ms": latency_ms,
                "params_m": params_m,
                "estimated_total_seconds": time_budget.estimated_total_seconds,
            },
        }
    except EvalTimeException as exc:
        partial = exc.partial or {}
        loss_start = partial.get("loss_start", loss_start)
        loss_end = partial.get("loss_end", loss_end)
        loss_drop = partial.get("loss_drop", loss_drop)
        backward_ok = bool(partial.get("backward_ok", backward_ok))
        trained_step_ok = bool(partial.get("trained_step_ok", trained_step_ok))
        return _timeout_eval_result(
            error=f"{type(exc).__name__}: {exc}",
            estimated_total_seconds=exc.estimated_total_seconds,
            eval_limit_seconds=cfg.eval_limit_seconds,
            seed_accuracy_baseline=seed_accuracy_baseline,
            backbone_model_names=backbone_model_names,
            built_ok=built_ok,
            forward_ok=forward_ok,
            forward_shape_ok=forward_shape_ok,
            trained_step_ok=trained_step_ok,
            backward_ok=backward_ok,
            loss_start=loss_start,
            loss_end=loss_end,
            loss_drop=loss_drop,
            loss_drop_ok=False,
            latency_ms=latency_ms,
            params_m=params_m,
            kl_div=cfg.kl_div,
            weights=cfg.weights,
        )

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
        bool(getattr(cfg, "full_test_acc", False)),
    )


def _get_or_create_eval_worker_pool() -> _EvalWorkerPool:
    global _EVAL_WORKER_POOL
    if _EVAL_WORKER_POOL is not None:
        diagnostics = _EVAL_WORKER_POOL.diagnostics()
        workers = diagnostics.get("workers", [])
        if not workers or not all(bool(worker.get("alive", False)) for worker in workers):
            _EVAL_WORKER_POOL.close()
            _EVAL_WORKER_POOL = None
    if _EVAL_WORKER_POOL is None:
        _EVAL_WORKER_POOL = _EvalWorkerPool()
    return _EVAL_WORKER_POOL


def shutdown_eval_worker() -> None:
    global _EVAL_WORKER_POOL
    if _EVAL_WORKER_POOL is None:
        return
    info = _EVAL_WORKER_POOL.diagnostics()
    print(
        "[Reward Worker Pool] Shutdown "
        f"mode={info['mode']} "
        f"pool_size={info['pool_size']} "
        f"workers_per_gpu={info.get('workers_per_gpu', 1)} "
        f"worker_pids={info['worker_pids']} "
        f"total_rss_gib={info['total_rss_gib']:.2f}"
    )
    _EVAL_WORKER_POOL.close()
    _EVAL_WORKER_POOL = None


def get_eval_worker_diagnostics() -> Optional[Dict[str, Any]]:
    if _EVAL_WORKER_POOL is None:
        return None
    return _EVAL_WORKER_POOL.diagnostics()


def _worker_cuda_memory_gib() -> Tuple[Optional[float], Optional[float]]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    try:
        allocated = torch.cuda.memory_allocated() / float(1024 ** 3)
        reserved = torch.cuda.memory_reserved() / float(1024 ** 3)
        return allocated, reserved
    except RuntimeError:
        return None, None


def _persistent_eval_worker_entry(
    conn,
    assigned_gpu: Optional[int],
    assigned_cuda_visible_device: Optional[str],
) -> None:
    worker_device = "cuda:0" if assigned_gpu is not None else "cpu"
    try:
        conn.send(
            {
                "cmd": "worker_ready",
                "pid": os.getpid(),
                "assigned_gpu": assigned_gpu,
                "assigned_cuda_visible_device": assigned_cuda_visible_device,
                "worker_device": worker_device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
                "rss_gib": float(_read_process_rss_gib() or 0.0),
                "torch_home": os.environ.get("TORCH_HOME", ""),
            }
        )
        _persistent_eval_worker_loop(conn, worker_device=worker_device, assigned_gpu=assigned_gpu)
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
    reward_batch_index: Optional[int] = None,
    completion_index: Optional[int] = None,
    batch_last_item: bool = False,
) -> Dict[str, Any]:
    entry = _prepare_eval_request_entry(
        code=code,
        in_shape=in_shape,
        out_shape=out_shape,
        prm=prm,
        device=device,
        val_metric_baseline=val_metric_baseline,
        seed_accuracy_baseline=seed_accuracy_baseline,
        cfg=cfg,
        reward_batch_index=reward_batch_index,
        completion_index=completion_index,
        batch_last_item=batch_last_item,
    )
    worker_pool = _get_or_create_eval_worker_pool()
    return worker_pool.request_entry(entry, timeout=float(entry["request_timeout"]))


def _prepare_eval_request_entry(
    *,
    code: str,
    in_shape: Tuple[int, int, int, int],
    out_shape: Tuple[int, ...],
    prm: Optional[Dict[str, Any]],
    device: str,
    val_metric_baseline: Optional[float],
    seed_accuracy_baseline: Optional[float],
    cfg: Optional[EvalConfig],
    reward_batch_index: Optional[int],
    completion_index: Optional[int],
    batch_last_item: bool,
) -> Dict[str, Any]:
    worker_pool = _get_or_create_eval_worker_pool()
    worker_device = worker_pool.primary_device()
    effective_cfg = _build_effective_eval_cfg(
        cfg=cfg,
        prm=prm,
        device=worker_device,
        in_shape=in_shape,
        out_shape=out_shape,
    )
    return {
        "payload": {
            "cmd": "evaluate",
            "code": code,
            "in_shape": tuple(in_shape),
            "out_shape": tuple(out_shape),
            "prm": prm,
            "device": worker_device,
            "val_metric_baseline": val_metric_baseline,
            "seed_accuracy_baseline": seed_accuracy_baseline,
            "cfg": _serialize_eval_cfg(effective_cfg),
            "reward_batch_index": reward_batch_index,
            "completion_index": completion_index,
            "batch_last_item": batch_last_item,
        },
        "effective_cfg": effective_cfg,
        "request_timeout": _request_timeout_seconds(effective_cfg),
        "seed_accuracy_baseline": seed_accuracy_baseline,
        "backbone_model_names": _extract_backbone_model_names_from_code(code),
    }


def evaluate_code_and_reward_batch(
    request_specs: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    if not request_specs:
        return []
    entries = [
        _prepare_eval_request_entry(
            code=str(spec["code"]),
            in_shape=tuple(spec.get("in_shape", (8, 3, 32, 32))),
            out_shape=tuple(spec.get("out_shape", (10,))),
            prm=spec.get("prm"),
            device=str(spec.get("device", "cpu")),
            val_metric_baseline=spec.get("val_metric_baseline"),
            seed_accuracy_baseline=spec.get("seed_accuracy_baseline"),
            cfg=spec.get("cfg"),
            reward_batch_index=spec.get("reward_batch_index"),
            completion_index=spec.get("completion_index"),
            batch_last_item=bool(spec.get("batch_last_item", False)),
        )
        for spec in request_specs
    ]
    worker_pool = _get_or_create_eval_worker_pool()
    timeout = max(float(entry["request_timeout"]) for entry in entries)
    return worker_pool.map_entries(entries, timeout=timeout)


def _persistent_eval_worker_loop(conn, *, worker_device: str, assigned_gpu: Optional[int]) -> None:
    loader_cache: Dict[Tuple[Any, ...], Tuple[DataLoader, DataLoader]] = {}
    last_reward_batch_index = None
    last_completion_index = None
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
            train_loader = None
            val_loader = None
            if not _is_formal_nn_eval_enabled(cfg):
                loader_key = _loader_cache_key(cfg)
                if loader_key not in loader_cache:
                    loader_cache[loader_key] = _build_cifar10_loaders(cfg)
                train_loader, val_loader = loader_cache[loader_key]
            reward_batch_index = request.get("reward_batch_index")
            completion_index = request.get("completion_index")
            last_reward_batch_index = reward_batch_index
            last_completion_index = completion_index
            if bool(request.get("worker_batch_first_item")):
                cuda_allocated_gib, cuda_reserved_gib = _worker_cuda_memory_gib()
                print(
                    "[Reward Worker Memory] "
                    f"stage=batch_start "
                    f"pid={os.getpid()} "
                    f"worker_slot={request.get('worker_slot')} "
                    f"assigned_gpu={assigned_gpu} "
                    f"worker_device={worker_device} "
                    f"reward_batch_index={reward_batch_index} "
                    f"completion_index={completion_index} "
                    f"rss_gib={_format_mem_value(_read_process_rss_gib())} "
                    f"cuda_allocated_gib={_format_mem_value(cuda_allocated_gib)} "
                    f"cuda_reserved_gib={_format_mem_value(cuda_reserved_gib)} "
                    f"torch_home={os.environ.get('TORCH_HOME', '')!r}"
                )

            try:
                if worker_device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
                    eval_limit_seconds=cfg.eval_limit_seconds,
                    backbone_model_names=_extract_backbone_model_names_from_code(request.get("code", "")),
                )
                result["components"]["reward"] = -1.0
            result["assigned_gpu"] = assigned_gpu
            result["worker_device"] = worker_device
            result["worker_slot"] = request.get("worker_slot", None)
            if bool(request.get("worker_batch_last_item")):
                if worker_device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cuda_allocated_gib, cuda_reserved_gib = _worker_cuda_memory_gib()
                print(
                    "[Reward Worker Memory] "
                    f"stage=batch_end "
                    f"pid={os.getpid()} "
                    f"worker_slot={request.get('worker_slot')} "
                    f"assigned_gpu={assigned_gpu} "
                    f"worker_device={worker_device} "
                    f"reward_batch_index={reward_batch_index} "
                    f"completion_index={completion_index} "
                    f"rss_gib={_format_mem_value(_read_process_rss_gib())} "
                    f"cuda_allocated_gib={_format_mem_value(cuda_allocated_gib)} "
                    f"cuda_reserved_gib={_format_mem_value(cuda_reserved_gib)} "
                    f"torch_home={os.environ.get('TORCH_HOME', '')!r}"
                )

            conn.send(result)
    finally:
        cuda_allocated_gib, cuda_reserved_gib = _worker_cuda_memory_gib()
        print(
            "[Reward Worker Memory] "
            f"stage=shutdown "
            f"pid={os.getpid()} "
            f"worker_slot={request.get('worker_slot') if 'request' in locals() and isinstance(request, dict) else None} "
            f"assigned_gpu={assigned_gpu} "
            f"worker_device={worker_device} "
            f"reward_batch_index={last_reward_batch_index} "
            f"completion_index={last_completion_index} "
            f"rss_gib={_format_mem_value(_read_process_rss_gib())} "
            f"cuda_allocated_gib={_format_mem_value(cuda_allocated_gib)} "
            f"cuda_reserved_gib={_format_mem_value(cuda_reserved_gib)} "
            f"torch_home={os.environ.get('TORCH_HOME', '')!r}"
        )
        loader_cache.clear()
        gc.collect()
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
    backbone_model_names = _extract_backbone_model_names_from_code(code)
    cfg = _build_effective_eval_cfg(
        cfg=cfg,
        prm=prm,
        device=device,
        in_shape=in_shape,
        out_shape=out_shape,
    )

    if _is_formal_nn_eval_enabled(cfg):
        try:
            frozen_result = _formal_eval_with_nn_dataset(
                code,
                prm=prm,
                cfg=cfg,
                freeze_backbones=True,
                seed_accuracy_baseline=seed_accuracy_baseline,
                backbone_model_names=backbone_model_names,
            )
            unfrozen_result = None
            if bool(getattr(cfg, "run_unfrozen_backbone_eval", False)):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                unfrozen_result = _formal_eval_with_nn_dataset(
                    code,
                    prm=prm,
                    cfg=cfg,
                    freeze_backbones=False,
                    seed_accuracy_baseline=seed_accuracy_baseline,
                    backbone_model_names=backbone_model_names,
                )
            return _merge_dual_eval_results(
                frozen_result=frozen_result,
                unfrozen_result=unfrozen_result,
                cfg=cfg,
            )
        except Exception as e:
            error_type = type(e).__name__
            error_msg = f"{error_type}: {e}"
            failure_result = _empty_eval_result(
                error=error_msg,
                seed_accuracy_baseline=seed_accuracy_baseline,
                eval_limit_seconds=cfg.eval_limit_seconds,
                backbone_model_names=backbone_model_names,
            )
            failure_result["frozen_eval"] = _nested_eval_payload(
                failure_result,
                eval_mode="frozen",
                backbone_frozen=True,
            )
            if bool(getattr(cfg, "run_unfrozen_backbone_eval", False)):
                failure_result["unfrozen_eval"] = _nested_eval_payload(
                    failure_result,
                    eval_mode="unfrozen",
                    backbone_frozen=False,
                )
            return failure_result

    try:
        builder = build_fn_from_code(code, in_shape, out_shape, prm, device)
    except Exception as e:
        # Pass through error type so reward_fn can assign layered partial rewards
        error_type = type(e).__name__
        error_msg = f"{error_type}: {e}"
        failure_result = _empty_eval_result(
            error=error_msg,
            seed_accuracy_baseline=seed_accuracy_baseline,
            eval_limit_seconds=cfg.eval_limit_seconds,
            backbone_model_names=backbone_model_names,
        )
        failure_result["frozen_eval"] = _nested_eval_payload(
            failure_result,
            eval_mode="frozen",
            backbone_frozen=True,
        )
        if bool(getattr(cfg, "run_unfrozen_backbone_eval", False)):
            failure_result["unfrozen_eval"] = _nested_eval_payload(
                failure_result,
                eval_mode="unfrozen",
                backbone_frozen=False,
            )
        return failure_result
    try:
        frozen_result = evaluate_and_reward(
            build_fn=builder,
            train_loader=train_loader,
            val_loader=val_loader,
            val_metric_baseline=val_metric_baseline,
            seed_accuracy_baseline=seed_accuracy_baseline,
            cfg=cfg,
            backbone_model_names=backbone_model_names,
            freeze_backbones=True,
        )
        unfrozen_result = None
        if bool(getattr(cfg, "run_unfrozen_backbone_eval", False)):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            unfrozen_result = evaluate_and_reward(
                build_fn=builder,
                train_loader=train_loader,
                val_loader=val_loader,
                val_metric_baseline=val_metric_baseline,
                seed_accuracy_baseline=seed_accuracy_baseline,
                cfg=cfg,
                backbone_model_names=backbone_model_names,
                freeze_backbones=False,
            )
        return _merge_dual_eval_results(
            frozen_result=frozen_result,
            unfrozen_result=unfrozen_result,
            cfg=cfg,
        )
    except Exception as e:
        error_type = type(e).__name__
        error_msg = f"{error_type}: {e}"
        failure_result = _empty_eval_result(
            error=error_msg,
            seed_accuracy_baseline=seed_accuracy_baseline,
            eval_limit_seconds=cfg.eval_limit_seconds,
            backbone_model_names=backbone_model_names,
        )
        failure_result["frozen_eval"] = _nested_eval_payload(
            failure_result,
            eval_mode="frozen",
            backbone_frozen=True,
        )
        if bool(getattr(cfg, "run_unfrozen_backbone_eval", False)):
            failure_result["unfrozen_eval"] = _nested_eval_payload(
                failure_result,
                eval_mode="unfrozen",
                backbone_frozen=False,
            )
        return failure_result


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
