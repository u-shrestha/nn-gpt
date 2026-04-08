import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, os.path.dirname(__file__))

if "pandas" not in sys.modules:
    import types

    pandas_stub = types.ModuleType("pandas")
    pandas_stub.read_pickle = lambda *_args, **_kwargs: {}
    sys.modules["pandas"] = pandas_stub

if "ab.nn.util.Const" not in sys.modules:
    import types

    ab_nn_mod = types.ModuleType("ab.nn")
    ab_nn_util_mod = types.ModuleType("ab.nn.util")
    ab_nn_const_mod = types.ModuleType("ab.nn.util.Const")
    ab_nn_const_mod.base_module = "ab"
    ab_nn_const_mod.ab_root_path = Path(os.path.dirname(__file__)).resolve()
    ab_nn_const_mod.out_dir = Path(tempfile.gettempdir()) / "nngpt-test-out"
    ab_nn_const_mod.default_epoch_limit_minutes = 1
    ab_nn_util_runtime_mod = types.ModuleType("ab.nn.util.Util")
    ab_nn_util_runtime_mod.release_memory = lambda: None
    ab_nn_util_runtime_mod.uuid4 = lambda _content: "stub-uuid"
    sys.modules["ab.nn"] = ab_nn_mod
    sys.modules["ab.nn.util"] = ab_nn_util_mod
    sys.modules["ab.nn.util.Const"] = ab_nn_const_mod
    sys.modules["ab.nn.util.Util"] = ab_nn_util_runtime_mod

if "torch" not in sys.modules:
    import types

    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
        mem_get_info=lambda *_args, **_kwargs: (0, 0),
        get_device_name=lambda *_args, **_kwargs: "",
        set_device=lambda *_args, **_kwargs: None,
        empty_cache=lambda: None,
        ipc_collect=lambda: None,
    )
    sys.modules["torch"] = torch_mod

from ab.gpt import NNEval
from ab.gpt.util import nneval_worker_pool as worker_pool
from ab.gpt.util.CycleResults import collect_cycle_metrics


class NNEvalRuntimeTest(unittest.TestCase):
    def test_extract_accuracy_from_saved_eval_payload(self):
        self.assertEqual(
            NNEval._extract_accuracy_from_eval_payload({"eval_results": {"accuracy": 0.42}}),
            0.42,
        )
        self.assertEqual(
            NNEval._extract_accuracy_from_eval_payload({"eval_results": {"epochs": [{"acc": 0.33}]}}),
            0.33,
        )
        self.assertIsNone(NNEval._extract_accuracy_from_eval_payload({"eval_results": {}}))

    def test_load_existing_success_result_reads_eval_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "eval_info.json").write_text(
                json.dumps({"eval_results": {"accuracy": 0.55}}),
                encoding="utf-8",
            )
            result = NNEval._load_existing_success_result(model_dir)
        self.assertEqual(result["model_id"], model_dir.name)
        self.assertTrue(result["success"])
        self.assertTrue(result["skipped"])
        self.assertEqual(result["accuracy"], 0.55)

    def test_build_nneval_worker_plan_uses_all_visible_gpus(self):
        env = {
            "CUDA_VISIBLE_DEVICES": "2,5",
            "NNGPT_NNEVAL_WORKERS_PER_GPU": "2",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.object(worker_pool.torch.cuda, "is_available", return_value=True):
                with mock.patch.object(worker_pool.torch.cuda, "device_count", return_value=2):
                    plan = worker_pool.build_nneval_worker_plan(use_all_visible_gpus=True)
        self.assertEqual(plan["mode"], "fixed_worker_pool")
        self.assertEqual(plan["target_gpu_tokens"], ["2", "5"])
        self.assertEqual(plan["per_gpu_worker_counts"], [2, 2])
        self.assertEqual(plan["eval_gpu_tokens"], ["2", "5", "2", "5"])

    def test_build_nneval_worker_plan_uses_aux_tokens_when_requested(self):
        env = {
            "CUDA_VISIBLE_DEVICES": "0,1,3",
            "NNGPT_TRAIN_GPU_TOKENS": "0",
            "NNGPT_AUX_GPU_TOKENS": "1,3",
            "NNGPT_NNEVAL_WORKERS_PER_GPU": "1",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.object(worker_pool.torch.cuda, "is_available", return_value=True):
                with mock.patch.object(worker_pool.torch.cuda, "device_count", return_value=3):
                    plan = worker_pool.build_nneval_worker_plan(use_all_visible_gpus=False)
        self.assertEqual(plan["mode"], "fixed_worker_pool")
        self.assertEqual(plan["target_gpu_tokens"], ["1", "3"])
        self.assertEqual(plan["per_gpu_worker_counts"], [0, 1, 1])
        self.assertEqual(plan["eval_gpu_tokens"], ["1", "3"])

    def test_collect_cycle_metrics_reads_new_eval_info_accuracy_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_base_dir = Path(tmpdir)
            model_dir = models_base_dir / "B000"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "eval_info.json").write_text(
                json.dumps({"eval_results": {"accuracy": 0.61}}),
                encoding="utf-8",
            )
            eval_results_list, model_dirs_list, successful_models, failed_models = collect_cycle_metrics(
                models_base_dir,
                models_base_dir,
            )
        self.assertEqual(len(eval_results_list), 1)
        self.assertEqual(eval_results_list[0]["accuracy"], 0.61)
        self.assertEqual(model_dirs_list, [model_dir])
        self.assertEqual(successful_models, [model_dir])
        self.assertEqual(failed_models, [])

    def test_evaluate_model_entries_adds_worker_command(self):
        captured = {}

        class DummyPool:
            def map_entries(self, entries, *, timeout):
                captured["entries"] = entries
                captured["timeout"] = timeout
                return [{"success": True, "model_id": entries[0]["payload"]["model_id"]}]

        with mock.patch.object(worker_pool, "_await_nneval_worker_pool", return_value=DummyPool()):
            results = worker_pool.evaluate_model_entries(
                [{"payload": {"model_id": "B000", "model_dir": "/tmp/model", "code_file": "/tmp/model/new_nn.py"}}],
                use_all_visible_gpus=True,
            )

        self.assertEqual(results, [{"success": True, "model_id": "B000"}])
        self.assertEqual(captured["entries"][0]["payload"]["cmd"], "evaluate_model")
        self.assertEqual(captured["entries"][0]["payload"]["model_id"], "B000")


if __name__ == "__main__":
    unittest.main()
