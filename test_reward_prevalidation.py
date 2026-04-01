import importlib
import os
import sys
import types
import unittest


def _install_reward_stubs():
    def _decorator_passthrough(fn=None):
        if fn is None:
            return lambda inner: inner
        return fn

    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = object
    torch_mod.device = lambda name: name
    torch_mod.no_grad = _decorator_passthrough
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
        memory_allocated=lambda *args, **kwargs: 0.0,
        memory_reserved=lambda *args, **kwargs: 0.0,
        empty_cache=lambda: None,
        ipc_collect=lambda: None,
        get_device_properties=lambda *_args, **_kwargs: types.SimpleNamespace(total_memory=0),
        mem_get_info=lambda *args, **kwargs: (0, 0),
    )
    torch_mod.distributed = types.SimpleNamespace(
        is_available=lambda: False,
        is_initialized=lambda: False,
    )

    nn_mod = types.ModuleType("torch.nn")
    nn_mod.Module = object
    nn_mod.Parameter = object
    functional_mod = types.ModuleType("torch.nn.functional")
    nn_mod.functional = functional_mod
    torch_mod.nn = nn_mod

    utils_mod = types.ModuleType("torch.utils")
    data_mod = types.ModuleType("torch.utils.data")

    class _DummyLoader:
        pass

    data_mod.DataLoader = _DummyLoader
    data_mod.TensorDataset = _DummyLoader
    data_mod.Subset = _DummyLoader
    utils_mod.data = data_mod
    torch_mod.utils = utils_mod

    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = nn_mod
    sys.modules["torch.nn.functional"] = functional_mod
    sys.modules["torch.utils"] = utils_mod
    sys.modules["torch.utils.data"] = data_mod


VALID_CODE = """
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device):
        self.device = device
        self.use_amp = False
        self._input_spec = tuple(in_shape[1:])
        self.infer_dimensions_dynamically(out_shape[0])

    def infer_dimensions_dynamically(self, num_classes):
        return None

    def forward(self, x, is_probing: bool = False):
        return x
""".strip()


LEGACY_INFER_DIMENSIONS_CODE = """
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device):
        self.device = device
        self.use_amp = False
        self._input_spec = tuple(in_shape[1:])
        self.infer_dimensions(in_shape, out_shape[0])

    def infer_dimensions_dynamically(self, num_classes):
        return None

    def forward(self, x, is_probing: bool = False):
        return x
""".strip()


WRONG_DYNAMIC_SIGNATURE_CODE = """
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device):
        self.device = device
        self.use_amp = False
        self._input_spec = tuple(in_shape[1:])
        self.infer_dimensions_dynamically(in_shape, out_shape[0])

    def infer_dimensions_dynamically(self, num_classes):
        return None

    def forward(self, x, is_probing: bool = False):
        return x
""".strip()


class RewardPrevalidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_reward_stubs()
        sys.path.insert(0, os.path.dirname(__file__))
        cls.reward = importlib.import_module("ab.gpt.util.Reward")

    def _preview(self, code: str):
        return self.reward._preview_eval_request(
            code=code,
            in_shape=(1, 3, 224, 224),
            out_shape=(10,),
            prm={"lr": 0.01, "epoch": 1, "batch": 64},
            device="cpu",
            seed_accuracy_baseline=0.1,
            cfg=None,
        )

    def test_valid_code_passes_cpu_prevalidation(self):
        preview = self._preview(VALID_CODE)
        self.assertIsNone(preview["prevalidated_result"])

    def test_legacy_infer_dimensions_call_is_rejected_before_gpu(self):
        preview = self._preview(LEGACY_INFER_DIMENSIONS_CODE)
        result = preview["prevalidated_result"]
        self.assertIsNotNone(result)
        self.assertEqual(result["error_stage"], "cpu_prevalidate")
        self.assertIn("infer_dimensions", result["error"])

    def test_wrong_infer_dimensions_dynamically_arity_is_rejected(self):
        preview = self._preview(WRONG_DYNAMIC_SIGNATURE_CODE)
        result = preview["prevalidated_result"]
        self.assertIsNotNone(result)
        self.assertEqual(result["error_stage"], "cpu_prevalidate")
        self.assertIn("takes 2 positional arguments but 3 were given", result["error"])

    def test_distributed_reward_worker_plan_uses_local_gpu_only(self):
        reward = self.reward
        original_env = {key: os.environ.get(key) for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK")}
        original_cuda = reward.torch.cuda
        try:
            os.environ["WORLD_SIZE"] = "8"
            os.environ["RANK"] = "3"
            os.environ["LOCAL_RANK"] = "3"
            reward.torch.cuda = types.SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 8,
                memory_allocated=lambda *args, **kwargs: 0.0,
                memory_reserved=lambda *args, **kwargs: 0.0,
                empty_cache=lambda: None,
                ipc_collect=lambda: None,
                get_device_properties=lambda *_args, **_kwargs: types.SimpleNamespace(total_memory=24 * (1024 ** 3)),
                mem_get_info=lambda device_index=0, *args, **kwargs: (
                    20 * (1024 ** 3) if int(device_index) == 3 else 2 * (1024 ** 3),
                    24 * (1024 ** 3),
                ),
            )

            plan = reward.get_reward_worker_plan()
        finally:
            reward.torch.cuda = original_cuda
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(plan["mode"], "distributed_local_dynamic_pool")
        self.assertEqual(plan["reward_gpu_indices"], [3])
        self.assertEqual(plan["pool_size"], 1)

    def test_distributed_reward_worker_plan_waits_when_local_gpu_has_no_headroom(self):
        reward = self.reward
        original_env = {key: os.environ.get(key) for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK")}
        original_cuda = reward.torch.cuda
        try:
            os.environ["WORLD_SIZE"] = "8"
            os.environ["RANK"] = "3"
            os.environ["LOCAL_RANK"] = "3"
            reward.torch.cuda = types.SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 8,
                memory_allocated=lambda *args, **kwargs: 0.0,
                memory_reserved=lambda *args, **kwargs: 0.0,
                empty_cache=lambda: None,
                ipc_collect=lambda: None,
                get_device_properties=lambda *_args, **_kwargs: types.SimpleNamespace(total_memory=24 * (1024 ** 3)),
                mem_get_info=lambda device_index=0, *args, **kwargs: (
                    8 * (1024 ** 3) if int(device_index) == 3 else 20 * (1024 ** 3),
                    24 * (1024 ** 3),
                ),
            )

            plan = reward.get_reward_worker_plan()
        finally:
            reward.torch.cuda = original_cuda
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(plan["mode"], "distributed_gpu_wait")
        self.assertEqual(plan["reward_gpu_indices"], [])


if __name__ == "__main__":
    unittest.main()
