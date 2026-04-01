import contextlib
import importlib
import io
import os
import sys
import types
import unittest


def _install_module_stubs():
    def _decorator_passthrough(fn=None):
        if fn is None:
            return lambda inner: inner
        return fn

    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = object
    torch_mod.dtype = object
    nn_mod = types.ModuleType("torch.nn")
    nn_mod.Module = object
    functional_mod = types.ModuleType("torch.nn.functional")
    nn_mod.functional = functional_mod
    torch_mod.nn = nn_mod
    torch_mod.no_grad = _decorator_passthrough
    torch_mod.bfloat16 = "bfloat16"
    torch_mod.float16 = "float16"
    torch_mod.distributed = types.SimpleNamespace(is_available=lambda: False, is_initialized=lambda: False)
    torch_mod.cuda = types.SimpleNamespace(
        empty_cache=lambda: None,
        is_available=lambda: False,
        is_bf16_supported=lambda: False,
    )
    torch_mod.device = lambda name: name
    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = nn_mod
    sys.modules["torch.nn.functional"] = functional_mod

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoTokenizer = object
    transformers_mod.AutoModelForCausalLM = object
    transformers_mod.TrainingArguments = object
    sys.modules["transformers"] = transformers_mod

    peft_mod = types.ModuleType("peft")
    peft_mod.LoraConfig = object
    peft_mod.get_peft_model = lambda model, config: model
    peft_mod.PeftModel = object
    peft_mod.prepare_model_for_kbit_training = lambda model, *args, **kwargs: model
    sys.modules["peft"] = peft_mod

    trl_mod = types.ModuleType("trl")
    trl_trainer_mod = types.ModuleType("trl.trainer")
    trl_grpo_trainer_mod = types.ModuleType("trl.trainer.grpo_trainer")
    trl_grpo_trainer_mod.GRPOTrainer = object
    trl_grpo_config_mod = types.ModuleType("trl.trainer.grpo_config")
    trl_grpo_config_mod.GRPOConfig = object
    sys.modules["trl"] = trl_mod
    sys.modules["trl.trainer"] = trl_trainer_mod
    sys.modules["trl.trainer.grpo_trainer"] = trl_grpo_trainer_mod
    sys.modules["trl.trainer.grpo_config"] = trl_grpo_config_mod

    datasets_mod = types.ModuleType("datasets")
    datasets_mod.Dataset = object
    sys.modules["datasets"] = datasets_mod

    reward_mod = types.ModuleType("ab.gpt.util.Reward")
    reward_mod.PersistentEvalWorkerError = RuntimeError
    reward_mod.evaluate_code_and_reward = lambda *args, **kwargs: None
    reward_mod.evaluate_code_and_reward_batch = lambda *args, **kwargs: []
    reward_mod.get_distributed_runtime_info = lambda: {
        "distributed": False,
        "world_size": 1,
        "rank": 0,
        "raw_local_rank": 0,
        "local_rank": 0,
        "visible_gpu_count": 0,
        "visible_gpu_tokens": [],
        "train_gpu": None,
        "train_gpu_token": None,
    }
    reward_mod.get_eval_worker_diagnostics = lambda: None
    reward_mod.shutdown_eval_worker = lambda: None
    util_pkg_mod = types.ModuleType("ab.gpt.util")
    util_pkg_mod.__path__ = [os.path.join(os.path.dirname(__file__), "ab", "gpt", "util")]
    sys.modules["ab.gpt.util"] = util_pkg_mod
    sys.modules["ab.gpt.util.Reward"] = reward_mod
    util_pkg_mod.Reward = reward_mod

    util_mod = types.ModuleType("ab.gpt.util.Util")
    util_mod.extract_str = lambda text, start, end: text.split(start, 1)[1].split(end, 1)[0] if start in text and end in text else ""
    sys.modules["ab.gpt.util.Util"] = util_mod
    util_pkg_mod.Util = util_mod

    const_mod = types.ModuleType("ab.gpt.util.Const")
    const_mod.conf_train_dir = lambda *args, **kwargs: ""
    const_mod.conf_test_dir = lambda *args, **kwargs: ""
    const_mod.epoch_dir = lambda *args, **kwargs: ""
    const_mod.new_nn_file = "new_nn.py"
    const_mod.new_out_file = "new_out.txt"
    const_mod.synth_dir = lambda path: path
    sys.modules["ab.gpt.util.Const"] = const_mod
    util_pkg_mod.Const = const_mod

    sftutil_mod = types.ModuleType("ab.gpt.util.SFTUtil")
    sftutil_mod.legacy_patterns = []
    sftutil_mod.available_backbones = ["resnet18", "mobilenet_v2", "efficientnet_b0"]
    sftutil_mod.open_discovery_goal_profiles = []
    sftutil_mod.open_discovery_prompt_template = ""
    sftutil_mod.open_discovery_rl_prompt_template = ""
    sftutil_mod.open_discovery_skeleton_code = """
import torch
from torch import nn

def adaptive_pool_flatten(x):
    return x

def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):
    return nn.Identity()

class TorchVision(nn.Module):
    def __init__(self, model=None, in_channels=3):
        super().__init__()

    def forward(self, x):
        return x

class FractalBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()

    def infer_dimensions_dynamically(self, n_classes):
        return None

    def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
        return x
"""
    sys.modules["ab.gpt.util.SFTUtil"] = sftutil_mod
    util_pkg_mod.SFTUtil = sftutil_mod

    ab_nn_mod = types.ModuleType("ab.nn")
    sys.modules["ab.nn"] = ab_nn_mod
    ab_nn_api_mod = types.ModuleType("ab.nn.api")
    ab_nn_api_mod.data = lambda *args, **kwargs: None
    sys.modules["ab.nn.api"] = ab_nn_api_mod
    ab_nn_mod.api = ab_nn_api_mod

    ab_nn_util_mod = types.ModuleType("ab.nn.util")
    sys.modules["ab.nn.util"] = ab_nn_util_mod
    ab_nn_util_util_mod = types.ModuleType("ab.nn.util.Util")
    ab_nn_util_util_mod.create_file = lambda *args, **kwargs: None
    sys.modules["ab.nn.util.Util"] = ab_nn_util_util_mod
    ab_nn_util_mod.Util = ab_nn_util_util_mod


def _make_completion(model_a: str, model_b: str, pattern: str) -> str:
    return f"""
<block>
def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):
    return nn.Identity()
</block>
<init>
def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
    super().__init__()
    self.pattern = "{pattern}"
    self.device = device
    self.use_amp = False
    self._input_spec = tuple(in_shape[1:])
    self.backbone_a = TorchVision(model="{model_a}", in_channels=in_shape[1])
    self.backbone_b = TorchVision(model="{model_b}", in_channels=in_shape[1])
    self.classifier = nn.Identity()
    self.infer_dimensions_dynamically(out_shape[0])
</init>
<forward>
def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
    x_a = adaptive_pool_flatten(self.backbone_a(x)).flatten(1)
    x_b = adaptive_pool_flatten(self.backbone_b(x)).flatten(1)
    fused = torch.cat([x_a, x_b], dim=1)
    return self.classifier(fused)
</forward>
""".strip()


class _DummyLogger:
    def log_to_file(self, *_args, **_kwargs):
        return None

    def log_generation(self, *_args, **_kwargs):
        return None


class TuneRLRewardLogicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_module_stubs()
        sys.path.insert(0, os.path.dirname(__file__))
        cls.tunerl = importlib.import_module("ab.gpt.TuneRL")

    def setUp(self):
        self.tunerl.code_logger = _DummyLogger()
        self.tunerl.create_file = lambda *args, **kwargs: None
        self.tunerl.graph_archive_counts.clear()
        self.tunerl.family_archive_counts.clear()
        self.tunerl.family_hash_archive_counts.clear()
        self.tunerl.family_metric_best.clear()
        self.tunerl.motif_name_counts.clear()
        self.tunerl.saved_graph_counts.clear()
        self.tunerl.saved_family_hash_counts.clear()
        self.tunerl.goal_graph_archive_counts.clear()
        self.tunerl.goal_family_hash_archive_counts.clear()
        self.tunerl.saved_goal_family_hash_counts.clear()
        self.tunerl.train_graph_hashes.clear()
        self.tunerl.train_family_hashes.clear()
        self.tunerl.train_descriptor_keys.clear()

    def test_reward_order_invariance_for_same_family(self):
        completion_a = _make_completion("resnet18", "mobilenet_v2", "MotifA")
        completion_b = _make_completion("efficientnet_b0", "mobilenet_v2", "MotifB")

        def fake_eval(*args, **kwargs):
            baseline = kwargs["seed_accuracy_baseline"]
            return {
                "reward": 0.0,
                "components": {},
                "val_metric": 0.12,
                "built_ok": True,
                "forward_ok": True,
                "forward_shape_ok": True,
                "trained_step_ok": True,
                "backward_ok": True,
                "loss_start": 1.0,
                "loss_end": 0.9,
                "loss_drop": 0.1,
                "loss_drop_ok": True,
                "train_acc": baseline + 0.05,
                "accuracy_baseline": baseline,
                "train_acc_gain": 0.05,
                "train_acc_improved": True,
                "latency_ms": None,
                "params_m": None,
            }

        self.tunerl.evaluate_code_and_reward = fake_eval
        info_a = self.tunerl.extract_graph_info(*self.tunerl.extract_completion_blocks(completion_a)[1:], legacy_patterns=[])
        info_b = self.tunerl.extract_graph_info(*self.tunerl.extract_completion_blocks(completion_b)[1:], legacy_patterns=[])
        self.assertEqual(info_a.family_hash, info_b.family_hash)
        self.assertNotEqual(info_a.graph_hash, info_b.graph_hash)
        self.tunerl.saved_graph_counts[info_a.graph_hash] = 1
        self.tunerl.saved_graph_counts[info_b.graph_hash] = 1

        prompts_ab = ["Discovery Target Tags: stem", "Discovery Target Tags: stem"]
        with contextlib.redirect_stdout(io.StringIO()):
            scores_ab = self.tunerl.compute_reward(
                prompts_ab,
                [completion_a, completion_b],
                accuracy=[0.6, 0.6],
            )

        self.setUp()
        self.tunerl.evaluate_code_and_reward = fake_eval
        self.tunerl.saved_graph_counts[info_a.graph_hash] = 1
        self.tunerl.saved_graph_counts[info_b.graph_hash] = 1
        with contextlib.redirect_stdout(io.StringIO()):
            scores_ba = self.tunerl.compute_reward(
                prompts_ab,
                [completion_b, completion_a],
                accuracy=[0.6, 0.6],
            )

        score_by_completion_ab = {completion_a: scores_ab[0], completion_b: scores_ab[1]}
        score_by_completion_ba = {completion_b: scores_ba[0], completion_a: scores_ba[1]}
        self.assertEqual(score_by_completion_ab[completion_a], score_by_completion_ba[completion_a])
        self.assertEqual(score_by_completion_ab[completion_b], score_by_completion_ba[completion_b])

    def test_trainset_novelty_requires_loss_drop(self):
        completion = _make_completion("resnet18", "mobilenet_v2", "NovelMotif")

        def fake_eval_no_drop(*args, **kwargs):
            baseline = kwargs["seed_accuracy_baseline"]
            return {
                "reward": 0.0,
                "components": {},
                "val_metric": 0.11,
                "built_ok": True,
                "forward_ok": True,
                "forward_shape_ok": True,
                "trained_step_ok": True,
                "backward_ok": True,
                "loss_start": 1.0,
                "loss_end": 0.99,
                "loss_drop": 0.01,
                "loss_drop_ok": False,
                "train_acc": baseline + 0.10,
                "accuracy_baseline": baseline,
                "train_acc_gain": 0.10,
                "train_acc_improved": True,
                "latency_ms": None,
                "params_m": None,
            }

        def fake_eval_with_drop(*args, **kwargs):
            result = fake_eval_no_drop(*args, **kwargs)
            result["loss_end"] = 0.9
            result["loss_drop"] = 0.1
            result["loss_drop_ok"] = True
            return result

        self.tunerl.evaluate_code_and_reward = fake_eval_no_drop
        no_drop = self.tunerl.base_discovery_reward_fn(
            completion,
            seed_accuracy_baseline=0.6,
            prompt_goal_tags=[],
        )
        self.assertEqual(no_drop["reward"], 0.0)
        self.assertEqual(no_drop["open_discovery"]["r_trainset_novelty"], 0.0)
        self.assertFalse(no_drop["open_discovery"]["novel_vs_trainset_family"])

        self.tunerl.evaluate_code_and_reward = fake_eval_with_drop
        with_drop = self.tunerl.base_discovery_reward_fn(
            completion,
            seed_accuracy_baseline=0.6,
            prompt_goal_tags=[],
        )
        self.assertLess(with_drop["reward"], no_drop["reward"])
        self.assertAlmostEqual(with_drop["open_discovery"]["r_trainset_novelty"], 0.04, places=6)
        self.assertTrue(with_drop["open_discovery"]["novel_vs_trainset_family"])
        self.assertTrue(with_drop["open_discovery"]["novel_vs_trainset_graph"])

    def test_compute_reward_requires_accuracy_column(self):
        completion = _make_completion("resnet18", "mobilenet_v2", "MotifA")

        with self.assertRaisesRegex(ValueError, "accuracy"):
            self.tunerl.compute_reward(["Discovery Target Tags: stem"], [completion])

    def test_reward_tracks_sample_level_train_acc_gain(self):
        completion = _make_completion("resnet18", "mobilenet_v2", "GainMotif")
        baselines_seen = []

        def fake_eval(*args, **kwargs):
            baseline = kwargs["seed_accuracy_baseline"]
            baselines_seen.append(baseline)
            train_acc = baseline + 0.04
            return {
                "reward": 0.0,
                "components": {},
                "val_metric": 0.05,
                "built_ok": True,
                "forward_ok": True,
                "forward_shape_ok": True,
                "trained_step_ok": True,
                "backward_ok": True,
                "loss_start": 1.0,
                "loss_end": 0.8,
                "loss_drop": 0.2,
                "loss_drop_ok": True,
                "train_acc": train_acc,
                "accuracy_baseline": baseline,
                "train_acc_gain": train_acc - baseline,
                "train_acc_improved": True,
                "latency_ms": None,
                "params_m": None,
            }

        self.tunerl.evaluate_code_and_reward = fake_eval
        graph_info = self.tunerl.extract_graph_info(
            *self.tunerl.extract_completion_blocks(completion)[1:],
            legacy_patterns=[],
        )
        self.tunerl.saved_graph_counts[graph_info.graph_hash] = 1
        with contextlib.redirect_stdout(io.StringIO()):
            rewards = self.tunerl.compute_reward(
                ["Discovery Target Tags: stem"],
                [completion],
                accuracy=[0.73],
            )

        self.assertEqual(baselines_seen, [0.73])
        self.assertAlmostEqual(rewards[0], 0.1075, places=6)

    def test_resolve_generation_plan_requires_divisible_target(self):
        original = os.environ.get("NNGPT_RL_NUM_GENERATIONS")
        try:
            os.environ["NNGPT_RL_NUM_GENERATIONS"] = "8"
            plan = self.tunerl.resolve_generation_plan(
                {"world_size": 8},
                env_name="NNGPT_RL_NUM_GENERATIONS",
                default=8,
            )
            self.assertEqual(plan["trainer_num_generations"], 1)
            self.assertEqual(plan["effective_global_num_generations"], 8)

            with self.assertRaisesRegex(ValueError, "must be divisible by WORLD_SIZE=3"):
                self.tunerl.resolve_generation_plan(
                    {"world_size": 3},
                    env_name="NNGPT_RL_NUM_GENERATIONS",
                    default=8,
                )
        finally:
            if original is None:
                os.environ.pop("NNGPT_RL_NUM_GENERATIONS", None)
            else:
                os.environ["NNGPT_RL_NUM_GENERATIONS"] = original

    def test_distributed_compute_reward_precomputes_shards_before_global_scoring(self):
        completion = _make_completion("resnet18", "mobilenet_v2", "DistMotif")
        remote_completion = _make_completion("efficientnet_b0", "mobilenet_v2", "DistRemote")
        precompute_calls = []
        gather_calls = []

        original_distributed_initialized = self.tunerl._distributed_initialized
        original_distributed_world_size = self.tunerl._distributed_world_size
        original_distributed_rank = self.tunerl._distributed_rank
        original_is_main_process = self.tunerl.is_main_process
        original_all_gather_object = self.tunerl._all_gather_object
        original_broadcast_object = self.tunerl._broadcast_object
        original_precompute_eval_results = self.tunerl._precompute_eval_results
        original_score_reward_entries = self.tunerl._score_reward_entries
        original_finalize_scored_results = self.tunerl._finalize_scored_results
        original_print_discovery_metrics = self.tunerl._print_discovery_metrics
        try:
            self.tunerl._distributed_initialized = lambda: True
            self.tunerl._distributed_world_size = lambda: 2
            self.tunerl._distributed_rank = lambda: 0
            self.tunerl.is_main_process = lambda: True
            remote_graph_info = self.tunerl.extract_graph_info(
                *self.tunerl.extract_completion_blocks(remote_completion)[1:],
                legacy_patterns=[],
            )

            def fake_all_gather_object(payload):
                gather_calls.append(payload)
                if len(gather_calls) == 1:
                    remote_entry = {
                        "rank": 1,
                        "local_index": 0,
                        "prompt": "Discovery Target Tags: stem",
                        "completion": remote_completion,
                        "graph_info": remote_graph_info,
                        "prompt_goal_tags": ["stem"],
                        "goal_key": "stem",
                        "seed_accuracy_baseline": 0.6,
                        "precomputed_eval_result": None,
                    }
                    return [payload, [remote_entry]]
                remote_result = {
                    "reward": 0.5,
                    "components": {},
                    "val_metric": 0.2,
                    "built_ok": True,
                    "forward_ok": True,
                    "forward_shape_ok": True,
                    "trained_step_ok": True,
                    "backward_ok": True,
                    "loss_start": 1.0,
                    "loss_end": 0.8,
                    "loss_drop": 0.2,
                    "loss_drop_ok": True,
                    "train_acc": 0.65,
                    "accuracy_baseline": 0.6,
                    "train_acc_gain": 0.05,
                    "train_acc_improved": True,
                    "latency_ms": None,
                    "params_m": None,
                }
                remote_entry = {
                    "rank": 1,
                    "local_index": 0,
                    "global_index": 1,
                    "prompt": "Discovery Target Tags: stem",
                    "completion": remote_completion,
                    "graph_info": remote_graph_info,
                    "prompt_goal_tags": ["stem"],
                    "goal_key": "stem",
                    "seed_accuracy_baseline": 0.6,
                    "precomputed_eval_result": remote_result,
                }
                return [payload, [remote_entry]]

            self.tunerl._all_gather_object = fake_all_gather_object
            self.tunerl._broadcast_object = lambda payload, src=0: payload
            def fake_precompute(entries, **kwargs):
                precompute_calls.append([entry["global_index"] for entry in entries])
                for entry in entries:
                    entry["precomputed_eval_result"] = {
                        "reward": 0.25,
                        "components": {},
                        "val_metric": 0.1,
                        "built_ok": True,
                        "forward_ok": True,
                        "forward_shape_ok": True,
                        "trained_step_ok": True,
                        "backward_ok": True,
                        "loss_start": 1.0,
                        "loss_end": 0.85,
                        "loss_drop": 0.15,
                        "loss_drop_ok": True,
                        "train_acc": 0.64,
                        "accuracy_baseline": 0.6,
                        "train_acc_gain": 0.04,
                        "train_acc_improved": True,
                        "latency_ms": None,
                        "params_m": None,
                    }

            self.tunerl._precompute_eval_results = fake_precompute
            def fake_score_reward_entries(entries, **kwargs):
                self.assertEqual([entry["global_index"] for entry in entries], [0, 1])
                self.assertTrue(all(entry.get("precomputed_eval_result") is not None for entry in entries))
                return [
                    {
                        "rank": int(entry["rank"]),
                        "local_index": int(entry["local_index"]),
                        "prompt": entry["prompt"],
                        "completion": entry["completion"],
                        "graph_info": entry["graph_info"],
                        "goal_key": entry["goal_key"],
                        "result": {"reward": 0.25 if int(entry["rank"]) == 0 else 0.5},
                        "score": 0.25 if int(entry["rank"]) == 0 else 0.5,
                    }
                    for entry in entries
                ]

            self.tunerl._score_reward_entries = fake_score_reward_entries
            self.tunerl._finalize_scored_results = lambda scored_results: None
            self.tunerl._print_discovery_metrics = lambda: None

            with contextlib.redirect_stdout(io.StringIO()):
                rewards = self.tunerl.compute_reward(
                    ["Discovery Target Tags: stem"],
                    [completion],
                    accuracy=[0.6],
                )
        finally:
            self.tunerl._distributed_initialized = original_distributed_initialized
            self.tunerl._distributed_world_size = original_distributed_world_size
            self.tunerl._distributed_rank = original_distributed_rank
            self.tunerl.is_main_process = original_is_main_process
            self.tunerl._all_gather_object = original_all_gather_object
            self.tunerl._broadcast_object = original_broadcast_object
            self.tunerl._precompute_eval_results = original_precompute_eval_results
            self.tunerl._score_reward_entries = original_score_reward_entries
            self.tunerl._finalize_scored_results = original_finalize_scored_results
            self.tunerl._print_discovery_metrics = original_print_discovery_metrics

        self.assertEqual(precompute_calls, [[0]])
        self.assertEqual(rewards, [0.25])


if __name__ == "__main__":
    unittest.main()
