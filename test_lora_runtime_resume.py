import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


def _install_lora_stubs():
    original_modules = {
        name: sys.modules.get(name)
        for name in (
            "torch",
            "torch.nn",
            "datasets",
            "peft",
            "transformers",
            "trl",
            "trl.trainer",
            "trl.trainer.sft_trainer",
            "ab.nn.util.Util",
        )
    }

    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)

    class _DummyLinear:
        pass

    nn_mod = types.ModuleType("torch.nn")
    nn_mod.modules = types.SimpleNamespace(linear=types.SimpleNamespace(Linear=_DummyLinear))
    torch_mod.nn = nn_mod

    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = nn_mod

    datasets_mod = types.ModuleType("datasets")
    datasets_mod.Dataset = object
    sys.modules["datasets"] = datasets_mod

    peft_mod = types.ModuleType("peft")

    class DummyPeftConfig:
        def __init__(self, r=8, lora_alpha=16, target_modules=None, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"):
            self.r = r
            self.lora_alpha = lora_alpha
            self.target_modules = list(target_modules or [])
            self.lora_dropout = lora_dropout
            self.bias = bias
            self.task_type = task_type

    peft_mod.LoraConfig = DummyPeftConfig
    peft_mod.get_peft_model = lambda model, config: model
    peft_mod.prepare_model_for_kbit_training = lambda model: model
    sys.modules["peft"] = peft_mod

    transformers_mod = types.ModuleType("transformers")

    class DummyTrainingArguments:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def to_dict(self):
            return dict(self.__dict__)

    class DummyTrainer:
        last_instance = None

        def __init__(self, *args, **kwargs):
            DummyTrainer.last_instance = self
            self.model = kwargs["model"]
            self.args = kwargs["args"]
            self.callbacks = []
            self.resume_from_checkpoint = None

        def add_callback(self, callback):
            self.callbacks.append(callback)

        def train(self, resume_from_checkpoint=None):
            self.resume_from_checkpoint = resume_from_checkpoint
            state = types.SimpleNamespace(global_step=5)
            control = object()
            for callback in self.callbacks:
                callback.on_save(self.args, state, control)
            return types.SimpleNamespace(metrics={"loss": 0.1})

        def log_metrics(self, *args, **kwargs):
            return None

        def save_metrics(self, *args, **kwargs):
            return None

        def save_state(self):
            return None

    transformers_mod.TrainingArguments = DummyTrainingArguments
    transformers_mod.PreTrainedModel = object
    transformers_mod.PreTrainedTokenizerBase = object
    transformers_mod.Trainer = DummyTrainer
    transformers_mod.TrainerCallback = object
    sys.modules["transformers"] = transformers_mod

    trl_mod = types.ModuleType("trl")

    class DummySFTConfig(DummyTrainingArguments):
        pass

    class DummyCollator:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    trl_mod.SFTConfig = DummySFTConfig
    trl_mod.SFTTrainer = DummyTrainer
    trl_mod.DataCollatorForCompletionOnlyLM = DummyCollator
    sys.modules["trl"] = trl_mod

    trl_trainer_mod = types.ModuleType("trl.trainer")
    sys.modules["trl.trainer"] = trl_trainer_mod

    trl_sft_mod = types.ModuleType("trl.trainer.sft_trainer")
    trl_sft_mod.DataCollatorForLanguageModeling = DummyCollator
    trl_sft_mod.DataCollatorForCompletionOnlyLM = DummyCollator
    sys.modules["trl.trainer.sft_trainer"] = trl_sft_mod

    util_mod = types.ModuleType("ab.nn.util.Util")
    util_mod.release_memory = lambda: None
    sys.modules["ab.nn.util.Util"] = util_mod
    return original_modules


class _DummyParam:
    requires_grad = True
    dtype = "float32"

    def numel(self):
        return 10


class _DummyModel:
    def __init__(self):
        self.config = types.SimpleNamespace(use_cache=True)

    def gradient_checkpointing_enable(self):
        return None

    def named_parameters(self):
        return [("weight", _DummyParam())]

    def named_modules(self):
        return []

    def save_pretrained(self, output_dir, access_token=None):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "adapter.bin").write_text("ok", encoding="utf-8")


class _DummyTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "</s>"


class _DummyDataset:
    column_names = ["text"]

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"text": "sample prompt"}

    def train_test_split(self, test_size=0.1):
        return {"train": self, "test": self}


class LoRARuntimeResumeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._original_modules = _install_lora_stubs()
        sys.path.insert(0, os.path.dirname(__file__))
        from ab.gpt.util import training_runtime as runtime
        from ab.gpt.util.LoRA import LoRA
        from peft import LoraConfig
        from trl import SFTConfig
        from transformers import Trainer

        cls.runtime = runtime
        cls.LoRA = LoRA
        cls.LoraConfig = LoraConfig
        cls.SFTConfig = SFTConfig
        cls.Trainer = Trainer

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("ab.gpt.util.LoRA", None)
        for name, module in cls._original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_lora_train_restores_runtime_state_and_uses_trainer_resume(self):
        restored_states = []
        reset_calls = []

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_out = Path(tmpdir) / "trainer_out"
            resume_checkpoint = Path(tmpdir) / "checkpoint-3"
            resume_checkpoint.mkdir(parents=True)
            (resume_checkpoint / "runtime_state.json").write_text(
                json.dumps({"reward_batch_index": 3}),
                encoding="utf-8",
            )

            training_args = self.SFTConfig(output_dir=str(trainer_out))
            peft_config = self.LoraConfig(target_modules=["q_proj"])
            lora = self.LoRA(
                _DummyModel(),
                _DummyTokenizer(),
                training_args=training_args,
                peft_config=peft_config,
            )
            hooks = self.runtime.RuntimeStateHooks(
                capture=lambda: {"reward_batch_index": 5},
                restore=lambda state: restored_states.append(state),
                reset=lambda: reset_calls.append(True),
            )

            lora.train(
                _DummyDataset(),
                _DummyTokenizer(),
                str(Path(tmpdir) / "final_model"),
                resume_from_checkpoint=str(resume_checkpoint),
                runtime_state_hooks=hooks,
            )

            trainer_instance = self.Trainer.last_instance
            self.assertIsNotNone(trainer_instance)
            self.assertEqual(restored_states, [{"reward_batch_index": 3}])
            self.assertEqual(reset_calls, [])
            self.assertEqual(trainer_instance.resume_from_checkpoint, str(resume_checkpoint.resolve()))
            self.assertEqual(
                json.loads((trainer_out / "checkpoint-5" / "runtime_state.json").read_text(encoding="utf-8")),
                {"reward_batch_index": 5},
            )


if __name__ == "__main__":
    unittest.main()
