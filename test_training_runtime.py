import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


sys.path.insert(0, os.path.dirname(__file__))

from ab.gpt.util import training_runtime as runtime


class TrainingRuntimeTest(unittest.TestCase):
    def test_resolve_role_plan_auto_split(self):
        plan = runtime.resolve_role_plan(visible_gpu_tokens=["0", "1"], requested_mode="auto")
        self.assertEqual(plan.requested_mode, "auto")
        self.assertEqual(plan.resolved_mode, "split")
        self.assertEqual(plan.train_gpu_tokens, ["0"])
        self.assertEqual(plan.aux_gpu_tokens, ["0", "1"])

    def test_resolve_resume_spec_normalizes_adapter_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "stage1"
            adapter_dir = checkpoint_dir / "adapter"
            adapter_dir.mkdir(parents=True)
            (checkpoint_dir / "reward_state.json").write_text("{}", encoding="utf-8")

            spec = runtime.resolve_resume_spec(
                stage_checkpoint_dir=str(adapter_dir),
                legacy_state_filenames=("reward_state.json",),
            )

        self.assertEqual(spec.mode, "stage")
        self.assertEqual(spec.stage_checkpoint_dir, checkpoint_dir.resolve())
        self.assertEqual(spec.stage_adapter_dir, adapter_dir.resolve())

    def test_restore_or_reset_runtime_state_uses_legacy_alias(self):
        restored = []
        reset_calls = []
        hooks = runtime.RuntimeStateHooks(
            restore=lambda state: restored.append(state),
            reset=lambda: reset_calls.append(True),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            (checkpoint_dir / "reward_state.json").write_text(
                json.dumps({"reward_batch_index": 7}),
                encoding="utf-8",
            )
            restored_path = runtime.restore_or_reset_runtime_state(
                checkpoint_dir,
                hooks,
                legacy_state_filenames=("reward_state.json",),
            )

        self.assertEqual(restored, [{"reward_batch_index": 7}])
        self.assertEqual(reset_calls, [])
        self.assertEqual(restored_path, (checkpoint_dir / "reward_state.json").resolve())

    def test_restore_or_reset_runtime_state_resets_on_fresh_run(self):
        reset_calls = []
        hooks = runtime.RuntimeStateHooks(reset=lambda: reset_calls.append(True))
        restored_path = runtime.restore_or_reset_runtime_state(None, hooks)
        self.assertIsNone(restored_path)
        self.assertEqual(reset_calls, [True])

    def test_build_trainer_checkpoint_callback_writes_runtime_and_alias_state(self):
        original_transformers = sys.modules.get("transformers")
        transformers_stub = types.ModuleType("transformers")

        class TrainerCallback:
            pass

        transformers_stub.TrainerCallback = TrainerCallback
        sys.modules["transformers"] = transformers_stub

        try:
            callback = runtime.build_trainer_checkpoint_callback(
                runtime.RuntimeStateHooks(capture=lambda: {"reward_batch_index": 11}),
                state_aliases=("reward_state.json",),
            )
            self.assertIsNotNone(callback)

            with tempfile.TemporaryDirectory() as tmpdir:
                args = types.SimpleNamespace(output_dir=tmpdir)
                state = types.SimpleNamespace(global_step=12)
                control = object()
                callback.on_save(args, state, control)

                runtime_state_path = Path(tmpdir) / "checkpoint-12" / "runtime_state.json"
                reward_state_path = Path(tmpdir) / "checkpoint-12" / "reward_state.json"
                self.assertEqual(json.loads(runtime_state_path.read_text(encoding="utf-8")), {"reward_batch_index": 11})
                self.assertEqual(json.loads(reward_state_path.read_text(encoding="utf-8")), {"reward_batch_index": 11})
        finally:
            if original_transformers is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = original_transformers

    def test_save_runtime_checkpoint_writes_manifest_aliases(self):
        hooks = runtime.RuntimeStateHooks(capture=lambda: {"stage": "stage2"})
        manifest = {"event": "entered", "stage_name": "stage2"}

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "stage2"
            runtime.save_runtime_checkpoint(
                checkpoint_dir,
                hooks=hooks,
                manifest=manifest,
                state_aliases=("reward_state.json",),
                manifest_aliases=("stage_manifest.json",),
            )

            self.assertTrue((checkpoint_dir / "runtime_state.json").exists())
            self.assertTrue((checkpoint_dir / "reward_state.json").exists())
            self.assertEqual(
                json.loads((checkpoint_dir / "runtime_manifest.json").read_text(encoding="utf-8")),
                manifest,
            )
            self.assertEqual(
                json.loads((checkpoint_dir / "stage_manifest.json").read_text(encoding="utf-8")),
                manifest,
            )


if __name__ == "__main__":
    unittest.main()
