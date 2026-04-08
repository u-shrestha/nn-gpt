import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, os.path.dirname(__file__))


class _DummyFrame:
    columns = []


class EvalRuntimeIsolationTest(unittest.TestCase):
    def test_evaluate_isolates_tmp_module_root_per_process(self):
        captured = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            api_mod = types.ModuleType("ab.nn.api")

            def _data():
                return _DummyFrame()

            _data.cache_clear = lambda: None
            api_mod.data = _data

            def _check_nn(code, task, dataset, metric, prm, save_to_db=True, prefix=None, save_path=None):
                import ab.nn.util.Train as train_runtime
                from ab.nn.util.Const import ab_root_path

                captured["train_out_during_call"] = train_runtime.out
                isolated_root = Path(ab_root_path) / train_runtime.out
                (isolated_root / "nn" / "tmp").mkdir(parents=True, exist_ok=True)
                (isolated_root / "nn" / "tmp" / "__init__.py").write_text("", encoding="utf-8")
                sys.modules[f"{train_runtime.out}.nn.tmp.fake"] = types.ModuleType("fake")
                return ("fake-model", 0.5, 0.1, 1.0)

            api_mod.check_nn = _check_nn

            util_mod = types.ModuleType("ab.nn.util.Util")
            util_mod.uuid4 = lambda _code: "checksum"

            const_mod = types.ModuleType("ab.nn.util.Const")
            const_mod.base_module = "ab"
            const_mod.ab_root_path = root
            const_mod.out_dir = root / "out"

            train_mod = types.ModuleType("ab.nn.util.Train")
            train_mod.out = "out"

            gpt_util_mod = types.ModuleType("ab.gpt.util.Util")
            gpt_util_mod.read_py_file_as_string = lambda path: Path(path).read_text(encoding="utf-8")

            nn_mod = types.ModuleType("ab.nn")
            nn_mod.__path__ = []
            nn_util_mod = types.ModuleType("ab.nn.util")
            nn_util_mod.__path__ = []

            with mock.patch.dict(
                sys.modules,
                {
                    "ab.nn": nn_mod,
                    "ab.nn.api": api_mod,
                    "ab.nn.util": nn_util_mod,
                    "ab.nn.util.Util": util_mod,
                    "ab.nn.util.Const": const_mod,
                    "ab.nn.util.Train": train_mod,
                    "ab.gpt.util.Util": gpt_util_mod,
                },
                clear=False,
            ):
                eval_module = importlib.import_module("ab.gpt.util.Eval")
                eval_module = importlib.reload(eval_module)

                model_dir = root / "model"
                model_dir.mkdir(parents=True, exist_ok=True)
                nn_file = model_dir / "new_nn.py"
                nn_file.write_text(
                    "\n" + "\n".join(
                        [
                            "def supported_hyperparameters():",
                            "    return ['lr']",
                            "",
                            "def train_setup(prm):",
                            "    return prm['lr']",
                            "",
                            "def learn(train, test, device, prm):",
                            "    return prm['lr']",
                        ]
                    ),
                    encoding="utf-8",
                )

                evaluator = eval_module.Eval(
                    model_source_package=str(model_dir),
                    task="img-classification",
                    dataset="cifar-10",
                    metric="acc",
                    prm={"lr": 0.01, "batch": 8, "dropout": 0.2, "momentum": 0.9, "transform": "norm_256_flip", "epoch": 1},
                    save_to_db=False,
                    prefix="runtime",
                    save_path=str(model_dir),
                )
                result = evaluator.evaluate(nn_file)

                self.assertEqual(result, ("fake-model", 0.5, 0.1, 1.0))
                self.assertEqual(train_mod.out, "out")
                self.assertTrue(captured["train_out_during_call"].startswith("out_nneval_tmp_pid_"))
                self.assertFalse((root / captured["train_out_during_call"]).exists())
                self.assertNotIn(f"{captured['train_out_during_call']}.nn.tmp.fake", sys.modules)


if __name__ == "__main__":
    unittest.main()
