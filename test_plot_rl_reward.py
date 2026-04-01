import contextlib
import io
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

from ab.gpt import plot_rl_reward


def _make_record(
    reward: float = 0.5,
    *,
    built_ok: bool = True,
    forward_shape_ok: bool = True,
    backward_ok: bool = True,
    loss_drop_ok: bool = True,
    timed_out: bool = False,
    train_acc=0.72,
    seed_accuracy_baseline=0.68,
    seed_train_acc_gap=0.04,
    group_baseline_train_acc=0.70,
    group_train_acc_gain=0.02,
    group_train_acc_improved: bool = True,
    reward_batch_index: int = 1,
    reward_group_id: int = 0,
    group_warmup: bool = False,
    loss_start=1.0,
    loss_end=0.8,
    loss_drop=0.2,
    val_metric=0.15,
    estimated_total_seconds=None,
    eval_limit_seconds=270,
    warmup_dense_reward=None,
    anti_collapse_trainable_ok: bool | None = True,
    xml_tag_exact: bool | None = True,
    dual_backbone_ok: bool | None = True,
    class_count: int | None = 0,
    import_count: int | None = 0,
    bad_signature_count: int | None = 0,
    include_raw_extraction: bool = True,
    include_anti_collapse: bool = True,
    include_open_discovery: bool = True,
):
    api_result = {
        "built_ok": built_ok,
        "forward_ok": forward_shape_ok,
        "forward_shape_ok": forward_shape_ok,
        "backward_ok": backward_ok,
        "loss_drop_ok": loss_drop_ok,
        "train_acc": train_acc,
        "seed_accuracy_baseline": seed_accuracy_baseline,
        "seed_train_acc_gap": seed_train_acc_gap,
        "group_baseline_train_acc": group_baseline_train_acc,
        "group_train_acc_gain": group_train_acc_gain,
        "group_train_acc_improved": group_train_acc_improved,
        "reward_batch_index": reward_batch_index,
        "reward_group_id": reward_group_id,
        "group_warmup": group_warmup,
        "loss_start": loss_start,
        "loss_end": loss_end,
        "loss_drop": loss_drop,
        "val_metric": val_metric,
        "timed_out": timed_out,
        "estimated_total_seconds": estimated_total_seconds,
        "eval_limit_seconds": eval_limit_seconds,
        "warmup_dense_reward": warmup_dense_reward,
        "backbone_model_names": ["resnet18", "mobilenet_v3_small"],
    }
    if include_open_discovery:
        api_result["open_discovery"] = {"placeholder": True}
    if include_anti_collapse:
        api_result["anti_collapse"] = {"trainable_ok": anti_collapse_trainable_ok, "anti_collapse_delta": 0.0}
    if include_raw_extraction:
        api_result["raw_extraction"] = {
            "xml_tag_exact": xml_tag_exact,
            "dual_backbone_ok": dual_backbone_ok,
            "class_count": class_count,
            "import_count": import_count,
            "bad_signature_count": bad_signature_count,
        }
    return {
        "reward": reward,
        "api_result": api_result,
    }


class PlotRlRewardTest(unittest.TestCase):
    @contextlib.contextmanager
    def _matplotlib_stub(self):
        class _FakeAxes:
            def plot(self, *args, **kwargs):
                return None

            def axhline(self, *args, **kwargs):
                return None

            def set_ylim(self, *args, **kwargs):
                return None

            def set_title(self, *args, **kwargs):
                return None

            def set_xlabel(self, *args, **kwargs):
                return None

            def set_ylabel(self, *args, **kwargs):
                return None

            def grid(self, *args, **kwargs):
                return None

            def legend(self, *args, **kwargs):
                return None

            def axvspan(self, *args, **kwargs):
                return None

            def axvline(self, *args, **kwargs):
                return None

        class _FakeFigure:
            def suptitle(self, *args, **kwargs):
                return None

            def tight_layout(self, *args, **kwargs):
                return None

            def savefig(self, path, **kwargs):
                Path(path).write_bytes(b"fake-png")

        fake_matplotlib = types.ModuleType("matplotlib")
        fake_matplotlib.use = lambda *_args, **_kwargs: None

        fake_pyplot = types.ModuleType("matplotlib.pyplot")

        def _subplots(rows, cols, **kwargs):
            axes = [_FakeAxes() for _ in range(rows * cols)]
            return _FakeFigure(), axes

        fake_pyplot.subplots = _subplots
        fake_pyplot.close = lambda *_args, **_kwargs: None

        old_matplotlib = sys.modules.get("matplotlib")
        old_pyplot = sys.modules.get("matplotlib.pyplot")
        sys.modules["matplotlib"] = fake_matplotlib
        sys.modules["matplotlib.pyplot"] = fake_pyplot
        try:
            yield
        finally:
            if old_matplotlib is None:
                sys.modules.pop("matplotlib", None)
            else:
                sys.modules["matplotlib"] = old_matplotlib
            if old_pyplot is None:
                sys.modules.pop("matplotlib.pyplot", None)
            else:
                sys.modules["matplotlib.pyplot"] = old_pyplot

    def _write_log(self, log_dir: Path, records) -> Path:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "generation_samples.jsonl"
        with log_file.open("w", encoding="utf-8") as handle:
            for record in records:
                if isinstance(record, str):
                    handle.write(record + "\n")
                else:
                    handle.write(json.dumps(record) + "\n")
        return log_file

    def _write_progress_log(self, log_dir: Path, lines: list[str]) -> Path:
        progress_file = log_dir / "training_progress.log"
        with progress_file.open("w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")
        return progress_file

    def test_smoke_generates_dashboard_and_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "rl_output" / "sft"
            self._write_log(
                log_dir,
                [
                    _make_record(
                        reward=-0.2,
                        built_ok=False,
                        forward_shape_ok=False,
                        backward_ok=False,
                        loss_drop_ok=False,
                        train_acc=None,
                        seed_train_acc_gap=None,
                        group_baseline_train_acc=None,
                        group_train_acc_gain=None,
                        group_train_acc_improved=False,
                        reward_group_id=0,
                        group_warmup=True,
                        loss_start=None,
                        loss_end=None,
                        loss_drop=None,
                        val_metric=None,
                        warmup_dense_reward=None,
                    ),
                    _make_record(
                        reward=0.22,
                        train_acc=0.24,
                        seed_accuracy_baseline=0.10,
                        seed_train_acc_gap=0.14,
                        group_baseline_train_acc=None,
                        group_train_acc_gain=None,
                        group_train_acc_improved=False,
                        reward_group_id=0,
                        group_warmup=True,
                        warmup_dense_reward=0.22,
                    ),
                    _make_record(
                        reward=0.8,
                        train_acc=0.82,
                        seed_accuracy_baseline=0.68,
                        seed_train_acc_gap=0.14,
                        group_baseline_train_acc=0.70,
                        group_train_acc_gain=0.12,
                        group_train_acc_improved=True,
                        reward_group_id=1,
                        group_warmup=False,
                        estimated_total_seconds=180.0,
                    ),
                ],
            )
            self._write_progress_log(
                log_dir,
                [
                    "[12:47:38] progress: 10 generation，1 success，success rate 10.0% warmup_trainable_count=2 warmup_positive_count=1 timeout_count=0 improved_count=0",
                    "[12:59:10] progress: 20 generation，3 success，success rate 15.0% warmup_trainable_count=4 warmup_positive_count=2 timeout_count=1 improved_count=1",
                ],
            )

            output = io.StringIO()
            with self._matplotlib_stub(), contextlib.redirect_stdout(output):
                return_code = plot_rl_reward.main(["--log-dir", str(log_dir), "--window", "2"])

            self.assertEqual(return_code, 0)
            self.assertTrue((log_dir / "reward_dashboard.png").exists())
            stdout = output.getvalue()
            self.assertIn("Samples: 3", stdout)
            self.assertIn("Train accuracy:", stdout)
            self.assertIn("Training progress:", stdout)
            self.assertIn("Saved dashboard to:", stdout)

    def test_rolling_nanmean_handles_regular_and_nan_values(self):
        regular = plot_rl_reward.rolling_nanmean([1.0, 2.0, 3.0], window=2)
        self.assertEqual(len(regular), 3)
        self.assertAlmostEqual(regular[0], 1.0)
        self.assertAlmostEqual(regular[1], 1.5)
        self.assertAlmostEqual(regular[2], 2.5)

        with_nan = plot_rl_reward.rolling_nanmean([1.0, float("nan"), 3.0], window=2)
        self.assertAlmostEqual(with_nan[0], 1.0)
        self.assertAlmostEqual(with_nan[1], 1.0)
        self.assertAlmostEqual(with_nan[2], 3.0)

    def test_nan_values_are_kept_without_interpolation(self):
        series = [float("nan"), 2.0, float("nan")]
        rolled = plot_rl_reward.rolling_nanmean(series, window=2)
        self.assertTrue(math.isnan(rolled[0]))
        self.assertAlmostEqual(rolled[1], 2.0)
        self.assertAlmostEqual(rolled[2], 2.0)

    def test_summary_matches_current_sft_calculation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "rl_output" / "sft_summary"
            self._write_log(
                log_dir,
                [
                    _make_record(
                        reward=-0.5,
                        built_ok=False,
                        forward_shape_ok=False,
                        backward_ok=False,
                        loss_drop_ok=False,
                        timed_out=False,
                        train_acc=None,
                        seed_train_acc_gap=None,
                        group_baseline_train_acc=None,
                        group_train_acc_gain=None,
                        group_train_acc_improved=False,
                        reward_group_id=0,
                        group_warmup=True,
                        warmup_dense_reward=None,
                        anti_collapse_trainable_ok=None,
                    ),
                    _make_record(
                        reward=0.20,
                        timed_out=True,
                        train_acc=0.72,
                        seed_accuracy_baseline=0.68,
                        seed_train_acc_gap=0.04,
                        group_baseline_train_acc=None,
                        group_train_acc_gain=None,
                        group_train_acc_improved=False,
                        reward_group_id=0,
                        group_warmup=True,
                        estimated_total_seconds=301.0,
                        warmup_dense_reward=0.20,
                        anti_collapse_trainable_ok=True,
                    ),
                    _make_record(
                        reward=0.90,
                        timed_out=False,
                        train_acc=0.82,
                        seed_accuracy_baseline=0.70,
                        seed_train_acc_gap=0.12,
                        group_baseline_train_acc=0.60,
                        group_train_acc_gain=0.22,
                        group_train_acc_improved=True,
                        reward_group_id=1,
                        group_warmup=False,
                        estimated_total_seconds=180.0,
                        warmup_dense_reward=None,
                        anti_collapse_trainable_ok=True,
                    ),
                ],
            )
            self._write_progress_log(
                log_dir,
                [
                    "[12:59:10] progress: 20 generation，3 success，success rate 15.0% warmup_trainable_count=4 warmup_positive_count=2 timeout_count=1 improved_count=1",
                ],
            )

            data = plot_rl_reward.load_reward_log(log_dir)
            summary = plot_rl_reward.compute_summary(data, window=2)

            self.assertEqual(int(summary["count"]), 3)
            self.assertAlmostEqual(summary["reward_min"], -0.5)
            self.assertAlmostEqual(summary["reward_max"], 0.9)
            self.assertAlmostEqual(summary["reward_mean"], 0.2)
            self.assertAlmostEqual(summary["positive_reward_rate"], 2.0 / 3.0)
            self.assertAlmostEqual(summary["timed_out_rate"], 1.0 / 3.0)
            self.assertAlmostEqual(summary["train_acc_mean"], 0.77)
            self.assertAlmostEqual(summary["seed_accuracy_baseline_mean"], (0.68 + 0.68 + 0.70) / 3.0)
            self.assertAlmostEqual(summary["group_train_acc_gain_mean"], 0.22)
            self.assertAlmostEqual(summary["group_train_acc_improved_rate"], 1.0 / 3.0)
            self.assertAlmostEqual(summary["warmup_dense_reward_mean"], 0.20)
            self.assertAlmostEqual(summary["estimated_total_seconds_mean"], 240.5)
            self.assertAlmostEqual(summary["warmup_trainable_rate"], 0.5)
            self.assertAlmostEqual(summary["warmup_positive_rate"], 1.0)
            self.assertAlmostEqual(summary["anti_collapse_trainable_ok_rate"], 1.0)
            self.assertAlmostEqual(summary["progress_latest_success_rate"], 0.15)
            self.assertAlmostEqual(summary["progress_latest_timeout_count"], 1.0)
            self.assertAlmostEqual(summary["progress_latest_improved_count"], 1.0)

    def test_missing_optional_subobjects_are_supported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "rl_output" / "optional"
            self._write_log(
                log_dir,
                [
                    _make_record(
                        reward=0.5,
                        include_open_discovery=False,
                        include_raw_extraction=False,
                        include_anti_collapse=False,
                    ),
                ],
            )

            data = plot_rl_reward.load_reward_log(log_dir)
            summary = plot_rl_reward.compute_summary(data, window=2)

            self.assertEqual(data.count, 1)
            self.assertTrue(math.isnan(data.xml_tag_exact[0]))
            self.assertTrue(math.isnan(data.dual_backbone_ok[0]))
            self.assertTrue(math.isnan(data.format_violation[0]))
            self.assertTrue(math.isnan(data.anti_collapse_trainable_ok[0]))
            self.assertTrue(math.isnan(summary["xml_tag_exact_rate"]))
            self.assertTrue(math.isnan(summary["anti_collapse_trainable_ok_rate"]))

    def test_timeout_sample_is_tracked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "rl_output" / "timeout"
            self._write_log(
                log_dir,
                [
                    _make_record(
                        reward=0.0,
                        backward_ok=False,
                        loss_drop_ok=False,
                        timed_out=True,
                        estimated_total_seconds=312.5,
                        eval_limit_seconds=270,
                        warmup_dense_reward=None,
                    ),
                ],
            )

            data = plot_rl_reward.load_reward_log(log_dir)
            summary = plot_rl_reward.compute_summary(data, window=2)

            self.assertEqual(data.timed_out[0], 1.0)
            self.assertAlmostEqual(data.estimated_total_seconds[0], 312.5)
            self.assertAlmostEqual(summary["timed_out_rate"], 1.0)
            self.assertAlmostEqual(summary["estimated_total_seconds_mean"], 312.5)

    def test_forward_shape_ok_falls_back_to_forward_ok(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "rl_output" / "forward_fallback"
            record = _make_record(reward=0.25)
            del record["api_result"]["forward_shape_ok"]
            self._write_log(log_dir, [record])

            data = plot_rl_reward.load_reward_log(log_dir)
            summary = plot_rl_reward.compute_summary(data, window=2)

            self.assertEqual(data.forward_shape_ok[0], 1.0)
            self.assertAlmostEqual(summary["forward_shape_ok_rate"], 1.0)

    def test_missing_backward_and_timeout_fields_fall_back_safely(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "rl_output" / "legacy_fields"
            record = _make_record(reward=0.0, backward_ok=False, loss_drop_ok=False, timed_out=False)
            del record["api_result"]["backward_ok"]
            del record["api_result"]["loss_drop_ok"]
            del record["api_result"]["timed_out"]
            record["api_result"]["trained_step_ok"] = False
            self._write_log(log_dir, [record])

            data = plot_rl_reward.load_reward_log(log_dir)
            summary = plot_rl_reward.compute_summary(data, window=2)

            self.assertEqual(data.backward_ok[0], 0.0)
            self.assertEqual(data.loss_drop_ok[0], 1.0)
            self.assertEqual(data.timed_out[0], 0.0)
            self.assertAlmostEqual(summary["backward_ok_rate"], 0.0)
            self.assertAlmostEqual(summary["timed_out_rate"], 0.0)

    def test_missing_training_progress_log_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "rl_output" / "no_progress"
            self._write_log(log_dir, [_make_record(reward=0.1)])

            data = plot_rl_reward.load_reward_log(log_dir)
            summary = plot_rl_reward.compute_summary(data, window=2)

            self.assertEqual(data.progress_generation_total, [])
            self.assertTrue(math.isnan(summary["progress_latest_total"]))
            self.assertTrue(math.isnan(summary["progress_latest_success_rate"]))

    def test_generation_log_is_trimmed_to_current_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "rl_output" / "trimmed"
            self._write_log(
                log_dir,
                [
                    _make_record(reward=-0.9, train_acc=0.01),
                    _make_record(reward=-0.8, train_acc=0.02),
                    _make_record(reward=-0.7, train_acc=0.03),
                    _make_record(reward=0.4, train_acc=0.40),
                    _make_record(reward=0.6, train_acc=0.60),
                ],
            )
            self._write_progress_log(
                log_dir,
                [
                    "[13:00:01] Generation 1: Reward=0.4000",
                    "[13:00:02] Generation 2: Reward=0.6000",
                    "[13:00:10] progress: 2 generation，1 success，success rate 50.0% warmup_trainable_count=1 warmup_positive_count=1 timeout_count=0 improved_count=0",
                ],
            )

            data = plot_rl_reward.load_reward_log(log_dir)
            summary = plot_rl_reward.compute_summary(data, window=2)

            self.assertEqual(data.total_file_samples, 5)
            self.assertEqual(data.current_run_sample_count, 2)
            self.assertEqual(data.trimmed_stale_samples, 3)
            self.assertEqual(data.reward, [0.4, 0.6])
            self.assertEqual(data.sample_index, [1, 2])
            self.assertEqual(int(summary["count"]), 2)
            self.assertAlmostEqual(summary["trimmed_stale_samples"], 3.0)

    def test_missing_log_file_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_dir = Path(tmpdir) / "rl_output" / "missing"
            with self.assertRaisesRegex(FileNotFoundError, "generation_samples.jsonl"):
                plot_rl_reward.load_reward_log(missing_dir)

    def test_schema_validation_is_strict(self):
        cases = [
            ("missing api_result", [{"reward": 0.1}], "api_result"),
            ("nonnumeric reward", [{"reward": "bad", "api_result": {}}], "reward"),
            ("malformed json", ['{"reward": 0.1'], "invalid JSON"),
            (
                "invalid boolean field",
                [_make_record(reward=0.1)],
                "built_ok",
            ),
        ]

        for case_name, records, expected_pattern in cases:
            with self.subTest(case=case_name):
                with tempfile.TemporaryDirectory() as tmpdir:
                    log_dir = Path(tmpdir) / "rl_output" / "case"
                    if case_name == "invalid boolean field":
                        record = records[0]
                        record["api_result"]["built_ok"] = "bad"
                    self._write_log(log_dir, records)
                    with self.assertRaisesRegex(ValueError, expected_pattern):
                        plot_rl_reward.load_reward_log(log_dir)


if __name__ == "__main__":
    unittest.main()
