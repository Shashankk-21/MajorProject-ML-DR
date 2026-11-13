"""Deterministic, CPU-safe smoke tests for the StepZero diagnostic modules."""

from __future__ import annotations

import contextlib
import csv
import importlib
import io
import math
import py_compile
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEPZERO = ROOT / "notebooks" / "stepzero"
for path in (ROOT, STEPZERO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


Z, R, G, T = tuple(importlib.import_module(name) for name in
                   ("stepzero_lib", "rank_sweep", "grad_norm_probe", "timing_probe"))


class StepZeroTests(unittest.TestCase):
    def assertRaisesSystemExit(self, function):
        with self.assertRaises(SystemExit):
            function()

    def test_imports_compile_without_source_outputs(self):
        before = {p for p in STEPZERO.rglob("*") if p.is_file()}
        with tempfile.TemporaryDirectory() as temp:
            for name in ("stepzero_lib.py", "rank_sweep.py", "grad_norm_probe.py", "timing_probe.py"):
                py_compile.compile(str(STEPZERO / name), cfile=str(Path(temp) / (name + "c")), doraise=True)
        self.assertEqual(before, {p for p in STEPZERO.rglob("*") if p.is_file()})
        for module in (Z, R, G, T):
            self.assertEqual(Path(module.__file__).resolve().parent, STEPZERO)

    def test_rank_selection_and_validation(self):
        result = Z.select_rank({4: {"val_macro_f1": .79, "qwk": .80},
                                8: {"val_macro_f1": .81, "qwk": .81},
                                16: {"val_macro_f1": .805, "qwk": .70}})
        self.assertEqual(result["chosen"], 8)
        tied = Z.select_rank({4: {"val_macro_f1": .79, "qwk": .8},
                              8: {"val_macro_f1": .805, "qwk": .8},
                              16: {"val_macro_f1": .81, "qwk": .8}}, tie=.01)
        self.assertEqual((tied["tie_set"], tied["chosen"]), ([8, 16], 8))
        flagged = Z.select_rank({4: {"val_macro_f1": .80, "qwk": .70},
                                 8: {"val_macro_f1": .81, "qwk": .75}}, qwk_flag=.02)
        self.assertTrue(flagged["manual_review"])
        okay = Z.select_rank({4: {"val_macro_f1": .80, "qwk": .73},
                              8: {"val_macro_f1": .81, "qwk": .75}}, qwk_flag=.02)
        self.assertFalse(okay["manual_review"])
        for values, kwargs in [
            ({}, {}), ({0: {"val_macro_f1": .8, "qwk": .8}}, {}),
            ({1: {"val_macro_f1": math.nan, "qwk": .8}}, {}),
            ({1: {"val_macro_f1": .8, "qwk": .8}}, {"tie": -1}),
            ({1: {"val_macro_f1": .8, "qwk": .8}}, {"qwk_flag": math.inf}),
        ]:
            with self.subTest(values=values, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    Z.select_rank(values, **kwargs)

    def test_gradient_helpers(self):
        norms = np.array([1., 2., 3., 4., 5.])
        self.assertEqual(Z.percentile_summary(norms),
                         {"n": 5, "p25": 2., "p50": 3., "p75": 4., "p95": 4.8, "max": 5.})
        self.assertEqual(Z.fraction_clipped(norms, 3.), .4)
        self.assertAlmostEqual(Z.geometric_mean([1., 4., 16.]), 4.)
        rec = Z.recommend_max_grad_norm({"a": 2., "b": 8.}, flag_ratio=3.)
        self.assertEqual((rec["value"], rec["ratio"], rec["flag"]), (4., 4., True))
        self.assertEqual(Z.phase_ratio(2., 4.), {"ratio": 2., "flag": False})
        self.assertEqual(Z.round_sig(12345., 3), 12300.)
        self.assertRaises(ValueError, Z.percentile_summary, [])
        self.assertRaises(ValueError, Z.fraction_clipped, [1.], 0)

    def test_flat_per_sample_norms_numpy_and_tensor(self):
        a, b = np.array([[3., 4.], [0., 1.]]), np.array([[12.], [2.]])
        np.testing.assert_allclose(Z.flat_per_sample_norms([a, b]), [13., math.sqrt(5.)])
        with self.assertRaises(ValueError):
            Z.flat_per_sample_norms([])
        with self.assertRaises(ValueError):
            Z.flat_per_sample_norms([np.ones((2, 1)), np.ones((3, 1))])
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable; tensor path is not applicable")
        np.testing.assert_allclose(Z.flat_per_sample_norms([torch.tensor(a), torch.tensor(b)]),
                                   [13., math.sqrt(5.)])

    def test_timing_math_and_formatting(self):
        self.assertEqual(Z.epoch_seconds_nondp(10, 4, 1.5), 4.5)
        n, logical, physical = 5, 3, 2
        q = 1 / math.ceil(n / logical)
        expected = sum(math.comb(n, k) * q ** k * (1 - q) ** (n - k) * math.ceil(k / physical)
                       for k in range(n + 1))
        self.assertAlmostEqual(Z.expected_micro_batches_dp(n, logical, physical),
                               math.ceil(n / logical) * expected)
        self.assertAlmostEqual(Z.epoch_seconds_dp(n, logical, physical, 2.),
                               2 * Z.expected_micro_batches_dp(n, logical, physical))
        self.assertEqual(Z.largest_fitting({16: "ok", 32: "OOM", 48: "ok"}), 48)
        self.assertEqual(Z.fmt_hms(3661.4), "1:01:01")
        self.assertEqual(Z.format_table(["a", "b"], [[1, "x"]]), "a  b\n-  -\n1  x")

    def test_json_helpers(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nested" / "result.json"
            Z.atomic_write_json(path, {"answer": 42, "values": [1, 2]})
            self.assertEqual(Z.read_json(path), {"answer": 42, "values": [1, 2]})
            self.assertIsNone(Z.read_json(path.with_name("missing.json")))
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_rank_args_schedule_best_epoch_and_guard(self):
        args = R.parse_args(["--manifest-root", "m", "--checkpoint-root", "c"])
        self.assertEqual((args.ranks, args.wandb), ([4, 8, 16, 32], False))
        self.assertRaisesSystemExit(lambda: R.parse_args(["--manifest-root", "m", "--checkpoint-root", "c",
                                                           "--ranks", "4", "4"]))
        self.assertRaisesSystemExit(lambda: R.parse_args(["--manifest-root", "m", "--checkpoint-root", "c",
                                                           "--ranks", "0"]))
        cfg = {"lora-rank": 99, "lora-alpha": 1}
        base_kwargs = {"client_manifest_paths": ["x"], "checkpoint_dir": "old", "other": "kept"}
        with mock.patch("configs.loader.to_centralized_kwargs", return_value=base_kwargs):
            kwargs = R.build_kwargs(cfg, 8, Path("rank8"))
        self.assertEqual((kwargs["checkpoint_dir"], kwargs["max_epochs"], kwargs["warmup_epochs"],
                          kwargs["patience"], kwargs["seed"]), ("rank8", 4, .2, 99, 42))
        self.assertEqual(R._params_of({"client_manifest_paths": [], "checkpoint_dir": "x", "a": 1}), {"a": 1})
        state = {"epochs": {"1": {"val_macro_f1": .8, "qwk": .7}, "2": {"val_macro_f1": .8, "qwk": .7}}}
        self.assertEqual(R.best_epoch(state)["epoch"], 1)
        state["result"] = {"best_val_macro_f1": .8}
        R._check_best_recorded(state, 4, Path("state.json"), Path("rank4"))
        state["result"]["best_val_macro_f1"] = .9
        with self.assertRaises(SystemExit):
            R._check_best_recorded(state, 4, Path("state.json"), Path("rank4"))

    def test_rank_run_mocked_integration_resume_and_config_guard(self):
        cfg = {"lora-rank": 8, "lora-alpha": 16}
        kwargs = {
            "client_manifest_paths": ["manifest"],
            "checkpoint_dir": "ignored",
            "setting": "v"
        }
        calls = []

        def fake_trainer(**received):
            calls.append(received)
            callback = received["on_epoch_end"]

            for epoch, f1 in ((1, .70), (2, .90)):
                callback(epoch, {
                    "train_loss": 1.,
                    "val_loss": .4,
                    "val_macro_f1": f1,
                    "lr_lora": .01,
                    "lr_head": .02,
                    "val_confusion_matrix": [[2, 0], [0, 2]]
                })

            return {
                "best_val_macro_f1": .90,
                "final_epoch": 2,
                "best_checkpoint_path": "best.pt"
            }

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)

            trainer_module = types.ModuleType("training.train_centralized")
            trainer_module.run_centralized_training = fake_trainer

            tracker_module = types.ModuleType("utils.experiment_tracker")
            tracker_module.quadratic_weighted_kappa = lambda _matrix: .75

            with mock.patch.dict(sys.modules, {
                "training.train_centralized": trainer_module,
                "utils.experiment_tracker": tracker_module,
            }):
                with mock.patch(
                    "configs.loader.to_centralized_kwargs",
                    side_effect=lambda *_args, **_kwargs: dict(kwargs)
                ):
                    state = R.run_rank(8, cfg, output, False)
                    self.assertEqual(state["result"]["final_epoch"], 2)
                    self.assertEqual(state["epochs"]["2"]["qwk"], .75)
                    self.assertEqual(R.best_epoch(state)["epoch"], 2)

                    state2 = R.run_rank(8, cfg, output, False)
                    self.assertEqual(state2["result"]["final_epoch"], 2)

                self.assertEqual(len(calls), 1)

                with mock.patch(
                    "configs.loader.to_centralized_kwargs",
                    side_effect=lambda *_args, **_kwargs: dict(kwargs, setting="changed")
                ):
                    with self.assertRaises(SystemExit):
                        R.run_rank(8, cfg, output, False)

                self.assertEqual(len(calls), 1)

    def test_grad_args_summary_and_report(self):
        args = G.parse_args(["--manifest-root", "m", "--checkpoint-root", "c"])
        self.assertEqual((args.batches, args.phases), (20, ["init", "epoch1"]))
        self.assertRaisesSystemExit(lambda: G.parse_args(["--manifest-root", "m", "--checkpoint-root", "c",
                                                           "--batches", "0"]))
        self.assertRaisesSystemExit(lambda: G.parse_args(["--manifest-root", "m", "--checkpoint-root", "c",
                                                           "--batch-size", "1"]))
        phase = G.summarize_phase({"a": [1., 2., 3.], "b": [4., 8., 16.]})
        self.assertEqual(phase["summary"]["POOLED"]["n"], 6)
        self.assertAlmostEqual(phase["recommendation"]["value"], 4.)
        self.assertIn(phase["recommendation"]["value_rounded"], phase["candidates"])
        expected_candidates = sorted(set(G.CANDIDATES) | {phase["recommendation"]["value_rounded"]})
        self.assertEqual(phase["candidates"], expected_candidates)
        self.assertEqual(phase["clipped"]["a"]["2.0"], 1 / 3)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            G.report({"phases": {"epoch1": {"clients": {"a": {"norms": [1., 2., 3.]}}}}})
        self.assertIn("Phase 'epoch1'", output.getvalue())
        self.assertIn("max-grad-norm:", output.getvalue())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            G.report({"phases": {"init": {"clients": {"a": {"norms": [1., 2.]}}}}})
        self.assertIn("not measured: no value to paste yet", output.getvalue())

    def test_timing_args_sizes_stamped_errors_and_report(self):
        args = T.parse_args(["--manifest-root", "m", "--checkpoint-root", "c"])
        self.assertEqual((args.steps, args.warmup_steps, args.lora_rank, args.lora_alpha), (20, 3, 8, 16))
        for extra in (["--steps", "0"], ["--batch-sizes", "0"], ["--lora-rank", "0"]):
            self.assertRaisesSystemExit(lambda extra=extra: T.parse_args(
                ["--manifest-root", "m", "--checkpoint-root", "c", *extra]))
        self.assertEqual([T.dp_size_is_representative(64, b) for b in (16, 32, 48, 64)],
                         [True, True, False, True])
        stamps, syncs = [], []
        with mock.patch.object(T.time, "perf_counter", side_effect=[1., 2., 3.]):
            self.assertEqual(list(T._stamped(["a", "b", "c"], stamps, 2, lambda: syncs.append(1))), ["a", "b"])
        self.assertEqual((stamps, len(syncs)), ([1., 2.], 2))
        with mock.patch.object(T, "measure_config", side_effect=RuntimeError("CUDA out of memory")), \
             mock.patch.object(T, "_free_gpu"):
            self.assertEqual(T.run_one("nondp", 8)["status"], "OOM")
        with mock.patch.object(T, "measure_config", side_effect=RuntimeError("unrelated")), \
             mock.patch.object(T, "_free_gpu"):
            with self.assertRaisesRegex(RuntimeError, "unrelated"):
                T.run_one("nondp", 8)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["split"])
                writer.writeheader()
                writer.writerows([{"split": "train"}, {"split": "val"}, {"split": "train"}])
            self.assertEqual(T.train_sizes({"client": path}), {"client": 2})
        store = {"results": {
            "nondp-16": {"mode": "nondp", "physical_batch": 16, "status": "ok",
                         "seconds_per_step": .1, "peak_mem_gib": 1., "accumulation_steps": 1},
            "dp-16": {"mode": "dp", "physical_batch": 16, "status": "skipped"}}}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            T.report(store, {"a": 64}, ["a"], 1, 64, "cpu")
        self.assertIn("nondp: 16", output.getvalue())
        self.assertEqual(Z.largest_fitting({16: "ok", 32: "OOM"}), 16)


class PassFailResult(unittest.TextTestResult):
    def addSuccess(self, test):
        super().addSuccess(test)
        print(f"PASS: {test.id()}")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        print(f"FAIL: {test.id()}")

    def addError(self, test, err):
        super().addError(test, err)
        print(f"FAIL: {test.id()}")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        print(f"SKIP: {test.id()} ({reason})")


class PassFailRunner(unittest.TextTestRunner):
    resultclass = PassFailResult


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(StepZeroTests)
    result = PassFailRunner(verbosity=0, stream=sys.stderr).run(suite)
    skipped = len(result.skipped)
    passed = result.testsRun - len(result.failures) - len(result.errors) - skipped
    print(f"{passed}/{result.testsRun} passed, {skipped} skipped")
    sys.exit(0 if result.wasSuccessful() else 1)
