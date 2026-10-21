from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    import numpy as np
    import torch
except ImportError:  # pragma: no cover
    np = torch = None


@unittest.skipIf(torch is None, "Torch and numpy are required")
class QueryNumericsTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.device = torch.device("cpu")

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def model(self, fail=False):
        class Boundary(torch.nn.Module):
            def forward(self, value):
                return (value * float("inf") if fail else value, None)

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_items = 9
                self.item_embedding = torch.nn.Embedding(10, 4)
                self.dropout = torch.nn.Dropout(.4)
                self.boundary = Boundary()
                self.compressor = torch.nn.Identity()
                self.compressor.tail_size = 2
                self.register_buffer("calls", torch.zeros((), dtype=torch.long))

            def encode(self, ids, mask):
                self.calls.add_(1)
                value, _ = self.boundary(self.dropout(self.item_embedding(ids)))
                sources = torch.arange(ids.shape[1])[None].expand_as(ids)
                return value, mask, sources

        return Tiny().train()

    def batch(self):
        return {"item_ids": np.array([[1, 2, 3], [2, 3, 4]], dtype=np.int64),
                "valid_mask": np.ones((2, 3), dtype=bool),
                "targets": np.array([[2, 3, 4], [3, 4, 5]], dtype=np.int64),
                "negative_ids": np.array([[6, 7], [7, 8]], dtype=np.int64)}

    def hooks(self, model):
        return [(len(module._forward_hooks), len(module._forward_pre_hooks)) for module in model.modules()]

    def test_replay_locates_first_bad_nested_output_and_returns_finite_json(self):
        from rhpr.sasrec.query_numerics import capture_training_rng, diagnose_nonfinite_batch

        model = self.model(fail=True)
        before = self.hooks(model)
        report = diagnose_nonfinite_batch(model, self.batch(), self.device, amp=True,
                                         rng_state=capture_training_rng(self.device))
        self.assertTrue(report["parameters_finite"])
        for name in ("training_precision", "fp32"):
            replay = report[name]
            self.assertFalse(replay["loss_finite"])
            self.assertIsNone(replay["loss"])
            self.assertEqual(replay["first_nonfinite"]["module"], "boundary")
            self.assertEqual(replay["first_nonfinite"]["boundary"], "output")
            self.assertEqual(replay["first_nonfinite"]["tensor_path"], "[0]")
            self.assertGreater(replay["first_nonfinite"]["posinf"] + replay["first_nonfinite"]["neginf"], 0)
            self.assertFalse(replay["autocast_enabled"])  # Actual CUDA AMP is not emulated on CPU.
        self.assertEqual(self.hooks(model), before)
        self.assertNotIn("negative_ids\": [[", json.dumps(report, allow_nan=False))

    def test_exact_dropout_replay_preserves_all_rng_weights_gradients_buffers_and_modes(self):
        from rhpr.sasrec.query_numerics import capture_training_rng, diagnose_nonfinite_batch
        from rhpr.sasrec.long_training_profile import _loss

        model, arrays = self.model(), self.batch()
        model.boundary.eval()  # Mixed child modes must survive diagnostics too.
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.step()
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        batch = {key: torch.from_numpy(value) for key, value in arrays.items()}
        snapshot = capture_training_rng(self.device)
        expected = model.encode(batch["item_ids"], batch["valid_mask"])[0].detach()
        torch.rand(17)
        state = {key: value.clone() for key, value in model.state_dict().items()}
        gradients = [parameter.grad.clone() for parameter in model.parameters()]
        modes = [module.training for module in model.modules()]
        rng = torch.get_rng_state().clone()
        py_rng, np_rng = random.getstate(), np.random.get_state()
        seen = []

        def loss(*args):
            seen.append(args[1][0].clone())
            random.random()
            np.random.rand()
            return _loss(*args)

        with patch("rhpr.sasrec.query_numerics._loss", side_effect=loss):
            report = diagnose_nonfinite_batch(model, arrays, self.device, amp=False, rng_state=snapshot)
        self.assertTrue(report["training_precision"]["loss_finite"])
        self.assertTrue(report["fp32"]["loss_finite"])
        self.assertEqual(len(seen), 2)
        for value in seen:
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        for key, value in state.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        for parameter, gradient in zip(model.parameters(), gradients):
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
        self.assertEqual(modes, [module.training for module in model.modules()])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(py_rng, random.getstate())
        actual_np = np.random.get_state()
        self.assertEqual(np_rng[0], actual_np[0])
        np.testing.assert_array_equal(np_rng[1], actual_np[1])
        self.assertEqual(np_rng[2:], actual_np[2:])
        actual_optimizer = optimizer.state_dict()
        self.assertEqual(optimizer_state["param_groups"], actual_optimizer["param_groups"])
        for key, values in optimizer_state["state"].items():
            for name, value in values.items():
                torch.testing.assert_close(actual_optimizer["state"][key][name], value, rtol=0, atol=0)

    def test_precision_replays_are_separate_evidence_without_a_training_fallback(self):
        from rhpr.sasrec.query_numerics import diagnose_nonfinite_batch

        model = self.model()
        # Controlled loss outputs verify reporting/control flow, not CUDA arithmetic.
        with patch("rhpr.sasrec.query_numerics._loss", side_effect=[torch.tensor(float("nan")), torch.tensor(2.)]):
            report = diagnose_nonfinite_batch(model, self.batch(), self.device, amp=True)
        self.assertFalse(report["training_precision"]["loss_finite"])
        self.assertTrue(report["fp32"]["loss_finite"])
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["rng_reproduces_failed_forward"])
        self.assertEqual(int(model.calls), 0)

    def test_exception_reports_type_without_sensitive_text_and_removes_hooks(self):
        from rhpr.sasrec.query_numerics import capture_training_rng, diagnose_nonfinite_batch

        model = self.model()
        before, rng = self.hooks(model), capture_training_rng(self.device)
        with patch.object(model.boundary, "forward", side_effect=RuntimeError("private raw ID 123")):
            report = diagnose_nonfinite_batch(model, self.batch(), self.device, amp=False, rng_state=rng)
        self.assertEqual(report["training_precision"]["error_type"], "RuntimeError")
        self.assertEqual(report["fp32"]["error_type"], "RuntimeError")
        self.assertNotIn("123", json.dumps(report, allow_nan=False))
        self.assertEqual(before, self.hooks(model))
        self.assertTrue(torch.equal(rng["cpu"], torch.get_rng_state()))
        self.assertEqual(int(model.calls), 0)

    def test_parameter_nonfinites_counted_without_nonfinite_json_scalars(self):
        from rhpr.sasrec.query_numerics import diagnose_nonfinite_batch

        model = self.model()
        with torch.no_grad():
            model.item_embedding.weight[1, :3] = torch.tensor([float("nan"), float("inf"), -float("inf")])
        report = diagnose_nonfinite_batch(model, self.batch(), self.device, amp=False)
        row = report["parameters"]["item_embedding.weight"]
        self.assertEqual((row["nan"], row["posinf"], row["neginf"]), (1, 1, 1))
        self.assertFalse(report["parameters_finite"])
        self.assertEqual(report["training_precision"]["first_nonfinite"]["module"], "item_embedding")
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
