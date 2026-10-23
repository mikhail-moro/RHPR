"""Finite augmentation epochs, sparse validation, and truthful TPU diagnostics."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

_SOURCE = os.environ.get("RHPR_GRID_SOURCE")
_CAN = bool(_SOURCE) and all(importlib.util.find_spec(name) for name in
    ("torch", "numpy", "transformers", "lightning", "pyarrow", "hydra"))


@unittest.skipUnless(_CAN, "Requires pinned GRID source and GR dependencies")
class GridEpochIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from rhpr.gr.grid import load_grid
        cls.np, cls.torch = np, torch
        load_grid(_SOURCE)
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.old_threads)

    def fixture(self):
        from rhpr.gr.data import UserSequence
        from rhpr.gr.grid import build_grid_generator
        from rhpr.gr.sid import SidIndex
        index = SidIndex(self.np.array([[i, 0, 0] for i in range(4)]), width=4)
        users = [UserSequence(i, (i, (i + 1) % 4, (i + 2) % 4), 0, 1) for i in range(4)]
        model = build_grid_generator(index.sids, {"codebook_size": 4, "beam_width": 2,
            "d_model": 8, "d_ff": 16, "d_kv": 4, "num_heads": 2,
            "encoder_layers": 1, "decoder_layers": 1, "dropout": 0.,
            "mlp_layers": 2, "separator_token": True, "num_user_bins": None},
            seed=42, source_root=_SOURCE)
        config = {"seed": 42, "data": {"history_items": 2}, "training": {
            "schedule": "epochs", "max_epochs": 30, "validation_interval_epochs": 10,
            "patience_epochs": 10, "precision": "fp32", "evaluation_precision": "fp32",
            "evaluation_batch_size": 4, "learning_rate": 0.001, "weight_decay": 1e-6,
            "batch_size": 4, "physical_batch_size": 2, "source_user_batch_size": 2,
            "sequence_length": 12, "max_augmented_batch": 128, "grad_clip_norm": None,
            "max_seconds": 60}}
        return model, users, index, config

    def test_finite_epoch_matches_source_collator_and_retains_both_remainders(self):
        from rhpr.gr.paper_protocol import PaperEpochStream, collate_paper_rows
        from rhpr.gr.data import UserSequence
        _, _, index, _ = self.fixture()
        users = [UserSequence(10, (0, 1), 0, 0), UserSequence(11, (0, 1, 2), 0, 0),
                 UserSequence(12, (0, 1, 2, 3), 0, 0)]
        torch, np = self.torch, self.np
        seed = 17
        stream = PaperEpochStream(users, index.sids, seed=seed, source_batch_size=2,
            effective_batch_size=4, sequence_length=12, max_augmented_batch=128)
        before = torch.get_rng_state().clone()
        batches = list(stream)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        self.assertEqual([len(batch["future_ids"]) for batch in batches], [4, 4, 2])
        joined = {name: torch.cat([batch[name] for batch in batches]) for name in batches[0]}
        order = np.random.default_rng(seed).permutation(3)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            direct = [collate_paper_rows(users, index.sids, order[begin:begin + 2],
                sequence_length=12, max_augmented_batch=128) for begin in (0, 2)]
        for name in joined:
            torch.testing.assert_close(joined[name], torch.cat([part[name] for part in direct]), rtol=0, atol=0)
        report = stream.report()
        self.assertEqual(report["source_rows_consumed"], 3)
        self.assertEqual(report["model_examples_yielded"], 10)
        self.assertTrue(report["epoch_complete"])
        self.assertEqual(report["buffered_examples"], 0)
        with self.assertRaises(StopIteration):
            next(stream)

    def test_ten_epoch_validation_and_epoch_patience_restore_recall_selected_state(self):
        from rhpr.gr.paper_experiment import train_paper_generator
        model, users, index, config = self.fixture()
        captured = []
        def evaluate(model, *args, **kwargs):
            captured.append({name: value.detach().clone() for name, value in model.state_dict().items()})
            recall, ndcg = ((0.3, 0.4), (0.2, 0.3))[len(captured) - 1]
            return {"recall@10": recall, "ndcg@10": ndcg}, None, None
        with tempfile.TemporaryDirectory() as tmp:
            result = train_paper_generator(model, users, index, config, Path(tmp), "cpu", evaluator=evaluate)
            curves = json.loads((Path(tmp) / "learning-curves.json").read_text())
            self.assertEqual([row["epoch"] for row in curves if "validation" in row], [10, 20])
            self.assertEqual(result["epochs"], 20)
            self.assertEqual(result["updates"], 60)
            self.assertEqual(result["model_examples_consumed"], 240)
            self.assertEqual(result["selected_epoch"], 10)
            self.assertEqual(result["stop_reason"], "early_stopping")
            for name, value in model.state_dict().items():
                self.torch.testing.assert_close(value, captured[0][name], rtol=0, atol=0)

    def test_budget_stops_at_complete_epoch_with_one_explicit_final_check(self):
        from rhpr.gr.paper_experiment import train_paper_generator
        model, users, index, config = self.fixture()
        def evaluate(*args, **kwargs):
            return {"recall@10": 0.2, "ndcg@10": 0.1}, None, None
        with tempfile.TemporaryDirectory() as tmp:
            result = train_paper_generator(model, users, index, config, Path(tmp), "cpu",
                max_seconds=0, evaluator=evaluate)
            self.assertEqual(result["epochs"], 1)
            self.assertEqual(result["updates"], 3)
            self.assertEqual(result["stop_reason"], "time_budget")
            curves = json.loads((Path(tmp) / "learning-curves.json").read_text())
            self.assertEqual(curves[0]["validation_trigger"], "final_budget_or_epoch_boundary")

    def test_tpu_probe_rejects_cpu_xla_instead_of_claiming_hardware_compatibility(self):
        from rhpr.gr.tpu_probe import run_tpu_probe
        xla, core, xm, runtime = [ModuleType(name) for name in
            ("torch_xla", "torch_xla.core", "torch_xla.core.xla_model", "torch_xla.runtime")]
        xla.__version__ = "fake"
        xla.core, xla.runtime, core.xla_model = core, runtime, xm
        runtime.device_type = lambda: "CPU"
        modules = {"torch_xla": xla, "torch_xla.core": core,
                   "torch_xla.core.xla_model": xm, "torch_xla.runtime": runtime}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ), patch.dict("sys.modules", modules):
            result = run_tpu_probe({"name": "probe"}, Path(tmp))
            report = json.loads((Path(tmp) / "tpu-compatibility.json").read_text())
        self.assertFalse(result["generator_tpu_compatible"])
        self.assertEqual(result["last_stage"], "native_xla_import")
        self.assertIn("actual TPU hardware", report["failure"]["message"])


if __name__ == "__main__":
    unittest.main()
