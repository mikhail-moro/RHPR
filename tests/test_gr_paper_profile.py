"""Meaningful numerical, rollback and no-quality contracts for runtime profiling."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rhpr.gr.paper_profile import (
    PaperProfileBudget, benchmark_paper_generation, benchmark_paper_training,
    embedding_agreement, masked_mean, profile_encoder, profile_paper_operators,
    runtime_estimates, runtime_sid_index, validate_paper_profile,
)
from rhpr.gr.profiling import ProfileBudgetExhausted

_HAS_TORCH = all(importlib.util.find_spec(name) for name in ("torch", "numpy"))
_SOURCE = os.environ.get("RHPR_GRID_SOURCE")
_GRID_DEPS = ("torch", "numpy", "transformers", "lightning", "pytorch_lightning", "torchmetrics", "pyarrow", "hydra")
_CAN_INTEGRATE = bool(_SOURCE) and all(importlib.util.find_spec(name) for name in _GRID_DEPS)


class PaperProfileContracts(unittest.TestCase):
    def test_budget_and_registration_cannot_turn_profile_into_quality_training(self):
        ticks = iter([10, 11, 13])
        deadline = PaperProfileBudget(3, clock=lambda: next(ticks))
        deadline.check()
        with self.assertRaises(ProfileBudgetExhausted):
            deadline.check()
        valid = {"profile": {}, "training": {"batch_size": 256}}
        self.assertEqual(validate_paper_profile(valid)["max_seconds"], 900)
        self.assertEqual(validate_paper_profile(valid)["precisions"][0], "fp32")
        for change in ({"evaluation": {"test_after_selection": True}},
                       {"parent": {"kernel": "old/hybrid"}},
                       {"training": {"batch_size": 512}},
                       {"profile": {"encoder_batch_sizes": [8, 8]}},
                       {"profile": {"max_seconds": float("inf")}},
                       {"profile": {"precisions": ["fp16", "fp16"]}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_paper_profile({**valid, **change})

    @unittest.skipUnless(_HAS_TORCH, "Install torch/numpy for numeric contracts")
    def test_runtime_identifiers_are_injective_and_explicitly_untrained(self):
        import numpy as np
        a, b = runtime_sid_index(120, 8, 42), runtime_sid_index(120, 8, 42)
        np.testing.assert_array_equal(a.sids, b.sids)
        self.assertEqual(a.sids.shape, (120, 4))
        self.assertEqual(len(np.unique(a.sids, axis=0)), 120)
        self.assertTrue((a.sids[:, -1] == 0).all())
        with self.assertRaises(ValueError):
            runtime_sid_index(9, 2, 42)

    @unittest.skipUnless(_HAS_TORCH, "Install torch/numpy for numeric contracts")
    def test_pooling_includes_eos_excludes_padding_without_normalization(self):
        import torch
        states = torch.tensor([[[2., 4.], [6., 8.], [999., 999.]]])
        torch.testing.assert_close(masked_mean(states, torch.tensor([[1, 1, 0]])), torch.tensor([[4., 6.]]))
        self.assertEqual(embedding_agreement(states[:, 0], states[:, 0])["maximum_absolute_difference"], 0)
        with self.assertRaises(FloatingPointError):
            embedding_agreement(states[:, 0], torch.tensor([[float("nan"), 0.]]))

    def test_estimates_count_updates_and_full_validation_not_invented_epochs(self):
        report = {"counts": {"users": 2000},
                  "generation_cases": [{"status": "ok", "precision": "fp32", "seconds": 2., "users": 100}],
                  "training_cases": [{"status": "ok", "mode": "full_pipeline", "precision": "fp32",
                     "paper_effective_batch_preserved": True, "physical_batch_size": 128, "seconds_per_update": .5}]}
        result = runtime_estimates(report, {"training": {"validation_interval_updates": 100}})
        self.assertEqual([x["optimizer_updates"] for x in result], [100, 1000, 10000])
        self.assertEqual(result[0]["estimated_train_plus_validation_seconds"], 90)
        self.assertEqual(result[0]["training_examples"], 25600)
        self.assertNotIn("epochs", result[0])


@unittest.skipUnless(_CAN_INTEGRATE, "Set RHPR_GRID_SOURCE and install GR dependencies")
class PaperProfileActualGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from rhpr.gr.grid import load_grid
        cls.torch, cls.old_threads = torch, torch.get_num_threads()
        torch.set_num_threads(1)
        load_grid(Path(_SOURCE))

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.old_threads)

    def setUp(self):
        from rhpr.gr.data import UserSequence
        from rhpr.gr.grid import build_grid_generator
        from rhpr.gr.paper_protocol import PaperBatchStream
        self.index = runtime_sid_index(10, 4, 42)
        self.users = [UserSequence(i, tuple((i + j) % 10 for j in range(5)), (i + 5) % 10, -1)
                      for i in range(5)]
        self.model = build_grid_generator(self.index.sids, {
            "codebook_size": 4, "beam_width": 3, "d_model": 8, "d_ff": 16,
            "num_heads": 1, "d_kv": 8, "encoder_layers": 1, "decoder_layers": 1,
            "num_user_bins": None, "separator_token": True, "mlp_layers": 2, "dropout": .1,
        }, source_root=Path(_SOURCE))
        self.factory = lambda effective: PaperBatchStream(self.users, self.index.sids,
            seed=42, source_batch_size=2, effective_batch_size=effective,
            sequence_length=24, max_augmented_batch=8)
        self.training = {"batch_size": 8, "learning_rate": 5e-4,
                         "weight_decay": 1e-6, "grad_clip_norm": None}
        self.specification = {"warmup": 1, "iterations": 2, "seed": 42}

    def test_real_author_batches_backward_and_weights_rng_mode_rollback(self):
        torch = self.torch
        initial = {key: value.clone() for key, value in self.model.state_dict().items()}
        self.model.eval()
        rng = torch.get_rng_state().clone()
        for mode in ("full_pipeline", "gpu_resident"):
            result = benchmark_paper_training(self.model, self.factory, 4, "fp32", mode,
                self.training, self.specification, PaperProfileBudget(30), "cpu")
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["measured_updates_applied"], 2)
            self.assertEqual(result["sampling"]["sid_tokens_including_target"], 24)
            self.assertGreater(result["seconds_per_update"], 0)
            self.assertFalse(self.model.training)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            for key, value in self.model.state_dict().items():
                torch.testing.assert_close(value, initial[key], rtol=0, atol=0)

    def test_meta_encoder_loader_restores_exact_fp32_weights_and_alias(self):
        from transformers import T5Config, T5EncoderModel
        from rhpr.gr.paper_profile import _load_encoder
        torch = self.torch
        original = T5EncoderModel(T5Config(vocab_size=8, d_model=8, d_ff=16, d_kv=8,
                                          num_heads=1, num_layers=1))
        with tempfile.TemporaryDirectory() as tmp:
            original.save_pretrained(tmp)
            restored = _load_encoder(tmp, "cpu")
            for key, value in original.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
            self.assertIs(restored.shared.weight, restored.encoder.embed_tokens.weight)
            self.assertFalse(restored.training)
            original.half().save_pretrained(tmp)
            with self.assertRaisesRegex(ValueError, "remain FP32"):
                _load_encoder(tmp, "cpu")

    def test_completed_encoder_cases_survive_later_budget_exhaustion(self):
        torch = self.torch
        class SmallEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
            def forward(self, input_ids, attention_mask):
                shape = (*input_ids.shape, 2)
                return SimpleNamespace(last_hidden_state=torch.ones(shape) * self.weight)
        def tokenizer(texts, **options):
            return {"input_ids": torch.ones((len(texts), 8), dtype=torch.long),
                    "attention_mask": torch.ones((len(texts), 8), dtype=torch.long)}
        progress = []
        ticks = iter([0, 0, 0, 0, 3])
        with tempfile.TemporaryDirectory() as tmp, patch(
            "rhpr.gr.resources.verify_resources"
        ), patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), patch(
            "rhpr.gr.paper_profile._load_encoder", return_value=SmallEncoder()
        ), self.assertRaises(ProfileBudgetExhausted):
            profile_encoder({"seed": 42, "representation": {"path": tmp,
                "manifest_sha256": "fixture", "model": "fixture", "revision": "fixture", "max_length": 8}},
                ["item"] * 4, "cpu", {"encoder_items": 4, "encoder_batch_sizes": [1, 2],
                "encoder_precisions": ["fp32"], "warmup": 1, "iterations": 1},
                PaperProfileBudget(3, clock=lambda: next(ticks)),
                progress=lambda row: progress.append(copy.deepcopy(row)))
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0]["cases"][0]["status"], "ok")

    def test_nonfinite_update_restores_model_instead_of_reporting_fast_success(self):
        torch = self.torch
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        def invalid_loss(model, batch):
            return next(model.parameters()).sum() * float("nan")
        with patch("rhpr.gr.experiment._loss", side_effect=invalid_loss), self.assertRaises(FloatingPointError):
            benchmark_paper_training(self.model, self.factory, 4, "fp32", "full_pipeline",
                self.training, self.specification, PaperProfileBudget(30), "cpu")
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_beam_runtime_has_no_retrieval_quality_or_test_use(self):
        from rhpr.gr.data import build_examples
        examples = build_examples(self.users, 3, "validation")
        row, ids = benchmark_paper_generation(self.model, examples, self.index.sids, 2, "fp32",
            "cpu", self.specification, PaperProfileBudget(30))
        self.assertEqual(ids.shape, (5, 3, 4))
        self.assertEqual(row["status"], "ok")
        self.assertNotIn("ndcg@10", row)
        self.assertNotIn("recall@10", row)

    def test_operator_instrumentation_restores_state_and_reports_separate_timing(self):
        torch = self.torch
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        self.model.eval()
        rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as tmp:
            row = profile_paper_operators(self.model, self.factory, self.training,
                self.specification, PaperProfileBudget(30), "cpu", Path(tmp))
            self.assertEqual(row["status"], "ok")
            self.assertIn("cpu_author_augmentation_seconds", row["synchronized_stage_timing"])
            self.assertGreater(row["recognized_flops_per_update"], 0)
            self.assertTrue((Path(tmp) / "grid-paper-operators.txt").is_file())
        self.assertFalse(self.model.training)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
