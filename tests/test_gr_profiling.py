"""Bounded, validation-only profiling and reversible real training updates."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

from rhpr.gr.data import file_sha256
from rhpr.gr.profiling import (
    ProfileBudgetExhausted, ProfileDeadline, _profile_data,
    benchmark_evaluation, benchmark_training, detailed_training_profile,
    precision_supported, prediction_agreement, validate_profile, verified_parent,
)

_HAS_TORCH = all(importlib.util.find_spec(name) for name in ("torch", "numpy"))
_SOURCE = os.environ.get("RHPR_GRID_SOURCE")
_GRID_DEPS = ("torch", "numpy", "transformers", "lightning", "pytorch_lightning", "torchmetrics", "pyarrow", "hydra")
_CAN_INTEGRATE = bool(_SOURCE) and all(importlib.util.find_spec(name) for name in _GRID_DEPS)


class ProfileContractTests(unittest.TestCase):
    def test_deadline_stops_before_more_work_and_rejects_unbounded_budget(self):
        ticks = iter([100, 101, 103])
        budget = ProfileDeadline(3, clock=lambda: next(ticks))
        budget.check()
        with self.assertRaises(ProfileBudgetExhausted):
            budget.check()
        for seconds in (0, -1, float("nan"), float("inf"), 601):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                ProfileDeadline(seconds)

    def test_profile_requires_reference_and_existing_architecture_without_test(self):
        result = validate_profile({"profile": {"precisions": ["fp16"]}})
        self.assertEqual(result["precisions"], ["fp32", "fp16"])
        self.assertEqual(result["max_seconds"], 480)
        for config in (
            {"profile": {}, "evaluation": {"test_after_selection": True}},
            {"profile": {"architectures": [{"use_parent_checkpoint": False}]}},
            {"profile": {"train_batch_sizes": [128, 128]}},
            {"profile": {"train_batch_sizes": [True]}},
            {"profile": {"evaluation_batch_sizes": [[256]]}},
            {"profile": {"precisions": ["fp16", "fp16"]}},
            {"profile": {"precisions": [["fp16"]]}},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_profile(config)

    def parent_fixture(self, directory):
        for name, content in {
            "generator-best.pt": b"checkpoint bytes",
            "semantic-ids.npy": b"saved SID bytes",
            "run-manifest.json": b'{"git_sha":"pinned-source"}',
        }.items():
            (directory / name).write_bytes(content)
        return {
            "kernel": "owner/private-run", "path": str(directory),
            "source_code_sha": "pinned-source",
            "files": {name: {"sha256": file_sha256(directory / name),
                             "size_bytes": (directory / name).stat().st_size}
                      for name in ("generator-best.pt", "semantic-ids.npy", "run-manifest.json")},
        }

    def test_reuse_requires_all_pinned_bytes_and_original_source_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            parent = self.parent_fixture(directory)
            self.assertEqual(verified_parent(parent), directory.resolve())
            wrong_revision = {**parent, "source_code_sha": "different"}
            with self.assertRaisesRegex(ValueError, "source revision"):
                verified_parent(wrong_revision)
            missing_pin = copy.deepcopy(parent)
            del missing_pin["files"]["generator-best.pt"]
            with self.assertRaisesRegex(ValueError, "pin checkpoint"):
                verified_parent(missing_pin)
            (directory / "semantic-ids.npy").write_bytes(b"changed SID bytes")
            with self.assertRaisesRegex(ValueError, "differs"):
                verified_parent(parent)

    def test_profile_never_uses_or_validates_test_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "catalog.json").write_text(json.dumps([
                {"item_id": "a", "text": "A"}, {"item_id": "b", "text": "B"},
            ]))
            # Neither an unknown item nor a non-item test value can reach the
            # profile's arrays: only train and validation fields are read.
            (directory / "sequences.json").write_text(json.dumps([
                {"user_id": 9, "train": ["a", "b"], "validation": "a",
                 "test": {"do not read": "held-out target"}},
            ]))
            expected = {"sha256": {name: file_sha256(directory / name)
                                   for name in ("catalog.json", "sequences.json")},
                        "items": 2, "users": 1, "train_examples": 1}
            users, counts = _profile_data(directory, expected)
            self.assertEqual(users[0].train, (0, 1))
            self.assertEqual(users[0].validation, 0)
            self.assertEqual(users[0].test, -1)
            self.assertEqual(counts["train_examples"], 1)
            expected["users"] = 2
            with self.assertRaisesRegex(ValueError, "count differs"):
                _profile_data(directory, expected)


@unittest.skipUnless(_HAS_TORCH, "Install numpy and torch for profiling execution tests")
class ProfileExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        cls.np, cls.torch = np, torch
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.old_threads)

    def setUp(self):
        np, torch = self.np, self.torch
        from rhpr.gr.sid import SidIndex
        self.index = SidIndex(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]), width=4)
        self.examples = (np.array([[0, 1], [1, 2], [2, 3], [3, 0]] * 2),
                         np.arange(8, dtype=np.int64), np.arange(8, dtype=np.int64) % 4)
        self.training = {"batch_size": 4, "learning_rate": 1e-3,
                         "weight_decay": 1e-4, "grad_clip_norm": 1.0}
        self.specification = {"seed": 42, "warmup": 1, "iterations": 2}

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(4, 8)
                self.dropout = torch.nn.Dropout(0.3)
                self.decoder = torch.nn.Module()
                self.decoder.decoder_mlp = torch.nn.ModuleList([torch.nn.Linear(8, 4) for _ in range(4)])

            def forward(self, input_ids, **kwargs):
                context = self.dropout(self.embedding(input_ids.clamp_min(0)).mean(dim=1))
                return context[:, None, :].expand(-1, 5, -1)

        self.model = Model().eval()

    def test_training_restores_weights_rng_and_mode_and_preserves_effective_batch(self):
        torch = self.torch
        state = {key: value.clone() for key, value in self.model.state_dict().items()}
        random_state = torch.get_rng_state()
        records = []
        for mode in ("full_pipeline", "gpu_resident", "full_pipeline"):
            row = benchmark_training(self.model, self.examples, self.index.sids,
                                     self.np.arange(8), 2, "fp32", mode,
                                     self.training, self.specification, ProfileDeadline(30), "cpu")
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["effective_batch_size"], 4)
            self.assertEqual(row["microbatches_per_update"], 2)
            self.assertTrue(row["baseline_effective_batch_preserved"])
            self.assertEqual(row["measured_updates_applied"], 2)
            self.assertGreater(row["examples_per_second"], 0)
            self.assertFalse(self.model.training)
            self.assertTrue(all(torch.equal(state[key], value) for key, value in self.model.state_dict().items()))
            self.assertTrue(torch.equal(random_state, torch.get_rng_state()))
            self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
            records.append(row)
        # Fresh optimizer, weights and dropout RNG give the same actual losses.
        self.assertEqual([row["last_loss"] for row in records], [records[0]["last_loss"]] * 3)
        larger = benchmark_training(self.model, self.examples, self.index.sids,
                                    self.np.arange(8), 8, "fp32", "full_pipeline",
                                    self.training, self.specification, ProfileDeadline(30), "cpu")
        self.assertEqual(larger["effective_batch_size"], 8)
        self.assertFalse(larger["baseline_effective_batch_preserved"])

    def test_failed_training_restores_weights_instead_of_producing_a_fast_result(self):
        torch = self.torch
        original_forward = self.model.forward
        self.model.forward = lambda **kwargs: original_forward(**kwargs) * float("nan")
        state = {key: value.clone() for key, value in self.model.state_dict().items()}
        with self.assertRaises(FloatingPointError):
            benchmark_training(self.model, self.examples, self.index.sids,
                               self.np.arange(8), 2, "fp32", "full_pipeline",
                               self.training, self.specification, ProfileDeadline(30), "cpu")
        self.assertTrue(all(torch.equal(state[key], value) for key, value in self.model.state_dict().items()))
        self.assertFalse(self.model.training)

    def test_rank_change_is_separate_from_retrieved_set_agreement(self):
        left = self.index.sids[[[0, 1], [2, 3]]]
        right = self.index.sids[[[1, 0], [2, 3]]]
        metrics = prediction_agreement(left, right)
        self.assertEqual(metrics["top1_exact_agreement"], 0.5)
        self.assertEqual(metrics["ranked_beam_exact_agreement"], 0.5)
        self.assertEqual(metrics["top10_set_overlap"], 1.0)

    def test_cpu_does_not_claim_native_fp16_or_bf16_support(self):
        self.assertTrue(precision_supported("fp32", "cpu"))
        self.assertFalse(precision_supported("fp16", "cpu"))
        self.assertFalse(precision_supported("bf16", "cpu"))

    def test_evaluation_warms_real_generation_and_keeps_subset_in_order(self):
        torch = self.torch
        sids = torch.from_numpy(self.index.sids.copy())

        class Generator(torch.nn.Module):
            calls = 0

            def generate(self, input_ids, attention_mask, user_id):
                self.calls += 1
                targets = user_id[:, 0] % len(sids)
                ids = torch.stack((sids[targets], sids[(targets + 1) % len(sids)]), dim=1)
                return ids, torch.ones(len(targets), 2, dtype=torch.float32)

        model = Generator()
        result, predictions = benchmark_evaluation(model, self.examples, self.index, 3, "fp32", "cpu",
                                                   ProfileDeadline(30), warmup=2)
        self.assertEqual(model.calls, 5)  # two warmups and ceil(8/3) timed batches
        self.assertEqual(result["warmup_batches"], 2)
        self.assertEqual(result["validation_metrics"]["recall@5"], 1.0)
        self.np.testing.assert_array_equal(predictions[:, 0], self.index.sids[self.examples[2]])

    def test_operator_profile_restores_rng_weights_and_identifies_instrumentation(self):
        torch = self.torch
        state = {key: value.clone() for key, value in self.model.state_dict().items()}
        random_state = torch.get_rng_state()
        with tempfile.TemporaryDirectory() as tmp:
            result = detailed_training_profile(self.model, self.examples, self.index.sids,
                                               self.np.arange(8), 2, "fp32", self.training,
                                               "cpu", Path(tmp))
            self.assertTrue(result["synchronized_stage_timing"])
            self.assertGreater(result["recognized_operator_flops_per_physical_batch"], 0)
            self.assertTrue((Path(tmp) / "training-profile-operators.txt").is_file())
        self.assertTrue(all(torch.equal(state[key], value) for key, value in self.model.state_dict().items()))
        self.assertTrue(torch.equal(random_state, torch.get_rng_state()))


@unittest.skipUnless(_CAN_INTEGRATE, "Set RHPR_GRID_SOURCE and install GR dependencies for actual GRID profiling")
class GridProfileIntegrationTests(unittest.TestCase):
    def test_real_grid_training_and_validation_profile_use_parent_sids(self):
        import numpy as np
        import torch
        from rhpr.gr.grid import build_grid_generator
        from rhpr.gr.sid import SidIndex
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            index = SidIndex(np.array([[0, 0, 0], [1, 0, 0], [2, 1, 1], [3, 1, 0]]), width=4)
            model = build_grid_generator(index.sids, {
                "codebook_size": 4, "beam_width": 2, "d_model": 16, "d_ff": 32,
                "num_heads": 2, "d_kv": 8, "encoder_layers": 1, "decoder_layers": 1,
                "num_user_bins": 8, "dropout": 0.0,
            }, seed=42, source_root=Path(_SOURCE))
            examples = (np.array([[0, 1], [1, 2], [2, 3], [3, 0]]),
                        np.arange(4), np.arange(4))
            before = {key: value.clone() for key, value in model.state_dict().items()}
            result = benchmark_training(model, examples, index.sids, np.arange(4), 2,
                                        "fp32", "full_pipeline",
                                        {"batch_size": 4, "learning_rate": 1e-3,
                                         "weight_decay": 1e-4, "grad_clip_norm": 1.0},
                                        {"warmup": 1, "iterations": 2}, ProfileDeadline(30), "cpu")
            self.assertEqual(result["status"], "ok")
            self.assertTrue(all(torch.equal(before[key], value) for key, value in model.state_dict().items()))
            row, predictions = benchmark_evaluation(model, examples, index, 2, "fp32", "cpu", ProfileDeadline(30))
            self.assertEqual(predictions.shape, (4, 2, 4))
            self.assertTrue(0 <= row["validation_metrics"]["ndcg@10"] <= 1)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
