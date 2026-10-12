from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
    import polars as pl
    import torch
except ImportError:
    np = pl = torch = None


@unittest.skipIf(torch is None or pl is None, "Torch and Polars are required")
class TailAblationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self) -> None:
        torch.set_num_threads(self.previous_threads)

    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")

    def fixture(self, directory: Path):
        """Save real weights and source catalogue predictions without training."""
        from rhpr.sasrec.data import YambdaSequences, split_sequence
        from rhpr.sasrec.depth_diagnostics import _fit_bins, _index_features
        from rhpr.sasrec.diagnostics import extract_history_features
        from rhpr.sasrec.experiment import (
            _EvaluationDataset, _collate_evaluation, _full_catalog_ndcg, _loader,
        )
        from rhpr.sasrec.model import SASRecModel

        source = directory / "source"
        source.mkdir()
        lengths = (17, 128, 129, 1028)
        dataset = YambdaSequences(
            splits=tuple(split_sequence(user, (np.arange(length + 2) * user) % 23 + 1)
                         for user, length in zip((11, 12, 13, 14), lengths, strict=True)),
            catalog_item_ids=np.arange(1, 24),
        )
        recipe = {
            "kind": "yambda_sasrec", "seed": 42,
            "data": {"path": "unused-listens.parquet"},
            "model": {
                "max_seq_len": 1028, "d_model": 8, "n_heads": 2, "n_layers": 1,
                "ffn_multiplier": 1, "dropout": 0.0,
                "compressor_type": "recursive_pair_mlp", "compression_depth": 1,
                "tail_size": 128, "use_time_features": False,
            },
            "loss": {"name": "sampled_softmax", "tail_size": 128},
            "evaluation": {"k": 10, "evaluate_test": False},
        }
        torch.manual_seed(73)
        model = SASRecModel(num_items=dataset.num_items, **recipe["model"]).eval()
        torch.save(model.state_dict(), source / "best-model.pt")
        checksum = hashlib.sha256((source / "best-model.pt").read_bytes()).hexdigest()
        config = json.loads((ROOT / "experiments/yambda_mixed_depth_tail128_ablation.json").read_text())
        config["source"].update(path=str(source), checkpoint_sha256=checksum,
                                git_sha="a" * 40, expected_validation_users=4)
        config["runtime"].update(device="cpu", amp=False, workers=0)
        config["evaluation"].update(batch_size=2, item_chunk_size=7)
        config["benchmark"]["enabled"] = False
        config["bootstrap_samples"] = 20
        loader = _loader(_EvaluationDataset(dataset.splits, 1028, "validation"),
            batch_size=2, shuffle=False, workers=0, seed=42,
            collate_fn=_collate_evaluation, device=torch.device("cpu"))
        predictions = []
        with torch.no_grad():
            for depth in range(4):
                model.set_compression_depth(depth)
                rows = []
                _full_catalog_ndcg(model, loader, device=torch.device("cpu"), k=10,
                    item_chunk_size=7, amp=False, phase="validation", prediction_rows=rows,
                    model_label=f"depth_{depth}", seed=42)
                for row in rows:
                    row["compression_depth"] = depth
                predictions.extend(rows)
        features = extract_history_features(dataset.splits,
            model.item_embedding.weight.detach().numpy(), max_seq_len=1028, tail_size=128)
        values = {
            "model-config.json": recipe,
            "run-manifest.json": {"git_sha": "a" * 40,
                "config": {"kind": "yambda_mixed_depth", "seed": 42, "base": recipe}},
            "metrics.json": {"mixed_depth_complete": True, "test_evaluated": False,
                "seed": 42, "checkpoint_sha256": checksum, "best_epoch": 15,
                "epochs_trained": 15, "num_users": 4, "num_items": 23,
                "validation_examples": 4, "depth_0_validation_examples": 4,
                "depth_3_validation_examples": 4},
            "diagnostic-reference.json": {"checkpoint": "best-model.pt",
                "checkpoint_sha256": checksum, "git_sha": "a" * 40, "seed": 42,
                "best_epoch": 15},
            "feature_bins.json": _fit_bins(_index_features(features)),
            "history-features.json": features,
            "predictions.json": predictions,
        }
        for name, value in values.items():
            self.write_json(source / name, value)
        return config, dataset, source, model

    def test_mixed_length_selector_preserves_raw_tail_and_source_positions(self) -> None:
        from rhpr.sasrec.tail_ablation import TailOnlySequenceCompressor

        lengths = torch.tensor([17, 128, 129, 1028])
        embeddings = torch.arange(4 * 1028 * 3, dtype=torch.float32).reshape(4, 1028, 3)
        mask = torch.arange(1028)[None, :] < lengths[:, None]
        output, output_mask, sources = TailOnlySequenceCompressor()(embeddings, mask)
        self.assertEqual(output.shape, (4, 128, 3))
        for user, length in enumerate(lengths.tolist()):
            retained = min(length, 128)
            self.assertTrue(torch.equal(output[user, :retained], embeddings[user, length - retained:length]))
            self.assertTrue(torch.equal(sources[user, :retained], torch.arange(length - retained, length)))
            self.assertEqual(int(output_mask[user].sum()), retained)
            self.assertTrue(torch.all(sources[user, retained:] == -1))
            self.assertTrue(torch.all(output[user, retained:] == 0))

    def test_raw_recency_pe_and_short_history_states_are_unchanged(self) -> None:
        from rhpr.sasrec.model import SASRecModel
        from rhpr.sasrec.tail_ablation import TailOnlySequenceCompressor

        torch.manual_seed(5)
        model = SASRecModel(num_items=23, max_seq_len=1028, d_model=8, n_heads=2,
            n_layers=1, ffn_multiplier=1, dropout=0.0,
            compressor_type="recursive_pair_mlp", compression_depth=0).eval()
        lengths = torch.tensor([17, 128, 129, 1028])
        mask = torch.arange(1028)[None, :] < lengths[:, None]
        items = (torch.arange(1028)[None, :].expand(4, -1) % 23 + 1).masked_fill(~mask, 0)
        with torch.no_grad():
            full_states = model.last_state(items, mask)
            model.compressor = TailOnlySequenceCompressor()
            observed_positions = []
            hook = model.position_embedding.register_forward_pre_hook(
                lambda module, arguments: observed_positions.append(arguments[0].clone()))
            tail_states = model.last_state(items, mask)
            hook.remove()
        self.assertTrue(torch.allclose(full_states[:2], tail_states[:2], rtol=1e-5, atol=1e-6))
        expected = torch.arange(127, -1, -1)[None, :].expand(4, -1).clone()
        expected[0] = 0
        expected[0, :17] = torch.arange(16, -1, -1)
        self.assertTrue(torch.equal(observed_positions[0], expected))

    def test_real_catalogue_eval_reuses_checkpoint_and_never_reads_test_or_trains(self) -> None:
        from rhpr.sasrec import tail_ablation
        from rhpr.sasrec.data import YambdaSequences
        from rhpr.sasrec.diagnostics import context_hash

        class ValidationOnlySplit:
            def __init__(self, split):
                self.user_id, self.train, self.validation = split.user_id, split.train, split.validation

            @property
            def test(self):
                raise AssertionError("test target must remain unread")

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config, dataset, source, fixture_model = self.fixture(directory)
            safe_dataset = YambdaSequences(
                splits=tuple(ValidationOnlySplit(split) for split in dataset.splits),
                catalog_item_ids=dataset.catalog_item_ids)
            checkpoint_before = (source / "best-model.pt").read_bytes()
            expected_state = {key: value.clone() for key, value in fixture_model.state_dict().items()}
            instantiated = []
            actual_model_type = tail_ablation.SASRecModel

            def record_model(**kwargs):
                model = actual_model_type(**kwargs)
                instantiated.append(model)
                return model

            output = directory / "ablation"
            with mock.patch.object(tail_ablation, "SASRecModel", side_effect=record_model), \
                    mock.patch.object(torch.optim, "AdamW", side_effect=AssertionError("no optimizer")):
                metrics = tail_ablation.run_tail_ablation(config, output,
                                                         dataset=safe_dataset, source_dir=source)
            self.assertEqual(len(instantiated), 1)
            self.assertEqual(instantiated[0].state_dict().keys(), expected_state.keys())
            for key, value in instantiated[0].state_dict().items():
                self.assertTrue(torch.equal(value, expected_state[key]), key)
            self.assertTrue(all(parameter.grad is None for parameter in instantiated[0].parameters()))
            self.assertEqual((source / "best-model.pt").read_bytes(), checkpoint_before)
            self.assertEqual(list(output.rglob("*.pt")), [])
            self.assertIs(metrics["tail_ablation_complete"], True)
            self.assertIs(metrics["test_evaluated"], False)
            self.assertIs(metrics["training_performed"], False)
            self.assertEqual(metrics["training_epochs"], 0)
            self.assertEqual(metrics["num_users"], 4)
            self.assertEqual(metrics["num_items"], 23)
            for value in metrics.values():
                self.assertIsInstance(value, (int, float, str, bool))
                if isinstance(value, float):
                    self.assertTrue(math.isfinite(value))
            self.assertEqual(json.loads((output / "metrics.json").read_text()), metrics)
            rows = json.loads((output / "predictions.json").read_text())
            self.assertEqual(len(rows), 12)
            self.assertEqual(pl.read_parquet(output / "predictions.parquet").height, 12)
            indexed = {(row["model"], row["user_id"]): row for row in rows}
            for split in dataset.splits:
                for label in ("depth_0", "depth_3", "tail_128"):
                    row = indexed[label, split.user_id]
                    self.assertEqual(row["target"], split.validation)
                    self.assertEqual(row["context_hash"], context_hash(split.train))
                    self.assertEqual(row["reference_context_hash"], context_hash(split.train))
                    actual_input = split.train[-128:] if label == "tail_128" else split.train
                    self.assertEqual(row["input_context_hash"], context_hash(actual_input))
                    self.assertEqual(row["input_length"], len(actual_input))
                    self.assertEqual(len(row["top10"]), 10)
                    self.assertEqual(len(set(row["top10"])), 10)
                    self.assertTrue(all(1 <= item <= 23 for item in row["top10"]))
                if len(split.train) <= 128:
                    self.assertEqual(indexed["depth_0", split.user_id]["top10"],
                                     indexed["tail_128", split.user_id]["top10"])
            reproduction = json.loads((output / "reproduction-check.json").read_text())
            for label in ("depth_0", "depth_3"):
                self.assertEqual(reproduction["modes"][label]["top10_equal_count"], 4)
                self.assertAlmostEqual(reproduction["modes"][label]["delta_ndcg_at_10"], 0.0, places=7)
                self.assertEqual(metrics[label + "_checkpoint_sha256"], config["source"]["checkpoint_sha256"])
                self.assertEqual(metrics[label + "_validation_examples"], 4)
            self.assertEqual(json.loads((output / "history-features.json").read_text()),
                             json.loads((source / "history-features.json").read_text()))
            self.assertTrue((output / "tail-cohorts.json").is_file())

    def test_lineage_mismatch_fails_before_checkpoint_deserialization(self) -> None:
        from rhpr.sasrec import tail_ablation

        for field, message in (("checkpoint_sha256", "SHA256"), ("git_sha", "git revision")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                config, dataset, source, _ = self.fixture(directory)
                config["source"][field] = "0" * len(config["source"][field])
                with mock.patch.object(torch, "load", side_effect=AssertionError("must verify before loading")) as load:
                    with self.assertRaisesRegex(ValueError, message):
                        tail_ablation.run_tail_ablation(config, directory / "ablation",
                                                       dataset=dataset, source_dir=source)
                    load.assert_not_called()
                partial = json.loads((directory / "ablation/metrics.json").read_text())
                self.assertIs(partial["tail_ablation_complete"], False)

    def test_changed_full_context_or_target_is_rejected_before_loading(self) -> None:
        from rhpr.sasrec import tail_ablation

        for name, field, value in (("predictions", "target", 23),
                                    ("predictions", "context_hash", "incorrect-context"),
                                    ("history-features", "context_hash", "incorrect-context")):
            with self.subTest(name=name, field=field), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                config, dataset, source, _ = self.fixture(directory)
                path = source / f"{name}.json"
                rows = json.loads(path.read_text())
                if field == "target" and rows[0][field] == value:
                    value = 22
                rows[0][field] = value
                self.write_json(path, rows)
                with mock.patch.object(torch, "load", side_effect=AssertionError("must reject identity first")) as load:
                    with self.assertRaisesRegex(ValueError, "context|target"):
                        tail_ablation.run_tail_ablation(config, directory / "ablation",
                                                       dataset=dataset, source_dir=source)
                    load.assert_not_called()

    def test_config_cannot_enable_training_test_or_partial_validation(self) -> None:
        from rhpr.sasrec.tail_ablation import validate_tail_ablation_config

        valid = json.loads((ROOT / "experiments/yambda_mixed_depth_tail128_ablation.json").read_text())
        validate_tail_ablation_config(valid)
        candidates = []
        for field in ("training", "model", "data"):
            candidate = copy.deepcopy(valid)
            candidate[field] = {}
            candidates.append(candidate)
        for section, field, value in (("evaluation", "evaluate_test", True),
                                       ("evaluation", "max_validation_batches", 1),
                                       ("source", "checkpoint_sha256", "not-pinned")):
            candidate = copy.deepcopy(valid)
            candidate[section][field] = value
            candidates.append(candidate)
        for candidate in candidates:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                validate_tail_ablation_config(candidate)


if __name__ == "__main__":
    unittest.main()
