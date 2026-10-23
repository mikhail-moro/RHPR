"""Actual GRID fitting and validation selection for the bounded quality run."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

_SOURCE = os.environ.get("RHPR_GRID_SOURCE")
_CAN_INTEGRATE = bool(_SOURCE) and all(importlib.util.find_spec(name) for name in
    ("torch", "numpy", "transformers", "lightning", "pyarrow", "hydra"))


@unittest.skipUnless(_CAN_INTEGRATE, "Requires pinned GRID source and GR dependencies")
class PaperQualityIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from rhpr.gr.grid import load_grid
        cls.np, cls.torch = np, torch
        cls.grid = load_grid(Path(_SOURCE))
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.old_threads)

    def test_full_rk_fit_respects_rank_counts_init_budget_and_exported_assignments(self):
        from rhpr.gr.paper_fit import fit_paper_rkmeans
        np, torch = self.np, self.torch
        embeddings = np.random.default_rng(73).normal(size=(24, 8)).astype(np.float32)
        config = {"codebook_size": 4, "num_hierarchies": 2, "logical_ranks": 3,
            "batch_size_per_rank": 4, "steps_per_level": 6, "init_buffer_size": 6,
            "normalize_residuals": True, "distance_backend": "torch_cdist"}
        rng = torch.get_rng_state().clone()
        codes, centers, report = fit_paper_rkmeans(embeddings, config, seed=42, source_root=_SOURCE)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        again, again_centers, _ = fit_paper_rkmeans(embeddings, config, seed=42, source_root=_SOURCE)
        np.testing.assert_array_equal(codes, again)
        np.testing.assert_array_equal(centers, again_centers)
        source_codes, source_centers, _ = fit_paper_rkmeans(embeddings,
            dict(config, distance_backend="upstream"), seed=42, source_root=_SOURCE)
        np.testing.assert_array_equal(codes, source_codes)
        np.testing.assert_array_equal(centers, source_centers)
        residual = embeddings.copy()
        for level, stage in enumerate(report["levels"]):
            self.assertEqual(stage["initialization_steps"], 2)
            self.assertEqual(stage["steady_updates"], 4)
            self.assertEqual(stage["training_points_per_rank"], 16)
            self.assertEqual([sum(row) for row in stage["rank_cluster_counts"]], [16, 16, 16])
            residual /= np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1e-12)
            distances = ((residual[:, None] - centers[level][None]) ** 2).sum(-1)
            np.testing.assert_array_equal(codes[:, level], distances.argmin(1))
            residual -= centers[level][codes[:, level]]

    def test_catalogue_cycles_cover_every_row_without_dropping_remainder(self):
        from rhpr.gr.paper_fit import CatalogueStream
        stream = CatalogueStream(7, 42)
        first = stream.take(5).tolist()
        second = stream.take(9).tolist()
        self.assertEqual(sorted(first + second[:2]), list(range(7)))
        self.assertEqual(sorted(second[2:]), list(range(7)))

    def make_training(self):
        from rhpr.gr.data import UserSequence
        from rhpr.gr.grid import build_grid_generator
        from rhpr.gr.sid import SidIndex
        index = SidIndex(self.np.array([[i, 0, 0] for i in range(4)]), width=4)
        model = build_grid_generator(index.sids, {"codebook_size": 4, "beam_width": 2,
            "d_model": 8, "d_ff": 16, "d_kv": 4, "num_heads": 2,
            "encoder_layers": 1, "decoder_layers": 1, "dropout": 0.,
            "mlp_layers": 2, "separator_token": True, "num_user_bins": None},
            seed=42, source_root=_SOURCE)
        users = [UserSequence(i, (i, (i + 1) % 4, (i + 2) % 4), 0, 1) for i in range(4)]
        config = {"seed": 42, "data": {"history_items": 2}, "training": {
            "learning_rate": 0.001, "weight_decay": 1e-6, "precision": "fp32",
            "evaluation_precision": "fp32", "evaluation_batch_size": 4,
            "batch_size": 4, "physical_batch_size": 2, "source_user_batch_size": 2,
            "sequence_length": 12, "max_augmented_batch": 8,
            "max_seconds": 30, "max_updates": 5, "validation_interval_updates": 1,
            "patience_checks": 2, "grad_clip_norm": None}}
        return model, users, index, config

    def test_early_stopping_uses_ndcg_while_selected_weights_use_recall(self):
        from rhpr.gr.paper_experiment import train_paper_generator
        model, users, index, config = self.make_training()
        states = []
        scores = iter(((0.1, 0.5), (0.3, 0.4), (0.2, 0.4)))

        def evaluate(model, examples, sid_index, batch_size, device, precision):
            states.append({name: value.detach().clone() for name, value in model.state_dict().items()})
            recall, ndcg = next(scores)
            return {"recall@10": recall, "ndcg@10": ndcg}, None, None

        with tempfile.TemporaryDirectory() as tmp:
            result = train_paper_generator(model, users, index, config, Path(tmp), "cpu", evaluator=evaluate)
            self.assertEqual(result["stop_reason"], "early_stopping")
            self.assertEqual(result["updates"], 3)
            self.assertEqual(result["selected_update"], 2)
            self.assertEqual(result["best_validation_ndcg10"], 0.5)
            self.assertEqual(result["selected_validation"]["ndcg@10"], 0.4)
            self.assertEqual(result["model_examples_consumed"], 12)
            for name, value in model.state_dict().items():
                self.torch.testing.assert_close(value, states[1][name], rtol=0, atol=0)
            self.assertTrue((Path(tmp) / "generator-last.pt").is_file())

    def test_time_budget_still_selects_on_validation_and_saves_final_partial_interval(self):
        from rhpr.gr.paper_experiment import train_paper_generator
        model, users, index, config = self.make_training()
        config["training"]["validation_interval_updates"] = 100
        calls = []

        def evaluate(*args, **kwargs):
            calls.append(1)
            return {"recall@10": 0.1, "ndcg@10": 0.05}, None, None

        with tempfile.TemporaryDirectory() as tmp:
            result = train_paper_generator(model, users, index, config, Path(tmp), "cpu",
                max_seconds=0, evaluator=evaluate)
            self.assertEqual(result["stop_reason"], "time_budget")
            self.assertFalse(result["early_stopping_reached"])
            self.assertEqual(result["selected_update"], 1)
            self.assertEqual(len(calls), 1)
            curves = json.loads((Path(tmp) / "learning-curves.json").read_text())
            self.assertEqual(curves[0]["update"], 1)

    def test_complete_small_run_fits_real_codes_trains_and_evaluates_test_once(self):
        from rhpr.gr.experiment import _evaluate
        from rhpr.gr.paper_experiment import run_grid_paper_quality
        from rhpr.metrics import validate_metrics

        model, users, index, config = self.make_training()
        config.update(name="cpu-smoke", max_run_seconds=60,
            quantizer={"codebook_size": 4, "num_hierarchies": 3,
                "logical_ranks": 2, "batch_size_per_rank": 4, "embedding_dim": 8,
                "steps_per_level": 3, "init_buffer_size": 4,
                "normalize_residuals": True, "distance_backend": "torch_cdist", "max_seconds": 30},
            generator={"codebook_size": 4, "beam_width": 2, "d_model": 8, "d_ff": 16,
                "d_kv": 4, "num_heads": 2, "encoder_layers": 1, "decoder_layers": 1,
                "dropout": 0., "mlp_layers": 2, "separator_token": True, "num_user_bins": None},
            distance_profile={"quantizer_iterations": 1, "warmup": 1})
        config["max_run_seconds"] = 300  # Includes the runner's final-test reserve.
        config["training"].update(device="cpu", cpu_threads=1, max_updates=2)
        embeddings = self.np.random.default_rng(7).normal(size=(4, 8)).astype(self.np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "data/catalog.json").write_text(json.dumps([
                {"item_id": i, "text": f"item {i}"} for i in range(4)]))
            (root / "data/sequences.json").write_text(json.dumps([
                {"user_id": u.user_id, "train": list(u.train), "validation": u.validation,
                 "test": u.test} for u in users]))
            output = root / "output"
            output.mkdir()
            config["resources"] = {"path": str(root), "manifest_sha256": "smoke",
                                   "scratch_path": str(root / "scratch")}
            with patch("rhpr.gr.resources.verify_resources", return_value={}), patch(
                "rhpr.gr.resources.unpack_grid", return_value=Path(_SOURCE)), patch(
                "rhpr.gr.paper_experiment.encode_flan_catalogue",
                return_value=(embeddings, {"seconds": 0.})), patch.dict(os.environ), patch(
                "rhpr.gr.paper_experiment._evaluate", wraps=_evaluate) as test_evaluation:
                metrics = validate_metrics(run_grid_paper_quality(config, output))
            self.assertEqual(test_evaluation.call_count, 1)
            self.assertEqual(metrics["test_evaluations"], 1)
            self.assertEqual(metrics["updates"], 2)
            for name in ("semantic-ids.npy", "quantizer.json", "training-selection.json",
                         "test-predictions.npz", "test-metrics.json", "rk-distance-profile.json"):
                self.assertTrue((output / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
