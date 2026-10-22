"""Integrity checks and optional execution of pinned upstream GRID components."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rhpr.gr.grid import (
    GRID_REVISION,
    build_grid_generator,
    fit_grid_rkmeans,
    load_grid,
    prepare_grid_source,
)

_SOURCE = os.environ.get("RHPR_GRID_SOURCE")
_REQUIRED = ("torch", "numpy", "transformers", "lightning", "pytorch_lightning", "torchmetrics", "pyarrow", "hydra")
_CAN_INTEGRATE = bool(_SOURCE) and all(importlib.util.find_spec(name) is not None for name in _REQUIRED)


class GridSourceIntegrityTests(unittest.TestCase):
    def test_rejects_unpinned_source_without_importing_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "__init__.py").write_text("raise RuntimeError('do not import me')\n")
            with self.assertRaisesRegex(ValueError, "does not match pinned revision"):
                prepare_grid_source(root)


@unittest.skipUnless(_CAN_INTEGRATE, "Set RHPR_GRID_SOURCE and install GR dependencies for actual GRID integration")
class GridIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        cls.np = np
        cls.torch = torch
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.source = Path(_SOURCE).resolve()
        cls.grid = load_grid(cls.source)
        cls.sids = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [2, 1, 1, 0], [3, 1, 0, 0]], dtype=np.int64)
        cls.config = {
            "codebook_size": 4, "beam_width": 2, "d_model": 16, "d_ff": 32,
            "num_heads": 2, "d_kv": 8, "encoder_layers": 1, "decoder_layers": 1,
            "num_user_bins": 8, "dropout": 0.0,
        }

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.old_threads)

    def make_model(self):
        return build_grid_generator(self.sids, self.config, seed=42, source_root=self.source)

    def test_preparation_changes_only_annotation_import(self):
        prepared = self.grid.source_root
        self.assertNotEqual(prepared, self.source)
        changed = []
        for original in sorted((self.source / "src").rglob("*.py")):
            rel = original.relative_to(self.source)
            if original.read_bytes() != (prepared / rel).read_bytes():
                changed.append(rel.as_posix())
        self.assertEqual(changed, ["src/data/loading/components/interfaces.py"])
        patched = (prepared / changed[0]).read_text()
        self.assertIn("if TYPE_CHECKING:\n    from src.data.loading.components.iterators import RawDataIterator", patched)
        self.assertNotIn("src.data.loading.components.iterators", sys.modules)
        self.assertNotIn("tensorflow", sys.modules)
        self.assertEqual(self.grid.revision, GRID_REVISION)

    def test_prepared_copy_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = prepare_grid_source(self.source, Path(tmp) / "prepared")
            module = target / "src/models/modules/semantic_id/tiger_generation_model.py"
            module.write_text(module.read_text() + "\n# unexpected edit\n")
            with self.assertRaisesRegex(ValueError, "does not match pinned revision"):
                load_grid(target)

    def test_conflicting_src_namespace_is_not_overwritten(self):
        code = (
            "import sys, types; "
            "sys.modules['src']=types.ModuleType('src'); "
            "from rhpr.gr.grid import load_grid; "
            "load_grid(sys.argv[1])"
        )
        child_env = os.environ.copy()
        repository_src = str(Path(__file__).resolve().parents[1] / "src")
        prior_pythonpath = child_env.get("PYTHONPATH")
        child_env["PYTHONPATH"] = (
            repository_src + os.pathsep + prior_pythonpath
            if prior_pythonpath else repository_src
        )
        result = subprocess.run(
            [sys.executable, "-c", code, str(self.source)],
            capture_output=True, text=True, env=child_env,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already imported", result.stderr)

    def test_actual_generator_backpropagates_all_four_code_losses(self):
        torch = self.torch
        model = self.make_model()
        self.assertIsInstance(model, self.grid.SemanticIDEncoderDecoder)
        self.assertEqual(model.__class__.__module__, "src.models.modules.semantic_id.tiger_generation_model")
        x = torch.tensor([[0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, -1, -1, -1, -1]])
        target = torch.tensor([[2, 1, 1, 0], [3, 1, 0, 0]])
        model.train()
        output = model(
            attention_mask_encoder=(x != -1).long(), input_ids=x,
            user_id=torch.tensor([[0], [1]]), future_ids=target,
        )
        losses = [
            torch.nn.functional.cross_entropy(model.decoder.decoder_mlp[level](output[:, level]), target[:, level])
            for level in range(4)
        ]
        sum(losses).backward()
        self.assertTrue(all(torch.isfinite(loss).item() for loss in losses))
        self.assertGreater(model.item_sid_embedding_table_encoder.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.user_embedding.weight.grad.abs().sum().item(), 0)
        for head in model.decoder.decoder_mlp:
            self.assertTrue(torch.isfinite(head.weight.grad).all().item())
            self.assertGreater(head.weight.grad.abs().sum().item(), 0)

    def test_cached_beam_scores_match_independent_teacher_forcing(self):
        torch = self.torch
        model = self.make_model().eval()
        x = torch.tensor([[0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, -1, -1, -1, -1]])
        mask = (x != -1).long()
        users = torch.tensor([[0], [1]])
        with torch.no_grad():
            generated, scores = model.generate(mask, x, users)
            self.assertEqual(tuple(generated.shape), (2, 2, 4))
            for row in range(2):
                for beam in range(2):
                    codes = generated[row, beam].unsqueeze(0)
                    hidden = model(
                        attention_mask_encoder=mask[row:row + 1], input_ids=x[row:row + 1],
                        user_id=users[row:row + 1], future_ids=codes,
                    )
                    expected = 1.0
                    for level in range(4):
                        probs = model.decoder.decoder_mlp[level](hidden[:, level]).softmax(dim=-1)
                        expected *= probs[0, codes[0, level]].item()
                    self.assertAlmostEqual(scores[row, beam].item(), expected, delta=1e-6)

    def test_float32_beam_adapter_preserves_upstream_results_exactly(self):
        torch = self.torch
        model = self.make_model().eval()
        upstream = type(model)._beam_search_one_step
        wrapped_ids, wrapped_scores = None, None
        original_ids, original_scores = None, None
        generator = torch.Generator().manual_seed(314)
        for level in range(4):
            rows = 2 if level == 0 else 4
            logits = torch.randn(rows, 4, generator=generator)
            wrapped_ids, wrapped_scores, _ = model._beam_search_one_step(
                candidate_logits=logits.clone(), generated_ids=wrapped_ids,
                marginal_log_prob=wrapped_scores, past_key_values=None,
                hierarchy=level, batch_size=2,
            )
            original_ids, original_scores, _ = upstream(
                model, candidate_logits=logits.clone(), generated_ids=original_ids,
                marginal_log_prob=original_scores, past_key_values=None,
                hierarchy=level, batch_size=2,
            )
            self.assertTrue(torch.equal(wrapped_ids, original_ids))
            self.assertTrue(torch.equal(wrapped_scores, original_scores))

    def test_fp16_path_probabilities_do_not_underflow_at_four_256_code_levels(self):
        torch = self.torch
        config = dict(self.config, codebook_size=256)
        model = build_grid_generator(self.sids, config, seed=42, source_root=self.source).eval()
        upstream = type(model)._beam_search_one_step
        wrapped_ids, wrapped_scores = None, None
        original_ids, original_scores = None, None
        for level in range(4):
            rows = 1 if level == 0 else config["beam_width"]
            logits = torch.zeros(rows, 256, dtype=torch.float16)
            wrapped_ids, wrapped_scores, _ = model._beam_search_one_step(
                candidate_logits=logits.clone(), generated_ids=wrapped_ids,
                marginal_log_prob=wrapped_scores, past_key_values=None,
                hierarchy=level, batch_size=1,
            )
            original_ids, original_scores, _ = upstream(
                model, candidate_logits=logits.clone(), generated_ids=original_ids,
                marginal_log_prob=original_scores, past_key_values=None,
                hierarchy=level, batch_size=1,
            )
        self.assertTrue(torch.equal(original_scores, torch.zeros_like(original_scores)))
        self.assertEqual(wrapped_scores.dtype, torch.float32)
        self.assertTrue(torch.equal(wrapped_scores, torch.full_like(wrapped_scores, 256.0 ** -4)))
        self.assertEqual(model.rhpr_grid_provenance["beam_probability_dtype"], "float32")

    def test_autocast_generation_keeps_beam_probabilities_in_float32(self):
        torch = self.torch
        model = self.make_model().eval()
        x = torch.tensor([[0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, -1, -1, -1, -1]])
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            generated, scores = model.generate((x != -1).long(), x, torch.tensor([[0], [1]]))
        self.assertEqual(tuple(generated.shape), (2, 2, 4))
        self.assertEqual(scores.dtype, torch.float32)
        self.assertTrue(torch.isfinite(scores).all().item())
        self.assertTrue((scores > 0).all().item())

    def test_nonunique_complete_ids_are_rejected(self):
        duplicate = self.np.concatenate([self.sids, self.sids[:1]], axis=0)
        with self.assertRaisesRegex(ValueError, "uniquely identify"):
            build_grid_generator(duplicate, self.config, source_root=self.source)

    def test_rkmeans_is_reproducible_and_exported_codes_match_centroids(self):
        np = self.np
        embeddings = np.random.default_rng(42).normal(size=(32, 8)).astype(np.float32)
        config = {"codebook_size": 4, "num_hierarchies": 2, "passes_per_level": 2, "batch_size": 8, "init_buffer_size": 16}
        codes, centroids, report = fit_grid_rkmeans(embeddings, config, seed=73, source_root=self.source)
        repeated_codes, repeated_centroids, _ = fit_grid_rkmeans(embeddings, config, seed=73, source_root=self.source)
        np.testing.assert_array_equal(codes, repeated_codes)
        np.testing.assert_array_equal(centroids, repeated_centroids)
        self.assertEqual(codes.shape, (32, 2))
        self.assertEqual(centroids.shape, (2, 4, 8))
        self.assertEqual([entry["training_points"] for entry in report["levels"]], [64, 64])
        self.assertEqual([entry["updates"] for entry in report["levels"]], [8, 8])
        residual = embeddings.copy()
        for level in range(2):
            residual /= np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1e-12)
            nearest = ((residual[:, None, :] - centroids[level][None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
            np.testing.assert_array_equal(codes[:, level], nearest)
            residual -= centroids[level][nearest]
        _, scaled_centroids, _ = fit_grid_rkmeans(embeddings * 10, config, seed=73, source_root=self.source)
        np.testing.assert_allclose(centroids, scaled_centroids, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
