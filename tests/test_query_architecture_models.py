from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "Torch is required")
class QueryArchitectureModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def config(self):
        config = json.loads((ROOT / "experiments/yambda_query_prefix_1k_search.json").read_text())
        fixed = {"backbone_lr": .001, "compressor_lr": .001}
        config["architecture_trials"] = {
            "query_cross_attention_m4": [
                dict(fixed, name="ca_base", readout_depth=1, cross_attention_heads=4),
                dict(fixed, name="ca_read2", readout_depth=2, cross_attention_heads=4),
                dict(fixed, name="ca_heads8", readout_depth=1, cross_attention_heads=8)],
            "query_ssm_m4": [
                dict(fixed, name="q_base", d_inner=128, d_state=16, gate_rank=32),
                dict(fixed, name="q_inner64", d_inner=64, d_state=16, gate_rank=32),
                dict(fixed, name="q_state32", d_inner=128, d_state=32, gate_rank=32)],
        }
        return config

    def assert_state_equal(self, first, second):
        self.assertEqual(first.keys(), second.keys())
        for name, value in first.items():
            with self.subTest(parameter=name):
                torch.testing.assert_close(value, second[name], rtol=0, atol=0)

    def test_baselines_preserve_original_builder_weights_outputs_and_rng(self):
        from rhpr.sasrec.query_architecture_models import build_architecture_model
        from rhpr.sasrec.query_proxy_protocol import build_hpo_model, hpo_variants

        config = self.config()
        items = torch.arange(1, 270)[None, :] % 70 + 1
        mask = torch.ones_like(items, dtype=torch.bool)
        for variant in hpo_variants(config):
            with self.subTest(variant=variant["name"]):
                expected = build_hpo_model(config, variant, 80).eval()
                expected_rng = torch.get_rng_state().clone()
                actual = build_architecture_model(config, variant,
                    config["architecture_trials"][variant["name"]][0], 80).eval()
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                self.assertEqual(type(actual.compressor), type(expected.compressor))
                self.assert_state_equal(expected.state_dict(), actual.state_dict())
                with torch.no_grad():
                    for before, after in zip(expected.encode(items, mask), actual.encode(items, mask)):
                        torch.testing.assert_close(before, after, rtol=0, atol=0)
                self.assertEqual(actual.position_embedding.num_embeddings, 32768)

    def test_all_architectures_keep_backbone_query_ffn_weights_and_training_rng(self):
        from rhpr.sasrec.query_architecture_models import build_architecture_model
        from rhpr.sasrec.query_proxy_protocol import build_hpo_model, hpo_variants

        config = self.config()
        expected_counts = {"ca_base": 265856, "ca_read2": 332416,
                           "ca_heads8": 265856, "q_base": 249632,
                           "q_inner64": 230048, "q_state32": 254240}
        for variant in hpo_variants(config):
            expected = build_hpo_model(config, variant, 80)
            expected_rng = torch.get_rng_state().clone()
            shared = {name: value for name, value in expected.state_dict().items()
                      if not name.startswith("compressor.") or name.startswith(tuple(
                          f"compressor.{prefix}" for prefix in ("query_generator.", "memory_norm.",
                          "readout_norm.", "ffn_norm.", "ffn.")))}
            for trial in config["architecture_trials"][variant["name"]]:
                with self.subTest(trial=trial["name"]):
                    model = build_architecture_model(config, variant, trial, 80)
                    self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                    self.assert_state_equal(shared,
                        {name: model.state_dict()[name] for name in shared})
                    self.assertEqual(sum(value.numel() for value in model.compressor.parameters()),
                                     expected_counts[trial["name"]])
                    self.assertEqual(model.compressor.query_generator.attention.num_heads, 4)

    def test_eight_global_heads_change_only_attention_factorization(self):
        from rhpr.sasrec.query_architecture_models import build_architecture_model
        from rhpr.sasrec.query_proxy_protocol import hpo_variants

        config = self.config()
        variant = hpo_variants(config)[0]
        base_trial, _, head_trial = config["architecture_trials"][variant["name"]]
        base = build_architecture_model(config, variant, base_trial, 80).compressor.eval()
        changed = build_architecture_model(config, variant, head_trial, 80).compressor.eval()
        self.assert_state_equal(base.state_dict(), changed.state_dict())
        self.assertEqual(changed.cross_attention.num_heads, 8)
        self.assertEqual(changed.cross_attention.head_dim, 16)
        self.assertEqual(changed.query_generator.attention.num_heads, 4)
        values = torch.randn(2, 270, 128)
        mask = torch.ones((2, 270), dtype=torch.bool)
        with torch.no_grad():
            before, after = base(values, mask), changed(values, mask)
        self.assertFalse(torch.allclose(before[0][:, :4], after[0][:, :4]))
        self.assertTrue(torch.equal(before[1], after[1]))
        self.assertTrue(torch.equal(before[2], after[2]))
        torch.testing.assert_close(before[0][:, 4:], after[0][:, 4:], rtol=0, atol=0)

    def compressor(self, depth=2, heads=4):
        from rhpr.sasrec.query_architecture_models import IterativeQueryCrossAttentionCompressor

        return IterativeQueryCrossAttentionCompressor(d_model=16, summary_tokens=2,
            tail_size=6, query_context_size=4, n_heads=4, dropout=0,
            query_source="old_suffix", readout_depth=depth, cross_attention_heads=heads).eval()

    def test_depth_one_class_is_exact_original_even_for_ragged_histories(self):
        from rhpr.sasrec.query_ssm import QueryCrossAttentionCompressor

        torch.manual_seed(191)
        expected = QueryCrossAttentionCompressor(16, 2, tail_size=6,
            query_context_size=4, n_heads=4, dropout=0, query_source="old_suffix").eval()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(191)
        actual = self.compressor(depth=1)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assert_state_equal(expected.state_dict(), actual.state_dict())
        values = torch.randn(4, 19, 16)
        mask = torch.arange(19)[None, :] < torch.tensor([17, 10, 6, 2])[:, None]
        with torch.no_grad():
            for before, after in zip(expected(values, mask), actual(values, mask)):
                torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_two_reads_and_one_ffn_have_live_gradients_without_tail_or_padding_leakage(self):
        torch.manual_seed(193)
        compressor = self.compressor()
        calls = {"first": 0, "second": 0, "ffn": 0}

        def count(name):
            def hook(module, arguments, output):
                calls[name] += 1
            return hook

        handles = [module.register_forward_hook(count(name)) for name, module in
            (("first", compressor.cross_attention), ("second", compressor.extra_cross_attentions[0]),
             ("ffn", compressor.ffn))]
        values = torch.randn(2, 21, 16, requires_grad=True)
        mask = torch.arange(21)[None, :] < torch.tensor([17, 10])[:, None]
        output, output_mask, sources = compressor(values, mask)
        for handle in handles:
            handle.remove()
        self.assertEqual(calls, {"first": 1, "second": 1, "ffn": 1})
        loss = (output[:, :2].square() * torch.linspace(.2, 1.8, 16)).sum()
        loss.backward()
        for module in (compressor.cross_attention, compressor.extra_cross_attentions[0],
                       compressor.ffn, compressor.query_generator):
            self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in module.parameters()
                                   if parameter.grad is not None), 0)
        self.assertGreater(float(values.grad[0, :7].abs().sum()), 0)
        self.assertGreater(float(values.grad[0, 7:11].abs().sum()), 0)
        self.assertEqual(int(values.grad[0, 11:].count_nonzero()), 0)
        self.assertEqual(int(values.grad[1, 4:].count_nonzero()), 0)
        changed = values.detach().clone()
        changed[0, 11:] = torch.randn_like(changed[0, 11:]) * 100
        changed[1, 4:] = torch.randn_like(changed[1, 4:]) * 100
        with torch.no_grad():
            altered = compressor(changed, mask)
        torch.testing.assert_close(output[:, :2], altered[0][:, :2], rtol=0, atol=0)
        self.assertEqual(sources[:, :2].tolist(), [[10, 10], [3, 3]])
        self.assertEqual(int(output[~output_mask].count_nonzero()), 0)

    def test_two_reads_preserve_ragged_short_histories_observed_zero_and_batching(self):
        torch.manual_seed(197)
        compressor = self.compressor()
        lengths = torch.tensor([17, 11, 7, 6, 2])
        values = torch.randn(5, 21, 16)
        values[0, 8] = 0  # Observed OOV remains an occupied event.
        mask = torch.arange(21)[None, :] < lengths[:, None]
        changed = values.clone()
        changed[~mask] = torch.randn_like(changed[~mask]) * 1000
        with torch.no_grad():
            output, output_mask, sources = compressor(values, mask)
            altered = compressor(changed, mask)
            self.assertTrue(bool(torch.isfinite(output).all()))
            torch.testing.assert_close(output, altered[0], rtol=0, atol=0)
            for row, length in enumerate(lengths.tolist()):
                solo = compressor(values[row:row + 1, :length], mask[row:row + 1, :length])
                occupied = int(output_mask[row].sum())
                torch.testing.assert_close(output[row, :occupied], solo[0][0], rtol=2e-5, atol=2e-6)
                self.assertTrue(torch.equal(sources[row, :occupied], solo[2][0]))
                if length <= 6:
                    torch.testing.assert_close(output[row, :occupied], values[row, :length], rtol=0, atol=0)
        self.assertEqual(int(output[~output_mask].count_nonzero()), 0)
        self.assertTrue(bool((sources[~output_mask] == -1).all()))

    def test_two_reads_preserve_every_raw_tail_causal_prediction(self):
        from rhpr.sasrec.model import SASRecModel

        torch.manual_seed(199)
        model = SASRecModel(num_items=80, max_seq_len=32, d_model=16,
            n_heads=4, n_layers=2, ffn_multiplier=4, dropout=0).eval()
        model.compressor = self.compressor()
        items = torch.arange(1, 18)[None, :]
        mask = torch.ones_like(items, dtype=torch.bool)
        with torch.no_grad():
            output, output_mask, sources = model.encode(items, mask)
            for source in range(11, 16):
                changed = items.clone()
                changed[:, source + 1:] += 30
                after, after_mask, after_sources = model.encode(changed, mask)
                self.assertTrue(torch.equal(output_mask, after_mask))
                self.assertTrue(torch.equal(sources, after_sources))
                available = output_mask & (sources <= source)
                torch.testing.assert_close(output[available], after[available], rtol=0, atol=0)

    def test_ssm_dimensions_preserve_small_step_initialization_and_backward(self):
        from rhpr.sasrec.query_architecture_models import build_architecture_model
        from rhpr.sasrec.query_proxy_protocol import hpo_variants

        config = self.config()
        variant = hpo_variants(config)[1]
        for trial in config["architecture_trials"][variant["name"]][1:]:
            with self.subTest(trial=trial["name"]):
                compressor = build_architecture_model(config, variant, trial, 80).compressor
                self.assertEqual(compressor.A_log.shape, (trial["d_inner"], trial["d_state"]))
                initial_dt = torch.nn.functional.softplus(compressor.delta_projection.bias.detach())
                self.assertGreaterEqual(float(initial_dt.min()), 1e-4 - 1e-9)
                self.assertLessEqual(float(initial_dt.max()), .1)
                values = torch.randn(1, 130, 128, requires_grad=True)
                output, _, _ = compressor(values, torch.ones((1, 130), dtype=torch.bool))
                (output[:, :2].square() * torch.linspace(.2, 1.8, 128)).sum().backward()
                self.assertGreater(float(values.grad[:, :2].abs().sum()), 0)
                self.assertEqual(int(values.grad[:, 2:].count_nonzero()), 0)
                for name in ("A_log", "value_projection.weight", "delta_projection.weight",
                             "write_projection.weight", "read_projection.weight"):
                    gradient = dict(compressor.named_parameters())[name].grad
                    self.assertIsNotNone(gradient)
                    self.assertTrue(bool(torch.isfinite(gradient).all()))
                    self.assertGreater(float(gradient.abs().sum()), 0)

    def test_unregistered_trial_and_learning_rate_drift_fail_before_build(self):
        from rhpr.sasrec.query_architecture_models import build_architecture_model
        from rhpr.sasrec.query_proxy_protocol import hpo_variants

        config = self.config()
        variant = hpo_variants(config)[0]
        trial = config["architecture_trials"][variant["name"]][0]
        with self.assertRaisesRegex(ValueError, "registered trial"):
            build_architecture_model(config, variant, dict(trial, name="foreign"), 80)
        changed = copy.deepcopy(config)
        changed["architecture_trials"][variant["name"]][0]["compressor_lr"] = .003
        with self.assertRaisesRegex(ValueError, "fix both learning rates"):
            build_architecture_model(changed, variant,
                changed["architecture_trials"][variant["name"]][0], 80)
        for depth, heads in ((0, 4), (3, 4), (True, 4), (1, 3), (1, True)):
            with self.subTest(depth=depth, heads=heads), self.assertRaises(ValueError):
                self.compressor(depth=depth, heads=heads)


if __name__ == "__main__":
    unittest.main()
