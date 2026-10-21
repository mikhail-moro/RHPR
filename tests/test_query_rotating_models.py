from __future__ import annotations

from contextlib import ExitStack
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "Torch is required")
class QueryRotatingModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def config(self):
        return json.loads((ROOT / "experiments/yambda_query_rotating_windows.json").read_text())

    def arms(self, config):
        from rhpr.sasrec.query_rotating_protocol import rotating_trials, rotating_variants
        return [(variant, rotating_trials(config, variant)[1]) for variant in rotating_variants(config)]

    def assert_state_equal(self, expected, actual):
        self.assertEqual(expected.keys(), actual.keys())
        for name, value in expected.items():
            torch.testing.assert_close(value, actual[name], rtol=0, atol=0, msg=name)

    def test_fresh_factories_preserve_each_original_initializer_and_rng_without_loading_weights(self):
        from rhpr.sasrec.query_history_models import build_history_model
        from rhpr.sasrec.query_rotating_models import build_rotating_model

        config = self.config()
        for family, (variant, trial) in zip(("qssm", "fuxi"), self.arms(config), strict=True):
            with self.subTest(family=family):
                expected = build_history_model(config["history_sources"][family], 20)
                expected_rng = torch.get_rng_state().clone()
                with patch("torch.load", side_effect=AssertionError("fresh models must not read checkpoints")):
                    actual = build_rotating_model(config, variant, trial, 20)
                self.assert_state_equal(expected.state_dict(), actual.state_dict())
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                self.assertFalse(actual.history_source_checkpoint_loaded)
                self.assertFalse(actual.rotating_metadata["checkpoint_weights_loaded"])
                self.assertEqual(actual.rotating_metadata["initial_position_table_length"], 32768)
                self.assertEqual(actual.rotating_metadata["historical_fixed_precision_identical"], family == "qssm")
                if family == "qssm":
                    self.assertEqual(actual.compressor.summary_tokens, 32)
                    self.assertEqual(actual.position_embedding.num_embeddings, 32768)
                    self.assertEqual(actual.compressor.query_source, "old_suffix")
                else:
                    self.assertEqual(actual.encoder.layers[0]._channel_p._emb.shape, (32768, 32))

    def test_factory_rejects_historical_training_changed_seed_and_unregistered_learning_rates(self):
        from rhpr.sasrec.query_rotating_models import build_rotating_model
        from rhpr.sasrec.query_rotating_protocol import rotating_trials

        config = self.config()
        variant, trial = self.arms(config)[0]
        with self.assertRaisesRegex(ValueError, "saved comparison"):
            build_rotating_model(config, variant, rotating_trials(config, variant)[0], 20)
        with self.assertRaises(ValueError):
            build_rotating_model(config, variant, dict(trial, compressor_lr=.003), 20)
        config["history_sources"]["qssm"]["seed"] = 7
        with self.assertRaisesRegex(ValueError, "seed"):
            build_rotating_model(config, variant, trial, 20)

    def test_optimizer_starts_empty_owns_all_parameters_and_preserves_component_groups(self):
        from rhpr.sasrec.query_rotating_models import build_rotating_model, rotating_optimizer

        config = self.config()
        for variant, trial in self.arms(config):
            model = build_rotating_model(config, variant, trial, 20)
            optimizer = rotating_optimizer(model, config["training"], trial)
            self.assertFalse(optimizer.state)
            owned = [id(value) for group in optimizer.param_groups for value in group["params"]]
            self.assertEqual(len(owned), len(set(owned)))
            self.assertEqual(set(owned), {id(value) for value in model.parameters() if value.requires_grad})
            self.assertEqual([group["component"] for group in optimizer.param_groups],
                             ["backbone", "compressor"] if variant["kind"] == "query_ssm" else ["backbone"])
            self.assertTrue(all(group["lr"] == .001 and group["weight_decay"] == .00001
                                and not group["foreach"] for group in optimizer.param_groups))
        with self.assertRaises(ValueError):
            rotating_optimizer(model, config["training"], dict(trial, name="historical_fixed"))

    def cohort(self):
        return SimpleNamespace(num_items=20, users=[SimpleNamespace(user_id=uid,
            train=np.arange(length, dtype=np.int64) % 20 + 1)
            for uid, length in ((7, 2305), (11, 2817))])

    def test_both_real_encoders_keep_last128_sources_aligned_to_the_common_shifted_targets(self):
        from rhpr.sasrec.query_rotating_models import build_rotating_model
        from rhpr.sasrec.query_rotating_protocol import rotating_batch_arrays, rotating_schedule

        config, cohort = self.config(), self.cohort()
        schedule = rotating_schedule(cohort, seed=42, epoch=2)
        reference = None
        for variant, trial in self.arms(config):
            arrays = rotating_batch_arrays(cohort, schedule[:1], config, variant, old_length=1024)
            if reference is not None:
                for key in reference:
                    np.testing.assert_array_equal(reference[key], arrays[key])
            reference = arrays
            model = build_rotating_model(config, variant, trial, cohort.num_items).eval()
            ids = torch.from_numpy(arrays["item_ids"])
            with torch.no_grad():
                states, valid, sources = model.encode(ids, torch.from_numpy(arrays["valid_mask"]))
            self.assertTrue(bool(torch.isfinite(states).all()))
            selected_sources = sources[0, valid[0]][-128:]
            self.assertEqual(selected_sources.tolist(), list(range(1024, 1152)))
            np.testing.assert_array_equal(arrays["targets"][0, selected_sources],
                                          cohort.users[schedule[0].user_index].train[schedule[0].end - 127:schedule[0].end + 1])
            self.assertEqual(int(valid.sum()), 160 if variant["kind"] == "query_ssm" else 1152)

    def tiny_model(self, family):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.item_embedding = torch.nn.Embedding(21, 4, padding_idx=0)
                self.projection = torch.nn.Linear(4, 4)
                self.architecture = "sasrec" if family == "qssm" else "fuxi_linear"
                self.num_items, self.max_seq_len, self.d_model = 20, 32768, 4
                self.rotating_metadata = {"checkpoint_weights_loaded": False, "family": family}

            def encode(self, ids, valid):
                states = self.projection(self.item_embedding(ids))
                sources = torch.arange(ids.shape[1]).expand_as(ids).masked_fill(~valid, -1)
                return states, valid, sources

        return TinyModel()

    def _preflight(self, family, *, bad_updates=False, nonfinite_evaluation=False):
        from rhpr.sasrec.query_rotating_models import preflight_rotating
        from rhpr.sasrec.query_rotating_protocol import rotating_batch_arrays

        config, cohort = self.config(), self.cohort()
        variant, trial = self.arms(config)[0 if family == "qssm" else 1]
        config["training"].update(amp=False, effective_batch_size=2, num_negatives=2)
        config["rotating_batches"][variant["name"]]["training"] = 1
        model = self.tiny_model(family)
        before = copy.deepcopy(model.state_dict())
        seen, eval_shapes = [], []

        def arrays(*args, **kwargs):
            result = rotating_batch_arrays(*args, **kwargs)
            seen.append(result)
            return result

        def daily(actual_model, *args, context_length, batch_size, **kwargs):
            self.assertIs(actual_model, model)
            self.assertFalse(all(torch.equal(value, model.state_dict()[name]) for name, value in before.items()))
            eval_shapes.append((context_length, batch_size))
            if nonfinite_evaluation:
                raise RuntimeError("evaluation produced nonfinite catalogue scores")
            return {"anchors_per_second": 75.}

        module = "rhpr.sasrec.query_rotating_models."
        with ExitStack() as stack:
            stack.enter_context(patch(module + "build_rotating_model", return_value=model))
            stack.enter_context(patch("rhpr.sasrec.query_rotating_protocol.rotating_batch_arrays", side_effect=arrays))
            stack.enter_context(patch(module + "_profile_daily_rate", side_effect=daily))
            stack.enter_context(patch(module + "_observe_prefix_flops", return_value=(None, {"history_flops": 123})))
            major = stack.enter_context(patch(module + "_major_flops", return_value={"per_user": 456, "scope": "all_native_channels"}))
            if bad_updates:
                stack.enter_context(patch(module + "train_prefix_pass", return_value={
                    "successful_updates": 0, "amp_skipped_update_attempts": 1}))
            row = preflight_rotating(config, cohort, None, variant, trial, torch.device("cpu"))
            self.assertEqual(major.call_count, 0 if family == "qssm" else 1)
        return row, seen, eval_shapes

    def test_preflight_performs_two_real_effective_adam_updates_at1152_and_keeps_catalogue_shape(self):
        for family in ("qssm", "fuxi"):
            with self.subTest(family=family):
                row, seen, eval_shapes = self._preflight(family)
                self.assertEqual(len(seen), 4)  # Two users x two effective updates.
                self.assertTrue(all(arrays["item_ids"].shape == (1, 1152) for arrays in seen))
                self.assertEqual(eval_shapes, [(1152, 32)])
                self.assertEqual(row["warmup"]["successful_updates"], 1)
                self.assertEqual(row["training"]["successful_updates"], 1)
                self.assertEqual(row["training"]["amp_skipped_update_attempts"], 0)
                self.assertEqual(row["representative_b1_arithmetic"]["proxy"]["input_tokens"], 1152)
                self.assertFalse(row["source_checkpoint_weights_loaded"])
                self.assertTrue(row["fresh_initializer_profiled"])
                self.assertFalse(row["profile_weights_reused"])
                self.assertFalse(row["cross_family_flop_scopes_identical"])
                self.assertFalse(row["full_evaluation"]["profiled"])

    def test_preflight_isolates_rng_and_rejects_skipped_updates_or_nonfinite_catalogue(self):
        random.seed(301)
        np.random.seed(302)
        torch.manual_seed(303)
        # Fixture construction precedes the boundary; the builder is responsible
        # for resetting its own original initializer, not the test fixture RNG.
        from rhpr.sasrec.query_rotating_models import preflight_rotating
        config, cohort = self.config(), self.cohort()
        variant, trial = self.arms(config)[0]
        python_before, numpy_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()

        def failure(*args, **kwargs):
            random.random()
            np.random.random()
            torch.rand(1)
            raise RuntimeError("deliberate builder failure")

        with patch("rhpr.sasrec.query_rotating_models.build_rotating_model", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "deliberate builder failure"):
                preflight_rotating(config, cohort, None, variant, trial, torch.device("cpu"))
        self.assertEqual(python_before, random.getstate())
        np.testing.assert_array_equal(numpy_before[1], np.random.get_state()[1])
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        with self.assertRaisesRegex(FloatingPointError, "two effective Adam"):
            self._preflight("qssm", bad_updates=True)
        with self.assertRaisesRegex(FloatingPointError, "nonfinite catalogue"):
            self._preflight("fuxi", nonfinite_evaluation=True)

    def test_budget_prices_both_fresh_shapes_at_maximum_without_retraining_historical_controls(self):
        from rhpr.sasrec.query_rotating_models import conservative_rotating_budget

        config = self.config()
        profiles = {variant["name"] + "-rotating": {"status": "success",
            "training": {"windows_per_second": train}, "proxy_evaluation": {"anchors_per_second": rate}}
            for (variant, _), train, rate in zip(self.arms(config), (100., 80.), (200., 160.), strict=True)}
        row = conservative_rotating_budget(config, profiles, 100., 4427, {"validation_anchors": 26335})
        work = sum(60 * 4427 / train + 6 * 26335 / rate
                   for train, rate in ((100., 200.), (80., 160.)))
        self.assertAlmostEqual(row["estimated_remaining_work_seconds"], work)
        self.assertAlmostEqual(row["conservative_total_seconds"], 100 + 1.25 * (work + 240))
        self.assertEqual(row["new_training_configurations"], 2)
        self.assertEqual(row["reused_controls"], 2)
        self.assertEqual(row["priced_fresh_shapes"], 2)
        self.assertEqual(row["test_evaluations"], 0)
        self.assertFalse(row["auto_reduces_trials_or_passes"])
        first = next(iter(profiles))
        profiles[first] = {"status": "preflight_numerical_failed"}
        failed = conservative_rotating_budget(config, profiles, 100., 4427, {"validation_anchors": 26335})
        self.assertEqual(failed["preflight_failed_shapes"], 1)
        self.assertEqual(failed["priced_fresh_shapes"], 1)
        for rate in (0., float("nan"), True):
            last = next(key for key in profiles if key != first)
            profiles[last]["training"]["windows_per_second"] = rate
            with self.assertRaisesRegex(ValueError, "positive finite"):
                conservative_rotating_budget(config, profiles, 100., 4427, {"validation_anchors": 26335})


if __name__ == "__main__":
    unittest.main()
