from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
    import torch
except ImportError:  # pragma: no cover
    np = torch = None


@unittest.skipIf(torch is None, "Torch and numpy are required")
class QueryPrefixProtocolTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def config(self):
        return json.loads((ROOT / "experiments/yambda_sasrec_prefix_query_pilot.json").read_text())

    def small_config(self):
        config = self.config()
        config["model"].update(d_model=8, n_layers=1, n_heads=2, dropout=0.)
        config["compressor"].update(d_inner=4, d_state=3, gate_rank=3, dropout=0.)
        config["training"].update(raw_context_length=512, serving_context_length=512,
                                  num_negatives=7, amp=False)
        return config

    def cohort(self):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser

        users = tuple(LongHistoryUser(uid, (np.arange(length) % 41 + 1).astype(np.int32),
                         np.arange(length, dtype=np.uint32)) for uid, length in ((17, 279), (4, 301)))
        return LongHistoryCohort(users, np.arange(41, dtype=np.int64), {})

    def test_registered_config_prevents_protocol_and_runtime_drift(self):
        from rhpr.sasrec.query_prefix_protocol import prefix_variants, validate_prefix_config

        config = self.config()
        validate_prefix_config(config)
        self.assertEqual(config["variants"], [variant["name"] for variant in prefix_variants(config)])
        for path, value in ((('kind',), 'yambda_query_ssm_block_pilot'),
                            (('seed',), True), (('summary_budgets',), [16]),
                            (('training', 'raw_context_length'), 16384),
                            (('training', 'serving_context_length'), 16384),
                            (('training', 'old_prefix_length'), 8000),
                            (('training', 'raw_tail_size'), 159),
                            (('training', 'supervised_targets'), 32),
                            (('training', 'early_stopping_patience'), 0),
                            (('training', 'physical_batch_sizes'), [3]),
                            (('training', 'max_passes'), 101),
                            (('training', 'budget_seconds'), float('inf')),
                            (('compressor', 'query_source'), 'recent_start'),
                            (('compressor', 'query_context_size'), 159),
                            (('model', 'use_time_features'), True),
                            (('model', 'n_layers'), 4),
                            (('runtime', 'target_torch_version'), '2.11.0'),
                            (('data', 'evaluation_content_sha256'), 'invalid'),
                            (('variants',), config['variants'][::-1])):
            bad = copy.deepcopy(config)
            cursor = bad
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_prefix_config(bad)

    def test_backbone_rng_is_shared_and_special_dt_initialization_survives(self):
        from rhpr.sasrec.query_prefix_protocol import build_prefix_model, prefix_variants

        config = self.small_config()
        models, rng = [], []
        for variant in prefix_variants(config):
            models.append(build_prefix_model(config, variant, 41))
            rng.append(torch.get_rng_state())
        backbone = {key: tensor for key, tensor in models[0].state_dict().items()
                    if not key.startswith("compressor.")}
        for model, state in zip(models[1:], rng[1:]):
            self.assertTrue(torch.equal(rng[0], state))
            for key, value in backbone.items():
                torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
            self.assertEqual(model.compressor.query_source, "old_suffix")
            self.assertEqual(model.compressor.tail_size, 128)
        for key, value in models[1].compressor.query_generator.state_dict().items():
            torch.testing.assert_close(models[2].compressor.query_generator.state_dict()[key], value, rtol=0, atol=0)
        ssm = models[2].compressor
        dt = torch.nn.functional.softplus(ssm.delta_projection.bias)
        self.assertTrue(bool((dt >= config["compressor"]["dt_min"] * .999).all()))
        self.assertTrue(bool((dt <= config["compressor"]["dt_max"] * 1.001).all()))
        self.assertGreater(float(ssm.delta_projection.bias.abs().min()), 1.)
        with self.assertRaisesRegex(ValueError, "unregistered"):
            build_prefix_model(config, {"kind": "uniform", "summary_tokens": 4}, 41)

    def test_passes_repeat_final_targets_but_change_order_and_negative_stream(self):
        from rhpr.sasrec.long_history_data import LongHistoryCohort
        from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule, prefix_schedule_summary

        config, cohort = self.small_config(), self.cohort()
        a = fixed_prefix_schedule(cohort, seed=42, epoch=1)
        b = fixed_prefix_schedule(cohort, seed=42, epoch=2)
        reversed_cohort = LongHistoryCohort(cohort.users[::-1], cohort.catalog_item_ids, {})
        reordered = fixed_prefix_schedule(reversed_cohort, seed=42, epoch=1)
        identity = lambda rows: [(w.user_id, w.end, w.negative_seed) for w in rows]
        self.assertEqual(identity(a), identity(reordered))
        self.assertEqual({w.user_id: w.end for w in a}, {w.user_id: w.end for w in b})
        self.assertTrue(all(w.end == len(cohort.users[w.user_index].train) - 1 for w in a))
        self.assertNotEqual({w.user_id: w.negative_seed for w in a},
                            {w.user_id: w.negative_seed for w in b})
        first, second = (prefix_schedule_summary(cohort, schedule, config) for schedule in (a, b))
        self.assertEqual(first["fixed_target_positions_sha256"], second["fixed_target_positions_sha256"])
        self.assertNotEqual(first["schedule_sha256"], second["schedule_sha256"])
        self.assertEqual(first["target_presentations"], 256)
        self.assertFalse(first["train_is_exhaustive_event_epoch"])

    def test_full_input_is_right_padded_and_recent_has_identical_last128_targets(self):
        from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule, prefix_batch_arrays, prefix_variants

        config, cohort = self.small_config(), self.cohort()
        cohort.users[0].train[7] = 0  # An occupied OOV observation must not become padding.
        schedule = fixed_prefix_schedule(cohort, seed=42, epoch=1)
        recent, ca, ssm = [prefix_batch_arrays(cohort, schedule, config, variant)
                           for variant in prefix_variants(config)]
        self.assertEqual(ca["item_ids"].shape, (2, 300))
        self.assertEqual(recent["item_ids"].shape, (2, 128))
        for index, window in enumerate(schedule):
            user = cohort.users[window.user_index]
            np.testing.assert_array_equal(ca["item_ids"][index, :window.end], user.train[:-1])
            np.testing.assert_array_equal(ca["targets"][index, :window.end], user.train[1:])
            self.assertTrue(bool(ca["valid_mask"][index, :window.end].all()))
            self.assertFalse(bool(ca["valid_mask"][index, window.end:].any()))
            np.testing.assert_array_equal(recent["item_ids"][index], user.train[-129:-1])
            np.testing.assert_array_equal(recent["targets"][index], user.train[-128:])
        for key in ca:
            np.testing.assert_array_equal(ca[key], ssm[key])
        np.testing.assert_array_equal(ca["negative_ids"], recent["negative_ids"])

    def test_safety_cap_rejects_instead_of_truncating_and_window_must_be_final(self):
        from dataclasses import replace
        from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule, prefix_batch_arrays, prefix_variants

        config, cohort = self.small_config(), self.cohort()
        schedule = fixed_prefix_schedule(cohort, seed=42, epoch=1)
        variant = prefix_variants(config)[1]
        config["training"]["raw_context_length"] = 299
        with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
            prefix_batch_arrays(cohort, schedule, config, variant)
        config["training"]["raw_context_length"] = 512
        with self.assertRaisesRegex(ValueError, "final train-only"):
            prefix_batch_arrays(cohort, [replace(schedule[0], end=schedule[0].end - 1)], config, variant)

    def test_128_loss_states_align_after_variable_length_packing(self):
        from rhpr.sasrec.long_training_profile import _loss
        from rhpr.sasrec.query_prefix_protocol import (
            build_prefix_model, fixed_prefix_schedule, prefix_batch_arrays, prefix_variants,
        )

        config, cohort = self.small_config(), self.cohort()
        schedule = fixed_prefix_schedule(cohort, seed=42, epoch=1)
        for variant in prefix_variants(config):
            with self.subTest(variant=variant["name"]):
                model = build_prefix_model(config, variant, cohort.num_items).eval()
                batch = {key: torch.from_numpy(value) for key, value in
                         prefix_batch_arrays(cohort, schedule, config, variant).items()}
                states, mask, sources = model.encode(batch["item_ids"], batch["valid_mask"])
                states.retain_grad()
                for index, window in enumerate(schedule):
                    end = 128 if variant["kind"] == "recent" else window.end
                    self.assertEqual(sources[index, -128:].tolist(), list(range(end - 128, end)))
                    self.assertTrue(bool(mask[index, -128:].all()))
                    if variant["kind"] != "recent":
                        self.assertEqual(sources[index, :4].tolist(), [window.end - 129] * 4)
                loss = _loss(model, (states, mask, sources), batch, tail_size=128, num_negatives=7)
                positives = torch.stack([torch.from_numpy(cohort.users[w.user_index].train[-128:].astype(np.int64))
                                         for w in schedule])
                selected = states[:, -128:]
                positive_scores = (selected * model.item_embedding(positives)).sum(-1, keepdim=True)
                negative_scores = torch.einsum("bld,bkd->blk", selected,
                                                model.item_embedding(batch["negative_ids"]))
                collisions = batch["negative_ids"][:, None, :] == positives[:, :, None]
                negative_scores = negative_scores.masked_fill(collisions, torch.finfo(negative_scores.dtype).min)
                expected = torch.nn.functional.cross_entropy(
                    torch.cat((positive_scores, negative_scores), dim=-1).flatten(0, 1),
                    torch.zeros(256, dtype=torch.long))
                torch.testing.assert_close(loss, expected, rtol=0, atol=0)
                loss.backward()
                self.assertTrue(bool(torch.isfinite(model.item_embedding.weight.grad).all()))
                self.assertEqual(int(states.grad[:, :-128].count_nonzero()), 0)
                if variant["kind"] != "recent":
                    self.assertGreater(float(model.compressor.query_generator.queries.grad.abs().sum()), 0.)

    def test_summaries_ignore_raw_tail_and_causal_states_ignore_later_tail_values(self):
        from rhpr.sasrec.query_prefix_protocol import build_prefix_model, prefix_variants

        config = self.small_config()
        for variant in prefix_variants(config)[1:]:
            with self.subTest(variant=variant["name"]):
                model = build_prefix_model(config, variant, 41).eval()
                items = (torch.arange(280)[None, :] % 41 + 1).long()
                changed = items.clone()
                changed[:, -100:] = changed[:, -100:] % 41 + 1
                mask = torch.ones_like(items, dtype=torch.bool)
                embeddings = model.item_embedding(items)
                alternate = model.item_embedding(changed)
                original_z = model.compressor(embeddings, mask)[0][:, :4]
                changed_z = model.compressor(alternate, mask)[0][:, :4]
                torch.testing.assert_close(original_z, changed_z, rtol=0, atol=0)
                original_states = model.encode(items, mask)[0]
                changed_states = model.encode(changed, mask)[0]
                # Fixed boundary and full sequence length: the earliest28 raw
                # states cannot read the100 changed future raw values.
                torch.testing.assert_close(original_states[:, :32], changed_states[:, :32], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
