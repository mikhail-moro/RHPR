from __future__ import annotations

import copy
from dataclasses import replace
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
class QueryOnlyProtocolTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def config(self):
        from rhpr.sasrec.query_only_protocol import QUERY_ONLY_KIND

        parent = json.loads((ROOT / "experiments/yambda_sasrec_prefix_query_pilot.json").read_text())
        parent["model"].update(d_model=8, n_layers=1, n_heads=2, dropout=0.)
        parent["compressor"].update(d_inner=4, d_state=3, gate_rank=3, dropout=0.)
        parent["training"].update(num_negatives=7, amp=False)
        return {"kind": QUERY_ONLY_KIND, "parent": {"experiment_config": parent},
                "training": {**parent["training"], "max_passes": 30}}

    def cohort(self):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser

        users = tuple(LongHistoryUser(uid, (np.arange(length) % 41 + 1).astype(np.int32),
                      np.arange(length, dtype=np.uint32)) for uid, length in ((17, 279), (4, 301)))
        return LongHistoryCohort(users, np.arange(41, dtype=np.int64), {})

    def test_shared_initialization_and_rng_keep_the_parent_position_table(self):
        from rhpr.sasrec.query_only_protocol import build_query_only_model, query_only_variant
        from rhpr.sasrec.query_prefix_protocol import build_prefix_model, prefix_variants

        config = self.config()
        before = copy.deepcopy(config)
        parent = config["parent"]["experiment_config"]
        models, rng = [], []
        for variant in prefix_variants(parent):
            models.append(build_prefix_model(parent, variant, 41))
            rng.append(torch.get_rng_state())
        short = build_query_only_model(config, query_only_variant(), 41)
        self.assertEqual(short.max_seq_len, 32768)
        self.assertEqual(short.position_embedding.num_embeddings, 32768)
        self.assertEqual(config, before)
        for key, value in short.state_dict().items():
            if not key.startswith("compressor."):
                for model in models:
                    torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
            else:
                for model in models[1:]:
                    torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        for state in rng:
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertFalse(hasattr(short.compressor, "memory_norm"))
        self.assertFalse(hasattr(short.compressor, "readout_norm"))
        self.assertFalse(hasattr(short.compressor, "cross_attention"))
        self.assertFalse(hasattr(short.compressor, "value_projection"))

    def test_crop_before_lookup_preserves_original_targets_negatives_and_windows(self):
        from rhpr.sasrec.query_only_protocol import query_only_batch_arrays, query_only_variant
        from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule, prefix_batch_arrays, prefix_variants

        config, cohort = self.config(), self.cohort()
        parent = config["parent"]["experiment_config"]
        cohort.users[0].train[-200] = 0  # Occupied OOV query event remains valid.
        schedule = fixed_prefix_schedule(cohort, seed=42, epoch=7)
        original = tuple(schedule)
        short = query_only_batch_arrays(cohort, schedule, config, query_only_variant())
        self.assertEqual(short["item_ids"].shape, (2, 256))
        self.assertTrue(bool(short["valid_mask"].all()))
        for variant in prefix_variants(parent):
            full = prefix_batch_arrays(cohort, schedule, parent, variant)
            np.testing.assert_array_equal(short["negative_ids"], full["negative_ids"])
            for index, window in enumerate(schedule):
                end = 128 if variant["kind"] == "recent" else window.end
                np.testing.assert_array_equal(short["targets"][index, -128:],
                                              full["targets"][index, end - 128:end])
        for index, window in enumerate(schedule):
            user = cohort.users[window.user_index]
            np.testing.assert_array_equal(short["item_ids"][index], user.train[-257:-1])
            np.testing.assert_array_equal(short["targets"][index], user.train[-256:])
            self.assertEqual(window.end, len(user.train) - 1)
        self.assertEqual(schedule, original)
        self.assertEqual([len(user.train) for user in cohort.users], [279, 301])

    def test_all128_supervised_states_match_manual_sampled_softmax(self):
        from rhpr.sasrec.long_training_profile import _loss
        from rhpr.sasrec.query_only_protocol import (
            build_query_only_model, query_only_batch_arrays, query_only_variant,
        )
        from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule

        config, cohort = self.config(), self.cohort()
        variant = query_only_variant()
        schedule = fixed_prefix_schedule(cohort, seed=42, epoch=1)
        batch = {key: torch.from_numpy(value) for key, value in
                 query_only_batch_arrays(cohort, schedule, config, variant).items()}
        model = build_query_only_model(config, variant, cohort.num_items).eval()
        states, mask, sources = model.encode(batch["item_ids"], batch["valid_mask"])
        self.assertEqual(states.shape, (2, 132, 8))
        self.assertEqual(sources[0, :4].tolist(), [127] * 4)
        self.assertEqual(sources[0, -128:].tolist(), list(range(128, 256)))
        states.retain_grad()
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
        self.assertEqual(int(states.grad[:, :4].count_nonzero()), 0)
        self.assertGreater(float(model.compressor.query_generator.queries.grad.abs().sum()), 0.)
        self.assertGreater(float(model.compressor.ffn[0].weight.grad.abs().sum()), 0.)
        self.assertTrue(bool(torch.isfinite(model.item_embedding.weight.grad).all()))

    def test_remote_history_is_ignored_and_summary_gradients_use_only_old_queries(self):
        from rhpr.sasrec.query_only_protocol import QueryOnlyPrefixCompressor

        torch.manual_seed(17)
        comp = QueryOnlyPrefixCompressor(8, summary_tokens=2, tail_size=4,
                                         query_context_size=3, n_heads=2).eval()
        inputs = torch.randn(1, 12, 8, requires_grad=True)
        mask = torch.ones((1, 12), dtype=torch.bool)
        original = comp(inputs, mask)[0]
        changed = inputs.detach().clone()
        changed[:, :5] = torch.randn_like(changed[:, :5]) * 100
        torch.testing.assert_close(comp(changed, mask)[0], original, rtol=0, atol=0)
        original[:, :2].square().sum().backward()
        self.assertEqual(int(inputs.grad[:, :5].count_nonzero()), 0)
        self.assertGreater(float(inputs.grad[:, 5:8].abs().sum()), 0.)
        self.assertEqual(int(inputs.grad[:, 8:].count_nonzero()), 0)
        inputs.grad.zero_()
        comp(inputs, mask)[0].square().sum().backward()
        self.assertGreater(float(inputs.grad[:, 8:].abs().sum()), 0.)

    def test_full_input_and_256_crop_have_identical_states_and_raw_tail_is_causal(self):
        from rhpr.sasrec.query_only_protocol import build_query_only_model, query_only_variant

        model = build_query_only_model(self.config(), query_only_variant(), 41).eval()
        items = (torch.arange(300)[None, :] % 41 + 1).long()
        full = model.encode(items, torch.ones_like(items, dtype=torch.bool))[0]
        cropped = items[:, -256:]
        crop_mask = torch.ones_like(cropped, dtype=torch.bool)
        short = model.encode(cropped, crop_mask)[0]
        torch.testing.assert_close(full, short, rtol=0, atol=0)
        alternate = cropped.clone()
        alternate[:, -100:] = alternate[:, -100:] % 41 + 1
        changed = model.encode(alternate, crop_mask)[0]
        # Four summaries and the first28 raw states cannot see100 changed future events.
        torch.testing.assert_close(changed[:, :32], short[:, :32], rtol=0, atol=0)
        self.assertGreater(float((changed[:, 32:] - short[:, 32:]).detach().abs().sum()), 0.)

    def test_short_ragged_rows_and_occupied_zero_embeddings_are_safe(self):
        from rhpr.sasrec.query_only_protocol import QueryOnlyPrefixCompressor

        torch.manual_seed(4)
        comp = QueryOnlyPrefixCompressor(8, summary_tokens=2, tail_size=4,
                                         query_context_size=3, n_heads=2).eval()
        inputs = torch.randn(3, 11, 8)
        lengths = [3, 6, 11]
        mask = torch.arange(11)[None, :] < torch.tensor(lengths)[:, None]
        inputs[1, 1] = 0  # A valid zero vector participates in the old query.
        values, valid, sources = comp(inputs, mask)
        self.assertEqual(valid.sum(1).tolist(), [3, 6, 6])
        self.assertEqual(sources[0, :3].tolist(), [0, 1, 2])
        self.assertEqual(sources[1, :2].tolist(), [1, 1])
        self.assertEqual(sources[2, :2].tolist(), [6, 6])
        self.assertTrue(bool(torch.isfinite(values).all()))
        self.assertEqual(int(values[~valid].count_nonzero()), 0)
        altered = inputs.clone()
        altered[~mask] = torch.randn_like(altered[~mask]) * 100
        torch.testing.assert_close(comp(altered, mask)[0], values, rtol=0, atol=0)
        for index, length in enumerate(lengths):
            single = comp(inputs[index:index + 1, :length], torch.ones((1, length), dtype=torch.bool))
            count = int(valid[index].sum())
            torch.testing.assert_close(single[0][0], values[index, :count], rtol=1e-6, atol=1e-6)
            self.assertEqual(single[2][0].tolist(), sources[index, :count].tolist())

    def test_nonfinal_windows_and_original_context_overflow_are_rejected(self):
        from rhpr.sasrec.query_only_protocol import (
            build_query_only_model, query_only_batch_arrays, query_only_variant,
        )
        from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule

        config, cohort = self.config(), self.cohort()
        schedule = fixed_prefix_schedule(cohort, seed=42, epoch=1)
        variant = query_only_variant()
        with self.assertRaisesRegex(ValueError, "original final"):
            query_only_batch_arrays(cohort, [replace(schedule[0], end=schedule[0].end - 1)], config, variant)
        bad = copy.deepcopy(config)
        bad["parent"]["experiment_config"]["training"]["raw_context_length"] = 256
        with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
            query_only_batch_arrays(cohort, schedule, bad, variant)
        with self.assertRaisesRegex(ValueError, "exactly query_only_m4"):
            build_query_only_model(config, {**variant, "summary_tokens": 16}, cohort.num_items)
        with self.assertRaisesRegex(ValueError, "nonempty physical batch"):
            query_only_batch_arrays(cohort, [], config, variant)


if __name__ == "__main__":
    unittest.main()
