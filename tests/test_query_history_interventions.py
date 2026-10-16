from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
    import torch
except ImportError:  # pragma: no cover
    np = torch = None


class ExampleHelpers:
    def example(self, *, length=43, uid=23, day=317):
        from rhpr.sasrec.daily_evaluation_data import DailyEvaluationExample

        context = (np.arange(length) % 41 + 1).astype(np.int32)
        # Occupied OOV slots occur in remote, query support, and raw tail.
        context[1] = context[-9] = context[-2] = 0
        return DailyEvaluationExample(uid, f"{uid}:{day}", day, context,
                                      np.arange(length, dtype=np.uint32),
                                      7, 1000, "a" * 64)

    def intervene(self, example, condition, **kwargs):
        from rhpr.sasrec.query_history_interventions import intervene_example

        return intervene_example(example, condition=condition, seed=42,
                                 tail_size=6, query_context_size=4, **kwargs)


@unittest.skipIf(np is None, "numpy is required")
class QueryHistoryInterventionsTest(ExampleHelpers, unittest.TestCase):
    def test_crop_preserves_complete_query_and_tail_including_occupied_oov(self):
        example = self.example()
        original_context, original_times = example.context.copy(), example.timestamps.copy()
        self.assertIs(self.intervene(example, "full"), example)
        cropped = self.intervene(example, "query_only_history")
        np.testing.assert_array_equal(cropped.context, original_context[-10:])
        np.testing.assert_array_equal(cropped.timestamps, original_times[-10:])
        self.assertEqual(int((cropped.context == 0).sum()), 2)
        self.assertFalse(np.shares_memory(cropped.context, example.context))
        self.assertFalse(np.shares_memory(cropped.timestamps, example.timestamps))
        for field in ("user_id", "anchor_id", "day", "target", "target_timestamp",
                      "reference_context_hash"):
            self.assertEqual(getattr(cropped, field), getattr(example, field))
        cropped.context[0] = 999
        np.testing.assert_array_equal(example.context, original_context)
        np.testing.assert_array_equal(example.timestamps, original_times)

    def test_remote_shuffle_keeps_support_multiset_q_r_and_time_slots(self):
        example = self.example()
        context, timestamps = example.context.copy(), example.timestamps.copy()
        shuffled = self.intervene(example, "shuffle_remote")
        self.assertEqual(shuffled.context.shape, context.shape)
        self.assertEqual(shuffled.context.dtype, context.dtype)
        self.assertFalse(np.array_equal(shuffled.context[:-10], context[:-10]))
        np.testing.assert_array_equal(np.sort(shuffled.context[:-10]), np.sort(context[:-10]))
        np.testing.assert_array_equal(shuffled.context[-10:], context[-10:])
        np.testing.assert_array_equal(shuffled.timestamps, timestamps)
        np.testing.assert_array_equal(example.context, context)
        self.assertEqual(shuffled.reference_context_hash, example.reference_context_hash)
        self.assertEqual(int((shuffled.context == 0).sum()), int((context == 0).sum()))

    def test_permutation_binds_only_seed_uid_and_anchor_and_does_not_use_global_rng(self):
        from rhpr.sasrec.query_history_interventions import intervene_example

        example = self.example()
        before_rng = np.random.get_state()
        original = self.intervene(example, "shuffle_remote")
        after_rng = np.random.get_state()
        for before, after in zip(before_rng, after_rng):
            if isinstance(before, np.ndarray):
                np.testing.assert_array_equal(before, after)
            else:
                self.assertEqual(before, after)
        # Target metadata must not even be converted to integers by the control.
        changed = replace(example, target=object(), target_timestamp=object(),
                          reference_context_hash="b" * 64,
                          timestamps=example.timestamps + 10000)
        np.testing.assert_array_equal(original.context,
                                      self.intervene(changed, "shuffle_remote").context)
        np.testing.assert_array_equal(original.context,
                                      self.intervene(example, "shuffle_remote").context)
        for changed in (replace(example, user_id=24), replace(example, anchor_id="23:318")):
            self.assertFalse(np.array_equal(original.context,
                                            self.intervene(changed, "shuffle_remote").context))
        other_seed = intervene_example(example, condition="shuffle_remote", seed=43,
                                       tail_size=6, query_context_size=4)
        self.assertFalse(np.array_equal(original.context, other_seed.context))

    def test_traversal_order_and_batch_boundaries_cannot_change_permutation(self):
        from rhpr.sasrec.query_history_interventions import iter_intervened_examples

        examples = [self.example(uid=uid, day=day) for uid, day in ((1, 317), (2, 318), (3, 317))]
        kwargs = dict(condition="shuffle_remote", seed=42, tail_size=6, query_context_size=4)
        forward = list(iter_intervened_examples(iter(examples), **kwargs))
        reverse = list(iter_intervened_examples(reversed(examples), **kwargs))
        split = list(iter_intervened_examples(examples[:1], **kwargs))
        split += list(iter_intervened_examples(examples[1:], **kwargs))
        self.assertEqual([row.anchor_id for row in forward], [row.anchor_id for row in examples])
        expected = {row.anchor_id: row.context for row in forward}
        for row in reverse + split:
            np.testing.assert_array_equal(row.context, expected[row.anchor_id])

    def test_zero_or_one_remote_slot_is_an_identity_control(self):
        for length in (10, 11):
            with self.subTest(length=length):
                example = self.example(length=length)
                self.assertIs(self.intervene(example, "shuffle_remote"), example)

    def test_declared_policy_is_json_safe_and_invalid_inputs_fail_closed(self):
        from rhpr.sasrec.query_history_interventions import (
            INTERVENTION_CONDITIONS, intervention_spec, intervene_example,
        )

        example = self.example()
        for condition in INTERVENTION_CONDITIONS:
            policy = intervention_spec(condition)
            self.assertTrue(policy["requires_time_features_off"])
            self.assertTrue(policy["model_weights_and_architecture_unchanged"])
            self.assertFalse(policy["target_values_or_timestamps_used_for_permutation"])
            self.assertEqual(json.loads(json.dumps(policy)), policy)
        for kwargs in ({"condition": "query_only"}, {"seed": True}, {"seed": -1},
                       {"tail_size": 0}, {"query_context_size": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                intervene_example(example, **(dict(condition="full", seed=42,
                                                    tail_size=6, query_context_size=4) | kwargs))
        for invalid in (replace(example, context=example.context.astype(np.float32)),
                        replace(example, context=example.context[:, None]),
                        replace(example, context=-np.ones(43, dtype=np.int32)),
                        replace(example, timestamps=example.timestamps[:-1]),
                        replace(example, user_id=True), replace(example, anchor_id=""),
                        self.example(length=9)):
            with self.subTest(invalid=type(invalid)), self.assertRaises(ValueError):
                self.intervene(invalid, "full")


@unittest.skipIf(torch is None, "Torch and numpy are required")
class QueryHistoryPositionControlTest(ExampleHelpers, unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def model(self, kind):
        from rhpr.sasrec.model import SASRecModel
        from rhpr.sasrec.query_ssm import make_query_compressor

        torch.manual_seed(208)
        model = SASRecModel(num_items=41, max_seq_len=64, d_model=8,
                            n_heads=2, n_layers=1, ffn_multiplier=4, dropout=0)
        model.compressor = make_query_compressor(kind=kind, d_model=8, summary_tokens=2,
            tail_size=6, query_context_size=4, n_heads=2, d_inner=4, d_state=3,
            gate_rank=3, dropout=0, query_source="old_suffix")
        return model.eval()

    def tensors(self, example):
        items = torch.tensor(example.context.astype(np.int64))[None, :]
        return items, torch.ones_like(items, dtype=torch.bool)

    def test_queries_retained_memory_and_predictor_positions_are_identical(self):
        from rhpr.sasrec.query_history_interventions import INTERVENTION_CONDITIONS

        example = self.example(length=23)
        for kind in ("query_cross_attention", "query_ssm"):
            with self.subTest(kind=kind), torch.no_grad():
                model = self.model(kind)
                parameters = {name: tensor.clone() for name, tensor in model.state_dict().items()}
                expected_queries = expected_memory = expected_recency = None
                for condition in INTERVENTION_CONDITIONS:
                    transformed = self.intervene(example, condition)
                    items, mask = self.tensors(transformed)
                    inputs = model.compressor._inputs(model.item_embedding(items), mask)
                    _, old_lengths, _, queries, _, memory, _ = inputs
                    retained_memory = memory[:, int(old_lengths[0]) - 4:int(old_lengths[0])]
                    _, compressed_mask, sources = model.encode(items, mask)
                    recency = mask.sum(1)[:, None] - 1 - sources
                    self.assertTrue(bool(compressed_mask.all()))
                    self.assertEqual(recency.tolist(), [[6, 6, 5, 4, 3, 2, 1, 0]])
                    if expected_queries is None:
                        expected_queries, expected_memory, expected_recency = queries, retained_memory, recency
                    else:
                        torch.testing.assert_close(queries, expected_queries, rtol=0, atol=0)
                        torch.testing.assert_close(retained_memory, expected_memory, rtol=0, atol=0)
                        torch.testing.assert_close(recency, expected_recency, rtol=0, atol=0)
                for name, tensor in model.state_dict().items():
                    torch.testing.assert_close(tensor, parameters[name], rtol=0, atol=0)

    def test_when_history_readout_is_zero_all_controls_have_identical_predictions(self):
        """A query-residual-only model must not react to remote input controls."""
        from rhpr.sasrec.query_history_interventions import INTERVENTION_CONDITIONS

        example = self.example(length=23)
        for kind in ("query_cross_attention", "query_ssm"):
            with self.subTest(kind=kind), torch.no_grad():
                model = self.model(kind)
                projection = (model.compressor.cross_attention.out_proj if kind == "query_cross_attention"
                              else model.compressor.output_projection)
                projection.weight.zero_()
                projection.bias.zero_()
                states = [model.last_state(*self.tensors(self.intervene(example, condition)))
                          for condition in INTERVENTION_CONDITIONS]
                for state in states[1:]:
                    torch.testing.assert_close(state, states[0], rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
