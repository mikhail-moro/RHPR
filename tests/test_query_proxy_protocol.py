from __future__ import annotations

import copy
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


@unittest.skipIf(torch is None, "Torch and numpy are required")
class QueryProxyProtocolTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def config(self, *, tiny=False):
        config = json.loads((ROOT / "experiments/yambda_query_prefix_proxy_hpo.json").read_text())
        if tiny:
            parent = config["parent"]["experiment_config"]
            parent["model"].update(d_model=8, n_layers=1, n_heads=2, dropout=0.)
            parent["compressor"].update(d_inner=4, d_state=3, gate_rank=3, dropout=0.)
            parent["training"].update(num_negatives=7, amp=False)
        return config

    def cohort(self):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser

        users = tuple(LongHistoryUser(uid, (np.arange(length) % 41 + 1).astype(np.int32),
                      np.arange(length, dtype=np.uint32)) for uid, length in ((17, 279), (4, 533)))
        return LongHistoryCohort(users, np.arange(41, dtype=np.int64), {})

    def test_registered_design_and_exact_parent_drift_are_checked(self):
        from rhpr.sasrec.query_proxy_protocol import (
            hpo_trials, hpo_variants, validate_proxy_hpo_config,
        )

        config = self.config()
        validate_proxy_hpo_config(config)
        self.assertEqual([v["name"] for v in hpo_variants(config)], config["variants"])
        self.assertEqual([t["name"] for t in hpo_trials(config)],
                         ["current_lr", "lower_lr", "faster_compressor"])
        trials = hpo_trials(config)
        trials[0]["backbone_lr"] = 99
        self.assertEqual(config["hpo"]["trials"][0]["backbone_lr"], .001)
        mutations = [
            ("hpo", "proxy_old_length", 2048), ("hpo", "proxy_passes", 10),
            ("hpo", "final_passes", 40), ("hpo", "physical_batch_size", 16),
            ("hpo", "trials", list(reversed(config["hpo"]["trials"]))),
            ("training", "budget_seconds", 999999), ("training", "num_negatives", 128),
            ("model", "n_layers", 3), ("compressor", "d_inner", 64),
            ("runtime", "target_torch_version", "2.5.1"),
            ("data", "catalog_sha256", "0" * 64),
            ("parent", "source_code_sha", "0" * 40),
            ("parent", "manifest_sha256", "0" * 64),
        ]
        for section, key, value in mutations:
            bad = copy.deepcopy(config)
            bad[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_proxy_hpo_config(bad)
        # Mutually changing both copies must not bypass the original run pin.
        bad = copy.deepcopy(config)
        bad["seed"] = bad["parent"]["experiment_config"]["seed"] = 7
        with self.assertRaisesRegex(ValueError, "exact completed parent"):
            validate_proxy_hpo_config(bad)
        for field in ("baselines", "checkpoints"):
            bad = copy.deepcopy(config)
            bad["parent"][field] = {}
            with self.assertRaisesRegex(ValueError, "exact completed parent"):
                validate_proxy_hpo_config(bad)

    def test_parent_initializer_and_rng_are_identical_for_both_compressors(self):
        from rhpr.sasrec.query_prefix_protocol import build_prefix_model
        from rhpr.sasrec.query_proxy_protocol import build_hpo_model, hpo_variants

        config = self.config(tiny=True)
        unchanged = copy.deepcopy(config)
        for variant in hpo_variants(config):
            parent = build_prefix_model(config["parent"]["experiment_config"], variant, 41)
            parent_rng = torch.get_rng_state().clone()
            proxy = build_hpo_model(config, variant, 41)
            self.assertEqual(proxy.max_seq_len, 32768)
            self.assertEqual(proxy.position_embedding.num_embeddings, 32768)
            for key, value in parent.state_dict().items():
                torch.testing.assert_close(proxy.state_dict()[key], value, rtol=0, atol=0)
            self.assertTrue(torch.equal(parent_rng, torch.get_rng_state()))
        self.assertEqual(config, unchanged)

    def test_cpu_left_crop_keeps_qr_labels_negative_draws_and_original_schedule(self):
        from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule, prefix_batch_arrays
        from rhpr.sasrec.query_proxy_protocol import hpo_batch_arrays, hpo_variants

        config, cohort = self.config(tiny=True), self.cohort()
        cohort.users[0].train[-200] = 0  # An occupied OOV remains a valid slot.
        schedule = fixed_prefix_schedule(cohort, seed=42, epoch=7)
        original = tuple(schedule)
        variant = hpo_variants(config)[0]
        full = prefix_batch_arrays(cohort, schedule, config["parent"]["experiment_config"], variant)
        uncropped = hpo_batch_arrays(cohort, schedule, config, variant, old_length=None)
        for key in full:
            np.testing.assert_array_equal(uncropped[key], full[key])
        for old_length in (128, 256, 1024):
            short = hpo_batch_arrays(cohort, schedule, config, variant, old_length=old_length)
            np.testing.assert_array_equal(short["negative_ids"], full["negative_ids"])
            for index, window in enumerate(schedule):
                count = min(window.end, old_length + 128)
                self.assertEqual(int(short["valid_mask"][index].sum()), count)
                np.testing.assert_array_equal(short["item_ids"][index, :count],
                                              full["item_ids"][index, window.end - count:window.end])
                np.testing.assert_array_equal(short["targets"][index, count - 128:count],
                                              full["targets"][index, window.end - 128:window.end])
                np.testing.assert_array_equal(short["item_ids"][index, count - 256:count],
                                              full["item_ids"][index, window.end - 256:window.end])
                # Summary/raw recency PE stay exactly those of the full history.
                self.assertEqual(count - 1 - (count - 129), 128)
                np.testing.assert_array_equal(count - 1 - np.arange(count - 128, count),
                                              window.end - 1 - np.arange(window.end - 128, window.end))
                self.assertFalse(bool(short["valid_mask"][index, count:].any()))
            zeros = (short["item_ids"] == 0) & short["valid_mask"]
            self.assertTrue(bool(zeros.any()))
        self.assertEqual(schedule, original)

    def test_nonfinal_windows_overflow_and_query_dropping_are_rejected(self):
        from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule
        from rhpr.sasrec.query_proxy_protocol import hpo_batch_arrays, hpo_variants

        config, cohort = self.config(tiny=True), self.cohort()
        windows = fixed_prefix_schedule(cohort, seed=42, epoch=1)
        variant = hpo_variants(config)[0]
        for length in (0, 127, True, 4096.):
            with self.assertRaisesRegex(ValueError, "query suffix"):
                hpo_batch_arrays(cohort, windows, config, variant, old_length=length)
        with self.assertRaisesRegex(ValueError, "nonempty physical"):
            hpo_batch_arrays(cohort, [], config, variant, old_length=256)
        with self.assertRaisesRegex(ValueError, "original final"):
            hpo_batch_arrays(cohort, [replace(windows[0], end=windows[0].end - 1)],
                             config, variant, old_length=256)
        config["parent"]["experiment_config"]["training"]["raw_context_length"] = 256
        with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
            hpo_batch_arrays(cohort, windows, config, variant, old_length=256)

    def test_optimizer_has_disjoint_complete_groups_and_parent_decay(self):
        from rhpr.sasrec.query_proxy_protocol import build_hpo_model, hpo_optimizer, hpo_trials, hpo_variants

        config = self.config(tiny=True)
        for variant in hpo_variants(config):
            model = build_hpo_model(config, variant, 41)
            for trial in hpo_trials(config):
                optimizer = hpo_optimizer(model, config["training"], trial)
                self.assertIs(optimizer.defaults["foreach"], False)
                all_ids = []
                for group in optimizer.param_groups:
                    self.assertEqual(group["lr"], trial[group["component"] + "_lr"])
                    self.assertEqual(group["weight_decay"], config["training"]["weight_decay"])
                    expected = {id(p) for name, p in model.named_parameters()
                                if name.startswith("compressor.") == (group["component"] == "compressor")}
                    self.assertEqual({id(p) for p in group["params"]}, expected)
                    all_ids.extend(id(p) for p in group["params"])
                self.assertEqual(len(all_ids), len(set(all_ids)))
                self.assertEqual(set(all_ids), {id(p) for p in model.parameters()})

    def test_current_lr_matches_parent_adamw_updates_including_flagged_parameters(self):
        from rhpr.sasrec.long_quality import _optimizer
        from rhpr.sasrec.query_proxy_protocol import build_hpo_model, hpo_optimizer, hpo_trials, hpo_variants

        config = self.config(tiny=True)
        for variant in hpo_variants(config):
            parent = build_hpo_model(config, variant, 41)
            proxy = build_hpo_model(config, variant, 41)
            # Even a no-decay marker is intentionally ignored by the parent.
            next(parent.parameters())._no_weight_decay = True
            next(proxy.parameters())._no_weight_decay = True
            parent_opt = _optimizer(parent, config["training"])
            proxy_opt = hpo_optimizer(proxy, config["training"], hpo_trials(config)[0])
            for step in (1, 2):
                for index, (left, right) in enumerate(zip(parent.parameters(), proxy.parameters())):
                    grad = torch.full_like(left, (index + 1) * step * .0001)
                    left.grad, right.grad = grad, grad.clone()
                parent_opt.step()
                proxy_opt.step()
            for left, right in zip(parent.parameters(), proxy.parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_evaluation_crop_retains_anchor_reference_and_qr_without_aliasing(self):
        from rhpr.sasrec.daily_evaluation_data import DailyEvaluationExample, daily_context_sha256
        from rhpr.sasrec.query_proxy_protocol import transform_hpo_example

        items = np.arange(700, dtype=np.int32)
        times = np.arange(700, dtype=np.uint32) * 7
        reference = daily_context_sha256(17, items, times)
        example = DailyEvaluationExample(17, "17:3", 3, items, times, 41, 5000, reference)
        self.assertIs(transform_hpo_example(example, None), example)
        for length in (128, 256, 1024):
            cropped = transform_hpo_example(example, length)
            width = min(length + 128, len(items))
            np.testing.assert_array_equal(cropped.context, items[-width:])
            np.testing.assert_array_equal(cropped.timestamps, times[-width:])
            np.testing.assert_array_equal(cropped.context[-256:], example.context[-256:])
            self.assertEqual(cropped.reference_context_hash, reference)
            self.assertEqual((cropped.user_id, cropped.anchor_id, cropped.day, cropped.target,
                              cropped.target_timestamp), (17, "17:3", 3, 41, 5000))
            self.assertFalse(np.shares_memory(cropped.context, items))
            self.assertFalse(np.shares_memory(cropped.timestamps, times))
        with self.assertRaisesRegex(ValueError, "query128 and raw128"):
            transform_hpo_example(replace(example, context=items[:255], timestamps=times[:255]), 128)


if __name__ == "__main__":
    unittest.main()
