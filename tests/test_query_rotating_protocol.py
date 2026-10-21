from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
    import torch
except ImportError:  # pragma: no cover
    np = torch = None


@unittest.skipIf(torch is None, "Torch and numpy are required")
class QueryRotatingProtocolTest(unittest.TestCase):
    def config(self):
        return json.loads((ROOT / "experiments/yambda_query_rotating_windows.json").read_text())

    def cohort(self, lengths=((17, 2049), (4, 1665))):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        users = tuple(LongHistoryUser(uid, (np.arange(length) % 41 + 1).astype(np.int32),
            np.arange(length, dtype=np.uint32)) for uid, length in lengths)
        return LongHistoryCohort(users, np.arange(41, dtype=np.int64), {})

    def test_exact_two_fresh_recipes_and_pinned_historical_sources(self):
        from rhpr.sasrec.query_rotating_protocol import (
            rotating_evaluation_batch, rotating_trials, rotating_variants, validate_rotating_config,
        )
        config = self.config()
        before = copy.deepcopy(config)
        validate_rotating_config(config)
        for family, history in config["history_sources"].items():
            self.assertEqual(history, json.loads((ROOT /
                f"experiments/yambda_history_continuation_{family}.json").read_text()))
        variants = rotating_variants(config)
        self.assertEqual([row["name"] for row in variants], ["query_ssm_m32", "fuxi_linear"])
        for variant in variants:
            trials = rotating_trials(config, variant)
            self.assertEqual([row["name"] for row in trials], ["historical_fixed", "rotating"])
            self.assertTrue(all(row["backbone_lr"] == row["compressor_lr"] == .001 for row in trials))
            self.assertEqual(rotating_evaluation_batch(config, variant, 1024), 32)
            trials[0]["backbone_lr"] = 2.
        variants[0]["summary_tokens"] = 1
        self.assertEqual(config, before)

    def test_registration_rejects_schedule_architecture_optimization_and_provenance_drift(self):
        from rhpr.sasrec.query_rotating_protocol import validate_rotating_config
        mutations = (
            lambda c: c.update(extra=True), lambda c: c.pop("schedule"),
            lambda c: c.update(kind="yambda_query_mamba_ablation"),
            lambda c: c.update(name="other"), lambda c: c.update(summary_budgets=[4]),
            lambda c: c["variants"].reverse(),
            lambda c: c["schedule"].update(first_pass="random"),
            lambda c: c["schedule"].update(target_block_size=64),
            lambda c: c["hpo"].update(proxy_old_length=2048),
            lambda c: c["hpo"].update(max_passes=100),
            lambda c: c["hpo"].update(patience=3),
            lambda c: c["hpo"].update(full_length_validation=True),
            lambda c: c["hpo"].update(test_evaluations=1),
            lambda c: c["hpo"]["trials"][1].update(compressor_lr=.003),
            lambda c: c["training"].update(num_negatives=128),
            lambda c: c["training"].update(effective_batch_size=64),
            lambda c: c["training"].update(budget_seconds=21600),
            lambda c: c["rotating_batches"]["fuxi_linear"].update(training=32),
            lambda c: c["model"].update(n_layers=4),
            lambda c: c["data"].update(catalog_sha256="0" * 64),
            lambda c: c["kaggle"].update(kernel_sources=[]),
            lambda c: c["history_sources"].pop("fuxi"),
            lambda c: c["history_sources"]["qssm"]["selected_source"]["checkpoint"].update(sha256="0" * 64),
            lambda c: c["history_sources"]["fuxi"]["training"].update(budget_seconds=21600),
        )
        for mutate in mutations:
            config = self.config()
            mutate(config)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                validate_rotating_config(config)

    def test_first_pass_matches_historical_inputs_targets_user_order_and_negatives(self):
        from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule
        from rhpr.sasrec.query_proxy_protocol import hpo_batch_arrays, hpo_variants
        from rhpr.sasrec.query_rotating_protocol import (
            rotating_batch_arrays, rotating_schedule, rotating_variants,
        )
        config, cohort = self.config(), self.cohort()
        old = fixed_prefix_schedule(cohort, seed=42, epoch=1)
        new = rotating_schedule(cohort, seed=42, epoch=1)
        self.assertEqual(old, new)
        expected = hpo_batch_arrays(cohort, old, config, hpo_variants(config)[0], old_length=1024)
        for variant in rotating_variants(config):
            actual = rotating_batch_arrays(cohort, new, config, variant, old_length=1024)
            for field in expected:
                np.testing.assert_array_equal(actual[field], expected[field])

    def test_blocks_do_not_repeat_within_cycles_and_are_invariant_to_cohort_order(self):
        from rhpr.sasrec.long_history_data import LongHistoryCohort
        from rhpr.sasrec.query_rotating_protocol import rotating_schedule
        cohort = self.cohort()
        reversed_cohort = LongHistoryCohort(cohort.users[::-1], cohort.catalog_item_ids, {})
        history = {user.user_id: [] for user in cohort.users}
        for epoch in range(1, 25):
            windows = rotating_schedule(cohort, seed=42, epoch=epoch)
            reordered = rotating_schedule(reversed_cohort, seed=42, epoch=epoch)
            identity = lambda rows: [(w.user_id, w.end, w.negative_seed) for w in rows]
            self.assertEqual(identity(windows), identity(reordered))
            self.assertEqual(windows, rotating_schedule(cohort, seed=42, epoch=epoch))
            for window in windows:
                history[window.user_id].append(window.end)
        for user in cohort.users:
            count = (len(user.train) - 1025) // 128
            expected = {len(user.train) - 1 - 128 * j for j in range(count)}
            for start in range(0, 24 - count + 1, count):
                actual = history[user.user_id][start:start + count]
                self.assertEqual(set(actual), expected)
                self.assertEqual(len(set(actual)), count)
            first_cycle = history[user.user_id][:count]
            labels = [set(range(end - 127, end + 1)) for end in first_cycle]
            self.assertEqual(len(set.union(*labels)), count * 128)
            self.assertTrue(all(end >= 1152 and end < len(user.train) for end in expected))
        seed_a = [w.end for epoch in range(2, 7) for w in rotating_schedule(cohort, seed=42, epoch=epoch)]
        seed_b = [w.end for epoch in range(2, 7) for w in rotating_schedule(cohort, seed=43, epoch=epoch)]
        self.assertNotEqual(seed_a, seed_b)

    def test_arrays_shift_train_only_labels_and_preserve_occupied_oov_for_both_models(self):
        from rhpr.sasrec.long_quality_schedule import quality_negative_ids
        from rhpr.sasrec.query_rotating_protocol import (
            rotating_batch_arrays, rotating_schedule, rotating_variants,
        )
        config, cohort = self.config(), self.cohort()
        config["training"]["num_negatives"] = 7
        for user in cohort.users:
            user.train[::17] = 0
        windows = rotating_schedule(cohort, seed=42, epoch=3)
        qssm, fuxi = [rotating_batch_arrays(cohort, windows, config, variant, old_length=1024)
                      for variant in rotating_variants(config)]
        for field in qssm:
            np.testing.assert_array_equal(qssm[field], fuxi[field])
        self.assertEqual(qssm["item_ids"].shape, (2, 1152))
        self.assertTrue(bool(qssm["valid_mask"].all()))
        self.assertGreater(int((qssm["item_ids"] == 0).sum()), 0)
        for index, window in enumerate(windows):
            user = cohort.users[window.user_index]
            np.testing.assert_array_equal(qssm["item_ids"][index], user.train[window.end - 1152:window.end])
            np.testing.assert_array_equal(qssm["targets"][index], user.train[window.end - 1151:window.end + 1])
            np.testing.assert_array_equal(qssm["targets"][index, -128:], user.train[window.end - 127:window.end + 1])
            np.testing.assert_array_equal(qssm["negative_ids"][index], quality_negative_ids(
                window, num_items=cohort.num_items, num_negatives=7))

    def test_summary_counts_exact_coverage_without_confusing_end_positions_with_input_width(self):
        from rhpr.sasrec.query_rotating_protocol import rotating_schedule, rotating_schedule_summary
        config, cohort = self.config(), self.cohort(((17, 16386), (4, 17000)))
        windows = tuple(w for epoch in range(1, 61)
                        for w in rotating_schedule(cohort, seed=42, epoch=epoch))
        summary = rotating_schedule_summary(cohort, windows, config)
        self.assertEqual(summary["unique_target_positions"], 2 * 60 * 128)
        self.assertEqual(summary["target_presentations"], 2 * 60 * 128)
        self.assertEqual(summary["repeated_endpoint_presentations"], 0)
        self.assertEqual(summary["training_input_length_max"], 1152)
        self.assertGreater(summary["sampled_end_max"], 1152)
        self.assertGreater(summary["target_block_events_behind_latest_max"], 0)
        self.assertEqual(summary["available_full_target_blocks"], 120 + 124)
        self.assertEqual(summary["available_train_transitions"],
            summary["unavailable_initial_context_transitions"]
            + summary["unavailable_block_remainder_transitions"]
            + summary["available_target_positions_in_full_blocks"])
        reordered = rotating_schedule_summary(cohort, windows[::-1], config)
        self.assertEqual(summary["schedule_sha256"], reordered["schedule_sha256"])
        self.assertEqual(summary["target_positions_sha256"], reordered["target_positions_sha256"])
        self.assertNotEqual(summary["ordered_schedule_sha256"], reordered["ordered_schedule_sha256"])
        self.assertFalse(summary["train_is_exhaustive_event_epoch"])
        small = self.cohort(((17, 1281),))
        repeated = tuple(w for epoch in range(1, 6) for w in rotating_schedule(small, seed=42, epoch=epoch))
        counts = rotating_schedule_summary(small, repeated, config)
        self.assertEqual(counts["target_presentations"], 640)
        self.assertEqual(counts["unique_target_positions"], 256)
        self.assertEqual(counts["repeated_endpoint_presentations"], 3)

    def test_invalid_contexts_endpoints_identities_and_lengths_are_rejected(self):
        from rhpr.sasrec.long_history_data import LongHistoryCohort
        from rhpr.sasrec.query_rotating_protocol import (
            rotating_batch_arrays, rotating_evaluation_batch, rotating_schedule,
            rotating_schedule_summary, rotating_variants,
        )
        config, cohort = self.config(), self.cohort()
        variant = rotating_variants(config)[0]
        windows = rotating_schedule(cohort, seed=42, epoch=1)
        for old in (None, 2048, 128, 0, True, 1024.):
            with self.subTest(old=old), self.assertRaises(ValueError):
                rotating_evaluation_batch(config, variant, old)
        for window in (replace(windows[0], end=windows[0].end - 1),
                       replace(windows[0], end=1151),
                       replace(windows[0], end=windows[0].end + 128),
                       replace(windows[0], user_id=999), replace(windows[0], user_index=999),
                       replace(windows[0], epoch=0), replace(windows[0], negative_seed=-1)):
            with self.subTest(window=window), self.assertRaises(ValueError):
                rotating_batch_arrays(cohort, [window], config, variant, old_length=1024)
        with self.assertRaisesRegex(ValueError, "one window per user"):
            rotating_schedule_summary(cohort, [windows[0], windows[0]], config)
        with self.assertRaises(ValueError):
            rotating_schedule(self.cohort(((17, 1152),)), seed=42, epoch=1)
        with self.assertRaises(ValueError):
            rotating_schedule(cohort, seed=True, epoch=1)
        with self.assertRaises(ValueError):
            rotating_schedule(cohort, seed=42, epoch=0)
        duplicated = LongHistoryCohort((cohort.users[0], cohort.users[0]), cohort.catalog_item_ids, {})
        with self.assertRaisesRegex(ValueError, "unique"):
            rotating_schedule(duplicated, seed=42, epoch=1)

    def control_fixture(self, directory, *, nonfinite=False):
        from rhpr.sasrec.query_rotating_protocol import rotating_variants
        config, metadata, reports = self.config(), {}, {}
        output = Path(directory) / "output"
        output.mkdir()
        for family, variant, report_name, experiment_name, stage in (
                ("qssm", rotating_variants(config)[0], "query-qssm-search.json",
                 "yambda_query_qssm_search.json", "qssm_summary_memory_search"),
                ("fuxi", rotating_variants(config)[1], "query-baseline-screen.json",
                 "yambda_query_baseline_screen.json", "matched_baseline_screen")):
            history = config["history_sources"][family]
            source = json.loads((ROOT / "experiments" / experiment_name).read_text())
            trial = history["selected_source"]["trial"]
            epoch = history["selected_source"]["checkpoint"]["best_epoch"]
            model = torch.nn.Linear(3, 2)
            if nonfinite and family == "qssm":
                with torch.no_grad():
                    model.weight[0, 0] = float("nan")
            checkpoint_config = dict(source, active_trial=copy.deepcopy(trial),
                active_stage=stage, active_old_length=1024, active_pass_cap=60)
            payload = dict(format_version=3, source_code_sha=history["selected_source"]["source_code_sha"],
                experiment_config=checkpoint_config, model_config=source["model"], variant=variant,
                num_items=11, best_epoch=epoch, selection_score=.25, state_dict=model.state_dict())
            path = output / f"control-{family}.pt"
            torch.save(payload, path)
            record = dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size_bytes=path.stat().st_size, best_epoch=epoch, trial=copy.deepcopy(trial), precision="float32")
            provenance = dict(source_code_sha=history["selected_source"]["source_code_sha"],
                weights_copied=True, weight_bytes_verified=True, test_ranks_read=False)
            metadata[family] = dict(checkpoint_path=str(path), checkpoint_record=record,
                source_config=copy.deepcopy(source), source_checkpoint_config=checkpoint_config,
                source_variant=variant, source_trial=copy.deepcopy(trial), source_best_epoch=epoch,
                source_best_validation={"ndcg10": .25, "num_items": 11}, provenance=provenance,
                validation_artifacts={"validation_ranks": {"file": f"source-validation/{family}.json.gz"}},
                stopping_policy_replay={"best_epoch": epoch})
            result = dict(status="success" if family == "qssm" else "full_transfer_failed", complete=True,
                trial_name=trial["name"], trial=copy.deepcopy(trial), best_epoch=epoch,
                best_validation={"ndcg10": .25, "num_items": 11}, passes_completed=60,
                best_checkpoint=record, best_validation_rank_file="old-short.json.gz",
                best_full_validation={"ndcg10": .99}, full_validation_rank_file="DO-NOT-READ-full.json.gz",
                full_validation_wall_seconds=100., full_validation_is_frozen_1k_checkpoint=True,
                resumable_checkpoint={"path": "DO-NOT-READ-resume.pt"}, failure={"stage": "full_length_validation"})
            report_path = Path(directory) / report_name
            report_path.write_text(json.dumps({"trials": {variant["name"]: {trial["name"]: result}}}))
            reports[report_name] = report_path
        return config, output, metadata, reports

    def test_both_historical_controls_load_actual_weights_without_rng_changes_or_full_rank_reads(self):
        from rhpr.sasrec.query_history_ablation import _verified_file as verify_file
        from rhpr.sasrec.query_rotating_protocol import load_rotating_controls, rotating_variants
        with tempfile.TemporaryDirectory() as directory:
            config, output, metadata, reports = self.control_fixture(directory)
            rng = torch.get_rng_state().clone()
            def verify(root, name, *args):
                return reports[name] if name in reports else verify_file(root, name, *args)
            with (patch("rhpr.sasrec.query_rotating_protocol.prepare_history_source",
                        side_effect=lambda source, _evaluation, _output: metadata[source["family"]]) as prepare,
                  patch("rhpr.sasrec.query_baseline_protocol._resolve_kernel_root", return_value=Path(directory)),
                  patch("rhpr.sasrec.query_history_ablation._verified_file", side_effect=verify),
                  patch("rhpr.sasrec.query_history_models.build_history_model",
                        side_effect=lambda _source, _items: torch.nn.Linear(3, 2)) as build):
                controls = load_rotating_controls(config, object(), output)
            self.assertEqual(prepare.call_count, 2)
            self.assertEqual(build.call_count, 2)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            for family, variant in zip(("qssm", "fuxi"), rotating_variants(config)):
                control = controls[variant["name"]]
                self.assertEqual(control["trial_name"], "historical_fixed")
                self.assertEqual(control["best_checkpoint"]["trial"], control["trial"])
                self.assertEqual(control["reused_source_trial"], config["history_sources"][family]["selected_source"]["trial"])
                self.assertEqual(control["best_validation_rank_file"], f"source-validation/{family}.json.gz")
                self.assertTrue(control["reused_provenance"]["model_parameters_loaded_strictly"])
                self.assertFalse(control["training_performed_in_this_run"])
                self.assertFalse(control["reused_provenance"]["test_ranks_read"])
                for field in ("best_full_validation", "full_validation_rank_file", "full_validation_wall_seconds",
                              "full_validation_is_frozen_1k_checkpoint", "resumable_checkpoint", "failure"):
                    self.assertNotIn(field, control)
            self.assertFalse(controls["fuxi_linear"]["reused_provenance"]["schedule_effect_is_causally_isolated"])

    def test_nonfinite_source_weights_and_registration_drift_fail_before_training(self):
        from rhpr.sasrec.query_history_ablation import _verified_file as verify_file
        from rhpr.sasrec.query_rotating_protocol import load_rotating_controls
        with tempfile.TemporaryDirectory() as directory:
            config, output, metadata, reports = self.control_fixture(directory, nonfinite=True)
            def verify(root, name, *args):
                return reports[name] if name in reports else verify_file(root, name, *args)
            with (patch("rhpr.sasrec.query_rotating_protocol.prepare_history_source",
                        side_effect=lambda source, _evaluation, _output: metadata[source["family"]]),
                  patch("rhpr.sasrec.query_baseline_protocol._resolve_kernel_root", return_value=Path(directory)),
                  patch("rhpr.sasrec.query_history_ablation._verified_file", side_effect=verify),
                  patch("rhpr.sasrec.query_history_models.build_history_model",
                        side_effect=lambda _source, _items: torch.nn.Linear(3, 2))):
                with self.assertRaisesRegex(ValueError, "finite FP32"):
                    load_rotating_controls(config, object(), output)
            config["training"]["budget_seconds"] = 1
            with patch("rhpr.sasrec.query_rotating_protocol.prepare_history_source") as prepare:
                with self.assertRaises(ValueError):
                    load_rotating_controls(config, object(), output)
                prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
