from __future__ import annotations

import copy
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
class QueryMambaProtocolTest(unittest.TestCase):
    def config(self):
        return json.loads((ROOT / "experiments/yambda_query_mamba_ablation.json").read_text())

    def test_three_independent_features_reuse_exact_selected_source_without_protocol_drift(self):
        from rhpr.sasrec.query_mamba_protocol import (
            mamba_source_config, mamba_trials, mamba_variants, validate_mamba_config,
        )
        config = self.config()
        before = copy.deepcopy(config)
        validate_mamba_config(config)
        self.assertEqual(mamba_source_config(config), json.loads(
            (ROOT / "experiments/yambda_query_qssm_search.json").read_text()))
        self.assertEqual(config["control_history_config"], json.loads(
            (ROOT / "experiments/yambda_history_continuation_qssm.json").read_text()))
        variant = mamba_variants(config)[0]
        self.assertEqual(variant, {"name": "query_ssm_m32", "kind": "query_ssm", "summary_tokens": 32})
        trials = mamba_trials(config, variant)
        self.assertEqual([trial["name"] for trial in trials], ["base", "conv4", "output_gate", "skip_d"])
        self.assertEqual([trial["feature"] for trial in trials], ["none", "conv4", "output_gate", "skip_d"])
        self.assertTrue(all((trial["backbone_lr"], trial["compressor_lr"]) == (.001, .001)
                            for trial in trials))
        self.assertFalse(config["hpo"]["full_length_validation"])
        self.assertEqual(config["hpo"]["test_evaluations"], 0)
        self.assertEqual(config["training"]["budget_seconds"], 10800)
        self.assertEqual(config["hpo"]["min_passes"], 20)
        self.assertEqual(config["hpo"]["max_passes"], 60)
        trials[0]["feature"] = "conv4"
        variant["summary_tokens"] = 4
        source = mamba_source_config(config)
        source["hpo"]["max_passes"] = 1
        self.assertEqual(config, before)

    def test_registration_rejects_architecture_policy_data_and_provenance_mutations(self):
        from rhpr.sasrec.query_mamba_protocol import validate_mamba_config
        mutations = (
            lambda c: c.update(extra=True), lambda c: c.pop("source_search_config"),
            lambda c: c.update(kind="yambda_query_qssm_search"),
            lambda c: c.update(name="other"), lambda c: c.update(summary_budgets=[16]),
            lambda c: c.update(variants=["query_ssm_m4"]),
            lambda c: c["architecture"].update(d_inner=128),
            lambda c: c["architecture"].update(dt_max=.1),
            lambda c: c["hpo"].update(proxy_old_length=2048),
            lambda c: c["hpo"].update(max_passes=40),
            lambda c: c["hpo"].update(patience=1),
            lambda c: c["hpo"].update(full_length_validation=True),
            lambda c: c["hpo"].update(test_evaluations=1),
            lambda c: c["hpo"]["trials"][1].update(feature="conv4+output_gate"),
            lambda c: c["hpo"]["trials"][2].update(compressor_lr=.003),
            lambda c: c["hpo"]["trials"].reverse(),
            lambda c: c["training"].update(budget_seconds=7200),
            lambda c: c["training"].update(num_negatives=128),
            lambda c: c["training"].update(effective_batch_size=64),
            lambda c: c["mamba_batches"]["query_ssm_m32"].update(training=16),
            lambda c: c["mamba_batches"]["query_ssm_m32"].update(full=4),
            lambda c: c["model"].update(n_layers=4),
            lambda c: c["compressor"].update(dropout=.1),
            lambda c: c["data"].update(catalog_sha256="0" * 64),
            lambda c: c["kaggle"].update(kernel_sources=[]),
            lambda c: c["control_history_config"]["selected_source"]["checkpoint"].update(sha256="0" * 64),
            lambda c: c["control_history_config"]["continuation"].update(schedule_epoch_offset=20),
            lambda c: c["source_search_config"]["hpo"].update(max_passes=40),
            lambda c: c["source_search_config"]["reference_search"].update(source_code_sha="0" * 40),
        )
        for mutate in mutations:
            config = self.config()
            mutate(config)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                validate_mamba_config(config)

    def test_no_full_history_or_unregistered_batch_shape_is_accepted(self):
        from rhpr.sasrec.query_mamba_protocol import (
            mamba_evaluation_batch, mamba_trials, mamba_variants,
        )
        config = self.config()
        variant = mamba_variants(config)[0]
        self.assertEqual(mamba_evaluation_batch(config, variant, 1024), 32)
        for old in (None, 2048, 128, 0, True, 1024.):
            with self.subTest(old=old), self.assertRaises(ValueError):
                mamba_evaluation_batch(config, variant, old)
        for invalid in ("query_ssm_m4", dict(variant, summary_tokens=4),
                        dict(variant, kind="query_cross_attention")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                mamba_trials(config, invalid)

    def test_inputs_preserve_source_occupied_oov_positions_targets_and_negative_identities(self):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        from rhpr.sasrec.query_mamba_protocol import mamba_batch_arrays, mamba_variants
        from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule
        from rhpr.sasrec.query_qssm_protocol import qssm_batch_arrays
        config = self.config()
        # This checks only the data adapter and intentionally avoids expensive
        # production-sized negatives while retaining source identity binding.
        config["source_search_config"]["parent"]["experiment_config"]["training"]["num_negatives"] = 7
        users = tuple(LongHistoryUser(uid, (np.arange(length) % 41 + 1).astype(np.int32),
            np.arange(length, dtype=np.uint32)) for uid, length in ((17, 279), (4, 1533)))
        for user in users:
            user.train[-200] = user.train[-20] = 0
        cohort = LongHistoryCohort(users, np.arange(41, dtype=np.int64), {})
        variant = mamba_variants(config)[0]
        for epoch in (1, 7):
            windows = fixed_prefix_schedule(cohort, seed=42, epoch=epoch)
            expected = qssm_batch_arrays(cohort, windows, config["source_search_config"], variant,
                                         old_length=1024)
            actual = mamba_batch_arrays(cohort, windows, config, variant, old_length=1024)
            for field in expected:
                np.testing.assert_array_equal(actual[field], expected[field])
            for index, window in enumerate(windows):
                count = int(actual["valid_mask"][index].sum())
                user = cohort.users[window.user_index]
                np.testing.assert_array_equal(actual["targets"][index, count-128:count], user.train[-128:])
                self.assertTrue(bool(((actual["item_ids"][index] == 0) & actual["valid_mask"][index]).any()))
            for old in (None, 4096):
                with self.assertRaises(ValueError):
                    mamba_batch_arrays(cohort, windows, config, variant, old_length=old)

    def test_selection_uses_finite_short_ndcg10_and_registered_tie_order(self):
        from rhpr.sasrec.query_mamba_protocol import mamba_trials, select_mamba_candidate
        trials = mamba_trials(self.config(), "query_ssm_m32")
        rows = [dict(trial=trial, trial_name=trial["name"], complete=True, status="success",
                     best_validation={"ndcg10": .2}, best_full_validation={"ndcg10": 1. - i * .1})
                for i, trial in enumerate(trials)]
        rows[2]["best_validation"]["ndcg10"] = .3
        self.assertEqual(select_mamba_candidate(trials, rows[::-1]), trials[2])
        rows[0]["best_validation"]["ndcg10"] = .3
        self.assertEqual(select_mamba_candidate(trials, rows[::-1]), trials[0])
        for invalid in (float("nan"), float("inf"), True, -.1, 1.1):
            rows[0]["best_validation"]["ndcg10"] = invalid
            self.assertEqual(select_mamba_candidate(trials, rows), trials[2])
        rows[2].update(complete=False)
        self.assertEqual(select_mamba_candidate(trials, rows), trials[1])

    def control_fixture(self, directory, *, nonfinite=False):
        config = self.config()
        source = config["source_search_config"]
        history = config["control_history_config"]
        trial = history["selected_source"]["trial"]
        variant = {"name": "query_ssm_m32", "kind": "query_ssm", "summary_tokens": 32}
        output = Path(directory) / "output"
        output.mkdir()
        model = torch.nn.Linear(3, 2)
        if nonfinite:
            with torch.no_grad():
                model.weight[0, 0] = float("nan")
        checkpoint_config = dict(source, active_trial=copy.deepcopy(trial),
            active_stage="qssm_summary_memory_search", active_old_length=1024, active_pass_cap=60)
        payload = dict(format_version=3, source_code_sha=history["selected_source"]["source_code_sha"],
            experiment_config=checkpoint_config, model_config=source["model"], variant=variant,
            num_items=11, best_epoch=20, selection_score=.25, state_dict=model.state_dict())
        path = output / "control.pt"
        torch.save(payload, path)
        record = dict(path="control.pt", sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            size_bytes=path.stat().st_size, best_epoch=20, trial=copy.deepcopy(trial), precision="float32")
        provenance = dict(source_code_sha=history["selected_source"]["source_code_sha"],
            weights_copied=True, weight_bytes_verified=True, test_ranks_read=False)
        metadata = dict(checkpoint_path=str(path), checkpoint_record=record,
            source_config=copy.deepcopy(source), source_checkpoint_config=checkpoint_config,
            source_variant=variant, source_trial=copy.deepcopy(trial), source_best_epoch=20,
            source_best_validation={"ndcg10": .25, "num_items": 11}, provenance=provenance,
            validation_artifacts={"validation_ranks": {"file": "source-validation/short.json.gz"}},
            stopping_policy_replay={"best_epoch": 20})
        result = dict(status="success", complete=True, trial_name="dt_slow", trial=copy.deepcopy(trial),
            best_epoch=20, best_validation={"ndcg10": .25, "num_items": 11}, passes_completed=40,
            best_checkpoint=record, best_validation_rank_file="old-short.json.gz",
            best_full_validation={"ndcg10": .99}, full_validation_rank_file="DO-NOT-READ-full.json.gz",
            full_validation_wall_seconds=100., full_validation_is_frozen_1k_checkpoint=True,
            resumable_checkpoint={"path": "DO-NOT-READ-resume.pt"})
        report = {"trials": {"query_ssm_m32": {"dt_slow": result}},
            "preflight": {"query_ssm_m32-dt_slow": {"training": {"windows_per_second": 100.}}}}
        report_path = Path(directory) / "report.json"
        report_path.write_text(json.dumps(report))
        return config, output, metadata, report_path

    def test_reused_control_strictly_loads_real_finite_weights_without_advancing_rng_or_opening_full_ranks(self):
        from rhpr.sasrec.query_history_ablation import _verified_file as verify_file
        from rhpr.sasrec.query_mamba_protocol import load_mamba_controls, mamba_trials
        with tempfile.TemporaryDirectory() as directory:
            config, output, metadata, report = self.control_fixture(directory)
            rng = torch.get_rng_state().clone()
            def verify(root, name, *args):
                return report if name == "query-qssm-search.json" else verify_file(root, name, *args)
            with (patch("rhpr.sasrec.query_mamba_protocol.prepare_history_source", return_value=metadata) as prepare,
                  patch("rhpr.sasrec.query_baseline_protocol._resolve_kernel_root", return_value=Path(directory)),
                  patch("rhpr.sasrec.query_history_ablation._verified_file", side_effect=verify),
                  patch("rhpr.sasrec.query_history_models.build_history_model",
                        side_effect=lambda _config, _items: torch.nn.Linear(3, 2)) as build):
                # Permit the report helper mock while the strict checkpoint
                # loader independently verifies its real serialized payload.
                controls = load_mamba_controls(config, object(), output)
            prepare.assert_called_once()
            build.assert_called_once_with(config["control_history_config"], 11)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            control = controls["query_ssm_m32"]
            base = mamba_trials(config, "query_ssm_m32")[0]
            self.assertEqual(control["trial"], base)
            self.assertEqual(control["trial_name"], "base")
            self.assertEqual(control["best_checkpoint"]["trial"], base)
            self.assertEqual(control["best_checkpoint"]["sha256"], metadata["checkpoint_record"]["sha256"])
            self.assertEqual(control["reused_source_trial"]["name"], "dt_slow")
            self.assertTrue(control["reused_provenance"]["model_parameters_loaded_strictly"])
            self.assertFalse(control["training_performed_in_this_run"])
            self.assertFalse(control["reused_provenance"]["optimizer_state_inherited"])
            self.assertFalse(control["reused_provenance"]["test_ranks_read"])
            self.assertEqual(control["best_validation_rank_file"], "source-validation/short.json.gz")
            self.assertTrue(control["full_transfer_not_requested"])
            for field in ("best_full_validation", "full_validation_rank_file", "full_validation_wall_seconds",
                          "full_validation_is_frozen_1k_checkpoint", "resumable_checkpoint"):
                self.assertNotIn(field, control)

    def test_reused_control_rejects_nonfinite_weight_payload_and_registration_drift_before_io(self):
        from rhpr.sasrec.query_history_ablation import _verified_file as verify_file
        from rhpr.sasrec.query_mamba_protocol import load_mamba_controls
        with tempfile.TemporaryDirectory() as directory:
            config, output, metadata, report = self.control_fixture(directory, nonfinite=True)
            def verify(root, name, *args):
                return report if name == "query-qssm-search.json" else verify_file(root, name, *args)
            with (patch("rhpr.sasrec.query_mamba_protocol.prepare_history_source", return_value=metadata),
                  patch("rhpr.sasrec.query_baseline_protocol._resolve_kernel_root", return_value=Path(directory)),
                  patch("rhpr.sasrec.query_history_ablation._verified_file", side_effect=verify),
                  patch("rhpr.sasrec.query_history_models.build_history_model",
                        side_effect=lambda _config, _items: torch.nn.Linear(3, 2))):
                with self.assertRaisesRegex(ValueError, "finite FP32"):
                    load_mamba_controls(config, object(), output)
            config["training"]["budget_seconds"] = 1
            with patch("rhpr.sasrec.query_mamba_protocol.prepare_history_source") as prepare:
                with self.assertRaises(ValueError):
                    load_mamba_controls(config, object(), output)
                prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
