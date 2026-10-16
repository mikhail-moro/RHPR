from __future__ import annotations

import copy
import hashlib
import json
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
    import torch
except ImportError:  # pragma: no cover
    np = torch = None


@unittest.skipIf(torch is None, "Torch and numpy are required")
class QueryRestartStateTest(unittest.TestCase):
    def config(self):
        config = json.loads((ROOT / "experiments/yambda_query_lr_restart.json").read_text())
        config["training"]["physical_batch_sizes"] = [32]
        return config

    def fixture(self, temporary, *, parent_name="recent128", layout=""):
        config = self.config()
        root = Path(temporary)
        slug = "rhpr-query-ssm-block-pilot"
        config["restart"]["checkpoint_root"] = str(root / slug)
        output = root / layout / slug
        checkpoint = output / config["restart"]["checkpoints"][parent_name]["file"]
        checkpoint.parent.mkdir(parents=True)
        model = torch.nn.Linear(3, 2)
        original = json.loads((ROOT / "experiments/yambda_query_ssm_block_pilot.json").read_text())
        payload = {"format_version": 1, "source_code_sha": config["restart"]["source_code_sha"],
            "experiment_config": original, "model_config": config["model"],
            "variant": {"name": parent_name, "kind": "recent" if parent_name == "recent128"
                else "query_cross_attention", "summary_tokens": 0 if parent_name == "recent128" else 4},
            "num_items": 17, "best_epoch": 48, "selection_score": .03125,
            "state_dict": {key: value.detach().clone() for key, value in model.state_dict().items()}}
        self.write_payload(config, parent_name, checkpoint, payload)
        return config, checkpoint, payload, model

    @staticmethod
    def write_payload(config, name, checkpoint, payload):
        torch.save(payload, checkpoint)
        spec = config["restart"]["checkpoints"][name]
        spec["size_bytes"] = checkpoint.stat().st_size
        spec["sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def test_registered_config_arms_and_restart_boundaries(self):
        from rhpr.sasrec.query_restart_state import restart_arms, validate_restart_config

        config = self.config()
        validate_restart_config(config)
        arms = restart_arms(config)
        self.assertEqual([arm["name"] for arm in arms], ["recent128_lr0p001", "recent128_lr0p003",
            "query_cross_attention_m4_lr0p001", "query_cross_attention_m4_lr0p003"])
        self.assertEqual([arm["learning_rate"] for arm in arms], [.001, .003, .001, .003])
        self.assertEqual([arm["summary_tokens"] for arm in arms], [0, 0, 4, 4])
        for path, value in (
                (("restart", "learning_rates"), [.001]),
                (("restart", "parent_epoch"), 47),
                (("restart", "source_code_sha"), "0" * 40),
                (("restart", "parent_kernel"), "other/kernel"),
                (("restart", "checkpoints", "recent128", "file"), "../recent128.pt"),
                (("restart", "checkpoints", "recent128", "sha256"), "x" * 64),
                (("restart", "checkpoints", "recent128", "size_bytes"), True),
                (("training", "max_additional_passes"), 96),
                (("training", "validation_every_additional_passes"), 16),
                (("training", "early_stopping_min_additional_passes"), 8),
                (("training", "early_stopping_patience"), 2),
                (("training", "physical_batch_sizes"), [8, 16, 32]),
                (("training", "supervised_targets"), 1),
                (("model", "n_layers"), 3),
                (("kaggle", "kernel_sources"), []),
                (("variants",), ["recent128"]),
        ):
            bad = copy.deepcopy(config)
            cursor = bad
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_restart_config(bad)

    def test_bounded_mount_resolution_accepts_classic_and_notebooks_layouts(self):
        from rhpr.sasrec.query_restart_state import load_parent_checkpoint, resolve_parent_checkpoint

        for layout in ("", "carasik", "notebooks/carasik", "kernels/private/carasik"):
            for name in ("recent128", "query_cross_attention_m4"):
                with self.subTest(layout=layout, name=name), tempfile.TemporaryDirectory() as temporary:
                    config, checkpoint, payload, model = self.fixture(temporary, parent_name=name, layout=layout)
                    self.assertEqual(resolve_parent_checkpoint(config, name, input_root=temporary), checkpoint)
                    target = torch.nn.Linear(3, 2)
                    loaded = load_parent_checkpoint(config, name, num_items=17, model=target, input_root=temporary)
                    self.assertEqual(loaded["selection_score"], .03125)
                    for key, expected in model.state_dict().items():
                        torch.testing.assert_close(target.state_dict()[key], expected, rtol=0, atol=0)
                    self.assertEqual(loaded["experiment_config"]["training"]["physical_batch_sizes"], [8, 16, 32])
                    self.assertFalse(any("optimizer" in key for key in loaded))

    def test_mount_resolution_rejects_duplicates_missing_and_outside_input(self):
        from rhpr.sasrec.query_restart_state import resolve_parent_checkpoint

        with tempfile.TemporaryDirectory() as temporary:
            config, checkpoint, _, _ = self.fixture(temporary)
            other = Path(temporary) / "notebooks/carasik/rhpr-query-ssm-block-pilot/checkpoints/recent128.pt"
            other.parent.mkdir(parents=True)
            shutil.copyfile(checkpoint, other)
            with self.assertRaisesRegex(ValueError, "found2|found 2"):
                resolve_parent_checkpoint(config, "recent128", input_root=temporary)
            other.unlink()
            checkpoint.unlink()
            with self.assertRaisesRegex(ValueError, "found 0"):
                resolve_parent_checkpoint(config, "recent128", input_root=temporary)
            config["restart"]["checkpoint_root"] = "/tmp/unrelated/rhpr-query-ssm-block-pilot"
            with self.assertRaisesRegex(ValueError, "inside input_root"):
                resolve_parent_checkpoint(config, "recent128", input_root=temporary)

    def test_checkpoint_bytes_are_verified_before_deserialization(self):
        from rhpr.sasrec.query_restart_state import load_parent_checkpoint

        with tempfile.TemporaryDirectory() as temporary:
            config, checkpoint, _, _ = self.fixture(temporary)
            for field, value, message in (("size_bytes", checkpoint.stat().st_size + 1, "size_bytes"),
                                          ("sha256", "0" * 64, "SHA256")):
                bad = copy.deepcopy(config)
                bad["restart"]["checkpoints"]["recent128"][field] = value
                with self.subTest(field=field), patch("torch.load") as deserialize:
                    with self.assertRaisesRegex(ValueError, message):
                        load_parent_checkpoint(bad, "recent128", num_items=17, input_root=temporary)
                    deserialize.assert_not_called()

    def test_verified_parent_metadata_and_tensor_checks_reject_without_model_mutation(self):
        from rhpr.sasrec.query_restart_state import load_parent_checkpoint

        cases = (
            ("source", lambda p: p.update(source_code_sha="0" * 40)),
            ("epoch", lambda p: p.update(best_epoch=47)),
            ("catalogue", lambda p: p.update(num_items=18)),
            ("variant", lambda p: p["variant"].update(summary_tokens=4)),
            ("model", lambda p: p["model_config"].update(dropout=.3)),
            ("data", lambda p: p["experiment_config"]["data"].update(catalog_sha256="0" * 64)),
            ("loss", lambda p: p["experiment_config"]["training"].update(num_negatives=256)),
            ("nonfinite", lambda p: p["state_dict"]["weight"].fill_(float("nan"))),
            ("half", lambda p: p["state_dict"].update(weight=p["state_dict"]["weight"].half())),
            ("shape", lambda p: p["state_dict"].update(weight=torch.zeros(3, 2))),
            ("missing", lambda p: p["state_dict"].pop("bias")),
            ("extra", lambda p: p["state_dict"].update(extra=torch.zeros(1))),
            ("badscore", lambda p: p.update(selection_score=float("nan"))),
        )
        for name, mutate in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                config, checkpoint, payload, _ = self.fixture(temporary)
                # Payload model_config must not alias the expected config.
                payload = copy.deepcopy(payload)
                mutate(payload)
                self.write_payload(config, "recent128", checkpoint, payload)
                target = torch.nn.Linear(3, 2)
                before = {key: value.detach().clone() for key, value in target.state_dict().items()}
                with self.assertRaises(ValueError):
                    load_parent_checkpoint(config, "recent128", num_items=17, model=target, input_root=temporary)
                for key, value in target.state_dict().items():
                    torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_verified_content_allows_mount_path_changes_but_not_protocol_changes(self):
        from rhpr.sasrec.query_restart_state import load_parent_checkpoint

        with tempfile.TemporaryDirectory() as temporary:
            config, checkpoint, payload, _ = self.fixture(temporary)
            payload["experiment_config"]["data"].update(prepared_path="/old/cohort", evaluation_path="/old/eval")
            payload["experiment_config"]["runtime"].update(wheel_path="/old/wheels", target_site="/old/site")
            self.write_payload(config, "recent128", checkpoint, payload)
            loaded = load_parent_checkpoint(config, "recent128", num_items=17, input_root=temporary)
            self.assertEqual(loaded["experiment_config"]["data"]["prepared_path"], "/old/cohort")

    def test_restart_rng_resets_python_numpy_and_torch_independently_of_prior_work(self):
        from rhpr.sasrec.query_restart_state import reset_restart_rng

        def sample():
            return random.random(), np.random.rand(4), torch.rand(4)

        reset_restart_rng(42)
        expected = sample()
        for _ in range(11):
            sample()
        reset_restart_rng(42)
        actual = sample()
        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        for bad in (-1, True):
            with self.assertRaises(ValueError):
                reset_restart_rng(bad)

    def test_early_stopping_keeps_parent_and_requires_three_checks_and_minimum24(self):
        from rhpr.sasrec.query_restart_state import ValidationEarlyStopping

        stopping = ValidationEarlyStopping(.4)
        self.assertFalse(stopping.observe(8, .4)["should_stop"])
        self.assertFalse(stopping.observe(16, .39)["should_stop"])
        result = stopping.observe(24, .4)
        self.assertTrue(result["should_stop"])
        self.assertFalse(result["improved"])
        self.assertEqual(result["best_additional_passes"], 0)
        self.assertEqual(result["best_score"], .4)
        self.assertEqual(result["bad_checks"], 3)
        with self.assertRaises(ValueError):
            stopping.observe(32, .8)

    def test_early_stopping_strict_improvement_resets_patience_and_ties_keep_earliest(self):
        from rhpr.sasrec.query_restart_state import ValidationEarlyStopping

        stopping = ValidationEarlyStopping(.4)
        self.assertFalse(stopping.observe(8, .3)["improved"])
        self.assertTrue(stopping.observe(16, .5)["improved"])
        self.assertEqual(stopping.bad_checks, 0)
        self.assertFalse(stopping.observe(24, .5)["should_stop"])
        self.assertFalse(stopping.observe(32, .49)["should_stop"])
        result = stopping.observe(40, .48)
        self.assertTrue(result["should_stop"])
        self.assertEqual(result["best_additional_passes"], 16)
        self.assertEqual(result["best_score"], .5)

    def test_early_stopping_enforces_finite_scores_and_complete_validation_schedule(self):
        from rhpr.sasrec.query_restart_state import ValidationEarlyStopping

        with self.assertRaises(ValueError):
            ValidationEarlyStopping(float("nan"))
        stopping = ValidationEarlyStopping(0.)
        for bad_pass in (0, 7, 16, True):
            with self.assertRaises(ValueError):
                stopping.observe(bad_pass, .2)
        with self.assertRaises(ValueError):
            stopping.observe(8, float("inf"))
        for epoch in range(8, 49, 8):
            result = stopping.observe(epoch, epoch / 100.)
            self.assertTrue(result["improved"])
            self.assertFalse(result["should_stop"])
        self.assertEqual(stopping.best_additional_passes, 48)
        with self.assertRaises(ValueError):
            stopping.observe(56, .8)


if __name__ == "__main__":
    unittest.main()
