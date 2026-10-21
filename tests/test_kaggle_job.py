from __future__ import annotations

import unittest
import tempfile
import copy
import json
import re
import subprocess
from pathlib import Path
from unittest.mock import patch

from scripts.kaggle_job import copy_result_file, execute, parse_kernel_status, write_collection_manifest


class KaggleJobTest(unittest.TestCase):
    @staticmethod
    def reused_checkpoint_manifest() -> dict:
        kernel = "carasik/rhpr-query-prefix-1k-search"
        source = "66a0322b46b17cf9b480952512d49810c1efcd8e"
        checkpoints, models, trials = [], {}, {}
        for family, name, digest, size in (
            ("query_cross_attention_m4", "ca_base",
             "f5d774c7c483484037144b78db264f0c3362d70f498e6a53b7773780dd50c6f3", 556857385),
            ("query_ssm_m4", "q_base",
             "466c6924a4e10e81c789bd5fea0582e3bac24b75e3bb41750c25d9732f7c0ff8", 556793733),
        ):
            original = f"checkpoints/{family}-current_lr.pt"
            mounted = f"/kaggle/input/notebooks/{kernel}/{original}"
            trial = {"name": name, "backbone_lr": 0.001, "compressor_lr": 0.001}
            original_trial = dict(trial, name="current_lr")
            checkpoints.append({
                "active_old_length": 1024, "best_epoch": 50, "precision": "float32",
                "training_model": family, "trial": trial, "path": mounted,
                "sha256": digest, "size_bytes": size, "reused": True,
                "external_kernel": kernel, "source_code_sha": source,
                "original_path": original, "original_trial": original_trial,
                "verified_mount_path": mounted,
            })
            models[family] = {"original_trial": "current_lr", "checkpoint": {
                "file": original, "sha256": digest, "size_bytes": size, "best_epoch": 50,
            }}
            trials[family] = [trial]
        return {"format_version": 3, "checkpoints": checkpoints, "config": {
            "reused_controls": {"kernel": kernel, "source_code_sha": source,
                                "checkpoint_root": "/kaggle/input/rhpr-query-prefix-1k-search",
                                "models": models},
            "architecture_trials": trials,
        }}

    def test_parses_plain_status(self) -> None:
        self.assertEqual(parse_kernel_status('kernel has status "complete"'), "complete")

    def test_parses_namespaced_status(self) -> None:
        output = 'kernel has status "KernelWorkerStatus.ERROR"'
        self.assertEqual(parse_kernel_status(output), "error")

    def test_rejects_missing_status(self) -> None:
        with self.assertRaises(RuntimeError):
            parse_kernel_status("unexpected output")

    def test_suite_prefers_root_metrics_over_variant_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            downloaded, destination = root / "downloaded", root / "artifacts"
            (downloaded / "runs" / "full").mkdir(parents=True)
            destination.mkdir()
            (downloaded / "metrics.json").write_text('{"pilot_complete": true}')
            (downloaded / "runs" / "full" / "metrics.json").write_text('{"variant": "full"}')
            copy_result_file(downloaded, "metrics.json", destination)
            self.assertEqual((destination / "metrics.json").read_text(), '{"pilot_complete": true}')

    def test_output_pattern_is_forwarded_without_changing_default_download(self) -> None:
        for pattern in (None, "", r".*\.(json|json\.gz|md|log|zip)$"):
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                output = root / "artifacts"
                commands = []

                def fake_run(command, *, check=True):
                    commands.append(command)
                    if command[1:3] == ["kernels", "output"]:
                        downloaded = output / "downloaded"
                        (downloaded / "metrics.json").write_text('{"pilot_complete": true}')
                        (downloaded / "summary.md").write_text("Completed\n")
                        (downloaded / "run-manifest.json").write_text('{"git_sha": "source"}')
                    stdout = "Kernel version 7 successfully pushed. Please check progress" if command[2] == "push" else ""
                    return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

                with patch("scripts.kaggle_job.run", side_effect=fake_run), patch(
                    "scripts.kaggle_job.kernel_status", return_value=("complete", "complete")
                ), patch("scripts.kaggle_job.time.sleep"):
                    execute(root / "bundle", "owner/kernel", output, 1, output_file_pattern=pattern)

                expected_command = [
                    "kaggle", "kernels", "output", "owner/kernel", "-p",
                    str(output / "downloaded"), "--force",
                ]
                if pattern:
                    expected_command.extend(["--file-pattern", pattern])
                self.assertEqual(commands[-1], expected_command)
                manifest = json.loads((output / "output-collection-manifest.json").read_text())
                self.assertEqual(manifest["filtered_download"], bool(pattern))
                self.assertEqual(manifest["output_file_pattern"], pattern or None)
                self.assertEqual(manifest["kernel_version"], 7)
                self.assertEqual(manifest["versioned_kernel_id"], "owner/kernel/7")

    def test_invalid_output_pattern_does_not_submit_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch("scripts.kaggle_job.run") as run:
            with self.assertRaises(re.error):
                execute(Path(temporary), "owner/kernel", Path(temporary) / "outputs", 1,
                        output_file_pattern="[")
            run.assert_not_called()

    def test_failed_kernel_retains_partial_outputs_and_reports_missing_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "artifacts"
            commands = []
            pattern = r".*\.(json|json\.gz|md|log|zip)$"

            def fake_run(command, *, check=True):
                commands.append(command)
                if command[2] == "output":
                    downloaded = output / "downloaded"
                    (downloaded / "trials" / "finished").mkdir(parents=True)
                    (downloaded / "trials" / "finished" / "metrics.json").write_text('{"ndcg": 0.1}')
                    (downloaded / "proxy-hpo-progress.json").write_text('{"finished_trials": 6}')
                    (downloaded / "run-manifest.json").write_text('{"git_sha": "source"}')
                    (downloaded / "checkpoint-manifest.json").write_text('{"checkpoints": [')
                    (downloaded / "kernel-slug.log").write_text("Training failed\n")
                return subprocess.CompletedProcess(command, 0, stdout="Kernel version 4 successfully pushed.", stderr="")

            with patch("scripts.kaggle_job.run", side_effect=fake_run), patch(
                "scripts.kaggle_job.kernel_status", return_value=("error", "error")
            ), patch("scripts.kaggle_job.time.sleep"):
                with self.assertRaisesRegex(RuntimeError, "finished with status error"):
                    execute(root / "bundle", "owner/kernel-slug", output, 1, output_file_pattern=pattern)

            self.assertEqual(sum(command[2] == "push" for command in commands), 1)
            self.assertEqual(commands[-1][-2:], ["--file-pattern", pattern])
            self.assertFalse((output / "metrics.json").exists())
            self.assertEqual(json.loads((output / "run-manifest.json").read_text()), {"git_sha": "source"})
            self.assertEqual((output / "downloaded" / "kernel.log").read_text(), "Training failed\n")
            manifest = json.loads((output / "output-collection-manifest.json").read_text())
            self.assertEqual(manifest["checkpoint_manifest_error_type"], "JSONDecodeError")
            self.assertIn("proxy-hpo-progress.json", [entry["path"] for entry in manifest["downloaded_files"]])
            report = json.loads((output / "terminal-status.json").read_text())
            self.assertFalse(report["completed_successfully"])
            self.assertEqual(report["terminal_status"], "error")
            self.assertEqual(report["kernel_version"], 4)
            self.assertEqual(report["missing_root_files"], ["metrics.json", "summary.md"])
            self.assertEqual(report["copied_root_files"], ["run-manifest.json"])
            self.assertTrue(report["kernel_log_downloaded"])

    def test_polling_failure_is_preserved_when_both_diagnostic_downloads_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "artifacts"
            commands = []
            primary = RuntimeError("status API failed with secret-primary")

            def fake_run(command, *, check=True):
                commands.append(command)
                if command[2] == "output":
                    raise subprocess.CalledProcessError(1, command, stderr="secret-secondary")
                return subprocess.CompletedProcess(command, 0, stdout="Kernel version 2 successfully pushed.", stderr="")

            with patch("scripts.kaggle_job.run", side_effect=fake_run), patch(
                "scripts.kaggle_job.kernel_status", side_effect=primary
            ), patch("scripts.kaggle_job.time.sleep"):
                with self.assertRaises(RuntimeError) as caught:
                    execute(root / "bundle", "owner/kernel", output, 1)

            self.assertIs(caught.exception, primary)
            self.assertEqual(sum(command[2] == "push" for command in commands), 1)
            self.assertEqual(sum(command[2] == "output" for command in commands), 2)
            self.assertEqual(commands[-1][-2:], ["--file-pattern", r"(?!)"])
            manifest = json.loads((output / "output-collection-manifest.json").read_text())
            self.assertEqual(manifest["downloaded_file_count"], 0)
            report_text = (output / "terminal-status.json").read_text()
            self.assertNotIn("secret-", report_text)
            report = json.loads(report_text)
            self.assertEqual(report["terminal_status"], "status_poll_error")
            self.assertEqual([entry["stage"] for entry in report["collection_errors"]],
                             ["download_outputs", "download_kernel_log"])

    def test_failed_output_download_still_collects_kernel_log_and_available_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "artifacts"
            commands = []

            def fake_run(command, *, check=True):
                commands.append(command)
                if command[2] == "output":
                    downloaded = output / "downloaded"
                    if command[-1] != r"(?!)":
                        (downloaded / "metrics.json").write_text('{"completed_trials": 5}')
                        raise subprocess.CalledProcessError(1, command)
                    (downloaded / "kernel.log").write_text("Failure details\n")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch("scripts.kaggle_job.run", side_effect=fake_run), patch(
                "scripts.kaggle_job.kernel_status", return_value=("cancelled", "cancelled")
            ), patch("scripts.kaggle_job.time.sleep"):
                with self.assertRaisesRegex(RuntimeError, "finished with status cancelled"):
                    execute(root / "bundle", "owner/kernel", output, 1, output_file_pattern=r".*\.json$")

            self.assertEqual(json.loads((output / "metrics.json").read_text()), {"completed_trials": 5})
            report = json.loads((output / "terminal-status.json").read_text())
            self.assertEqual(report["terminal_status"], "cancelled")
            self.assertTrue(report["kernel_log_downloaded"])
            self.assertEqual(report["collection_errors"],
                             [{"stage": "download_outputs", "error_type": "CalledProcessError"}])

    def test_timeout_preserves_timeout_after_cancel_and_download_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "artifacts"
            commands = []

            def fake_run(command, *, check=True):
                commands.append((command, check))
                if command[2] == "cancel":
                    self.assertFalse(check)
                    raise OSError("cancel transport failed")
                if command[2] == "output":
                    raise OSError("download transport failed")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch("scripts.kaggle_job.run", side_effect=fake_run), patch(
                "scripts.kaggle_job.kernel_status", return_value=("running", "running")
            ), patch("scripts.kaggle_job.time.sleep"), patch(
                "scripts.kaggle_job.time.monotonic", side_effect=[0, 61]
            ):
                with self.assertRaisesRegex(TimeoutError, "exceeded 1 minutes"):
                    execute(root / "bundle", "owner/kernel", output, 1)

            self.assertEqual(sum(command[2] == "push" for command, _ in commands), 1)
            self.assertEqual([command[2] for command, _ in commands], ["push", "cancel", "output", "output"])
            report = json.loads((output / "terminal-status.json").read_text())
            self.assertEqual(report["terminal_status"], "timeout")
            self.assertEqual(report["last_observed_kernel_status"], "running")
            self.assertEqual(report["primary_error_type"], "TimeoutError")
            self.assertEqual(report["collection_errors"][0], {"stage": "cancel", "error_type": "OSError"})

    def test_failed_submission_does_not_poll_or_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary = subprocess.CalledProcessError(1, ["kaggle", "kernels", "push"])
            with patch("scripts.kaggle_job.run", side_effect=primary) as run, patch(
                "scripts.kaggle_job.kernel_status"
            ) as status:
                with self.assertRaises(subprocess.CalledProcessError) as caught:
                    execute(root / "bundle", "owner/kernel", root / "outputs", 1)
            self.assertIs(caught.exception, primary)
            run.assert_called_once()
            status.assert_not_called()

    def test_success_still_requires_each_aggregate_output(self) -> None:
        for omitted in ("metrics.json", "summary.md", "run-manifest.json"):
            with self.subTest(omitted=omitted), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                output = root / "artifacts"

                def fake_run(command, *, check=True):
                    if command[2] == "output":
                        for name, contents in {"metrics.json": "{}", "summary.md": "Done",
                                               "run-manifest.json": "{}"}.items():
                            if name != omitted:
                                (output / "downloaded" / name).write_text(contents)
                    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

                with patch("scripts.kaggle_job.run", side_effect=fake_run), patch(
                    "scripts.kaggle_job.kernel_status", return_value=("complete", "complete")
                ), patch("scripts.kaggle_job.time.sleep"):
                    with self.assertRaisesRegex(RuntimeError, f"expected one {re.escape(omitted)}, found 0"):
                        execute(root / "bundle", "owner/kernel", output, 1)

    def test_collection_manifest_distinguishes_remote_checkpoints_from_downloaded_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            downloaded = root / "downloaded"
            (downloaded / "checkpoints").mkdir(parents=True)
            (downloaded / "checkpoints" / "small.pt").write_bytes(b"weights")
            (downloaded / "checkpoint-manifest.json").write_text(json.dumps({
                "checkpoints": [
                    {"path": "checkpoints/small.pt", "size_bytes": 7, "sha256": "small-hash"},
                    {"path": "checkpoints/large.pt", "size_bytes": 536_870_912, "sha256": "large-hash"},
                ]
            }))
            write_collection_manifest(downloaded, root, "owner/kernel/3", r".*\.json$")
            manifest = json.loads((root / "output-collection-manifest.json").read_text())
            self.assertEqual(manifest["kernel_id"], "owner/kernel/3")
            self.assertTrue(manifest["checkpoint_manifest_downloaded"])
            collected, retained = manifest["checkpoints"]
            self.assertEqual(collected["storage"], "downloaded")
            self.assertEqual(collected["downloaded_size_bytes"], 7)
            self.assertEqual(retained["storage"], "retained_on_private_kaggle")
            self.assertFalse(retained["downloaded"])
            self.assertIsNone(retained["downloaded_size_bytes"])
            self.assertEqual(retained["kaggle_size_bytes"], 536_870_912)
            self.assertEqual(manifest["downloaded_file_count"], 2)
            self.assertEqual(manifest["downloaded_size_bytes"],
                             sum(path.stat().st_size for path in downloaded.rglob("*") if path.is_file()))

    def test_collection_manifest_rejects_checkpoint_paths_outside_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "checkpoint-manifest.json").write_text(json.dumps({
                "checkpoints": [{"path": "../outside.pt", "size_bytes": 10}],
            }))
            with self.assertRaisesRegex(ValueError, "within kernel outputs"):
                write_collection_manifest(root, root, "owner/kernel", None)

    def test_collection_manifest_records_reused_input_weights_separately_from_current_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.reused_checkpoint_manifest()
            # The completed architecture run has two reused controls and four new checkpoints.
            for name in ("ca_read2", "ca_heads8", "q_inner64", "q_state32"):
                source["checkpoints"].append({"path": f"checkpoints/{name}.pt", "size_bytes": 100,
                                              "sha256": "a" * 64})
            (root / "checkpoints").mkdir()
            (root / "checkpoints" / "ca_read2.pt").write_bytes(b"new weights")
            # A similarly named local file must never turn an external input into a downloaded output.
            (root / source["checkpoints"][0]["original_path"]).write_bytes(b"unrelated output")
            (root / "checkpoint-manifest.json").write_text(json.dumps(source))
            write_collection_manifest(root, root, "owner/current-run", r".*\.json$")
            rows = json.loads((root / "output-collection-manifest.json").read_text())["checkpoints"]
            self.assertEqual(len(rows), 6)
            for expected, collected in zip(source["checkpoints"][:2], rows[:2]):
                self.assertEqual(collected["storage"], "retained_on_private_parent_kaggle")
                self.assertFalse(collected["downloaded"])
                self.assertIsNone(collected["downloaded_size_bytes"])
                for key in ("path", "original_path", "verified_mount_path", "external_kernel",
                            "source_code_sha", "training_model", "original_trial", "sha256", "reused"):
                    self.assertEqual(collected[key], expected[key])
                self.assertEqual(collected["kaggle_size_bytes"], expected["size_bytes"])
            self.assertEqual(rows[2]["storage"], "downloaded")
            self.assertEqual(rows[2]["downloaded_size_bytes"], len(b"new weights"))
            self.assertEqual(rows[3]["storage"], "retained_on_private_kaggle")

    def test_reused_checkpoint_accepts_only_bounded_input_mount_layouts_without_resolving_them(self) -> None:
        for mount in ("/kaggle/input/rhpr-query-prefix-1k-search",
                      "/kaggle/input/carasik/rhpr-query-prefix-1k-search",
                      "/kaggle/input/notebooks/carasik/rhpr-query-prefix-1k-search",
                      "/kaggle/input/kernels/carasik/rhpr-query-prefix-1k-search"):
            with self.subTest(mount=mount), tempfile.TemporaryDirectory() as temporary:
                root, source = Path(temporary), self.reused_checkpoint_manifest()
                for row in source["checkpoints"]:
                    row["path"] = row["verified_mount_path"] = f"{mount}/{row['original_path']}"
                (root / "checkpoint-manifest.json").write_text(json.dumps(source))
                with patch.object(Path, "resolve", side_effect=AssertionError("must not resolve input mount")):
                    write_collection_manifest(root, root, "owner/current-run", None)

    def test_reused_checkpoint_rejects_mismatched_source_or_checkpoint_metadata(self) -> None:
        changes = {"external_kernel": "wrong/kernel", "source_code_sha": "b" * 40,
                   "sha256": "b" * 64, "size_bytes": 12,
                   "original_path": "checkpoints/other.pt", "best_epoch": 40,
                   "training_model": "other_model", "original_trial": {"name": "other_trial"},
                   "trial": {"name": "ca_heads8"}, "verified_mount_path": "/kaggle/input/other.pt"}
        for field, value in changes.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root, source = Path(temporary), self.reused_checkpoint_manifest()
                source["checkpoints"][0][field] = value
                (root / "checkpoint-manifest.json").write_text(json.dumps(source))
                with self.assertRaisesRegex(ValueError, "pinned reused_controls"):
                    write_collection_manifest(root, root, "owner/current-run", None)

    def test_reused_checkpoint_requires_explicit_boolean_and_all_provenance_fields(self) -> None:
        fixture = self.reused_checkpoint_manifest()
        changes = [{"reused": flag} for flag in (False, 1, "true", None)]
        changes.extend({field: None} for field in ("external_kernel", "source_code_sha", "original_path",
                                                  "original_trial", "verified_mount_path"))
        changes.append({"reused": "missing"})
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root, source = Path(temporary), copy.deepcopy(fixture)
                if change == {"reused": "missing"}:
                    del source["checkpoints"][0]["reused"]
                else:
                    source["checkpoints"][0].update(change)
                (root / "checkpoint-manifest.json").write_text(json.dumps(source))
                with self.assertRaises(ValueError):
                    write_collection_manifest(root, root, "owner/current-run", None)

    def test_reused_checkpoint_rejects_escaping_mounts_and_unpinned_configurations(self) -> None:
        mounts = ("/tmp/weights.pt", "/kaggle/input/../outside.pt",
                  "/kaggle/input/notebooks/wrong-owner/rhpr-query-prefix-1k-search/checkpoints/weights.pt",
                  "/kaggle/input/unknown/rhpr-query-prefix-1k-search/checkpoints/weights.pt",
                  "/kaggle/input/rhpr-query-prefix-1k-search/../checkpoints/weights.pt")
        fixtures = []
        for mount in mounts:
            source = self.reused_checkpoint_manifest()
            source["checkpoints"][0]["path"] = source["checkpoints"][0]["verified_mount_path"] = mount
            fixtures.append(source)
        for missing in ("config", "reused_controls", "models", "architecture_trials"):
            source = self.reused_checkpoint_manifest()
            if missing == "config":
                del source["config"]
            elif missing in ("reused_controls", "architecture_trials"):
                del source["config"][missing]
            else:
                del source["config"]["reused_controls"][missing]
            fixtures.append(source)
        for invalid_root in ("/tmp/rhpr-query-prefix-1k-search",
                             "/kaggle/input/../rhpr-query-prefix-1k-search"):
            source = self.reused_checkpoint_manifest()
            source["config"]["reused_controls"]["checkpoint_root"] = invalid_root
            fixtures.append(source)
        source = self.reused_checkpoint_manifest()
        row = source["checkpoints"][0]
        row["original_path"] = source["config"]["reused_controls"]["models"][row["training_model"]][
            "checkpoint"]["file"] = "../weights.pt"
        fixtures.append(source)
        for index, source in enumerate(fixtures):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "checkpoint-manifest.json").write_text(json.dumps(source))
                with self.assertRaises(ValueError):
                    write_collection_manifest(root, root, "owner/current-run", None)

    def test_invalid_reused_checkpoint_preserves_strict_and_best_effort_collection_semantics(self) -> None:
        for strict in (True, False):
            with self.subTest(strict=strict), tempfile.TemporaryDirectory() as temporary:
                root, source = Path(temporary), self.reused_checkpoint_manifest()
                source["checkpoints"].insert(0, {"path": "checkpoints/valid.pt", "size_bytes": 7})
                source["checkpoints"][1]["sha256"] = "wrong-hash"
                (root / "checkpoint-manifest.json").write_text(json.dumps(source))
                if strict:
                    with self.assertRaises(ValueError):
                        write_collection_manifest(root, root, "owner/current-run", None)
                    self.assertFalse((root / "output-collection-manifest.json").exists())
                else:
                    write_collection_manifest(root, root, "owner/current-run", None,
                                              strict_checkpoint_manifest=False)
                    result = json.loads((root / "output-collection-manifest.json").read_text())
                    self.assertEqual(result["checkpoints"], [])
                    self.assertEqual(result["checkpoint_manifest_error_type"], "ValueError")
                    self.assertTrue(result["checkpoint_manifest_downloaded"])
                    self.assertEqual(result["downloaded_file_count"], 1)

    def test_completed_kernel_collection_succeeds_with_pinned_reused_checkpoint_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "artifacts"

            def fake_run(command, *, check=True):
                if command[2] == "output":
                    for name, contents in {"metrics.json": '{"complete": true}', "summary.md": "Done",
                                           "run-manifest.json": "{}",
                                           "checkpoint-manifest.json": json.dumps(self.reused_checkpoint_manifest())}.items():
                        (output / "downloaded" / name).write_text(contents)
                return subprocess.CompletedProcess(command, 0, stdout="Kernel version 1 successfully pushed.", stderr="")

            with patch("scripts.kaggle_job.run", side_effect=fake_run), patch(
                "scripts.kaggle_job.kernel_status", return_value=("complete", "complete")
            ), patch("scripts.kaggle_job.time.sleep"):
                execute(root / "bundle", "owner/current-run", output, 1)
            result = json.loads((output / "output-collection-manifest.json").read_text())
            self.assertEqual(result["kernel_version"], 1)
            self.assertEqual(len(result["checkpoints"]), 2)
            self.assertTrue(all(row["storage"] == "retained_on_private_parent_kaggle"
                                for row in result["checkpoints"]))


if __name__ == "__main__":
    unittest.main()
