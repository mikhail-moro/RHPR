"""Reconstruct private, past-only test-history features for a completed run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rhpr.sasrec.history_features import (
    CATALOGUE_FREQUENCY_FEATURE_FIELDS, HISTORY_FEATURE_FIELDS, history_features,
)
from rhpr.sasrec.long_evaluation_data import (
    COHORT_SIGNATURES, iter_long_evaluation_examples, load_prepared_long_evaluation,
)
from rhpr.sasrec.long_history_data import load_prepared_cohort


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(rows: list[dict], name: str) -> dict:
    values = np.asarray([row[name] for row in rows], dtype=np.float64)
    quantiles = np.quantile(values, [0, .1, .25, .5, .75, .9, 1])
    return {
        "mean": float(values.mean()), "std": float(values.std()),
        "quantiles": {key: float(value) for key, value in zip(
            ("min", "p10", "p25", "p50", "p75", "p90", "max"), quantiles)},
        "distinct_values": int(np.unique(values).size),
    }


def prepare_history_features(
    *, cohort_directory: Path, evaluation_directory: Path, run_output: Path, output: Path,
) -> dict:
    """Verify run identity and save descriptors, never observed item arrays.

    The parquet is private because it contains user IDs. Its manifest contains
    aggregate distributions and hashes only. All catalogue frequencies use the
    complete eligible training cohort before the immutable training cutoff.
    Target identities/support and timestamps verify alignment and chronology;
    neither enters the feature table or feature computation.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    started = time.perf_counter()
    cohort_directory, evaluation_directory = Path(cohort_directory), Path(evaluation_directory)
    run_output, output = Path(run_output), Path(output)
    cohort = load_prepared_cohort(cohort_directory)
    evaluation = load_prepared_long_evaluation(evaluation_directory, cohort)
    config = json.loads((run_output / "experiment.json").read_text(encoding="utf-8"))
    run_cohort = json.loads((run_output / "cohort-summary.json").read_text(encoding="utf-8"))
    run_evaluation = json.loads((run_output / "evaluation-summary.json").read_text(encoding="utf-8"))
    for key in (*COHORT_SIGNATURES, "train_end_timestamp"):
        if run_cohort.get(key) != cohort.summary.get(key):
            raise ValueError(f"run training-cohort mismatch: {key}")
        if key in config["data"] and config["data"][key] != cohort.summary.get(key):
            raise ValueError(f"run configuration training-cohort mismatch: {key}")
    for key in (*COHORT_SIGNATURES, "train_end_timestamp", "validation_end_timestamp",
                "test_end_timestamp", "evaluation_content_sha256"):
        if run_evaluation.get(key) != evaluation.summary.get(key):
            raise ValueError(f"run evaluation mismatch: {key}")
    if config["data"].get("evaluation_content_sha256") != evaluation.summary.get("evaluation_content_sha256"):
        raise ValueError("run configuration evaluation-content mismatch")
    raw_context_length = config["training"]["raw_context_length"]
    tail_size = config["training"]["raw_tail_size"]
    if (type(raw_context_length) is not int or raw_context_length < 128
            or type(tail_size) is not int or tail_size != 128):
        raise ValueError("history feature extraction requires an integer raw window and recent-128 tail")

    prediction_path = run_output / "recent128-test-users.json"
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    if not predictions:
        raise ValueError("test prediction rows must not be empty")
    if any(type(row.get("user_id")) is not int or type(row.get("target_known")) is not bool
           for row in predictions):
        raise ValueError("prediction user IDs and target support must have valid domains")
    baseline = {row["user_id"]: row for row in predictions}
    if len(baseline) != len(predictions):
        raise ValueError("test predictions contain duplicate user IDs")

    counts = np.zeros(cohort.num_items + 1, dtype=np.int64)
    for user in cohort.users:
        counts += np.bincount(user.train, minlength=counts.size)
    if counts[0] != 0 or int(counts.sum()) != cohort.train_events or np.any(counts[1:] == 0):
        raise ValueError("training-only catalogue frequency counts disagree with the fixed cohort")

    training_users = {user.user_id: user for user in cohort.users}
    evaluation_users = {user.user_id: user for user in evaluation.users}
    contexts_digest = hashlib.sha256()
    ids_digest, timestamps_digest, uid_digest = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    feature_names = HISTORY_FEATURE_FIELDS + CATALOGUE_FREQUENCY_FEATURE_FIELDS
    rows, supports, first_times, last_times, anchor_times = [], [], [], [], []
    checks = {"contexts_nondecreasing": 0, "last_context_strictly_before_anchor": 0,
              "target_support_matches_prediction_rows": 0, "constant_raw_context_length": 0}
    for example in iter_long_evaluation_examples(
        cohort, evaluation, split_name="test", max_seq_len=raw_context_length,
    ):
        uid = example.user_id
        if uid not in baseline or bool(example.target > 0) != baseline[uid]["target_known"]:
            raise ValueError("reconstructed evaluation UID or target support differs from test predictions")
        if example.context.size != example.timestamps.size or example.context.size != raw_context_length:
            raise ValueError("reconstructed test context has an unexpected visible window length")
        if (np.any(example.timestamps[1:] < example.timestamps[:-1])
                or int(example.timestamps[-1]) >= example.target_timestamp):
            raise ValueError("reconstructed context must be chronological and strictly before its test anchor")
        for key in checks:
            checks[key] += 1
        train, history = training_users[uid], evaluation_users[uid]
        observed_count = int(train.train.size + history.test_history.size)
        features = history_features(example.context, example.timestamps,
                                    train_event_count=int(train.train.size), catalogue_counts=counts)
        if tuple(features) != feature_names or not all(np.isfinite(value) for value in features.values()):
            raise ValueError("history feature schema or finite-value contract failed")
        rows.append({
            "user_id": uid, **features,
            "observed_context_length": int(example.context.size),
            "old_prefix_length": int(example.context.size - tail_size),
            "train_event_count": int(train.train.size),
            "observed_event_count_before_cropping": observed_count,
            "log_observed_event_count": float(np.log1p(observed_count)),
        })
        uid_bytes = np.asarray([uid], dtype="<i8").tobytes()
        header = np.asarray([uid, example.context.size], dtype="<i8").tobytes()
        item_bytes = np.asarray(example.context, dtype="<i4").tobytes()
        time_bytes = np.asarray(example.timestamps, dtype="<u4").tobytes()
        uid_digest.update(uid_bytes)
        ids_digest.update(header + item_bytes)
        timestamps_digest.update(header + time_bytes)
        contexts_digest.update(header + item_bytes + time_bytes)
        supports.append(example.target > 0)
        first_times.append(int(example.timestamps[0]))
        last_times.append(int(example.timestamps[-1]))
        anchor_times.append(example.target_timestamp)
    if {row["user_id"] for row in rows} != set(baseline) or len(rows) != len(baseline):
        raise ValueError("reconstructed test users differ from the prediction user set")

    diagnostics = ["observed_context_length", "old_prefix_length", "train_event_count",
                   "observed_event_count_before_cropping", "log_observed_event_count"]
    schema = pa.schema([
        pa.field("user_id", pa.int64()),
        *[pa.field(name, pa.float64()) for name in feature_names],
        *[pa.field(name, pa.int64()) for name in diagnostics[:-1]],
        pa.field("log_observed_event_count", pa.float64()),
    ], metadata={
        b"feature_policy": b"past_only_item_identity_not_semantics_train_catalogue_counts_only",
        b"unknown_item_policy": b"zero_id_occupied_observed_chronological_slot",
        b"raw_context_length": str(raw_context_length).encode(),
        b"recent_tail_length": str(tail_size).encode(),
    })
    output.mkdir(parents=True, exist_ok=True)
    path = output / "history-features.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd", use_dictionary=False)
    manifest = {
        "format_version": 1,
        "source_kind": "verified_immutable_test_contexts_and_eligible_user_training_only_popularity",
        "baseline_kind": "HSTU_Recent128_model_uid_support_checks_only",
        "split": "test", "users": len(rows), "warm_targets_checked": int(sum(supports)),
        "unsupported_targets_checked": int(len(rows) - sum(supports)),
        "feature_fields": list(feature_names), "diagnostic_fields": diagnostics,
        "targets_or_target_timestamps_in_features": False,
        "context_policy": {
            "max_seq_len": raw_context_length, "recent_tail_size": tail_size,
            "old_prefix_size": raw_context_length - tail_size,
            "window_length_constant_across_users": True,
            "unknown_identity_zero_keeps_observed_chronological_slot": True,
            "identity_diversity_is_not_semantic_diversity": True,
            "train_event_count_is_full_train_activity_proxy_not_visible_length": True,
            "observed_event_count_includes_known_and_unknown_posttrain_bridge_events": True,
            "catalogue_frequencies_scope": "all_eligible_pre_cutoff_training_user_events_only",
            "frequency_vector_one_based": True,
        },
        "checks": checks,
        "provenance": {
            "train_directory": str(cohort_directory), "evaluation_directory": str(evaluation_directory),
            "run_output_directory": str(run_output),
            "training_catalogue_sha256": cohort.summary["catalog_sha256"],
            "cohort_train_item_sha256": cohort.summary["cohort_mapped_train_items_sha256"],
            "cohort_train_timestamp_sha256": cohort.summary["cohort_train_timestamps_sha256"],
            "evaluation_content_sha256": evaluation.summary["evaluation_content_sha256"],
            "ordered_uid_sha256": uid_digest.hexdigest(),
            "ordered_uid_context_items_sha256": ids_digest.hexdigest(),
            "ordered_uid_context_timestamps_sha256": timestamps_digest.hexdigest(),
            "ordered_uid_context_items_and_timestamps_sha256": contexts_digest.hexdigest(),
            "catalogue_count_vector_sha256": hashlib.sha256(counts.astype("<i8").tobytes()).hexdigest(),
            "catalogue_train_event_count": int(counts.sum()),
            "catalogue_eligible_training_users": len(cohort.users),
            "feature_module_sha256": _file_sha256(ROOT / "src/rhpr/sasrec/history_features.py"),
            "extractor_sha256": _file_sha256(Path(__file__)),
            "baseline_prediction_file_sha256": _file_sha256(prediction_path),
        },
        "observed_timestamp_ranges": {
            "first_context_min": min(first_times), "first_context_max": max(first_times),
            "last_context_min": min(last_times), "last_context_max": max(last_times),
            "anchor_min": min(anchor_times), "anchor_max": max(anchor_times),
        },
        "distributions": {name: _distribution(rows, name) for name in (
            *feature_names, "train_event_count", "observed_event_count_before_cropping")},
        "output": {"name": path.name, "bytes": path.stat().st_size, "sha256": _file_sha256(path)},
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "history-features-manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True, help="Verified prepared training cohort directory")
    parser.add_argument("--evaluation", type=Path, required=True, help="Verified held-out evaluation directory")
    parser.add_argument("--run-output", type=Path, required=True, help="Completed long-compression run output directory")
    parser.add_argument("--output", type=Path, required=True, help="Private output directory for features and manifest")
    args = parser.parse_args()
    manifest = prepare_history_features(cohort_directory=args.cohort,
        evaluation_directory=args.evaluation, run_output=args.run_output, output=args.output)
    print(json.dumps({"users": manifest["users"], "feature_fields": manifest["feature_fields"],
                      "elapsed_seconds": manifest["elapsed_seconds"], "output": manifest["output"]},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
