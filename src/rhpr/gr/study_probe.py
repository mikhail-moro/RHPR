"""Fixed train/validation probes that never change optimizer or RNG state."""
from pathlib import Path
import random
import time

import numpy as np

from rhpr.gr.data import file_sha256


def validate_probe(spec):
    if (not isinstance(spec.get("target_items"), list) or not spec["target_items"]
            or len(set(spec["target_items"])) != len(spec["target_items"])
            or any(type(i) is not int or i < 0 for i in spec["target_items"])):
        raise ValueError("Probe target_items must be distinct nonnegative integers")
    for key in ("interval_updates", "train_examples_per_item", "validation_examples_per_item", "batch_size"):
        if type(spec.get(key)) is not int or spec[key] < 1:
            raise ValueError(f"Invalid training_probe {key}")


def prepare_probe_cases(users, cases, windows, config, output, store):
    import pyarrow as pa
    import pyarrow.parquet as pq
    spec = config["diagnostics"]["training_probe"]
    targets = set(spec["target_items"])
    if max(targets) >= len(store.read("items")):
        raise ValueError("Probe target outside the catalogue")
    by_user = {u.user_id: u for u in users}
    source_epoch = config["training"]["max_epochs"]
    manifest = pq.read_table(windows[source_epoch]).to_pylist()
    result, counts = [], {}
    for row in manifest:
        item = row["target_item"]
        if item in targets and counts.get(item, 0) < spec["train_examples_per_item"]:
            result.append({"probe_id": f"train:{source_epoch}:{row['ordinal']}", "split": "train",
                "user_id": row["user_id"], "target_item": item,
                "history": list(by_user[row["user_id"]].train[row["visible_start"]:row["end"] - 1]),
                "source_epoch": source_epoch, "source_ordinal": row["ordinal"]})
            counts[item] = counts.get(item, 0) + 1
    counts = {}
    for row in sorted(cases, key=lambda c: c["case_id"]):
        item = row["target_item"]
        if row["split"] == "validation" and item in targets and counts.get(item, 0) < spec["validation_examples_per_item"]:
            result.append({"probe_id": row["case_id"], "split": "validation", "user_id": row["user_id"],
                "target_item": item, "history": row["visible_history"], "source_epoch": None, "source_ordinal": None})
            counts[item] = counts.get(item, 0) + 1
    if not result:
        raise ValueError("No matching fixed probe cases")
    schema = pa.schema([("probe_id", pa.string()), ("split", pa.string()), ("user_id", pa.int64()),
        ("target_item", pa.int64()), ("history", pa.list_(pa.int64())), ("source_epoch", pa.int64()), ("source_ordinal", pa.int64())])
    path = Path(output) / "probes" / "cases.parquet"
    path.parent.mkdir()
    pq.write_table(pa.Table.from_pylist(result, schema=schema), path, compression="zstd")
    store.write("artifacts", [{"path": str(path.relative_to(store.root)), "sha256": file_sha256(path), "role": "fixed_probe_cases"}])
    return result


class TrainingProbe:
    def __init__(self, config, output, store, run_id):
        import pyarrow.parquet as pq
        self.spec = config["diagnostics"]["training_probe"]
        self.config, self.store, self.run_id = config, store, run_id
        self.rows = pq.read_table(store.root / "probes" / "cases.parquet").to_pylist()
        self.directory = Path(output) / "probes"
        self.directory.mkdir()
        self.last_step = None
        self.seconds = 0.

    def sample(self, model, index, epoch, attempted_updates, effective_updates):
        import torch
        import pyarrow as pa
        import pyarrow.parquet as pq
        from rhpr.gr.sid_study import make_batch
        from rhpr.gr.study_search import ModelScorer, target_diagnostics
        from rhpr.gr.experiment import _event
        if self.last_step == attempted_updates:
            return
        was_training = model.training
        device = next(model.parameters()).device
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        started = time.monotonic()
        records = []
        try:
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
                model.eval()
                for begin in range(0, len(self.rows), self.spec["batch_size"]):
                    selected = self.rows[begin:begin + self.spec["batch_size"]]
                    batch = make_batch([r["history"] for r in selected], [r["target_item"] for r in selected],
                        [r["user_id"] for r in selected], index, self.config["training"]["input_tokens"], device)
                    scorer = ModelScorer(model, batch, index)
                    paths, _ = target_diagnostics(scorer, index.sids[[r["target_item"] for r in selected]],
                        [r["probe_id"] for r in selected], self.run_id, epoch)
                    lookup = {r["probe_id"]: r for r in selected}
                    for row in paths:
                        case = lookup[row.pop("case_id")]
                        records.append({**row, "probe_id": case["probe_id"], "split": case["split"],
                            "target_item": case["target_item"], "attempted_updates": attempted_updates,
                            "effective_updates": effective_updates})
        finally:
            model.train(was_training)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        # A complete snapshot is atomic and independently readable, even when
        # a subsequent model fails. It is separate from checkpoint-level tables.
        path = self.directory / f"step-{attempted_updates:06d}.parquet"
        temporary = path.with_suffix(".pending")
        schema = pa.schema([field for field in self.store.schema["target_path"] if field.name != "case_id"] + [
            pa.field("probe_id", pa.string()), pa.field("split", pa.string()), pa.field("target_item", pa.int64()),
            pa.field("attempted_updates", pa.int64()), pa.field("effective_updates", pa.int64())])
        pq.write_table(pa.Table.from_pylist(records, schema=schema), temporary, compression="zstd")
        temporary.replace(path)
        self.store.write("artifacts", [{"path": str(path.relative_to(self.store.root)),
            "sha256": file_sha256(path), "role": "fixed_training_probe"}])
        elapsed = time.monotonic() - started
        self.seconds += elapsed
        self.last_step = attempted_updates
        _event("sid_study_probe", run_id=self.run_id, epoch=epoch, attempted_updates=attempted_updates,
            effective_updates=effective_updates, cases=len(self.rows), seconds=elapsed, total_probe_seconds=self.seconds)
