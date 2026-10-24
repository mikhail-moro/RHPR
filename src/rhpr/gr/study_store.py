"""Versioned, portable Parquet evidence store; DuckDB is an optional reader.

Only completed atomic parts are visible. Tables exist even when a measurement
has not been made; nullable ranks never masquerade as a zero-quality item.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import uuid


# All positions/levels are zero based, ranks are one based, scores are natural
# log probabilities. JSON fields preserve heterogeneous source metadata.
FIELDS = {
    "items": "item_id:i source_id:i text:s text_sha256:s train_events:i train_users:i",
    "item_features": "item_id:i name:s value_json:s provenance:s status:s",
    "events": "user_id:i position:i item_id:i split:s timestamp:s",
    "cases": "case_id:s user_id:i split:s target_position:i target_item:i history:li visible_history:li target_repeats:i target_last_lag:i exact_panel:b panel_stratum:s panel_probability:f",
    "case_history": "case_id:s position:i item_id:i visible:b input_position:i",
    "training_windows": "epoch:i ordinal:i source_group:i user_id:i start:i end:i visible_start:i target_item:i",
    "train_edges": "source_item:i target_item:i kind:s count:i source_support:i target_support:i",
    "runs": "run_id:s quantizer_seed:i model_seed:i data_seed:i initial_state_sha256:s config_json:s",
    "item_codes": "run_id:s item_id:i sid:li",
    "geometry": "run_id:s item_id:i level:i token:i nearest_token:i alternative_token:i assigned_distance:f alternative_distance:f distance_margin:f residual_norm_before:f residual_norm_after:f",
    "prefixes": "run_id:s prefix:s parent:s level:i token:i is_suffix:b items:i train_events:i train_users:i child_count:i target_events:i child_entropy:f child_max_share:f",
    "token_usage": "run_id:s level:i token:i items:i parent_count:i train_events:i",
    "index_comparisons": "run_a:s run_b:s level:i prefix_ari:f prefix_nmi:f token_ari:f token_nmi:f",
    "case_index": "run_id:s case_id:s level:i distinct_prefixes:i prefix_entropy:f target_shared_events:i target_shared_last_lag:i",
    "checkpoints": "run_id:s epoch:i optimizer_updates:i examples:i train_loss:f seconds:f path:s sha256:s",
    "training_epochs": "run_id:s epoch:i examples:i attempted_updates:i effective_updates:i skipped_updates:i mean_loss:f learning_rate:f scaler_scale:f grad_norm_mean:f training_seconds:f evaluation_seconds:f cuda_peak_bytes:i",
    "evaluations": "run_id:s epoch:i split:s cases:i exact_cases:i seconds:f metrics_json:s",
    "model_inputs": "run_id:s case_id:s input_ids:li attention_mask:li",
    "exposures": "run_id:s epoch:i item_id:i context_count:i target_count:i target_loss_sum:f target_level_loss_sums:lf",
    "target_path": "run_id:s epoch:i case_id:s level:i prefix:s token:i logits:lf raw_logprob:f local_logprob:f token_rank:i best_other_token:i valid_mass:f",
    "predictions": "run_id:s epoch:i case_id:s scoring:s search:s beam_width:i rank:i item_id:i sid:li log_score:f",
    "case_metrics": "run_id:s epoch:i case_id:s scoring:s search:s beam_width:i target_rank:i raw_target_logprob:f valid_catalogue_mass:f recall5:f recall10:f ndcg5:f ndcg10:f exact_top10_overlap:f target_lost10:b target_promoted10:b",
    "beam_trace": "run_id:s epoch:i case_id:s scoring:s beam_width:i level:i prefix:s rank:i log_score:f target_prefix:b",
    "interventions": "intervention_id:s run_id:s epoch:i case_id:s kind:s positions:li scoring:s baseline_target_logprob:f changed_target_logprob:f beam_width:i baseline_target_rank:i changed_target_rank:i",
    "intervention_predictions": "intervention_id:s condition:s rank:i item_id:i log_score:f",
    "artifacts": "path:s sha256:s role:s",
}


def schemas():
    import pyarrow as pa
    types = {"s": pa.string(), "i": pa.int64(), "f": pa.float64(), "b": pa.bool_(),
             "li": pa.list_(pa.int64()), "lf": pa.list_(pa.float32())}
    return {name: pa.schema([(key, types[kind]) for key, kind in
                             (column.split(":") for column in columns.split())])
            for name, columns in FIELDS.items()}


def prefix_key(prefix):
    return "/".join(map(str, prefix))


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite measurement; use null only for explicitly missing observations")
    if isinstance(value, (list, tuple)):
        for part in value:
            _finite(part)


class StudyStore:
    def __init__(self, root, *, shard=None):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.root = Path(root)
        self.tables = self.root / "tables"
        self.schema = schemas()
        self.pending = {}
        self.shard = shard
        manifest = self.root / "schema.json"
        declaration = {"version": 1, "positions": "zero based", "ranks": "one based",
                       "scores": "natural log probability", "tables": FIELDS}
        if shard is not None:
            if not shard or Path(shard).name != shard or shard in {".", ".."} or "\\" in shard:
                raise ValueError("Writer shard must be a safe basename")
            # Parent initializes the store once. Workers never rewrite shared
            # schema or schema-only files and own separate output directories.
            if not manifest.is_file() or json.loads(manifest.read_text()) != declaration:
                raise ValueError("Worker requires an initialized matching store")
            for name in self.schema:
                (self.tables / name / shard).mkdir(parents=True, exist_ok=False)
            return
        self.tables.mkdir(parents=True, exist_ok=True)
        for name, schema in self.schema.items():
            directory = self.tables / name
            directory.mkdir(exist_ok=True)
            if not (directory / "schema.parquet").exists():
                pq.write_table(pa.Table.from_pylist([], schema=schema), directory / "schema.parquet")
        if manifest.exists() and json.loads(manifest.read_text()) != declaration:
            raise ValueError("Evidence schema mismatch")
        manifest.write_text(json.dumps(declaration, indent=2) + "\n")

    def write(self, name, rows):
        import pyarrow as pa
        import pyarrow.parquet as pq
        rows = list(rows)
        if not rows:
            return
        schema = self.schema[name]
        for row in rows:
            if set(row) != set(schema.names):
                raise ValueError(f"{name}: expected exactly {schema.names}, received {list(row)}")
            for value in row.values():
                _finite(value)
        limits = {"target_path": 256, "predictions": 10000, "beam_trace": 20000,
                  "case_metrics": 10000, "case_index": 10000, "model_inputs": 2048, "artifacts": 1000}
        if name in limits:
            pending = self.pending.setdefault(name, [])
            pending.extend(rows)
            if len(pending) < limits[name]:
                return
            rows, self.pending[name] = pending, []
        return self._commit(name, rows)

    def _commit(self, name, rows):
        import pyarrow as pa
        import pyarrow.parquet as pq
        schema = self.schema[name]
        table = pa.Table.from_pylist(rows, schema=schema)
        directory = self.tables / name
        if self.shard is not None:
            directory /= self.shard
        path = directory / (uuid.uuid4().hex + ".parquet")
        temporary = path.with_suffix(".pending")
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
        return path

    def flush(self):
        for name, rows in self.pending.items():
            if rows:
                self._commit(name, rows)
                self.pending[name] = []

    def read(self, name):
        import pyarrow.parquet as pq
        self.flush()
        return pq.read_table(sorted((self.tables / name).rglob("*.parquet"))).to_pylist()

    def connect(self, database=":memory:"):
        """Create relocatable views; re-open with this method after moving files."""
        import duckdb
        self.flush()
        connection = duckdb.connect(str(database))
        for name in FIELDS:
            path = str((self.tables / name / "**" / "*.parquet").resolve()).replace("'", "''")
            connection.execute(f'CREATE OR REPLACE VIEW "{name}" AS SELECT * FROM read_parquet(\'{path}\', hive_partitioning=false)')
        connection.execute("""CREATE OR REPLACE VIEW item_quality AS
            SELECT m.run_id, m.epoch, c.split, c.target_item AS item_id,
                   m.scoring, m.search, m.beam_width, count(*) AS support,
                   avg(m.recall10) AS recall10, avg(m.ndcg10) AS ndcg10
            FROM case_metrics m JOIN cases c USING(case_id)
            GROUP BY ALL""")
        connection.execute("""CREATE OR REPLACE VIEW paired_quality AS
            SELECT a.run_id AS run_a, b.run_id AS run_b, a.case_id, a.epoch,
                   a.scoring, a.search, a.beam_width,
                   b.ndcg10-a.ndcg10 AS delta_ndcg10
            FROM case_metrics a JOIN case_metrics b
            ON a.case_id=b.case_id AND a.epoch=b.epoch AND a.scoring=b.scoring
            AND a.search=b.search AND a.beam_width IS NOT DISTINCT FROM b.beam_width AND a.run_id<b.run_id""")
        return connection

    def inventory(self):
        from rhpr.gr.data import file_sha256
        import pyarrow.parquet as pq
        self.flush()
        files = [{"path": str(p.relative_to(self.root)), "sha256": file_sha256(p),
                  "rows": pq.ParquetFile(p).metadata.num_rows}
                 for p in sorted(self.tables.rglob("*.parquet"))]
        (self.root / "inventory.json").write_text(json.dumps(files, indent=2) + "\n")
        return files
