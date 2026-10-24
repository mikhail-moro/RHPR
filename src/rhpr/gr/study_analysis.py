"""Read, audit, and replay SID study evidence without an application/UI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

from rhpr.gr.data import file_sha256
from rhpr.gr.study_store import StudyStore


PRIMARY_KEYS = {
    "items": "item_id", "events": "user_id,position", "cases": "case_id",
    "case_history": "case_id,position", "training_windows": "epoch,ordinal",
    "runs": "run_id", "item_codes": "run_id,item_id", "geometry": "run_id,item_id,level",
    "prefixes": "run_id,prefix", "token_usage": "run_id,level,token",
    "index_comparisons": "run_a,run_b,level", "case_index": "run_id,case_id,level",
    "checkpoints": "run_id,epoch", "training_epochs": "run_id,epoch",
    "exposures": "run_id,epoch,item_id", "model_inputs": "run_id,case_id",
    "target_path": "run_id,epoch,case_id,level",
    "case_metrics": "run_id,epoch,case_id,scoring,search,beam_width",
    "predictions": "run_id,epoch,case_id,scoring,search,beam_width,rank",
    "beam_trace": "run_id,epoch,case_id,scoring,beam_width,level,prefix",
    "interventions": "intervention_id", "item_features": "item_id,name",
    "train_edges": "source_item,target_item,kind", "artifacts": "path",
    "intervention_predictions": "intervention_id,condition,rank",
    "evaluations": "run_id,epoch,split",
}


def audit(store, *, verify_files=False):
    """Fail on duplicated observations, broken joins and unaligned code paths."""
    errors = []
    with store.connect() as db:
        counts = {name: db.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0] for name in store.schema}
        for table, keys in PRIMARY_KEYS.items():
            n = db.execute(f'SELECT count(*) FROM (SELECT {keys} FROM "{table}" GROUP BY {keys} HAVING count(*)>1)').fetchone()[0]
            if n:
                errors.append(f"{table}: {n} duplicated keys")
        relations = [("case_history", "case_id", "cases", "case_id"),
                     ("cases", "target_item", "items", "item_id"),
                     ("events", "item_id", "items", "item_id"),
                     ("item_codes", "run_id", "runs", "run_id"),
                     ("item_codes", "item_id", "items", "item_id"),
                     ("case_metrics", "case_id", "cases", "case_id"),
                     ("case_metrics", "run_id", "runs", "run_id")]
        for table, schema in store.schema.items():
            for key, parent in (("run_id", "runs"), ("case_id", "cases"), ("item_id", "items")):
                relation = (table, key, parent, key)
                if key in schema.names and table != parent and relation not in relations:
                    relations.append(relation)
        for child, key, parent, foreign in relations:
            n = db.execute(f'SELECT count(*) FROM "{child}" c ANTI JOIN "{parent}" p ON c.{key}=p.{foreign}').fetchone()[0]
            if n:
                errors.append(f"{child}.{key}: {n} missing {parent} rows")
        n = db.execute("SELECT count(DISTINCT initial_state_sha256) FROM runs").fetchone()[0]
        if n > 1:
            errors.append("Generator initialization differs between runs")
        # A full SID must map to one item in each realization.
        n = db.execute("SELECT count(*) FROM (SELECT run_id,sid FROM item_codes GROUP BY ALL HAVING count(*)>1)").fetchone()[0]
        if n:
            errors.append("Non-unique complete semantic IDs")
        n = db.execute("""SELECT count(*) FROM target_path t JOIN cases c USING(case_id)
            JOIN item_codes i ON i.run_id=t.run_id AND i.item_id=c.target_item
            WHERE t.token<>list_extract(i.sid,t.level+1)""").fetchone()[0]
        if n:
            errors.append("Target path token differs from indexed target")
        n = db.execute("""SELECT count(*) FROM training_windows w LEFT JOIN events e
            ON w.user_id=e.user_id AND w.end-1=e.position
            WHERE e.item_id IS NULL OR e.item_id<>w.target_item OR e.split<>'train'
            OR w.visible_start<w.start OR w.visible_start>=w.end-1""").fetchone()[0]
        if n:
            errors.append("Training windows have invalid target positions/context boundaries")
        for run_id, epoch, split, cases, exact_cases in db.execute("SELECT run_id,epoch,split,cases,exact_cases FROM evaluations").fetchall():
            params = [run_id, epoch, split]
            actual = db.execute("""SELECT count(*) FROM target_path t JOIN cases c USING(case_id)
                WHERE run_id=? AND epoch=? AND split=?""", params).fetchone()[0]
            depth = db.execute("SELECT len(sid) FROM item_codes WHERE run_id=? LIMIT 1", [run_id]).fetchone()[0]
            if actual != cases * depth:
                errors.append(f"Incomplete correct-path evaluation: {run_id}/{epoch}/{split}")
            actual = db.execute("""SELECT count(*) FROM case_metrics m JOIN cases c USING(case_id)
                WHERE run_id=? AND epoch=? AND split=? AND search='exact'""", params).fetchone()[0]
            if actual != exact_cases * 2:
                errors.append(f"Incomplete exact evaluation: {run_id}/{epoch}/{split}")
    if verify_files:
        for row in store.read("artifacts") + store.read("checkpoints"):
            path = store.root / row["path"]
            if not path.resolve().is_relative_to(store.root.resolve()) or file_sha256(path) != row["sha256"]:
                errors.append(f"Artifact digest differs: {row['path']}")
    if errors:
        raise ValueError("; ".join(errors))
    return {"passed": True, "rows": counts, "files_verified": verify_files}


def replay_intervention(root, run_id, epoch, case_id, positions, source_path, *, device="cpu", beam_width=10):
    """Cache an explicit fixed-position masking intervention on a saved model.

    No claim about causal real-world behaviour. One call needs a fresh process
    because the upstream source uses a top-level package named src.
    """
    import numpy as np
    import torch
    from rhpr.gr.genpas import activate_source
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import build_study_model
    from rhpr.gr.study_search import ModelScorer, mask_history_events, target_diagnostics, beam_search
    import tempfile
    store = StudyStore(root)
    with store.connect() as db:
        case = db.execute("SELECT target_item,user_id FROM cases WHERE case_id=?", [case_id]).fetchone()
        checkpoint = db.execute("SELECT path,sha256 FROM checkpoints WHERE run_id=? AND epoch=?", [run_id, epoch]).fetchone()
        inputs = db.execute("SELECT input_ids,attention_mask FROM model_inputs WHERE run_id=? AND case_id=?", [run_id, case_id]).fetchone()
        codes = db.execute("SELECT sid FROM item_codes WHERE run_id=? ORDER BY item_id", [run_id]).fetchall()
    if not case or not checkpoint or not inputs or not codes:
        raise ValueError("Unknown case, checkpoint or index")
    path = store.root / checkpoint[0]
    if file_sha256(path) != checkpoint[1]:
        raise ValueError("Checkpoint digest mismatch")
    config = json.loads((store.root / "study-config.json").read_text())
    index = SidIndex(np.asarray([row[0][:-1] for row in codes]), width=config["quantizer"]["codebook_size"])
    state = torch.load(path, map_location="cpu", weights_only=False)
    np.testing.assert_array_equal(state["sids"], index.sids)
    source = activate_source(source_path)
    with tempfile.TemporaryDirectory() as temporary:
        model = build_study_model(index, source, Path(temporary), config["generator"])
        model.load_state_dict(state["model"], strict=True)
        model.to(device).eval()
        batch = {"input_ids": torch.tensor([inputs[0]], device=device),
            "attention_mask_encoder": torch.tensor([inputs[1]], device=device),
            "user_id": torch.tensor([[case[1]]], device=device)}
        changed = mask_history_events(batch, positions, index.sid_length)
        target = index.sids[[case[0]]]
        original = ModelScorer(model, batch, index)
        modified = ModelScorer(model, changed, index)
        before, _ = target_diagnostics(original, target, [case_id], run_id, epoch)
        after, _ = target_diagnostics(modified, target, [case_id], run_id, epoch)
        identifier = uuid.uuid4().hex
        result = []
        for scoring, field in (("raw", "raw_logprob"), ("local", "local_logprob")):
            ranks = []
            for condition, scorer in (("baseline", original), ("changed", modified)):
                beams, _ = beam_search(scorer, beam_width, scoring, targets=target)
                items = [index.sid_to_item[sid] for sid, score in beams[0]]
                ranks.append(items.index(case[0]) + 1 if case[0] in items else None)
                store.write("intervention_predictions", [{"intervention_id": identifier + "-" + scoring,
                    "condition": condition, "rank": rank, "item_id": index.sid_to_item[sid], "log_score": score}
                    for rank, (sid, score) in enumerate(beams[0], 1)])
            result.append({"intervention_id": identifier + "-" + scoring, "run_id": run_id,
                "epoch": epoch, "case_id": case_id, "kind": "mask_events_keep_positions",
                "positions": sorted(set(positions)), "scoring": scoring,
                "baseline_target_logprob": sum(row[field] for row in before),
                "changed_target_logprob": sum(row[field] for row in after), "beam_width": beam_width,
                "baseline_target_rank": ranks[0], "changed_target_rank": ranks[1]})
    store.write("interventions", result)
    store.inventory()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("audit")
    check.add_argument("--verify-files", action="store_true")
    commands.add_parser("database")
    query = commands.add_parser("query")
    query.add_argument("sql")
    intervention = commands.add_parser("mask")
    intervention.add_argument("--run", required=True)
    intervention.add_argument("--epoch", required=True, type=int)
    intervention.add_argument("--case", required=True)
    intervention.add_argument("--positions", nargs="+", required=True, type=int)
    intervention.add_argument("--source", type=Path, required=True)
    intervention.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    intervention.add_argument("--beam-width", default=10, type=int)
    args = parser.parse_args()
    if not (args.root / "schema.json").is_file():
        parser.error("Not a SID study directory")
    store = StudyStore(args.root)
    if args.command == "audit":
        result = audit(store, verify_files=args.verify_files)
    elif args.command == "database":
        with store.connect(args.root / "study.duckdb"):
            pass
        result = {"database": str(args.root / "study.duckdb")}
    elif args.command == "query":
        with store.connect() as db:
            cursor = db.execute(args.sql)
            result = [dict(zip((c[0] for c in cursor.description), row)) for row in cursor.fetchall()]
    else:
        result = replay_intervention(args.root, args.run, args.epoch, args.case, args.positions, args.source,
                                     device=args.device, beam_width=args.beam_width)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
