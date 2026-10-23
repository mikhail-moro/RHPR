"""Finite source-user epochs retaining GRID's sampled window augmentation."""
from __future__ import annotations

import time

from rhpr.gr.data import build_examples
from rhpr.gr.experiment import _evaluate, _event, _write


def train_epoch_generator(model, users, sid_index, config, output, device, *,
                          max_seconds=None, evaluator=_evaluate, clock=time.monotonic):
    import torch
    from rhpr.gr.paper_profile import _update
    from rhpr.gr.paper_protocol import PaperEpochStream

    spec = config["training"]
    for name in ("max_epochs", "validation_interval_epochs", "patience_epochs"):
        if type(spec.get(name)) is not int or spec[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    optimizer = torch.optim.Adam(model.parameters(), lr=spec["learning_rate"],
                                 weight_decay=spec["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=spec["precision"] == "fp16", init_scale=1024)
    validation = build_examples(users, config["data"]["history_items"], "validation")
    started = clock()
    limit = spec["max_seconds"] if max_seconds is None else max_seconds
    updates, skipped, examples = 0, 0, 0
    best_recall, best_ndcg = -1.0, -1.0
    selected_epoch, selected_update, best_ndcg_epoch = 0, 0, 0
    train_seconds, validation_seconds = 0.0, 0.0
    curves = []
    reason = "epoch_limit"
    last_validated = 0

    def validate(row, trigger):
        nonlocal best_recall, best_ndcg, selected_epoch, selected_update
        nonlocal best_ndcg_epoch, validation_seconds, last_validated
        before = clock()
        scores, _, _ = evaluator(model, validation, sid_index, spec["evaluation_batch_size"],
            device, precision=spec["evaluation_precision"])
        validation_seconds += clock() - before
        row.update(validation=scores, validation_trigger=trigger)
        if scores["recall@10"] > best_recall:
            best_recall, selected_epoch, selected_update = scores["recall@10"], row["epoch"], updates
            torch.save({"model": model.state_dict(), "epoch": selected_epoch, "update": updates,
                "validation": scores, "criterion": "validation_recall@10"}, output / "generator-best.pt")
        if scores["ndcg@10"] > best_ndcg:
            best_ndcg, best_ndcg_epoch = scores["ndcg@10"], row["epoch"]
        last_validated = row["epoch"]
        row.update(best_recall=best_recall, best_ndcg=best_ndcg, best_ndcg_epoch=best_ndcg_epoch)
        _event("grid_epoch_validation", **row)

    for epoch in range(1, spec["max_epochs"] + 1):
        # Stop at a complete source epoch. The mean gives a bounded estimate;
        # the independent Kaggle server timeout remains the hard ceiling.
        mean_epoch = train_seconds / len(curves) if curves else 0.0
        if curves and clock() - started + mean_epoch >= limit:
            reason = "time_budget"
            break
        stream = PaperEpochStream(users, sid_index.sids, seed=config["seed"] + epoch,
            source_batch_size=spec["source_user_batch_size"], effective_batch_size=spec["batch_size"],
            sequence_length=spec["sequence_length"], max_augmented_batch=spec["max_augmented_batch"])
        model.train()
        before, epoch_loss, epoch_examples, epoch_updates = clock(), 0.0, 0, 0
        for batch in stream:
            count = len(batch["future_ids"])
            result = _update(model, batch, spec["physical_batch_size"], optimizer, scaler,
                             spec["precision"], device, spec["grad_clip_norm"])
            examples += count
            epoch_examples += count
            epoch_loss += result["loss"] * count
            if result["optimizer_update_applied"]:
                updates += 1
                epoch_updates += 1
            else:
                skipped += 1
                if skipped > 16:
                    raise FloatingPointError("repeated GRID epoch AMP gradient overflows")
        seconds = clock() - before
        train_seconds += seconds
        stream_report = stream.report()
        if not stream_report["epoch_complete"] or stream_report["source_rows_consumed"] != len(users):
            raise RuntimeError("source-user epoch did not consume every user exactly once")
        row = {"epoch": epoch, "update": updates, "epoch_updates": epoch_updates,
            "epoch_examples": epoch_examples, "model_examples_consumed": examples,
            "train_loss_sum_over_codes": epoch_loss / epoch_examples,
            "epoch_training_seconds": seconds, "training_seconds": train_seconds,
            "amp_skipped_updates": skipped, "source_users": len(users)}
        curves.append(row)
        if epoch % spec["validation_interval_epochs"] == 0:
            validate(row, "epoch_interval")
        _write(output / "learning-curves.json", curves)
        _write(output / "training-stream.json", stream_report)
        # Epoch-local sampling is reconstructed from seed+next_epoch on resume.
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch, "update": updates,
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
            "next_epoch_seed": config["seed"] + epoch + 1,
            "best_recall": best_recall, "best_ndcg": best_ndcg,
            "best_ndcg_epoch": best_ndcg_epoch, "stream_report": stream_report,
            "automatic_resume": False}, output / "generator-last.pt")
        _event("grid_epoch_complete", **row)
        if last_validated == epoch and epoch - best_ndcg_epoch >= spec["patience_epochs"]:
            reason = "early_stopping"
            break
    if curves and last_validated != curves[-1]["epoch"]:
        validate(curves[-1], "final_budget_or_epoch_boundary")
        _write(output / "learning-curves.json", curves)
    checkpoint = torch.load(output / "generator-best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result = {"epochs": len(curves), "updates": updates, "selected_epoch": selected_epoch,
        "selected_update": selected_update, "best_ndcg_epoch": best_ndcg_epoch,
        "checkpoint_metric": "validation_recall@10", "early_stopping_metric": "validation_ndcg@10",
        "best_validation_recall10": best_recall, "best_validation_ndcg10": best_ndcg,
        "selected_validation": checkpoint["validation"], "stop_reason": reason,
        "early_stopping_reached": reason == "early_stopping", "training_seconds": train_seconds,
        "validation_seconds": validation_seconds, "elapsed_seconds": clock() - started,
        "validation_checks": sum("validation" in row for row in curves),
        "model_examples_consumed": examples, "amp_skipped_updates": skipped,
        "precision": spec["precision"], "epoch_definition": stream_report["epoch_definition"],
        "validation_interval_epochs": spec["validation_interval_epochs"],
        "patience_epochs": spec["patience_epochs"]}
    _write(output / "training-selection.json", result)
    return result
