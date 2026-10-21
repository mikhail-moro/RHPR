# Long-history next-item evaluation data

> Historical research, closed 2026-10-21. See the [closure archive](research/closed-history-compression/README.md) for final results and status.

The compression pilot reuses the train-only cohort and catalogue from the
[training-cost profile](long-history-data.md). Eligibility, mapped item IDs and
model training depend only on events before day 286 (`24,710,400` seconds).
Held-out activity does not change the set of 4,427 training users or the
1,049,612-item catalogue.

The global evaluation periods are:

| Split | Inclusive beginning | Exclusive end | Query selection |
|---|---:|---:|---|
| Validation | day 286 (`24,710,400`) | day 293 (`25,315,200`) | First chronological event per user in the period |
| Test | day 293 (`25,315,200`) | day 300 (`25,920,000`) | First chronological event per user in the period |

Each query predicts one next item using only events before that event's
chronological index. Validation uses the training history. Test uses the
training history plus all intervening observed validation-period events, with
model weights frozen. Actual target timestamps are diagnostic metadata and are
never inputs to the model. Repeats, skips and non-organic listens are retained,
matching the training task. These rules define a rolling next-item protocol;
they differ from both the earlier leave-two-out experiments and the Yambda
paper's Listen+ evaluation.

The first event is selected before item mapping. An unsupported target maps to
zero and receives zero ranking credit in the primary all-anchor metric. A
secondary warm-target metric can expose performance conditional on catalogue
support. Users without an event in a period remain in training but have no
query for that period; reports must show this support. User-macro averages use
one query per evaluable user. Paired model comparisons use identical user IDs
and targets.

An unseen **observed context** item also maps to zero. Its chronological slot
and timestamp remain, with a zero item embedding consistently for every model.
Its observed slot remains valid for HSTU and position indexing; padding is
determined from context length, not `item_id != 0`. Deleting unknown events
would change recency and chronological geometry. Zero is never a catalogue
candidate. The catalogue remains fixed from training, without adding unseen
future items or learning their embeddings.

The supplementary private data artifact contains `evaluation.parquet` and
`evaluation-summary.json`. It stores all fixed cohort user IDs, nullable
validation/test targets and timestamps, and the intervening context events
needed by test. Training arrays are reused from the existing prepared dataset;
the supplementary artifact does not duplicate them. Its summary records the
existing four train/catalogue fingerprints, anchor rule, boundaries, support,
unknown counts, canonical content fingerprint and file byte hash. Loading
checks these contracts against the verified training cohort.

```bash
python scripts/prepare_long_evaluation.py \
  --source /workspace/yambda-500m-source/listens.partial.parquet \
  --train-cohort /workspace/yambda500m-profile-dataset \
  --output /workspace/yambda500m-long-evaluation-dataset \
  --validation-end-timestamp 25315200 \
  --test-end-timestamp 25920000
```

The source projection is sparse: only UID, item and timestamp columns should
be read. Preparation verifies every selected source training prefix against
the existing prepared arrays before selecting future anchors.

`iter_long_evaluation_examples(cohort, evaluation_data, split_name=...,
max_seq_len=...)` yields chronological cropped contexts lazily. Its
`target_timestamp` field is diagnostic and must not be forwarded into the
predictor. The training profile selected users with a full **total** 16,384
context, not 16,384 old events plus 128 recent ones. Experiments should declare
the total raw context length and any left padding explicitly.
