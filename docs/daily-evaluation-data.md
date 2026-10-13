# Daily next-item evaluation on the fixed 16K cohort

The new evaluation keeps the same 4,427 training users, 89,191,505 pre-cutoff
events, 1,049,612 training-catalogue items and day-286 training boundary.
Training events and targets do not move into validation or test. The model's
weights remain frozen during evaluation.

For each user, select the first chronological listen of each occupied day:
validation covers days 286–292; test covers days 293–299. Selection happens
before catalogue mapping. A target outside the fixed training catalogue has
mapped ID zero and receives zero full-catalogue ranking credit. This provides
up to seven distinct prediction moments per user in each period. It is a
rolling temporal evaluation, not seven predictions made from one unchanged
context.

The prepared supplement has 26,335 validation anchors from 4,384 users and
25,571 test anchors from 4,349 users. Unsupported targets account for 1,400
validation anchors (5.32%) and 2,059 test anchors (8.05%). There are 2,185 test
users with all seven daily anchors and 900 with six. The first available daily
anchor for every user in each period exactly matches the previous single-anchor
evaluation, including its target, timestamp and full reference context.

Before an anchor, context contains all observed events preceding that anchor,
including previous validation/test anchors and other intervening events. Crop
to the last 16,384 events. Compressed models retain 128 recent events and
compress the older prefix; Recent128 receives the last 128 events of exactly
the same reference context. Never supply the anchor timestamp to the model.
An unknown observed item keeps a chronological slot with zero item embedding;
padding masks use observed lengths rather than `item_id != 0`.

The private supplement stores post-cutoff observations once per user in
`daily-future.parquet` and compact day/offset anchor records in
`daily-anchors.parquet`. It reuses the existing train dataset instead of
duplicating full 16K contexts for every anchor. Every anchor has stable ID
`UID:DAY` and a SHA256 of its full reference context, unchanged if a consumer
crops its input to 128 events. File hashes, training-cohort signatures, the
exact source training prefix and canonical first-per-day anchors are checked
during preparation and loading.

`history-features.parquet` contains the existing 20 model-independent
descriptors for each full reference context. Item frequencies use only the
unchanged pre-cutoff training cohort. Target identity, target support and
target timestamp are excluded from this feature table. User ID, day, split,
anchor ID and context hash are diagnostic join fields, not routing features.
Learn routing rules using validation outcomes; hold test outcomes aside until
the rule is fixed. Aggregate daily outcomes per user before interpreting a
user's preferred compression budget, and bootstrap users rather than treating
correlated anchors as independent users.

Prepare the private supplement with:

```bash
python scripts/prepare_daily_evaluation.py \
  --source /workspace/yambda-500m-source/listens.partial.parquet \
  --train-cohort /workspace/yambda500m-profile-dataset \
  --previous-evaluation /workspace/yambda500m-long-evaluation-dataset \
  --output /workspace/yambda500m-daily-evaluation-dataset \
  --validation-end-timestamp 25315200 \
  --test-end-timestamp 25920000
```

Only UID/item/timestamp column chunks are read from the projected source.
Private event arrays, user features and credentials must not be committed.
