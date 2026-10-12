# Train-visible long-history profiling data

`rhpr.sasrec.long_history_data` prepares the input shared by the four initial
training-cost measurements. Eligibility and the item mapping use training
events only. No validation/test targets are selected by this profiling step.

The source is the official sequential Yambda-500M `listens.parquet`, which has
one row per user and singular list columns `item_id` and `timestamp`. The Kaggle
`ploshkin/yambda` mirror used by earlier runs exposes 50M only; preparation from
the official 500M file therefore happens before uploading a private prepared
dataset. Pin the official source revision and download identity in the caller's
manifest. The loader records source path, size, and modification time.

For the initial profile the explicit training boundary is `24_710_400` seconds
(day 286). The preparation contract is:

- Retain all listens with observed `timestamp < 24_710_400`, including skips,
  non-organic listens, and repeated items. Sort timestamps stably if necessary.
- Select users with at least **16,385 training events**, enough for 16,384
  shifted input events and their next-item targets. Full source history length
  alone does not establish eligibility.
- Build a sorted catalogue from those users' training prefixes only. External
  item IDs are mapped to `1..N` as `int32`; `0` is padding. Store observed input
  timestamps as `uint32` seconds.
- Decode only UID, item, and timestamp columns in small Arrow batches. Keep
  compact NumPy arrays rather than Python event lists. The source's single large
  row group may still require substantial decoder memory; batch size is not a
  guarantee that compressed-column buffers fit within the batch size.

Example preparation:

```python
from rhpr.sasrec.long_history_data import (
    load_long_history_cohort, write_prepared_cohort,
)

cohort = load_long_history_cohort(
    "/workspace/yambda-500m-source/listens.parquet",
    train_end_timestamp=24_710_400,
    max_seq_len=16_384,
    output_dir="/workspace/long-history-profile/cohort-diagnostics",
)
write_prepared_cohort(cohort, "/workspace/prepared-long-history-cohort")
```

Preparation diagnostics are aggregate `cohort-summary.json` and
`per-user-profile.parquet` (one count row per source UID). The prepared private
dataset contains only:

| File | Contents |
|---|---|
| `train.parquet` | `uid:int64`, mapped `item_id:list[int32]`, observed `timestamp:list[uint32]` |
| `catalog.parquet` | Sorted training-only `external_item_id:int64` |
| `cohort-summary.json` | Boundaries, counts, distributions, canonical content signatures, Parquet byte hashes |

`load_prepared_cohort(directory)` checks file SHA256 and sizes, schema,
eligibility, timestamp ordering and cutoff, mapped-ID range, sorted catalogue,
and canonical UID/length, timestamp, item, and catalogue signatures. The GPU
profiler must also check that its configured cutoff/window matches this summary.
Files containing event arrays are private data artifacts and never belong in
the source repository or public experiment output.

`make_long_training_pair(user, max_seq_len=16_384, end_index=e)` interprets `e`
as an exclusive offset into the **training prefix including the last target**.
It returns inputs `train[e-16_385:e-1]`, next-item targets
`train[e-16_384:e]`, and timestamps aligned only to the input events. Valid
endings are `16_385 <= e <= len(train)`. The schedule can vary `e` across
training visits while remaining identical for all architectures.
