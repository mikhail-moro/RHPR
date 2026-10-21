> Historical report from 2026-10-20; research branch closed on 2026-10-21. Any proposed runs or next steps below are historical and cancelled. See the [closure record](../README.md) for the final status.

# Organic versus recommendation-driven repeats on original validation

Read-only CPU audit. Only uid/is_organic source column chunks were downloaded; no item/time/playback columns or GPU inference.

Training organic fraction: 51.00% of events. Validation all-event organic fraction: 50.86%.

| Target origin | Anchors | Share | Repeat R128 | Seen1152 | FrequencyR128 NDCG10 | FuXi NDCG10 | QSSM2 NDCG10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| organic | 14,660 | 55.67% | 51.47% | 75.53% | 0.163124 | 0.037202 | 0.025320 |
| recommendation_driven | 11,675 | 44.33% | 6.22% | 50.56% | 0.010786 | 0.007292 | 0.006791 |

Repeat shares are anchor fractions. Quality is macro-user within each origin group. Users overlap and within-group denominators differ; global-weight contributions in JSON reconstruct the original metrics exactly.

The original source event is joined by UID and original parallel-array offset (train length + future index), with pinned-source provenance and explicit binary/length checks. Actual model input hashes match for every anchor.

The local Parquet container is sparse: its logical size preserves original offsets, while only requested chunks/footer/header are physically stored. No full4GB source file was transferred. See the fetch manifest for exact network and allocated byte counts.

Limitations:
- Origin is the recorded event source, not randomized assignment; origin groups differ in users/items/context. Their quality differences are descriptive, not causal recommendation effects.
- An organic event does not necessarily mean explicit choice of this particular track, and a recommendation event does not prove satisfaction.
- Only user/item/time and origin are used here; playback completion and track duration were not fetched or analysed.
- Target-origin grouping happens after prediction and must not become an input that leaks the target-generation process.
- Users have exceptionally long histories; results do not estimate all listeners. Validation selects the first event per occupied day.
- Within-origin macro-user denominators differ and users overlap groups. Global-denominator contributions are supplied separately.
- All models are frozen, validation-selected p60 single-seed runs; no new test or neural inference was run.

[Full JSON](../aggregates/music-origin-audit.json), [fetch manifest](../aggregates/music-origin-fetch-manifest.json), [analysis script](../analysis/analyze-music-origin.py).
