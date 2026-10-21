> Historical report from 2026-10-20; research branch closed on 2026-10-21. Any proposed runs or next steps below are historical and cancelled. See the [closure record](../README.md) for the final status.

# Artist/album return patterns on the existing validation anchors

Read-only CPU analysis; no training or neural inference. Mappings were downloaded from the exact pinned Yambda revision.

Support: 26,335 anchors, 4,384 users, 24,935 known targets and 1,400 OOV targets.

| Pattern | Count | Fraction of all anchors |
|---|---:|---:|
| track_seen_R128 | 8,272 | 31.41% |
| artist_seen_R128 | 14,937 | 56.72% |
| album_seen_R128 | 11,386 | 43.24% |
| track_unseen_R128_but_artist_seen_R128 | 6,665 | 25.31% |
| track_unseen_R128_but_album_seen_R128 | 3,114 | 11.82% |
| track_unseen_visible1152 | 7,960 | 30.23% |
| track_unseen_visible1152_but_artist_seen_visible1152 | 4,435 | 16.84% |
| track_unseen_visible1152_but_album_seen_visible1152 | 2,514 | 9.55% |

Metadata coverage:
- artist: 99.4993% catalogue items; 99.9880% known targets; 351 catalogue items have multiple associated entities.
  Target ambiguity: 0.00% of covered targets have multiple entities. Among 24,932 single-entity targets, entity overlap with R128 is 59.91%; unseen R128 track but familiar entity is 26.73%.
- album: 99.9248% catalogue items; 100.0000% known targets; 223,610 catalogue items have multiple associated entities.
  Target ambiguity: 36.77% of covered targets have multiple entities. Among 15,767 single-entity targets, entity overlap with R128 is 42.77%; unseen R128 track but familiar entity is 9.03%.

The actual pinned sequential500m listens Parquet schema, read by just2125 footer bytes, includes `is_organic`, `played_ratio_pct` and `track_length_seconds` in addition to user/item/timestamp. Earlier partial-projection manifests do not describe the full source schema. These fields can therefore be restored from the same source revision. No full listening file was downloaded.

Interpretation: track-level novelty and artist/album-level familiarity can differ. This motivates separate multi-resolution diagnostics; it does not identify moods or prove a particular recommendation objective.

Limitations:
- Artist and album IDs are observed metadata identities; they do not label playlist, genre, intent or mood.
- Artist/album overlap uses any associated entity for many-to-many mappings, never an arbitrary first entity. Multiple album links may include compilations/editions; any-shared-album is a relation diagnostic, not proof of intentional album listening.
- OOV observed IDs were already mapped to zero in prepared data; their original entity identities cannot be reconstructed here. Unknown target entities remain unassessed.
- An entity unseen in the visible1152 may have been present earlier in the complete user history; this analysis does not call it globally new.
- User cohort was selected for exceptionally long histories and is not representative of all music listeners.
- All-target fractions retain missing entities and OOV as no observed overlap; covered-target fractions give complementary coverage-aware descriptions.
- Only original validation anchors analysed, no test metrics or neural logits were computed.

[Full JSON](../aggregates/music-metadata-audit.json), [fetch manifest](../aggregates/music-metadata-fetch-manifest.json), [analysis script](../analysis/analyze-music-metadata.py).
