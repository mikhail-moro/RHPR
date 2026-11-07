# MovieLens100K names-in-text and ID-only SASRec: completed results

Private notebook v1 COMPLETE; executable `8c9200133eee6473776836e3b5344e02446c05a6`. All6fresh fits completed20epochs; experiment body 48.29min. No new GPU work.

| Model | Test NDCG@10 mean ± SD | Test Recall@10 mean | Selected val NDCG@10 mean |
|---|---:|---:|---:|
| TIGER plain text RQ | 0.04143 ± 0.00335 | 0.08766 | 0.05866 |
| TIGER AE text | 0.04016 ± 0.00261 | 0.08165 | 0.05684 |
| TIGER text+poster+plot | 0.04292 ± 0.00305 | 0.09084 | 0.05925 |
| TIGER + graph people | 0.04326 ± 0.00636 | 0.08766 | 0.05676 |
| TIGER + names in text | 0.03772 ± 0.00357 | 0.08059 | 0.05928 |
| SASRec ID-only | 0.05261 ± 0.00261 | 0.11099 | 0.06463 |

## New controls by seed

| Model | Seed42 | Seed101 | Seed202 |
|---|---:|---:|---:|
| TIGER + names in text | 0.03891 | 0.03371 | 0.04055 |
| SASRec ID-only | 0.05411 | 0.05412 | 0.04960 |

## Paired comparisons

| Comparison | Mean relative test change | Positive seeds | Conditional paired-user95% interval, absolute NDCG delta |
|---|---:|---:|---:|
| Names vs no people | -12.1% | 0/3 | [-0.01079, +0.00034] |
| Names vs graph | -12.8% | 1/3 | [-0.01126, +0.00005] |
| SASRec vs names | +39.5% | 3/3 | [+0.00747, +0.02256] |
| SASRec vs graph | +21.6% | 3/3 | [+0.00174, +0.01695] |
| SASRec vs no people | +22.6% | 3/3 | [+0.00218, +0.01720] |
| SASRec vs plain text GR | +27.0% | 3/3 | [+0.00332, +0.01931] |

Names-in-text hurts test versus the same multimodal setup without people on all3seeds. It also loses on average versus graph-people, with mixed signs. This is an observation about this encoder/tokenizer/training recipe, not proof that Sentence-T5 cannot encode names. Both representations change the quantized index; there is no controlled mechanism attribution.

ID-only SASRec wins on test in all3seeds versus the names-in-text control, the stronger graph-people reference and the no-people multimodal reference. Its selected validation mean is also higher. This establishes the current baseline result, not a universal ranking of dense and generative retrieval or an independently tuned SASRec optimum.

## Capacity and speed

| Predictor | Active nonembedding parameters | Supported forward GFLOPs | Mean epoch seconds |
|---|---:|---:|---:|
| TIGER + names in text | 4590464 | 0.42737 | 50.75 |
| SASRec ID-only | 4609536 | 0.18737 | 21.45 |

Parameter capacity differs by0.415%; actual FLOPs do not match. The counter includes supported B1training-forward operations for20valid history items and output heads, FMA=2, without backward/beam/offline AE or encoding. SASRec has full-catalogue item CE; TIGER has summed SID-token CE. Loss scales are not comparable.

| Predictor | Batch size | Mean of three seed median latencies, ms/batch | Seed median range, ms |
|---|---:|---:|---:|
| TIGER + names in text | 1 | 43.51 | 42.71–45.03 |
| TIGER + names in text | 32 | 69.16 | 67.95–70.24 |
| SASRec ID-only | 1 | 8.32 | 7.96–8.73 |
| SASRec ID-only | 32 | 8.86 | 8.60–9.14 |

FP32 resident-model recommendation on the same32real training histories, each20valid events;10warmups/50samples per batch shape, same case digest across all6fits. SASRec exact catalogue scoring+top20 versus TIGER free beam20. Excludes input H2D, CPU decoding and offline index preparation. This is one2T4session with concurrent workers, not a dedicated serving deployment. The catalogue has only1273items: exhaustive dense scoring is cheap here; this does not establish efficiency at production catalogue sizes.

## Selection, diagnostics and verification

SASRec best epochs: {42: 10, 101: 5, 202: 10}. Names-in-text TIGER best epochs: {42: 10, 101: 20, 202: 15}. All models nevertheless completed20epochs and were selected using validation only. SASRec validation peaks early in this budget; its larger capacity is not evidence that20epochs is an optimal training regime.

Names-in-text quantizers remain well occupied: {42: (1272, 244), 101: (1273, 240), 202: (1273, 254)} (unique triples, used first-level codes). All1273fullSIDs remain unique after suffix. This again shows occupancy alone does not guarantee better NDCG.

Independently rescored 90 saved validation/test files, including all30newfiles, against catalogue item IDs or each model's own SID map. Same943users/targets/split;20epochs,1709400examples,6680updates,zeroAMPskips per fit. Initial hashes pinned per predictor/seed; native capacity report matches requested tolerance. All3AE transfers/fresh optimizers/initial latents verified. Frozen names vectors equal oldstage07T5vectors exactly; poster/plot/masks unchanged; graph input inactive.

All ratings are retained, with strict earlier-timestamp contexts, timestamp-group holdouts and full-catalogue evaluation; this is next rating in MovieLens, not actual VOD viewing chronology. SASRec uses no seen-item mask or sampled negatives. TIGER retains its hashed-user input; SASRec is item-ID-only. Repeated generator trajectories are not guaranteed bitwise deterministic under the existing numerical policy.

Conditional user bootstrap intervals in comparisons.json exclude training/runtime uncertainty and multiple-comparison correction. Three fitted seeds on943shared users are not2829independent users.

Artifacts: `/workspace/artifacts/movielens100k-names-sasrec-controls/analysis` (report,CSV,plots,compressed ranks,verification); full numerical bundle without large checkpoints: `/tmp/movielens100k-names-sasrec-controls-numerical-evidence.zip`. Large weights remain private Kaggle.
