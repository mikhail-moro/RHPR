# Seed42 text AE warm-up: completed results

Private notebook COMPLETE, executable `444ba8238b2e244b498c4bb23ada89c769723b63`. Experiment body 20.25min. Only one text-only fit; no new GPU work.

| Model | Selected val NDCG@10 | Epoch | Test NDCG@10 | Test Recall@10 | Semantic triples | Used level-1 codes |
|---|---:|---:|---:|---:|---:|---:|
| Plain RQ | 0.06900 | 20 | 0.04341 | 0.09120 | 1271 | 140/256 |
| Heads, direct RQ | 0.05402 | 10 | 0.03438 | 0.06999 | 1025 | 15/256 |
| Heads, AE then RQ | 0.06102 | 15 | 0.04210 | 0.09014 | 1273 | 252/256 |

Test NDCG@10 improves +22.5% versus direct headed control; -3.0% versus plain RQ. Validation improves relative to headed control but remains below plain RQ. One seed does not establish a stable gain.

AE stops at 4500 updates by spectral plateau in 69.6s. Eligible active-mode counts: [(2000, 7), (2500, 7), (3000, 6), (3500, 6), (4000, 6), (4500, 6)]. Raw reconstruction is still improving; this is a spectral stopping criterion, not reconstruction convergence.

Continuous AE text MSE: 1.0014 → 0.7488. Final quantized MSE 0.7250, versus direct 0.8351. Both use identical fixed-normalized targets; do not compare this raw loss with the old plain recipe.

Final books use [252, 238, 231] codes. All 1273 semantic triples are unique already at RQ update500; the direct control ends at 1025 triples and [15, 152, 188] used codes. Suffix resolves any direct-control collisions; no items were dropped.

Native RQ takes 91.5s versus direct 94.1s. TIGER mean epoch takes 50.73s versus 50.95s. The incremental AE costs ~70s; generation speed has not materially changed.

All 15 saved prediction files across three models were independently rescored. Same943users/targets/split; same frozen text, normalization and headed quantizer initial parameters; same generator initialization; all20epochs/1709400examples/6680updates/zeroAMPskips. All1273initial RQ latents exactly match AE exports; books untouched during AE and optimizer accumulators fresh.

This supports the staged recipe as a way to repair the weak headed control. It does not establish that parameter count was the cause, that occupancy explains NDCG, or that the spectral stopping rule is optimal. Warm-up, native initialization on trained latents, first-batch eval behavior and extra compute change together. The next direct application would be the multimodal variants under this recipe, with no launch authorized by this analysis.

Paired user differences and conditional bootstrap intervals are in paired-comparisons.json; they omit training-seed variation. Full checkpoint weights remain in the private Kaggle output. Numerical outputs, hashes, curves, prediction files and SID maps are preserved in numerical-evidence.zip.

Conditional paired-user bootstrap 95% interval for test NDCG@10 delta versus direct headed control: [-0.00235,+0.01794], crossing zero. This does not establish a stable improvement even conditional on the fitted models. Against plain RQ: [-0.01252,+0.01014].
