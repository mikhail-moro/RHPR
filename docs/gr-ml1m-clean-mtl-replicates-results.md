# Clean ML1M MTL replication: three pipeline seeds

Collected 2026-11-05. Additional seeds101/202 completed in private Kaggle `carasik/rhpr-ml1m-clean-mtl-replicates/1`, executable `fc2a62bdd57af294caadd939bf1bbbda2de7032f`; [submission workflow](https://github.com/mikhail-moro/RHPR/actions/runs/42033993480). Combined with completed seed42 (`carasik/rhpr-ml1m-clean-mtl/1`, executable `20a4c08534cd6853e3fd64340f2dc429e6bed3a6`), this gives three matched pipeline seeds per method.

**The apparent PCGrad advantage on seed42 did not replicate. No method has a consistent downstream advantage over the ordinary sum in this 30-epoch setting.** GradNorm has the highest mean test score, but only wins on seed101; it loses on the other two seeds. This does not establish that MTL cannot help under other configurations, and it does not prove equivalence.

## Shared protocol and completion

Same frozen metadata Sentence-T5 + poster + description-only MPNet features, no people graph; same catalogue2,874, training151,166 cases, validation/test2,786 users each. Original timeline eligibility with an uninterrupted covered20-item block followed by singleton validation/test timestamp groups. Training uses strict-past clean windows, with no splicing across missing-feature barriers or invented within-timestamp order. The full input file hashes and all case hashes match the original screen.

Every method/seed trains its own tokenizer from scratch: continuous AE3,500 steps → RQ-VAE3,000 steps in FP32, three256-entry codebooks plus uniqueness suffix. Same heads, normalization, initialization policy, optimizers and budgets. TIGER4+4,d128,FP16/nativeSDPA,batch512/512,LR0.001,30epochs,val5. Same free beam20, FP32 evaluation, no seen-item filtering. Each final test uses the checkpoint selected solely by validation NDCG@10.

The trial seed controls tokenizer initialization/training, generator initialization and generator epoch permutations. These are **pipeline-seed** replicates; they do not isolate quantizer randomness from generator randomness. The trial-local configuration copy correctly applies seed202 rather than the parent seed101 to generator training. All methods share initialization pins within each seed.

All **12** fits completed30epochs,8,880optimizer updates and6validation checks, with **zero AMP-skipped updates**. No failed trials. Frozen feature arrays are exactly equal between sessions; the feature archive SHA256 is identical: `fdb8e49551ba896032b93d97a9be51adeb5a776fd160dc199ac5b874b77198e4`.

## Selected checkpoint results

Mean ± sample standard deviation over the three pipeline seeds:

| Method | Val NDCG@10 | Test NDCG@10 | Test mean vs sum |
|---|---:|---:|---:|
| Sum | 0.066740 ± 0.005208 | 0.059173 ± 0.006228 | — |
| UW | 0.066528 ± 0.002722 | 0.058160 ± 0.006260 | −1.71% |
| GradNorm | 0.066932 ± 0.003714 | 0.060195 ± 0.001684 | +1.73% |
| PCGrad | 0.067324 ± 0.003333 | 0.058395 ± 0.008523 | −1.32% |

Relative change above is the ratio of means, not the mean of per-seed percentages. Standard deviation describes variation across three pipeline fits, not uncertainty across users.

| Seed | Method | Selected epoch | Val NDCG@10 | Test NDCG@10 | Test vs same-seed sum |
|---:|---|---:|---:|---:|---:|
| 42 | Sum | 25 | 0.066660 | 0.061652 | — |
| 42 | UW | 30 | 0.068528 | 0.056512 | −8.34% |
| 42 | GradNorm | 30 | 0.065226 | 0.060147 | −2.44% |
| 42 | PCGrad | 30 | 0.070168 | 0.065146 | +5.67% |
| 101 | Sum | 25 | 0.061572 | 0.052087 | — |
| 101 | UW | 30 | 0.063428 | 0.052889 | +1.54% |
| 101 | GradNorm | 25 | 0.064377 | 0.058536 | +12.38% |
| 101 | PCGrad | 30 | 0.063657 | 0.048818 | −6.28% |
| 202 | Sum | 30 | 0.071987 | 0.063779 | — |
| 202 | UW | 25 | 0.067627 | 0.065079 | +2.04% |
| 202 | GradNorm | 30 | 0.071192 | 0.061903 | −2.94% |
| 202 | PCGrad | 30 | 0.068147 | 0.061221 | −4.01% |

Validation means differ from sum by UW−0.32%,GradNorm+0.29%,PCGrad+0.88%. The previously observed PCGrad test+5.67% becomes a negative three-seed mean after replication. GradNorm's mean+1.73% is driven by one seed, so selecting it as a winner based on this test mean would be unjustified. Its narrower observed test standard deviation is also based on just three fits and does not establish improved stability.

Eight of12selected checkpoints are at epoch30. These conclusions apply to the fixed budget, not converged optima. No automatic extensions, tuning or extra seeds were launched.

## Paired uncertainty

User bootstrap resamples the **same user indices jointly across all three seeds**, after averaging their within-user method-minus-sum deltas. It does not pretend that3×2,786are independent users. With5,000draws, RNG20261105:

| Method | Mean test delta | Conditional user95% interval | Test seeds better/worse |
|---|---:|---|---|
| UW | −0.001013 | [−0.004860,+0.002758] | 2 / 1 |
| GradNorm | +0.001022 | [−0.002874,+0.004714] | 1 / 2 |
| PCGrad | −0.000778 | [−0.004604,+0.003034] | 1 / 2 |

All intervals include zero, including those for validation. They are conditional on the three fitted pipelines and fixed split; they exclude further training-seed variation and target-item dependence. A descriptive paired-seed exact sign-flip check enumerates all8sign patterns (two-sided p=1.0for every contrast here); with only three seeds it has very low resolution and is not definitive inference. There is neither robust positive evidence nor proof of equal performance.

## Mechanism and diagnostics

PCGrad was active on all seeds:

| Seed | Projected updates / 6,500 | Fraction | AE updates | RQ updates |
|---:|---:|---:|---:|---:|
| 42 | 652 | 10.03% | 489 | 163 |
| 101 | 702 | 10.80% | 502 | 200 |
| 202 | 450 | 6.92% | 368 | 82 |

It therefore changes the optimization, but correcting reconstruction-gradient conflicts is not enough to guarantee better next-item ranking. The projection counts do not explain downstream quality by themselves. Sampled last-layer gradient cosines cover only one matrix every500steps; PCGrad operates on all shared parameters at every step. These scopes must not be conflated.

Code usage, collisions, per-modality reconstruction, final MTL weights and all diagnostic curves are retained in JSON and raw outputs. Better reconstruction/occupancy is not established as a downstream proxy. UW also changes the total reconstruction scale, so its outcome does not isolate only relative modality balancing.

## Verification, time and artifacts

CPU replay independently rescored **all84prediction files** (72validation +12test) from raw SID beams, checking NDCG/Recall@5/@10, original user/target arrays, suffix SID reconstruction, codebook counts, initialization pins, warm-up transfer flags, complete epoch/update counts and validation-only selection. All case files and input manifest hashes were checked. `metrics.json` files are finite flat scalar summaries and match the suite/trial metrics. No duplicate candidates;39invalid candidates across4,680,480saved beams (6in seed42,33in the added-seed session). Invalid candidates retain their original ranks and score as misses; no filtering/refill protocol change.

Added-seed session: **8,785.28seconds =146.42minutes**, started `2026-11-05T13:15:54.887375+00:00`, finished `2026-11-05T15:42:20.172240+00:00`. Generator training epochs average about58seconds (validation/preparation excluded). This agrees with the2.5-hour estimate. Seed42session was73.34minutes; these are concurrent-worker wall times, not isolated latency benchmarks.

Report JSON: `docs/gr-ml1m-clean-mtl-replicates-results.json`. CPU replay: `scripts/research/analyze_ml1m_mtl_replicates.py`. Numerical evidence bundle: `/tmp/ml1m-three-seed-mtl-evidence.zip`, containing both sessions' numerical/log outputs, exact input dataset, analysis with user-level metrics/ranks, all training/MTL curves, figures, verification inventories, source audits and reports. Large `.pt` checkpoints remain in private versioned Kaggle outputs and are excluded from the compact bundle.

Replay from the source checkout with NumPy/Matplotlib, without GPU/network/model inference:

```bash
PYTHONPATH=src:. python scripts/research/analyze_ml1m_mtl_replicates.py \
  /path/to/bundle/seed42 /path/to/bundle/seeds101-202 \
  /path/to/bundle/input /path/to/new-analysis
```

For the current setup the ordinary sum remains a reasonable working control. Record MTL as an ablation without demonstrated improvement; these results are not a basis for promoting PCGrad or GradNorm to a new default.
