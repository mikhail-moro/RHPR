# ML1M clean contiguous holdouts: multimodal MTL screening

Collected 2026-11-05. Private Kaggle `carasik/rhpr-ml1m-clean-mtl/1` completed successfully. Executable commit `20a4c08534cd6853e3fd64340f2dc429e6bed3a6`; submitted by [workflow 42020589648](https://github.com/mikhail-moro/RHPR/actions/runs/42020589648). This report changes no experiment defaults and launches no further training.

PCGrad is the most promising candidate in this one-seed screen: +5.26% validation and +5.67% test NDCG@10 relative to the ordinary sum. Paired-user confidence intervals include zero. These results do not yet establish a reproducible MTL advantage.

## Shared protocol

- Seed 42, four fresh tokenizer/generator pipelines: sum, uncertainty weighting (UW), GradNorm and PCGrad. Two T4 workers, two sequential pipelines per worker.
- Fixed catalogue of 2,874 films; 2,786 users in each of validation/test; 151,166 training windows. Raw timestamps and missing-feature events determine eligibility before filtering. Validation and test are adjacent original singleton timestamp groups following an uninterrupted covered block of 20 items. No histories are spliced across missing-feature barriers. Training windows use strictly earlier timestamp groups; tied training targets share the same strict past.
- Metadata Sentence-T5 (title including year, genres, year, runtime, tags), poster vectors and description-only MPNet. Frozen shared features; no people graph. The old four-slot encoder capacity is retained with a zero, unobserved people slot. Metadata has no truncated items (maximum 298 tokens); plot uses 384-token chunks without truncation (2,909 chunks, maximum document 724 tokens).
- AE 3,500 steps followed by RQ-VAE 3,000 steps, both FP32; transferred neural weights and fresh RQ optimizer verified. Three codebooks of 256 entries plus collision suffix.
- TIGER: 4 encoder + 4 decoder layers, d_model 128, d_ff 1024, 6 attention heads, d_kv 64, dropout 0.1. Native Transformers 5.22.1 SDPA, FP16 training, FP32 evaluation, batch 512/512, LR 0.001. Existing 2,000 user-token hash bins retained.
- Every generator completed 30 epochs, 8,880 optimizer updates, six validation checks (every five epochs), zero AMP-skipped updates. Checkpoint selected using validation NDCG@10; exactly one final test evaluation per pipeline. Free beam 20, no seen-item filtering. No early stopping or scheduler in this bounded screen.

All methods share generator initialization pin `d57fc2e783c0a04bfc567c0082541f757e241d1c53d7e79b3e8f281bf2eb9337`, tokenizer initialization pin `54253fbb9f6837546a0c29291db31a12cb0be65f75609a6d8f7c3d89aeccc69b` and frozen feature-file SHA256 `fdb8e49551ba896032b93d97a9be51adeb5a776fd160dc199ac5b874b77198e4`.

## Downstream results

| Tokenizer loss method | Selected epoch | Val NDCG@10 | Test NDCG@10 | Test vs sum | Test Recall@10 |
|---|---:|---:|---:|---:|---:|
| Sum | 25 | 0.066660 | 0.061652 | — | 0.118090 |
| UW | 30 | 0.068528 | 0.056512 | −8.34% | 0.112706 |
| GradNorm | 30 | 0.065226 | 0.060147 | −2.44% | 0.118808 |
| PCGrad | 30 | 0.070168 | 0.065146 | +5.67% | 0.127064 |

PCGrad test absolute NDCG gain is 0.003493, with 241 users improving, 215 worsening and 2,330 tied. Conditional paired-user bootstrap (5,000 draws, seed 20261105) gives a 95% interval of **[−0.003231, +0.010149]**. Validation gain is 0.003508, interval **[−0.003848, +0.010696]**. Every method-vs-sum interval includes zero; see JSON for all comparisons.

These intervals condition on the fitted seed and fixed cohort. They exclude training-seed uncertainty, dependence through shared target items and correction for exploratory multiple comparisons. Test is reported for assessment; future method selection should continue to use validation. This does not show that clean filtering caused any quality change relative to ML100K: dataset, catalogue, batch size and runtime also differ.

PCGrad leads at epochs 20 and 30, but loses to sum at 25. UW improves validation but degrades test. Three selected checkpoints are at the 30-epoch budget cap, so convergence has not been established. The economical follow-up would replicate **sum and PCGrad** with two additional seeds; this is a proposal, not a submitted job.

## Tokenizer diagnostics

| Method | Final reconstruction weights: text / poster / plot | Used codes by level | Unique semantic triples / 2,874 | Tokenizer training seconds |
|---|---|---|---:|---:|
| Sum | 1 / 1 / 1 | 255 / 252 / 231 | 2,871 | 242.85 |
| UW | 1.185 / 1.179 / 1.117 | 254 / 251 / 238 | 2,869 | 244.48 |
| GradNorm | 0.921 / 0.926 / 1.154 | 254 / 247 / 248 | 2,871 | 302.66 |
| PCGrad | 1 / 1 / 1 | 253 / 252 / 244 | 2,870 | 408.21 |

No semantic collision group exceeds two items; suffixes produce unique final SIDs. No evidence of gross codebook collapse here. Small differences in usage or reconstruction do not establish why downstream quality differs.

PCGrad projected gradients on **652 / 6,500 optimizer steps (10.03%)**, with 2,240 individual pair projections: 489 affected AE steps and another 163 RQ steps. Thus PCGrad actually changes optimization in this experiment. Earlier ML100K had 44 / 19,500 affected steps across three pipelines (0.23%); that comparison is observational and confounded, not a causal explanation of the gain.

The stored diagnostic cosines sample one shared encoder weight matrix every 500 steps, whereas PCGrad uses all shared parameters on every step. PCGrad's sampled last-layer pairs are positive; this does not contradict projections elsewhere or between probes. UW has one negative sampled pair. Zero projection counters on the other methods mean the projection algorithm was not applied, not that their gradients never conflict.

UW's final weights sum to 3.4805 rather than 3: much of the change is a common reconstruction scale, so this is not a clean experiment isolating only relative modality balancing. GradNorm weights sum to approximately 3.

## Timing and verification

Experiment manifest duration: **4,400.63 seconds (73.34 minutes)**; start `2026-11-05T11:39:47.933653+00:00`, finish `2026-11-05T12:53:08.567361+00:00`. Submit-to-finish is about 74.4 minutes. Mean generator training epoch: sum 56.81 s, UW 57.85 s, GradNorm 56.77 s, PCGrad 57.51 s. Validation and preparation are excluded from these per-epoch training timings. These are observed concurrent-worker timings, not isolated performance benchmarks.

CPU replay independently verified:

- Input manifest/file SHA256 and sizes, 151,166/2,786/2,786 case shapes/hashes, aligned user/target arrays, and test context equal to shifted validation context plus the observed validation target.
- Shared source IDs/features, no truncation, exact initial pins, warm-up transfer flags, full epoch/update counts and validation-based checkpoint selection.
- Reconstructed suffix SIDs and codebook usage against semantic codes.
- **All 28 saved prediction files** (24 validation + 4 test), recomputing NDCG/Recall@5/@10 from raw SID beams. No duplicated candidates. Six invalid candidate SIDs across all 1,560,160 candidates: UW val15=2, val30=1, test=1; GradNorm val25=2. Invalid beams remain failed candidates at their original ranks; no filtering/refill changes the protocol. This tiny incidence does not explain the method gaps.

The submission audit also verified all 203 embedded source files and the exact config/dataset source set. Prelaunch checks: 982 regression passes, 13 skips, nine new protocol tests, four native end-to-end probes and two multibatch probes. Collection required no retraining.

## Preserved artifacts and CPU replay

Machine-readable report: `docs/gr-ml1m-clean-mtl-results.json`. Replay script: `scripts/research/analyze_ml1m_clean_mtl.py`.

Evidence bundle: `/tmp/ml1m-clean-mtl-evidence.zip`. It contains downloaded numerical outputs/logs, the exact input dataset, source audit, submission receipt, report, replay script and analysis: `summary.json`, `metrics.csv`, `users.csv`, `training-curves.json`, `mtl-curves.json`, verification/hash inventories and PNG/PDF figures. Large `.pt` checkpoints remain in the private versioned Kaggle output and are not included in this compact bundle.

To replay after extracting, use the executable source commit (or a descendant with the same GR modules), NumPy and Matplotlib:

```bash
PYTHONPATH=src:. python scripts/research/analyze_ml1m_clean_mtl.py \
  /path/to/bundle/raw /path/to/bundle/input /path/to/replayed-analysis
```

The replay uses no model inference, GPU, credentials or network.
