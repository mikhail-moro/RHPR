# ML1M frozen ALS channel: completed seed42 screen

Collected2026-11-05. Private Kaggle `carasik/rhpr-ml1m-als-channel/1` COMPLETE, executable `6ecbdaa0b54bd8d353270af203cf6d9d01646a24`; [workflow](https://github.com/mikhail-moro/RHPR/actions/runs/42076145430). No failures. Both new fits completed30epochs/8,880updates/6validation checks/zeroAMPskips. Validation selected epoch25 in both arms; reused content-only ordinary-sum seed42 also selected25.

**Adding frozen ALS as an equally weighted reconstruction channel did not improve downstream quality on this seed. True ALS loses to both the existing content-only control and shuffled ALS.** One-seed evidence is exploratory; all conditional user-bootstrap intervals include zero. It does not establish that collaborative information is useless or that an InfoNCE objective would fail.

## Results

| Tokenizer input | Val NDCG@10 | Test NDCG@10 | Test vs content | Test Recall@10 |
|---|---:|---:|---:|---:|
| Metadata + poster + plot (reused) |0.066660|0.061652|—|0.118090|
| +true ALS |0.064178|0.055118|−10.60%|0.109117|
| +shuffled ALS |0.068415|0.059330|−3.77%|0.118449|

True ALS vs shuffled: validation−6.19%,test−7.10%. True ALS loses to the content-only control at each validation checkpoint after epoch5 (10,15,20,25,30). Shuffled ALS wins validation at25/30 but loses selected test NDCG. Selection uses only validation; no seen-item filtering or beam refill.

Paired-user bootstrap5,000draws,RNG20261105,2,786aligned users per split:

| Test comparison | Absolute NDCG delta |95%conditional interval| Better/worse/tied users |
|---|---:|---|---|
| True ALS − content |−0.006534|[−0.013059,+0.000080]|194/240/2352|
| Shuffled − content |−0.002323|[−0.008604,+0.003903]|209/213/2364|
| True ALS − shuffled |−0.004212|[−0.010608,+0.002239]|196/234/2356|

Intervals condition on fitted models/cohort; they omit training-seed uncertainty, shared target-item dependence and exploratory multiplicity. Even the largest negative contrast narrowly includes zero. Do not infer a confirmed negative population effect or equivalence from this screen.

## What was actually tested

Same clean raw-timeline20-item protocol and case hashes,151166training cases/2786VAL/2786TEST,2874films. Frozen metadataSentenceT5,poster,description-onlyMPNet exactly match the content-only control archive (`fdb8e49551ba896032b93d97a9be51adeb5a776fd160dc199ac5b874b77198e4`). Same neural initialization pins and architecture, AE3500/RQ3000FP32; TIGER4+4d128,FP16/nativeSDPA,batch512/512,30epochs,val5. Details: `gr-ml1m-als-channel.md`.

ALS128factors: implicit0.7.3,20iterations,regularization0.1,alpha20,seed42. Fit on303199binary distinct edges from admitted TRAIN contexts and targets only.2777supported films;97zero/unobserved. No validation/test input is read by factor preparation. Shuffled is a fixed derangement only among supported rows, preserving vector multiset and missingness. ALS reflects observed ratings, including low ratings, and does not model order; it is not a sequential next-item teacher.

ALS occupies the old inactive128D`people` slot; **there is no people/graph input**. Total architecture and parameter initialization match the existing control. Both new arms activate that head and add a unit-weight reconstruction objective: coefficient sum3→4. Thus true-vs-shuffled controls this extra objective/active head, while comparison to content alone also includes the change in objective scale and active capacity. It is not an isolated test of relative loss balancing.

## Reconstruction and index diagnostics

Final normalized reconstruction MSE:

| Channel | Content | +true ALS | +shuffled ALS |
|---|---:|---:|---:|
| Metadata |0.82563|0.80647|0.81517|
| Poster |0.84761|0.85002|0.83792|
| Plot |0.88948|0.88560|0.87642|
| ALS (internal people slot) |—|0.90029|0.97150|

True ALS is more reconstructible than the shuffled assignment and metadata MSE also improves, while downstream quality declines. This demonstrates that those reconstruction improvements do not guarantee ranking improvements in these runs; it does not identify objective mismatch as the cause.

Code usage by level: content255/252/231,trueALS256/256/248,shuffled256/252/249. Unique semantic triples2871/2871/2873 of2874,largest collision group2 for each; suffix SIDs remain unique. No gross occupancy/collision collapse is visible.

All78sampled off-diagonal last-shared-layer gradient cosine pairs in each new arm are positive (minimum0.159trueALS/0.177shuffled), across13probes. These sparse probes cover one matrix, not all parameters or steps; they neither establish nor rule out global gradient conflict or dominance. MTL modes were ordinary sum throughout. Contrastive behavior-based training and joint TIGER/tokenizer optimization were not tested.

## Time, verification and saved evidence

New session38.07minutes; finished2026-11-05T18:14:48.305286+00:00 (21:14:48Moscow). Training epochs mean59.66s for trueALS,58.09s shuffled; tokenizer training289.03s/284.73s. Timings include concurrentT4execution, not isolated latency benchmarks.

CPU replay independently checks source/config, input manifest/case hashes, shared content archive, ALS manifest/permutation/missingness, factor provenance, initial model pins, warm-up transfer/optimizer flags, full epoch/update counts, validation checkpoint selection, reconstructed SIDs/codebook counts and raw beam metrics.42prediction files rescored:28from the prior four-method seed42 run plus14from the new two fits;21of these belong to the three models reported here. Full NDCG/Recall@5/@10 verified without inference or retraining. Selected results all correspond to epoch25.

Artifacts: `docs/gr-ml1m-als-channel-results.json`; CPU replay `scripts/research/analyze_ml1m_als.py`; local analysis `/tmp/ml1m-als-analysis` includes PNG/PDF,per-user metrics/ranks,curves,verification/hash inventories. Numerical evidence ZIP `/tmp/ml1m-als-evidence.zip` includes both downloaded sessions, original input, ALS input and reports; `.pt` checkpoints remain in private Kaggle outputs.

```bash
PYTHONPATH=src:. python scripts/research/analyze_ml1m_als.py \
  /path/to/bundle/control /path/to/bundle/als \
  /path/to/bundle/input /path/to/bundle/cf /path/to/new-analysis
```

Working conclusion: no positive evidence to promote the equal-weight ALS reconstruction channel. If pursuing collaborative learning, an explicit behavior-based objective would test a different hypothesis; this screen provides no authorization for new training.
