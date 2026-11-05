# ALS objective ablations: completed seed42 results

Private `carasik/rhpr-ml1m-als-objectives/1` COMPLETE, executable `a102ceac9a374f5c2627cc3c09cb673afdecf1e5`; [workflow](https://github.com/mikhail-moro/RHPR/actions/runs/42091502981). Both fresh fits completed30epochs/8880updates/6validation checks/zeroAMPskips. No failures or additional training. Runtime37.53minutes, finished2026-11-05T19:50:00.395036+00:00 (22:50Moscow).

**Neither GradNorm nor removing CF reconstruction recovered ranking quality on seed42.** GradNorm slightly increased the ALS task weight, while the lambda0 arm kept a trained, influential ALS input head and correctly left its reconstruction head unchanged. This does not support the specific hypothesis that excessive CF reconstruction alone explains the prior degradation. It does not rule out other forms of modality interference or benefits under different seeds/objectives.

## Selected results

All checkpoints selected solely by validation NDCG@10. Same2874films/2786VAL/2786TEST/151166TRAINcases,rawclean20+val+test,frozenfeatures and initialization pins;AE3500/RQ3000FP32;TIGER4+4d128,FP16/nativeSDPA,batch512/512,30epochs,val5,FP32freebeam20test. Existing references were reused without fitting.

| Variant | Best epoch | Val NDCG@10 | Test NDCG@10 | Test vs content | Test vs ALS sum |
|---|---:|---:|---:|---:|---:|
| Content only (reused) |25|0.066660|0.061652|—|—|
| ALS ordinary sum (reused) |25|0.064178|0.055118|−10.60%|—|
| Shuffled ALS sum (reused) |25|0.068415|0.059330|−3.77%|—|
| ALS GradNorm |25|0.062935|0.053425|−13.34%|−3.07%|
| ALS lambda_CF=0 |20|0.065669|0.048492|−21.35%|−12.02%|

Removing CF reconstruction improves validation2.32% relative to ALS sum but degrades selected test12.02%. Thus an improved validation endpoint in this exploratory one-seed comparison does not establish improved generalization. No checkpoint was selected by test. There is no gross codebook collapse: usedcodes GradNorm256/254/248,lambda0 256/254/252; unique triples2872/2873of2874,largest collision group2.

Conditional paired-user bootstrap5000draws,RNG20261105:

| Test contrast | Absolute delta |95% interval |
|---|---:|---|
| GradNorm − content |−0.008227|[−0.014696,−0.001723]|
| GradNorm − ALS sum |−0.001693|[−0.008240,+0.004665]|
| Lambda0 − content |−0.013160|[−0.019746,−0.006419]|
| Lambda0 − ALS sum |−0.006626|[−0.013186,+0.000395]|

The new models are worse than the fitted content control beyond conditional user-sampling uncertainty, but neither difference against fitted ALS sum excludes zero. These intervals omit training-seed variation, target-item dependence and exploratory multiplicity; they do not establish a seed-robust negative effect. All validation contrast intervals include zero. This remains one pipeline seed42, not a multi-seed conclusion.

## GradNorm behavior

Final weights in metadata/poster/plot/ALS order: **0.877889 /0.929638 /1.127218 /1.065255**, sum4. GradNorm balances reconstruction tasks using the existing alpha1.5 adaptation, at AE and RQ stages. It did not learn to suppress ALS; its weight increased6.53% relative to unit weight. This is compatible with its objective, which sees reconstruction rates/gradient norms, not NDCG. Final ALS MSE0.89550 vs0.90029 for ALS sum, yet test quality worsens. Better ALS reconstruction again did not guarantee better ranking in this run.

Raw per-head MSE, adaptive weights, gradient probes, code usage and losses are preserved in numerical outputs. The ordinary reconstruction gradient probes from the earlier run cover one shared matrix, not global gradient dominance; these ablations do not constitute a causal proof of dominance or objective mismatch.

## Lambda0 implementation and checkpoint verification

All four inputs remain active and observed (ALS2777supported films,97masked), same neural architecture/initialization. Fixed reconstruction weights1/1/1/0; ALS raw target MSE remains logged (1.13416final) but contributes zero to the objective. The ordinary fixed-weight native loss path is used because the MTL adapter would override fixed weights. `mtl_final_weights=null` and no adapter learning-curve file in this arm are expected. Raw head/AE/codebook/generator curves remain available.

Prelaunch explicit gradient check established zero ALS decoder reconstruction gradient while its encoder head receives nonzero gradient from content objectives. Collection additionally verifies the actual saved quantizer checkpoint SHA `713dc465bbe67977207cf624c24989724f957c12bb3dee6836779463cabc6a4f` against its output report. Using its trained neural modules on CPU:

- Maximum ALS decoder-head parameter change from exact seed42 initialization: **0**.
- Maximum ALS encoder-head parameter change: **0.07322**, confirming it trained.
- Masking the full ALS encoder block after its MLP changes latents for **all2777supported items**; median relative latent L2 change **7.25%**,95th percentile20.11%.

This is a read-only tokenizer sensitivity probe, not generator inference or evidence that the surviving signal is useful. Input influences the latent even without its own reconstruction task. No generator checkpoint was downloaded or evaluated. One quantizer checkpoint is retained for audit.

Compared with ALS sum, lambda0 also changes total fixed reconstruction coefficient3vs4 and reconstruction strength relative to quantization. GradNorm keeps coefficient sum4. These differences and one seed limit causal attribution. The conclusion is that these particular interventions did not recover quality, not that CF reconstruction is necessary or all modality dominance explanations are false.

## Verification and artifacts

CPU replay independently rescored35prediction files across the five reported models (sixvalidation+onetest each), verifying user/target arrays,NDCG/Recall@5/@10,invalid/duplicate beams,validation-only best selection,full budget,case/manifest hashes,exact shared content feature archive,pins,warm-up transfer flags,SID/codebook consistency,ALS factor provenance and objective configuration. Content archive remains `fdb8e49551ba896032b93d97a9be51adeb5a776fd160dc199ac5b874b77198e4`.

New runtime source/config204files were audited before launch. Collection regression:1000tests,91optional skips,0failures. No new GPU work or training launched while collecting.

Report JSON `docs/gr-ml1m-als-objectives-results.json`; replay `scripts/research/analyze_ml1m_als_objectives.py`. Local `/tmp/ml1m-als-objectives-analysis` contains summary,user-level scores,allcurves,prediction verification,PNG/PDFcomparison and input-audit JSON. Evidence ZIP `/tmp/ml1m-als-objectives-evidence.zip` preserves raw numeric outputs and required inputs/references; generator weights remain private on Kaggle.

```bash
PYTHONPATH=src:. python scripts/research/analyze_ml1m_als_objectives.py \
  /path/to/control /path/to/channel /path/to/objectives \
  /path/to/input /path/to/cf /path/to/analysis
```

Original content-only sum remains the working reference. There is no positive evidence here to promote ALS GradNorm or lambda0, and no extra seed/HPO/contrastive run is authorized by completion of this experiment.
