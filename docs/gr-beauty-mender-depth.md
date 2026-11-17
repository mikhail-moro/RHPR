# Beauty: a two-fit generator architecture pilot

Authorized2026-11-17: test the generator configuration from Preference
Discerning with LLM-Enhanced Generative Retrieval on Recent3 and Recent20.
Exactly TWO fresh generator fits,seed42,one per T4. Config:
`experiments/beauty_mender_depth_2t4.json`;runner:`rhpr.gr.beauty_pairs`.
Existing Recent12 session stays untouched;no extra HPO or seeds.

## Verified source and scope

Paper:https://arxiv.org/html/2412.08604v2 (PDFdownloaded and Figure7visually
inspected). Official code revision`cf2d58f014477ca3702d63c7937289cf3c2f9ade`:
[TIGER_Beauty.yaml](https://github.com/facebookresearch/preference_discerning/blob/cf2d58f014477ca3702d63c7937289cf3c2f9ade/configs/setting/TIGER_Beauty.yaml).

Author configuration:6encoder+6decoder layers,d_model128,d_ff1024,6heads,
d_kv64,dropout.2,ReLU. Their script also disables USERembeddings,uses
Sentence-T5-XXL/modifiedRQ-VAE,includes first-target examples,beam30/invalid
filtering,AdamWlr.0003/weight_decay.035/batch256,warmup10000/max200000steps.
These differences are NOT adopted. We borrow ONLY6+6depth anddropout.2.
We do not migrate the codebase or claim exact reproduction of their metrics.

Original Beauty/1 generator:4+4/dropout.1,5108992parameters. Adapted generator:
6+6/dropout.2,7338496parameters (+43.64%). This is depth PLUS regularization,
not an isolated causal test of capacity. A positive result cannot distinguish
the effects of depth and dropout without a later ablation.

Figure7(left) shows absolute NumberofMatches versus sequence-length bucket,
not per-bucket Recall. Short buckets are much larger in Beauty. Their
observation does not establish normalized quality degradation or a causal
effect of adding context. Comparing two context caps on the same users is a
different experiment. Neither their graph nor ours proves a capacity limit.
The paper inherits a20-item cap rather than demonstrating its optimality.

## Fixed data, index and training

All original22363users/12101items/198502events;131413TRAINprefixes,22363VAL
andTESTtargets,leave-last-two;TESTobservesVAL. Same supplied metadata,
Sentence-T5-base,max256/nativeGRIDnormalizedRQ-KMeans3x256/seed42,
mandatory3+1uniqueSID. Recompute singles from the SAME two existing offline
datasets. Require original SIDfileSHA
`624a0e23fc8c4ca844972beb4718800121d23f9a12a85c8135a4c26767dd02ab`
before any generator training;check original raw case/target hashes too.
No pair encoding/fitting/cache transfer/newdataset or parentkernel. Empty
pairlookup artifacts are unused;unused pairinput parameter table stays in
both models to preserve architecture comparability with earlier controls.

Exactly `mender_recent20_42` onworker0 and `mender_recent3_42` onworker1.
Same initializer WITHIN the new architecture:
`42d133f62430a906aeb64942d36247f8614eee7b26b87dd91c5e3b22af9d8ecb`.
It necessarily differs from the old architecture's initializer. Shared
training seed42controls constructor/dropout/epoch order;all TRAINtargets
appear once per epoch. No compression/augmentation/random tokenization.

Keep USER2000,noSEP,SDPA,FP16compiled512/512,Adam.003/wd1e-4/clip1,
warmup2/cosine60to.001,floorafter60,min40/max120,VALevery5,
EMAalpha.5/min_delta.0005/patience20epochs. RawVALitemNDCG10selects checkpoint;
EMAonlystops. Save full best/last optimizer/scaler/scheduler/RNGstates,
allVALpredictions/lossesbySIDlevel/curves/padding/time/memory/FLOPs.
TESTonce after selection,FP32beam20/batch64/all12101items,no seenfilter,
samplednegatives or invalid-path reranking. No selection fromTEST.

Fitlimit10000s/server21600s;preserve checkpoint on failure,mark incomplete,
no automatic fallback/retry/extension. Epochcap is censoring,not plateau.
Estimate1.5-2.5hours after bootstrap,dependent on stopping/hardware contention;
not a benchmark promise. Quota26.13GPUh before preparation. Oneprivateoffline
2T4session,twoexistingdatasets,no notifications. Do not duplicate Recent12.

## Compute and interpretation

Actual saved originalSIDs used for CPUpreflight,locally only. Real supported
mathSDPAteacher-forced4codeforward/backward finite at both widths;initial
weights unchanged. B1FMA2forward excludes beam/backward/text/preparation:

| History | Old4+4 GFLOPs | New6+6 GFLOPs | EncoderpositionsinclUSER |
|---|---:|---:|---:|
|Recent3|.082200520|.123168070|13|
|Recent20|.427088200|.640482218|81|

Reproducer:`scripts/research/preflight_beauty_recent12.py --config
experiments/beauty_mender_depth_2t4.json --output
docs/gr-beauty-mender-depth-preflight.json`.

Compare newRecent20minusRecent3 against historicalBeauty/1Recent20minus
Recent3 on the SAME VALusers and targets;also report overall changes at each
cap and actual length groups<=3,4-6,7-10,11-20. Inspect learning curves,
teacher-forced losses and cap censoring. TESTis descriptive confirmation,
not tuning feedback or a source of new subgroup rules. Paired user intervals
condition on this fixed seed/index/checkpoint selection and do not establish
seed robustness. Improvement of both caps equally is general model gain,
not evidence that the bigger model benefits more from history. No effect
does not prove history useless or rule out other optimizers/tokenizers.

Historical controls are reused;no rerun or inference on their checkpoints:
Recent3VAL.0439901876/TEST.0299792672,best50/stop75;
Recent20VAL.0457632662/TEST.0317360948,best55/stop75.
Their original source is`7828d110117889667c3a5e5877f7be9bd0222c13`.

CPUpreflight passed;full1170tests total,152optional skips,0failures,
120.990seconds. Private/offline bundle built. Recent12serverstatusCOMPLETE
was observed before this submission;its quality audit is separate.

SubmittedONCEprivate`carasik/rhpr-beauty-mender-depth/1`,source
`a68b70dbfe15b386d40552923afbe9801a7f5b90`,workflow43987690723SUCCESS
(submissiononly),serverRUNNING. All218deployedPythonfiles/exactconfig/
privateoffline/original2datasets/no parentsverified. Receipt:
`gr-beauty-mender-depth-submission.json`. No trainingoutcomes yet.
