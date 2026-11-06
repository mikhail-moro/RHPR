# Frozen projection versus full shared RQ adaptation

Authorized2026-11-06: two30epochfits on existing clean ML1M151166TRAINwindows,
seed42,VAL5,2T4,about40minutes maximum quota. User clarified additional arm must
have NO projection and must train the entire RQ encoder/decoder with ordinary
reconstruction/quantization loss in addition to TIGER's NIP. No direct NIP gradient
through discrete assignments into the encoder is claimed.

projected_frozen: keep pretrained SUMseed42 native64D RQ encoder,decoder andbooks
frozen,SIDs fixed. TIGER input lookup = learned shared Linear64->128(codevector).
Projection starts as identity in first64coordinates andzeros elsewhere,suffix
keeps ordinary trainable128D embeddings. No RQ loss in this arm.

joint_rq: no learned bridge or projection between books and TIGER. Actual native
RQcentroid Parameters are TIGER's input code vectors. Expand RQlatent64->128 by
zero-padding encoder final linear output rows/bias,decoder first input columns,
and centroids. Remaining pretrained encoder/decoder/heads/BNstates copiedexactly.
This preserves initial encoder's first64coordinates,decoder reconstruction and
nearest-neighbor assignments; addedcoordinates can subsequently be modeled by
both encoder and decoder. All RQneuralparameters andcentroids train. Initial
TIGERlookup values and base Transformer weights match the projectioncontrol and
previous initialization experiment,despite different full model parameter shapes.

Joint objective = NIP tokenCEsum + reconstructionSUM + native betaquantizationSUM,
weights1/1/1. Use GRID STE strategy and native BetaQuantizationLoss(beta.25,mean).
RQ encoder receives reconstruction/commitment gradients,decoder receives
reconstruction,books receive NIP+codebookquantization gradients. Native STE blocks
reconstruction gradient into selected codevectors. No gradient through argmin.
Auxiliary512item catalogue-uniform sample with replacement per optimizerstep,
independent NumPyRNG424242. Frozen text/poster/plot features and catalogue scaling
reused exactly;no rating holdout information enters RQobjective. Native RQ math
runs FP32 under an explicit autocast-off scope,TIGER FP16/nativeSDPA. AdamLR.001,
weightdecay.0001 for trainableparameters as previous generator;this is not the
original RQpretraining Adagrad schedule. No new loss/optimizer tuning. RQloss mean
uses128latentdimensions in jointarm;zero expansion initially changes its mean
scale relative to64D pretraining. This is an exploratory combined intervention,
not an isolated causal test of encoder freezing or a matched parameter-count study.

At each joint epochSTART,encode the catalogue using currentRQweights in evalmode,
requantize with current128Dcentroids and rebuild catalogue-order suffixes. Train
with this snapshot for the epoch. Keep all epochSID arrays,churn/occupancy/collision
curves and per-epochNIP/Rec/VQlosses. Persistent activeSIDbuffer restores the
best-by-VAL index along with complete model state before one finalTEST per arm.
Frozenprojection arm keeps originalSIDthroughout. Explicit trial-start/complete/
failure logs. Failed assignments retain candidatecodes/books/latents. Fail on
suffixoverflow or incomplete budget;no fallback,retry or additionalfits.

Same4+4TIGER,d128,FF1024,H6,kv64,dropout.1,512batch,FP32evaluation,
30epochs/8880updates,6VALchecks,VALNDCG10selection,noES/scheduler. Sourceencoder
andfeature checkpoints stay the same pretrainedSUMseed42;do not retrain upstream.
Nominal/active parameters projected8,157,123/4,887,808,joint8,214,531/8,214,531;
inactive people modules remain unused compatibility components. All original
TIGER outputheads remain ordinary separate GRID heads,not tied to input table.

Private carasik/rhpr-ml1m-codebook-adaptation,offline,three existingdatasets and
one existingprivatecleanMTLparent. No new data upload,Telegram or hardwarefallback.
Quota0.65h checked;servercap37min,perfit2050s;may fail explicitly if30epochs do not
fit. Do not silently shorten or extend training. No additional compute authorized.
Config experiments/ml1m_codebook_adaptation_2t4.json.
