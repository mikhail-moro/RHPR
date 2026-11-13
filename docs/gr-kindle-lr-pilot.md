# Kindle Recent20 learning-rate pilot

Two authorized fresh fits on the fixed seed42 item index from the completed
Kindle dense100K pair pilot. Control is the existing constant AdamLR0.001
Recent20 trajectory, not a new training job. No Sentence-T5 encoding, pair
encoding, KMeans fitting, Pair20 fitting, new seeds or TEST evaluation.

Both arms use the same initial parameters as the original Recent20 (including
its unused pair-input table), same92133 TRAIN prefixes/2625VAL cases, seed42
shuffle/dropout protocol, mini-TIGER4+4/d64/FF512, FP16/nativeSDPA/batch512,
Adamweight_decay0.0001/clip1. Full60epochs, VAL every5, best VAL NDCG@10
selection, no early stop before60. VAL teacher-forced CE is additionally logged
for diagnosis; evaluation preserves RNG and does not select by this loss.

- cosine_003: peak0.003, final0.0003.
- cosine_010: peak0.01, final0.001.

Use standard PyTorch LambdaLR: linear warm-up over360successful optimizer
updates (two nominal epochs), then half-cosine to the floor at update10800.
The first applied LR is peak/360, update360 uses the peak, and update10800
uses the floor. FP16-skipped updates do not advance the scheduler; if skips
occur, the end LR can remain slightly above the floor at epoch60. Log first,
last and next LR per epoch plus the exact scheduler state, Adam/scaler and RNG
in last checkpoints. This same frozen60-epoch schedule can be resumed after
interruption; extending its horizon later would be a separate training policy.
No automatic precision/LR fallback or failed-arm retries.

Use one isolated worker per T4; wall-clock estimate30–45min including startup,
not a guarantee. Server cap80min, per-fit budget3300s, stop only at epoch
boundaries. Failure to finish60epochs/12VAL checks is a failed pilot rather
than a shortened successful fit. Preserve best/last weights and all VAL
predictions. TEST stays closed even after selection.

Pin parent manifest/config/item-SIDs/control-curves by SHA256/size and source
8b575bbe21ea3e477cf78539fad92f9488c08719. New initializer must equal original
145019d618c46a5a0455496d44a0d517eaf82e8c234651551a15b64019d21277.
Prepared data hashes and generator/data/training invariants are checked before
workers train. The parent kernel contains TEST artifacts, but the pilot copies
only item SIDs and its old validation curves and never builds TEST examples.

Compare best and fixed VAL scores versus consumed optimizer updates, examples
and actual training time; report numerical failures, skipped updates and loss.
Primary practical question: can60epochs of a new schedule approach/exceed the
old120-epoch selected validation result, and improve on its same-budget60epoch
result? This is one model/seed and examined validation set, not proof of a
portable schedule or downstream TEST improvement. Do not relaunch the nine-fit
matrix automatically after this screen.

Preflight:1082 full regression tests passed,152optional skips. Actual saved
parent artifacts passed byte/size pins. Native CPU mini-TIGER forward/backward
on prepared Kindle TRAIN cases produced finite loss/gradients, matched initial
weights for both arms, and the expected first scheduler steps. GPU initializer
matching remains an enforced precondition in each worker. No TEST cases built.

Submitted once as private carasik/rhpr-kindle-recent20-lr-pilot/version1,
RUNNING at server check. GitHub workflow43517280237success, executable
ac7fc6a3aa587aa0366de504182d9c831f0c00ff. Pulled source/config/private/offline/
T4/dataset/parent metadata verified. Receipt:gr-kindle-lr-pilot-submission.json.
Submission is not training completion; collect two completed VAL trajectories
and verify LR/updates/pins before comparing. No additional runs authorized.
