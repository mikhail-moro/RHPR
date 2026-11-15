# Frozen Kindle pair-alignment and single-input probe

Authorized2026-11-15:repeat the earlier ML1M pair-boundary inference study on
current Kindle Mixed32 and original Pair32,adding no-pair Full32. No training,
checkpoint selection,new seed,tokenizer refit or notifications. Both checkpoints
are the previously raw-VAL-selected epoch55,seed42. Exactly28inferences:
two models xVAL/TEST xseven predeclared variants,all958users and original targets.
FP32/SDPA,beam20,batch64,no seen filtering. One model perT4,private/offline kernel
`carasik/rhpr-kindle-alignment`;config`experiments/kindle_alignment_2t4.json`.

|Layout|Original events|Old pairs|Old singles|Recent singles|Encoder positions inclUSER|
|---|---:|---:|---:|---:|---:|
|32_reference|32|8|0|16|97|
|32_edges|32|7|2|16|101|
|30_first|30|7|1,first old|15|93|
|30_last|30|7|1,last old|15|93|
|34_first|34|8|1,first old|17|105|
|34_last|34|8|1,last old|17|105|
|32_single (Full32)|32|0|16|16|129|

All pair AND single SIDs use3+1codes;separate pair-input namespace. Chronological
events preserved once per chosen window;no target/future event used. Windows
30/34 change both original context and recent-tail size;do not attribute their
effects solely to alignment. Every user has34observed past events,so no cohort
selection/recovered-prefix ambiguity. Native widths,not padding toFull32.

On rolling VAL->TEST,reference->reference shares zero ordered old pairs;
VALreference->TESTedges shares7,->TEST34_first shares8,->TEST30_first shares6.
Report all same-layout and cross-layout contrasts,without choosing a winner
onTEST. Cross-layout agreement changes boundary treatment/tail/positions too;
not isolated proof of cache utility,novel-pair generalization or train-format fix.

Freeze all existing pair keys and full3+1SIDs byte-for-byte. CPU coverage finds
only838missing keys,all inTEST30_first. Encode only these descriptions with the
same frozen Sentence-T5,max_seq_length2048 and exact First/Second book template.
Apply saved normalized-residual GRID centroids without fitting;append suffix
ordinals after max existing suffix per prefix in sorted new-key order. Retain
new keys,embeddings,codes and diagnostics. Never renumber original suffixes.

19parent files and10data files pinned,including both selected weights,
original four prediction files and saved pair centroids. Replay reference
ranked SIDs exactly against saved VAL/TEST predictions before interpreting
variants;record score differences. Check model state unchanged and gradients
disabled. Save28prediction files,per-case/checkpoint hashes,metrics and B1
supported-forward FLOPs under parent CPU math-SDPA definition (FMA2,teacher
forced4code output heads;excludes beam/backward/text encoding).

Interpret as exploratory sensitivity of already inspected frozen models.
Full32 is trained format forMixed,OOD forPair-only. Boundary layouts are new
inference formats forboth;the experiment does not estimate a retrained model's
quality. Report per-user ranks,gains/losses,support/previous recent-popularity
groups and conditional paired-user uncertainty;not independent seed evidence.

Prelaunch GPUquota37.36h. Expected5–15min after environment preparation;server
timeout60min is a hard worker lifetime,not the CI polling timeout. No additional
training/epochs/seeds/variants or automatic duplicate launch.

CPUpreflight:19parent and10input pins verified. Actual selected checkpoints
produce identical ranked SIDs for8saved VAL examples/model;maximum score
difference1.1e-7. All7native-layout forward losses finite,model unchanged.
Supported forward GFLOPs respectively.516501/.539346/.493853/.493853/
.562388/.562388/.704770. See`gr-kindle-alignment-preflight.json`.

Submitted once:private version1,RUNNING,executable sourcea2de37f0d0a3,
workflow43657473213SUCCESS(submission only),job132639118028. Actual worker
timeout3600s verified. All212deployed Python files,exact config,private/offline
metadata and both completed parent sources byte-verified.1124tests pass,
152optional skips. See`gr-kindle-alignment-submission.json`;results pending.
