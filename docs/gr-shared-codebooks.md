# RQ codebook initialization and live sharing in TIGER

Authorized2026-11-06: two30epochfits,onegeneratorseed42,VALevery5,2T4.
Reuse clean ML1M151166TRAINwindows,2786VAL/TESTusers,2874catalogueitems.
The phrase150K dataset means the existing clean raw-contiguous20+VAL+TEST
cases,not a new corpus. SameSUMseed42 metadata/poster/plot RQ-VAE checkpoint
from carasik/rhpr-ml1m-clean-mtl;no new quantizer or feature pretraining.

Arm1 codebook_init: initialize the first3x256TIGER embedding rows from saved
RQ-VAE centroids,then train TIGER normally with original fixed SID assignments.
Arm2 live_codebook: exactly the same initialTIGER,optimizer and train permutations;
the first768embedding rows themselves are the live RQ centroids. Frozen catalogue
latents are computed once from the frozen final RQ encoder,with frozen catalogue
normalization. Reassign all catalogue items by residual squared Euclidean distance
at the START of each epoch,then rebuild deterministic catalogue-order suffixes.
No backpropagation through hard argmin. NIP gradients reach codebook vectors by
normal TIGER embedding lookup; this is alternating reassignment,not a claim of
end-to-end differentiable index learning. No reconstruction or quantization loss.

OriginalRQ books/latents64D,TIGER128D. Use a fixed zero-padding isometric lift
for both initial books and frozenlatents,so original distances/assignments match.
After training all128coordinates of shared rows can move; do not claim this is
unchanged64D RQ geometry. User/BOS/suffix embeddings and ordinary GRID output
classification heads remain as usual. GRID shares encoder/decoder input lookup,
but uses its usual separate per-level output heads. No output-head tying added.
This experiment tests semantic initialization and live reassignment; it does not
assume RQ reconstruction vectors and NIP embeddings encode identical information.

Cache epoch-specific SID arrays and index churn/occupancy/collisions; evaluate
VAL with the same snapshot used in that epoch. Store the active snapshot as a
persistent model buffer. Restore the best-by-VAL checkpoint AND matching index
before finalTEST; never decode an old generator with a newer index. Thirtyepochs,
8880updates,sixVALchecks,noES/scheduler,FP16training/FP32eval,nativeSDPA,512batch,
4+4layers,d128,FF1024,H6,kv64,AdamLR.001,weightdecay.0001,clip1. No extra trials,
retries,automaticprecision/LRfallback,Telegram or hardware fallback. Fail explicitly
on suffix overflow/nonfinite loss or incomplete budget. Original random-initSUMseed42
fullTIGER metrics already exist as a historical control; do not retrain a thirdfit.
Onepipeline/generatorseed is exploratory,not evidence of a robust improvement.

Private carasik/rhpr-ml1m-shared-codebooks,offline,three existingdatasets plus one
existing notebook parent. Servercap60minutes,eachfit timebudget3000s. Quota checked
1.18h before submission;expected35–45min. Config ml1m_shared_codebooks_2t4.json.
Save weights,matching best and epochSID snapshots,TRAIN/VALcurves and finalTEST
predictions. No new research-data upload. Native CPU probes are correctness checks,
not hardware or quality estimates.
