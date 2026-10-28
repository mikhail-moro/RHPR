# Context-ALS SID placement: matched20epoch TIGER screen

User authorized2026-10-27:several variants,20complete epochs,validation every5.
Exactly four fresh fits in one private2×T4session:

| Trial | Frozen item-to-SID mapping | Worker/order |
|---|---|---|
| `rqkmeans_original` | Exact original T5RQ-KMeans paths | GPU0first |
| `item_als` | Exact prior ALSuniform256-swap mapping | GPU1first |
| `context_als_20` | TRAIN preceding-context ALS means,window20 | GPU0second |
| `context_als_5` | Same,window5 | GPU1second |

This incorporates the requested ALS20control rather than reconstructing it in
an isolated GPU session. Original20is also refitted so every candidate has a
matched control with retained20epoch weights for future context-length replay.
No original/legacy co-occurrence or other RQ-VAE variants are added.

## Frozen inputs

See `docs/gr-sid-context-signal.md` for prototype aggregation,short/rare handling,
positive cosine neighbor policy,confidence weighting and completed CPU evidence.
Use the audited cosine-neighbor candidates,not the earlier weighted-cosine
ranking sensitivity. All swapped indices have256accepted moves;original0.
Changed items are480itemALS,482context20,481context5. Semantic distortion and
changed sets are not equal. Context confidence policy is another difference from
the old ALS graph,so this is not a causal isolation of representation alone.

`scripts/package_sid_context_indices.py` checks completed CPU file hashes and
the prior ALSresource manifest,then checks exact complete paths/owner bijections.
It exports15small numerical/provenance files,about2MB,without reviews,raw text,
names or credentials. Private input:`carasik/rhpr-sid-context-screen`.
Manifest SHA256:
`8878eb9f8b3f5f3174a763e501d8387f1c85b298651e6485b6638b9c87782209`.
Original and old ALSarrays preserve the previous input bytes. No upstream
encoder,ALS,graph,search or quantizer refit runs in the GPU notebook.

## Training and evaluation

`experiments/amazon_beauty_sid_context_screen_2t4.json` reuses the existing
frozen-SID generator runner and runtime. Sharedseed42,GRID4encoder+4decoder,
d128/FF1024/dkv64/6heads/dropout0.1/userbins2000/no separator.4977920neural
parameters. Same pinnedBeauty catalogue12101items/22363users and TRAIN split.
All131413noninitial TRAINpositions once per epoch,latest20context events,
identical seed-plus-epoch permutations and neural initialization across trials.
Expected training-case hash and initialization hash remain the previous controls'.

FP16training,Adam0.001/weightdecay0.0001/clip1,batch256/physical128. Exactly
20epochs and2628260consumed examples per fit,10280planned optimizer batches
with any AMP skips reported separately. No early stopping before the endpoint.
Maximum7200seconds per trial is a failure guard,not an intended shorter budget.
One isolated process per GPU,one CPUthread per process,two fits sequentially.
An insufficient device count fails;there is no single-GPU/TPU/precision fallback.

Full22363-user validation at5/10/15/20only. Same FP32batch32/freebeam20,static
full catalogue,no seen-item masking. Primary comparison:itemNDCG@10at20;
also report Recall@5/10,NDCG@5 and full fixed-epoch trajectories. Best-validation
selection is saved separately and must not replace the fixed20epoch endpoint.
No test example construction or evaluation,extra tuning,seeds,automatic retries,
extensions or Telegram notifications.

Retain all16per-user validation prediction files,full curves,best and last
checkpoints for each trial. Last20includes model,optimizer,AMP scaler,RNG state,
SIDhash and exact epoch/update counters. This avoids the previous missing20epoch
weights. Intermediate5/10/15weights are not separately archived;their predictions
are. No heavy SID-study telemetry is added.

## Checks and submission

Two new tests cover the four-fit/cadence/protocol contract and real packaged
index import,including complete suffix ownership and every prefix occupancy.
Both passed against the actual new inputs. The existing five ALSscreen tests
also passed with the native pinned GRIDruntime,including actual tiny training
fits showing validation cadence/prediction capture preserve model,optimizer
and RNG. Local kernel bundle metadata verified private/offline2T4and two input
datasets. Package inputs are hash-checked again inside the GPU run.

Before submission quota showed12.26GPUhours remaining and the previous replay
was COMPLETE. Previous60epoch fits used~4790–4801optimizer-training seconds
per model plus~49seconds per validation. For two waves of20epochs expect about
60–80minutes execution including validation/startup;queue time is additional.

Kernel:`carasik/rhpr-context-als-tiger-screen`. Submit once through the existing
GitHubActions Kaggle workflow,with Telegram disabled and submit-only mode;do
not poll training until completion or user requests a status update.

Submitted once:GitHubActions run
`https://github.com/mikhail-moro/RHPR/actions/runs/40541334398` completed
successfully in submit-only mode. Executable source commit
`9dcee38f1fe77cff18daf093246c7c81e9dc0751` is confirmed in Kaggle's returned
source blob. Kaggle version1metadata confirms private,GPUenabled,T4shape,
internetdisabled,TPUdisabled and exactly the two required dataset mounts.
Submission status was QUEUED;no training results are available yet. Read-only
receipt:`/workspace/artifacts/sid-context-screen-submission.json`. Do not
resubmit merely because this session waits in the GPUqueue.
