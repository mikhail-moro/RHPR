# Context40 replication: seeds101and202

Authorized2026-11-03: user requested two more seeds after the seed42context
pilot's contradictory validation/test results. Repeat only the40item setting:
no-people control and actor graph.1 xseeds101/202 =4fresh predictors.

Exactly the seed42pilot's30epoch budget and training/evaluation policy:
TIGER4+4/d128/FF1024, original eagerT5attention, FP16AMP+GradScaler,
effective256/physical128, same data seed and85470targets, same943heldout users,
unfiltered full-catalogue beam20FP32validation/test. First20epochs val5/10/15/20,
verified optimizer/scaler/RNGresume, then val every epoch21–30and original
ReduceLRschedule from21. Test once using the best validation checkpoint.
The30epoch cap is not proof of convergence. No HPO, additional windows or seeds.

Freeze the original SIDindex for EACH corresponding pipeline seed from
`carasik/rhpr-ml100k-actor-full-training v1`; no AE/RQor graph retraining.
Predictors start from their original constructor seeds101/202. Thus this
replicates pipeline variability across fixed historical seed-specific indices
and generator initializations; it does not isolate generator-only variance
on one common index. All parent artifacts are SHA256-pinned.

Keep completed seed42results unchanged. Aggregate three seeds only after all
new fits complete, including unfavorable outcomes; compare validation and test
separately. Original20window validation is available under the same first30epoch
budget. Historical20test can use a differently selected checkpoint; inspect
its budget-match flag per seed, not assume all are comparable.

Config: experiments/movielens100k_context40_seeds101_202.json.
Private NEWkernel: carasik/rhpr-ml100k-context40-replication.
Two T4workers, each processes two models. Before submission quota6.42GPU-hours;
measured prior40item epoch93–98seconds suggests roughly1.5–2hours elapsed.
No FlashAttention, seen mask, wider beam, batch/precision/hardware fallback,
automatic retries or extensions. Collect this run after user reports completion.

These seeds reuse the same previously examined validation/test users and targets.
They check training/pipeline robustness, not a new independent holdout; window40
was selected after exploratory results and all outcomes must be retained.

Validation before submission:956repository tests passed(91skipped). Four real
constructor/backward checks matched original101/202initializers, preserved
weights and had finite gradients. No GPU trial was launched by preflight.
