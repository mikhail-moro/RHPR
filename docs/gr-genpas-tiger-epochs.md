# TIGER baseline using GenPAS source, with finite epochs

The active baseline uses the actual GenPAS TIGER model, **without GenPAS weighted
sampling**. Configuration:
[`amazon_beauty_genpas_tiger_epochs.json`](../experiments/amazon_beauty_genpas_tiger_epochs.json).
This is a separate runner; historical GRID experiments and their artifacts remain
unchanged.

## Sampling and epoch definition

The adapter calls the released `collate_functions.collate_with_sid_causal_duplicate`,
the TIGER/GRID contiguous-window baseline. It never calls
`general_sequence_sampling`, and does not use alpha/beta/gamma, mixup or other
GenPAS augmentation. A window provides a context and the next complete item SID.
The released window sampler itself is sometimes called augmentation; here
“ordinary” refers to this baseline rather than the GenPAS method.

Each epoch visits every source training user once, in a shuffle determined by
seed42 and the epoch number. Source groups contain up to512 users. The author
collator selects up to512 windows per group; repeated sampled indices are emitted
once, preserving its implementation. Both the final smaller source group and
final accumulation group are retained. A source-user epoch does **not** enumerate
all possible targets/windows, and its generated example count can vary.

The window RNG is isolated from dropout and validation. Dataloaders are rebuilt
each epoch; their seed is assigned before Lightning can prefetch the first batch.
The initial implementation uses zero loader workers and one accelerator device.
It caches the small Beauty rows after the author's preprocessing to avoid
repeated TFRecord parsing. Epochs preserve the original training split; validation
and test are not resampled or regenerated.

## Model, optimization and selection

Pin: [GenPAS7803b541](https://github.com/snap-research/GenPAS/tree/7803b541f646237084ac01692accc8e97053491b).
Both Python and YAML trees are verified before import. The copied source differs
only in deferring an annotation-only TensorFlow iterator import, and includes
Snap's `License+Attribution.txt`. The author model, loss, optimizer, scheduler,
metrics and constrained beam generation are used directly, not reconstructed.

The Beauty model preset remains4+4 layers, width128, FF1024,6heads, d_kv64,
dropout.15, two hidden FF layers, no user token/separator and beam10. Its input
budget is100 SID positions including four target slots, leaving at most24 history
items. It expects the author's **Flan-T5-XXL RK-Means3×256 SIDs with collision
suffix**, not our older Flan-T5-XL GRID tokenizer. No encoder or quantizer is trained
in this runner.

Native config behavior is retained where the paper differs: Adam (not AdamW),
nominal LR.001, weight decay.0001, and `WarmupLinearScheduler` registered with
`interval="epoch"`. Initial effective LR is.0003. Making epochs finite now advances
that scheduler at each epoch end; it previously stayed at its initial value in
the infinite stream. Its1000-unit warmup and320000-unit schedule are retained
explicitly; this is not a silent correction to update-based scheduling.

The next GPU config uses FP16 rather than source BF16, source batch512 and
accumulation16 on one device, aiming at the source's nominal8192 examples per
full update across four devices. Actual window counts can be smaller. Lightning
flushes the final partial accumulation group using its native loss normalization;
exact effective batches and update trajectories are not claimed equivalent.
Validation/test disable autocast inside the unchanged author evaluation step:
its beam search multiplies four probabilities, which can underflow in FP16.
Training remains mixed FP16; evaluation uses the retained FP32 weights.

- Training is limited by1000 source-user epochs, with no optimizer-step cap.
- Validate only after epochs10,20,30…; no sanity validation or epoch1 check.
- Select and save the best checkpoint by validation Recall@5 on those same boundaries.
- Early stopping patience is10 validation checks, i.e.100 epochs without improvement.
- Require the epoch cap to be a multiple of10; do not test unvalidated final weights.
- Evaluate test once, after restoring the validation-selected checkpoint.

There is no automatic resume or multi-device/TPU support in this adapter yet.
The former GRID TPU diagnostic does not establish GenPAS compatibility.

## Execution and evidence

`kaggle/run.py` remains the experiment entry point (`kind=gr_genpas_tiger_epochs`).
The committed input paths describe the required staging layout; a Kaggle resource
dataset has **not** been published for this new baseline. Stage the pinned source
and the unmodified author data from the
[repository README](https://github.com/snap-research/GenPAS#-datasets), preserving
the `training/`, `evaluation/`, `testing/`, and `sids/` directories. Local development
requires the GR dependencies plus `pip install -e '.[genpas-data]'` for TFRecord
reading. An offline Kaggle wheel manifest still needs preparation; the old GRID
resource bundle must not be treated as this baseline's complete runtime.

The runner records source revision, input hashes, resolved model config, actual
source-user/example/update counts, learning rates, validation epochs and selected
checkpoint. Full-split TFRecord reads use the author per-row iterator, avoiding
its batched iterator's hidden `drop_remainder=True`. Unexpected filtered or
duplicate user rows fail rather than silently changing epoch coverage.

Verification uses the actual pinned GenPAS class, collator and optimizer on a
tiny CPU fixture:20 epochs, validation at10 and20, final partial batches,
checkpoint restoration and one test pass. Run it in isolation from GRID's `src`
namespace:

```bash
RHPR_GENPAS_SOURCE=/path/to/GenPAS PYTHONPATH=src \
  python -m unittest discover -s tests -p test_gr_genpas_epochs.py -v
```

This verifies scheduling/component integration, not Beauty quality, full author
archive ingestion, GPU throughput or TPU compatibility. Changing sampling,
example exposure, scheduler progression and checkpoint cadence can affect metrics;
this is a declared baseline protocol, not an exact published reproduction.
