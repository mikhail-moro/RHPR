# Finite GRID augmentation epochs and TPU compatibility

The first quality run completed on T4:11,600 updates,5,939,200 examples,
test Recall@10 .04073693 / NDCG@10 .02138411. It stopped by early stopping.
Validation took73.52min versus102.39min training. Source revision:
`fb195577861331b90981bf76500ccfbd1bb0c370`.

## Author augmentation

[GRID §4.2, Table6](https://arxiv.org/html/2507.22224v1#S4.SS2) studies sliding
windows: every contiguous subsequence of at least two items can become a
context/next-item example. Beauty's ablation gives Recall@10 .0597 versus .0447
without augmentation, and NDCG@10 .0328 versus .0226. These ablation rows use
the section's default tokenizer; they are not Table1's RK-Means target.

The completed run already called the unchanged author
`collate_with_sid_causal_duplicate`. Its released implementation enumerates
possible windows in each source-user batch, samples at most512 indices, then
truncates selected windows to120 SID tokens including the four target slots.
Sampling is with replacement, but membership-based enumeration emits duplicate
selected indices once; the adapter deliberately preserves that source behavior.
The full Beauty training rows contain1,197,012 possible windows before
truncation, but the cap means one source-user pass does not enumerate all of them.

## Epoch contract

Configuration:
[`amazon_beauty_grid_epochs_fp16_b512.json`](../experiments/amazon_beauty_grid_epochs_fp16_b512.json).

- One epoch visits each of22,363 source users exactly once, in a seeded shuffle.
- Use the same author window sampling; resample with `seed+epoch` each epoch.
- Flush the final smaller source batch and model batch; discard no examples
  and carry no windows into the next epoch. Report actual example/update counts.
- Keep FP16, effective batch up to512, source batches32 and the existing model/LR.
- Validate at epochs10,20,30…; make a labelled final validation if the time or
  epoch budget ends between these boundaries. There is no extra epoch1 check.
- Stop after20 complete epochs without a better validation NDCG@10, evaluated
  only at validation boundaries. Select the checkpoint by validation Recall@10.
- Save the last model/optimizer/scaler and RNG state at every completed epoch.
  Automatic resume is not implemented by this change.

This is a finite source-user epoch definition, not a fixed number of unique
training pairs. It also differs from both the old full-prefix epochs and the
paper's unbounded training stream. Last partial model batches have their actual
size. Changing validation frequency and defining boundaries changes the training
and selection protocol; no exact optimization-trajectory equivalence is claimed.

The registered next run initializes the generator afresh while reusing the
completed tokenizer. It mounts the private parent output and pins the source
revision and artifact hashes. Catalogue order, source data hashes, representation
and quantizer settings, SID shape and collision suffix are checked before reuse.
The Flan/RK stages are recorded as reused, with their original costs separately
retained. They are not rerun or charged as newly measured encoding/fitting time.

## TPU check scope

[`gr_grid_tpu_compatibility.json`](../experiments/gr_grid_tpu_compatibility.json)
is a separate bounded diagnostic. It requests actual TPU hardware, retains the
native torch/torch_xla pair and uses only compatible verified dependency wheels
offline. CPU/XLA execution cannot count as a passing TPU result.

Checks: native matrix calculation, actual pinned GRID FP32 loss against CPU,
three BF16 Adam updates with changed weights, then FP32 beam search. Results and
the exact failure stage go to `tpu-compatibility.json`. The diagnostic reports
no recommendation quality and does not train on validation/test targets.

Passing those checks establishes generator compatibility on one TPU device.
It does not establish that the current full GPU pipeline can run unchanged:
Flan/RK preparation, epoch-runner XLA optimizer/synchronization, fixed-shape
padding for final batches, checkpoint handling and multi-device scaling still
need explicit support. The production GPU runner continues to use CUDA AMP;
the TPU probe uses XLA BF16 without a CUDA GradScaler.

Submitted diagnostic: private kernel `carasik/rhpr-gr-grid-tpu-probe`,
[workflow39951344759](https://github.com/mikhail-moro/RHPR/actions/runs/39951344759),
source `352e7f95644b3aa02aca99a151706eac3a2511ba`, server timeout20min.
At2026-10-23 12:14UTC Kaggle CLI reported QUEUED; the MCP status response was
empty. The TPU quota had no reservation and no hardware test had executed yet.
Submission success is not a compatibility result. No GPU epoch run was
submitted: only2,374.585s (~39.6min) GPU quota remained, versus roughly50min to
the first scheduled ten-epoch validation at the preceding run's throughput.
