# ML1M field representations: learning curves of frozen SID indexes

User requested a test of whether the apparent advantage of a representation
changes with generator training time. This is an exploratory validation-only
study. A curve gap closing is consistent with a difference in learning speed;
it does not establish asymptotic equivalence or causal influence of a feature.

## Matrix and controls

Frozen Sentence-T5 input for the same 2,874 ML1M items:

* `title`: original title INCLUDING release year.
* `genres`: title plus observed ML1M genre set.
* `duration`: title plus observed TMDB duration.
* `shuffled_duration`: title plus an item-level random permutation of observed
  durations; the same shuffled map is reused across all quantizer seeds. The
  five `unknown` values stay with their original items. This preserves marginal
  duration counts, not joint distributions with genre, popularity or year;
  it is a useful control, not a perfect causal intervention.

No data upload is needed: the byte-pinned `metadata-texts.json` in
`carasik/rhpr-ml1m-clean-mtl-input` provides the fields. The source text parser
was compared against all 2,874 independently prepared source records. The
permutation has seed20261109, maps 2863 item positions to a different item and
changes the duration value for 2823 items. Texts, owner array, and hashes are
saved. Same Sentence-T5 max2048 and no truncation; observed maximum 43 tokens.

Each field gets three fresh native GRID RQ-VAEs, 3×256, 3000 updates, seeds
42/101/202, and a uniqueness suffix. `generator` seed31415 is used on all 12
indices; for the fixed q42 `title` and `genres` indices, seeds27182 and161803
are also trained: 16 generators, 12 frozen indices, 2 isolated T4 workers.
Token seed and model seed are separate; the small crossed subset does not
estimate their interaction across every index. Same-seed pin checks compare
fresh model initial parameters across all indices, and epoch permutations are
seed-controlled; GPU operations need not be bitwise deterministic.

## Common training and evaluation

* Existing raw-contiguous clean ML1M cases: 151,166 training windows, 2,786
  validation users, context≤20, identical hashes and catalogue for all fits.
  Validation is one target/user, ratings timestamps are not view timestamps.
* Narrow mini-TIGER: GRID T5 4+4, d_model64, d_ff512, d_kv32, six heads,
  1,375,040 parameters, 2000 user bins, FP16 and native HF 5.22.1 SDPA,
  effective/physical batch512; FP32 free beam20 validation; no seen filter.
* Fresh models, same Adam LR0.001 and weight decay0.0001, no LR scheduler;
  validation every10 epochs, min30, max100, stop after30 epochs without a new
  strict NDCG@10 best. A stop is operational, not proof of convergence. The
  epoch cap is reported as censored. The trainer has a 5100-second per-fit
  epoch-boundary limit; a time-budget exit fails the fit with checkpoint intact.
* Save each epoch's training curve and best/last resumable checkpoints, and
  every validation's top20 predictions, NDCG/Recall and token-level teacher-
  forced cross entropy. These values support fixed-epoch comparisons when
  both fits reached that epoch and best-checkpoint comparisons separately.
* No TEST inference or selection. Earlier ML1M val/test were already studied;
  neither should be represented as an untouched confirmation set.

First local preflight verified catalogue and train/val digests, all three
narrow generator initialization pins using two different code assignments,
all source fields/metadata, permutation distribution, CPU Sentence-T5 output
for each field, and native small RQ-VAE and real narrow SDPA training smoke.
The actual long experiment and GPU integration still require the Kaggle run.
