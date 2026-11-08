# ML1M: raw clean windows versus filtering and joining history

CPU-only audit2026-11-05. No GPU, model training, dataset upload, or production
protocol change. This evaluates candidate ML1M preprocessing, not the cause of
observed ML100K downstream metric variation.

Fixed catalogue:2874films under the existing
`tmdb_rich4_and_zenodo_plot_vit` coverage mask. It is the prior common
multimodal source-coverage subset; it does not claim all newly required text
vectors are already computed. Raw ML1M:6040users,1000209ratings;929772ratings
have eligible catalogue films. All rating values retained. Times are rating
submission times, not actual viewing times.

## What filtering can change

Previously proposed ML1M processing removes uncovered item events, then selects
the latest two singleton timestamp groups in that FILTERED history. This gives
5672users and839388strict-past training examples, matching the preceding audit.
It is a next-eligible-item task, whose chronology differs from raw next-item.

Among these5672users:

-450validation targets and407test targets are singletons only AFTER filtering;
 their ORIGINAL timestamp groups contain multiple rated films.799users(14.1%)
 have this issue at one or both endpoints. This is857/11344target cases(7.55%).
-4086raw validation contexts and4173raw test contexts include an uncovered event
 among up to20preceding raw events.4440users(78.3%)have at least one affected
 context. Removing those events can splice observations across gaps and bring
 older eligible events into the model's20item context.
-If any partially touched timestamp group at the left context edge is also
 checked in full,4498users(79.3%)have a context touching an uncovered event.

These are measured structural changes in the candidate ML1M protocol, NOT
proof of a quality penalty or explanation of instability on trained ML100K
models. Filtering is legitimate if next-eligible-item is the intended task;
raw next-item conclusions require spelling out the difference.

## Candidate cohorts and supervised examples

Every NEW policy selects targets that are singleton groups in the ORIGINAL
raw timeline, before item filtering. A timestamp group containing ANY uncovered
film is a barrier; do not use movie-ID order to salvage the covered half of that
second. The whole timestamp group touched at a20item window's left edge must
also be covered. Within fully covered tied context groups, deterministic item
ordering still does not reveal true viewing order.

| Policy | Users / targets per split | Training examples |
|---|---:|---:|
|Previous filter-events-first proposal|5672|839388|
|Initial prefix ending before first uncovered timestamp group|2600|35730|
|Same initial prefix, requiring20past events before val|753|21989|
|Raw clean windows up to20, permitting naturally short ORIGINAL prefixes|4146|174378|
|Raw clean windows, requiring full20before BOTH val/test|3158|166912|
|Literal contiguous20context+val+test block, endpoints adjacent raw events|2786|151166|

The last two rows retain the natural short TRAIN prefixes of the up-to20rule;
only evaluation requires full20. Requiring full20for ALL training cases as
well gives3012users and134695training examples. This is a different, stricter
choice, not used silently in the main comparison.

For context, restricting each user to ONE latest uninterrupted clean run gives
5139users/70980examples with short prefixes allowed; requiring20past events
before val gives3092users/96796examples. Using eligible windows from all earlier
clean regions preserves more supervision than selecting one run.

For raw-window cohorts, reported JSON `train_events` counts RAW rows before val,
including discarded examples/barriers. It is NOT retained supervision. Use
`train_cases` to estimate generator work. Users without at least one admitted
training example before val are excluded in the table:24such users excluded
from the full20evaluation-window candidate and20from literal22blocks. Initial
prefix and previous-filter cohorts require at least2earlier train timestamp
groups. Every row has one val and one later test target for the same users.

## Proposed rule for discussion

Prefer raw clean windows rather than dropping the remainder of an entire user:

1. Retain the ORIGINAL timeline and source-coverage flags. Do not first delete
   unknown item events and then generate windows.
2. For any target, take at most20strictly earlier raw events. Admit an example
   only when target and all touched context timestamp groups are covered. For
   training, admit all targets in a fully covered same-time group using the
   same strict-past context; do not introduce within-group target order.
3. A short context is allowed only at the natural start of the original user's
   history when its entire available past is covered. Do not manufacture a
   short prefix immediately after an uncovered event. Once that event falls
   outside the20event window (including touched boundary groups), later valid
   examples become usable again.
4. For evaluation, require raw singleton target groups. Choose the latest two
   eligible endpoints deterministically, independent of model outputs. The
   up-to20rule retains4146users; an explicitly stricter full20rule retains3158.
5. Train only on eligible examples STRICTLY before chosen val, with at least
   one admitted training example per user. Test uses its actual raw past,
   including preceding val/intervening observations, and its own clean20window.
   Val and test need not be adjacent or lie in the same uninterrupted run;
   literal contiguous22blocks are the separate2786user alternative.
6. Fix the same catalogue, cohort, target timestamps and example schedule for
   every model. Newly selected earlier heldouts require fresh models; reuse
   of weights trained on those events would contaminate evaluation.

This avoids invented adjacency without throwing away all later events after
an early missing film. Its costs are fewer examples, endpoint backtracking and
selection towards well-covered histories/films. It does not remove natural
next-item uncertainty, sparse single-target evaluation, seed variability, or
unknown ordering of ratings entered in the same second. It is a concrete
candidate protocol, not an approved full-training run or universal solution.

## Reproducibility

`scripts/research/audit_ml1m_clean_histories.py` takes `--movielens`, `--masks`,
`--output`. It independently reconstructs raw timestamp groups, verifies the
old5672/852295/839388counts, and writes policy-specific per-user CSVs plus the
raw-target/context audit. CLI replay reproduced every policy/audit count.
`gr-ml1m-clean-history-audit.json` pins source/mask SHA256 and detailed counts.

Local evidence `/tmp/ml1m-clean-history-evidence.zip`: raw MovieLens ZIP,
coverage-mask Parquet, per-user CSVs, JSON summaries, replay script and report.
No model calls, runtime source modifications or new GPU experiments.
