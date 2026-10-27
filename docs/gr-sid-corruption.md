# Random whole-SID corruption screen

Authorized 2026-10-28: five fresh TIGER fits, 20 epochs, validation at 5/10/15/20,
one neural/order seed 42, one permutation seed 2753, in one private two-T4 session.
Config: `experiments/amazon_beauty_sid_corruption_2t4.json`.

The original byte-pinned Sentence-T5 / RQ-KMeans mapping is the same control as
the preceding screens. Fractions 0%, 5%, 20%, 50%, 100% correspond to exactly
0, 605, 2,420, 6,050, 12,101 moved items (rounding N*fraction). A fixed uniform
random priority defines nested selected item sets. For each selected count,
an independent SeedSequence([2753,count]) samples uniform permutations until
there are no fixed points. All unselected items remain unchanged. Swap complete
four-token paths including the original uniqueness suffix. This is item-to-path
reassignment, not global renaming of code labels. Each dose starts from the
original table; it is not applied cumulatively to an already corrupted table.

Every catalogue prefix, occupancy, valid full path, and collision-group-size
distribution remains identical. TRAIN-weighted token/prefix frequencies are
allowed to change and are explicitly reported, so this does not isolate semantic
content from every distributional property of the autoregressive task.

## Inputs and protocol

Reuse three existing private datasets: `carasik/rhpr-gr-beauty-phase0`,
`carasik/rhpr-sid-context-screen`, and `carasik/rhpr-sid-als-screen`.
No dataset upload, embedding generation, quantizer fit, ALS fit, or graph search.
The third supplies the existing Sentence-T5 positive-cosine top-20 neighbor graph.
Its manifest and catalogue/data identity are verified before use. This diagnostic
measures T5 neighborhood grouping, not independently established true semantics.

The parent derives and saves all maps, owners, selected item IDs, item TRAINtarget
frequencies, and per-dose diagnostics before generator fitting. Full path/owner
file hashes are pinned from CPU preflight and checked at runtime. Each generator
uses its table consistently for histories, labels, and item-level decoding.
All five named-neural-parameter initialization hashes match the previous
`5f4a24fa8c9b5f7063739fd44822fd0a5f98eff91c7e882563d32529b0c5abff`.
Data/order hashes and all hyperparameters match the pure-index20 screen.

GRID 4+4 layers, d128, FF1024, six heads, 4,977,920 parameters, dropout0.1.
Adam1e-3/wd1e-4/clip1, batch256/physical128, FP16 training; FP32 evaluation,
batch32, free beam20. All131413TRAINprefixes once per epoch, latest20events,
2628260examples/model. Validation22363users, full catalogue12101, no seen mask.
Primary endpoint epoch20; retain earlier checkpoints' predictions as dynamics,
not independent replications or grounds to choose a favorable dose/epoch.

Two isolated GPU workers run [0%,20%,100%] and [5%,50%]. Execution is expected
about85–110minutes plus queue. Each model has the unchanged7200second failure
guard, not permission to extend epochs. No test, early stop, retry, extra seeds,
automatic extension, or Telegram. Paused TPU changes remain untouched.

## Measurements

- Save all20validation beam files, training curves, and ten best/last model states.
- At the same four checks, save per-example teacher-forced NLL and true-token
  rank for all four levels, plus aggregate losses and top1rates. True preceding
  SIDtokens are supplied; each head retains its full vocabulary, including
  invalid continuations. Ranks use one plus count of strictly greater logits.
  This adds an evaluation-only pass, restores model mode, and does not update
  weights or sample data. It is separate from free beam-search retrieval quality.
- Record weighted semantic-neighbor prefix sharing and normalized LCP, changed
  prefix counts, event-weighted moved-item mass, and TRAINprefix entropy,
  largest branch mass, and total variation against the original mapping.
- Catalogue occupancy equality is asserted at all four depths, not assumed.

CPU preflight first-prefix semantic-neighbor sharing:
0%:0.53224,5%:0.48065,20%:0.33980,50%:0.13683,100%:0.00624.
TRAINfirst-prefix total variation respectively:
0,0.01690,0.05815,0.10519,0.18072.

A declining curve supports a benefit from the original organization under this
protocol, with changed TRAINtoken frequencies as a possible contributor.
Slower early learning followed by convergence suggests optimization effects.
A flat20epoch curve does not prove semantics universally irrelevant; this is
one permutation realization, one fitted seed, and one dataset/budget.

Implementation: `src/rhpr/gr/sid_corruption.py`, optional preparation in
`generator_suite.py`, and an opt-in teacher-forced sidecar in `experiment.py`.
CPU preflight and launch receipts: `/workspace/artifacts/sid-corruption/`.
