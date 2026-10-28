# TRAIN-only preceding-context ALS signal for SID swaps

Implemented2026-10-27 following the user's proposal. No GPU jobs were submitted.
The next generator experiment should include the missing ALS20 control rather
than spending a separate GPU session reconstructing it now.

## Definition

Keep the existing frozen64-dimensional ALS item factors `z_i`. For every
noninitial TRAIN event with target `i`, construct its preceding context
`H = train[max(0,t-window):t]`. First take the arithmetic mean of the available
ALS vectors in H; then take the arithmetic mean of these context vectors over
all TRAIN positions targeting i. These are two different averages: longer
contexts do not receive more outer weight. Raw ALS factor norms are retained;
cosine normalization happens when comparing the resulting prototypes. Store
the original item factors by their pinned reference, and the new context
vectors as a separate item-aligned array. No concatenation with the target's
own embedding or direct-item ALS fallback is used.

Window20 is primary because TIGER observes at most20events. Window5 is a
prespecified CPU sensitivity, not selected by recommender validation. Within
short H, average the available events, with no PAD in the denominator. The
current target and future events never enter H. Earlier repeated occurrences
of the same item remain legitimate context. Items with fewer than2ALS-fit
users have their factor excluded from event pooling. Empty-factor contexts
are skipped; partially represented contexts retain the valid events. These
choices and coverage are logged.

Fit prototypes and graph on the original16772TRAIN-user partition only. The
5591disjoint TRAIN users supply independent context observations for diagnostics,
using the same fixed ALS encoder. No recommender val/test target is used to fit,
evaluate or choose these CPU candidates. Source hashes, user partition, binary
ALS input CSR hash, factor-array hash and catalogue/data identity are checked.
The upstream factors are reused, not refitted.

For each item record usable target-context count, distinct context-user count,
sum of actual H lengths and number of embedded events. Let `u_i` be distinct
context-user support. Fewer than2users gives zero CF reliability; otherwise
`r_i = u_i/(u_i+10)`. Select top20positive-cosine neighbors among eligible
endpoints, then set `w_ij = r_i*r_j*max(cos(mu_i,mu_j),0)` on those edges.
Support controls confidence rather than asserting that a short context is
intrinsically uninformative. Repeated epochs do not add evidence.

An item without CF edges is not frozen as a path donor: the semantic objective
can still justify exchanging its path with a supported item. This is distinct
from a hard no-movement rule for rare items, which was not implemented.

Use the unchanged exact global `context_LCP +0.3*semantic_LCP` swap objective,
normalized separately by each graph's edge mass. Uniform proposals/seed2702,
maximum100000checks, cap256accepted moves. Freeze the original T5semantic
graph. Exchange complete RQ-KMeans paths including their uniqueness suffix;
verify bijection, all catalogue prefix occupancies and full path preservation.
The suffix is excluded from the normalized3-level LCP objective as before.

## CPU evidence and graph-policy audit

Artifacts: `/workspace/artifacts/sid-context-signal/`. The initial probe ranked
neighbors by confidence-weighted cosine; it is retained as the parent directory,
not discarded. This put41.46%of incoming graph weight on the top100items,
versus4.32%for the original direct-item ALS graph. Its window20candidate had
249accepted swaps/454changed items, window5had254/472, with heldout count-model
gain0.444268/0.441509bits.

The corrected default ranks neighbors by cosine, then applies reliability to
edge weights. Results live in `cosine-neighbors/`. Window20top100concentration
is11.91%, effective weighted targets2405rather than365in the first probe
(direct ALS7418). Thus popularity concentration remains higher than the old
graph; it is reduced, not eliminated. Both policies are explicit in the API.
This change was driven by TRAIN graph diagnostics, without generator outcomes.

Final default CPU preparation took65.9seconds; first probe80.7seconds. Both
final candidates reached256accepted swaps:20window74912proposals/482changed
items;5window77879/481. The original direct ALS control changes480items.

| Candidate | Heldout TRAIN gain, bits | Heldout context LCP, window20 graph | Semantic graph LCP | Price R² |
|---|---:|---:|---:|---:|
| Original RQ-KMeans |0.411252|0.031825|0.199945|0.277913|
| Existing direct-item ALS swaps |0.413672|0.032013|0.196909|0.276802|
| Context swaps, window20 |0.447136|0.034451|0.193862|0.275638|

Window5context swaps give heldout gain0.444777bits; its own heldout context
graph score changes0.032819to0.036009. Scores of different graphs are not a
common absolute scale. The gain is the existing smoothed last-item count
predictor's improvement over its own first-SID unigram on heldout TRAIN users,
not TIGER, NDCG, an InfoNCE objective or a calibrated quality proxy. For20window,
its conditional NLL changes7.041920to6.980055bits, while unigram NLL changes
7.453173to7.427191bits. These are diagnostic first-level predictions.

98339of98398fit target contexts are usable.34.05%contain1–2events; mean actual
length6.69for window20.219items have no usable fit context,875have contexts from
one user;11007items can contribute CF edges. Fit-vs-heldout same-item prototype
cosine rises with support:0.254for2–4fit users,0.4907for10–19,0.7674for20+.
Different-item cosine within the same fit-support bins is0.1381/0.2257/0.3021;
the excess is0.1159/0.2650/0.4653. This removes a common-direction baseline,
but not dependence on the shared frozen encoder. Low-support prototypes remain
noisy, and missing heldout support is explicitly reported rather than imputed.

No inference of better downstream ranking follows yet. A single mean loses
order and mixes multiple item usage modes. Reliability changes relative to
the old graph are another difference besides context representation. Semantic
LCP falls3.04%, versus1.52%under the old ALS swaps. Accepted edit counts match,
but semantic distortion and changed item sets do not. A clean future ablation
can apply the same confidence policy to direct ALS; no extra fits are launched.

## Reproduction and retained data

Run with the project native Python environment and `PYTHONPATH=src`:

```bash
python scripts/prepare_sid_context_screen.py --config experiments/amazon_beauty_sid_context_prepare_cpu.json
python scripts/report_sid_context_signal.py --directory /workspace/artifacts/sid-context-signal/cosine-neighbors --plot
```

Output directories must be new. The optional `reuse_primary_vectors` is a
disk-saving hardlink optimization: the freshly computed vectors must match
exactly; if that path is absent they are saved normally. The prior factor files
and their resource manifest remain required and hash-checked. No implicit fit
or GPU dependency is needed for this preparation.

Saved: original factor/source references, config/provenance, primary context
vectors and weighted graph, per-item support/length arrays for both windows,
complete candidate paths, owner permutations, accepted swaps, search reports,
all snapshot proxy curves, independent TRAIN support diagnostics, file hashes,
README and four-panel `curves.png`. The reporting step adds
`reliability-audit.json`; the preparation's `files.json` covers preparation
outputs only. Original weighted-ranking results are retained separately.

Verification:8context-specific tests,5existing exact-swap tests and4ALS screen
tests passed. One optional native generator test was skipped because its pinned
GRID runtime was not supplied; generator code did not change. The new tests
cover causal exclusion, short windows, equal context weights, repeated targets,
missing factors, distinct-user support, both neighbor policies, factor
immutability, exact objective and complete-path ownership. No GPU quota used.
