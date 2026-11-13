# Kindle Store: dense 100K-event pair pilot

The user requested Amazon Reviews2023 Kindle Store, ratings >=4, title+categories,
Sentence-T5 -> normalized residual RQ-KMeans3x256 plus collision suffix, ordinary
leave-last-two-out, approximately100K interactions sampled by complete users.

The initial uniform user sample had100015 events,5560users,69308items. Titles and
categories were present for every selected item, but52.2% of validation and55.2%
of test targets were absent from all generator training contexts/targets. Only823
users had20past items before validation. No GPU was launched for that sample.
The user explicitly selected a denser sampling rule rather than retaining this
cold-start-dominated pilot.

The revised, fixed rule selects the5000 most frequent books among positive
interactions in the official5core rating-only file (frequency descending, ASIN
ascending for ties). Metadata are joined by parent_asin. Keep users with>=22
positive records in this catalogue, shuffle their IDs with NumPy seed42, and take
the whole-user prefix closest to100K events. Every selected user keeps their
complete positive history inside this restricted catalogue. No outcome from a
trained model enters selection. Catalogue selection uses the full static dataset
counts, including eventual heldouts: this is dataset construction, not a
historical/production catalogue-availability or unbiased population evaluation.
The result represents active readers of popular books and next-item prediction
inside that catalogue; it does not describe all Kindle customers or all books.

Sort retained histories by timestamp then parent_asin; last target=test,
penultimate=validation, all earlier noninitial prefix positions=train. Contexts
contain up to20 observed items, with natural short training prefixes padded.
Test includes the observed validation event. Preserve heldout targets even when
they never appeared in training, and record this incidence separately. Short
history users are excluded before splitting, not based on any model result.

Compare Recent20, Recent15 and Pair20. Pair combines adjacent non-overlapping
pairs in the older floor(history_length/2) events, leaves an unpaired old event
and the newer half as single items. Item and pair SIDs both have3semantic codes
plus one unique collision ordinal. Pair token embeddings have a separate input
table, installed in all controls to preserve within-seed parameter initialization.
Pair RQ fits on unique TRAIN pair embeddings only; validation/test-only pairs are
encoded by frozen centroids. Collision ordinals are allocated train, then val,
then test without changing previous assignments. Item content uses the entire
fixed static catalogue. Pair descriptions use `First book: ... Second book: ...`.

Use paired pipeline seeds42/101/202; same item index, initial model weights,
supervised targets and epoch shuffle per seed. Mini-TIGER4+4/d64/FF512,
FP16/nativeSDPA/batch512, Adam1e-3/wd1e-4/clip1, beam20, no seen-item filter,
validation every5epochs, min50/max120/patience30epochs, best val NDCG@10
selection. Test is evaluated only after selection. Best and full resumable last
checkpoints, per-validation predictions, item/pair embeddings and frozen indexes
are retained. Tokenizer and generator seeds are coupled, so this is not a
variance decomposition. Epoch limits are censoring, not demonstrated convergence.

The training implementation is the existing pair runner with catalogue size
read from the pinned configuration, preserving the ML100K numerical protocol.
Prepared files and provenance remain private Kaggle inputs, outside Git.

Prepared sample: 2625 users, 4969 books, 100008 events, 92133 supervised training
prefixes and 2625 cases per heldout split. Median retained history31, range22–275.
All selected books have title/categories; validation/test timestamp ties0.
Unseen-in-training targets: validation8 (0.305%), test19 (0.724%).
Source URLs/digests are in the private input's source-provenance.json; every
prepared file is SHA256-pinned in experiments/kindle_dense100k_pair_2t4.json.

Preflight passed on the actual prepared arrays: all heldout contexts have20
items, IDs/ranges/counts are valid, and the book pair template is verified.
54310 unique TRAIN pairs, 60128 pairs across all splits; heldout pairs never
enter centroid fitting. Full local regression:1078 tests,152 optional skips,
zero failures. Private input status ready before submission.

Submitted once as private kernel carasik/rhpr-kindle-dense100k-pair/version1,
RUNNING at the server check. GitHub Actions43481734588 submitted executable
8b575bbe21ea3e477cf78539fad92f9488c08719 successfully. Pulled deployed source
was independently checked against every local src/rhpr Python file and the
exact configuration, plus private/offline/T4/input metadata. See the adjacent
gr-kindle-dense-pair-submission.json receipt. Submission success is not
training completion; collect outputs and verify all nine fits before analysis.
