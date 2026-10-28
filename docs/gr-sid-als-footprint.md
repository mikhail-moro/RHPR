# What the ALS SID swaps actually change

CPU-only audit of the frozen `t5_rqkmeans_uniform` mapping:480of12101items
(3.97%),256accepted transpositions. This is the index used in the60epoch study
and the conditional-index experiment. Other variants are summarized separately;
similar moved-item percentages do not imply similar traffic exposure.

## Histories, targets, and training examples

Current Beauty protocol:22363users,one validation target/user,latest20TRAINitems
visible. Training has131413distinct noninitial prefix-target positions per epoch.

| Exposure measure | Count | Percentage |
|---|---:|---:|
| Users with a moved item anywhere in full TRAIN history |4441/22363|19.86%|
| Validation visible history contains a moved item |4306/22363|19.26%|
| Validation target is a moved item |757/22363|3.39%|
| Validation history or target affected |4927/22363|22.03%|
| Training prefix input or target affected |28268/131413|21.51%|

The validation union consists of4170history-only rows,621target-only rows,
and136with both.17436rows have unchanged input and target assignments. There
are4947changed-item occurrences in validation contexts and29334in training
contexts. Changes to the first SIDtoken occur in3398validation histories.

These are local input/label changes. All users can receive different predictions
through globally retrained shared parameters. Dividing aggregate NDCGchange by
the22%exposure does not isolate a causal effect on the affected subgroup.

## Depth of final assignment changes

Counts below classify each moved item by the earliest differing token between
its original and final complete SID;they describe final assignments,not step counts.

| Earliest changed token | Items | Share of480 |
|---|---:|---:|
| First token |401|83.54%|
| First preserved,second changed |77|16.04%|
| First two preserved,third changed |2|0.42%|
| Only uniqueness suffix changed |0|0%|

Thus this is not mostly leaf movement. All256accepted moves were replayed and
reconstruct the exact archived owner permutation. Their union contains480items.

## Semantic displacement

T5vectors do not change. For item i,measure cosine between its own frozen
embedding and the embedding of the item that originally occupied its new path,
`owner[i]`. This is not distance between learned codebook centroids. Final
assignment and actual accepted pair distances are saved separately.

Final assignment cosine:median0.87917,mean0.88064,IQR0.85845–0.90181,
range0.79573–0.97097. Median cosine distance1−cosine is0.12083;median normalized
Euclidean distance is0.49159. Accepted-step pair cosine median is0.87841.

High absolute cosine is common in this catalogue. Random nonself catalogue
pairs have median0.84860. Across100uniform random derangements of the exact
same480items and their original path inventory,pooled median is0.85161.
ALS therefore selects closer assignments on average,but not just nearest
neighbors:the median original owner of the destination path ranks1304.5among
12100other items. Only33of480assignments are top20neighbors;76are top100.
Ranks are optimistic for ties and measured on normalized T5vectors directly.

The100same-item-set controls exactly match moved items,available paths,
popularity distribution,and history/target exposure. They are untrained index
diagnostics;they do not supply a recommendation-quality control. Their movement
depth is not matched:mean475.49of480change first token,versus401under ALS.

| Weighted graph normalized LCP | Original | ALS swap | Random same480items,mean |
|---|---:|---:|---:|
| T5semantic-neighbor graph |0.199945|0.196909|0.192671|
| Fit ALS-neighbor graph |0.045589|0.047448|0.044082|

ALS loses about1.52%of semantic graph coherence,versus3.64%for these random
controls. This supports semantic preservation relative to random reassignment.
Fit ALSgraph gain is selected by the optimizer and is not independent evidence
of recommendation improvement. Random draw ranges in`audit.json`describe100
mapping draws,not confidence intervals for recommender quality.

## Popularity of the480items

Popularity here counts original TRAINnext-item target positions,not repetitions
across epochs. Moved items have median5positions versus6for unmoved items;
mean9.43versus10.92. Distinct TRAINusers have medians6versus7. The moved set
carries3.45%of all TRAINtarget positions,slightly less than its3.97%catalogue share.

| TRAINtarget positions/item | Moved items | Share of480 |
|---|---:|---:|
|0|2|0.42%|
|1–4|190|39.58%|
|5–9|180|37.50%|
|10–19|63|13.13%|
|20–49|35|7.29%|
|50+|10|2.08%|

There is a modest tilt toward less popular items,not a selection consisting
only of unsupported or unused items. The dataset itself has many rare items.

## Different4%variants have different footprints

| Index | Changed items | Median TRAINtargets/moved item | TRAINtarget mass | Validation history-or-target exposure |
|---|---:|---:|---:|---:|
| Main ALSuniform |480|5|3.45%|22.03%|
| ALSpriority |472|12|7.61%|34.68%|
| Context ALSwindow20 |482|7|7.02%|31.51%|
| Context ALSwindow5 |481|8|7.46%|32.33%|
| New random5%screen |605|6|4.81%|28.55%|

Priority and its trained random control use the exact same472moved items and
paths,so their exposure/popularity match. The main uniform480and the random5%
screen do not match. Comparing their aggregate quality losses alone cannot
attribute the difference to useful collaborative optimization.

## Quality on the affected rows

All values below reuse saved paired validation ranks. No new generator fits,
forward passes,test evaluation,or GPU sessions were performed.

| Comparison | All-user relative NDCG10 | Affected-row relative NDCG10 |
|---|---:|---:|
| Fresh20epoch main ALSvs original |−4.86%|−8.04% on4927rows|
| Epoch20of the60epoch pair |−0.15%|−0.97% on4927rows|
| Epoch60of that same pair |+2.25%|+5.97% on4927rows|
| Historical priority10vs its same-item-set random control |+0.94%|+8.81% on7756rows|

The first20epoch affected-row absolute delta is−0.003014;paired-user95%interval
[−0.005851,−0.000247],target-cluster sensitivity[−0.006791,+0.000450]. The60epoch
affected-row delta is+0.002410;user interval[−0.001396,+0.005929],target-cluster
[−0.001744,+0.006821]. Thus a stable positive effect of the main index has not
been established even within locally affected examples.

There is an exploratory lead in the existing priority10comparison:affected-row
absolute delta+0.003292,user interval[+0.000562,+0.005869]. Its target-cluster
interval includes zero:[−0.000883,+0.007508];the all-user delta interval also
includes zero. This compares ALSagainst random,not against original,and uses
472items rather than the main480. It is not an independent confirmation of
main-index benefit. All intervals use2000paired resamples,are post-hoc and
unadjusted,and condition on fitted weights;they omit neural-seed uncertainty.

The supported conclusion is that ALS makes more semantically conservative
assignments than unconstrained random reassignment,while useful downstream
collaborative improvement remains unconfirmed. A future discriminating control
would use the same moved-item set/path inventory and comparable semantic
displacement and depth of movement while omitting CFoptimization. No new
training for such a control is authorized or launched here.

## Reproducibility and retained data

`scripts/analyze_als_swap_footprint.py` checks pinned embedding/SIDhashes,
both frozen input manifests,data/catalogue identity,owner permutations,and
exact replay of256moves. Saved users and targets are aligned across quality
comparisons;the fresh20global metrics match the archived suite. Other ranks
reuse the prior independently verified replay/long-run analyses.

Artifacts:`/workspace/artifacts/sid-als-footprint/`:

- `audit.json`:all counts,quantiles,reference mapping draws,variant comparisons,
  quality tables and intervals,definitions and limitations.
- `moved-items.parquet`:all480items with catalogue text,original/final SID,
  destination's original item ID,T5cosine/distance/rank,and popularity fields.
- `accepted-swaps.parquet`:all256steps with item IDs,pre-swap path LCP,T5cosine,
  and objective delta.
- `user-exposure.npz`:all22363users/targets,history/target/first-token exposure
  flags and full-history changed occurrence counts.
- `footprint.png/svg`:depth,similarity,popularity,and exposure visualizations.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/rhpr-mpl-cache PYTHONPATH=src /workspace/rhpr-check/bin/python scripts/analyze_als_swap_footprint.py --plot
```
