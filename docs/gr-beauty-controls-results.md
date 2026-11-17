# Beauty Recent12 and 6+6 generator: completed results

Collected2026-11-17. Both authorized Kaggle sessions COMPLETE and audited:

- `carasik/rhpr-beauty-recent12/1`,source
  `413f2afb993b5b56f93534a7ccef155313e89938`,2518.518s/41.98min.
- `carasik/rhpr-beauty-mender-depth/1`,source
  `a68b70dbfe15b386d40552923afbe9801a7f5b90`,6989.028s/116.48min.

Durations are measured run-manifest time after bootstrap,excluding queue.
Exactly3fresh fits total. No new GPU training,inference or hyperparameter
search was performed during collection. Historical controls below reuse
Beauty/1,source`7828d110117889667c3a5e5877f7be9bd0222c13`.

## Quality and forward cost

Same complete22363users/12101items,TRAIN131413prefixes,22363VAL/TESTtargets,
same original frozen item index,seed42,rawVALNDCG10checkpoint selection.

| Model | VALNDCG10 | TESTNDCG10 | TESTRecall10 | ForwardGFLOPs | Best/stopepoch |
|---|---:|---:|---:|---:|---:|
|4+4Recent3|.04399019|.02997927|.05442025|.082200520|50/75|
|4+4Recent12,NEW|.04445329|.03137152|.05907079|.257705640|60/80|
|4+4Recent20|.04576327|.03173609|.05938380|.427088200|55/75|
|4+4RandomTail3|.04730982|.03149949|.05889192|.257705640|85/90|
|4+4RandomTail5|.04564402|.03162411|.05969682|.278189884|50/75|
|6+6Recent3,NEW|.03905005|.02819985|.05388365|.123168026|110/110|
|6+6Recent20,NEW|.04207304|.02978388|.05468855|.640482218|75/95|

Forwardcost is supported B1mathSDPAteacher-forced4codeFMA2 at the fixed padded
width,not beam-serving latency;excludes backward/text/cache costs. Runtime
Recent3profiler differs by44FLOPs from its CPUpreflight. Both measured values
are preserved,not silently replaced. Generator dimensions/parameter count
match actual checkpoints.

## Larger generator did not improve overall quality

We adopted6+6layers anddropout.2 from the official MenderTIGERBeautyconfig,
keeping our other settings,includingUSER2000. Oldgenerator4+4/dropout.1,
5108992params;new7338496params. Thus depth AND regularization changed;
this is not an isolated capacity ablation or exact paper reproduction.

Relative to the old architecture at the same cap,newRecent20loses8.06%VAL
and6.15%TEST;newRecent3loses11.23%VALand5.94%TEST. Conditional paired-user
TEST95%intervals for absolute deltas:

- new20minusold20:−.00195221,[−.00349475,−.00039022].
- new3minusold3:−.00177942,[−.00330241,−.00026078].

Within4+4,Recent20overRecent3is+4.03%VAL/+5.86%TEST. Within6+6,it is
+7.74%VAL/+5.62%TEST. The paired four-model contrast
`(new20−new3)−(old20−old3)` is+.00124991VAL,CI[−.00122390,+.00371744],
and−.00017279TEST,CI[−.00227096,+.00191736]. Therefore no reliable aggregate
increase in history benefit was established. These intervals condition on
one index/generator seed and selected checkpoints;they are not seed-level
replication,multiplicity-adjusted inference or population guarantees.

Both architectures have a positive overall20minus3TESTmean. The current
results do not imply that all old history is ignored. The transferred
architecture configuration failed;they do not prove that a bigger generator
cannot help with other regularization/optimization.

## Length subgroup nuance

Actual raw visible length,cap20;allTESTusers have>3events. Same users and
targets within each row,not independently filtered model samples:

| TESTcontext | Users | OldRecent3 | OldRecent20 | NewRecent3 | NewRecent20 |
|---|---:|---:|---:|---:|---:|
|4–6|14063|.0262622|.0292371|.0255414|.0265728|
|7–10|4753|.0297020|.0317430|.0302525|.0317509|
|11–20|3547|.0450880|.0416347|.0359893|.0398792|

On11–20inputs,the within-architecture direction reverses:old20is7.66%below
old3;new20is10.81%above new3. VALalso reverses:old3.0664093vsold20.0606317,
new3.0545131vsnew20.0558824. This is a descriptive sign change,not an overall
improvement:in this TESTgroup,new20still loses4.22%toold20,andnew3loses20.18%
toold3. Higher relative history benefit is partly a weaker short-context
model,not proof that added depth recovers a better recommendation solution.
No validated length router;one seed and exploratory subgroup comparisons.

## Recent12 closes the compute-matched baseline gap

Recent12andRandomTail3have exactly49padded encoder positions and.257705640GF.
RandomTail3overRecent12:VAL+6.43%(.00285653absolute,
CI[+.00092589,+.00472474]);TESTonly+.41%(.00012797,
CI[−.00144844,+.00175220]). There is no establishedTESTadvantage over this
simple truncation control,and the uncertainty still allows several percent
loss or gain. Similar means are not a formal noninferiority result.

Recent12versusRecent20:VAL−2.86%,TEST−1.15%,forward−39.66%. RandomTail3versus
Recent20:TEST−.75%,same−39.66%forward. Both yield closeTESTmeans at lower
fixed widths,butcompression has not shown an advantage over truncation.
The nominal positiveVALcontrast is not robust evidence across splits/seeds.
Recent3remains an even cheaper baseline;it is not compute-matched toTail3.

WarmTRAINepoch:Recent1217.25s,RandomTail328.30s,Recent2029.64s. Thus equal
maximum inference width does not make their training equally fast. PairTRAIN
uses weaker random compression and CPUtokenization;timings also come from
different sessions/worker placement. No production latency comparison.

## Learning behavior and stopping

All3new fits stopped by declared EMAearly stopping;none hit120epoch cap.
Recent12best60/stop80;new20best75/stop95;new3best110/stop110. The last case
is valid under the protocol:checkpoint selection uses rawVALNDCG,but stopping
usesEMAwithmin_delta.0005. Last EMAreference improvement was90;rawNDCG110
can be the best without resetting that counter. It is not a stale checkpoint
or an off-by-one error. Operational stopping does not prove global convergence.

Minimum teacher-forcedVALCE:Recent1240(8.64979),new2060(8.63068),new375
(8.69755). Old20min40(8.64029),old3min40(8.75180). New models reach similar
or better minimumCE while ranking quality is lower. CEandselectedNDCGare
different objectives;these measurements alone do not identify the mechanism.
Loss rises after its minimum rather than showing a uniformly improving tail.

New20warmTRAIN45.80sversusold2029.64s;new312.84sversusold39.16s.
NewskippedAMPupdates:Recent122,new204,new36;no numerical failure/fallback.
OptimizerTRAINseconds1453.75/4494.37/1539.52;VALseconds893.45/2244.57/1216.88.

## Audit and retained evidence

All new configurations/manifests/suites/finiteflatmetrics/requiredsummaries
verified. Both recomputed embedding matrices,centroids and3+1itemSIDarrays
are byte-identical to the original Beauty/1arrays. All rawTRAIN/VAL/TESTcases,
targets,per-architectureinitializer,schedule/updatecounts,EMA/checkpoint
selection and selected model hashes verified. All6new best/lastweights finite;
last optimizer/scaler/scheduler/RNG/encodedcase hashes retained and checked.

All57newVALpredictionfiles,3newTESTfiles and8historicalselectedVAL/TESTfiles
independently replayed:68files/30413680candidate paths,25632invalid paths
retained at original ranks,0duplicates. Historical full learning/augmentation
audit remains in`gr-beauty-pairs-results.md`;we did not repeat its63VALchecks.

Raw outputroots:`/workspace/artifacts/beauty-recent12-results` and
`/workspace/artifacts/beauty-mender-depth-results`. Combinedanalysis:
`/workspace/artifacts/beauty-controls-analysis`,including7modelmetrics,
313082user-model-split rows,paired contrasts,selectedranks andPNG/PDFfigures.
Both figures visually inspected. All data/weights remain outsideGit.

Reproduce with CPUonly:

```
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/history-mpl PYTHONPATH=src \
/tmp/history-pilot-venv/bin/python scripts/research/analyze_beauty_controls.py
```

Reports:`gr-beauty-controls-results.json`,`gr-beauty-controls-completion.json`.
Full1170tests total,152optional skips,0failures,143.955seconds.
No new training,inference or additional seeds were launched.
