# Completed frequency-aware and stronger compression pilot

Collected2026-11-15fromprivate`carasik/rhpr-kindle-frequency/1`,source
`7a9a7bd430362777eeecbb058a9cebf339f9a2cd`. COMPLETE,all3requested fits
finished60epochs;no new GPU fits/inference were submitted to collect/analyze.
Experiment runtime4202.054s=70.03min after initial bootstrap,excluding queue.
TwoT4s;frequency5thenrandom8onGPU0,random5onGPU1.

## Outcome

The fixed-budget frequency priority did **not** improve overall NDCG10 in this
pilot:VAL−12.12%/TEST−3.08%against matched random5 at identical.319748392GF.
Conditional paired-user intervals include zero for both comparisons,so neither
benefit nor seed-robust harm is established. Rare-target quality did not improve
on both splits. No implementation or coverage error was found by CPU replay.

More aggressive8pairs saves19.40%forward FLOPs relative to5pairs,but TESTNDCG
falls24.63% (the test NDCG/GFLOP ratio falls6.49%). Its VALNDCG falls only7.17%,
so the VALratio improves15.18% while TESTratio worsens. This split disagreement
prevents a general quality/compute claim. ExistingRecent10is both cheaper
(.217327232GF)and higher quality on VAL/TEST than all3new fitted models by point
estimate. These are selected single-seed results,not population noninferiority.

| Model | Best epoch | VAL NDCG10 | VAL hits10 | TEST NDCG10 | TEST hits10 | Forward GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| Frequency5 — preserve rare context items | 55 | 0.012243 | 20 | 0.009904 | 21 | 0.319748 |
| Random5 — same budget / first16eligible | 60 | 0.013931 | 30 | 0.010219 | 23 | 0.319748 |
| Random8 — eight old pairs / four recent singles | 55 | 0.012932 | 25 | 0.007702 | 14 | 0.257705 |
| Previous Random20 — old10paired/recent10single | 60 | 0.014796 | 24 | 0.015643 | 29 | 0.319748 |
| Recent10 | 60 | 0.014931 | 29 | 0.014623 | 26 | 0.217327 |
| Recent20 | 55 | 0.020492 | 38 | 0.014722 | 24 | 0.427088 |
| Recent32 | 55 | 0.016899 | 30 | 0.014447 | 28 | 0.704770 |

The previous Random20 had5pairs limited toold10and10recent singles. At the
same FLOPs,newrandom5is−5.85%VAL/−34.68%TEST. New policies allow merging six
more recent events and also change the distribution/diversity of TRAIN layouts
and fixed evaluation placement. Thus the observations favor preserving the
recent tail in this setup,but do not isolate a causal effect of recency alone.
The exact matching of train counts/FLOPs applies to frequency5vsrandom5.

## Did the frequency policy actually protect rare contextual items?

Yes. Rarity is<=5distinct original TRAINusers,includingfirstevents,excluding
VAL/TEST,before prefix expansion. On VAL merged rare-item occurrences are
397vs1779(−77.68%);TEST406vs1814(−77.62%). The total pair count remains5per
example for both policies. Frequency5preserves every rare context item in
77.66%VAL/76.72%TESTcases,versus25.99%/23.70%forrandom5.

Across all60TRAINepochs both budget5arms receive exactly20,273,000pair
presentations and the same per-example merge count/padding schedule.
Frequency5merges287,246rare-item occurrences vs5,820,476forrandom5(−95.06%).
This confirms that the manipulation worked;it does not establish that rare
context merging was responsible for previous rare-target errors. The number
of single-item next-item targets and original users is unchanged.

Random8receives31,560,557pair presentations and9,045,165merged rare occurrences.
On evaluation it merges all6events in positions11–16of the latest20;
frequency5merges3.60VAL/3.56TESTof those events on average,random5about3.72,
where the previous Random20merged none of the latest10.

## Target rarity and context rarity give different answers

The table below uses TRAIN-user rarity for both targets and context;older
reports used TRAINnext-item-position support,which differs slightly because
initial events were excluded. Both versions are retained in the machine report.
Rare context means the first16of the latest20;group thresholds are exploratory.

| Split | Group | Model | Users | NDCG10 | Hits10 |
|---|---|---|---:|---:|---:|
| validation | target_train_users_1_5 | frequency5 | 400 | 0.002468 | 2 |
| validation | old16_no_rare | frequency5 | 147 | 0.016237 | 3 |
| validation | old16_rare1_2 | frequency5 | 332 | 0.010898 | 5 |
| validation | old16_rare3plus | frequency5 | 479 | 0.011950 | 12 |
| validation | target_train_users_1_5 | random5 | 400 | 0.001643 | 2 |
| validation | old16_no_rare | random5 | 147 | 0.029499 | 12 |
| validation | old16_rare1_2 | random5 | 332 | 0.013057 | 10 |
| validation | old16_rare3plus | random5 | 479 | 0.009760 | 8 |
| validation | target_train_users_1_5 | random8 | 400 | 0.004675 | 5 |
| validation | old16_no_rare | random8 | 147 | 0.007401 | 3 |
| validation | old16_rare1_2 | random8 | 332 | 0.018766 | 11 |
| validation | old16_rare3plus | random8 | 479 | 0.010586 | 11 |
| test | target_train_users_1_5 | frequency5 | 425 | 0.000910 | 1 |
| test | old16_no_rare | frequency5 | 140 | 0.017941 | 6 |
| test | old16_rare1_2 | frequency5 | 336 | 0.010359 | 8 |
| test | old16_rare3plus | frequency5 | 482 | 0.007252 | 7 |
| test | target_train_users_1_5 | random5 | 425 | 0.002464 | 3 |
| test | old16_no_rare | random5 | 140 | 0.007685 | 3 |
| test | old16_rare1_2 | random5 | 336 | 0.018984 | 14 |
| test | old16_rare3plus | random5 | 482 | 0.004845 | 6 |
| test | target_train_users_1_5 | random8 | 425 | 0.000708 | 1 |
| test | old16_no_rare | random8 | 140 | 0.009208 | 2 |
| test | old16_rare1_2 | random8 | 336 | 0.010574 | 7 |
| test | old16_rare3plus | random8 | 482 | 0.005262 | 5 |

There is a weak,consistent-direction subgroup observation:with>=3rare context
items,frequency5vsrandom5improvesVAL.009760->.011950(12vs8hits,n479),
TEST.004845->.007252(7vs6hits,n482). Conditional CIs of the differences are
[-.006374,+.010799]and[-.002946,+.008007],both cross0. The TESTdifference is
only one extra hit plus ranking changes. In that same group frequency5still
trails previousRandom20(.008793)andRecent10(.008158)onTEST.

When at least2rare merges are avoided,frequency5also has a positive NDCGdelta
on both splits (n433/434),but both CIs cross0. This correlates with the context
rarity groups and is not independent replication. With no rare old-context
items,the sign reverses across splits. No usable router is established.

Rare targets themselves have2vs2hits onVAL(n400),1vs3onTEST(n425).
Thus the stronger statement "protect rare context items to recover rare
recommendations"is unsupported. It remains possible that soft temperature
would balance representation exposure,but this pilot tests a joint TRAIN+EVAL
placement policy and does not isolate training-only frequency weighting.

## Selection, learning curves and timing

All3models reached the60epoch cap;bestepochs55/60/55,not proven convergence.
Each sees141137original prefix targets per epoch=8,468,220targets in total,
958validation and958test users,one target each. Each has16558successful
updates+2AMPskips. Skipsfrequency5epochs52/59,random5andrandom8epochs51/59.
No LR/precision fallback or compiler-cache fallback was found.

Warm epoch medians29.55s/30.37s/26.10s;optimizer totals2308.57s/2328.35s/
1727.75s. First epochs369.49s/369.46s/122.80s;random8runs second in a worker,
so cold compilation/cache conditions differ and its first-epoch time is not a
fair standalone startup comparison. Peak allocated2.039/2.039/2.057GiB;
reserved4.764/4.764/4.563GiB. Historical control times are not simultaneous
benchmarks. FLOPs are B1supported matmul forward mathSDPA,teacherforced4codes,
FMA2;exclude beam/backward/offline embeddings and CPU placement,not serving
latency. Both budget5models have exactly61encoderpositions;budget8has49.

All3minimum teacher-forced VALCE occur at55:9.191580/9.189854/9.247453.
At60each CE worsens. Budget8VALNDCG drops.012932->.007485from55to60;
its selected TEST still uses55,so the reported test loss was not caused by
accidentally evaluating the final60epoch checkpoint. Budget5frequency uses55,
random5uses60by the predeclared rawVALNDCG rule. No alternative checkpoint was
chosen usingTEST. Curves and selected quality/compute points:
`/workspace/artifacts/kindle-pilot/frequency-results/frequency-curves.png`and.pdf.

## Paired uncertainty

5000paired-user bootstrap draws with fixed seed4242;conditional on fitted
models and selected checkpoints. These intervals omit generator/index seed
variation,item clustering and model-selection uncertainty;no multiplicity
correction. TESThas already been examined across this research series.

| Split | Difference | Delta NDCG10 | Conditional user95%CI |
|---|---|---:|---|
| validation | frequency5 − random5 | -0.001688 | [-0.008209, +0.004822] |
| validation | random8 − random5 | -0.000999 | [-0.007448, +0.005234] |
| validation | frequency5 − random20 | -0.002553 | [-0.009610, +0.004339] |
| validation | random5 − random20 | -0.000865 | [-0.007391, +0.005370] |
| validation | random8 − random20 | -0.001864 | [-0.009259, +0.005257] |
| validation | frequency5 − recent20 | -0.008249 | [-0.015735, -0.001199] |
| validation | random5 − recent20 | -0.006561 | [-0.014131, +0.000547] |
| validation | random8 − recent20 | -0.007560 | [-0.015083, -0.000602] |
| validation | random8 − recent10 | -0.001999 | [-0.008696, +0.004862] |
| test | frequency5 − random5 | -0.000315 | [-0.005642, +0.004712] |
| test | random8 − random5 | -0.002517 | [-0.007695, +0.002553] |
| test | frequency5 − random20 | -0.005739 | [-0.011936, -0.000034] |
| test | random5 − random20 | -0.005424 | [-0.011024, +0.000293] |
| test | random8 − random20 | -0.007941 | [-0.014086, -0.001930] |
| test | frequency5 − recent20 | -0.004818 | [-0.011113, +0.000991] |
| test | random5 − recent20 | -0.004503 | [-0.011291, +0.002153] |
| test | random8 − recent20 | -0.007020 | [-0.013630, -0.001119] |
| test | random8 − recent10 | -0.006921 | [-0.014125, -0.000016] |

For frequency5vsrandom5:VAL11gained/21lost/9shared hits;TEST13gained/15lost/
8shared. No consistent overall advantage. Several intervals comparing the
aggressive model to historical controls are below0,but exploratory conditional
intervals do not establish a universal harmful compression threshold.

## Reproducibility and artifact audit

CPU reproducer:`scripts/research/analyze_kindle_frequency.py`. Exact source,
69parent/cachepins+10inputpins,958users/targets/case hashes and same initializer,
all learning-rate schedules/EMA/stopping/best selection verified. All180actual
TRAINepochinput hashes and exposure counters replayed,including exact matching
of both budget5count/padding schedules. All6saved full best/last checkpoints
were downloaded and checked finite,epochs,SHA,scheduler counters.

Actual5055newSentence-T5vectors are finite and their RQassignments exactly
reproduced onCPUusing original centroids. TRAINthenVALthenTESTsuffix allocation
reproduces all126467pairs;all121412previous pair SIDs remain unchanged,
maxsuffix31,no tokenizer refit. Actual evaluation arrays/pair starts/rare counts
replay for all3models/bothsplits. All91prediction files across new fits and
reused controls replay exactly:1,743,560ranked paths,2882invalid,0duplicates.

Private raw/derived outputs,full weights,analysis.json,model-metrics.csv,
user-ranks.npz and PNG/PDF are under
`/workspace/artifacts/kindle-pilot/frequency-results`. Portable scalar/group
analysis:`gr-kindle-frequency-results.json`;completion receipt:
`gr-kindle-frequency-completion.json`. Full1145testsPASS,152optional skips,
0failures,109.228s. No new fits,inference variants,HPO or notifications.
