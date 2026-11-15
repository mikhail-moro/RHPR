# Frozen Kindle pair alignment: completed results

Private `carasik/rhpr-kindle-alignment/1`, executable source
`a2de37f0d0a30fe4953b5e42b41b174c0c7e6902`. COMPLETE28/28evaluations,
107.065seconds measured experiment execution after initial runtime bootstrap,zero failures.
No training,checkpoint reselection,new seeds or quantizer fitting. Frozen
raw-VAL-selected55 Mixed32 and Pair32,seed42;958users per split,FP32SDPA,
beam20,batch64,no seen-item filter. Existing TEST previously examined,
so these are exploratory checkpoint sensitivity checks.

## All predeclared layouts

All item and pair SIDs retain3+1codes. The older half is paired;the recent half
remains single. `first`/`last` names identify the singleton at the boundary of
an odd-length older half. `edges` leaves BOTH older boundaries single.
`single` is Full32 without pair SIDs. Native lengths includeUSER.

|Layout|Positions|Forward GFLOPs|Mixed VAL NDCG10|Mixed TEST NDCG10|Pair VAL NDCG10|Pair TEST NDCG10|
|---|---:|---:|---:|---:|---:|---:|
|32_reference|97|0.516501|0.018937|0.009874|0.015966|0.010651|
|32_edges|101|0.539346|0.018437|0.010408|0.012739|0.007946|
|30_first|93|0.493853|0.019140|0.011194|0.015325|0.008708|
|30_last|93|0.493853|0.018992|0.010789|0.015001|0.008948|
|34_first|105|0.562388|0.017828|0.010771|0.014083|0.008382|
|34_last|105|0.562388|0.019372|0.009733|0.013630|0.009933|
|32_single|129|0.704770|0.017882|0.011408|0.005868|0.003593|

## Main findings

No layout repairs the original between-split instability. Mixed is comparatively
insensitive to these inference changes,while original Pair-only deteriorates
on BOTH splits under every alternative layout. This is descriptive evidence
of format tolerance in these fitted Mixed weights,not multi-seed efficacy.
All12 within-Mixed layout-versus-reference conditional user95%CIs cross0.

Mixed Full32 raises TEST.009874->.011408(+15.54%),20->24hits;VAL
.018937->.017882(-5.57%),35->35hits. TEST gains8hits and loses4;shared-hit
rank changes partly offset the gains. DeltaTEST+.0015345,conditional paired
user95%CI[-.001832,+.005413];deltaVAL-.0010557,CI[-.004776,+.002702].
Full32 costs.704770vs.516501GF(+36.45%);no established gain or Pareto claim.
For Pair-only,Full32 is an untrained input format:TEST.010651->.003593,
VAL.015966->.005868. Do not interpret that as intrinsic harm from single SIDs.

At the SAME32events,edges shifts the pairs but retains recent16singles.
MixedTEST.009874->.010408(20->23hits),VAL.018937->.018437(35->35).
TEST delta+.0005342,CI[-.001686,+.002981]. Pair-onlyTEST.010651->.007946
(21->18hits),VAL.015966->.012739(31->24). Positions97->101 and FLOPs
.516501->.539346 also change,so this is not a position-matched ablation.

## Stable pair identities do not give a consistent rescue

Each user has exactly the expected ordered-pair overlap between VALreference
and the altered TEST layout:edges7/8,34_first8/8,30_first6/8;
reference->reference0/8. Despite retaining ALL8prior pairs,34_firstTEST gives
Mixed.010771 and Pair.008382. The latter is below its reference.010651.
Keeping validation pair identities is not itself beneficial to a stateless
model;this is a sensitivity check on which representations it receives.

|TEST layout|Pair occurrences|Not observed in original TRAIN pair inputs|Mixed TEST|Pair TEST|
|---|---:|---:|---:|---:|
|32_reference|7664|847|0.009874|0.010651|
|32_edges|6706|830|0.010408|0.007946|
|30_first|6706|1673|0.011194|0.008708|
|30_last|6706|847|0.010789|0.008948|
|34_first|7664|830|0.010771|0.008382|
|34_last|7664|0|0.009733|0.009933|
|32_single|0|0|0.011408|0.003593|

The34_last layout has ZERO novel complete pair identities atTEST,but does not
improve either checkpoint over reference. Conversely30_first has1673novel
occurrences vs847reference,and is among the highest MixedTEST point scores.
This does not support a simple monotone novelty explanation. These are
confounded comparisons:30/34 also change event count,recent tail and positions.
Complete-pair novelty does not mean individual code embeddings are unseen.
All141137TRAINexamples were assigned Pair view at least once in Mixed across
60epochs,so the set of presented originalTRAINpair identities is retained;
the number of presentations is still roughly halved.

## The original Recent32 control is still stronger on TEST

Reusing already saved predictions,no additional evaluation:Recent32VAL
.016899/TEST.014447(28hits). Mixed Full32 uses the same single-item inputs
and.704770GF,yetTEST.011408(24hits),a21.03%point deficit. The paired delta
is-.0030383,CI[-.008561,+.002237];not established harm/equivalence. VAL
MixedFull32 is+.0009826,CI[-.004203,+.006280]. Thus compressed TEST input
alone is not a sufficient account of the fitted-model difference;the cause
of the validation/test reversal remains unidentified.

## Groups and uncertainty

MixedFull32TEST rare targets(1-5TRAINpositions):4->5hits;6-20:10->12;
>20:6->7;cold39users:0->0. Lower-popularity recent contexts(n502)go
.004994->.007243,7->10hits;other456go.015246->.015994,13->14.
AtVAL,the same Full32 change does not consistently help these groups.
Group observations are sparse,previously inspected,and exploratory.
All comparisons condition on selected fixed checkpoints and5000paired-user
resamples;they exclude training/index/checkpoint-selection uncertainty and
multiplicity correction. Do not pick a layout fromTEST or call it a router.

## Verification and artifacts

All28prediction files independently replayed,536480ranked paths:584invalid,
zero duplicates. Fourreference runs reproduce EVERY rankedSID AND score
bit-for-bit. Verified19parent pins+10data hashes+2savedRecent32prediction pins,
full input/target hashes,model-state hashes against preflight and FLOPs.
Both frozen model states unchanged,gradients disabled. All838newpair768D
embeddings retained;applying saved normalized-residualcentroids onCPU exactly
reproduces GPUcodes and appended collision suffixes. All116401originalfull
3+1pairSIDs remain byte-identical;extended index117239is unique.

Private complete raw outputs and derived arrays:
`/workspace/artifacts/kindle-pilot/alignment-results`.
Includes28predictions,metadata/logs,newpairvectors/codes,model-metrics.csv,
user-ranks.npz and analysis.json. No checkpoint copies or private raw data
committed. Aggregate JSON:`gr-kindle-alignment-results.json`;completion:
`gr-kindle-alignment-completion.json`. CPU reproduction:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/kindle-pairs-mpl PYTHONPATH=src \
  /tmp/unique4-resume-venv/bin/python scripts/research/analyze_kindle_alignment.py \
  --results /workspace/artifacts/kindle-pilot/alignment-results \
  --pair /workspace/artifacts/kindle-pilot/long-pairs-results \
  --mixed /workspace/artifacts/kindle-pilot/mixed-resume-results \
  --data /workspace/artifacts/kindle-pilot/long-input
```

No new GPU work,training,additional variants,notifications or pending runs.

Final regression:1124tests pass,152optional skips,0failures,79.589s.
