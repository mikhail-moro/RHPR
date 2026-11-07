# Codebook adaptation: completed exploratory comparison

Private carasik/rhpr-ml1m-codebook-adaptation/1 COMPLETE,33.951min.
Executable b828588a3bb1ca228d73507d875df4bda69d7add. Both arms30epochs,
8880updates,6VALchecks,151166TRAINwindows/epoch,0AMPskips. One seed42.
Top-level metrics,summary andmanifest present andvalid. No further compute.

| Arm | Best VAL NDCG10 | Selected TEST NDCG10 | Best epoch |
|---|---:|---:|---:|
| Historical independent lookup SUMcontrol |.06665966|.06165244|25|
| Historical copiedcentroids,fixedSID |.05971500|.05188044|25|
| Frozen64DRQ + learned shared64->128projection |.03686649|.03114151|30|
| Full128DRQ,NIP+Rec+VQ,sharedcentroids,no projection |.02118170|.01895965|25|

Projection remains substantially worse on this30epochbudget and was best at the
finalepoch;this does not establish its converged quality. It constrains the768
codeword inputvectors to a frozen dictionary transformed by one8192parameter
linear map. Independent lookup can move codeword vectors separately. If the
fixed vectors remain distinct,this is a geometry/optimization restriction,not
necessarily loss of discrete token identity. The extra frozenRQnetwork parameters
do not add active NIPcapacity. FulljointRQ changes objectives,latentdimensionality,
assignments andactiveparameters;not a clean causal test of any one factor.

## Index and objective dynamics

RQloss prevented the previous catastrophic overflow. At joint epoch30: occupied
codes27/24/30,1756unique3token semanticIDs,largestsemanticcollision9;fullSIDs remain
unique via suffix. Original occupied255/252/231,2871unique semanticIDs,maxcollision2.
Joint median SID churn between epochs2–30:84.93%;epoch30changes2385/2874items.
No numericalfailure or suffixoverflow. Projection index stays exactlyfixed.

Joint mean training RecLoss2.6305(epoch1)->2.3620(epoch30),VQLoss.1850->1.4835,
NIP12.3174->7.3160. RQmath isFP32;NIPFP16. These are changing assignments and
sampled-catalogue auxiliarylosses,not comparable fixed-label NIPlearning curves
across all arms. No gradient conflict angle/norm study was performed. Under GRID
STE,books receive directNIP+VQgradients;Rec trains encoder/decoder. Therefore the
failure does not directly measure an NIP-versus-reconstruction gradient conflict.
The earlier NIP-only live run already collapsed without any RecLoss.

## Information redistribution in the SID

Empirical entropy over the SAME151166TRAINtargetpositions,weighted by targetitem
frequency. Each row is the incremental conditional entropy given previous SID
tokens;userhistory is NOT conditioned on. This is NOT teacher-forced CE/NIPLoss.

| Increment | Original index bits | Joint epoch30 bits |
|---|---:|---:|
| First token |7.66010|4.41679|
| Second given first |2.76905|3.20974|
| Third given firsttwo |.05791|2.22077|
| Suffix given semanticprefix |.00072|.64048|
| Total |10.48779|10.48779|

This invariance follows from the fullSID being an injective itemidentifier. For
any FIXEDbijective encoding,ideal conditionalentropy H(SID|history)=H(item|history)
as well,when historyinformation is preserved. A finite generator and beamsearch
can find different encodings easier or harder;their quality need not be invariant.
Collapsingearlyprefixes shifts disambiguation to later tokens/suffix rather than
removing itemprediction uncertainty. Without uniqueness constraints,actual loss
of itemidentity could indeed yield a degenerate easier classification target.

NIP gradient in this implementation treats discrete assignments as fixed during
each epoch. It does not differentiate the effect of moving a centroid on future
history/targetlabels. Reassignment minimizes residualdistance,not generator CE.
Thus a centroidstep improving lookup usefulness need not improve the composed
objective after the index changes. Coupled Adamweightdecay and LR also changed
centroid geometry;no ablation isolates a deliberate preference of NIP for a
smallerindex. The results do not disprove jointlearning with different couplings,
separate taskheads,soft/STEbridges,stable assignments or different schedules.

## Verification and saved evidence

Independently replayed14newpredictionfiles(12VAL+2TEST)against matchingepochSID
snapshots,users/targets andsavedNDCG/Recall5/10/invalid/duplicatecounts. Config,
sourceSHA,casehashes,initializationpins,selectedsnapshot andbudgetsverified.
Empirical fullSIDentropy conservation asserted numerically. No new training,
generatorinference or GPUdiagnostic. Existing implementation checks1021tests,
91optional skips,0fail passed before submission;this collection changes docs only.

Full JSON docs/gr-codebook-adaptation-results.json. Raw local
/tmp/codebook-adaptation-completed;analysis /tmp/codebook-adaptation-analysis;
plot codebook-adaptation.png/.pdf. EvidenceZIP /tmp/codebook-adaptation-evidence.zip
contains numericaloutputs,verification,entropy,plots,TRAIN/VAL/TESTcases and
CPUreplay scripts. Weights remain on the private Kaggle output.
