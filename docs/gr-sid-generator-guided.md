# Frozen-generator-guided SID reassignment

Authorized2026-10-28:one bounded likelihood search using the existing frozen
original RQ-KMeans20epoch generator,then two fresh20epoch fits on2T4:original
and optimized indices. Validation5/10/15/20,primary20,no semantic penalty or test.
Config:`experiments/amazon_beauty_sid_generator_guided_2t4.json`.

Submitted once as private [Kaggle v1](https://www.kaggle.com/code/carasik/rhpr-generator-guided-sid),
now COMPLETE; see [results](gr-sid-generator-guided-results.md).
[Actions40663408472](https://github.com/mikhail-moro/RHPR/actions/runs/40663408472)
completed successfully. Executable commit`c3459221dfae7163b46de0d9ec28c4003a679735`;
the actual notebook configuration and all179embedded package files match that
commit. Package SHA256`d64fd323278a1b422b1bd07ba403b0e10880578a2ad99216d7a3a75731fd19f4`.
Private/offline/T4 settings,two dataset mounts and frozen-source kernel mount
were verified. No second submission or automatic continuation. Local preflight,
submission receipt and source audit are in`/workspace/artifacts/sid-generator-guided`.

This is a JTM-inspired pairwise whole-path search,not a reproduction of JTM's
segmented assignment algorithm. The scientific question is whether a tree
optimized for one frozen generator transfers to a newly initialized generator.

## Pinned source

Existing private kernel`carasik/rhpr-sid-corruption-screen`,executable source
`1e87396aa65fd8b2dc9259a650e501cec25eec35`,trial`random_p000`.
Checkpoint`trials/random_p000/generator-best.pt`,19977481bytes,SHA256
`deadfd084b77f4a68a0f39097a423d187316b94a9b2a4d1ea563639efbf303c9`.
Contents independently checked:epoch20,10280updates,original-index validation
NDCG10=0.03644568500575003. Its config matches the source run. Runtime verifies
the parent manifest,checkpoint,source result and separate original SIDtable by
pinned size/hash,source revision,epoch,updates,data,generator,seed and training
protocol. Source checkpoint is loaded weights-only and never fitted again.

Reuse existing datasets`carasik/rhpr-gr-beauty-phase0`and
`carasik/rhpr-sid-context-screen`,plus the existing source notebook as a private
kernel input. No dataset upload,encoder,quantizer,ALS,or semantic graph fitting.

## Search objective and batching

Freeze theta and maximize the event-weighted TRAINobjective

J(pi)=sum_(H,y,u in TRAIN)logP_theta(pi(y)|pi(H),u).

Each likelihood is the sum of four raw full-head teacher-forced logprobabilities,
including suffix. Candidate paths supply their own earlier decoder tokens;
histories and targets both use the current candidate mapping. Native per-head
full-vocabulary softmax is retained,no valid-child renormalization or beam score.
This matches the existing training likelihood weighting rather than equal
weighting of per-item means.

Build raw-item inverted incidence over all131413TRAINprefix cases. A swap affects
all rows whose target is either item OR whose visible history contains either
item. Cache current per-row logprobabilities. Group the affected rows by their
original fixed64-row minibatches and rescore those complete groups with the
new table. Batch composition/padding are identical to the original cache pass,
including the final partial batch. This avoids arbitrary subset layouts deciding
small likelihood differences. All cached entries of accepted batch groups are
updated;rejected proposals leave mapping and cache unchanged.

Accept delta>max(0.01total nats,1e-5nats*affected-row count). Source/model arithmetic
is FP32;per-example four-term sums and aggregate deltas use FP64. An identity
rescore must agree within1e-5. At the end,fully rescore all TRAINcases in the
original batch grid:per-row cache drift<=1e-5 and summed drift<=0.02nats. Source
state and CPU/CUDA RNGstates must remain unchanged. Original complete path set,
owner permutation and catalogue prefix occupancy at every depth are checked.

## Proposal mining and budgets

Mine once from16384uniformly sampled TRAINpositions,seed2758,using the frozen
original mapping. Native greedy decoding uses one path(no prefix constraint),
and only valid catalogue full paths are retained. Its probability products are
ignored;proposal confidence is rescored with exact teacher-forced logP of the
proposed path. Keep a wrong item if logP(wrong)-logP(observed)>=log(1.25)and both
items have at least2TRAINtargets. This filters proposals,not acceptance data.

Aggregate unordered item pairs,rank by summed confidence gaps clipped at5nats,
then mistake count and item IDs for deterministic ties. This is approximate
candidate retrieval,not exact catalogue argmax. The candidate list stays fixed
while acceptance always evaluates the current mapping on all affected cases,
including cases that were originally predicted correctly. No validation/test
examples or labels are used for proposing or accepting moves.

Stop at the first cap:2000candidate proposals,256accepted transpositions,
2000000search-scored rows,or1200search seconds. Time is checked between complete
proposals with the previous proposal's cost as a guard;one expensive proposal
may overrun. Initial complete-cache pass,mining,and final full audit are separate
stages and are timed/logged separately. Do not force acceptance or weaken a
threshold. If no swaps are accepted,record that result and skip the two identical
fresh fits. There is no retry,extra seed,or automatic budget extension.

## Fresh comparison and saved evidence

After freezing the new mapping,two isolated workers each fit one generator
from the same initial named-neural-parameter hash
`5f4a24fa8c9b5f7063739fd44822fd0a5f98eff91c7e882563d32529b0c5abff`,seed42,
same raw cases and epoch order. GRID4+4/d128/FF1024/6heads,4977920parameters,
Adam1e-3/wd1e-4/clip1,batch256/physical128,FP16training,FP32validation batch32,
free beam20,no seen mask. All131413prefixes once per epoch,20epochs,
2628260examples/model. Do not warm-start either fresh fit from the teacher.

Save all eight validation beam files,best and full last states,and teacher-forced
per-level NLL/ranks at all four checks. Save the optimization report,all candidate
pairs,mining probe row IDs/predictions/margins,one JSONLrecord per evaluated move,
accepted moves,initial/final TRAINlikelihood cache,and final SID/owner arrays.
Every record includes actual counts and stop reason,including no-change runs.

No semantic regularizer is used. A distance penalty would discourage dissimilar
swaps and needs a new calibrated scale;oldlambda0.3does not transfer automatically.
Improved teacher TRAINlikelihood is only a search sanity check. Downstream NDCG
can fail to improve after fresh training,and pairwise search may find a local
optimum. One fitted seed/candidate-mining seed is a screen,not seed replication.

Expected GPUexecution roughly45–65minutes including bounded search and parallel
fresh fits;queue and source-mount startup add uncertainty. Exact GPUthroughput
is measured in the session. Local CPUprofiling is not used to infer GPUtiming.
No Telegram,test evaluation,additional kernels or continuation are authorized.

## Verification before submission

Core tests show that improving target-item examples can be outweighed by harm
to other targets through histories. Cache tests preserve original batch shapes,
accept/reject behavior and full-rescore equality. Native tests train a small
source model,exercise exact likelihood/greedy mining and frozen weight/RNGguards,
run local search,then launch two fresh isolated CPUworkers with identical initial
parameters. The zero-pair branch skips both fits. Real source preflight verifies
epoch/updates,initial neural hash,native loss equality,exact repeated scores,
unchanged source state/RNGand shared case/order hashes.

Local artifacts:`/workspace/artifacts/sid-generator-guided/`.
