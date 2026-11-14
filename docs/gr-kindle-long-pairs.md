# Kindle Pair32/Pair64 pilot, seed42

Question: can merging older adjacent descriptions preserve or improve quality
on longer windows with the existing wider TIGER? An improvement would establish
an effect of this representation pipeline under this budget, not identify noise,
capacity or attention distances as the unique cause.

Train exactly two new generators:Pair64 and Pair32,one perT4. Reuse completed
wide Recent20/32 controls from private `carasik/rhpr-kindle-capacity-pilot/1`,
source `6d2919a155893e4c6bbb859a1fbcc0c800573c6c`. Control64 is deferred;
the narrow historical Recent64 is not a matched wide control.

Same cohort958users/16177books/141137train next-item prefixes. Same targets,
split,shuffles,seed42,frozen item3+1 SID and wide4+4/d128/FF1024/dkv64/H6.
Same initial Transformer and attached pair-input table as the completed wide
controls;now the pair table is actually used. Nominal parameter count is equal,
but active input embeddings differ. Same FP16/SDPA/compile,batch512/512,
FP32 validation beam20/batch64/every5,LR.003→.001,warmup2/cosine60,
EMAalpha.5/delta.0005/patience15,min40/max60. Raw best VAL NDCG@10 selects
weights;EMA only controls stopping. Keep full best/last checkpoints,curves,
per-VAL predictions and losses,FLOPs,timing and CUDA memory. A time exit is
incomplete,not evidence of convergence. No model or positional-bias retuning.

For each example first take the latest32 or64 observed items,remove left
padding,then merge nonoverlapping ordered adjacent pairs in the older
floor(n/2) items. Any orphan remains single. The recent half is always single.
Pair descriptions are `First book: {title+categories} Second book: {...}`,
encoded once by the same local Sentence-T5(max2048tokens),then cached.
Fit a separate GRID residual KMeans3x256 on uniformly sampled UNIQUE TRAIN
pairs from the UNION of the two windows. Both models use that one frozen
pair index;the item index is copied,never refit. Pair keys encode order.

Real counts:114727TRAIN pairs,22403VAL-context pairs,22397TEST-context pairs,
116401union. Fit only TRAIN vectors;assign other pairs with frozen centers.
Each pair also has its own uniqueness suffix:3+1 codes,independent input
namespace. Allocate suffixes TRAIN first,then new VAL,then new TEST keys;
adding evaluation contexts cannot rewrite previous codes. No target content
or labels enter fitting. TEST contexts include the observed VAL item,as before.
The same catalogue/transductive preprocessing limitations as the parent apply.
No fallback for suffix overflow or numerical failure is authorized.

| Representation | Raw full window | Encoder SID tokens | Plus USER |
|---|---:|---:|---:|
| Existing Recent20 |20|80|81|
| Existing Recent32 |32|128|129|
| New Pair32 |32|96|97|
| New Pair64 |64|192|193|
| Deferred wide Recent64 |64|256|257|

These are not equal-FLOP comparisons:Pair32 lies between Recent20/32 and
Pair64 has more compute than Recent32. CPU math-SDPA supported forward
operation counts are measured using actual full examples;FMA=2,teacher-forced
four-code prediction including output heads,excluding beam/backward/text
encoding. Pair encoding/KMeans preparation time and storage are additional,
cached offline costs,not zero-cost deployment claims.

Preflight full-shaped CPU forwards with synthetic pair-code fixtures give
Pair32=.516500848GF and Pair64=1.119081520GF;runtime profiles repeat this on
the actually fitted pair SIDs. Matched initializer and finite backward verified.
All38 parent pins verified. Eight actual VAL examples per old control replay
the entire beam20 SID ranking exactly versus saved GPU predictions;weights
unchanged. Full CPU replay was stopped for cost,not a failed numerical check;
all44 parent trajectory prediction files had already been replayed numerically
in the capacity analysis. Full1109tests/152optional skips pass. See
`gr-kindle-long-pairs-preflight.json`.

Open TEST for exactly four predeclared checkpoints:the new VAL-selected Pair32/
Pair64 and already VAL-selected wide Recent20/32. Controls receive inference
only,no extra updates,epochs or reselection. FP32 beam20,batch64,original ranks,
no seen-item filter. Save raw predictions/scores/users/targets and checkpoint
hashes;report NDCG/Recall5/10,invalid and duplicate counts. TEST is secondary
final reporting,not an optimization or stopping signal. After viewing it,
further experiments on this cohort must be described as exploratory rather
than independent confirmation. The two old control best weights are byte-pinned
along with38 parent artifacts including both full VAL trajectories.

Compare common fixed40,best within common budgets,and VAL-selected curves
before interpreting TEST. One fixed previously examined seed and958targets
provide a cheap screening result,not a robust multi-seed or causal explanation.
Merging changes token semantics,positions,sequence length and the active input
table at once. Lower NDCG at a selected epoch does not establish that attention
cannot model old items;relative T5 buckets do not truncate full attention.

Private/offline kernel `carasik/rhpr-kindle-long-pairs`;no notifications.
Config `experiments/kindle_long_pairs_2t4.json`. Session cap4h;individual fit
cap2h. Expected roughly1.5–2.5h including pair-cache preparation,with uncertainty
in ST5 throughput and compile overhead. Quota before launch40.12h GPU remaining.
