# ALS and prioritized SID search: ten-epoch screening

User authorized this run on2026-10-27:ten complete TIGER epochs,validation at5/10.
The candidates investigate whether graph placement helps under a different CF
signal/proposal mechanism. Exactly six new fits,one private two-T4 session.

## CPU preparation

`experiments/amazon_beauty_sid_als_prepare_cpu.json` and
`scripts/prepare_sid_als_screen.py` run locally. Reuse the exact frozen T5/native
RQ-VAE and T5/normalized-residual RQ-KMeans whole-path indices and semantic graph.
The encoder and quantizers are never updated. Same fixed75/25 TRAIN-user split.

Use `implicit==0.7.3` CPU ALS:64factors,regularization0.1,alpha20,20iterations,
seed2712,two CPU threads. Rows are users,columns items;binary TRAIN incidence,
no validation/test signal. Record input CSR hash,implementation/version,factors
and source-array hashes. Item factors with fewer than2fit users are masked;
construct positive-cosine top20neighbours. GPU training has no implicit dependency.

For each initial index compare:

1. ALS graph with uniform random pair proposals.
2. Same ALS graph with prioritized proposals:refresh every500checks;anchor
   priority is local CF error times `frequency/(frequency+10)`,sample within
   the top25%anchors.20%uniform exploration. Select a CF friend's other first
   branch,then a random partner among its8most semantically similar occupants,
   excluding the friend. Accept only an exact global CF+0.3semantic improvement.
3. Random SID reassignment on exactly the priority-changed item set,with no
   fixed points and the same available original paths.

Same k20/lambda0.3;maximum100kpair checks and cap256accepted moves per guided
trajectory. Common per-family budget is the minimum available guided count,
at most256;all four guided runs reached256here. Full trie/prefix occupancies,
catalogue code frequencies,collisions and supplied suffix ownership are fixed.

Random controls match affected items/paths with priority,not semantic distortion
or number of transpositions. Uniform/priority have the same accepted-move budget
but slightly different numbers of distinct changed items. These are limitations
of the comparison,not hidden equalities.

Observed preparation:37.9seconds. RQ-VAE uniform/priority change489/480items;
RQ-KMeans480/472. KMeans uniform needed99092checks versus8986priority checks.
The oldcooc20kindices changeddifferent numbers of items,so their endpoint metrics
are contextual references rather than a pure same-budget CF-source ablation.

The private input dataset is `carasik/rhpr-sid-als-screen`,about10MB. It contains
only derived numerical arrays and metadata from public GRID/P5 Amazon Beauty,
already used by the existing private research inputs. No raw reviews,text,
names,email or credentials. All transferred files are audited and SHA256-pinned.

## GPU screening

`experiments/amazon_beauty_sid_als_screen_2t4.json` trains exactly six candidates.
Reuse original and legacycooc-swap validation metrics at epoch10;no repeated
control fits and no original epoch5metrics are implied.

Keep original generator/data seed42,GRID4+4/d128,4977920parameters,FP16,
Adam0.001/wd0.0001/clip1,batch256/physical128. All131413TRAINprefix targets
exactly once per epoch,latest20context.10full epochs/5140updates/1314130examples,
validation FP32batch32freebeam20at5and10. No test data construction/evaluation.

Rank by **validation at epoch10**,not the best5/10checkpoint. Preserve best
validation selection separately. Save user predictions at both checks,curves,
best/lastweights,Adam,AMP,TorchCPU/CUDA/Python/NumPy RNG and SID matrix hash.
These states support later continuation;this run does not automatically train
beyond10or launch finalists. Early rankings can flip by20epochs,as observed in
the previous cooc experiment. This is screening,not final proof of superiority.

The extra validation/capture must not change learning:the native test compares
final weights,optimizer state and RNG across two validation cadences. Another
guard makes test labels unreadable during validation-only training.

Keep the run private/offline. No encoder/quantizer training,ALS hyperparameter
search,otherlambda/k,extra generator seeds,automatic retries or Telegram.
