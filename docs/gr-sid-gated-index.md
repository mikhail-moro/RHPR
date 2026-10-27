# Original/ALS conditional index: matched20epoch experiment

User authorized2026-10-27:20epochs,validation every5,with a special branch token
at the beginning. Exactly two fresh models in one private2×T4session:

1. `original_with_branch`:original RQ-KMeans SIDmapping for every example,
   conditioned on the same SHORT/LONGfeature used by the experiment.
2. `gated_original_als`:original mapping when the available prefix has fewer
   than10events,exact previous frozen item-ALSswap mapping otherwise.

The threshold10was fixed by the user. This is one shared generator per model,
not an ensemble or a learned router. Earlier cached routing of two separately
trained models gave+0.53%at20with an interval including0and negative earlier
checks;it did not measure a jointly trained model or guarantee no quality loss.
See local`/workspace/artifacts/sid-context-screen/analysis/two-index-proposal.md`.

## Known branch prefix and causal routing

Each supervised prefix H independently determines `g = observed_length(H)>=10`.
Both its complete input history AND its next-item labels are encoded through
the same selected table. The user's eventual full TRAINhistory length is never
used to choose the branch for early training examples. At validation/inference,
the same available-context rule chooses the table and the item decoder.

Inputs remain left-padded to20items,each with its complete4token SIDincluding
the original collision suffix. The batch transports a leading0/1branch column.
The model adapter strips it before applying GRID's repeating hierarchy offsets.
It then prepends a dedicated learned SHORTorLONGembedding before the native
user token and SIDembeddings. Encoder layout is `[branch,user,20*4SIDpositions]`,
82positions including PAD. A plain raw code cannot be prepended through the
unmodified native adapter because that would shift every hierarchy offset.

The two branch embeddings add256parameters at d128:4978176total,versus4977920
previously. They are separate from the shared SIDembedding vocabulary. Native
user hashing,item embedding preparation,decoder BOS,four prediction heads,
decoder KVcache and FP32beam arithmetic are preserved. Decoder cross-attention
observes the branch through encoder outputs;the branch is known and is not an
extra predicted target. One forward/beam search handles mixed-branch batches.

Control and switched models use precisely the same two-token feature and model
architecture. Only the LONGitem-to-SID mapping differs. Both trees have the
same complete path set;only480of12101item assignments differ. This comparison
controls for adding an explicit length feature and for the extra encoder token.

`src/rhpr/gr/conditional_index.py` owns routing,batch labels,item decoding and
the instance-level GRIDencoder adapter. The ordinary trainer and generator suite
use optional index hooks;their ordinary single-table path stays available.
The coordinator verifies both CPUneural initialization and shared TRAINcases.
The branch-token adapter is installed before optimizer creation and checkpoint
loading,so its parameters are trained and saved normally.

## Immutable inputs and matched training

Reuse existing private input`carasik/rhpr-sid-context-screen`,manifest
`8878eb9f8b3f5f3174a763e501d8387f1c85b298651e6485b6638b9c87782209`.
No new input upload or encoder/ALS/graph/quantizer fitting. Select only original
and exact prior item-ALSarrays;context20/context5candidates are not trained here.
Original suffix ownership,full path sets and catalogue identities stay pinned.

`experiments/amazon_beauty_sid_gated_index_2t4.json` preserves seed42,Beauty
12101items/22363users,all131413TRAINprefixes per epoch,latest20context,exact
same case/order hashes,GRID4+4/d128/FF1024/dkv64/6heads/dropout0.1/userbins2000.
Adam0.001/wd0.0001/clip1,batch256/physical128,FP16training,FP32batch32freebeam20
evaluation,no sampled catalogue or seen-item filtering.

Each epoch contains98608SHORTand32805LONGexamples. The shared case hash remains
`6f7ba06c70a6a218d4e620ac77ce41142baf762f7e62902c9fc9b6f7b45d847f`.
The branch assignment hash/int64is
`5a0670622fababf0a56e6ec94eb3edb35d819b0b6d76aa14f5e77fe6db50a76a`.
Shared conditioned neural initialization hash:
`b5700937900c8bfd4f0665ca20937da4727a4b2e7c32346835915338eab81b6d`.
The factory separately checks the original base initialization hash5f4a24fa…
before adding the branch embedding. Extra parameters intentionally change the
full initialization hash from the historical unconditioned models'.

Two isolated workers,one model per T4,one CPUthread each. Exactly20complete
epochs,2628260consumed examples/10280planned updates per model,with AMPskips
reported separately. Validation5/10/15/20only,no early stopping before20.
Primary endpoint NDCG@10at20;best-validation checkpoint is separate. Test
construction/evaluation,extra candidates/seeds,threshold tuning,automatic
extensions/retries and Telegram are disabled. Max7200seconds per model is a
failure guard,not an intended shorter training budget.

## Branch-aware evaluation and retained states

Generated paths decode under that row's chosen index. A path can name different
items across the two mappings,so evaluating all rows under the LONGtable would
be wrong. Invalid/duplicate paths retain original beam ranks as before. Metrics
combine exact per-branch item-level counts and report branch counts/NDCG10.
Validation preserves original user/batch order;there is no separate model pass
per branch. Each of eight saved validation files includes `branch_ids` and
`observed_context_lengths` as well as full beams,scores,users and item targets.

Save best/full-last checkpoints and all training curves. Last includes learned
branch embeddings,Adam,AMP,all RNGstates,epoch/update counters and a digest of
BOTHfrozen mapping matrices. `conditional-index.json` records mapping hashes,
branch semantics,threshold and changed-item count. The control's two matrices
are identical. Future inference/replay must load both mappings and install this
adapter before loading model state;the legacy single-index evaluator is not a
valid decoder for these outputs. The20epoch last states are preserved;no extra
intermediate5/10/15weight snapshots are promised.

## Checks and submission

Six new tests passed,including an actual pinned-GRIDsubprocess. Coverage includes
early prefixes of long users,context/target table consistency,padding,branch-aware
item decoding and duplicates,same-prefix-path invariants,exact preservation of
the native item/user embedding preparation,nonzero gradients for BOTHbranch
embeddings,teacher-forced loss,cached beam generation,two tiny actual fits,
identical initial parameters,checkpoint restore and two-table checkpoint hashes.
The native test refuses access to test targets. Five previous ALSscreen tests
(including native training/cadence),three previous long-run guards and two actual
context-input import tests also passed:16tests total.

Preflight on the real catalogue imported both exact mapping assets,verified the
base/conditioned initialization and case hashes,counted branches,and measured
4978176parameters for both models. Receipt:
`/workspace/artifacts/sid-gated-index/preflight.json`.
Local bundle metadata checked:private/offline/T4/two existing dataset mounts.
No new dependencies are needed.

Before submission,11.24GPUhours remained and the previous context screen was
COMPLETE. Expected about35–45minutes execution after GPUassignment,queue extra.
Kernel:`carasik/rhpr-gated-sid-tiger-screen`. Exactly one submission;no Telegram.
Do not launch a replacement while this version is queued/running or poll training
repeatedly.

## Submission receipt

GitHubActions dispatch returned HTTP401and was not accepted;no workflow run was
created. The authenticated GitHubconnector remained available for repository
access. Submitted once with the official KaggleCLI directly,creating version1.
Status confirmed RUNNING. This is not a replacement for an already accepted job.

Executable sourcecommit`45d7dab1d35a55821585e17423f32eb197a5cce3`,treeba42770e….
To exclude unrelated local paused edits,the176embedded source/resource files
were independently matched to the recursive GitHubtree by GitblobSHA1. The
one differing unrelated file was substituted only inside the in-memory package;
the user's local worktree was preserved. Runner builder and experiment JSON were
also matched to the commit. This avoids labeling a dirty local package as the
published source revision.

Kaggle's returned source was parsed without executing it. Its exact experiment
JSON and embedded package match the submitted artifacts. Package SHA256:
`c358125a17c2a447d96aa56f142c1718771285861f08ec59ad538fdcc4592f85`.
Metadata confirms version1/private/GPU/T4/internetoff/TPUoff and the two existing
dataset mounts. Receipts:
`/workspace/artifacts/sid-gated-index/source-audit.json` and `submission.json`.
No new GPU results are claimed at submission time. Wait for user completion or
status request;do not resubmit or continuously poll this session.
