# Frozen Pair32 / Recent32 prefix-mask probe

User authorized2026-11-15. No training,tokenizer fitting,model selection or new
seeds. Use only saved raw-VAL-selected epoch55 Pair32 and Recent32,seed42.
One private/offline twoT4 kernel,one frozen model/worker. Exactly8inferences:
two models ×VAL/TEST ×full/masked_old,all958users/split,FP32/SDPA/beam20/batch64,
no seen-item filter. Both datasets and TEST have already been examined;this is
an exploratory mechanism probe,not an independent generalization test.

Pair32 input96SID positions:8old pair summaries ×4codes,then16recent singles
×4codes. Mask the first32SID positions. Recent32 input128SID positions:16old
singles ×4codes,then the SAME16recent singles ×4codes. Mask the first64SID
positions. In both models sequence length,USER token atposition0,and ALL64
recent SID positions remain exactly where they were. Models therefore retain
97/129encoder positions,respectively;no compaction or relative-bias retuning.

Set only old encoded positions to−1,which the existing InputIndex4 translates
into a zero encoder/cross-attention mask. The replacement lookup value0is
masked,not an active dummy SID. USER remains visible. Masked queries can still
be computed,but are never visible as keys to USER/recent queries or the decoder.
Native real-checkpoint test changes every masked ID and obtains BIT-IDENTICAL
visible encoder states AND generated ranked SIDs/scores. Recent values and
attention positions are unchanged;no claim that the USER representation is
unchanged after removing its access to old keys.

Before evaluating masks,replay each full baseline against its original saved
VAL/TEST predictions:require all ranked SIDs identical;record score differences.
Stop rather than reinterpret a mismatched baseline. Verify loaded model state
unchanged after all evaluations,all parameters require_grad=False,no gradients.
No optimizer or training functions are used.16parent artifacts byte-pinned,
including both best weights,indexes and four reference prediction files.

Inspect aggregate NDCG/Recall5/10,paired gains/losses and original lost hits.
Use the previously fixed recent-context popularity feature:mean log(1+TRAIN
target count) over latest16books,VAL cutoff2.300820444821813 reused onTEST.
Inspect whether the two original rank1→outside-beam20 losses recover. Compare
Pair32 mask effect against the same intervention in Recent32;the fitted models
have different weights and position geometry,so even a difference in response
does not isolate compression as the sole cause.

Positive rescue would indicate harmful reliance on the old input under THESE
weights. A degradation would indicate that deleting this input is not a remedy
for this checkpoint;it would not prove that pair representations are optimal.
Masking alters the inference distribution and is not equivalent to training a
Recent16 model. Do not select checkpoints or claim improved deployed quality
from these already inspected targets. No extra mask variants,HPO or retraining.

Preflight:1112tests pass,152optional skips;8CPU original VAL examples/model
exactly replay saved GPU rankings;16parent pins verified;real no-leak mask test
passes for both models. See `gr-kindle-prefix-mask-preflight.json`.

Config:`experiments/kindle_prefix_mask_2t4.json`. Parent Pair32:
`carasik/rhpr-kindle-long-pairs/1`,source33754d70acc0;control:
`carasik/rhpr-kindle-capacity-pilot/1`,source6d2919a15589. Same3input datasets
and offline HF5.22.1overlay. No ST5/RQ work or checkpoint copies in outputs.
Save all8prediction files,case/checkpoint/model hashes,metrics and original
replay checks. Private kernel:`carasik/rhpr-kindle-prefix-mask`,no notifications.
Inference should take under a minute after environment/model loading;queue and
startup can dominate. Prelaunch quota38.40GPU hours remaining.
