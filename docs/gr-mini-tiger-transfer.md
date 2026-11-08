# Fixed narrow mini-TIGER transfer check

The pilot's narrow4+4 generator at epoch10had pooled rho0.833,but group-specific
correlations0.8/−0.2 and wide conditional user uncertainty. This follow-up tests
the chosen proxy on different existing indexes before any further shrinking or
joint optimization. The pilot is not resumed or completed by this run.

## Fixed protocol before observing proxy results

- Four saved SUM/UW/GradNorm/PCGrad indexes at tokenizer seed202,from private
  `carasik/rhpr-ml1m-clean-mtl-replicates/1`. All full-reference artifacts and
  parent manifest/plan are byte-pinned. No data upload,new quantizer training,
  content encoding,or full-TIGER training.
- One architecture:narrow4+4,d_model64,d_ff512,6heads,d_kv32,dropout0.1;
  1,375,040parameters. Same architecture/seed initialization pins as the pilot.
- Two fresh generator seeds31415/27182 crossed with all four indexes:8fits.
  Index assignments remain unchanged. Each worker runs four fits on its T4.
- Exactly10epochs,VALevery5,min_epochs10,patience_epochs10,no scheduler/HPO.
  Primary proxy score is fixed epoch10VAL NDCG@10,averaged across the two
  generator seeds. Per-seed results also reported. Epoch5is diagnostic only;
  the internally VAL-selected checkpoint is not used to choose the proxy score.
- Same151166TRAINcases,2786VALusers,2874items,window20,raw clean timestamp
  protocol,FP16training/FP32evaluation,nativeHF5.22.1SDPA,512/512batch,
  AdamLR.001/wd.0001,clip1,2000userhashbins,original3x256SID+suffix,freebeam20.
- Primary reference is each existing full TIGER's epoch30VAL NDCG@10;
  best-of30is secondary only. Report Spearman,top2overlap,and target NDCG regret
  for the proxy-selected index. Preserve raw per-user predictions for replay
  and conditional user uncertainty. TEST is not evaluated.
- Compare early full checkpoints as descriptive baselines:they share the
  target's training trajectory,whereas the proxy uses independent generator seeds.

## Scope and budget

The four candidate indexes were not used to select the proxy architecture/epoch.
Their full-model outcomes were already part of earlier MTL research,so this is
not an untouched dataset. Only four related indexes and one full-generator seed
per index (202,coupled to original tokenizer seed) exist. This is a limited
transfer check,not a universal correlation or significance claim. Do not pool
the pilot and transfer candidates and call the result independent validation.

Measured pilot speed predicts roughly30minutes wall on2T4. Internal deadline
3000seconds prevents starting more fits;server timeout60minutes is the hard cap.
No automatic retries,epoch extension,additional seeds/shapes,hardware fallback,
new dataset publication,or Telegram notifications. User authorized this bounded
follow-up2026-11-06. Incomplete matrices fail visibly and retain finished trials.

Configuration:`experiments/ml1m_mini_tiger_transfer_2t4.json`.
Private kernel:`carasik/rhpr-ml1m-mini-transfer`.
Readiness checks and submission provenance are saved alongside this protocol.
