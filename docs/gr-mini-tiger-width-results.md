# Mini-TIGER width calibration results

All48freshfits completed in139.76min on2T4;10epochs,2960updates,2VALchecks,151166TRAINexamples/epoch and0AMPskips each. Finished2026-11-06T10:00:44.373790UTC (13:00Moscow). Private carasik/rhpr-ml1m-mini-width/1. No TEST or new quantizer training. Executable d80787c82e1cb93c016359df7e069ea0b8753207.

## Fixed epoch10 comparison

Both generator seeds averaged before ranking against saved full-TIGER fixedepoch30 VAL NDCG@10. Historical narrow baseline replayed on the same12indexes and first10epochs.

| Model | Parameters | Train sec/epoch | Speed vs narrow | Overall rho (12 indexes) | SUM rho (3 indexes) | Full-target regret |
|---|---:|---:|---:|---:|---:|---:|
| narrow | 1,375,040 | 38.85 | 1.00x | 0.902 | 1.0 | 0.00% |
| ffn_small | 1,112,896 | 36.37 | 1.07x | 0.385 | 0.5 | 9.89% |
| attention_small | 719,552 | 30.19 | 1.29x | -0.126 | 1.0 | 6.66% |

The smaller FFN increases its own NDCG on several indexes but worsens agreement with the full model. A stronger small recommender is not automatically a better tokenizer proxy. FFN reduction saves only6.4% training time; further attention reduction saves22.3% vs narrow. Historical full epoch57.91s gives speedups1.49x/1.59x/1.92x respectively. Two-seed averaging doubles per-index compute. Do not compare mixed10/20epoch narrow total-fit means with the new10epochfits.

## Seed and group stability

| Model | Overall rho seed31415 / seed27182 | Within index seed42 /101 /202 | SUM rho seed31415 /27182 |
|---|---|---|---|
| narrow | 0.706 / 0.790 | 0.8 / -0.2 / 0.8 | 1.0 / 1.0 |
| ffn_small | 0.385 / 0.364 | -1.0 / -0.2 / -0.4 | 0.5 / 0.5 |
| attention_small | -0.399 / 0.007 | -1.0 / 1.0 / -0.4 | 0.5 / 1.0 |

Narrow and attention_small two-seed means order SUM indexes202>42>101,matching fullTIGER. Narrow does this for both individual generator seeds. attention_small seed31415 orders42>202>101,its other seed202>42>101. FFNsmall picks42>202>101 with both seeds; this selects a full target9.89%below the best.

Across all indexes narrow selects sum_seed202 (fullNDCG.0719875),FFNsmall selects sum_seed42 (.0648702),attention_small selects uw_seed202 (.0671957). The attention_small result on only three SUM indexes does not establish a reliable general proxy.

## Uncertainty

Conditional shared-user bootstrap5000draws,identical resampled users for mini/full across allindexes:

| Model | Overall rho95% interval | SUM rho95% interval |
|---|---|---|
| narrow | [0.22377622377622378, 0.8811188811188811] | [0.5, 1.0] |
| ffn_small | [-0.22377622377622378, 0.5944055944055944] | [-0.5, 1.0] |
| attention_small | [-0.48951048951048953, 0.2937062937062937] | [-0.5, 1.0] |

These percentile ranges are conditional on fitted models/indexes,not confidence intervals for generalization to new tokenizers. They exclude training randomness,index selection and target-item dependence. User resampling can reorder near-tied full scores; a percentile interval need not contain the original plug-in correlation. Only12relatedindexes and3SUMindexes; fullpipeline seeds jointly change tokenizer and fullgenerator. Architecture screen follows inspection of earlier narrowresults. The pooled narrowrho.902coexists with within-seed101rho-.2; do not replace group-level limitations with the pooled number.

## Verification and artifacts

Independent replay decoded156savedVALprediction files:96newmini,48historicalnarrow,12full;verifieduser/targetarrays,SID andreferencehashes,initializationpins,modelshapes,parameters,budgets,AMPskips,NDCG/Recall5/10,invalid/duplicatecounts and correlations. All48fits and worker result sets match. Fullregression1012tests,91optional skips,0fail. No inference/training was performed for this analysis.

Replay: scripts/research/analyze_mini_width.py. JSON: docs/gr-mini-tiger-width-results.json. Local graph/evidence: /tmp/mini-width-analysis/width-calibration.png and /tmp/mini-width-evidence.zip. Evidence includes raw numerical outputs,predictionfiles,TRAIN/VALcases,selected historical narrow/full references and analysis; generator checkpoints stay on privateKaggle.

Conclusion: retain original narrow4+4,d64,FF512,H6,kv32,epoch10,two-seed mean as a provisional proxy within this index family. Neither width reduction is supported as a general replacement by this calibration. attention_small remains an exploratory SUM-only candidate,requiring additional independent indexes before trusting selection. No new run authorized or pending.
