# Identical RQ-KMeans index, different generator seeds

Authorized2026-10-28:two fresh20epoch TIGERfits in one private2T4session.
Config:`experiments/amazon_beauty_sid_seed_oracle_2t4.json`.

Both use the byte-identical original Sentence-T5/RQ-KMeans complete SIDtable
from the existing private context-screen input. No quantizer/graph fitting,
swaps,branch token or conditional index. Neural initialization/dropout seeds
42and43 differ;raw TRAINcases and epoch-order seed42 remain fixed. Thus this
isolates neural-training diversity at fixed data order,not all possible sources
of seed diversity. Native GRID4+4/d128/FF1024/6heads,4977920parameters,FP16train,
Adam1e-3/wd1e-4/clip1,batch256/physical128,FP32validation batch32/freebeam20.

Every131413TRAINprefixes appear once per epoch,2628260examples/model. Same
22363validation users/targets and catalogue12101. Latest20events observed.
No test,early stop,automatic retry,extension,extra seeds or Telegram. Validation
at5/10/15/20;fixed epoch20primary. Save all8beam files,best and full last states,
per-trial neural initialization hashes and one shared TRAINstream manifest.
The parent computes CPU initialization controls per trial and workers verify
their own seed hashes. Both original SIDfile hashes must agree before fitting.

After fitting,the CPU-only oracle independently checks saved users/targets and
scores each target against the same SIDtable,matching logged NDCG10. Per user,
choose the complete model ranking with highest NDCG10(equivalently minimum
target rank). No list merging or target promotion. Save`paired-oracle.json`,
`oracle-user-ranks.npz`and flat endpoint oracle metrics. Report absolute/relative
headroom versus the better fixed model,Recall10 and both/exclusive/missed hits.
Rank999means absent from saved20beams;all ranks>10score zero here.

Historical pure original/ALS20pair:0.03694259/0.03514883,oracle0.04729654,
headroom+28.03%. A similar gain here would show that a large oracle alone does
not establish benefits from different hierarchies. It does not prove the two
mechanisms identical or quantify a causal fraction. One pair,one validation
population;epochs are not independent replicates. GPU nondeterminism can change
the fresh seed42control despite byte-identical initial weights/data/order.

Expected execution about30–40minutes plus queue,one fit per T4. Exactly one
kernel`carasik/rhpr-sid-seed-oracle`;two existing private datasets,offline runtime.

## Submission

Submitted once via [GitHub Actions40589179704](https://github.com/mikhail-moro/RHPR/actions/runs/40589179704),
submit-only;no Telegram. Private [Kaggle notebook](https://www.kaggle.com/code/carasik/rhpr-sid-seed-oracle)
version1reports RUNNING. Executablecommit`178883fe87351ee81ea70830688782129370584d`.
Actualsource config and all177embedded Gitblob hashes independently match that
commit;packageSHA256`321c269addc30abe861ca4b3a8197e0e2031872d9d928b5c33279b961b80b4b0`.
OriginalSIDfileSHA256`b196711dceedce81a991d995aa5c86ccd35288d5cdfe3f7aedb174de86470dae`.

Preflight verifies131413cases and original file identity;CPU neural hashes42/43
are pinned in config. Standard suite893testsOK(76optional skips);targeted3OK,
including actual parent orchestration,two isolated CPUworkers,tiny training fits,
checkpointed budgets,independent oracle and mismatch rejection. No result yet.
Receipts:`/workspace/artifacts/sid-seed-oracle/`. No retries or extra fits.
