# ML1M Last-8: finish windows 16 and 64, exclude 256

User stopped `carasik/rhpr-ml1m-last8-windows/2` and requested the remaining
windows on 2026-11-18 (Europe/Moscow). The five-window comparison is now
8/16/32/64/128, generator/index seed42. No further256 training, extra seeds,
ML10M, HPO, or notifications are authorized in this step.

The stopped run actually completed8/32/128, including selected-checkpoint TEST.
They remain on the original kernel and are downloaded to
`/tmp/ml1m-last8-stopped-v2`. Do not rerun them. Windows16/64 never started:
they were queued after256 in worker0. Consequently no resume or transfer of
generator weights is needed for the two missing fits.

| Window | VAL NDCG@10 | TEST NDCG@10 | Best / stop epoch |
|---|---:|---:|---:|
| 8 | 0.05199936 | 0.04578558 | 35 / 60 |
| 32 | 0.04988699 | 0.04138927 | 40 / 60 |
| 128 | 0.06705911 | 0.05449579 | 35 / 70 |

All three stopped by the registered EMA criterion, with no AMP skips.
Window128 is19.02% higher than8 on TEST in this single-seed comparison, but
the relation is nonmonotonic. This is not a multi-seed robustness result.
The original target/timestamp-tie caveats still apply. The256 fit has32
completed epochs, no final selected-checkpoint TEST, and is excluded.

CPU audit: `scripts/research/audit_ml1m_last8_partial.py`; machine-readable
`gr-ml1m-last8-partial-results.json`. It checks shared raw cases/index/model
initialization, raw-VAL checkpoint selection, completed epochs/update counts,
replays the six selected VAL/TEST files (724800paths), and checks all six
completed-model checkpoints for finite weights and continuation-state fields.
No new model inference or GPU work is performed by this audit.

New registration: `experiments/ml1m_last8_remaining_2t4.json`, exactly TWO
fits64/16 on two isolated T4 workers. Intended new private/offline kernel:
`carasik/rhpr-ml1m-last8-remaining`. Keeping a separate slug preserves the
stopped run's outputs. Model, training, microbatches, LR/EMA/VAL/TEST policy,
all users/items/targets and source/runtime datasets are unchanged. The
preparation array still stores256 past item slots to preserve the original
case hashes; the two models only receive their respective64/16-item suffixes.

Recompute ST5/RQ-KMeans once from the original ML1M metadata; no new dataset,
kernel dependency, fitted-vector upload or weights transfer. Before any
generator fit, the regenerated SID-file SHA256 must equal the completed
models' `80422aa36fbcabbd0c1133a0695547513c1a795f908c72b4ed412e9cc2c2b672`.
Compile remains Inductor with CUDA Graphs disabled. The pre-fit compiled
gradient-accumulation probe remains in place; it succeeded in the stopped run.

Quota before this continuation:20.25GPUhours remaining. Keep the original
8h session ceiling,20,000s per-fit budget,40minimum/120maximum epochs,
VALevery5/EMApatience20. Caps are censored, not convergence. Only windows16/64
are launched; combine their outcomes with the three already completed models
after the run. There is no need to wait behind256 or retrain128.
