# Compiled Recent20 LR0.003→0.001 / EMA: completed

Private carasik/rhpr-kindle-lr003-floor001-ema/version1, source
6b9e4c6277b80f3fe8b40ae66f532bb6f5a4add1. One fresh seed42 fit on the pinned
Kindle item index, train512/FP16/SDPA/compiled-forward and VAL256/FP32. No TEST.

60epochs,10800updates,0AMP skips. Best raw VAL NDCG@10=0.05133097 at50.
EMA reference last improved at55; its stale counter was only5epochs at60,
so EMA did not trigger an early exit. Reached configured epoch cap60.

Session17min00s; fit16min19s including validation/checkpoints. First training
epoch92.33s includes lazy compilation of full/tail shapes; later median13.88s.
The previous eager LR0.003→0.0003 fit needed24min54s:1.525x per-fit speedup.
Its best VAL was0.05025496: new+2.14%. Versus the fixed0.001 control's best
within60(+3.92%) and within120(−8.42%), this remains a budget-limited pilot.
LR floor, compiler/dropout numerics, VAL batch and stopping all change together;
no isolated LR-floor effect or TEST gain is established by one seed.

Independently replayed all12VAL prediction files: NDCG/Recall5/10 match1e-12.
Raw-best checkpoint, initializer/SID/case/source pins, all180updates/epoch,
every first/last/next LR and causal EMA/reference/improvement state verified.
No TEST predictions created or evaluated. Full best/last weights remain in the
private kernel; local prediction/curve evidence is retained in
/workspace/artifacts/kindle-pilot/lr-ema-results.
