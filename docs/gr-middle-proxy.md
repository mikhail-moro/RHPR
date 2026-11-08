# Intermediate proxy architecture screen

Authorized2026-11-07: two proposed architectures on six existing frozen indexes,
two existing mini generator seeds31415/27182,10epochs/VAL5. Exactly24fresh fits.
No baseline retraining,full-model/tokenizer/content fitting,TEST,extra seeds,
HPO,retries,Colab fallback,new datasets or notifications.

| Shape | Encoder/decoder layers | d_model | FFN | Heads | d_kv |
|---|---|---:|---:|---:|---:|
| middle_width |4+4|96|768|6|48|
| half_depth |2+2|128|1024|6|64|

Both are intermediate in capacity between historical narrow1.375M and full4.978M.
Exact parameter counts and all24 initializer checks are recorded in preflight.
One native tiny10epoch/twoVAL training probe per shape/seed(4probes).

Frozen SUM/GradNorm indexes x42/101/202. Same raw-clean ML1M151166TRAINwindows,
2786VALusers,2874items,history20. Same GRID/nativeHF5.22.1 SDPA,FP16train/FP32eval,
512physical/effectivebatch,Adam.001/wd.0001,dropout.1,unmaskedbeam20,userbins2000,
SID3x256+suffix. Per-shape/seed initialization matches across all six indexes;
different architectures have different initial parameter shapes and values.

Primary: fixed proxy epoch10 two-seed mean versus the already completed full-TIGER
selected VAL scores under min60/max100/patience20,seed161803. Secondary epoch5 and
individual seeds;report regret,shortlist retention,method/index-seed groups and
time. Historical narrow10 and early-full selected checkpoints reused as controls.
Proxy checkpoint selection does not substitute best-of10 for primary fixed10.
No adaptive architecture/epoch choice or additional fits inside this run.

Full plateau scores,early-full scores,best/stop epochs,SID hashes and source
generator/selection/curve/best-prediction hashes are embedded as a small pinned
reference object. Its SHA256 is fixed before training. These scalar references
were independently replayed previously;no new full-output parent mount/upload
is needed. Same two completed cleanMTL parent sources and three existing datasets.

This is architecture development on six related,already examined indexes and
the same validation cohort. It is not held-out confirmation or a new quantizer
family/dataset transfer. A favorable result needs fresh candidate validation.
Two-candidate seed groups are descriptive. An epoch or architecture picked after
viewing results must be labeled post-hoc. No numerical rho threshold automatically
authorizes another run.

Private carasik/rhpr-ml1m-middle-proxy,2T4,one worker per mini seed,12fits/worker.
Expectedroughly1.5–2h session,not measured;server150min/internal8400s/perfit1200s.
GPUquota40.79h observed before submission. Missing fits fail the suite;no silent
partial calibration,precision change or shortened matrix. All predictions,losses,
curves,weights,case/index/initialization pins and config/source revisions saved.
The bundle-size guard requires source<1MiB;global-zlib packaging preserves source
bytes and configuration values. Config experiments/ml1m_middle_proxy_2t4.json.

## Completed collection

All24 fits completed in86.18min,no failures,no extraGPU runs. Primary
epoch10 rho is−.20/−.60 for middle_width/half_depth versus reused narrow+.4286.
Both improve their own NDCG but do not improve proxy ranking transfer here.
See [verified results](gr-middle-proxy-results.md) for costs,selection regret,
secondary checks and limitations;no portable proxy recipe established.
