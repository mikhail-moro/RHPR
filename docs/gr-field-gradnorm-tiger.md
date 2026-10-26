# Frozen GradNorm field-AE indices → TIGER

User requested this downstream check after the nine-intervention proxy study.
The research focus is field-level item encoders;title-only is a control for
useful textual semantics,not a replacement research direction. Configuration:
`experiments/amazon_beauty_field_gradnorm_tiger_2t4.json`.

One private two-T4 session,exactly two new TIGER fits. Reuse the completed
`rhpr-ae-ablation-probes` source d13dd8aada6c1f60ce36409147c261af5a09b608,
trials gradnorm_s42/43,checkpoint5000. Field inputs are brand/categories/price
plus frozen Sentence-T5 title. Native AE latent64,GradNorm alpha1.5/weight
Adagrad LR0.025,normalized field objectives. The selected indices were already
fitted with sequential raw-latent RQ-KMeans3×256,n_init1,max300,tol1e-4,
seed+level,CPU thread1. Reuse saved codes and books;do not run an encoder or
refit/re-export nearest-centroid assignments.

Pin parent manifest/provenance/suite and each candidate code array,books,
KMeans report and AE report. Validate catalogue order,source trial,variant,
seed,step5000,objective settings,checkpoint hash,quantizer spec and shapes.
Cross-check saved semantic statistics against the codes. Copy books,reports
and source hashes into the generator output;append the existing ordinal suffix
without repairs. Parent max collision group is3for both seeds.

Keep the completed control's generator and evaluation exactly:

- Shared generator/data seed42,early GRID4encoder+4decoder,d128,FF1024,
  d_kv64,6heads,dropout0.1,noSEP,user bins2000,4977920parameters.
- FP16 training,Adam LR0.001,wd0.0001,clip1,batch256/physical128.
- All131413noninitial training positions once per epoch,latest20context,
  fixed seed+epoch permutation. Twenty complete epochs,validation10/20,
  selection by validation item NDCG@10,one final FP32free-beam20test.
- Assert shared initial neural hash and training-case hash before workers
  start,and initial neural hash inside each worker. Save best/last,curves,
  optimizer/scaler/RNG,per-user test predictions and provenance.

Reuse completed`rhpr-ae-rqkmeans-tiger` source2ba30f85977472cafefff621542f2d320972c2ca
controls:fixed-normalized AE3500→RQ-KMeans,test NDCG10approximately0.0200315/
0.0191125. No repeated control training. Its AE budget is3500,whereas the new
GradNorm AE is5000;this is an operational candidate comparison,not an isolated
same-budget causal loss ablation. A matched5000-step fixed-normalized generator
control could be a later separate study,not part of this authorization.

Two index seeds with shared generator seed are not independent generator-seed
repeats. The candidates were selected exploratorily using TRAIN proxies;this
run checks downstream transfer under the existing split,not a new-dataset
confirmatory study. No AE/quantizer fit,generator tuning,title-only generator,
extra controls,automatic retries or Telegram.
