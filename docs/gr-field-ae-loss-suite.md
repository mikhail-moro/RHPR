# Continuous-AE loss hypotheses on two T4s

The completed continuous AE reconstructs brand much better than the RQ arm,
but title MSE increases from 0.001843 at initialization to 0.009174 after5000
updates. The catalogue-mean predictor achieves0.00028725. Uncertainty log
variance for title changes only from-0.5 to-0.639 under native Adagrad0.001.
These observations motivate objective/optimization controls before changing
the quantizer. They do not establish that task-gradient conflict causes failure.

Authorized follow-up:one private Kaggle session with two isolated GPU processes,
two trials each. Configuration:
`experiments/amazon_beauty_field_ae_loss_suite_2t4.json`.

|Trial|Intervention|Question|
|---|---|---|
|uw_lr_003|Adagrad0.03 for UW log variances only|Does faster UW adaptation recover fields?|
|uw_lr_01|Adagrad0.1 for UW log variances only|Is the same effect present at a larger rate?|
|fixed_normalized|Sum each raw loss divided by its catalogue constant-predictor error|Does explicit loss-scale calibration help?|
|title_only|Title MSE only; retain brand/categories/price/title inputs|Can the same network reconstruct titles when other tasks carry zero objective weight?|

All four use fresh seed42 initialization,identical native neural modules,
Adagrad0.001 for neural weights,FP32,batch2048,5000updates and evaluations at0,
every500updates and finally. No scheduler,clipping,early stopping,automatic
fallback,retries,codebooks or generator. The title-only arm has inactive other
output heads;their metrics are diagnostic,not evidence of failed training.
Fixed/title-only arms freeze unused UW parameters. Faster UW arms retain the
LibMTL formula and initialization;only the UW parameter-group LR changes.

Fixed normalizers are fitted once from observed catalogue metadata:brand
empirical-prior cross entropy,categories mean per-label Bernoulli entropy,
price Huber error to the catalogue median,and title MSE to the catalogue mean
vector. Divide by max(error,1e-6);record errors,weights and clamped fields.
These constants put losses in units of reference error;they do not guarantee
equal gradients or equal task importance. Price's median is a reference
predictor,not a claim about the exact Huber-optimal constant.

Frozen title arrays are mounted from completed
`carasik/rhpr-field-ae-title-5k`,source9792ff38ba9733fa15822e7ec0fe61d8b0af7e2f.
Pin bytes and sizes of its fields,schema,report and run manifest. The ordinary
phase0 data manifest also pins catalogue IDs/order. No repeated text encoding
or baseline fit. The completed original UW control supplies the comparison;
this is a one-seed diagnostic screen,not a replicated effectiveness study.

The parent process verifies inputs and prepares pinned GRID once. Workers
start via `python -m rhpr.gr.ae_suite`,inherit the installed offline wheel site
and embedded package,and each see exactly one assigned CUDA device. Requesting
two GPUs fails explicitly if unavailable. Separate CPU mode verifies process
orchestration locally;it is not a GPU performance test. Worker0 runs UW0.03 then
fixed normalization;worker1 runs UW0.1 then title-only. Models,optimizers,RNGs,
checkpoints and output directories are independent. Compare exact initial-state
hashes across successful trials. Cancellation terminates both subprocesses.

Artifacts:each `trials/<id>/` contains `autoencoder.pt`,raw catalogue arrays,
schema,input/latent embeddings,field curves,reconstruction curves,final report
and trial manifest. Checkpoints contain policy,normalizers,optimizer groups and
initial-state hash. Restore by building the same model,calling
`configure_objective(model,arrays,checkpoint['loss_policy'])`,then loading its
state dict. `workers/` records device hardware,status and full logs;
`suite-results.json` aggregates results/failures. Failed trials retain a stack
trace and do not erase successful siblings;the overall suite fails if any
trial fails or initial hashes differ. No silent restart.

Compare raw errors,brand accuracy,category F1,price MAE,title cosine/MSE relative
to constant predictors,learning curves,latent norms and effective task weights.
Weighted total losses differ in meaning across arms and should not be ranked.
A successful title-only arm would show available capacity under that objective;
it would not identify a unique multitask failure mechanism or establish better
SID recommendation quality. No recommendation validation/test is performed.
