# Bounded Kindle mini-TIGER runtime profile

The user requested LR0.003→0.001 with EMA for intermediate experiments and an
investigation of throughput, batches, data loading, compilation and padding.
Profile first on the saved Recent20 seed42 cosine_003 best checkpoint, without
another full fit or TEST access. All updates are disposable timing warm-ups and
measurements; checkpoint/RNG are restored between cases, no trained weights saved.

One private two-T4 session: GPU0 measures FP16 train physical batches256/512/
1024/2048, each CPU-assembly+transfer versus fixed GPU-resident batch. Effective
batch remains512 for physical256/512 and changes for1024/2048; bigger-batch
throughput does not establish unchanged optimization. Eight warm-up and20timed
updates per case; measure current loss/finite updates, throughput, peak allocated/
reserved memory. Synchronized stage breakdown plus CUDA/CPU operator and Chrome
trace identify copy/forward/backward/Adam and actual SDPA kernels. Instrumented
stage timings are not uninstrumented throughput. GPU residency benchmarks one
cached batch, not random indexing of the entire training set.

GPU1 measures FP32 beam20 validation batches32/64/128/256 on all2625VAL cases,
two passes per size after warm-up, saves IDs and batch32 agreement. Same trained
weights and targets; changing batch can expose numerical differences near ties.
No TEST cases loaded or evaluated. nvidia-smi samples GPU/controller utilization
and used/total VRAM at200ms; utilization counters have coarse hardware sampling
and are not occupancy measurements. Match timestamps to individual cases.

After core timing, independently bounded120s cold torch.compile(model,
mode=reduce-overhead,fullgraph=False) probe on GPU0, then warm training throughput.
Timeout/failure is recorded separately, not hidden by fallback. Compile CPU and
child processes are killed together on timeout. A failed compiler does not
invalidate completed eager measurements. Worker deadlines420s; server cap15min.

The current data are RAM-resident, not streamed from disk during updates. Last
LR runs spend~94%of fit time in train,~5.6%in validation,~0.24%outside these.
92133training contexts average14.587/20valid items:27.07%padding. Every random
512-case batch in a checked shuffle includes a full20item context, so dynamic
trimming alone does not remove padding. All VAL contexts have20items. Sorted
length buckets could reduce work but alter SGD ordering. With current USER then
left-padding layout, naive trimming also changes T5 USER/item relative biases;
there is no claim of prediction-preserving bucketing. Raw training arrays occupy
16.2MB; preencoded model tensors for all cases would occupy~121.6MB on GPU,
excluding model, optimizer, activations and allocator/context overhead.

Reuse pinned runtime/dataset and saved model/index parent. No encoder/KMeans,
new indexes, HPO, long training, hardware fallback or automatic retries.
