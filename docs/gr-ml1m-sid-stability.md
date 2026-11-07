# ML1M: tokenizer and generator stability

User-authorized 2026-11-08: assess seed variability on exactly the same
catalogue and cases. Six fixed SID indexes crossed with three fresh narrow
mini-TIGER generator seeds: 18 models, two T4 workers, nine models per worker.

- `meta_kmeans`: metadata Sentence-T5 vectors; the same sklearn Lloyd residual
  KMeans recipe as the earlier Beauty SID study, three levels ×256 with L2
  normalization of each incoming residual. Two seeds42/101.
- `meta_rqvae`: same Sentence-T5 vectors, 4-slot field encoder with only text
  active, text reconstruction, same 3500-step AE warm-up and 3000-step GRID
  RQ-VAE, three ×256. Two seeds42/101.
- `multimodal_rqvae`: exact frozen SUM controls from completed clean-ML1M
  seed42/101 runs (metadata, poster, plot), no additional MTL. Matching
  3500-step AE/3000-step RQ-VAE recipe.

All six get the same uniqueness suffix. KMeans vs RQ-VAE changes both the
quantizer and representation pipeline; text vs multimodal changes inputs and
active reconstruction heads. Neither comparison isolates a single operation.
Per-pipeline two token seeds and per-index three generator seeds do not give
precise estimates of variance or establish generality to other datasets.

Fixed ML1M: 2874 items, 151166 train windows, 2786 VAL users, history20.
TRAIN/VAL cases and catalogue/embedding hashes verified before fitting. TEST
is not used for fitting, checkpoint selection or evaluation. Standard resource
verification does read the test file bytes for its pre-existing checksum.
Frozen shared Sentence-T5/MPNet/poster arrays are reused from the parent output;
no text embedding recomputation and no multimodal-tokenizer refitting.

Generator: narrow 4 encoder +4 decoder,d_model64,d_ff512,6 heads,d_kv32,
1375040 parameters, user hash2000; native HF5.22.1 SDPA, FP16 train/FP32 eval,
Adam LR.001 wd.0001, batch512, free beam20. Fresh seed31415/27182/161803
controls initialization and data order. Every index is frozen before any
training. Compare initial parameter hashes across indexes for the same seed.

Stop policy: minimum30, maximum60 epochs, evaluation every10 and patience20
without strict VAL NDCG@10 improvement. The chosen checkpoint, all validation
predictions, validation CE and train curves, best/last states and timings are
saved. Mark epoch-cap models budget-censored rather than converged. Report
fixed30 and best-selected VAL separately, per-index generator sample SD and
per-tokenizer-seed variation, with shared-user paired bootstrap where possible.
CI and uncertainty should not treat three generator fits as three independent
tokenizer constructions. No TEST or extra seed/sweep is authorized.

Private kernel `carasik/rhpr-ml1m-sid-stability`, workflow submit-only; 540
minute server timeout, 27000s internal suite deadline after index preparation,
3500s per fit. Kaggle reports 39.34 GPU hours remaining. Prior measured mini
train38.82s/epoch; 18×30–60 epochs implies roughly 5.82–11.64 GPU hours of
training, plus evaluation/index staging; approximately 3–7 hours on 2T4.
These are estimates and counts can differ at early stopping.

Preflight: three generator parameter initialization pins tested on both
actual SID controls; all case and source hashes verified; native GRID text-only
AE3→RQ6 smoke on each tokenizer seed; real 2874-item RQ-KMeans one seed; native
GRID/HF5.22 CPU generator30epoch/3VAL tiny-data smoke. The fake smoke sizes and
precision do not replace a T4 integration test. Full regression 1040 tests,
91 optional skips, 0 failures; focused8 protocol/integration tests pass. Output
config `experiments/ml1m_tokenizer_generator_stability_2t4.json`.
