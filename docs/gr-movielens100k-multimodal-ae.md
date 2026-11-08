# MovieLens100K cumulative modalities with continuous AE warm-up

Repeat the completed four-stage multimodal ladder with the two-stage quantizer
recipe tested by the seed42 text-control experiment. Twelve fresh independent
fits: seeds 42/101/202 × text, text+poster, text+poster+description,
text+poster+description+people. Each fit gets its own AE, codebooks and generator.
The completed text seed42 fit is retained as prior evidence and repeated in this
uniform run; it is not silently imported or counted as a new independent seed.

Config: `experiments/movielens100k_multimodal_ae_2t4.json`.
Private target: `carasik/rhpr-ml100k-multimodal-ae`.

## Exact comparison

Input vectors are reused from completed private `carasik/rhpr-ml100k-multimodal`
v2 through Kaggle `kernel_sources`; no uploads, new embeddings or Dataset.
The parent executable, run manifest, features and catalogue-order hashes are
checked before any fit. This preserves Sentence-T5 metadata text (no appended
people), poster ViT CLS, MPNET **description only**, shared DeepWalk people
features and observation masks. One missing description stays masked.

Same 1,273 films/943 users and timestamp-group train/validation/test cases,
85,470 strict-past targets per full epoch, maximum history20, full catalogue,
free beam20, no seen-item mask. Same per-seed initial neural weights for both
headed quantizer and TIGER, pinned against the completed direct-training ladder.
All modality heads remain instantiated in all stages; inactive inputs are zero
and excluded from loss, preserving the architecture and initialization.

## Quantizer recipe

1. Continuous AE through modality encoder heads/shared encoder/shared decoder/
   modality decoder heads, bypassing all quantization. Native Adagrad0.001,
   FP32, fixed catalogue mean/scalar-RMS normalization. Sum observed-item,
   dimension-mean MSEs of active modalities; no UW or new loss weighting.
2. Full-catalogue eval latent spectrum every500 updates. Same prior spectral
   plateau heuristic as the successful text control: minimum2000 updates,
   four consecutive eligible probes with equal active-mode count at aggregate
   nominal24bit residual rate. Earliest possible stop3500, hard cap5000.
   Each stage/seed stops independently. This criterion does not establish an
   optimal recommendation representation or reconstruction convergence.
3. Retain all neural weights and BatchNorm buffers. Native GRID residual KMeans
   initializes untouched codebooks from trained eval latents. Keep neural
   modules in eval during initialization and first reconstruction batch, then
   jointly train using fresh native Adagrad accumulators. Same RQ3×256/latent64,
   3000 Adagrad steps; separate initialization SGD does not advance that counter.
   AE Torch RNG is isolated from subsequent native initialization randomness.

Assert untouched books during AE, exact neural/BN transfer, fresh RQ accumulators
and unchanged eval latents for the first native initialization. Quantizer initial
hash is selected by each trial's seed; never compare seed101/202 to seed42.
No architectural changes to the AE/RQ implementation used in the text pilot.

## Generator and artifacts

Fresh TIGER4+4/d_model128, same effective/physical batch256/128, FP16 train and
FP32 eval, 20 full epochs, validation5/10/15/20. Test once using the checkpoint
selected by validation NDCG@10. All85,470cases/epoch and epoch permutations
match the direct ladder within each pipeline seed; no warm-started generators.

Keep per-fit AE checkpoint/optimizer/latent vectors/spectrum and head-MSE curves,
native RQ checkpoint, per-head/Q losses, occupancy/collisions, final books/SIDs,
TIGER checkpoints and all validation/test predictions. Compare against both the
direct headed ladder and the plain RQ text reference. Assess per-seed paired
results and validation/test separately. More occupied books are a diagnostic,
not a proven quality proxy; AE extra compute and initialization effects remain
coupled, and this experiment does not isolate which mechanism matters.

## Execution budget

One private session with2×T4, one isolated worker per GPU, six fits per worker.
Expected body roughly2–2.5hours plus environment setup/queue, based on the prior
117.5min ladder and ~70sec extra AE in the text pilot; multimodal AE can cost more.
No automatic training retries, tuning, extra seeds, TPU fallback or Telegram.
Old direct-ladder and text-pilot notebooks remain intact for comparison.
