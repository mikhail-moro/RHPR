# MovieLens100K: cumulative multimodal RQ-VAE heads

User-approved ladder: text → +poster → +plot → +people. Twelve fresh fits at
pipeline seeds42/101/202, four stages each, two independent T4 workers. Frozen
features are shared across seeds. Keep the exact previous1273films/943users,
strict timestamp-group split, embargo, case digests and latest20context items.

## Frozen inputs

- Text768D: exact previous Sentence-T5 stage5 vectors: original title INCLUDING
  year, genres, explicit year, runtime, TMDB keywords. No actor/director names.
- Poster768D: Zenodo15403972 `json/vit_cls.json`, joined by explicit original
  ML1M IDs through the verified ML100K bridge. All1273films observed. Do not use
  the2980-row MMRec NPY arrays: their row positions are not original film IDs.
- Plot768D: raw `description` from Zenodo `ml1m_text.tsv` (actually CSV), using
  pinned Sentence-T5. Other columns (genres/tags/reviews) are ignored. Encode
  nonoverlapping chunks of at most2048tokens, weighted by content token count,
  then L2-normalize the pooled vector. No silent truncation. Compute once on
  GPU0before independent fit workers. There are1272observed plots; keep the one
  missing film, zero its input and mask its reconstruction loss.
- People128D: weighted DeepWalk of the SAME catalogue, with TMDB person IDs.
  One shared64Dperson table; concatenate separate cast/director mean pools.
  Each random-walk step selects an incident film uniformly, then another
  participant uniformly. This implements W_ij=sum1/(n_film-1), preventing large
  casts from dominating by their number of pair edges. No user interactions.
  Gensim4.4.0SGNS,10walks/node,40tokens/walk,window5,negative5,sample0,
  three SGNS epochs,one CPU worker,seed/PYTHONHASHSEED42.16692people.

Source archives are checksum-verified. `prepare_movielens_modalities.py` aligns
all arrays to the source-ID order, saves provenance/observation masks and a
hashed resource manifest. `prepare_movie_people_deepwalk.py` reproduces the
frozen people vectors. CPU dependency: gensim==4.4.0 (not needed on Kaggle).

## Quantizer

Each modality has an MLP encoder d_m→256→128 and an MLP decoder512→256→d_m.
All four head pairs exist in all stages, preserving parameter shapes and neural
initialization within each seed. Inactive channels are not read: their128D
encoder blocks are zero, and their decoder losses are omitted. Missing observed
inputs are also zeroed AFTER encoder heads to suppress biases.

Concatenate the four128Dblocks. The shared native GRID trunk is512→768→256→128→64,
then native RQ3×256 with STE and beta0.25, then the mirrored64→128→256→768→512
shared decoder and modality reconstruction heads. Native codebook initialization,
Adagrad1e-3, FP32 and3000optimizer updates are retained. No AE warm-up or UW.

For each frozen channel, fit a catalogue mean vector and one scalar RMS
coordinate deviation on observed items. Center and divide by that scalar.
This preserves geometry up to translation/uniform scale and gives a zero
predictor dimension-mean MSE of1for every channel. The reconstruction objective
is exactly the sum of dimension-mean MSEs over observed items in each active
head. Fixed statistics are saved; no learned loss weights. Native quantization
loss remains added with its previous coefficient. More active heads therefore
increase total reconstruction weight, as requested; log each component.

Report per-head reconstruction and semantic code occupancy every500updates,
plus final losses/codebooks/checkpoints. A suffix overflow is a genuine recorded
failure; do not silently widen it or rerun with changed settings.

## Generator and comparison

The generator is unchanged: prior TIGER4+4/d128/4977920parameters,FP16training,
FP32evaluation,20completeepochs,validation5/10/15/20,one test at the best
validation NDCG@10checkpoint,freebeam20/no seen-item mask. Every fit has85470
training cases/epoch. Per-seed initial generator hashes are pinned; aggregate
initialization checks group by seed. All modalities use the same train order
within each pipeline seed. The new text-only headed quantizer is the primary
control; historical plain RQ-VAE stage5 is a separate architectural reference.

Estimated session2h–2h20plus queue/setup, based on previous50–55second generator
epochs. Quota inspection found25.79GPU-hours unreserved. One private session
`carasik/rhpr-ml100k-multimodal`, no tuning/retries/extra seeds/TPU/Telegram.

## Checks

Native tests cover equal baseline reconstruction scales, missing masks,
identical initial head weights across active-channel subsets, exact invariance
to inactive inputs, gradients to active heads only, real native RQ-VAE fitting,
checkpoint/code export and a twenty-epoch tiny temporal TIGER with test output.
Real pinned-encoder chunking smoke passed: an8400-token document was encoded
without truncation, pooled outputs have unit norm, and a missing plot stays
zero/masked.
