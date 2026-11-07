# Paired graph reconstruction loss pilot

User authorized weakening graph reconstruction after the three-seed full-training
comparison. One private two-T4 session, two fresh seed-42 pipelines:

| Trial | Text / poster / plot weights | People-graph weight |
|---|---|---:|
| `graph_weight_1_seed42` | 1 / 1 / 1 | 1 |
| `graph_weight_01_seed42` | 1 / 1 / 1 | 0.1 |

Config: `experiments/movielens100k_graph_loss_weight_2t4.json`.
Private target: `carasik/rhpr-ml100k-graph-loss-weight`.

## Controlled intervention

Weight the dimension-mean observed-item graph reconstruction MSE in both the
continuous AE and native RQ-VAE stages. Keep graph features active in both arms;
do not scale the raw vectors or change input normalization, heads, latent size,
codebooks, quantization/commitment loss coefficients or optimizers. Default and
explicit all-one weights retain the historical sum operation. Weights add no
trainable parameters and must preserve pinned neural initialization hashes.

Both arms use exactly 3,500 AE updates, with probes every 500 updates, followed
by 3,000 native RQ updates. Fixing AE duration prevents reconstruction weighting
from also choosing a different spectral stopping point. This matches the actual
old seed-42 graph warm-up length but is a new fixed-duration comparison; the old
adaptive run remains a historical reference. Preserve the AE transfer checks,
untouched codebooks, native residual KMeans initialization and fresh Adagrad.

Frozen features and catalogue order come from the existing completed private
`carasik/rhpr-ml100k-multimodal` notebook, with manifest/executable/vector/order
hashes pinned. No new feature encoding, graph walk training or data upload.
Same text, poster, MPNet description and DeepWalk cast/director vectors.

## Generator training

Each arm trains a fresh, identically initialized TIGER 4+4/d128 on its resulting
frozen SID index. Preserve the full-training policy used for previous models:

- Epochs 1–20: validation at 5/10/15/20, unchanged Adam LR 0.001.
- At 20, preserve prefix curves/selection and last-checkpoint hash. Continue
  the last model, Adam/scaler and RNG through the existing verified resume
  path. Do not evaluate test at the phase boundary.
- Epochs 21 onward: validation every epoch; ReduceLROnPlateau halves after
  five failed checks, threshold 0.0001, minimum LR 0.00001. Seed scheduler with
  historical best and zero bad checks.
- Minimum 30 / maximum 100 total epochs, early stopping 20 epochs since best
  raw validation NDCG@10. Test once after checkpoint selection. Flag epoch-cap
  fits as budget-censored. A time-budget exit fails without autoextension.

Same 1,273 films / 943 users, timestamp-group holdouts, history 20, 85,470 cases
per epoch, effective/physical batch 256/128, Adam weight decay 0.0001, FP16
generator training / FP32 evaluation, beam 20, full catalogue, no seen mask or
sampled negatives. Quantizer remains FP32. One trial per T4, CPU thread 1.

## Diagnostics and decision

Keep raw reconstruction MSE per head and weighted reconstruction totals, AE
spectrum/latents/weights, codebook curves, occupancy and codes, selected/last
generator checkpoints, prefix/resume audit, LR, per-level validation CE and
validation/test predictions. Weighted totals across arms are different
objectives; compare raw per-head errors separately.

Primary comparison: selected validation/test item NDCG@10 and Recall@10, paired
within the one pipeline seed. Check whether weaker graph pressure recovers text
reconstruction and text-neighbor prefix preservation while retaining some graph
structure. These diagnostics are not validated recommendation-quality proxies.
A positive pilot would require replication; no extra seeds or weight sweep are
included. The initializations match, but CUDA training need not be bitwise
deterministic.

Practical estimate: roughly 45–75 minutes plus setup/queue if stopping resembles
the previous fits; near-cap training may take about 1.5–2 hours. No retries,
automatic extensions, split changes, new graph representations, TPU or Telegram.
Preserve all earlier outputs and paused TPU edits.
