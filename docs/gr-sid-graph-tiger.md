# Frozen graph placements → TIGER

User authorized a downstream check on2026-10-27. Use
`experiments/amazon_beauty_sid_graph_tiger_2t4.json`. One private two-T4 session,
three new generator fits: original RQ-KMeans,lambda0.3-swapped RQ-KMeans and
lambda0.3-swapped native RQ-VAE. Pair the two RQ-KMeans runs on the same worker.
Reuse the completed native RQ-VAE control from `rhpr-rqvae-tiger-beauty` source
6e37388a2dcd7f786bc49d47ac381459ef1de323,testNDCG10=0.021541507543033203.
Its seed,data,generator and training configurations are exactly equal. The old
RQ-KMeans run had up to100epochs/validation5,so it cannot supply the20epoch control.

Reuse exact final CPU candidates,not fresh graph optimization:Sentence-T5 full
metadata;RQ-VAE3x256 and normalized-residual RQ-KMeans3x256;graph k20,lambda0.3,
20kproposals,75%TRAIN users. Native RQ-VAE accepted258swaps/moved4.06%items;
RQ-KMeans accepted62/moved1.01%. Proxy candidates were selected exploratorily.
First-token held-out gain rises0.32394→0.32838 and0.41125→0.41202,respectively;
paired user-bootstrap intervals overlap zero. These are not downstream gains.

`package_sid_graph_indices.py` uploads no encoders or checkpoints. It packages
complete original/swapped paths,owner permutations,source catalogue order,CPU
manifest/config/split and recorded candidate statistics into a small private
Kaggle dataset. Its content manifest and every file are SHA256-pinned. Loader
checks family,role,lambda,catalogue,data identity and exact original[owner]
equality. Keep supplied suffix ownership using `SidIndex.from_full_sids`;do not
rebuild suffixes or nearest-centroid assignments. All path occupancies are fixed.

Preserve the established TIGER protocol:generator/data seed42,4encoder+4decoder,
d128/FF1024/dkv64/6heads/dropout0.1/noSEP/userbins2000,4977920parameters.
FP16 train,AdamLR0.001/wd0.0001/clip1,batch256/physical128. Every one of131413
TRAIN-prefix targets once per epoch,latest20context,fixed seed+epoch permutation.
20complete epochs,validation only10/20,selection by validation itemNDCG10,
one final FP32free-beam20test per model. Full12101catalogue,no seen-item masking;
invalid beams retain miss ranks. Save best/last/checkpoint states,curves and
per-user test predictions. Assert original neural and case hashes before workers
start and neural hash inside each worker. The original standalone RQ-VAE run
did not log initialization hash:reconstruct it from its pinned unchanged builder
and config;new workers check5f4a24fa…/case6f7ba06c…exactly.

No quantizer,encoder or graph search during GPU run. No InfoNCE,otherlambda/k,
tuning,extra seeds,automatic retries,TPU or Telegram. One generator seed cannot
establish superiority or noninferiority. Similar quality would support feasibility
of this placement procedure,not demonstrate a benefit by itself. Meaningful gains
would motivate later matched repeats and stronger index objectives.
