# Actor-only DeepWalk and reconstruction-weight screening

User-authorized2026-11-02. Ten fresh pipelines: two seeds42/101 crossed
with five candidates. No new director-only fitting. Three fresh actor-only
graph weights1/.3/.1, no-people control, and cast readout of the old shared
people graph at weight1. Same original1273movies and fixed temporal cases.

The actor-only DeepWalk is prepared once on CPU from the saved TMDB person
IDs, using ONLY cast membership. A person who is also a director remains
if credited as an actor; director-role edges are excluded. Same walk/SGNS
settings as previously: gensim4.4.0,64D,10walks per node,length40,window5,
negative5,sample0,3SGNSepochs,seed42,oneworker,PYTHONHASHSEED42. There are
15998actor nodes. Mean actor vectors are placed in the first64coordinates
of the original128D people slot, with zero director padding. Raw people
vectors, pooled film vectors, sourceIDs and provenance are saved in the
new PRIVATE carasik/rhpr-ml100k-actor-deepwalk dataset. Each input file hash
and catalogue order is pinned and verified. No interactions or holdout
labels enter graph training. The graph seed stays fixed across pipeline
seeds: this screen does not measure graph-training-seed uncertainty.

All arms retain the same Sentence-T5 text without names, ViT poster,
MPNet plot, native RQ3x256/latent64 and TIGER4+4/d128. Existing four-head
schema/initial weights are preserved. Actor coordinates alone are catalogue-
centered and RMS-normalized; only those64decoder outputs enter people MSE.
Other active reconstruction weights are1; candidate people weight applies
in BOTH the3500fixed-step continuous AE warm-up and3000step native RQ fit.
Quantization/commitment losses unchanged. No-people skips the entire block
and people loss. The old-graph cast control uses the exact saved old vectors.

Generator budget is FIXED30epochs: no early stopping before30, no LR
scheduler, constant Adam.001, FP16train/FP32eval, same batches256/physical128,
85470cases per epoch, last20strictly-past events. Validation ONLY5/10/15/20/
25/30, select on valNDCG10 and evaluate selected test once. This is a cheap
screen, not full convergence or an untouched final test. Primary candidate
comparison uses mean validation NDCG over the two seeds; do not choose the
weight from test. No automatic winner extension or extra seeds are authorized.

The two workers each hold one seed and run all5candidates, to keep hardware
assignment matched within seed. Same pinned quantizer/generator initial
parameters; fixed CUDA seeds do not guarantee bitwise training trajectories.
No-people and old-shared-graph arms are fresh controls under the same30epoch
schedule; earlier ES/full-training metrics are historical references only.

Preserve raw per-head and weighted losses, graph/input hashes, AE/RQ curves,
codebooks/maps/codes, generator curves/checkpoints and per-user val/test beams.
Old sharedgraph versus new actorgraph at weight1 diagnoses changed graph
training; weights1/.3/.1 compare graph influence within the new representation.
This is not a full topology×weight factorial, and does not establish DeepWalk
benefit over a person-identity baseline. High director sampleSD from only3
pipeline seeds did not by itself identify instability caused by directors.

Private kernel target carasik/rhpr-ml100k-actor-screen,2T4,expected roughly
2–3hours plus startup/queue at recent speed. No TPU, notifications, automatic
retries/extensions, extra weights or additional graph seeds.

Configuration: experiments/movielens100k_actor_screen_2t4.json.
