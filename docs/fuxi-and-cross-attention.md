# Interpreting FuXi and cross-attention results

A strong FuXi-Linear result in this repository does not establish a universal
advantage over cross-attention (CA), and industrial CA deployments do not imply
that CA must win this experiment.

## Different architectures and tasks

Our FuXi adapter is a complete sequence predictor: its native blocks process the
old history and recent suffix together. Our CA and QuerySSM branches first
compress the old history into a fixed number of summaries, then pass those
summaries and the recent suffix to SASRec. Comparing these systems is useful, but
does not isolate the compression operator: the predictor, information bottleneck
and allocation of computation also differ. Equal width and layer counts do not
equalize capacity or computational cost. See the
[baseline protocol](query-baseline-screen.md).

Our objective is next-item retrieval: one user representation scores the entire
catalogue. Industrial target-attention examples often instead rank already
retrieved candidates. In Douyin's STCA, each candidate supplies its own query,
allowing history selection specifically for that candidate. Its cost is linear
in history length per candidate; request-level batching shares the user/history
path across candidates. This differs from constructing reusable user summaries
without knowing the next item. The paper reports online experiments, rather
than establishing an architecture-independent preference for CA.
[STCA, sections 3.1–3.2 and 4.5](https://arxiv.org/html/2511.06077v1).

Cross-attention with K queries over L history tokens has an attention-map cost
proportional to K times L, excluding projections and feature widths. With fixed
K it is already linear in history length. A linear-attention backbone therefore
does not automatically have a better asymptotic history cost than a fixed-query
compressor; constants, candidate count, caching and kernel efficiency matter.

Industrial attention systems also include serving engineering. TWIN separates
features so that historical projections can be precomputed and cached while
preserving target-behavior relevance. Consequently, a comparison based only on
uncached forward time misses part of its deployment design.
[TWIN, abstract](https://arxiv.org/abs/2302.02352).

## Scope of the implementations and evidence

Our lightweight CA compressor is not a reproduction of LONGER. LONGER combines
token merging, an intra-group transformer, global tokens, cross-causal attention,
subsequent self-attention layers and KV-cache serving. A result for our compressor
does not transfer directly to that architecture.
[LONGER, sections 3.2–3.6](https://arxiv.org/html/2505.04421v1).

FuXi-Linear's February 2026 paper evaluates three public datasets; it does not
report an online A/B test or production deployment. Its offline results therefore
cannot establish why a particular production system should replace CA.
[FuXi-Linear, section 6](https://arxiv.org/html/2602.23671v1).
Our adapter retains the official native blocks, but supplies zero timestamps,
does not reproduce the paper's training recipe, and does not implement a fused
kernel or recurrent serving cache. See the
[implementation and provenance notes](../src/rhpr/vendor/fuxi/README.md).

## What the next experiment can establish

The rotating-window QSSM/FuXi experiment tests a training-protocol change:
replacing repeated final training blocks with different historical target blocks
at the registered update budget. It can show whether their quality gap persists
under that protocol. CA is excluded to conserve compute, so this experiment
does not directly settle the CA comparison. A retained gap would motivate further
controlled architectural work; a reduced gap would not by itself prove that
fixed targets caused every previous result.
