# SID research ledger

Research stages below are ordered conceptually and chronologically. Paths refer
to `analyses/` inside the companion archive. Original reports retain their full
numbers, uncertainty, selection criteria and scripts. Statements here are final
interpretations as of 2026-10-25, not a claim of independent replication.

| Stage | Question / check | Outcome and limitation | Evidence directory |
|---|---|---|---|
| 1 | Locate quality differences across six quantizer seeds | Per-item likelihood/rank differences exist; common generator initialization is not an independent seed replication | `sid-first-pass` |
| 2 | Per-level likelihood, support, geometry, behavioral coherence | Support predicts likelihood better than distortion; measured properties did not predict held-out NDCG gains reliably | `sid-first-pass` |
| 3 | Beam versus exact, matched scoring | Beam10 differs from exact; wider beams recover most observed panel differences. Exact model ranking can still be wrong | `sid-first-pass` |
| 4 | Mask/shuffle/recent-only/zero-memory history interventions | History affects the fixed generator; later levels use it less in absolute average nats. OOD interventions are not trained competitors | `sid-context-probe` |
| 5 | Extreme items and users | Local branch failures and context sensitivity found; selected extremes do not establish prevalence | `sid-extremes` |
| 6 | Good/bad item cohorts with frequency controls | Quality differences persist, particularly at level2; purity, distortion and collisions do not supply a simple explanation | `sid-cohorts` |
| 7 | At which level are brand/category features distinguishable? | Early features are distinguishable; within-item feature-location quality effect is not strong or universal | `sid-feature-location` |
| 8 | Global late-token semantics versus conditional parent meaning | Associations can conflict globally while being coherent within the actual prefix. No causal interference mechanism established | `sid-semantic-conflicts` |
| 9 | Does decoder prefer globally semantically attractive wrong children? | Paired wrong-child test 52.9%, interval47.4–58.4%; no decisive preference | `sid-child-choice` |
| 10 | Quantizer × generator seed follow-up | Some item-specific index effects replicate; global winner changes. Common data order controls verified | `sid-training-variance` |
| 11 | Rare branch2166/3500 trajectory and recent batches | Switching repeats and tracks recent targets; not proof of catastrophic forgetting or a semantic cause | `sid-training-variance` |
| 12 | How prevalent are switches and early misses? | Canonical rare target is only14 validation cases; 96.6% of beam10 misses lose the true prefix at level1/2, not necessarily due to beam approximation | `sid-error-prevalence` |
| 13 | Perfectly repair one conditional choice | Target-aware ceiling shows early leverage; impossible oracle intervention is not a deployable method | `sid-decision-ceiling` |
| 14 | Predict first code versus continue true prefix | Teacher-forced continuation and free retrieval separated; local correctness does not imply end-to-end success | `sid-conditional-decoding` |
| 15 | Quality of individual tokens and first-two-code pairs | Stable group means depend strongly on member composition; no universal token quality rule | `sid-token-quality` |
| 16 | Catalogue support, singleton prevalence | Most pairs are singletons; about50.4% of validation cases use one. This is exposure prevalence, not excess risk | `sid-prefix-support`, `sid-density-semantics` |
| 17 | Pair item count versus original/sampled supervision | Raw association is confounded; matched within-item size benefit uncertain and unstable across tolerances | `sid-size-vs-support` |
| 18 | Is low-support quality simply incomplete optimization? | Data scarcity and finite optimization are different; late deterioration exists, but convergence of all pairs is unproven | `sid-pair-learning` |
| 19 | Construct semantic atlas | Token/prefix catalogue associations linked to real cases; lookup instrument, not evidence the network uses those features | `sid-density-semantics` |
| 20 | Supported persistent errors + history/target feature inspection |1880 case-index candidates,81 resolved by another index at≥2/3 new seeds; no established semantic collision mechanism | `sid-semantic-errors` |
| 21 | Exact feature marginals and metadata audit | Mask retains69% Skin Care mass despite hair-oil top1; SHANY gap mostly missing brand; EltaMD history rule has no train support; color sample inadequate | `sid-semantic-errors` |

## Hypotheses left open rather than disproved

- Representation objective and behavioral neighbor consistency may influence
  how useful early clusters are to a finite generator.
- Shared token parameters may create conditional interference, but observed
  global semantic disagreement is insufficient to demonstrate it.
- Sparse prefix supervision, optimizer state, LR and user-grouped minibatches
  may jointly affect conditional calibration; no causal data-order run was made.
- Pair size may matter under another index or supervision regime; current
  matched comparisons lack a reliable independent benefit.
- Larger or differently trained generators might change the hierarchy of
  errors. Current20-epoch curves do not establish an architecture ceiling.

None of these is pending authorized GPU work. The user chose to pause this
direction and return to representation/encoder comparisons.

## Claims explicitly not supported

“Text encoders are the weakest component”; “semantic purity guarantees good
SID”; “singleton pairs cause the quality loss”; “beam search causes96.6% of
errors”; “the model ignores history”; “a metadata-brand mismatch proves encoder
failure”; “extra epochs solve everything”; “our source hybrid reproduced TIGER”.
