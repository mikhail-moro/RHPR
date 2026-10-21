> Historical report from 2026-10-20; research branch closed on 2026-10-21. Any proposed runs or next steps below are historical and cancelled. See the [closure record](../README.md) for the final status.

# Post-hoc target-repeat diagnostic

All three frozen pass60 checkpoints were evaluated on the same H1024+R128 contexts. This CPU-only analysis reconstructs those observed contexts, verifies their saved input hashes, and joins existing validation ranks. No model weights, model forward, or GPU compute were used.

The groups below use the held-out target for analysis after prediction. They are not permissible serving/routing features and do not identify periodicity. A target absent from the visible1152 events may still occur in the user’s earlier full history.

| Group | Anchors | Share | Users | FuXi NDCG@10 | QSSM NDCG@10 | QSSM2 NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| Target seen in R128 | 8,272 | 31.41% | 3,097 | 0.059256 | 0.043744 | 0.042570 |
| Target seen only in H1024 | 8,703 | 33.05% | 3,519 | 0.009301 | 0.008258 | 0.007455 |
| Known target absent from visible1152 | 7,960 | 30.23% | 3,492 | 0.005258 | 0.003923 | 0.004639 |
| OOV target | 1,400 | 5.32% | 1,054 | 0.000000 | 0.000000 | 0.000000 |

Each within-group score is macro-averaged over users having at least one anchor in that group. Users may occur in multiple groups; these means cannot be combined using anchor shares.

To locate the actual overall quality gap, each group’s contribution instead retains the original global macro-user denominator: each anchor contributes NDCG divided by that user’s total number of validation anchors, then divided by all4384 users. These group contributions sum exactly to the overall macro-user gap.

| Contribution to FuXi − compressor NDCG@10 | Against QSSM | Against QSSM2 |
|---|---:|---:|
| Target seen in R128 | 0.004931 (82.9%) | 0.005767 (86.1%) |
| Target seen only in H1024 | 0.000539 (9.1%) | 0.000677 (10.1%) |
| Known target absent from visible1152 | 0.000481 (8.1%) | 0.000258 (3.8%) |
| OOV target | 0.000000 (0.0%) | 0.000000 (0.0%) |

The largest part of FuXi’s observed advantage is on targets already present in R128: about83% against QSSM and86% against QSSM2. The simple story that the advantage mostly comes from recommending targets outside the recent tail is therefore not supported. This does **not** show that old history is unnecessary: older profile or order information can still determine which recently observed item is repeated.

Conditional paired bootstrap intervals, FuXi minus QSSM2, NDCG@10 (5000 user draws, seed42):

| Group | Difference | 95% interval |
|---|---:|---:|
| Target seen in R128 | 0.016686 | [0.012764, 0.020810] |
| Target seen only in H1024 | 0.001846 | [0.000249, 0.003469] |
| Known target absent from visible1152 | 0.000619 | [-0.000809, 0.002125] |

The new-target-within-window interval includes zero. All slices are exploratory and receive no multiple-comparison correction. The bootstrap reflects user sampling of the already selected frozen checkpoints; it excludes training-seed and checkpoint-selection uncertainty. No causal attribution to sequence order, calendar time, or compression follows from these groups alone.

NDCG@100, exact support fingerprints, full paired intervals for FuXi against both compressors, and exact global-gap contributions are saved in [the JSON analysis](../aggregates/qssm2-research-repeat-slices.json).
