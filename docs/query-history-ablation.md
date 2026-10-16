# Frozen-checkpoint history interventions

Configuration: `experiments/yambda_query_history_ablation.json`, seed 42.
This validation-only diagnostic tests whether the completed SASRec Query SSM
and cross-attention models depend on events before their old-suffix queries.
It performs no optimization and reads no test anchors for evaluation.

The parent is the private kernel `carasik/rhpr-sasrec-prefix-query-pilot`, source
revision `660c7705ed3e005f35b339313c8c865255e36ede`. QSSM uses its selected
pass-20 weights; cross-attention uses its selected pass-30 weights. Configuration,
checkpoint size and SHA256, parent manifest and saved validation anchors are
verified before inference. A kernel-source reference does not itself pin a
version; the file hashes bind the required content.

For an observed history `X = [remote, Q, R]`, both Q and R contain 128 events:

| Condition | Predictor input history |
| --- | --- |
| `full` | Complete observed history, in original order |
| `query_only_history` | Keep Q and R; pass them through the same trained compressor |
| `shuffle_remote` | Permute only remote item IDs, preserving their multiset and length |

`query_only_history` is an inference intervention on the original QSSM/CA.
It is distinct from training a QueryOnly architecture without a scan/readout.
Both interventions preserve the query inputs and raw tail byte for byte.
Recency relative to H_end preserves query positions, retained memory positions,
summary positions and raw-tail positions after cropping. Summary values may
change; that is the quantity being tested.

The permutation is deterministic from the seed, user ID and anchor identity,
independent of model, batching and traversal order. It never consults target
values or target timestamps. Item IDs move between remote chronological slots;
timestamps remain on those slots. Time features must remain OFF. Unknown item
ID zero continues to occupy an observed slot rather than becoming padding.

The same 26,335 validation anchors, train-only catalogue, OOV policy and
full-catalogue Top100 ranking remain fixed. Metrics are macro-user NDCG@10 and
NDCG@100; all of each user's daily anchors stay together in paired bootstrap.
The original reference-context hash binds common evaluation support, while
separate hashes describe actual intervened inputs. Full inputs must reproduce
the saved parent validation ranks and metrics before interventions are accepted.
Source and native CUDA forward/backward audits are mandatory.

The primary family comprises two interventions against each model's full
input, four NDCG@10 contrasts. NDCG@100 is secondary. An absolute margin of
0.0005 is declared for each cutoff to distinguish a potentially negligible
fixed-weight change from an inconclusive non-significant result. Intervals and
any multiplicity correction are recorded explicitly; failure to reject zero
alone is not evidence of equivalence.

Interpretation is limited to these trained weights. Deleting distant history
tests dependence on the preceding state/readout. Shuffling retains item counts
but changes order and the assignment of remote recency features. Neither
intervention estimates how well a model trained from scratch without that
history would work. Distribution shift can worsen an otherwise useful model;
an independently trained QueryOnly control remains a separate next experiment.

The execution budget is 2,400 seconds, with an independent 90-minute Actions
timeout. All inference is uncached. Outputs retain per-anchor ranks, input
fingerprints, checkpoint verification, paired differences and observed timing,
plus standard flat scalar metrics, summary and run manifest. Parent weights
remain in the private kernel source and are not copied into new outputs.
