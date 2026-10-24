# SID index sensitivity study

Status: implemented with CPU verification; no GPU experiment submitted. On
2026-10-24 the user cancelled TPU reproduction work and chose GPU diagnostics.
No UI is planned. This study asks which index changes explain changes in the
prediction of the SAME item/case, and which effects belong to scoring/search.

## Frozen protocol

Configuration: `experiments/amazon_beauty_sid_index_study.json`.

| Component | Choice |
| --- | --- |
| Data | Original GRID/P5 Amazon2014 Beauty, 22,363 users, 12,101 items; original leave-last-two split |
| Representation | Complete frozen Sentence-T5-base 768D embeddings from private baseline v2; byte hash and catalogue order checked |
| Quantizer | sklearn Lloyd residual KMeans; normalize incoming residual at each level; 3×256, k-means++/n_init1/max_iter100; mandatory catalogue-order collision suffix |
| Varying factor | Quantizer seeds11/22/33/44/55/66; each level uses seed+level |
| Fixed RNG | Model seed42, data seed4242; identical initial tensor-content hash; same raw-item windows/order in every run |
| Generator | Pinned GenPAS T5, **3 encoder +3 decoder layers**, d128, FF1024, 6 heads with d_kv64, dropout.1; standard FF, no user embedding or extra GenPAS FF layer |
| Training | Two T4 GPUs, one independent seed worker per card, FP16/scaler, FP32 master weights/loss; Adam lr5e-4/wd1e-4, clip1, batch512; constant LR, partial batches retained |
| Budget | 20 epochs, no early stopping; checkpoints/validation5/10/15/20; compare epoch20 rather than each model's best checkpoint |
| Epoch | Every source user once, seeded groups32, ordinary TIGER/GRID sampled contiguous subsequences, cap512 draws/group; duplicate sampled indices emitted once |
| Context | At most24 train items, original20 evaluation items; full SIDs, right padding to100 input tokens, target excluded |
| Evaluation | Entire static catalogue; no sampled negatives/seen-item masking; one held-out target/user/split; test only at final epoch |
| Search | Valid beam10 for every evaluated case under two scoring rules; beams50/100 plus exact search on fixed panels at the final epoch |

`amazon_beauty_sid_index_study_10x30.json` is a separate extended budget with
ten index seeds and30epochs, validation every5 and exact diagnostics at30.
Neither config dispatches compute by itself. The implementation supports smaller
seed lists with unchanged data/model seeds for a staged launch. Preserve all
other protocol settings when splitting execution; do not concatenate duplicate
shared tables blindly when combining outputs for analysis.

The cap is NOT512 examples guaranteed per group. The real first two manifests
contain291,611 and292,502 examples:570/572 optimizer updates at batch512.
Each epoch's exact sample membership/order is stored once in `training_windows`
and loaded one epoch at a time. For `[start,end)`, the target is item `end-1`,
visible history is `[max(start,end-1-24),end-1)`. The sampler/truncation is tested
against the original author collator on short/long histories and duplicate
draws. No GenPAS weighted sampling is enabled.

Canonical right padding excludes the source's unused masked target placeholder;
no placeholder is attended. Full input/mask parity is tested. `d_kv` is not
inferred from d_model/heads. This is a declared TIGER-style hybrid, not an exact
reproduction of a paper's scores, tokenizer or convergence schedule.

Six index seeds are NOT independent generator seeds. SID changes
affect both history encoding and target factorization; this experiment does
not isolate their separate contributions. Fixed twenty/thirty-epoch results
also cannot establish converged rankings if learning curves still improve.
No fallback changes LR, precision or batch size after a numerical failure.

## Evidence contract

All entities use stable raw item/user/case IDs shared across runs. Levels and
positions are zero based, ranks one based, scores natural-log probabilities.
Prefix keys are slash-separated codes; the root is empty with level−1. The
fourth level is a disambiguation suffix, not learned semantic content.

| Tables | Evidence |
| --- | --- |
| `items`, `item_features` | Every item, original ID, exact text/hash, train event/user support; title/brand/categories/price with extraction provenance and missing/unparsed status |
| `events` | Full original train/validation/test events, user and absolute position; timestamps null because not supplied |
| `cases`, `case_history` | Full history, actual input context, target, repeats/lag, visibility and input positions; panel stratum/inclusion probability |
| `training_windows` | Epoch/order/source group/user/start/end/visible start/target; independent of SID assignment |
| `train_edges` | Directed adjacent transitions and directed co-user pairs; counts/endpoint supports computed only from train |
| `runs`, `model_inputs` | Seeds, config, initial-state hash; exact evaluated input IDs/masks for each index realization |
| `item_codes` | Complete item SID including collision suffix |
| `geometry` | Assignment, nearest/alternative centroid, distances/margin, incoming/outgoing residual norms at each level |
| `prefixes` | ALL trie nodes, parent/token, item/train-user/event support, child count, next-child TRAIN target entropy/concentration |
| `token_usage` | Global vocabulary-token reuse at each level across parents; item/event and parent counts |
| `index_comparisons` | Prefix-partition and global level-token ARI/NMI for every pair of indices |
| `case_index` | History prefix diversity/entropy, history-target prefix matches and lags |
| `exposures` | Actual item context/target counts and per-level target loss sums every epoch |
| `training_epochs`, `checkpoints` | Examples, attempted/effective/skipped updates, AMP scale, loss/gradient norm, training/evaluation time, CUDA peak memory; scheduled checkpoints and hashes |
| `evaluations` | Completed snapshot coverage: split, cases/panel cases, duration and metrics |
| `target_path` | Full FP32 logit vector at each correct-prefix decision, token rank, strongest alternative, raw/local probability and valid-child mass |
| `predictions`, `case_metrics` | Ranked item IDs/SIDs/scores; Recall/NDCG5/10; exact target ranks, beam/exact agreement and lost/promoted targets |
| `beam_trace` | Every retained prefix/score at every level; explicit null-rank correct prefix if pruned |
| `interventions`, `intervention_predictions` | Optional fixed-position event masks, before/after target scores/ranks and ranked outputs |
| `artifacts` | Paths/hashes/roles for representations, centroids, codes, configs, original input data and full-catalogue score vectors |

Texts remain authoritative. Feature provenance is
`parsed_from_encoder_text_v1`, NOT independently verified source metadata.
Unknown brand/missing fields are explicit. No dates/features are invented.
Original catalogue/sequences are copied into `inputs/`.

`prefixes.target_events` counts each noninitial TRAIN position once, not
sampled-window multiplicity. Actual supervision is in `exposures`; join codes
to aggregate by token/prefix. Loss sums are train-mode values before updates,
including dropout and attempted AMP-skipped steps. Divide by target_count;
zero support means undefined mean, not zero loss.

Geometry distances use each NORMALIZED incoming residual. Norms are before
normalization and after subtraction. They are not naive original-space additive
reconstruction errors: preserve embeddings/centroids to replay the recursion.
Numerical code labels have no identity across runs. Complete-SID prefix
partitions are always singletons, so their ARI1 is expected rather than evidence.

## Scoring versus search

Raw item score is `sum_l log p(z_l | history,z_<l)` using each head's FULL
vocabulary, matching training CE. Released GenPAS masks invalid child logits
BEFORE softmax. Its score is the sum of

`log p(z_l | ...) - log sum_valid_children p(child | ...)`.

A singleton child gets local probability1 even if raw probability was.01.
This may reorder items. It differs from normalizing raw catalogue scores with
ONE final normalizer, which preserves ranks. Both scoring rules are retained.

Exact scoring visits every valid prefix once, reuses the history encoding and
saves ALL raw/local leaf scores as per-case NPZ vectors. DB predictions retain
top100 plus exact target ranks. Raw valid-catalogue probability mass is saved;
the local catalogue distribution must sum to1. Exact raw target scores must
agree with one-pass teacher forcing.

Beams run in FP32/log space, with variable valid beam counts and no invalid
filler paths. Beam ties use lexicographic SID prefix; exact ties use item ID.
Local search is checked against nontied native source model outputs. Tie
conventions can affect discrete comparisons; full scores remain available.

Compare beam/exact under MATCHING scoring. Save both lost exact-top10 targets
and lucky beam promotions despite worse exact rank. Exact search can lower
recommendation metrics by recovering high-probability wrong items.

## Diagnostic budget and persistence

Every evaluated case gets full true-path head logits and beam10 under both
scores. All teacher-forced levels share one causal decoder pass. Exact search
and beams50/100 run only at the final epoch on512 preselected cases PER split,
shared across all models. Selection balances context-length and TRAIN target
popularity strata without looking at predictions. Validation panel selection
does not use test labels. Inclusion probabilities are stored: unweighted
stratified-panel averages are not population estimates. Main beam10 uses all
cases and is unaffected by panel weighting.

Exact search may cost substantially; its GPU cost is not yet measured. A rough
training-only extrapolation from the old T4 run suggests40–60minutes/20epochs
or60–90minutes/30epochs per model, with considerable uncertainty. It excludes
all new diagnostics. The old4+4 run trained131k examples/epoch in roughly70sec;
the new3+3 model processes292k/epoch with a larger batch/context. Do not sell
these extrapolations as profiled timings. Ten×30 means300 training epochs,
three times the original five×20 budget (before the user increased the default to six seeds), and twice as many final exact panels.
Measure first models before queuing the entire extended budget.

No full attention matrices/per-step gradient dumps. Scheduled checkpoints keep
model/optimizer/scaler, CPU/CUDA/NumPy/Python RNG, SID matrix, budget and config.
More diagnostics can be replayed selectively. Current replay supports fixed
position masking and beam regeneration, not automatic exhaustive interventions
or automatic training resume.

Parquet parts are compressed and atomically committed. Batches are buffered
to limit tiny files; evaluation wrappers flush on failure and every epoch
flushes. Abrupt process termination cannot guarantee uncommitted measurements.
`pipeline-status.json` distinguishes running/failed/complete; absence of an
`evaluations` row means incomplete measurement, not zero quality.
`inventory.json` hashes committed table parts.

## Read and replay

Install GR dependencies plus `pip install -e '.[sid-analysis]'` locally.
DuckDB is optional on the GPU machine: Parquet is the canonical portable store.

```bash
PYTHONPATH=src python -m rhpr.gr.study_analysis /path/to/output audit --verify-files
PYTHONPATH=src python -m rhpr.gr.study_analysis /path/to/output database
PYTHONPATH=src python -m rhpr.gr.study_analysis /path/to/output query 'SELECT * FROM item_quality LIMIT 10'
PYTHONPATH=src python -m rhpr.gr.study_analysis /path/to/output mask \
  --run rq-11 --epoch 20 --case validation:123:8 --positions 0 \
  --source /path/to/pinned/GenPAS --device cpu --beam-width 10
```

Rebuild DuckDB views with `database` after moving files. `item_quality` includes
support; an item never used as a target has no estimate. LEFT JOIN items to
preserve missing quality. `paired_quality` joins the same case/checkpoint/
scoring/search/width across runs, including null width for exact search.
Audit checks keys, joins, target tokens, window bounds, completed snapshot
coverage and optional artifact hashes. More examples: `sid-index-analysis.sql`.

Mask positions refer to ACTUAL input events. Removing attention without moving
survivors leaves a positional gap, which is part of the intervention. This is
an effect on a fixed model, not user-behaviour causality. Mask one occurrence
or specify all explicitly. Padding/all-history masks are rejected. Only replay
trusted checkpoints produced by this pipeline.

Explore validation first; freeze hypotheses before confirming on test. Paired
comparisons control unchanged item/case properties but do not prove that one
tree statistic caused the improvement. User bootstrap is conditional on the
fitted models; run-level N remains the number of index realizations.

The default is now six×20 epochs on two GPUs; the earlier ten×30 config is an
optional extended budget, not the selected run. The config builds through the stable Kaggle entry point with existing private
data/wheels and hash-pinned embeddings. A future authorized GPU launch must
inspect quota, preserve private outputs and collect the evidence store, not
only metrics.json. Implementation alone does not authorize dispatch.


## Two-GPU execution

The active config explicitly requests `machine_shape=NvidiaTeslaT4` via the
Kaggle CLI metadata, and checks that both requested logical CUDA devices exist
BEFORE preparing data. A single-GPU allocation fails; no serial/CPU fallback.
`execution.worker_devices=["cuda:0","cuda:1"]` assigns seeds round-robin:
GPU0:11/33/55; GPU1:22/44/66. This is independent model training, not DDP, so
per-model batch512, LR, losses, examples and update order do not change.

The parent prepares shared data, windows, all quantizers and cross-index
statistics once, then creates a CPU reference initialization with a tensor
hash. Two fresh subprocesses each see only their assigned GPU through
CUDA_VISIBLE_DEVICES (preserving an inherited GPU/UUID mapping). Model
construction uses the same initialization threads/seed and is checked against
the reference hash. Training and diagnostics use one CPU thread per worker;
the offline wheel site and embedded RHPR package are passed to subprocesses.
The original pinned source is independently activated in each fresh process.

Workers write disjoint table shards under `tables/<name>/worker-N/` and
separate `rq-<seed>` checkpoints/scores. All share the same evidence schema,
item/case IDs and raw-window files. Recursive Parquet/DuckDB views read one
logical database, with no duplicated global tables or concurrent DuckDB writes.
Only the parent builds the final inventory/database and aggregate metrics.

`workers/plan.json` records assignments and shared inputs; per-worker atomic
JSON status, logs and per-seed result/failure files retain progress. A failed
seed is not retried with different settings; other seeds continue. The parent
reports incomplete status if any attempt fails or a process dies, while keeping
partial successful outputs and their inventory. Parent cancellation terminates
child processes. No automatic training resume is provided.

The exact same process orchestration can be explicitly tested with CPU/FP32
workers. Tests compare native serial versus two-process model weights, RNG,
exposures and item metrics, and exercise failure isolation. CPU tests do not
establish CUDA compatibility, two-T4 throughput or near-2x speedup. CPU and disk
are shared; every model still runs20epochs and its own complete diagnostics.
