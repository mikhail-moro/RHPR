# Three-modality tokenizer MTL screening

User-authorized first screen: metadata Sentence-T5, frozen poster and description-only
MPNet; no people input. Seed42 only, four FRESH pipelines: sum, UW, GradNorm, PCGrad.
Same frozen input bytes, catalogue, temporal split and initial AE/generator parameters.
Inactive people block remains zero for compatibility with historical architecture.

AE3500 fixed updates -> native residual KMeans initialization -> RQ-VAE3000 optimizer
steps (including native initialization-only batch);3x256/latent64. All objectives use
catalogue-centred/RMS-normalized targets and dimension-mean observed-item MSE.
MLP heads/encoder/decoder transfer across phases; fresh RQ Adagrad accumulators.
MTL applies to reconstruction in BOTH phases. GradNorm reference losses restart at RQ;
learned task weights transfer. Quantization and commitment coefficients remain native.

- Sum: three equally weighted MSEs, fresh matched baseline.
- UW: sum exp(-s_m)*MSE_m+s_m, initial s=0. Same native Adagrad LR0.001 for
  weights and neural parameters; learned weights are unconstrained in total scale.
  Thus UW also changes reconstruction/quantization balance; not pure relative weighting.
- GradNorm: alpha1.5, positive softmax weights sum3, initial1 each; same native
  Adagrad LR0.001. Norms measured at last shared encoder Linear weight. Weight
  objective uses detached norms/targets, neural objective detached task weights.
- PCGrad: published random-order pairwise projections on coordinates shared by ALL
  reconstruction tasks; shared projected gradients SUMMED (not averaged) to preserve
  summed control scale, task-private gradients summed. Quantization gradients added
  afterwards untouched. Private seeded Python RNG independent of model/data streams.
  Algorithm implemented locally; no claim of using an external PCGrad package.

Save per-head MSEs, weights, gradient norms/cosines every500 optimizer calls, AE/RQ
phase, book usage/collisions/spectra, codes/books, checkpoints and provenance. Fixed
budgets avoid different spectral stopping times confounding the loss comparison.

TIGER4+4/d128, history20, FP16train/FP32eval, batch256/physical128, LR.001 constant,
30epochs, val5/10/15/20/25/30, no scheduler/early stop, best VAL NDCG10 checkpoint,
one final full-catalogue unmasked beam20 test. Same943users/1273films/85470cases per
training epoch. Existing heldouts reused; seed42 is exploratory, no novelty or
robust quality gain established. Do not select methods on test or count old models
as matched controls. Two T4 workers: sum+GradNorm / UW+PCGrad. Estimated55–75min,
not measured; remaining quota1.43h at preparation, no automatic retry/fallback.
