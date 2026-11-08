# ML1M SID-by-generator stability: results

Private `carasik/rhpr-ml1m-sid-stability` COMPLETE, executable source
`2e3e56f7512150fa44f847c3377760582542cc46`. All 18 fits finished in
22535.88s (6.26h session, 2×T4), with no failed trials. Validation only.

Six frozen indexes (two tokenizer seeds per pipeline), three generator seeds
per fixed index. Same 2874 catalogue items,151166 training windows,2786 VAL
users/history20; narrow mini-TIGER, FP16/SDPA/512, validation10/min30/max60/
patience20. Five models satisfied the operational early-stop rule (four at
60, one at50); **13/18 reached the 60-epoch budget without satisfying it**.
The fixed50 comparisons cover all18 fits. Operational stopping does not prove
convergence. TEST has not been evaluated.

## Fixed epoch: mean VAL NDCG10 and generator spread

Means average two tokenizer indexes × three generator seeds. Spread is the
sample SD of three generator seeds separately within each fixed index; below,
the two within-index SDs are averaged. It is descriptive, not a precision
estimate from independent tokenizer replications.

| Pipeline | Mean NDCG30 | Mean NDCG50 | Mean within-index SD30 | Mean within-index SD50 |
|---|---:|---:|---:|---:|
| metadata → RQ-KMeans | .059574 | .064622 | .001725 | .002513 |
| metadata → RQ-VAE | .057837 | .059003 | .001985 | .005325 |
| metadata+poster+plot → RQ-VAE | .054482 | .057701 | .003369 | .002198 |

SD by index42/index101: KMeans30 .001369/.002081,50 .002844/.002183;
textRQ30 .001259/.002711,50 .005323/.005327; multimodal30 .003844/.002895,
50 .001840/.002556. Multimodal has larger observed SD on epoch30, but the
ordering changes on epoch50; these data do not establish that multimodality
causes higher or lower seed variance. Three generator seeds/two fixed indexes
per method are too few to measure population variances reliably.

Best-selected (possibly different epochs) mean VAL is .065377/.061544/.059092;
mean within-index SD .002638/.003036/.002961, respectively. Selection uses the
same VAL cohort, can bias maxima upward and is not the fixed-epoch comparison.
Do not present this as a full-TIGER result: transfer of mini-TIGER rankings to
full training is not reliably established in prior experiments.

17/18 last NDCG values exceed their epoch30 values; validation CE decreases
from epoch30 to last for14/18. Each measure is descriptive, and a favourable
late value neither establishes convergence nor a TEST gain. The plotted epoch60
text-RQVAE mean has n=5 because one model stops at epoch50; all epoch≤50 means
use n=6.

## Paired comparisons and uncertainty

KMeans vs multimodal: epoch30 delta+.005092, positive in5/6 matching labels;
conditional shared-user bootstrap95% interval [+ .001771,+ .008399]. At
fixed50, delta+.006921, positive6/6, interval [+ .00360,+ .01023]. Text RQ-VAE
vs multimodal: epoch30 delta+.003356, interval [+ .000364,+ .006402]; at50
+.001301, interval [− .001736,+ .004318]. These 5000 shared-user bootstraps
condition on these fixed indexes, trained models and same VAL cohort; they do
not include tokenizer or generator training randomness, target-item clustering,
checkpoint/epoch selection or multiple comparisons. Six runs are only two
independent tokenizer indices per method, not six independent tokenizers.
Same-named random seeds across different algorithms do not guarantee matched
randomness. Do not infer statistical proof, causation, or out-of-sample wins.

KMeans and RQ-VAE pipelines differ in latent network and reconstruction as well
as quantization. Text and multimodal RQ-VAE differ in input channels and active
heads. Observed KMeans quality is an exploratory mini-TIGER signal. The two
RQ-VAE representations do not support a robust effect of multimodal input on
quality or generator stability.

## Verification and provenance

Independent CPU replay re-scored every **107 validation prediction files**,
reproducing NDCG/Recall@5/@10, per-user target identities and invalid/duplicate
counts to1e−12. All 18 fits, 9+9 assignments, source/config/data hashes,
frozen parent feature/SID hashes, initial states, tokenizer warm-up/fits,
epoch/stop criteria, prediction availability and AMP counters verified.
15 AMP skips in316720 planned updates; 27 invalid candidate paths in5962040
beams,0 duplicates. Invalid paths retain their original rank; TEST not used.

Numerical analysis `docs/gr-ml1m-sid-stability-results.json`, CPU replay
`scripts/research/analyze_ml1m_sid_stability.py`, protocol/config
`docs/gr-ml1m-sid-stability.md` / `experiments/ml1m_tokenizer_generator_stability_2t4.json`.
Predictions/curves/indexes/graphs retained in the numerical evidence bundle;
large best/last model checkpoint weights stay on private Kaggle. No automatic
GPU continuation or new runs were sent during collection.

The matched-user bootstrap/summary CLI is
`scripts/research/analyze_tokenizer_generator_stability.py --root <results>
--data <input> --output <analysis>`; it recalculates epoch30,50 and selected
per-user scores. The graph CLI is
`scripts/research/plot_tokenizer_generator_stability.py --root <results>
--output <analysis>`. Both operate on saved VAL artifacts and use no GPU.
