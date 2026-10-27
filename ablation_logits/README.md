# Logits vs. probabilities ablation

CalArena fits every calibrator on **predicted probabilities**. Logit-based methods (TS, VS, MS, ...) are applied to the log-probabilities `log(p)`.
This is a deliberate design choice:
- we treat the base model as a black box that exposes probabilities, the common interface of all classifiers;
- TabRepo and TabArena only store predicted probabilities.

This folder reproduces the ablation study from the paper's appendix ("Calibrating on probabilities vs. logits"). It checks that this choice does not change the conclusions of the benchmark.
For the computer vision benchmarks we have access to the raw model logits, so we compare three inputs for the logit-based calibrators:

| Variant | Input to the calibrator | Information lost |
|---|---|---|
| `{method}-probs` | `log(softmax(logits))`, computed from the float32 probabilities of the main benchmark | dimensional + numerical compression |
| `{method}-scaled-logits` | `logits - logsumexp(logits)`, computed without going through the softmax | dimensional compression only |
| `{method}-logits` | raw model logits | none |

- **Dimensional compression:** the softmax is invariant to adding a constant to all logits, so probabilities lose one dimension of the logit space.
- **Numerical compression:** very confident predictions saturate to probabilities of exactly 0 or 1 in finite precision, so different logits can map to the same probability.

Scaled logits and probabilities are in one-to-one correspondence mathematically. Comparing them therefore isolates numerical compression, and comparing scaled logits with raw logits isolates dimensional compression.
Temperature scaling is invariant to a per-sample shift of the logits, so `TS-logits` and `TS-scaled-logits` only differ numerically.

Methods and benchmarks:
- `cv-multiclass`: TS, VS and MS;
- `imagenet-multiclass`: TS and VS (MS is too slow with 1000 classes, as in the main benchmark).


## Files

```
ablation_logits/
├── generate_logits_benchmarks.py  # Builds data/{benchmark}-{logits,scaled-logits}.h5 from cv_data/
├── run_ablation.py                # Fits TS/VS/MS on logits through the probmetrics torch interface
├── ablation.batch                 # SLURM array, one job per (benchmark, calibrator, variant)
├── make_figures.py                # Winrate leaderboards and tables for the paper appendix
└── results/{benchmark}/{method}-{variant}.csv
```

The `-probs` variant is not re-run: it is the main benchmark result (`results/{benchmark}/{method}.csv`).


## Reproducing the ablation

All commands are run from the repository root.

1. Generate the logit benchmarks (requires the raw CV logits under `cv_data/`, see the main README):
   ```bash
   python ablation_logits/generate_logits_benchmarks.py
   ```
2. Run the calibrators, either locally
   ```bash
   python ablation_logits/run_ablation.py --benchmark cv-multiclass --calibrator VS --variant scaled-logits
   ```
   or with SLURM: `sbatch ablation_logits/ablation.batch`.
3. Produce the figures (`figures/ablation_logits_{cv,imagenet}_winrates.pdf`) and the tables:
   ```bash
   python ablation_logits/make_figures.py
   ```

Note that probmetrics calibrators validate that the inputs of `fit()` / `predict_proba()` are probabilities.
`run_ablation.py` therefore passes logits through the torch interface: `fit_torch(CategoricalLogits(z), y)` and `predict_proba_torch(CategoricalLogits(z))`.
