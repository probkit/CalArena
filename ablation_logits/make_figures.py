"""
Builds the figures and tables of the logits vs. probabilities ablation (paper appendix).

The {calibrator}-probs variant is the main benchmark result (results/{benchmark}/),
the -logits and -scaled-logits variants are read from ablation_logits/results/.

Usage (from the repository root):
    python ablation_logits/make_figures.py
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from utils import load_benchmark_results, compute_winrates, plot_leaderboard

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
})

VARIANTS = ["probs", "logits", "scaled-logits"]

# Methods of the main benchmark that are not part of the ablation
CV_OTHER_METHODS = [
    "Base-model", "Hist-uniform", "Hist-quantile", "Isotonic", "CIR", "Venn-Abers",
    "BBQ", "Spline", "Platt-logits", "ETS", "SVS", "SMS", "Dirichlet", "Kernel",
    "LightGBM", "XGBoost", "CatBoost",
]
IMAGENET_OTHER_METHODS = [
    "Base-model", "Hist-uniform", "Hist-quantile", "Isotonic", "CIR", "Platt-logits",
    "ETS", "SVS",
]

ABLATIONS = {
    "CV-multiclass": (["TS", "VS", "MS"], CV_OTHER_METHODS, "cv"),
    "ImageNet-multiclass": (["TS", "VS"], IMAGENET_OTHER_METHODS, "imagenet"),
}


def load_ablation_results(benchmark_name, ablated, other_methods):
    # Main benchmark results, renaming the ablated methods to {method}-probs
    results = load_benchmark_results(benchmark_name, other_methods + ablated)
    results = results.rename(
        columns={
            col: col.replace(f"{m}_", f"{m}-probs_", 1)
            for m in ablated
            for col in results.columns
            if col.startswith(f"{m}_")
        }
    )

    ablation = load_benchmark_results(
        benchmark_name,
        [f"{m}-{v}" for m in ablated for v in VARIANTS if v != "probs"],
        results_dir=str(HERE / "results"),
    )
    merge_cols = ["dataset", "model", "cal_size", "test_size", "n_classes"]
    return results.merge(ablation, on=merge_cols)


for benchmark_name, (ablated, other_methods, short_name) in ABLATIONS.items():
    np.random.seed(0)

    results = load_ablation_results(benchmark_name, ablated, other_methods)
    methods = other_methods + [f"{m}-{v}" for m in ablated for v in VARIANTS]

    winrates = compute_winrates(
        results, methods, metric="brier", bootstrap_datasets=False, n_bootstraps=1000
    )

    plot_leaderboard(
        winrates,
        title=benchmark_name,
        y_label="Average winrate",
        y_lim=(0, 1),
        color="green",
        figsize=(9, 4),
        savefile=f"figures/ablation_logits_{short_name}_winrates.pdf",
    )

    print(f"\n### {benchmark_name}\n")
    print(winrates.round(3).to_markdown())
    print()
    print(winrates.round(3).to_latex())
