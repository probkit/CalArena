"""
Reproduces all the figures and tables of the paper.

Usage (from the repository root):
    python make_figures.py

Figures are saved to figures/ and the absolute improvement tables of the appendix to
figures/tables/ (LaTeX tabulars, included in the paper with \\input).
"""

import os
import sys
from pathlib import Path

# The critical difference diagrams (scikit-posthocs) iterate over Python sets, whose
# order depends on string hashing. Fixing the hash seed, which must be set before the
# interpreter starts, makes the generated PDFs byte-identical across runs.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import (
    load_benchmark_results,
    compute_winrates,
    compute_elo_scores,
    compute_absolute_improvements,
    plot_leaderboard,
    plot_grouped_leaderboards,
    plot_cd_diagram,
    save_and_show,
)

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

FIGURES_DIR = Path("figures")
TABLES_DIR = FIGURES_DIR / "tables"

BINARY_METHODS = [
    "Base-model",
    "Hist-uniform",
    "Hist-quantile",
    "Scaling-Binning",
    "BBQ",
    "Isotonic",
    "CIR",
    "Venn-Abers",
    "TS",
    "ETS",
    "Platt-probs",
    "Platt-logits",
    "Quadratic",
    "Beta",
    "Spline",
    "CDF-Spline",
    "Kernel",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]

MULTICLASS_METHODS = [
    "Base-model",
    "Hist-uniform",
    "Hist-quantile",
    "Isotonic",
    "CIR",
    "Venn-Abers",
    "BBQ",
    "Spline",
    "Platt-logits",
    "TS",
    "ETS",
    "VS",
    "SVS",
    "MS",
    "SMS",
    "Dirichlet",
    "Kernel",
    "LightGBM",
    "XGBoost",
    "CatBoost",
]

# Methods that are too slow for 1000 classes are excluded
IMAGENET_METHODS = [
    "Base-model",
    "Hist-uniform",
    "Hist-quantile",
    "Isotonic",
    "CIR",
    "Platt-logits",
    "TS",
    "ETS",
    "VS",
    "SVS",
]

BINARY_METRICS = {
    "brier": "min",
    "logloss": "min",
    "kuiper": "min",
    "ece-15": "min",
    "accuracy": "max",
}

MULTICLASS_METRICS = {
    "brier": "min",
    "logloss": "min",
    "ece-15": "min",
    "accuracy": "max",
}


METRIC_NAMES = {
    "brier": "Brier",
    "logloss": "Logloss",
    "kuiper": "Kuiper",
    "ece-15": "ECE-15",
    "accuracy": "Accuracy",
}


def write_latex_table(table, metrics, path):
    """
    Writes a table from compute_absolute_improvements() as a LaTeX tabular. The paper
    includes it with \\input inside a table environment holding the caption.
    """
    header = " & ".join(
        ["Method"]
        + [
            rf"$\phantom{{-}} 100 \times \Phi_\mathrm{{{METRIC_NAMES[metric]}}}$"
            for metric in metrics
        ]
    )
    rows = [
        " & ".join([method, *row.astype(str)]) + r" \\"
        for method, row in table.iterrows()
    ]
    lines = [
        r"\begin{tabular}{l" + "l" * len(metrics) + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n")


def benchmark_results(
    benchmark_name, methods, metrics, bootstrap_datasets, use_num_datasets, color
):
    """
    Winrate and Elo leaderboards, critical difference diagram and absolute
    improvement table for one benchmark. Returns the results, winrates and Elo scores.
    """
    modality, source = benchmark_name.lower().split("-")[::-1]
    prefix = f"{modality}_{source}"

    results = load_benchmark_results(benchmark_name=benchmark_name, methods=methods)

    winrates = compute_winrates(
        results,
        methods,
        metric="brier",
        bootstrap_datasets=bootstrap_datasets,
        n_bootstraps=1000,
    )
    plot_leaderboard(
        winrates,
        title=benchmark_name,
        y_label="Average winrate",
        y_lim=(0, 1),
        color=color,
        savefile=FIGURES_DIR / f"{prefix}_winrates.pdf",
    )

    elo = compute_elo_scores(
        results, methods, bootstrap_datasets=bootstrap_datasets, num_bootstrap=100
    )
    plot_leaderboard(
        elo,
        title=benchmark_name,
        y_label="Elo score",
        color=color,
        savefile=FIGURES_DIR / f"{prefix}_elo.pdf",
    )

    plot_cd_diagram(
        results,
        methods,
        metric="brier",
        groupby_dataset=False,
        title=benchmark_name,
        savefile=FIGURES_DIR / f"{prefix}_cd.pdf",
    )

    absolute = compute_absolute_improvements(
        results=results,
        methods=methods,
        metrics=metrics,
        use_num_datasets=use_num_datasets,
        precision=2,
        factor=100,
    )
    write_latex_table(absolute, metrics, TABLES_DIR / f"{prefix}_absolute.tex")

    return results, winrates, elo


def plot_sms_vs_spline():
    """Winrate of SMS against Spline as a function of the number of classes."""
    results = load_benchmark_results(
        benchmark_name="TabRepo-multiclass", methods=["Spline", "SMS"]
    )
    n_classes = results.n_classes.unique()
    n_classes.sort()

    sms_winrate = []
    for n_class in n_classes:
        results_filter = results[results.n_classes <= n_class]
        sms_winrate.append(
            (results_filter["SMS_brier"] < results_filter["Spline_brier"]).mean().item()
        )

    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.plot(n_classes, sms_winrate, marker="o", color="#277a3e")

    ax.set_xlabel("Maximum number of classes")
    ax.set_ylabel("SMS winrate over Spline")

    ax.set_title("SMS vs Spline winrate, TabRepo-multiclass")

    plt.tight_layout()

    save_and_show(FIGURES_DIR / "sms_spline_winrate.pdf")


def plot_catboost_ablation():
    """Calibration design principles added to a CatBoost classifier."""
    methods = ["Base-model", "CB", "CB-tiny", "CB-monotone", "CB-tiny-monotone"]
    results = load_benchmark_results(benchmark_name="TabRepo-binary", methods=methods)
    winrates = compute_winrates(
        results, methods, metric="brier", bootstrap_datasets=True, n_bootstraps=1000
    )
    # Short labels (monotone -> arrow) so that they fit horizontally
    winrates = winrates.rename(
        index=lambda method: method.replace("monotone", r"$\nearrow$")
    )
    plot_leaderboard(
        winrates,
        title="TabRepo-binary",
        y_label="Average winrate",
        y_lim=(0, 1),
        savefile=FIGURES_DIR / "catboost_ablation.pdf",
        figsize=(5, 3.5),
        xticks_rotation=0,
    )


def plot_runtimes(results, methods, multiclass, color, title, savefile):
    """Average runtime per 1000 calibration samples (and per class for multiclass)."""
    methods = [method for method in methods if method != "Base-model"]
    results = results.reset_index(drop=True)

    df = results[[f"{method}_time" for method in methods]].copy()
    df = df.rename(columns={f"{method}_time": method for method in methods})

    for method in methods:
        df[method] = 1000 * df[method] / results["cal_size"]
        if multiclass:
            df[method] = df[method] / results["n_classes"]

    table = pd.DataFrame({
        "time_mean": df[methods].mean(),
        "time_err": 1.96 * df[methods].sem(),
    })

    table = table.sort_values(by="time_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        table.index,
        table["time_mean"],
        yerr=table["time_err"],
        color=color,
        capsize=2,
        edgecolor="black",
        linewidth=0.5,
        error_kw={
            "elinewidth": 0.5,
            "capthick": 0.5,
            "ecolor": "gray",
        },
    )
    ax.grid(True, axis="y", which="minor", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.grid(True, axis="y", which="major", linestyle="-", linewidth=1.0, alpha=0.8)
    ax.set_axisbelow(True)

    ax.set_yscale("log")
    if multiclass:
        ax.set_ylabel("Avg runtime per 1000 samples per class (log scale)")
    else:
        ax.set_ylabel("Avg runtime per 1000 samples (log scale)")

    ax.set_title(title)

    ax.yaxis.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    plt.xticks(rotation=40, ha="right")

    plt.tight_layout()

    save_and_show(savefile)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Fix the seed so that bootstrapped confidence intervals are reproducible
    np.random.seed(0)

    # Per-benchmark results. Confidence intervals are obtained by bootstrapping
    # datasets when the benchmark contains enough of them, experiments otherwise.
    binary = {}
    for benchmark_name, bootstrap_datasets in [
        ("TabRepo-binary", True),
        ("TabArena-binary", True),
        ("CV-binary", False),
    ]:
        binary[benchmark_name] = benchmark_results(
            benchmark_name,
            BINARY_METHODS,
            BINARY_METRICS,
            bootstrap_datasets=bootstrap_datasets,
            use_num_datasets=bootstrap_datasets,
            color="blue",
        )

    multiclass = {}
    for benchmark_name, methods, bootstrap_datasets in [
        ("TabRepo-multiclass", MULTICLASS_METHODS, True),
        ("TabArena-multiclass", MULTICLASS_METHODS, False),
        ("CV-multiclass", MULTICLASS_METHODS, False),
        ("ImageNet-multiclass", IMAGENET_METHODS, False),
    ]:
        multiclass[benchmark_name] = benchmark_results(
            benchmark_name,
            methods,
            MULTICLASS_METRICS,
            bootstrap_datasets=bootstrap_datasets,
            use_num_datasets=bootstrap_datasets,
            color="green",
        )

    # Grouped leaderboards (Figure 1 and Elo results). ImageNet is reported separately
    # since many methods cannot be applied.
    for modality, benchmarks, color, title, elo_lim in [
        ("binary", binary, "blue", "Binary benchmark results", (500, 1500)),
        ("multiclass", multiclass, "green", "Multiclass benchmark results", (500, 1700)),
    ]:
        benchmarks = {
            name.split("-")[0]: values
            for name, values in benchmarks.items()
            if not name.startswith("ImageNet")
        }

        plot_grouped_leaderboards(
            {name: winrates for name, (_, winrates, _) in benchmarks.items()},
            title=title,
            y_label="Average winrate",
            y_lim=(0, 1),
            savefile=[
                FIGURES_DIR / f"{modality}_winrates.pdf",
                FIGURES_DIR / f"{modality}_winrates.png",  # README
            ],
            color=color,
        )
        plot_grouped_leaderboards(
            {name: elo for name, (_, _, elo) in benchmarks.items()},
            title=title,
            y_label="Elo score",
            y_lim=elo_lim,
            savefile=FIGURES_DIR / f"{modality}_elo.pdf",
            color=color,
        )

    # Takeaways experiments
    plot_sms_vs_spline()
    plot_catboost_ablation()

    # Runtimes, over the TabRepo, TabArena and CV benchmarks
    plot_runtimes(
        pd.concat([results for results, _, _ in binary.values()]),
        BINARY_METHODS,
        multiclass=False,
        color="#2066a8",
        title="Binary methods runtime comparison",
        savefile=FIGURES_DIR / "binary_runtimes.pdf",
    )
    plot_runtimes(
        pd.concat(
            [
                results
                for name, (results, _, _) in multiclass.items()
                if not name.startswith("ImageNet")
            ]
        ),
        MULTICLASS_METHODS,
        multiclass=True,
        color="#277a3e",
        title="Multiclass methods runtime comparison",
        savefile=FIGURES_DIR / "multiclass_runtimes.pdf",
    )


if __name__ == "__main__":
    main()
