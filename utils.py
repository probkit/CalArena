import itertools
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from tqdm import tqdm
from scipy.stats import friedmanchisquare
from arena_rank.utils.data_utils import PairDataset
from arena_rank.models.bradley_terry import BradleyTerry


def load_benchmark_results(benchmark_name: str, methods: list) -> pd.DataFrame:
    allowed_benchmarks = {
        "tabarena-binary",
        "tabrepo-binary",
        "cv-binary",
        "tabarena-multiclass",
        "tabrepo-multiclass",
        "cv-multiclass",
        "imagenet-multiclass",
    }

    benchmark_name_ = benchmark_name.lower()

    if benchmark_name_ not in allowed_benchmarks:
        raise ValueError(
            f"Invalid benchmark name: '{benchmark_name_}'. "
            f"Allowed values are: {', '.join(sorted(allowed_benchmarks))}"
        )

    if not methods:
        raise ValueError("The 'methods' list cannot be empty.")

    modality = benchmark_name_.split("-")[-1]
    base_path = f"results/{benchmark_name_}"

    merge_cols = ["dataset", "model", "cal_size", "test_size"]
    if modality != "binary":
        merge_cols.append("n_classes")

    results = pd.read_csv(f"{base_path}/{methods[0]}.csv")
    for method in methods[1:]:
        df = pd.read_csv(f"{base_path}/{method}.csv")
        results = results.merge(df, on=merge_cols)

    return results


def compute_winrates(
    results, methods, metric="brier", bootstrap_datasets=True, n_bootstraps=1000
):
    """
    Computes mean win rates and 95% Confidence Intervals using bootstrapping.

    Args:
        results: DataFrame containing benchmark results and a "dataset" column.
        methods: List of method names.
        metric: The metric to evaluate (default "brier").
        bootstrap_datasets: True (resample blocks) or False (resample rows).
        n_bootstraps: Number of bootstrap iterations.
    """
    # Prepare data
    df = results[[f"{method}_{metric}" for method in methods]].copy()
    df = df.rename(columns={f"{method}_{metric}": method for method in methods})

    # Compute the win rate for each method, per experiment
    winrates_list = []
    for col in df.columns:
        other_cols = df.columns.difference([col])
        is_winning = df[other_cols].gt(df[col], axis=0)
        win_rate_per_experiment = is_winning.mean(axis=1)
        winrates_list.append(win_rate_per_experiment.rename(col))

    df_winrates = pd.concat(winrates_list, axis=1)

    # Attach the dataset column for dataset-level grouping
    df_winrates["dataset"] = results["dataset"].values

    # Base point estimate
    mean_winrates = df_winrates[methods].mean()

    # Bootstrapping
    boot_means = []

    if bootstrap_datasets:
        # Block bootstrap: Pre-compute sums and counts per dataset for speed
        dataset_sums = df_winrates.groupby("dataset")[methods].sum()
        dataset_counts = df_winrates.groupby("dataset").size()

        unique_datasets = dataset_sums.index.values
        n_datasets = len(unique_datasets)

        for _ in range(n_bootstraps):
            # Sample datasets with replacement
            sampled_ds = np.random.choice(
                unique_datasets, size=n_datasets, replace=True
            )

            # Compute the total wins and total experiments in this bootstrap sample
            total_wins = dataset_sums.loc[sampled_ds].sum()
            total_experiments = dataset_counts.loc[sampled_ds].sum()

            # The new mean win rate is total wins / total experiments
            boot_means.append(total_wins / total_experiments)

    else:
        # Standard bootstrap: resample N rows with replacement
        vals = df_winrates[methods].values
        n = len(vals)
        for _ in range(n_bootstraps):
            indices = np.random.choice(n, size=n, replace=True)
            boot_means.append(vals[indices].mean(axis=0))

    boot_means = np.array(boot_means)

    # Calculate 95% CIs (2.5th and 97.5th percentiles)
    ci_lower = np.percentile(boot_means, 2.5, axis=0)
    ci_upper = np.percentile(boot_means, 97.5, axis=0)

    leaderboard = pd.DataFrame(
        {
            "score": mean_winrates,
            "lower": pd.Series(ci_lower, index=methods),
            "upper": pd.Series(ci_upper, index=methods),
        }
    ).sort_values(by="score", ascending=False)

    return leaderboard


def compute_elo_scores(
    results, methods, metric="brier", bootstrap_datasets=True, num_bootstrap=100
):
    methods = sorted(methods)
    n_methods = len(methods)

    datasets = results.dataset.unique().tolist()
    n_datasets = len(datasets)

    # 1. Construct Pairwise Comparisons
    pairwise_data = []
    for model_a, model_b in itertools.combinations(methods, 2):
        pair_chunk = results[
            ["dataset", f"{model_a}_{metric}", f"{model_b}_{metric}"]
        ].copy()
        pair_chunk = pair_chunk.rename(
            columns={f"{model_a}_{metric}": "model_a", f"{model_b}_{metric}": "model_b"}
        )

        # A model wins if its score is strictly lower
        conditions = [
            (pair_chunk["model_a"] < pair_chunk["model_b"]),
            (pair_chunk["model_b"] < pair_chunk["model_a"]),
        ]
        choices = ["model_a", "model_b"]

        pair_chunk["winner"] = np.select(conditions, choices, default="tie")
        pair_chunk["model_a"] = model_a
        pair_chunk["model_b"] = model_b

        pairwise_data.append(pair_chunk)

    df_pairwise = pd.concat(pairwise_data, ignore_index=True)
    arena_dataset = PairDataset.from_pandas(df_pairwise)
    model = BradleyTerry(n_competitors=n_methods)

    # 2. Compute Ratings and Confidence Intervals
    if not bootstrap_datasets:
        # Experiment-level bootstrapping
        res = model.compute_ratings_and_cis(
            arena_dataset,
            significance_level=0.05,
            ci_method="bootstrap",
            num_bootstrap=num_bootstrap,
        )
        competitors = res["competitors"]
        score = res["ratings"]
        lower = res["rating_lower"]
        upper = res["rating_upper"]

    else:
        # Dataset-level bootstrapping manually implemented

        # Fit base model to acquire definitive point estimates (scores)
        model.fit(arena_dataset)
        base_ratings = model.params["ratings"]
        base_scaled_ratings = base_ratings * model.alpha + model.init_rating

        # Map base ratings to the canonical 'methods' list to prevent misalignment
        score = np.full(n_methods, np.nan)
        for idx, comp in enumerate(arena_dataset.competitors):
            score[methods.index(comp)] = base_scaled_ratings[idx]

        bootstrap_scores = np.zeros((num_bootstrap, n_methods))
        dataset_groups = {ds: df for ds, df in df_pairwise.groupby("dataset")}

        for i in tqdm(range(num_bootstrap), desc="Bootstrapping datasets"):
            sampled_datasets = np.random.choice(datasets, size=n_datasets, replace=True)
            blocks = [dataset_groups[ds] for ds in sampled_datasets]
            df_bootstrap = pd.concat(blocks, ignore_index=True)

            bs_dataset = PairDataset.from_pandas(df_bootstrap)
            bs_model = BradleyTerry(n_competitors=n_methods)
            bs_model.fit(bs_dataset)

            bs_ratings = bs_model.params["ratings"]
            bs_scaled_ratings = bs_ratings * bs_model.alpha + bs_model.init_rating

            # Extract scores and map to global method indices
            iter_scores = np.full(n_methods, np.nan)
            for idx, comp in enumerate(bs_dataset.competitors):
                iter_scores[methods.index(comp)] = bs_scaled_ratings[idx]

            bootstrap_scores[i, :] = iter_scores

        # Vectorized percentile extraction across the 0th axis (iterations)
        # Using nanpercentile guarantees survival if a method vanishes in a resample
        lower = np.nanpercentile(bootstrap_scores, 2.5, axis=0)
        upper = np.nanpercentile(bootstrap_scores, 97.5, axis=0)
        competitors = methods

    # 3. Construct Leaderboard
    leaderboard = pd.DataFrame(
        {
            "score": pd.Series(score, index=competitors),
            "lower": pd.Series(lower, index=competitors),
            "upper": pd.Series(upper, index=competitors),
        }
    ).sort_values(by="score", ascending=False)

    return leaderboard


def compute_absolute_improvements(
    results, methods, metrics, use_num_datasets=True, precision=2, factor=100
):
    """
    metrics: Expected as a dictionary, e.g., {"Accuracy": "max", "ErrorRate": "min"}.
    Uses the first metric passed as the main metric for sorting the table.
    Adds 95% Confidence Intervals to the estimates.
    """
    # Extract the first metric and its orientation using an iterator
    first_metric, first_orient = next(iter(metrics.items()))
    base_col_first = results[f"Base-model_{first_metric}"]

    # Calculate the raw mean improvements for the first metric to establish sorting
    # order
    first_diffs = {}
    for method in methods:
        method_col = results[f"{method}_{first_metric}"]

        # Orient differences so that higher is ALWAYS better
        if first_orient == "min":
            diffs = base_col_first - method_col
        else:  # "max"
            diffs = method_col - base_col_first

        first_diffs[method] = diffs.mean()

    # Sort methods based on the computed improvements
    sorted_methods = sorted(methods, key=lambda m: first_diffs[m], reverse=True)

    if use_num_datasets:
        N = len(results.dataset.unique())
    else:
        N = len(results)

    # Build the table using the newly sorted methods list
    table_data = {}

    for metric, orient in metrics.items():
        base_col = results[f"Base-model_{metric}"]

        mean_diffs = {}
        ci_margins = {}

        for method in methods:
            method_col = results[f"{method}_{metric}"]

            # Orient differences so that higher is ALWAYS better
            if orient == "min":
                diff_series = (base_col - method_col) * factor
            else:  # "max"
                diff_series = (method_col - base_col) * factor

            # Extract statistics for CI calculation
            mean_diff = diff_series.mean()
            std_diff = diff_series.std()

            # Calculate the 95% CI margin (1.96 * Standard Error)
            # Re-added np.sqrt() to N for correct standard error math
            ci_margin = 1.96 * (std_diff / np.sqrt(N)) if N > 0 else 0.0

            mean_diffs[method] = mean_diff
            ci_margins[method] = ci_margin

        mean_diff_series = pd.Series(mean_diffs)

        # Calculate ranks. Higher is always better now, so max number gets rank 1.
        ranks = mean_diff_series.rank(ascending=False, method="min").astype(int)

        formatted_col = []
        for m in sorted_methods:
            mean_val = mean_diffs[m]
            ci_val = ci_margins[m]
            rank_val = ranks[m]

            # If the number is positive or exactly zero, add the invisible minus sign
            if mean_val >= 0:
                mean_str = f"\\phantom{{-}}{mean_val:.{precision}f}"
            else:
                # Python naturally includes the minus sign for negative numbers
                mean_str = f"{mean_val:.{precision}f}"

            # Combine everything using the LaTeX sizing trick
            cell_string = (
                f"${mean_str}$ {{\\tiny $\\pm {ci_val:.{precision}f}$}} (\\#{rank_val})"
            )
            formatted_col.append(cell_string)

        # Store in our dictionary with a capitalized column name
        table_data[metric.capitalize()] = formatted_col

    # Create the final DataFrame with the sorted index
    table = pd.DataFrame(table_data, index=sorted_methods)
    table.index.name = "Method"

    return table


### Plotting ###


def plot_leaderboard(
    leaderboard,
    title=None,
    savefile=None,
    y_label=None,
    y_lim=None,
    color="blue",
    figsize=(7, 4),
    xticks_rotation=40,
):
    # Color palette
    if color == "blue":
        c = "#2066a8"
    elif color == "green":
        c = "#277a3e"
    elif color == "red":
        c = "#ae282c"
    elif color == "orange":
        c = "#cc5a14"
    else:
        print("[!] Unknwon color, choose from ['blue', 'green', 'red', 'orange']")

    # Separate the Base-model from the rest of the table
    has_base = "Base-model" in leaderboard.index
    if has_base:
        base_stats = leaderboard.loc["Base-model"]
        plot_leaderboard = leaderboard.drop(index="Base-model")
    else:
        plot_leaderboard = leaderboard.copy()

    # Calculate error margins for the models being plotted as bars
    plot_leaderboard = plot_leaderboard.sort_values(by="score", ascending=True)
    errors = np.array(
        [
            plot_leaderboard["score"] - plot_leaderboard["lower"],
            plot_leaderboard["upper"] - plot_leaderboard["score"],
        ]
    )

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        plot_leaderboard.index,
        plot_leaderboard["score"],
        yerr=errors,
        color=c,
        capsize=2,
        edgecolor="black",
        linewidth=1.0,
        error_kw={
            "elinewidth": 0.5,
            "capthick": 0.5,
            "ecolor": "gray",
        },
    )

    # Plot the Base-model as a horizontal line
    if has_base:
        ax.axhline(
            y=base_stats["score"],
            color="black",
            linestyle="-.",
            linewidth=1.5,
            label="Base-model",
        )

        ax.axhspan(
            ymin=base_stats["lower"],
            ymax=base_stats["upper"],
            color="black",
            alpha=0.1,
            linewidth=0,
        )

        ax.legend(loc="best", frameon=False)

    ax.yaxis.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    if y_label is not None:
        ax.set_ylabel(y_label)

    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])
    else:
        min_val = leaderboard["lower"].min()
        max_val = leaderboard["upper"].max()
        y_padding = (max_val - min_val) * 0.15
        ax.set_ylim(min_val - y_padding, max_val + y_padding)

    if title is not None:
        ax.set_title(title)

    plt.xticks(rotation=xticks_rotation, ha="right")

    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile, bbox_inches="tight")

    plt.show()


def plot_grouped_leaderboards(
    leaderboards_dict,
    title=None,
    y_label=None,
    y_lim=None,
    savefile=None,
    figsize=(10, 4),
    xticks_rotation=40,
    color="blue",
):
    """
    Plots a grouped bar chart for several benchmarks.
    leaderboards_dict: dict of format
    {"Benchmark 1 Name": df1, "Benchmark 2 Name": df2, ...}
    """

    # Separate the Base-model from the rest of the tables
    base_stats_dict = {}
    processed_tables = {}

    for b_name, table in leaderboards_dict.items():
        if "Base-model" in table.index:
            base_stats_dict[b_name] = table.loc["Base-model"]
            processed_tables[b_name] = table.drop(index="Base-model")
        else:
            processed_tables[b_name] = table.copy()

    # Get a unique list of all methods across all tables
    all_methods = set()
    for table in processed_tables.values():
        all_methods.update(table.index.tolist())
    all_methods = list(all_methods)

    # Sort methods by their average "Mean Win Rate" across all benchmarks
    sort_metrics = []
    for method in all_methods:
        mean_val = np.mean(
            [
                table.loc[method, "score"]
                for table in processed_tables.values()
                if method in table.index
            ]
        )
        sort_metrics.append((method, mean_val))

    sort_metrics.sort(key=lambda x: x[1])
    sorted_methods = [x[0] for x in sort_metrics]

    fig, ax = plt.subplots(figsize=figsize)

    # Define plotting parameters for grouped bars
    n_benchmarks = len(processed_tables)
    width = 0.8 / n_benchmarks
    x_positions = np.arange(len(sorted_methods))

    # Color palette
    if color == "blue":
        colors = ["#2066a8", "#8ec1da", "#cde1ec"]
    elif color == "green":
        colors = ["#277a3e", "#7fbf86", "#cce6d0"]
    elif color == "red":
        colors = ["#ae282c", "#d47264", "#f6d6c2"]
    elif color == "orange":
        colors = ["#cc5a14", "#e8955c", "#fae0ce"]
    else:
        print("[!] Unknwon color, choose from ['blue', 'green', 'red', 'orange']")

    for i, (b_name, table) in enumerate(processed_tables.items()):
        color = colors[i % len(colors)]

        # Calculate offset so groups are centered on the ticks
        offset = (i - n_benchmarks / 2 + 0.5) * width

        # Extract data strictly in the order of sorted_methods
        means = []
        err_lower = []
        err_upper = []

        for method in sorted_methods:
            if method in table.index:
                m = table.loc[method, "score"]
                l = m - table.loc[method, "lower"]
                u = table.loc[method, "upper"] - m
                means.append(m)
                err_lower.append(l)
                err_upper.append(u)
            else:
                # Handle cases where a method isn't in this specific benchmark
                means.append(0)
                err_lower.append(0)
                err_upper.append(0)

        errors = np.array([err_lower, err_upper])

        # Plot the grouped bars
        ax.bar(
            x_positions + offset,
            means,
            width=width,
            yerr=errors,
            label=b_name,
            color=color,
            capsize=2,
            edgecolor="black",
            linewidth=0.0,
            error_kw={
                "elinewidth": 0.5,
                "capthick": 0.5,
                "ecolor": "gray",
            },
        )

        # Plot the Base-model as horizontal lines with matching colors
        if b_name in base_stats_dict:
            base_stats = base_stats_dict[b_name]
            ax.axhline(
                y=base_stats["score"],
                color=color,
                linestyle="-.",
                linewidth=1.5,
                alpha=1.0,
            )

            ax.axhspan(
                ymin=base_stats["lower"],
                ymax=base_stats["upper"],
                color=color,
                alpha=0.1,
                linewidth=0,
            )

    ax.set_xlim(-0.6, len(sorted_methods) - 0.5)

    ax.yaxis.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # Set x-ticks exactly at the center of the groups
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_methods, rotation=xticks_rotation, ha="right")

    # Legend handling: Add the base-model indicator if applicable
    handles, labels = ax.get_legend_handles_labels()
    if len(base_stats_dict) > 0:
        base_line = mlines.Line2D(
            [], [], color="black", linestyle="-.", label="Base-model"
        )
        handles.append(base_line)
        labels.append("Base-model")

    ax.legend(handles=handles, labels=labels, loc="best", frameon=False)

    if y_label is not None:
        ax.set_ylabel(y_label)

    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile, bbox_inches="tight")

    plt.show()


def plot_cd_diagram(
    results,
    methods,
    metric="brier",
    groupby_dataset=False,
    figsize=(9, 3),
    title=None,
    savefile=None,
):
    df = pd.DataFrame(None)
    for method in methods:
        df[method] = results[f"{method}_{metric}"] - results[f"Base-model_{metric}"]

    if groupby_dataset:
        df["dataset"] = results["dataset"]
        df = df.groupby("dataset").mean().reset_index()
        df = df.drop("dataset", axis=1)

    stat, p_value = friedmanchisquare(*[df[col] for col in df.columns])

    print(f"\nFriedman Test Statistic: {stat:.3f}")
    print(f"P-Value: {p_value:.5e}")

    if p_value < 0.05:
        print("Result: Significant difference found. Proceeding to Nemenyi test.")
    else:
        print("Result: No significant difference found. Stop here.")

    nemenyi_results = sp.posthoc_nemenyi_friedman(df)

    ranks = df.rank(axis=1, method="average")
    avg_ranks = ranks.mean(axis=0)

    plt.figure(figsize=figsize)

    sp.critical_difference_diagram(
        ranks=avg_ranks, sig_matrix=nemenyi_results, label_props={"fontweight": "bold"}
    )

    if title is not None:
        plt.title(title)

    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile, bbox_inches="tight")

    plt.show()
