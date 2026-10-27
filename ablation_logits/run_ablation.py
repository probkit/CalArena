"""
Logits vs. probabilities ablation.

Fits logit-based multiclass calibrators (TS, VS, MS) on raw logits or scaled logits
(logits - logsumexp(logits)) instead of the predicted probabilities used in the main
benchmark. The probabilities variant is the main benchmark itself
(results/{benchmark}/{calibrator}.csv).

Usage (from the repository root):
    python ablation_logits/run_ablation.py --benchmark cv-multiclass --calibrator TS --variant logits

Results are saved to ablation_logits/results/{benchmark}/{calibrator}-{variant}.csv
"""

import sys
import time
import argparse
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

# Importing run_benchmark also sets the thread-control environment variables.
from run_benchmark import MULTICLASS_CALIBRATORS

import h5py
import torch
import pandas as pd
from tqdm import tqdm
from probmetrics.distributions import CategoricalLogits
from probmetrics.metrics import Metrics

BENCHMARKS = ["cv-multiclass", "imagenet-multiclass"]
CALIBRATORS = ["TS", "VS", "MS"]
VARIANTS = ["logits", "scaled-logits"]


def test_calibrator_on_logits(cal, name, metrics, results, z_cal, y_cal, z_test, y_test):
    # The torch interface accepts logits directly, whereas fit() / predict_proba()
    # expect probabilities.
    start = time.perf_counter()
    cal.fit_torch(
        CategoricalLogits(torch.as_tensor(z_cal)),
        torch.as_tensor(y_cal, dtype=torch.long),
    )
    preds_test = cal.predict_proba_torch(
        CategoricalLogits(torch.as_tensor(z_test))
    ).get_probs()
    end = time.perf_counter()
    runtime = end - start

    metrics = metrics.compute_all_from_labels_probs(
        torch.tensor(y_test, dtype=torch.int64), preds_test
    )
    results.update({f"{name}_{key}": value.item() for key, value in metrics.items()})
    results[f"{name}_time"] = runtime

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run the logits vs. probabilities ablation."
    )
    parser.add_argument("--benchmark", type=str, required=True, choices=BENCHMARKS)
    parser.add_argument("--calibrator", type=str, required=True, choices=CALIBRATORS)
    parser.add_argument("--variant", type=str, required=True, choices=VARIANTS)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(HERE / "data"),
        help="Directory containing the logit HDF5 files.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(HERE / "results"),
        help="Base directory to use for saving ablation results.",
    )
    args = parser.parse_args()

    experiments_file = HERE.parent / "calibration_benchmarks" / f"{args.benchmark}-experiments.csv"
    h5_file = Path(args.data_dir) / f"{args.benchmark}-{args.variant}.h5"

    if not h5_file.exists():
        print(f"\n[Error] Logit benchmark file not found: {h5_file}")
        print("Generate it with: python ablation_logits/generate_logits_benchmarks.py")
        sys.exit(1)

    results_dir = Path(args.results_dir) / args.benchmark
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = Metrics.from_names(["brier", "logloss", "ece-15", "accuracy"])
    name = f"{args.calibrator}-{args.variant}"
    factory = MULTICLASS_CALIBRATORS[args.calibrator]

    experiments = pd.read_csv(experiments_file)
    calibrator_results = []

    print(f"\n=== Running {args.benchmark} ablation for {name} ===")

    with h5py.File(h5_file, "r") as h5:
        for _, row in tqdm(experiments.iterrows(), total=len(experiments)):
            dataset, model = row["dataset"], row["model"]

            grp = h5[f"{dataset}/{model}"]
            z_cal = grp["logits_cal"][:]
            y_cal = grp["labels_cal"][:]
            z_test = grp["logits_test"][:]
            y_test = grp["labels_test"][:]

            results = {
                "dataset": dataset,
                "model": model,
                "cal_size": len(z_cal),
                "test_size": len(z_test),
                "n_classes": z_cal.shape[1],
            }

            results = test_calibrator_on_logits(
                factory(), name, metrics, results, z_cal, y_cal, z_test, y_test
            )
            calibrator_results.append(results)

    df = pd.DataFrame(calibrator_results)
    df.to_csv(results_dir / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
