import os
import sys

# Set environment variables for thread control before numerical imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import h5py
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from probmetrics.calibrators import *
from probmetrics.metrics import Metrics

# ==========================================
# CALIBRATOR DEFINITIONS
# ==========================================

BINARY_CALIBRATORS = {
    # No calibration
    "Base-model": lambda: None,

    # Binning
    "Hist-uniform": lambda: BinaryHistogramBinningCalibrator(strategy="uniform"),
    "Hist-quantile": lambda: BinaryHistogramBinningCalibrator(strategy="quantile"),
    "Scaling-Binning": lambda: BinaryPlattBinnerCalibrator(),
    "BBQ": lambda: NetcalBBQCalibrator(),

    # Isotonic regression and derivatives
    "Isotonic": lambda: SklearnCalibrator(method="isotonic", cv="prefit"),
    "CIR": lambda: CenteredIsotonicRegressionCalibrator(),
    "Venn-Abers": lambda: BinaryVennAbersCalibrator(),
    "ENIR": lambda: NetcalENIRCalibrator(),  # TODO: Very slow for large datasets.

    # Logistic-based methods
    "TS": lambda: TemperatureScalingCalibrator(),
    "ETS": lambda: ETSCalibrator(),
    "Platt-probs": lambda: SklearnCalibrator(method="sigmoid", cv="prefit"),
    "Platt-logits": lambda: BinaryLogisticCalibrator(type="affine"),
    "Quadratic": lambda: BinaryLogisticCalibrator(type="quadratic"),
    "Beta": lambda: BetacalCalibrator(),

    # Other non-parametric methods
    "Spline": lambda: MLISplineCalibrator(),
    "CDF-Spline": lambda: CDFSplineCalibrator(),
    "Kernel": lambda: KernelCalibrator(),

    # Tree based
    "XGBoost": lambda: XGBoostCalibrator(),
    "LightGBM": lambda: LightGBMCalibrator(),
    "CatBoost": lambda: CatBoostCalibrator(),

    # CatBoost ablation experiment
    "CB": lambda: BinaryCatBoostCalibrator(),
    "CB-tiny": lambda: BinaryCatBoostCalibrator(tiny=True),
    "CB-monotone": lambda: BinaryCatBoostCalibrator(monotone=True),
    "CB-tiny-monotone": lambda: BinaryCatBoostCalibrator(tiny=True, monotone=True),
}

MULTICLASS_CALIBRATORS = {
    # No calibration
    "Base-model": lambda: None,

    # Binary methods applied in OvR fashion
    "Hist-uniform": lambda: MulticlassOneVsRestCalibrator(
        BinaryHistogramBinningCalibrator(strategy="uniform")
    ),
    "Hist-quantile": lambda: MulticlassOneVsRestCalibrator(
        BinaryHistogramBinningCalibrator(strategy="quantile")
    ),
    "Isotonic": lambda: SklearnCalibrator(method="isotonic", cv="prefit"),
    "CIR": lambda: MulticlassOneVsRestCalibrator(
        CenteredIsotonicRegressionCalibrator()
    ),
    "Venn-Abers": lambda: VennAbersCalibrator(),
    "BBQ": lambda: NetcalBBQCalibrator(),
    "Spline": lambda: MLISplineCalibrator(),

    # Natively multiclass methods
    "TS": lambda: TemperatureScalingCalibrator(),
    "ETS": lambda: ETSCalibrator(),
    "VS": lambda: VectorScalingCalibrator(),
    "SVS": lambda: SVSCalibrator(),
    "MS": lambda: MatrixScalingCalibrator(),
    "SMS": lambda: SMSCalibrator(),
    "Dirichlet": lambda: DirichletCalibrator(n_cv=0, reg_lambda=1e-3, reg_mu=1e-3),
    "Kernel": lambda: KernelCalibrator(),

    # Tree based
    "XGBoost": lambda: XGBoostCalibrator(),
    "LightGBM": lambda: LightGBMCalibrator(),
    "CatBoost": lambda: CatBoostCalibrator(),  # TODO very slow for high dim datasets.
}


def load_custom_calibrators() -> dict:
    try:
        from custom_calibrators import CUSTOM_CALIBRATORS

        return CUSTOM_CALIBRATORS
    except ImportError:
        return {}


def test_calibrator(cal, name, metrics, results, p_cal, y_cal, p_test, y_test):
    # If no calibrator specified, give results without post hoc calibration
    if cal is None:
        if len(p_test.shape) == 1:
            p_test = np.stack([1.0 - p_test, p_test], axis=1)
        elif p_test.shape[1] == 1:
            p_test = np.concatenate([1.0 - p_test, p_test], axis=1)
        metrics = metrics.compute_all_from_labels_probs(
            torch.tensor(y_test, dtype=torch.int64), torch.tensor(p_test)
        )
        results.update(
            {f"{name}_{key}": value.item() for key, value in metrics.items()}
        )
        results[f"{name}_time"] = 0
        return results

    # Some calibrators are fitted instantly but are slow at prediction time so we
    # measure this as well
    start = time.perf_counter()
    cal.fit(p_cal, y_cal)
    preds_test = cal.predict_proba(p_test)
    end = time.perf_counter()
    runtime = end - start

    metrics = metrics.compute_all_from_labels_probs(
        torch.tensor(y_test, dtype=torch.int64), torch.tensor(preds_test)
    )
    results.update({f"{name}_{key}": value.item() for key, value in metrics.items()})
    results[f"{name}_time"] = runtime

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run calibration benchmarks (Binary or Multiclass)."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Name of the benchmark to run, following the pattern '{name}-{modality}' (e.g., 'tabrepo-binary', 'cv-multiclass').",
    )
    parser.add_argument(
        "--calibrator",
        type=str,
        default=None,
        help="Name of the calibrator to run. Omit to run all calibrators for the given modality.",
    )
    parser.add_argument(
        "--benchmarks_dir",
        type=str,
        default="calibration_benchmarks",
        help="Base directory to use for loading benchmark files.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Base directory to use for saving benchmark results.",
    )
    args = parser.parse_args()

    benchmarks_dir = args.benchmarks_dir

    benchmark_name = args.benchmark
    benchmark_name = benchmark_name.lower()

    # Infer Modality
    try:
        modality = benchmark_name.split("-")[-1].lower()
        if modality not in ["binary", "multiclass"]:
            raise ValueError
    except ValueError:
        raise ValueError(
            f"Invalid benchmark name '{benchmark_name}'. Must end with '-binary' or '-multiclass'."
        )

    # Assign appropriate calibrator dictionary and metrics
    if modality == "binary":
        base_calibrators = BINARY_CALIBRATORS
        metrics_list = ["brier", "logloss", "kuiper", "ece-15", "accuracy"]
    else:
        base_calibrators = MULTICLASS_CALIBRATORS
        metrics_list = ["brier", "logloss", "ece-15", "accuracy"]

    metrics = Metrics.from_names(metrics_list)
    calibrator_factories = {**base_calibrators, **load_custom_calibrators()}

    # Filter specific calibrator if requested
    if args.calibrator is not None:
        if args.calibrator not in calibrator_factories:
            available = ", ".join(calibrator_factories.keys())
            raise ValueError(
                f"Unknown calibrator '{args.calibrator}' for modality '{modality}'. "
                f"Available calibrators: {available}"
            )
        calibrator_factories = {args.calibrator: calibrator_factories[args.calibrator]}

    # Standardize output directory layout
    # Yields e.g., results/tabrepo-binary/
    results_dir = Path(args.results_dir) / benchmark_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Note: File naming relies entirely on the `{name}-{modality}` benchmark convention.
    experiments_file = Path(f"{benchmarks_dir}/{benchmark_name}-experiments.csv")
    h5_file = Path(f"{benchmarks_dir}/{benchmark_name}.h5")

    # File Existence Check
    if not experiments_file.exists() or not h5_file.exists():
        print(f"\n[Error] Benchmark files not found in directory: '{benchmarks_dir}'")
        print(f"  Expected experiment file : {experiments_file}")
        print(f"  Expected HDF5 data file  : {h5_file}")
        print("\nPlease ensure the files exist, or specify the correct directory using the --benchmarks_dir argument.")
        print("Example: python run_benchmark.py --benchmark tabrepo-binary --benchmarks_dir /path/to/my/benchmarks")
        sys.exit(1)

    experiments = pd.read_csv(experiments_file)

    with h5py.File(h5_file, "r") as h5:
        for cal_name, factory in calibrator_factories.items():
            print(f"\n=== Running {benchmark_name} benchmark for {cal_name} calibrator ===")
            calibrator_results = []

            for _, row in tqdm(experiments.iterrows(), total=len(experiments)):
                dataset, model = row["dataset"], row["model"]

                grp = h5[f"{dataset}/{model}"]
                p_cal = grp["probas_cal"][:]
                y_cal = grp["labels_cal"][:]
                p_test = grp["probas_test"][:]
                y_test = grp["labels_test"][:]

                # Extract dimensions contextually based on inferred modality
                results = {
                    "dataset": dataset,
                    "model": model,
                    "cal_size": len(p_cal),
                    "test_size": len(p_test),
                }

                if modality == "multiclass":
                    # Assumes p_cal shape is (n_samples, n_classes)
                    results["n_classes"] = p_cal.shape[1]

                cal = factory()
                results = test_calibrator(
                    cal, cal_name, metrics, results, p_cal, y_cal, p_test, y_test
                )

                calibrator_results.append(results)

            df = pd.DataFrame(calibrator_results)
            df.to_csv(results_dir / f"{cal_name}.csv", index=False)


if __name__ == "__main__":
    main()
