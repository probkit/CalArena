from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tabarena import EvaluationRepository
from tabarena.nips2025_utils.artifacts import tabarena_method_metadata_collection

HERE = Path(__file__).parent


def find_best_config_tabarena(repo, dataset, fold, configs):
    metrics = repo.metrics(datasets=[dataset], configs=configs, folds=[fold])
    return metrics.loc[metrics.metric_error_val.idxmin()].name[2]


# On the leaderboard https://huggingface.co/spaces/TabArena/leaderboard:
# Models (with imputation) > All Repeats > Classification > All Datasets
# We select all models for which the tuned or default version has >= 1300 ELO, as of
# April 1st 2026 (this is not a joke).
tabarena_models = [
    "TabPFN-v2.6",
    "TabICLv2",
    "RealTabPFN-v2.5",
    "TabICL_GPU",
    "LimiX_GPU",
    "TabM_GPU",
    "RealMLP_GPU",
    "BetaTabPFN_GPU",
    "ModernNCA_GPU",
    "Mitra_GPU",
    "TabDPT_GPU",
]

tabarena_fold = 0

binary_experiments, multiclass_experiments = [], []

with (
    h5py.File(HERE / "tabarena-binary.h5", "w") as f_bin,
    h5py.File(HERE / "tabarena-multiclass.h5", "w") as f_multi,
):
    f_bin.attrs["source"] = "tabarena"
    f_bin.attrs["problem_type"] = "binary"
    f_multi.attrs["source"] = "tabarena"
    f_multi.attrs["problem_type"] = "multiclass"

    for model in tabarena_models:
        method_metadata = tabarena_method_metadata_collection.get_method_metadata(
            method=model
        )

        if not method_metadata.path_processed_exists:
            print(
                f"Downloading processed data to {method_metadata.path_processed} ... "
                f"Ensure you have a fast internet connection. This download can be up to 15 GB."
            )
            method_metadata.method_downloader().download_processed()

        repo: EvaluationRepository = method_metadata.load_processed()

        if method_metadata.method_type != "config":
            raise AssertionError(
                f"This only supports config methods. "
                f"(method={method_metadata.method!r}, method_type={method_metadata.method_type!r})"
            )

        configs = repo.configs()
        datasets = repo.datasets()
        for dataset in datasets:
            dataset_info = repo.dataset_info(dataset=dataset)

            if dataset_info["problem_type"] == "binary":
                tabarena_config = find_best_config_tabarena(
                    repo, dataset, tabarena_fold, configs
                )

                p_cal = repo.predict_val(
                    dataset=dataset,
                    fold=tabarena_fold,
                    config=tabarena_config,
                    binary_as_multiclass=False,
                )
                y_cal = repo.labels_val(dataset=dataset, fold=tabarena_fold)
                p_test = repo.predict_test(
                    dataset=dataset,
                    fold=tabarena_fold,
                    config=tabarena_config,
                    binary_as_multiclass=False,
                )
                y_test = repo.labels_test(dataset=dataset, fold=tabarena_fold)

                grp = f_bin.create_group(f"{dataset}/{model}")
                grp.create_dataset(
                    "probas_cal", data=p_cal.astype(np.float32), compression="gzip"
                )
                grp.create_dataset(
                    "labels_cal", data=y_cal.astype(np.int32), compression="gzip"
                )
                grp.create_dataset(
                    "probas_test", data=p_test.astype(np.float32), compression="gzip"
                )
                grp.create_dataset(
                    "labels_test", data=y_test.astype(np.int32), compression="gzip"
                )

                binary_experiments.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "cal_size": len(p_cal),
                        "test_size": len(p_test),
                        "tabarena_fold": tabarena_fold,
                        "tabarena_config": tabarena_config,
                    }
                )

            elif dataset_info["problem_type"] == "multiclass":
                tabarena_config = find_best_config_tabarena(
                    repo, dataset, tabarena_fold, configs
                )

                p_cal = repo.predict_val(
                    dataset=dataset, fold=tabarena_fold, config=tabarena_config
                )
                y_cal = repo.labels_val(dataset=dataset, fold=tabarena_fold)
                p_test = repo.predict_test(
                    dataset=dataset, fold=tabarena_fold, config=tabarena_config
                )
                y_test = repo.labels_test(dataset=dataset, fold=tabarena_fold)

                n_cal, k = p_cal.shape

                grp = f_multi.create_group(f"{dataset}/{model}")
                grp.create_dataset(
                    "probas_cal", data=p_cal.astype(np.float32), compression="gzip"
                )
                grp.create_dataset(
                    "labels_cal", data=y_cal.astype(np.int32), compression="gzip"
                )
                grp.create_dataset(
                    "probas_test", data=p_test.astype(np.float32), compression="gzip"
                )
                grp.create_dataset(
                    "labels_test", data=y_test.astype(np.int32), compression="gzip"
                )

                multiclass_experiments.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "cal_size": n_cal,
                        "test_size": len(y_test),
                        "n_classes": k,
                        "tabarena_fold": tabarena_fold,
                        "tabarena_config": tabarena_config,
                    }
                )

df_binary = pd.DataFrame(binary_experiments)
df_binary.to_csv(HERE / "tabarena-binary-experiments.csv", index=False)

df_multiclass = pd.DataFrame(multiclass_experiments)
df_multiclass.to_csv(HERE / "tabarena-multiclass-experiments.csv", index=False)
