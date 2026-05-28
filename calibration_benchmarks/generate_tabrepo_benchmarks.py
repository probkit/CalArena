from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from tabrepo import load_repository

HERE = Path(__file__).parent


def find_best_config_tabrepo(repo, dataset, fold, model):
    configs = [c for c in repo.configs() if c.split("_")[0] == model]
    metrics = repo.metrics(datasets=[dataset], configs=configs, folds=[fold])
    return metrics.loc[metrics.metric_error_val.idxmin()].name[2]


repo = load_repository("D244_F3_C1530_200")

tabrepo_models = [
    "CatBoost",
    "ExtraTrees",
    # "FTTransformer",  # Missing data
    # "KNeighbors",     # Missing data
    "LightGBM",
    "LinearModel",
    "NeuralNetFastAI",
    "NeuralNetTorch",
    "RandomForest",
    "XGBoost",
]

tabrepo_fold = 0

binary_experiments, multiclass_experiments = [], []

with (
    h5py.File(HERE / "tabrepo-binary.h5", "w") as f_bin,
    h5py.File(HERE / "tabrepo-multiclass.h5", "w") as f_multi,
):
    f_bin.attrs["source"] = "tabrepo"
    f_bin.attrs["problem_type"] = "binary"
    f_multi.attrs["source"] = "tabrepo"
    f_multi.attrs["problem_type"] = "multiclass"

    for dataset in tqdm(repo.datasets()):
        problem_type = repo.dataset_info(dataset)["problem_type"]

        if (
            problem_type == "binary" and dataset != "MiniBooNE"
        ):  # TabRepo returns an error for this dataset.
            for model in tabrepo_models:
                tabrepo_config = find_best_config_tabrepo(
                    repo=repo, dataset=dataset, fold=tabrepo_fold, model=model
                )

                p_cal = repo.predict_val(
                    dataset=dataset,
                    fold=tabrepo_fold,
                    config=tabrepo_config,
                    binary_as_multiclass=False,
                )
                y_cal = repo.labels_val(dataset=dataset, fold=tabrepo_fold)
                p_test = repo.predict_test(
                    dataset=dataset,
                    fold=tabrepo_fold,
                    config=tabrepo_config,
                    binary_as_multiclass=False,
                )
                y_test = repo.labels_test(dataset=dataset, fold=tabrepo_fold)

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
                        "tabrepo_fold": tabrepo_fold,
                        "tabrepo_config": tabrepo_config,
                    }
                )

        elif problem_type == "multiclass" and dataset not in (
            "jannis",
            "kropt",
            "shuttle",
        ):  # TabRepo returns an error for these 3 datasets.
            for model in tabrepo_models:
                tabrepo_config = find_best_config_tabrepo(
                    repo=repo, dataset=dataset, fold=tabrepo_fold, model=model
                )

                p_cal = repo.predict_val(
                    dataset=dataset, fold=tabrepo_fold, config=tabrepo_config
                )
                y_cal = repo.labels_val(dataset=dataset, fold=tabrepo_fold)
                p_test = repo.predict_test(
                    dataset=dataset, fold=tabrepo_fold, config=tabrepo_config
                )
                y_test = repo.labels_test(dataset=dataset, fold=tabrepo_fold)

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
                        "tabrepo_fold": tabrepo_fold,
                        "tabrepo_config": tabrepo_config,
                    }
                )

df_binary = pd.DataFrame(binary_experiments)
df_binary.to_csv(HERE / "tabrepo-binary-experiments.csv", index=False)

df_multiclass = pd.DataFrame(multiclass_experiments)
df_multiclass.to_csv(HERE / "tabrepo-multiclass-experiments.csv", index=False)
