"""
Generates the logit versions of the CV-multiclass and ImageNet-multiclass benchmarks
used in the logits vs. probabilities ablation.

For each benchmark, two HDF5 files are written to ablation_logits/data/:
  - {benchmark}-logits.h5        : raw model logits
  - {benchmark}-scaled-logits.h5 : logits - logsumexp(logits), i.e. log-probabilities
                                   computed without going through the softmax

Both files have the hierarchy {dataset}/{model}/ -> {logits_cal, labels_cal,
logits_test, labels_test}. The experiment lists are the ones of the main benchmark
(calibration_benchmarks/{benchmark}-experiments.csv).

Requires the raw CV logits under cv_data/ (see the main README).
"""

import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from safetensors import safe_open
from scipy.special import logsumexp

HERE = Path(__file__).parent
ROOT = HERE.parent
CV_DATA_DIR = ROOT / "cv_data"
DATA_DIR = HERE / "data"


### utils (same loaders as calibration_benchmarks/generate_cv_benchmarks.py) ###

def unpickle_logits(file):
    with open(file, "rb") as f:
        (logits_cal, labels_cal), (logits_test, labels_test) = pickle.load(f)
    return ((logits_cal, labels_cal), (logits_test, labels_test))


def load_safetensors_logits(file):
    with safe_open(file, framework="pt", device="cpu") as sf:
        tensors = {k: sf.get_tensor(k) for k in sf.keys()}
    logits_cal = tensors["logits_val"].detach().cpu().numpy()
    logits_test = tensors["logits_test"].detach().cpu().numpy()
    labels_cal = tensors["labels_val"].detach().cpu().numpy()
    labels_test = tensors["labels_test"].detach().cpu().numpy()
    return ((logits_cal, labels_cal), (logits_test, labels_test))


def load_cv_multiclass(dataset, model):
    if dataset in ["c10", "c100", "birds", "SVHN"]:
        return unpickle_logits(CV_DATA_DIR / "Markus" / f"{model}_{dataset}_logits.p")
    return load_safetensors_logits(CV_DATA_DIR / "Hekler" / dataset / f"{model}.safetensors")


def load_imagenet(dataset, model):
    if model in ["resnet152", "densenet161"]:
        return unpickle_logits(CV_DATA_DIR / "Markus" / f"{model}_imgnet_logits.p")
    return load_safetensors_logits(CV_DATA_DIR / "Hekler" / "imagenet" / f"{model}.safetensors")


def scale_logits(logits):
    return logits - logsumexp(logits, axis=1, keepdims=True)


### Generate benchmarks ###

DATA_DIR.mkdir(exist_ok=True)

for benchmark, loader in [
    ("cv-multiclass", load_cv_multiclass),
    ("imagenet-multiclass", load_imagenet),
]:
    experiments = pd.read_csv(ROOT / "calibration_benchmarks" / f"{benchmark}-experiments.csv")

    with (
        h5py.File(DATA_DIR / f"{benchmark}-logits.h5", "w") as f_raw,
        h5py.File(DATA_DIR / f"{benchmark}-scaled-logits.h5", "w") as f_scaled,
    ):
        for f in (f_raw, f_scaled):
            f.attrs["source"] = benchmark.split("-")[0]
            f.attrs["problem_type"] = "multiclass"

        for _, row in experiments.iterrows():
            dataset, model = row["dataset"], row["model"]
            print(f"{benchmark}: {dataset}/{model}")

            (logits_cal, labels_cal), (logits_test, labels_test) = loader(dataset, model)

            for f, transform in [(f_raw, lambda x: x), (f_scaled, scale_logits)]:
                grp = f.create_group(f"{dataset}/{model}")
                grp.create_dataset(
                    "logits_cal",
                    data=transform(logits_cal).astype(np.float32),
                    compression="gzip",
                )
                grp.create_dataset(
                    "labels_cal",
                    data=labels_cal.flatten().astype(np.int32),
                    compression="gzip",
                )
                grp.create_dataset(
                    "logits_test",
                    data=transform(logits_test).astype(np.float32),
                    compression="gzip",
                )
                grp.create_dataset(
                    "labels_test",
                    data=labels_test.flatten().astype(np.int32),
                    compression="gzip",
                )
