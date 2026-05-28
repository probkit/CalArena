import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

import h5py
import pickle
import numpy as np
import pandas as pd
from safetensors import safe_open
from scipy.special import logsumexp

CV_DATA_DIR = HERE.parent / "cv_data"


### utils ###

def softmax(logits: np.ndarray) -> np.ndarray:
    l = logits - np.max(logits, axis=1, keepdims=True)
    exp_l = np.exp(l)
    return exp_l / np.sum(exp_l, axis=1, keepdims=True)


def unpickle_logits(file):
    with open(file, "rb") as f:
        (logits_cal, labels_cal), (logits_test, labels_test) = pickle.load(f)
    return ((logits_cal, labels_cal), (logits_test, labels_test))


def convert_cifar10_to_binary(logits, labels):
    """
    Converts 10-class CIFAR-10 logits and labels to Binary (Animal vs Machine).
    Animal = Class 1, Machine = Class 0.
    """
    machine_idx = [0, 1, 8, 9]
    animal_idx = [2, 3, 4, 5, 6, 7]

    binary_labels = np.zeros_like(labels)
    binary_labels[np.isin(labels, animal_idx)] = 1

    logits_animals = logits[:, animal_idx]
    logits_machines = logits[:, machine_idx]

    binary_logits = logsumexp(logits_animals, axis=1) - logsumexp(
        logits_machines, axis=1
    )
    binary_probas = 1 / (1 + np.exp(-binary_logits))

    return binary_probas, binary_labels


### Generate binary benchmark ###

# The CV experiment CSVs are pre-committed and serve as input (list of what to process).
# We write the HDF5 and overwrite the CSV with updated sizes.
binary_configs = pd.read_csv(HERE / "cv-binary-experiments.csv")

binary_experiments = []

with h5py.File(HERE / "cv-binary.h5", "w") as f:
    f.attrs["source"] = "cv"
    f.attrs["problem_type"] = "binary"

    for _, row in binary_configs.iterrows():
        dataset, model = row["dataset"], row["model"]

        if dataset == "c10":
            (logits_cal, labels_cal), (logits_test, labels_test) = unpickle_logits(
                CV_DATA_DIR / "Markus" / f"{model}_c10_logits.p"
            )
            probas_cal, labels_cal = convert_cifar10_to_binary(logits_cal, labels_cal)
            probas_test, labels_test = convert_cifar10_to_binary(
                logits_test, labels_test
            )
        else:
            with safe_open(
                CV_DATA_DIR / "Hekler" / dataset / f"{model}.safetensors",
                framework="pt",
                device="cpu",
            ) as sf:
                tensors = {k: sf.get_tensor(k) for k in sf.keys()}
            logits_cal = tensors["logits_val"].detach().cpu().numpy()
            logits_test = tensors["logits_test"].detach().cpu().numpy()
            labels_cal = tensors["labels_val"].detach().cpu().numpy()
            labels_test = tensors["labels_test"].detach().cpu().numpy()
            probas_cal = softmax(logits_cal)[:, 1]
            probas_test = softmax(logits_test)[:, 1]

        grp = f.create_group(f"{dataset}/{model}")
        grp.create_dataset(
            "probas_cal", data=probas_cal.astype(np.float32), compression="gzip"
        )
        grp.create_dataset(
            "labels_cal", data=labels_cal.flatten().astype(np.int32), compression="gzip"
        )
        grp.create_dataset(
            "probas_test", data=probas_test.astype(np.float32), compression="gzip"
        )
        grp.create_dataset(
            "labels_test",
            data=labels_test.flatten().astype(np.int32),
            compression="gzip",
        )

        binary_experiments.append(
            {
                "dataset": dataset,
                "model": model,
                "cal_size": len(probas_cal),
                "test_size": len(probas_test),
            }
        )

df_binary = pd.DataFrame(binary_experiments)
df_binary.to_csv(HERE / "cv-binary-experiments.csv", index=False)


### Generate multiclass benchmark ###

multiclass_configs = pd.read_csv(HERE / "cv-multiclass-experiments.csv")

multiclass_experiments = []

with h5py.File(HERE / "cv-multiclass.h5", "w") as f:
    f.attrs["source"] = "cv"
    f.attrs["problem_type"] = "multiclass"

    for _, row in multiclass_configs.iterrows():
        dataset, model = row["dataset"], row["model"]

        if dataset in ["c10", "c100", "birds", "SVHN"]:
            (logits_cal, labels_cal), (logits_test, labels_test) = unpickle_logits(
                CV_DATA_DIR / "Markus" / f"{model}_{dataset}_logits.p"
            )
        elif dataset in ["derma", "oct"]:
            with safe_open(
                CV_DATA_DIR / "Hekler" / dataset / f"{model}.safetensors",
                framework="pt",
                device="cpu",
            ) as sf:
                tensors = {k: sf.get_tensor(k) for k in sf.keys()}
            logits_cal = tensors["logits_val"].detach().cpu().numpy()
            logits_test = tensors["logits_test"].detach().cpu().numpy()
            labels_cal = tensors["labels_val"].detach().cpu().numpy()
            labels_test = tensors["labels_test"].detach().cpu().numpy()

        probas_cal = softmax(logits_cal)
        probas_test = softmax(logits_test)

        n_cal, k = probas_cal.shape

        grp = f.create_group(f"{dataset}/{model}")
        grp.create_dataset(
            "probas_cal", data=probas_cal.astype(np.float32), compression="gzip"
        )
        grp.create_dataset(
            "labels_cal", data=labels_cal.flatten().astype(np.int32), compression="gzip"
        )
        grp.create_dataset(
            "probas_test", data=probas_test.astype(np.float32), compression="gzip"
        )
        grp.create_dataset(
            "labels_test",
            data=labels_test.flatten().astype(np.int32),
            compression="gzip",
        )

        multiclass_experiments.append(
            {
                "dataset": dataset,
                "model": model,
                "cal_size": n_cal,
                "test_size": len(labels_test),
                "n_classes": k,
            }
        )

df_multiclass = pd.DataFrame(multiclass_experiments)
df_multiclass.to_csv(HERE / "cv-multiclass-experiments.csv", index=False)


### Generate imagenet benchmark ###

imagenet_configs = pd.read_csv(HERE / "imagenet-multiclass-experiments.csv")

imagenet_experiments = []

with h5py.File(HERE / "imagenet-multiclass.h5", "w") as f:
    f.attrs["source"] = "imagenet"
    f.attrs["problem_type"] = "multiclass"

    for _, row in imagenet_configs.iterrows():
        model = row["model"]

        if model in ["resnet152", "densenet161"]:
            (logits_cal, labels_cal), (logits_test, labels_test) = unpickle_logits(
                CV_DATA_DIR / "Markus" / f"{model}_imgnet_logits.p"
            )
        else:
            with safe_open(
                CV_DATA_DIR / "Hekler" / "imagenet" / f"{model}.safetensors",
                framework="pt",
                device="cpu",
            ) as sf:
                tensors = {k: sf.get_tensor(k) for k in sf.keys()}
            logits_cal = tensors["logits_val"].detach().cpu().numpy()
            logits_test = tensors["logits_test"].detach().cpu().numpy()
            labels_cal = tensors["labels_val"].detach().cpu().numpy()
            labels_test = tensors["labels_test"].detach().cpu().numpy()

        probas_cal = softmax(logits_cal)
        probas_test = softmax(logits_test)

        n_cal, k = probas_cal.shape

        # imagenet has a single dataset; we use "imagenet" as the dataset key for
        # consistency with other benchmarks
        grp = f.create_group(f"imagenet/{model}")
        grp.create_dataset(
            "probas_cal", data=probas_cal.astype(np.float32), compression="gzip"
        )
        grp.create_dataset(
            "labels_cal", data=labels_cal.flatten().astype(np.int32), compression="gzip"
        )
        grp.create_dataset(
            "probas_test", data=probas_test.astype(np.float32), compression="gzip"
        )
        grp.create_dataset(
            "labels_test",
            data=labels_test.flatten().astype(np.int32),
            compression="gzip",
        )

        imagenet_experiments.append(
            {
                "dataset": "imagenet",
                "model": model,
                "cal_size": n_cal,
                "test_size": len(labels_test),
                "n_classes": k,
            }
        )

df_imagenet = pd.DataFrame(imagenet_experiments)
df_imagenet.to_csv(HERE / "imagenet-multiclass-experiments.csv", index=False)
