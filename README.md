# CalArena

**A large-scale benchmark for post-hoc calibration of classification models.**

CalArena systematically evaluates 20 binary and 19 multiclass calibration methods across
hundreds of (dataset, model) pairs spanning classical tabular models, state-of-the-art
tabular foundation models, and computer vision networks. All calibrator implementations
are provided by the [probmetrics](https://github.com/probkit/probmetrics) package.

📄 Read the paper here: [CalArena: A Large-Scale Post-Hoc Calibration Benchmark](https://arxiv.org/abs/2605.30188)

## Leaderboards

**Binary Calibration Winrates**
![Binary Winrates Leaderboard](figures/binary_winrates.png)

**Multiclass Calibration Winrates**
![Multiclass Winrates Leaderboard](figures/multiclass_winrates.png)

Above, we plot results for binary and multiclass post-hoc calibration benchmarks.
Each bar represents the winrate of the calibration method, averaged over all experiments
in the benchmark with 95% Confidence Intervals (CIs) constructed by bootstrapping entire
datasets (TabRepo and TabArena-binary benchmarks) or experiments directly
(TabArena-multiclass and CV benchmarks).
Methods are ranked based on the average winrate over the three benchmarks.


## Repository structure

```
CalArena/
├── run_benchmark.py              # Run calibration experiments
├── utils.py                      # Analysis utilities and plotting helpers
├── custom_calibrators.py         # Register your own calibrators
├── paper_figures.ipynb           # Reproduce all paper figures
├── calibration_benchmarks/       # Benchmark data and generation scripts
│   ├── generate_tabrepo_benchmarks.py
│   ├── generate_tabarena_benchmarks.py
│   ├── generate_cv_benchmarks.py
│   └── {benchmark}-experiments.csv  # Benchmark experiments index
├── batch_scripts/                # SLURM job scripts for cluster execution
├── results/                      # Benchmark results (one CSV per calibrator)
│   └── {benchmark}/{calibrator}.csv
└── figures/                      # Paper figures (PDF)
```

> **Note:** The HDF5 files storing benchmark data (`calibration_benchmarks/*.h5`) are listed in `.gitignore` due to their size (~1.7 GB in total).
> Download them from [HuggingFace](https://huggingface.co/datasets/probkit/CalArena).


## Benchmarks overview

CalArena includes **7 benchmarks** across three data modalities:

| Benchmark | Problem type | Base models | Datasets | #Experiments |
|---|---|---|---|---|
| `tabrepo-binary` | Binary | 8 classical tabular models | 104 tabular datasets | 832 |
| `tabarena-binary` | Binary | 11 advanced tabular models | 30 tabular datasets | 314 |
| `cv-binary` | Binary | 9 deep CV models | CIFAR-10 (Animal vs Machine), Breast, Pneumonia | 13 |
| `tabrepo-multiclass` | Multiclass | 8 classical tabular models | 65 tabular datasets | 520 |
| `tabarena-multiclass` | Multiclass | 11 modern tabular models | 8 tabular datasets | 84 |
| `cv-multiclass` | Multiclass | 10 deep CV models | CIFAR-10/100, Birds, SVHN, Derma, OCT | 20 |
| `imagenet-multiclass` | Large scale multiclass | 8 deep CV models | ImageNet | 8 |

Each benchmark is stored as
- a single HDF5 file (`calibration_benchmarks/{benchmark}.h5`) with the hierarchy `{dataset}/{model}/` → `{probas_cal, labels_cal, probas_test, labels_test}`
- a companion `{benchmark}-experiments.csv` experiments index file listing dataset, model, calibration set size, test set size, number of classes

**TabRepo** base models: CatBoost, ExtraTrees, LightGBM, LinearModel, NeuralNetFastAI, NeuralNetTorch, RandomForest, XGBoost.

**TabArena** base models (≥ 1300 ELO on [TabArena leaderboard](https://huggingface.co/spaces/TabArena/leaderboard), as of April 1 2026, v0.1.3.1): TabPFN-v2.6, TabICLv2, RealTabPFN-v2.5, TabICL\_GPU, LimiX\_GPU, TabM\_GPU, RealMLP\_GPU, BetaTabPFN\_GPU, ModernNCA\_GPU, Mitra\_GPU, TabDPT\_GPU.


## Calibrators overview

💻 Implementations for every calibrator listed can be found in the [probmetrics](https://github.com/probkit/probmetrics) package.

### Binary (20 methods)

| Name | Method | Paper |
|---|---|---|
| `Base-model` | No calibration (raw model probabilities) | |
| `Hist-uniform` | Histogram binning with fixed sized bins (10 bins) | [1] |
| `Hist-quantile` | Histogram binning with fixed number of points per bin (10 bins) | [1] |
| `Scaling-Binning` | Platt scaling + histogram binning | [2] |
| `BBQ` | Bayesian Binning into Quantiles | [3] |
| `Isotonic` | Isotonic regression | [4] |
| `CIR` | Centered Isotonic Regression | [5] |
| `Venn-Abers` | Venn-Abers predictor | [6] |
| `TS` | Temperature Scaling | [7] |
| `ETS` | Ensemble Temperature Scaling | [8] |
| `Platt-probs` | Platt Scaling on top-class probabilities | [9] |
| `Platt-logits` | Platt Scaling on top-class logits | [9] |
| `Quadratic` | Quadratic logistic calibration | [10] |
| `Beta` | Beta calibration | [11] |
| `Spline` | Spline based calibration | [12] |
| `CDF-Spline` | Spline based calibration on the CDF | [13] |
| `Kernel` | Kernel calibration | [14] |
| `XGBoost` | Post-hoc calibration using a binary classifier | |
| `LightGBM` | Post-hoc calibration using a binary classifier | |
| `CatBoost` | Post-hoc calibration using a binary classifier | |

### Multiclass (19 methods)

| Name | Method | Paper |
|---|---|---|
| `Base-model` | No calibration | |
| `Hist-uniform` | Histogram binning with fixed sized bins (OvR) | [1] |
| `Hist-quantile` | Histogram binning with fixed number of points per bin (OvR) | [1] |
| `BBQ` | Bayesian Binning into Quantiles (OvR) | [3] |
| `Isotonic` | Isotonic regression (OvR) | [4] |
| `CIR` | Centered Isotonic Regression (OvR) | [5] |
| `Venn-Abers` | Venn-Abers predictor (OvR) | [6] |
| `Spline` | Spline based calibration (OvR) | [12] |
| `TS` | Temperature Scaling | [7] |
| `ETS` | Ensemble Temperature Scaling | [8] |
| `VS` | Vector Scaling | [7] |
| `SVS` | Structured Vector Scaling | [10] |
| `MS` | Matrix Scaling | [7] |
| `SMS` | Structured Matrix Scaling | [10] |
| `Dirichlet` | Dirichlet calibration | [15] |
| `Kernel` | Kernel calibration | [14] |
| `XGBoost` | Post-hoc calibration using a multiclass classifier | |
| `LightGBM` | Post-hoc calibration using a multiclass classifier | |
| `CatBoost` | Post-hoc calibration using a multiclass classifier | |

[1] [Obtaining calibrated probability estimates
from decision trees and naive Bayesian classifiers](https://cseweb.ucsd.edu/~elkan/calibrated.pdf)  
[2] [Verified Uncertainty Calibration](https://proceedings.neurips.cc/paper_files/paper/2019/hash/f8c0c968632845cd133308b1a494967f-Abstract.html)  
[3] [Obtaining Well Calibrated Probabilities Using Bayesian Binning](https://pubmed.ncbi.nlm.nih.gov/25927013/)  
[4] [Transforming Classifier Scores into Accurate Multiclass Probability Estimates](https://dl.acm.org/doi/pdf/10.1145/775047.775151)  
[5] [Centered Isotonic Regression: Point and Interval Estimation for Dose–Response Studies](https://www.tandfonline.com/doi/full/10.1080/19466315.2017.1286256)  
[6] [Large-scale probabilistic predictors with and without
guarantees of validity](https://proceedings.neurips.cc/paper/2015/file/a9a1d5317a33ae8cef33961c34144f84-Paper.pdf)  
[7] [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)  
[8] [Mix-n-Match : Ensemble and Compositional Methods for Uncertainty Calibration in Deep Learning](https://proceedings.mlr.press/v119/zhang20k.html)  
[9] [Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods](https://www.researchgate.net/profile/John-Platt-2/publication/2594015_Probabilistic_Outputs_for_Support_Vector_Machines_and_Comparisons_to_Regularized_Likelihood_Methods/links/004635154cff5262d6000000/Probabilistic-Outputs-for-Support-Vector-Machines-and-Comparisons-to-Regularized-Likelihood-Methods.pdf)  
[10] [Structured Matrix Scaling for Multi-Class Calibration](https://arxiv.org/abs/2511.03685)  
[11] [Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers](https://proceedings.mlr.press/v54/kull17a.html)  
[12] [Spline-Based Probability Calibration](https://arxiv.org/abs/1809.07751)  
[13] [Calibration of Neural Networks using Splines](https://openreview.net/forum?id=eQe8DEWNN2W)  
[14] [Consistent and Asymptotically Unbiased Estimation of Proper
Calibration Errors](https://proceedings.mlr.press/v238/popordanoska24a/popordanoska24a.pdf)  
[15] [Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with Dirichlet calibration](https://proceedings.neurips.cc/paper/2019/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html)  


## Installation

To run the benchmark, you need to install probmetrics with extra dependencies:

```bash
# "extra" and "dirichletcal" install other calibration package to support all
# calibrators in the library.
pip install 'probmetrics[extra,dirichletcal]'

# To load the benchmark data, you also need h5py
pip install h5py
```

For results analysis and plotting you need extra packages
```bash
pip install scikit-posthocs # Statistical analysis and CD diagrams
pip install arena-rank # To compute Elo scores
```


## Collecting benchmark data

### Downloading (recommended)

The benchmark files (HDF5 + experiment CSVs) are available on HuggingFace:

> **[https://huggingface.co/datasets/probkit/CalArena](https://huggingface.co/datasets/probkit/CalArena)**

Download the files and place them under `calibration_benchmarks/`.

### Generating the benchmarks from the original data sources (disk space hungry)

For the sake of completeness, we provide the scripts that we ran to generate the benchmarks from the original data sources.
Each script writes its output directly into `calibration_benchmarks/`, you only need to run these once.

```bash
python calibration_benchmarks/generate_tabrepo_benchmarks.py # requires installing tabrepo
python calibration_benchmarks/generate_tabarena_benchmarks.py # requires installing tabarena
python calibration_benchmarks/generate_cv_benchmarks.py # requires downloading the data from original sources
```

⚠️ **We strongly recommend using the files downloaded from HuggingFace as accessing model predictions from TabRepo and TabArena requires downloading hundred gigabytes of raw data locally.**

**TabRepo data:** Model predictions (we use context `D244_F3_C1530_200`, ~120 GB total) are downloaded automatically on first run. Ensure a fast internet connection.

**TabArena data:** Model predictions (~60 GB total) are downloaded automatically on first run. Ensure a fast internet connection.

**CV data:** `generate_cv_benchmarks.py` reads raw logits from `cv_data/` (gitignored). Place the Markus pickle files under `cv_data/Markus/` and the Hekler safetensors files under `cv_data/Hekler/` before running. The experiment-list CSVs (`cv-binary-experiments.csv`, etc.) are already committed and serve as the index of which datasets and models to process.


## Running the benchmarks

Once the HDF5 and CSV files are downloaded and stored under `calibration_benchmarks/`, you can run the benchmarks using the following python scripts.

### Run all calibrators

```bash
python run_benchmark.py --benchmark tabrepo-binary
```

Results are saved to `results/{benchmark}/{calibrator}.csv`, one file per calibrator.

### Run a single calibrator

```bash
python run_benchmark.py --benchmark tabrepo-binary --calibrator TS
```

Results are saved to `results/{benchmark}/{calibrator}.csv`.

### CLI reference

```
--benchmark       Name of the benchmark, e.g. tabrepo-binary, cv-multiclass   [required]
--calibrator      Name of a single calibrator to run (default: run all)
--benchmarks_dir  Directory containing benchmark HDF5/CSV files (default: calibration_benchmarks)
--results_dir     Directory to write results (default: results)
```


## Cluster execution (SLURM)

The recommended cluster strategy is to submit one job per calibrator so that runtimes are isolated.
Ready-to-use SLURM scripts for each benchmark are in `batch_scripts/`.
Before submitting, uncomment and adjust the `conda activate` line for your environment:

```bash
sbatch batch_scripts/tabrepo-binary.batch
```

Example script (abbreviated):

```bash
#!/bin/bash
#SBATCH --array=0-19
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%A_%a.out

CALIBRATORS=(
  "Base-model"
  "Hist-uniform" "Hist-quantile" "Scaling-Binning" "BBQ"
  "Isotonic" "CIR" "Venn-Abers"
  "TS" "ETS" "Platt" "Affine" "Quadratic" "Beta"
  "Spline" "CDF-Spline" "Kernel"
  "XGBoost" "LightGBM" "CatBoost"
)

source /home/$USER/.bashrc
cd $HOME/CalArena

python run_benchmark.py \
    --benchmark tabrepo-binary \
    --calibrator "${CALIBRATORS[$SLURM_ARRAY_TASK_ID]}"
```

SLURM output logs are written to `logs/` (gitignored).


## Adding your own calibrator

### Step 1: Implement the calibrator

A calibrator must expose two methods:

```python
def fit(self, p_cal: np.ndarray, y_cal: np.ndarray) -> None: ...
def predict_proba(self, p_test: np.ndarray) -> np.ndarray: ...
```

For binary tasks, `p_cal` / `p_test` are 1-D arrays of positive-class probabilities (or shape `(n, 2)`). For multiclass tasks they are `(n, k)` arrays.

The cleanest approach is to subclass the `probmetrics` base class, which accepts either a NumPy or PyTorch implementation:

```python
from probmetrics.calibrators import Calibrator

class MyCalibrator(Calibrator):
    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        '''Implement either this or the following'''
        ...

    def _fit_torch_impl(
        self, y_pred: CategoricalDistribution, y_true_labels: torch.Tensor
    ) -> None:
        '''Implement either this or the previous'''
        ...
    
    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        '''Implement either this or the following'''
        ...

    def _predict_proba_torch_impl(
        self, y_pred: CategoricalDistribution
    ) -> CategoricalDistribution:
        '''Implement either this or the previous'''
        ...
```

However, any object that satisfies `.fit()` / `.predict_proba()` works, inheriting from `Calibrator` is optional but recommend for smooth integration into the library.

### Step 2: Register it in `custom_calibrators.py`

```python
from my_module import MyCalibrator

CUSTOM_CALIBRATORS = {
    "MyCalibrator": lambda: MyCalibrator(),
}
```

The dictionary key is the **display name** used in plot legends and result filenames.

### Step 3: Run it

```bash
python run_benchmark.py --benchmark tabrepo-binary --calibrator MyCalibrator
```

Results are saved to `results/tabrepo-binary/MyCalibrator.csv`, exactly like a built-in method.


## Reproducing the paper figures

All figures are generated by [`paper_figures.ipynb`](paper_figures.ipynb).
Pre-generated PDFs are in `figures/`.

Results are loaded via `load_benchmark_results()` from [`utils.py`](utils.py):

```python
from utils import load_benchmark_results

df = load_benchmark_results(
    benchmark_name="tabrepo-binary",
    methods=["TS", "Platt", "Isotonic", "Base-model"],
)
```

`utils.py` also provides:
- `compute_winrates()` — mean win rates with 95% bootstrap CIs (dataset-level or experiment-level)
- `compute_elo_scores()` — Bradley-Terry Elo ratings with bootstrap CIs
- `compute_absolute_improvements()` — per-metric improvements over `Base-model`, LaTeX-formatted
- `plot_leaderboard()`, `plot_grouped_leaderboards()`, `plot_cd_diagram()` — reproducing publication figures


## Contributing a calibrator to `probmetrics`

All built-in calibrators live in the [probmetrics](https://github.com/probkit/probmetrics) package.
To contribute a new method and include it in the official benchmark, implement it there following the `Calibrator` interface and open a pull request.


## Citation

```bibtex
@article{calarena2026,
  title     = {CalArena: A Large-Scale Post-Hoc Calibration Benchmark},
  author    = {Eug{\`e}ne Berta and David Holzm{\"u}ller and Francis Bach and Michael I. Jordan},
  journal   = {arXiv preprint arXiv:2605.30188},
  year      = {2026},
  url       = {https://arxiv.org/abs/2605.30188},
}
```
