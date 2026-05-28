# Benchmark data

Pre-generated HDF5 files and experiment CSVs are available on HuggingFace:

> **[https://huggingface.co/datasets/probkit/CalArena](https://huggingface.co/datasets/probkit/CalArena)**

Download the `.h5` files and experiment CSVs (`*-experiments.csv`) and place them here.

To regenerate the HDF5 files from scratch, run the generation scripts from the repo root:

```bash
python calibration_benchmarks/generate_tabrepo_benchmarks.py
python calibration_benchmarks/generate_tabarena_benchmarks.py
python calibration_benchmarks/generate_cv_benchmarks.py
```

See the main [README](../README.md) for full instructions and data source requirements.
