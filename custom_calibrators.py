"""
Register your custom calibrators here to include them in the benchmark.

Each entry in CUSTOM_CALIBRATORS maps a display name (used in plot legends and
result filenames) to a factory function that returns a fresh calibrator instance.

A calibrator must implement two methods:
  - fit(p_cal: np.ndarray, y_cal: np.ndarray) -> None
  - predict_proba(p_test: np.ndarray) -> np.ndarray

The easiest way to get the interface right is to inherit from the base probmetrics
class:

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

But any object that implements .fit() and .predict_proba() will work.

Usage in the benchmark:
    python run_benchmark.py --benchmark tabrepo-binary --calibrator MyCalibrator
"""

# from my_module import MyCalibrator

CUSTOM_CALIBRATORS = {
    # "MyCalibrator": lambda: MyCalibrator(),
}
