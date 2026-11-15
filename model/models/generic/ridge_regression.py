"""Linear Ridge regression for flattened time-series windows."""

from sklearn.linear_model import Ridge

from model.models.generic.sklearn_adapter import SklearnEstimatorAdapter


class RidgeRegressionAdapter(SklearnEstimatorAdapter):
    MODEL_TYPE = "ridge_regression"
    MODEL_VERSION = 1

    def __init__(self, *args, n_classes: int = 1, use_scaler: bool = True, **kwargs):
        if n_classes != 1:
            raise ValueError("Ridge regression requires one output")
        super().__init__(*args, n_classes=n_classes, use_scaler=use_scaler, **kwargs)

    def _build_estimator(
        self,
        alpha: float = 10.0,
        solver: str = "lsqr",
        max_iter: int = 1000,
        tol: float = 1e-3,
        **kwargs,
    ):
        if kwargs:
            raise ValueError(f"Unsupported Ridge parameters: {sorted(kwargs)}")
        if alpha < 0 or max_iter < 1 or tol <= 0:
            raise ValueError("Ridge alpha, max_iter and tol must be valid")
        hp = {"alpha": alpha, "solver": solver, "max_iter": max_iter, "tol": tol}
        return Ridge(**hp), hp
