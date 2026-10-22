import lightgbm as lgb

from model.models.generic.sklearn_adapter import SklearnEstimatorAdapter


class LightGBMAdapter(SklearnEstimatorAdapter):
    """CPU LightGBM classifier for the single-head training pipeline."""

    MODEL_TYPE = "lightgbm"
    MODEL_VERSION = 1

    def __init__(self, *args, use_scaler: bool = False, **kwargs):
        super().__init__(*args, use_scaler=use_scaler, **kwargs)

    def _build_estimator(
        self,
        num_leaves: int = 31,
        max_depth: int = -1,
        n_estimators: int = 300,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        subsample_freq: int = 1,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        min_child_samples: int = 20,
        **kwargs,
    ):
        if kwargs:
            print(f"[LightGBMAdapter] Ignored kwargs: {list(kwargs.keys())}")
        hp = {
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "subsample_freq": subsample_freq,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "min_child_samples": min_child_samples,
        }
        estimator = lgb.LGBMClassifier(
            **hp,
            device_type="cpu",
            n_jobs=-1,
            verbosity=-1,
        )
        return estimator, hp
