from functools import lru_cache
import logging

import lightgbm as lgb
import numpy as np

from model.regression_objective import (
    lightgbm_asymmetric_huber_objective,
)
from model.models.generic.sklearn_adapter import SklearnEstimatorAdapter


@lru_cache(maxsize=1)
def _lightgbm_gpu_available() -> bool:
    probe = lgb.LGBMClassifier(
        n_estimators=1,
        max_bin=15,
        min_data_in_bin=1,
        min_data_in_leaf=1,
        device_type="gpu",
        verbosity=-1,
        n_jobs=1,
    )
    try:
        probe.fit(
            np.array([[0.0], [1.0], [0.2], [0.8]], dtype=np.float32),
            np.array([0, 1, 0, 1]),
        )
    except lgb.basic.LightGBMError:
        return False
    return True


class LightGBMAdapter(SklearnEstimatorAdapter):
    """LightGBM adapter for classification and robust regression tasks."""

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
        device_type = "cpu"
        if self.device.type == "cuda":
            if _lightgbm_gpu_available():
                device_type = "gpu"
            else:
                logging.getLogger(__name__).warning(
                    "LightGBM GPU training is unavailable; falling back to CPU"
                )
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
        if self.n_classes == 1:
            estimator = lgb.LGBMRegressor(
                **hp,
                objective=lightgbm_asymmetric_huber_objective,
                metric="None",
                device_type=device_type,
                n_jobs=-1,
                verbosity=-1,
            )
            return estimator, hp

        estimator = lgb.LGBMClassifier(
            **hp, device_type=device_type, n_jobs=-1, verbosity=-1
        )
        return estimator, hp
