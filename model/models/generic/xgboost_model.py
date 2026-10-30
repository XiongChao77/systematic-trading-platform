import logging
import math

import numpy as np
import xgboost as xgb

from model.regression_objective import (
    xgboost_asymmetric_huber_metric,
    xgboost_asymmetric_huber_objective,
)
from model.models.generic.sklearn_adapter import SklearnEstimatorAdapter


def positive_precision(y_true, y_score, sample_weight=None) -> float:
    """Measure unweighted positive-class precision at a fixed decision rule."""
    del sample_weight
    scores = np.asarray(y_score)
    predictions = (scores >= 0.5).astype(np.int64) if scores.ndim == 1 else scores.argmax(axis=1)
    positive_class = 1 if scores.ndim == 1 else scores.shape[1] - 1
    mask = predictions == positive_class
    return float(np.mean(np.asarray(y_true)[mask] == positive_class)) if mask.any() else 0.0


_ASYMMETRIC_HUBER_OBJECTIVE = "asymmetric_huber"
_CLASSIFICATION_EVAL_METRICS = {"logloss", "mlogloss"}
_REGRESSION_OBJECTIVE_METRICS = {
    "reg:squarederror": "rmse",
    "reg:absoluteerror": "mae",
    "reg:pseudohubererror": "mphe",
}


class XGBoostAdapter(SklearnEstimatorAdapter):
    """
    Single-estimator XGBoost adapter (sklearn API), plugged into the same
    single-task training flow as every other model (train_single /
    run_single_model_3class_task). Trees don't need feature scaling, so
    use_scaler defaults to False.

    Model version changes when estimator training semantics change because the
    serialized sklearn pipeline contains the fitted XGBoost objective.
    """

    MODEL_TYPE = "xgboost"
    MODEL_VERSION = 1

    def __init__(self, *args, use_scaler: bool = False, **kwargs):
        super().__init__(*args, use_scaler=use_scaler, **kwargs)
        self.best_iteration: int | None = None

    def _build_estimator(
        self,
        xgb_depth: int = 4,
        xgb_estimators: int = 300,
        xgb_early_stopping_rounds: int = 30,
        class_weight_power: float = 1.0,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        objective: str | None = None,
        tree_method: str = "hist",
        eval_metric: str = "logloss",
        **kwargs,
    ):
        xgb_device = self.device.type
        if xgb_device == "cuda" and not xgb.build_info().get("USE_CUDA", False):
            logging.getLogger(__name__).warning(
                "Selected CUDA device is unavailable in this XGBoost build; using CPU"
            )
            xgb_device = "cpu"
        if xgb_early_stopping_rounds < 1:
            raise ValueError("xgb_early_stopping_rounds must be positive")
        if not math.isfinite(class_weight_power) or not 0 <= class_weight_power <= 1:
            raise ValueError("class_weight_power must be finite and between 0 and 1")
        if not math.isfinite(min_child_weight) or min_child_weight < 0:
            raise ValueError("min_child_weight must be finite and non-negative")
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and non-negative")
        if self.n_classes != 1 and objective is not None:
            raise ValueError("objective is only configurable for regression models")
        if kwargs:
            print(f"[XGBoostAdapter] Ignored kwargs: {list(kwargs.keys())}")
        hp = {
            "xgb_depth": xgb_depth,
            "xgb_estimators": xgb_estimators,
            "xgb_early_stopping_rounds": xgb_early_stopping_rounds,
            "class_weight_power": class_weight_power,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "tree_method": tree_method,
            "eval_metric": eval_metric,
        }
        if objective is not None:
            hp["objective"] = objective
        if self.n_classes == 1:
            estimator_objective, self.loss_metric, estimator_eval_metric = (
                self._regression_objective_settings(objective, eval_metric)
            )
            estimator = xgb.XGBRegressor(
                objective=estimator_objective,
                max_depth=xgb_depth,
                n_estimators=xgb_estimators,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_lambda=reg_lambda,
                reg_alpha=reg_alpha,
                min_child_weight=min_child_weight,
                gamma=gamma,
                tree_method=tree_method,
                device=xgb_device,
                eval_metric=estimator_eval_metric,
                n_jobs=-1,
            )
            return estimator, hp

        self.loss_metric = "mlogloss" if self.n_classes > 2 else "logloss"
        metrics = [self.loss_metric]
        if eval_metric not in {"logloss", "mlogloss", self.loss_metric}:
            metrics.append(eval_metric)
        metrics.append(positive_precision)
        estimator = xgb.XGBClassifier(
            max_depth=xgb_depth,
            n_estimators=xgb_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            min_child_weight=min_child_weight,
            gamma=gamma,
            tree_method=tree_method,
            device=xgb_device,
            eval_metric=metrics,
            n_jobs=-1,
        )
        return estimator, hp

    @staticmethod
    def _regression_objective_settings(
        objective: str | None,
        eval_metric: str,
    ):
        """Resolve the estimator objective and the metric used for early stopping."""
        objective_name = objective or _ASYMMETRIC_HUBER_OBJECTIVE
        if objective_name == _ASYMMETRIC_HUBER_OBJECTIVE:
            custom_metric = xgboost_asymmetric_huber_metric
            return (
                xgboost_asymmetric_huber_objective,
                custom_metric.__name__,
                custom_metric,
            )
        default_metric = _REGRESSION_OBJECTIVE_METRICS.get(objective_name)
        if default_metric is None:
            supported = ", ".join(
                [_ASYMMETRIC_HUBER_OBJECTIVE, *_REGRESSION_OBJECTIVE_METRICS]
            )
            raise ValueError(
                f"Unsupported regression objective {objective_name!r}; "
                f"expected one of: {supported}"
            )
        metric = (
            default_metric
            if eval_metric in _CLASSIFICATION_EVAL_METRICS
            else eval_metric
        )
        return objective_name, metric, metric

    def _class_sample_weights(self, y):
        """Compute mean-one weights using only the task's training labels."""
        power = self.hp["class_weight_power"]
        if self.n_classes == 1 or power == 0:
            return None
        labels = self._to_numpy_labels(y)
        if labels.ndim != 1 or not len(labels):
            raise ValueError("Class weighting requires non-empty one-dimensional training labels")
        classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        weights = (len(labels) / (len(classes) * counts)) ** power
        weights /= np.dot(counts, weights) / len(labels)
        logging.getLogger(__name__).info(
            "XGBoost class weighting | power=%s | counts=%s | weights=%s",
            power,
            {int(label): int(count) for label, count in zip(classes, counts)},
            {int(label): float(weight) for label, weight in zip(classes, weights)},
        )
        return weights[inverse].astype(np.float32)

    def fit(self, X, y, sample_weight=None):
        """Apply class weights unless explicit sample weights are supplied."""
        if sample_weight is None:
            sample_weight = self._class_sample_weights(y)
        return super().fit(X, y, sample_weight=sample_weight)

    def fit_with_validation(
        self, X, y, validation_X, validation_y, test_X, test_y
    ) -> list[dict[str, float]]:
        X_train = self._flatten(X)
        y_train = self._to_numpy_labels(y)
        X_validation = self._flatten(validation_X)
        y_validation = self._to_numpy_labels(validation_y)
        X_test = self._flatten(test_X)
        y_test = self._to_numpy_labels(test_y)
        sample_weight = self._class_sample_weights(y_train)

        estimator = self.pipeline.named_steps["clf"]
        loss_metric = self.loss_metric
        estimator.set_params(
            callbacks=[
                xgb.callback.EarlyStopping(
                    rounds=self.hp["xgb_early_stopping_rounds"],
                    metric_name=loss_metric,
                    data_name="validation_1",
                    maximize=False,
                    save_best=True,
                )
            ]
        )
        if self.use_scaler:
            scaler = self.pipeline.named_steps["scaler"]
            X_train = scaler.fit_transform(X_train)
            X_validation = scaler.transform(X_validation)
            X_test = scaler.transform(X_test)

        estimator.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[
                (X_train, y_train),
                (X_validation, y_validation),
                (X_test, y_test),
            ],
            verbose=False,
        )
        self.is_fitted = True

        results = estimator.evals_result()
        train_losses = results["validation_0"][loss_metric]
        validation_losses = results["validation_1"][loss_metric]
        test_losses = results["validation_2"][loss_metric]
        # Training uses NumPy-backed CPU inputs, so align the fitted booster with
        # those inputs before validation prediction and checkpointing.
        estimator.get_booster().set_param({"device": "cpu"})
        best_round = min(
            range(len(validation_losses)),
            key=validation_losses.__getitem__,
        )
        self.best_iteration = best_round
        history = []
        for round_index, (train_loss, validation_loss, test_loss) in enumerate(
            zip(train_losses, validation_losses, test_losses)
        ):
            entry = {
                "epoch": float(round_index + 1),
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
                "test_loss": float(test_loss),
            }
            if self.n_classes > 1:
                for name, result_key in (
                    ("train", "validation_0"),
                    ("validation", "validation_1"),
                    ("test", "validation_2"),
                ):
                    entry[f"{name}_precision"] = float(
                        results[result_key]["positive_precision"][round_index]
                    )
            history.append(entry)
        return history
