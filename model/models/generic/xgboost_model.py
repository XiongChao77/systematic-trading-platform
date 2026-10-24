import xgboost as xgb

from model.models.generic.sklearn_adapter import SklearnEstimatorAdapter


class XGBoostAdapter(SklearnEstimatorAdapter):
    """
    Single-estimator XGBoost adapter (sklearn API), plugged into the same
    single-task training flow as every other model (train_single /
    run_single_model_3class_task). Trees don't need feature scaling, so
    use_scaler defaults to False.

    Note: MODEL_VERSION bumped to 2 because the previous "dual-head" (2x
    XGBClassifier fused trigger+direction) implementation under this same
    MODEL_TYPE was dead code (no call site ever invoked its .fit(), and
    running it through train.py's gradient loop would crash on an empty
    model.parameters() list). No historical checkpoints depend on v1.
    """

    MODEL_TYPE = "xgboost"
    MODEL_VERSION = 2

    def __init__(self, *args, use_scaler: bool = False, **kwargs):
        super().__init__(*args, use_scaler=use_scaler, **kwargs)
        self.best_iteration: int | None = None

    def _build_estimator(
        self,
        xgb_depth: int = 4,
        xgb_estimators: int = 300,
        xgb_early_stopping_rounds: int = 30,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        tree_method: str = "hist",
        xgb_device: str = "cpu",
        eval_metric: str = "logloss",
        **kwargs,
    ):
        if xgb_device not in {"cpu", "cuda"}:
            raise ValueError("xgb_device must be 'cpu' or 'cuda'")
        if xgb_device == "cuda" and not xgb.build_info().get("USE_CUDA", False):
            raise RuntimeError("GPU training requested, but this XGBoost build has no CUDA support")
        if xgb_early_stopping_rounds < 1:
            raise ValueError("xgb_early_stopping_rounds must be positive")
        if kwargs:
            print(f"[XGBoostAdapter] Ignored kwargs: {list(kwargs.keys())}")
        hp = {
            "xgb_depth": xgb_depth,
            "xgb_estimators": xgb_estimators,
            "xgb_early_stopping_rounds": xgb_early_stopping_rounds,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "tree_method": tree_method,
            "xgb_device": xgb_device,
            "eval_metric": eval_metric,
        }
        loss_metric = "mlogloss" if self.n_classes > 2 else "logloss"
        metrics = [loss_metric]
        if eval_metric not in {"logloss", "mlogloss", loss_metric}:
            metrics.append(eval_metric)
        estimator = xgb.XGBClassifier(
            max_depth=xgb_depth,
            n_estimators=xgb_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            tree_method=tree_method,
            device=xgb_device,
            eval_metric=metrics,
            n_jobs=-1,
        )
        return estimator, hp

    def fit_with_validation(self, X, y, validation_X, validation_y) -> list[dict[str, float]]:
        X_train = self._flatten(X)
        y_train = self._to_numpy_labels(y)
        X_validation = self._flatten(validation_X)
        y_validation = self._to_numpy_labels(validation_y)

        estimator = self.pipeline.named_steps["clf"]
        loss_metric = "mlogloss" if self.n_classes > 2 else "logloss"
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

        estimator.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_validation, y_validation)],
            verbose=False,
        )
        self.is_fitted = True

        results = estimator.evals_result()
        train_losses = results["validation_0"][loss_metric]
        validation_losses = results["validation_1"][loss_metric]
        # Training uses NumPy-backed CPU inputs, so align the fitted booster with
        # those inputs before validation/test prediction and checkpointing.
        estimator.get_booster().set_param({"device": "cpu"})
        best_round = min(
            range(len(validation_losses)),
            key=validation_losses.__getitem__,
        )
        self.best_iteration = best_round
        return [
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
            }
            for epoch, (train_loss, validation_loss) in enumerate(
                zip(train_losses, validation_losses), start=1
            )
        ]
