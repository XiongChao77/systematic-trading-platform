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

    def _build_estimator(
        self,
        xgb_depth: int = 4,
        xgb_estimators: int = 300,
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
        if kwargs:
            print(f"[XGBoostAdapter] Ignored kwargs: {list(kwargs.keys())}")
        hp = {
            "xgb_depth": xgb_depth,
            "xgb_estimators": xgb_estimators,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "tree_method": tree_method,
            "xgb_device": xgb_device,
            "eval_metric": eval_metric,
        }
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
            eval_metric=eval_metric,
            n_jobs=-1,
        )
        return estimator, hp
