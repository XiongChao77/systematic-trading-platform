from abc import abstractmethod

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from model.models.model_base import BaseTimeSeriesModel


class SklearnEstimatorAdapter(BaseTimeSeriesModel):
    """
    Generic adapter that lets any single sklearn-compatible classifier
    (fit(X, y) / predict_proba(X)) plug into the same ModelFactory /
    train.py / ModelHandler pipeline as the torch models.

    Compatibility tricks:
      - forward(x) returns log(predict_proba(x)) as "logits", so the existing
        torch.softmax(logits) calls in train.py recover the exact probabilities
        (softmax(log(p)) == p since p already sums to 1).
      - ModelHandler adds the common inference wrapper after loading the
        checkpoint.
      - state_dict()/load_state_dict() hold the fitted sklearn Pipeline
        directly; BaseTimeSeriesModel.save_checkpoint/load_checkpoint pickle
        it via torch.save/torch.load like any other object.
    """

    TRAINING_BACKEND = "sklearn"

    def __init__(
        self,
        input_size: int,
        n_classes: int = 2,
        seq_len: int = None,
        use_scaler: bool = True,
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.n_classes = int(n_classes)
        self.seq_len = int(seq_len) if seq_len is not None else None
        self.use_scaler = bool(use_scaler)
        self.device = torch.device(device)

        estimator, self.hp = self._build_estimator(**kwargs)
        steps = [("scaler", StandardScaler())] if self.use_scaler else []
        steps.append(("clf", estimator))
        self.pipeline = Pipeline(steps)
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Subclass hook: build the raw estimator, return (estimator, hp_used)
    # so hp_used can be persisted in export_meta() and replayed on reload.
    # ------------------------------------------------------------------
    @abstractmethod
    def _build_estimator(self, **kwargs):
        pass

    # ------------------------------------------------------------------
    # Data shaping
    # ------------------------------------------------------------------
    def _flatten(self, x) -> np.ndarray:
        """[N, T, F] -> [N, T*F]; also accepts already-2D input."""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x)

        if x.ndim == 3:
            N, T, F = x.shape
            if self.seq_len is not None:
                assert T == self.seq_len, f"Shape mismatch: expected seq_len={self.seq_len}, got T={T}"
            assert F == self.input_size, f"Shape mismatch: expected input_size={self.input_size}, got F={F}"
            x = x.reshape(N, -1)
        return x

    @staticmethod
    def _to_numpy_labels(y) -> np.ndarray:
        if isinstance(y, torch.Tensor):
            return y.detach().cpu().numpy()
        return np.asarray(y)

    # ------------------------------------------------------------------
    # Train / infer
    # ------------------------------------------------------------------
    def fit(self, X, y, sample_weight=None):
        Xf = self._flatten(X)
        y_np = self._to_numpy_labels(y)
        fit_kwargs = {"clf__sample_weight": sample_weight} if sample_weight is not None else {}
        self.pipeline.fit(Xf, y_np, **fit_kwargs)
        self.is_fitted = True

    def predict_proba(self, X) -> np.ndarray:
        Xf = self._flatten(X)
        if not self.is_fitted:
            return np.full((Xf.shape[0], self.n_classes), 1.0 / self.n_classes)
        return self.pipeline.predict_proba(Xf)

    def predict_values(self, X) -> np.ndarray:
        Xf = self._flatten(X)
        if not self.is_fitted:
            raise RuntimeError("Cannot run regression inference before the estimator is fitted")
        return np.asarray(self.pipeline.predict(Xf), dtype=np.float64).reshape(-1)

    def forward(self, x):
        if self.n_classes == 1:
            values = self.predict_values(x)
            return torch.as_tensor(values, dtype=torch.float32, device=x.device).reshape(-1, 1)
        probs_np = self.predict_proba(x)
        probs = torch.as_tensor(probs_np, dtype=torch.float32, device=x.device)
        return torch.log(probs.clamp_min(1e-8))

    # ------------------------------------------------------------------
    # Persistence (sklearn objects aren't torch parameters/buffers)
    # ------------------------------------------------------------------
    def state_dict(self):
        return {"pipeline": self.pipeline, "is_fitted": self.is_fitted}

    def load_state_dict(self, state_dict):
        self.pipeline = state_dict["pipeline"]
        self.is_fitted = state_dict["is_fitted"]

    def export_meta(self, **extra) -> dict:
        return {
            "model_type": self.MODEL_TYPE,
            "model_version": self.MODEL_VERSION,
            "input_size": self.input_size,
            "n_classes": self.n_classes,
            "seq_len": self.seq_len,
            "use_scaler": self.use_scaler,
            "hp": self.hp,
            **extra,
        }

    @classmethod
    def build_from_meta(cls, meta: dict, state: dict, device):
        model = cls(
            input_size=meta["input_size"],
            n_classes=meta.get("n_classes", 2),
            seq_len=meta.get("seq_len"),
            use_scaler=meta.get("use_scaler", True),
            device=device,
            **meta.get("hp", {}),
        )
        model.load_state_dict(state["state_dict"])
        return model


class SVCAdapter(SklearnEstimatorAdapter):
    """
    sklearn SVC adapter. probability=True is required to get predict_proba,
    which makes fitting noticeably slower (internal 5-fold CV calibration).
    """

    MODEL_TYPE = "svc"
    MODEL_VERSION = 1

    def _build_estimator(self, C: float = 1.0, kernel: str = "rbf", gamma="scale", **kwargs):
        if kwargs:
            print(f"[SVCAdapter] Ignored kwargs: {list(kwargs.keys())}")
        hp = {"C": C, "kernel": kernel, "gamma": gamma}
        estimator = SVC(probability=True, class_weight="balanced", **hp)
        return estimator, hp


class LogisticRegressionSklearnAdapter(SklearnEstimatorAdapter):
    """
    sklearn LogisticRegression adapter. Named distinctly from the existing
    torch-based LogisticRegressionTS_V1 (model_type="logistic_regression"),
    which is a different (gradient-trained) implementation.
    """

    MODEL_TYPE = "logistic_regression_sklearn"
    MODEL_VERSION = 1

    def _build_estimator(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        **kwargs,
    ):
        if kwargs:
            print(f"[LogisticRegressionSklearnAdapter] Ignored kwargs: {list(kwargs.keys())}")
        hp = {"C": C, "penalty": penalty, "solver": solver, "max_iter": max_iter}
        estimator = LogisticRegression(class_weight="balanced", **hp)
        return estimator, hp
