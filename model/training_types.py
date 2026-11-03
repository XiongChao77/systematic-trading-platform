from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TensorSplit(Dataset):
    """A time-ordered split shared by every training task."""

    X: torch.Tensor
    y: torch.Tensor
    returns: torch.Tensor
    assets: torch.Tensor | None = None

    def __post_init__(self) -> None:
        size = len(self.X)
        if len(self.y) != size or len(self.returns) != size:
            raise ValueError("X, y and returns must contain the same number of samples")

        if self.assets is not None:
            if self.assets.ndim != 1 or len(self.assets) != size:
                raise ValueError("assets must contain one integer ID per sample")
            if self.assets.dtype not in {
                torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
            }:
                raise ValueError("assets must use an integer tensor dtype")

    def asset_groups(self) -> list[np.ndarray]:
        if self.assets is None:
            return [np.arange(len(self))]
        assets = self.assets.detach().cpu().numpy()
        return [np.flatnonzero(assets == asset) for asset in np.unique(assets)]

    def balanced_sample_weights(self, sample_weight=None) -> np.ndarray:
        """Normalize existing weights to equal asset totals and an overall mean of one."""
        if sample_weight is None:
            weights = np.ones(len(self), dtype=np.float64)
        elif isinstance(sample_weight, torch.Tensor):
            weights = sample_weight.detach().cpu().numpy().astype(np.float64, copy=True)
        else:
            weights = np.array(sample_weight, dtype=np.float64, copy=True)
        if weights.shape != (len(self),) or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Sample weights must be finite, non-negative and match the split")
        groups = self.asset_groups()
        for indices in groups:
            total = weights[indices].sum()
            if total <= 0:
                raise ValueError("Each asset must have positive total sample weight")
            weights[indices] *= len(self) / (len(groups) * total)
        return weights

    @property
    def labels(self) -> np.ndarray:
        return self.y.detach().cpu().numpy()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index], self.returns[index]


@dataclass(frozen=True)
class DataSplits:
    train: TensorSplit
    validation: TensorSplit
    test: TensorSplit


def temporal_split_bounds(
    sample_count: int,
    *,
    train_ratio: float,
    val_ratio: float,
    purge_overlap: bool,
    seq_len: int,
    stride: int,
) -> dict[str, tuple[int, int]]:
    """Return the exact half-open sample ranges used by time-series training."""
    train_end = int(sample_count * train_ratio)
    validation_end = int(sample_count * (train_ratio + val_ratio))
    purge = (
        max(math.ceil(seq_len / max(stride, 1)) - 1, 0)
        if purge_overlap
        else 0
    )
    validation_start = train_end + purge
    test_start = validation_end + purge
    if not (
        0 < train_end <= validation_start < validation_end <= test_start < sample_count
    ):
        raise ValueError(
            "Dataset is too small for the requested train/validation/test split "
            f"and purge gap ({purge} samples)"
        )
    return {
        "train": (0, train_end),
        "valid": (validation_start, validation_end),
        "test": (test_start, sample_count),
    }


@dataclass
class FitResult:
    selection_metric: str
    validation_score: float
    validation_metrics: dict[str, float]
    history: list[dict[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class PredictionResult:
    y_true: np.ndarray
    probabilities: np.ndarray | None
    loss: float | None = None
    predictions: np.ndarray | None = None
    predicted_labels: np.ndarray | None = None

    @property
    def y_pred(self) -> np.ndarray:
        if self.predicted_labels is not None:
            return self.predicted_labels
        if self.probabilities is None:
            raise ValueError("Classification results require probabilities or predicted_labels")
        return self.probabilities.argmax(axis=1)
