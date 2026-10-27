from __future__ import annotations

import numpy as np
import torch


HUBER_DELTA = 1.0
_HESSIAN_FLOOR = 1e-6


def _forgiven_overshoots(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    return ((targets == 1.0) & (predictions > 1.0)) | (
        (targets == -1.0) & (predictions < -1.0)
    )


def asymmetric_huber_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if predictions.shape != targets.shape:
        raise ValueError("Regression predictions and targets must have matching shapes")

    residual = predictions - targets
    forgiven = _forgiven_overshoots(targets, predictions)
    residual = np.where(forgiven, 0.0, residual)
    absolute_residual = np.abs(residual)
    quadratic = np.minimum(absolute_residual, HUBER_DELTA)
    linear = absolute_residual - quadratic
    return 0.5 * quadratic**2 + HUBER_DELTA * linear


def asymmetric_huber_gradient_hessian(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if predictions.shape != targets.shape:
        raise ValueError("Regression predictions and targets must have matching shapes")

    residual = predictions - targets
    forgiven = _forgiven_overshoots(targets, predictions)
    residual = np.where(forgiven, 0.0, residual)
    gradient = np.clip(residual, -HUBER_DELTA, HUBER_DELTA)
    hessian = np.where(
        np.abs(residual) < HUBER_DELTA,
        1.0,
        _HESSIAN_FLOOR,
    )
    gradient[forgiven] = 0.0
    hessian[forgiven] = _HESSIAN_FLOOR
    return gradient, hessian


def asymmetric_huber_torch_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    predictions = predictions.reshape(-1)
    targets = targets.to(dtype=predictions.dtype).reshape(-1)
    if predictions.shape != targets.shape:
        raise ValueError("Regression predictions and targets must have matching shapes")

    residual = predictions - targets
    forgiven = ((targets == 1.0) & (predictions > 1.0)) | (
        (targets == -1.0) & (predictions < -1.0)
    )
    residual = residual.masked_fill(forgiven, 0.0)
    absolute_residual = residual.abs()
    quadratic = absolute_residual.clamp(max=HUBER_DELTA)
    linear = absolute_residual - quadratic
    return (0.5 * quadratic.square() + HUBER_DELTA * linear).mean()


def xgboost_asymmetric_huber_objective(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return asymmetric_huber_gradient_hessian(targets, predictions)


def xgboost_asymmetric_huber_metric(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> float:
    return float(asymmetric_huber_loss(predictions, targets).mean())


def lightgbm_asymmetric_huber_objective(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return asymmetric_huber_gradient_hessian(targets, predictions)

