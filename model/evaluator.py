from __future__ import annotations

import logging
import os
from collections import Counter

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
)

from model.training_types import FitResult, PredictionResult


def plot_precision_recall_curve(
    result: PredictionResult,
    labels: list[int],
    output_path: str,
) -> None:
    """Save a binary PR curve or one-vs-rest curves for multiclass predictions."""
    if len(labels) < 2:
        raise ValueError("Precision-recall curves require at least two classes")
    if result.probabilities.shape != (len(result.y_true), len(labels)):
        raise ValueError("Prediction probabilities must have one column per class")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    curve_labels = labels[1:] if len(labels) == 2 else labels
    figure = Figure(figsize=(8, 6))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    try:
        for class_index, class_id in enumerate(labels):
            if class_id not in curve_labels:
                continue
            binary_truth = result.y_true == class_id
            precision, recall, _ = precision_recall_curve(
                binary_truth,
                result.probabilities[:, class_index],
            )
            average_precision = average_precision_score(
                binary_truth,
                result.probabilities[:, class_index],
            )
            axis.plot(
                recall,
                precision,
                label=f"Class {class_id} (AP={average_precision:.3f})",
            )

        axis.set(
            xlabel="Recall",
            ylabel="Precision",
            title="Precision-Recall Curve",
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.05),
        )
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
        figure.tight_layout()
        figure.savefig(output_path, dpi=160)
    finally:
        figure.clear()


def plot_threshold_precision_recall_curve(
    result: PredictionResult,
    labels: list[int],
    output_path: str,
) -> None:
    """Save precision and recall against decision thresholds."""
    if len(labels) < 2:
        raise ValueError("Threshold precision-recall curves require at least two classes")
    if result.probabilities.shape != (len(result.y_true), len(labels)):
        raise ValueError("Prediction probabilities must have one column per class")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    curve_labels = labels[1:] if len(labels) == 2 else labels
    figure = Figure(figsize=(8, 6))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    try:
        for class_index, class_id in enumerate(labels):
            if class_id not in curve_labels:
                continue
            precision, recall, thresholds = precision_recall_curve(
                result.y_true == class_id,
                result.probabilities[:, class_index],
            )
            axis.plot(
                thresholds,
                precision[:-1],
                label=f"Class {class_id} Precision",
            )
            axis.plot(
                thresholds,
                recall[:-1],
                linestyle="--",
                label=f"Class {class_id} Recall",
            )

        axis.set(
            xlabel="Decision threshold",
            ylabel="Score",
            title="Threshold vs. Precision/Recall",
            ylim=(0.0, 1.05),
        )
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
        figure.tight_layout()
        figure.savefig(output_path, dpi=160)
    finally:
        figure.clear()


def plot_training_curves(fit_result: FitResult, output_path: str) -> None:
    """Save per-epoch training and validation loss curves."""
    if not fit_result.history:
        return
    epochs = [entry["epoch"] for entry in fit_result.history]
    train_losses = [entry["train_loss"] for entry in fit_result.history]
    validation_losses = [entry["validation_loss"] for entry in fit_result.history]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    figure = Figure(figsize=(8, 6))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    try:
        axis.plot(epochs, train_losses, label="Train loss")
        axis.plot(epochs, validation_losses, label="Validation loss")
        axis.set(
            xlabel="Epoch",
            ylabel="Loss",
            title="Training and Validation Loss",
        )
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
        figure.tight_layout()
        figure.savefig(output_path, dpi=160)
    finally:
        figure.clear()


def evaluate_classification(result: PredictionResult, labels: list[int]) -> dict:
    y_true = result.y_true
    y_pred = result.y_pred
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    total = len(y_true)
    per_class = {}

    for class_id in labels:
        key = str(class_id)
        base_rate = true_counts[class_id] / total if total else 0.0
        predicted = y_pred == class_id
        precision = float(report.get(key, {}).get("precision", 0.0))
        per_class[key] = {
            "precision": precision,
            "recall": float(report.get(key, {}).get("recall", 0.0)),
            "f1": float(report.get(key, {}).get("f1-score", 0.0)),
            "support": int(report.get(key, {}).get("support", 0)),
            "base_rate": float(base_rate),
            "pred_ratio": float(pred_counts[class_id] / total) if total else 0.0,
            "precision_lift": float(precision / base_rate) if base_rate else None,
            "avg_probability_when_predicted": (
                float(result.probabilities[predicted, class_id].mean()) if predicted.any() else None
            ),
        }

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "loss": None if result.loss is None else float(result.loss),
        "sample_count": total,
        "labels": labels,
        "true_distribution": {
            str(label): {
                "count": int(true_counts[label]),
                "ratio": float(true_counts[label] / total) if total else 0.0,
            }
            for label in labels
        },
        "pred_distribution": {
            str(label): {
                "count": int(pred_counts[label]),
                "ratio": float(pred_counts[label] / total) if total else 0.0,
            }
            for label in labels
        },
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "matrix": matrix.tolist()},
    }


def confusion_frame(metrics: dict) -> pd.DataFrame:
    labels = metrics["confusion_matrix"]["labels"]
    return pd.DataFrame(
        metrics["confusion_matrix"]["matrix"],
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )


def format_classification_table(metrics: dict) -> str:
    labels = metrics["labels"]
    per_class = metrics["per_class"]
    total = metrics["sample_count"]

    header = f"{'Class':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}"
    separator = "-" * len(header)
    lines = [header, separator]
    for label in labels:
        row = per_class[str(label)]
        lines.append(
            f"{str(label):<10} | {row['precision']:>10.4f} | {row['recall']:>10.4f} | "
            f"{row['f1']:>10.4f} | {row['support']:>8d}"
        )
    lines.append(separator)
    lines.append(f"{'Accuracy':<10} | {'':<10} | {'':<10} | {metrics['accuracy']:>10.4f} | {total:>8d}")
    lines.append(
        f"{'Macro avg':<10} | {metrics['macro_precision']:>10.4f} | {metrics['macro_recall']:>10.4f} | "
        f"{metrics['macro_f1']:>10.4f} | {total:>8d}"
    )
    lines.append(
        f"{'Weighted avg':<10} | {metrics['weighted_precision']:>10.4f} | {metrics['weighted_recall']:>10.4f} | "
        f"{metrics['weighted_f1']:>10.4f} | {total:>8d}"
    )
    return "\n".join(lines)


def log_evaluation_report(
    logger: logging.Logger,
    *,
    task_type: str,
    metrics: dict,
    save_dir: str,
    model_label: str = "Model",
) -> None:
    """Print a classification report table plus label proportions and the save path."""
    kind = "Binary" if len(metrics["labels"]) == 2 else "Multiclass"
    logger.info("%s Evaluating %s Model: %s | Task: %s %s", "=" * 20, kind, model_label, task_type, "=" * 20)
    logger.info("\n" + format_classification_table(metrics))
    loss_str = f"{metrics['loss']:.4f}" if metrics["loss"] is not None else "n/a"
    logger.info("Test macro-F1: %.4f | Test Loss: %s", metrics["macro_f1"], loss_str)

    proportions = ", ".join(
        f"{label}: {metrics['true_distribution'][label]['ratio'] * 100:.2f}%"
        for label in (str(l) for l in metrics["labels"])
    )
    logger.info("[%s] True label proportion (Test set): %s", model_label, proportions)
    logger.info("🎉 %s task evaluation complete. Artifacts saved to: %s", kind, save_dir)
