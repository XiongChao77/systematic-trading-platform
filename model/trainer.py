from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.tasks.strategies import TaskStrategy
from model.train_config import TrainConfig
from model.training_types import FitResult, PredictionResult, TensorSplit


def _state_dict_on_cpu(model: torch.nn.Module) -> dict:
    source = model._orig_mod if hasattr(model, "_orig_mod") else model
    return {key: value.detach().cpu().clone() for key, value in source.state_dict().items()}


class TorchTrainer:
    def __init__(self, config: TrainConfig, device: torch.device, logger: logging.Logger):
        self.config = config
        self.device = device
        self.logger = logger

    def _loader(self, split: TensorSplit, *, training: bool) -> DataLoader:
        sampler = self.task.sampler(split.labels) if training else None
        return DataLoader(
            split,
            batch_size=self.config.batch_size,
            sampler=sampler,
            shuffle=training and sampler is None,
            num_workers=0,
        )

    def _optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        gate = [parameter for name, parameter in model.named_parameters() if "feature_weighter" in name]
        regular = [parameter for name, parameter in model.named_parameters() if "feature_weighter" not in name]
        groups = [{"params": regular, "lr": self.config.lr}]
        if gate:
            groups.append({"params": gate, "lr": self.config.gate_lr})
        return torch.optim.AdamW(groups, weight_decay=self.config.weight_decay)

    def fit(
        self,
        model: torch.nn.Module,
        task: TaskStrategy,
        train_split: TensorSplit,
        validation_split: TensorSplit,
        test_split: TensorSplit,
    ) -> FitResult:
        if not len(train_split) or not len(validation_split) or not len(test_split):
            raise ValueError("Training, validation and test splits must not be empty")

        self.task = task
        task.configure(train_split.labels)
        train_loader = self._loader(train_split, training=True)
        validation_loader = self._loader(validation_split, training=False)
        optimizer = self._optimizer(model)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(1, self.config.patience // 2)
        )

        selection_metric = self.config.selection_metric
        maximize = selection_metric == "macro_f1"
        best_score = -float("inf") if maximize else float("inf")
        best_state = None
        best_validation = {}
        history = []
        stale_epochs = 0

        for epoch in range(1, self.config.epochs + 1):
            model.train()
            total_loss = 0.0
            total_samples = 0

            progress = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.epochs}", leave=False)
            for inputs, labels, returns in progress:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                returns = returns.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                output = task.forward(model, inputs)
                loss = task.loss(output, labels, returns, epoch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                batch_size = len(inputs)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                progress.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = total_loss / max(total_samples, 1)
            validation = self.predict(model, task, validation_split)
            test = self.predict(model, task, test_split)
            validation_f1 = f1_score(
                validation.y_true,
                validation.y_pred,
                labels=task.labels,
                average="macro",
                zero_division=0,
            )
            validation_loss = float(validation.loss)
            validation_values = (
                validation.predictions if task.is_regression else validation.y_pred
            )
            validation_mae = float(
                mean_absolute_error(validation.y_true, validation_values)
            )
            scheduler.step(validation_loss)

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "test_loss": float(test.loss),
                "validation_macro_f1": validation_f1,
            }
            if task.is_regression:
                epoch_metrics["validation_mae"] = validation_mae
            else:
                # Multiclass tasks need per-class precision; the positive class is
                # the last configured label for both binary and three-class tasks.
                class_precisions = precision_score(
                    test.y_true,
                    test.y_pred,
                    labels=list(task.labels),
                    average=None,
                    zero_division=0,
                )
                epoch_metrics["test_precision"] = float(class_precisions[-1])
            history.append(epoch_metrics)
            self.logger.info(
                "Epoch %03d | train_loss %.4f | validation_loss %.4f | validation_macro_f1 %.4f%s",
                epoch,
                train_loss,
                validation_loss,
                validation_f1,
                f" | validation_mae {validation_mae:.4f}" if task.is_regression else "",
            )

            score = (
                validation_f1
                if maximize
                else validation_mae
                if selection_metric == "mae"
                else validation_loss
            )
            improved = score > best_score + 1e-6 if maximize else score < best_score - 1e-6
            if improved:
                best_score = score
                best_state = _state_dict_on_cpu(model)
                best_validation = {
                    "loss": validation_loss,
                    "macro_f1": validation_f1,
                }
                if task.is_regression:
                    best_validation["mae"] = validation_mae
                    best_validation["rmse"] = float(
                        mean_squared_error(validation.y_true, validation.predictions) ** 0.5
                    )
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    self.logger.info("Early stopping at epoch %d", epoch)
                    break

        if best_state is None:
            raise RuntimeError("Training did not produce a checkpoint")

        model.load_state_dict(best_state)
        model.to(self.device).eval()
        return FitResult(
            selection_metric=selection_metric,
            validation_score=float(best_score),
            validation_metrics=best_validation,
            history=history,
        )

    @torch.no_grad()
    def predict(self, model: torch.nn.Module, task: TaskStrategy, split: TensorSplit) -> PredictionResult:
        if not len(split):
            raise ValueError("Prediction split must not be empty")

        model.eval()
        loader = DataLoader(split, batch_size=self.config.batch_size, shuffle=False)
        probabilities = []
        predictions = []
        predicted_labels = []
        truths = []
        total_loss = 0.0
        total_samples = 0

        for inputs, labels, returns in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            returns = returns.to(self.device, non_blocking=True)
            output = task.forward(model, inputs)
            loss = task.loss(output, labels, returns, epoch=0)
            probs = task.probabilities(output)

            batch_size = len(inputs)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            outputs = probs.detach().cpu().numpy()
            if task.is_regression:
                predictions.append(outputs.reshape(-1))
                predicted_labels.append(task.decode(outputs))
            else:
                probabilities.append(outputs)
            truths.append(labels.detach().cpu().numpy())

        raw_truths = np.concatenate(truths)
        if task.is_regression:
            predicted_values = np.concatenate(predictions)
            decoded_predictions = np.concatenate(predicted_labels)
            probability_values = None
        else:
            probability_values = np.concatenate(probabilities)
            raw_truths = task.decode_class_indices(raw_truths.astype(np.int64))
            decoded_predictions = task.decode_class_indices(
                probability_values.argmax(axis=1)
            )
            predicted_values = None

        return PredictionResult(
            y_true=raw_truths,
            probabilities=probability_values,
            loss=total_loss / max(total_samples, 1),
            predictions=predicted_values,
            predicted_labels=decoded_predictions,
        )


class SklearnTrainer:
    def __init__(self, config: TrainConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def fit(
        self,
        model: Any,
        task: TaskStrategy,
        train_split: TensorSplit,
        validation_split: TensorSplit,
        test_split: TensorSplit,
    ) -> FitResult:
        task.configure(train_split.labels)
        fit_with_validation = getattr(model, "fit_with_validation", None)
        if callable(fit_with_validation):
            history = fit_with_validation(
                train_split.X, train_split.y,
                validation_split.X, validation_split.y,
                test_split.X, test_split.y,
            )
            # for metrics in history:
            #     self.logger.info(
            #         "Round %03d | Train loss %.5f | Validation loss %.5f",
            #         int(metrics["epoch"]),
            #         metrics["train_loss"],
            #         metrics["validation_loss"],
            #     )
            if history and model.best_iteration is not None:
                best = history[model.best_iteration]
                self.logger.info(
                    "XGBoost early stopping | best_round %03d | best_validation_loss %.5f",
                    int(best["epoch"]),
                    best["validation_loss"],
                )
        else:
            model.fit(train_split.X, train_split.y)
            history = []
        validation = self.predict(model, task, validation_split)
        macro_f1 = f1_score(
            validation.y_true,
            validation.y_pred,
            labels=task.labels,
            average="macro",
            zero_division=0,
        )
        validation_metrics = {"loss": float(validation.loss), "macro_f1": float(macro_f1)}
        validation_values = (
            validation.predictions if task.is_regression else validation.y_pred
        )
        validation_metrics["mae"] = float(
            mean_absolute_error(validation.y_true, validation_values)
        )
        if task.is_regression:
            validation_metrics["rmse"] = float(
                mean_squared_error(validation.y_true, validation.predictions) ** 0.5
            )
        selection_metric = self.config.selection_metric
        score = (
            macro_f1
            if selection_metric == "macro_f1"
            else validation_metrics["mae"]
            if selection_metric == "mae"
            else float(validation.loss)
        )
        return FitResult(
            selection_metric=selection_metric,
            validation_score=float(score),
            validation_metrics=validation_metrics,
            history=history,
        )

    def predict(self, model: Any, task: TaskStrategy, split: TensorSplit) -> PredictionResult:
        if task.is_regression:
            y_true = split.labels
            predictions = model.predict_values(split.X).reshape(-1)
            return PredictionResult(
                y_true=y_true,
                probabilities=None,
                loss=float(mean_squared_error(y_true, predictions)),
                predictions=predictions,
                predicted_labels=task.decode(predictions),
            )
        probabilities = model.predict_proba(split.X)
        class_indices = split.labels.astype(np.int64)
        y_true = task.decode_class_indices(class_indices)
        predicted_labels = task.decode_class_indices(probabilities.argmax(axis=1))
        return PredictionResult(
            y_true=y_true,
            probabilities=probabilities,
            loss=log_loss(
                class_indices,
                probabilities,
                labels=list(range(probabilities.shape[1])),
            ),
            predicted_labels=predicted_labels,
        )


def build_trainer(model: Any, config: TrainConfig, device: torch.device, logger: logging.Logger) -> TorchTrainer | SklearnTrainer:
    if getattr(model, "TRAINING_BACKEND", "gradient") == "sklearn":
        return SklearnTrainer(config, logger)
    return TorchTrainer(config, device, logger)
