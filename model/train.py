#!/usr/bin/env python3
from __future__ import annotations

import logging
import os,sys
import random
import time
import argparse
import json

import pandas as pd
from dataclasses import asdict, dataclass

import numpy as np
import torch

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir,'..'))
from data_process import common
from model.artifacts import save_fusion_run, save_single_run
from model.data_loader import TimeSeriesWindowDataset
from model.evaluator import evaluate_classification, evaluate_regression, evaluate_numeric_regression, log_evaluation_report, plot_rolling_binary_oos_curves
from model.model_factory import ModelFactory
from model.tasks.strategies import build_task, validate_preparation_config
from model.train_config import (
    DataConfig,
    RollingConfig,
    SingleModelConfig,
    TrainConfig,
    TrainTask,
)
from model.tasks.strategies import TaskStrategy
from model.trainer import build_trainer
from model.training_types import TensorSplit, PredictionResult
from model.window_predictions import annotate_predictions, validate_predictions


@dataclass(frozen=True)
class WindowTrainingData:
    train: TensorSplit
    validation: TensorSplit
    feature_names: list[str]
    requested_features: list[str]
    feature_count: int
    prep_output_dir: str


@dataclass
class FittedWindowModel:
    task_type: str
    model: torch.nn.Module
    report: dict
    save_dir: str
    task: TaskStrategy
    feature_names: list[str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _output_dim(task_type: str) -> int:
    if task_type in TrainTask.REGRESSION_TASKS:
        return 1
    return 2 if task_type in TrainTask.BINARY_TASKS else 3


def _build_model(config: TrainConfig, task_type: str, input_size: int, device: torch.device) -> torch.nn.Module:
    model_parameters = asdict(config.model_cfg)
    if task_type == TrainTask.MARTINGALE_REGRESSION and model_parameters["model_type"] == "xgboost":
        if model_parameters.get("objective") == "asymmetric_huber":
            raise ValueError("Martingale regression requires a numeric regression objective")
        if model_parameters.get("objective") is None:
            model_parameters["objective"] = "reg:squarederror"
    return ModelFactory.build_for_training(
        device=device,
        input_size=input_size,
        n_classes=_output_dim(task_type),
        **model_parameters,
    )


def _task_training_splits(task: TaskStrategy, raw: WindowTrainingData):
    if task.source_labels is not None:
        missing_train_labels = set(task.source_labels) - set(raw.train.labels.tolist())
    else:
        missing_train_labels = set()
    if missing_train_labels:
        raise ValueError(
            f"Training split is missing labels required by the task: {sorted(missing_train_labels)}"
        )

    train = task.dataset_split(raw.train)
    validation = task.dataset_split(raw.validation)
    for name, split in (
        ("train", train),
        ("validation", validation),
    ):
        if not len(split):
            raise ValueError(f"{name} split is empty after task transformation")
    missing_train_labels = (
        set()
        if task.is_regression or task.source_labels is not None
        else set(task.labels) - set(train.labels.tolist())
    )
    if missing_train_labels:
        raise ValueError(
            f"Training split is missing labels required by the task: {sorted(missing_train_labels)}"
        )
    return train, validation


def _fit_window_model(
    *,
    task_type: str,
    config: TrainConfig,
    prepared: WindowTrainingData,
    save_dir: str,
    device: torch.device,
    logger: logging.Logger,
) -> FittedWindowModel:
    if task_type in TrainTask.COMBO_TASKS:
        raise ValueError(f"{task_type} is a combo task, not a single-model task")

    set_seed(config.seed)
    model = _build_model(config, task_type, prepared.feature_count, device)
    task = build_task(task_type, config, device, model=model)
    train_split, validation_split = _task_training_splits(task, prepared)
    trainer = build_trainer(model, config, device, logger)

    fit_result = trainer.fit(
        model, task, train_split, validation_split,
        None,
    )
    prediction = trainer.predict(
        model, task, validation_split,
    )
    metrics = _evaluation_metrics(prediction, task)
    report = save_single_run(
        model=model,
        task_type=task_type,
        fit_result=fit_result,
        prediction=prediction,
        metrics=metrics,
        save_dir=save_dir,
        prep_output_dir=prepared.prep_output_dir,
        feature_names=prepared.feature_names,
        feature_list=prepared.requested_features,
        label_col=config.data_cfg.label_col,
        seq_len=config.model_cfg.seq_len,
        train_config=asdict(config),
        evaluation_split="validation",
    )
    logger.info(
        "Completed %s: validation_%s=%.4f",
        task_type,
        fit_result.selection_metric,
        fit_result.validation_score,
    )
    model_label = f"Best_{fit_result.selection_metric.upper()}"
    log_evaluation_report(
        logger,
        task_type=task_type,
        metrics=metrics,
        save_dir=save_dir,
        model_label=model_label,
    )
    return FittedWindowModel(task_type=task_type, model=model, report=report, save_dir=save_dir,
                     task=task, feature_names=prepared.feature_names)


def fusion_trigger_dir(logger: logging.Logger, trigger_dir: str, direction_dir: str, fusion_dir: str) -> dict:
    """Fuse saved window predictions from two independent training roots."""
    from model.window_fusion import fuse_window_runs
    return fuse_window_runs(
        task_type=TrainTask.TRIGGER_DIRECTION,
        role_directories={"trigger": trigger_dir, "direction": direction_dir},
        fusion_dir=fusion_dir, logger=logger,
    )


def fusion_long_short_ovr(logger: logging.Logger, long_ovr_dir: str, short_ovr_dir: str, fusion_dir: str) -> dict:
    """Fuse saved window predictions from two independent training roots."""
    from model.window_fusion import fuse_window_runs
    return fuse_window_runs(
        task_type=TrainTask.LONG_SHORT_OVR,
        role_directories={"long_ovr": long_ovr_dir, "short_ovr": short_ovr_dir},
        fusion_dir=fusion_dir, logger=logger,
    )


def single_window(times, config: DataConfig, interval):
    """Generate one chronological development window from timestamp fractions."""
    times = pd.DatetimeIndex(times)
    train_end = int(len(times) * config.train_ratio)
    valid_end = int(len(times) * (config.train_ratio + config.val_ratio))
    if not 0 < train_end < valid_end < len(times):
        raise ValueError("Insufficient timestamps for a single training window")
    return [{
        "fold_id": "0000", "phase": "development",
        "train_start": times[0].isoformat(),
        "validation_start": times[train_end].isoformat(),
        "oos_start": times[valid_end].isoformat(),
        "oos_end": (times[-1] + interval).isoformat(),
    }]


def validate_windows(windows, forward_start):
    """Normalize windows and reject overlapping OOS or unsafe phase boundaries."""
    normalized = []
    ids = set()
    last_end = None
    for index, item in enumerate(windows):
        fold = dict(item)
        fold_id = fold.get("fold_id", f"{index:04d}")
        if not isinstance(fold_id, str) or not fold_id or fold_id in (".", "..") or any(
            separator in fold_id for separator in ("/", "\\")
        ) or fold_id in ids:
            raise ValueError(f"Invalid or duplicate fold_id: {fold_id!r}")
        boundaries = []
        for name in ("train_start", "validation_start", "oos_start", "oos_end"):
            stamp = pd.Timestamp(fold[name])
            if pd.isna(stamp):
                raise ValueError(f"Invalid {name} for fold {fold_id}")
            stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
            fold[name] = stamp.isoformat()
            boundaries.append(stamp)
        if not all(left < right for left, right in zip(boundaries, boundaries[1:])):
            raise ValueError(f"Window boundaries must be strictly increasing: {fold_id}")
        if last_end is not None and boundaries[2] < last_end:
            raise ValueError("OOS windows must be chronological and non-overlapping")
        phase = "development" if boundaries[2] < forward_start else "forward"
        if fold.get("phase", phase) != phase:
            raise ValueError(f"Window phase disagrees with the Forward boundary: {fold_id}")
        if phase == "development" and boundaries[3] > forward_start:
            raise ValueError("Development OOS cannot cross the Forward boundary")
        fold.update(fold_id=fold_id, phase=phase)
        normalized.append(fold)
        ids.add(fold_id)
        last_end = boundaries[3]
    return normalized


def label_maturity(frame, preparation, interval):
    """Return the close of the last candle used to determine each target."""
    if preparation.label_type == "MARTINGALE":
        return pd.to_datetime(frame["martingale_exit_time"], unit="ms", utc=True) + interval
    times = pd.to_datetime(frame["open_time_date_utc"], utc=True)
    if preparation.label_type == "PURE_BINARY":
        if "reach_time" not in frame:
            raise ValueError("PURE_BINARY window training requires persisted reach_time")
        return times + pd.to_timedelta((frame["reach_time"] + 1) * interval.value, unit="ns")
    return times + interval * (preparation.predict_num + 1)


def _evaluation_metrics(result, task):
    if task.numeric_regression:
        return evaluate_numeric_regression(result, task.labels)
    if task.is_regression:
        return evaluate_regression(result, task.labels)
    return evaluate_classification(result, task.labels)


def _combine_predictions(results):
    count = sum(len(result.y_true) for result in results)
    def concat(name):
        return np.concatenate([getattr(result, name) for result in results]) if getattr(results[0], name) is not None else None
    return PredictionResult(
        y_true=concat("y_true"), probabilities=concat("probabilities"),
        predictions=concat("predictions"), predicted_labels=concat("predicted_labels"),
        loss=sum(result.loss * len(result.y_true) for result in results) / count
        if all(result.loss is not None for result in results) else None,
    )


def _evaluate_oos_models(configs, fitted_models, symbol_frames, preparation, interval_ms,
                         oos_start, oos_end, fold_dir, fold, device, logger):
    """Evaluate frozen models after selection, with the same maturity rules as fitting."""
    interval = pd.Timedelta(milliseconds=interval_ms)
    results = {}
    fold["oos_metrics"] = {}
    for config in configs:
        fitted = fitted_models[config.train_task]
        predictions = []
        for sub in symbol_frames.values():
            times = pd.to_datetime(sub["open_time_date_utc"], utc=True)
            part = sub.loc[(times >= oos_start - interval * config.model_cfg.seq_len) & (times < oos_end)]
            dataset = TimeSeriesWindowDataset(
                part, interval_ms, fitted.feature_names, config.data_cfg.label_col,
                config.model_cfg.seq_len, stride=config.stride, show_feature_distribution=False,
            )
            sample_rows = part.loc[dataset.indices]
            sample_times = pd.to_datetime(sample_rows["open_time_date_utc"], utc=True)
            maturity = label_maturity(sample_rows, preparation, interval)
            evaluation_start = oos_start + interval * (
                config.model_cfg.seq_len - 1 if config.data_cfg.purge_overlap else 0
            )
            mask = torch.from_numpy(((sample_times >= evaluation_start) & (maturity <= oos_end)).to_numpy(copy=True))
            raw = TensorSplit(dataset.X[mask], dataset.y[mask], dataset.returns[mask])
            split = fitted.task.dataset_split(raw)
            if not len(split):
                continue
            trainer = build_trainer(fitted.model, config, device, logger)
            predictions.append(trainer.predict(fitted.model, fitted.task, split))
        if not predictions:
            logger.warning("Window %s has no mature OOS labels for %s", fold["fold_id"], config.train_task)
            continue
        result = _combine_predictions(predictions)
        metrics = _evaluation_metrics(result, fitted.task)
        role_dir = os.path.join(fold_dir, config.train_task)
        metrics_path = os.path.join(role_dir, "metrics.json")
        with open(metrics_path, encoding="utf-8") as handle:
            artifact = json.load(handle)
        artifact["test"] = metrics
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2)
        torch.save(result, os.path.join(role_dir, "oos_evaluation.pt"))
        fold["oos_metrics"][config.train_task] = metrics
        fold["sample_counts"][config.train_task]["test"] = len(result.y_true)
        log_evaluation_report(logger, task_type=config.train_task, metrics=metrics,
                              save_dir=role_dir, model_label="Frozen_Window_OOS")
        results[config.train_task] = result
    return results


def rolling_windows(start, end, forward_start, config):
    """Anchor calendar windows to the persisted Forward boundary."""
    start, end, forward_start = map(pd.Timestamp, (start, end, forward_start))
    def history_start(oos_start):
        return oos_start - config.validation_offset() - config.training_offset()
    offset = 1
    development = []
    while True:
        oos_start = forward_start - config.update_offset(offset)
        if history_start(oos_start) < start:
            break
        development.append(oos_start)
        offset += 1
    starts = list(reversed(development))
    offset = 0
    while forward_start + config.update_offset(offset) < end:
        starts.append(forward_start + config.update_offset(offset))
        offset += 1
    fixed_train_start = history_start(starts[0]) if starts else None
    for index, oos_start in enumerate(starts):
        if history_start(oos_start) < start:
            raise ValueError("Insufficient history for the first Forward training window")
        phase = "development" if oos_start < forward_start else "forward"
        boundary = forward_start if phase == "development" else end
        yield {
            "fold_id": f"{index:04d}", "phase": phase,
            "train_start": (
                fixed_train_start if config.mode == "expanding"
                else history_start(oos_start)
            ).isoformat(),
            "validation_start": (oos_start - config.validation_offset()).isoformat(),
            "oos_start": oos_start.isoformat(),
            "oos_end": min(
                starts[index + 1] if index + 1 < len(starts) else end, boundary,
            ).isoformat(),
        }


def _save_rolling_binary_oos_curves(stitched, manifest, configs, save_dir, logger):
    """Save pooled breach curves for all OOS folds and each research phase."""
    if len(configs) != 1 or configs[0].train_task != TrainTask.MARTINGALE_BINARY:
        return
    phase_folds = {
        phase: {fold["fold_id"] for fold in manifest["folds"] if fold["phase"] == phase}
        for phase in ("development", "forward")
    }
    curves = {}
    for name, frame in (
        ("all_oos", stitched),
        ("development_oos", stitched.loc[stitched["fold_id"].isin(phase_folds["development"])]),
        ("forward_oos", stitched.loc[stitched["fold_id"].isin(phase_folds["forward"])]),
    ):
        if frame.empty:
            continue
        summary = plot_rolling_binary_oos_curves(
            frame, configs[0].max_layers, save_dir, name,
        )
        if summary is None:
            logger.warning("Skipping pooled OOS curves for %s: one class or no valid labels", name)
            continue
        curves[name] = {
            **summary,
            "threshold_precision_recall": f"{name}_threshold_precision_recall.png",
            "precision_recall": f"{name}_pr_curve.png",
        }
        logger.info("Pooled OOS breach curves saved | phase=%s | rows=%d | events=%d | AP=%.6f",
                    name, summary["rows"], summary["events"], summary["average_precision"])
    manifest["binary_oos_curves"] = curves


def _execute_windows(
    *, configs, folds, save_dir, provenance, preparation, interval_ms,
    prep_output_dir, symbols, frame, manifest, write_manifest, device, logger,
) -> dict:
    """Execute every window through one single- or multiasset training flow.

    Windows are generated independently per symbol (no cross-symbol sequences)
    and concatenated with a per-sample asset id. Equal asset totals are enforced
    through TensorSplit.balanced_sample_weights. OOS inference runs per symbol and
    predictions are keyed by (symbol, timestamp).
    """
    from model.train_config import COMBO_SUB_TASKS
    from model.model_loader import ModelHandler

    tasks = tuple(config.train_task for config in configs)
    combo = next((name for name, heads in COMBO_SUB_TASKS.items() if tasks == heads), None)
    if "symbol" not in frame.columns:
        raise ValueError("Multiasset preparation must persist a 'symbol' column")
    interval = pd.Timedelta(milliseconds=interval_ms)

    # Snapshot the available feature columns so the full merged frame can be
    # released once the per-symbol frames are built (memory saving).
    available_features = set(frame.columns)

    symbol_frames = {}
    symbol_times = {}
    for symbol in symbols:
        sub = frame.loc[frame["symbol"] == symbol]
        if sub.empty:
            raise ValueError(f"No prepared rows for symbol {symbol}")
        sub = sub.sort_values("open_time_date_utc").reset_index(drop=True)
        times = pd.to_datetime(sub["open_time_date_utc"], utc=True)
        if not times.is_unique or not times.is_monotonic_increasing:
            raise ValueError(f"Symbol {symbol} timestamps must be unique and increasing")
        symbol_frames[symbol] = sub
        symbol_times[symbol] = times
    asset_ids = {symbol: index for index, symbol in enumerate(symbols)}
    del frame

    predictions = []
    window_predictions = []
    development_predictions = []
    evaluations = {config.train_task: [] for config in configs}
    for fold in folds:
        logger.info("Rolling fold %s | %s | OOS %s -> %s", fold["fold_id"],
                    fold["phase"], fold["oos_start"], fold["oos_end"])
        train_start, valid_start, oos_start, oos_end = (
            pd.Timestamp(fold[key]) for key in
            ("train_start", "validation_start", "oos_start", "oos_end")
        )
        fold_dir = os.path.join(save_dir, fold["fold_id"])
        role_dirs = {}
        fitted_models = {}
        fold["sample_counts"] = {}
        feature_names = None
        for config in configs:
            if config.feature_conf_list is None:
                factory = common.FeatureFactory(interval_ms)
                requested = [name for name in dict.fromkeys(
                    [*factory.base_features, *factory.all_feature_list]
                ) if name in available_features]
            else:
                requested = list(config.feature_conf_list)
            seq_len = config.model_cfg.seq_len

            def _history_frame(symbol):
                sub = symbol_frames[symbol]
                times = symbol_times[symbol]
                history_mask = (
                    (times >= train_start - interval * seq_len)
                    & (times < oos_start)
                )
                return sub.loc[history_mask], times

            # Pass 1 (light): count per-symbol windows and resolve the feature
            # schema without materializing the full window tensor.
            total = 0
            for symbol in symbols:
                history_frame, _ = _history_frame(symbol)
                if history_frame.empty:
                    continue
                dataset = TimeSeriesWindowDataset(
                    history_frame, interval_ms, requested, config.data_cfg.label_col,
                    seq_len, stride=config.stride,
                    materialize_windows=False, show_feature_distribution=False,
                )
                if len(dataset) == 0:
                    continue
                if feature_names is None:
                    feature_names = list(dataset.feature_names)
                elif feature_names != list(dataset.feature_names):
                    raise ValueError("Fusion feature schemas do not match")
                total += len(dataset)
            if feature_names is None:
                raise ValueError(f"No samples materialized for task {config.train_task}")
            feature_count = len(feature_names)

            # Pass 2 (fill): materialize each symbol's windows into one shared
            # buffer, releasing each symbol's tensor as soon as it is copied.
            ys, rets, ts, assets, maturities = [], [], [], [], []
            X = torch.empty((total, seq_len, feature_count), dtype=torch.float32)
            cursor = 0
            for symbol in symbols:
                history_frame, times = _history_frame(symbol)
                if history_frame.empty:
                    continue
                dataset = TimeSeriesWindowDataset(
                    history_frame, interval_ms, requested, config.data_cfg.label_col,
                    seq_len, stride=config.stride,
                    show_feature_distribution=False,
                )
                if len(dataset) == 0:
                    continue
                count = len(dataset)
                X[cursor:cursor + count] = dataset.X
                ys.append(dataset.y)
                rets.append(dataset.returns)
                ts.append(times.loc[dataset.indices].to_numpy(dtype="datetime64[ns]"))
                event_end = label_maturity(history_frame.loc[dataset.indices], preparation, interval)
                maturities.append(event_end.to_numpy(dtype="datetime64[ns]"))
                assets.append(np.full(count, asset_ids[symbol], dtype=np.int64))
                cursor += count
                del dataset
            y = torch.cat(ys, dim=0)
            returns = torch.cat(rets, dim=0)
            times_all = np.concatenate(ts)
            assets_all = torch.from_numpy(np.concatenate(assets)) if len(symbols) > 1 else None
            maturity = np.concatenate(maturities)

            def select(lower, upper):
                lower64 = pd.Timestamp(lower).to_datetime64()
                upper64 = pd.Timestamp(upper).to_datetime64()
                mask = torch.from_numpy(
                    (times_all >= lower64) & (times_all < upper64) & (maturity <= upper64)
                )
                return TensorSplit(X[mask], y[mask], returns[mask], assets_all[mask] if assets_all is not None else None)

            train_split = select(train_start, valid_start)
            valid_lower = valid_start + interval * (seq_len - 1 if config.data_cfg.purge_overlap else 0)
            valid_split = select(valid_lower, oos_start)
            if not len(train_split) or not len(valid_split):
                raise ValueError(f"Empty train/validation split for task {config.train_task}")
            fold["sample_counts"][config.train_task] = {
                "train": len(train_split), "validation": len(valid_split),
            }
            role_dir = os.path.join(fold_dir, config.train_task)
            prepared = WindowTrainingData(
                train_split, valid_split, feature_names,
                requested, len(feature_names), prep_output_dir,
            )
            fitted = _fit_window_model(
                task_type=config.train_task, config=config, prepared=prepared,
                save_dir=role_dir, device=device, logger=logger,
            )
            fitted_models[config.train_task] = fitted
            role_dirs[config.train_task.lower()] = role_dir
            with open(os.path.join(role_dir, "window_metadata.json"), "w", encoding="utf-8") as handle:
                json.dump({**fold, "data_id": provenance["data_id"],
                           "training_root": os.path.relpath(save_dir, role_dir)}, handle, indent=2)
            del prepared, train_split, valid_split, X, y, returns, ys, rets
        oos_results = _evaluate_oos_models(
            configs, fitted_models, symbol_frames, preparation, interval_ms,
            oos_start, oos_end, fold_dir, fold, device, logger,
        )
        for task_type, result in oos_results.items():
            evaluations[task_type].append(result)
        task_strategies = {task_type: fitted.task for task_type, fitted in fitted_models.items()}
        del fitted, fitted_models
        model_dir = next(iter(role_dirs.values()))
        if combo:
            model_dir = os.path.join(fold_dir, "fusion")
            save_fusion_run(task_type=combo, role_directories=role_dirs, fusion_dir=model_dir)
            with open(os.path.join(model_dir, "window_metadata.json"), "w", encoding="utf-8") as handle:
                json.dump({**fold, "data_id": provenance["data_id"],
                           "training_root": os.path.relpath(save_dir, model_dir)}, handle, indent=2)
        handler = ModelHandler(model_dir, device)
        # The first fold also predicts its own train/validation span so the
        # development backtest can compare in-sample against OOS behaviour.
        first_fold = fold["fold_id"] == folds[0]["fold_id"]
        window_start = train_start if first_fold else oos_start
        segment_parts = []
        columns = ["open_time_date_utc", "pred", "direction_pred", "pred_prob",
                   "prob_short", "prob_neutral", "prob_long", "net_score", "regression_score"]
        for symbol in symbols:
            sub = symbol_frames[symbol]
            times = symbol_times[symbol]
            inference_mask = (
                (times >= window_start - interval * (handler.seq_len + 1))
                & (times < oos_end)
            )
            inference_frame = sub.loc[inference_mask].copy()
            features_only = inference_frame.drop(columns=[handler.label_col, "trend_strength"], errors="ignore")
            dataset = TimeSeriesWindowDataset(
                features_only, interval_ms, handler.feature_cols, None, handler.seq_len,
                is_live=False, materialize_windows=False, show_feature_distribution=False,
            )
            predict = (handler.predict_binary_component_with_ds
                       if handler.task_type in (TrainTask.TRIGGER, TrainTask.DIRECTION,
                                                TrainTask.LONG_OVR, TrainTask.SHORT_OVR)
                       else handler.predict_with_ds)
            predicted, _, _, sub_predictions = predict(
                dataset, features_only, is_live=False, batch_size=configs[0].batch_size,
                evaluate_sub_models=False, collect_sub_predictions=True,
            )
            if handler.task_type == TrainTask.MARTINGALE_BINARY:
                binary_columns = columns + ["survival_probability", "breach_probability"]
            else:
                binary_columns = columns
            segment = predicted.loc[(times.loc[predicted.index] >= window_start), binary_columns].copy()
            if preparation.label_type == "MARTINGALE":
                segment[handler.label_col] = inference_frame.loc[segment.index, handler.label_col]
                segment["martingale_label"] = segment[handler.label_col]
                if "open_time_ms_utc" in inference_frame:
                    segment["open_time_ms_utc"] = inference_frame.loc[segment.index, "open_time_ms_utc"]
            segment["fold_id"] = fold["fold_id"]
            if segment.empty or not segment["pred"].notna().any():
                continue
            if len(sub_predictions.columns):
                segment = segment.join(sub_predictions)
            segment["symbol"] = symbol
            segment_parts.append(segment)
        if not segment_parts:
            raise RuntimeError("Rolling window has no valid predictions")
        window = pd.concat(segment_parts, ignore_index=True)
        window = annotate_predictions(window, fold)
        validate_predictions(window, {"folds": [*manifest["folds"], fold]}, allow_in_sample=True)
        window_predictions.append(window)
        window_path = os.path.join(fold_dir, "window_predictions.pkl")
        window.to_pickle(window_path)
        fold["window_predictions"] = os.path.relpath(window_path, save_dir)
        if fold["phase"] == "development":
            development_predictions.append(window)
        segment_times = pd.to_datetime(window["open_time_date_utc"], utc=True)
        segment = window.loc[segment_times >= oos_start]
        if segment.empty:
            raise RuntimeError("Rolling OOS has no valid predictions")
        fold_prediction_path = os.path.join(fold_dir, "oos_predictions.pkl")
        segment.to_pickle(fold_prediction_path)
        fold["predictions"] = os.path.relpath(fold_prediction_path, save_dir)
        predictions.append(segment)
        fold["model_dir"] = os.path.relpath(model_dir, save_dir)
        fold["prediction_count"] = len(segment)
        manifest["folds"].append(fold)
        write_manifest()
        del handler
    stitched = pd.concat(predictions, ignore_index=True)
    validate_predictions(stitched, manifest)
    if stitched.duplicated(subset=["symbol", "open_time_date_utc"]).any():
        raise RuntimeError("Rolling OOS predictions overlap")
    stitched.to_pickle(os.path.join(save_dir, manifest["predictions"]))
    development = pd.concat(development_predictions, ignore_index=True) if development_predictions else stitched.iloc[:0].copy()
    if development.duplicated(subset=["symbol", "open_time_date_utc"]).any():
        raise RuntimeError("Rolling development predictions overlap")
    development.to_pickle(os.path.join(save_dir, "development_predictions.pkl"))
    all_windows = pd.concat(window_predictions, ignore_index=True)
    validate_predictions(all_windows, manifest, allow_in_sample=True)
    all_windows.to_pickle(os.path.join(save_dir, "window_predictions.pkl"))
    manifest["window_predictions"] = "window_predictions.pkl"
    manifest["development_predictions"] = "development_predictions.pkl"
    manifest["prediction_columns"] = list(stitched.columns)
    manifest["oos_metrics"] = {}
    for task_type, results in evaluations.items():
        if results:
            combined = _combine_predictions(results)
            metrics = _evaluation_metrics(combined, task_strategies[task_type])
            manifest["oos_metrics"][task_type] = metrics
            torch.save(combined, os.path.join(save_dir, f"{task_type}_oos_evaluation.pt"))
    with open(os.path.join(save_dir, "oos_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest["oos_metrics"], handle, indent=2)
    _save_rolling_binary_oos_curves(stitched, manifest, configs, save_dir, logger)
    manifest["status"] = "complete"
    write_manifest()
    logger.info("Rolling training complete | folds=%d | OOS rows=%d", len(folds), len(stitched))
    logger.info("Rolling artifact directory: %s", os.path.abspath(save_dir))
    return manifest


def train_windows(
    *, configs=None, windows=None, rolling_config=None,
    prep_output_dir=common.DATA_OUT_DIR, save_dir=None, logger=None,
    include_forward=True,
) -> dict:
    """Train fresh models on one explicit window or a sequence of windows.

    With no explicit windows, a RollingConfig generates calendar windows;
    otherwise the development history generates one ratio-based window.
    Model selection is independent of the window mode.
    """
    from model.train_config import COMBO_SUB_TASKS
    from model.artifacts import copy_data_config_meta

    logger = logger or logging.getLogger("train")
    configs = list(configs if configs is not None else [SingleModelConfig])
    if len(configs) not in (1, 2):
        raise ValueError("Window training requires one model or two ordered fusion heads")
    tasks = tuple(config.train_task for config in configs)
    combo = next((name for name, heads in COMBO_SUB_TASKS.items() if tasks == heads), None)
    if len(configs) == 2 and combo is None:
        raise ValueError("Fusion heads must be an ordered supported task pair")
    if any(config.model_cfg.seq_len != configs[0].model_cfg.seq_len or
           config.feature_conf_list != configs[0].feature_conf_list or
           config.data_cfg != configs[0].data_cfg for config in configs):
        raise ValueError("Fusion heads must share sequence length, features and split configuration")
    if windows is not None and rolling_config is not None:
        raise ValueError("Pass explicit windows or rolling_config, not both")
    provenance = common.load_data_manifest_from_dir(prep_output_dir)
    preparation = common.load_pre_params_from_dir(prep_output_dir)
    for config in configs:
        validate_preparation_config(preparation, config.train_task)
    interval_ms = common.get_interval_ms(preparation.interval)
    interval = pd.Timedelta(milliseconds=interval_ms)
    frame = pd.concat([
        common.load_train_df_from_dir(prep_output_dir),
        common.load_test_df_from_dir(prep_output_dir),
    ], ignore_index=True)
    symbols = tuple(preparation.symbols)
    if "symbol" not in frame:
        if len(symbols) != 1:
            raise ValueError("Multiasset preparation must persist a symbol column")
        frame["symbol"] = symbols[0]
    if set(frame["symbol"].unique()) != set(symbols):
        raise ValueError("Prepared symbols do not match the preparation configuration")
    times = pd.to_datetime(frame["open_time_date_utc"], utc=True)
    for symbol in symbols:
        asset_times = times[frame["symbol"] == symbol]
        if not asset_times.is_unique or not asset_times.is_monotonic_increasing:
            raise ValueError(f"Symbol {symbol} timestamps must be unique and increasing")
    forward_start = pd.Timestamp(provenance["split_time"])
    if forward_start.tzinfo is None:
        forward_start = forward_start.tz_localize("UTC")
    mode = "explicit" if windows is not None else rolling_config.mode if rolling_config else "single"
    if windows is not None:
        folds = list(windows)
    elif rolling_config is not None:
        folds = list(rolling_windows(times.min(), times.max() + interval, forward_start, rolling_config))
    else:
        history = times[times < forward_start].sort_values().unique()
        folds = single_window(history, configs[0].data_cfg, interval)
    folds = validate_windows(folds, forward_start)
    if not include_forward:
        folds = [fold for fold in folds if fold["phase"] == "development"]
    if not folds:
        raise ValueError("No eligible training windows")
    if save_dir is None:
        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ")
        save_dir = os.path.join(common.TRAIN_OUT_DIR, f"{mode}_{stamp}")
    if os.path.exists(os.path.join(save_dir, "rolling_manifest.json")):
        raise FileExistsError("Training output already exists; select a new save_dir")
    copy_data_config_meta(prep_output_dir, save_dir)
    manifest = {
        "status": "running", "data_id": provenance["data_id"],
        "prediction_contract": 1,
        "include_forward": include_forward, "window_mode": mode,
        "forward_start": forward_start.isoformat(),
        "rolling_config": asdict(rolling_config) if rolling_config else None,
        "configs": [asdict(config) for config in configs],
        "planned_windows": [dict(fold) for fold in folds],
        "folds": [], "predictions": "oos_predictions.pkl",
    }

    def write_manifest():
        temporary = os.path.join(save_dir, "rolling_manifest.json.tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        os.replace(temporary, os.path.join(save_dir, "rolling_manifest.json"))

    with open(os.path.join(save_dir, "train_config.json"), "w", encoding="utf-8") as handle:
        json.dump(asdict(configs[0]), handle, indent=2)
    write_manifest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Window training | mode=%s | folds=%d | tasks=%s | device=%s",
                mode, len(folds), ",".join(tasks), device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    try:
        return _execute_windows(
            configs=configs, folds=folds, save_dir=save_dir,
            provenance=provenance, preparation=preparation, interval_ms=interval_ms,
            prep_output_dir=prep_output_dir, symbols=symbols,
            frame=frame, manifest=manifest, write_manifest=write_manifest,
            device=device, logger=logger,
        )
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        write_manifest()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models on one or more chronological windows")
    planning = parser.add_mutually_exclusive_group()
    planning.add_argument("--rolling", dest="rolling", action="store_true",
                          help="Use rolling training windows (default)")
    planning.add_argument("--single", dest="rolling", action="store_false",
                          help="Use one ratio-based training window")
    planning.add_argument("--windows-file", help="JSON list of explicit chronological windows")
    parser.set_defaults(rolling=True)
    parser.add_argument("--mode", choices=("sliding", "expanding"), default="expanding",
                        help="Training window mode used with --rolling")
    parser.add_argument("--prep-output-dir", default=common.DATA_OUT_DIR)
    parser.add_argument("--save-dir")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--validation-months", type=int, default=2)
    parser.add_argument("--oos-months", type=int, default=3)
    parser.add_argument("--oos-days", type=int, help="Retrain every N days; overrides --oos-months")
    parser.add_argument("--train-days", type=int, help="Training history in days; overrides --train-months")
    parser.add_argument("--validation-days", type=int, help="Validation history in days; overrides --validation-months")
    args = parser.parse_args()
    windows = None
    if args.windows_file:
        with open(args.windows_file, encoding="utf-8") as handle:
            windows = json.load(handle)
        if not isinstance(windows, list):
            parser.error("--windows-file must contain a JSON list")
    started_at = time.perf_counter()
    session_logger, _ = common.setup_session_logger(sub_folder="train", file_level=logging.DEBUG)
    try:
        train_windows(
            configs=[SingleModelConfig],
            windows=windows,
            rolling_config=RollingConfig(
                mode=args.mode, train_months=args.train_months,
                validation_months=args.validation_months, oos_months=args.oos_months,
                oos_days=args.oos_days, train_days=args.train_days, validation_days=args.validation_days,
            ) if args.rolling and windows is None else None,
            prep_output_dir=args.prep_output_dir, save_dir=args.save_dir, logger=session_logger,
        )
    finally:
        session_logger.info("Total run time: %.2f s", time.perf_counter() - started_at)
