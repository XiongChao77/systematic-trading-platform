#!/usr/bin/env python3
from __future__ import annotations

import logging
import os,sys
import random
import time
import argparse
import json

import pandas as pd
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir,'..'))
from data_process import common
from model.artifacts import save_fusion_run, save_single_run
from model.data_loader import TimeSeriesWindowDataset
from model.evaluator import evaluate_classification, evaluate_regression, log_evaluation_report
from model.model_factory import ModelFactory
from model.tasks.strategies import build_task
from model.train_config import (
    DataConfig,
    LightGBMConfig,
    SingleModelConfig,
    TrainConfig,
    TrainTask,
)
from model.tasks.strategies import TaskStrategy
from model.trainer import build_trainer
from model.training_types import DataSplits, TensorSplit, temporal_split_bounds


@dataclass(frozen=True)
class PreparedData:
    splits: DataSplits
    feature_names: list[str]
    requested_features: list[str]
    feature_count: int
    prep_output_dir: str


@dataclass
class SingleRun:
    task_type: str
    model: torch.nn.Module
    report: dict
    save_dir: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_time(
    dataset: TimeSeriesWindowDataset,
    data_config: DataConfig,
) -> DataSplits:
    """Split once on the original timeline; every task derives from these splits."""
    bounds = temporal_split_bounds(
        len(dataset),
        train_ratio=data_config.train_ratio,
        val_ratio=data_config.val_ratio,
        purge_overlap=data_config.purge_overlap,
        seq_len=dataset.seq_len,
        stride=dataset.stride,
    )

    def make(start: int, end: int) -> TensorSplit:
        return TensorSplit(
            dataset.X[start:end],
            dataset.y[start:end],
            dataset.returns[start:end],
        )

    return DataSplits(
        train=make(*bounds["train"]),
        validation=make(*bounds["valid"]),
        test=make(*bounds["test"]),
    )


def prepare_data(
    *,
    config: TrainConfig,
    prep_output_dir: str,
    cache_dir: str,
    logger: logging.Logger,
) -> PreparedData:
    data_manifest = common.load_data_manifest_from_dir(prep_output_dir)
    logger.info(
        "Loaded prepared data provenance: data_id=%s, time=%s -> %s",
        data_manifest["data_id"],
        data_manifest["time"]["start"],
        data_manifest["time"]["end"],
    )
    frame = common.load_train_df_from_dir(prep_output_dir)
    preparation = common.load_pre_params_from_dir(prep_output_dir)
    interval_ms = common.get_interval_ms(preparation.interval)
    if config.feature_conf_list is None:
        feature_factory = common.FeatureFactory(interval_ms)
        candidate_features = dict.fromkeys(
            [*feature_factory.base_features, *feature_factory.all_feature_list]
        )
        requested_features = [
            name for name in candidate_features if name in frame.columns
        ]
        if not requested_features:
            raise ValueError("No model features are available in the prepared dataset")
    else:
        requested_features = list(config.feature_conf_list)

    dataset = TimeSeriesWindowDataset(
        df=frame,
        kline_interval_ms=interval_ms,
        feature_cols=requested_features,
        label_col=config.data_cfg.label_col,
        seq_len=config.model_cfg.seq_len,
        stride=config.stride,
        cache_path=os.path.join(cache_dir, "train_cache.pt"),
        use_cache=config.use_cache,
        show_feature_distribution=True,
    )
    splits = split_by_time(dataset, config.data_cfg)
    logger.info(
        "Prepared %d samples: train=%d validation=%d test=%d, features=%d",
        len(dataset),
        len(splits.train),
        len(splits.validation),
        len(splits.test),
        dataset.feature_count,
    )
    return PreparedData(
        splits=splits,
        feature_names=list(dataset.feature_names),
        requested_features=requested_features,
        feature_count=dataset.feature_count,
        prep_output_dir=prep_output_dir,
    )


def _output_dim(task_type: str) -> int:
    if task_type == TrainTask.REGRESSION:
        return 1
    return 2 if task_type in TrainTask.BINARY_TASKS else 3


def _build_model(config: TrainConfig, task_type: str, input_size: int, device: torch.device) -> torch.nn.Module:
    model_parameters = asdict(config.model_cfg)
    return ModelFactory.build_for_training(
        device=device,
        input_size=input_size,
        n_classes=_output_dim(task_type),
        **model_parameters,
    )


def _dataset_splits(task:TaskStrategy, raw: DataSplits) -> DataSplits:
    if task.source_labels is not None:
        missing_train_labels = set(task.source_labels) - set(raw.train.labels.tolist())
    else:
        missing_train_labels = set()
    if missing_train_labels:
        raise ValueError(
            f"Training split is missing labels required by the task: {sorted(missing_train_labels)}"
        )

    transformed = DataSplits(
        train=task.dataset_split(raw.train),
        validation=task.dataset_split(raw.validation),
        test=task.dataset_split(raw.test),
    )
    for name, split in (
        ("train", transformed.train),
        ("validation", transformed.validation),
        ("test", transformed.test),
    ):
        if not len(split):
            raise ValueError(f"{name} split is empty after task transformation")
    missing_train_labels = (
        set()
        if task.is_regression or task.source_labels is not None
        else set(task.labels) - set(transformed.train.labels.tolist())
    )
    if missing_train_labels:
        raise ValueError(
            f"Training split is missing labels required by the task: {sorted(missing_train_labels)}"
        )
    return transformed


def train_single(
    *,
    task_type: str,
    config: TrainConfig,
    prepared: PreparedData,
    save_dir: str,
    device: torch.device,
    logger: logging.Logger,
    evaluate_test: bool = True,
) -> SingleRun:
    if task_type in TrainTask.COMBO_TASKS:
        raise ValueError(f"{task_type} is a combo task, not a single-model task")

    set_seed(config.seed)
    model = _build_model(config, task_type, prepared.feature_count, device)
    task = build_task(task_type, config, device, model=model)
    splits = _dataset_splits(task, prepared.splits)
    trainer = build_trainer(model, config, device, logger)

    fit_result = trainer.fit(
        model, task, splits.train, splits.validation,
        splits.test if evaluate_test else None,
    )
    prediction = trainer.predict(
        model, task, splits.test if evaluate_test else splits.validation,
    )
    metrics = (
        evaluate_regression(prediction, task.labels)
        if task.is_regression
        else evaluate_classification(prediction, task.labels)
    )
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
        evaluation_split="test" if evaluate_test else "validation",
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
    return SingleRun(task_type=task_type, model=model, report=report, save_dir=save_dir)


def fusion_trigger_dir(logger: logging.Logger, trigger_dir: str, direction_dir: str, fusion_dir: str) -> dict:
    """Fuse two independently trained models (trigger + direction) into one artifact dir.

    Both models must already be trained (e.g. via `train()` with `TrainTask.TRIGGER`
    and `TrainTask.DIRECTION`). This does not retrain or evaluate anything; it only
    records where each checkpoint lives so inference code can load and fuse them.
    """
    description = save_fusion_run(
        task_type=TrainTask.TRIGGER_DIRECTION,
        role_directories={"trigger": trigger_dir, "direction": direction_dir},
        fusion_dir=fusion_dir,
    )
    logger.info("Fused TRIGGER_DIRECTION: trigger=%s direction=%s -> %s", trigger_dir, direction_dir, fusion_dir)
    return description


def fusion_long_short_ovr(logger: logging.Logger, long_ovr_dir: str, short_ovr_dir: str, fusion_dir: str) -> dict:
    """Fuse two independently trained models (long OvR + short OvR) into one artifact dir.

    Both models must already be trained (e.g. via `train()` with `TrainTask.LONG_OVR`
    and `TrainTask.SHORT_OVR`). This does not retrain or evaluate anything; it only
    records where each checkpoint lives so inference code can load and fuse them.
    """
    description = save_fusion_run(
        task_type=TrainTask.LONG_SHORT_OVR,
        role_directories={"long_ovr": long_ovr_dir, "short_ovr": short_ovr_dir},
        fusion_dir=fusion_dir,
    )
    logger.info("Fused LONG_SHORT_OVR: long_ovr=%s short_ovr=%s -> %s", long_ovr_dir, short_ovr_dir, fusion_dir)
    return description


def train(
    *,
    config: TrainConfig,
    prep_output_dir: str = common.DATA_OUT_DIR,
    save_dir: str = common.TRAIN_OUT_DIR,
    logger: logging.Logger | None = None,
) -> dict:
    """Single public training entry point: trains exactly one model for one TrainConfig.

    To combine two single-head models (trigger+direction or long/short OvR), train each
    with its own TrainConfig via this function, then fuse the saved checkpoints with
    fusion_trigger_dir / fusion_long_short_ovr.
    """
    logger = logger or logging.getLogger("train")
    task_type = config.train_task
    if task_type in TrainTask.COMBO_TASKS:
        raise ValueError(
            f"{task_type} trains two models jointly and is no longer supported here; "
            "train each sub-task separately with train() and fuse the saved checkpoints "
            "with fusion_trigger_dir/fusion_long_short_ovr"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_device = device.type
    device_label = {
        "cuda": "GPU (CUDA)",
        "cpu": "CPU",
    }.get(training_device, training_device)
    logger.info("Training device: %s", device_label)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    prepared = prepare_data(
        config=config,
        prep_output_dir=prep_output_dir,
        cache_dir=save_dir,
        logger=logger,
    )
    return train_single(
        task_type=task_type,
        config=config,
        prepared=prepared,
        save_dir=save_dir,
        device=device,
        logger=logger,
    ).report


def rolling_windows(start, end, forward_start, config):
    """Anchor calendar windows to the persisted Forward boundary."""
    start, end, forward_start = map(pd.Timestamp, (start, end, forward_start))
    history = config.train_months + config.validation_months
    offset = 1
    development = []
    while True:
        oos_start = forward_start - pd.DateOffset(months=offset * config.oos_months)
        if oos_start - pd.DateOffset(months=history) < start:
            break
        development.append(oos_start)
        offset += 1
    starts = list(reversed(development))
    offset = 0
    while forward_start + pd.DateOffset(months=offset * config.oos_months) < end:
        starts.append(forward_start + pd.DateOffset(months=offset * config.oos_months))
        offset += 1
    fixed_train_start = starts[0] - pd.DateOffset(months=history) if starts else None
    for index, oos_start in enumerate(starts):
        if oos_start - pd.DateOffset(months=history) < start:
            raise ValueError("Insufficient history for the first Forward training window")
        phase = "development" if oos_start < forward_start else "forward"
        boundary = forward_start if phase == "development" else end
        yield {
            "fold_id": f"{index:04d}", "phase": phase,
            "train_start": (
                fixed_train_start if config.mode == "expanding"
                else oos_start - pd.DateOffset(months=history)
            ).isoformat(),
            "validation_start": (oos_start - pd.DateOffset(months=config.validation_months)).isoformat(),
            "oos_start": oos_start.isoformat(),
            "oos_end": min(
                starts[index + 1] if index + 1 < len(starts) else end, boundary,
            ).isoformat(),
        }


def _train_rolling_multiasset(
    *, configs, rolling_config, save_dir, provenance, preparation, interval_ms,
    prep_output_dir, symbols, frame, manifest, write_manifest, device, logger,
) -> dict:
    """Rolling joint training over multiple symbols sharing one calendar.

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
    horizon = int(preparation.predict_num)

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

    start = pd.Timestamp(min(t.iloc[0] for t in symbol_times.values()))
    end = pd.Timestamp(max(t.iloc[-1] for t in symbol_times.values())) + interval
    folds = list(rolling_windows(start, end, pd.Timestamp(provenance["split_time"]), rolling_config))
    if not folds or not any(fold["phase"] == "development" for fold in folds):
        raise ValueError("Insufficient history for development OOS evaluation")

    predictions = []
    for fold in folds:
        logger.info("Rolling fold %s | %s | OOS %s -> %s", fold["fold_id"],
                    fold["phase"], fold["oos_start"], fold["oos_end"])
        train_start, valid_start, oos_start, oos_end = (
            pd.Timestamp(fold[key]) for key in
            ("train_start", "validation_start", "oos_start", "oos_end")
        )
        fold_dir = os.path.join(save_dir, fold["fold_id"])
        role_dirs = {}
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
            ys, rets, ts, assets = [], [], [], []
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
                assets.append(np.full(count, asset_ids[symbol], dtype=np.int64))
                cursor += count
                del dataset
            y = torch.cat(ys, dim=0)
            returns = torch.cat(rets, dim=0)
            times_all = np.concatenate(ts)
            assets_all = torch.from_numpy(np.concatenate(assets))
            maturity = times_all + np.timedelta64(interval_ms * (horizon + 1), "ms")

            def select(lower, upper):
                lower64 = pd.Timestamp(lower).to_datetime64()
                upper64 = pd.Timestamp(upper).to_datetime64()
                mask = torch.from_numpy(
                    (times_all >= lower64) & (times_all < upper64) & (maturity <= upper64)
                )
                return TensorSplit(X[mask], y[mask], returns[mask], assets_all[mask])

            train_split = select(train_start, valid_start)
            valid_split = select(valid_start, oos_start)
            if not len(train_split) or not len(valid_split):
                raise ValueError(f"Empty train/validation split for task {config.train_task}")
            fold["sample_counts"][config.train_task] = {
                "train": len(train_split), "validation": len(valid_split),
            }
            role_dir = os.path.join(fold_dir, config.train_task)
            prepared = PreparedData(
                DataSplits(train_split, valid_split, valid_split), feature_names,
                requested, len(feature_names), prep_output_dir,
            )
            train_single(task_type=config.train_task, config=config, prepared=prepared,
                         save_dir=role_dir, device=device, logger=logger, evaluate_test=False)
            role_dirs[config.train_task.lower()] = role_dir
        model_dir = next(iter(role_dirs.values()))
        if combo:
            model_dir = os.path.join(fold_dir, "fusion")
            save_fusion_run(task_type=combo, role_directories=role_dirs, fusion_dir=model_dir)
        handler = ModelHandler(model_dir, device)
        segment_parts = []
        columns = ["open_time_date_utc", "pred", "direction_pred", "pred_prob",
                   "prob_short", "prob_neutral", "prob_long", "net_score"]
        for symbol in symbols:
            sub = symbol_frames[symbol]
            times = symbol_times[symbol]
            inference_mask = (
                (times >= oos_start - interval * (handler.seq_len + 1))
                & (times < oos_end)
            )
            inference_frame = sub.loc[inference_mask].copy()
            features_only = inference_frame.drop(columns=[handler.label_col, "trend_strength"], errors="ignore")
            dataset = TimeSeriesWindowDataset(
                features_only, interval_ms, handler.feature_cols, None, handler.seq_len,
                is_live=False, materialize_windows=False, show_feature_distribution=False,
            )
            predicted, _, _, sub_predictions = handler.predict_with_ds(
                dataset, features_only, is_live=False, batch_size=configs[0].batch_size,
                evaluate_sub_models=False, collect_sub_predictions=True,
            )
            segment = predicted.loc[(times.loc[predicted.index] >= oos_start), columns].copy()
            if segment.empty or not segment["pred"].notna().any():
                continue
            if len(sub_predictions.columns):
                segment = segment.join(sub_predictions)
            segment["symbol"] = symbol
            segment_parts.append(segment)
        if not segment_parts:
            raise RuntimeError("Rolling OOS has no valid predictions")
        segment = pd.concat(segment_parts, ignore_index=True)
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
    if stitched.duplicated(subset=["symbol", "open_time_date_utc"]).any():
        raise RuntimeError("Rolling OOS predictions overlap")
    stitched.to_pickle(os.path.join(save_dir, manifest["predictions"]))
    manifest["prediction_columns"] = list(stitched.columns)
    manifest["status"] = "complete"
    write_manifest()
    logger.info("Rolling training complete | folds=%d | OOS rows=%d", len(folds), len(stitched))
    logger.info("Rolling artifact directory: %s", os.path.abspath(save_dir))
    return manifest


def train_rolling(
    *, configs=None, rolling_config=None,
    prep_output_dir=common.DATA_OUT_DIR, save_dir=None, logger=None,
) -> dict:
    """Retrain fresh heads for each monthly OOS, including the Forward period."""
    from model.train_config import COMBO_SUB_TASKS, RollingConfig, TriggerModel, DirectionModel
    from model.artifacts import copy_data_config_meta
    from model.model_loader import ModelHandler

    logger = logger or logging.getLogger("train")
    configs = list(configs if configs is not None else (TriggerModel, DirectionModel))
    rolling_config = rolling_config or RollingConfig()
    if len(configs) not in (1, 2):
        raise ValueError("Rolling training requires one model or two fusion heads")
    tasks = tuple(config.train_task for config in configs)
    combo = next((name for name, heads in COMBO_SUB_TASKS.items() if tasks == heads), None)
    if len(configs) == 2 and combo is None:
        raise ValueError("Rolling fusion heads must be an ordered supported task pair")
    if any(config.model_cfg.seq_len != configs[0].model_cfg.seq_len or
           config.feature_conf_list != configs[0].feature_conf_list for config in configs):
        raise ValueError("Fusion heads must share sequence length and ordered features")
    save_dir = save_dir or os.path.join(
        common.TRAIN_OUT_DIR, "rolling" if rolling_config.mode == "sliding" else "expanding",
    )
    if os.path.exists(os.path.join(save_dir, "rolling_manifest.json")):
        raise FileExistsError("Rolling output already exists; select a new save_dir")
    provenance = common.load_data_manifest_from_dir(prep_output_dir)
    preparation = common.load_pre_params_from_dir(prep_output_dir)
    if preparation.label_type == "PURE_BINARY":
        raise ValueError("Rolling training requires bounded-horizon labels")
    interval_ms = common.get_interval_ms(preparation.interval)
    frame = pd.concat([
        common.load_train_df_from_dir(prep_output_dir),
        common.load_test_df_from_dir(prep_output_dir),
    ], ignore_index=True)
    symbols = tuple(preparation.symbols)
    multiasset = len(symbols) > 1
    forward_start = pd.Timestamp(provenance["split_time"])
    interval = pd.Timedelta(milliseconds=interval_ms)
    if not multiasset:
        times = pd.to_datetime(frame["open_time_date_utc"], utc=True)
        if not times.is_unique or not times.is_monotonic_increasing:
            raise ValueError("Prepared timestamps must be unique and increasing")
        folds = list(rolling_windows(times.iloc[0], times.iloc[-1] + interval,
                                     forward_start, rolling_config))
        if not folds or not any(fold["phase"] == "development" for fold in folds):
            raise ValueError("Insufficient history for development OOS evaluation")
    else:
        folds = []

    copy_data_config_meta(prep_output_dir, save_dir)
    manifest = {
        "status": "running", "data_id": provenance["data_id"],
        "forward_start": forward_start.isoformat(),
        "rolling_config": asdict(rolling_config),
        "configs": [asdict(config) for config in configs],
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
    if multiasset:
        return _train_rolling_multiasset(
            configs=configs, rolling_config=rolling_config, save_dir=save_dir,
            provenance=provenance, preparation=preparation, interval_ms=interval_ms,
            prep_output_dir=prep_output_dir, symbols=symbols,
            frame=frame, manifest=manifest, write_manifest=write_manifest,
            device=device, logger=logger,
        )
    predictions = []
    for fold in folds:
        logger.info("Rolling fold %s | %s | OOS %s -> %s", fold["fold_id"],
                    fold["phase"], fold["oos_start"], fold["oos_end"])
        train_start, valid_start, oos_start, oos_end = (
            pd.Timestamp(fold[key]) for key in
            ("train_start", "validation_start", "oos_start", "oos_end")
        )
        fold_dir = os.path.join(save_dir, fold["fold_id"])
        role_dirs = {}
        fold["sample_counts"] = {}
        feature_names = None
        for config in configs:
            # Only historical rows are materialized; OOS never enters fitting.
            history_mask = (times >= train_start - interval * config.model_cfg.seq_len) & (times < oos_start)
            history_frame = frame.loc[history_mask]
            if config.feature_conf_list is None:
                factory = common.FeatureFactory(interval_ms)
                requested = [name for name in dict.fromkeys(
                    [*factory.base_features, *factory.all_feature_list]
                ) if name in frame.columns]
            else:
                requested = list(config.feature_conf_list)
            dataset = TimeSeriesWindowDataset(
                history_frame, interval_ms, requested, config.data_cfg.label_col,
                config.model_cfg.seq_len, stride=config.stride,
                show_feature_distribution=False,
            )
            sample_times = times.loc[dataset.indices]
            # Include the full last future candle when deciding label maturity.
            maturity = sample_times + interval * (preparation.predict_num + 1)

            def select(lower, upper):
                mask = torch.tensor(
                    ((sample_times >= lower) & (sample_times < upper) & (maturity <= upper)).to_numpy(),
                    dtype=torch.bool,
                )
                return TensorSplit(dataset.X[mask], dataset.y[mask], dataset.returns[mask])

            train_split = select(train_start, valid_start)
            valid_split = select(valid_start, oos_start)
            fold["sample_counts"][config.train_task] = {
                "train": len(train_split), "validation": len(valid_split),
            }
            if feature_names is not None and feature_names != dataset.feature_names:
                raise ValueError("Fusion feature schemas do not match")
            feature_names = list(dataset.feature_names)
            role_dir = os.path.join(fold_dir, config.train_task)
            prepared = PreparedData(
                DataSplits(train_split, valid_split, valid_split), feature_names,
                requested, dataset.feature_count, prep_output_dir,
            )
            train_single(task_type=config.train_task, config=config, prepared=prepared,
                         save_dir=role_dir, device=device, logger=logger, evaluate_test=False)
            role_dirs[config.train_task.lower()] = role_dir
            del dataset, prepared, train_split, valid_split
        model_dir = next(iter(role_dirs.values()))
        if combo:
            model_dir = os.path.join(fold_dir, "fusion")
            save_fusion_run(task_type=combo, role_directories=role_dirs, fusion_dir=model_dir)
        handler = ModelHandler(model_dir, device)
        inference_mask = (times >= oos_start - interval * (handler.seq_len + 1)) & (times < oos_end)
        inference_frame = frame.loc[inference_mask].copy()
        # Labels and forward-derived returns cannot affect inference eligibility.
        features_only = inference_frame.drop(columns=[handler.label_col, "trend_strength"], errors="ignore")
        dataset = TimeSeriesWindowDataset(
            features_only, interval_ms, handler.feature_cols, None, handler.seq_len,
            is_live=False, materialize_windows=False, show_feature_distribution=False,
        )
        predicted, _, _, sub_predictions = handler.predict_with_ds(
            dataset, features_only, is_live=False, batch_size=configs[0].batch_size,
            evaluate_sub_models=False, collect_sub_predictions=True,
        )
        columns = ["open_time_date_utc", "pred", "direction_pred", "pred_prob",
                   "prob_short", "prob_neutral", "prob_long", "net_score"]
        segment = predicted.loc[(times.loc[predicted.index] >= oos_start), columns].copy()
        if segment.empty or not segment["pred"].notna().any():
            raise RuntimeError("Rolling OOS has no valid predictions")
        if len(sub_predictions.columns):
            segment = segment.join(sub_predictions)
        fold_prediction_path = os.path.join(fold_dir, "oos_predictions.pkl")
        segment.to_pickle(fold_prediction_path)
        fold["predictions"] = os.path.relpath(fold_prediction_path, save_dir)
        predictions.append(segment)
        fold["model_dir"] = os.path.relpath(model_dir, save_dir)
        fold["prediction_count"] = len(segment)
        manifest["folds"].append(fold)
        write_manifest()
        del handler, dataset
    stitched = pd.concat(predictions, ignore_index=True)
    if not stitched["open_time_date_utc"].is_unique:
        raise RuntimeError("Rolling OOS predictions overlap")
    stitched.to_pickle(os.path.join(save_dir, manifest["predictions"]))
    manifest["prediction_columns"] = list(stitched.columns)
    manifest["status"] = "complete"
    write_manifest()
    logger.info("Rolling training complete | folds=%d | OOS rows=%d", len(folds), len(stitched))
    logger.info("Rolling artifact directory: %s", os.path.abspath(save_dir))
    logger.info("Rolling summary manifest: %s", os.path.abspath(os.path.join(save_dir, "rolling_manifest.json")))
    logger.info("Combined OOS predictions: %s", os.path.abspath(os.path.join(save_dir, manifest["predictions"])))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train single or rolling models")
    parser.add_argument("--rolling", action="store_true")
    parser.add_argument("--mode", choices=("sliding", "expanding"), default="expanding",
                        help="Training window mode used with --rolling")
    parser.add_argument("--prep-output-dir", default=common.DATA_OUT_DIR)
    parser.add_argument("--save-dir")
    args = parser.parse_args()
    config = SingleModelConfig
    started_at = time.perf_counter()
    session_logger, _ = common.setup_session_logger(sub_folder="train", file_level=logging.DEBUG)
    try:
        if args.rolling:
            from model.train_config import RollingConfig

            train_rolling(rolling_config=RollingConfig(mode=args.mode),
                          prep_output_dir=args.prep_output_dir,
                          save_dir=args.save_dir, logger=session_logger)
        else:
            train(config=config, prep_output_dir=args.prep_output_dir,
                  save_dir=args.save_dir or os.path.join(common.TRAIN_OUT_DIR, config.train_task),
                  logger=session_logger)
    finally:
        session_logger.info("Total run time: %.2f s", time.perf_counter() - started_at)
