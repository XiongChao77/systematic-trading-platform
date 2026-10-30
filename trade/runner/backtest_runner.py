from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import datetime as datetime_module
import contextlib
import fcntl
import hashlib
import os, sys, time, json, pickle, torch
import backtrader as bt
import backtrader.analyzers as btanalyzers
import pandas as pd
import logging
from dataclasses import asdict, replace
from typing import Any, Optional, Type

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", ".."))

# Import project modules
from data_process.common import *
from data_process import common
from data_process.utils import TaskIdentity, config_from_dict_train, param_hash
from model import model_loader
from model import data_loader
from model.training_types import temporal_split_bounds
from trade.runner import cus_analyzer
from trade.runner.analyze_backtest_report import (
    analyze_backtest_report,
    log_ftmo_challenge_summary,
    plot_equity_curves,
    simulate_ftmo_challenges,
)
from trade.runner.config import (
    BrokerConfig,
    BacktestEngineConfig,
    CsvDataConfig,
    ExperimentContext,
    ModelDataConfig,
    RunnerConfig,
)
from trade.runner.frontend_report_store import write_latest_backtest_report
from trade.venue.bt import cus_comminfo
from model import train_config
from trade.venue.bt.bt_venue_bbm import BbmBtVenue
from trade.feed.prediction_feed import PredictionFeed
from trade.strategy.strategy_bbm import BbmStrategyConfig

log_file = os.path.join(TEMPORARY_DIR, "trade_log_ftmo")


class TradeResult:
    def __init__(self) -> None:
        self.times = 0


def log_parameters(params_obj, logger):
    """
    Inspect an arbitrary params object and log all attributes in groups of 4.
    """
    # 1. Filter out Python internal __xx__ attributes and methods (callables).
    #    This works whether parameters are defined in __init__ or on the class.
    all_keys = [k for k in dir(params_obj) if not k.startswith("__") and not callable(getattr(params_obj, k))]

    # 2. Optionally sort by name for easier inspection
    # all_keys.sort()

    items_per_line = 4

    for i in range(0, len(all_keys), items_per_line):
        chunk_keys = all_keys[i : i + items_per_line]

        # 3. Build \"key: value\" strings using getattr for safety
        para_parts = []
        for k in chunk_keys:
            val = getattr(params_obj, k)
            para_parts.append(f"{k}: {val}")

        para_str = " | ".join(para_parts)

        # 4. Log with a prefix aligned with SUMMARY/EXPOSURE sections
        logger(f"Para    | {para_str}")


def log_metrics_block(logger, prefix: str, metrics: dict, items_per_line: int = 4):
    """
    Generic metric printing: dump any {key: scalar} mapping N items per line.
    Every strategy returns different fields, so nothing is assumed about them here.
    """
    if not metrics:
        return
    keys = list(metrics.keys())
    for i in range(0, len(keys), items_per_line):
        parts = []
        for k in keys[i : i + items_per_line]:
            v = metrics[k]
            parts.append(f"{k}: {v:.4g}" if isinstance(v, float) else f"{k}: {v}")
        logger.info(f"{prefix:<8}| " + " | ".join(parts))


def create_backtest_cerebro(
    *,
    venue_cls: Type[bt.Strategy],
    strategy_config: Any,
    broker_config: BrokerConfig,
    data: bt.feed.DataBase,
    predict_num: Optional[int] = None,
    data_interval_ms: int,
    engine_config: Optional[BacktestEngineConfig] = None,
) -> bt.Cerebro:
    """Create the shared Backtrader runtime and preserve the original analyzers."""
    engine_config = engine_config or BacktestEngineConfig()
    cerebro = bt.Cerebro(
        runonce=engine_config.runonce,
        cheat_on_open=engine_config.cheat_on_open,
        maxcpus=engine_config.max_cpus,
    )
    params = dict(
        strategy_config=strategy_config,
        initial_equity=broker_config.initial_equity,
        margin_warn_pct=broker_config.margin_warn_pct,
        data_interval_ms=data_interval_ms,
    )
    if predict_num is not None:
        params["predict_num"] = predict_num
    cerebro.addstrategy(venue_cls, **params)
    cerebro.adddata(data)
    cerebro.broker.set_coc(engine_config.cheat_on_close)
    cerebro.broker.setcash(broker_config.initial_equity)
    cerebro.broker.addcommissioninfo(cus_comminfo.CommInfo_Cryptocurrency(commission=broker_config.commission_pct, leverage=broker_config.leverage))

    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, compression=1, annualize=True, factor=365)
    cerebro.addanalyzer(btanalyzers.Returns, _name="returns", tann=365)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(cus_analyzer.CusAnalyzer, _name="customize")
    return cerebro


def _venue_for_strategy(strategy_config):
    venue_by_config = ((BbmStrategyConfig, BbmBtVenue),)
    for config_type, venue_cls in venue_by_config:
        if isinstance(strategy_config, config_type):
            return venue_cls
    raise TypeError(f"Unsupported strategy config: {type(strategy_config).__name__}")


_STRATEGY_CONFIG_TYPES = {config_type.__name__: config_type for config_type in (BbmStrategyConfig,)}


def strategy_config_to_dict(strategy_config) -> dict:
    """Serialize a strategy config with its Python class as JSON metadata."""
    config_type = type(strategy_config)
    if _STRATEGY_CONFIG_TYPES.get(config_type.__name__) is not config_type:
        raise TypeError(f"Unsupported strategy config: {config_type.__name__}")
    return {
        "config_type": config_type.__name__,
        **asdict(strategy_config),
    }


def strategy_config_from_dict(params: dict):
    params = dict(params)
    config_type_name = params.pop("config_type", None)

    # Backward compatibility for reports created before config_type metadata.
    legacy_type = params.pop("strategy_type", None)
    if config_type_name is None:
        config_type_name = {
            None: MlStrategyConfig.__name__,
            "ml": MlStrategyConfig.__name__,
            "bbm": BbmStrategyConfig.__name__,
        }.get(legacy_type, legacy_type)

    config_type = _STRATEGY_CONFIG_TYPES.get(config_type_name)
    if config_type is None:
        raise ValueError(f"Unsupported strategy config type: {config_type_name}")
    return config_type(**params)


def strategy_config_for_preparation(strategy_config, pre_para: Optional[common.PreparationConfig]):
    """Return the effective strategy config implied by preparation settings.

    A TBM label and ``BbmStrategyConfig`` share the same volatility barriers.
    Those four barrier multipliers are therefore preparation-derived values,
    not independent strategy sweep parameters.
    """
    if pre_para is None or pre_para.label_type != "TBM" or not isinstance(strategy_config, BbmStrategyConfig):
        return strategy_config

    if pre_para.stop_multiplier_rate_long is None or pre_para.stop_multiplier_rate_short is None:
        raise ValueError("TBM with BbmStrategyConfig requires both preparation stop " "multiplier rates")

    return replace(
        strategy_config,
        threshold_long=pre_para.vol_multiplier_long,
        threshold_short=pre_para.vol_multiplier_short,
        stop_loss_long=(pre_para.vol_multiplier_long * pre_para.stop_multiplier_rate_long),
        stop_loss_short=(pre_para.vol_multiplier_short * pre_para.stop_multiplier_rate_short),
    )


def _resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name) if isinstance(device_name, str) else device_name


_MODEL_TIME_REGION_CACHE: dict[tuple, dict] = {}
_PREDICTION_CACHE_SCHEMA = 5
_PREDICTION_COLUMNS = (
    "pred",
    "direction_pred",
    "pred_prob",
    "prob_short",
    "prob_neutral",
    "prob_long",
    "net_score",
)


def _prediction_artifact_files(train_output_dir: str) -> list[str]:
    """Resolve model artifacts, including model directories referenced by a fusion run."""
    files: set[str] = set()
    visited: set[str] = set()

    def visit(directory: str) -> None:
        directory = os.path.abspath(directory)
        if directory in visited:
            return
        visited.add(directory)
        description_path = os.path.join(directory, "task_description.json")
        if os.path.isfile(description_path):
            files.add(description_path)
            with open(description_path, "r", encoding="utf-8") as handle:
                description = json.load(handle)
            for value in (description.get("models") or {}).values():
                if isinstance(value, str):
                    visit(os.path.join(directory, value))
                elif isinstance(value, dict):
                    for key in ("model", "meta"):
                        relative_path = value.get(key)
                        if relative_path:
                            files.add(os.path.abspath(os.path.join(directory, relative_path)))
        for filename in ("model.pt", "meta.json", "train_config.json"):
            path = os.path.join(directory, filename)
            if os.path.isfile(path):
                files.add(path)

    visit(train_output_dir)
    return sorted(files)


def _validate_period(period: str) -> str:
    if period not in {"long", "forward", "all"}:
        raise ValueError("period must be one of: 'long', 'forward', 'all'")
    return period


def _prediction_cache_path(
    data_config: ModelDataConfig,
    train_output_dir: str,
    period: str,
) -> str:
    period = _validate_period(period)
    data_filenames = {
        "long": ["train_data.csv" if common.CONF_DF == "to_csv" else "train_data.feather"],
        "forward": ["test_data.csv" if common.CONF_DF == "to_csv" else "test_data.feather"],
        "all": [
            "train_data.csv" if common.CONF_DF == "to_csv" else "train_data.feather",
            "test_data.csv" if common.CONF_DF == "to_csv" else "test_data.feather",
        ],
    }[period]
    source_files = _prediction_artifact_files(train_output_dir)
    source_files.extend(os.path.join(data_config.prep_output_dir, filename) for filename in data_filenames)
    signatures = []
    for path in sorted(set(map(os.path.abspath, source_files))):
        stat = os.stat(path)
        signatures.append((path, stat.st_size, stat.st_mtime_ns))
    identity = {
        "schema": _PREDICTION_CACHE_SCHEMA,
        "period": period,
        "prep_output_dir": os.path.abspath(data_config.prep_output_dir),
        "train_output_dir": os.path.abspath(train_output_dir),
        "sources": signatures,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    cache_dir = data_config.prediction_cache_dir
    if cache_dir is None:
        cache_dir = os.path.join(train_output_dir, "prediction_cache")
    elif not os.path.isabs(cache_dir):
        cache_dir = os.path.join(PROJECT_DIR, cache_dir)
    return os.path.join(cache_dir, f"{period}_{digest}.pkl")


@contextlib.contextmanager
def _exclusive_file_lock(lock_path: str):
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_prediction_cache(cache_path: str):
    if not os.path.isfile(cache_path):
        return None
    with open(cache_path, "rb") as handle:
        payload = pickle.load(handle)
    if payload.get("schema") != _PREDICTION_CACHE_SCHEMA:
        return None
    return payload


def _write_prediction_cache(cache_path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    temp_path = f"{cache_path}.{os.getpid()}.tmp"
    try:
        with open(temp_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, cache_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def combine_model_period_frames(
    long_frame: pd.DataFrame,
    forward_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Join prepared long/forward data into one strict chronological frame."""
    time_column = "open_time_date_utc"
    if time_column not in long_frame.columns or time_column not in forward_frame.columns:
        raise ValueError(f"Both model periods must contain {time_column!r}")
    if long_frame.empty or forward_frame.empty:
        raise ValueError("Both long and forward periods must contain data")

    long_columns = list(long_frame.columns)
    if set(long_columns) != set(forward_frame.columns):
        missing = sorted(set(long_columns) - set(forward_frame.columns))
        extra = sorted(set(forward_frame.columns) - set(long_columns))
        raise ValueError("long/forward columns do not match " f"(missing in forward={missing}, extra in forward={extra})")

    long_part = long_frame.copy()
    forward_part = forward_frame[long_columns].copy()
    long_part[time_column] = pd.to_datetime(long_part[time_column], utc=True)
    forward_part[time_column] = pd.to_datetime(forward_part[time_column], utc=True)
    if not long_part[time_column].is_monotonic_increasing or not long_part[time_column].is_unique:
        raise ValueError("long period timestamps are not strictly increasing")
    if not forward_part[time_column].is_monotonic_increasing or not forward_part[time_column].is_unique:
        raise ValueError("forward period timestamps are not strictly increasing")
    if long_part[time_column].iloc[-1] >= forward_part[time_column].iloc[0]:
        raise ValueError("long and forward periods overlap or duplicate timestamps")

    return pd.concat([long_part, forward_part], ignore_index=True)


def _load_model_period_frame(
    data_config: ModelDataConfig,
    period: str,
) -> pd.DataFrame:
    period = _validate_period(period)
    if period == "long":
        return common.load_train_df_from_dir(data_config.prep_output_dir)
    if period == "forward":
        return common.load_test_df_from_dir(data_config.prep_output_dir)
    if period == "all":
        return combine_model_period_frames(
            common.load_train_df_from_dir(data_config.prep_output_dir),
            common.load_test_df_from_dir(data_config.prep_output_dir),
        )
    raise AssertionError("validated period was not handled")


def _model_time_regions(
    data_config: ModelDataConfig,
    train_cfg: train_config.TrainConfig,
    handler: model_loader.ModelHandler,
    current_frame: pd.DataFrame,
    period: str,
) -> dict[str, dict[str, Any]]:
    """Calculate the exact train/valid/test/OOS timestamp regions."""
    data_cfg = train_cfg.data_cfg
    cache_key = (
        os.path.abspath(data_config.prep_output_dir),
        tuple(handler.feature_cols),
        handler.label_col,
        handler.seq_len,
        train_cfg.stride,
        data_cfg.train_ratio,
        data_cfg.val_ratio,
        data_cfg.purge_overlap,
    )
    cached = _MODEL_TIME_REGION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    train_frame = current_frame if period == "long" else common.load_train_df_from_dir(data_config.prep_output_dir)
    oos_frame = current_frame if period == "forward" else common.load_test_df_from_dir(data_config.prep_output_dir)
    valid_indices = data_loader.valid_window_end_indices(
        train_frame,
        feature_cols=handler.feature_cols,
        label_col=handler.label_col,
        seq_len=handler.seq_len,
        stride=train_cfg.stride,
    )
    bounds = temporal_split_bounds(
        len(valid_indices),
        train_ratio=data_cfg.train_ratio,
        val_ratio=data_cfg.val_ratio,
        purge_overlap=data_cfg.purge_overlap,
        seq_len=handler.seq_len,
        stride=train_cfg.stride,
    )

    def as_iso(value) -> str:
        # Prepared feather frames can hold timezone-naive timestamps, so the
        # region boundaries must be normalized before they are compared with
        # the UTC-aware backtest frame.
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp.isoformat()

    def sample_time(sample_position: int) -> str:
        row_index = valid_indices[sample_position]
        return as_iso(train_frame.loc[row_index, "open_time_date_utc"])

    regions = {
        name: {
            "start": sample_time(start),
            "end": sample_time(end - 1),
            "sample_count": end - start,
        }
        for name, (start, end) in bounds.items()
    }
    oos_times = pd.to_datetime(oos_frame["open_time_date_utc"], utc=True)
    if oos_times.empty:
        raise ValueError("OOS dataset is empty; cannot calculate plot boundaries")
    regions["oos"] = {
        "start": as_iso(oos_times.iloc[0]),
        "end": as_iso(oos_times.iloc[-1]),
    }
    _MODEL_TIME_REGION_CACHE[cache_key] = regions
    return regions


def _infer_prediction_payload(
    logger,
    data_config: ModelDataConfig,
    train_cfg: train_config.TrainConfig,
    frame: pd.DataFrame,
    train_output_dir: str,
    interval_ms: int,
    period: str,
    inference_batch_size: int = 512,
) -> dict:
    handler = model_loader.ModelHandler(
        tarin_out_path=train_output_dir,
        device=_resolve_device(data_config.device),
    )
    time_regions = _model_time_regions(
        data_config,
        train_cfg,
        handler,
        frame,
        period,
    )
    dataset = data_loader.TimeSeriesWindowDataset(
        df=frame,
        kline_interval_ms=interval_ms,
        feature_cols=handler.feature_cols,
        label_col=handler.label_col,
        seq_len=handler.seq_len,
        is_live=False,
        materialize_windows=False,
    )
    predicted_frame, model_stats, sub_model_stats, sub_model_predictions = (
        handler.predict_with_ds(
            dataset,
            frame,
            is_live=False,
            batch_size=inference_batch_size,
            diff_thresh=None,
            evaluate_sub_models=True,
        )
    )
    first_valid_idx = predicted_frame["pred"].first_valid_index()
    if first_valid_idx is None:
        raise RuntimeError("No valid predictions found in the entire dataset")

    predictions = predicted_frame.loc[:, list(_PREDICTION_COLUMNS)].copy()
    if len(sub_model_predictions.columns):
        predictions = predictions.join(sub_model_predictions)
    stage_model_stats = handler.stage_performance(
        predicted_frame,
        time_regions,
        sub_model_predictions=sub_model_predictions,
    )
    return {
        "schema": _PREDICTION_CACHE_SCHEMA,
        "predictions": predictions,
        "first_valid_idx": first_valid_idx,
        "model_stats": model_stats,
        "sub_model_stats": sub_model_stats,
        "stage_model_stats": stage_model_stats,
        "time_regions": time_regions,
    }


def _prediction_inputs(data_config: ModelDataConfig, period: str):
    pre_para: BaseDefine = common.load_pre_params_from_dir(data_config.prep_output_dir)
    frame = _load_model_period_frame(data_config, period).copy()
    frame["open_time_date_utc"] = pd.to_datetime(frame["open_time_date_utc"], utc=True)
    train_output_dir = data_config.train_output_dir
    if not os.path.isabs(train_output_dir):
        train_output_dir = os.path.join(PROJECT_DIR, train_output_dir)
    return pre_para, frame, train_output_dir, common.get_interval_ms(pre_para.interval)


def precompute_prediction_cache(
    logger,
    data_config: ModelDataConfig,
    train_cfg: train_config.TrainConfig,
    period: str,
    *,
    inference_batch_size: int,
) -> str:
    """Generate one prediction cache before CPU backtest workers are started."""
    period = _validate_period(period)
    pre_para, frame, train_output_dir, interval_ms = _prediction_inputs(
        data_config,
        period,
    )
    del pre_para
    cache_path = _prediction_cache_path(data_config, train_output_dir, period)
    with _exclusive_file_lock(cache_path + ".lock"):
        payload = _read_prediction_cache(cache_path)
        if payload is None:
            logger.info("Generating prediction cache: %s", cache_path)
            payload = _infer_prediction_payload(
                logger,
                data_config,
                train_cfg,
                frame,
                train_output_dir,
                interval_ms,
                period,
                inference_batch_size,
            )
            _write_prediction_cache(cache_path, payload)
            logger.info("Prediction cache saved: %s", cache_path)
        else:
            logger.info("Prediction cache already exists: %s", cache_path)
    return cache_path


def _load_model_data(
    logger,
    data_config: ModelDataConfig,
    train_cfg: train_config.TrainConfig,
    period: str,
):
    pre_para, frame, train_output_dir, interval_ms = _prediction_inputs(
        data_config,
        period,
    )
    logger.info(f"prep_output_dir:{data_config.prep_output_dir}, " f"train_output_dir:{data_config.train_output_dir}")
    logger.info("Loaded model data | period=%s | bars=%d", period, len(frame))

    if data_config.use_prediction_cache:
        cache_path = _prediction_cache_path(data_config, train_output_dir, period)
        with _exclusive_file_lock(cache_path + ".lock"):
            try:
                payload = _read_prediction_cache(cache_path)
            except Exception as exc:
                raise RuntimeError(f"Prediction cache is unreadable: {cache_path}") from exc
            if payload is None:
                raise FileNotFoundError("Prediction cache is missing or incompatible; call " f"precompute_prediction_cache() before backtesting: {cache_path}")
            logger.info("Prediction cache hit; skipping model inference: %s", cache_path)
    else:
        payload = _infer_prediction_payload(
            logger,
            data_config,
            train_cfg,
            frame,
            train_output_dir,
            interval_ms,
            period,
        )

    predictions = payload["predictions"]
    if not predictions.index.equals(frame.index):
        raise RuntimeError("Prediction cache index does not match the prepared market data; " "refusing to use misaligned signals")
    missing_columns = [column for column in _PREDICTION_COLUMNS if column not in predictions.columns]
    if missing_columns:
        raise RuntimeError(f"Prediction cache is missing required columns: {missing_columns}")
    # Sub-model columns (composite models only) are carried alongside the fused
    # predictions so every stage can be scored component by component.
    for column in predictions.columns:
        frame[column] = predictions[column]
    frame = frame.loc[payload["first_valid_idx"] :].copy()
    model_stats = payload["model_stats"]
    sub_model_stats = payload["sub_model_stats"]
    stage_model_stats = payload.get("stage_model_stats") or {}
    time_regions = payload["time_regions"]
    logger.info(
        "Model time regions | train=%s | valid=%s | test=%s | OOS=%s",
        time_regions["train"]["start"],
        time_regions["valid"]["start"],
        time_regions["test"]["start"],
        time_regions["oos"]["start"],
    )
    logger.info(f"Backtest range: {frame['open_time_date_utc'].min()} " f"to {frame['open_time_date_utc'].max()}")
    return (
        frame,
        model_stats,
        sub_model_stats,
        pre_para,
        time_regions,
        stage_model_stats,
    )


def load_backtest_prediction_frame(
    logger,
    data_config: ModelDataConfig,
    period: str,
) -> pd.DataFrame:
    """Return timestamped predictions produced by the backtest inference path."""

    train_cfg = _load_train_config(data_config.train_output_dir)
    frame, _, _, _, _, _ = _load_model_data(
        logger,
        data_config,
        train_cfg,
        period,
    )
    result = frame.loc[:, ["close_time_ms_utc", *_PREDICTION_COLUMNS]].copy()
    result.insert(
        1,
        "close_time_date_utc",
        pd.to_datetime(result["close_time_ms_utc"], unit="ms", utc=True),
    )
    return result.reset_index(drop=True)


def _load_csv_data(logger, data_config: CsvDataConfig):
    data_path = common.market_data_path(data_config, root_dir=PROJECT_DATA_DIR)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    frame = pd.read_csv(data_path, encoding="utf-8")
    if "open_time_date_utc" not in frame.columns:
        raise ValueError("'open_time_date_utc' column missing from market data")
    frame["open_time_date_utc"] = pd.to_datetime(frame["open_time_date_utc"], utc=True)
    logger.info(f"Loaded {len(frame)} bars from {data_path}")
    return frame, {}, {}, None, None, {}


def _build_feed(frame: pd.DataFrame, data_config):
    from_date = data_config.from_date if isinstance(data_config, CsvDataConfig) else None
    to_date = data_config.to_date if isinstance(data_config, CsvDataConfig) else None
    feed_params = dict(
        dataname=frame,
        datetime="open_time_date_utc",
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        openinterest=-1,
        nocase=True,
        fromdate=pd.Timestamp(from_date).to_pydatetime() if from_date else None,
        todate=pd.Timestamp(to_date).to_pydatetime() if to_date else None,
    )
    feed_params["expected_vol"] = "expected_vol" if "expected_vol" in frame.columns else None
    return PredictionFeed(**feed_params)


def slice_stage_frame(frame: pd.DataFrame, region: dict) -> pd.DataFrame:
    """Return the bars whose open time falls inside one model time region."""
    times = pd.to_datetime(frame["open_time_date_utc"], utc=True)
    start = pd.Timestamp(region["start"])
    end = pd.Timestamp(region["end"])
    return frame.loc[(times >= start) & (times <= end)].copy()


_STAGE_PREDICTION_COLUMNS = (
    "open_time_date_utc",
    "close",
    "label",
    "pred",
    "pred_prob",
    "prob_short",
    "prob_neutral",
    "prob_long",
    "net_score",
    "direction_pred",
)


def write_stage_predictions(save_dir: Optional[str], stage: str, stage_frame: pd.DataFrame) -> Optional[str]:
    """Persist the per-bar model output used to score one backtest stage.

    Composite models also expose their sub-model class probabilities here, which
    is what makes the fused signal explainable per stage.
    """
    if not save_dir:
        return None

    columns = [column for column in _STAGE_PREDICTION_COLUMNS if column in stage_frame.columns]
    sub_model_names = sorted(
        column[len("sub_") : -len("_pred")]
        for column in stage_frame.columns
        if column.startswith("sub_") and column.endswith("_pred")
    )
    for name in sub_model_names:
        columns.extend(
            column
            for column in (
                f"sub_{name}_pred",
                f"sub_{name}_prob_0",
                f"sub_{name}_prob_1",
            )
            if column in stage_frame.columns
        )

    output_dir = os.path.join(save_dir, "stage_predictions")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{stage}_predictions.csv")
    stage_frame.loc[:, columns].to_csv(output_path, index=False)
    return output_path


_STAGE_SUMMARY_FILENAME = "stage_summary.json"
_STAGE_SUMMARY_MARKDOWN_FILENAME = "stage_summary.md"
# Stage name -> time-region key used by ModelHandler.STAGE_REGIONS.
_STAGE_REGION_KEYS = dict(model_loader.ModelHandler.STAGE_REGIONS)


def _number(value, default: float = 0.0) -> float:
    """Return a float for logging/metrics even when the source value is missing."""
    return default if value is None else float(value)


def _baseline_from_stats(stats: dict) -> tuple[Optional[float], Optional[int]]:
    """Return the majority-class share of the true labels and that class.

    Always predicting this class is the naive accuracy baseline: a model that
    scores below it is worse than a constant predictor on the same rows.
    """
    proportions = stats.get("label_proportions_true") or {}
    if not proportions:
        return None, None
    class_key = max(proportions, key=proportions.get)
    return float(proportions[class_key]), int(class_key)


def _compact_classification(report: dict) -> dict:
    """Keep per-class and aggregate precision/recall/F1 rows only."""
    compact = {}
    for key, value in (report or {}).items():
        if not isinstance(value, dict):
            continue
        compact[key] = {
            name: value.get(name)
            for name in ("precision", "recall", "f1-score", "support")
            if name in value
        }
    return compact


def _model_stage_summary(stage_stats: dict) -> dict:
    """Summarize the fused model's prediction quality for one stage."""
    fused = stage_stats.get("fused") or {}
    signal = fused.get("signal") or {}
    baseline, baseline_class = _baseline_from_stats(fused)
    accuracy = fused.get("accuracy")
    return {
        "bar_count": stage_stats.get("bar_count"),
        "predicted_count": stage_stats.get("predicted_count"),
        "accuracy": accuracy,
        "baseline_accuracy": baseline,
        "baseline_class": baseline_class,
        "accuracy_minus_baseline": (
            float(accuracy) - baseline
            if accuracy is not None and baseline is not None
            else None
        ),
        "macro_f1": fused.get("f1_macro"),
        "weighted_f1": fused.get("f1_weighted"),
        "weighted_precision": fused.get("precision_weighted"),
        "weighted_recall": fused.get("recall_weighted"),
        "coverage": signal.get("coverage"),
        "signal_count": signal.get("signal_count"),
        "directional_accuracy": signal.get("directional_accuracy"),
        "long_precision": signal.get("long_win_rate"),
        "short_precision": signal.get("short_win_rate"),
        "long_signal_count": signal.get("long_count"),
        "short_signal_count": signal.get("short_count"),
        "precision_lift": fused.get("precision_lift"),
        "label_proportions_true": fused.get("label_proportions_true"),
        "prediction_distribution": fused.get("label_distribution_pred"),
        "confusion_matrix": fused.get("confusion_matrix"),
        "classification_report": _compact_classification(fused.get("classification_report")),
    }


def _sub_model_stage_summary(stats: dict) -> dict:
    """Summarize one sub-model (trigger/direction or long/short OvR) for a stage."""
    report = stats.get("classification_report") or {}
    baseline, baseline_class = _baseline_from_stats(stats)
    return {
        "task_type": stats.get("task_type"),
        "sample_count": stats.get("sample_count"),
        "accuracy": stats.get("accuracy"),
        "baseline_accuracy": baseline,
        "baseline_class": baseline_class,
        "macro_f1": stats.get("f1_macro"),
        "positive_precision": (report.get("1") or {}).get("precision"),
        "positive_recall": (report.get("1") or {}).get("recall"),
        "negative_precision": (report.get("0") or {}).get("precision"),
        "negative_recall": (report.get("0") or {}).get("recall"),
        "confusion_matrix": stats.get("confusion_matrix"),
        "label_distribution_true": stats.get("label_distribution_true"),
        "label_distribution_pred": stats.get("label_distribution_pred"),
    }


_STAGE_STRATEGY_KEYS = ("time", "performance", "drawdown", "exposure", "trades", "strategy")


def _strategy_stage_summary(stage_result: dict) -> dict:
    """Keep the standalone trading performance blocks of one stage."""
    return {key: stage_result.get(key) for key in _STAGE_STRATEGY_KEYS}


def _optional_product(*values):
    """Multiply values, returning None when any factor is missing."""
    if any(value is None for value in values):
        return None
    result = 1.0
    for value in values:
        result *= float(value)
    # Avoid binary floating-point noise (e.g. 12 * 0.6 -> 7.199999999999999).
    return round(result, 12)


def _labeling_summary(pre_para) -> dict:
    """Capture the barrier geometry used to build the labels.

    TBM labels take profit at ``vol_multiplier`` expected volatility; the stop
    barrier only invalidates the same-side take-profit candidate, so
    ``stop_multiplier_rate`` is not a negative-class definition.
    """
    if pre_para is None:
        return {}
    vol_long = getattr(pre_para, "vol_multiplier_long", None)
    vol_short = getattr(pre_para, "vol_multiplier_short", None)
    stop_rate_long = getattr(pre_para, "stop_multiplier_rate_long", None)
    stop_rate_short = getattr(pre_para, "stop_multiplier_rate_short", None)
    return {
        "label_type": getattr(pre_para, "label_type", None),
        "take_profit_vol_multiple_long": vol_long,
        "take_profit_vol_multiple_short": vol_short,
        "stop_multiplier_rate_long": stop_rate_long,
        "stop_multiplier_rate_short": stop_rate_short,
        "stop_loss_vol_multiple_long": _optional_product(vol_long, stop_rate_long),
        "stop_loss_vol_multiple_short": _optional_product(vol_short, stop_rate_short),
        "label_window_bars": getattr(pre_para, "predict_num", None),
        "min_expected_move_pct": getattr(pre_para, "min_expected_move_pct", None),
        "tbm_intrabar_interval": getattr(pre_para, "tbm_intrabar_interval", None),
    }


_EXECUTION_FIELDS = (
    "compound",
    "risk_per_trade_pct",
    "fixed_hold_bars",
    "threshold_long",
    "threshold_short",
    "stop_loss_long",
    "stop_loss_short",
    "allow_long",
    "allow_short",
    "prob_thresh",
    "min_expected_move_pct",
    "max_daily_loss_pct",
)


def _execution_summary(strategy_config) -> dict:
    """Capture the effective strategy parameters used by the backtests."""
    if strategy_config is None:
        return {}
    summary = {"config_type": type(strategy_config).__name__}
    for name in _EXECUTION_FIELDS:
        if not hasattr(strategy_config, name):
            continue
        value = getattr(strategy_config, name)
        summary[name] = round(value, 12) if isinstance(value, float) else value
    return summary


def _broker_summary(broker_config) -> dict:
    """Capture broker-level assumptions behind the reported PnL."""
    if broker_config is None:
        return {}
    return {
        name: getattr(broker_config, name)
        for name in ("initial_equity", "commission_pct", "leverage", "margin_warn_pct")
        if hasattr(broker_config, name)
    }


def _artifact_task_type(train_output_dir: Optional[str]) -> Optional[str]:
    """Read the trained task type from the artifact's task_description.json."""
    if not train_output_dir:
        return None
    artifact_dir = (
        train_output_dir
        if os.path.isabs(train_output_dir)
        else os.path.join(PROJECT_DIR, train_output_dir)
    )
    task_desc_path = os.path.join(artifact_dir, model_loader.TASK_DESCRIPTION_FILENAME)
    if not os.path.isfile(task_desc_path):
        return None
    with open(task_desc_path, "r", encoding="utf-8") as task_file:
        return json.load(task_file).get("task_type")


def build_stage_summary(
    *,
    stage_results: dict,
    stage_model_stats: dict,
    stage_prediction_files: dict,
    pre_para,
    train_cfg,
    model_task_type: Optional[str],
    period: str,
    analysis_mode: str,
    overall_result: dict,
    strategy_config=None,
    broker_config=None,
) -> dict:
    """Assemble the model/baseline/sub-model/trading summary of every stage."""
    stages = {}
    for stage, stage_result in stage_results.items():
        stats = stage_model_stats.get(stage) or {}
        stages[stage] = {
            "model": _model_stage_summary(stats),
            "sub_models": {
                name: _sub_model_stage_summary(sub_stats)
                for name, sub_stats in (stats.get("sub_models") or {}).items()
            },
            "strategy": _strategy_stage_summary(stage_result),
            "predictions_file": stage_prediction_files.get(stage),
        }

    return {
        "schema_version": 2,
        "generated_at": datetime_module.datetime.now(datetime_module.timezone.utc).isoformat(),
        "period": period,
        "analysis_mode": analysis_mode,
        "baseline": {
            "definition": (
                "accuracy obtained by always predicting the most frequent label "
                "class of the stage"
            ),
            "source": "label_proportions_true",
            "usage": (
                "model.accuracy_minus_baseline above zero means the model beats a "
                "constant predictor; compare long/short precision with "
                "label_proportions_true to judge directional skill"
            ),
        },
        "model": {
            "task_type": model_task_type
            or (train_cfg.train_task if train_cfg is not None else None),
            "model_type": getattr(
                getattr(train_cfg, "model_cfg", None), "model_type", None
            ),
            "model_version": getattr(
                getattr(train_cfg, "model_cfg", None), "model_version", None
            ),
            "seq_len": getattr(getattr(train_cfg, "model_cfg", None), "seq_len", None),
            "stride": getattr(train_cfg, "stride", None),
            "symbol": getattr(pre_para, "symbol", None),
            "interval": getattr(pre_para, "interval", None),
            "horizon_bars": getattr(pre_para, "predict_num", None),
            "sub_models": sorted(
                {
                    name
                    for stats in stage_model_stats.values()
                    for name in (stats.get("sub_models") or {})
                }
            ),
            "labeling": _labeling_summary(pre_para),
            "execution": _execution_summary(strategy_config),
            "broker": _broker_summary(broker_config),
        },
        "stages": stages,
        "overall": {
            **_strategy_stage_summary(overall_result),
            "ftmo_challenge": overall_result.get("ftmo_challenge"),
        },
    }


def write_stage_summary(save_dir: Optional[str], summary: dict) -> Optional[str]:
    """Persist the stage summary next to the experiment report."""
    if not save_dir:
        return None
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, _STAGE_SUMMARY_FILENAME)
    with open(output_path, "w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2, default=str)
    return output_path


# -----------------------------------------------------------------------------
# Human-readable rendering of the stage summary
# -----------------------------------------------------------------------------


def _md_text(value) -> str:
    return "n/a" if value is None else str(value)


def _md_float(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _md_pct(value, digits: int = 2) -> str:
    """Format a fraction (0.25 -> 25.00%)."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def _md_raw_pct(value, digits: int = 2) -> str:
    """Format a value that is already expressed in percent units."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def _md_money(value, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _md_int(value) -> str:
    if value is None:
        return "n/a"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _md_bool(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _md_lookup(mapping, key, default=None):
    """Read a mapping whose keys may be ints in memory and strings after JSON."""
    if not mapping:
        return default
    if key in mapping:
        return mapping[key]
    string_key = str(key)
    if string_key in mapping:
        return mapping[string_key]
    try:
        int_key = int(key)
    except (TypeError, ValueError):
        return default
    return mapping.get(int_key, default)


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _md_key_value_table(rows: list[tuple[str, str]]) -> str:
    return _md_table(["item", "value"], [[name, value] for name, value in rows])


def _md_confusion(confusion: dict) -> str:
    """Render a confusion matrix dict as a markdown table."""
    if not confusion:
        return "_n/a_"
    labels = confusion.get("labels") or []
    matrix = confusion.get("matrix") or []
    if not labels or not matrix:
        return "_n/a_"
    rows = [
        [str(true_label)] + [str(cell) for cell in row]
        for true_label, row in zip(labels, matrix)
    ]
    return _md_table(["true \\ pred"] + [str(label) for label in labels], rows)


def _md_class_label(class_key) -> str:
    return {
        "-1": "short (-1)",
        "0": "neutral (0)",
        "1": "long (+1)",
    }.get(str(class_key), str(class_key))


def _md_stage_time_range(stage: dict) -> tuple[str, str]:
    time_block = (stage.get("strategy") or {}).get("time") or {}
    return _md_text(time_block.get("start")), _md_text(time_block.get("end"))


def _md_stage_region(stage_name: str, overall: dict) -> dict:
    regions = (overall.get("time") or {}).get("regions") or {}
    region_key = _STAGE_REGION_KEYS.get(stage_name, stage_name)
    return regions.get(region_key) or {}


def _md_analysis_mode_usage(analysis_mode) -> str:
    return {
        "none": "model predictions are used as-is",
        "trigger_perfect": (
            "labels decide action/no-action and the direction sub-model decides "
            "the side; rows without a model prediction stay unsignaled"
        ),
        "direction_perfect": (
            "the model trigger decides action/no-action and the true label "
            "decides the side"
        ),
    }.get(str(analysis_mode), "unknown analysis mode")


def build_stage_summary_markdown(summary: dict) -> str:
    """Render the stage summary as a human-readable markdown document."""
    stages = summary.get("stages") or {}
    model_info = summary.get("model") or {}
    labeling = model_info.get("labeling") or {}
    execution = model_info.get("execution") or {}
    broker = model_info.get("broker") or {}
    overall = summary.get("overall") or {}
    sub_model_names = list(model_info.get("sub_models") or [])
    for stage in stages.values():
        for name in (stage.get("sub_models") or {}):
            if name not in sub_model_names:
                sub_model_names.append(name)

    lines: list[str] = []
    lines.append("# Backtest Stage Summary")
    lines.append("")
    lines.append(
        "Human-readable companion of `stage_summary.json` in the same directory. "
        "Machine-readable values live in that JSON file; per-bar predictions live "
        "in `stage_predictions/<stage>_predictions.csv`."
    )
    lines.append("")

    # ---------------------------------------------------------------- overview
    lines.append("## Run overview")
    lines.append("")
    lines.append(
        _md_key_value_table(
            [
                ("generated_at (UTC)", _md_text(summary.get("generated_at"))),
                ("period", _md_text(summary.get("period"))),
                ("analysis_mode", _md_text(summary.get("analysis_mode"))),
                ("analysis_mode usage", _md_analysis_mode_usage(summary.get("analysis_mode"))),
                ("task_type", _md_text(model_info.get("task_type"))),
                (
                    "model",
                    "{} (version {})".format(
                        _md_text(model_info.get("model_type")),
                        _md_text(model_info.get("model_version")),
                    ),
                ),
                (
                    "symbol / interval",
                    "{} / {}".format(
                        _md_text(model_info.get("symbol")),
                        _md_text(model_info.get("interval")),
                    ),
                ),
                ("sequence length (bars)", _md_int(model_info.get("seq_len"))),
                ("stride", _md_int(model_info.get("stride"))),
                ("label window (bars)", _md_int(model_info.get("horizon_bars"))),
                ("stages", _md_text(", ".join(stages) if stages else "none")),
                (
                    "sub-models",
                    ", ".join(sub_model_names) if sub_model_names else "none",
                ),
            ]
        )
    )
    lines.append("")

    if labeling:
        lines.append("### Label geometry")
        lines.append("")
        lines.append(
            "Barriers are multiples of `expected_vol`. In TBM labeling the take-profit "
            "barriers decide short/long; the stop barrier only invalidates the same-side "
            "take-profit candidate, so a stop alone yields the neutral class, not the "
            "opposite class."
        )
        lines.append("")
        lines.append(
            _md_key_value_table(
                [
                    ("label type", _md_text(labeling.get("label_type"))),
                    (
                        "take-profit multiple, long",
                        _md_float(labeling.get("take_profit_vol_multiple_long"), 2)
                        + " x expected_vol",
                    ),
                    (
                        "take-profit multiple, short",
                        _md_float(labeling.get("take_profit_vol_multiple_short"), 2)
                        + " x expected_vol",
                    ),
                    (
                        "stop multiplier rate, long",
                        _md_float(labeling.get("stop_multiplier_rate_long"), 2),
                    ),
                    (
                        "stop multiplier rate, short",
                        _md_float(labeling.get("stop_multiplier_rate_short"), 2),
                    ),
                    (
                        "effective stop multiple, long",
                        _md_float(labeling.get("stop_loss_vol_multiple_long"), 2)
                        + " x expected_vol",
                    ),
                    (
                        "effective stop multiple, short",
                        _md_float(labeling.get("stop_loss_vol_multiple_short"), 2)
                        + " x expected_vol",
                    ),
                    ("label window", _md_int(labeling.get("label_window_bars")) + " bars"),
                    (
                        "min expected move",
                        _md_pct(labeling.get("min_expected_move_pct")) + " of price",
                    ),
                    (
                        "intrabar interval",
                        _md_text(labeling.get("tbm_intrabar_interval")),
                    ),
                ]
            )
        )
        lines.append("")

    if execution:
        lines.append("### Execution settings")
        lines.append("")
        lines.append(
            _md_key_value_table(
                [
                    ("strategy config", _md_text(execution.get("config_type"))),
                    (
                        "take-profit trigger, long / short",
                        "{} / {} x expected_vol".format(
                            _md_float(execution.get("threshold_long"), 2),
                            _md_float(execution.get("threshold_short"), 2),
                        ),
                    ),
                    (
                        "stop-loss, long / short",
                        "{} / {} x expected_vol".format(
                            _md_float(execution.get("stop_loss_long"), 2),
                            _md_float(execution.get("stop_loss_short"), 2),
                        ),
                    ),
                    ("risk per trade", _md_pct(execution.get("risk_per_trade_pct"))),
                    ("max daily loss", _md_pct(execution.get("max_daily_loss_pct"))),
                    ("fixed hold bars", _md_int(execution.get("fixed_hold_bars"))),
                    ("min expected move", _md_pct(execution.get("min_expected_move_pct"))),
                    ("probability threshold", _md_float(execution.get("prob_thresh"), 4)),
                    ("allow long", _md_bool(execution.get("allow_long"))),
                    ("allow short", _md_bool(execution.get("allow_short"))),
                    ("compounding", _md_bool(execution.get("compound"))),
                ]
            )
        )
        lines.append("")

    if broker:
        lines.append("### Broker settings")
        lines.append("")
        lines.append(
            _md_key_value_table(
                [
                    ("initial equity", _md_money(broker.get("initial_equity"))),
                    ("commission", _md_raw_pct(broker.get("commission_pct")) + " per side"),
                    ("leverage", _md_float(broker.get("leverage"), 1) + "x"),
                    ("margin warning", _md_pct(broker.get("margin_warn_pct"))),
                ]
            )
        )
        lines.append("")

    # ---------------------------------------------------------------- stages
    lines.append("## 1. Stage windows")
    lines.append("")
    rows = []
    for stage_name, stage in stages.items():
        start, end = _md_stage_time_range(stage)
        model = stage.get("model") or {}
        region = _md_stage_region(stage_name, overall)
        rows.append(
            [
                stage_name,
                _md_int(region.get("sample_count")),
                start,
                end,
                _md_int(model.get("bar_count")),
                _md_int(model.get("predicted_count")),
                os.path.basename(str(stage.get("predictions_file") or "n/a")),
            ]
        )
    lines.append(
        _md_table(
            [
                "stage",
                "split samples",
                "first bar",
                "last bar",
                "backtest bars",
                "predicted bars",
                "per-bar csv",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "`backtest bars` is the number of bars actually replayed for that stage "
        "(after dropping rows without usable model input). `split samples` comes "
        "from the training split manifest and is available for train/valid/test "
        "only; for the forward stage use `predicted bars` as the reference. Each "
        "stage runs its own broker, so the trading numbers below are standalone "
        "stage runs."
    )
    lines.append("")

    # ------------------------------------------------------- fused model
    lines.append("## 2. Fused model prediction quality")
    lines.append("")
    lines.append("### Signal quality")
    lines.append("")
    rows = []
    for stage_name, stage in stages.items():
        model = stage.get("model") or {}
        rows.append(
            [
                stage_name,
                _md_int(model.get("bar_count")),
                _md_int(model.get("predicted_count")),
                _md_pct(model.get("coverage")),
                _md_int(model.get("signal_count")),
                _md_int(model.get("long_signal_count")),
                _md_pct(model.get("long_precision")),
                _md_int(model.get("short_signal_count")),
                _md_pct(model.get("short_precision")),
                _md_pct(model.get("directional_accuracy")),
            ]
        )
    lines.append(
        _md_table(
            [
                "stage",
                "bars",
                "predictions",
                "coverage",
                "signals",
                "long signals",
                "long precision",
                "short signals",
                "short precision",
                "directional accuracy",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "`coverage` is the share of bars with a non-neutral prediction. "
        "`long precision` / `short precision` are the share of that side's signals "
        "whose true label matches the signal. `directional accuracy` is accuracy "
        "restricted to signaled bars."
    )
    lines.append("")

    lines.append("### Classification quality")
    lines.append("")
    rows = []
    for stage_name, stage in stages.items():
        model = stage.get("model") or {}
        rows.append(
            [
                stage_name,
                _md_float(model.get("accuracy")),
                _md_float(model.get("baseline_accuracy")),
                _md_float(model.get("accuracy_minus_baseline"), 4),
                _md_float(model.get("macro_f1")),
                _md_float(model.get("weighted_f1")),
                _md_float(model.get("weighted_precision")),
                _md_float(model.get("weighted_recall")),
                _md_text(model.get("baseline_class")),
            ]
        )
    lines.append(
        _md_table(
            [
                "stage",
                "accuracy",
                "baseline (majority class)",
                "accuracy - baseline",
                "macro F1",
                "weighted F1",
                "weighted precision",
                "weighted recall",
                "baseline class",
            ],
            rows,
        )
    )
    lines.append("")

    lines.append("### Per-class metrics")
    lines.append("")
    rows = []
    for stage_name, stage in stages.items():
        model = stage.get("model") or {}
        report = model.get("classification_report") or {}
        lifts = model.get("precision_lift") or {}
        for class_key in sorted(report, key=lambda key: str(key)):
            metrics = report[class_key]
            lift = _md_lookup(lifts, class_key, {}) or {}
            rows.append(
                [
                    stage_name,
                    _md_class_label(class_key),
                    _md_float(metrics.get("precision")),
                    _md_float(metrics.get("recall")),
                    _md_float(metrics.get("f1-score")),
                    _md_int(metrics.get("support")),
                    _md_pct(lift.get("base_rate")),
                    _md_float(lift.get("lift"), 3),
                    _md_raw_pct(lift.get("lift_pct"), 1),
                ]
            )
    lines.append(
        _md_table(
            [
                "stage",
                "class",
                "precision",
                "recall",
                "F1",
                "support",
                "base rate",
                "precision lift",
                "lift vs base rate",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "`precision lift` = precision / base rate. A lift above 1 means the class "
        "is predicted more often than its natural occurrence and pays off more often "
        "than a random pick on the same rows."
    )
    lines.append("")

    # --------------------------------------------------------- baseline
    lines.append("## 3. Baseline and prediction distribution")
    lines.append("")
    lines.append(
        "Baseline = accuracy of always predicting the most frequent true label class "
        "of that stage (source: `label_proportions_true`). `accuracy - baseline` "
        "above zero means the model beats a constant predictor."
    )
    lines.append("")
    rows = []
    for stage_name, stage in stages.items():
        model = stage.get("model") or {}
        proportions = model.get("label_proportions_true") or {}
        rows.append(
            [
                stage_name,
                _md_pct(_md_lookup(proportions, -1)),
                _md_pct(_md_lookup(proportions, 0)),
                _md_pct(_md_lookup(proportions, 1)),
                _md_text(model.get("baseline_class")),
                _md_pct(model.get("baseline_accuracy")),
                _md_float(model.get("accuracy")),
                _md_float(model.get("accuracy_minus_baseline")),
            ]
        )
    lines.append(
        _md_table(
            [
                "stage",
                "true short (-1)",
                "true neutral (0)",
                "true long (+1)",
                "majority class",
                "baseline accuracy",
                "model accuracy",
                "accuracy - baseline",
            ],
            rows,
        )
    )
    lines.append("")

    lines.append("### Prediction distribution")
    lines.append("")
    rows = []
    for stage_name, stage in stages.items():
        model = stage.get("model") or {}
        distribution = model.get("prediction_distribution") or {}
        total = sum(
            value for value in distribution.values() if isinstance(value, (int, float))
        )
        cell = lambda key: _md_pct(
            (_md_lookup(distribution, key) or 0) / total if total else None
        )
        rows.append(
            [
                stage_name,
                _md_int(_md_lookup(distribution, -1)),
                cell(-1),
                _md_int(_md_lookup(distribution, 0)),
                cell(0),
                _md_int(_md_lookup(distribution, 1)),
                cell(1),
            ]
        )
    lines.append(
        _md_table(
            [
                "stage",
                "pred short count",
                "pred short share",
                "pred neutral count",
                "pred neutral share",
                "pred long count",
                "pred long share",
            ],
            rows,
        )
    )
    lines.append("")

    lines.append("### Confusion matrices (rows = true, columns = predicted)")
    lines.append("")
    for stage_name, stage in stages.items():
        model = stage.get("model") or {}
        lines.append(f"**{stage_name}**")
        lines.append("")
        lines.append(_md_confusion(model.get("confusion_matrix") or {}))
        lines.append("")

    # ------------------------------------------------------- sub models
    if sub_model_names:
        lines.append("## 4. Sub-model quality")
        lines.append("")
        lines.append(
            "Sub-model metrics are evaluated with the sub-model's own binary target "
            "(trigger = action vs no action, direction = long vs short). They are "
            "independent of the fused three-class decisions."
        )
        lines.append("")
        rows = []
        for name in sub_model_names:
            for stage_name, stage in stages.items():
                stats = (stage.get("sub_models") or {}).get(name)
                if not stats:
                    continue
                rows.append(
                    [
                        name,
                        _md_text(stats.get("task_type")),
                        stage_name,
                        _md_int(stats.get("sample_count")),
                        _md_float(stats.get("accuracy")),
                        _md_float(stats.get("baseline_accuracy")),
                        _md_float(
                            stats["accuracy"] - stats["baseline_accuracy"]
                            if stats.get("accuracy") is not None
                            and stats.get("baseline_accuracy") is not None
                            else None
                        ),
                        _md_float(stats.get("macro_f1")),
                        _md_float(stats.get("negative_precision")),
                        _md_float(stats.get("negative_recall")),
                        _md_float(stats.get("positive_precision")),
                        _md_float(stats.get("positive_recall")),
                    ]
                )
        lines.append(
            _md_table(
                [
                    "sub-model",
                    "task",
                    "stage",
                    "n",
                    "accuracy",
                    "baseline",
                    "accuracy - baseline",
                    "macro F1",
                    "p(neg)",
                    "r(neg)",
                    "p(pos)",
                    "r(pos)",
                ],
                rows,
            )
        )
        lines.append("")

        lines.append("### Sub-model prediction distribution")
        lines.append("")
        rows = []
        for name in sub_model_names:
            for stage_name, stage in stages.items():
                stats = (stage.get("sub_models") or {}).get(name)
                if not stats:
                    continue
                true_dist = stats.get("label_distribution_true") or {}
                pred_dist = stats.get("label_distribution_pred") or {}
                true_total = sum(
                    value
                    for value in true_dist.values()
                    if isinstance(value, (int, float))
                )
                pred_total = sum(
                    value
                    for value in pred_dist.values()
                    if isinstance(value, (int, float))
                )
                rows.append(
                    [
                        name,
                        stage_name,
                        _md_pct(
                            (_md_lookup(true_dist, 1) or 0) / true_total
                            if true_total
                            else None
                        ),
                        _md_pct(
                            (_md_lookup(true_dist, 0) or 0) / true_total
                            if true_total
                            else None
                        ),
                        _md_pct(
                            (_md_lookup(pred_dist, 1) or 0) / pred_total
                            if pred_total
                            else None
                        ),
                        _md_pct(
                            (_md_lookup(pred_dist, 0) or 0) / pred_total
                            if pred_total
                            else None
                        ),
                    ]
                )
        lines.append(
            _md_table(
                [
                    "sub-model",
                    "stage",
                    "true positive",
                    "true negative",
                    "pred positive",
                    "pred negative",
                ],
                rows,
            )
        )
        lines.append("")

        lines.append("### Sub-model confusion matrices (rows = true, columns = predicted)")
        lines.append("")
        for name in sub_model_names:
            for stage_name, stage in stages.items():
                stats = (stage.get("sub_models") or {}).get(name)
                if not stats:
                    continue
                lines.append(f"**{name} / {stage_name}**")
                lines.append("")
                lines.append(_md_confusion(stats.get("confusion_matrix") or {}))
                lines.append("")

    # ------------------------------------------------------- trading
    lines.append("## 5. Trading performance per stage")
    lines.append("")
    lines.append(
        "Each stage runs its own broker, so these numbers are the standalone "
        "performance of that stage, not a slice of the combined run. `equity return` "
        "is the final equity change (already net of commission); the gross PnL table "
        "below reconstructs pre-cost totals as `avg gross x trades`."
    )
    lines.append("")

    lines.append("### Returns and risk")
    lines.append("")
    rows = []
    for stage_name, stage in stages.items():
        strategy_block = stage.get("strategy") or {}
        performance = strategy_block.get("performance") or {}
        drawdown = strategy_block.get("drawdown") or {}
        rows.append(
            [
                stage_name,
                _md_pct(performance.get("gross_return")),
                _md_pct(performance.get("cagr")),
                _md_float(performance.get("sharpe"), 2),
                _md_float(performance.get("calmar"), 2),
                _md_raw_pct(drawdown.get("max_dd_pct")),
                _md_pct(drawdown.get("max_daily_dd")),
                _md_pct(drawdown.get("robust_max_daily_loss")),
                _md_int(drawdown.get("max_hwm_duration_days")),
            ]
        )
    lines.append(
        _md_table(
            [
                "stage",
                "equity return",
                "CAGR",
                "Sharpe",
                "Calmar",
                "max DD",
                "max daily DD",
                "robust daily loss",
                "max recovery days",
            ],
            rows,
        )
    )
    lines.append("")

    rows = []
    for stage_name, stage in stages.items():
        strategy_block = stage.get("strategy") or {}
        performance = strategy_block.get("performance") or {}
        drawdown = strategy_block.get("drawdown") or {}
        rc = performance.get("rc_summary") or {}
        rows.append(
            [
                stage_name,
                _md_money(performance.get("start_value")),
                _md_money(performance.get("end_value")),
                _md_money(drawdown.get("max_dd_amt")),
                _md_text(drawdown.get("max_daily_date")),
                _md_int(drawdown.get("dd_3_pct_days")),
                _md_int(drawdown.get("dd_4_pct_days")),
                _md_int(drawdown.get("dd_5_pct_days")),
                _md_int(rc.get("rc_n")),
                _md_float(rc.get("rc_median"), 2),
                _md_float(rc.get("rc_q05"), 2),
                _md_float(rc.get("rc_q95"), 2),
                _md_pct(rc.get("rc_pos_ratio")),
            ]
        )
    lines.append(
        _md_table(
            [
                "stage",
                "start equity",
                "end equity",
                "max DD amount",
                "worst day",
                "days DD >3%",
                "days DD >4%",
                "days DD >5%",
                "rolling calmar n",
                "rolling calmar median",
                "rolling calmar P05",
                "rolling calmar P95",
                "rolling calmar >0 share",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "Rolling-calmar statistics are computed over rolling windows and measure "
        "persistence: a high `>0 share` means the strategy stays profitable across "
        "most windows instead of relying on a few spikes."
    )
    lines.append("")

    lines.append("### Exposure")
    lines.append("")
    rows = []
    for stage_name, stage in stages.items():
        strategy_block = stage.get("strategy") or {}
        exposure = strategy_block.get("exposure") or {}
        rows.append(
            [
                stage_name,
                _md_pct(exposure.get("avg_pos")),
                _md_pct(exposure.get("max_pos")),
                _md_pct(exposure.get("p95_pos")),
                _md_pct(exposure.get("risk_per_trade_pct")),
            ]
        )
    lines.append(
        _md_table(
            [
                "stage",
                "average exposure",
                "maximum exposure",
                "P95 exposure",
                "risk per trade",
            ],
            rows,
        )
    )
    lines.append("")

    lines.append("### Trade activity")
    lines.append("")
    rows = []
    for stage_name, stage in stages.items():
        strategy_block = stage.get("strategy") or {}
        trades = strategy_block.get("trades") or {}
        holding = trades.get("holding_period_bars") or {}
        rows.append(
            [
                stage_name,
                _md_int(trades.get("total")),
                _md_float(trades.get("daily_freq"), 2),
                _md_pct(trades.get("win_rate")),
                _md_float(trades.get("profit_factor"), 3),
                _md_float(holding.get("mean"), 1),
                _md_float(holding.get("p50"), 1),
                _md_float(holding.get("p90"), 1),
                _md_int(holding.get("max")),
                _md_int(trades.get("lost_longest")),
                _md_int(trades.get("won_longest")),
            ]
        )
    lines.append(
        _md_table(
            [
                "stage",
                "trades",
                "trades/day",
                "win rate",
                "profit factor",
                "hold bars mean",
                "hold bars P50",
                "hold bars P90",
                "hold bars max",
                "longest losing streak",
                "longest winning streak",
            ],
            rows,
        )
    )
    lines.append("")

    rows = []
    for stage_name, stage in stages.items():
        strategy_block = stage.get("strategy") or {}
        trades = strategy_block.get("trades") or {}
        rows.append(
            [
                stage_name,
                _md_money(trades.get("avg_pnl_gross")),
                _md_raw_pct(trades.get("avg_pct_gross"), 3),
                _md_money(trades.get("avg_pnl_net")),
                _md_raw_pct(trades.get("avg_pct_net"), 3),
                _md_money(trades.get("avg_cost")),
                _md_money(trades.get("long_pnl")),
                _md_raw_pct(trades.get("long_win_rate"), 1),
                _md_money(trades.get("short_pnl")),
                _md_raw_pct(trades.get("short_win_rate"), 1),
            ]
        )
    lines.append(
        _md_table(
            [
                "stage",
                "avg gross $",
                "avg gross %",
                "avg net $",
                "avg net %",
                "avg cost $",
                "long PnL $",
                "long win rate",
                "short PnL $",
                "short win rate",
            ],
            rows,
        )
    )
    lines.append("")

    lines.append("### Trade totals and cost drag")
    lines.append("")
    rows = []
    for stage_name, stage in stages.items():
        trades = (stage.get("strategy") or {}).get("trades") or {}
        total = trades.get("total") or 0
        avg_gross = trades.get("avg_pnl_gross")
        avg_cost = trades.get("avg_cost")
        total_gross = avg_gross * total if avg_gross is not None else None
        total_cost = avg_cost * total if avg_cost is not None else None
        total_net = (
            total_gross - total_cost
            if total_gross is not None and total_cost is not None
            else None
        )
        rows.append(
            [
                stage_name,
                _md_int(total),
                _md_money(total_gross),
                _md_money(total_cost),
                _md_money(total_net),
                _md_money(
                    total_net / total if total and total_net is not None else None
                ),
                _md_money(
                    total_cost / total if total and total_cost is not None else None
                ),
            ]
        )
    lines.append(
        _md_table(
            [
                "stage",
                "trades",
                "gross PnL $",
                "commission $",
                "net PnL $",
                "net per trade $",
                "cost per trade $",
            ],
            rows,
        )
    )
    lines.append("")

    lines.append("### Strategy diagnostics")
    lines.append("")
    rows = []
    for stage_name, stage in stages.items():
        diagnostics = (stage.get("strategy") or {}).get("strategy") or {}
        rows.append(
            [
                stage_name,
                _md_text(diagnostics.get("strategy")),
                _md_float(diagnostics.get("input_f1")),
                _md_float(diagnostics.get("label_long_consistency")),
                _md_float(diagnostics.get("label_short_consistency")),
                _md_int(diagnostics.get("label_long_count")),
                _md_int(diagnostics.get("label_short_count")),
                _md_int(diagnostics.get("entries")),
                _md_int(diagnostics.get("fixed_hold_bars")),
                _md_int(diagnostics.get("meltdown_days")),
                _md_int(diagnostics.get("unexplained_meltdown")),
                _md_int(diagnostics.get("skipped_missing_threshold")),
                _md_int(diagnostics.get("skipped_small_threshold")),
                _md_int(diagnostics.get("skipped_size")),
                _md_float(diagnostics.get("max_margin_level"), 4),
            ]
        )
    lines.append(
        _md_table(
            [
                "stage",
                "venue",
                "input F1",
                "long label consistency",
                "short label consistency",
                "long label count",
                "short label count",
                "entries",
                "fixed hold bars",
                "meltdown days",
                "unexplained meltdown",
                "skipped (no threshold)",
                "skipped (small threshold)",
                "skipped (size)",
                "max margin level",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "`input F1` is recomputed on the signals the strategy actually saw. "
        "`label consistency` audits the label alignment: it compares the label "
        "lookback price with the current price and should stay near 100%."
    )
    lines.append("")

    # ------------------------------------------------------- overall
    lines.append("## 6. Whole-period run (all stages combined)")
    lines.append("")
    performance = overall.get("performance") or {}
    drawdown = overall.get("drawdown") or {}
    exposure = overall.get("exposure") or {}
    trades = overall.get("trades") or {}
    time_block = overall.get("time") or {}
    rows = [
        [
            _md_text(time_block.get("start")),
            _md_text(time_block.get("end")),
            _md_pct(performance.get("gross_return")),
            _md_pct(performance.get("cagr")),
            _md_float(performance.get("sharpe"), 2),
            _md_float(performance.get("calmar"), 2),
            _md_raw_pct(drawdown.get("max_dd_pct")),
            _md_pct(drawdown.get("max_daily_dd")),
            _md_int(trades.get("total")),
            _md_pct(trades.get("win_rate")),
            _md_float(trades.get("profit_factor"), 3),
            _md_pct(exposure.get("avg_pos")),
            _md_pct(exposure.get("max_pos")),
        ]
    ]
    lines.append(
        _md_table(
            [
                "start",
                "end",
                "equity return",
                "CAGR",
                "Sharpe",
                "Calmar",
                "max DD",
                "max daily DD",
                "trades",
                "win rate",
                "profit factor",
                "avg exposure",
                "max exposure",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "The whole-period run replays all stages back-to-back with one broker. It "
        "is the number used for FTMO statistics and for the frontend report; compare "
        "it with the per-stage runs above to see whether one stage dominates."
    )
    lines.append("")

    ftmo = overall.get("ftmo_challenge") or {}
    ftmo_periods = ftmo.get("periods") or {}
    if ftmo_periods:
        lines.append("### FTMO barrier simulation")
        lines.append("")
        rows = []
        for period_name, period_stats in ftmo_periods.items():
            duration = (period_stats.get("duration_days") or {}).get("resolved") or {}
            rows.append(
                [
                    period_name,
                    _md_int(period_stats.get("start_count")),
                    _md_int(period_stats.get("resolved_count")),
                    _md_pct(period_stats.get("profit_target_rate_of_resolved")),
                    _md_pct(period_stats.get("loss_limit_rate")),
                    _md_pct(period_stats.get("unresolved_rate")),
                    _md_float(duration.get("p50"), 1),
                    _md_float(duration.get("mean"), 1),
                ]
            )
        lines.append(
            _md_table(
                [
                    "segment",
                    "start dates",
                    "resolved",
                    "target hit share of resolved",
                    "loss-limit share",
                    "unresolved share",
                    "median days to resolution",
                    "mean days to resolution",
                ],
                rows,
            )
        )
        lines.append("")
        lines.append(
            "Method: `{}`, equity resolution `{}`, profit target {}, loss limit {}.".format(
                _md_text(ftmo.get("method")),
                _md_text(ftmo.get("equity_resolution")),
                _md_pct(ftmo.get("profit_target_pct")),
                _md_pct(ftmo.get("loss_limit_pct")),
            )
        )
        lines.append("")

    # ------------------------------------------------------- notes
    lines.append("## 7. Notes")
    lines.append("")
    lines.append(
        "- Baseline = share of the most frequent true label class. It is a naive "
        "constant predictor, not a tradable strategy."
    )
    lines.append(
        "- `equity return` is net of commission; pre-cost PnL in the totals table is "
        "`avg gross x trades`."
    )
    lines.append(
        "- Stage backtests are independent runs (own broker/cash); FTMO statistics "
        "exist only for the whole-period report."
    )
    lines.append(
        "- Per-bar model output (including sub-model probabilities) is saved in "
        "`stage_predictions/<stage>_predictions.csv`."
    )
    lines.append(
        "- Long/short precision is not comparable to 50%: compare it with the "
        "per-class base rate (`precision lift`) to judge directional skill."
    )
    lines.append(
        "- In TBM labeling a stop-loss touch alone does not create the opposite "
        "class; the opposite class needs a reversed take-profit barrier first. "
        "This is why some stopped-out trades still carry the neutral label."
    )
    lines.append(
        "- Sub-model targets are task-specific: `trigger` = action vs no action, "
        "`direction` = long vs short, `long_ovr` / `short_ovr` = one side vs the rest."
    )
    lines.append(
        "- `min_expected_move` is a fraction of price. The labeling filter and the "
        "strategy filter are reported separately; when they differ, the dataset "
        "rejected smaller barrier moves than the backtest strategy does."
    )
    lines.append("")
    return "\n".join(lines)


def write_stage_summary_markdown(save_dir: Optional[str], summary: dict) -> Optional[str]:
    """Persist the human-readable stage summary next to the JSON summary."""
    if not save_dir:
        return None
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, _STAGE_SUMMARY_MARKDOWN_FILENAME)
    with open(output_path, "w", encoding="utf-8") as output:
        output.write(build_stage_summary_markdown(summary))
    return output_path


_ANALYSIS_MODES = ("none", "trigger_perfect", "direction_perfect")


def _apply_analysis_mode(frame: pd.DataFrame, analysis_mode: str) -> pd.DataFrame:
    """Apply idealized assumptions while preserving the other predicted component.

    Trigger-perfect mode uses the label for action/no-action and the direction
    model for active labels. Rows without a model prediction (dropped by the
    continuity/label filters) keep no signal because no direction is available.
    Direction-perfect mode corrects direction only when the predicted trigger is
    active; neutral and invalid labels remain predicted.
    """
    if analysis_mode not in _ANALYSIS_MODES:
        raise ValueError(f"analysis_mode must be one of: {', '.join(_ANALYSIS_MODES)}")
    if analysis_mode == "none":
        return frame
    if "label" not in frame.columns:
        raise ValueError(f"Analysis mode {analysis_mode!r} requires a 'label' column")

    labels = pd.to_numeric(frame["label"], errors="coerce")
    valid_labels = labels.isin(
        [int(Signal.NEGATIVE), int(Signal.NEUTRAL), int(Signal.POSITIVE)]
    )
    directional_labels = valid_labels & labels.ne(int(Signal.NEUTRAL))
    analyzed = frame.copy()

    if analysis_mode == "trigger_perfect":
        if "direction_pred" not in frame.columns:
            raise ValueError(
                "Trigger-perfect analysis requires direction predictions from a "
                "TRIGGER_DIRECTION model"
            )
        direction_preds = pd.to_numeric(frame["direction_pred"], errors="coerce")
        has_direction = direction_preds.isin(
            [int(Signal.NEGATIVE), int(Signal.POSITIVE)]
        )
        # Windows dropped by the continuity/label checks have no prediction at
        # all, so they cannot be traded and have no direction to retain.
        predicted_rows = pd.to_numeric(frame["pred"], errors="coerce").isin(
            [int(Signal.NEGATIVE), int(Signal.NEUTRAL), int(Signal.POSITIVE)]
        )
        missing_direction = directional_labels & predicted_rows & ~has_direction
        if missing_direction.any():
            raise ValueError(
                "Trigger-perfect analysis is missing direction predictions for "
                f"{int(missing_direction.sum())} predicted directional labels"
            )
        analyzed.loc[valid_labels & labels.eq(int(Signal.NEUTRAL)), "pred"] = int(Signal.NEUTRAL)
        analyzed.loc[directional_labels & has_direction, "pred"] = direction_preds.loc[
            directional_labels & has_direction
        ]
    else:
        predicted_trigger = pd.to_numeric(frame["pred"], errors="coerce").isin(
            [int(Signal.NEGATIVE), int(Signal.POSITIVE)]
        )
        correctly_triggered = directional_labels & predicted_trigger
        analyzed.loc[correctly_triggered, "pred"] = labels.loc[correctly_triggered]

    return analyzed


def _write_json_file(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as output:
        json.dump(
            payload,
            output,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )


def _run_backtest(
    logger,
    frame: pd.DataFrame,
    config: RunnerConfig,
    strategy_config,
    venue_cls,
    pre_para,
    stage: Optional[str] = None,
):
    """Build the feed/cerebro pair for one frame and execute the strategy."""
    feed = _build_feed(frame, config.data_config)
    cerebro = create_backtest_cerebro(
        venue_cls=venue_cls,
        strategy_config=strategy_config,
        broker_config=config.broker_config,
        data=feed,
        predict_num=pre_para.predict_num if isinstance(pre_para, BaseDefine) else None,
        data_interval_ms=common.get_interval_ms(pre_para.interval if pre_para is not None else config.data_config.interval),
        engine_config=config.engine_config,
    )
    stage_label = f"stage={stage} | " if stage else ""
    logger.info(
        f"Starting backtest | {stage_label}data={type(config.data_config).__name__} "
        f"| venue={venue_cls.__name__} | bars={len(frame)}"
    )
    return cerebro.run()[0]


def _evaluate_stages(
    *,
    logger,
    frame: pd.DataFrame,
    config: RunnerConfig,
    strategy_config,
    venue_cls,
    pre_para,
    period: str,
    analysis_mode: str,
    saved_train_config,
    data_manifest,
    time_regions: dict,
    stage_model_stats: dict,
) -> tuple[dict, dict, dict]:
    """Run one independent backtest per model stage and collect its results.

    Each stage gets its own broker, so equity, drawdown and trade statistics are
    that stage's standalone performance instead of a slice of the combined run.
    """
    stage_results = {}
    stage_details = {}
    stage_prediction_files = {}

    for stage, region_key in model_loader.ModelHandler.STAGE_REGIONS:
        region = time_regions.get(region_key)
        if not region:
            logger.info("Stage %s has no model time region; skipping", stage)
            continue
        stage_frame = slice_stage_frame(frame, region)
        if stage_frame.empty:
            logger.info("Stage %s has no bars in the backtest frame; skipping", stage)
            continue

        stage_strat = _run_backtest(
            logger,
            stage_frame,
            config,
            strategy_config,
            venue_cls,
            pre_para,
            stage=stage,
        )
        stage_metrics = stage_model_stats.get(stage, {})
        stage_report, stage_detail = generate_backtest_report(
            logger,
            stage_strat,
            period=period,
            model_stats=stage_metrics.get("fused", {}),
            strategy_config=strategy_config,
            broker_config=config.broker_config,
            pre_para=pre_para,
            train_cfg=saved_train_config,
            sub_model_stats=stage_metrics.get("sub_models", {}),
            data_config=config.data_config,
            time_regions={stage: region},
            data_manifest=data_manifest,
            experiment_context=config.experiment_context,
            analysis_mode=analysis_mode,
            stage=stage,
        )
        stage_results[stage] = stage_report["results"][stage]
        stage_details[stage] = stage_detail["results"][stage]

        prediction_path = write_stage_predictions(config.save_dir, stage, stage_frame)
        if prediction_path:
            stage_prediction_files[stage] = prediction_path
            logger.info("Stage %s predictions saved: %s", stage, prediction_path)

        fused = stage_metrics.get("fused") or {}
        if fused:
            signal = fused.get("signal") or {}
            baseline, baseline_class = _baseline_from_stats(fused)
            logger.info(
                "STAGE %s MODEL | accuracy=%.4f | baseline=%.4f (class %s) | "
                "macro_f1=%.4f | coverage=%.4f | long_precision=%.4f | "
                "short_precision=%.4f | directional_accuracy=%.4f",
                stage.upper(),
                _number(fused.get("accuracy")),
                _number(baseline),
                baseline_class,
                _number(fused.get("f1_macro")),
                _number(signal.get("coverage")),
                _number(signal.get("long_win_rate")),
                _number(signal.get("short_win_rate")),
                _number(signal.get("directional_accuracy")),
            )
        for name, sub_stats in (stage_metrics.get("sub_models") or {}).items():
            sub_report = sub_stats.get("classification_report") or {}
            logger.info(
                "STAGE %s SUB %-9s | acc=%.4f | macro_f1=%.4f | "
                "p(pos)=%.4f | r(pos)=%.4f | p(neg)=%.4f | r(neg)=%.4f | n=%s",
                stage.upper(),
                name,
                _number(sub_stats.get("accuracy")),
                _number(sub_stats.get("f1_macro")),
                _number((sub_report.get("1") or {}).get("precision")),
                _number((sub_report.get("1") or {}).get("recall")),
                _number((sub_report.get("0") or {}).get("precision")),
                _number((sub_report.get("0") or {}).get("recall")),
                sub_stats.get("sample_count"),
            )

    if stage_results:
        logger.info(
            "Stage summary | stage   | bars    | trades | gross_ret | win_rate | accuracy | macro_f1"
        )
        for stage, stage_result in stage_results.items():
            stage_metrics = stage_model_stats.get(stage, {})
            fused = stage_metrics.get("fused") or {}
            logger.info(
                "Stage summary | %-7s | %7d | %6d | %9.4f | %8.4f | %8.4f | %8.4f",
                stage,
                stage_metrics.get("bar_count", 0),
                stage_result["trades"]["total"],
                stage_result["performance"]["gross_return"],
                stage_result["trades"]["win_rate"],
                fused.get("accuracy", 0.0),
                fused.get("f1_macro", 0.0),
            )

    return stage_results, stage_details, stage_prediction_files


def main(
    logger: logging.Logger,
    config: RunnerConfig,
    period: str,
    analysis_mode: str = "none",
    *,
    evaluate_stages: bool = False,
):
    """Standalone runner: load data/model, execute the strategy and build the report."""
    period = _validate_period(period)
    if analysis_mode not in _ANALYSIS_MODES:
        raise ValueError(f"analysis_mode must be one of: {', '.join(_ANALYSIS_MODES)}")
    if isinstance(config.data_config, ModelDataConfig):
        preparation_manifest = common.load_data_manifest_from_dir(
            config.data_config.prep_output_dir,
        )
        data_manifest = preparation_manifest["source"]
        saved_train_config = _load_train_config(config.data_config.train_output_dir)
        (
            frame,
            model_stats,
            sub_model_stats,
            pre_para,
            time_regions,
            stage_model_stats,
        ) = _load_model_data(
            logger,
            config.data_config,
            saved_train_config,
            period,
        )
    elif isinstance(config.data_config, CsvDataConfig):
        data_manifest = None
        saved_train_config = None
        (
            frame,
            model_stats,
            sub_model_stats,
            pre_para,
            time_regions,
            stage_model_stats,
        ) = _load_csv_data(
            logger,
            config.data_config,
        )
    else:
        raise TypeError(f"Unsupported data config: {type(config.data_config).__name__}")

    frame = _apply_analysis_mode(frame, analysis_mode)
    if analysis_mode != "none":
        logger.info("Running backtest analysis mode: %s", analysis_mode)
    strategy_config = strategy_config_for_preparation(
        config.strategy_config,
        pre_para,
    )
    venue_cls = _venue_for_strategy(strategy_config)
    strat = _run_backtest(
        logger,
        frame,
        config,
        strategy_config,
        venue_cls,
        pre_para,
    )
    report, report_details = generate_backtest_report(
        logger,
        strat,
        period=period,
        model_stats=model_stats,
        strategy_config=strategy_config,
        broker_config=config.broker_config,
        pre_para=pre_para,
        train_cfg=saved_train_config,
        sub_model_stats=sub_model_stats,
        data_config=config.data_config,
        time_regions=time_regions,
        data_manifest=data_manifest,
        experiment_context=config.experiment_context,
        analysis_mode=analysis_mode,
    )

    stage_results = {}
    stage_details = {}
    stage_prediction_files = {}
    if evaluate_stages and isinstance(config.data_config, ModelDataConfig):
        if not time_regions:
            logger.info("Stage analysis requires model time regions; skipping")
        else:
            stage_results, stage_details, stage_prediction_files = _evaluate_stages(
                logger=logger,
                frame=frame,
                config=config,
                strategy_config=strategy_config,
                venue_cls=venue_cls,
                pre_para=pre_para,
                period=period,
                analysis_mode=analysis_mode,
                saved_train_config=saved_train_config,
                data_manifest=data_manifest,
                time_regions=time_regions,
                stage_model_stats=stage_model_stats or {},
            )
            report["stage_results"] = stage_results
            report_details["stage_results"] = stage_details
            report["stage_model_metrics"] = stage_model_stats or {}
            if stage_prediction_files:
                report["stage_prediction_files"] = stage_prediction_files
            if stage_results:
                summary = build_stage_summary(
                    stage_results=stage_results,
                    stage_model_stats=stage_model_stats or {},
                    stage_prediction_files=stage_prediction_files,
                    pre_para=pre_para,
                    train_cfg=saved_train_config,
                    model_task_type=_artifact_task_type(
                        config.data_config.train_output_dir
                    ),
                    period=period,
                    analysis_mode=analysis_mode,
                    overall_result=report["results"][period],
                    strategy_config=strategy_config,
                    broker_config=config.broker_config,
                )
                summary_path = write_stage_summary(config.save_dir, summary)
                if summary_path:
                    report["stage_summary_file"] = summary_path
                    logger.info("Stage summary saved: %s", summary_path)
                markdown_path = write_stage_summary_markdown(config.save_dir, summary)
                if markdown_path:
                    report["stage_summary_markdown"] = markdown_path
                    logger.info("Stage summary markdown saved: %s", markdown_path)

    candle_columns = ["open_time_date_utc", "open", "high", "low", "close", "volume"]
    candle_columns.extend(name for name in ("pred", "label") if name in frame.columns)
    candles = frame[candle_columns].copy()
    if "pred" in candles:
        candles["pred"] = candles["pred"].fillna(float(Signal.INVALID))
    if "label" in candles:
        candles["label"] = candles["label"].fillna(int(Signal.INVALID)).astype(int)
    candles.rename(columns={"open_time_date_utc": "time"}, inplace=True)
    candles["time"] = candles["time"].apply(lambda dt: int(dt.timestamp()))

    result = {
        "schema_version": 2,
        "generated_at": datetime_module.datetime.now(datetime_module.timezone.utc).isoformat(),
        "candles": candles.to_dict(orient="records"),
        "report": report,
        "report_details": report_details,
    }
    return result


def _load_train_config(train_output_dir: str) -> train_config.TrainConfig:
    """Restore the exact TrainConfig persisted with the model artifact.

    Composite/fusion artifacts (e.g. TRIGGER_DIRECTION) are assembled from
    existing single-model runs and carry no train_config.json of their own;
    fall back to the first sub-model dir referenced by task_description.json.
    """

    artifact_dir = train_output_dir
    if not os.path.isabs(artifact_dir):
        artifact_dir = os.path.join(PROJECT_DIR, artifact_dir)
    config_path = os.path.join(artifact_dir, "train_config.json")
    if not os.path.isfile(config_path):
        config_path = _fusion_train_config_path(artifact_dir)
    if config_path is None:
        raise FileNotFoundError(
            f"train_config.json not found in training output: {artifact_dir}",
        )
    with open(config_path, "r", encoding="utf-8") as config_file:
        payload = json.load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"train_config.json must contain an object: {config_path}")
    return config_from_dict_train(payload)


def _fusion_train_config_path(artifact_dir: str) -> Optional[str]:
    """Locate a train_config.json inside a fusion artifact's sub-model dirs."""
    task_desc_path = os.path.join(artifact_dir, model_loader.TASK_DESCRIPTION_FILENAME)
    if not os.path.isfile(task_desc_path):
        return None
    with open(task_desc_path, "r", encoding="utf-8") as task_file:
        task_desc = json.load(task_file)
    models = task_desc.get("models", {})
    if "main" in models:
        # Single-run layout; train_config.json is genuinely missing.
        return None
    for role_dir in models.values():
        candidate = os.path.join(artifact_dir, role_dir, "train_config.json")
        if os.path.isfile(candidate):
            return candidate
    return None


def build_daily_df(daily_stats):
    """Convert daily account summaries to a date-sorted DataFrame."""
    if not daily_stats:
        return pd.DataFrame()
    df = pd.DataFrame(daily_stats)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


# 6 months (180 or 365 recommended for crypto)
def rolling_calmar(df: pd.DataFrame, window_days: int = 180, step_days: int = 30):
    """
    df must contain: date, end_equity
    """
    if df is None or df.empty or "date" not in df.columns or "end_equity" not in df.columns:
        return pd.DataFrame()

    df = df[["date", "end_equity"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["end_equity"] = pd.to_numeric(df["end_equity"], errors="coerce")
    df = df.dropna(subset=["date", "end_equity"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    results = []

    dates = df["date"]
    equity = df["end_equity"].values

    start_idx = 0
    n = len(df)

    while start_idx < n:
        start_date = dates.iloc[start_idx]
        end_date = start_date + pd.Timedelta(days=window_days)

        # Find the index where the window ends
        candidate_idx = df.index[df["date"] <= end_date]
        if len(candidate_idx) == 0:
            break
        end_idx = int(candidate_idx.max())
        if end_idx <= start_idx:
            break

        eq_start = equity[start_idx]
        eq_end = equity[end_idx]

        # CAGR
        years = (dates.iloc[end_idx] - dates.iloc[start_idx]).days / 365
        if years <= 0 or eq_start <= 0 or eq_end <= 0:
            next_date = start_date + pd.Timedelta(days=step_days)
            next_idx = df.index[df["date"] >= next_date].min()
            if pd.isna(next_idx):
                break
            start_idx = int(next_idx)
            continue

        cagr = (eq_end / eq_start) ** (1 / years) - 1

        # Max drawdown (inside the window)
        window_eq = equity[start_idx : end_idx + 1]
        if np.any(window_eq <= 0):
            next_date = start_date + pd.Timedelta(days=step_days)
            next_idx = df.index[df["date"] >= next_date].min()
            if pd.isna(next_idx):
                break
            start_idx = int(next_idx)
            continue
        peak = np.maximum.accumulate(window_eq)
        dd = (window_eq - peak) / peak
        max_dd = dd.min()

        calmar = cagr / abs(max_dd) if max_dd < 0 else np.inf

        results.append(
            {
                "start": dates.iloc[start_idx],
                "end": dates.iloc[end_idx],
                "cagr": cagr,
                "max_dd": max_dd,
                "calmar": calmar,
            }
        )

        # Roll forward
        next_date = start_date + pd.Timedelta(days=step_days)
        next_idx = df.index[df["date"] >= next_date].min()
        if pd.isna(next_idx):
            break
        start_idx = int(next_idx)

    return pd.DataFrame(results)


def summarize_rolling_calmar(rc_df: pd.DataFrame) -> dict:
    """
    rc_df columns: start,end,cagr,max_dd,calmar
    Returns finer grained rolling metrics: distribution + tail risk + persistence + robustness
    """
    if rc_df is None or len(rc_df) == 0:
        return {"rc_n": 0}

    s = rc_df["calmar"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return {"rc_n": 0}

    # Basic distribution
    q = s.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).to_dict()

    # Persistence: longest consecutive run of "bad" / "goos" windows (in window order)
    # Note: the windows overlap, but a run still shows "unusable for a long stretch"
    flags_neg = (s < 0).to_numpy()
    flags_ge1 = (s >= 1).to_numpy()
    flags_ge2 = (s >= 2).to_numpy()

    def longest_run(flags: np.ndarray) -> int:
        best = cur = 0
        for f in flags:
            cur = cur + 1 if f else 0
            best = max(best, cur)
        return int(best)

    # Tail risk: expected shortfall (conditional mean)
    def expected_shortfall(x: pd.Series, alpha: float) -> float:
        thr = x.quantile(alpha)
        tail = x[x <= thr]
        return float(tail.mean()) if len(tail) else float("nan")

    # Robustness: MAD / IQR etc.
    median = float(q[0.50])
    mad = float((s - median).abs().median())  # median absolute deviation
    iqr = float(q[0.75] - q[0.25])

    # Share of "qualifying" windows (you prefer Calmar>=2)
    out = {
        "rc_n": int(len(s)),  # Total number of valid rolling calmar values
        # Quantiles (finer grained)
        "rc_q01": float(q[0.01]),  # 1st percentile of rolling calmar
        "rc_q05": float(q[0.05]),  # 5th percentile of rolling calmar
        "rc_q10": float(q[0.10]),  # 10th percentile of rolling calmar
        "rc_q25": float(q[0.25]),  # 25th percentile (lower quartile) of rolling calmar
        "rc_median": float(q[0.50]),  # Median (50th percentile) of rolling calmar
        "rc_q75": float(q[0.75]),  # 75th percentile (upper quartile) of rolling calmar
        "rc_q90": float(q[0.90]),  # 90th percentile of rolling calmar
        "rc_q95": float(q[0.95]),  # 95th percentile of rolling calmar
        "rc_q99": float(q[0.99]),  # best 99th percentile of rolling calmar
        # Ratios
        "rc_pos_ratio": float((s > 0).mean()),  # Proportion of positive rolling calmar values
        "rc_neg_ratio": float((s < 0).mean()),  # Proportion of negative rolling calmar values
        "rc_ge_1_ratio": float((s >= 1).mean()),  # Proportion of rolling calmar values >= 1
        "rc_ge_2_ratio": float((s >= 2).mean()),  # Proportion of rolling calmar values >= 2
        "rc_ge_3_ratio": float((s >= 3).mean()),  # Proportion of rolling calmar values >= 3
        # Tail (average level of the worst windows, more robust than min)
        "rc_es_05": expected_shortfall(s, 0.05),  # Average of the worst 5% rolling calmar values
        "rc_es_10": expected_shortfall(s, 0.10),  # Average of the worst 10% rolling calmar values
        # Persistence (ability to cross a death zone such as 2022-2023)
        "rc_longest_neg_run": longest_run(flags_neg),  # Longest consecutive run of negative rolling calmar values
        "rc_longest_ge1_run": longest_run(flags_ge1),  # Longest consecutive run of rolling calmar values >= 1
        "rc_longest_ge2_run": longest_run(flags_ge2),  # Longest consecutive run of rolling calmar values >= 2
        # Robustness / dispersion
        "rc_mean": float(s.mean()),  # Mean of rolling calmar values
        "rc_std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,  # Standard deviation of rolling calmar values
        "rc_mad": mad,  # Median absolute deviation of rolling calmar values
        "rc_iqr": iqr,  # Interquartile range (Q3 - Q1) of rolling calmar values
        "rc_cv": (
            float(s.std(ddof=1) / s.mean()) if len(s) > 1 and s.mean() != 0 else float("nan")
        ),  # Coefficient of variation (std/mean) of rolling calmar values
    }

    # Optional: carry the distribution of the rolling cagr / max_dd as well (more interpretable)
    if "cagr" in rc_df.columns:
        c = rc_df["cagr"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(c):
            out.update(
                {
                    "rc_cagr_median": float(c.median()),
                    "rc_cagr_q10": float(c.quantile(0.10)),
                    "rc_cagr_q25": float(c.quantile(0.25)),
                }
            )
    if "max_dd" in rc_df.columns:
        d = rc_df["max_dd"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(d):
            out.update(
                {
                    "rc_dd_median": float(d.median()),
                    "rc_dd_q90": float(d.quantile(0.90)),  # drawdowns are negative, so q90 is the closest to 0
                    "rc_dd_min": float(d.min()),  # worst drawdown
                }
            )

    return out


def summarize_holding_periods(hold_bars) -> dict:
    """Summarize the exact Backtrader bar lengths of closed trades."""
    keys = ("p10", "p25", "p50", "p75", "p90", "p95", "p99")
    empty = {
        "count": 0,
        "mean": 0.0,
        "min": 0,
        "max": 0,
        **{key: 0.0 for key in keys},
    }
    if not hold_bars:
        return empty

    bars = np.asarray(hold_bars, dtype=float)
    bars = bars[np.isfinite(bars) & (bars >= 0)]
    if not len(bars):
        return empty

    quantiles = np.quantile(
        bars,
        [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
    )
    return {
        "count": int(len(bars)),
        "mean": float(bars.mean()),
        "min": int(bars.min()),
        "max": int(bars.max()),
        **{key: float(value) for key, value in zip(keys, quantiles)},
    }


def generate_backtest_report(
    logger,
    strat,
    strategy_config: Any,
    broker_config: BrokerConfig,
    period: str,
    model_stats=None,
    pre_para: Optional[BaseDefine] = None,
    train_cfg=None,
    sub_model_stats=None,
    data_config=None,
    time_regions=None,
    data_manifest=None,
    experiment_context: Optional[ExperimentContext] = None,
    analysis_mode: str = "none",
    stage: Optional[str] = None,
):
    """
    Fixed report generator:
    1. Corrected the profit factor formula (Gross Won / Gross Lost)
    2. Corrected the PnL% extraction (handles backtrader's nested list structure)

    sub_model_stats: per sub-model metrics for composite tasks (TRIGGER_DIR/LONG_SHORT_OVR)
    (see ModelHandler.evaluate_sub_models); an empty dict for single model tasks.

    A ``stage`` report scores one model time region (train/valid/test/forward)
    instead of a full long/forward/all period, so it uses the stage name as its
    results key and skips the FTMO challenge, which needs a full period.
    """
    model_stats = model_stats or {}
    sub_model_stats = sub_model_stats or {}
    period = _validate_period(period) if stage is None else stage
    if analysis_mode not in _ANALYSIS_MODES:
        raise ValueError(f"analysis_mode must be one of: {', '.join(_ANALYSIS_MODES)}")
    if experiment_context is None:
        raise ValueError("experiment_context is required for reproducible reports")

    # ========== 1. Pull the analyzers ==========
    perf = strat.analyzers.customize.get_analysis()
    dd = strat.analyzers.dd.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()
    ret_analyzer = strat.analyzers.returns.get_analysis()

    # ----- basic numbers -----
    start_value = strat.broker.startingcash  # starting capital
    end_value = strat.broker.getvalue()  # final capital
    gross_return = (end_value - start_value) / start_value
    cagr = ret_analyzer.get("rnorm", 0.0)
    sr = sharpe.get("sharperatio", 0.0) or 0.0

    # ----- max drawdown -----
    maxdd_pct = dd.get("max", {}).get("drawdown", 0.0)
    maxdd_amt = dd.get("max", {}).get("moneydown", 0.0)
    maxdd_len = dd.get("max", {}).get("len", 0)
    calmar = (cagr * 100 / abs(maxdd_pct)) if maxdd_pct > 0 else 0.0
    # rolling calmar
    daily_account = perf.get("daily_account", [])
    if daily_account:
        df = build_daily_df(daily_account)
        rc_df = rolling_calmar(df, window_days=180, step_days=30)
        rc_summary = summarize_rolling_calmar(rc_df)
    else:
        rc_summary = {"rc_n": 0}

    lost_longest = safe_get(trades, ["streak", "lost", "longest"], 0)
    won_longest = safe_get(trades, ["streak", "won", "longest"], 0)
    # --- read the intraday drawdown data ---
    max_daily_dd = perf.get("max_daily_dd", 0.0)  # e.g. -0.045
    max_daily_date = perf.get("max_daily_dd_date", "N/A")
    max_violation_days = perf.get("daily_dd_max_violation_days", 0)
    max_3_violation_days = perf.get("daily_dd_max_3_violation_days", 0)
    violation_days = perf.get("daily_dd_violation_days", 0)
    # 1. Global minimum equity
    global_min_equity = perf.get("global_min_equity", 0.0)
    max_hwm_duration_days = perf.get("max_hwm_duration_days", 0)
    # 2. Distance from the starting capital (FTMO max loss)
    start_cash = strat.broker.startingcash
    dist_to_start_pct = (global_min_equity - start_cash) / start_cash
    # 3. Log it
    logger.info(f"FTMO LINE | Dist to Start: {dist_to_start_pct*100:.2f}% (Limit: -10%)")

    if dist_to_start_pct < -0.10:
        logger.warning("❌ FAILED: account equity fell below 10% of initial capital")
    # Print it into the log
    logger.info(f"RISK(Daily)| Worst Day: {max_daily_dd*100:.2f}% ({max_daily_date}) | >4% Days: {violation_days}")

    if max_daily_dd < -0.05:
        logger.warning("❌ Critical warning: daily drawdown breached the FTMO 5% limit")

    # ----- trade statistics (overall) -----
    total_trades = safe_get(trades, ["total", "closed"], 0)
    total_won = safe_get(trades, ["won", "total"], 0)
    total_lost = safe_get(trades, ["lost", "total"], 0)
    win_rate = (total_won / total_trades) if total_trades > 0 else 0.0
    closed_trade_hold_bars = list(getattr(strat, "closed_trade_hold_bars", []))
    holding_period_stats = summarize_holding_periods(closed_trade_hold_bars)

    # Backtrader classifies winning and losing trades by commission-adjusted PnL.
    # Profit factor is total winning PnL divided by absolute total losing PnL.
    winning_pnl_total = safe_get(trades, ["won", "pnl", "total"], 0.0)
    losing_pnl_total = abs(safe_get(trades, ["lost", "pnl", "total"], 0.0))
    profit_factor = winning_pnl_total / losing_pnl_total if losing_pnl_total > 0 else None

    # ----- raw absolute PnL -----
    avg_pnl_net = safe_get(trades, ["pnl", "net", "average"], 0.0)  # after costs
    avg_pnl_gross = safe_get(trades, ["pnl", "gross", "average"], 0.0)  # before costs
    avg_cost = avg_pnl_gross - avg_pnl_net

    # ============================================================
    # === [fix 2] per-trade return percentage (robust iteration) ===
    # ============================================================

    # 1. Daily frequency
    if len(strat.datas) > 0 and len(strat.datas[0]) > 0:
        t_start = bt.num2date(strat.datas[0].datetime.array[0])
        t_end = bt.num2date(strat.datas[0].datetime.array[-1])
        duration = t_end - t_start
        total_days = max(duration.days + (duration.seconds / 86400), 1)
    else:
        total_days = 1
    daily_trades = total_trades / total_days

    # Averages (*100 to get percent)
    if total_trades > 0 and abs(avg_cost) > 1e-12:
        avg_pct_gross = avg_pnl_gross / (avg_cost / 2) * broker_config.commission_pct
        avg_pct_net = avg_pnl_net / (avg_cost / 2) * broker_config.commission_pct
    else:
        avg_pct_gross = 0.0
        avg_pct_net = 0.0

    # ============================================================
    # --- 1. Long statistics ---
    long_total = safe_get(trades, ["long", "total"], 0)
    long_won = safe_get(trades, ["long", "won"], 0)  # number of wins
    # Long total pnl (amount)
    long_pnl_total = safe_get(trades, ["long", "pnl", "total"], 0.0)
    # Long win rate
    long_win_rate = (long_won / long_total * 100) if long_total > 0 else 0.0
    # --- 2. Short statistics ---
    short_total = safe_get(trades, ["short", "total"], 0)
    short_won = safe_get(trades, ["short", "won"], 0)
    # Short total pnl (amount)
    short_pnl_total = safe_get(trades, ["short", "pnl", "total"], 0.0)
    # Short win rate
    short_win_rate = (short_won / short_total * 100) if short_total > 0 else 0.0

    # Position information (already computed by CusAnalyzer)
    avg_pos = perf.get("avg_pos_ratio", 0)
    max_pos = perf.get("max_pos_ratio", 0)
    p95_pos = perf.get("p95_pos_ratio", 0)

    # ============================================================
    # === [added: robust risk statistics] only this block changed, to support the Top-N display ===
    # ============================================================
    daily_account = perf.get("daily_account", [])
    top_10_str = "N/A"
    robust_max_loss = 0.0

    if daily_account:
        # Keep negative intraday drawdowns and sort them worst first.
        losses_values = []
        for item in daily_account:
            if isinstance(item, dict):
                val = cus_analyzer.daily_drawdown_pct(item)
            else:
                val = item
            if val < 0:
                losses_values.append(val)

        sorted_losses = sorted(losses_values)
        displayed_losses = sorted_losses[:10]
        top_10_str = " | ".join([f"{l*100:.2f}%" for l in displayed_losses])

        # Robust max loss: drop the #1 outlier, average ranks 2-5
        robust_losses = sorted_losses[1:5]
        if robust_losses:
            robust_max_loss = sum(robust_losses) / len(robust_losses)
        else:
            robust_max_loss = sorted_losses[0] if sorted_losses else 0.0

    # ---- strategy specific statistics ----
    strategy_summary, strategy_details = strat.strategy_metrics() if hasattr(strat, "strategy_metrics") else ({}, {})

    params_snapshot = {
        "strategy": strategy_config_to_dict(strategy_config),
        "broker": asdict(broker_config),
        "analysis_mode": analysis_mode,
    }
    if data_config is not None:
        data_snapshot = asdict(data_config)
        params_snapshot["data"] = data_snapshot
    if pre_para is not None:
        params_snapshot["common"] = asdict(pre_para)
    if train_cfg is not None:
        params_snapshot["train"] = asdict(train_cfg)
    if data_manifest is not None:
        params_snapshot["data_manifest"] = data_manifest

    if pre_para is not None and train_cfg is not None:
        task_identity = TaskIdentity.from_configs(
            strategy_config=strategy_config,
            broker_config=broker_config,
            common=pre_para,
            train=train_cfg,
        )
        params_hash = (
            task_identity.full_hash
            if analysis_mode == "none"
            else param_hash(
                {
                    "identity": task_identity.as_dict(),
                    "analysis_mode": analysis_mode,
                }
            )
        )
        params_snapshot["identity"] = task_identity.as_dict()
    else:
        params_hash = param_hash(params_snapshot)
    params_snapshot.update(
        hash=params_hash,
        git_commit=experiment_context.git_commit,
    )

    size_param = getattr(
        strategy_config,
        "risk_per_trade_pct",
        getattr(strategy_config, "base_order_pct", None),
    )
    period_report = {
        f"time": {
            f"start": bt.num2date(strat.datas[0].datetime.array[0]),
            f"end": bt.num2date(strat.datas[0].datetime.array[-1]),
            f"regions": time_regions or {},
        },
        f"performance": {
            f"gross_return": gross_return,
            f"cagr": cagr,
            f"calmar": calmar,
            f"sharpe": sr,
            f"start_value": start_value,
            f"end_value": end_value,
            f"rc_summary": rc_summary,
        },
        f"drawdown": {
            f"max_dd_pct": maxdd_pct,
            f"max_dd_amt": maxdd_amt,
            f"max_daily_dd": max_daily_dd,
            f"max_daily_date": max_daily_date,
            f"robust_max_daily_loss": robust_max_loss,
            f"dd_3_pct_days": max_3_violation_days,
            f"dd_4_pct_days": violation_days,
            f"dd_5_pct_days": max_violation_days,
            f"max_hwm_duration_days": max_hwm_duration_days,
        },
        f"exposure": {
            f"avg_pos": avg_pos,
            f"max_pos": max_pos,
            f"p95_pos": p95_pos,
            f"risk_per_trade_pct": size_param,
        },
        f"trades": {
            f"total": total_trades,
            f"daily_freq": daily_trades,
            f"win_rate": win_rate,
            f"holding_period_bars": holding_period_stats,
            f"lost_longest": lost_longest,
            f"won_longest": won_longest,
            f"avg_pnl_gross": avg_pnl_gross,
            f"avg_pct_gross": avg_pct_gross,
            f"avg_pnl_net": avg_pnl_net,
            f"avg_pct_net": avg_pct_net,
            f"avg_cost": avg_cost,
            f"profit_factor": profit_factor,
            f"long_pnl": long_pnl_total,
            f"long_win_rate": long_win_rate,
            f"short_pnl": short_pnl_total,
            f"short_win_rate": short_win_rate,
        },
        f"strategy": strategy_summary,
        f"model_metrics": model_stats,
        f"sub_model_metrics": sub_model_stats,
    }
    if stage is None:
        ftmo_input = dict(period_report)
        ftmo_input["raw_analyzer"] = {
            "customize": {"daily_account": daily_account},
        }
        period_report["ftmo_challenge"] = simulate_ftmo_challenges(
            ftmo_input,
            period,
        )

    persisted_perf = dict(perf)
    period_details = {
        f"raw_analyzer": {
            f"customize": persisted_perf,
        },
        f"strategy_detail": strategy_details,
        f"trade_logs": list(getattr(strat, "trade_logs", [])),
    }

    # common.dump_params_json(train_cfg,logger)
    # common.dump_params_json(para,logger)
    # common.dump_params_json(pre_para,logger)

    # summary output
    if stage is not None:
        logger.info("=" * 25 + f" STAGE {stage.upper()} " + "=" * 25)
    logger.info("-" * 29 + f"PARAMS_HASH | {params_hash}" + "-" * 29)

    logger.info(f"RISK(Daily)| Top 10 Losses: [{top_10_str}]")
    if robust_max_loss:  # Only log if we calculated it
        logger.info(f"RISK(Daily)| Robust Max Loss (Avg 2nd-5th): " f"{robust_max_loss*100:.2f}%")
    logger.info(
        f"RISK(Daily)| Worst Day: "
        f"{period_report['drawdown']['max_daily_dd']*100:.2f}% "
        f"({period_report['drawdown']['max_daily_date']}) | "
        f">3% Days: {period_report['drawdown']['dd_3_pct_days']} | "
        f">4% Days: {period_report['drawdown']['dd_4_pct_days']} | "
        f">5% Days: {period_report['drawdown']['dd_5_pct_days']}"
    )

    logger.info(
        f"Time    | {period_report['time']['start']} --> {period_report['time']['end']} "
        f"| CAGR: {period_report['performance']['cagr']*100:.2f}% "
        f"| Calmar: {period_report['performance']['calmar']:.2f}"
    )

    logger.info(
        f"SUMMARY | GrossRet: {period_report['performance']['gross_return']*100:.2f}% "
        f"| Sharpe: {period_report['performance']['sharpe']:.3f} "
        f"| MaxDD: {period_report['drawdown']['max_dd_pct']:.2f}% "
        f"({period_report['drawdown']['max_dd_amt']:.0f}) "
        f"| end value {period_report['performance']['end_value']}"
    )

    logger.info(
        f"EXPOSURE| Avg Pos: {period_report['exposure']['avg_pos']*100:.2f}% "
        f"| Max Pos: {period_report['exposure']['max_pos']*100:.2f}% "
        f"| P95 Pos: {period_report['exposure']['p95_pos']*100:.2f}% "
        f"| risk_per_trade_pct: {period_report['exposure']['risk_per_trade_pct']}"
    )

    logger.info(
        f"TRADES  | Total: {period_report['trades']['total']} "
        f"| Freq: {period_report['trades']['daily_freq']:.2f} trades/day "
        f"| WinRate: {period_report['trades']['win_rate']*100:.2f}% "
        f"| lost_longest: {period_report['trades']['lost_longest']} "
        f"| won_longest: {period_report['trades']['won_longest']}"
    )

    hold = period_report["trades"]["holding_period_bars"]
    logger.info(
        f"HOLD PCT| P10: {hold['p10']:.2f} | P25: {hold['p25']:.2f} "
        f"| P50: {hold['p50']:.2f} "
        f"| P75: {hold['p75']:.2f} | P90: {hold['p90']:.2f} "
        f"| P95: {hold['p95']:.2f} | P99: {hold['p99']:.2f} bars"
    )

    if stage is None:
        log_ftmo_challenge_summary(period_report["ftmo_challenge"], logger=logger)

    logger.info(
        f"PNL($)  | Avg Gross: {period_report['trades']['avg_pnl_gross']:.2f}"
        f"({period_report['trades']['avg_pct_gross']:.3f}%) "
        f"| Avg Net: {period_report['trades']['avg_pnl_net']:.2f}"
        f"({period_report['trades']['avg_pct_net']:.3f}%) "
        f"(Cost: {period_report['trades']['avg_cost']:.2f}/trade)"
    )

    log_metrics_block(logger, "STRATEGY", period_report["strategy"])

    logger.info(
        f"DETAILS | Long: {period_report['trades']['long_pnl']} "
        f"Winrate: {period_report['trades']['long_win_rate']:.1f}% | "
        f"Short: {period_report['trades']['short_pnl']} "
        f"Winrate: {period_report['trades']['short_win_rate']:.1f}%"
    )

    logger.info("-" * 80)

    report = {
        "params": params_snapshot,
        "results": {period: period_report},
    }
    report_details = {
        "results": {period: period_details},
    }
    return report, report_details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a model backtest.")
    parser.add_argument(
        "--analysis-mode",
        choices=_ANALYSIS_MODES,
        default="none",
        help=(
            "Choose none, trigger_perfect (actual label controls trigger and "
            "model direction is retained), or direction_perfect (correct the "
            "direction when the model triggered)."
        ),
    )
    parser.add_argument(
        "--skip-stages",
        action="store_true",
        help="Skip the standalone train/valid/test/forward stage backtests.",
    )
    args = parser.parse_args()
    train_output_dir = os.path.join(common.TRAIN_OUT_DIR, train_config.TrainTask.TRIGGER_DIRECTION)
    start_time = time.time()
    pre_para: BaseDefine = common.load_pre_params_from_dir(train_output_dir)
    exp_dir = common.create_experiment_dir(os.path.join(common.PERSISTENCE_DIR, "simulation"), pre_para.symbol, pre_para.interval)
    logger, _ = common.setup_session_logger(log_file_path=os.path.join(exp_dir, "experiment.log"), console_level=logging.INFO, file_level=logging.INFO)
    strategy_config = BbmStrategyConfig(fixed_hold_bars = pre_para.predict_num)#int(pre_para.predict_num))
    broker_config = BrokerConfig()
    runner_config = RunnerConfig(
        strategy_config=strategy_config,
        broker_config=broker_config,
        save_dir=exp_dir,
        experiment_context=ExperimentContext(git_commit=common.git_revision()),
        data_config=ModelDataConfig(prep_output_dir=common.DATA_OUT_DIR, train_output_dir=train_output_dir, device="auto"),
    )
    result = main(
        logger,
        runner_config,
        "all",
        analysis_mode=args.analysis_mode,
        evaluate_stages=not args.skip_stages,
    )
    output_path = os.path.join(exp_dir, "report.jsonl")
    append_jsonl(output_path, result["report"])

    identity = TaskIdentity(**result["report"]["params"]["identity"])
    detail_output_path = os.path.join(
        os.path.dirname(output_path),
        identity.prep_hash,
        identity.train_hash,
        identity.sim_hash,
        "report_details.json",
    )
    os.makedirs(os.path.dirname(detail_output_path), exist_ok=True)
    _write_json_file(detail_output_path, result["report_details"])

    plot_equity_curves(
        result,
        exp_dir,
        file_name="equity_curve.png",
        normalize_equity=False,
        logger=logger,
    )
    frontend_report_path = write_latest_backtest_report(result)
    logger.info("Frontend report published to %s", frontend_report_path)
    end_time = time.time()
    run_time = end_time - start_time
    logger.info(f": run_time: {run_time:.4f} s, report saved to {output_path}")
