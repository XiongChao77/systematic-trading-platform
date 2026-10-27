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
_PREDICTION_CACHE_SCHEMA = 3
_PREDICTION_COLUMNS = (
    "pred",
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
    """Calculate the exact train/valid/test/OOD timestamp regions."""
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
    ood_frame = current_frame if period == "forward" else common.load_test_df_from_dir(data_config.prep_output_dir)
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
        return pd.Timestamp(value).isoformat()

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
    ood_times = pd.to_datetime(ood_frame["open_time_date_utc"], utc=True)
    if ood_times.empty:
        raise ValueError("OOD dataset is empty; cannot calculate plot boundaries")
    regions["ood"] = {
        "start": as_iso(ood_times.iloc[0]),
        "end": as_iso(ood_times.iloc[-1]),
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
    predicted_frame, model_stats, sub_model_stats = handler.predict_with_ds(
        dataset,
        frame,
        is_live=False,
        batch_size=inference_batch_size,
        diff_thresh=None,
        evaluate_sub_models=True,
    )
    first_valid_idx = predicted_frame["pred"].first_valid_index()
    if first_valid_idx is None:
        raise RuntimeError("No valid predictions found in the entire dataset")
    return {
        "schema": _PREDICTION_CACHE_SCHEMA,
        "predictions": predicted_frame.loc[:, list(_PREDICTION_COLUMNS)].copy(),
        "first_valid_idx": first_valid_idx,
        "model_stats": model_stats,
        "sub_model_stats": sub_model_stats,
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
    for column in _PREDICTION_COLUMNS:
        frame[column] = predictions[column]
    frame = frame.loc[payload["first_valid_idx"] :].copy()
    model_stats = payload["model_stats"]
    sub_model_stats = payload["sub_model_stats"]
    time_regions = payload["time_regions"]
    logger.info(
        "Model time regions | train=%s | valid=%s | test=%s | OOD=%s",
        time_regions["train"]["start"],
        time_regions["valid"]["start"],
        time_regions["test"]["start"],
        time_regions["ood"]["start"],
    )
    logger.info(f"Backtest range: {frame['open_time_date_utc'].min()} " f"to {frame['open_time_date_utc'].max()}")
    return frame, model_stats, sub_model_stats, pre_para, time_regions


def load_backtest_prediction_frame(
    logger,
    data_config: ModelDataConfig,
    period: str,
) -> pd.DataFrame:
    """Return timestamped predictions produced by the backtest inference path."""

    train_cfg = _load_train_config(data_config.train_output_dir)
    frame, _, _, _, _ = _load_model_data(
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
    return frame, {}, {}, None, None


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


def _write_json_file(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as output:
        json.dump(
            payload,
            output,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )


def main(logger: logging.Logger, config: RunnerConfig, period: str):
    """Standalone runner: load data/model, execute the strategy and build the report."""
    period = _validate_period(period)
    if isinstance(config.data_config, ModelDataConfig):
        preparation_manifest = common.load_data_manifest_from_dir(
            config.data_config.prep_output_dir,
        )
        data_manifest = preparation_manifest["source"]
        saved_train_config = _load_train_config(config.data_config.train_output_dir)
        frame, model_stats, sub_model_stats, pre_para, time_regions = _load_model_data(
            logger,
            config.data_config,
            saved_train_config,
            period,
        )
    elif isinstance(config.data_config, CsvDataConfig):
        data_manifest = None
        saved_train_config = None
        frame, model_stats, sub_model_stats, pre_para, time_regions = _load_csv_data(
            logger,
            config.data_config,
        )
    else:
        raise TypeError(f"Unsupported data config: {type(config.data_config).__name__}")

    strategy_config = strategy_config_for_preparation(
        config.strategy_config,
        pre_para,
    )
    venue_cls = _venue_for_strategy(strategy_config)
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
    logger.info(f"Starting backtest | data={type(config.data_config).__name__} " f"| venue={venue_cls.__name__}")
    strat = cerebro.run()[0]
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
    )

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

    # Persistence: longest consecutive run of "bad" / "good" windows (in window order)
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
):
    """
    Fixed report generator:
    1. Corrected the profit factor formula (Gross Won / Gross Lost)
    2. Corrected the PnL% extraction (handles backtrader's nested list structure)

    sub_model_stats: per sub-model metrics for composite tasks (TRIGGER_DIR/LONG_SHORT_OVR)
    (see ModelHandler.evaluate_sub_models); an empty dict for single model tasks.
    """
    model_stats = model_stats or {}
    sub_model_stats = sub_model_stats or {}
    period = _validate_period(period)
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
        params_hash = task_identity.full_hash
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
    train_output_dir = os.path.join(common.TRAIN_OUT_DIR, train_config.TrainTask.TRIGGER_DIRECTION)
    start_time = time.time()
    pre_para: BaseDefine = common.load_pre_params_from_dir(train_output_dir)
    exp_dir = common.create_experiment_dir(os.path.join(common.PERSISTENCE_DIR, "simulation"), pre_para.symbol, pre_para.interval)
    logger, _ = common.setup_session_logger(log_file_path=os.path.join(exp_dir, "experiment.log"), console_level=logging.INFO, file_level=logging.INFO)
    strategy_config = BbmStrategyConfig(fixed_hold_bars=60*24)#int(pre_para.predict_num))
    broker_config = BrokerConfig()
    runner_config = RunnerConfig(
        strategy_config=strategy_config,
        broker_config=broker_config,
        save_dir=exp_dir,
        experiment_context=ExperimentContext(git_commit=common.git_revision()),
        data_config=ModelDataConfig(prep_output_dir=common.DATA_OUT_DIR, train_output_dir=train_output_dir, device="auto"),
    )
    result = main(logger, runner_config, "all")
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
