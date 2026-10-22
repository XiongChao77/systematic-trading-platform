from enum import IntEnum, Enum
from functools import lru_cache
from dataclasses import dataclass
import hashlib, logging, math, re, git
import pandas as pd
import numpy as np
import os, colorlog, logging, json, platform
from dataclasses import asdict, is_dataclass, fields, replace
from typing import Literal, Optional, Union
from datetime import datetime
from data_process.utils import *
from data_process.feature import *
from numba import njit


class Signal(IntEnum):
    INVALID = -1
    NEGATIVE = 0
    NEUTRAL = 1
    POSITIVE = 2


eps = 1e-8
# Volatility multiplier guidance (typically 0.5 ~ 1.0)
"""
Multiplier, sigma threshold, interpretation
VOL_MULTIPLIER=1.0, 1σ, ~31.8% of price moves exceed this threshold (two tails).
VOL_MULTIPLIER=0.5, 0.5σ, ~61.7% of price moves exceed this threshold; moderate signal frequency.
VOL_MULTIPLIER=1.5, 1.5σ, only ~13.4% of price moves exceed this threshold.
VOL_MULTIPLIER=2.0, 2σ, only ~4.6% of price moves exceed this threshold.
"""
DATA_PROCESS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.environ.get(
    "FINANCIAL_ML_ORIGINAL_PROJECT_DIR",
    os.path.dirname(DATA_PROCESS_DIR),
)
TEMPORARY_DIR = os.path.join(PROJECT_DIR, "output")
if platform.system().lower() != "windows":
    os.makedirs("/dev/shm/quant", exist_ok=True)
    if not os.path.islink(TEMPORARY_DIR):
        os.symlink("/dev/shm/quant", TEMPORARY_DIR)  # Linux/Ubuntu: map temporary output to shared memory
else:
    os.makedirs(TEMPORARY_DIR, exist_ok=True)
PERSISTENCE_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "quant_output")
os.makedirs(PERSISTENCE_DIR, exist_ok=True)
DATA_OUT_DIR = os.path.join(TEMPORARY_DIR, "data")
os.makedirs(DATA_OUT_DIR, exist_ok=True)


@dataclass
class MarketDataSourceConfig:
    """Fields that uniquely locate one raw market-data file."""

    market_category: str = "Cryptocurrency"  # cryptocurrency / Stock / Forex
    data_source: str = "binance_public_data"  # binance / yahoo / dukascopy
    symbol: str = "DOGEUSDT"  # BTCUSDT ETHUSDT DOGEUSDT XAUUSD
    interval: str = "30m"
    trading_type: str = "um"  # spot  / um(USDT-M Futures) / cm    (Coin-M Futures)


@dataclass
class BaseDefine(MarketDataSourceConfig):
    # label
    label_type: str = "FTHL"  # FTHL / TBM / TBM_TREND / BBM
    # model / data
    vol_ewma_span: int = 20
    predict_num: int = 32
    # risk / vol
    vol_multiplier_long: float = 1.7
    stop_multiplier_rate_long: Optional[float] = None
    vol_multiplier_short: float = 1.7
    stop_multiplier_rate_short: Optional[float] = None
    tbm_take_profit_price: Optional[Literal["close", "high_low"]] = None
    min_expected_move_pct: float = 0.01
    version: float = 0.1

    def __post_init__(self):
        if self.label_type == "PURE_BINARY":
            raise ValueError("PURE_BINARY requires PureBinaryDefine")


@dataclass
class PureBinaryDefine(MarketDataSourceConfig):
    """Symmetric first-hit labels without volatility or prediction-window settings."""

    label_type: Literal["PURE_BINARY"] = "PURE_BINARY"
    pure_binary_threshold_pct: float = 0.01  # Fractional return: 0.01 means 1% on both sides.
    version: float = 0.1

    def __post_init__(self):
        if self.label_type != "PURE_BINARY":
            raise ValueError("PureBinaryDefine only supports PURE_BINARY")


PreparationConfig = Union[BaseDefine, PureBinaryDefine]


def preparation_config_from_dict(parameters: dict) -> PreparationConfig:
    """Restore the preparation type identified by its serialized label type."""
    config_type = PureBinaryDefine if parameters.get("label_type") == "PURE_BINARY" else BaseDefine
    return config_type(**parameters)


vol_multiplier = 1.8
stop_multiplier_rate = 0.4
DOGE_1m = BaseDefine(
    market_category="Cryptocurrency",
    data_source="binance_public_data",
    symbol="DOGEUSDT",
    interval="1m",
    trading_type="um",
    label_type="FTHL",
    predict_num=16,
    vol_ewma_span=80,
    vol_multiplier_long=vol_multiplier,
    stop_multiplier_rate_long=stop_multiplier_rate,
    vol_multiplier_short=vol_multiplier,
    stop_multiplier_rate_short=stop_multiplier_rate,
)
DOGE_5m = replace(
    DOGE_1m,
    interval="5m",
)
DOGE_15m = replace(
    DOGE_1m,
    interval="15m",
)
DOGE_30m = replace(
    DOGE_1m,
    interval="30m",
)
DOGE_1h = replace(
    DOGE_1m,
    interval="1h",
)
XLM_30m = BaseDefine(
    market_category="Cryptocurrency",
    data_source="binance_public_data",
    symbol="XLMUSDT",
    interval="30m",
    trading_type="um",
    label_type="FTHL",
    predict_num=16,
    vol_ewma_span=80,
    vol_multiplier_long=1.7,
    stop_multiplier_rate_long=None,
    vol_multiplier_short=1.7,
    stop_multiplier_rate_short=None,
)
XAU_15m = BaseDefine(
    market_category="Forex",
    data_source="dukascopy",
    symbol="XAUUSD",
    interval="15m",
    trading_type="spot",
    label_type="FTHL",
    vol_ewma_span=80,
    predict_num=16,
    vol_multiplier_long=1.7,
    stop_multiplier_rate_long=None,
    vol_multiplier_short=1.7,
    stop_multiplier_rate_short=None,
)

log_level = logging.INFO

PROJECT_DATA_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "QuantData")


def market_data_path(
    source: MarketDataSourceConfig,
    root_dir: Optional[str] = None,
) -> str:
    """Resolve the canonical raw CSV path for a market data source."""
    root_dir = PROJECT_DATA_DIR if root_dir is None else root_dir
    return os.path.join(
        root_dir,
        source.market_category,
        source.data_source,
        source.trading_type,
        f"{source.symbol}_{source.interval}.csv",
    )


train_data_path = os.path.join(DATA_OUT_DIR, "train_data.csv")
test_data_path = os.path.join(DATA_OUT_DIR, "test_data.csv")
data_config_path = os.path.join(DATA_OUT_DIR, "data_config_meta.json")
TRAIN_OUT_DIR = os.path.join(TEMPORARY_DIR, "train")
os.makedirs(TRAIN_OUT_DIR, exist_ok=True)
EXPERIMENT_DIR = os.path.join(PROJECT_DATA_DIR, "experiment")
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

CONF_DF = "to_feather"  # /'to_feather'/'to_csv'


# ---------- Per-directory read/write (for batch multiprocessing: each preparation uses its own directory) ----------
def _data_path_in_dir(base_dir: str, name: str) -> str:
    return os.path.join(base_dir, name)

    # return os.path.join('/home/chao/work/Quant/data_process/label_viewer/public', name)


def save_train_df_to_dir(df: pd.DataFrame, base_dir: str) -> None:
    os.makedirs(base_dir, exist_ok=True)
    path = _data_path_in_dir(base_dir, "train_data.csv" if CONF_DF == "to_csv" else "train_data.feather")
    if os.path.exists(path):
        os.remove(path)
    if CONF_DF == "to_csv":
        df.to_csv(path, index=False, encoding="utf-8")
    else:
        df.columns = df.columns.astype(str)
        df.to_feather(path)


def save_test_df_to_dir(df: pd.DataFrame, base_dir: str) -> None:
    os.makedirs(base_dir, exist_ok=True)
    path = _data_path_in_dir(base_dir, "test_data.csv" if CONF_DF == "to_csv" else "test_data.feather")
    if os.path.exists(path):
        os.remove(path)
    if CONF_DF == "to_csv":
        df.to_csv(path, index=False, encoding="utf-8")
    else:
        df.columns = df.columns.astype(str)
        df.to_feather(path)


def load_train_df_from_dir(base_dir: str) -> pd.DataFrame:
    path = _data_path_in_dir(base_dir, "train_data.csv" if CONF_DF == "to_csv" else "train_data.feather")
    if CONF_DF == "to_csv":
        return pd.read_csv(path, encoding="utf-8")
    return pd.read_feather(path)


def load_test_df_from_dir(base_dir: str) -> pd.DataFrame:
    path = _data_path_in_dir(base_dir, "test_data.csv" if CONF_DF == "to_csv" else "test_data.feather")
    if CONF_DF == "to_csv":
        return pd.read_csv(path, encoding="utf-8")
    return pd.read_feather(path)


def get_data_config_path_in_dir(base_dir: str) -> str:
    return _data_path_in_dir(base_dir, "data_config_meta.json")


DATA_MANIFEST_FILENAME = "data_manifest.json"


def get_data_manifest_path_in_dir(base_dir: str) -> str:
    return _data_path_in_dir(base_dir, DATA_MANIFEST_FILENAME)


def get_train_data_path_in_dir(base_dir: str) -> str:
    return _data_path_in_dir(
        base_dir,
        "train_data.csv" if CONF_DF == "to_csv" else "train_data.feather",
    )


def get_test_data_path_in_dir(base_dir: str) -> str:
    return _data_path_in_dir(
        base_dir,
        "test_data.csv" if CONF_DF == "to_csv" else "test_data.feather",
    )


def sha256_file(path: str, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def data_manifest_id(payload: dict) -> str:
    canonical = json.dumps(
        json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_data_manifest_from_dir(base_dir: str) -> dict:
    """Load provenance recorded by preparation without validating its contents."""
    manifest_path = get_data_manifest_path_in_dir(base_dir)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pre_params_from_dir(base_dir: str) -> PreparationConfig:
    """Load interval settings from data_config_meta.json under base_dir (no global paths; multiprocessing-friendly)."""
    config_path = get_data_config_path_in_dir(base_dir)
    if not os.path.exists(config_path):
        raise RuntimeError(f"❌ Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        para = preparation_config_from_dict(meta)
    return para


def attach_attr(df, feature_group_list, feature_conf_list=[], para=BaseDefine):
    # 1. Basic preprocessing
    # df.drop('ignore', axis=1, inplace=True)
    # --- 2. Indicator computation (generate raw, unscaled feature columns) ---
    # df = add_relative_features(df)
    kline_interval_ms = get_interval_ms(para.interval)
    return FeatureFactory(kline_interval_ms, feature_group_list, feature_conf_list).generate(df)


# for Forex
def add_bars_to_gap(
    df,
    interval_ms,
    time_col="open_time_ms_utc",
    col_name="bars_to_close",
):
    df = df.sort_values(time_col).reset_index(drop=True).copy()

    t = df[time_col].to_numpy(np.int64)
    n = len(df)

    if n == 0:
        df[col_name] = np.array([], dtype=float)
        return df

    # 1. Turn it into a time series numbered by interval
    time_seq = t // interval_ms

    # 2. A difference other than 1 between neighbours means a gap follows the current bar
    gap_after = np.zeros(n, dtype=bool)
    gap_after[:-1] = np.diff(time_seq) != 1

    # 3. Find the closest gap bar index after every position
    idx = np.arange(n)
    gap_idx_marker = np.where(gap_after, idx, np.inf)

    next_gap_idx = np.minimum.accumulate(gap_idx_marker[::-1])[::-1]

    # 4. Distance from the current bar to the gap bar; the gap bar itself is 1
    bars_to_close = next_gap_idx - idx + 1

    df[col_name] = bars_to_close

    return df


def attch_open_time_sn(para: MarketDataSourceConfig, df):
    interval_ms = get_interval_ms(para.interval)
    if para.market_category == "Forex":
        df = add_bars_to_gap(df, interval_ms)
        df["open_time_sn"] = np.arange(len(df), dtype=np.int64)
    else:
        df["bars_to_close"] = np.inf
        df["open_time_sn"] = df["open_time_ms_utc"] // interval_ms
    return df


def attach_label(df, para=BaseDefine, label_col="label"):
    df = attch_open_time_sn(para, df)
    if isinstance(para, PureBinaryDefine):
        return attach_pure_binary_label(df, para=para, label_col=label_col)
    if para.label_type == "FTHL":
        df = attach_fthl_label(df, para=para, label_col=label_col)
    elif para.label_type == "TBM":
        df = attach_triple_barrier_label(df, para=para, label_col=label_col)
    elif para.label_type == "TBM_TREND":
        df = attach_triple_barrier_trend_label(df, para=para, label_col=label_col)
    elif para.label_type == "BBM":
        df = attach_binary_barrier_label(df, para=para, label_col=label_col, verbose=True)
    return df


def attach_fthl_label(df, para=BaseDefine, label_col="label"):
    """
    Path-dependent asymmetric labeling logic.
    """
    time_col = "open_time_ms_utc"
    time_values = df[time_col].values

    # 1. Compute asymmetric dynamic thresholds
    df = calculate_thresholds(df, para)

    # 2. Physical time anchoring (unchanged)
    interval_ms = get_interval_ms(para.interval)
    target_times = time_values + (para.predict_num * interval_ms)
    target_indices = np.searchsorted(time_values, target_times, side="left")
    in_bounds = target_indices < len(df)
    safe_idx = np.where(in_bounds, target_indices, 0)
    final_valid_mask = in_bounds & (time_values[safe_idx] == target_times)

    # 3. Compute forward return and extreme moves (unchanged)
    future_close = np.where(final_valid_mask, df["close"].values[safe_idx], np.nan)
    pct_final = np.log(future_close / df["close"])

    high_mtx = np.column_stack([df["high"].shift(-i).values for i in range(1, para.predict_num + 1)])
    low_mtx = np.column_stack([df["low"].shift(-i).values for i in range(1, para.predict_num + 1)])

    steps = (target_indices - np.arange(len(df))).clip(1, para.predict_num)
    future_high_max = np.maximum.accumulate(high_mtx, axis=1)[np.arange(len(df)), steps - 1]
    future_low_min = np.minimum.accumulate(low_mtx, axis=1)[np.arange(len(df)), steps - 1]

    max_drawdown = (future_low_min - df["close"]) / df["close"]
    max_runup = (future_high_max - df["close"]) / df["close"]

    # 4. Apply asymmetric logic
    # Long: use long-side thresholds
    cond_long = final_valid_mask & (pct_final > df["threshold_long"]) & (max_drawdown > -df["stop_threshold_long"])

    # Short: use short-side thresholds
    cond_short = final_valid_mask & (pct_final < -df["threshold_short"]) & (max_runup < df["stop_threshold_short"])

    # 5. Build labels
    conditions = [~final_valid_mask, cond_short, cond_long]
    choices = [Signal.INVALID, Signal.NEGATIVE, Signal.POSITIVE]
    df[label_col] = np.select(conditions, choices, default=Signal.NEUTRAL).astype(int)

    # volatility normalized return
    df["trend_strength"] = np.where(pct_final >= 0, pct_final / (df["threshold_long"] + eps), np.abs(pct_final) / (df["threshold_short"] + eps))

    # Handle invalid rows (out-of-bounds in physical time)
    df.loc[~final_valid_mask, "trend_strength"] = np.nan

    return df


def calculate_thresholds(df, para=BaseDefine, **kwargs):
    """
    Compute dynamic volatility thresholds using Rogers–Satchell + EWMA.
    """

    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    # ===== 1️⃣ Rogers–Satchell single-period variance =====
    log_ho = np.log(df["high"] / df["open"])
    log_hc = np.log(df["high"] / df["close"])
    log_lo = np.log(df["low"] / df["open"])
    log_lc = np.log(df["low"] / df["close"])

    rs_var = log_hc * log_ho + log_lc * log_lo

    # Guard against tiny negatives due to numeric errors (variance should be non-negative)
    rs_var = rs_var.clip(lower=0)

    # ===== 2️⃣ EWMA-smoothed variance =====
    span = para.vol_ewma_span
    ewma_var = rs_var.ewm(span=span, adjust=False).mean()
    # Sqrt -> volatility
    expected_vol = np.sqrt(ewma_var)
    # ===== 3️⃣ Save the current one-bar expected volatility =====
    df["expected_vol"] = expected_vol

    # ===== 4️⃣ Asymmetric thresholds =====
    df["threshold_long"] = expected_vol * para.vol_multiplier_long
    df["threshold_short"] = expected_vol * para.vol_multiplier_short

    if para.stop_multiplier_rate_long is not None:
        df["stop_threshold_long"] = df["threshold_long"] * para.stop_multiplier_rate_long
    else:
        df["stop_threshold_long"] = np.inf

    if para.stop_multiplier_rate_short is not None:
        df["stop_threshold_short"] = df["threshold_short"] * para.stop_multiplier_rate_short
    else:
        df["stop_threshold_short"] = np.inf

    return df


def print_zret_statistics(df, label_col="label"):
    print("\n================ trend_strength Statistics ================\n")

    valid = df["trend_strength"].notna()

    overall = df.loc[valid, "trend_strength"]

    print("Overall trend_strength distribution:")
    print(overall.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

    print("\nBy label:")

    for label in sorted(df[label_col].unique()):
        sub = df.loc[(df[label_col] == label) & valid, "trend_strength"]

        if len(sub) == 0:
            continue

        print(f"\nLabel {label}  count={len(sub)}")
        print(sub.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))


def _tbm_take_profit_uses_high_low(para) -> bool:
    mode = para.tbm_take_profit_price
    if mode is None:
        raise ValueError("tbm_take_profit_price is required when label_type is " f"{para.label_type!r}; use 'close' or 'high_low'")
    if mode == "close":
        return False
    if mode == "high_low":
        return True
    raise ValueError("tbm_take_profit_price must be 'close' or 'high_low', " f"got {mode!r}")


@njit(cache=True)
def fast_triple_barrier_kernel(
    close,
    high,
    low,
    thresholds,
    window,
    take_profit_use_high_low=False,
):
    n = len(close)
    labels = np.ones(n, dtype=np.int32)  # neutral (1) by default
    reach_times = np.full(n, window, dtype=np.int32)

    l_tp_p = thresholds[:, 0]
    l_sl_p = thresholds[:, 1]
    s_tp_p = thresholds[:, 2]
    s_sl_p = thresholds[:, 3]

    for i in range(n - window):
        p0 = close[i]

        # Independent price barriers
        l_tp = p0 * (1 + l_tp_p[i])
        l_sl = p0 * (1 - l_sl_p[i])
        s_tp = p0 * (1 - s_tp_p[i])
        s_sl = p0 * (1 + s_sl_p[i])

        # Tracks whether this sample ever hit its target inside the window
        first_l_tp = window + 1
        first_s_tp = window + 1

        # Tracks whether this side already "died" on its stop loss
        l_active = True
        s_active = True

        for j in range(1, window + 1):
            curr_idx = i + j
            h, l, c = high[curr_idx], low[curr_idx], close[curr_idx]

            # --- long path ---
            if l_active:
                long_tp_price = h if take_profit_use_high_low else c
                if long_tp_price >= l_tp:  # TP hit first
                    first_l_tp = j
                    l_active = False  # target reached, stop updating
                elif l <= l_sl:  # SL hit first
                    l_active = False  # dead

            # --- short path ---
            if s_active:
                short_tp_price = l if take_profit_use_high_low else c
                if short_tp_price <= s_tp:  # TP hit first
                    first_s_tp = j
                    s_active = False
                elif h >= s_sl:  # SL hit first
                    s_active = False

            # Leave early once both sides have a result (TP or SL)
            if not l_active and not s_active:
                break

        # --- final decision ---
        # Only a side that "won" its own path (TP before SL) may take part in the comparison
        if first_l_tp <= window and first_l_tp < first_s_tp:
            labels[i] = 2  # Signal.POSITIVE
            reach_times[i] = first_l_tp
        elif first_s_tp <= window and first_s_tp < first_l_tp:
            labels[i] = 0  # Signal.NEGATIVE
            reach_times[i] = first_s_tp
        else:
            labels[i] = 1  # Signal.NEUTRAL
            # reach_times keeps its default or is set to the time of the first SL

    return labels, reach_times


def attach_triple_barrier_label(df, para=BaseDefine, label_col="label"):
    # 1. Base thresholds
    df = calculate_thresholds(df, para)

    # 2. Prepare the underlying data
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)

    thresholds = np.column_stack(
        [df["threshold_long"].values, df["stop_threshold_long"].values, df["threshold_short"].values, df["stop_threshold_short"].values]
    ).astype(np.float64)

    window = int(para.predict_num)

    # 3. Call the numba accelerated kernel
    labels, reach_times = fast_triple_barrier_kernel(
        close,
        high,
        low,
        thresholds,
        window,
        _tbm_take_profit_uses_high_low(para),
    )

    # 4. Write the results back
    df[label_col] = labels
    df["reach_time"] = reach_times

    # 5. Physical time check (masking)
    # Makes sure the timestamp predict_num steps later lines up with physical time
    interval_ms = get_interval_ms(para.interval)
    time_values = df["open_time_ms_utc"].values
    target_times = time_values + (window * interval_ms)
    target_indices = np.searchsorted(time_values, target_times, side="left")

    in_bounds = target_indices < len(df)
    time_match = np.zeros(len(df), dtype=np.bool_)
    valid_idx = np.where(in_bounds)[0]
    time_match[valid_idx] = time_values[target_indices[valid_idx]] == target_times[valid_idx]

    # Handle the invalid rows
    df.loc[~time_match, label_col] = -1  # Signal.INVALID
    df.loc[~time_match, "reach_time"] = -1  # invalid reach time

    return df


@njit(cache=True)
def fast_triple_barrier_trend_kernel(
    close,
    high,
    low,
    thresholds,
    window,
    conflict_policy=0,
    take_profit_use_high_low=False,
):
    """
    Triple Barrier Trend Label Kernel.

    Difference from normal Triple Barrier:
    - For each anchor point i, search path i+1 ... i+window.
    - If anchor i independently hits POSITIVE/NEGATIVE at step j,
      then assign the same trend label to [i, i+j].
    - Labels propagated from a previous anchor do NOT create new anchors.
      Only the current i's own triple-barrier result can propagate further.

    conflict_policy:
        0 = first anchor wins.
            If a point was already labeled by an earlier trend segment,
            do not overwrite it.
        1 = last anchor wins.
            Later independent anchors can overwrite existing propagated labels.

    Returns:
        trend_labels:
            Final trend-expanded labels.
        anchor_labels:
            Raw triple-barrier labels at each anchor i before propagation.
        anchor_reach_times:
            Reach time of the raw anchor.
        trend_source_idx:
            Which anchor assigned the final trend label.
            -1 means no trend source.
    """
    n = len(close)

    # final expanded label
    trend_labels = np.ones(n, dtype=np.int32)  # default NEUTRAL = 1

    # raw anchor result, not propagated
    anchor_labels = np.ones(n, dtype=np.int32)  # default NEUTRAL = 1
    anchor_reach_times = np.full(n, window, dtype=np.int32)

    # source index of propagated trend label
    trend_source_idx = np.full(n, -1, dtype=np.int32)

    l_tp_p = thresholds[:, 0]
    l_sl_p = thresholds[:, 1]
    s_tp_p = thresholds[:, 2]
    s_sl_p = thresholds[:, 3]

    for i in range(n - window):
        p0 = close[i]

        # Long-side barriers
        l_tp = p0 * (1.0 + l_tp_p[i])
        l_sl = p0 * (1.0 - l_sl_p[i])

        # Short-side barriers
        s_tp = p0 * (1.0 - s_tp_p[i])
        s_sl = p0 * (1.0 + s_sl_p[i])

        first_l_tp = window + 1
        first_s_tp = window + 1

        l_active = True
        s_active = True

        # ===== 1. Raw triple barrier path search =====
        for j in range(1, window + 1):
            curr_idx = i + j

            h = high[curr_idx]
            l = low[curr_idx]
            c = close[curr_idx]

            # Long path
            if l_active:
                long_tp_price = h if take_profit_use_high_low else c
                if long_tp_price >= l_tp:
                    first_l_tp = j
                    l_active = False
                elif l <= l_sl:
                    l_active = False

            # Short path
            if s_active:
                short_tp_price = l if take_profit_use_high_low else c
                if short_tp_price <= s_tp:
                    first_s_tp = j
                    s_active = False
                elif h >= s_sl:
                    s_active = False

            if not l_active and not s_active:
                break

        # ===== 2. Decide raw anchor label =====
        out_label = 1
        reach_time = window

        if first_l_tp <= window and first_l_tp < first_s_tp:
            out_label = 2  # POSITIVE
            reach_time = first_l_tp

        elif first_s_tp <= window and first_s_tp < first_l_tp:
            out_label = 0  # NEGATIVE
            reach_time = first_s_tp

        else:
            out_label = 1  # NEUTRAL
            reach_time = window

        anchor_labels[i] = out_label
        anchor_reach_times[i] = reach_time

        # ===== 3. Trend propagation =====
        # Only POSITIVE / NEGATIVE anchors can propagate.
        # Propagated labels themselves do not create new anchors.
        if out_label == 0 or out_label == 2:
            end_idx = i + reach_time
            if end_idx >= n:
                end_idx = n - 1

            for k in range(i, end_idx + 1):
                if conflict_policy == 0:
                    # first anchor wins
                    if trend_source_idx[k] == -1:
                        trend_labels[k] = out_label
                        trend_source_idx[k] = i
                else:
                    # last anchor wins
                    trend_labels[k] = out_label
                    trend_source_idx[k] = i

    return trend_labels, anchor_labels, anchor_reach_times, trend_source_idx


def attach_triple_barrier_trend_label(
    df,
    para=BaseDefine,
    label_col="label",
    conflict_policy="first",
):
    """
    Attach Triple Barrier Trend labels.

    This label is different from normal triple barrier:

    Normal TBM:
        label[i] only describes whether point i itself hits TP/SL
        within predict_num bars.

    Trend TBM:
        if point A independently hits POSITIVE/NEGATIVE at time B,
        then labels A...B are assigned the same direction.

    Important:
        A propagated label at time B does NOT become a new signal source.
        Only B's own independent triple-barrier result can propagate further.

    conflict_policy:
        "first":
            Earlier trend segment has priority.
            This preserves the statement: once A->B is judged as one trend,
            labels from A to B stay the same.
        "last":
            Later independent anchors can overwrite earlier propagated labels.
            This is more reactive, but less strict.
    """

    df = df.copy()

    # 1. Calculate thresholds using the same logic as normal TBM
    df = calculate_thresholds(df, para)

    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)

    thresholds = np.column_stack(
        [
            df["threshold_long"].values,
            df["stop_threshold_long"].values,
            df["threshold_short"].values,
            df["stop_threshold_short"].values,
        ]
    ).astype(np.float64)

    window = int(para.predict_num)

    if conflict_policy == "first":
        conflict_code = 0
    elif conflict_policy == "last":
        conflict_code = 1
    else:
        raise ValueError(f"Unsupported conflict_policy={conflict_policy}. " "Use 'first' or 'last'.")

    # 2. Run trend kernel
    trend_labels, anchor_labels, anchor_reach_times, trend_source_idx = fast_triple_barrier_trend_kernel(
        close=close,
        high=high,
        low=low,
        thresholds=thresholds,
        window=window,
        conflict_policy=conflict_code,
        take_profit_use_high_low=_tbm_take_profit_uses_high_low(para),
    )

    # 3. Physical time alignment check
    interval_ms = get_interval_ms(para.interval)
    time_values = df["open_time_ms_utc"].values

    target_times = time_values + (window * interval_ms)
    target_indices = np.searchsorted(time_values, target_times, side="left")

    in_bounds = target_indices < len(df)

    time_match = np.zeros(len(df), dtype=np.bool_)
    valid_idx = np.where(in_bounds)[0]
    time_match[valid_idx] = time_values[target_indices[valid_idx]] == target_times[valid_idx]

    # 4. Apply invalid mask
    trend_labels[~time_match] = int(Signal.INVALID)
    anchor_labels[~time_match] = int(Signal.INVALID)
    anchor_reach_times[~time_match] = -1
    trend_source_idx[~time_match] = -1

    # 5. Save output columns
    df[label_col] = trend_labels.astype(int)

    # raw independent TBM result before trend propagation
    df["tb_anchor_label"] = anchor_labels.astype(int)
    df["tb_anchor_reach_time"] = anchor_reach_times.astype(int)

    # propagated trend source
    df["trend_source_idx"] = trend_source_idx.astype(int)

    # True means this row itself independently triggered TBM direction.
    df["is_tb_anchor"] = (df["tb_anchor_label"] == int(Signal.POSITIVE)) | (df["tb_anchor_label"] == int(Signal.NEGATIVE))

    # True means this row's final label comes from a previous anchor,
    # not from itself.
    df["is_trend_propagated"] = (df["trend_source_idx"] >= 0) & (df["trend_source_idx"] != np.arange(len(df)))

    return df


@njit(cache=True)
def fast_binary_barrier_kernel(
    close,
    high,
    low,
    time_sn,
    tp_long,
    tp_short,
    window,
    min_expected_move_pct=0.0,
):
    """
    Binary Barrier kernel: a simplified triple barrier with take-profit only
    and a fixed time barrier.

    For each anchor i, walk forward bar by bar until one of the two
    take-profit barriers is touched:
        upper = close[i] * (1 + tp_long[i])
        lower = close[i] * (1 - tp_short[i])

    - upper touched first  -> POSITIVE (2)
    - lower touched first  -> NEGATIVE (0)
    - both touched inside the same bar (order unknown) -> INVALID (-1)
    - no touch within ``window`` bars -> use the terminal close versus the
      anchor close to choose POSITIVE / NEGATIVE
    - equal terminal and anchor closes -> INVALID (-1)
    - insufficient future data / a time gap -> INVALID (-1)

    Returns:
        labels:      per-anchor label
        reach_times: number of bars from the anchor to the touching bar
                     (-1 when INVALID)
        scan_lens:   number of bars actually examined before stopping.
                     For INVALID rows this is how far the scan got.
        invalid_reasons: 0 = valid, 1 = both barriers hit in the same bar,
                     2 = incomplete window (data end / time gap),
                     3 = a barrier is below min_expected_move_pct,
                     4 = terminal close equals anchor close
    """
    n = len(close)

    labels = np.full(n, -1, dtype=np.int32)  # INVALID by default
    reach_times = np.full(n, -1, dtype=np.int32)
    scan_lens = np.zeros(n, dtype=np.int32)
    invalid_reasons = np.full(n, 2, dtype=np.int32)  # exhausted by default

    for i in range(n):
        if tp_long[i] < min_expected_move_pct or tp_short[i] < min_expected_move_pct:
            invalid_reasons[i] = 3
            continue

        p0 = close[i]

        upper = p0 * (1.0 + tp_long[i])
        lower = p0 * (1.0 - tp_short[i])

        limit = n - 1 - i
        if window < limit:
            limit = window

        base_sn = time_sn[i]

        for j in range(1, limit + 1):
            curr_idx = i + j

            # Contiguity check: a time gap makes the forward path unusable
            if time_sn[curr_idx] - base_sn != j:
                break

            scan_lens[i] = j

            hit_up = high[curr_idx] >= upper
            hit_dn = low[curr_idx] <= lower

            if hit_up and hit_dn:
                # Same bar touches both sides, order cannot be resolved
                labels[i] = -1  # INVALID
                reach_times[i] = -1
                invalid_reasons[i] = 1
                break
            elif hit_up:
                labels[i] = 2  # POSITIVE
                reach_times[i] = j
                invalid_reasons[i] = 0
                break
            elif hit_dn:
                labels[i] = 0  # NEGATIVE
                reach_times[i] = j
                invalid_reasons[i] = 0
                break

        # A complete window without a barrier touch is still a valid binary
        # target: classify it by the close at the fixed time barrier.  Do not
        # apply this fallback to truncated or discontinuous windows.
        if invalid_reasons[i] == 2 and scan_lens[i] == window:
            terminal_close = close[i + window]
            if terminal_close > p0:
                labels[i] = 2  # POSITIVE
                reach_times[i] = window
                invalid_reasons[i] = 0
            elif terminal_close < p0:
                labels[i] = 0  # NEGATIVE
                reach_times[i] = window
                invalid_reasons[i] = 0
            else:
                invalid_reasons[i] = 4

    return labels, reach_times, scan_lens, invalid_reasons


@njit(cache=True)
def fast_pure_binary_kernel(close, high, low, threshold_pct):
    """Find the first future bar reaching either configured inclusive barrier.

    Scan from the next row through the data end, without a time limit or
    time-gap restriction. Same-bar dual hits and no hits are INVALID (-1).
    Valid labels are POSITIVE (2) and NEGATIVE (0). Invalid reason codes
    are 0 for valid, 1 for same-bar dual hits, and 2 for no future hit.
    """
    n = len(close)
    labels = np.full(n, -1, dtype=np.int32)
    reach_times = np.full(n, -1, dtype=np.int32)
    scan_lens = np.zeros(n, dtype=np.int32)
    invalid_reasons = np.full(n, 2, dtype=np.int32)

    for i in range(n):
        upper = close[i] * (1.0 + threshold_pct)
        lower = close[i] * (1.0 - threshold_pct)
        for curr_idx in range(i + 1, n):
            scan_lens[i] = curr_idx - i
            hit_up = high[curr_idx] >= upper
            hit_down = low[curr_idx] <= lower
            if hit_up and hit_down:
                invalid_reasons[i] = 1
                break
            if hit_up or hit_down:
                labels[i] = 2 if hit_up else 0
                reach_times[i] = curr_idx - i
                invalid_reasons[i] = 0
                break

    return labels, reach_times, scan_lens, invalid_reasons


def attach_pure_binary_label(df, para: Optional[PureBinaryDefine] = None, label_col="label", verbose=True):
    """Attach first-hit labels using configured symmetric return barriers.

    pure_binary_threshold_pct is a fractional return (0.01 means 1%).
    Neither predict_num nor volatility thresholds limit the search. Input
    rows must be ordered chronologically. The anchor bar itself is excluded.
    """
    if para is None:
        para = PureBinaryDefine()
    if not isinstance(para, PureBinaryDefine):
        raise TypeError("Pure binary labeling requires PureBinaryDefine")
    threshold_pct = float(para.pure_binary_threshold_pct)
    if not np.isfinite(threshold_pct) or not 0.0 < threshold_pct < 1.0:
        raise ValueError("pure_binary_threshold_pct must be finite and between 0 and 1 (exclusive)")
    df = df.copy()
    labels, reach_times, scan_lens, invalid_reasons = fast_pure_binary_kernel(
        df["close"].to_numpy(dtype=np.float64),
        df["high"].to_numpy(dtype=np.float64),
        df["low"].to_numpy(dtype=np.float64),
        threshold_pct,
    )
    df[label_col] = labels.astype(int)
    df["reach_time"] = reach_times.astype(int)
    df["scan_len"] = scan_lens.astype(int)
    df["invalid_reason"] = invalid_reasons.astype(int)
    if verbose:
        print(f"PURE_BINARY inclusive return threshold: {threshold_pct:.4%}")
        print_pure_binary_stats(df, label_col=label_col)
    return df


def print_pure_binary_stats(df, label_col="label"):
    """Print class shares of all samples and per-direction event-time percentiles.

    Event times count subsequent bars from the anchor, with the next bar
    counted as one. Percentiles use NumPy's default linear interpolation.
    INVALID samples are excluded from event-time statistics.
    """
    print("\n==================== PURE_BINARY Statistics ====================")
    total = len(df)
    print(f"Total Samples: {total}")
    if total == 0:
        print("Empty dataframe.\n")
        return

    labels = df[label_col].to_numpy()
    groups = [
        ("POSITIVE", labels == int(Signal.POSITIVE)),
        ("NEGATIVE", labels == int(Signal.NEGATIVE)),
        ("INVALID", labels == int(Signal.INVALID)),
    ]
    print(f"{'Label':<12} | {'Count':>10} | {'Share of all samples':>20}")
    for name, mask in groups:
        count = int(mask.sum())
        print(f"{name:<12} | {count:>10} | {count / total:>19.2%}")

    print("Reach Time Percentiles (bars after anchor; valid events only):")
    percentiles = [1, 5, 25, 50, 75, 90, 95, 99]
    for name, mask in groups[:2]:
        values = df.loc[mask, "reach_time"].to_numpy(dtype=float)
        if len(values) == 0:
            print(f"{name}: No events.")
            continue
        quantiles = np.percentile(values, percentiles)
        percentile_text = " | ".join(f"P{pct}: {value:.2f}" for pct, value in zip(percentiles, quantiles))
        print(
            f"{name}: Count: {len(values)} | Mean: {values.mean():.2f} | "
            f"Min: {values.min():.0f} | {percentile_text} | Max: {values.max():.0f}"
        )
    print("=" * 64 + "\n")


def attach_binary_barrier_label(
    df,
    para=BaseDefine,
    label_col="label",
    verbose=True,
):

    df = df.copy()

    window = int(para.predict_num)
    if window <= 0:
        raise ValueError(f"predict_num must be positive for BBM labels, got {window}")

    # The barrier distance comes from expected volatility and the directional
    # multipliers. predict_num is the fixed maximum search/time-barrier length.
    df = calculate_thresholds(df, para)

    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)

    tp_long = df["threshold_long"].values.astype(np.float64)
    tp_short = df["threshold_short"].values.astype(np.float64)

    # 2. Bar sequence numbers, used to detect physical time gaps
    time_sn = df["open_time_sn"].values.astype(np.int64)

    # 3. Run the kernel
    labels, reach_times, scan_lens, invalid_reasons = fast_binary_barrier_kernel(
        close,
        high,
        low,
        time_sn,
        tp_long,
        tp_short,
        window,
        float(para.min_expected_move_pct),
    )

    # 4. Write the results back
    df[label_col] = labels.astype(int)
    df["reach_time"] = reach_times.astype(int)
    # How many bars the forward scan actually consumed (meaningful for INVALID rows)
    df["scan_len"] = scan_lens.astype(int)
    # 0 = valid, 1 = both barriers hit in the same bar,
    # 2 = incomplete window, 3 = threshold below min_expected_move_pct,
    # 4 = terminal close unchanged
    df["invalid_reason"] = invalid_reasons.astype(int)

    if verbose:
        print_binary_barrier_stats(df, label_col=label_col)

    return df


THRESHOLD_PERCENTILES = [1, 5, 25, 50, 75, 90, 95, 99]


def log_threshold_percentiles(df, logger=None, cols=("threshold_long", "threshold_short")):
    """
    Log the percentile distribution of the dynamic thresholds.
    Falls back to print() when no logger is given.
    """
    emit = logger.info if logger is not None else print

    header = " | ".join(f"P{p}".rjust(8) for p in THRESHOLD_PERCENTILES)
    emit(f"{'Threshold':<18} | {header}")

    for col in cols:
        if col not in df.columns:
            continue
        vals = df[col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        pcts = np.percentile(vals, THRESHOLD_PERCENTILES)
        row = " | ".join(f"{v:8.4f}" for v in pcts)
        emit(f"{col:<18} | {row}")


def print_binary_barrier_stats(df, label_col="label"):
    """
    Print the Binary Barrier label distribution and reach-time percentiles
    (P50 / P90 / P95 / P99 / max) for POSITIVE / NEGATIVE / INVALID.

    For INVALID rows reach_time does not exist, so the scanned length
    (scan_len) is reported instead - it says how far the forward scan got
    before hitting an ambiguous bar or running out of data.
    """
    print("\n" + "=" * 20 + " 📊 Binary Barrier Statistics " + "=" * 20)

    total = len(df)
    if total == 0:
        print("Empty dataframe.")
        print("=" * 70 + "\n")
        return

    labels = df[label_col].values
    groups = [
        ("POSITIVE", labels == int(Signal.POSITIVE), "reach_time"),
        ("NEGATIVE", labels == int(Signal.NEGATIVE), "reach_time"),
        ("INVALID  (No/Ambig)", labels == int(Signal.INVALID), "scan_len"),
    ]

    print(f"Total Samples: {total}")
    print("-" * 70)

    # 1. Label distribution
    print(f"{'Label Type':<22} | {'Count':>10} | {'Percentage':>10}")
    for name, mask, _ in groups:
        cnt = int(mask.sum())
        print(f"{name:<22} | {cnt:>10} | {cnt / total * 100:>9.2f}%")

    # Breakdown of why rows are INVALID
    if "invalid_reason" in df.columns:
        reason = df["invalid_reason"].values
        inv_mask = labels == int(Signal.INVALID)
        same_bar = int((inv_mask & (reason == 1)).sum())
        exhausted = int((inv_mask & (reason == 2)).sum())
        below_min_move = int((inv_mask & (reason == 3)).sum())
        unchanged_close = int((inv_mask & (reason == 4)).sum())
        print(f"{'  - same-bar conflict':<22} | {same_bar:>10} | {same_bar / total * 100:>9.2f}%")
        print(f"{'  - incomplete window':<22} | {exhausted:>10} | {exhausted / total * 100:>9.2f}%")
        print(f"{'  - move below minimum':<22} | {below_min_move:>10} | {below_min_move / total * 100:>9.2f}%")
        print(f"{'  - unchanged close':<22} | {unchanged_close:>10} | {unchanged_close / total * 100:>9.2f}%")

    print("-" * 70)

    # 2. Scan-length distribution across every anchor. This makes it visible
    # whether most labels resolve early or pile up at the time barrier.
    if "scan_len" in df.columns:
        scan_len = df["scan_len"].to_numpy(dtype=float)
        scan_len = scan_len[np.isfinite(scan_len) & (scan_len >= 0)]
        print("🔎 Scan Length Percentiles (All Samples):")
        if len(scan_len):
            percentiles = np.percentile(scan_len, THRESHOLD_PERCENTILES)
            percentile_text = " | ".join(f"P{pct}: {value:.0f}" for pct, value in zip(THRESHOLD_PERCENTILES, percentiles))
            print(f"Count: {len(scan_len)} | Mean: {scan_len.mean():.2f} | " f"Min: {scan_len.min():.0f} | {percentile_text} | " f"Max: {scan_len.max():.0f}")
        else:
            print("No scan_len values.")
        print("-" * 70)

    # 3. Reach time percentiles
    print("⏱️ Reach Time Percentiles (Steps):")
    print(f"{'Group':<22} | {'Count':>8} | {'Mean':>8} | {'P50':>7} | {'P90':>7} | {'P95':>7} | {'P99':>7} | {'Max':>7}")

    for name, mask, col in groups:
        if col not in df.columns:
            continue
        vals = df.loc[mask, col].values
        vals = vals[vals >= 0]
        if len(vals) == 0:
            print(f"{name:<22} | {0:>8} | {'-':>8} | {'-':>7} | {'-':>7} | {'-':>7} | {'-':>7} | {'-':>7}")
            continue
        p50, p90, p95, p99 = np.percentile(vals, [50, 90, 95, 99])
        suffix = "" if col == "reach_time" else "  (scan_len)"
        print(f"{name:<22} | {len(vals):>8} | {vals.mean():>8.2f} | " f"{p50:>7.0f} | {p90:>7.0f} | {p95:>7.0f} | {p99:>7.0f} | {vals.max():>7}{suffix}")

    print("=" * 70 + "\n")


def print_label_performance_stats(df, para=BaseDefine):
    """
    Print the label distribution and detailed statistics of the reach time
    """
    print("\n" + "=" * 20 + " 📊 Triple Barrier Statistics " + "=" * 20)

    # 1. Basic information
    total_len = len(df)
    valid_df = df[df["label"] != -1].copy()  # drop INVALID (-1)
    predict_num = para.predict_num

    print(f"Total Samples: {total_len}")
    print(f"Valid Samples: {len(valid_df)} ({(len(valid_df)/total_len)*100:.2f}%)")
    print(f"Max Window (predict_num): {predict_num}")
    print("-" * 50)

    # 2. Label distribution
    label_counts = valid_df["label"].value_counts().sort_index()
    label_map = {0: "NEGATIVE (Short Win)", 1: "NEUTRAL (Time-out/SL)", 2: "POSITIVE (Long Win)"}

    print(f"{'Label Type':<25} | {'Count':<10} | {'Percentage':<10}")
    for lbl, count in label_counts.items():
        name = label_map.get(lbl, "Unknown")
        pct = (count / len(valid_df)) * 100
        print(f"{name:<25} | {count:<10} | {pct:>8.2f}%")

    print("-" * 50)

    # 3. Reach time statistics (non neutral labels only)
    print("⏱️ Reach Time Descriptive Statistics (Steps):")

    # Descriptive statistics of reach_time per group
    stats = valid_df.groupby("label")["reach_time"].describe(percentiles=[0.25, 0.5, 0.75, 0.9])
    # Rename the index for readability
    stats.index = stats.index.map(label_map)
    print(stats[["count", "mean", "min", "50%", "90%", "max"]])

    # 4. Efficiency analysis: fast trigger vs slow trigger
    print("\n🚀 Efficiency Analysis (Speed of Signal):")
    for lbl in [0, 2]:
        sub = valid_df[valid_df["label"] == lbl]
        if len(sub) > 0:
            name = label_map[lbl]
            # A "fast trigger" is defined as reaching the target within the first 25% of the window
            fast_threshold = predict_num * 0.25
            fast_hits = len(sub[sub["reach_time"] <= fast_threshold])
            fast_pct = (fast_hits / len(sub)) * 100

            # A "buzzer beater" is defined as reaching it only in the last 10% of the window
            slow_threshold = predict_num * 0.9
            slow_hits = len(sub[sub["reach_time"] >= slow_threshold])

            print(f"[{name}]")
            print(f"  - Fast Hits (<= {fast_threshold:.0f} steps): {fast_hits} ({fast_pct:.2f}%)")
            print(f"  - Slow Hits (>= {slow_threshold:.0f} steps): {slow_hits} ({(slow_hits/len(sub))*100:.2f}%)")
            print(f"  - Median Reach Time: {sub['reach_time'].median():.0f} steps")

    print("=" * 60 + "\n")


def attach_macd_event_lifecycle_label(
    df,
    interval_ms,
    para=BaseDefine,
):
    """
    Strict time-aligned MACD event lifecycle labels (auto-detect feature column names).
    min_threshold logic removed.
    """
    # --- 1. Auto-detect MACD feature column names ---
    dif_cols = [c for c in df.columns if c.startswith("MACD_") and c.endswith("_DIF")]
    dea_cols = [c for c in df.columns if c.startswith("MACD_") and c.endswith("_DEA")]

    if not dif_cols or not dea_cols:
        raise ValueError("❌ MACD feature columns not found (expected suffixes: _DIF and _DEA)")

    dif_name = dif_cols[0]
    prefix = dif_name.replace("_DIF", "")
    dea_name = f"{prefix}_DEA"

    if dea_name not in df.columns:
        raise ValueError(f"❌ Cannot find matching DEA column for {dif_name}: {dea_name}")

    print(f"🔍 [MACD Match] Auto-detected feature columns: {dif_name} / {dea_name}")

    time_col = "open_time_ms_utc"
    time_values = df[time_col].values

    # 2. Identify crossover points
    dif = df[dif_name]
    dea = df[dea_name]
    cross_mask = (dif > dea) != (dif.shift(1) > dea.shift(1))
    cross_mask.iloc[0] = False
    event_indices = df.index[cross_mask].tolist()

    # 3. Initialize and compute dynamic thresholds
    df["label"] = Signal.INVALID
    df = calculate_thresholds(df, para)

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    thresholds = df["threshold"].values
    sl_thresholds = df["stop_threshold"].values

    # 4. Iterate crossover events
    for i in range(len(event_indices)):
        curr_idx = event_indices[i]
        if i + 1 >= len(event_indices):
            df.at[curr_idx, "label"] = Signal.INVALID
            continue

        next_idx = event_indices[i + 1]

        # --- Time alignment check ---
        expected_gap_ms = (next_idx - curr_idx) * interval_ms
        actual_gap_ms = time_values[next_idx] - time_values[curr_idx]

        if actual_gap_ms != expected_gap_ms:
            df.at[curr_idx, "label"] = Signal.INVALID
            continue

        # --- Business logic ---
        is_long_event = dif.iloc[curr_idx] > dea.iloc[curr_idx]
        entry_price = closes[curr_idx]
        tp_target = thresholds[curr_idx]
        sl_target = sl_thresholds[curr_idx]

        pnl_at_exit = (closes[next_idx] - entry_price) / entry_price if is_long_event else (entry_price - closes[next_idx]) / entry_price

        window_highs = highs[curr_idx + 1 : next_idx + 1]
        window_lows = lows[curr_idx + 1 : next_idx + 1]

        if is_long_event:
            hit_stop = np.any(window_lows <= entry_price * (1 - sl_target))
        else:
            hit_stop = np.any(window_highs >= entry_price * (1 + sl_target))

        if pnl_at_exit >= tp_target and not hit_stop:
            df.at[curr_idx, "label"] = Signal.POSITIVE if is_long_event else Signal.NEGATIVE
        else:
            df.at[curr_idx, "label"] = Signal.NEUTRAL

    return df


def attach_boll_event_lifecycle_label(
    df,
    interval_ms,
    para=BaseDefine,
):
    """
    Mean-reversion Bollinger Band lifecycle labels.
    min_threshold logic removed.
    """
    upper_cols = [c for c in df.columns if c.startswith("BOLL_UPPER_")]
    lower_cols = [c for c in df.columns if c.startswith("BOLL_LOWER_")]
    middle_cols = [c for c in df.columns if c.startswith("BOLL_MIDDLE_")]

    if not (upper_cols and lower_cols and middle_cols):
        raise ValueError("❌ Incomplete BOLL feature columns detected")

    u_name, l_name, m_name = upper_cols[0], lower_cols[0], middle_cols[0]
    print(f"🔍 [BOLL Match] Auto-detected feature columns: {u_name}, {l_name}, {m_name}")

    time_col = "open_time_ms_utc"
    time_values = df[time_col].values

    long_trigger = df["close"] < df[l_name]
    short_trigger = df["close"] > df[u_name]
    event_mask = long_trigger | short_trigger
    event_indices = df.index[event_mask].tolist()

    # 3. Initialize and compute dynamic thresholds
    df["label"] = Signal.INVALID
    df = calculate_thresholds(df, para)

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    middles = df[m_name].values
    thresholds = df["threshold"].values
    sl_thresholds = df["stop_threshold"].values

    # 4. Iterate events
    for curr_idx in event_indices:
        is_long_event = long_trigger.iloc[curr_idx]
        entry_price = closes[curr_idx]
        tp_target = thresholds[curr_idx]
        sl_limit = sl_thresholds[curr_idx]

        if is_long_event:
            exit_candidates = np.where(closes[curr_idx + 1 :] >= middles[curr_idx + 1 :])[0]
        else:
            exit_candidates = np.where(closes[curr_idx + 1 :] <= middles[curr_idx + 1 :])[0]

        if len(exit_candidates) == 0:
            df.at[curr_idx, "label"] = Signal.INVALID
            continue

        next_idx = curr_idx + 1 + exit_candidates[0]

        if (time_values[next_idx] - time_values[curr_idx]) != (next_idx - curr_idx) * interval_ms:
            df.at[curr_idx, "label"] = Signal.INVALID
            continue

        pnl_at_exit = (closes[next_idx] - entry_price) / entry_price if is_long_event else (entry_price - closes[next_idx]) / entry_price

        window_highs = highs[curr_idx + 1 : next_idx + 1]
        window_lows = lows[curr_idx + 1 : next_idx + 1]

        if is_long_event:
            hit_stop = np.any(window_lows <= entry_price * (1 - sl_limit))
        else:
            hit_stop = np.any(window_highs >= entry_price * (1 + sl_limit))

        if pnl_at_exit >= tp_target and not hit_stop:
            df.at[curr_idx, "label"] = Signal.POSITIVE if is_long_event else Signal.NEGATIVE
        else:
            df.at[curr_idx, "label"] = Signal.NEUTRAL

    _boll_audit(df, event_indices)
    return df


def _boll_audit(df, event_indices):
    total = len(event_indices)
    stats = df.loc[event_indices, "label"].value_counts()
    print(f"\n📊 [BOLL Lifecycle Audit]")
    print(f"  - Total triggers: {total}")
    print(f"  - POSITIVE  (2) valid: {stats.get(Signal.POSITIVE , 0)} ({(stats.get(Signal.POSITIVE , 0)/total)*100:.2f}%)")
    print(f"  - NEGATIVE (0) valid: {stats.get(Signal.NEGATIVE, 0)} ({(stats.get(Signal.NEGATIVE, 0)/total)*100:.2f}%)")
    print(f"  - NEUTRAL (1) noise: {stats.get(Signal.NEUTRAL, 0)}")


def attach_sma_7_25_crossover_label(
    df,
    interval_ms,
    para=BaseDefine,
):
    """
    SMA 7/25 crossover lifecycle labels.
    min_threshold logic removed.
    """
    fast_ma_name = "SMA_7B"
    slow_ma_name = "SMA_25B"

    if fast_ma_name not in df.columns or slow_ma_name not in df.columns:
        raise ValueError(f"❌ SMA columns not found: {fast_ma_name} or {slow_ma_name}. Please check FeatureMA configuration.")

    time_col = "open_time_ms_utc"
    time_values = df[time_col].values

    fast_ma = df[fast_ma_name]
    slow_ma = df[slow_ma_name]
    cross_mask = (fast_ma > slow_ma) != (fast_ma.shift(1) > slow_ma.shift(1))
    cross_mask.iloc[0] = False
    event_indices = df.index[cross_mask].tolist()

    df["label"] = Signal.INVALID
    df = calculate_thresholds(df, para)

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    thresholds = df["threshold"].values
    sl_thresholds = df["stop_threshold"].values

    for i in range(len(event_indices)):
        curr_idx = event_indices[i]
        if i + 1 >= len(event_indices):
            continue

        next_idx = event_indices[i + 1]

        expected_gap = (next_idx - curr_idx) * interval_ms
        actual_gap = time_values[next_idx] - time_values[curr_idx]

        if actual_gap != expected_gap:
            continue

        is_long_event = fast_ma.iloc[curr_idx] > slow_ma.iloc[curr_idx]
        entry_price = closes[curr_idx]

        pnl_rate = (closes[next_idx] - entry_price) / entry_price if is_long_event else (entry_price - closes[next_idx]) / entry_price

        window_highs = highs[curr_idx + 1 : next_idx + 1]
        window_lows = lows[curr_idx + 1 : next_idx + 1]

        sl_limit = sl_thresholds[curr_idx]
        if is_long_event:
            hit_stop = np.any(window_lows <= entry_price * (1 - sl_limit))
        else:
            hit_stop = np.any(window_highs >= entry_price * (1 + sl_limit))

        if pnl_rate >= thresholds[curr_idx] and not hit_stop:
            df.at[curr_idx, "label"] = Signal.POSITIVE if is_long_event else Signal.NEGATIVE
        else:
            df.at[curr_idx, "label"] = Signal.NEUTRAL

    return df


def clean_data_quality_auto(df: pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("Starting automated data quality scan...")
    initial_count = len(df)
    na_rows = df.isna().any(axis=1).sum()
    if na_rows > 0:
        logger.warning(f"Detected {na_rows} rows containing NaN values; dropping them.")

    zero_check_cols = [col for col in ["open", "close", "high", "low"] if col in df.columns]
    if zero_check_cols:
        zero_mask = (df[zero_check_cols] == 0).any(axis=1)
        zero_rows = zero_mask.sum()
    else:
        zero_mask = pd.Series(False, index=df.index)
        zero_rows = 0
        logger.warning("OHLC columns not found; skipping zero-value check.")

    if zero_rows > 0:
        zero_stats = (df[zero_check_cols] == 0).sum()
        logger.warning(f"Detected {zero_rows} rows containing zero values in OHLC columns {zero_check_cols}. " f"Distribution:\n{zero_stats[zero_stats > 0]}")

    condition = df.isna().any(axis=1) | zero_mask
    df_cleaned = df[~condition].copy()
    df_cleaned.reset_index(drop=True, inplace=True)

    final_count = len(df_cleaned)
    dropped_count = initial_count - final_count

    if dropped_count > 0:
        logger.info(f"✅ Cleaning done: {initial_count} rows -> {final_count} rows (dropped {dropped_count})")
    else:
        logger.info("✅ Scan complete: no NaN or zero values found.")

    return df_cleaned


def float_range(start, end, step):
    values = []
    v = start
    eps = step / 10
    while v <= end + eps:
        values.append(round(v, 10))
        v += step
    return values


@lru_cache(maxsize=1)
def load_interval_ms(config_path=data_config_path):
    if not os.path.exists(config_path):
        raise RuntimeError(f"❌ Config file not found: {config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        interval_ms = meta.get("interval_ms")
        if interval_ms is None:
            raise RuntimeError("⚠️ Missing 'interval_ms' field in config file!")
        return interval_ms
    except Exception as e:
        raise RuntimeError(f"💥 Unexpected error while reading JSON: {e}")


def setup_session_logger(
    sub_folder: str = None, log_file_path=None, symbol: str = "", console_level: int = logging.INFO, file_level: int = logging.INFO
) -> tuple[logging.Logger, str]:
    if log_file_path == None:
        assert sub_folder != None
        log_dir = os.path.join(PERSISTENCE_DIR, "log", sub_folder)
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sym_str = f"_{symbol}" if symbol else ""
        log_filename = f"session{sym_str}_{timestamp}.log"
        log_file_path = os.path.join(log_dir, log_filename)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    if root_logger.handlers:
        root_logger.handlers = []
    log_format_console = "%(log_color)s%(asctime)s.%(msecs)03d-%(name)s-%(levelname)s- %(message)s"
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    color_formatter = colorlog.ColoredFormatter(
        log_format_console,
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "RECORD": "blue",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red,bg_yellow",
        },
    )
    ch.setFormatter(color_formatter)
    root_logger.addHandler(ch)
    fh = logging.FileHandler(log_file_path, encoding="utf-8")
    fh.setLevel(file_level)
    file_formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(file_formatter)
    root_logger.addHandler(fh)
    root_logger.info(f"Session Logger Initialized. Log file: {log_file_path}")

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    return root_logger, log_file_path


def get_interval_from_filename(path: str) -> str:
    """
    Extract interval string from a file path (e.g. ETHUSDT_3m.csv -> 3m).
    """
    filename = os.path.basename(path)
    # Match formats like 1s, 15s, 1m, 3m... 1M
    match = re.search(r"_(\d+[smhdwM])\.csv", filename)
    if match:
        return match.group(1)
    return "unknown"


def get_interval_ms(interval_str: str) -> int:
    """
    Convert an interval string to milliseconds.
    Supported: 1s, 15s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    """
    # Base units in milliseconds
    units = {
        "s": 1000,
        "m": 60 * 1000,
        "h": 60 * 60 * 1000,
        "d": 24 * 60 * 60 * 1000,
        "w": 7 * 24 * 60 * 60 * 1000,
        "M": 30 * 24 * 60 * 60 * 1000,  # Approximate month as 30 days
    }

    # Split number and unit via regex
    match = re.match(r"(\d+)([smhdwM])", interval_str)
    if not match:
        return 0

    value, unit = match.groups()
    return int(value) * units[unit]


def validate_kline_source(
    df: pd.DataFrame,
    interval_ms: int,
    *,
    source: str = "kline source",
    logger=None,
) -> dict:
    """Validate that a dataframe is one non-overlapping, contiguous K-line series.

    The open-time sequence alone is not sufficient: a file can alternate hourly
    candles with half-hour candles while still having an apparently perfect 30m
    open-time cadence.  Therefore this validation also checks every candle's
    close time and adjacent time ranges.

    Returns a small validation summary.  Raises ``ValueError`` on any structural
    or OHLC error so callers do not continue with contaminated market data.
    """
    required_columns = {
        "open_time_ms_utc",
        "close_time_ms_utc",
        "open",
        "high",
        "low",
        "close",
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"K-line validation failed for {source}: missing columns {missing_columns}")
    if not isinstance(interval_ms, (int, np.integer)) or interval_ms <= 0:
        raise ValueError(f"Invalid K-line interval_ms for {source}: {interval_ms!r}")
    if df.empty:
        raise ValueError(f"K-line validation failed for {source}: data is empty")

    def format_utc_ms(value: int) -> str:
        return pd.to_datetime(value, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")

    open_time = pd.to_numeric(df["open_time_ms_utc"], errors="coerce")
    close_time = pd.to_numeric(df["close_time_ms_utc"], errors="coerce")
    invalid_timestamp = open_time.isna() | close_time.isna() | (open_time % 1 != 0) | (close_time % 1 != 0)

    errors = []
    samples = []
    invalid_timestamp_count = int(invalid_timestamp.sum())
    if invalid_timestamp_count:
        errors.append(f"invalid timestamps={invalid_timestamp_count}")
        samples.extend(
            f"row={idx} open_time={df.at[idx, 'open_time_ms_utc']!r} " f"close_time={df.at[idx, 'close_time_ms_utc']!r}"
            for idx in df.index[invalid_timestamp][:3]
        )

    summary = {
        "source": source,
        "row_count": int(len(df)),
        "interval_ms": int(interval_ms),
        "invalid_timestamp_count": invalid_timestamp_count,
        "duplicate_open_time_count": 0,
        "out_of_order_count": 0,
        "unexpected_spacing_count": 0,
        "missing_candle_count": 0,
        "invalid_duration_count": 0,
        "overlap_count": 0,
        "invalid_ohlc_count": 0,
    }

    if not invalid_timestamp_count:
        open_time = open_time.astype("int64")
        close_time = close_time.astype("int64")
        delta = open_time.diff()

        duplicate_mask = open_time.duplicated(keep=False)
        out_of_order_mask = delta < 0
        spacing_mask = delta.notna() & (delta != interval_ms)
        forward_gap_mask = delta > interval_ms
        aligned_gap_mask = forward_gap_mask & (delta % interval_ms == 0)
        duration = close_time - open_time + 1
        invalid_duration_mask = duration != interval_ms
        overlap_mask = open_time.iloc[1:].to_numpy() <= close_time.iloc[:-1].to_numpy()

        summary["duplicate_open_time_count"] = int(duplicate_mask.sum())
        summary["out_of_order_count"] = int(out_of_order_mask.sum())
        summary["unexpected_spacing_count"] = int(spacing_mask.sum())
        summary["missing_candle_count"] = int(((delta.loc[aligned_gap_mask] // interval_ms) - 1).sum())
        summary["invalid_duration_count"] = int(invalid_duration_mask.sum())
        summary["overlap_count"] = int(overlap_mask.sum())

        if summary["duplicate_open_time_count"]:
            errors.append(f"duplicate open times={summary['duplicate_open_time_count']}")
        if summary["out_of_order_count"]:
            errors.append(f"out-of-order rows={summary['out_of_order_count']}")
        if summary["unexpected_spacing_count"]:
            errors.append("unexpected open-time spacing=" f"{summary['unexpected_spacing_count']} " f"(missing candles={summary['missing_candle_count']})")
            for idx in df.index[spacing_mask][:3]:
                pos = df.index.get_loc(idx)
                previous_open = int(open_time.iloc[pos - 1])
                current_open = int(open_time.iloc[pos])
                delta_ms = int(delta.iloc[pos])
                if delta_ms > interval_ms and delta_ms % interval_ms == 0:
                    missing_count = delta_ms // interval_ms - 1
                    first_missing = previous_open + interval_ms
                    last_missing = current_open - interval_ms
                    gap_detail = f"missing_range={format_utc_ms(first_missing)} -> " f"{format_utc_ms(last_missing)} " f"missing_count={missing_count}"
                else:
                    gap_detail = "missing_range=unaligned_spacing"
                samples.append(
                    f"rows={df.index[pos - 1]}->{idx} "
                    f"previous_open={format_utc_ms(previous_open)} ({previous_open}) "
                    f"current_open={format_utc_ms(current_open)} ({current_open}) "
                    f"{gap_detail} delta_ms={delta_ms}"
                )
        if summary["invalid_duration_count"]:
            errors.append(f"invalid candle duration={summary['invalid_duration_count']}")
            for idx in df.index[invalid_duration_mask][:3]:
                samples.append(
                    f"row={idx} open_time={int(open_time.loc[idx])} " f"close_time={int(close_time.loc[idx])} " f"duration_ms={int(duration.loc[idx])}"
                )
        if summary["overlap_count"]:
            errors.append(f"overlapping candles={summary['overlap_count']}")
            for pos in np.flatnonzero(overlap_mask)[:3]:
                samples.append(
                    f"rows={df.index[pos]}->{df.index[pos + 1]} " f"previous_close={int(close_time.iloc[pos])} " f"next_open={int(open_time.iloc[pos + 1])}"
                )

    ohlc = df[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    finite_ohlc = np.isfinite(ohlc.to_numpy()).all(axis=1)
    positive_ohlc = (ohlc > 0).all(axis=1).to_numpy()
    price_bounds = (
        (ohlc["high"] >= ohlc[["open", "close"]].max(axis=1)) & (ohlc["low"] <= ohlc[["open", "close"]].min(axis=1)) & (ohlc["high"] >= ohlc["low"])
    ).to_numpy()
    invalid_ohlc_mask = ~(finite_ohlc & positive_ohlc & price_bounds)
    summary["invalid_ohlc_count"] = int(invalid_ohlc_mask.sum())
    if summary["invalid_ohlc_count"]:
        errors.append(f"invalid OHLC rows={summary['invalid_ohlc_count']}")
        for pos in np.flatnonzero(invalid_ohlc_mask)[:3]:
            idx = df.index[pos]
            samples.append(f"row={idx} o={df.at[idx, 'open']!r} h={df.at[idx, 'high']!r} " f"l={df.at[idx, 'low']!r} c={df.at[idx, 'close']!r}")

    if errors:
        message = f"K-line validation failed for {source}: " + "; ".join(errors)
        if samples:
            message += ". Samples: " + " | ".join(samples[:6])
        if logger is not None:
            logger.error(message)
        raise ValueError(message)

    if logger is not None:
        logger.info(f"K-line validation passed: source={source}, rows={len(df)}, " f"interval_ms={interval_ms}, missing_candles=0, overlaps=0")
    return summary


def git_revision(*, require_clean: bool = False) -> str:
    """Return the repository revision, optionally requiring an entirely clean tree."""
    snapshot_revision = os.environ.get("FINANCIAL_ML_SOURCE_REVISION")
    if snapshot_revision:
        return snapshot_revision
    repo = git.Repo(PROJECT_DIR)
    if require_clean and repo.is_dirty(untracked_files=True):
        status = repo.git.status("--short")
        raise RuntimeError(
            "Git working tree is not clean; commit or remove all staged, unstaged, " f"and untracked changes before starting an experiment:\n{status}"
        )
    return repo.head.object.hexsha


def build_dataclass(cls, data: dict):
    """
    Build a dataclass from a dict (supports nested dataclasses).
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue

        val = data[f.name]

        # Nested dataclass
        if is_dataclass(f.type) and isinstance(val, dict):
            kwargs[f.name] = build_dataclass(f.type, val)
        else:
            kwargs[f.name] = val

    return cls(**kwargs)


def load_parameters(path, cls):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return build_dataclass(cls, data["strategy"])


def load_common_define(path, cls):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return build_dataclass(cls, data["common"])


def create_experiment_dir(base_dir, symbol, interval, now=None):
    """
    Create experiment directory:
        base_dir / SYMBOL_INTERVAL / YYYY-MM-DD / HH_MM_SS
    Returns the final experiment directory path.
    """
    now = now or datetime.now()

    date_dir = now.strftime("%Y-%m-%d")
    time_dir = now.strftime("%H_%M_%S")
    sym_interval_dir = f"{symbol}_{interval}"

    exp_dir = os.path.join(base_dir, sym_interval_dir, date_dir, time_dir)
    os.makedirs(exp_dir, exist_ok=True)

    return exp_dir


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())  # Optional but recommended
