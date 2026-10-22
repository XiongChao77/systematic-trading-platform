"""Time-series factor diagnostics for preparation's train/test Feather files."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from fnmatch import fnmatchcase
from functools import wraps
import json
import logging
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.feature_selection import mutual_info_regression
from threadpoolctl import threadpool_limits
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
import os
import multiprocessing as mp

os.environ["LOKY_MAX_CPU_COUNT"] = "8"

LOGGER = logging.getLogger(__name__)
_TIMING_PATH = ContextVar("factor_analysis_timing_path", default=())
_TIMING_STATS = ContextVar("factor_analysis_timing_stats", default=None)


@contextmanager
def timed_step(name: str, **details):
    """Log wall time, including failed steps, with nested execution context."""
    path = (*_TIMING_PATH.get(), name)
    token = _TIMING_PATH.set(path)
    label = " > ".join(path)
    context = " ".join(f"{key}={value}" for key, value in details.items())
    started = perf_counter()
    status = "DONE"
    try:
        yield
    except BaseException:
        status = "FAILED"
        raise
    finally:
        elapsed = perf_counter() - started
        stats = _TIMING_STATS.get()
        if stats is not None:
            entry = stats.setdefault(name, [0, 0.0, 0.0])
            entry[0] += 1
            entry[1] += elapsed
            entry[2] = max(entry[2], elapsed)
        LOGGER.debug("%s %s %s elapsed=%.3fs", status, label, context, elapsed)
        _TIMING_PATH.reset(token)


def timed(name: str):
    """Time a function; the outermost call also prints aggregate statistics."""

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            owner = _TIMING_STATS.get() is None
            if owner:
                if not LOGGER.handlers and not logging.getLogger().handlers:
                    handler = logging.StreamHandler()
                    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
                    LOGGER.addHandler(handler)
                    LOGGER.setLevel(logging.INFO)
                    LOGGER.propagate = False
                token = _TIMING_STATS.set({})
            try:
                with timed_step(name):
                    return function(*args, **kwargs)
            finally:
                if owner:
                    LOGGER.info(
                        "TIMING SUMMARY (inclusive times; nested steps overlap)"
                    )
                    for step, (count, total, maximum) in sorted(
                        _TIMING_STATS.get().items(),
                        key=lambda item: item[1][1],
                        reverse=True,
                    ):
                        LOGGER.info(
                            "TIMING step=%s calls=%d total=%.3fs mean=%.3fs max=%.3fs",
                            step,
                            count,
                            total,
                            total / count,
                            maximum,
                        )
                    _TIMING_STATS.reset(token)

        return wrapped

    return decorate


EXCLUDED = set(
    """
open_time_ms_utc open_time_date_utc close_time_ms_utc ignore
bars_to_close open_time_sn label reach_time scan_len invalid_reason
trend_strength tb_anchor_label tb_anchor_reach_time trend_source_idx
is_tb_anchor is_trend_propagated expected_vol threshold_long threshold_short
stop_threshold_long stop_threshold_short threshold stop_threshold
""".split()
)
STANDARD_RETURN_PERIODS = (
    (60, "1m"),
    (300, "5m"),
    (1800, "30m"),
    (3600, "1h"),
    (14400, "4h"),
    (86400, "1d"),
)


@dataclass(frozen=True)
class AnalysisConfig:
    horizons: tuple[int, ...] | None = None
    rolling_window: int = 252
    rolling_step: int = 100
    quantiles: int = 5
    lags: int = 20
    min_samples: int = 30
    diagnostic_max_samples: int = 10000
    mi_max_samples: int = 3000
    mi_shifts: int = 19
    seed: int = 42
    transform: str = "raw"
    workers: int = 4

    def __post_init__(self):
        if self.horizons is not None and (
            not self.horizons or any(h < 1 for h in self.horizons)
        ):
            raise ValueError("Horizons must be positive bar counts")
        if self.min_samples < 10 or self.rolling_window < self.min_samples:
            raise ValueError("Require rolling_window >= min_samples >= 10")
        if self.rolling_step < 1 or self.lags < 1 or self.quantiles < 2:
            raise ValueError("Invalid rolling step, lag count or quantile count")
        if min(self.diagnostic_max_samples, self.mi_max_samples) < self.min_samples:
            raise ValueError("Sample caps must be at least min_samples")
        if self.mi_shifts < 0:
            raise ValueError("Invalid MI shift count")
        if self.transform not in ("raw", "diff"):
            raise ValueError("transform must be raw or diff")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")


@timed("load_data")
def load_prepared_data(data_dir: str | Path) -> pd.DataFrame:
    frames = []
    for split in ("train", "test"):
        frame = pd.read_feather(Path(data_dir) / f"{split}_data.feather")
        if frame.empty or not frame.columns.is_unique:
            raise ValueError(f"{split} must have rows and unique column names")
        if "open_time_ms_utc" in frame:
            times = pd.to_datetime(frame["open_time_ms_utc"], unit="ms", utc=True)
        else:
            times = pd.to_datetime(frame["open_time_date_utc"], utc=True)
        if times.isna().any():
            raise ValueError(f"Missing timestamps in {split}")
        frame = frame.assign(_time=times, _split=split)
        frames.append(frame)
    if set(frames[0].columns) != set(frames[1].columns):
        raise ValueError("Train and test schemas differ")
    if frames[0]._time.max() >= frames[1]._time.min():
        raise ValueError("Train must end strictly before test begins")
    df = (
        pd.concat(frames, ignore_index=True).sort_values("_time").reset_index(drop=True)
    )
    if df._time.duplicated().any():
        raise ValueError(
            "Duplicate timestamps: analyze one asset and interval at a time"
        )
    close = pd.to_numeric(df["close"], errors="coerce")
    if not (np.isfinite(close) & (close > 0)).all():
        raise ValueError("Close prices must be finite and positive")
    df["close"] = close
    return df


def resolve_return_horizons(df: pd.DataFrame, configured: tuple[int, ...] | None):
    """Resolve the requested clock periods into observed-bar return horizons."""
    deltas = df["_time"].diff().dt.total_seconds()
    bar_seconds = float(deltas[deltas > 0].median())
    if not np.isfinite(bar_seconds) or bar_seconds <= 0:
        raise ValueError("Cannot infer the bar interval from timestamps")
    requested = configured or tuple(
        max(1, int(round(seconds / bar_seconds)))
        for seconds, _ in STANDARD_RETURN_PERIODS
        if seconds >= bar_seconds * 0.5
    )
    horizons = tuple(sorted(set(requested)))
    labels = {}
    for horizon in horizons:
        actual_seconds = horizon * bar_seconds
        nearest = min(
            STANDARD_RETURN_PERIODS, key=lambda item: abs(item[0] - actual_seconds)
        )
        if abs(nearest[0] - actual_seconds) <= max(bar_seconds * 0.51, 1):
            labels[str(horizon)] = nearest[1]
        else:
            labels[str(horizon)] = f"{horizon} bars"
    return horizons, labels, bar_seconds


def select_factors(df: pd.DataFrame, factors: list[str] | None = None) -> list[str]:
    eligible = [
        name
        for name in df.select_dtypes(include="number").columns
        if name not in EXCLUDED and not name.startswith("_")
    ]
    if factors is None:
        selected = eligible
    else:
        eligible_lookup = {name.casefold(): name for name in eligible}
        selected = []
        for query in dict.fromkeys(factors):
            query = query.strip()
            if not query:
                raise ValueError("Factor names and patterns cannot be empty")
            exact = eligible_lookup.get(query.casefold())
            if exact is not None:
                matches = [exact]
            else:
                if query in df.columns:
                    raise ValueError(f"Not an eligible factor: {query}")
                has_glob = any(char in query for char in "*?[]")
                folded_query = query.casefold()
                matches = [
                    name
                    for name in eligible
                    if (
                        fnmatchcase(name.casefold(), folded_query)
                        if has_glob
                        else folded_query in name.casefold()
                    )
                ]
            if not matches:
                raise ValueError(f"No eligible factor matches: {query}")
            selected.extend(matches)
        selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError("No eligible numeric factors")
    return selected


def longest_valid_run(frame: pd.DataFrame | pd.Series):
    valid = frame.notna()
    if isinstance(valid, pd.DataFrame):
        valid = valid.all(axis=1)
    boundaries = np.flatnonzero(np.diff(np.r_[False, valid.to_numpy(), False]))
    if not len(boundaries):
        return frame.iloc[:0]
    starts, ends = boundaries[::2], boundaries[1::2]
    longest = np.argmax(ends - starts)
    return frame.iloc[starts[longest] : ends[longest]]


def correlation(x, y, method="spearman") -> float:
    if len(x) < 2 or pd.Series(x).nunique() < 2 or pd.Series(y).nunique() < 2:
        return np.nan
    if method == "spearman":
        return float(spearmanr(x, y)[0])
    return float(np.corrcoef(x, y)[0, 1])


@timed("diagnostics")
def diagnostics(x: pd.Series, cfg: AnalysisConfig) -> tuple[dict, list[dict]]:
    values = longest_valid_run(x).iloc[-cfg.diagnostic_max_samples :]
    result = {
        "n_total": len(x),
        "n_finite": int(x.notna().sum()),
        "diagnostic_n": len(values),
        "missing_fraction": float(x.isna().mean()),
    }
    if len(values) < cfg.min_samples or values.nunique() < 2:
        return {**result, "status": "constant_or_insufficient"}, []

    values = (values - values.mean()) / values.std()
    lag = min(cfg.lags, len(values) // 2 - 2)
    messages = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name, operation in (
            (
                "adf",
                lambda: adfuller(values, maxlag=lag, autolag="AIC", regression="c"),
            ),
            ("kpss", lambda: kpss(values, regression="c", nlags="auto")),
        ):
            try:
                outcome = operation()
                result.update(
                    {
                        f"{name}_stat": outcome[0],
                        f"{name}_pvalue": outcome[1],
                        f"{name}_lags": outcome[2],
                    }
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                messages.append(f"{name}: {exc}")
        messages.extend(str(w.message) for w in caught)

    a, q, p = acf(values, nlags=lag, fft=True, qstat=True)
    try:
        pa = pacf(values, nlags=lag, method="ywm")
    except (ValueError, np.linalg.LinAlgError) as exc:
        pa = np.full(lag + 1, np.nan)
        messages.append(f"pacf: {exc}")

    result.update(status="ok", warnings="; ".join(messages), acf_1=a[1])
    if "adf_pvalue" in result and "kpss_pvalue" in result:
        adf_reject = result["adf_pvalue"] < 0.05
        kpss_reject = result["kpss_pvalue"] < 0.05
        result["stationarity"] = (
            "stationary"
            if adf_reject and not kpss_reject
            else "nonstationary" if kpss_reject and not adf_reject else "inconclusive"
        )

    rows = [
        {
            "lag": i,
            "acf": a[i],
            "pacf": pa[i],
            "ljung_box_stat": q[i - 1],
            "ljung_box_pvalue": p[i - 1],
        }
        for i in range(1, lag + 1)
    ]
    return result, rows


def _row_correlations(x, y, valid, counts):
    """Compute paired Pearson correlations along rows without Python windows."""
    # 避免重复的 np.where 掩码和 max/min 遍历
    x_clean = np.where(valid, x, 0.0)
    y_clean = np.where(valid, y, 0.0)

    mean_x = x_clean.sum(axis=1, keepdims=True) / counts[:, None]
    mean_y = y_clean.sum(axis=1, keepdims=True) / counts[:, None]

    dx = np.where(valid, x_clean - mean_x, 0.0)
    dy = np.where(valid, y_clean - mean_y, 0.0)

    numerator = np.einsum("ij,ij->i", dx, dy)
    var_x = np.einsum("ij,ij->i", dx, dx)
    var_y = np.einsum("ij,ij->i", dy, dy)
    denominator = np.sqrt(var_x * var_y)

    result = np.full(len(x), np.nan)
    # 只要分母大于0即可，不需要额外的变量极值检测
    np.divide(numerator, denominator, out=result, where=(denominator > 0))
    return np.clip(result, -1, 1)


@timed("rolling_ic")
def rolling_ic(pair: pd.DataFrame, times: pd.Series, cfg: AnalysisConfig) -> list[dict]:
    rows = []
    if len(pair) < cfg.rolling_window:
        return rows

    x_np = pair.x.to_numpy(dtype=float)
    y_np = pair.y.to_numpy(dtype=float)
    times_np = times.to_numpy()

    x_windows = np.lib.stride_tricks.sliding_window_view(x_np, cfg.rolling_window)[
        :: cfg.rolling_step
    ]
    y_windows = np.lib.stride_tricks.sliding_window_view(y_np, cfg.rolling_window)[
        :: cfg.rolling_step
    ]

    batch_size = max(1, 250_000 // cfg.rolling_window)
    for start in range(0, len(x_windows), batch_size):
        x = x_windows[start : start + batch_size]
        y = y_windows[start : start + batch_size]
        valid = ~np.isnan(x) & ~np.isnan(y)
        counts = valid.sum(axis=1)
        keep = counts >= cfg.min_samples
        if not keep.any():
            continue

        x, y, valid, counts = x[keep], y[keep], valid[keep], counts[keep]
        rx = rankdata(np.where(valid, x, np.inf), axis=1, method="average")
        ry = rankdata(np.where(valid, y, np.inf), axis=1, method="average")
        rank_ic = _row_correlations(rx, ry, valid, counts)
        pearson_ic = _row_correlations(x, y, valid, counts)

        stops = (
            (start + np.flatnonzero(keep)) * cfg.rolling_step + cfg.rolling_window - 1
        )
        # 直接通过 numpy 获取 window_end，减少 pandas 索引开销
        w_ends = times_np[stops]

        rows.extend(
            pd.DataFrame(
                {
                    "window_end": w_ends,
                    "n": counts,
                    "rank_ic": rank_ic,
                    "pearson_ic": pearson_ic,
                }
            ).to_dict("records")
        )
    return rows


@timed("mutual_information")
def mutual_information(pair: pd.DataFrame, cfg: AnalysisConfig, horizon: int) -> dict:
    block = longest_valid_run(pair).iloc[-cfg.mi_max_samples :]
    if len(block) < cfg.min_samples or block.x.nunique() < 2 or block.y.nunique() < 2:
        return {"mi_n": len(block), "mi_status": "constant_or_insufficient"}

    x, y = block.x.to_numpy(), block.y.to_numpy()
    discrete = len(np.unique(x)) <= 20

    def estimate(target):
        return float(
            mutual_info_regression(
                x[:, None], target, discrete_features=discrete, random_state=cfg.seed
            )[0]
        )

    score = estimate(y)
    result = {
        "mi_n": len(block),
        "mi_nats": score,
        "mi_discrete": discrete,
        "mi_status": "ok",
    }

    margin = max(horizon, cfg.lags) + 1
    shifts = np.arange(margin, len(block) - margin)
    if cfg.mi_shifts and len(shifts):
        rng = np.random.default_rng(cfg.seed)
        shifts = rng.choice(shifts, min(cfg.mi_shifts, len(shifts)), replace=False)
        null = np.array([estimate(np.roll(y, int(s))) for s in shifts])
        result.update(
            mi_shift_pvalue=(1 + int((null >= score).sum())) / (1 + len(null)),
            mi_null_mean=float(null.mean()),
            mi_shifts=len(null),
        )
    return result


@timed("predictive_metrics")
def predictive_metrics(
    pair: pd.DataFrame, edges: np.ndarray, cfg: AnalysisConfig, horizon: int
) -> tuple[dict, list[dict]]:
    clean = pair.dropna()
    result = {"n": len(clean)}
    if len(clean) < cfg.min_samples or clean.x.nunique() < 2 or clean.y.nunique() < 2:
        return {**result, "status": "constant_or_insufficient"}, []

    x_np, y_np = clean.x.to_numpy(dtype=float), clean.y.to_numpy(dtype=float)

    result.update(
        status="ok",
        rank_ic=correlation(x_np, y_np),
        pearson_ic=correlation(x_np, y_np, "pearson"),
    )

    # Newey-West
    block = longest_valid_run(pair)
    result["nw_n"] = len(block)
    if (
        len(block) >= cfg.min_samples
        and block.x.nunique() > 1
        and block.y.nunique() > 1
    ):
        x = (block.x - block.x.mean()) / block.x.std()
        bandwidth = min(
            len(block) - 2, max(horizon - 1, int(4 * (len(block) / 100) ** (2 / 9)))
        )
        fit = sm.OLS(block.y.to_numpy(), sm.add_constant(x.to_numpy())).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": bandwidth, "use_correction": True},
            use_t=True,
        )
        result.update(
            nw_beta=fit.params[1],
            nw_se=fit.bse[1],
            nw_t=fit.tvalues[1],
            nw_pvalue=fit.pvalues[1],
            nw_lags=bandwidth,
        )

    result.update(mutual_information(pair, cfg, horizon))

    # Quantile Analysis (用 Numpy 替代 Pandas 的 cut/groupby)
    groups = []
    if len(edges) < 3:
        result["quantile_status"] = "insufficient_distinct_train_edges"
        return result, groups

    bins = np.digitize(x_np, edges) - 1
    for i in range(len(edges) - 1):
        mask = bins == i
        cnt = int(mask.sum())
        if cnt > 0:
            qy = y_np[mask]
            groups.append(
                {
                    "quantile": i + 1,
                    "n": cnt,
                    "mean_return": float(qy.mean()),
                    "median_return": float(np.median(qy)),
                    "positive_fraction": float((qy > 0).mean()),
                }
            )
        else:
            groups.append(
                {
                    "quantile": i + 1,
                    "n": 0,
                    "mean_return": np.nan,
                    "median_return": np.nan,
                    "positive_fraction": np.nan,
                }
            )

    populated = [r for r in groups if r["n"] > 0]
    if len(populated) > 1:
        result["quantile_monotonicity"] = correlation(
            np.arange(len(populated)), np.array([r["mean_return"] for r in populated])
        )
        result["top_minus_bottom"] = (
            groups[-1]["mean_return"] - groups[0]["mean_return"]
        )

    return result, groups


_FACTOR_WORKER_CONTEXT = None
_FACTOR_THREADPOOL_LIMITER = None


def _initialize_factor_worker(close, times, train_count, cfg, limit_threads=True):
    """Cache split targets once per process rather than sending them per factor."""
    global _FACTOR_THREADPOOL_LIMITER, _FACTOR_WORKER_CONTEXT
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(processName)s %(message)s",
        )
    if limit_threads:
        _FACTOR_THREADPOOL_LIMITER = threadpool_limits(limits=1)

    split_slices = {
        "train": slice(0, train_count),
        "test": slice(train_count, None),
        "combined": slice(None),
    }
    split_times, split_targets = {}, {}
    for split, section in split_slices.items():
        split_close = pd.Series(close[section])
        split_times[split] = pd.Series(
            pd.to_datetime(times[section], unit="ns", utc=True)
        )
        split_targets[split] = {
            horizon: (split_close.shift(-horizon) / split_close - 1).replace(
                [np.inf, -np.inf], np.nan
            )
            for horizon in cfg.horizons
        }

    _FACTOR_WORKER_CONTEXT = {
        "cfg": cfg,
        "split_slices": split_slices,
        "split_times": split_times,
        "split_targets": split_targets,
        "train_count": train_count,
    }


@timed("factor")
def _analyze_factor_task(factor, x_values):
    """Analyze one factor across all splits and horizons in the current process."""
    context = _FACTOR_WORKER_CONTEXT
    if context is None:
        raise RuntimeError("Factor worker was not initialized")
    cfg = context["cfg"]
    train_count = context["train_count"]
    split_slices = context["split_slices"]
    split_times = context["split_times"]
    split_targets = context["split_targets"]
    tables = {
        name: []
        for name in (
            "diagnostics",
            "autocorrelation",
            "predictive",
            "quantiles",
            "rolling_ic",
        )
    }

    LOGGER.info("Factor started: %s", factor)
    with timed_step("prepare_factor"):
        x = pd.Series(np.asarray(x_values, dtype=float)).replace(
            [np.inf, -np.inf], np.nan
        )
        if cfg.transform == "diff":
            x = x.diff()
        train = x.iloc[:train_count].dropna()
        cuts = np.unique(
            train.quantile(np.linspace(0, 1, cfg.quantiles + 1)[1:-1]).dropna()
        )
        cuts = cuts[(cuts > train.min()) & (cuts < train.max())]
        if train.nunique() > 1 and not len(cuts):
            cuts = np.array([train.min() / 2 + train.max() / 2])
        edges = (
            np.r_[-np.inf, cuts, np.inf]
            if train.nunique() > 1
            else np.array([])
        )

    for split in ("train", "test", "combined"):
        values = x.iloc[split_slices[split]].reset_index(drop=True)
        key = {"factor": factor, "split": split}
        diag, lags = diagnostics(values, cfg)
        tables["diagnostics"].append({**key, **diag})
        tables["autocorrelation"].extend({**key, **row} for row in lags)

        for horizon in cfg.horizons:
            target = split_targets[split][horizon]
            pair = pd.DataFrame({"x": values, "y": target}).replace(
                [np.inf, -np.inf], np.nan
            )
            hkey = {**key, "horizon": horizon}
            metrics, groups = predictive_metrics(pair, edges, cfg, horizon)
            rolling = rolling_ic(pair, split_times[split], cfg)
            rank_values = pd.Series(
                [row["rank_ic"] for row in rolling], dtype=float
            ).dropna()
            std = rank_values.std()
            metrics.update(
                rolling_ic_count=len(rank_values),
                rolling_ic_mean=rank_values.mean(),
                rolling_ic_std=std,
                icir=(rank_values.mean() / std if std > 0 else np.nan),
            )
            tables["predictive"].append({**hkey, **metrics})
            tables["quantiles"].extend({**hkey, **row} for row in groups)
            tables["rolling_ic"].extend({**hkey, **row} for row in rolling)

    LOGGER.info("Factor finished: %s", factor)
    return factor, cuts.tolist() if len(edges) else [], tables


@timed("analysis_total")
def analyze(
    data_dir: str | Path,
    output_dir: str | Path | None = None,
    factors: list[str] | None = None,
    config: AnalysisConfig | None = None,
) -> dict[str, pd.DataFrame]:
    cfg = config or AnalysisConfig()
    df = load_prepared_data(data_dir)
    horizons, horizon_labels, bar_seconds = resolve_return_horizons(df, cfg.horizons)
    cfg = replace(cfg, horizons=horizons)
    LOGGER.info(
        "Forward return horizons: %s",
        ", ".join(f"{horizon_labels[str(h)]}={h} bars" for h in horizons),
    )
    selected = select_factors(df, factors)

    output = Path(output_dir) if output_dir else Path(data_dir) / "factor_analysis"
    tables = {
        name: []
        for name in (
            "diagnostics",
            "autocorrelation",
            "predictive",
            "quantiles",
            "rolling_ic",
        )
    }
    thresholds = {}

    with timed_step("cache_splits_and_returns"):
        train_count = int(df._split.eq("train").sum())
        close_values = df.close.to_numpy(dtype=np.float64, copy=True)
        time_values = df._time.astype("int64").to_numpy(copy=True)

    worker_count = min(cfg.workers, len(selected))
    LOGGER.info(
        "Factor workload: factors=%d workers=%d splits=3 horizons=%d",
        len(selected),
        worker_count,
        len(cfg.horizons),
    )

    def collect_result(result, completed):
        factor, cuts, factor_tables = result
        thresholds[factor] = cuts
        for name, rows in factor_tables.items():
            tables[name].extend(rows)
        if completed % 10 == 0 or completed == len(selected):
            LOGGER.info("Factor progress %d/%d: %s", completed, len(selected), factor)

    if worker_count == 1:
        _initialize_factor_worker(
            close_values, time_values, train_count, cfg, limit_threads=False
        )
        for index, factor in enumerate(selected, 1):
            factor_values = df[factor].astype(float).to_numpy(
                dtype=np.float64, copy=True
            )
            collect_result(_analyze_factor_task(factor, factor_values), index)
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_factor_worker,
            initargs=(close_values, time_values, train_count, cfg),
        ) as pool:
            pending = {}
            next_index = 0
            completed = 0
            while next_index < len(selected) or pending:
                while next_index < len(selected) and len(pending) < worker_count:
                    factor = selected[next_index]
                    factor_values = df[factor].astype(float).to_numpy(
                        dtype=np.float64, copy=True
                    )
                    future = pool.submit(_analyze_factor_task, factor, factor_values)
                    pending[future] = next_index
                    next_index += 1

                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    completed += 1
                    collect_result(future.result(), completed)

    with timed_step("build_reports_and_fdr"):
        results = {name: pd.DataFrame(rows) for name, rows in tables.items()}
        empty_columns = {
            "autocorrelation": [
                "factor",
                "split",
                "lag",
                "acf",
                "pacf",
                "ljung_box_stat",
                "ljung_box_pvalue",
            ],
            "quantiles": [
                "factor",
                "split",
                "horizon",
                "quantile",
                "n",
                "mean_return",
                "median_return",
                "positive_fraction",
            ],
            "rolling_ic": [
                "factor",
                "split",
                "horizon",
                "window_end",
                "n",
                "rank_ic",
                "pearson_ic",
            ],
        }
        for name, columns in empty_columns.items():
            if results[name].empty:
                results[name] = pd.DataFrame(columns=columns)

        prediction = results["predictive"]
        if "nw_pvalue" in prediction:
            prediction["nw_fdr_bh"] = np.nan
            for _, group in prediction.groupby("split"):
                valid = group.nw_pvalue.dropna()
                if len(valid):
                    prediction.loc[valid.index, "nw_fdr_bh"] = multipletests(
                        valid, method="fdr_bh"
                    )[1]

        results["decay"] = prediction[
            [
                c
                for c in (
                    "factor",
                    "split",
                    "horizon",
                    "n",
                    "rank_ic",
                    "pearson_ic",
                    "nw_t",
                    "nw_pvalue",
                    "nw_fdr_bh",
                )
                if c in prediction
            ]
        ].copy()

    output.mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        with timed_step("write_csv", table=name, rows=len(table)):
            table.to_csv(output / f"{name}.csv", index=False)

    with timed_step("write_metadata"):
        metadata = {
            "config": asdict(cfg),
            "horizon_labels": horizon_labels,
            "factors": selected,
            "train_quantile_cuts": thresholds,
            "data_dir": str(Path(data_dir).resolve()),
            "row_count": len(df),
            "start": df._time.iloc[0].isoformat(),
            "end": df._time.iloc[-1].isoformat(),
            "median_bar_seconds": bar_seconds,
            "unequal_time_steps": int(df._time.diff().dropna().nunique()) > 1,
            "return_definition": "close[t+h] / close[t] - 1; horizons count observed bars",
            "combined_is_in_sample": True,
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
        )
    LOGGER.info("Saved analysis to %s", output.resolve())
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "output" / "data",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--factors",
        nargs="+",
        metavar="FACTOR",
        help="Limit analysis to factor names or patterns (substring or glob); omit to analyze all eligible factors",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=AnalysisConfig().workers,
        help="Maximum number of worker processes; each task processes one factor",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help="Forward horizons in observed bars; defaults to 1m, 5m, 30m, 1h, 4h, 1d when available",
    )
    for name in (
        "rolling_window",
        "rolling_step",
        "quantiles",
        "lags",
        "min_samples",
        "diagnostic_max_samples",
        "mi_max_samples",
        "mi_shifts",
        "seed",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=int,
            default=getattr(AnalysisConfig(), name),
        )
    parser.add_argument("--transform", choices=["raw", "diff"], default="raw")
    args = vars(parser.parse_args())
    data_dir, output_dir, factors = (
        args.pop(name) for name in ("data_dir", "output_dir", "factors")
    )
    args["horizons"] = tuple(args["horizons"]) if args["horizons"] else None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    analyze(data_dir, output_dir, factors, AnalysisConfig(**args))


if __name__ == "__main__":
    main()
