"""
Volatility parameter analysis for the project's current volatility estimator.

Current estimator (kept consistent with data_process.common.calculate_thresholds):
    1. Rogers-Satchell single-bar variance
    2. EWMA(span=M) of RS variance
    3. sqrt(EWMA variance) * sqrt(N)

For every (M, N):
    predicted_vol[t] = sqrt(EWMA_M(RS_var)[t]) * sqrt(N)
    realized_vol[t]  = sqrt(sum(RS_var[t+1 : t+N+1]))

Outputs:
    - Pearson correlation heatmap
    - Symmetric prediction-accuracy heatmap
    - Spearman / MAE / RMSE / calibration-ratio matrices as CSV
    - Long-form metrics CSV

The public entry point is intentionally ``main(pare_para, ...)`` so it can be
called directly from preparation.py with the same BaseDefine instance.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from data_process import common


DEFAULT_M_VALUES = [5, 10, 20, 40, 80, 160, 320]
DEFAULT_N_VALUES = [1, 2, 4, 8, 16, 32, 64]
EPS = 1e-12


def _normalize_grid(values: Iterable[int], current_value: int) -> list[int]:
    """Return sorted positive unique integers and always include current value."""
    out = {int(v) for v in values if int(v) > 0}
    if int(current_value) > 0:
        out.add(int(current_value))
    return sorted(out)


def _rogers_satchell_variance(df: pd.DataFrame) -> pd.Series:
    """Same single-period Rogers-Satchell variance used by common.calculate_thresholds."""
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")

    log_ho = np.log(df["high"] / df["open"])
    log_hc = np.log(df["high"] / df["close"])
    log_lo = np.log(df["low"] / df["open"])
    log_lc = np.log(df["low"] / df["close"])

    rs_var = log_hc * log_ho + log_lc * log_lo
    return rs_var.clip(lower=0)


def _future_realized_volatility(
    rs_var: pd.Series,
    time_ms: pd.Series,
    interval_ms: int,
    horizon: int,
) -> pd.Series:
    """
    Realized volatility over the NEXT N bars using the same RS variance measure.

    At row t this is sqrt(sum(rs_var[t+1] ... rs_var[t+N])).
    Rows crossing a physical-time gap (e.g. a Forex weekend) are invalidated.
    """
    horizon = int(horizon)
    shifted = rs_var.shift(-1)
    future_var = shifted.rolling(horizon, min_periods=horizon).sum().shift(-(horizon - 1))

    # Require exactly N physical bar intervals from anchor t to row t+N.
    target_time = time_ms.shift(-horizon)
    contiguous = (target_time - time_ms) == (horizon * interval_ms)
    future_var = future_var.where(contiguous)

    return np.sqrt(future_var.clip(lower=0))


def _prediction_metrics(predicted: pd.Series, actual: pd.Series) -> dict[str, float]:
    valid = predicted.notna() & actual.notna()
    p = predicted.loc[valid].astype(float)
    a = actual.loc[valid].astype(float)

    finite = np.isfinite(p.to_numpy()) & np.isfinite(a.to_numpy())
    p = p.iloc[np.flatnonzero(finite)]
    a = a.iloc[np.flatnonzero(finite)]

    if len(p) < 3:
        return {
            "samples": float(len(p)),
            "pearson": np.nan,
            "spearman": np.nan,
            "symmetric_accuracy": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "median_actual_pred_ratio": np.nan,
        }

    pearson = p.corr(a, method="pearson")
    # Ranking first avoids a scipy dependency while giving Spearman correlation.
    spearman = p.rank(method="average").corr(a.rank(method="average"), method="pearson")

    error = p - a
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(np.square(error)))

    # 1 - sMAPE/2.  Range is [0, 1] for non-negative volatility values.
    # 1.0 = exact prediction; lower values = worse magnitude calibration.
    symmetric_accuracy = np.mean(
        1.0 - (np.abs(p - a) / (np.abs(p) + np.abs(a) + EPS))
    )

    nonzero_pred = p > EPS
    ratio = (a.loc[nonzero_pred] / p.loc[nonzero_pred]) if nonzero_pred.any() else pd.Series(dtype=float)
    median_ratio = float(ratio.median()) if len(ratio) else np.nan

    return {
        "samples": float(len(p)),
        "pearson": float(pearson),
        "spearman": float(spearman),
        "symmetric_accuracy": float(symmetric_accuracy),
        "mae": float(mae),
        "rmse": float(rmse),
        "median_actual_pred_ratio": median_ratio,
    }


def _plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    colorbar_label: str,
    output_path: str,
    value_format: str = ".3f",
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", origin="lower")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Future horizon N (bars)")
    ax.set_ylabel("EWMA span M (bars)")
    ax.set_title(title)

    # Put the metric value into every cell; useful when grids are not too large.
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            val = values[row, col]
            if np.isfinite(val):
                ax.text(col, row, format(val, value_format), ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_price_and_volatility(
    pare_para,
    volatility_span: Iterable[int],
    output_path: Optional[str] = None,
    logger=None,
) -> dict[str, object]:
    """
    Plot price and volatility for multiple EWMA spans.

    Every span produces an individual dual-axis chart. A final combined image
    contains all charts in a subplot grid. Price uses the left y-axis;
    volatility uses the right y-axis and is shown as a percentage. The same
    Rogers-Satchell estimator as ``common.calculate_thresholds`` is used.
    """
    if isinstance(volatility_span, (str, bytes)):
        raise TypeError("volatility_span must be an iterable of positive integers")
    try:
        raw_spans = list(volatility_span)
    except TypeError as exc:
        raise TypeError(
            "volatility_span must be an iterable of positive integers"
        ) from exc

    spans: list[int] = []
    for raw_span in raw_spans:
        if isinstance(raw_span, bool):
            raise ValueError("volatility_span values must be positive integers")
        span = int(raw_span)
        if span <= 0 or float(raw_span) != span:
            raise ValueError("volatility_span values must be positive integers")
        if span not in spans:
            spans.append(span)
    if not spans:
        raise ValueError("volatility_span must contain at least one value")

    emit = logger.info if logger is not None else print
    source_file = common.market_data_path(pare_para)
    df = pd.read_csv(source_file)

    interval_ms = common.get_interval_ms(pare_para.interval)
    if pare_para.market_category == "Cryptocurrency":
        common.validate_kline_source(
            df,
            interval_ms,
            source=source_file,
            logger=logger,
        )

    if "open_time_ms_utc" in df.columns:
        df = df.sort_values("open_time_ms_utc").reset_index(drop=True)
        time_axis = pd.to_datetime(
            df["open_time_ms_utc"],
            unit="ms",
            utc=True,
            errors="coerce",
        )
    elif "open_time_date_utc" in df.columns:
        time_axis = pd.to_datetime(
            df["open_time_date_utc"],
            utc=True,
            errors="coerce",
        )
        order = np.argsort(time_axis.to_numpy())
        df = df.iloc[order].reset_index(drop=True)
        time_axis = time_axis.iloc[order].reset_index(drop=True)
    else:
        raise ValueError(
            "Market data must contain open_time_ms_utc or open_time_date_utc"
        )

    rs_var = _rogers_satchell_variance(df)
    close_price = pd.to_numeric(df["close"], errors="coerce")
    volatility_by_span = {
        span: np.sqrt(
            rs_var.ewm(span=span, adjust=False).mean().clip(lower=0)
        ) * 100.0
        for span in spans
    }

    if output_path is None:
        output_dir = os.path.join(
            common.PERSISTENCE_DIR,
            "analyse",
            "volatility_prediction",
            f"{pare_para.symbol}_{pare_para.interval}",
        )
        output_path = os.path.join(
            output_dir,
            f"price_volatility_spans_{'_'.join(map(str, spans))}.png",
        )
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    def draw_span(price_ax, span: int) -> None:
        volatility_pct = volatility_by_span[span]
        valid = (
            time_axis.notna()
            & close_price.notna()
            & volatility_pct.notna()
            & np.isfinite(close_price)
            & np.isfinite(volatility_pct)
        )
        if not valid.any():
            raise ValueError(
                f"No finite price/volatility samples are available for span={span}"
            )

        volatility_ax = price_ax.twinx()
        price_line = price_ax.plot(
            time_axis.loc[valid],
            close_price.loc[valid],
            color="tab:blue",
            linewidth=1.0,
            label="Close price",
        )
        volatility_line = volatility_ax.plot(
            time_axis.loc[valid],
            volatility_pct.loc[valid],
            color="tab:orange",
            linewidth=1.0,
            alpha=0.85,
            label=f"RS EWMA volatility ({span} bars)",
        )
        price_ax.set_xlabel("Time (UTC)")
        price_ax.set_ylabel("Price", color="tab:blue")
        volatility_ax.set_ylabel("Volatility (%)", color="tab:orange")
        price_ax.tick_params(axis="y", labelcolor="tab:blue")
        volatility_ax.tick_params(axis="y", labelcolor="tab:orange")
        price_ax.grid(True, alpha=0.25)
        price_ax.set_title(
            f"{pare_para.symbol} {pare_para.interval} - EWMA span={span}"
        )
        lines = price_line + volatility_line
        price_ax.legend(
            lines,
            [line.get_label() for line in lines],
            loc="upper left",
        )

    individual_paths: dict[int, str] = {}
    for span in spans:
        individual_path = os.path.join(
            output_dir,
            f"price_volatility_span_{span}.png",
        )
        fig, price_ax = plt.subplots(figsize=(15, 7))
        draw_span(price_ax, span)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(individual_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        individual_paths[span] = individual_path
        emit(
            f"[VolatilityAnalysis] price/volatility chart span={span} "
            f"output={individual_path}"
        )

    column_count = min(2, len(spans))
    row_count = int(np.ceil(len(spans) / column_count))
    combined_fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(9 * column_count, 5.5 * row_count),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for price_ax, span in zip(flat_axes, spans):
        draw_span(price_ax, span)
    for unused_ax in flat_axes[len(spans):]:
        combined_fig.delaxes(unused_ax)

    combined_fig.suptitle(
        f"Price and Volatility by EWMA Span - "
        f"{pare_para.symbol} {pare_para.interval}",
        fontsize=15,
    )
    combined_fig.autofmt_xdate()
    combined_fig.tight_layout(rect=(0, 0, 1, 0.97))
    combined_fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(combined_fig)
    emit(f"[VolatilityAnalysis] combined chart output={output_path}")

    return {
        "spans": spans,
        "individual_charts": individual_paths,
        "combined_chart": output_path,
    }


def analyse_volatility_parameters(
    pare_para,
    m_values: Optional[Iterable[int]] = None,
    n_values: Optional[Iterable[int]] = None,
    output_dir: Optional[str] = None,
    logger=None,
) -> dict[str, object]:
    """
    Generate M/N volatility correlation and prediction-accuracy heatmaps.

    Parameters
    ----------
    pare_para:
        Existing common.BaseDefine instance. market source / symbol / interval are
        taken directly from it. Its current vol_ewma_span and predict_num are
        automatically included in the grid.
    m_values:
        EWMA spans M to test. Default: DEFAULT_M_VALUES + current M.
    n_values:
        Future horizons N to test. Default: DEFAULT_N_VALUES + current N.
    output_dir:
        Optional output directory. Defaults to
        quant_output/analyse/volatility_prediction/{symbol}_{interval}.
    logger:
        Optional logger; print() is used when omitted.
    """
    emit = logger.info if logger is not None else print

    m_values = _normalize_grid(
        DEFAULT_M_VALUES if m_values is None else m_values,
        pare_para.vol_ewma_span,
    )
    n_values = _normalize_grid(
        DEFAULT_N_VALUES if n_values is None else n_values,
        pare_para.predict_num,
    )

    source_file = common.market_data_path(pare_para)
    emit(f"[VolatilityAnalysis] source={source_file}")
    emit(f"[VolatilityAnalysis] M={m_values}")
    emit(f"[VolatilityAnalysis] N={n_values}")

    df = pd.read_csv(source_file)
    interval_ms = common.get_interval_ms(pare_para.interval)

    if pare_para.market_category == "Cryptocurrency":
        common.validate_kline_source(
            df,
            interval_ms,
            source=source_file,
            logger=logger,
        )

    # Keep preprocessing aligned with preparation.py.
    if logger is not None:
        df = common.clean_data_quality_auto(df, logger)
    else:
        # clean_data_quality_auto expects a logger, so use a minimal no-op adapter.
        class _Logger:
            def info(self, *_args, **_kwargs):
                pass

            def warning(self, *_args, **_kwargs):
                pass

        df = common.clean_data_quality_auto(df, _Logger())

    df = df.sort_values("open_time_ms_utc").reset_index(drop=True)
    rs_var = _rogers_satchell_variance(df)
    time_ms = pd.to_numeric(df["open_time_ms_utc"], errors="coerce")

    # Future realized volatility only depends on N, so calculate once per horizon.
    future_vol_by_n = {
        n: _future_realized_volatility(rs_var, time_ms, interval_ms, n)
        for n in n_values
    }

    records: list[dict[str, float]] = []

    for m in m_values:
        # Use the project's own calculate_thresholds implementation with N=1,
        # then scale by sqrt(N). This exactly matches current expected_vol logic
        # while avoiding recalculating the same EWMA for every N.
        m_para = replace(pare_para, vol_method="ewma", vol_ewma_span=int(m), predict_num=1)
        one_bar = common.calculate_thresholds(df.copy(), para=m_para)["expected_vol"]

        for n in n_values:
            predicted_vol = one_bar * np.sqrt(n)
            actual_vol = future_vol_by_n[n]
            metrics = _prediction_metrics(predicted_vol, actual_vol)
            records.append({"M": int(m), "N": int(n), **metrics})

    metrics_df = pd.DataFrame(records)

    def pivot(metric: str) -> pd.DataFrame:
        return (
            metrics_df.pivot(index="M", columns="N", values=metric)
            .reindex(index=m_values, columns=n_values)
        )

    matrices = {
        "pearson": pivot("pearson"),
        "spearman": pivot("spearman"),
        "symmetric_accuracy": pivot("symmetric_accuracy"),
        "mae": pivot("mae"),
        "rmse": pivot("rmse"),
        "median_actual_pred_ratio": pivot("median_actual_pred_ratio"),
        "samples": pivot("samples"),
    }

    if output_dir is None:
        output_dir = os.path.join(
            common.PERSISTENCE_DIR,
            "analyse",
            "volatility_prediction",
            f"{pare_para.symbol}_{pare_para.interval}",
        )
    os.makedirs(output_dir, exist_ok=True)

    metrics_csv = os.path.join(output_dir, "volatility_MN_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    for name, matrix in matrices.items():
        matrix.to_csv(os.path.join(output_dir, f"{name}_matrix.csv"))

    pearson_png = os.path.join(output_dir, "pearson_correlation_heatmap.png")
    accuracy_png = os.path.join(output_dir, "prediction_accuracy_heatmap.png")

    _plot_heatmap(
        matrices["pearson"],
        title=f"Volatility Forecast Pearson Correlation - {pare_para.symbol} {pare_para.interval}",
        colorbar_label="Pearson correlation",
        output_path=pearson_png,
    )
    _plot_heatmap(
        matrices["symmetric_accuracy"],
        title=f"Volatility Forecast Symmetric Accuracy - {pare_para.symbol} {pare_para.interval}",
        colorbar_label="Accuracy (1 = exact)",
        output_path=accuracy_png,
    )

    # Useful diagnostics for the user's mean-reversion concern:
    # actual/predicted < 1 means the current estimator systematically over-predicts.
    ratio_png = os.path.join(output_dir, "actual_predicted_ratio_heatmap.png")
    _plot_heatmap(
        matrices["median_actual_pred_ratio"],
        title=f"Median Actual / Predicted Volatility - {pare_para.symbol} {pare_para.interval}",
        colorbar_label="Median actual / predicted",
        output_path=ratio_png,
    )

    current_row = metrics_df[
        (metrics_df["M"] == int(pare_para.vol_ewma_span))
        & (metrics_df["N"] == int(pare_para.predict_num))
    ]
    if not current_row.empty:
        row = current_row.iloc[0]
        emit(
            "[VolatilityAnalysis] current params "
            f"M={int(row['M'])}, N={int(row['N'])}: "
            f"Pearson={row['pearson']:.4f}, Spearman={row['spearman']:.4f}, "
            f"Accuracy={row['symmetric_accuracy']:.4f}, "
            f"MedianActual/Predicted={row['median_actual_pred_ratio']:.4f}"
        )

    emit(f"[VolatilityAnalysis] outputs={output_dir}")

    return {
        "metrics": metrics_df,
        "matrices": matrices,
        "output_dir": output_dir,
        "pearson_heatmap": pearson_png,
        "accuracy_heatmap": accuracy_png,
        "ratio_heatmap": ratio_png,
        "metrics_csv": metrics_csv,
    }


def main(
    pare_para,
    m_values: Optional[Iterable[int]] = None,
    n_values: Optional[Iterable[int]] = None,
    output_dir: Optional[str] = None,
    logger=None,
):
    """preparation.py-friendly entry point."""
    return analyse_volatility_parameters(
        pare_para=pare_para,
        m_values=m_values,
        n_values=n_values,
        output_dir=output_dir,
        logger=logger,
    )


if __name__ == "__main__":
    logger, _ = common.setup_session_logger(sub_folder="analyse")
    pare_para = common.DOGE_1h
    main(pare_para=pare_para, logger=logger)
