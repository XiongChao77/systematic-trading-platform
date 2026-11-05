"""Where the martingale lost money, drawn on the price series.

Two different things go wrong in a martingale run, and the chart marks both:

* a **losing trade** - the ladder was driven through every layer and the cycle
  was stopped out.  Frequent, survivable, and the thing that actually grinds an
  account down.
* an **account failure** - liquidation, or equity through the ruin threshold.
  Rare, terminal, and always preceded by some of the above.

One figure, four stacked panels sharing the time axis:

1. the market, with every losing trade and every account failure on it;
2. one horizontal segment per run, from its random entry to its exit;
3. losing trades per time bucket;
4. account failures per time bucket, split by cause.

Panels 3 and 4 are small multiples, not a dual axis: losses outnumber failures
by two orders of magnitude, so sharing a y-scale would flatten one of them.

Colour never carries meaning alone here - each class also has its own marker
shape and a legend entry.  The failure causes use the reserved status colours
(serious, critical); a losing trade is not a status, so it takes a categorical
slot instead of a third, lighter red that no one could tell apart.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")   # headless: this module only ever writes files

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D


# Validated against the light chart surface (#fcfcfb) with --pairs all:
# worst CVD ΔE 13.0, worst normal-vision ΔE 15.7, both clear of the floors.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"
PRICE_LINE = "#52514e"
PRICE_BAND = "#c9c8c4"

LOSS_STYLE = {"color": "#4a3aa7", "marker": ".", "label": "Losing trade"}
FAILURE_STYLE = {
    "ruin_threshold": {"color": "#ec835a", "marker": "v", "label": "Ruin threshold"},
    "liquidation": {"color": "#d03b3b", "marker": "X", "label": "Liquidation"},
    "grid_infeasible": {"color": "#9b4f96", "marker": "P", "label": "Grid infeasible"},
}
SURVIVED = {"color": "#2a78d6", "marker": "o", "label": "Survived"}
RESET_RUN = {"color": "#d47a16", "marker": "D", "label": "Failed, reset & continued"}

# Above this many losing trades the price panel becomes an ink blot and the
# render crawls, so it draws an evenly spaced subsample - and says so.
MAX_LOSS_MARKERS = 20_000


def _decimate(times: np.ndarray, low: np.ndarray, high: np.ndarray,
              close: np.ndarray, max_points: int = 3_000):
    """Bucket a long series down to a drawable width, keeping the extremes.

    A stride would drop the spikes that killed the accounts, which are the
    whole point of the chart; per-bucket min/max keeps them.
    """
    count = len(close)
    if count <= max_points:
        return times, low, high, close

    edges = np.unique(np.linspace(0, count, max_points + 1).astype(int))
    starts, stops = edges[:-1], edges[1:]
    keep = stops > starts
    starts, stops = starts[keep], stops[keep]

    return (
        times[starts],
        np.minimum.reduceat(low, starts),
        np.maximum.reduceat(high, starts),
        close[stops - 1],
    )


def _style_axis(axis, ylabel: str):
    axis.set_facecolor(SURFACE)
    axis.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    axis.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    axis.set_axisbelow(True)
    axis.tick_params(colors=INK_MUTED, labelsize=8)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)


def _failure_points(runs: Sequence[dict], bars) -> dict:
    """Group every account-level failure by cause into time/price arrays."""
    grouped: dict[str, dict] = {}
    for run in runs:
        events = [
            event for event in run.get("grid_failures", [])
            if event.get("reason") != "grid_break"
        ]
        if not events and run.get("ruined"):
            # Compatibility with reports created before reset/event tracking.
            reason = run.get("exit_reason", "liquidation")
            price = run.get("failure_price")
            if price is None or not np.isfinite(price):
                price = float(bars.close[run["end_index"]])
            events = [{
                "reason": reason,
                "time": run["end_time"],
                "price": price,
            }]
        for event in events:
            reason = event["reason"]
            bucket = grouped.setdefault(reason, {"time": [], "price": []})
            bucket["time"].append(event["time"])
            bucket["price"].append(float(event["price"]))
    return grouped


def _loss_points(trades: Sequence[dict]) -> dict:
    """Every cycle that closed at a loss, at the time and price it closed."""
    losses = [trade for trade in trades if trade["net_pnl"] < 0.0]
    return {
        "time": [trade["closed_at"] for trade in losses],
        "price": [float(trade["exit_price"]) for trade in losses],
        "net_pnl": [float(trade["net_pnl"]) for trade in losses],
        "full_ladder": sum(bool(trade.get("full_layers")) for trade in losses),
        "count": len(losses),
    }


def _subsample(values: Sequence, limit: int):
    """Evenly spaced subsample, deterministic and order preserving."""
    if len(values) <= limit:
        return list(values), False
    keep = np.linspace(0, len(values) - 1, limit).astype(int)
    return [values[index] for index in keep], True


def _bucket_counts(times, edges) -> np.ndarray:
    if not len(times):
        return np.zeros(len(edges) - 1)
    counts, _ = np.histogram(
        mdates.date2num(pd.to_datetime(list(times))), bins=mdates.date2num(edges)
    )
    return counts


def _safe_time_name(value) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    name = timestamp.strftime("%Y%m%d_%H%M%S")
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", name)


def _trade_identity(trade: dict, position: int) -> str:
    run_id = trade.get("run_id", trade.get("monte_carlo_run_id", "run"))
    account = trade.get("account_sequence", "acct")
    trade_id = trade.get("index", position + 1)
    return f"run{run_id}_acct{account}_trade{trade_id}"


def _event_path(directory: str, trade: dict, position: int) -> str:
    base = f"{_safe_time_name(trade['closed_at'])}_{_trade_identity(trade, position)}"
    return os.path.join(directory, f"{base}.png")


def _format_duration(start, end) -> str:
    seconds = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()
    if not np.isfinite(seconds) or seconds < 0.0:
        return "n/a"
    if seconds < 60.0:
        return f"{seconds:.0f}s"
    minutes = seconds / 60.0
    if minutes < 60.0:
        return f"{minutes:.1f}m"
    hours = minutes / 60.0
    if hours < 48.0:
        return f"{hours:.2f}h"
    days = hours / 24.0
    return f"{days:.2f}d"


def plot_failures(
    logger: logging.Logger,
    bars,
    result: dict,
    path: str,
    *,
    title: Optional[str] = None,
    max_points: int = 3_000,
    max_loss_markers: int = MAX_LOSS_MARKERS,
    buckets: int = 60,
) -> Optional[str]:
    """Draw the market with every losing trade and account failure marked."""
    runs = result.get("monte_carlo_runs", result.get("runs", []))
    if not runs:
        logger.warning(f"No Monte Carlo runs to plot, skipping {path}")
        return None

    times = pd.to_datetime(bars.time)
    plot_time, plot_low, plot_high, plot_close = _decimate(
        times.to_numpy(), bars.low, bars.high, bars.close, max_points
    )
    plot_time = pd.to_datetime(plot_time)

    losses = _loss_points(result.get("trades", []))
    failures = _failure_points(runs, bars)
    failed_total = sum(len(group["time"]) for group in failures.values())

    figure, (price_axis, span_axis, loss_axis, fail_axis) = plt.subplots(
        4, 1,
        figsize=(15, 12.5),
        height_ratios=(3.0, 1.9, 0.95, 0.95),
        sharex=True,
        constrained_layout=True,
    )
    figure.patch.set_facecolor(SURFACE)

    # ---------- 1. the market, with both kinds of loss on it ----------
    _style_axis(price_axis, "Market price")
    price_axis.fill_between(
        plot_time, plot_low, plot_high,
        color=PRICE_BAND, linewidth=0, alpha=0.9, zorder=1,
    )
    price_axis.plot(plot_time, plot_close, color=PRICE_LINE, linewidth=0.9, zorder=2)

    handles = []
    if losses["count"]:
        shown_time, trimmed = _subsample(losses["time"], max_loss_markers)
        shown_price, _ = _subsample(losses["price"], max_loss_markers)
        price_axis.scatter(
            pd.to_datetime(shown_time), shown_price,
            s=9, marker=LOSS_STYLE["marker"], color=LOSS_STYLE["color"],
            alpha=0.30, linewidths=0, zorder=3,
        )
        label = (
            f"{LOSS_STYLE['label']} ({losses['count']}, "
            f"{losses['full_ladder']} at full ladder)"
        )
        if trimmed:
            logger.info(
                f"Price panel shows {len(shown_time)} of {losses['count']} "
                f"losing trades (evenly spaced subsample)"
            )
            label += f" — showing {len(shown_time)}"
        handles.append(
            Line2D([], [], color=LOSS_STYLE["color"], marker="o", linestyle="none",
                   markersize=5, label=label)
        )

    # Failures go on top: two orders of magnitude rarer, and terminal.
    for reason in ("ruin_threshold", "liquidation", "grid_infeasible"):
        group = failures.get(reason)
        if not group:
            continue
        style = FAILURE_STYLE[reason]
        price_axis.scatter(
            pd.to_datetime(group["time"]), group["price"],
            s=70, marker=style["marker"], color=style["color"],
            edgecolors=SURFACE, linewidths=0.8, zorder=4,
        )
        handles.append(
            Line2D([], [], color=style["color"], marker=style["marker"],
                   linestyle="none", markersize=8,
                   label=f"{style['label']} ({len(group['time'])})")
        )
    if handles:
        price_axis.legend(
            handles=handles, loc="upper left", frameon=False,
            fontsize=9, labelcolor=INK_MUTED,
        )
    price_axis.set_title(
        title or "Martingale losses and account failures on the price series",
        color=INK, fontsize=12, loc="left", pad=8,
    )
    aggregate = result.get("aggregate", {})
    price_axis.text(
        0.995, 0.02,
        f"{aggregate.get('monte_carlo_runs', len(runs))} MC paths · "
        f"{aggregate.get('runs', len(runs))} strategy runs · "
        f"{failed_total} failed ({aggregate.get('ruin_rate', 0.0) * 100:.1f}%) · "
        f"{aggregate.get('total_trades', 0)} trades",
        transform=price_axis.transAxes, ha="right", va="bottom",
        color=INK_MUTED, fontsize=9,
    )

    # ---------- 2. one segment per run ----------
    _style_axis(span_axis, "Run (by entry)")
    ordered = sorted(runs, key=lambda run: pd.Timestamp(run["start_time"]))
    observed = {}
    for row, run in enumerate(ordered):
        had_failure = bool(run.get("grid_failure_count", int(run.get("ruined", False))))
        style = RESET_RUN if had_failure else SURVIVED
        observed[style["label"]] = style
        start, end = pd.Timestamp(run["start_time"]), pd.Timestamp(run["end_time"])
        span_axis.plot(
            [start, end], [row, row],
            color=style["color"], linewidth=1.1, alpha=0.65, solid_capstyle="butt",
        )
        span_axis.plot(
            [end], [row],
            marker=style["marker"], color=style["color"],
            markersize=4.5, markeredgewidth=0.0,
        )
    span_axis.set_ylim(-1, len(ordered))
    # Only outcomes that actually happened: an unused legend entry reads as a
    # category with zero observations rather than one that never applied.
    span_axis.legend(
        handles=[
            Line2D([], [], color=style["color"], marker=style["marker"],
                   linewidth=1.6, markersize=6, label=label)
            for label, style in observed.items()
        ],
        loc="upper left", frameon=False, fontsize=9,
        ncol=len(observed), labelcolor=INK_MUTED,
    )

    # ---------- 3 & 4. counts per bucket, as small multiples ----------
    edges = pd.date_range(times.min(), times.max(), periods=buckets + 1)
    centres = mdates.date2num(edges[:-1])
    width = (edges[1] - edges[0]).total_seconds() / 86400.0 * 0.9

    _style_axis(loss_axis, "Losing trades")
    loss_axis.bar(
        centres + width / 2, _bucket_counts(losses["time"], edges),
        width=width, color=LOSS_STYLE["color"], linewidth=0.8, edgecolor=SURFACE,
    )

    _style_axis(fail_axis, "Account failures")
    bottom = np.zeros(len(edges) - 1)
    for reason in ("ruin_threshold", "liquidation", "grid_infeasible"):
        group = failures.get(reason)
        if not group:
            continue
        style = FAILURE_STYLE[reason]
        counts = _bucket_counts(group["time"], edges)
        fail_axis.bar(
            centres + width / 2, counts, width=width, bottom=bottom,
            color=style["color"], linewidth=0.8, edgecolor=SURFACE,
        )
        bottom = bottom + counts
    fail_axis.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(mdates.AutoDateLocator())
    )

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    figure.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(figure)
    logger.info(
        f"Loss/failure chart saved to {path} "
        f"({losses['count']} losing trades, {failed_total} account failures)"
    )
    return path



def _equity_plot_points(curve: dict, bars, max_points: int):
    """Bound plotting cost while preserving every reset discontinuity."""
    indices = np.asarray(curve["indices"], dtype=int)
    values = np.asarray(curve.get("balances", curve["values"]), dtype=float)
    if len(indices) != len(values):
        raise ValueError("equity curve indices and values must have equal length")
    if len(indices) <= max_points:
        keep = np.arange(len(indices))
    else:
        keep = set(np.linspace(0, len(indices) - 1, max_points).astype(int))
        # A reset is represented by two adjacent values at one bar. Keep both
        # ends even when the surrounding long path is decimated.
        repeated = np.flatnonzero(indices[1:] == indices[:-1]) + 1
        for position in repeated:
            keep.update((position - 1, position))
        keep = np.asarray(sorted(keep), dtype=int)
    return pd.to_datetime(bars.time[indices[keep]]), values[keep]


def _numbered_path(path: str, part: int, total_parts: int) -> str:
    if total_parts == 1:
        return path
    root, extension = os.path.splitext(path)
    return f"{root}_{part:03d}{extension or '.png'}"


def plot_equity_paths(
    logger: logging.Logger,
    bars,
    result: dict,
    path: str,
    *,
    curves_per_plot: int,
    title: Optional[str] = None,
    max_points: int = 3_000,
) -> list[str]:
    """Plot every Monte Carlo equity path against the same market timeline.

    Paths are split into deterministic run-id groups so no simulation is
    omitted. Each Grid Failure is visible both as an X on the market and as a
    vertical jump from failed equity back to initial equity.
    """
    if curves_per_plot < 1:
        raise ValueError("curves_per_plot must be >= 1")
    curves = sorted(result.get("equity_curves", []), key=lambda row: row["run_id"])
    path_summaries = result.get("monte_carlo_runs", result.get("runs", []))
    summaries = {run["run_id"]: run for run in path_summaries}
    if not curves:
        logger.warning(f"No equity curves to plot, skipping {path}")
        return []

    groups = [
        curves[offset:offset + curves_per_plot]
        for offset in range(0, len(curves), curves_per_plot)
    ]
    written = []
    color_map = plt.get_cmap("turbo")

    for part, group in enumerate(groups, start=1):
        output_path = _numbered_path(path, part, len(groups))
        colors = color_map(np.linspace(0.04, 0.96, len(group)))
        first_index = min(int(np.min(curve["indices"])) for curve in group)
        last_index = max(int(np.max(curve["indices"])) for curve in group)
        market_time, market_low, market_high, market_close = _decimate(
            bars.time[first_index:last_index + 1],
            bars.low[first_index:last_index + 1],
            bars.high[first_index:last_index + 1],
            bars.close[first_index:last_index + 1],
            max_points,
        )
        market_time = pd.to_datetime(market_time)

        figure, price_axis = plt.subplots(
            figsize=(16, 8.5),
            constrained_layout=True,
        )
        figure.patch.set_facecolor(SURFACE)
        _style_axis(price_axis, "Market price")
        equity_axis = price_axis.twinx()
        _style_axis(equity_axis, "Trading equity / initial")
        equity_axis.yaxis.set_label_position("right")
        equity_axis.yaxis.tick_right()
        equity_axis.spines["left"].set_visible(False)
        equity_axis.spines["right"].set_visible(True)
        equity_axis.spines["right"].set_color(GRID)
        equity_axis.patch.set_visible(False)
        price_axis.set_zorder(1)
        equity_axis.set_zorder(2)
        price_axis.fill_between(
            market_time, market_low, market_high,
            color=PRICE_BAND, linewidth=0, alpha=0.75, zorder=1,
        )
        price_axis.plot(
            market_time, market_close,
            color=PRICE_LINE, linewidth=0.85, zorder=2,
        )

        initial_equity = summaries[group[0]["run_id"]]["initial_equity"]
        max_multiple = 1.0
        min_multiple = 1.0

        for curve, color in zip(group, colors):
            run_id = curve["run_id"]
            summary = summaries[run_id]
            curve_time, equity = _equity_plot_points(curve, bars, max_points)
            run_initial_equity = float(summary["initial_equity"])
            if run_initial_equity <= 0.0:
                raise ValueError("initial_equity must be positive for equity plot")
            equity_multiple = equity / run_initial_equity
            failures = summary.get("grid_failures", [])
            if len(equity_multiple):
                max_multiple = max(max_multiple, float(np.nanmax(equity_multiple)))
                min_multiple = min(min_multiple, float(np.nanmin(equity_multiple)))
            label = f"Run {run_id} ({len(failures)} failures)"
            equity_axis.plot(
                curve_time, equity_multiple,
                color=color, linewidth=1.05, alpha=0.88, label=label,
            )
            if failures:
                failure_time = pd.to_datetime([item["time"] for item in failures])
                failure_equity = [
                    item["equity_before_reset"] / run_initial_equity
                    for item in failures
                ]
                reset_equity = [
                    item["reset_equity"] / run_initial_equity
                    for item in failures
                ]
                max_multiple = max(
                    max_multiple, max(failure_equity), max(reset_equity)
                )
                min_multiple = min(
                    min_multiple, min(failure_equity), min(reset_equity)
                )
                equity_axis.scatter(
                    failure_time, failure_equity,
                    color=[color], marker="X", s=42,
                    edgecolors=SURFACE, linewidths=0.6, zorder=4,
                )
                equity_axis.scatter(
                    failure_time, reset_equity,
                    color=[color], marker="D", s=30,
                    edgecolors=SURFACE, linewidths=0.6, zorder=4,
                )
                price_axis.scatter(
                    failure_time,
                    [item["price"] for item in failures],
                    color=[color], marker="X", s=38,
                    edgecolors=SURFACE, linewidths=0.6, zorder=4,
                )
                for item, when in zip(failures, failure_time):
                    label_text = f"GF{item['sequence']} {item['reason']}"
                    equity_axis.annotate(
                        label_text,
                        xy=(when, item["equity_before_reset"] / run_initial_equity),
                        xytext=(5, 8),
                        textcoords="offset points",
                        color=color,
                        fontsize=7,
                        ha="left",
                        va="bottom",
                        arrowprops={
                            "arrowstyle": "-",
                            "color": color,
                            "linewidth": 0.45,
                            "alpha": 0.8,
                        },
                    )

        equity_axis.axhline(
            1.0,
            color=INK_MUTED, linestyle="--", linewidth=0.8, alpha=0.6,
        )
        top = max(1.05, max_multiple * 1.08)
        bottom = max(0.0, min(0.95, min_multiple * 0.92))
        equity_axis.set_ylim(bottom=bottom, top=top)
        equity_axis.legend(
            loc="upper left", frameon=False, fontsize=8,
            ncol=min(3, len(group)), labelcolor=INK_MUTED,
        )
        equity_axis.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(mdates.AutoDateLocator())
        )
        price_axis.set_title(
            (title or "Monte Carlo equity paths over market price")
            + (f" — {part}/{len(groups)}" if len(groups) > 1 else ""),
            color=INK, fontsize=12, loc="left", pad=8,
        )

        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        figure.savefig(output_path, dpi=140, facecolor=SURFACE)
        plt.close(figure)
        written.append(output_path)

    logger.info(
        f"Equity/market charts saved: {len(written)} files, "
        f"{len(curves)} paths, up to {curves_per_plot} paths per file"
    )
    return written


ADVERSE_BOUNDARY_KINDS = {"adverse", "grid_break"}


def _minimum_visible_cmap(name: str, source: str, low: float = 0.42):
    """Use the darker part of a sequential cmap so single hits remain visible."""
    base = plt.get_cmap(source)
    return LinearSegmentedColormap.from_list(
        name,
        base(np.linspace(low, 1.0, 256)),
    )


def _adverse_boundary_points(trades: Sequence[dict]) -> pd.DataFrame:
    """One row per layer whose own adverse boundary was reached."""
    rows = []
    for trade in trades:
        for fill in trade.get("fills", ()):
            if fill.get("boundary_kind") not in ADVERSE_BOUNDARY_KINDS:
                continue
            boundary_bar = fill.get("boundary_bar")
            boundary_time = fill.get("boundary_time")
            if boundary_bar is None or boundary_time is None:
                continue
            rows.append({
                "run_id": trade.get("run_id"),
                "trade_index": trade.get("index"),
                "layer": int(fill["layer"]),
                "boundary_bar": int(boundary_bar),
                "boundary_time": pd.Timestamp(boundary_time),
                "boundary_price": float(fill.get(
                    "boundary_price",
                    fill.get("execution_next_adverse_price", np.nan),
                )),
                "boundary_kind": fill.get("boundary_kind"),
                "bars_held": float(fill.get("boundary_bars_held", np.nan)),
            })
    return pd.DataFrame(rows)


def plot_adverse_boundaries(
    logger: logging.Logger,
    bars,
    result: dict,
    path: str,
    *,
    title: Optional[str] = None,
    max_points: int = 3_000,
    density_bins: int = 240,
) -> Optional[str]:
    """Market chart plus one horizontal timeline per layer's adverse hits.

    Each dot is the first time that layer's own adverse boundary was reached.
    Dot colour is the count of hits for the same layer inside the same time
    bucket, so denser clusters appear darker.
    """
    points = _adverse_boundary_points(result.get("trades", ()))
    if points.empty:
        logger.warning(f"No adverse boundary hits to plot, skipping {path}")
        return None

    first_index = max(0, int(points["boundary_bar"].min()))
    last_index = min(len(bars) - 1, int(points["boundary_bar"].max()))
    market_time, market_low, market_high, market_close = _decimate(
        bars.time[first_index:last_index + 1],
        bars.low[first_index:last_index + 1],
        bars.high[first_index:last_index + 1],
        bars.close[first_index:last_index + 1],
        max_points,
    )
    market_time = pd.to_datetime(market_time)

    layers = np.array(sorted(points["layer"].unique()), dtype=int)
    layer_to_y = {layer: offset for offset, layer in enumerate(layers, start=1)}
    y_values = points["layer"].map(layer_to_y).to_numpy(dtype=float)

    bin_count = max(1, min(density_bins, max(1, last_index - first_index + 1)))
    bin_edges = np.linspace(first_index, last_index + 1, bin_count + 1)
    time_bins = np.clip(
        np.digitize(points["boundary_bar"].to_numpy(dtype=float), bin_edges) - 1,
        0,
        bin_count - 1,
    )
    density_keys = list(zip(points["layer"].to_numpy(dtype=int), time_bins))
    density = {}
    for key in density_keys:
        density[key] = density.get(key, 0) + 1
    density_values = np.asarray([density[key] for key in density_keys], dtype=float)
    density_cmap = _minimum_visible_cmap("adverse_density", "Blues", low=0.46)
    density_norm = Normalize(
        vmin=0.0,
        vmax=max(1.0, float(density_values.max())),
    )

    figure, (price_axis, layer_axis) = plt.subplots(
        2, 1,
        figsize=(16, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, max(1.3, min(4.0, len(layers) * 0.28))]},
        constrained_layout=True,
    )
    figure.patch.set_facecolor(SURFACE)
    _style_axis(price_axis, "Market price")
    _style_axis(layer_axis, "Adverse boundary layer")

    price_axis.fill_between(
        market_time, market_low, market_high,
        color=PRICE_BAND, linewidth=0, alpha=0.75, zorder=1,
    )
    price_axis.plot(
        market_time, market_close,
        color=PRICE_LINE, linewidth=0.85, zorder=2,
    )
    for layer, y in layer_to_y.items():
        layer_axis.hlines(
            y,
            market_time[0],
            market_time[-1],
            color=GRID,
            linewidth=0.9,
            zorder=1,
        )

    scatter = layer_axis.scatter(
        points["boundary_time"],
        y_values,
        c=density_values,
        cmap=density_cmap,
        norm=density_norm,
        s=13,
        alpha=0.96,
        linewidths=0.28,
        edgecolors="#08306b",
        zorder=2,
    )
    layer_axis.set_yticks([layer_to_y[layer] for layer in layers])
    layer_axis.set_yticklabels([f"L{layer}" for layer in layers])
    layer_axis.set_ylim(0.4, len(layers) + 0.6)
    layer_axis.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(mdates.AutoDateLocator())
    )
    colorbar = figure.colorbar(scatter, ax=layer_axis, pad=0.01, fraction=0.035)
    colorbar.set_label("Hits in same layer/time bucket", color=INK_MUTED, fontsize=8)
    colorbar.ax.tick_params(colors=INK_MUTED, labelsize=8)

    kind_counts = points["boundary_kind"].value_counts().to_dict()
    price_axis.set_title(
        (title or "Adverse boundary hits by layer")
        + f" | hits {len(points)} | kinds {kind_counts}",
        color=INK,
        fontsize=12,
        loc="left",
        pad=8,
    )

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    figure.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(figure)
    logger.info(
        f"Adverse boundary chart saved to {path} "
        f"({len(points)} hits across {len(layers)} layers)"
    )
    return path


def plot_grid_break_events(
    logger: logging.Logger,
    bars,
    result: dict,
    directory: str,
    *,
    context_bars: int = 200,
    title_prefix: Optional[str] = None,
) -> Optional[str]:
    """Write one market-window chart for every realized full-grid break."""
    trades = [
        trade for trade in result.get("trades", [])
        if trade.get("reason") == "grid_break"
    ]
    if not trades:
        logger.warning(f"No grid breaks to plot, skipping {directory}")
        return None
    os.makedirs(directory, exist_ok=True)

    times = pd.to_datetime(bars.time)
    written = 0
    for position, trade in enumerate(trades):
        entry_bar = int(trade.get("entry_bar", trade.get("exit_bar", 0)))
        exit_bar = int(trade.get("exit_bar", entry_bar))
        if exit_bar < 0 or entry_bar >= len(bars.close):
            continue
        entry_bar = max(entry_bar, 0)
        exit_bar = min(exit_bar, len(bars.close) - 1)
        start = max(0, entry_bar - context_bars)
        stop = min(len(bars.close), exit_bar + context_bars + 1)
        if stop <= start:
            continue

        window_time = times[start:stop]
        output_path = _event_path(directory, trade, position)
        figure, axis = plt.subplots(figsize=(14, 6.8), constrained_layout=True)
        figure.patch.set_facecolor(SURFACE)
        _style_axis(axis, "Market price")
        axis.fill_between(
            window_time,
            bars.low[start:stop],
            bars.high[start:stop],
            color=PRICE_BAND,
            linewidth=0,
            alpha=0.85,
            zorder=1,
            label="high-low",
        )
        axis.plot(
            window_time,
            bars.close[start:stop],
            color=PRICE_LINE,
            linewidth=1.1,
            zorder=2,
            label="close",
        )

        entry_time = times[entry_bar]
        exit_time = times[exit_bar]
        duration = _format_duration(
            trade.get("opened_at", entry_time),
            trade.get("closed_at", exit_time),
        )
        bars_held = trade.get("bars_held", exit_bar - entry_bar + 1)
        axis.axvspan(entry_time, exit_time, color="#f3b35f", alpha=0.16, zorder=0)
        axis.axvline(entry_time, color="#2a78d6", linewidth=1.2, alpha=0.85)
        axis.axvline(exit_time, color="#d03b3b", linewidth=1.2, alpha=0.9)

        fills = trade.get("fills") or []
        fill_times = []
        fill_prices = []
        fill_layers = []
        for fill in fills:
            bar = int(fill.get("bar", entry_bar))
            if start <= bar < stop:
                price = float(fill.get("price", np.nan))
                if not np.isfinite(price):
                    continue
                fill_times.append(times[bar])
                fill_prices.append(price)
                fill_layers.append(int(fill.get("layer", 0)))
        if fill_times:
            axis.scatter(
                fill_times,
                fill_prices,
                s=26,
                color="#2a78d6",
                edgecolors=SURFACE,
                linewidths=0.6,
                zorder=4,
                label="layer fills",
            )
            for fill_time, fill_price, layer in zip(fill_times, fill_prices, fill_layers):
                axis.annotate(
                    f"L{layer}",
                    (fill_time, fill_price),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=7,
                    color=INK_MUTED,
                )

        axis.scatter(
            [entry_time],
            [float(trade.get("first_entry_price", bars.close[entry_bar]))],
            s=58,
            marker="o",
            color="#2a78d6",
            edgecolors=SURFACE,
            linewidths=0.9,
            zorder=5,
            label="base entry",
        )
        axis.scatter(
            [exit_time],
            [float(trade.get("exit_price", bars.close[exit_bar]))],
            s=72,
            marker="X",
            color="#d03b3b",
            edgecolors=SURFACE,
            linewidths=0.9,
            zorder=5,
            label="grid break",
        )

        axis.set_title(
            (
                f"{title_prefix + ' | ' if title_prefix else ''}"
                f"grid break {pd.Timestamp(trade['closed_at'])} "
                f"| {trade.get('direction', '')} "
                f"| held {bars_held} bars / {duration} "
                f"| PnL {float(trade.get('net_pnl', 0.0)):.2f}"
            ),
            color=INK,
            fontsize=12,
            loc="left",
            pad=8,
        )
        axis.text(
            0.995,
            0.02,
            f"context {context_bars} bars · "
            f"entry→break {duration} · "
            f"{pd.Timestamp(entry_time)} → {pd.Timestamp(exit_time)}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            color=INK_MUTED,
            fontsize=8,
        )
        axis.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK_MUTED)
        locator = mdates.AutoDateLocator()
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

        figure.savefig(output_path, dpi=140, facecolor=SURFACE)
        plt.close(figure)
        written += 1

    logger.info(f"Grid-break event charts saved: {written} files in {directory}")
    return directory


# ============================================================
# Where and at what scale the grid breaks pile up
# ============================================================
# Diverging pair for the pressure map: blue (calmer than random) through a
# neutral gray midpoint to red (more grid breaks than random).  Both arms are
# monotone in lightness and step-matched, so neither pole dominates the other.
DIVERGING_LOW = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
DIVERGING_MID = "#f0efec"
DIVERGING_HIGH = ["#fbd7d5", "#f4b0ac", "#ec8a85", "#e34948", "#bf3232", "#952727", "#6b1c1c"]
# Single-hue ramp for a magnitude (probability) heatmap.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# Ordinal steps for the quantile curves: they are an ordered family, so they
# take one hue by lightness rather than four unrelated colours.
ORDINAL = ["#86b6ef", "#3987e5", "#256abf", "#0d366b"]


def _diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "pressure", DIVERGING_LOW + [DIVERGING_MID] + DIVERGING_HIGH
    )


def _sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("density", SEQUENTIAL)


def scale_grid(min_bars: int, max_bars: int, count: int) -> np.ndarray:
    """Geometric ladder of window sizes, in bars, with no repeats."""
    if max_bars <= min_bars:
        return np.array([max(1, min_bars)], dtype=int)
    grid = np.geomspace(max(1, min_bars), max_bars, count)
    return np.unique(np.round(grid).astype(int))


def _loss_and_exposure(runs: Sequence[dict], trades: Sequence[dict], total_bars: int):
    """Per-bar count of grid breaks, and how many runs were trading that bar.

    Exposure matters: runs start at random bars and die, so a period that simply
    had more accounts alive would otherwise look like a period of more danger.
    """
    losses = np.zeros(total_bars, dtype=float)
    for trade in trades:
        if trade["net_pnl"] < 0.0:
            index = int(trade["exit_bar"])
            if 0 <= index < total_bars:
                losses[index] += 1.0

    # Difference array: +1 at each run's first bar, -1 just past its last.
    edges = np.zeros(total_bars + 1, dtype=float)
    for run in runs:
        start = max(0, int(run["start_index"]))
        stop = min(total_bars, int(run["end_index"]) + 1)
        if stop > start:
            edges[start] += 1.0
            edges[stop] -= 1.0
    return losses, np.cumsum(edges[:-1])


def _rolling_sums(values: np.ndarray, centres: np.ndarray, window: int):
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    half = window // 2
    starts = np.clip(centres - half, 0, len(values) - 1)
    stops = np.clip(starts + window, 0, len(values))
    starts = np.clip(stops - window, 0, len(values))
    return cumulative[stops] - cumulative[starts]


def pressure_field(
    runs: Sequence[dict],
    trades: Sequence[dict],
    total_bars: int,
    scales: np.ndarray,
    columns: int = 900,
):
    """Z(t, m): grid breaks in a window of m bars, in sigmas above random.

    The null is a Poisson process whose rate is the pooled losses per
    active-run-bar, so Z compares each window against what that much trading
    would have produced if losses were scattered uniformly.
    """
    losses, exposure = _loss_and_exposure(runs, trades, total_bars)
    total_exposure = exposure.sum()
    if total_exposure <= 0.0 or losses.sum() <= 0.0:
        return None

    rate = losses.sum() / total_exposure
    centres = np.unique(np.linspace(0, total_bars - 1, min(columns, total_bars)).astype(int))
    field = np.full((len(scales), len(centres)), np.nan)
    for row, window in enumerate(scales):
        observed = _rolling_sums(losses, centres, int(window))
        expected = rate * _rolling_sums(exposure, centres, int(window))
        usable = expected > 1e-9
        field[row, usable] = (observed[usable] - expected[usable]) / np.sqrt(expected[usable])
    return {
        "z": field,
        "centres": centres,
        "scales": scales,
        "exposure": exposure,
        "rate": rate,
    }


def plot_pressure_heatmap(
    logger: logging.Logger,
    bars,
    result: dict,
    path: str,
    *,
    title: Optional[str] = None,
    min_scale: int = 8,
    max_scale: Optional[int] = None,
    scale_count: int = 40,
    columns: int = 900,
) -> Optional[str]:
    """Time x window-scale map of standardized stop-loss pressure."""
    runs, trades = result.get("runs", []), result.get("trades", [])
    total_bars = len(bars)
    if not runs or not trades:
        logger.warning(f"Nothing to map, skipping {path}")
        return None

    scales = scale_grid(min_scale, max_scale or max(min_scale * 2, total_bars // 8), scale_count)
    field = pressure_field(runs, trades, total_bars, scales, columns)
    if field is None:
        logger.warning(f"No grid breaks to map, skipping {path}")
        return None

    times = pd.to_datetime(bars.time)
    x = times[field["centres"]]
    finite = field["z"][np.isfinite(field["z"])]
    # Symmetric limits so the neutral midpoint really sits at Z = 0, clipped at
    # a high percentile so one extreme window cannot flatten the whole map.
    limit = float(np.percentile(np.abs(finite), 99)) if len(finite) else 1.0
    limit = max(limit, 1.0)

    figure, (exposure_axis, heat_axis) = plt.subplots(
        2, 1, figsize=(15, 7.5), height_ratios=(1.0, 4.0),
        sharex=True, constrained_layout=True,
    )
    figure.patch.set_facecolor(SURFACE)

    _style_axis(exposure_axis, "Runs active")
    exposure_axis.fill_between(
        times[field["centres"]], field["exposure"][field["centres"]],
        color=PRICE_BAND, linewidth=0,
    )
    exposure_axis.set_title(
        title or "Stop-loss pressure by time and window scale",
        color=INK, fontsize=12, loc="left", pad=8,
    )

    _style_axis(heat_axis, "Window m (bars, log)")
    heat_axis.grid(False)
    mesh = heat_axis.pcolormesh(
        x, field["scales"], np.ma.masked_invalid(field["z"]),
        cmap=_diverging_cmap(), vmin=-limit, vmax=limit, shading="nearest",
    )
    heat_axis.set_yscale("log")
    heat_axis.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(mdates.AutoDateLocator())
    )
    bar = figure.colorbar(mesh, ax=heat_axis, pad=0.01)
    bar.set_label(
        "Z: sigmas above a random process of the same rate",
        color=INK_MUTED, fontsize=9,
    )
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    bar.outline.set_edgecolor(GRID)
    heat_axis.text(
        0.995, 0.03,
        f"{int(sum(t['net_pnl'] < 0 for t in trades))} grid breaks · "
        f"rate {field['rate'] * 1000:.2f} per 1000 run-bars · "
        f"|Z| clipped at {limit:.1f}",
        transform=heat_axis.transAxes, ha="right", va="bottom",
        color=INK_MUTED, fontsize=9,
    )

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    figure.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(figure)
    logger.info(f"Pressure heatmap saved to {path}")
    return path


# ============================================================
# How small clusters merge into big ones as the gap allowance grows
# ============================================================
def cluster_sizes(positions: Sequence[float], max_gap: float) -> np.ndarray:
    """Single-linkage cluster sizes: split wherever the gap exceeds max_gap."""
    ordered = np.sort(np.asarray(positions, dtype=float))
    if not len(ordered):
        return np.array([], dtype=int)
    breaks = np.flatnonzero(np.diff(ordered) > max_gap) + 1
    return np.diff(np.concatenate(([0], breaks, [len(ordered)]))).astype(int)


def cluster_size_by_scale(trades: Sequence[dict], scales: np.ndarray) -> dict:
    """Pool cluster sizes over every run, once per scale.

    Clustering is done inside a run: two accounts being stopped out at the same
    moment are not one cluster, they are the same market event hitting twice.
    """
    positions_by_run: dict[int, list] = {}
    for trade in trades:
        if trade["net_pnl"] < 0.0:
            positions_by_run.setdefault(trade.get("run_id", 0), []).append(
                trade["exit_bar"] - trade.get("start_index", 0)
            )
    runs = [np.sort(np.asarray(values, dtype=float)) for values in positions_by_run.values()]
    if not runs:
        return {}

    sizes_by_scale, summary = {}, []
    for scale in scales:
        sizes = np.concatenate([cluster_sizes(run, float(scale)) for run in runs])
        sizes_by_scale[int(scale)] = sizes
        summary.append({
            "m": int(scale),
            "clusters": int(len(sizes)),
            "median": float(np.median(sizes)),
            "p75": float(np.percentile(sizes, 75)),
            "p95": float(np.percentile(sizes, 95)),
            "max": int(sizes.max()),
            "mean": float(sizes.mean()),
        })
    return {"sizes": sizes_by_scale, "curves": summary}


def plot_cluster_scales(
    logger: logging.Logger,
    result: dict,
    path: str,
    *,
    title: Optional[str] = None,
    min_scale: int = 1,
    max_scale: int = 5_000,
    scale_count: int = 40,
) -> Optional[str]:
    """Cluster-size distribution and quantile curves against the gap allowance."""
    trades = result.get("trades", [])
    scales = scale_grid(min_scale, max_scale, scale_count)
    analysis = cluster_size_by_scale(trades, scales)
    if not analysis:
        logger.warning(f"No grid breaks to cluster, skipping {path}")
        return None

    curves = pd.DataFrame(analysis["curves"])
    largest = max(int(sizes.max()) for sizes in analysis["sizes"].values())
    edges = np.unique(np.round(np.geomspace(1, max(largest, 2) + 1, 45)).astype(int))

    # Column-normalized: each m is its own distribution, so the panel answers
    # "what share of clusters have this size", not "which m had more losses".
    density = np.full((len(edges) - 1, len(scales)), np.nan)
    for column, scale in enumerate(scales):
        counts, _ = np.histogram(analysis["sizes"][int(scale)], bins=edges)
        total = counts.sum()
        if total:
            density[:, column] = counts / total

    figure, (heat_axis, curve_axis, count_axis) = plt.subplots(
        3, 1, figsize=(13, 11), height_ratios=(2.4, 1.8, 1.0),
        sharex=True, constrained_layout=True,
    )
    figure.patch.set_facecolor(SURFACE)

    _style_axis(heat_axis, "Cluster size (grid breaks)")
    heat_axis.grid(False)
    mesh = heat_axis.pcolormesh(
        scales, edges[:-1], np.ma.masked_invalid(density),
        cmap=_sequential_cmap(), shading="nearest",
    )
    heat_axis.set_yscale("log")
    heat_axis.set_title(
        title or "How grid breaks merge into clusters as the gap allowance grows",
        color=INK, fontsize=12, loc="left", pad=8,
    )
    bar = figure.colorbar(mesh, ax=heat_axis, pad=0.01)
    bar.set_label("Share of clusters at this m", color=INK_MUTED, fontsize=9)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    bar.outline.set_edgecolor(GRID)

    _style_axis(curve_axis, "Cluster size")
    series = (("median", "Median"), ("p75", "P75"), ("p95", "P95"), ("max", "Max"))
    endings = []
    for (column, label), colour in zip(series, ORDINAL):
        curve_axis.plot(
            curves["m"], curves[column], color=colour, linewidth=2.0, label=label,
        )
        endings.append([float(curves[column].iloc[-1]), label, colour])

    # Direct labels at the right end, decluttered: P95 and Max converge once m
    # swallows every gap, and stacked text at the same y is unreadable.
    endings.sort(key=lambda item: item[0])
    minimum_ratio = 1.18   # the axis is log, so separation is multiplicative
    for lower, upper in zip(endings, endings[1:]):
        if upper[0] < lower[0] * minimum_ratio:
            upper[0] = lower[0] * minimum_ratio
    for y, label, colour in endings:
        curve_axis.annotate(
            label, (curves["m"].iloc[-1], y),
            xytext=(8, 0), textcoords="offset points",
            color=colour, fontsize=9, va="center", annotation_clip=False,
        )
    curve_axis.set_xscale("log")
    curve_axis.set_yscale("log")
    curve_axis.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_MUTED)

    _style_axis(count_axis, "Clusters")
    count_axis.plot(curves["m"], curves["clusters"], color=ORDINAL[1], linewidth=2.0)
    count_axis.set_xscale("log")
    count_axis.set_yscale("log")
    count_axis.set_xlabel("Gap allowance m (bars, log)", color=INK_MUTED, fontsize=9)

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    figure.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(figure)
    logger.info(
        f"Cluster-scale chart saved to {path} "
        f"(m from {scales[0]} to {scales[-1]} bars, largest cluster {largest})"
    )
    return path


# ============================================================
# Realized grid breaks against the volatility that produced them
# ============================================================
BREACH_COLOR = "#d03b3b"
VOLATILITY_COLOR = "#4a3aa7"


def _bars_per_hour(periods_per_year: float) -> float:
    if periods_per_year <= 0.0 or not np.isfinite(periods_per_year):
        return 0.0
    return float(periods_per_year) / (365.25 * 24.0)


def _rank_correlation(left: pd.Series, right: pd.Series) -> float:
    """Spearman rho without scipy: ranks, then Pearson."""
    joined = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(joined) < 3:
        return float("nan")
    left_rank = joined["left"].rank().to_numpy()
    right_rank = joined["right"].rank().to_numpy()
    if np.std(left_rank) <= 0.0 or np.std(right_rank) <= 0.0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def rolling_annualized_volatility(bars, *, window_hours: float = 4.0):
    """Re-exported from the backtest module so charts and gates cannot drift."""
    from trade.sim.martingale_backtest import rolling_annualized_volatility as impl

    return impl(bars, window_hours=window_hours)


def _traded_window(trades, total_bars: int) -> Optional[tuple]:
    """First entry and last exit across the supplied trades, in bar indices.

    Monte Carlo paths start at random bars, so bars before the first entry
    cannot contain a break and must not be counted as calm observations: they
    would inflate the low-volatility buckets. One continuous full-history pass
    makes this span the whole series.
    """
    entries = []
    exits = []
    for trade in trades:
        entry = trade.get("entry_bar")
        exit_ = trade.get("exit_bar")
        if entry is not None and np.isfinite(float(entry)):
            entries.append(int(entry))
        if exit_ is not None and np.isfinite(float(exit_)):
            exits.append(int(exit_))
    if not entries or not exits:
        return None
    start = max(0, min(entries))
    end = min(total_bars - 1, max(exits))
    if end <= start:
        return None
    return start, end


def _breach_points(trades, times) -> pd.DataFrame:
    """Every realized full-grid break, one row per bar, ascending in time.

    Monte Carlo paths overlap, so the same ladder breaking inside the same
    bar is one event; the first record for that bar wins. That keeps the
    timeline honest without one run drawing the same point twice.
    """
    rows = []
    seen_bars = set()
    for trade in trades:
        if trade.get("reason") != "grid_break":
            continue
        bar = int(trade.get("exit_bar", -1))
        if bar < 0 or bar >= len(times) or bar in seen_bars:
            continue
        seen_bars.add(bar)
        rows.append({
            "bar": bar,
            "time": pd.Timestamp(times[bar]),
            "exit_price": float(trade.get("exit_price", np.nan)),
            "layers": int(trade.get("layers", 0)),
            "net_pnl": float(trade.get("net_pnl", 0.0)),
        })
    columns = ["bar", "time", "exit_price", "layers", "net_pnl"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("bar").reset_index(drop=True)


def breach_volatility_analysis(
    bars,
    trades,
    *,
    window_hours: float = 4.0,
    horizons_hours: Sequence[float] = (1.0, 4.0, 24.0),
    buckets: int = 10,
) -> dict:
    """Conditional grid-break probability per trailing-volatility bucket.

    For every bar the trailing volatility is measured on data that precedes
    it, then the bar is placed in one of ``buckets`` equal-count bins. The
    statistic is P(a grid break happens within the next N bars | this
    bucket). Buckets by rank instead of fixed cutoffs, because volatility is
    right-skewed and a quantile grid would leave empty cells.

    This is the per-bar counterpart of a monthly average: it removes the
    smoothing that makes a calm month with one spike look identical to a
    volatile month with one spike.
    """
    times = pd.to_datetime(bars.time)
    volatility, window_bars = rolling_annualized_volatility(
        bars, window_hours=window_hours
    )
    breaches = _breach_points(trades, times)

    mask = np.zeros(len(times), dtype=bool)
    if len(breaches):
        mask[breaches["bar"].to_numpy()] = True
    cumulative = np.concatenate(([0], np.cumsum(mask)))
    positions = np.arange(len(times))
    traded = np.zeros(len(times), dtype=bool)
    window = _traded_window(trades, len(times))
    if window is not None:
        traded[window[0]:window[1] + 1] = True
    bars_per_hour = _bars_per_hour(bars.periods_per_year)
    bucket_count = max(2, int(buckets))

    rows = []
    baseline_rows = []
    for hours in horizons_hours:
        horizon = max(1, int(round(hours * bars_per_hour))) if bars_per_hour else 1
        future = cumulative[np.minimum(positions + horizon, len(times))] - cumulative[positions]
        valid = (
            np.isfinite(volatility)
            & traded
            & (positions + horizon <= len(times))
        )
        valid_positions = np.flatnonzero(valid)
        if not len(valid_positions):
            continue
        indicator = (future[valid_positions] > 0).astype(float)
        baseline = float(indicator.mean())
        values = volatility[valid_positions]
        order = np.argsort(values, kind="stable")
        bucket_ids = np.empty(len(values), dtype=int)
        bucket_ids[order] = np.minimum(
            np.arange(len(values)) * bucket_count // len(values), bucket_count - 1
        )
        for bucket in range(bucket_count):
            selected = bucket_ids == bucket
            probability = float(indicator[selected].mean())
            rows.append({
                "horizon_hours": float(hours),
                "horizon_bars": horizon,
                "bucket": bucket + 1,
                "samples": int(selected.sum()),
                "vol_low_pct": float(values[selected].min() * 100.0),
                "vol_high_pct": float(values[selected].max() * 100.0),
                "mean_vol_pct": float(values[selected].mean() * 100.0),
                "breach_probability": probability,
                "baseline_probability": baseline,
                "lift": probability / baseline if baseline > 0.0 else float("nan"),
            })
        baseline_rows.append({
            "horizon_hours": float(hours),
            "horizon_bars": horizon,
            "samples": int(len(valid_positions)),
            "baseline_probability": baseline,
        })

    conditional = pd.DataFrame(rows)
    baselines = pd.DataFrame(baseline_rows)

    # Months are calendar buckets, not instants, so the timezone is dropped
    # explicitly instead of letting to_period() warn about it.
    month_times = times.tz_convert(None) if times.tz is not None else times
    month_index = month_times.to_period("M")
    monthly = pd.DataFrame({
        "bars": pd.Series(traded.astype(int), index=month_index).groupby(level=0).sum(),
        "breaches": pd.Series(mask.astype(int), index=month_index).groupby(level=0).sum(),
        "mean_vol_pct": pd.Series(
            np.where(traded, volatility, np.nan), index=month_index
        ).groupby(level=0).mean() * 100.0,
    })
    # Months fully outside the traded window would read as zero breaks at zero
    # volatility and drag the correlation to meaninglessness.
    monthly = monthly[monthly["bars"] > 0]
    monthly.index.name = "month"
    monthly = monthly.reset_index()
    monthly["month"] = monthly["month"].astype(str)
    monthly["breaches_per_1000_bars"] = (
        monthly["breaches"] / monthly["bars"].clip(lower=1) * 1000.0
    )
    monthly_rho = _rank_correlation(monthly["mean_vol_pct"], monthly["breaches"])

    breach_frame = breaches.copy()
    if len(breach_frame):
        finite = np.sort(volatility[np.isfinite(volatility)])
        breach_vol = volatility[breach_frame["bar"].to_numpy()]
        breach_frame["volatility_pct"] = breach_vol * 100.0
        breach_frame["volatility_percentile"] = (
            np.searchsorted(finite, breach_vol, side="right") / max(len(finite), 1)
        )

    return {
        "times": times,
        "volatility": volatility,
        "window_bars": window_bars,
        "window_hours": float(window_hours),
        "breaches": breach_frame,
        "conditional": conditional,
        "baselines": baselines,
        "monthly": monthly,
        "monthly_rho": monthly_rho,
    }


def plot_breach_timeline(
    logger: logging.Logger,
    bars,
    analysis: dict,
    path: str,
    *,
    title: Optional[str] = None,
    max_points: int = 3_000,
) -> Optional[str]:
    """Price, volatility and monthly break count on one shared time axis."""
    breaches = analysis["breaches"]
    if not len(breaches):
        logger.warning(f"No grid breaks to plot, skipping {path}")
        return None

    times = np.asarray(analysis["times"])
    price_times, price_low, price_high, price_close = _decimate(
        times,
        np.asarray(bars.low, dtype=float),
        np.asarray(bars.high, dtype=float),
        np.asarray(bars.close, dtype=float),
        max_points=max_points,
    )
    volatility = np.asarray(analysis["volatility"], dtype=float)
    stride = max(1, int(np.ceil(len(times) / max_points)))
    vol_times = times[::stride]
    vol_values = volatility[::stride]
    breach_times = times[breaches["bar"].to_numpy()]
    breach_vol = volatility[breaches["bar"].to_numpy()]
    finite = np.isfinite(breach_vol)
    # Deep ladders break a few hundred times and read as individual events;
    # a 2-layer ladder breaks often enough to paint a band, so its markers
    # shrink and go translucent while the price line stays on top.
    crowded = len(breaches) > 5_000
    marker_size = 4 if crowded else 9
    marker_alpha = 0.45 if crowded else 0.75

    figure, (price_axis, vol_axis, count_axis) = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"height_ratios": (2.2, 1.2, 0.9)}, constrained_layout=True,
    )
    figure.patch.set_facecolor(SURFACE)

    _style_axis(price_axis, "Market price")
    price_axis.fill_between(
        price_times, price_low, price_high,
        color=PRICE_BAND, linewidth=0, alpha=0.85, zorder=1,
    )
    price_axis.plot(
        price_times, price_close, color=PRICE_LINE, linewidth=1.0, zorder=5,
    )
    price_axis.scatter(
        breach_times, breaches["exit_price"], s=marker_size, marker="X",
        color=BREACH_COLOR, linewidths=0.0, alpha=marker_alpha, zorder=4,
        label=f"Grid break ({len(breaches)})",
    )
    price_axis.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_MUTED)
    price_axis.set_title(
        title or "Every realized grid break against trailing volatility",
        color=INK, fontsize=12, loc="left", pad=8,
    )
    percentile = float(breaches["volatility_percentile"].mean() * 100.0)
    price_axis.text(
        0.995, 0.02,
        f"{len(breaches)} breaks · mean breach volatility percentile {percentile:.1f}%",
        transform=price_axis.transAxes, ha="right", va="bottom",
        color=INK_MUTED, fontsize=9,
    )

    _style_axis(vol_axis, "Annualized volatility")
    vol_axis.plot(
        vol_times, vol_values * 100.0,
        color=VOLATILITY_COLOR, linewidth=0.9,
    )
    if finite.any():
        vol_axis.scatter(
            breach_times[finite], breach_vol[finite] * 100.0,
            s=marker_size, marker="X", color=BREACH_COLOR, linewidths=0.0,
            alpha=marker_alpha, zorder=4,
        )
    vol_axis.text(
        0.995, 0.90,
        f"trailing {analysis['window_bars']}-bar "
        f"({analysis['window_hours']:g}h) window",
        transform=vol_axis.transAxes, ha="right", va="top",
        color=INK_MUTED, fontsize=8,
    )

    _style_axis(count_axis, "Grid breaks / month")
    monthly = analysis["monthly"]
    if len(monthly):
        month_times = pd.to_datetime(monthly["month"])
        count_axis.bar(
            month_times, monthly["breaches"], width=20,
            color=BREACH_COLOR, alpha=0.85, linewidth=0,
        )
    count_axis.set_xlabel("Time", color=INK_MUTED, fontsize=9)
    locator = mdates.AutoDateLocator()
    count_axis.xaxis.set_major_locator(locator)
    count_axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    figure.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(figure)
    logger.info(f"Breach timeline saved to {path} ({len(breaches)} grid breaks)")
    return path


def plot_breach_volatility(
    logger: logging.Logger,
    analysis: dict,
    path: str,
    *,
    title: Optional[str] = None,
) -> Optional[str]:
    """Conditional break probability vs volatility, and monthly correlation."""
    conditional = analysis["conditional"]
    if conditional.empty:
        logger.warning(f"No volatility buckets to plot, skipping {path}")
        return None

    figure, (curve_axis, scatter_axis) = plt.subplots(
        1, 2, figsize=(14, 5.6), constrained_layout=True,
    )
    figure.patch.set_facecolor(SURFACE)

    _style_axis(curve_axis, "P(grid break within horizon) (%)")
    horizons = sorted(conditional["horizon_hours"].unique())
    for colour, hours in zip(ORDINAL, horizons):
        subset = conditional[conditional["horizon_hours"] == hours].sort_values("mean_vol_pct")
        curve_axis.plot(
            subset["mean_vol_pct"], subset["breach_probability"] * 100.0,
            color=colour, marker="o", markersize=4, linewidth=1.8,
            label=f"next {hours:g}h ({int(subset['horizon_bars'].iloc[0])} bars)",
        )
        baseline = float(subset["baseline_probability"].iloc[0]) * 100.0
        curve_axis.axhline(
            baseline, color=colour, linestyle="--", linewidth=1.0, alpha=0.55,
        )
    curve_axis.set_xlabel(
        "Trailing annualized volatility (%)", color=INK_MUTED, fontsize=9
    )
    curve_axis.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_MUTED)
    curve_axis.set_title(
        title or "Breaks cluster in high-volatility bars",
        color=INK, fontsize=12, loc="left", pad=8,
    )
    curve_axis.text(
        0.995, 0.02,
        "dashed = unconditional rate · decile buckets",
        transform=curve_axis.transAxes, ha="right", va="bottom",
        color=INK_MUTED, fontsize=8,
    )

    _style_axis(scatter_axis, "Grid breaks per month")
    monthly = analysis["monthly"]
    if len(monthly):
        scatter_axis.scatter(
            monthly["mean_vol_pct"], monthly["breaches"],
            s=26, color=VOLATILITY_COLOR, alpha=0.75, edgecolors=SURFACE,
            linewidths=0.5,
        )
        rho = analysis.get("monthly_rho", float("nan"))
        scatter_axis.text(
            0.03, 0.96,
            f"Spearman rho = {rho:.3f}\n{len(monthly)} months",
            transform=scatter_axis.transAxes, ha="left", va="top",
            color=INK_MUTED, fontsize=9,
        )
    scatter_axis.set_xlabel("Monthly mean volatility (%)", color=INK_MUTED, fontsize=9)

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    figure.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(figure)
    logger.info(f"Breach-volatility chart saved to {path}")
    return path


def plot_breach_analysis(
    logger: logging.Logger,
    bars,
    trades,
    directory: str,
    *,
    title: Optional[str] = None,
    window_hours: float = 4.0,
    horizons_hours: Sequence[float] = (1.0, 4.0, 24.0),
    buckets: int = 10,
) -> Optional[str]:
    """Write both charts and their underlying tables for every grid break."""
    analysis = breach_volatility_analysis(
        bars, trades,
        window_hours=window_hours,
        horizons_hours=horizons_hours,
        buckets=buckets,
    )
    if not len(analysis["breaches"]):
        logger.warning(f"No grid breaks found, skipping breach analysis in {directory}")
        return None
    os.makedirs(directory, exist_ok=True)
    plot_breach_timeline(
        logger, bars, analysis,
        os.path.join(directory, "breach_timeline_vs_volatility.png"),
        title=title,
    )
    plot_breach_volatility(
        logger, analysis,
        os.path.join(directory, "breach_vs_volatility.png"),
        title=title,
    )
    analysis["breaches"].to_csv(
        os.path.join(directory, "breach_points_volatility.csv"), index=False
    )
    analysis["conditional"].to_csv(
        os.path.join(directory, "breach_volatility_buckets.csv"), index=False
    )
    analysis["monthly"].to_csv(
        os.path.join(directory, "monthly_breaches_vs_volatility.csv"), index=False
    )
    logger.info(
        f"Breach analysis saved to {directory} "
        f"({len(analysis['breaches'])} breaks, "
        f"monthly Spearman rho {analysis['monthly_rho']:.3f})"
    )
    return directory


# ============================================================
# What each annualized volatility level looks like on the chart
# ============================================================
# The bins are the annualized percentages the entry gate sweeps, so a band
# read here is a threshold actually used in a sweep.
REGIME_THRESHOLDS = (0.40, 0.60, 0.80, 1.10, 1.40)
REGIME_LABELS = ("< 40%", "40-60%", "60-80%", "80-110%", "110-140%", ">= 140%")
# The quantity is risk, so the ramp escalates from cool to warm instead of
# cycling through categorical hues.
REGIME_COLORS = ("#8fb4cf", "#4f86b5", "#1f5e8f", "#d9a441", "#cf6a3a", "#b03a2e")


def _exemplar_windows(
    volatility: np.ndarray,
    periods_per_year: float,
    exemplar_days: float,
    targets: Sequence[float],
) -> list[tuple[int, int, float]]:
    """Pick non-overlapping windows whose mean volatility is nearest a target."""
    bars_per_day = periods_per_year / 365.25
    length = max(2, int(round(exemplar_days * bars_per_day)))
    means = pd.Series(volatility).rolling(length, min_periods=length).mean()
    occupied = np.zeros(len(volatility), dtype=bool)
    selected: list[tuple[int, int, float]] = []
    for target in targets:
        score = np.array((means - target).abs().to_numpy(), dtype=float, copy=True)
        score[~np.isfinite(score)] = np.inf
        score[occupied] = np.inf
        if not np.isfinite(score).any():
            break
        end = int(np.argmin(score))
        begin = max(0, end - length + 1)
        half = length // 4
        occupied[max(0, begin - half):min(len(volatility), end + half)] = True
        selected.append((begin, end, float(means.iloc[end])))
    return selected


def plot_volatility_regimes(
    logger: logging.Logger,
    bars,
    path: str,
    *,
    window_hours: float = 4.0,
    exemplar_days: float = 3.0,
    max_points: int = 5_000,
    exemplar_targets: Sequence[float] = (0.35, 0.90, 2.20),
) -> Optional[str]:
    """Price coloured by trailing volatility, with one window per regime.

    The top panel answers "which market produced that volatility": every
    stretch of the price is drawn in the colour of the trailing window's
    annualized volatility. The middle panel is that volatility itself with the
    sweep thresholds marked, on a log axis so the ordinary 20-80% range is not
    flattened by the occasional spike. The bottom row is three three-day
    windows whose mean volatility is closest to a calm, a normal and a
    stressed target, so the bars inside a regime can be compared in detail.

    Both panels use log scales: DOGE spans three orders of magnitude in price
    and its volatility ranges from a few percent to several hundred percent
    annualized, so a linear axis would hide everything except the extremes.
    """
    from trade.sim.martingale_backtest import rolling_annualized_volatility

    times = pd.to_datetime(bars.time)
    volatility, window_bars = rolling_annualized_volatility(
        bars, window_hours=window_hours
    )
    if not np.isfinite(volatility).any():
        logger.warning(f"No volatility values to plot, skipping {path}")
        return None

    close = np.asarray(bars.close, dtype=float)
    low = np.asarray(bars.low, dtype=float)
    high = np.asarray(bars.high, dtype=float)
    count = len(close)

    edges = np.unique(np.linspace(0, count, max_points + 1).astype(int))
    starts, stops = edges[:-1], edges[1:]
    keep = stops > starts
    starts, stops = starts[keep], stops[keep]
    sample_times = times[starts]
    sample_low = np.minimum.reduceat(low, starts)
    sample_high = np.maximum.reduceat(high, starts)
    sample_close = close[stops - 1]
    # One colour per drawn segment: the mean of the trailing volatility over
    # the bucket, not its last value, so a single spike cannot recolour a day.
    sample_vol = np.full(len(starts), np.nan)
    for index, (begin, end) in enumerate(zip(starts, stops)):
        window = volatility[begin:end]
        window = window[np.isfinite(window)]
        if window.size:
            sample_vol[index] = float(window.mean())
    sample_bins = np.where(
        np.isfinite(sample_vol),
        np.digitize(np.nan_to_num(sample_vol), REGIME_THRESHOLDS),
        -1,
    )

    figure = plt.figure(figsize=(14, 11.6), constrained_layout=True)
    grid = figure.add_gridspec(3, 3, height_ratios=(3.2, 1.4, 2.0))
    price_axis = figure.add_subplot(grid[0, :])
    vol_axis = figure.add_subplot(grid[1, :], sharex=price_axis)
    figure.patch.set_facecolor(SURFACE)

    _style_axis(price_axis, "Market price (log)")
    price_axis.set_yscale("log")
    price_axis.yaxis.set_major_locator(
        mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0))
    )
    price_axis.yaxis.set_major_formatter(mticker.ScalarFormatter())
    price_axis.yaxis.set_minor_formatter(mticker.NullFormatter())
    price_axis.fill_between(
        sample_times, sample_low, sample_high,
        color=PRICE_BAND, linewidth=0, alpha=0.35, zorder=1,
    )

    # One line per contiguous regime run; a new run starts whenever the bucket
    # changes. Joining with one extra point keeps the path continuous.
    boundaries = np.concatenate(
        ([0], np.flatnonzero(np.diff(sample_bins) != 0) + 1, [len(sample_bins)])
    )
    for run in range(len(boundaries) - 1):
        begin, end = boundaries[run], boundaries[run + 1]
        bucket = int(sample_bins[begin])
        colour = PRICE_BAND if bucket < 0 else REGIME_COLORS[bucket]
        price_axis.plot(
            sample_times[begin:end + 1], sample_close[begin:end + 1],
            color=colour, linewidth=1.3, zorder=2,
        )

    present = sorted({int(bucket) for bucket in sample_bins if bucket >= 0})
    handles = [
        Line2D([0], [0], color=REGIME_COLORS[bucket], linewidth=2,
               label=REGIME_LABELS[bucket])
        for bucket in present
    ]
    price_axis.legend(
        handles=handles, loc="upper left", ncol=len(handles), frameon=False,
        fontsize=8, labelcolor=INK_MUTED, handlelength=1.6, columnspacing=1.2,
        borderaxespad=0.15,
    )
    price_axis.set_title(
        "Market price coloured by trailing annualized volatility",
        color=INK, fontsize=12, loc="left", pad=10,
    )

    _style_axis(vol_axis, "Annualized volatility (log)")
    vol_axis.set_yscale("log")
    # A fixed window around the sweep thresholds: the extremes reach four
    # digits but would squash the 40-140% bands that decide entries into a
    # few pixels.
    vol_axis.set_ylim(5.0, 300.0)
    edges_pct = [0.0] + [threshold * 100.0 for threshold in REGIME_THRESHOLDS] + [300.0]
    for index, colour in enumerate(REGIME_COLORS):
        vol_axis.axhspan(
            edges_pct[index], edges_pct[index + 1],
            color=colour, alpha=0.07, linewidth=0, zorder=0,
        )
    # The drawn line is the same daily mean the price is coloured by, not the
    # raw 4h series: at 662k bars a raw trace is an unreadable solid band.
    vol_axis.plot(
        sample_times, np.where(np.isfinite(sample_vol), sample_vol * 100.0, np.nan),
        color=VOLATILITY_COLOR, linewidth=1.1, zorder=2,
    )
    for threshold in REGIME_THRESHOLDS:
        vol_axis.axhline(
            threshold * 100.0, color=INK_MUTED, linestyle="--",
            linewidth=0.8, alpha=0.55,
        )
    vol_axis.yaxis.set_major_locator(
        mticker.FixedLocator([threshold * 100.0 for threshold in REGIME_THRESHOLDS])
    )
    vol_axis.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    vol_axis.yaxis.set_minor_locator(mticker.NullLocator())
    vol_axis.set_xlabel("Time", color=INK_MUTED, fontsize=9)
    vol_axis.text(
        0.995, 0.94,
        f"trailing {window_bars}-bar ({window_hours:g}h) window",
        transform=vol_axis.transAxes, ha="right", va="top",
        color=INK_MUTED, fontsize=8,
    )
    locator = mdates.AutoDateLocator()
    vol_axis.xaxis.set_major_locator(locator)
    vol_axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    # Calm / normal / stressed windows, picked by mean volatility. Occupied
    # spans are excluded so the three examples never overlap.
    selected = _exemplar_windows(
        volatility, bars.periods_per_year, exemplar_days, exemplar_targets
    )

    for column, (begin, end, mean_vol) in enumerate(selected):
        panel = figure.add_subplot(grid[2, column])
        panel.set_facecolor(SURFACE)
        span = slice(begin, end + 1)
        panel.fill_between(
            times[span], low[span], high[span],
            color=PRICE_BAND, linewidth=0, alpha=0.35,
        )
        panel.plot(times[span], close[span], color=PRICE_LINE, linewidth=1.0)
        panel.set_title(
            f"{column + 1} · {pd.Timestamp(times[begin]):%Y-%m-%d}"
            f" · {mean_vol * 100.0:.0f}% annualized",
            color=INK, fontsize=10, loc="left",
        )
        panel.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
        panel.tick_params(colors=INK_MUTED, labelsize=8)
        for side in ("top", "right"):
            panel.spines[side].set_visible(False)
        panel.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

        # Three days are a hairline at a six-year zoom, so the window is marked
        # by its centre only; the detail lives in the panel below.
        centre = times[begin] + (times[end] - times[begin]) / 2
        price_axis.axvline(
            centre, color=INK_MUTED, linestyle=":", linewidth=0.9,
            alpha=0.7, zorder=0,
        )
        price_axis.annotate(
            str(column + 1),
            xy=(centre, 0.04),
            xycoords=("data", "axes fraction"),
            color=INK_MUTED, fontsize=9, ha="center", va="bottom",
        )

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    figure.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(figure)
    logger.info(
        f"Volatility regime chart saved to {path} "
        f"(bins {list(REGIME_LABELS)}, windows {[start for start, _, _ in selected]})"
    )
    return path


# ============================================================
# Interactive HTML version: 5-minute bars, coloured by volatility
# ============================================================
# 662k bars cannot be inspected at 5-minute resolution in a static PNG, so the
# HTML embeds the whole series and draws only the visible range. Each bar is
# packed as uint32 time offset, four uint32 prices (scale 1e-6) and one uint16
# volatility (0.1% units, 65535 = not yet available): 22 bytes per bar.
REGIME_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --surface: #fcfcfb;
  --panel: #ffffff;
  --ink: #0b0b0b;
  --muted: #52514e;
  --line: #e3e2df;
  --accent: #4a3aa7;
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0;
  background: var(--surface);
  color: var(--ink);
  font: 13px/1.45 ui-sans-serif, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  display: flex;
  flex-direction: column;
}
header { padding: 14px 18px 0; }
h1 { font-size: 15px; margin: 0; font-weight: 600; letter-spacing: .01em; }
#subtitle { margin: 4px 0 0; color: var(--muted); font-size: 12px; }
main { flex: 1; display: flex; flex-direction: column; min-height: 0; padding: 10px 18px 14px; gap: 9px; }
.toolbar { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; }
.group { display: flex; gap: 4px; align-items: center; }
.group .tag { color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .07em; margin-right: 2px; }
button {
  font: inherit; font-size: 12px; padding: 4px 9px;
  border: 1px solid var(--line); background: var(--panel); color: var(--ink);
  border-radius: 6px; cursor: pointer;
}
button:hover { border-color: #bcbab4; }
button[aria-pressed="true"] { border-color: var(--ink); font-weight: 600; }
.chart-shell {
  position: relative; flex: 1; min-height: 320px;
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px; overflow: hidden;
}
canvas { display: block; width: 100%; height: 100%; touch-action: none; cursor: crosshair; }
.readout { font-size: 12px; color: var(--muted); min-height: 18px; font-variant-numeric: tabular-nums; }
.legend { display: flex; flex-wrap: wrap; gap: 6px; font-size: 11px; color: var(--muted); }
.chip { display: flex; gap: 6px; align-items: center; padding: 2px 8px; border: 1px solid var(--line); border-radius: 999px; }
.swatch { width: 10px; height: 10px; border-radius: 2px; }
.share { color: var(--ink); font-variant-numeric: tabular-nums; }
.note { color: var(--muted); font-size: 11px; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <p id="subtitle"></p>
</header>
<main>
  <div class="toolbar" id="toolbar"></div>
  <div class="chart-shell"><canvas id="chart"></canvas></div>
  <div class="readout" id="readout">Wheel to zoom, drag to pan, double-click to reset.</div>
  <div class="legend" id="legend"></div>
  <p class="note">
    Bar colour is the trailing annualized volatility of the 4h window ending at that bar
    (causal: it only uses bars up to the close). Bands are the entry-gate sweep thresholds.
    Filled body = up bar, hollow = down bar.
  </p>
</main>
<script type="text/plain" id="payload">__DATA__</script>
<script>
const META = __META__;
const $ = (id) => document.getElementById(id);
const TH = META.thresholds, LBL = META.labels, CLR = META.colors;
const T0 = META.start_sec, SCALE = META.price_scale, BANDS = CLR.length;

let N, TA, OA, HA, LA, CA, VA;
(function decodePayload() {
  const text = $('payload').textContent.trim();
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  const buffer = bytes.buffer;
  N = META.count;
  TA = new Uint32Array(buffer, 0, N);
  OA = new Uint32Array(buffer, 4 * N, N);
  HA = new Uint32Array(buffer, 8 * N, N);
  LA = new Uint32Array(buffer, 12 * N, N);
  CA = new Uint32Array(buffer, 16 * N, N);
  VA = new Uint16Array(buffer, 20 * N, N);
})();

const openAt = (i) => OA[i] / SCALE;
const highAt = (i) => HA[i] / SCALE;
const lowAt = (i) => LA[i] / SCALE;
const closeAt = (i) => CA[i] / SCALE;
const secAt = (i) => T0 + TA[i];
const volAt = (i) => (VA[i] === 65535 ? NaN : VA[i] / 10);

function binOf(pct) {
  if (!isFinite(pct)) return -1;
  const value = pct / 100;
  let band = 0;
  while (band < TH.length && value >= TH[band]) band += 1;
  return band;
}

const canvas = $('chart');
const ctx = canvas.getContext('2d');
const PAD = { l: 68, r: 18, t: 14, b: 26 };
const GAP = 30;
let CW = 0, CH = 0, cache = null, hover = null, dragState = null;
const totalFrom = T0, totalTo = secAt(N - 1) + 300;
let view = { from: totalFrom, to: totalTo };

function layout() {
  const plotW = CW - PAD.l - PAD.r;
  const plotH = CH - PAD.t - PAD.b - GAP;
  const priceH = Math.max(120, Math.round(plotH * 0.70));
  return {
    x0: PAD.l, plotW,
    priceTop: PAD.t, priceBottom: PAD.t + priceH, priceH,
    volTop: PAD.t + priceH + GAP, volBottom: CH - PAD.b,
    volH: Math.max(48, CH - PAD.b - (PAD.t + priceH + GAP)),
  };
}

function pad2(value) { return String(value).padStart(2, '0'); }

function isoDate(sec) {
  const date = new Date(sec * 1000);
  return `${date.getUTCFullYear()}-${pad2(date.getUTCMonth() + 1)}-${pad2(date.getUTCDate())}`;
}

function isoTime(sec) {
  const date = new Date(sec * 1000);
  return `${isoDate(sec)} ${pad2(date.getUTCHours())}:${pad2(date.getUTCMinutes())}`;
}

function fmtPrice(value) {
  if (value >= 1) return value.toFixed(2);
  if (value >= 0.1) return value.toFixed(3);
  if (value >= 0.01) return value.toFixed(4);
  return value.toFixed(5);
}

function fmtPct(value) {
  return `${value >= 100 ? value.toFixed(0) : value.toFixed(1)}%`;
}

function indexAtSec(sec) {
  let lo = 0, hi = N - 1;
  if (sec <= T0) return 0;
  if (sec >= secAt(N - 1)) return N - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (secAt(mid) < sec) lo = mid + 1; else hi = mid;
  }
  return lo;
}

function timeTicks(from, to) {
  const ladder = [300, 900, 3600, 4 * 3600, 6 * 3600, 12 * 3600, 86400,
                  3 * 86400, 7 * 86400, 14 * 86400, 30 * 86400, 90 * 86400,
                  180 * 86400, 365 * 86400];
  const span = to - from;
  let step = ladder[ladder.length - 1];
  for (const candidate of ladder) {
    if (span / candidate <= 7) { step = candidate; break; }
  }
  const ticks = [];
  for (let t = Math.ceil(from / step) * step; t <= to; t += step) ticks.push(t);
  return { step, ticks };
}

function rebuild() {
  const L = layout();
  const cols = Math.max(1, Math.floor(L.plotW));
  const span = view.to - view.from;
  const first = Math.max(0, indexAtSec(view.from) - 1);
  const last = Math.min(N, indexAtSec(view.to) + 2);
  const open = new Float64Array(cols);
  const high = new Float64Array(cols);
  const low = new Float64Array(cols);
  const close = new Float64Array(cols);
  const count = new Uint32Array(cols);
  const volSum = new Float64Array(cols);
  const volCount = new Uint32Array(cols);
  const shares = new Uint32Array(BANDS);
  let priced = 0, totalVol = 0, pmin = Infinity, pmax = -Infinity;

  for (let i = first; i < last; i += 1) {
    const t = secAt(i);
    if (t < view.from || t > view.to) continue;
    const col = Math.min(cols - 1, Math.max(0, Math.floor((t - view.from) / span * cols)));
    const o = openAt(i), h = highAt(i), l = lowAt(i), c = closeAt(i);
    if (!(o > 0) || !(c > 0)) continue;
    if (count[col] === 0) { open[col] = o; high[col] = h; low[col] = l; }
    else {
      if (h > high[col]) high[col] = h;
      if (l < low[col]) low[col] = l;
    }
    close[col] = c;
    count[col] += 1;
    if (h > pmax) pmax = h;
    if (l < pmin) pmin = l;
    priced += 1;
    const vol = volAt(i);
    if (isFinite(vol)) { volSum[col] += vol; volCount[col] += 1; }
    const band = binOf(vol);
    if (band >= 0) { shares[band] += 1; totalVol += 1; }
  }

  if (!isFinite(pmin) || !isFinite(pmax)) { pmin = 1; pmax = 1; }
  cache = {
    cols, open, high, low, close, count, volSum, volCount,
    shares, totalVol, priced,
    pmin: pmin * 0.985, pmax: pmax * 1.015,
  };
  const bars = document.querySelectorAll('.share');
  bars.forEach((element) => {
    const band = Number(element.dataset.band);
    element.textContent = totalVol ? `${(shares[band] / totalVol * 100).toFixed(1)}%` : '—';
  });
}

function paintAxes(L) {
  ctx.font = '11px ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif';
  ctx.fillStyle = '#52514e';
  ctx.strokeStyle = '#e3e2df';
  ctx.lineWidth = 1;
  const { step, ticks } = timeTicks(view.from, view.to);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const t of ticks) {
    const x = L.x0 + (t - view.from) / (view.to - view.from) * L.plotW;
    if (x < L.x0 - 1 || x > L.x0 + L.plotW + 1) continue;
    ctx.beginPath();
    ctx.moveTo(x, L.priceTop);
    ctx.lineTo(x, L.volBottom);
    ctx.stroke();
    const label = step < 86400
      ? `${isoTime(t).slice(5, 10)} ${isoTime(t).slice(11)}`
      : (step < 90 * 86400 ? isoDate(t).slice(5) : isoDate(t).slice(0, 7));
    ctx.fillText(label, x, L.volBottom + 7);
  }
}

function paintPrice(L) {
  const { cols, open, high, low, close, count, volSum, volCount, pmin, pmax } = cache;
  const lo = Math.log(pmin), hi = Math.log(pmax);
  const yOf = (price) => L.priceBottom - (Math.log(price) - lo) / (hi - lo) * L.priceH;

  const decade = Math.floor(Math.log10(pmin));
  const ticks = [];
  for (let d = decade - 1; d <= Math.ceil(Math.log10(pmax)); d += 1) {
    for (const mult of [1, 1.5, 2, 3, 4, 5, 6, 8, 9, 10]) {
      const value = mult * Math.pow(10, d);
      if (value >= pmin && value <= pmax) ticks.push(value);
    }
  }
  if (ticks.length < 2) {
    // A narrow range can miss every round number; fall back to equal steps in
    // log space so the axis is never unlabelled.
    ticks.length = 0;
    for (let step = 0; step <= 3; step += 1) {
      ticks.push(Math.exp(lo + (hi - lo) * step / 3));
    }
  } else if (ticks.length > 7) {
    const stride = Math.ceil(ticks.length / 6);
    const thinned = ticks.filter((_, index) => index % stride === 0);
    ticks.length = 0;
    thinned.forEach((value) => ticks.push(value));
  }
  ctx.font = '11px ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#52514e';
  ctx.strokeStyle = '#e3e2df';
  for (const value of ticks) {
    const y = yOf(value);
    ctx.beginPath();
    ctx.moveTo(L.x0, y);
    ctx.lineTo(L.x0 + L.plotW, y);
    ctx.stroke();
    ctx.fillText(fmtPrice(value), L.x0 - 6, y);
  }

  const colW = L.plotW / cols;
  for (let col = 0; col < cols; col += 1) {
    if (!count[col]) continue;
    const x = L.x0 + (col + 0.5) * colW;
    const yHigh = yOf(high[col]), yLow = yOf(low[col]);
    const yOpen = yOf(open[col]), yClose = yOf(close[col]);
    const band = binOf(volCount[col] ? volSum[col] / volCount[col] : NaN);
    const colour = band < 0 ? META.neutral : CLR[band];
    const bodyWidth = Math.max(1, Math.min(9, colW - 1));
    ctx.strokeStyle = colour;
    ctx.globalAlpha = 0.9;
    ctx.lineWidth = Math.max(0.7, Math.min(1.3, colW / 3));
    ctx.beginPath();
    ctx.moveTo(x, yHigh);
    ctx.lineTo(x, yLow);
    ctx.stroke();
    const top = Math.min(yOpen, yClose);
    const height = Math.max(1, Math.abs(yClose - yOpen));
    ctx.globalAlpha = 1;
    if (colW >= 3 && close[col] < open[col]) {
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(x - bodyWidth / 2, top, bodyWidth, height);
      ctx.strokeStyle = colour;
      ctx.lineWidth = 1;
      ctx.strokeRect(x - bodyWidth / 2, top, bodyWidth, height);
    } else {
      ctx.fillStyle = colour;
      ctx.fillRect(x - bodyWidth / 2, top, bodyWidth, height);
    }
  }
  ctx.globalAlpha = 1;
}

function paintVolatility(L) {
  const { cols, volSum, volCount } = cache;
  const floor = Math.log(META.vol_floor), ceiling = Math.log(META.vol_ceiling);
  const yOf = (pct) => {
    const clamped = Math.min(META.vol_ceiling, Math.max(META.vol_floor, pct));
    const ratio = (Math.log(clamped) - floor) / (ceiling - floor);
    return L.volBottom - ratio * L.volH;
  };

  const boundaries = [0].concat(TH.map((value) => value * 100), [META.vol_ceiling]);
  for (let band = 0; band < BANDS; band += 1) {
    const top = yOf(Math.min(boundaries[band + 1], META.vol_ceiling));
    const bottom = yOf(boundaries[band]);
    ctx.fillStyle = CLR[band];
    ctx.globalAlpha = 0.07;
    ctx.fillRect(L.x0, top, L.plotW, Math.max(0, bottom - top));
  }
  ctx.globalAlpha = 1;

  ctx.strokeStyle = '#52514e';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.font = '11px ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const threshold of TH) {
    const y = yOf(threshold * 100);
    ctx.beginPath();
    ctx.moveTo(L.x0, y);
    ctx.lineTo(L.x0 + L.plotW, y);
    ctx.stroke();
    ctx.fillStyle = '#52514e';
    ctx.fillText(fmtPct(threshold * 100), L.x0 - 6, y);
    ctx.strokeStyle = '#52514e';
  }
  ctx.setLineDash([]);

  const colW = L.plotW / cols;
  ctx.strokeStyle = '#4a3aa7';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  let drawing = false;
  let lastCol = -1;
  for (let col = 0; col < cols; col += 1) {
    if (!volCount[col]) continue;
    const x = L.x0 + (col + 0.5) * colW;
    const y = yOf(volSum[col] / volCount[col]);
    // When zoomed in there is only one bar per pixel column, so adjacent
    // columns are a pixel apart. Bridging empty columns keeps the strip a
    // line instead of a scatter of isolated dots; a wide gap restarts it.
    if (!drawing || col - lastCol > 12) { ctx.moveTo(x, y); drawing = true; }
    else { ctx.lineTo(x, y); }
    lastCol = col;
  }
  ctx.stroke();

  ctx.fillStyle = '#52514e';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText('4h annualized volatility (log)', L.x0 + 4, L.volTop - 14);
}

function paintExemplars(L) {
  const span = view.to - view.from;
  ctx.font = '600 11px ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif';
  META.exemplars.forEach((example, index) => {
    const centre = (example.start_sec + example.end_sec) / 2;
    const x = L.x0 + (centre - view.from) / span * L.plotW;
    if (x < L.x0 || x > L.x0 + L.plotW) return;
    ctx.strokeStyle = '#52514e';
    ctx.setLineDash([2, 3]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, L.priceTop);
    ctx.lineTo(x, L.volBottom);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#52514e';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(String(index + 1), x, L.priceBottom - 2);
  });
}

function paint() {
  if (!CW || !CH) return;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, CW, CH);
  if (!cache) rebuild();
  const L = layout();
  if (!cache.priced) {
    ctx.fillStyle = '#52514e';
    ctx.font = '12px ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No bars in this range', CW / 2, CH / 2);
    return;
  }
  paintAxes(L);
  paintPrice(L);
  paintVolatility(L);
  paintExemplars(L);
  paintHover(L);
}

function paintHover(L) {
  if (!hover) return;
  const x = L.x0 + (hover.sec - view.from) / (view.to - view.from) * L.plotW;
  if (x < L.x0 || x > L.x0 + L.plotW) return;
  ctx.strokeStyle = '#0b0b0b';
  ctx.globalAlpha = 0.35;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, L.priceTop);
  ctx.lineTo(x, L.volBottom);
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function updateReadout() {
  if (!hover) {
    $('readout').textContent = 'Wheel to zoom, drag to pan, double-click to reset.';
    return;
  }
  const i = hover.index;
  const vol = volAt(i);
  const band = binOf(vol);
  const bandText = band < 0 ? 'volatility not available yet' : `band ${LBL[band]}`;
  $('readout').textContent =
    `${isoTime(secAt(i))} UTC  ·  close ${fmtPrice(closeAt(i))}  ·  ` +
    `4h vol ${isFinite(vol) ? fmtPct(vol) : 'n/a'}  ·  ${bandText}`;
}

function setView(from, to, pressed) {
  const span = Math.max(300 * 60, to - from);
  const limit = totalTo - totalFrom;
  let start = Math.max(totalFrom, from);
  if (start + span > totalTo) start = Math.max(totalFrom, totalTo - span);
  const clampedSpan = Math.min(span, limit);
  view = { from: start, to: start + clampedSpan };
  cache = null;
  hover = null;
  toolbarButtons.forEach((button) => {
    button.setAttribute('aria-pressed', button.dataset.key === pressed ? 'true' : 'false');
  });
  paint();
}

function zoomAt(anchorSec, factor) {
  const span = Math.min(totalTo - totalFrom, Math.max(300 * 60, (view.to - view.from) * factor));
  const ratio = (anchorSec - view.from) / (view.to - view.from);
  const from = anchorSec - ratio * span;
  setView(from, from + span, null);
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

const toolbarButtons = [];
function addButtons(group, items) {
  items.forEach((item) => {
    const button = document.createElement('button');
    button.textContent = item.text;
    button.dataset.key = item.key;
    button.title = item.title || '';
    button.setAttribute('aria-pressed', item.key === 'all' ? 'true' : 'false');
    button.addEventListener('click', item.action);
    group.appendChild(button);
    toolbarButtons.push(button);
  });
}

function buildToolbar() {
  const toolbar = $('toolbar');
  const ranges = document.createElement('div');
  ranges.className = 'group';
  const rangeTag = document.createElement('span');
  rangeTag.className = 'tag';
  rangeTag.textContent = 'range';
  ranges.appendChild(rangeTag);
  const presets = [
    { text: 'All', key: 'all', days: null },
    { text: '1y', key: '1y', days: 365 },
    { text: '90d', key: '90d', days: 90 },
    { text: '30d', key: '30d', days: 30 },
    { text: '7d', key: '7d', days: 7 },
    { text: '1d', key: '1d', days: 1 },
  ];
  addButtons(ranges, presets.map((preset) => ({
    text: preset.text,
    key: preset.key,
    action: () => {
      if (!preset.days) { setView(totalFrom, totalTo, preset.key); return; }
      const to = totalTo;
      setView(to - preset.days * 86400, to, preset.key);
    },
  })));
  toolbar.appendChild(ranges);

  const jumps = document.createElement('div');
  jumps.className = 'group';
  const jumpTag = document.createElement('span');
  jumpTag.className = 'tag';
  jumpTag.textContent = '3-day windows';
  jumps.appendChild(jumpTag);
  addButtons(jumps, META.exemplars.map((example, index) => ({
    text: String(index + 1),
    key: `example-${index}`,
    title: `${isoDate(example.start_sec)} · mean ${fmtPct(example.mean_pct)}`,
    action: () => {
      const span = example.end_sec - example.start_sec;
      const pad = span * 0.35;
      setView(example.start_sec - pad, example.end_sec + pad, `example-${index}`);
    },
  })));
  toolbar.appendChild(jumps);
}

function buildLegend() {
  const legend = $('legend');
  for (let band = 0; band < BANDS; band += 1) {
    const chip = document.createElement('span');
    chip.className = 'chip';
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = CLR[band];
    const label = document.createElement('span');
    label.textContent = LBL[band];
    const share = document.createElement('span');
    share.className = 'share';
    share.dataset.band = String(band);
    share.textContent = '—';
    chip.append(swatch, label, share);
    legend.appendChild(chip);
  }
}

function resizeCanvas() {
  DPR = window.devicePixelRatio || 1;
  const box = canvas.parentElement.getBoundingClientRect();
  CW = Math.max(320, box.width);
  CH = Math.max(240, box.height);
  canvas.width = Math.round(CW * DPR);
  canvas.height = Math.round(CH * DPR);
  cache = null;
  paint();
}

canvas.addEventListener('wheel', (event) => {
  event.preventDefault();
  const point = canvasPoint(event);
  const L = layout();
  const ratio = (point.x - L.x0) / L.plotW;
  if (ratio < 0 || ratio > 1) return;
  const anchor = view.from + ratio * (view.to - view.from);
  zoomAt(anchor, event.deltaY > 0 ? 1.25 : 0.8);
}, { passive: false });

canvas.addEventListener('pointerdown', (event) => {
  dragState = { x: event.clientX, from: view.from, to: view.to };
  canvas.setPointerCapture(event.pointerId);
  canvas.style.cursor = 'grabbing';
});

canvas.addEventListener('pointerup', () => {
  dragState = null;
  canvas.style.cursor = 'crosshair';
});

canvas.addEventListener('pointermove', (event) => {
  const L = layout();
  if (dragState) {
    const shift = (event.clientX - dragState.x) / L.plotW * (dragState.to - dragState.from);
    setView(dragState.from - shift, dragState.to - shift, null);
    return;
  }
  const point = canvasPoint(event);
  if (point.x < L.x0 || point.x > L.x0 + L.plotW || point.y > L.volBottom) {
    hover = null;
  } else {
    const sec = view.from + (point.x - L.x0) / L.plotW * (view.to - view.from);
    const index = indexAtSec(sec);
    hover = { sec, index };
  }
  updateReadout();
  paint();
});

canvas.addEventListener('pointerleave', () => { hover = null; updateReadout(); paint(); });
canvas.addEventListener('dblclick', () => setView(totalFrom, totalTo, 'all'));

new ResizeObserver(() => resizeCanvas()).observe(canvas.parentElement);

$('subtitle').textContent =
  `${META.label} · ${META.count.toLocaleString('en-US')} bars · ` +
  `${isoDate(T0)} → ${isoDate(totalTo)} · trailing ${META.window_bars}-bar ` +
  `(${META.window_hours}h) annualized volatility`;

buildToolbar();
buildLegend();
resizeCanvas();
</script>
</body>
</html>
"""


def write_volatility_regimes_html(
    logger: logging.Logger,
    bars,
    path: str,
    *,
    label: str = "DOGEUSDT 5m",
    window_hours: float = 4.0,
    exemplar_days: float = 3.0,
    exemplar_targets: Sequence[float] = (0.35, 0.90, 2.20),
) -> str:
    """Write a self-contained HTML chart of every bar, coloured by regime.

    The PNG version has to decimate 662k bars down to a few thousand points,
    which hides the 5-minute structure. Here the whole series travels with the
    file and the browser draws only the visible range, so zooming in reaches
    individual candles. The payload costs 22 bytes per bar.
    """
    from trade.sim.martingale_backtest import rolling_annualized_volatility

    volatility, window_bars = rolling_annualized_volatility(
        bars, window_hours=window_hours
    )
    times = pd.DatetimeIndex(pd.to_datetime(bars.time))
    if times.tz is None:
        times = times.tz_localize("UTC")
    seconds = np.asarray(
        (times - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds(),
        dtype=np.int64,
    )
    offsets = (seconds - seconds[0]).astype(np.uint32)

    def prices(values) -> np.ndarray:
        return np.rint(np.asarray(values, dtype=float) * 1_000_000.0).astype(np.uint32)

    percentage = volatility * 100.0
    units = np.where(
        np.isfinite(percentage),
        np.clip(np.rint(percentage * 10.0), 0, 65534),
        65535,
    ).astype(np.uint16)

    payload = np.concatenate(
        [offsets, prices(bars.open), prices(bars.high), prices(bars.low),
         prices(bars.close)]
    ).astype(np.uint32, copy=False).tobytes() + units.tobytes()
    blob = base64.b64encode(payload).decode("ascii")

    step = int(np.median(np.diff(seconds))) if len(seconds) > 1 else 300
    selected = _exemplar_windows(
        volatility, bars.periods_per_year, exemplar_days, exemplar_targets
    )
    exemplars = [
        {
            "start_sec": int(seconds[begin]),
            "end_sec": int(seconds[min(end + 1, len(seconds) - 1)]) + step,
            "mean_pct": round(float(mean_vol) * 100.0, 1),
        }
        for begin, end, mean_vol in selected
    ]

    meta = {
        "label": label,
        "count": int(len(offsets)),
        "start_sec": int(seconds[0]),
        "price_scale": 1_000_000,
        "thresholds": list(REGIME_THRESHOLDS),
        "labels": list(REGIME_LABELS),
        "colors": list(REGIME_COLORS),
        "neutral": PRICE_BAND,
        "vol_floor": 5.0,
        "vol_ceiling": 300.0,
        "window_hours": window_hours,
        "window_bars": int(window_bars),
        "exemplars": exemplars,
    }

    title = f"{label} · price by annualized volatility"
    html = (
        REGIME_HTML_TEMPLATE
        .replace("__TITLE__", title)
        .replace("__META__", json.dumps(meta))
        .replace("__DATA__", blob)
    )
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(html)
    logger.info(
        f"Volatility regime HTML saved to {path} "
        f"({meta['count']} bars, {len(html) / 1e6:.1f} MB, {len(exemplars)} windows)"
    )
    return path
