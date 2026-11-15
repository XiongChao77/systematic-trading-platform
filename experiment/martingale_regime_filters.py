"""Describe causal entry filters against previously scanned independent cycles."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment.martingale_layer_scan import _data_params
from experiment.martingale_report import validate_scan_sources
from trade.sim import martingale_backtest as mb
from trade.sim.martingale_scan import GRID_STOP, UNRESOLVED

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def wilder(values, period):
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return result
    start = finite[0] + period - 1
    result[start] = np.mean(values[finite[0]:start + 1])
    for i in range(start + 1, len(values)):
        result[i] = result[i - 1] + (values[i] - result[i - 1]) / period
    return result


CLUSTERING_WINDOWS = (12, 24, 48, 96)


def clustering_series(bars, window):
    moves = np.log(pd.Series(bars.close)).diff().abs()
    reference = moves.rolling(2016, min_periods=2016).quantile(0.80).shift(1)
    high = moves.gt(reference).astype(float).where(reference.notna())
    return high.rolling(window, min_periods=window).mean().shift(1)


def indicators(bars):
    close = pd.Series(bars.close)
    clustering = clustering_series(bars, CLUSTERING_WINDOWS[0])
    high_change = pd.Series(bars.high).diff().to_numpy()
    low_change = -pd.Series(bars.low).diff().to_numpy()
    plus = np.where((high_change > low_change) & (high_change > 0), high_change, 0.)
    minus = np.where((low_change > high_change) & (low_change > 0), low_change, 0.)
    previous = close.shift(1).to_numpy()
    tr = np.maximum.reduce([bars.high - bars.low, np.abs(bars.high - previous), np.abs(bars.low - previous)])
    plus[0] = minus[0] = np.nan
    atr = wilder(tr, 14)
    pdi = 100 * wilder(plus, 14) / atr
    mdi = 100 * wilder(minus, 14) / atr
    denominator = pdi + mdi
    dx = np.divide(100 * np.abs(pdi - mdi), denominator, out=np.zeros(len(close)), where=denominator > 0)
    dx[~np.isfinite(denominator)] = np.nan
    adx = pd.Series(wilder(dx, 14)).shift(1)
    change = close.diff().to_numpy()
    gain = wilder(np.maximum(change, 0), 14)
    loss = wilder(np.maximum(-change, 0), 14)
    total = gain + loss
    rsi = pd.Series(np.divide(100 * gain, total, out=np.full(len(close), 50.), where=total > 0))
    rsi[~np.isfinite(total)] = np.nan
    highest = pd.Series(bars.high).rolling(14, min_periods=14).max()
    lowest = pd.Series(bars.low).rolling(14, min_periods=14).min()
    spread = highest - lowest
    wr = (-100 * (highest - close) / spread.where(spread > 0)).where(spread != 0, -50.)
    return {"clustering": clustering, "adx": adx, "rsi": rsi.shift(1), "wr": wr.shift(1)}


def export_deciles(bars, sources):
    features = indicators(bars)
    clusterings = {window: clustering_series(bars, window) for window in CLUSTERING_WINDOWS}
    common = np.logical_and.reduce([value.notna().to_numpy() for value in list(features.values()) + list(clusterings.values())])
    series = []
    for name, feature in features.items():
        if name == "clustering":
            for window, values in clusterings.items():
                array = values.to_numpy()
                series.append((name, window, array, np.quantile(array[common], np.linspace(0, 1, 11))))
        else:
            array = np.abs(feature.to_numpy()) if name == "wr" else feature.to_numpy()
            series.append((name, None, array, np.linspace(0.0, 100.0, 11)))
    runs = {}
    for side, directory in sources.items():
        for layers in range(2, 7):
            cycles = pd.read_csv(Path(directory) / f"cycles_{side}_{layers}additions.csv", usecols=["start_index", "terminal"])
            assert np.array_equal(cycles.start_index, np.arange(len(common)))
            terminal = cycles.terminal.to_numpy()
            runs[(side, layers)] = (terminal, float(np.mean(terminal[common] == GRID_STOP) * 100))
    rows = []
    for name, window, values, edges in series:
        assignments = np.searchsorted(edges[1:-1], values, side="left")
        for (side, layers), (terminal, base_rate) in runs.items():
            for group in range(10):
                mask = common & (assignments == group)
                count = int(mask.sum())
                breaches = int(np.sum(terminal[mask] == GRID_STOP))
                rows.append(dict(indicator=name, window=window, side=side, layers=layers, group=group + 1,
                    lower=float(edges[group]), upper=float(edges[group + 1]),
                    entry_min=float(values[mask].min()) if count else None,
                    entry_max=float(values[mask].max()) if count else None,
                    entry_mean=float(values[mask].mean()) if count else None,
                    starts=count, coverage_pct=count / int(common.sum()) * 100, breaches=breaches,
                    unresolved=int(np.sum(terminal[mask] == UNRESOLVED)),
                    breach_rate_pct=breaches / count * 100 if count else None, baseline_rate_pct=base_rate))
    result = dict(meta=dict(eligible_bars=int(common.sum()), start=str(bars.time[np.flatnonzero(common)[0]]),
        end=str(bars.time[-1]), grouping="Volatility clustering uses full-history deciles per window; ADX/RSI/WR use fixed 0-100 bins (WR absolute); ties stay together; empty groups retained",
        windows=list(CLUSTERING_WINDOWS),
        denominator="All starts in each group, including unresolved cycles", sources=sources), rows=rows)
    payload = json.dumps(result, allow_nan=False)
    (OUT / "assets/indicator-deciles-data.js").write_text("window.INDICATOR_DECILES=" + payload + ";\n")
    pd.DataFrame(rows).to_csv(OUT / "indicator-deciles.csv", index=False)
    print(json.dumps(result["meta"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("regime", "oscillator", "deciles"), default="regime")
    args = parser.parse_args()
    prefix = "oscillator-filter" if args.group == "oscillator" else "regime-filter"
    source = (OUT / "assets/data.js").read_text()
    meta = json.loads(source[source.index("=") + 1:].strip().rstrip(";"))["meta"]
    bars = mb.load_bars(logging.getLogger(__name__), _data_params())
    validate_scan_sources(bars, meta["sources"])
    if args.group == "deciles":
        export_deciles(bars, meta["sources"])
        return
    names = ("rsi", "wr") if args.group == "oscillator" else ("clustering", "adx")
    features = {name: value for name, value in indicators(bars).items() if name in names}
    years = pd.DatetimeIndex(bars.time).year.to_numpy()
    masks = {}
    thresholds = {}
    common = np.ones(len(years), dtype=bool)
    for name, feature in features.items():
        history = feature.shift(1).rolling(25920, min_periods=25920)
        for fraction in ((.8,) if args.group == "oscillator" else (.2, .3, .4, .5, .8)):
            for direction in ("low", "high"):
                quantile = fraction if direction == "low" else 1 - fraction
                threshold = history.quantile(quantile)
                key = f"{name}_{direction}_{int(fraction * 100)}"
                masks[key] = (feature <= threshold if direction == "low" else feature >= threshold).to_numpy()
                thresholds[key] = {"indicator": name, "selection": direction, "target_coverage": fraction * 100, "kind": "trailing_90d_quantile"}
                common &= threshold.notna().to_numpy()
        fixed = {"clustering": (.1, .2, .3, .4), "adx": (15, 20, 25, 30), "rsi": (30, 70), "wr": (-80, -20)}
        for threshold in fixed[name]:
            for selection in (("low", "high") if args.group == "oscillator" else ("low",)):
                operator = "le" if selection == "low" else "ge"
                key = f"{name}_{operator}_{threshold}"
                masks[key] = (feature.le(threshold) if selection == "low" else feature.ge(threshold)).to_numpy()
                thresholds[key] = {"indicator": name, "selection": selection, "threshold": threshold, "kind": "fixed"}
    masks = {"baseline": common, **{key: value & common for key, value in masks.items()}}
    records = []
    for side, directory in meta["sources"].items():
        for layers in (4, 5, 6):
            cycles = pd.read_csv(Path(directory) / f"cycles_{side}_{layers}additions.csv", usecols=["start_index", "terminal", "net_return_pct", "bars_held"])
            assert np.array_equal(cycles.start_index, np.arange(len(years)))
            terminal = cycles.terminal.to_numpy()
            returns = cycles.net_return_pct.to_numpy() * 100
            held = cycles.bars_held.to_numpy()
            for year in [0, *np.unique(years).tolist()]:
                year_mask = np.ones(len(years), dtype=bool) if year == 0 else years == year
                base = common & year_mask
                base_count = int(base.sum())
                if not base_count:
                    continue
                base_rate = np.mean(terminal[base] == GRID_STOP) * 100
                base_return = np.mean(returns[base])
                for key, full_mask in masks.items():
                    mask = full_mask & year_mask
                    count = int(mask.sum())
                    if not count:
                        continue
                    rate = np.mean(terminal[mask] == GRID_STOP) * 100
                    records.append(dict(filter=key, side=side, layers=layers, year=int(year), starts=count,
                        coverage_pct=count / base_count * 100, breaches=int(np.sum(terminal[mask] == GRID_STOP)),
                        breach_rate_pct=rate, breach_rate_change_pct=(rate / base_rate - 1) * 100,
                        mean_return_pct=float(np.mean(returns[mask])), return_delta_pp=float(np.mean(returns[mask]) - base_return),
                        mean_bars_held=float(np.mean(held[mask])), unresolved=int(np.sum(terminal[mask] == UNRESOLVED))))
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / f"{prefix}-results.csv", index=False)
    result = dict(meta=dict(bars=len(years), eligible_bars=int(common.sum()), start=str(bars.time[np.flatnonzero(common)[0]]), end=str(bars.time[-1]),
        clustering="Previous 48 bars: share of absolute log returns above their own preceding 2016-bar 75th percentile",
        rsi="Wilder RSI(14), previous completed bar", wr="Williams %R(14), range -100 to 0, previous completed bar",
        adx="Wilder ADX(14), previous completed bar", adaptive_threshold="Preceding 25920 indicator values; low and high tails separately",
        scope="Descriptive full-history independent overlapping cycles; not rolling OOS or a portfolio backtest",
        costs="Inherited scan costs; zero fees, slippage and funding", sources=meta["sources"]), filters=thresholds, rows=records)
    (OUT / f"{prefix}-results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    plot(frame, names, prefix)
    if args.group == "oscillator":
        print(frame[(frame.year == 0) & frame["filter"].isin(["baseline", "rsi_low_80", "rsi_high_80", "wr_low_80", "wr_high_80"])].to_string(index=False))
        return
    print(frame[(frame.year == 0) & frame["filter"].isin(["baseline", "clustering_low_30", "clustering_high_30", "adx_low_30", "adx_high_30", "adx_le_20"])].to_string(index=False))
    print("YEARLY LONG K6")
    print(frame[(frame.year.between(2021, 2025)) & (frame.side == "long") & (frame.layers == 6) & frame["filter"].isin(["baseline", "clustering_low_30", "adx_low_30", "adx_le_20"])][["filter", "year", "coverage_pct", "breach_rate_pct", "mean_return_pct", "return_delta_pp"]].to_string(index=False))


def plot(frame, names, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    styles = [(names[0], "#15803d", "o"), (names[1], "#2563eb", "s")]
    for row, side in enumerate(("long", "short")):
        for col, layers in enumerate((4, 5, 6)):
            ax = axes[row, col]
            subset = frame[(frame.year == 0) & (frame.side == side) & (frame.layers == layers)]
            baseline = subset[subset["filter"] == "baseline"].iloc[0]
            for index, (indicator, color, marker) in enumerate(styles):
                for offset, (selection, linestyle) in enumerate((("low", "-"), ("high", "--"))):
                    selected = subset[subset["filter"].isin([f"{indicator}_{selection}_{p}" for p in (20, 30, 40, 50, 80)])].sort_values("coverage_pct")
                    if prefix == "oscillator-filter":
                        value = selected.iloc[0]
                        position = 2 * index + offset
                        ax.bar(position, value.mean_return_pct, color=color, alpha=1 if offset == 0 else .55)
                        ax.set_xticks(range(4), ["RSI low", "RSI high", "WR low", "WR high"])
                    else:
                        ax.plot(selected.coverage_pct, selected.mean_return_pct, color=color, marker=marker, linestyle=linestyle, label=f"{indicator} {selection}")
            ax.axhline(baseline.mean_return_pct, color="#64748b", label="Unfiltered", linewidth=1)
            ax.axhline(0, color="#cbd5e1", linewidth=.7)
            ax.set(title=f"{side.title()} K={layers}", xlabel="Retained tail (target 80%)" if prefix == "oscillator-filter" else "Actual retained starts (%)", ylabel="Mean cycle return (%)")
            ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("DOGE 5m | Separate entry filters | Trailing 90-day quantile thresholds", fontsize=14)
    fig.savefig(OUT / f"{prefix}-comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
