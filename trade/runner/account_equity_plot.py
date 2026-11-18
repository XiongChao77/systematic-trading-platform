"""Plot persisted continuous-account equity, including model-stage boundaries."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


CURVE_FILES = {"absolute": "equity_curve_absolute.png", "log": "equity_curve_log.png"}


def save_account_equity_curves(history_path, *, regions=None, title="Backtest equity", output_dir=None):
    """Save separate linear/log equity plots with drawdowns from all recorded bars.

    Equity uses daily last observations for readability. Drawdown uses each
    day's worst recorded bar, with the initial account equity included in the
    high-water mark. The plots describe the full continuous account; vertical
    lines mark model stage boundaries without resetting equity or positions.
    """
    history_path = Path(history_path)
    output = Path(output_dir) if output_dir is not None else history_path.parent
    output.mkdir(parents=True, exist_ok=True)
    account = pd.read_csv(history_path, usecols=["time", "previous_equity", "equity"])
    times = pd.to_datetime(account["time"], utc=True)
    values = account.equity.to_numpy(float)
    if account.empty or times.isna().any() or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError(f"Account history must contain unique chronological timestamps: {history_path}")
    baseline = float(account.previous_equity.iloc[0])
    if not np.isfinite(values).all() or not np.isfinite(baseline) or baseline <= 0:
        raise ValueError(f"Account history contains invalid equity: {history_path}")
    index = pd.DatetimeIndex(times)
    equity = pd.Series(values, index=index)
    highs = np.maximum.accumulate(np.r_[baseline, values])[1:]
    drawdown = pd.Series(np.where(highs > 0, (values / highs - 1) * 100, 0.), index=index)
    daily_equity = equity.groupby(equity.index.date, sort=True).tail(1)
    daily_drawdown = drawdown.resample("1D").min().dropna()
    # Include the first observation and starting balance even for a one-day run.
    plot_equity = pd.concat([pd.Series([baseline], index=[index[0]]), daily_equity])
    plot_equity = plot_equity.sort_index(kind="stable")
    boundaries = []
    for label, key in (("Validation", "valid"), ("OOS", "test"), ("Forward", "oos")):
        region = (regions or {}).get(key)
        if region and region.get("start"):
            stamp = pd.to_datetime(region["start"], utc=True)
            if index[0] <= stamp <= index[-1]:
                boundaries.append((label, stamp))
    paths = {}
    for scale, filename in CURVE_FILES.items():
        figure = Figure(figsize=(11, 5.5), layout="constrained")
        FigureCanvasAgg(figure)
        top, bottom = figure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        top.plot(plot_equity.index, plot_equity.values, color="#176b99", linewidth=1.2)
        top.axhline(baseline, color="gray", linestyle=":", linewidth=.8)
        if scale == "log":
            if np.all(values > 0):
                if plot_equity.min() == plot_equity.max():
                    top.set_ylim(baseline * .9, baseline * 1.1)
                top.set_yscale("log")
            else:
                top.set_yscale("symlog", linthresh=max(baseline * .001, .01))
                top.text(.01, .04, "Non-positive equity: symmetric log scale", transform=top.transAxes, fontsize=8)
        top.set_title(f"{title}\nFull continuous account | daily closing equity | {scale} scale", fontsize=10)
        top.set_ylabel("Account equity")
        bottom.fill_between(daily_drawdown.index, daily_drawdown.values, 0, color="#b44747", alpha=.65)
        bottom.set_ylabel("Drawdown (%)")
        bottom.set_xlabel("Date (UTC)")
        for label, stamp in boundaries:
            for axis in (top, bottom):
                axis.axvline(stamp, color="#666666", linestyle="--", linewidth=.8)
            top.text(stamp, .98, f" {label}", transform=top.get_xaxis_transform(), va="top", fontsize=8)
        for axis in (top, bottom):
            axis.grid(alpha=.2)
        path = output / filename
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            figure.savefig(temporary, format="png", dpi=120)
            os.replace(temporary, path)
        finally:
            figure.clear()
            temporary.unlink(missing_ok=True)
        paths[scale] = str(path.absolute())
    return paths


def _backfill_one(item):
    history, regions, title, identity, period = item
    paths = save_account_equity_curves(history, regions=regions, title=title)
    return {"hash": identity, "period": period, **paths}


def backfill_experiment(experiment_dir, *, workers=8):
    """Render every current report's stored account without replaying trades."""
    root = Path(experiment_dir).absolute()
    items, seen = [], set()
    with (root / "reports.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            report = json.loads(line)
            identity = report["params"]["hash"]
            for period, result in report["results"].items():
                key = (identity, period)
                if key in seen:
                    raise ValueError(f"Duplicate report identity: {key}")
                seen.add(key)
                history = root / "sim_output" / identity / period / "account_history.csv.gz"
                if not history.is_file():
                    raise FileNotFoundError(f"Stored account history is missing: {history}")
                title = f"{period} | run {identity}"
                if report.get("fusion_hash"):
                    trigger = report["TRIGGER"]["train_params"]["model_cfg"] if "TRIGGER" in report else None
                    direction = report["DIRECTION"]["train_params"]["model_cfg"] if "DIRECTION" in report else None
                    if trigger and direction:
                        title += (f" | seq={trigger['seq_len']} | Trigger={trigger['xgb_depth']}/{trigger['xgb_estimators']}"
                                  f" | Direction={direction['xgb_depth']}/{direction['xgb_estimators']}")
                title += f" | hold={report['params']['strategy'].get('fixed_hold_bars')} bars"
                items.append((str(history), result["time"].get("regions", {}), title, identity, period))
    logging.info("Rendering equity curves for %d backtests with %d workers", len(items), workers)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for row in executor.map(_backfill_one, items, chunksize=1):
            rows.append(row)
            if len(rows) % 100 == 0 or len(rows) == len(items):
                logging.info("Equity curves complete: %d/%d", len(rows), len(items))
    path = root / "equity_curve_index.csv"
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["hash", "period", *CURVE_FILES])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    logging.info("Equity curve index saved: %s", path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render equity curves from persisted batch accounts")
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    backfill_experiment(args.experiment_dir, workers=args.workers)
