#!/usr/bin/env python3
"""Compare 5-day and 1/2/3/6-month retraining for Trigger, Direction and their fusion.

Reuse prepared artifacts; no downloading, labeling, trading or hyperparameter search.
Run from the repository root:
    python experiment/retraining_frequency_comparison.py --plan-only
    python experiment/retraining_frequency_comparison.py
    python experiment/retraining_frequency_comparison.py --include-forward
    python experiment/retraining_frequency_comparison.py --out <existing-run> --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import copy
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_process import common
from model.train import train_windows, validate_windows, label_maturity
from model.train_config import TriggerModel, DirectionModel

LOG = logging.getLogger("retraining_frequency")
CADENCES = ("5d", "1m", "2m", "3m", "6m")
BASELINE = "1m"
TASKS = ("trigger", "direction", "fusion")
KEYS = ["symbol", "open_time_date_utc"]
PROBABILITY_COLUMNS = ["sub_trigger_prob_1", "sub_direction_prob_1"]


def write_json(path, payload):
    def clean(value):
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [clean(item) for item in value]
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    Path(path).write_text(json.dumps(clean(payload), indent=2, default=str, allow_nan=False) + "\n")


def utc(value):
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def cadence_offset(cadence, count=1):
    value, unit = int(cadence[:-1]), cadence[-1]
    if unit not in ("d", "m") or value < 1:
        raise ValueError(f"Invalid cadence: {cadence}")
    return pd.DateOffset(**{("days" if unit == "d" else "months"): value * count})


def make_windows(first, end, cadence, train_start, validation_months, mode, train_months):
    """Anchor every cadence at the same first OOS timestamp; truncate only its tail."""
    result = []
    index = 0
    while first + cadence_offset(cadence, index) < end:
        start = first + cadence_offset(cadence, index)
        valid_start = start - pd.DateOffset(months=validation_months)
        result.append(dict(
            fold_id=f"{index:04d}",
            train_start=(train_start if mode == "expanding" else valid_start - pd.DateOffset(months=train_months)).isoformat(),
            validation_start=valid_start.isoformat(), oos_start=start.isoformat(),
            oos_end=min(first + cadence_offset(cadence, index + 1), end).isoformat(),
        ))
        index += 1
    return result


def load_inputs(directory):
    manifest = common.load_data_manifest_from_dir(str(directory))
    para = common.load_pre_params_from_dir(str(directory))
    if para.label_type not in ("TBM", "BBM", "FTHL"):
        raise ValueError("This comparison requires bounded-horizon Signal labels: TBM, BBM or FTHL")
    frame = pd.concat([
        common.load_train_df_from_dir(str(directory)), common.load_test_df_from_dir(str(directory)),
    ], ignore_index=True)
    if "symbol" not in frame:
        frame["symbol"] = para.symbol
    frame["open_time_date_utc"] = pd.to_datetime(frame.open_time_date_utc, utc=True)
    if frame.duplicated(KEYS).any():
        raise ValueError("Prepared symbol/time keys must be unique")
    if set(frame.symbol.unique()) != set(para.symbols):
        raise ValueError("Prepared symbols disagree with the metadata")
    return manifest, para, frame.sort_values(KEYS).reset_index(drop=True)


def design(args, manifest, para, frame, configs):
    interval = pd.Timedelta(milliseconds=common.get_interval_ms(para.interval))
    starts = frame.groupby("symbol").open_time_date_utc.min()
    ends = frame.groupby("symbol").open_time_date_utc.max() + interval
    history_start, history_end = starts.max(), ends.min()
    forward = utc(manifest["split_time"])
    requested_end = utc(args.end) if args.end else history_end
    if requested_end > history_end:
        raise ValueError("Requested end exceeds the common symbol history")
    development_end = min(forward, requested_end)
    earliest = history_start + pd.DateOffset(months=args.train_months + args.validation_months)
    default_start = earliest
    if args.development_months is not None:
        default_start = max(earliest, development_end - pd.DateOffset(months=args.development_months))
    first = utc(args.start) if args.start else default_start
    if first < earliest or first >= development_end:
        raise ValueError("Development start must allow full train/validation history and precede the Forward boundary")
    train_start = first - pd.DateOffset(months=args.train_months + args.validation_months)
    phases = {"development": (first, development_end)}
    if args.include_forward:
        if requested_end <= forward:
            raise ValueError("No Forward observations available within the requested end")
        phases["forward"] = (forward, requested_end)
    windows = {
        phase: {str(cadence): validate_windows(
            make_windows(begin, end, cadence, train_start, args.validation_months, args.mode, args.train_months), forward,
        ) for cadence in CADENCES}
        for phase, (begin, end) in phases.items()
    }
    state_boundaries = {}
    if "expected_vol" in frame:
        cutoff = first - pd.DateOffset(months=args.validation_months)
        for symbol in para.symbols:
            history = frame.loc[(frame.symbol == symbol) & (frame.open_time_date_utc >= train_start)
                & (frame.open_time_date_utc < cutoff), "expected_vol"].to_numpy(float)
            history = history[np.isfinite(history)]
            if len(history):
                state_boundaries[symbol] = np.quantile(history, [.1, .9]).tolist()
    return dict(
        schema_version=2, data_id=manifest["data_id"], preparation=asdict(para),
        symbols=list(para.symbols), configs=[asdict(config) for config in configs],
        phases={phase: {"start": begin.isoformat(), "end_exclusive": end.isoformat()}
                for phase, (begin, end) in phases.items()},
        mode=args.mode, initial_train_months=args.train_months, validation_months=args.validation_months,
        historical_volatility_state_boundaries=state_boundaries,
        cadences=CADENCES, windows=windows, seed=args.seed,
        bootstrap_replicates=args.bootstrap, bootstrap_block_days=[1, 7, 30],
        backtest=dict(enabled=not args.skip_backtest, task="fusion", symbols=list(para.symbols),
            initial_equity=args.initial_equity, commission_pct=args.commission_pct,
            leverage=args.leverage, risk_per_trade_pct=args.risk_per_trade_pct,
            fixed_hold_bars=para.predict_num, model_gate="existing hard-gate fusion",
            interval="Full common phase OHLC; finite-probability keys shared across all arms; no future-label filtering",
            execution="Existing Backtrader runner defaults, including cheat_on_close=True and OHLC bracket matching"),
        selection="Lowest development pooled Fusion logloss; all other tasks/metrics diagnostic",
        evaluation="Same finite-probability symbol/time keys in all arms; valid labels mature by phase end",
        direction_scope="True non-neutral labels, independent of Trigger predictions",
        inference="Every valid candle; no per-retraining-boundary evaluation purge",
        probability_rule="Range=1-trigger; Up=trigger*direction; Down=trigger*(1-direction)",
        fusion_decision="Current deployed hard gate: Trigger > 0.5, then Direction > 0.5; soft product probabilities used for primary logloss",
        phase_boundary_policy="Reset all update calendars at Forward start, enabling clean phase separation",
        baseline=BASELINE, primary_metric="soft_fusion_logloss", threshold_rule="Binary argmax; Fusion current hard gate",
        statistical_scope="Paired non-circular time-block Bootstrap, all symbols resampled together; pointwise and simultaneous intervals",
        forward_caveat="Historical Forward data is a chronological holdout, not a claim of prospectively untouched data",
        args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
              if key not in ("resume", "plan_only", "out")},
    )


def task_values(frame, task):
    labels = frame.label.to_numpy(int)
    trigger = frame.sub_trigger_prob_1.to_numpy(float)
    direction = frame.sub_direction_prob_1.to_numpy(float)
    if task == "trigger":
        y = (labels != 0).astype(int)
        probabilities = np.column_stack([1 - trigger, trigger])
        keep = np.ones(len(frame), dtype=bool)
    elif task == "direction":
        y = (labels == 1).astype(int)
        probabilities = np.column_stack([1 - direction, direction])
        keep = labels != 0
    else:
        y = labels + 1
        probabilities = np.column_stack([trigger * (1 - direction), 1 - trigger, trigger * direction])
        keep = np.ones(len(frame), dtype=bool)
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    clipped /= clipped.sum(axis=1, keepdims=True)
    loss = -np.log(clipped[np.arange(len(y)), y])
    return y, probabilities, keep, loss


def metrics(frame, task):
    y, p, keep, loss = task_values(frame, task)
    y, p, loss = y[keep], p[keep], loss[keep]
    if not len(y):
        return dict(samples=0)
    soft_predicted = p.argmax(axis=1)
    predicted = soft_predicted
    if task == "fusion":
        predicted = np.where(frame.sub_trigger_prob_1.to_numpy() > .5,
            np.where(frame.sub_direction_prob_1.to_numpy() > .5, 2, 0), 1)[keep]
    classes = list(range(p.shape[1]))
    result = dict(
        samples=len(y), logloss=float(loss.mean()), accuracy=float(accuracy_score(y, predicted)),
        balanced_accuracy=float(recall_score(y, predicted, labels=np.unique(y), average="macro", zero_division=0)),
        true_class_count=int(len(np.unique(y))),
        macro_f1=float(f1_score(y, predicted, labels=classes, average="macro", zero_division=0)),
        confusion_matrix=confusion_matrix(y, predicted, labels=classes).tolist(),
        precision_by_class=precision_score(y, predicted, labels=classes, average=None, zero_division=0).tolist(),
        recall_by_class=recall_score(y, predicted, labels=classes, average=None, zero_division=0).tolist(),
    )
    if task != "fusion":
        result.update(
            brier=float(((p[:, 1] - y) ** 2).mean()),
            pr_auc=float(average_precision_score(y, p[:, 1])) if (y == 1).any() else np.nan,
            roc_auc=float(roc_auc_score(y, p[:, 1])) if len(np.unique(y)) == 2 else np.nan,
            positive_rate=float(y.mean()), positive_precision=result["precision_by_class"][1],
            positive_recall=result["recall_by_class"][1],
        )
    else:
        result["directional_coverage"] = float((predicted != 1).mean())
        result["soft_argmax_macro_f1"] = float(f1_score(y, soft_predicted, labels=classes, average="macro", zero_division=0))
        trigger = frame.sub_trigger_prob_1.to_numpy()[keep]
        direction = frame.sub_direction_prob_1.to_numpy()[keep]
        gate = trigger > .5
        hard_p = np.column_stack([np.where(gate, 1 - direction, 0), (~gate).astype(float), np.where(gate, direction, 0)])
        result["hard_gate_logloss_clipped"] = float(-np.log(np.clip(hard_p[np.arange(len(y)), y], 1e-12, 1)).mean())
    return result


def align_predictions(predictions, truth, begin, end, interval_ms):
    """Enforce identical evaluation keys after labels mature at the phase boundary."""
    interval = pd.Timedelta(milliseconds=interval_ms)
    eligible = truth.loc[
        (truth.open_time_date_utc >= begin) & (truth.open_time_date_utc < end)
        & truth.label.isin([-1, 0, 1])
        & (truth.open_time_date_utc + interval * (truth.attrs["predict_num"] + 1) <= end),
        KEYS + ["label"] + (["volatility_state"] if "volatility_state" in truth else []),
    ].set_index(KEYS).sort_index()
    common_keys = eligible.index
    coverage = []
    for cadence, frame in predictions.items():
        if frame.duplicated(KEYS).any():
            raise ValueError(f"Duplicate OOS keys in cadence {cadence}")
        indexed = frame.set_index(KEYS)
        values = indexed[PROBABILITY_COLUMNS].to_numpy(float)
        valid = np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1) & (values <= 1).all(axis=1)
        keys = indexed.index[valid].intersection(eligible.index)
        common_keys = common_keys.intersection(keys)
        coverage.append(dict(cadence=cadence, eligible=len(eligible), available=len(keys)))
    if not len(common_keys):
        raise ValueError("No common valid evaluation timestamps across cadences")
    result = {}
    for cadence, frame in predictions.items():
        result[cadence] = frame.set_index(KEYS).loc[common_keys].join(eligible).reset_index().sort_values(KEYS)
    for row in coverage:
        row.update(common_samples=len(common_keys), coverage=len(common_keys) / len(eligible) if len(eligible) else 0)
    return result, coverage


def paired_bootstrap(aligned, interval_ms, horizon, replicates, seed):
    """Bootstrap calendar-time blocks of paired losses, preserving missing/state runs."""
    baseline = aligned[BASELINE]
    competitors = [cadence for cadence in CADENCES if cadence != BASELINE]
    width = len(competitors)
    times = pd.DatetimeIndex(sorted(baseline.open_time_date_utc.unique()))
    grid = pd.date_range(times[0], times[-1], freq=pd.Timedelta(milliseconds=interval_ms))
    block_lengths = sorted(set(max(horizon, int(days * 86400000 / interval_ms)) for days in (1, 7, 30)))
    result = []
    for task in TASKS:
        _, _, valid, base_loss = task_values(baseline, task)
        contributions = []
        for cadence in competitors:
            _, _, keep, loss = task_values(aligned[cadence], task)
            if not np.array_equal(valid, keep):
                raise ValueError("Task evaluation samples differ between arms")
            per_time = pd.DataFrame({
                "time": baseline.open_time_date_utc.to_numpy(),
                "numerator": np.where(keep, loss - base_loss, 0), "denominator": keep.astype(float),
            }).groupby("time")[["numerator", "denominator"]].sum().reindex(grid, fill_value=0)
            contributions.append(per_time.to_numpy())
        values = np.stack(contributions, axis=1)
        observed = values[:, :, 0].sum(axis=0) / values[:, :, 1].sum(axis=0)
        prefix = np.concatenate([np.zeros((1, width, 2)), values.cumsum(axis=0)])
        n = len(grid)
        for length in block_lengths:
            if length > n:
                LOG.warning("Skipping Bootstrap block %d: only %d time bars", length, n)
                continue
            count, tail = divmod(n, length)
            rng = np.random.default_rng(seed + length)
            draws = np.empty((replicates, width))
            for start in range(0, replicates, 32):
                size = min(32, replicates - start)
                indices = rng.integers(0, n - length + 1, (size, count))
                sums = (prefix[indices + length] - prefix[indices]).sum(axis=1)
                if tail:
                    indices = rng.integers(0, n - tail + 1, size)
                    sums += prefix[indices + tail] - prefix[indices]
                draws[start:start + size] = np.divide(sums[:, :, 0], sums[:, :, 1],
                    out=np.full((size, width), np.nan), where=sums[:, :, 1] > 0)
            draws = draws[np.isfinite(draws).all(axis=1)]
            if len(draws) < replicates * .8:
                LOG.warning("Sparse Bootstrap support for %s: %d/%d draws", task, len(draws), replicates)
            if len(draws) < 2:
                continue
            se = draws.std(axis=0, ddof=1)
            critical = np.quantile(np.max(np.abs((draws - observed) / np.where(se > 0, se, 1)), axis=1), .95)
            low, high = np.quantile(draws, [.025, .975], axis=0)
            for column, cadence in enumerate(competitors):
                result.append(dict(task=task, cadence=cadence, baseline=BASELINE,
                    loss="logloss", block_bars=length, approximate_blocks=n / length,
                    mean_difference=observed[column], ci_low=low[column], ci_high=high[column],
                    simultaneous_ci_low=observed[column] - critical * se[column],
                    simultaneous_ci_high=observed[column] + critical * se[column], valid_replicates=len(draws)))
    return result


def evaluate_phase(phase, predictions, truth, bounds, interval_ms, args, output):
    begin, end = utc(bounds["start"]), utc(bounds["end_exclusive"])
    aligned, coverage = align_predictions(predictions, truth, begin, end, interval_ms)
    records = []
    for cadence, frame in aligned.items():
        for task in TASKS:
            records.append(dict(phase=phase, cadence=cadence, task=task, scope="overall", group="all", **metrics(frame, task)))
        monthly = frame.open_time_date_utc.dt.strftime("%Y-%m")
        yearly = frame.open_time_date_utc.dt.strftime("%Y")
        age = frame.age_month.astype(str)
        groups = [("month", monthly), ("year", yearly), ("model_age_month", age), ("model_age_day", frame.age_day.astype(int).astype(str)), ("symbol", frame.symbol)]
        if "volatility_state" in frame:
            groups.append(("volatility_state", frame.volatility_state))
        for scope, grouping in groups:
            for group in sorted(grouping.unique()):
                subset = frame.loc[grouping == group]
                for task in TASKS:
                    records.append(dict(phase=phase, cadence=cadence, task=task, scope=scope, group=group, **metrics(subset, task)))
        selected = frame.loc[frame.sub_trigger_prob_1 > .5]
        diagnostic = metrics(selected, "direction")
        records.append(dict(phase=phase, cadence=cadence, task="direction", scope="trigger_selected_diagnostic", group="all", **diagnostic))
        exported = frame.assign(cadence=cadence, phase=phase)
        trigger, direction = exported.sub_trigger_prob_1, exported.sub_direction_prob_1
        exported["soft_prob_down"] = trigger * (1 - direction)
        exported["soft_prob_range"] = 1 - trigger
        exported["soft_prob_up"] = trigger * direction
        exported.to_parquet(output / f"{phase}_{cadence}_evaluated_predictions.parquet", index=False)
    bootstrap = paired_bootstrap(aligned, interval_ms, truth.attrs["predict_num"], args.bootstrap, args.seed)
    pd.DataFrame(coverage).to_csv(output / f"{phase}_coverage.csv", index=False)
    if any(row["coverage"] < 1 for row in coverage):
        LOG.warning("%s uses common prediction coverage below 100%%; see coverage CSV", phase)
    pd.DataFrame(records).to_csv(output / f"{phase}_metrics.csv", index=False)
    pd.DataFrame(bootstrap).to_csv(output / f"{phase}_bootstrap.csv", index=False)
    return records, bootstrap


def train_phase(phase, plan, configs, args, output):
    predictions, costs = {}, []
    for cadence in CADENCES:
        directory = output / phase / cadence
        windows = plan["windows"][phase][str(cadence)]
        manifest_path = directory / "rolling_manifest.json"
        clock = time.perf_counter()
        reused = False
        if args.resume and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            if manifest["status"] != "complete":
                raise ValueError(f"Incomplete training arm cannot be reused: {directory}; use a new output directory")
            if manifest["data_id"] != plan["data_id"] or manifest["planned_windows"] != windows or manifest["configs"] != json.loads(json.dumps(plan["configs"])):
                raise ValueError(f"Existing arm differs from the frozen plan: {directory}")
            reused = True
        else:
            LOG.info("Training %s cadence=%s folds=%d", phase, cadence, len(windows))
            manifest = train_windows(configs=configs, windows=windows,
                prep_output_dir=str(args.prep_output_dir), save_dir=str(directory), logger=LOG)
        elapsed = time.perf_counter() - clock
        frame = pd.read_pickle(directory / manifest["predictions"])
        frame["open_time_date_utc"] = pd.to_datetime(frame.open_time_date_utc, utc=True)
        for fold in manifest["folds"]:
            start, end = utc(fold["oos_start"]), utc(fold["oos_end"])
            selected = (frame.open_time_date_utc >= start) & (frame.open_time_date_utc < end)
            stamp = frame.loc[selected, "open_time_date_utc"]
            frame.loc[selected, "model_updated_at_utc"] = start.isoformat()
            # Count exact calendar anniversaries of the update timestamp.
            age = sum((stamp >= start + pd.DateOffset(months=month)).astype(int)
                      for month in range(1, int(cadence[:-1]))) if cadence.endswith("m") and int(cadence[:-1]) > 1 else np.zeros(len(stamp), dtype=int)
            frame.loc[selected, "age_month"] = np.asarray(age) + 1
            frame.loc[selected, "age_day"] = ((stamp - start) / pd.Timedelta(days=1)).astype(int).to_numpy() + 1
        predictions[cadence] = frame
        cost = dict(phase=phase, cadence=cadence, retrainings=len(windows), fitted_models=2 * len(windows),
                    elapsed_seconds=elapsed, reused=reused, oos_prediction_rows=len(frame))
        cost_path = directory / "experiment_cost.json"
        if reused and cost_path.is_file():
            cost = json.loads(cost_path.read_text()) | {"reused": True}
        else:
            write_json(cost_path, cost)
        costs.append(cost)
    return predictions, costs


def backtest_phase(phase, plan, predictions, truth, args, output):
    """Backtest frozen fused OOS signals through the existing strategy/venue runner."""
    from trade.runner.backtest_entry import main as run_backtest
    from trade.runner.analyze_backtest_report import plot_equity_curves
    from trade.runner.config import RunnerConfig, ModelDataConfig, BrokerConfig, ExperimentContext
    from trade.strategy.strategy_bbm import BbmStrategyConfig

    begin, end = (utc(plan["phases"][phase][key]) for key in ("start", "end_exclusive"))
    data_dir = output / phase / "backtest_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # Retain INVALID-label and immature tail candles: labels must never gate trades.
    market = truth.loc[(truth.open_time_date_utc >= begin) & (truth.open_time_date_utc < end)].copy()
    if market.empty:
        raise ValueError("Backtest has no common phase candles")
    forward = utc(common.load_data_manifest_from_dir(str(args.prep_output_dir))["split_time"])
    common.save_train_df_to_dir(market.loc[market.open_time_date_utc < forward].reset_index(drop=True), str(data_dir))
    common.save_test_df_to_dir(market.loc[market.open_time_date_utc >= forward].reset_index(drop=True), str(data_dir))
    for name in ("data_config_meta.json", "data_manifest.json"):
        shutil.copyfile(args.prep_output_dir / name, data_dir / name)
    parent_manifest = json.loads((data_dir / "data_manifest.json").read_text())
    parent_manifest["experiment_market_slice"] = dict(
        parent_data_id=plan["data_id"], start=begin.isoformat(), end_exclusive=end.isoformat(),
    )
    write_json(data_dir / "data_manifest.json", parent_manifest)
    common_keys = None
    for frame in predictions.values():
        finite = np.isfinite(frame[PROBABILITY_COLUMNS].to_numpy(float)).all(axis=1)
        keys = pd.MultiIndex.from_frame(frame.loc[finite, KEYS])
        common_keys = keys if common_keys is None else common_keys.intersection(keys)
    if common_keys is None or not len(common_keys):
        raise ValueError("No common inference keys for backtesting")
    rows = []
    for cadence in CADENCES:
        arm = output / phase / cadence
        view = arm / "backtest_oos_view"
        view.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((arm / "rolling_manifest.json").read_text())
        view_manifest = copy.deepcopy(manifest)
        frame = predictions[cadence].copy()
        keys = pd.MultiIndex.from_frame(frame[KEYS])
        prediction_columns = [name for name in frame if name in (
            "pred", "direction_pred", "pred_prob", "prob_short", "prob_neutral", "prob_long", "net_score", "regression_score",
        ) or name.startswith("sub_")]
        frame.loc[~keys.isin(common_keys), prediction_columns] = np.nan
        frame.to_pickle(view / "oos_predictions.pkl")
        view_manifest.update(predictions="oos_predictions.pkl", development_predictions="oos_predictions.pkl",
                             window_predictions="oos_predictions.pkl")
        for fold in view_manifest["folds"]:
            fold["model_dir"] = str((arm / fold["model_dir"]).resolve())
        write_json(view / "rolling_manifest.json", view_manifest)
        for name in ("train_config.json", "data_config_meta.json", "data_manifest.json"):
            shutil.copyfile(arm / name, view / name)
        for symbol in plan["symbols"]:
            backtest_dir = arm / "backtest" / symbol
            backtest_dir.mkdir(parents=True, exist_ok=True)
            summary_path = backtest_dir / "experiment_summary.json"
            if args.resume and summary_path.is_file():
                rows.append(json.loads(summary_path.read_text()))
                continue
            LOG.info("Backtesting %s %s %s on OOS-only signals", phase, cadence, symbol)
            config = RunnerConfig(
                strategy_config=BbmStrategyConfig(fixed_hold_bars=plan["preparation"]["predict_num"],
                    risk_per_trade_pct=args.risk_per_trade_pct),
                broker_config=BrokerConfig(initial_equity=args.initial_equity,
                    commission_pct=args.commission_pct, leverage=args.leverage),
                data_config=ModelDataConfig(prep_output_dir=str(data_dir), train_output_dir=str(view),
                    symbol=symbol, device="cpu"),
                save_dir=str(backtest_dir), experiment_context=ExperimentContext(git_commit=common.git_revision()),
            )
            runner_period = "long" if phase == "development" else "forward"
            result = run_backtest(LOG, config, period=runner_period, evaluate_stages=False)
            price_file = data_dir / f"{symbol}_price.csv"
            market.loc[market.symbol == symbol, ["open_time_date_utc", "close"]].to_csv(price_file, index=False)
            chart_result = copy.deepcopy(result)
            if runner_period == "long":
                for section in ("report", "report_details"):
                    chart_result[section]["results"]["development"] = chart_result[section]["results"].pop("long")
            equity_path = plot_equity_curves(chart_result, str(backtest_dir), file_name="equity_curve.png",
                price_file=str(price_file), include_oos=True, logger=LOG)
            if equity_path is None:
                raise RuntimeError(f"Backtest did not produce an equity curve: {backtest_dir}")
            write_json(backtest_dir / "equity_payload.json", {
                "report": chart_result["report"], "report_details": chart_result["report_details"],
            })
            common.append_jsonl(str(backtest_dir / "reports.jsonl"), result["report"])
            overall = result["report"]["results"][runner_period]
            performance, drawdown, trades = (overall.get(name) or {} for name in ("performance", "drawdown", "trades"))
            row = dict(phase=phase, cadence=cadence, symbol=symbol, task="fusion",
                cagr_pct=float(performance.get("cagr") or 0) * 100,
                sharpe=performance.get("sharpe"), calmar=performance.get("calmar"),
                max_drawdown_pct=drawdown.get("max_dd_pct"), trades=int(trades.get("total") or 0),
                win_rate=trades.get("win_rate"), backtest_dir=str(backtest_dir),
                gross_return=performance.get("gross_return"), final_equity=performance.get("end_value"),
                actual_start=str(overall["time"]["start"]), actual_end=str(overall["time"]["end"]),
                start=begin.isoformat(), end_exclusive=end.isoformat())
            write_json(summary_path, row)
            rows.append(row)
    for symbol in plan["symbols"]:
        symbol_rows = [row for row in rows if row["symbol"] == symbol]
        paths = [Path(row["backtest_dir"]) / "equity_payload.json" for row in symbol_rows]
        payloads = [json.loads(path.read_text()) for path in paths]
        for scale, suffix in (("linear", "absolute"), ("log", "log")):
            equity_path = plot_equity_curves(
                payloads, str(output), file_name=f"{phase}_{symbol}_equity_comparison_{suffix}.png",
                price_file=str(data_dir / f"{symbol}_price.csv"), equity_scale=scale,
                include_oos=True, label_by_index=True, logger=LOG,
            )
            if equity_path is None:
                raise RuntimeError(f"Backtest comparison did not produce a {scale} equity curve: {phase}/{symbol}")
        write_json(output / f"{phase}_{symbol}_equity_legend.json", {
            f"S{index}": row["cadence"] for index, row in enumerate(symbol_rows)
        })
    return rows


def export_report(output, records, uncertainty, costs, selection):
    frame = pd.DataFrame(records)
    overall = frame[frame.scope == "overall"]
    overall.to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(costs).to_csv(output / "training_costs.csv", index=False)
    pd.DataFrame(uncertainty).to_csv(output / "paired_bootstrap.csv", index=False)
    for phase in overall.phase.unique():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for task, axis in zip(TASKS, axes):
            data = overall[(overall.phase == phase) & (overall.task == task)].sort_values("cadence")
            data = data.set_index("cadence").reindex(CADENCES)
            axis.plot(range(len(CADENCES)), data.logloss, marker="o")
            axis.set(title=task.capitalize(), xlabel="Retraining interval", ylabel="Pooled OOS logloss", xticks=range(len(CADENCES)), xticklabels=CADENCES)
            axis.grid(alpha=.3)
        fig.suptitle(f"Retraining frequency comparison: {phase}")
        fig.tight_layout()
        fig.savefig(output / f"{phase}_logloss.png", dpi=170)
        plt.close(fig)
    lines = ["# Retraining frequency comparison", "",
        "The only experimental change is the retraining/OOS interval: 5 days or 1, 2, 3, 6 calendar months.",
        "All arms use the same initial fit, validation length, model configurations and common OOS evaluation keys.",
        "Trigger and Direction are independently fitted at each update, then fused without fitting a third model.", "",
        f"Development-only selected cadence: **{selection['selected_cadence']}**, by pooled Fusion logloss.",
        "This descriptive selection does not establish statistical superiority. No Forward-based reselection is performed.", "",
        "| Phase | Cadence | Task | Samples | Logloss | Macro-F1 | Balanced accuracy |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in overall.itertuples():
        lines.append(f"| {row.phase} | {row.cadence} | {row.task} | {row.samples} | {row.logloss:.6f} | {row.macro_f1:.4f} | {row.balanced_accuracy:.4f} |")
    lines += ["", "## Paired loss uncertainty versus monthly retraining", "",
        "Negative differences favor the candidate cadence. Intervals include all declared block sensitivities.", "",
        "| Phase | Task | Cadence | Block bars | Mean difference | Pointwise 95% CI | Simultaneous 95% CI |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in uncertainty:
        lines.append(
            f"| {row['phase']} | {row['task']} | {row['cadence']} | {row['block_bars']} | "
            f"{row['mean_difference']:.6f} | [{row['ci_low']:.6f}, {row['ci_high']:.6f}] | "
            f"[{row['simultaneous_ci_low']:.6f}, {row['simultaneous_ci_high']:.6f}] |")
    lines += ["", "## Computation cost", "",
        "Wall time includes fitting, validation, per-window evaluation and prediction export; it is not isolated estimator fit time.", "",
        "| Phase | Cadence | Retrainings | Fitted models | Pipeline seconds |",
        "| --- | --- | --- | --- | --- |"]
    for row in costs:
        lines.append(f"| {row['phase']} | {row['cadence']} | {row['retrainings']} | {row['fitted_models']} | {row['elapsed_seconds']:.2f} |")
    lines += ["", "## Evaluation rules and limitations", "",
        "- Direction is scored only on true non-neutral labels; Trigger decisions never change its primary evaluation population.",
        "- pr_auc reports average precision (the step-weighted precision-recall integral), not trapezoidal interpolation.",
        "- Primary Fusion logloss uses soft joint probabilities [trigger*(1-direction), 1-trigger, trigger*direction] for [Down, Range, Up]. Fusion classification metrics use the existing deployed hard gate (Trigger > 0.5, then Direction > 0.5). Soft-argmax F1 and clipped hard-gate logloss are separate diagnostics. The deployed fusion implementation is not changed.",
        "- A model predicts every eligible candle until the next update. Training/validation labels must mature before their own upper boundary; OOS labels mature by the common phase end, not each update boundary.",
        "- Metrics are recomputed from common stitched predictions. Per-fold reports from train_windows are not used for cadence ranking.",
        "- Month/year/model-age/symbol diagnostics are in each phase's metrics CSV. Where expected_vol exists, Low/Normal/High states use fixed per-symbol initial-training 10%/90% thresholds; no future labels or later test data determine them. Trigger-selected Direction metrics are diagnostic only.",
        "- Paired moving-block Bootstrap resamples consecutive calendar-time blocks, all symbols together. Direction denominators are resampled with losses; neutral/missing rows are not compressed out of time.",
        "- Logloss differences are candidate minus monthly baseline: negative favors the candidate. Pointwise 95% and simultaneous four-cadence intervals are reported per task/block length. They do not jointly cover all tasks and block lengths; Fusion is the primary task.",
        "- Block lengths are at least the label horizon, with 1/7/30-day sensitivity. Blocks longer than the available timeline are skipped. Few long blocks limit confidence; F1/AUC results are descriptive, without independent-sample significance claims.",
        "- In expanding mode later training histories grow; in sliding mode the training length stays fixed. Validation always precedes each update. All arms reset their update calendar at the Forward boundary.",
        "- Natural historical Forward data may have been inspected previously; this study is not a claim of prospectively untouched data.",
        "- Report all costs and interval sensitivity before recommending an update schedule; lowest observed loss alone is insufficient.",
        "- Fusion backtests reuse the existing strategy and broker with fixed settings, OOS-only signals and no future-label filtering. Trigger alone has no direction and is not treated as a standalone trading strategy. Each symbol is an independent account, not a pooled portfolio.",
        "- No transaction-cost assumptions, labels or trading parameters are optimized. Backtests retain the runner's existing OHLC execution assumptions; model metrics are the selection criterion, not strategy profit.", ""]
    (output / "comparison.md").write_text("\n".join(lines))


def run(args):
    output = args.out or Path(common.PERSISTENCE_DIR) / "retraining_frequency" / pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ")
    if output.exists() and not args.resume:
        raise FileExistsError("Output already exists; choose a new directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(output / "experiment.log"),
    ])
    manifest, para, truth = load_inputs(args.prep_output_dir)
    configs = [replace(config, seed=args.seed) for config in (TriggerModel, DirectionModel)]
    plan = design(args, manifest, para, truth, configs)
    plan_path = output / "experiment_plan.json"
    canonical = json.loads(json.dumps(plan, default=str))
    if args.resume and plan_path.is_file() and json.loads(plan_path.read_text()) != canonical:
        raise ValueError("Resume arguments/data/configurations differ from the frozen experiment plan")
    write_json(plan_path, plan)
    write_json(output / "source.json", {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preparation_manifest_sha256": common.sha256_file(str(args.prep_output_dir / "data_manifest.json")),
    })
    LOG.info("Frozen experiment plan: %s", plan_path)
    for phase, arms in plan["windows"].items():
        LOG.info("%s folds: %s", phase, {cadence: len(windows) for cadence, windows in arms.items()})
    if args.plan_only:
        return output
    if plan["historical_volatility_state_boundaries"]:
        truth["volatility_state"] = "Unavailable"
        for symbol, (low, high) in plan["historical_volatility_state_boundaries"].items():
            selected = truth.symbol == symbol
            historical = truth.loc[selected, "expected_vol"].to_numpy(float)
            truth.loc[selected, "volatility_state"] = np.where(~np.isfinite(historical), "Unavailable",
                np.where(historical <= low, "Low", np.where(historical > high, "High", "Normal")))
    truth.attrs["predict_num"] = para.predict_num
    all_records, all_uncertainty, all_costs, backtests = [], [], [], []
    for phase in plan["phases"]:
        predictions, costs = train_phase(phase, plan, configs, args, output)
        records, uncertainty = evaluate_phase(phase, predictions, truth, plan["phases"][phase],
            common.get_interval_ms(para.interval), args, output)
        all_records.extend(records)
        all_uncertainty.extend(dict(phase=phase, **row) for row in uncertainty)
        all_costs.extend(costs)
        if phase == "development":
            candidates = [row for row in records if row["task"] == "fusion" and row["scope"] == "overall"]
            selected = min(candidates, key=lambda row: row["logloss"])
            selection = dict(selected_cadence=selected["cadence"], criterion="development pooled Fusion logloss",
                data_id=manifest["data_id"], selected_before_forward=True)
            write_json(output / "development_selection.json", selection)
        if not args.skip_backtest:
            backtests.extend(backtest_phase(phase, plan, predictions, truth, args, output))
            pd.DataFrame(backtests).to_csv(output / "backtest_comparison.csv", index=False)
    export_report(output, all_records, all_uncertainty, all_costs, selection)
    if backtests:
        lines = ["", "## Fusion OOS strategy backtests", "",
            "Fixed strategy/fees across cadences; independent accounts per symbol. See backtest_comparison.csv and per-arm equity curves. Comparison charts are exported separately as *_equity_comparison_absolute.png (linear equity) and *_equity_comparison_log.png (log equity).", "",
            "| Phase | Cadence | Symbol | CAGR % | Sharpe | Max DD % | Trades | Win rate |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in backtests:
            lines.append(f"| {row['phase']} | {row['cadence']} | {row['symbol']} | {row['cagr_pct']:.2f} | {row['sharpe']} | {row['max_drawdown_pct']} | {row['trades']} | {row['win_rate']} |")
        with (output / "comparison.md").open("a") as handle:
            handle.write("\n".join(lines) + "\n")
    LOG.info("Completed: %s", output / "comparison.md")
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-output-dir", type=Path, default=Path(common.DATA_OUT_DIR))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--mode", choices=("expanding", "sliding"), default="expanding")
    parser.add_argument("--train-months", type=int, default=12, help="Initial training months; fixed length in sliding mode")
    parser.add_argument("--validation-months", type=int, default=2)
    parser.add_argument("--development-months", type=int, default=None,
                        help="Explicitly limit development OOS history; default uses all available history")
    parser.add_argument("--start", help="Common first development OOS timestamp (UTC)")
    parser.add_argument("--end", help="Common exclusive end timestamp (UTC)")
    parser.add_argument("--include-forward", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true", help="Only compare model metrics")
    parser.add_argument("--initial-equity", type=float, default=10000.0)
    parser.add_argument("--commission-pct", type=float, default=.05, help="Commission percentage, not a fraction")
    parser.add_argument("--leverage", type=float, default=20.0)
    parser.add_argument("--risk-per-trade-pct", type=float, default=.015, help="Fraction of equity risked per trade")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed matching arms; incomplete arms require a new output directory")
    args = parser.parse_args(argv)
    if min(args.train_months, args.validation_months) < 1 or (
        args.development_months is not None and args.development_months < 1
    ) or args.bootstrap < 1000:
        parser.error("Month lengths must be positive and --bootstrap must be at least 1000")
    if args.initial_equity <= 0 or args.leverage < 1 or args.commission_pct < 0 or not 0 < args.risk_per_trade_pct < 1:
        parser.error("Invalid broker or risk settings")
    return args


if __name__ == "__main__":
    run(parse_args())
