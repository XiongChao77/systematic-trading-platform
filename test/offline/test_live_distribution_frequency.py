"""Compare captured predictions and filled entry frequency with saved backtests."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def proportions(values):
    counts = values.value_counts().reindex([0, 1, 2], fill_value=0)
    if int(counts.sum()) != len(values):
        raise ValueError("Unexpected prediction classes")
    return counts.to_numpy(float) / len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execution-dir")
    parser.add_argument("--start-utc")
    args = parser.parse_args()
    source, output = Path(args.parity_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((source / "summary.json").read_text())
    execution_dir = Path(args.execution_dir) if args.execution_dir else output / "reconciled"
    executions = pd.read_csv(execution_dir / "executions.csv", low_memory=False)
    fills = pd.read_csv(execution_dir / "fills.csv", low_memory=False)
    executions = executions.drop_duplicates(["instance_id", "execution_id"])
    executions = executions.loc[executions.order_role.eq("entry") & executions.filled_quantity.gt(0)].copy()
    executions["event_time"] = pd.to_datetime(executions.decision_at_utc, utc=True, format="mixed").fillna(
        pd.to_datetime(executions.first_fill_at_utc, utc=True, format="mixed"))
    executions["candle_close"] = (executions.event_time.dt.floor("15min") - pd.Timedelta(milliseconds=1)).dt.as_unit("ms").astype("int64")
    by_model, by_instance, metadata, reports = {}, {}, {}, {}
    aliases = {}
    for record in summary["strategies"]:
        identity, model_hash = record["instance_id"], record["hash_id"]
        detail = pd.read_csv(source / f"{identity}.csv", float_precision="round_trip")
        if args.start_utc:
            start_ms = pd.Timestamp(args.start_utc).value // 1_000_000
            detail = detail.loc[detail.close_time_ms_utc.ge(start_ms)]
        if detail.empty:
            continue
        by_model.setdefault(model_hash, []).append(detail)
        venue, account, symbol, interval, _ = identity.split("|")
        if interval != "15m":
            raise ValueError("This audit expects 15-minute traces")
        canonical = identity
        if venue == "ctrader":
            candidates = fills.loc[fills.execution_id.astype(str).str.startswith(f"ctrader-{account}-"), "instance_id"]
            accounts = {value.split("|")[1] for value in candidates}
            if len(accounts) == 1:
                canonical = "|".join([venue, accounts.pop(), symbol, interval, model_hash])
        aliases[identity] = canonical
        by_instance.setdefault(canonical, []).append(detail)
        metadata[model_hash] = record
        if model_hash not in reports:
            root = Path(record["model_path"]).parent.parent
            paths = [root / "selected_configs.jsonl"] + sorted(root.glob("selected_configs_*.jsonl"))
            for path in paths:
                if not path.exists():
                    continue
                for line in path.read_text().splitlines():
                    candidate = json.loads(line)
                    if candidate["params"]["hash"] == model_hash:
                        reports[model_hash] = (path, candidate)
                        break
                if model_hash in reports:
                    break
            if model_hash not in reports:
                raise ValueError(f"Missing historical report: {model_hash}")

    def combine(frames):
        frame = pd.concat(frames, ignore_index=True)
        if frame.groupby("close_time_ms_utc").pred_live.nunique().gt(1).any():
            raise ValueError("Conflicting predictions for the same model and candle")
        return frame.drop_duplicates("close_time_ms_utc").sort_values("close_time_ms_utc")

    distribution = []
    for model_hash, frames in by_model.items():
        live = combine(frames)
        report_path, report = reports[model_hash]
        live_p, replay_p = proportions(live.pred_live), proportions(live.pred_backtest)
        for period in ["long", "forward"]:
            historical = report["results"][period]
            counts = historical["model_metrics"]["label_distribution_pred"]
            hist_p = np.array([counts.get(str(c), 0) for c in [0, 1, 2]], dtype=float)
            hist_n = hist_p.sum()
            hist_p /= hist_n
            row = {
                "symbol": metadata[model_hash]["symbol"], "hash_id": model_hash,
                "period": period, "live_bars": len(live), "historical_bars": int(hist_n),
                "live_start_utc": pd.to_datetime(live.close_time_ms_utc.iloc[0], unit="ms", utc=True).isoformat(),
                "live_end_utc": pd.to_datetime(live.close_time_ms_utc.iloc[-1], unit="ms", utc=True).isoformat(),
                "historical_start_utc": historical["time"]["start"],
                "historical_end_utc": historical["time"]["end"],
                "report_path": str(report_path.resolve()),
                "same_period_class_mismatches": int(live.pred_live.ne(live.pred_backtest).sum()),
                "total_variation": float(np.abs(live_p - hist_p).sum() / 2),
                "live_directional_share": float(live_p[0] + live_p[2]),
                "historical_directional_share": float(hist_p[0] + hist_p[2]),
                "live_probability_mean": float(live.pred_prob_live.mean()),
                "replay_probability_mean": float(live.pred_prob_backtest.mean()),
                "live_net_score_mean": float(live.net_score_live.mean()),
                "replay_net_score_mean": float(live.net_score_backtest.mean()),
            }
            for i, name in enumerate(["short", "neutral", "long"]):
                row[f"live_{name}_share"] = live_p[i]
                row[f"historical_{name}_share"] = hist_p[i]
                row[f"replay_{name}_share"] = replay_p[i]
            distribution.append(row)
    pd.DataFrame(distribution).to_csv(output / "prediction_distribution.csv", index=False)

    frequencies = []
    daily_rows = []
    for identity, frames in by_instance.items():
        live = combine(frames)
        venue, _, symbol, _, model_hash = identity.split("|")
        first, last = int(live.close_time_ms_utc.iloc[0]), int(live.close_time_ms_utc.iloc[-1])
        calendar_days = (last - first + 900_000) / 86_400_000
        observed_days = len(live) / 96
        entry_executions = executions.loc[executions.instance_id.eq(identity) & executions.candle_close.between(first, last)]
        # Recovered cTrader split children may have separate synthetic execution
        # IDs. Count one filled directional entry per strategy candle, while
        # retaining raw execution counts for auditability.
        entries = entry_executions.drop_duplicates(["candle_close", "side"])
        observed_entries = entries.loc[entries.candle_close.isin(live.close_time_ms_utc)]
        report_path, report = reports[model_hash]
        historical = report["results"]["forward"]
        start, end = pd.to_datetime(historical["time"]["start"], utc=True), pd.to_datetime(historical["time"]["end"], utc=True)
        hist_days = max((end - start).total_seconds() / 86400, 1)
        trade_path = report_path.parent / "sim_output" / model_hash / "report_details.json"
        trade_logs = pd.DataFrame(json.loads(trade_path.read_text())["results"]["forward"]["trade_logs"])
        opens = trade_logs.loc[trade_logs.role.eq("open")].drop_duplicates("order_ref")
        hist_rate = len(opens) / hist_days
        # Historical rolling calendar-window counts show the natural variability
        # of trade rates without assuming independent candles or Poisson events.
        hist_times = pd.to_datetime(opens.dt, unit="s", utc=True).sort_values().dt.as_unit("ns").astype("int64").to_numpy()
        window_starts = pd.date_range(start, end - pd.Timedelta(days=calendar_days), freq="D").as_unit("ns").astype("int64").to_numpy()
        window_ends = window_starts + int(calendar_days * 86400 * 1e9)
        counts = np.searchsorted(hist_times, window_ends) - np.searchsorted(hist_times, window_starts)
        rates = counts / calendar_days
        rate = len(entries) / calendar_days
        row = {
            "instance_id": identity, "symbol": symbol, "venue": venue, "hash_id": model_hash,
            "calendar_days": calendar_days, "observed_prediction_days": observed_days,
            "prediction_coverage": observed_days / calendar_days,
            "filled_execution_records": len(entry_executions),
            "consolidated_split_records": len(entry_executions) - len(entries),
            "filled_entry_count": len(entries), "entries_on_observed_candles": len(observed_entries),
            "entries_on_unobserved_candles": len(entries) - len(observed_entries),
            "live_entries_per_calendar_day": rate,
            "live_entries_per_observed_day": len(observed_entries) / observed_days,
            "historical_open_entries": len(opens), "historical_days": hist_days,
            "historical_entries_per_day": hist_rate,
            "historical_report_closed_trades_per_day": historical["trades"]["daily_freq"],
            "calendar_frequency_ratio": rate / hist_rate,
            "observed_frequency_ratio": len(observed_entries) / observed_days / hist_rate,
            "historical_window_rate_p05": float(np.quantile(rates, .05)),
            "historical_window_rate_p95": float(np.quantile(rates, .95)),
            "historical_window_rate_max": float(np.max(rates)),
            "historical_window_percentile": float(np.mean(rates <= rate)),
            "historical_window_count": len(rates),
            "live_buy_share": float(entries.side.eq("buy").mean()) if len(entries) else None,
            "historical_buy_share": float(opens.is_buy.mean()),
        }
        frequencies.append(row)
        days = pd.to_datetime(live.close_time_ms_utc, unit="ms", utc=True).dt.floor("D")
        for day, bars in live.groupby(days):
            es = observed_entries.loc[observed_entries.event_time.dt.floor("D").eq(day)]
            daily_rows.append({"instance_id": identity, "date_utc": day.date().isoformat(),
                               "prediction_bars": len(bars), "filled_entries": len(es)})
    pd.DataFrame(frequencies).to_csv(output / "trade_frequency.csv", index=False)
    pd.DataFrame(daily_rows).to_csv(output / "daily_observations.csv", index=False)
    distribution_frame = pd.DataFrame(distribution)
    frequency_frame = pd.DataFrame(frequencies)
    overview = []
    for symbol, group in distribution_frame.loc[distribution_frame.period.eq("forward")].groupby("symbol"):
        frequency = frequency_frame.loc[frequency_frame.symbol.eq(symbol)]
        row = {"symbol": symbol, "models": len(group), "instances": len(frequency),
               "model_candle_observations": int(group.live_bars.sum()),
               "filled_entries": int(frequency.filled_entry_count.sum()),
               "observed_coverage": frequency.observed_prediction_days.sum() / frequency.calendar_days.sum(),
               "live_entries_per_observed_day": frequency.entries_on_observed_candles.sum() / frequency.observed_prediction_days.sum(),
               "historical_entries_per_day": np.average(frequency.historical_entries_per_day, weights=frequency.observed_prediction_days),
               "frequency_ratio": frequency.entries_on_observed_candles.sum() / (frequency.historical_entries_per_day * frequency.observed_prediction_days).sum()}
        for prefix in ["live", "historical", "replay"]:
            for name in ["short", "neutral", "long"]:
                key = f"{prefix}_{name}_share"
                row[key] = np.average(group[key], weights=group.live_bars)
        overview.append(row)
    pd.DataFrame(overview).to_csv(output / "overview.csv", index=False)
    (output / "method.json").write_text(json.dumps({
        "prediction_source": str(source.resolve()),
        "start_utc_filter": args.start_utc,
        "historical_baseline": "Saved forward-period reports; long-period distributions are secondary",
        "prediction_aggregation": "One observation per model hash and candle; account duplicates removed",
        "trade_count": "Filled directional entry candles; executions deduplicated by instance and execution ID, then grouped by instance, candle and side to consolidate recovered split children",
        "calendar_alignment": "Decision time, falling back to first fill time, floored to 15 minutes minus 1 ms",
        "frequency_denominators": "Calendar span and observed prediction bars / 96, reported separately",
        "aliases_inferred_from_fill_execution_ids": aliases,
        "unresolved_models": summary["unresolved"],
        "limitations": ["Recorded prediction coverage is not an authoritative uptime measure",
                        "Historical and live periods contain different market conditions",
                        "Rolling historical trade windows are descriptive, not an independence-based significance test",
                        "Historical probability-score distributions are not available in these summary reports"],
    }, indent=2) + "\n")
    print(f"Wrote {len(distribution)} distribution rows and {len(frequencies)} frequency rows")


if __name__ == "__main__":
    main()
