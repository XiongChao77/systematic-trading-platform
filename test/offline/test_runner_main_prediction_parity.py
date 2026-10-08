"""Audit consolidated traces against batch inference and sampled live replay."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from data_process import common
from model.data_loader import TimeSeriesWindowDataset
from test.offline.test_verify_prediction_traces import compare_rows
from trade.recording.prediction_trace import MARKET_COLUMNS, PREDICTION_COLUMNS
from trade.runner.live_runner import (
    LiveRunner,
    _prepare_market_frame,
    load_live_strategy_specs,
    load_params_from_report,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(detail):
    return {
        "rows": len(detail),
        "mismatched_rows": int((~detail.matches).sum()),
        "mismatch_count": {
            c: int((~detail[f"{c}_matches"]).sum()) for c in PREDICTION_COLUMNS
        },
        "max_abs_diff": {
            c: float(detail[f"{c}_abs_diff"].max()) for c in PREDICTION_COLUMNS
        },
    }


def refine_mismatches(output_dir):
    """Recompute every strict mismatch with batch size one and causal features."""
    torch.set_num_threads(4)
    output = Path(output_dir)
    summary = json.loads((output / "summary.json").read_text())
    results = []
    for record in summary["strategies"]:
        if not record["capture_vs_batch"]["mismatched_rows"]:
            continue
        identity = record["instance_id"]
        trace_path = output / "input_snapshot" / Path(record["trace"]).name
        if digest(trace_path) != record["trace_sha256"]:
            raise ValueError(f"Trace snapshot changed: {trace_path}")
        for relative, expected in record["model_sha256"].items():
            if digest(Path(record["model_path"]) / relative) != expected:
                raise ValueError(f"Model artifact changed: {relative}")
        trace = pd.read_csv(trace_path, float_precision="round_trip")
        detail = pd.read_csv(output / f"{identity}.csv", float_precision="round_trip")
        mismatches = detail.loc[~detail.matches]
        _, _, _, interval, model_hash = identity.split("|")
        spec = SimpleNamespace(
            model_path=record["model_path"], device="cpu", hash_id=model_hash)
        report_path = Path(record["model_path"]).parent.parent / "compare_reports.jsonl"
        load_params_from_report([spec], str(report_path))
        model = LiveRunner._load_model(spec)
        factory = LiveRunner._create_feature_generator(spec)
        full = factory.generate(_prepare_market_frame(trace[list(MARKET_COLUMNS)].copy(), spec.base_define))
        pipeline = SimpleNamespace(model=model, interval_ms=common.get_interval_ms(interval))
        rows = []
        for _, mismatch in mismatches.iterrows():
            position = int(np.flatnonzero(trace.close_time_ms_utc.eq(mismatch.close_time_ms_utc))[0])
            raw = trace.iloc[max(0, position - record["cache_bars"] + 1):position + 1]
            rolling = factory.generate(_prepare_market_frame(raw[list(MARKET_COLUMNS)].copy(), spec.base_define))
            one = LiveRunner._predict(None, pipeline, full.iloc[:position + 1]).iloc[-1]
            replay = LiveRunner._predict(None, pipeline, rolling).iloc[-1]
            window = full.iloc[position - model.seq_len + 1:position + 1].copy()
            dataset = TimeSeriesWindowDataset(
                df=window, kline_interval_ms=pipeline.interval_ms,
                feature_cols=model.feature_cols, label_col=model.label_col,
                seq_len=model.seq_len, is_live=False, show_feature_distribution=False)
            offline, _ = model.predict_with_ds(dataset, window, is_live=False,
                                               batch_size=1, diff_thresh=None)
            offline_one = offline.iloc[-1]
            row = {"instance_id": identity, "close_time_ms_utc": int(mismatch.close_time_ms_utc)}
            for c in PREDICTION_COLUMNS:
                row[f"{c}_capture"] = float(mismatch[f"{c}_live"])
                row[f"{c}_batch"] = float(mismatch[f"{c}_backtest"])
                row[f"{c}_single_full"] = float(one[c])
                row[f"{c}_rolling"] = float(replay[c])
                row[f"{c}_backtest_single"] = float(offline_one[c])
            rows.append(row)
        results.extend(rows)
        pd.DataFrame(results).to_csv(output / "mismatch_replay.csv", index=False)
        print(f"Refined {identity}: {len(rows)} rows", flush=True)
    frame = pd.DataFrame(results)
    comparisons = {}
    if not frame.empty:
        for left, right in [("capture", "backtest_single"), ("capture", "single_full"),
                            ("capture", "rolling"), ("single_full", "rolling"),
                            ("batch", "backtest_single")]:
            comparisons[f"{left}_vs_{right}"] = {}
            for column in PREDICTION_COLUMNS:
                a = frame[f"{column}_{left}"].to_numpy()
                b = frame[f"{column}_{right}"].to_numpy()
                comparisons[f"{left}_vs_{right}"][column] = {
                    "exact_mismatches": int((a != b).sum()),
                    "strict_mismatches": int((~np.isclose(a, b, rtol=1e-6, atol=1e-7)).sum()),
                    "max_abs_diff": float(np.abs(a - b).max()),
                }
    (output / "refinement_summary.json").write_text(json.dumps(
        {"rows": len(frame), "comparisons": comparisons}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="LiveTrading/live_config.json")
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--archive-root", default="LiveTrading/market")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replay-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.replay_samples < 1 or args.batch_size < 1:
        parser.error("Sample count and batch size must be positive")
    torch.set_num_threads(4)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    specs = load_live_strategy_specs(args.config)
    archive = Path(args.archive_root)
    summary = {
        "method": "All captured predictions versus full-history batch inference; sampled causal rolling-window LiveRunner._predict replay",
        "limitations": "Consolidated market data and currently preserved model artifacts; original startup states and historical artifact versions are not recoverable from these traces alone",
        "rtol": 1e-6, "atol": 1e-7,
        "config_sha256": digest(args.config),
        "strategies": [], "unresolved": [],
    }

    def save():
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    paths = sorted(Path(args.trace_dir).glob("*.csv"))
    if not paths:
        raise ValueError("No trace CSV files found")
    for path in paths:
        content = path.read_bytes()
        trace_sha256 = hashlib.sha256(content).hexdigest()
        snapshot = output / "input_snapshot"
        snapshot.mkdir(exist_ok=True)
        (snapshot / path.name).write_bytes(content)
        trace = pd.read_csv(io.BytesIO(content), float_precision="round_trip")
        warmup = trace.is_warmup.astype(str).str.lower()
        if not warmup.isin(["true", "false"]).all():
            raise ValueError(f"Invalid warmup flags: {path}")
        times = trace.close_time_ms_utc
        if not times.is_unique or not times.is_monotonic_increasing:
            raise ValueError(f"Duplicate or unordered candles: {path}")
        ids = [c.removesuffix("__pred") for c in trace if c.endswith("__pred")]
        resolved = {}
        for identity in ids:
            _, _, symbol, interval, model_hash = identity.split("|")
            matches = [s for s in specs if s.hash_id == model_hash
                       and s.base_define.symbol == symbol
                       and s.base_define.interval == interval]
            if matches:
                resolved[identity] = replace(matches[0], instance_id=identity, device="cpu")
                continue
            candidates = list(archive.glob(f"**/models/{model_hash}"))
            templates = [s for s in specs if s.base_define.symbol == symbol
                         and s.base_define.interval == interval]
            if len(candidates) == 1 and templates:
                artifact = candidates[0]
                spec = replace(templates[0], instance_id=identity, hash_id=model_hash,
                               model_path=str(artifact.resolve()), device="cpu")
                load_params_from_report([spec], str(artifact.parent.parent / "compare_reports.jsonl"))
                resolved[identity] = spec
            else:
                summary["unresolved"].append({
                    "instance_id": identity,
                    "recorded_rows": int(trace[f"{identity}__pred"].notna().sum()),
                    "reason": "No uniquely matching preserved model artifact",
                })
                save()
        # The feed cache is shared across all strategies for the same market.
        models = {}
        factories = {}
        for spec in resolved.values():
            if spec.hash_id not in models:
                models[spec.hash_id] = LiveRunner._load_model(spec)
                factories[spec.hash_id] = LiveRunner._create_feature_generator(spec)
        if not models:
            continue
        cache_bars = max(factories[h].get_global_min_history() + 2 * m.seq_len
                         for h, m in models.items()) + 500
        batch_cache = {}
        replay_cache = {}
        for identity, spec in resolved.items():
            print(f"Verifying {identity}", flush=True)
            model = models[spec.hash_id]
            factory = factories[spec.hash_id]
            interval_ms = common.get_interval_ms(spec.base_define.interval)
            if not trace.open_time_ms_utc.diff().dropna().eq(interval_ms).all():
                raise ValueError(f"Discontinuous market data: {path}")
            columns = [f"{identity}__{c}" for c in PREDICTION_COLUMNS]
            present = trace[columns].notna().any(axis=1)
            mask = warmup.eq("false") & present
            if (warmup.eq("true") & present).any() or not mask.any():
                raise ValueError(f"Invalid captured prediction coverage: {identity}")
            live = trace.loc[mask, ["close_time_ms_utc", *columns]].rename(
                columns=dict(zip(columns, PREDICTION_COLUMNS)))
            if spec.hash_id not in batch_cache:
                features = factory.generate(_prepare_market_frame(
                    trace[list(MARKET_COLUMNS)].copy(), spec.base_define))
                first_live = int(warmup.eq("false").idxmax())
                features = features.iloc[max(0, first_live - model.seq_len + 1):].copy()
                dataset = TimeSeriesWindowDataset(
                    df=features, kline_interval_ms=interval_ms,
                    feature_cols=model.feature_cols, label_col=model.label_col,
                    seq_len=model.seq_len, is_live=False,
                    show_feature_distribution=False)
                offline, _ = model.predict_with_ds(dataset, features, is_live=False,
                                                   batch_size=args.batch_size, diff_thresh=None)
                batch_cache[spec.hash_id] = offline
            offline = batch_cache[spec.hash_id]
            detail = compare_rows(live, offline, rtol=1e-6, atol=1e-7)
            detail.to_csv(output / f"{identity}.csv", index=False)
            positions = np.flatnonzero(mask)
            selected = positions[np.unique(np.linspace(
                0, len(positions) - 1, min(args.replay_samples, len(positions)), dtype=int))]
            replay_rows = []
            pipeline = SimpleNamespace(model=model, interval_ms=interval_ms)
            for position in selected:
                key = (spec.hash_id, int(position))
                if key not in replay_cache:
                    raw = trace.iloc[max(0, position - cache_bars + 1):position + 1]
                    features = factory.generate(_prepare_market_frame(
                        raw[list(MARKET_COLUMNS)].copy(), spec.base_define))
                    predicted = LiveRunner._predict(None, pipeline, features)
                    replay_cache[key] = predicted.iloc[-1][
                        ["close_time_ms_utc", *PREDICTION_COLUMNS]].to_dict()
                replay_rows.append(replay_cache[key])
            replay = pd.DataFrame(replay_rows)
            replay.to_csv(output / f"{identity}.replay.csv", index=False)
            replay_batch = compare_rows(replay, offline, rtol=1e-6, atol=1e-7)
            replay_capture = compare_rows(replay, live, rtol=1e-6, atol=1e-7)
            replay_batch.to_csv(output / f"{identity}.replay_vs_batch.csv", index=False)
            replay_capture.to_csv(output / f"{identity}.replay_vs_capture.csv", index=False)
            record = {
                "instance_id": identity, "symbol": spec.base_define.symbol,
                "hash_id": spec.hash_id, "model_path": spec.model_path,
                "model_sha256": {str(p.relative_to(spec.model_path)): digest(p)
                                 for p in sorted(Path(spec.model_path).rglob("*"))
                                 if p.is_file() and p.suffix in {".pt", ".json", ".pkl"}},
                "trace": str(path.resolve()), "trace_sha256": trace_sha256,
                "start_utc": pd.to_datetime(live.close_time_ms_utc.iloc[0], unit="ms", utc=True).isoformat(),
                "end_utc": pd.to_datetime(live.close_time_ms_utc.iloc[-1], unit="ms", utc=True).isoformat(),
                "unrecorded_nonwarmup_rows": int((warmup.eq("false") & ~present).sum()),
                "partial_recorded_rows": int(trace.loc[mask, columns].isna().any(axis=1).sum()),
                "cache_bars": cache_bars, "seq_len": model.seq_len,
                "capture_vs_batch": metrics(detail),
                "sampled_replay_vs_batch": metrics(replay_batch),
                "sampled_replay_vs_capture": metrics(replay_capture),
            }
            summary["strategies"].append(record)
            save()
            print(json.dumps({k: record[k] for k in ["instance_id", "capture_vs_batch",
                                                   "sampled_replay_vs_batch"]}), flush=True)
    refine_mismatches(output)
    failed = any(s[k]["mismatched_rows"] for s in summary["strategies"]
                 for k in ["capture_vs_batch", "sampled_replay_vs_batch", "sampled_replay_vs_capture"])
    raise SystemExit(1 if failed else 2 if summary["unresolved"] else 0)


if __name__ == "__main__":
    main()
