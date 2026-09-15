"""Compare captured live predictions with offline batch inference on captured bars.

No runner, venue, order submission, or network feed is initialized.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_process import common
from model.data_loader import TimeSeriesWindowDataset
from trade.recording.prediction_trace import MARKET_COLUMNS, PREDICTION_COLUMNS
from trade.runner.live_runner import (
    LiveRunner,
    _prepare_market_frame,
    load_live_strategy_specs,
)


def compare_rows(live, offline, *, rtol, atol):
    """Retain missing values and require finite predictions on both sides."""
    columns = ["close_time_ms_utc", *PREDICTION_COLUMNS]
    result = live[columns].merge(
        offline[columns],
        on="close_time_ms_utc",
        how="left",
        suffixes=("_live", "_backtest"),
        validate="one_to_one",
    )
    result["matches"] = True
    for column in PREDICTION_COLUMNS:
        left = result[f"{column}_live"].to_numpy(float)
        right = result[f"{column}_backtest"].to_numpy(float)
        finite = np.isfinite(left) & np.isfinite(right)
        match = (
            left == right
            if column == "pred"
            else np.isclose(left, right, rtol=rtol, atol=atol)
        )
        difference = np.full(len(left), np.nan)
        difference[finite] = np.abs(left[finite] - right[finite])
        result[f"{column}_abs_diff"] = difference
        result[f"{column}_matches"] = finite & match
        result["matches"] &= finite & match
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-7)
    args = parser.parse_args()
    if args.rtol < 0 or args.atol < 0 or args.batch_size < 1 or args.threads < 1:
        parser.error(
            "Tolerances must be nonnegative; batch size and threads must be positive"
        )
    torch.set_num_threads(args.threads)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    specs = {s.strategy_id: s for s in load_live_strategy_specs(args.config)}
    summary = {
        "method": "Captured market bars -> full-history features -> TimeSeriesWindowDataset -> predict_with_ds(is_live=False)",
        "scope": "Model output parity only; no labels, historical report replay, or execution simulation",
        "rtol": args.rtol,
        "atol": args.atol,
        "batch_size": args.batch_size,
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "strategies": [],
    }
    paths = sorted(Path(args.trace_dir).glob("*.csv"))
    if not paths:
        raise ValueError("No trace CSV files found")
    for path in paths:
        content = path.read_bytes()
        trace = pd.read_csv(io.BytesIO(content), float_precision="round_trip")
        warmup = trace["is_warmup"].astype(str).str.lower()
        if not warmup.isin(["true", "false"]).all():
            raise ValueError(f"Invalid warmup flags: {path}")
        live_mask = warmup.eq("false")
        if (
            not live_mask.any()
            or warmup.iloc[: int(live_mask.idxmax())].ne("true").any()
            or warmup[live_mask.idxmax() :].ne("false").any()
        ):
            raise ValueError(f"Expected warmup prefix followed by live rows: {path}")
        times = trace.close_time_ms_utc
        if not times.is_unique or not times.is_monotonic_increasing:
            raise ValueError(f"Duplicate or unordered candles: {path}")
        ids = [c.removesuffix("__pred") for c in trace.columns if c.endswith("__pred")]
        if not ids:
            raise ValueError(f"No strategy predictions found: {path}")
        for strategy_id in ids:
            spec = specs[strategy_id]
            print(f"Verifying {strategy_id}", flush=True)
            model = LiveRunner._load_model(spec)
            factory = LiveRunner._create_feature_generator(spec)
            features = factory.generate(
                _prepare_market_frame(trace[list(MARKET_COLUMNS)], spec.base_define)
            )
            first_live = int(live_mask.idxmax())
            features = features.iloc[max(0, first_live - model.seq_len + 1) :].copy()
            interval_ms = common.get_interval_ms(spec.base_define.interval)
            dataset = TimeSeriesWindowDataset(
                df=features,
                kline_interval_ms=interval_ms,
                feature_cols=model.feature_cols,
                label_col=model.label_col,
                seq_len=model.seq_len,
                is_live=False,
            )
            offline, _ = model.predict_with_ds(
                dataset,
                features,
                is_live=False,
                batch_size=args.batch_size,
                diff_thresh=None,
            )
            live = trace.loc[
                live_mask,
                [
                    "close_time_ms_utc",
                    *[f"{strategy_id}__{c}" for c in PREDICTION_COLUMNS],
                ],
            ].rename(columns={f"{strategy_id}__{c}": c for c in PREDICTION_COLUMNS})
            detail = compare_rows(live, offline, rtol=args.rtol, atol=args.atol)
            detail.to_csv(output / f"{strategy_id}.csv", index=False)
            artifacts = {}
            for artifact in sorted(Path(spec.model_path).rglob("*")):
                if artifact.is_file() and artifact.suffix in {
                    ".pt",
                    ".json",
                    ".txt",
                    ".pkl",
                }:
                    artifacts[str(artifact.relative_to(spec.model_path))] = (
                        hashlib.sha256(artifact.read_bytes()).hexdigest()
                    )
            record = {
                "strategy_id": strategy_id,
                "hash_id": spec.hash_id,
                "model_path": spec.model_path,
                "model_sha256": artifacts,
                "device": str(model.device),
                "seq_len": model.seq_len,
                "trace": str(path.resolve()),
                "trace_sha256": hashlib.sha256(content).hexdigest(),
                "warmup_rows": first_live,
                "live_rows": len(detail),
                "start_utc": pd.to_datetime(
                    times[live_mask].iloc[0], unit="ms", utc=True
                ).isoformat(),
                "end_utc": pd.to_datetime(
                    times.iloc[-1], unit="ms", utc=True
                ).isoformat(),
                "gaps": int(
                    (trace.open_time_ms_utc.diff().dropna() != interval_ms).sum()
                ),
                "mismatched_rows": int((~detail.matches).sum()),
                "missing_live_rows": int(
                    live[list(PREDICTION_COLUMNS)].isna().any(axis=1).sum()
                ),
                "missing_backtest_rows": int(
                    detail[[f"{c}_backtest" for c in PREDICTION_COLUMNS]]
                    .isna()
                    .any(axis=1)
                    .sum()
                ),
                "max_abs_diff": {
                    c: float(detail[f"{c}_abs_diff"].max()) for c in PREDICTION_COLUMNS
                },
                "mismatch_count": {
                    c: int((~detail[f"{c}_matches"]).sum()) for c in PREDICTION_COLUMNS
                },
            }
            summary["strategies"].append(record)
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        k: record[k]
                        for k in [
                            "strategy_id",
                            "live_rows",
                            "mismatched_rows",
                            "max_abs_diff",
                        ]
                    }
                ),
                flush=True,
            )
    raise SystemExit(int(any(s["mismatched_rows"] for s in summary["strategies"])))


if __name__ == "__main__":
    main()
