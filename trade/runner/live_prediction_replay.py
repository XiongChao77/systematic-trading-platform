"""Replay historical candles through the live inference path and compare output."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from data_process import common
from trade.feed.bt_feed_mock import BtDataFeedMock
from trade.runner.live_runner import (
    LiveRunner,
    LiveStrategySpec,
    StrategyPipeline,
    load_live_strategy_specs,
)
from trade.venue.mock_venue import MockVenue


PREDICTION_COLUMNS = ("pred", "pred_prob", "net_score")


@dataclass(frozen=True)
class PredictionComparison:
    compared_rows: int
    missing_backtest_rows: int
    mismatched_rows: int

    @property
    def matches(self) -> bool:
        return self.missing_backtest_rows == 0 and self.mismatched_rows == 0


class PredictionRecorder:
    """Collect the latest live-mode prediction at every replayed candle."""

    def __init__(self):
        self._records: list[dict] = []

    def __call__(
        self,
        pipeline: StrategyPipeline,
        candle_open_time_ms: int,
        row: pd.Series,
    ) -> None:
        row_open_time_ms = int(row["open_time_ms_utc"])
        if row_open_time_ms != int(candle_open_time_ms):
            raise ValueError(
                "Prediction candle does not match the callback open time: "
                f"callback={candle_open_time_ms}, prediction={row_open_time_ms}"
            )
        candle_close_time_ms = int(row["close_time_ms_utc"])
        record = {
            "instance_id": pipeline.spec.instance_id,
            "hash_id": pipeline.spec.hash_id,
            "close_time_ms_utc": int(candle_close_time_ms),
            "close_time_date_utc": pd.Timestamp(
                candle_close_time_ms,
                unit="ms",
                tz="UTC",
            ).isoformat(),
        }
        for column in PREDICTION_COLUMNS:
            value = row.get(column)
            record[column] = None if pd.isna(value) else float(value)
        self._records.append(record)

    def to_frame(self) -> pd.DataFrame:
        columns = [
            "instance_id",
            "hash_id",
            "close_time_ms_utc",
            "close_time_date_utc",
            *PREDICTION_COLUMNS,
        ]
        return pd.DataFrame.from_records(self._records, columns=columns)


def write_prediction_frame(frame: pd.DataFrame, path: str) -> None:
    """Persist a timestamped prediction frame in a supported table format."""

    output_path = os.path.abspath(path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    extension = os.path.splitext(output_path)[1].casefold()
    if extension == ".csv":
        frame.to_csv(output_path, index=False)
    elif extension in {".feather", ".ft"}:
        frame.to_feather(output_path)
    elif extension in {".parquet", ".pq"}:
        frame.to_parquet(output_path, index=False)
    elif extension in {".pkl", ".pickle"}:
        frame.to_pickle(output_path)
    else:
        raise ValueError(
            "Prediction output must be CSV, Feather, Parquet, or Pickle"
        )


def read_prediction_frame(path: str) -> pd.DataFrame:
    """Load a prediction frame written as a supported table format."""

    extension = os.path.splitext(path)[1].casefold()
    if extension == ".csv":
        return pd.read_csv(path)
    if extension in {".feather", ".ft"}:
        return pd.read_feather(path)
    if extension in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if extension in {".pkl", ".pickle"}:
        value = pd.read_pickle(path)
        if isinstance(value, pd.DataFrame):
            return value
        if isinstance(value, dict) and isinstance(value.get("predictions"), pd.DataFrame):
            return value["predictions"]
        raise TypeError(f"Prediction pickle does not contain a DataFrame: {path}")
    raise ValueError(f"Unsupported prediction file extension: {extension!r}")


def _with_candle_close_times(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    prepared = frame.copy()
    if prepared.empty:
        raise ValueError(f"{name} predictions are empty")
    if "close_time_ms_utc" in prepared.columns:
        prepared["close_time_ms_utc"] = pd.to_numeric(
            prepared["close_time_ms_utc"],
            errors="raise",
        ).astype("int64")
    elif "close_time_date_utc" in prepared.columns:
        timestamps = pd.to_datetime(
            prepared["close_time_date_utc"],
            utc=True,
            errors="raise",
        )
        prepared["close_time_ms_utc"] = timestamps.astype("int64") // 1_000_000
    else:
        raise ValueError(
            f"{name} predictions require close_time_ms_utc or close_time_date_utc"
        )
    if prepared["close_time_ms_utc"].duplicated().any():
        raise ValueError(f"{name} predictions contain duplicate candle times")
    return prepared


def prediction_frame_for_strategy(
    frame: pd.DataFrame,
    *,
    instance_id: Optional[str],
    name: str,
) -> pd.DataFrame:
    """Select standard prediction columns from long or captured wide data."""

    prepared = frame.copy()
    wide_columns = (
        {
            column: f"{instance_id}__{column}"
            for column in PREDICTION_COLUMNS
        }
        if instance_id is not None
        else {}
    )
    if wide_columns and all(
        source in prepared.columns for source in wide_columns.values()
    ):
        if "is_warmup" not in prepared.columns:
            raise ValueError(f"{name} captured predictions require is_warmup")
        warmup = prepared["is_warmup"].astype(str).str.strip().str.lower()
        prepared = prepared.loc[~warmup.isin({"true", "1"})].copy()
        prepared.rename(
            columns={source: target for target, source in wide_columns.items()},
            inplace=True,
        )
        prepared = prepared.loc[prepared["pred"].notna()].copy()
        prepared["instance_id"] = str(instance_id)
        return prepared

    if "instance_id" in prepared.columns:
        instance_ids = prepared["instance_id"].dropna().astype(str).unique()
        selected_instance_id = instance_id
        if selected_instance_id is None:
            if len(instance_ids) != 1:
                raise ValueError(
                    f"instance_id is required when {name} contains multiple strategies"
                )
            selected_instance_id = instance_ids[0]
        prepared = prepared.loc[
            prepared["instance_id"].astype(str) == str(selected_instance_id)
        ].copy()
    return prepared


def compare_prediction_frames(
    replay: pd.DataFrame,
    backtest: pd.DataFrame,
    *,
    instance_id: Optional[str] = None,
    rtol: float = 1e-6,
    atol: float = 1e-7,
) -> PredictionComparison:
    """Compare every replay prediction with the same backtest candle."""

    replay_frame = prediction_frame_for_strategy(
        replay,
        instance_id=instance_id,
        name="Replay",
    )
    backtest_frame = prediction_frame_for_strategy(
        backtest,
        instance_id=instance_id,
        name="Backtest",
    )
    replay_frame = _with_candle_close_times(replay_frame, "Replay")
    backtest_frame = _with_candle_close_times(backtest_frame, "Backtest")
    missing_columns = [
        column
        for column in PREDICTION_COLUMNS
        if column not in replay_frame.columns or column not in backtest_frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Prediction comparison is missing columns: {sorted(set(missing_columns))}"
        )

    merged = replay_frame[
        ["close_time_ms_utc", *PREDICTION_COLUMNS]
    ].merge(
        backtest_frame[["close_time_ms_utc", *PREDICTION_COLUMNS]],
        on="close_time_ms_utc",
        how="left",
        suffixes=("_replay", "_backtest"),
        indicator=True,
        validate="one_to_one",
    )
    missing_mask = merged["_merge"] != "both"
    mismatch_mask = pd.Series(False, index=merged.index)
    both = ~missing_mask
    if both.any():
        replay_pred = pd.to_numeric(merged.loc[both, "pred_replay"], errors="coerce")
        backtest_pred = pd.to_numeric(
            merged.loc[both, "pred_backtest"],
            errors="coerce",
        )
        mismatch_mask.loc[both] |= replay_pred.ne(backtest_pred)
        for column in ("pred_prob", "net_score"):
            replay_values = pd.to_numeric(
                merged.loc[both, f"{column}_replay"],
                errors="coerce",
            ).to_numpy(float)
            backtest_values = pd.to_numeric(
                merged.loc[both, f"{column}_backtest"],
                errors="coerce",
            ).to_numpy(float)
            mismatch_mask.loc[both] |= ~np.isclose(
                replay_values,
                backtest_values,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )

    return PredictionComparison(
        compared_rows=int(both.sum()),
        missing_backtest_rows=int(missing_mask.sum()),
        mismatched_rows=int(mismatch_mask.sum()),
    )


def compare_prediction_files(
    replay_path: str,
    backtest_path: str,
    *,
    instance_id: Optional[str] = None,
    rtol: float = 1e-6,
    atol: float = 1e-7,
) -> PredictionComparison:
    """Load and compare replay and backtest prediction files."""

    return compare_prediction_frames(
        read_prediction_frame(replay_path),
        read_prediction_frame(backtest_path),
        instance_id=instance_id,
        rtol=rtol,
        atol=atol,
    )


def run_prediction_replay(
    specs: list[LiveStrategySpec],
    output_path: str,
    *,
    data_path_resolver: Optional[
        Callable[[common.MarketDataSourceConfig], str]
    ] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Replay all configured feeds chronologically and persist live predictions."""

    resolve_path = data_path_resolver or common.market_data_path
    recorder = PredictionRecorder()

    def feed_factory(
        market_config: common.MarketDataSourceConfig,
        max_len: int,
    ) -> BtDataFeedMock:
        return BtDataFeedMock(resolve_path(market_config), max_len=max_len)

    def venue_factory(spec: LiveStrategySpec, _logger: logging.Logger) -> MockVenue:
        return MockVenue(initial_equity=spec.broker_config.initial_equity)

    runner = LiveRunner(
        specs,
        logger=logger,
        feed_factory=feed_factory,
        venue_factory=venue_factory,
        prediction_callback=recorder,
    )
    expected_records = 0
    try:
        runner.initialize()
        while True:
            candidates = [
                (group.feed.peek_next_close_time_ms(), group)
                for group in runner.feed_groups
            ]
            candidates = [
                (close_time, group)
                for close_time, group in candidates
                if close_time is not None
            ]
            if not candidates:
                break
            next_close_time = min(close_time for close_time, _ in candidates)
            for close_time, group in candidates:
                if close_time != next_close_time:
                    continue
                candle_close_time_ms = group.feed.advance()
                if candle_close_time_ms is None:
                    raise RuntimeError("Replay feed ended before its advertised candle")
                expected_records += len(group.pipelines)
            runner.process_pending_events()

        predictions = recorder.to_frame()
        write_prediction_frame(predictions, output_path)
        if len(predictions) != expected_records:
            raise RuntimeError(
                "Live prediction replay did not produce one prediction per enabled "
                f"pipeline candle: expected={expected_records}, actual={len(predictions)}"
            )
        return predictions
    finally:
        runner.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay raw candles through LiveRunner and compare predictions",
    )
    parser.add_argument("--config", required=True, help="Live runner JSON config")
    parser.add_argument("--output", required=True, help="Replay prediction output file")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optional root containing canonical raw market-data paths",
    )
    parser.add_argument(
        "--market-trace",
        default=None,
        help="Prediction trace CSV used as the market source for one feed group",
    )
    parser.add_argument(
        "--backtest-predictions",
        default=None,
        help="Optional backtest prediction file to compare",
    )
    parser.add_argument("--instance-id", default=None)
    args = parser.parse_args()

    logger = logging.getLogger("trade.live_replay")
    specs = load_live_strategy_specs(args.config)
    if args.backtest_predictions:
        from trade.runner.identity import spec_for_trace

        reference = read_prediction_frame(args.backtest_predictions)
        trace_ids = {column.removesuffix("__pred") for column in reference.columns if column.endswith("__pred")}
        if "instance_id" in reference:
            trace_ids.update(reference["instance_id"].dropna().astype(str))
        if args.instance_id:
            trace_ids = {args.instance_id}
        for trace_id in sorted(trace_ids):
            if not trace_id.startswith("ctrader|"):
                continue
            matched = spec_for_trace(specs, trace_id)
            specs = [matched if (spec.venue_config.venue == "ctrader"
                                 and spec.venue_config.trader_login == matched.venue_config.trader_login
                                 and spec.hash_id == matched.hash_id) else spec for spec in specs]
    resolver = None
    if args.market_trace:
        market_keys = {
            (
                spec.base_define.data_source,
                spec.base_define.trading_type,
                spec.base_define.symbol,
                spec.base_define.interval,
            )
            for spec in specs
        }
        if len(market_keys) != 1:
            raise ValueError(
                "--market-trace requires every enabled strategy to share one feed group"
            )
        trace_path = os.path.abspath(args.market_trace)
        resolver = lambda _market: trace_path  # noqa: E731
    elif args.data_root:
        resolver = lambda market: common.market_data_path(  # noqa: E731
            market,
            root_dir=args.data_root,
        )
    replay = run_prediction_replay(
        specs,
        args.output,
        data_path_resolver=resolver,
        logger=logger,
    )
    if args.backtest_predictions:
        comparison = compare_prediction_frames(
            replay,
            read_prediction_frame(args.backtest_predictions),
            instance_id=args.instance_id,
        )
        print(json.dumps(asdict(comparison), separators=(",", ":")))
        if not comparison.matches:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
