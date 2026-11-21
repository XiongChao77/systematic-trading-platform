#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download Binance Vision kline or aggTrades ZIP files by date range.

Data source:
    https://data.binance.vision/

Main behavior:
    - Prefer existing ZIP files. For uncached ranges, prefer monthly ZIP files
      for complete months and daily ZIP files for partial months.
    - If a daily ZIP is absent, use its monthly archive as a fallback.
    - If --start is not specified, reuse the cached earliest data date for the
      market/symbol, or query the Binance Vision S3 XML API when no date is cached.
      Explicit date ranges use local archives first without listing remote files.
    - If --end is not specified, default to the latest completed UTC day.
    - Treat local ZIP files and confirmed HTTP 404 statuses as the download cache.
      download_info.json is an audit report and is never used as downloader input.
    - Resume automatically from valid cached archives, downloading only missing
      sources. Invalid cached archives are downloaded again.
    - Skip a series when its final CSV already exists, unless --force is used.
    - Infer availability boundaries from paginated S3 XML listings; gaps remain
      eligible for monthly recovery.
    - Skip dates whose daily archives return HTTP 404 and have no monthly data.
      Keep later available data and report gaps; download failures remain fatal.
    - Log download and conversion progress per archive, with per-month payload
      progress and ETA, so long multi-gigabyte conversions stay observable.
    - Validate candle structure and OHLC values in the merged K-line sequence.

Examples:
    # Download multiple symbols and intervals (Cartesian product)
    python download_from_zip.py --symbols DOGEUSDT BTCUSDT --intervals 15m 30m

    # Spot DOGEUSDT 30m, explicit range
    python download_from_zip.py --symbols DOGEUSDT --intervals 30m --start 2021-01-01 --end 2021-12-31

    # Binance USDⓈ-M futures / UM futures
    python download_from_zip.py --market um --symbols DOGEUSDT --intervals 30m

    # Download aggregate trades instead of klines
    python download_from_zip.py --market um --symbols BTCUSDT --data-type aggTrades --start 2024-01-01 --end 2024-01-31

Output:
    <out_dir>/<market>/<symbol>/<interval>/
        download_info.json
        zips/<symbol>-<interval>-YYYY-MM-DD.zip
        monthly_zips/<symbol>-<interval>-YYYY-MM.zip
    <out_dir>/earliest_dates.json stores earliest dates for all market/symbols.

    K-line CSV files are moved to common.market_data_path(para). aggTrades are
    written as one Parquet file per calendar year in the canonical data
    directory.

    For um/cm markets, monthly fundingRate ZIPs are also downloaded (once per
    symbol, cached under <out_dir>/<market>/<symbol>/fundingRate/) and merged to
    <market_data dir>/<symbol>_fundingRate.csv for kline downloads. Use
    --no-funding-rate to skip.
"""

import argparse
import calendar
import csv
import io
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import requests
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as parquet
from requests.adapters import HTTPAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_process import common


LOGGER = logging.getLogger(__name__)


BASE_URL = "https://data.binance.vision"
S3_LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
HTTP_TIMEOUT_SECONDS = 30
HTTP_RETRIES = 3
RETRY_DELAY_SECONDS = 1
DOWNLOAD_DELAY_SECONDS = 0.3
FUNDING_API_MIN_INTERVAL_SECONDS = 1.0
FUNDING_API_RATE_LIMIT_FALLBACK_SECONDS = 300.0
INFO_FILENAME = "download_info.json"
MISSING_SOURCES_FILENAME = "missing_sources.json"
EARLIEST_DATES_FILENAME = "earliest_dates.json"
ZIP_DIRNAME = "zips"
MONTHLY_ZIP_DIRNAME = "monthly_zips"
PROGRESS_LOG_INTERVAL_SECONDS = 20.0
DOWNLOAD_CHUNK_BYTES = 4 * 1024 * 1024
DOWNLOAD_PROGRESS_LOG_INTERVAL_SECONDS = 10.0


class NoArchiveError(RuntimeError):
    """Raised when Binance Vision has no daily archive for a market symbol."""

VALID_MARKETS = {
    "spot": "data/spot/daily/klines",
    "um": "data/futures/um/daily/klines",   # USDⓈ-M futures
    "cm": "data/futures/cm/daily/klines",   # COIN-M futures
}

MARKET_ROOTS = {
    "spot": "data/spot",
    "um": "data/futures/um",
    "cm": "data/futures/cm",
}

SYMBOL_ALIASES = {
    "um": {
        "SHIBUSDT": "1000SHIBUSDT",
    },
}

# Fallback only when the S3 listing cannot provide an earliest date.
FALLBACK_START = {
    "spot": date(2017, 8, 17),
    "um": date(2020, 1, 1),
    "cm": date(2020, 8, 1),
}

OUTPUT_COLUMNS = [
    "open_time_ms_utc",
    "open_time_date_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_ms_utc",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]
AGG_TRADES_SOURCE_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
    "is_best_match",
]
AGG_TRADES_SCHEMA = pa.schema([
    pa.field("agg_trade_id", pa.int64()),
    pa.field("price", pa.string()),
    pa.field("quantity", pa.string()),
    pa.field("first_trade_id", pa.int64()),
    pa.field("last_trade_id", pa.int64()),
    pa.field("transact_time_ms_utc", pa.int64()),
    pa.field("transact_time_date_utc", pa.string()),
    pa.field("is_buyer_maker", pa.string()),
    pa.field("is_best_match", pa.string()),
])
AGG_TRADES_READ_BLOCK_BYTES = 32 * 1024 * 1024
AGG_TRADES_INTEGRITY_VERSION = 1


def load_active_kline_minutes(directory: Path, symbol: str, start: date, end: date) -> set[int]:
    """Return minutes with trades in a local 1m kline file, if available."""
    path = directory / f"{symbol}_1m.csv"
    if not path.is_file():
        return set()
    start_ms, end_ms = day_window_ms(start, end)
    result: set[int] = set()
    for chunk in pd.read_csv(
        path, usecols=["open_time_ms_utc", "number_of_trades"], chunksize=250_000,
    ):
        timestamps = chunk["open_time_ms_utc"].to_numpy()
        mask = (
            (timestamps >= start_ms) & (timestamps < end_ms)
            & (chunk["number_of_trades"].to_numpy() > 0)
        )
        result.update(int(value) for value in timestamps[mask] // 60_000)
    return result


class AggTradeYearBuilder:
    """Stage daily shards, then atomically publish a sorted, unique yearly file."""

    def __init__(self, directory: Path, symbol: str, year: int):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{symbol}_aggTrades_{year}.parquet"
        self.temporary = tempfile.TemporaryDirectory(
            prefix=f".{symbol}_aggTrades_{year}-", dir=directory,
        )
        self.shards: dict[date, list[Path]] = {}
        self.writers: dict[date, parquet.ParquetWriter] = {}

    def append(self, table: pa.Table) -> None:
        timestamps = table["transact_time_ms_utc"].to_numpy()
        days = timestamps // 86_400_000
        for value in np.unique(days):
            day = date(1970, 1, 1) + timedelta(days=int(value))
            if day not in self.writers:
                paths = self.shards.setdefault(day, [])
                path = Path(self.temporary.name) / f"{day}-{len(paths)}.parquet"
                paths.append(path)
                self.writers[day] = parquet.ParquetWriter(
                    path, AGG_TRADES_SCHEMA, compression="zstd", use_dictionary=False,
                )
            self.writers[day].write_table(table.filter(pa.array(days == value)))

    def close_shards(self) -> None:
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()

    def read_day(self, day: date, columns=None) -> pa.Table:
        return pa.concat_tables([
            parquet.ParquetFile(path).read(columns=columns) for path in self.shards[day]
        ])

    def inspect(self, start: date, end: date, active_minutes: set[int], previous=None) -> dict:
        """Find ID gaps, empty dates and disagreements with local 1m klines."""
        self.close_shards()
        gaps = []
        present_minutes: set[int] = set()
        last = previous
        for day in sorted(self.shards):
            table = self.read_day(day, ["agg_trade_id", "transact_time_ms_utc"])
            ids = table["agg_trade_id"].to_numpy()
            timestamps = table["transact_time_ms_utc"].to_numpy()
            if any(column.null_count for column in table.columns) or np.any(ids < 1):
                raise ValueError(f"Invalid aggregate trade IDs or timestamps on {day}")
            present_minutes.update(int(value) for value in np.unique(timestamps // 60_000))
            ids, indices = np.unique(ids, return_index=True)
            timestamps = timestamps[indices]
            if last is not None:
                if int(ids[0]) <= last[0]:
                    raise ValueError(
                        f"Aggregate trade IDs overlap across UTC dates at {day}: "
                        f"previous={last[0]}, next={int(ids[0])}"
                    )
                ids = np.concatenate(([last[0]], ids))
                timestamps = np.concatenate(([last[1]], timestamps))
            for index in np.flatnonzero(np.diff(ids) > 1):
                gaps.append({
                    "before_id": int(ids[index]), "after_id": int(ids[index + 1]),
                    "missing_ids": int(ids[index + 1] - ids[index] - 1),
                    "before_ms": int(timestamps[index]), "after_ms": int(timestamps[index + 1]),
                })
            last = (int(ids[-1]), int(timestamps[-1]))
        return {
            "id_gaps": gaps,
            "missing_dates": [day.isoformat() for day in date_range(start, end) if day not in self.shards],
            "missing_kline_minutes": sorted(active_minutes - present_minutes),
            "last_trade": last,
        }

    @staticmethod
    def normalise(table: pa.Table) -> tuple[pa.Table, int]:
        """Reject conflicting IDs, remove identical duplicates and sort the result."""
        table = table.take(pc.sort_indices(table, sort_keys=[("agg_trade_id", "ascending")]))
        ids = table["agg_trade_id"].to_numpy()
        duplicate_indices = np.flatnonzero(ids[1:] == ids[:-1]) + 1
        if len(duplicate_indices):
            current = table.take(pa.array(duplicate_indices))
            previous = table.take(pa.array(duplicate_indices - 1))
            for name in table.column_names:
                equal = pc.or_(
                    pc.fill_null(pc.equal(current[name], previous[name]), False),
                    pc.and_(pc.is_null(current[name]), pc.is_null(previous[name])),
                ).to_numpy()
                if not np.all(equal):
                    trade_id = int(ids[duplicate_indices[np.flatnonzero(~equal)[0]]])
                    raise ValueError(f"Conflicting agg_trade_id={trade_id}: field {name} differs")
            keep = np.concatenate(([True], ids[1:] != ids[:-1]))
            table = table.filter(pa.array(keep))
        table = table.take(pc.sort_indices(table, sort_keys=[
            ("transact_time_ms_utc", "ascending"), ("agg_trade_id", "ascending"),
        ]))
        return table, len(duplicate_indices)

    def publish(self) -> dict:
        self.close_shards()
        output = Path(self.temporary.name) / "final.parquet"
        rows = duplicates = 0
        with parquet.ParquetWriter(
            output, AGG_TRADES_SCHEMA, compression="zstd", use_dictionary=False,
        ) as writer:
            for day in sorted(self.shards):
                table, removed = self.normalise(self.read_day(day))
                writer.write_table(table, row_group_size=500_000)
                rows += table.num_rows
                duplicates += removed
        os.replace(output, self.path)
        return {"rows": rows, "duplicates_removed": duplicates}

    def cleanup(self) -> None:
        self.close_shards()
        self.temporary.cleanup()


def parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()


def ms_to_dt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def interval_to_ms(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])

    scales = {
        "s": 1000,
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
    }

    if unit == "M":
        raise ValueError("Monthly interval '1M' is not supported for fixed gap checking.")

    if unit not in scales:
        raise ValueError(f"Unsupported interval: {interval}")

    return value * scales[unit]


def normalize_ts_to_ms(value: str) -> int:
    """
    Normalize Binance Vision timestamps to milliseconds.

    Typical old files use milliseconds: 13 digits.
    Some newer spot files may use microseconds: 16 digits.
    """
    ts = int(float(value))
    digits = len(str(abs(ts)))

    if digits >= 16:       # microseconds
        return ts // 1000
    if digits <= 10:       # seconds, just in case
        return ts * 1000
    return ts              # milliseconds


def date_range(start: date, end: date) -> Iterable[date]:
    curr = start
    while curr <= end:
        yield curr
        curr += timedelta(days=1)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def format_byte_size(num_bytes: float) -> str:
    """Render a byte count with a compact unit suffix."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{int(value)} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} PB"


def format_count(value: int) -> str:
    """Render a large integer with a compact magnitude suffix."""
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def format_rows(value: int) -> str:
    """Render a row count with correct pluralization."""
    return f"{format_count(value)} row" if value == 1 else f"{format_count(value)} rows"


def format_duration(seconds: float) -> str:
    """Render an elapsed time as a short human-readable string."""
    total = int(max(0.0, seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class ArchiveConversionProgress:
    """Log throttled row progress while one archive is converted into Parquet."""

    def __init__(
        self,
        label: str,
        interval_seconds: float = PROGRESS_LOG_INTERVAL_SECONDS,
    ) -> None:
        self.label = label
        self.interval_seconds = interval_seconds
        self.started_at = time.monotonic()
        self.last_report_at = self.started_at
        self.rows = 0

    def update(self, rows: int) -> None:
        """Record rows read from the source and log at most once per interval."""
        self.rows += rows
        now = time.monotonic()
        if now - self.last_report_at < self.interval_seconds:
            return
        self.last_report_at = now
        elapsed = max(now - self.started_at, 1e-9)
        LOGGER.info(
            "[%s] %s read in %s (%s rows/s)",
            self.label,
            format_rows(self.rows),
            format_duration(elapsed),
            format_count(int(self.rows / elapsed)),
        )

    def summary(self) -> tuple[int, float]:
        """Return the total rows read and the elapsed seconds."""
        return self.rows, max(time.monotonic() - self.started_at, 1e-9)


def build_output_dir(base_dir: Path, market: str, symbol: str, interval: str) -> Path:
    return base_dir / market / symbol / interval


def zip_filename(symbol: str, interval: str, d: date) -> str:
    return f"{symbol}-{interval}-{d:%Y-%m-%d}.zip"


def build_zip_path(output_dir: Path, symbol: str, interval: str, d: date) -> Path:
    return output_dir / ZIP_DIRNAME / zip_filename(symbol, interval, d)


def build_csv_path(output_dir: Path, symbol: str, interval: str) -> Path:
    return output_dir / f"{symbol}_{interval}.csv"


def build_info_path(output_dir: Path) -> Path:
    return output_dir / INFO_FILENAME


def build_earliest_dates_path(base_dir: Path) -> Path:
    return base_dir / EARLIEST_DATES_FILENAME


def load_earliest_dates(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict) or not isinstance(payload.get("earliest_dates"), dict):
        raise ValueError(f"Invalid earliest-date file format: {path}")
    earliest_dates = payload["earliest_dates"]
    for key, value in earliest_dates.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"Invalid earliest-date entry in {path}: {key!r}={value!r}")
        parse_date(value)
    return earliest_dates


def record_earliest_date(
    path: Path,
    market: str,
    symbol: str,
    discovered_date: date,
) -> date:
    path.parent.mkdir(parents=True, exist_ok=True)
    earliest_dates = load_earliest_dates(path)
    key = f"{market}:{symbol.upper()}"
    existing_date = parse_date(earliest_dates.get(key))
    earliest_date = min(existing_date, discovered_date) if existing_date else discovered_date
    earliest_dates[key] = earliest_date.isoformat()

    temp_path = path.with_suffix(path.suffix + ".temp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "schema_version": 1,
                "earliest_dates": earliest_dates,
                "last_updated_utc": utc_now_iso(),
            },
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")
    os.replace(temp_path, path)
    return earliest_date


def write_download_info(info_path: Path, info: dict[str, Any]) -> None:
    info_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = info_path.with_suffix(info_path.suffix + ".temp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, info_path)


def load_known_missing_sources(
    cache_path: Path,
    market: str,
    symbol: str,
    interval: str,
) -> set[str]:
    if not cache_path.is_file():
        return set()
    try:
        with cache_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Could not read source status cache %s: %s", cache_path, exc)
        return set()

    if not isinstance(payload, dict):
        return set()
    if any(payload.get(key) != expected for key, expected in (
        ("market", market), ("symbol", symbol), ("interval", interval)
    )):
        return set()
    sources = payload.get("sources")
    if not isinstance(sources, list) or not all(isinstance(key, str) for key in sources):
        return set()
    return set(sources)


def write_known_missing_sources(
    cache_path: Path,
    market: str,
    symbol: str,
    interval: str,
    sources: set[str],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".temp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "market": market,
                "symbol": symbol,
                "interval": interval,
                "sources": sorted(sources),
            },
            file,
            indent=2,
        )
        file.write("\n")
    os.replace(temporary_path, cache_path)


def make_download_info(
    *,
    market: str,
    symbol: str,
    interval: str,
    interval_ms: int,
    output_dir: Path,
    csv_path: Path,
    start: date,
    end: date,
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "purpose": "Human-readable download audit only; never used as downloader input.",
        "data_source": "binance_vision",
        "base_url": BASE_URL,
        "market": market,
        "symbol": symbol,
        "interval": interval,
        "interval_ms": interval_ms,
        "dataset_start_date": start.isoformat(),
        "dataset_end_date": end.isoformat(),
        "scan_start_date": start.isoformat(),
        "scan_end_date": end.isoformat(),
        "download_dir": str(output_dir),
        "zip_dir": ZIP_DIRNAME,
        "csv_file": csv_path.name,
        "scan_started_utc": utc_now_iso(),
        "last_updated_utc": utc_now_iso(),
        "source_files": {},
    }


def make_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "binance-vision-kline-downloader/2.0"
    })
    return session


class FundingApiLimiter:
    """Space funding API requests and retain cooldowns across symbols."""

    def __init__(self) -> None:
        self.next_allowed_at = 0.0

    def wait(self) -> None:
        remaining = self.next_allowed_at - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        self.next_allowed_at = time.monotonic() + FUNDING_API_MIN_INTERVAL_SECONDS

    def defer(self, retry_after: Optional[str]) -> float:
        delay = FUNDING_API_RATE_LIMIT_FALLBACK_SECONDS
        if retry_after:
            try:
                delay = max(0.0, float(retry_after))
                if not math.isfinite(delay):
                    raise ValueError("Non-finite Retry-After value")
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    delay = max(0.0, retry_at.timestamp() - time.time())
                except (TypeError, ValueError, OverflowError):
                    delay = FUNDING_API_RATE_LIMIT_FALLBACK_SECONDS
        self.next_allowed_at = max(self.next_allowed_at, time.monotonic() + delay)
        return delay


def build_prefix(market: str, symbol: str, interval: str) -> str:
    return f"{VALID_MARKETS[market]}/{symbol}/{interval}/"


def build_archive_prefix(
    market: str,
    frequency: str,
    symbol: str,
    interval: str,
) -> str:
    if frequency not in {"daily", "monthly"}:
        raise ValueError(f"Unsupported archive frequency: {frequency}")
    return f"{MARKET_ROOTS[market]}/{frequency}/klines/{symbol}/{interval}/"


def build_agg_trades_prefix(market: str, frequency: str, symbol: str) -> str:
    if frequency not in {"daily", "monthly"}:
        raise ValueError(f"Unsupported archive frequency: {frequency}")
    return f"{MARKET_ROOTS[market]}/{frequency}/aggTrades/{symbol}/"


def agg_trades_zip_filename(symbol: str, day_or_month: date, *, monthly: bool) -> str:
    date_format = "%Y-%m" if monthly else "%Y-%m-%d"
    return f"{symbol}-aggTrades-{day_or_month:{date_format}}.zip"


def build_agg_trades_url(
    market: str,
    symbol: str,
    day_or_month: date,
    *,
    monthly: bool,
) -> str:
    frequency = "monthly" if monthly else "daily"
    filename = agg_trades_zip_filename(symbol, day_or_month, monthly=monthly)
    return f"{BASE_URL}/{build_agg_trades_prefix(market, frequency, symbol)}{filename}"


def read_agg_trades_zip_layout(
    zip_path: Path,
    source_name: str,
) -> tuple[str, list[str], bool]:
    """Return the CSV member, its column names and whether it carries a header line."""
    with zipfile.ZipFile(zip_path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV file inside ZIP: {source_name}")
        csv_name = csv_names[0]
        with archive.open(csv_name) as file:
            first_line = file.readline()

    fields = first_line.decode("utf-8").strip().split(",")
    if len(fields) < 7:
        raise ValueError(
            f"Bad aggTrades row with {len(fields)} fields in {source_name}: {fields[:3]}"
        )
    column_count = min(len(fields), len(AGG_TRADES_SOURCE_COLUMNS))
    return (
        csv_name,
        AGG_TRADES_SOURCE_COLUMNS[:column_count],
        not fields[0].strip().lstrip("-").isdigit(),
    )


def build_agg_trades_table(
    table: pa.Table,
    start_ms: int,
    end_ms: int,
    day_filter: Optional[pa.Array] = None,
) -> tuple[pa.Table, Optional[pa.Array], pa.Array]:
    """Convert one raw aggTrades batch into canonical columns trimmed to the date window."""
    transact_time = table.column("transact_time")
    # Binance Vision mixes seconds, milliseconds and microseconds across vintages.
    transact_time = pc.if_else(
        pc.less(transact_time, pa.scalar(10**10, pa.int64())),
        pc.multiply(transact_time, 1000),
        transact_time,
    )
    transact_time = pc.if_else(
        pc.greater_equal(transact_time, pa.scalar(10**15, pa.int64())),
        pc.divide(transact_time, 1000),
        transact_time,
    )
    timestamps = pc.cast(transact_time, pa.timestamp("ms"))

    mask = pc.and_(
        pc.greater_equal(transact_time, pa.scalar(start_ms, pa.int64())),
        pc.less(transact_time, pa.scalar(end_ms, pa.int64())),
    )
    days = None
    if day_filter is not None:
        days = pc.cast(timestamps, pa.date32())
        mask = pc.and_(mask, pc.is_in(days, value_set=day_filter))

    if not pc.all(mask).as_py():
        table = table.filter(mask)
        transact_time = transact_time.filter(mask)
        timestamps = timestamps.filter(mask)
        if days is not None:
            days = days.filter(mask)
    years = pc.year(timestamps)
    is_best_match = (
        table.column("is_best_match")
        if "is_best_match" in table.column_names
        else pc.fill_null(pa.nulls(table.num_rows, pa.string()), "")
    )

    return (
        pa.Table.from_arrays(
            [
                table.column("agg_trade_id"),
                table.column("price"),
                table.column("quantity"),
                table.column("first_trade_id"),
                table.column("last_trade_id"),
                transact_time,
                pc.cast(pc.cast(timestamps, pa.timestamp("s"), safe=False), pa.string()),
                table.column("is_buyer_maker"),
                is_best_match,
            ],
            schema=AGG_TRADES_SCHEMA,
        ),
        days,
        years,
    )


def iter_agg_trades_tables(
    zip_path: Path,
    source_name: str,
    start_ms: int,
    end_ms: int,
    only_days: Optional[set[date]] = None,
    progress: Optional[ArchiveConversionProgress] = None,
) -> Iterable[tuple[int, pa.Table, Optional[pa.Array]]]:
    """Stream aggTrades rows from a ZIP archive as canonical yearly Arrow tables."""
    csv_name, columns, has_header = read_agg_trades_zip_layout(zip_path, source_name)
    read_options = pacsv.ReadOptions(
        column_names=columns,
        skip_rows=1 if has_header else 0,
        block_size=AGG_TRADES_READ_BLOCK_BYTES,
    )
    convert_options = pacsv.ConvertOptions(column_types={
        name: pa.string() if name in {"price", "quantity", "is_buyer_maker", "is_best_match"}
        else pa.int64()
        for name in columns
    })
    day_filter = None if only_days is None else pa.array(sorted(only_days), type=pa.date32())

    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(csv_name) as file:
            reader = pacsv.open_csv(
                file,
                read_options=read_options,
                convert_options=convert_options,
            )
            for batch in reader:
                if progress is not None:
                    progress.update(batch.num_rows)
                table, days, years = build_agg_trades_table(
                    pa.Table.from_batches([batch]),
                    start_ms,
                    end_ms,
                    day_filter,
                )
                if not table.num_rows:
                    continue
                batch_years = pc.unique(years).to_pylist()
                if len(batch_years) == 1:
                    yield batch_years[0], table, days
                    continue
                for year in batch_years:
                    year_mask = pc.equal(years, year)
                    yield (
                        year,
                        table.filter(year_mask),
                        None if days is None else days.filter(year_mask),
                    )


def inspect_agg_trades_archive_days(zip_path: Path, source_name: str) -> set[date]:
    """Read only timestamps from an aggTrades ZIP to identify covered UTC dates."""
    csv_name, columns, has_header = read_agg_trades_zip_layout(zip_path, source_name)
    read_options = pacsv.ReadOptions(
        column_names=columns,
        skip_rows=1 if has_header else 0,
        block_size=AGG_TRADES_READ_BLOCK_BYTES,
    )
    convert_options = pacsv.ConvertOptions(
        include_columns=["transact_time"],
        column_types={"transact_time": pa.int64()},
    )
    result: set[date] = set()
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(csv_name) as file:
            reader = pacsv.open_csv(
                file,
                read_options=read_options,
                convert_options=convert_options,
            )
            for batch in reader:
                timestamps = batch.column("transact_time")
                timestamps = pc.if_else(
                    pc.less(timestamps, pa.scalar(10**10, pa.int64())),
                    pc.multiply(timestamps, 1000),
                    timestamps,
                )
                timestamps = pc.if_else(
                    pc.greater_equal(timestamps, pa.scalar(10**15, pa.int64())),
                    pc.divide(timestamps, 1000),
                    timestamps,
                )
                days = pc.cast(pc.cast(timestamps, pa.timestamp("ms")), pa.date32())
                result.update(pc.unique(days).to_pylist())
    return result


def check_zip_integrity(zip_bytes: bytes, source_name: str) -> None:
    """Validate ZIP container metadata without decompressing its contents."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            if not archive.infolist():
                raise ValueError(f"ZIP archive contains no files: {source_name}")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Corrupted ZIP archive: {source_name}: {exc}") from exc


def check_zip_path_integrity(zip_path: Path) -> None:
    """Validate a ZIP archive on disk without decompressing its contents."""
    try:
        with zipfile.ZipFile(zip_path) as archive:
            if not archive.infolist():
                raise ValueError(f"ZIP archive contains no files: {zip_path}")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Corrupted ZIP archive: {zip_path}: {exc}") from exc


def day_window_ms(start: date, end: date) -> tuple[int, int]:
    """Return the half-open UTC millisecond window covering the given day range."""
    start_ms = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp() * 1000)
    end_exclusive = end + timedelta(days=1)
    end_ms = int(
        datetime(
            end_exclusive.year,
            end_exclusive.month,
            end_exclusive.day,
            tzinfo=timezone.utc,
        ).timestamp()
        * 1000
    )
    return start_ms, end_ms


def resolve_agg_trades_start_end(
    session: requests.Session,
    market: str,
    symbol: str,
    start_arg: Optional[str],
    end_arg: Optional[str],
    *,
    skip_discovery: bool = False,
    local_cache_dir: Optional[Path] = None,
) -> tuple[date, date]:
    requested_start = parse_date(start_arg)
    end = parse_date(end_arg) or (datetime.now(timezone.utc).date() - timedelta(days=1))
    if requested_start is not None:
        return requested_start, end

    if skip_discovery:
        candidates: list[date] = []
        if local_cache_dir is not None:
            filename_pattern = re.compile(
                rf"{re.escape(symbol)}-aggTrades-(\d{{4}}-\d{{2}}(?:-\d{{2}})?)\.zip"
            )
            for cache_dir in (local_cache_dir / ZIP_DIRNAME, local_cache_dir / MONTHLY_ZIP_DIRNAME):
                if not cache_dir.is_dir():
                    continue
                for archive_path in cache_dir.glob(f"{symbol}-aggTrades-*.zip"):
                    match = filename_pattern.fullmatch(archive_path.name)
                    if match is None:
                        continue
                    date_text = match.group(1)
                    date_format = "%Y-%m-%d" if len(date_text) == 10 else "%Y-%m"
                    candidates.append(datetime.strptime(date_text, date_format).date())
        if candidates:
            start = min(candidates)
            LOGGER.info(
                "Using earliest local aggTrades archive date %s; Binance Vision date discovery skipped",
                start,
            )
            return start, end

        start = FALLBACK_START[market]
        LOGGER.warning(
            "Binance Vision date discovery skipped and no local aggTrades ZIPs found; "
            "using market fallback start %s",
            start,
        )
        return start, end

    prefix = build_agg_trades_prefix(market, "daily", symbol)
    pattern = re.compile(
        rf"{re.escape(prefix)}{re.escape(symbol)}-aggTrades-(\d{{4}}-\d{{2}}-\d{{2}})\.zip"
    )
    LOGGER.info("Querying Binance Vision S3 listing for the earliest daily aggTrades ZIP:")
    LOGGER.info(f"   {S3_LIST_URL}?prefix={prefix}")
    for key in list_s3_keys(session, prefix):
        match = pattern.fullmatch(key)
        if match is not None:
            start = datetime.strptime(match.group(1), "%Y-%m-%d").date()
            LOGGER.info(f"Earliest daily aggTrades ZIP: {start}")
            return start, end
    raise NoArchiveError(
        f"No Binance Vision daily aggTrades ZIPs found for market={market}, symbol={symbol}."
    )


def read_response_bytes(
    response: requests.Response,
    url: str,
    *,
    label: str,
) -> bytes:
    """Read a streamed body in chunks, logging size and speed while it arrives."""
    declared_length = response.headers.get("Content-Length")
    total_bytes = int(declared_length) if declared_length and declared_length.isdigit() else 0
    payload = bytearray()
    started_at = time.monotonic()
    last_report_at = started_at

    for chunk in response.iter_content(DOWNLOAD_CHUNK_BYTES):
        if not chunk:
            continue
        payload.extend(chunk)
        now = time.monotonic()
        if now - last_report_at < DOWNLOAD_PROGRESS_LOG_INTERVAL_SECONDS:
            continue
        last_report_at = now
        elapsed = max(now - started_at, 1e-9)
        fraction = f" ({len(payload) / total_bytes:.0%})" if total_bytes else ""
        LOGGER.info(
            "[%s] received %s%s of %s at %s/s",
            label,
            format_byte_size(len(payload)),
            fraction,
            format_byte_size(total_bytes) if total_bytes else "unknown size",
            format_byte_size(len(payload) / elapsed),
        )

    elapsed = max(time.monotonic() - started_at, 1e-9)
    if total_bytes and len(payload) != total_bytes:
        raise ValueError(
            f"Short read for {url}: {len(payload)} of {total_bytes} bytes"
        )
    LOGGER.info(
        "[%s] download complete: %s in %s (%s/s)",
        label,
        format_byte_size(len(payload)),
        format_duration(elapsed),
        format_byte_size(len(payload) / elapsed),
    )
    return bytes(payload)


def download_agg_trades_zip(
    session: requests.Session,
    url: str,
    *,
    label: str,
) -> tuple[Optional[bytes], Optional[str]]:
    for attempt in range(HTTP_RETRIES + 1):
        try:
            response = session.get(url, timeout=HTTP_TIMEOUT_SECONDS, stream=True)
            if response.status_code == 404:
                response.close()
                return None, "missing"
            with response:
                response.raise_for_status()
                content = read_response_bytes(response, url, label=label)
            check_zip_integrity(content, url)
            return content, None
        except Exception as exc:
            if attempt < HTTP_RETRIES:
                LOGGER.warning(
                    "[%s] request failed (%s: %s); retry %d/%d in %ss",
                    label,
                    type(exc).__name__,
                    exc,
                    attempt + 1,
                    HTTP_RETRIES,
                    RETRY_DELAY_SECONDS,
                )
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            return None, f"failed: {type(exc).__name__}: {exc}"
        finally:
            time.sleep(DOWNLOAD_DELAY_SECONDS)

    return None, "failed: unknown error"


def download_symbol_agg_trades(
    args: argparse.Namespace,
    session: requests.Session,
    symbol: str,
) -> Optional[Path]:
    market = args.market
    symbol = symbol.upper()
    interval = "aggTrades"
    para = common.MarketDataSourceConfig(
        market_category="Cryptocurrency",
        data_source="binance_public_data",
        symbols=(symbol,),
        interval=interval,
        trading_type=market,
    )
    market_data_path = Path(common.market_data_path(para)).expanduser().resolve()
    parquet_dir = market_data_path.parent
    LOGGER.info("[%s aggTrades] Output Parquet directory: %s", symbol, parquet_dir)

    output_dir = build_output_dir(
        Path(args.out_dir).expanduser().resolve(), market, symbol, interval
    )
    daily_zip_dir = output_dir / ZIP_DIRNAME
    monthly_zip_dir = output_dir / MONTHLY_ZIP_DIRNAME
    info_path = build_info_path(output_dir)
    missing_sources_path = output_dir / MISSING_SOURCES_FILENAME
    known_missing_sources = (
        set() if args.force
        else load_known_missing_sources(missing_sources_path, market, symbol, interval)
    )
    start, end = resolve_agg_trades_start_end(
        session,
        market,
        symbol,
        args.start,
        args.end,
        skip_discovery=args.skip_start_date_discovery or args.offline,
        local_cache_dir=output_dir,
    )
    LOGGER.info(
        "[%s aggTrades] Resolved date range: %s -> %s UTC (%d days); checking existing Parquet coverage",
        symbol,
        start,
        end,
        (end - start).days + 1,
    )
    if start > end:
        raise ValueError(f"start must be <= end, got start={start}, end={end}")
    expected_parquet_paths = [
        parquet_dir / f"{symbol}_aggTrades_{year}.parquet"
        for year in range(start.year, end.year + 1)
    ]
    year_spans = {
        year: (max(start, date(year, 1, 1)), min(end, date(year, 12, 31)))
        for year in range(start.year, end.year + 1)
    }
    previous_coverage: dict[str, Any] = {}
    if info_path.is_file() and not args.force:
        try:
            with info_path.open("r", encoding="utf-8") as file:
                cached_info = json.load(file)
        except (OSError, json.JSONDecodeError):
            cached_info = {}
        if (
            isinstance(cached_info, dict)
            and cached_info.get("data_type") == "aggTrades"
            and cached_info.get("market") == market
            and cached_info.get("symbol") == symbol
            and cached_info.get("daily_gap_scan_version") == 1
            and cached_info.get("agg_trade_integrity_version") == AGG_TRADES_INTEGRITY_VERSION
            and isinstance(cached_info.get("parquet_year_coverage"), dict)
        ):
            previous_coverage = cached_info["parquet_year_coverage"]

    def has_complete_year(year: int) -> bool:
        """Return whether the audit record confirms a finished yearly file for the request."""
        path = parquet_dir / f"{symbol}_aggTrades_{year}.parquet"
        if not path.is_file():
            return False
        entry = previous_coverage.get(str(year))
        if not isinstance(entry, dict) or entry.get("complete") is not True:
            return False
        fingerprint = entry.get("file_fingerprint")
        stat = path.stat()
        if fingerprint != {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
            return False
        try:
            covered_start = date.fromisoformat(entry["start"])
            covered_end = date.fromisoformat(entry["end"])
        except (KeyError, TypeError, ValueError):
            return False
        requested_start, requested_end = year_spans[year]
        return covered_start <= requested_start and covered_end >= requested_end

    pending_years = {year for year in year_spans if not has_complete_year(year)}
    # Reused years must survive in the audit file: an interrupted run rebuilds only
    # the pending years, and dropping the others would make the next run redo them.
    reused_year_coverage = {
        str(year): previous_coverage[str(year)]
        for year in sorted(year_spans)
        if year not in pending_years and isinstance(previous_coverage.get(str(year)), dict)
    }
    LOGGER.info(
        "[%s aggTrades] Existing Parquet coverage checked: %d/%d year(s) reusable; pending years: %s",
        symbol,
        len(year_spans) - len(pending_years),
        len(year_spans),
        ", ".join(str(year) for year in sorted(pending_years)) or "none",
    )
    if not pending_years:
        LOGGER.info(
            "[%s aggTrades] Reusing existing yearly Parquet files: %s",
            symbol,
            ", ".join(str(path) for path in expected_parquet_paths),
        )
        return expected_parquet_paths[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    daily_zip_dir.mkdir(parents=True, exist_ok=True)
    monthly_zip_dir.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {
        "schema_version": 1,
        "daily_gap_scan_version": 1,
        "agg_trade_integrity_version": AGG_TRADES_INTEGRITY_VERSION,
        "data_source": "binance_vision",
        "data_type": "aggTrades",
        "market": market,
        "symbol": symbol,
        "interval": interval,
        "dataset_start_date": start.isoformat(),
        "dataset_end_date": end.isoformat(),
        "download_dir": str(output_dir),
        "daily_zip_dir": ZIP_DIRNAME,
        "monthly_zip_dir": MONTHLY_ZIP_DIRNAME,
        "parquet_files": [path.name for path in expected_parquet_paths],
        "parquet_year_coverage": reused_year_coverage,
        "scan_started_utc": utc_now_iso(),
        "last_updated_utc": utc_now_iso(),
        "source_files": {},
    }
    info_path.parent.mkdir(parents=True, exist_ok=True)
    write_download_info(info_path, info)

    counters = {
        "monthly_cached": 0,
        "monthly_downloaded": 0,
        "monthly_missing": 0,
        "daily_cached": 0,
        "daily_downloaded": 0,
        "daily_missing": 0,
        "failed": 0,
    }

    def record_source(
        key: str,
        frequency: str,
        status: str,
        path: Path,
        url: str,
        error: Optional[str] = None,
    ) -> None:
        entry: dict[str, Any] = {
            "frequency": frequency,
            "status": status,
            "file": str(path.relative_to(output_dir)),
            "url": url,
            "checked_at_utc": utc_now_iso(),
        }
        if path.exists():
            entry["bytes"] = path.stat().st_size
        if error:
            entry["error"] = error
        if status == "missing":
            known_missing_sources.add(key)
        elif status in {"cached", "downloaded"}:
            known_missing_sources.discard(key)
        if status in {"missing", "cached", "downloaded"}:
            write_known_missing_sources(
                missing_sources_path, market, symbol, interval, known_missing_sources
            )
        info["source_files"][key] = entry
        info["last_updated_utc"] = utc_now_iso()
        write_download_info(info_path, info)

    builders: dict[int, AggTradeYearBuilder] = {}
    raw_rows = 0
    unavailable_dates: list[date] = []
    active_minutes = load_active_kline_minutes(parquet_dir, symbol, start, end)

    def write_parquet_table(year: int, table: pa.Table) -> None:
        nonlocal raw_rows
        if table.num_rows == 0:
            return
        raw_rows += table.num_rows
        if year not in builders:
            builders[year] = AggTradeYearBuilder(parquet_dir, symbol, year)
        builders[year].append(table)

    def consume_source(
        key: str,
        frequency: str,
        path: Path,
        url: str,
        start_ms: int,
        end_ms: int,
        only_days: Optional[set[date]] = None,
    ) -> Optional[set[date]]:
        """Stream one archive into the yearly Parquet files; None means it is unavailable."""
        temporary_path = path.with_suffix(path.suffix + ".temp")
        cached = False
        for candidate in (path, temporary_path):
            if not candidate.is_file():
                continue
            try:
                check_zip_path_integrity(candidate)
            except ValueError as exc:
                LOGGER.warning("Ignoring invalid cache: %s", exc)
                continue
            if candidate == temporary_path:
                os.replace(temporary_path, path)
            cached = True
            counters[f"{frequency}_cached"] += 1
            record_source(key, frequency, "cached", path, url)
            LOGGER.info(
                "[%s aggTrades] %s: reusing cached ZIP %s (%s)",
                symbol,
                key,
                path.name,
                format_byte_size(path.stat().st_size),
            )
            break

        if not cached:
            if args.offline:
                counters[f"{frequency}_missing"] += 1
                record_source(
                    key,
                    frequency,
                    "unavailable_offline",
                    path,
                    url,
                    "Archive is not present in the local cache and offline mode is enabled.",
                )
                LOGGER.warning(
                    "[%s aggTrades] %s: unavailable locally; skipped because offline mode is enabled",
                    symbol,
                    key,
                )
                return None

            if key in known_missing_sources:
                counters[f"{frequency}_missing"] += 1
                record_source(key, frequency, "missing", path, url)
                LOGGER.info(
                    "[%s aggTrades] %s: previously confirmed missing, skipping",
                    symbol,
                    key,
                )
                return None

            LOGGER.info("[%s aggTrades] Downloading %s: %s", symbol, key, url)
            content, error = download_agg_trades_zip(
                session, url, label=f"{symbol} aggTrades {key}"
            )
            if content is None:
                if error == "missing":
                    counters[f"{frequency}_missing"] += 1
                    record_source(key, frequency, "missing", path, url)
                    LOGGER.info(
                        "[%s aggTrades] %s: archive absent (HTTP 404), skipping",
                        symbol,
                        key,
                    )
                    return None
                counters["failed"] += 1
                record_source(key, frequency, "failed", path, url, error)
                raise RuntimeError(f"[{symbol} aggTrades] Failed to download {url}: {error}")
            write_zip_cache(path, content)
            counters[f"{frequency}_downloaded"] += 1
            record_source(key, frequency, "downloaded", path, url)
            LOGGER.info(
                "[%s aggTrades] %s: cached ZIP saved (%s)",
                symbol,
                key,
                format_byte_size(path.stat().st_size),
            )

        source_bytes = path.stat().st_size
        LOGGER.info(
            "[%s aggTrades] %s: converting %s of CSV into Parquet",
            symbol,
            key,
            format_byte_size(source_bytes),
        )
        progress = ArchiveConversionProgress(f"{symbol} aggTrades {key}")
        written_days: set[date] = set()
        for year, table, days in iter_agg_trades_tables(
            path, str(path), start_ms, end_ms, only_days, progress=progress
        ):
            write_parquet_table(year, table)
            if days is not None:
                written_days.update(days.to_pylist())
        rows, elapsed = progress.summary()
        LOGGER.info(
            "[%s aggTrades] %s: converted %s from %s in %s (%s rows/s)",
            symbol,
            key,
            format_rows(rows),
            format_byte_size(source_bytes),
            format_duration(elapsed),
            format_count(int(rows / elapsed)),
        )
        return written_days

    def planned_payload_bytes(month: date) -> int:
        """Return the cached archive bytes expected for one month, for progress estimates."""
        first_day, last_day = month_bounds(month)
        segment_start = max(start, first_day)
        segment_end = min(end, last_day)
        monthly_path = monthly_zip_dir / agg_trades_zip_filename(symbol, month, monthly=True)
        complete_month = segment_start == first_day and segment_end == last_day
        if complete_month and monthly_path.is_file():
            return monthly_path.stat().st_size
        return sum(
            daily_path.stat().st_size
            for daily_path in (
                daily_zip_dir / agg_trades_zip_filename(symbol, day, monthly=False)
                for day in date_range(segment_start, segment_end)
            )
            if daily_path.is_file()
        )

    def process_month(month: date) -> None:
        """Convert every available archive of one month into the yearly Parquet files."""
        first_day, last_day = month_bounds(month)
        segment_start = max(start, first_day)
        segment_end = min(end, last_day)
        complete_month = segment_start == first_day and segment_end == last_day
        segment_start_ms, segment_end_ms = day_window_ms(segment_start, segment_end)
        monthly_path = monthly_zip_dir / agg_trades_zip_filename(symbol, month, monthly=True)
        monthly_url = build_agg_trades_url(market, symbol, month, monthly=True)
        monthly_key = f"monthly:{month_key(month)}"

        monthly_unavailable = False
        if complete_month:
            monthly_days: Optional[set[date]] = None
            has_local_daily_archives = any(
                (daily_zip_dir / agg_trades_zip_filename(symbol, day, monthly=False)).is_file()
                for day in date_range(segment_start, segment_end)
            )
            if monthly_path.is_file() and has_local_daily_archives:
                try:
                    LOGGER.info(
                        "[%s aggTrades] Checking date coverage in cached %s before conversion",
                        symbol,
                        monthly_path.name,
                    )
                    monthly_days = inspect_agg_trades_archive_days(
                        monthly_path, str(monthly_path)
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "[%s aggTrades] Could not inspect monthly archive dates (%s: %s); "
                        "using normal monthly processing",
                        symbol,
                        type(exc).__name__,
                        exc,
                    )

            if monthly_days is not None:
                missing_days = [
                    day for day in date_range(segment_start, segment_end)
                    if day not in monthly_days
                ]
                if missing_days:
                    LOGGER.warning(
                        "[%s aggTrades] %s has no rows for %d day(s); "
                        "using daily ZIPs for those dates and monthly ZIPs for the rest",
                        symbol,
                        monthly_key,
                        len(missing_days),
                    )
                    missing_set = set(missing_days)
                    run_start: Optional[date] = None
                    run_days: set[date] = set()

                    def flush_monthly_run() -> None:
                        nonlocal run_start, run_days
                        if run_start is None or not run_days:
                            return
                        run_end = max(run_days)
                        run_start_ms, run_end_ms = day_window_ms(run_start, run_end)
                        consume_source(
                            monthly_key,
                            "monthly",
                            monthly_path,
                            monthly_url,
                            run_start_ms,
                            run_end_ms,
                            only_days=run_days,
                        )
                        run_start = None
                        run_days = set()

                    for day in date_range(segment_start, segment_end):
                        if day in missing_set:
                            flush_monthly_run()
                            day_path = daily_zip_dir / agg_trades_zip_filename(
                                symbol, day, monthly=False
                            )
                            day_url = build_agg_trades_url(
                                market, symbol, day, monthly=False
                            )
                            day_start_ms, day_end_ms = day_window_ms(day, day)
                            daily_key = f"daily:{day.isoformat()}"
                            daily_days = consume_source(
                                daily_key,
                                "daily",
                                day_path,
                                day_url,
                                day_start_ms,
                                day_end_ms,
                            )
                            if daily_days is None:
                                unavailable_dates.append(day)
                                continue
                            info["source_files"][daily_key]["checked_for_monthly_gap"] = monthly_key
                            if day in daily_days:
                                info["source_files"][daily_key]["status"] = "filled_from_daily"
                                info["source_files"][monthly_key].setdefault(
                                    "recovered_from_daily", []
                                ).append(day.isoformat())
                            continue

                        if run_start is None:
                            run_start = day
                        run_days.add(day)

                    flush_monthly_run()
                    info["source_files"][monthly_key]["dates_without_rows"] = [
                        day.isoformat() for day in missing_days
                    ]
                    write_download_info(info_path, info)
                    return

            written_days = consume_source(
                monthly_key,
                "monthly",
                monthly_path,
                monthly_url,
                segment_start_ms,
                segment_end_ms,
            )
            if written_days is not None:
                return
            monthly_unavailable = True

        missing_days: list[date] = []
        for day in date_range(segment_start, segment_end):
            day_path = daily_zip_dir / agg_trades_zip_filename(symbol, day, monthly=False)
            day_url = build_agg_trades_url(market, symbol, day, monthly=False)
            day_start_ms, day_end_ms = day_window_ms(day, day)
            written_days = consume_source(
                f"daily:{day.isoformat()}",
                "daily",
                day_path,
                day_url,
                day_start_ms,
                day_end_ms,
            )
            if written_days is not None:
                continue
            missing_days.append(day)
            if args.show_missing:
                LOGGER.warning("[%s aggTrades] Daily archive missing: %s", symbol, day)

        if not missing_days:
            return

        if monthly_unavailable:
            filled_days: set[date] = set()
        else:
            filled_days = consume_source(
                monthly_key,
                "monthly",
                monthly_path,
                monthly_url,
                segment_start_ms,
                segment_end_ms,
                only_days=set(missing_days),
            ) or set()

        for missing_day in missing_days:
            if missing_day not in filled_days:
                unavailable_dates.append(missing_day)
                continue
            source_key = f"daily:{missing_day.isoformat()}"
            info["source_files"][source_key]["status"] = "filled_from_monthly"
            info["source_files"][source_key]["daily_archive_status"] = "missing"
            info["source_files"][source_key]["fallback_source"] = monthly_key
        write_download_info(info_path, info)

    planned_months = [
        month for month in iter_month_starts(start, end) if month.year in pending_years
    ]
    planned_bytes = {month: planned_payload_bytes(month) for month in planned_months}
    total_payload_bytes = sum(planned_bytes.values())
    reused_years = sorted(year for year in year_spans if year not in pending_years)

    LOGGER.info("=" * 80)
    LOGGER.info("[%s aggTrades] Binance Vision aggregate-trade downloader", symbol)
    LOGGER.info(
        "[%s aggTrades] Scan range     : %s -> %s UTC (%d days)",
        symbol,
        start,
        end,
        (end - start).days + 1,
    )
    LOGGER.info(
        "[%s aggTrades] Pending years  : %s",
        symbol,
        ", ".join(str(year) for year in sorted(pending_years)),
    )
    if reused_years:
        LOGGER.info(
            "[%s aggTrades] Reused years   : %s",
            symbol,
            ", ".join(str(year) for year in reused_years),
        )
    LOGGER.info(
        "[%s aggTrades] Pending months : %d (cached ZIP payload %s)",
        symbol,
        len(planned_months),
        format_byte_size(total_payload_bytes),
    )
    LOGGER.info("[%s aggTrades] Monthly cache  : %s", symbol, monthly_zip_dir)
    LOGGER.info("[%s aggTrades] Daily cache    : %s", symbol, daily_zip_dir)
    LOGGER.info("[%s aggTrades] Yearly Parquet : %s", symbol, parquet_dir)
    LOGGER.info("=" * 80)

    completed_year_coverage: dict[str, Any] = dict(reused_year_coverage)
    last_month_of_year: dict[int, date] = {}
    for planned_month in planned_months:
        last_month_of_year[planned_month.year] = planned_month

    def finalise_year(year: int) -> None:
        """Repair local gaps, deduplicate, sort and publish a finished year."""
        if year not in builders:
            builders[year] = AggTradeYearBuilder(parquet_dir, symbol, year)
        builder = builders[year]
        span_start, span_end = year_spans[year]
        start_ms, end_ms = day_window_ms(span_start, span_end)
        expected_minutes = {
            minute for minute in active_minutes if start_ms <= minute * 60_000 < end_ms
        }
        previous = None
        if year > start.year:
            prior_path = parquet_dir / f"{symbol}_aggTrades_{year - 1}.parquet"
            if prior_path.is_file():
                prior = parquet.ParquetFile(prior_path)
                if prior.metadata.num_rows:
                    last = prior.read_row_group(
                        prior.num_row_groups - 1,
                        columns=["agg_trade_id", "transact_time_ms_utc"],
                    )
                    previous = (
                        int(last["agg_trade_id"][-1].as_py()),
                        int(last["transact_time_ms_utc"][-1].as_py()),
                    )
        initial = builder.inspect(span_start, span_end, expected_minutes, previous)
        repair_days = {date.fromisoformat(day) for day in initial["missing_dates"]}
        repair_days.update(
            datetime.fromtimestamp(minute * 60, timezone.utc).date()
            for minute in initial["missing_kline_minutes"]
        )
        for gap in initial["id_gaps"]:
            first = datetime.fromtimestamp(gap["before_ms"] / 1000, timezone.utc).date()
            last = datetime.fromtimestamp(gap["after_ms"] / 1000, timezone.utc).date()
            repair_days.update(date_range(max(first, span_start), min(last, span_end)))
        repair_sources = []
        for day in sorted(repair_days):
            path = daily_zip_dir / agg_trades_zip_filename(symbol, day, monthly=False)
            temporary_path = path.with_suffix(path.suffix + ".temp")
            local_path = path if path.is_file() else temporary_path
            if not local_path.is_file():
                LOGGER.info(
                    "[%s aggTrades] %s: no local daily ZIP for gap recovery", symbol, day,
                )
                continue
            LOGGER.info(
                "[%s aggTrades] %s: supplementing gaps from local daily ZIP %s",
                symbol, day, local_path.name,
            )
            day_start_ms, day_end_ms = day_window_ms(day, day)
            for table_year, table, _ in iter_agg_trades_tables(
                local_path, str(local_path), day_start_ms, day_end_ms,
            ):
                if table_year != year:
                    raise ValueError(f"Unexpected year in local daily ZIP {local_path}")
                write_parquet_table(year, table)
            repair_sources.append(str(local_path.relative_to(output_dir)))

        integrity = builder.inspect(span_start, span_end, expected_minutes, previous)
        complete = not (
            integrity["id_gaps"] or integrity["missing_dates"]
            or integrity["missing_kline_minutes"]
        )
        stats = builder.publish()
        path = builder.path
        builder.cleanup()
        del builders[year]
        unavailable_dates[:] = [day for day in unavailable_dates if day.year != year]
        unavailable_dates.extend(date.fromisoformat(day) for day in integrity["missing_dates"])
        for gap in integrity["id_gaps"]:
            LOGGER.warning(
                "[%s aggTrades] Final year %d has an unresolved ID gap: %d missing IDs "
                "(%d..%d), %s -> %s UTC; local daily recovery exhausted",
                symbol, year, gap["missing_ids"], gap["before_id"] + 1,
                gap["after_id"] - 1,
                datetime.fromtimestamp(gap["before_ms"] / 1000, timezone.utc).isoformat(),
                datetime.fromtimestamp(gap["after_ms"] / 1000, timezone.utc).isoformat(),
            )
        if integrity["missing_dates"]:
            LOGGER.warning(
                "[%s aggTrades] Final year %d has dates with no records: %s",
                symbol, year, ", ".join(integrity["missing_dates"]),
            )
        if integrity["missing_kline_minutes"]:
            LOGGER.warning(
                "[%s aggTrades] Final year %d has %d minute(s) with trades in local "
                "1m klines but no aggTrades; first=%s UTC, last=%s UTC; "
                "see missing_kline_minutes in download_info.json",
                symbol, year, len(integrity["missing_kline_minutes"]),
                datetime.fromtimestamp(integrity["missing_kline_minutes"][0] * 60, timezone.utc).isoformat(),
                datetime.fromtimestamp(integrity["missing_kline_minutes"][-1] * 60, timezone.utc).isoformat(),
            )
        stat = path.stat()
        completed_year_coverage[str(year)] = {
            "start": span_start.isoformat(),
            "end": span_end.isoformat(),
            "complete": complete,
            **stats,
            **integrity,
            "local_daily_recovery_sources": repair_sources,
            "recovered_missing_ids": (
                sum(gap["missing_ids"] for gap in initial["id_gaps"])
                - sum(gap["missing_ids"] for gap in integrity["id_gaps"])
            ),
            "file_fingerprint": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        }
        info["parquet_year_coverage"] = {
            key: completed_year_coverage[key] for key in sorted(completed_year_coverage)
        }
        info["last_updated_utc"] = utc_now_iso()
        write_download_info(info_path, info)
        LOGGER.info(
            "[%s aggTrades] year %d finalised: %s in %s; %s duplicate rows removed (%s)",
            symbol,
            year,
            format_rows(parquet.ParquetFile(path).metadata.num_rows),
            format_byte_size(path.stat().st_size),
            format_count(stats["duplicates_removed"]),
            "coverage complete" if complete else "gaps recorded",
        )

    conversion_started_at = time.monotonic()
    processed_payload_bytes = 0
    for index, month in enumerate(planned_months, start=1):
        month_started_at = time.monotonic()
        month_bytes = planned_bytes[month]
        LOGGER.info(
            "[%s aggTrades] month %d/%d %s: starting (archive payload %s, %s converted so far)",
            symbol,
            index,
            len(planned_months),
            month_key(month),
            format_byte_size(month_bytes),
            format_byte_size(processed_payload_bytes),
        )
        try:
            process_month(month)
            if month == last_month_of_year.get(month.year):
                finalise_year(month.year)
        except BaseException:
            for builder in builders.values():
                builder.cleanup()
            raise

        processed_payload_bytes += month_bytes
        elapsed_month = max(time.monotonic() - month_started_at, 1e-9)
        if not total_payload_bytes:
            LOGGER.info(
                "[%s aggTrades] month %d/%d %s: done in %s (%s written)",
                symbol,
                index,
                len(planned_months),
                month_key(month),
                format_duration(elapsed_month),
                format_rows(raw_rows),
            )
            continue

        elapsed_total = max(time.monotonic() - conversion_started_at, 1e-9)
        speed = processed_payload_bytes / elapsed_total
        remaining_bytes = max(total_payload_bytes - processed_payload_bytes, 0)
        if not remaining_bytes:
            eta_text = "complete"
        elif index >= 2 and speed > 0:
            eta_text = format_duration(remaining_bytes / speed)
        else:
            eta_text = "unknown"
        LOGGER.info(
            "[%s aggTrades] month %d/%d %s: done in %s (%s written); "
            "payload %s/%s (%.0f%%), average %s/s, ETA %s",
            symbol,
            index,
            len(planned_months),
            month_key(month),
            format_duration(elapsed_month),
            format_rows(raw_rows),
            format_byte_size(processed_payload_bytes),
            format_byte_size(total_payload_bytes),
            100.0 * processed_payload_bytes / total_payload_bytes,
            format_byte_size(speed),
            eta_text,
        )

    if counters["failed"]:
        raise RuntimeError(f"Source download/read failures={counters['failed']}")

    info["unavailable_dates"] = [day.isoformat() for day in sorted(set(unavailable_dates))]

    available_paths = [path for path in expected_parquet_paths if path.is_file()]
    if not available_paths:
        info.update({
            "final_rows": 0,
            "last_updated_utc": utc_now_iso(),
            "summary": {**counters, "complete": False},
        })
        write_download_info(info_path, info)
        LOGGER.warning("[%s aggTrades] No data found for %s through %s", symbol, start, end)
        return None

    year_coverage: dict[str, Any] = dict(completed_year_coverage)
    final_rows = sum(
        parquet.ParquetFile(path).metadata.num_rows for path in available_paths
    )
    info.update({
        "market_data_paths": [str(path) for path in available_paths],
        "parquet_year_coverage": {key: year_coverage[key] for key in sorted(year_coverage)},
        "raw_rows": raw_rows,
        "final_rows": final_rows,
        "last_updated_utc": utc_now_iso(),
        "summary": {
            **counters,
            "unavailable_dates": len(info["unavailable_dates"]),
            "unresolved_id_gaps": sum(len(entry.get("id_gaps", [])) for entry in year_coverage.values()),
            "missing_ids": sum(
                gap["missing_ids"] for entry in year_coverage.values()
                for gap in entry.get("id_gaps", [])
            ),
            "duplicates_removed": sum(entry.get("duplicates_removed", 0) for entry in year_coverage.values()),
            "complete": not unavailable_dates and all(
                isinstance(year_coverage.get(str(year)), dict)
                and year_coverage[str(year)].get("complete") is True
                for year in year_spans
            ),
        },
    })
    write_download_info(info_path, info)

    LOGGER.info(
        "[%s aggTrades] Yearly Parquet files contain %s rows: %s",
        symbol,
        final_rows,
        ", ".join(str(path) for path in available_paths),
    )
    return available_paths[0]


def build_daily_url(market: str, symbol: str, interval: str, d: date) -> str:
    day = d.strftime("%Y-%m-%d")
    filename = f"{symbol}-{interval}-{day}.zip"
    return f"{BASE_URL}/{build_prefix(market, symbol, interval)}{filename}"


def month_key(d: date) -> str:
    return d.strftime("%Y-%m")


def month_bounds(d: date) -> tuple[date, date]:
    last_day = calendar.monthrange(d.year, d.month)[1]
    return date(d.year, d.month, 1), date(d.year, d.month, last_day)


def iter_month_starts(start: date, end: date) -> Iterable[date]:
    current = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while current <= last:
        yield current
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def monthly_zip_filename(symbol: str, interval: str, month: date) -> str:
    return f"{symbol}-{interval}-{month:%Y-%m}.zip"


def build_monthly_url(market: str, symbol: str, interval: str, month: date) -> str:
    filename = monthly_zip_filename(symbol, interval, month)
    prefix = build_archive_prefix(market, "monthly", symbol, interval)
    return f"{BASE_URL}/{prefix}{filename}"


def build_monthly_zip_path(
    output_dir: Path,
    symbol: str,
    interval: str,
    month: date,
) -> Path:
    return output_dir / MONTHLY_ZIP_DIRNAME / monthly_zip_filename(symbol, interval, month)


def list_s3_keys(session: requests.Session, prefix: str) -> list[str]:
    """Read all object keys from the paginated S3 XML listing."""
    keys: list[str] = []
    marker: Optional[str] = None
    while True:
        params = {"prefix": prefix, "max-keys": "1000"}
        if marker is not None:
            params["marker"] = marker
        for attempt in range(HTTP_RETRIES + 1):
            try:
                response = session.get(
                    S3_LIST_URL,
                    params=params,
                    timeout=HTTP_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                if attempt == HTTP_RETRIES:
                    raise RuntimeError(
                        f"S3 listing failed after {HTTP_RETRIES} retries: {exc}"
                    ) from exc
                time.sleep(RETRY_DELAY_SECONDS)
        root = ET.fromstring(response.content)
        page_keys = [
            element.text
            for element in root.iter()
            if element.tag.rsplit("}", 1)[-1] == "Key" and element.text
        ]
        keys.extend(page_keys)
        truncated = any(
            (element.text or "").lower() == "true"
            for element in root.iter()
            if element.tag.rsplit("}", 1)[-1] == "IsTruncated"
        )
        if not truncated:
            return keys
        if not page_keys or page_keys[-1] == marker:
            raise RuntimeError(f"S3 listing cannot advance: prefix={prefix}")
        marker = page_keys[-1]


def infer_available_dates_from_listing(
    session: requests.Session,
    market: str,
    symbol: str,
    interval: str,
) -> list[date]:
    """Find the first and last available daily ZIPs in the S3 XML listing."""
    prefix = build_archive_prefix(market, "daily", symbol, interval)
    LOGGER.info("Querying Binance Vision S3 listing for the earliest daily ZIP:")
    LOGGER.info(f"   {S3_LIST_URL}?prefix={prefix}")
    pattern = re.compile(
        rf"{re.escape(prefix)}{re.escape(symbol)}-{re.escape(interval)}-"
        rf"(\d{{4}}-\d{{2}}-\d{{2}})\.zip",
    )
    dates: list[date] = []
    for key in list_s3_keys(session, prefix):
        match = pattern.fullmatch(key)
        if match is not None:
            dates.append(datetime.strptime(match.group(1), "%Y-%m-%d").date())

    if dates:
        bounds = [min(dates), max(dates)]
        LOGGER.info(f"Daily ZIP availability: {bounds[0]} through {bounds[-1]}")
        return bounds

    LOGGER.warning("S3 listing contains no daily ZIP filename.")
    return []


def resolve_start_end(
    session: requests.Session,
    market: str,
    symbol: str,
    interval: str,
    start_arg: Optional[str],
    end_arg: Optional[str],
    cached_start: Optional[date] = None,
) -> tuple[date, date, bool]:
    requested_start = parse_date(start_arg)
    requested_end = parse_date(end_arg) or (datetime.now(timezone.utc).date() - timedelta(days=1))

    if cached_start is not None:
        effective_end = requested_end
        if cached_start > effective_end:
            raise ValueError(
                "No available data in requested date range: "
                f"cached_start={cached_start}, requested_end={requested_end}"
            )
        if requested_start is not None and requested_start != cached_start:
            LOGGER.info(
                f"Ignoring --start={requested_start}; using recorded earliest "
                f"date for {market}:{symbol}: {cached_start}"
            )
        else:
            LOGGER.info(f"Using cached earliest date for {market}:{symbol}: {cached_start}")
        return cached_start, effective_end, True

    dates = infer_available_dates_from_listing(session, market, symbol, interval)

    if dates:
        effective_end = min(requested_end, dates[-1])
        effective_start = dates[0]
        if effective_start > effective_end:
            raise ValueError(
                "No available ZIP files in requested date range: "
                f"earliest_available={effective_start}, requested_end={requested_end}"
            )
        if requested_start is not None and requested_start != effective_start:
            LOGGER.info(
                f"Ignoring --start={requested_start}; using earliest listed "
                f"date for {market}:{symbol}:{interval}: {effective_start}"
            )
        else:
            LOGGER.info(f"Using earliest available file date: {effective_start}")
        if effective_end < requested_end:
            LOGGER.warning(
                "Requested end date is later than the last listed daily ZIP; "
                "requested_end=%s, using_end=%s",
                requested_end,
                effective_end,
            )
        return effective_start, effective_end, True

    symbol_hint = (
        " The USD-M SHIB perpetual symbol is 1000SHIBUSDT."
        if market == "um" and symbol == "SHIBUSDT"
        else ""
    )
    raise NoArchiveError(
        "No Binance Vision daily ZIPs found for "
        f"market={market}, symbol={symbol}, interval={interval}. "
        f"Check the market symbol.{symbol_hint}"
    )


def read_kline_rows_from_zip(zip_bytes: bytes, source_name: str) -> list[list[str]]:
    rows: list[list[str]] = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV file inside ZIP: {source_name}")

        with zf.open(csv_names[0], "r") as f:
            text = io.TextIOWrapper(f, encoding="utf-8")
            reader = csv.reader(text)

            for row in reader:
                if not row:
                    continue

                first = row[0].strip().lower()
                if first in {"open_time", "open time", "open_time_ms_utc"}:
                    continue

                if len(row) < 12:
                    raise ValueError(f"Bad row with {len(row)} fields in {source_name}: {row[:3]}")

                open_time_ms = normalize_ts_to_ms(row[0])
                close_time_ms = normalize_ts_to_ms(row[6])

                rows.append([
                    str(open_time_ms),
                    ms_to_dt(open_time_ms),
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    str(close_time_ms),
                    row[7],
                    row[8],
                    row[9],
                    row[10],
                    row[11] if len(row) > 11 else "0",
                ])

    return rows


def download_zip_url(
    session: requests.Session,
    url: str,
) -> tuple[Optional[bytes], Optional[str]]:
    """Download and structurally validate one Binance Vision ZIP."""
    for attempt in range(HTTP_RETRIES + 1):
        try:
            response = session.get(url, timeout=HTTP_TIMEOUT_SECONDS)
            if response.status_code == 404:
                return None, "missing"
            response.raise_for_status()
            if not read_kline_rows_from_zip(response.content, url):
                raise ValueError("Downloaded archive contains no candles")
            return response.content, None
        except Exception as exc:
            if attempt < HTTP_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            return None, f"failed: {type(exc).__name__}: {exc}"
        finally:
            time.sleep(DOWNLOAD_DELAY_SECONDS)
    return None, "failed: unknown error"


def write_zip_cache(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".temp")
    with tmp_path.open("wb") as file:
        file.write(content)
    os.replace(tmp_path, path)


def filter_rows_by_dates(rows: Iterable[list[str]], start: date, end: date) -> list[list[str]]:
    start_ms = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp() * 1000)
    end_exclusive = end + timedelta(days=1)
    end_ms = int(
        datetime(
            end_exclusive.year,
            end_exclusive.month,
            end_exclusive.day,
            tzinfo=timezone.utc,
        ).timestamp()
        * 1000
    )
    return [row for row in rows if start_ms <= int(row[0]) < end_ms]


def find_incomplete_intraday_dates(
    rows: Iterable[list[str]],
    start: date,
    end: date,
    interval_ms: int,
) -> list[date]:
    """Return dates whose UTC-aligned intraday candles are not fully covered."""
    day_ms = 86_400_000
    if interval_ms > day_ms or day_ms % interval_ms != 0:
        return []

    actual_open_times = {int(row[0]) for row in rows}
    expected_per_day = day_ms // interval_ms
    incomplete: list[date] = []
    for current_date in date_range(start, end):
        day_start_ms = int(
            datetime(
                current_date.year,
                current_date.month,
                current_date.day,
                tzinfo=timezone.utc,
            ).timestamp()
            * 1000
        )
        if any(
            day_start_ms + offset * interval_ms not in actual_open_times
            for offset in range(expected_per_day)
        ):
            incomplete.append(current_date)
    return incomplete


def merge_rows_by_open_time(rows: Iterable[list[str]]) -> tuple[list[list[str]], int]:
    by_open_time: dict[int, list[str]] = {}
    duplicates = 0

    for row in rows:
        open_time = int(row[0])
        if open_time in by_open_time:
            duplicates += 1
        by_open_time[open_time] = row

    merged = [by_open_time[k] for k in sorted(by_open_time)]
    return merged, duplicates


def write_output_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".temp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)
        writer.writerows(rows)

    os.replace(tmp_path, path)


def download_symbol_interval(
    args: argparse.Namespace,
    session: requests.Session,
    symbol: str,
    interval: str,
) -> Optional[Path]:
    """Download available archives and validate the merged candle data."""
    market = args.market
    symbol = symbol.upper()
    para = common.MarketDataSourceConfig(
        market_category="Cryptocurrency",
        data_source="binance_public_data",
        symbols=(symbol,),
        interval=interval,
        trading_type=market,
    )
    market_data_path = Path(common.market_data_path(para)).expanduser().resolve()
    if market_data_path.is_file() and not args.force:
        LOGGER.info(f"[{symbol} {interval}] Skipping entire task because final CSV exists: {market_data_path}")
        return market_data_path

    interval_ms = interval_to_ms(interval)
    output_dir = build_output_dir(
        Path(args.out_dir).expanduser().resolve(), market, symbol, interval
    )
    daily_zip_dir = output_dir / ZIP_DIRNAME
    monthly_zip_dir = output_dir / MONTHLY_ZIP_DIRNAME
    out_path = build_csv_path(output_dir, symbol, interval)
    info_path = build_info_path(output_dir)
    missing_sources_path = output_dir / MISSING_SOURCES_FILENAME
    earliest_dates_path = build_earliest_dates_path(
        Path(args.out_dir).expanduser().resolve()
    )
    cached_start = parse_date(
        load_earliest_dates(earliest_dates_path).get(f"{market}:{symbol}")
    )
    known_missing_sources = (
        set() if args.force
        else load_known_missing_sources(missing_sources_path, market, symbol, interval)
    )

    start, end, earliest_date_confirmed = resolve_start_end(
        session,
        market,
        symbol,
        interval,
        args.start,
        args.end,
        cached_start=cached_start,
    )
    if start > end:
        raise ValueError(f"start must be <= end, got start={start}, end={end}")

    output_dir.mkdir(parents=True, exist_ok=True)
    daily_zip_dir.mkdir(parents=True, exist_ok=True)
    monthly_zip_dir.mkdir(parents=True, exist_ok=True)
    info = make_download_info(
        market=market,
        symbol=symbol,
        interval=interval,
        interval_ms=interval_ms,
        output_dir=output_dir,
        csv_path=out_path,
        start=start,
        end=end,
    )
    info["archive_policy"] = (
        "existing_archives_first; monthly_for_uncached_complete_months; "
        "daily_for_partial_months; "
        "monthly_fallback_for_missing_daily"
    )
    info["monthly_zip_dir"] = MONTHLY_ZIP_DIRNAME
    write_download_info(info_path, info)

    counters = {
        "monthly_cached": 0,
        "monthly_downloaded": 0,
        "monthly_missing": 0,
        "monthly_incomplete": 0,
        "monthly_gap_dates": 0,
        "monthly_gap_dates_recovered_from_daily": 0,
        "daily_cached": 0,
        "daily_downloaded": 0,
        "daily_missing": 0,
        "failed": 0,
    }

    def record_source(
        key: str,
        *,
        frequency: str,
        status: str,
        path: Path,
        url: str,
        error: Optional[str] = None,
    ) -> None:
        entry: dict[str, Any] = {
            "frequency": frequency,
            "status": status,
            "file": str(path.relative_to(output_dir)),
            "url": url,
            "checked_at_utc": utc_now_iso(),
        }
        if path.exists():
            entry["bytes"] = path.stat().st_size
        if error:
            entry["error"] = error
        if status == "missing":
            known_missing_sources.add(key)
        elif status in {"cached", "downloaded"}:
            known_missing_sources.discard(key)
        if status in {"missing", "cached", "downloaded"}:
            write_known_missing_sources(
                missing_sources_path, market, symbol, interval, known_missing_sources
            )
        info["source_files"][key] = entry
        info["last_updated_utc"] = utc_now_iso()
        write_download_info(info_path, info)

    def load_source(
        key: str, frequency: str, path: Path, url: str
    ) -> tuple[Optional[list[list[str]]], Optional[str]]:
        # A completed atomic-write temporary file can survive an interruption.
        temporary_path = path.with_suffix(path.suffix + ".temp")
        for candidate in (path, temporary_path):
            if not candidate.is_file():
                continue
            try:
                rows = read_kline_rows_from_zip(candidate.read_bytes(), str(candidate))
                if not rows:
                    raise ValueError("Cached archive contains no candles")
            except Exception as exc:
                LOGGER.warning(f"Ignoring invalid cache {candidate}: {type(exc).__name__}: {exc}")
                continue
            if candidate == temporary_path:
                os.replace(temporary_path, path)
            counters[f"{frequency}_cached"] += 1
            record_source(key, frequency=frequency, status="cached", path=path, url=url)
            LOGGER.info(
                f"[{symbol} {interval}] {key}: reusing cached ZIP "
                f"{path.name} ({len(rows)} candles)"
            )
            return rows, None

        if key in known_missing_sources:
            counters[f"{frequency}_missing"] += 1
            record_source(key, frequency=frequency, status="missing", path=path, url=url)
            LOGGER.info(f"[{symbol} {interval}] Skipping previously confirmed missing source: {key}")
            return None, "missing"

        LOGGER.info(f"[{symbol} {interval}] Downloading {key}: {url}")
        content, error = download_zip_url(session, url)
        if content is not None:
            write_zip_cache(path, content)
            counters[f"{frequency}_downloaded"] += 1
            record_source(key, frequency=frequency, status="downloaded", path=path, url=url)
            LOGGER.info(f"[{symbol} {interval}] {key}: downloaded ZIP {len(content)} bytes")
            return read_kline_rows_from_zip(content, url), None
        if error == "missing":
            counters[f"{frequency}_missing"] += 1
            record_source(key, frequency=frequency, status="missing", path=path, url=url)
            return None, "missing"
        counters["failed"] += 1
        record_source(
            key, frequency=frequency, status="failed", path=path, url=url, error=error
        )
        raise RuntimeError(f"[{symbol} {interval}] Failed to download source {url}: {error}")

    def load_monthly(month: date) -> tuple[Optional[list[list[str]]], Optional[str]]:
        return load_source(
            f"monthly:{month_key(month)}", "monthly",
            build_monthly_zip_path(output_dir, symbol, interval, month),
            build_monthly_url(market, symbol, interval, month),
        )

    def load_daily(day: date) -> tuple[Optional[list[list[str]]], Optional[str]]:
        return load_source(
            f"daily:{day.isoformat()}", "daily",
            build_zip_path(output_dir, symbol, interval, day),
            build_daily_url(market, symbol, interval, day),
        )

    LOGGER.info("=" * 80)
    LOGGER.info("Binance Vision monthly-first kline downloader")
    LOGGER.info(f"Market       : {market}")
    LOGGER.info(f"Symbol       : {symbol}")
    LOGGER.info(f"Interval     : {interval}")
    LOGGER.info(f"Scan range   : {start} -> {end} UTC")
    LOGGER.info(f"Output dir   : {output_dir}")
    LOGGER.info(f"Monthly cache: {monthly_zip_dir}")
    LOGGER.info(f"Daily cache  : {daily_zip_dir}")
    LOGGER.info(f"Info file    : {info_path}")
    LOGGER.info(f"Output CSV   : {out_path}")
    LOGGER.info("=" * 80)

    all_rows: list[list[str]] = []
    unavailable_dates: list[date] = []
    month_starts = list(iter_month_starts(start, end))

    # Inventory file names before opening archives or requesting missing files.
    cached_paths = {
        path for directory in (daily_zip_dir, monthly_zip_dir)
        for path in directory.iterdir() if path.is_file()
    }

    def has_cache(path: Path) -> bool:
        return path in cached_paths or path.with_suffix(path.suffix + ".temp") in cached_paths

    month_plans: list[tuple[date, date, date, list[date], bool]] = []
    planned_paths: list[Path] = []
    for month in month_starts:
        first_day, last_day = month_bounds(month)
        segment_start = max(start, first_day)
        segment_end = min(end, last_day)
        daily_dates = list(date_range(segment_start, segment_end))
        monthly_path = build_monthly_zip_path(output_dir, symbol, interval, month)
        monthly_cached = has_cache(monthly_path)
        daily_cached = any(
            has_cache(build_zip_path(output_dir, symbol, interval, day))
            for day in daily_dates
        )
        complete_month = segment_start == first_day and segment_end == last_day
        use_monthly = monthly_cached or (complete_month and not daily_cached)
        month_plans.append((month, segment_start, segment_end, daily_dates, use_monthly))
        if use_monthly:
            planned_paths.append(monthly_path)
        else:
            planned_paths.extend(
                build_zip_path(output_dir, symbol, interval, day)
                for day in daily_dates
            )

    cached_count = sum(has_cache(path) for path in planned_paths)
    LOGGER.info(
        f"Local ZIP inventory: {cached_count} present, "
        f"{len(planned_paths) - cached_count} missing"
    )

    for number, (month, segment_start, segment_end, daily_dates, use_monthly) in enumerate(
        month_plans, start=1
    ):
        first_day, last_day = month_bounds(month)
        complete_month = segment_start == first_day and segment_end == last_day
        LOGGER.info(
            f"[{number}/{len(month_starts)}] {month_key(month)}: "
            f"{'monthly preferred' if complete_month else 'partial month, daily preferred'}"
        )

        monthly_base_used = False
        # Reuse monthly coverage even for partial ranges. If only daily files
        # exist, continue that download instead of fetching an overlapping month.
        if use_monthly:
            monthly_rows, monthly_error = load_monthly(month)
            if monthly_rows is not None:
                monthly_segment = filter_rows_by_dates(
                    monthly_rows, segment_start, segment_end
                )
                incomplete_dates = find_incomplete_intraday_dates(
                    monthly_segment,
                    segment_start,
                    segment_end,
                    interval_ms,
                )
                all_rows.extend(monthly_segment)
                if not incomplete_dates:
                    continue

                monthly_base_used = True
                daily_dates = incomplete_dates
                counters["monthly_incomplete"] += 1
                counters["monthly_gap_dates"] += len(incomplete_dates)
                monthly_source_key = f"monthly:{month_key(month)}"
                info["source_files"][monthly_source_key]["coverage_status"] = (
                    "incomplete"
                )
                info["source_files"][monthly_source_key]["incomplete_dates"] = [
                    item.isoformat() for item in incomplete_dates
                ]
                write_download_info(info_path, info)
                LOGGER.warning(
                    f"  monthly ZIP has {len(incomplete_dates)} incomplete date(s); "
                    "recovering those dates from daily ZIPs"
                )
            if monthly_error != "missing":
                if monthly_rows is None:
                    raise RuntimeError(
                        f"Failed to load monthly source {month_key(month)}: {monthly_error}"
                    )
            elif monthly_rows is None:
                LOGGER.warning(f"[{symbol} {interval}] Monthly ZIP unavailable; falling back to daily ZIPs")

        daily_rows: list[list[str]] = []
        missing_days: list[date] = []
        recovered_monthly_gap_dates = 0
        for day in daily_dates:
            rows, error = load_daily(day)
            if rows is not None:
                daily_rows.extend(rows)
                if monthly_base_used:
                    recovered_monthly_gap_dates += 1
            elif error == "missing":
                missing_days.append(day)
                if args.show_missing:
                    LOGGER.warning(f"[{symbol} {interval}] {day}: daily ZIP missing")
            else:
                raise RuntimeError(f"Failed to load daily source {day}: {error}")

        all_rows.extend(filter_rows_by_dates(daily_rows, segment_start, segment_end))
        counters["monthly_gap_dates_recovered_from_daily"] += (
            recovered_monthly_gap_dates
        )
        if not missing_days:
            continue

        if monthly_base_used:
            # The monthly source was already included and known to be incomplete.
            # If its repair daily file is also absent, no further source exists.
            unavailable_dates.extend(missing_days)
            continue

        # Missing daily objects may still be present in Binance's monthly archive.
        monthly_rows, monthly_error = load_monthly(month)
        if monthly_rows is None:
            if monthly_error != "missing":
                raise RuntimeError(
                    f"Failed monthly fallback for {month_key(month)}: {monthly_error}"
                )
            unavailable_dates.extend(missing_days)
            continue

        for missing_day in missing_days:
            replacement = filter_rows_by_dates(monthly_rows, missing_day, missing_day)
            if replacement:
                all_rows.extend(replacement)
                source_key = f"daily:{missing_day.isoformat()}"
                info["source_files"][source_key]["status"] = "filled_from_monthly"
                info["source_files"][source_key]["daily_archive_status"] = "missing"
                info["source_files"][source_key]["fallback_source"] = (
                    f"monthly:{month_key(month)}"
                )
            else:
                unavailable_dates.append(missing_day)
        write_download_info(info_path, info)

    if counters["failed"]:
        raise RuntimeError(f"Source download/read failures={counters['failed']}")
    unavailable_dates = sorted(set(unavailable_dates))
    info["unavailable_dates"] = [day.isoformat() for day in unavailable_dates]
    if unavailable_dates:
        LOGGER.warning(
            "[%s %s] Skipping %s date(s): daily archives returned HTTP 404 "
            "and monthly archives contain no replacement data: %s",
            symbol,
            interval,
            len(unavailable_dates),
            ", ".join(info["unavailable_dates"]),
        )

    # Fallback sources can overlap; validate the merged available candles.
    merged_rows, duplicate_rows = merge_rows_by_open_time(all_rows)
    filtered_rows = filter_rows_by_dates(merged_rows, start, end)
    if not filtered_rows:
        info.update({
            "final_rows": 0,
            "last_updated_utc": utc_now_iso(),
            "summary": {**counters, "unavailable_dates": len(unavailable_dates), "complete": False},
        })
        write_download_info(info_path, info)
        LOGGER.warning("[%s %s] No available data in the requested range; skipping CSV generation", symbol, interval)
        return None
    validation_df = pd.DataFrame(filtered_rows, columns=OUTPUT_COLUMNS)
    validation = common.validate_kline_source(
        validation_df,
        interval_ms,
        source=str(out_path),
        allowed_missing_dates=info["unavailable_dates"],
    )

    first_candle_ms = int(filtered_rows[0][0])
    last_candle_ms = int(filtered_rows[-1][0])
    first_candle_date = datetime.fromtimestamp(
        first_candle_ms / 1000, tz=timezone.utc
    ).date()
    last_candle_date = datetime.fromtimestamp(
        last_candle_ms / 1000, tz=timezone.utc
    ).date()
    write_output_csv(out_path, filtered_rows)
    earliest_asset_date = None
    if earliest_date_confirmed:
        earliest_asset_date = record_earliest_date(
            earliest_dates_path,
            market,
            symbol,
            first_candle_date,
        )

    recovered_daily = sum(
        entry["status"] == "filled_from_monthly"
        for entry in info["source_files"].values()
    )

    info.update({
        "market_data_path": str(market_data_path),
        "dataset_start_date": first_candle_date.isoformat(),
        "dataset_end_date": last_candle_date.isoformat(),
        "raw_rows": len(all_rows),
        "duplicate_rows": duplicate_rows,
        "final_rows": len(filtered_rows),
        "first_candle_open_time_ms_utc": first_candle_ms,
        "first_candle_open_time_utc": ms_to_dt(first_candle_ms),
        "last_candle_open_time_ms_utc": last_candle_ms,
        "last_candle_open_time_utc": ms_to_dt(last_candle_ms),
        "last_updated_utc": utc_now_iso(),
        "summary": {
            **counters,
            "daily_recovered_from_monthly": recovered_daily,
            "unavailable_dates": len(unavailable_dates),
            "allowed_gap_count": validation["allowed_gap_count"],
            "missing_candle_count": validation["missing_candle_count"],
            "complete": not unavailable_dates and validation["missing_candle_count"] == 0,
        },
    })
    if earliest_asset_date is not None:
        info["asset_earliest_date"] = earliest_asset_date.isoformat()
    write_download_info(info_path, info)

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("Download summary")
    LOGGER.info(f"Validated candle range : {first_candle_date} -> {last_candle_date} UTC")
    LOGGER.info(f"Monthly ZIPs cached    : {counters['monthly_cached']}")
    LOGGER.info(f"Monthly ZIPs downloaded: {counters['monthly_downloaded']}")
    LOGGER.info(f"Monthly ZIPs incomplete: {counters['monthly_incomplete']}")
    LOGGER.info(
        "Monthly gap dates fixed: "
        f"{counters['monthly_gap_dates_recovered_from_daily']}"
        f"/{counters['monthly_gap_dates']}"
    )
    LOGGER.info(f"Daily ZIPs cached      : {counters['daily_cached']}")
    LOGGER.info(f"Daily ZIPs downloaded  : {counters['daily_downloaded']}")
    LOGGER.info(f"Daily ZIPs recovered   : {recovered_daily}")
    LOGGER.info(f"Raw rows               : {len(all_rows)}")
    LOGGER.info(f"Duplicate rows         : {duplicate_rows}")
    LOGGER.info(f"Final rows             : {len(filtered_rows)}")
    LOGGER.info(f"Missing candles        : {validation['missing_candle_count']}")
    LOGGER.info(f"Overlapping candles    : {validation['overlap_count']}")
    LOGGER.info(f"Info file              : {info_path}")

    market_data_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_target = market_data_path.with_suffix(market_data_path.suffix + ".temp")
    shutil.copyfile(out_path, temporary_target)
    os.replace(temporary_target, market_data_path)
    out_path.unlink()
    LOGGER.info(f"Output CSV             : {market_data_path}")
    LOGGER.info("=" * 80)
    LOGGER.info("K-line source validation passed.")
    return market_data_path


def read_funding_rows_from_zip(zip_bytes: bytes, source_name: str) -> list[list[str]]:
    rows: list[list[str]] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV file inside ZIP: {source_name}")
        with zf.open(csv_names[0], "r") as f:
            for row in csv.reader(io.TextIOWrapper(f, encoding="utf-8")):
                if not row or not row[0].strip().lstrip("-").isdigit():
                    continue  # header or blank line
                if len(row) < 3:
                    raise ValueError(f"Bad funding row in {source_name}: {row}")
                calc_ms = normalize_ts_to_ms(row[0])
                rows.append([str(calc_ms), ms_to_dt(calc_ms), row[1], row[2]])
    return rows


def request_funding_rate_page(
    session: requests.Session,
    limiter: FundingApiLimiter,
    url: str,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    for attempt in range(HTTP_RETRIES + 1):
        limiter.wait()
        rate_limited = False
        try:
            response = session.get(url, params=params, timeout=HTTP_TIMEOUT_SECONDS)
            if response.status_code in {418, 429}:
                delay = limiter.defer(response.headers.get("Retry-After"))
                rate_limited = True
                LOGGER.warning(
                    f"[{params['symbol']} fundingRate] API returned {response.status_code}; "
                    f"cooldown: {delay:.1f}s"
                )
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt == HTTP_RETRIES:
                raise
            if not rate_limited:
                time.sleep(RETRY_DELAY_SECONDS)
    raise RuntimeError("Funding rate API retries exhausted")


def download_funding_rate(
    args: argparse.Namespace,
    session: requests.Session,
    symbol: str,
    limiter: FundingApiLimiter,
) -> Optional[Path]:
    """Download monthly fundingRate ZIPs (um/cm only) and write one merged CSV."""
    market = args.market
    if market not in {"um", "cm"}:
        return None
    symbol = symbol.upper()
    para = common.MarketDataSourceConfig(
        market_category="Cryptocurrency",
        data_source="binance_public_data",
        symbols=(symbol,),
        interval="fundingRate",
        trading_type=market,
    )
    target = Path(common.market_data_path(para)).expanduser().resolve()
    today = datetime.now(timezone.utc).date()
    end = parse_date(args.end) or (today - timedelta(days=1))
    start = parse_date(args.start) or FALLBACK_START[market]
    if target.is_file() and not args.force:
        try:
            last_ms = int(pd.read_csv(target, usecols=[0]).iloc[-1, 0])
            last_date = datetime.fromtimestamp(last_ms / 1000, tz=timezone.utc).date()
        except (OSError, ValueError, IndexError, KeyError):
            last_date = None
        if last_date is not None and last_date >= end - timedelta(days=1):
            LOGGER.info(f"[{symbol} fundingRate] Skipping up-to-date CSV: {target}")
            return target

    zip_dir = Path(args.out_dir).expanduser().resolve() / market / symbol / "fundingRate"
    base = f"{BASE_URL}/{MARKET_ROOTS[market]}/monthly/fundingRate/{symbol}"

    all_rows: list[list[str]] = []
    missing: list[str] = []
    for month in iter_month_starts(start, end):
        name = f"{symbol}-fundingRate-{month:%Y-%m}.zip"
        path = zip_dir / name
        content: Optional[bytes] = None
        if path.is_file() and not args.force:
            try:
                content = path.read_bytes()
                read_funding_rows_from_zip(content, name)
            except Exception:
                content = None
        if content is None:
            LOGGER.info(f"[{symbol} fundingRate] Downloading monthly ZIP: {name}")
            for attempt in range(HTTP_RETRIES + 1):
                try:
                    response = session.get(
                        f"{base}/{name}",
                        timeout=HTTP_TIMEOUT_SECONDS,
                    )
                    if response.status_code == 404:
                        break
                    response.raise_for_status()
                    read_funding_rows_from_zip(response.content, name)
                    content = response.content
                    write_zip_cache(path, content)
                    LOGGER.info(f"[{symbol} fundingRate] Downloaded monthly ZIP: {name} ({len(content)} bytes)")
                    break
                except Exception as exc:
                    if attempt == HTTP_RETRIES:
                        raise RuntimeError(f"Funding rate download failed for {name}: {exc}") from exc
                    time.sleep(RETRY_DELAY_SECONDS)
                finally:
                    time.sleep(DOWNLOAD_DELAY_SECONDS)
        if content is None:
            missing.append(f"{month:%Y-%m}")
            continue
        all_rows.extend(read_funding_rows_from_zip(content, name))

    # Vision only publishes monthly funding archives after the month ends,
    # so fill the remaining recent period from the REST API.
    all_rows.sort(key=lambda r: int(r[0]))
    last_interval = all_rows[-1][2] if all_rows else "8"
    start_ms = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp() * 1000)
    end_next = end + timedelta(days=1)
    end_ms = int(datetime(end_next.year, end_next.month, end_next.day, tzinfo=timezone.utc).timestamp() * 1000)
    cursor = max(start_ms, int(all_rows[-1][0]) + 1) if all_rows else start_ms
    api_url = (
        "https://fapi.binance.com/fapi/v1/fundingRate"
        if market == "um"
        else "https://dapi.binance.com/dapi/v1/fundingRate"
    )
    api_rows = 0
    while cursor < end_ms:
        try:
            data = request_funding_rate_page(
                session,
                limiter,
                api_url,
                {"symbol": symbol, "startTime": cursor, "endTime": end_ms - 1, "limit": 1000},
            )
        except Exception as exc:
            raise RuntimeError(f"Funding rate API failed for {symbol}: {exc}") from exc
        if not data:
            break
        for item in data:
            t = int(item["fundingTime"])
            all_rows.append([str(t), ms_to_dt(t), last_interval, str(item["fundingRate"])])
        api_rows += len(data)
        cursor = int(data[-1]["fundingTime"]) + 1
        if len(data) < 1000:
            break
    if api_rows:
        LOGGER.info(f"Funding rate rows from REST API: {api_rows}")

    if not all_rows:
        raise RuntimeError(f"No funding rate rows found for {symbol} ({market}).")

    by_time = {int(r[0]): r for r in all_rows}
    rows = [by_time[k] for k in sorted(by_time)]
    rows = filter_rows_by_dates(rows, start, end)
    if missing:
        LOGGER.warning(f"Funding rate monthly archives not available (tail filled by API): {', '.join(missing)}")

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".temp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "calc_time_ms_utc", "calc_time_date_utc",
            "funding_interval_hours", "last_funding_rate",
        ])
        writer.writerows(rows)
    os.replace(tmp, target)
    LOGGER.info(f"Funding rate CSV       : {target} ({len(rows)} rows)")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Binance Vision kline or aggTrades ZIP files and merge them."
    )
    parser.add_argument("--market", choices=VALID_MARKETS.keys(), default="um",
                        help="Market: spot, um=USDⓈ-M futures, cm=COIN-M futures. Default: um")
    parser.add_argument(
        "--data-type",
        choices=("klines", "aggTrades"),
        default="klines",
        help="Dataset to download. Default: klines.",
    )
    parser.add_argument(
        "--symbols",
        "--symbol",
        nargs="+",
        default=["DOGEUSDT","BTCUSDT","ETHUSDT","SOLUSDT","ZECUSDT","XRPUSDT","NEARUSDT","BNBUSDT","DOGEUSDT","UNIUSDT","AVAXUSDT","QNTUSDT","ADAUSDT","LINKUSDT","LTCUSDT","TRXUSDT","HBARUSDT","AAVEUSDT","FETUSDT","XLMUSDT","BCHUSDT","FILUSDT","INJUSDT","PAXGUSDT","DASHUSDT","DOTUSDT","ONEUSDT","VTHOUSDT","ICPUSDT","SANDUSDT","MOVRUSDT","RAYUSDT","RLCUSDT","CAKEUSDT","ALGOUSDT","ARUSDT","LSKUSDT","GTCUSDT","PHAUSDT","OGNUSDT","CRVUSDT","SHIBUSDT","RUNEUSDT"],#"ETHUSDT",
        help="One or more symbols, e.g. --symbols BTCUSDT DOGEUSDT",
    )
    parser.add_argument(
        "--intervals",
        "--interval",
        nargs="+",
        default=["1m","5m","15m"],#,"30m","1h","2h","4h"
        help="One or more intervals, e.g. --intervals 15m 30m 1h",
    )
    parser.add_argument(
        "--start",
        default=None,
        help=(
            "Fallback start day UTC, YYYY-MM-DD. A cached earliest date for "
            "market+symbol takes precedence; otherwise the earliest listed "
            "Binance Vision date takes precedence."
        ),
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End day UTC, YYYY-MM-DD. Default: latest completed UTC day (yesterday).",
    )
    parser.add_argument(
        "--skip-start-date-discovery",
        action="store_true",
        help=(
            "For aggTrades, skip the Binance Vision S3 earliest-date lookup and "
            "infer the start from local ZIP caches (market fallback if none exist)."
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Use only local ZIP caches, skip Binance requests and earliest-date discovery, "
            "and mark unavailable archives in the download audit."
        ),
    )
    parser.add_argument(
        "--out-dir",
        "--dir",
        dest="out_dir",
        default=str(Path(__file__).resolve().parent / "data"),
        help="ZIP cache and audit directory. Resume from existing ZIPs under <dir>/<market>/<symbol>/<interval-or-data-type>/. K-lines use CSV; aggTrades use yearly Parquet files.",
    )
    parser.add_argument("--show-missing", action="store_true", help="Print every missing ZIP URL/date, not only summary.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the final CSV and recheck previously missing archives.",
    )
    parser.add_argument(
        "--no-funding-rate",
        action="store_true",
        help="Do not download funding rates (downloaded by default for um/cm markets).",
    )
    args = parser.parse_args()

    _, log_file_path = common.setup_session_logger(sub_folder="data_process/binance_download")
    LOGGER.info("Download log file: %s", log_file_path)

    requested_symbols = list(dict.fromkeys(symbol.upper() for symbol in args.symbols))
    symbol_aliases = SYMBOL_ALIASES.get(args.market, {})
    symbols = list(dict.fromkeys(
        symbol_aliases.get(symbol, symbol) for symbol in requested_symbols
    ))
    for requested_symbol in requested_symbols:
        mapped_symbol = symbol_aliases.get(requested_symbol)
        if mapped_symbol is not None:
            LOGGER.info(
                "Mapping %s %s symbol to Binance contract symbol %s",
                args.market,
                requested_symbol,
                mapped_symbol,
            )
    intervals = list(dict.fromkeys(args.intervals))
    total_tasks = (
        len(symbols)
        if args.data_type == "aggTrades"
        else len(symbols) * len(intervals)
    )

    succeeded = 0
    failures: list[tuple[str, str, str]] = []
    funding_succeeded = 0
    funding_failures: list[tuple[str, str]] = []
    with make_session() as session:
        if args.data_type == "aggTrades":
            for task_number, symbol in enumerate(symbols, start=1):
                LOGGER.info(f"\n[Task {task_number}/{total_tasks}] {symbol} aggTrades")
                try:
                    download_symbol_agg_trades(args, session, symbol)
                    succeeded += 1
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    failures.append((symbol, "aggTrades", reason))
                    LOGGER.error(f"Failed {symbol} aggTrades: {reason}")
        else:
            funding_api_limiter = FundingApiLimiter()
            task_number = 0
            for symbol in symbols:
                no_kline_archives = False
                for interval in intervals:
                    task_number += 1
                    LOGGER.info(f"\n[Task {task_number}/{total_tasks}] {symbol} {interval}")
                    if no_kline_archives:
                        reason = "No daily archives found for this market symbol; remaining intervals skipped"
                        failures.append((symbol, interval, reason))
                        LOGGER.error(f"Failed {symbol} {interval}: {reason}")
                        continue
                    try:
                        download_symbol_interval(args, session, symbol, interval)
                        succeeded += 1
                    except Exception as exc:
                        reason = f"{type(exc).__name__}: {exc}"
                        failures.append((symbol, interval, reason))
                        LOGGER.error(f"Failed {symbol} {interval}: {reason}")
                        no_kline_archives = isinstance(exc, NoArchiveError)
                if no_kline_archives:
                    if args.market in {"um", "cm"} and not args.no_funding_rate:
                        reason = "No K-line archives found for this market symbol; funding download skipped"
                        funding_failures.append((symbol, reason))
                        LOGGER.error(f"Failed {symbol} fundingRate: {reason}")
                elif args.market in {"um", "cm"} and not args.no_funding_rate:
                    try:
                        download_funding_rate(args, session, symbol, funding_api_limiter)
                        funding_succeeded += 1
                    except Exception as exc:
                        reason = f"{type(exc).__name__}: {exc}"
                        funding_failures.append((symbol, reason))
                        LOGGER.error(f"Failed {symbol} fundingRate: {reason}")

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("All download tasks finished")
    task_label = "aggTrades" if args.data_type == "aggTrades" else "K-line"
    LOGGER.info(f"{task_label} tasks: total={total_tasks}, succeeded={succeeded}, failed={len(failures)}")
    LOGGER.info("Existing output CSVs reused without --force count as succeeded.")
    for symbol, interval, reason in failures:
        LOGGER.error(f"  Failed {symbol} {interval}: {reason}")
    if args.data_type == "klines" and args.market in {"um", "cm"} and not args.no_funding_rate:
        LOGGER.info(
            f"Funding rate tasks: total={len(symbols)}, "
            f"succeeded={funding_succeeded}, failed={len(funding_failures)}"
        )
        for symbol, reason in funding_failures:
            LOGGER.error(f"  Failed {symbol} fundingRate: {reason}")
    LOGGER.info("=" * 80)
    return 1 if failures or funding_failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user.")
        raise SystemExit(130)
