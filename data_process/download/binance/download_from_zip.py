#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download Binance Vision kline ZIP files by date range.

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
    - Treat local ZIP files as the cache. download_info.json is a human-readable
      audit report only and is never used as downloader input.
    - Resume automatically from valid cached archives, downloading only missing
      sources. Invalid cached archives are downloaded again.
    - Skip a series when its final CSV already exists, unless --force is used.
    - Infer --start from the first daily ZIP in the sorted S3 XML listing,
      without scanning the full history; gaps remain eligible for monthly recovery.
    - Aggregate the selected sources and strictly validate the final K-line sequence.

Examples:
    # Download multiple symbols and intervals (Cartesian product)
    python download_from_zip.py --symbols DOGEUSDT BTCUSDT --intervals 15m 30m

    # Spot DOGEUSDT 30m, explicit range
    python download_from_zip.py --symbols DOGEUSDT --intervals 30m --start 2021-01-01 --end 2021-12-31

    # Binance USDⓈ-M futures / UM futures
    python download_from_zip.py --market um --symbols DOGEUSDT --intervals 30m

Output:
    <out_dir>/<market>/<symbol>/<interval>/
        download_info.json
        zips/<symbol>-<interval>-YYYY-MM-DD.zip
        monthly_zips/<symbol>-<interval>-YYYY-MM.zip
    <out_dir>/earliest_dates.json stores earliest dates for all market/symbols.

    The validated CSV is moved to common.market_data_path(para).

    For um/cm markets, monthly fundingRate ZIPs are also downloaded (once per
    symbol, cached under <out_dir>/<market>/<symbol>/fundingRate/) and merged to
    <market_data dir>/<symbol>_fundingRate.csv. Use --no-funding-rate to skip.
"""

import argparse
import calendar
import csv
import io
import json
import os
import re
import shutil
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import requests
import pandas as pd
from requests.adapters import HTTPAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_process import common


BASE_URL = "https://data.binance.vision"
S3_LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
HTTP_TIMEOUT_SECONDS = 30
HTTP_RETRIES = 3
RETRY_DELAY_SECONDS = 1
DOWNLOAD_DELAY_SECONDS = 0.3
INFO_FILENAME = "download_info.json"
EARLIEST_DATES_FILENAME = "earliest_dates.json"
ZIP_DIRNAME = "zips"
MONTHLY_ZIP_DIRNAME = "monthly_zips"

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
    """Read object keys from a single S3 XML listing response."""
    for attempt in range(HTTP_RETRIES + 1):
        try:
            response = session.get(
                S3_LIST_URL,
                params={"prefix": prefix},
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
    return [
        element.text
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == "Key" and element.text
    ]


def infer_available_dates_from_listing(
    session: requests.Session,
    market: str,
    symbol: str,
    interval: str,
) -> list[date]:
    """Find the earliest daily ZIP from the lexicographically sorted S3 XML keys."""
    prefix = build_archive_prefix(market, "daily", symbol, interval)
    print("Querying Binance Vision S3 listing for the earliest daily ZIP:")
    print(f"   {S3_LIST_URL}?prefix={prefix}")
    pattern = re.compile(
        rf"{re.escape(prefix)}{re.escape(symbol)}-{re.escape(interval)}-"
        rf"(\d{{4}}-\d{{2}}-\d{{2}})\.zip",
    )
    # ISO dates sort chronologically; checksum objects and directory keys are ignored.
    for key in list_s3_keys(session, prefix):
        match = pattern.fullmatch(key)
        if match is not None:
            earliest = datetime.strptime(match.group(1), "%Y-%m-%d").date()
            print(f"Earliest daily ZIP: {earliest}")
            return [earliest]

    print("S3 listing contains no daily ZIP filename.")
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
            print(
                f"ℹ️ Ignoring --start={requested_start}; using recorded earliest "
                f"date for {market}:{symbol}: {cached_start}"
            )
        else:
            print(f"✅ Using cached earliest date for {market}:{symbol}: {cached_start}")
        return cached_start, effective_end, True

    dates = infer_available_dates_from_listing(session, market, symbol, interval)

    if dates:
        effective_end = requested_end
        effective_start = dates[0]
        if effective_start > effective_end:
            raise ValueError(
                "No available ZIP files in requested date range: "
                f"earliest_available={effective_start}, requested_end={requested_end}"
            )
        if requested_start is not None and requested_start != effective_start:
            print(
                f"ℹ️ Ignoring --start={requested_start}; using earliest listed "
                f"date for {market}:{symbol}:{interval}: {effective_start}"
            )
        else:
            print(f"✅ Using earliest available file date: {effective_start}")
        return effective_start, effective_end, True

    fallback = requested_start or FALLBACK_START[market]
    effective_start = fallback
    print(
        "⚠️ Failed to infer earliest date from listing. "
        f"Using fallback start date {effective_start}."
    )
    if effective_start > requested_end:
        raise ValueError(
            f"start must be <= end, got start={effective_start}, end={requested_end}"
        )
    return effective_start, requested_end, False


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
) -> Path:
    """Download one series with monthly-first archives and strict validation."""
    market = args.market
    symbol = symbol.upper()
    interval_ms = interval_to_ms(interval)
    para = common.MarketDataSourceConfig(
        market_category="Cryptocurrency",
        data_source="binance_public_data",
        symbol=symbol,
        interval=interval,
        trading_type=market,
    )
    market_data_path = Path(common.market_data_path(para)).expanduser().resolve()
    if market_data_path.is_file() and not args.force:
        print(f"Skipping existing output CSV: {market_data_path}")
        return market_data_path

    output_dir = build_output_dir(
        Path(args.out_dir).expanduser().resolve(), market, symbol, interval
    )
    daily_zip_dir = output_dir / ZIP_DIRNAME
    monthly_zip_dir = output_dir / MONTHLY_ZIP_DIRNAME
    out_path = build_csv_path(output_dir, symbol, interval)
    info_path = build_info_path(output_dir)
    earliest_dates_path = build_earliest_dates_path(
        Path(args.out_dir).expanduser().resolve()
    )
    cached_start = parse_date(
        load_earliest_dates(earliest_dates_path).get(f"{market}:{symbol}")
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
                print(f"Ignoring invalid cache {candidate}: {type(exc).__name__}: {exc}")
                continue
            if candidate == temporary_path:
                os.replace(temporary_path, path)
            counters[f"{frequency}_cached"] += 1
            record_source(key, frequency=frequency, status="cached", path=path, url=url)
            return rows, None

        content, error = download_zip_url(session, url)
        if content is not None:
            write_zip_cache(path, content)
            counters[f"{frequency}_downloaded"] += 1
            record_source(key, frequency=frequency, status="downloaded", path=path, url=url)
            print(f"{key}: downloaded ZIP {len(content)} bytes")
            return read_kline_rows_from_zip(content, url), None
        if error == "missing":
            counters[f"{frequency}_missing"] += 1
            record_source(key, frequency=frequency, status="missing", path=path, url=url)
            return None, "missing"
        counters["failed"] += 1
        record_source(
            key, frequency=frequency, status="failed", path=path, url=url, error=error
        )
        raise RuntimeError(f"Failed to download source {url}: {error}")

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

    print("=" * 80)
    print("Binance Vision monthly-first kline downloader")
    print(f"Market       : {market}")
    print(f"Symbol       : {symbol}")
    print(f"Interval     : {interval}")
    print(f"Scan range   : {start} -> {end} UTC")
    print(f"Output dir   : {output_dir}")
    print(f"Monthly cache: {monthly_zip_dir}")
    print(f"Daily cache  : {daily_zip_dir}")
    print(f"Info file    : {info_path}")
    print(f"Output CSV   : {out_path}")
    print("=" * 80)

    all_rows: list[list[str]] = []
    unresolved_dates: list[date] = []
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
    print(
        f"Local ZIP inventory: {cached_count} present, "
        f"{len(planned_paths) - cached_count} missing"
    )

    for number, (month, segment_start, segment_end, daily_dates, use_monthly) in enumerate(
        month_plans, start=1
    ):
        first_day, last_day = month_bounds(month)
        complete_month = segment_start == first_day and segment_end == last_day
        print(
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
                print(
                    f"  monthly ZIP has {len(incomplete_dates)} incomplete date(s); "
                    "recovering those dates from daily ZIPs"
                )
            if monthly_error != "missing":
                if monthly_rows is None:
                    raise RuntimeError(
                        f"Failed to load monthly source {month_key(month)}: {monthly_error}"
                    )
            elif monthly_rows is None:
                print("  monthly ZIP unavailable; falling back to daily ZIPs")

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
                    print(f"  {day}: daily ZIP missing")
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
            unresolved_dates.extend(missing_days)
            continue

        # Missing daily objects may still be present in Binance's monthly archive.
        monthly_rows, monthly_error = load_monthly(month)
        if monthly_rows is None:
            if monthly_error != "missing":
                raise RuntimeError(
                    f"Failed monthly fallback for {month_key(month)}: {monthly_error}"
                )
            unresolved_dates.extend(missing_days)
            continue

        for missing_day in missing_days:
            replacement = filter_rows_by_dates(monthly_rows, missing_day, missing_day)
            if replacement:
                all_rows.extend(replacement)
                source_key = f"daily:{missing_day.isoformat()}"
                info["source_files"][source_key]["status"] = "filled_from_monthly"
                info["source_files"][source_key]["fallback_source"] = (
                    f"monthly:{month_key(month)}"
                )
            else:
                unresolved_dates.append(missing_day)
        write_download_info(info_path, info)

    if counters["failed"]:
        raise RuntimeError(f"Source download/read failures={counters['failed']}")
    # Binance publishes daily archives with a delay, so when downloading up to
    # the present, a missing final day is expected and ignored.
    if end >= datetime.now(timezone.utc).date() - timedelta(days=1):
        unresolved_dates = [day for day in unresolved_dates if day != end]
    if unresolved_dates:
        dates_text = ", ".join(day.isoformat() for day in unresolved_dates[:10])
        raise RuntimeError(
            "Refusing to write an incomplete K-line source: daily and monthly "
            f"archives are both missing for {len(unresolved_dates)} date(s): {dates_text}"
        )
    if not all_rows:
        raise RuntimeError("No kline rows were found after aggregation.")

    # Fallback sources can overlap. The strict check applies to the merged
    # candle sequence rather than requiring every daily object to exist.
    merged_rows, duplicate_rows = merge_rows_by_open_time(all_rows)
    filtered_rows = filter_rows_by_dates(merged_rows, start, end)
    if not filtered_rows:
        raise RuntimeError("No kline rows remain inside the requested date range.")
    validation_df = pd.DataFrame(filtered_rows, columns=OUTPUT_COLUMNS)
    validation = common.validate_kline_source(
        validation_df,
        interval_ms,
        source=str(out_path),
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
            "unresolved_dates": len(unresolved_dates),
            "complete": True,
        },
    })
    if earliest_asset_date is not None:
        info["asset_earliest_date"] = earliest_asset_date.isoformat()
    write_download_info(info_path, info)

    print("\n" + "=" * 80)
    print("Download summary")
    print(f"Validated candle range : {first_candle_date} -> {last_candle_date} UTC")
    print(f"Monthly ZIPs cached    : {counters['monthly_cached']}")
    print(f"Monthly ZIPs downloaded: {counters['monthly_downloaded']}")
    print(f"Monthly ZIPs incomplete: {counters['monthly_incomplete']}")
    print(
        "Monthly gap dates fixed: "
        f"{counters['monthly_gap_dates_recovered_from_daily']}"
        f"/{counters['monthly_gap_dates']}"
    )
    print(f"Daily ZIPs cached      : {counters['daily_cached']}")
    print(f"Daily ZIPs downloaded  : {counters['daily_downloaded']}")
    print(f"Daily ZIPs recovered   : {recovered_daily}")
    print(f"Raw rows               : {len(all_rows)}")
    print(f"Duplicate rows         : {duplicate_rows}")
    print(f"Final rows             : {len(filtered_rows)}")
    print(f"Missing candles        : {validation['missing_candle_count']}")
    print(f"Overlapping candles    : {validation['overlap_count']}")
    print(f"Info file              : {info_path}")

    market_data_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_target = market_data_path.with_suffix(market_data_path.suffix + ".temp")
    shutil.copyfile(out_path, temporary_target)
    os.replace(temporary_target, market_data_path)
    out_path.unlink()
    print(f"Output CSV             : {market_data_path}")
    print("=" * 80)
    print("✅ K-line source validation passed.")
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


def download_funding_rate(
    args: argparse.Namespace,
    session: requests.Session,
    symbol: str,
) -> Optional[Path]:
    """Download monthly fundingRate ZIPs (um/cm only) and write one merged CSV."""
    market = args.market
    if market not in {"um", "cm"}:
        return None
    symbol = symbol.upper()
    para = common.MarketDataSourceConfig(
        market_category="Cryptocurrency",
        data_source="binance_public_data",
        symbol=symbol,
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
        except Exception:
            last_date = None
        # Funding settles every 1-8h, so the last row may be a day before `end`.
        if last_date is not None and last_date >= end - timedelta(days=1):
            print(f"Skipping existing funding rate CSV: {target}")
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
        data = None
        for attempt in range(HTTP_RETRIES + 1):
            try:
                response = session.get(
                    api_url,
                    params={"symbol": symbol, "startTime": cursor, "endTime": end_ms - 1, "limit": 1000},
                    timeout=HTTP_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                data = response.json()
                break
            except Exception as exc:
                if attempt == HTTP_RETRIES:
                    raise RuntimeError(f"Funding rate API failed for {symbol}: {exc}") from exc
                time.sleep(RETRY_DELAY_SECONDS)
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
        print(f"Funding rate rows from REST API: {api_rows}")

    if not all_rows:
        raise RuntimeError(f"No funding rate rows found for {symbol} ({market}).")

    by_time = {int(r[0]): r for r in all_rows}
    rows = [by_time[k] for k in sorted(by_time)]
    rows = filter_rows_by_dates(rows, start, end)
    if missing:
        print(f"ℹ️ Funding rate monthly archives not available (tail filled by API): {', '.join(missing)}")

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
    print(f"Funding rate CSV       : {target} ({len(rows)} rows)")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Binance Vision monthly-first kline ZIP files and merge them."
    )
    parser.add_argument("--market", choices=VALID_MARKETS.keys(), default="um",
                        help="Market: spot, um=USDⓈ-M futures, cm=COIN-M futures. Default: um")
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
        "--out-dir",
        "--dir",
        dest="out_dir",
        default=str(Path(__file__).resolve().parent / "data"),
        help="ZIP cache and audit directory. Resume automatically from existing ZIPs under <dir>/<market>/<symbol>/<interval>/. Validated CSV files are moved to common.market_data_path().",
    )
    parser.add_argument("--show-missing", action="store_true", help="Print every missing ZIP URL/date, not only summary.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the final CSV even if it already exists.",
    )
    parser.add_argument(
        "--no-funding-rate",
        action="store_true",
        help="Do not download funding rates (downloaded by default for um/cm markets).",
    )
    args = parser.parse_args()

    symbols = list(dict.fromkeys(symbol.upper() for symbol in args.symbols))
    intervals = list(dict.fromkeys(args.intervals))
    total_tasks = len(symbols) * len(intervals)

    succeeded = 0
    failures: list[tuple[str, str, str]] = []
    funding_succeeded = 0
    funding_failures: list[tuple[str, str]] = []
    with make_session() as session:
        task_number = 0
        for symbol in symbols:
            for interval in intervals:
                task_number += 1
                print(f"\n[Task {task_number}/{total_tasks}] {symbol} {interval}")
                try:
                    destination = download_symbol_interval(args, session, symbol, interval)
                    succeeded += 1
                    print(f"Completed {symbol} {interval}: {destination}")
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    failures.append((symbol, interval, reason))
                    print(f"Failed {symbol} {interval}: {reason}")
            if args.market in {"um", "cm"} and not args.no_funding_rate:
                try:
                    download_funding_rate(args, session, symbol)
                    funding_succeeded += 1
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    funding_failures.append((symbol, reason))
                    print(f"Failed {symbol} fundingRate: {reason}")

    print("\n" + "=" * 80)
    print("All download tasks finished")
    print(f"K-line tasks: total={total_tasks}, succeeded={succeeded}, failed={len(failures)}")
    print("Existing output CSVs reused without --force count as succeeded.")
    for symbol, interval, reason in failures:
        print(f"  Failed {symbol} {interval}: {reason}")
    if args.market in {"um", "cm"} and not args.no_funding_rate:
        print(
            f"Funding rate tasks: total={len(symbols)}, "
            f"succeeded={funding_succeeded}, failed={len(funding_failures)}"
        )
        for symbol, reason in funding_failures:
            print(f"  Failed {symbol} fundingRate: {reason}")
    print("=" * 80)
    return 1 if failures or funding_failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)
