"""Capture settled Bitget portfolio equity and publish daily performance."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import json
import logging
from math import isfinite, sqrt
import os
from pathlib import Path
import sqlite3
from statistics import mean, stdev
import subprocess
import sys
from tempfile import NamedTemporaryFile
import threading

import requests

from trade.recording.archive_bitget import BitgetHistory, configured_accounts, connect, sync_archive
from trade.recording.report_bitget import bill_checkpoints, complete_bill_dates, report
from trade.recording.strategy_curves import publish_strategy_curves

ZONE = timezone.utc
CAPTURE_WINDOW = timedelta(minutes=1)
DAY = timedelta(days=1)
REPORT_NAME = "equity_overview.json"
STRATEGY_START = date(2026, 9, 7)


def performance(valuations):
    if not valuations:
        return [], None

    first, last = valuations[0], valuations[-1]
    start, end = date.fromisoformat(first["date"]), date.fromisoformat(last["date"])
    days = (end - start).days
    rows = []
    returns = []
    peak, peak_day = first["value"], start
    previous = None
    longest = None
    for valuation in valuations:
        day = date.fromisoformat(valuation["date"])
        value = valuation["value"]
        if previous is not None and previous > 0:
            returns.append(value / previous - 1)
        previous = value

        recovered = value >= peak
        if value < peak or (rows and rows[-1]["drawdown"] < 0):
            duration = (day - peak_day).days
            if longest is None or duration >= longest["days"]:
                longest = {
                    "peak_date": str(peak_day), "end_date": str(day),
                    "days": duration, "recovered": recovered,
                }
        if recovered:
            peak, peak_day = value, day
        rows.append({**valuation, "drawdown": value / peak - 1 if peak else 0.0})

    total_return = annual_return = None
    if days and first["value"] > 0:
        growth = last["value"] / first["value"]
        total_return = growth - 1
        if growth >= 0:
            try:
                annual_return = growth ** (365 / days) - 1
            except OverflowError:
                pass
    drawdown = -min(row["drawdown"] for row in rows)
    volatility = stdev(returns) if len(returns) == days == len(rows) - 1 and days > 1 else 0
    calmar = annual_return / drawdown if drawdown and annual_return is not None else None
    return rows, {
        "start": str(start), "end": str(end), "days": days,
        "missing_days": days + 1 - len(rows),
        "total_return": total_return, "annualized_return": annual_return,
        "sharpe_zero_risk_free": mean(returns) * sqrt(365) / volatility if volatility else None,
        "daily_max_drawdown": drawdown,
        "calmar": calmar if calmar is not None and isfinite(calmar) else None,
        "longest_drawdown": longest,
    }


def fetch_equity(directory):
    with requests.Session() as session:
        result = BitgetHistory(session, directory).get("account/accounts")
    assets = [asset for asset in result["data"] if asset["marginCoin"] == "USDT"]
    if len(assets) != 1:
        raise ValueError("Expected one Bitget USDT equity record")
    equity = Decimal(assets[0]["accountEquity"])
    if not equity.is_finite() or equity < 0:
        raise ValueError("Invalid Bitget USDT equity")
    observed = datetime.fromtimestamp(int(result["requestTime"]) / 1000, ZONE)
    return observed, equity


def capture_equity(accounts, boundary):
    with ThreadPoolExecutor(max_workers=4) as pool:
        observations = list(pool.map(fetch_equity, accounts.values()))
    if not all(boundary <= observed < boundary + CAPTURE_WINDOW for observed, _ in observations):
        raise ValueError("Bitget equity observations fall outside the UTC midnight capture window")
    return {
        "valuation_at": boundary.isoformat(),
        "as_of": max(observed for observed, _ in observations).isoformat(),
        "equity": float(sum(equity for _, equity in observations)),
        "account_equity": {
            account: float(equity) for account, (_, equity) in zip(accounts, observations, strict=True)
        },
    }


def reconstruct_balances(db, accounts, end):
    """Rebuild complete UTC closes within the archived bill coverage."""
    if end < STRATEGY_START or not db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='checkpoints'"
    ).fetchone():
        return []

    try:
        first, last = complete_bill_dates(bill_checkpoints(db, accounts), ZONE)
        start, end = max(first, STRATEGY_START), min(last, end)
        if start > end:
            return []
        rows, _ = report(db, accounts, ZONE, start, end)
    except (ValueError, sqlite3.Error) as exc:
        logging.getLogger(__name__).warning("Cannot reconstruct UTC portfolio balances: %s", exc)
        return []
    return [{"date": str(day), "value": float(balance)} for day, balance, *_ in rows]


def publish(directory, accounts, day, runner_id):
    directory.mkdir(parents=True, exist_ok=True)
    boundary = datetime.combine(day, time.min, ZONE)
    settled_day = day - DAY
    account_set = json.dumps(sorted(accounts))
    capture_error = None

    with closing(connect(directory / "history.sqlite3")) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS utc_daily_equity (
            account_set TEXT NOT NULL, day TEXT NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY (account_set, day)
        )""")
        existing = db.execute(
            "SELECT 1 FROM utc_daily_equity WHERE account_set=? AND day=?",
            (account_set, str(settled_day)),
        ).fetchone()
        in_capture_window = boundary <= datetime.now(ZONE) < boundary + CAPTURE_WINDOW
        if not existing and settled_day >= STRATEGY_START and in_capture_window:
            try:
                snapshot = capture_equity(accounts, boundary)
            except Exception as exc:
                capture_error = exc
            else:
                with db:
                    db.execute(
                        "INSERT OR IGNORE INTO utc_daily_equity VALUES (?, ?, ?)",
                        (account_set, str(settled_day), json.dumps(snapshot, allow_nan=False)),
                    )
        db.execute("BEGIN")
        snapshots = [
            {"date": observed_day, **json.loads(payload)}
            for observed_day, payload in db.execute(
                """SELECT day, payload FROM utc_daily_equity
                   WHERE account_set=? AND day BETWEEN ? AND ? ORDER BY day""",
                (account_set, str(STRATEGY_START), str(settled_day)),
            )
        ]

        history_end = date.fromisoformat(snapshots[0]["date"]) - DAY if snapshots else settled_day
        history = reconstruct_balances(db, list(accounts), history_end)

    valuations = history + [{"date": row["date"], "value": row["equity"]} for row in snapshots]
    rows, metrics = performance(valuations)
    payload = {
        "runner_id": runner_id, "updated_at": datetime.now(ZONE).isoformat(),
        "account_ids": sorted(accounts), "strategy_start": str(STRATEGY_START),
        "metrics": metrics, "funds_series": rows,
        "equity_series": [{key: row[key] for key in ("date", "valuation_at", "as_of", "equity")}
                          for row in snapshots],
    }
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as handle:
        temp = Path(handle.name)
        try:
            json.dump(payload, handle, allow_nan=False)
            handle.close()
            temp.replace(directory / REPORT_NAME)
        finally:
            temp.unlink(missing_ok=True)
    if capture_error is not None:
        raise capture_error
    return snapshots


class DailyBitgetUpdate:
    def __init__(self, config, output, runner_id, logger):
        self.config = config
        self.output = Path(output)
        self.runner_id = runner_id
        self.logger = logger
        self.stop = threading.Event()
        self.refresh = threading.Event()
        self.threads = (
            threading.Thread(target=self._run, name="daily-bitget-update", daemon=True),
            threading.Thread(target=self._run_archive, name="daily-bitget-archive", daemon=True),
        )

    def start(self):
        for thread in self.threads:
            thread.start()

    def close(self):
        self.stop.set()
        self.refresh.set()
        for thread in self.threads:
            if thread.ident is not None:
                thread.join(timeout=8)

    def _run(self):
        while not self.stop.is_set():
            self.refresh.clear()
            day = datetime.now(ZONE).date()
            boundary = datetime.combine(day, time.min, ZONE)
            next_run = boundary + DAY
            try:
                result = self._update(day)
                if result and not self.stop.is_set():
                    self.logger.warning("Daily Bitget update failed | exit=%s", result)
                    now = datetime.now(ZONE)
                    retry = timedelta(seconds=5) if now < boundary + CAPTURE_WINDOW else timedelta(minutes=5)
                    next_run = min(next_run, now + retry)
            except Exception:
                self.logger.exception("Daily Bitget update failed")
                next_run = min(next_run, datetime.now(ZONE) + timedelta(minutes=5))
            self.refresh.wait(max(0, (next_run - datetime.now(ZONE)).total_seconds()))

    def _run_archive(self):
        while not self.stop.is_set():
            now = datetime.now(ZONE)
            boundary = datetime.combine(now.date(), time.min, ZONE)
            archive_at = boundary + CAPTURE_WINDOW
            if now < archive_at:
                self.stop.wait((archive_at - now).total_seconds())
                continue
            next_run = archive_at + DAY
            try:
                result = self._update(now.date(), archive_only=True)
                if result == 0:
                    self.refresh.set()
                elif not self.stop.is_set():
                    self.logger.warning("Daily Bitget archive failed | exit=%s", result)
                    next_run = min(next_run, datetime.now(ZONE) + timedelta(minutes=5))
            except Exception:
                self.logger.exception("Daily Bitget archive failed")
                next_run = min(next_run, datetime.now(ZONE) + timedelta(minutes=5))
            self.stop.wait(max(0, (next_run - datetime.now(ZONE)).total_seconds()))

    def _update(self, day, *, archive_only=False):
        command = [
            sys.executable, "-m", "trade.recording.daily_bitget",
            "--config", str(self.config), "--output", str(self.output),
            "--runner-id", self.runner_id, "--day", str(day),
        ]
        if archive_only:
            command.append("--archive-only")
        with subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2]) as child:
            while child.poll() is None:
                if self.stop.wait(0.5):
                    child.terminate()
                    try:
                        return child.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        return child.wait()
            return child.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runner-id", required=True)
    parser.add_argument("--day", required=True, type=date.fromisoformat)
    parser.add_argument("--archive-only", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    directory = args.output / "bitget_performance"
    accounts = configured_accounts(args.config)
    if args.archive_only:
        return sync_archive(directory / "history.sqlite3", accounts, args.output / "execution_traces")
    snapshots = publish(directory, accounts, args.day, args.runner_id)
    publish_strategy_curves(directory, args.config, accounts, snapshots, STRATEGY_START, args.day - DAY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
