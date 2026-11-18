"""Update and publish the daily Bitget portfolio report."""

from __future__ import annotations

import argparse
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile
import threading
from zoneinfo import ZoneInfo

from trade.recording.archive_bitget import configured_accounts, connect, sync_archive
from trade.recording.report_bitget import report

ZONE = ZoneInfo("Asia/Shanghai")
SCHEDULE = time(0, 10)
REPORT_NAME = "daily_overview.json"


def longest_drawdown(rows):
    peak_day, peak = rows[0][0] - timedelta(days=1), 1.0
    longest = None
    underwater = False

    def interval(end, recovered):
        return {
            "peak_date": str(peak_day), "end_date": str(end),
            "days": (end - peak_day).days, "recovered": recovered,
        }

    for day, _, nav, _, _ in rows:
        if nav < peak:
            underwater = True
            continue
        if underwater:
            candidate = interval(day, True)
            if longest is None or candidate["days"] >= longest["days"]:
                longest = candidate
        peak_day, peak, underwater = day, nav, False
    if underwater:
        candidate = interval(rows[-1][0], False)
        if longest is None or candidate["days"] >= longest["days"]:
            longest = candidate
    return longest


def publish(directory, accounts, began_ms, day, runner_id):
    db = connect(directory / "history.sqlite3")
    try:
        db.execute("""CREATE TABLE IF NOT EXISTS daily_equity (
            day TEXT PRIMARY KEY, account_set TEXT NOT NULL, payload TEXT NOT NULL
        )""")
        balances, metrics = report(db, list(accounts), ZONE, None, None)
        snapshots = db.execute(
            """SELECT account_id, observed_ms, equity, unrealized_pnl
               FROM equity_analysis WHERE margin_coin='USDT' AND observed_ms>=?
               ORDER BY observed_ms""", (began_ms,),
        )
        latest = {row[0]: row[1:] for row in snapshots if row[0] in accounts}
        if set(latest) != set(accounts):
            raise ValueError("Missing fresh USDT equity snapshots")
        as_of = datetime.fromtimestamp(max(row[0] for row in latest.values()) / 1000, ZONE)
        equity = sum(Decimal(row[1]) for row in latest.values())
        unrealized = sum(Decimal(row[2]) for row in latest.values())
        snapshot = {
            "as_of": as_of.isoformat(), "equity": float(equity),
            "unrealized_pnl": float(unrealized), "balance": float(equity - unrealized),
        }
        account_set = json.dumps(sorted(accounts))
        with db:
            db.execute(
                """INSERT INTO daily_equity VALUES (?,?,?)
                   ON CONFLICT(day) DO UPDATE SET
                   account_set=excluded.account_set, payload=excluded.payload""",
                (as_of.date().isoformat(), account_set, json.dumps(snapshot)),
            )
        observed = {
            date: json.loads(payload) for date, payload in db.execute(
                "SELECT day, payload FROM daily_equity WHERE account_set=? ORDER BY day",
                (account_set,),
            )
        }
    finally:
        db.close()

    payload = {
        "runner_id": runner_id, "updated_at": datetime.now(ZONE).isoformat(),
        "update_day": day, "account_ids": sorted(accounts),
        "snapshot": snapshot,
        "metrics": {**metrics, "longest_drawdown": longest_drawdown(balances)},
        "balance_series": [
            {"date": str(date), "balance": float(balance), "nav": nav, "drawdown": dd}
            for date, balance, nav, _, dd in balances
        ],
        "equity_series": [{"date": date, **row} for date, row in observed.items()],
    }
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as handle:
        temp = Path(handle.name)
        try:
            json.dump(payload, handle, allow_nan=False)
            handle.close()
            temp.replace(directory / REPORT_NAME)
        finally:
            temp.unlink(missing_ok=True)


class DailyBitgetUpdate:
    def __init__(self, config, output, runner_id, logger):
        self.config = config
        self.output = Path(output)
        self.runner_id = runner_id
        self.logger = logger
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="daily-bitget-update", daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=8)

    def _run(self):
        directory = self.output / "bitget_performance"
        marker = directory / REPORT_NAME
        while not self.stop.is_set():
            now = datetime.now(ZONE)
            day = now.date()
            if now.time() < SCHEDULE:
                self.stop.wait((datetime.combine(day, SCHEDULE, ZONE) - now).total_seconds())
                continue
            next_run = datetime.combine(day + timedelta(days=1), SCHEDULE, ZONE)
            try:
                accounts = sorted(configured_accounts(Path(self.config)))
                state = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
                if state.get("update_day") != str(day) or state.get("account_ids") != accounts:
                    result = self._update(day)
                    if result and not self.stop.is_set():
                        self.logger.warning("Daily Bitget archive failed | exit=%s", result)
                        next_run = datetime.now(ZONE) + timedelta(minutes=5)
            except Exception:
                self.logger.exception("Daily Bitget update failed")
                next_run = datetime.now(ZONE) + timedelta(minutes=5)
            self.stop.wait(max(0, (next_run - datetime.now(ZONE)).total_seconds()))

    def _update(self, day):
        command = [
            sys.executable, "-m", "trade.recording.daily_bitget",
            "--config", str(self.config), "--output", str(self.output),
            "--runner-id", self.runner_id, "--day", str(day),
        ]
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
    parser.add_argument("--day", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    directory = args.output / "bitget_performance"
    accounts = configured_accounts(args.config)
    began_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    status = sync_archive(directory / "history.sqlite3", accounts, args.output / "execution_traces")
    if status == 0:
        publish(directory, accounts, began_ms, args.day, args.runner_id)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
