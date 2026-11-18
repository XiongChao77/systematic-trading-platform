"""Calculate daily realized NAV and risk metrics from a Bitget archive."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import date, datetime, time, timedelta
from decimal import Decimal
import json
from math import sqrt
from pathlib import Path
import sqlite3
from statistics import mean, stdev
from zoneinfo import ZoneInfo

ZERO = Decimal(0)
TOLERANCE = Decimal("0.00001")
DAY = timedelta(days=1)
TRADING_BILLS = {
    "open_long", "open_short", "close_long", "close_short", "buy", "sell",
    "contract_settle_fee", "force_close_long", "force_close_short",
    "force_buy", "force_sell", "delivery_long", "delivery_short",
    "adl_close_long", "adl_close_short",
}


def bill_amount(row):
    return Decimal(row.get("amount") or "0") + Decimal(row.get("fee") or "0")


def report(db, accounts, zone, requested_start, requested_end):
    if not accounts:
        accounts = [
            row[0] for row in db.execute("SELECT account_id FROM configured_accounts")
        ]
    coverage = {
        row[0]: row[1:]
        for row in db.execute(
            "SELECT account_id, covered_from_ms, through_ms FROM checkpoints WHERE stream='bills'"
        )
        if not accounts or row[0] in accounts
    }
    if not coverage or (accounts and set(coverage) != set(accounts)):
        raise ValueError("Missing bill coverage for selected accounts")

    def stamp(day):
        return int(datetime.combine(day, time.min, zone).timestamp() * 1000)

    def local_day(milliseconds):
        return datetime.fromtimestamp(milliseconds / 1000, zone).date()

    first_ms = max(row[0] for row in coverage.values())
    first = local_day(first_ms)
    if stamp(first) < first_ms:
        first += DAY
    last = local_day(min(row[1] for row in coverage.values())) - DAY
    start, end = requested_start or first, requested_end or last
    if start < first or end > last or start > end:
        raise ValueError(f"Use complete dates within {first} through {last}")

    left, right = stamp(start), stamp(end + DAY)
    opening = ZERO
    events = defaultdict(lambda: [ZERO, ZERO])
    for account in coverage:
        gap = db.execute(
            """SELECT 1 FROM coverage_gaps WHERE account_id=? AND stream='bills'
               AND from_ms<? AND until_ms>? LIMIT 1""",
            (account, right, left),
        ).fetchone()
        if gap:
            raise ValueError(f"Bill coverage gap for account {account} in selected period")
        first_balance = None
        balance = None
        for created, payload in db.execute(
            """SELECT created_ms, payload FROM archive
               WHERE account_id=? AND stream='bills' AND created_ms>=? AND created_ms<?
                 AND json_extract(payload,'$.coin')='USDT'
               ORDER BY created_ms, length(record_id), record_id""",
            (account, left, right),
        ):
            bill = json.loads(payload)
            delta = bill_amount(bill)
            after = Decimal(bill["balance"])
            if balance is None:
                first_balance = after - delta
            elif abs(after - balance - delta) > TOLERANCE:
                raise ValueError(f"Bill balances do not reconcile for {account} at {created}")
            balance = after
            kind = bill["businessType"]
            events[created][0] += delta
            if kind.startswith(("trans_from_", "trans_to_")):
                events[created][1] += delta
            elif kind not in TRADING_BILLS:
                raise ValueError(f"Unclassified bill type {kind!r} for {account}")
        if first_balance is not None:
            opening += first_balance
            continue
        prior_gap = db.execute(
            """SELECT coalesce(max(until_ms),0) FROM coverage_gaps
               WHERE account_id=? AND stream='bills' AND until_ms<=?""",
            (account, left),
        ).fetchone()[0]
        previous = db.execute(
            """SELECT payload FROM archive WHERE account_id=? AND stream='bills'
               AND created_ms>=? AND created_ms<? AND json_extract(payload,'$.coin')='USDT'
               ORDER BY created_ms DESC, length(record_id) DESC, record_id DESC LIMIT 1""",
            (account, prior_gap, left),
        ).fetchone()
        if previous:
            opening += Decimal(json.loads(previous[0])["balance"])
            continue
        next_gap = db.execute(
            """SELECT coalesce(min(from_ms),?) FROM coverage_gaps
               WHERE account_id=? AND stream='bills' AND from_ms>=?""",
            (coverage[account][1] + 1, account, right),
        ).fetchone()[0]
        following = db.execute(
            """SELECT payload FROM archive WHERE account_id=? AND stream='bills'
               AND created_ms>=? AND created_ms<? AND json_extract(payload,'$.coin')='USDT'
               ORDER BY created_ms, length(record_id), record_id LIMIT 1""",
            (account, right, next_gap),
        ).fetchone()
        if following:
            bill = json.loads(following[0])
            opening += Decimal(bill["balance"]) - bill_amount(bill)
            continue
        raise ValueError(f"No USDT bill anchor for {account}")

    nav = event_peak = daily_peak = Decimal(1)
    balance = opening
    event_drawdown = ZERO
    flows = ZERO
    daily = {}
    for created, (delta, flow) in sorted(events.items()):
        capital = balance + flow
        if capital > TOLERANCE:
            nav *= 1 + (delta - flow) / capital
        elif abs(delta - flow) > TOLERANCE or capital < -TOLERANCE:
            raise ValueError(f"Cannot calculate return with nonpositive capital at {created}")
        balance += delta
        flows += flow
        event_peak = max(event_peak, nav)
        event_drawdown = max(event_drawdown, 1 - nav / event_peak)
        daily[local_day(created)] = (balance, nav)

    rows = []
    returns = []
    balance, nav = opening, Decimal(1)
    for offset in range((end - start).days + 1):
        day = start + offset * DAY
        close, current_nav = daily.get(day, (balance, nav))
        daily_peak = max(daily_peak, current_nav)
        ret = float(current_nav / nav - 1)
        if balance > TOLERANCE or close > TOLERANCE:
            returns.append(ret)
        rows.append((day, close, float(current_nav), ret, float(current_nav / daily_peak - 1)))
        balance, nav = close, current_nav
    drawdown = -min(row[4] for row in rows)
    annual_return = float(nav) ** (365 / len(returns)) - 1 if returns else None
    volatility = stdev(returns) if len(returns) > 1 else 0
    return rows, {
        "start": str(start), "end": str(end), "timezone": str(zone),
        "accounts": len(coverage), "funded_days": len(returns),
        "basis": "USDT realized NAV, external transfers adjusted",
        "opening_balance": float(opening), "closing_balance": float(balance),
        "net_transfers": float(flows), "total_return": float(nav - 1),
        "annualized_return": annual_return,
        "sharpe_zero_risk_free": mean(returns) * sqrt(365) / volatility if volatility else None,
        "daily_max_drawdown": drawdown, "bill_event_max_drawdown": float(event_drawdown),
        "calmar": annual_return / drawdown if drawdown and annual_return is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--account", action="append")
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    try:
        with sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True) as db:
            db.execute("BEGIN")
            rows, metrics = report(db, args.account, ZoneInfo(args.timezone), args.start, args.end)
    except (ValueError, sqlite3.Error) as exc:
        parser.error(str(exc))
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("date", "closing_balance_usdt", "nav", "daily_return", "drawdown"))
            writer.writerows(rows)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
