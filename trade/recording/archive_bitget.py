"""Archive Bitget Classic USDT futures history into a local SQLite database."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import hmac
import io
import json
import os
from pathlib import Path
import sqlite3
import time
from urllib.parse import urlencode

import requests

DAY_MS = 86_400_000
TRACE_TABLES = ("executions", "orders", "fills", "events")
ENDPOINTS = {
    "bills": ("account/bill", "bills", "billId", 20, {}),
    "orders": ("order/orders-history", "entrustedList", "orderId", 20, {}),
    "fills": ("order/fill-history", "fillList", "tradeId", 6, {}),
    **{
        plan: ("order/orders-plan-history", "entrustedList", "orderId", 20, {"planType": plan})
        for plan in ("normal_plan", "track_plan", "profit_loss")
    },
}
SCHEMA = """
CREATE TABLE IF NOT EXISTS configured_accounts (account_id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS archive (
    account_id TEXT NOT NULL, stream TEXT NOT NULL, record_id TEXT NOT NULL,
    created_ms INTEGER NOT NULL, updated_ms INTEGER NOT NULL,
    symbol TEXT, order_id TEXT, client_order_id TEXT, payload TEXT NOT NULL,
    PRIMARY KEY (account_id, stream, record_id)
);
CREATE INDEX IF NOT EXISTS archive_time ON archive(stream, account_id, created_ms);
CREATE INDEX IF NOT EXISTS archive_order ON archive(stream, account_id, order_id);
CREATE TABLE IF NOT EXISTS checkpoints (
    account_id TEXT, stream TEXT, covered_from_ms INTEGER NOT NULL,
    through_ms INTEGER NOT NULL, PRIMARY KEY (account_id, stream)
);
CREATE TABLE IF NOT EXISTS coverage_gaps (
    account_id TEXT, stream TEXT, from_ms INTEGER, until_ms INTEGER,
    PRIMARY KEY (account_id, stream, from_ms, until_ms)
);
CREATE TABLE IF NOT EXISTS local_trace (
    kind TEXT, row_hash TEXT, account_id TEXT NOT NULL,
    run_id TEXT NOT NULL, execution_id TEXT NOT NULL, order_id TEXT NOT NULL,
    payload TEXT NOT NULL, PRIMARY KEY (kind, row_hash)
);
CREATE INDEX IF NOT EXISTS trace_execution
    ON local_trace(kind, account_id, run_id, execution_id);
CREATE INDEX IF NOT EXISTS trace_order ON local_trace(kind, account_id, order_id);
CREATE VIEW IF NOT EXISTS fill_analysis AS
SELECT f.account_id, f.symbol, f.record_id AS trade_id, f.order_id,
       o.client_order_id, o.created_ms AS exchange_order_created_ms,
       f.created_ms AS fill_ms, f.created_ms - o.created_ms AS order_to_fill_ms,
       json_extract(f.payload, '$.side') AS side,
       json_extract(f.payload, '$.price') AS fill_price,
       json_extract(f.payload, '$.baseVolume') AS quantity,
       json_extract(f.payload, '$.quoteVolume') AS quote_volume,
       json_extract(f.payload, '$.tradeScope') AS liquidity,
       json_extract(f.payload, '$.profit') AS realized_pnl,
       json_extract(f.payload, '$.feeDetail') AS fees,
       f.payload AS raw_fill
FROM archive f
LEFT JOIN archive o ON o.stream = 'orders' AND o.account_id = f.account_id
    AND o.record_id = f.order_id
WHERE f.stream = 'fills';
CREATE VIEW IF NOT EXISTS execution_analysis AS
SELECT account_id, run_id, execution_id,
       json_extract(payload, '$.strategy_symbol') AS symbol,
       json_extract(payload, '$.strategy_hash') AS strategy_hash,
       json_extract(payload, '$.order_role') AS order_role,
       json_extract(payload, '$.status') AS status,
       json_extract(payload, '$.decision_at_utc') AS decision_at_utc,
       json_extract(payload, '$.submitted_at_utc') AS submitted_at_utc,
       json_extract(payload, '$.accepted_at_utc') AS accepted_at_utc,
       json_extract(payload, '$.first_fill_at_utc') AS first_fill_at_utc,
       json_extract(payload, '$.completed_at_utc') AS completed_at_utc,
       json_extract(payload, '$.side') AS side,
       json_extract(payload, '$.requested_quantity') AS requested_quantity,
       json_extract(payload, '$.submitted_quantity') AS submitted_quantity,
       json_extract(payload, '$.filled_quantity') AS filled_quantity,
       json_extract(payload, '$.decision_price') AS decision_price,
       json_extract(payload, '$.arrival_price') AS arrival_price,
       json_extract(payload, '$.fill_vwap') AS fill_vwap,
       json_extract(payload, '$.bid') AS bid,
       json_extract(payload, '$.ask') AS ask,
       CAST(NULLIF(json_extract(payload, '$.decision_slippage_bps'), '') AS REAL)
           AS decision_slippage_bps,
       CAST(NULLIF(json_extract(payload, '$.implementation_shortfall_bps'), '') AS REAL)
           AS implementation_shortfall_bps,
       CAST(NULLIF(json_extract(payload, '$.execution_slippage_vwap_bps'), '') AS REAL)
           AS execution_slippage_vwap_bps,
       CAST(NULLIF(json_extract(payload, '$.execution_slippage_worst_bps'), '') AS REAL)
           AS execution_slippage_worst_bps,
       CAST(NULLIF(json_extract(payload, '$.submit_to_first_fill_ms'), '') AS REAL)
           AS submit_to_first_fill_ms,
       CAST(NULLIF(json_extract(payload, '$.acknowledgement_latency_ms'), '') AS REAL)
           AS acknowledgement_latency_ms,
       payload
FROM local_trace WHERE kind = 'executions';
CREATE VIEW IF NOT EXISTS fill_execution_links AS
SELECT DISTINCT f.account_id, f.trade_id, f.order_id, t.run_id, t.execution_id
FROM fill_analysis f JOIN local_trace t ON t.account_id = f.account_id
    AND t.order_id = f.order_id AND t.kind IN ('orders', 'fills')
WHERE f.order_id != '';
CREATE VIEW IF NOT EXISTS equity_analysis AS
SELECT a.account_id, a.created_ms AS observed_ms,
       json_extract(asset.value, '$.marginCoin') AS margin_coin,
       json_extract(asset.value, '$.accountEquity') AS equity,
       json_extract(asset.value, '$.unrealizedPL') AS unrealized_pnl,
       json_extract(asset.value, '$.available') AS available,
       asset.value AS raw_account
FROM archive a JOIN json_each(a.payload, '$.records') asset
WHERE a.stream = 'equity';
"""


class BitgetHistory:
    def __init__(self, session, directory):
        self.session = session
        self.key, self.secret, self.passphrase = [
            (directory / name).read_text(encoding="utf-8").strip()
            for name in ("apikey", "secret_key", "passphrase")
        ]
        self.secret = self.secret.encode()

    def get(self, endpoint, **params):
        path = "/api/v2/mix/" + endpoint + "?" + urlencode(
            {"productType": "USDT-FUTURES", **params}
        )
        for delay in (0.2, 1, 3):
            time.sleep(delay)
            stamp = str(time.time_ns() // 1_000_000)
            signature = base64.b64encode(
                hmac.new(self.secret, (stamp + "GET" + path).encode(), hashlib.sha256).digest()
            ).decode()
            headers = {
                "ACCESS-KEY": self.key, "ACCESS-TIMESTAMP": stamp,
                "ACCESS-SIGN": signature, "ACCESS-PASSPHRASE": self.passphrase,
            }
            try:
                response = self.session.get(
                    "https://api.bitget.com" + path, headers=headers,
                    timeout=15, allow_redirects=False,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    continue
                result = response.json()
            except (requests.RequestException, ValueError):
                continue
            if result.get("code") == "00000":
                return result
            if result.get("code") != "429":
                raise RuntimeError(f"{endpoint}: Bitget code {result.get('code')}")
        raise RuntimeError(f"{endpoint}: request failed after retries")

    def pages(self, endpoint, list_key, start, end, extra):
        cursors = set()
        params = dict(startTime=start, endTime=end, limit=100, **extra)
        while True:
            data = self.get(endpoint, **params)["data"]
            rows = data[list_key] or []
            if not rows:
                return
            yield from rows
            cursor = data.get("endId")
            if not cursor and len(rows) < 100:
                return
            if not cursor or cursor in cursors:
                raise RuntimeError(f"{endpoint}: invalid pagination cursor")
            cursors.add(cursor)
            params["idLessThan"] = cursor


def encode(row):
    return json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def connect(path):
    return sqlite3.connect(path, timeout=60)


def archive_row(account, stream, record_id, created, updated, row):
    return (
        account, stream, str(record_id), created, updated, row.get("symbol"),
        row.get("executeOrderId") or row.get("orderId"), row.get("clientOid"), encode(row),
    )


def save_rows(db, rows):
    db.executemany(
        """INSERT INTO archive VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(account_id, stream, record_id) DO UPDATE SET
            updated_ms=excluded.updated_ms, symbol=excluded.symbol,
            order_id=excluded.order_id, client_order_id=excluded.client_order_id,
            payload=excluded.payload
        WHERE excluded.updated_ms >= archive.updated_ms AND excluded.payload != archive.payload""",
        rows,
    )


def sync_stream(db, api, account, stream, floor, now, overlap_days):
    endpoint, list_key, id_key, window_days, extra = ENDPOINTS[stream]
    checkpoint = db.execute(
        "SELECT through_ms FROM checkpoints WHERE account_id=? AND stream=?",
        (account, stream),
    ).fetchone()
    covered = checkpoint[0] if checkpoint else None
    left = max(floor, covered + 1 - overlap_days * DAY_MS) if checkpoint else floor
    total = 0
    while left <= now:
        right = min(now, left + window_days * DAY_MS - 1)
        rows = []
        for row in api.pages(endpoint, list_key, left, right, extra):
            created = int(row["cTime"])
            if row[id_key] in (None, ""):
                raise ValueError(f"{endpoint}: missing record ID")
            rows.append(archive_row(
                account, stream, row[id_key], created, int(row.get("uTime") or created), row
            ))
        with db:
            save_rows(db, rows)
            if covered is not None and left == floor and covered + 1 < floor:
                db.execute("INSERT OR IGNORE INTO coverage_gaps VALUES (?, ?, ?, ?)",
                           (account, stream, covered + 1, floor))
            db.execute(
                """INSERT INTO checkpoints VALUES (?, ?, ?, ?)
                   ON CONFLICT(account_id, stream) DO UPDATE SET
                   through_ms=max(checkpoints.through_ms, excluded.through_ms)""",
                (account, stream, left, right),
            )
        total += len(rows)
        left = right + 1
    return total


def sync_account(database, account, directory, now, snapshots_only, overlap_days):
    counts = {}
    errors = []
    with requests.Session() as session:
        api = BitgetHistory(session, directory)
        db = connect(database)
        try:
            if not snapshots_only:
                for stream in ENDPOINTS:
                    try:
                        # Workers may wait in the queue before starting an account.
                        floor = time.time_ns() // 1_000_000 - 90 * DAY_MS + 30_000
                        counts[stream] = sync_stream(
                            db, api, account, stream, floor, now, overlap_days
                        )
                    except (RuntimeError, KeyError, ValueError, TypeError) as exc:
                        errors.append(f"{stream}: {exc}")
            for stream, endpoint in (
                ("equity", "account/accounts"), ("positions", "position/all-position")
            ):
                try:
                    result = api.get(endpoint)
                    stamp = int(result["requestTime"])
                    row = archive_row(account, stream, stamp, stamp, stamp,
                                      {"records": result["data"]})
                    with db:
                        save_rows(db, [row])
                    counts[stream] = 1
                except (RuntimeError, KeyError, ValueError, TypeError) as exc:
                    errors.append(f"{stream}: {exc}")
        finally:
            db.close()
    return account, counts, errors


def import_traces(db, directory, accounts):
    counts = {}
    for kind in TRACE_TABLES:
        path = directory / f"{kind}.csv"
        if not path.is_file():
            continue
        # The live writer may append while this runs. Leave an unfinished line for next time.
        raw = path.read_bytes()
        complete = raw[:raw.rfind(b"\n") + 1].decode("utf-8-sig")
        added = db.total_changes
        with db:
            for row in csv.DictReader(io.StringIO(complete, newline="")):
                if None in row or None in row.values():
                    continue
                if row["venue"].lower() not in {"bitget", "bitgetvenue"}:
                    continue
                if row["account_id"] not in accounts:
                    continue
                payload = encode(row)
                digest = hashlib.sha256(payload.encode()).hexdigest()
                db.execute("INSERT OR IGNORE INTO local_trace VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (kind, digest, row["account_id"], row["run_id"], row["execution_id"],
                            row.get("order_id", ""), payload))
        counts[kind] = db.total_changes - added
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--snapshots-only", action="store_true")
    parser.add_argument(
        "--overlap-days", type=int, default=0,
        help="Re-fetch this many days before each checkpoint (default: 0).",
    )
    args = parser.parse_args()
    if not 0 <= args.overlap_days <= 90:
        parser.error("--overlap-days must be between 0 and 90")
    try:
        accounts = configured_accounts(args.config)
    except ValueError as exc:
        parser.error(str(exc))
    os.umask(0o077)
    return sync_archive(args.db, accounts, args.trace_dir, args.snapshots_only, args.overlap_days)


def configured_accounts(config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    accounts = {}
    for strategy in config["strategy"]:
        if strategy["venue"].strip().casefold() != "bitget" or not (
            strategy.get("enable", True) and strategy.get("run_live")
        ):
            continue
        account = str(strategy["account_id"]).strip()
        directory = (config_path.resolve().parent / strategy["bitget"]["path"]).resolve()
        if accounts.setdefault(account, directory) != directory:
            raise ValueError(f"Conflicting credential paths for account {account}")
    if not accounts:
        raise ValueError("No enabled Bitget live accounts")
    return accounts


def sync_archive(database, accounts, trace_dir=None, snapshots_only=False, overlap_days=0):
    database.parent.mkdir(parents=True, exist_ok=True)
    db = connect(database)
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(SCHEMA)
        with db:
            db.execute("DELETE FROM configured_accounts")
            db.executemany("INSERT INTO configured_accounts VALUES (?)",
                           ((account,) for account in accounts))
        now = time.time_ns() // 1_000_000
        failed = False
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = [
                pool.submit(sync_account, database, account, directory, now,
                            snapshots_only, overlap_days)
                for account, directory in accounts.items()
            ]
            for job in jobs:
                try:
                    account, counts, errors = job.result()
                    print(f"{account}: fetched {counts}", flush=True)
                    for error in errors:
                        print(f"{account}: {error}", flush=True)
                    failed |= bool(errors)
                except (OSError, sqlite3.Error) as exc:
                    failed = True
                    print(f"Account sync failed: {exc}", flush=True)
        if trace_dir is not None and not snapshots_only:
            print("New local trace rows:", import_traces(db, trace_dir, accounts))
        gaps = db.execute("SELECT COUNT(*) FROM coverage_gaps").fetchone()[0]
        if gaps:
            print(f"Retention gaps recorded in coverage_gaps: {gaps}")
        print(f"Archive: {database} | {datetime.now(timezone.utc).isoformat()}")
        return int(failed)
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
