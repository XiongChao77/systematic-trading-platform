"""Read-only verification of Bybit/cTrader order identity using existing keys.

Run directly; this manual diagnostic never places, modifies, or cancels orders.
Reports observed samples, not guarantees about every exchange-generated order.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DAY_MS = 86_400_000


def project(row, fields):
    return {field: row.get(field) for field in fields}


def summarize(observations):
    return dict(Counter(row["result"] for row in observations))


def error_details(exc):
    """Extract only status codes and the SDK error structure, never credentials."""
    result = {"error_type": type(exc).__name__}
    response = getattr(exc, "response", None)
    if response is not None:
        result["http_status"] = response.status_code
        try:
            code = response.json().get("retCode")
            if isinstance(code, int):
                result["api_code"] = code
        except (ValueError, AttributeError):
            pass
    match = re.fullmatch(r"Bybit API code (-?[0-9]+)", str(exc))
    if match:
        result["api_code"] = int(match[1])
    if type(exc).__name__ == "CTraderRequestError":
        match = re.match(
            r"cTrader request (ProtoOA[A-Za-z]+) failed: ([A-Z_0-9]+):", str(exc)
        )
        if match:
            result.update(request=match[1], api_code=match[2])
    return result


def inspect_bybit(key_path, start, end, symbol, testnet):
    import requests

    key = (key_path / "hmac_api_key").read_text().strip()
    secret = (key_path / "hmac_secret").read_text().strip()
    if not key or not secret:
        raise ValueError("Empty credentials")
    host = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
    session = requests.Session()

    def fetch(path, params):
        query = urlencode(params)
        timestamp = str(time.time_ns() // 1_000_000)
        window = "10000"
        signature = hmac.new(
            secret.encode(), (timestamp + key + window + query).encode(), hashlib.sha256
        ).hexdigest()
        response = session.get(
            host + path + "?" + query,
            headers={
                "X-BAPI-API-KEY": key,
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": window,
                "X-BAPI-SIGN": signature,
            },
            timeout=15,
            allow_redirects=False,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("retCode") != 0:
            raise RuntimeError(f"Bybit API code {body.get('retCode')}")
        return body["result"]

    def history(path):
        rows = []
        left = start
        while left <= end:
            right = min(end, left + 7 * DAY_MS - 1)
            params = dict(category="linear", startTime=left, endTime=right, limit=100)
            if symbol:
                params["symbol"] = symbol
            cursors = set()
            while True:
                data = fetch(path, params)
                rows.extend(data.get("list", []))
                cursor = data.get("nextPageCursor")
                if not cursor:
                    break
                if cursor in cursors:
                    raise RuntimeError("Repeated pagination cursor")
                cursors.add(cursor)
                params["cursor"] = cursor
                time.sleep(0.15)
            left = right + 1
        return rows

    try:
        orders = history("/v5/order/history")
        fills = history("/v5/execution/list")
    finally:
        session.close()
    orders = list({r["orderId"]: r for r in orders}.values())
    observations = []
    for row in orders:
        if not row.get("stopOrderType") or row["stopOrderType"] == "UNKNOWN":
            continue
        cid = row.get("orderLinkId", "")
        parent = row.get("parentOrderLinkId", "")
        matches = [
            r["orderId"]
            for r in orders
            if cid and r.get("orderLinkId") == cid and r["orderId"] != row["orderId"]
        ]
        observations.append(
            {
                "order_id": row["orderId"],
                "stop_order_type": row["stopOrderType"],
                "client_order_id": cid,
                "parent_client_order_id": parent,
                "other_order_ids_with_same_client_id": matches,
                "result": (
                    "explicit_parent_link"
                    if parent
                    else (
                        "shared_client_id_observed"
                        if matches
                        else (
                            "client_id_missing"
                            if not cid
                            else "inconclusive_parent_not_identified"
                        )
                    )
                ),
            }
        )
    return {
        "venue": "bybit",
        "environment": "testnet" if testnet else "live",
        "summary": summarize(observations),
        "observations": observations,
        "orders": [
            project(
                r,
                (
                    "orderId",
                    "orderLinkId",
                    "parentOrderLinkId",
                    "symbol",
                    "side",
                    "orderStatus",
                    "stopOrderType",
                    "createType",
                    "reduceOnly",
                    "createdTime",
                ),
            )
            for r in orders
        ],
        "fills": [
            project(
                r,
                ("execId", "orderId", "orderLinkId", "symbol", "execTime", "execType"),
            )
            for r in fills
        ],
        "status": (
            "observed" if observations else "inconclusive_no_conditional_order_samples"
        ),
    }


def inspect_ctrader(key_path, login, start, end):
    from trade.venue.live.ctrader.ctrader_venue import CTraderOpenApiConnection
    from ctrader_open_api.messages import OpenApiModelMessages_pb2 as models

    api = CTraderOpenApiConnection(str(key_path), environment="live", timeout=15)
    try:
        account, environment = api.resolve_account(login)
        if environment != api.environment:
            api.shutdown()
            api = CTraderOpenApiConnection(
                str(key_path), environment=environment, timeout=15
            )
        api.authenticate(account)

        def history(message, collection, left, right):
            fields = dict(
                ctidTraderAccountId=account, fromTimestamp=left, toTimestamp=right
            )
            if collection == "deal":
                fields["maxRows"] = 1000
            # Keep manual history reads well below the per-connection rate limit.
            time.sleep(1.0)
            response = api.request(message, **fields)
            if getattr(response, "hasMore", False):
                if left == right:
                    raise RuntimeError("History exceeds API limit in one millisecond")
                middle = (left + right) // 2
                return history(message, collection, left, middle) + history(
                    message, collection, middle + 1, right
                )
            return list(getattr(response, collection))

        orders, fills = [], []
        for left in range(start, end, DAY_MS):
            right = min(end, left + DAY_MS - 1)
            orders.extend(history("ProtoOAOrderListReq", "order", left, right))
            fills.extend(history("ProtoOADealListReq", "deal", left, right))
    finally:
        api.shutdown()
    orders = list({int(r.orderId): r for r in orders}.values())
    rows = [
        {
            "order_id": str(r.orderId),
            "position_id": str(r.positionId),
            "client_order_id": r.clientOrderId,
            "label": r.tradeData.label,
            "closing": r.closingOrder,
            "order_type": models.ProtoOAOrderType.Name(r.orderType),
            "order_status": models.ProtoOAOrderStatus.Name(r.orderStatus),
            "symbol_id": r.tradeData.symbolId,
        }
        for r in orders
    ]
    observations = []
    for row in rows:
        if not row["closing"] and row["order_type"] != "STOP_LOSS_TAKE_PROFIT":
            continue
        entries = [
            r
            for r in rows
            if r["position_id"] == row["position_id"]
            and r["position_id"] != "0"
            and not r["closing"]
            and r["order_type"] != "STOP_LOSS_TAKE_PROFIT"
        ]
        matched = [
            r
            for r in entries
            if row["client_order_id"] and r["client_order_id"] == row["client_order_id"]
        ]
        observations.append(
            {
                **row,
                "entry_order_ids": [r["order_id"] for r in entries],
                "matching_client_id_entry_order_ids": [r["order_id"] for r in matched],
                "entry_labels": sorted({r["label"] for r in entries}),
                "result": (
                    "inconclusive_entry_outside_sample"
                    if not entries
                    else (
                        "entry_client_id_inherited"
                        if matched
                        else (
                            "client_id_missing"
                            if not row["client_order_id"]
                            else "different_client_id"
                        )
                    )
                ),
            }
        )
    return {
        "venue": "ctrader",
        "trader_login": str(login),
        "environment": environment,
        "summary": summarize(observations),
        "observations": observations,
        "orders": rows,
        "fills": [
            {
                "deal_id": str(r.dealId),
                "order_id": str(r.orderId),
                "position_id": str(r.positionId),
                "executed_at_ms": r.executionTimestamp,
            }
            for r in {int(d.dealId): d for d in fills}.values()
        ],
        "status": "observed" if observations else "inconclusive_no_exit_order_samples",
    }


def targets(args):
    if args.key_path:
        if not args.venue or (args.venue == "ctrader" and not args.trader_login):
            raise ValueError(
                "Explicit keys require --venue and cTrader requires --trader-login"
            )
        return [(args.venue, Path(args.key_path).resolve(), args.trader_login)]
    path = Path(args.config).resolve()
    entries = json.loads(path.read_text())["strategy"]
    if isinstance(entries, dict):
        entries = entries.values()
    result = []
    for row in entries:
        venue = str(row.get("venue", "")).lower()
        if (
            venue not in {"bybit", "ctrader"}
            or not row.get("run_live")
            or (args.venue and args.venue != venue)
        ):
            continue
        settings = next(v for k, v in row.items() if k.lower() == venue)
        login = str(settings.get("trader_login", ""))
        if args.trader_login and login != args.trader_login:
            continue
        target = (venue, (path.parent / settings["path"]).resolve(), login)
        if target not in result:
            result.append(target)
    if not result:
        raise ValueError(
            "No matching configured accounts; use --key-path for an unconfigured account"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", default=str(ROOT / "LiveTrading/live_config.json"))
    source.add_argument("--key-path")
    parser.add_argument("--venue", choices=("bybit", "ctrader"))
    parser.add_argument("--trader-login")
    parser.add_argument("--symbol", help="Optional Bybit linear symbol filter")
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Bybit only; never inferred from key names",
    )
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument(
        "--output", required=True, help="New private JSON evidence file"
    )
    args = parser.parse_args()
    if not 1 <= args.days <= 90:
        parser.error("--days must be between 1 and 90")
    try:
        selected = targets(args)
    except Exception as exc:
        parser.error(str(exc) if isinstance(exc, ValueError) else type(exc).__name__)
    if args.testnet and any(v != "bybit" for v, _, _ in selected):
        parser.error("--testnet only applies to Bybit")
    # SDK logs and exception strings can contain authentication material.
    logging.disable(logging.CRITICAL)
    end = time.time_ns() // 1_000_000
    report = {
        "read_only": True,
        "from_ms": end - args.days * DAY_MS,
        "to_ms": end,
        "scope": "Observed history only; no matching samples is inconclusive.",
        "accounts": [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "x", opener=lambda p, flags: os.open(p, flags, 0o600)) as handle:
        for venue, key_path, login in selected:
            try:
                result = (
                    inspect_bybit(
                        key_path, report["from_ms"], end, args.symbol, args.testnet
                    )
                    if venue == "bybit"
                    else inspect_ctrader(key_path, login, report["from_ms"], end)
                )
            except Exception as exc:
                result = {"venue": venue, "status": "error", **error_details(exc)}
            result["account_reference"] = (
                str(login) if venue == "ctrader" else key_path.name
            )
            report["accounts"].append(result)
            handle.seek(0)
            json.dump(report, handle, indent=2)
            handle.truncate()
            handle.flush()
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in (
                            "venue",
                            "status",
                            "summary",
                            "error_type",
                            "http_status",
                            "api_code",
                            "request",
                        )
                        if k in result
                    }
                ),
                flush=True,
            )
    print(f"Report: {output.resolve()}")
    return 1 if any(r["status"] == "error" for r in report["accounts"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
