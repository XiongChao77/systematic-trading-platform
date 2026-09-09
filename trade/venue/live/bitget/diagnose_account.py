"""Read-only Bitget account diagnostics for rejected futures entries."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import time
from urllib.parse import urlencode

import requests

BASE_URL = "https://api.bitget.com"
ACCOUNT_INFO = "/api/v2/spot/account/info"
CONTRACTS = "/api/v2/mix/market/contracts"
FUTURES_ACCOUNT = "/api/v2/mix/account/account"
UTA_INFO = "/api/v3/account/info"
ALLOWED_PATHS = {ACCOUNT_INFO, CONTRACTS, FUTURES_ACCOUNT, UTA_INFO}


def fingerprint(value):
    return hashlib.sha256(str(value).encode()).hexdigest()[:16] if value else None


class DiagnosticClient:
    """Allowlisted GET requests without venue initialization or trading methods."""

    def __init__(self, key_path, *, session=None, timeout=10):
        self.session = session or requests.Session()
        self.timeout = timeout
        self.secrets = tuple(
            (Path(key_path) / name).read_text().strip()
            for name in ("apikey", "secret_key", "passphrase")
        )
        if not all(self.secrets):
            raise ValueError("Credential files must not be empty")
        self.redactions = list(self.secrets)

    def redact(self, value):
        text = str(value)
        for secret in sorted(self.redactions, key=len, reverse=True):
            text = text.replace(secret, "[REDACTED]")
        return text

    def get(self, path, params=None):
        if path not in ALLOWED_PATHS:
            raise ValueError("Endpoint is not allowed by the read-only diagnostic")
        query = urlencode(params or {})
        request_path = path + (f"?{query}" if query else "")
        headers = {"locale": "en-US", "Content-Type": "application/json"}
        if path != CONTRACTS:
            timestamp = str(int(time.time() * 1000))
            key, secret, passphrase = self.secrets
            signature = base64.b64encode(
                hmac.new(
                    secret.encode(),
                    (timestamp + "GET" + request_path).encode(),
                    hashlib.sha256,
                ).digest()
            ).decode()
            self.redactions.append(signature)
            headers.update(
                {
                    "ACCESS-KEY": key,
                    "ACCESS-PASSPHRASE": passphrase,
                    "ACCESS-TIMESTAMP": timestamp,
                    "ACCESS-SIGN": signature,
                }
            )
        started = time.monotonic()
        check = {"endpoint": path, "method": "GET", "ok": False}
        try:
            response = self.session.get(
                BASE_URL + request_path,
                headers=headers,
                timeout=self.timeout,
                allow_redirects=False,
            )
            check["http_status"] = response.status_code
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError("Expected an object response")
            check.update(
                code=self.redact(body.get("code", "")),
                message=self.redact(body.get("msg", "")),
                request_time_ms=body.get("requestTime"),
                ok=200 <= response.status_code < 300 and body.get("code") == "00000",
            )
            return check, body.get("data") if check["ok"] else None
        except Exception as exc:
            # Transport exceptions may retain request headers; never render them.
            check["error_type"] = type(exc).__name__
            return check, None
        finally:
            check["elapsed_ms"] = round((time.monotonic() - started) * 1000, 2)


def diagnose(client, strategy_id, symbol, *, include_uta=False):
    result = {
        "strategy_id": strategy_id,
        "symbol": symbol,
        "api_key_fingerprint": fingerprint(client.secrets[0]),
        "read_only": True,
        "order_submission_tested": False,
        "restriction_reason": "not_exposed_by_these_read_endpoints",
        "checks": [],
        "findings": [],
    }
    check, info = client.get(ACCOUNT_INFO)
    if check["ok"] and isinstance(info, dict):
        authorities = info.get("authorities")
        check["details"] = {
            "uid_fingerprint": fingerprint(info.get("userId")),
            "parent_uid_fingerprint": fingerprint(info.get("parentId")),
            "trader_type": info.get("traderType"),
            "ip_whitelist_configured": bool(info.get("ips")),
            "authorities": authorities,
        }
        if isinstance(authorities, list) and all(
            isinstance(item, str) for item in authorities
        ):
            allowed = "coow" in authorities
            check["details"]["futures_order_write_permission"] = allowed
            result["findings"].append(
                "classic_futures_order_write_permission_present"
                if allowed
                else "classic_futures_order_write_permission_missing"
            )
    elif check["ok"]:
        check.update(ok=False, error_type="InvalidAccountInfo")
    result["checks"].append(check)

    check, contracts = client.get(
        CONTRACTS, {"symbol": symbol, "productType": "USDT-FUTURES"}
    )
    if check["ok"]:
        contract = (
            next(
                (
                    r
                    for r in contracts
                    if isinstance(r, dict) and r.get("symbol") == symbol
                ),
                None,
            )
            if isinstance(contracts, list)
            else None
        )
        if contract is None:
            check.update(ok=False, error_type="ContractNotFound")
        else:
            check["details"] = {
                key: contract.get(key)
                for key in (
                    "symbolStatus",
                    "symbolType",
                    "supportMarginCoins",
                    "minTradeUSDT",
                    "minTradeNum",
                    "sizeMultiplier",
                    "maintainTime",
                    "limitOpenTime",
                    "offTime",
                )
            }
            result["findings"].append(
                "contract_status_normal"
                if contract.get("symbolStatus") == "normal"
                else "contract_status_not_normal"
            )
    result["checks"].append(check)

    check, account = client.get(
        FUTURES_ACCOUNT,
        {
            "symbol": symbol,
            "productType": "USDT-FUTURES",
            "marginCoin": "USDT",
        },
    )
    if check["ok"] and isinstance(account, dict):
        check["details"] = {
            key: account.get(key)
            for key in (
                "marginMode",
                "posMode",
                "assetMode",
                "crossedMarginLeverage",
                "isolatedLongLever",
                "isolatedShortLever",
            )
        }
        result["findings"].append("classic_futures_account_read_succeeded")
    elif check["ok"]:
        check.update(ok=False, error_type="InvalidFuturesAccount")
    result["checks"].append(check)

    if include_uta:
        check, info = client.get(UTA_INFO)
        if check["ok"] and isinstance(info, dict):
            check["details"] = {
                "uid_fingerprint": fingerprint(info.get("userId")),
                "parent_uid_fingerprint": fingerprint(info.get("parentId")),
                "permission_type": info.get("permType"),
                "permissions": info.get("permissions"),
            }
        elif check["ok"]:
            check.update(ok=False, error_type="InvalidUtaInfo")
        result["checks"].append(check)

    result["checks_passed"] = all(c["ok"] for c in result["checks"])
    # GET success is not proof that an account is permitted to submit an order.
    result["trading_access"] = "unverified"
    return result


def select_targets(config_path, strategy_ids):
    path = Path(config_path).resolve()
    rows = json.loads(path.read_text())["strategy"]
    targets = []
    for strategy_id in dict.fromkeys(strategy_ids):
        row = rows[strategy_id]
        venue = str(row["venue"]).lower()
        if venue != "bitget":
            raise ValueError(f"Strategy does not use Bitget: {strategy_id}")
        targets.append((strategy_id, (path.parent / row[venue]["path"]).resolve()))
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--key-path")
    source.add_argument("--config", help="Current live runner configuration")
    parser.add_argument("--strategy-id", action="append", default=[])
    parser.add_argument(
        "--symbol", default="DOGEUSDT", help="Contract checked for every selected key"
    )
    parser.add_argument("--include-uta", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.config and not args.strategy_id:
        parser.error("--config requires at least one --strategy-id")
    if args.key_path and len(args.strategy_id) > 1:
        parser.error("--key-path accepts at most one --strategy-id")
    try:
        targets = (
            select_targets(args.config, args.strategy_id)
            if args.config
            else [
                (
                    args.strategy_id[0] if args.strategy_id else "manual-check",
                    Path(args.key_path),
                )
            ]
        )
    except Exception as exc:
        parser.error(f"Cannot select diagnostic targets: {type(exc).__name__}")
    report = {
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "Current read-only observations cannot reconstruct past restrictions or prove order access.",
        "accounts": [],
    }
    for strategy_id, key_path in targets:
        client = None
        try:
            client = DiagnosticClient(key_path)
            result = diagnose(
                client,
                strategy_id,
                args.symbol.strip().upper(),
                include_uta=args.include_uta,
            )
            # Apply a final credential scrub to every serialized response field.
            result = json.loads(client.redact(json.dumps(result)))
        except Exception as exc:
            result = {
                "strategy_id": strategy_id,
                "checks_passed": False,
                "error_type": type(exc).__name__,
            }
        finally:
            if client is not None:
                client.session.close()
        report["accounts"].append(result)
        print(json.dumps(result), flush=True)
        time.sleep(1.05)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(
        output, "x", opener=lambda path, flags: os.open(path, flags, 0o600)
    ) as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"Report: {output.resolve()}")
    return 0 if all(r["checks_passed"] for r in report["accounts"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
