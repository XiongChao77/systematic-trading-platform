#!/usr/bin/env python3
"""Record public Binance USD-M order book snapshots, without analyzing them.

Uses only the standard library. No API keys or trading endpoints.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

if __package__:
    from .orderbook_common import parse_book, positive
else:
    from orderbook_common import parse_book, positive


BASE_URL = "https://fapi.binance.com"


def fetch_snapshot(symbol, depth, timeout):
    query = urlencode({"symbol": symbol, "limit": depth})
    request = Request(
        f"{BASE_URL}/fapi/v1/depth?{query}",
        headers={"User-Agent": "orderbook-slippage-recorder/1.0"},
    )
    started = time.monotonic()
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    received = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    if not isinstance(payload, dict) or "lastUpdateId" not in payload:
        raise ValueError("Unexpected Binance depth response")
    return {
        "received_at_utc": received,
        "symbol": symbol,
        "last_update_id": payload["lastUpdateId"],
        "exchange_time_ms": payload.get("T", payload.get("E")),
        "request_latency_ms": round((time.monotonic() - started) * 1000, 3),
        "depth": payload,
    }


def save_snapshot(directory, snapshot):
    """Append a validated raw snapshot; analysis belongs to a separate process."""
    parse_book(snapshot["depth"])
    day = snapshot["received_at_utc"][:10]
    path = directory / f"{snapshot['symbol']}_{day}.jsonl.gz"
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, separators=(",", ":")) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["DOGEUSDT", "XLMUSDT", "AAVEUSDT"])
    parser.add_argument("--depth", type=int, choices=[5, 10, 20, 50, 100, 500, 1000], default=1000)
    parser.add_argument("--interval", type=positive, default=5,
                        help="Minimum seconds between sampling rounds (default: 5)")
    parser.add_argument("--samples", type=int, default=0,
                        help="Sampling rounds; 0 records until Ctrl+C")
    parser.add_argument("--timeout", type=positive, default=10)
    parser.add_argument("--output", type=Path, default=Path("orderbook_output"))
    args = parser.parse_args(argv)
    if args.samples < 0:
        parser.error("--samples must be nonnegative")
    symbols = list(dict.fromkeys(s.upper() for s in args.symbols))
    if any(not s.isascii() or not s.isalnum() for s in symbols):
        parser.error("Symbols must contain only ASCII letters and digits")
    if args.interval < 1:
        parser.error("REST snapshot sampling requires --interval >= 1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = args.output / run_id
    directory.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), "output": str(directory), "symbols": symbols,
              "source": f"{BASE_URL}/fapi/v1/depth",
              "limitations": "Periodic snapshots, not every update. Symbols are sampled sequentially."}
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    logging.info("Recording to %s", directory.resolve())
    rounds, successes, failures = 0, 0, 0
    consecutive_failures = 0
    try:
        while args.samples == 0 or rounds < args.samples:
            started = time.monotonic()
            cooldown = 0.0
            for symbol in symbols:
                try:
                    snapshot = fetch_snapshot(symbol, args.depth, args.timeout)
                    # Disk errors are fatal rather than silently losing recordings.
                    parse_book(snapshot["depth"])
                except (HTTPError, URLError, TimeoutError, ValueError, KeyError, TypeError) as exc:
                    failures += 1
                    consecutive_failures += 1
                    logging.error("Snapshot failed for %s: %s", symbol, exc)
                    error = {"at_utc": datetime.now(timezone.utc).isoformat(),
                             "symbol": symbol, "error": str(exc)}
                    with (directory / "errors.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(error) + "\n")
                    if isinstance(exc, HTTPError) and exc.code in (400, 401, 403, 404, 451):
                        logging.error("Non-retryable HTTP error; check symbol and endpoint access")
                        return 1
                    cooldown = min(60, 2 ** min(consecutive_failures, 6))
                    if isinstance(exc, HTTPError) and exc.code in (418, 429):
                        try:
                            cooldown = max(60, float(exc.headers.get("Retry-After", "60")))
                        except ValueError:
                            cooldown = 60
                    break
                save_snapshot(directory, snapshot)
                successes += 1
                consecutive_failures = 0
                logging.info("%s snapshot=%s latency_ms=%.1f",
                             symbol, snapshot["last_update_id"], snapshot["request_latency_ms"])
            rounds += 1
            if args.samples == 0 or rounds < args.samples:
                time.sleep(max(cooldown, args.interval - (time.monotonic() - started), 0))
    except KeyboardInterrupt:
        logging.info("Stopped by user")
    finally:
        logging.info("Finished: rounds=%d snapshots=%d errors=%d", rounds, successes, failures)
    return 0 if successes and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
