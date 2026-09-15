#!/usr/bin/env python3
"""Analyze recorded order books offline for configurable USDT order notionals.

Amounts use snapshot midpoint, not margin. Exchange quantity filters, fees,
funding, latency and future liquidity changes are not modeled.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import tempfile
from pathlib import Path

if __package__:
    from .orderbook_common import parse_book, positive
else:
    from orderbook_common import parse_book, positive


FIELDS = (
    "received_at_utc", "symbol", "last_update_id", "exchange_time_ms",
    "request_latency_ms", "side", "target_notional_usdt", "mid_price",
    "best_price", "target_quantity", "filled_quantity", "coverage_pct",
    "status", "levels_used", "filled_quote_usdt", "vwap", "worst_price",
    "slippage_usdt_vs_best", "slippage_pct_vs_best", "slippage_bps_vs_best",
    "slippage_usdt_vs_mid", "slippage_pct_vs_mid",
)


def estimate(levels, side, notional, midpoint):
    """Walk visible depth; never extrapolate a full fill from insufficient depth."""
    if side not in {"buy", "sell"}:
        raise ValueError("Side must be buy or sell")
    if not all(math.isfinite(v) and v > 0 for v in (notional, midpoint)):
        raise ValueError("Notional and midpoint must be positive and finite")
    quantity = notional / midpoint
    remaining, quote, used, worst = quantity, 0.0, 0, None
    for price, available in levels:
        take = min(remaining, available)
        quote += take * price
        remaining -= take
        used += 1
        worst = price
        if remaining <= quantity * 1e-12:
            remaining = 0.0
            break
    filled = quantity - remaining
    complete = remaining == 0
    vwap = quote / filled if filled else None
    best = levels[0][0]
    direction = 1 if side == "buy" else -1
    # Full-order cost fields stay blank when only a partial fill is observable.
    best_cost = direction * (vwap - best) * quantity if complete else None
    mid_cost = direction * (vwap - midpoint) * quantity if complete else None
    best_pct = 100 * best_cost / (best * quantity) if complete else None
    return dict(
        side=side, target_notional_usdt=notional, mid_price=midpoint,
        best_price=best, target_quantity=quantity, filled_quantity=filled,
        coverage_pct=100 * filled / quantity,
        status="complete" if complete else "insufficient_depth",
        levels_used=used, filled_quote_usdt=quote, vwap=vwap, worst_price=worst,
        slippage_usdt_vs_best=best_cost, slippage_pct_vs_best=best_pct,
        slippage_bps_vs_best=best_pct * 100 if complete else None,
        slippage_usdt_vs_mid=mid_cost,
        slippage_pct_vs_mid=100 * mid_cost / notional if complete else None,
    )


def analyze_files(paths, output, amounts):
    """Stream snapshots to CSV, publishing the result only after complete success."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for path in paths:
                path = Path(path)
                opener = gzip.open if path.suffix == ".gz" else open
                with opener(path, "rt", encoding="utf-8") as source:
                    for line_number, line in enumerate(source, 1):
                        if not line.strip():
                            continue
                        try:
                            snapshot = json.loads(line)
                            bids, asks = parse_book(snapshot["depth"])
                            midpoint = (bids[0][0] + asks[0][0]) / 2
                            metadata = {key: snapshot[key] for key in FIELDS[:5]}
                            for side, levels in (("buy", asks), ("sell", bids)):
                                for amount in amounts:
                                    writer.writerow({
                                        **metadata, **estimate(levels, side, amount, midpoint),
                                    })
                                    count += 1
                        except (ValueError, KeyError, TypeError) as exc:
                            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if not count:
            raise ValueError("No snapshots found in input files")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="Recorded .jsonl[.gz] files or run directories")
    parser.add_argument("--amounts", nargs="+", type=positive,
                        default=[100, 500, 1000, 5000, 10000, 50000, 100000],
                        help="USDT notionals at midpoint (not margin)")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV")
    args = parser.parse_args(argv)
    paths = set()
    for item in args.inputs:
        if item.is_dir():
            paths.update(path.resolve() for path in item.rglob("*.jsonl.gz"))
        elif item.is_file() and (item.name.endswith(".jsonl.gz") or item.suffix == ".jsonl"):
            paths.add(item.resolve())
        else:
            parser.error(f"Not a recording file or directory: {item}")
    if not paths:
        parser.error("No .jsonl.gz recordings found")
    if args.output.suffix.lower() != ".csv":
        parser.error("--output must have a .csv extension")
    try:
        count = analyze_files(sorted(paths), args.output, args.amounts)
    except (OSError, ValueError, EOFError) as exc:
        parser.exit(1, f"Analysis failed: {exc}\n")
    print(f"Analyzed {len(paths)} files; wrote {count} rows to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

