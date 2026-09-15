"""Shared snapshot validation and CLI value parsing."""

import argparse
import math


def parse_book(payload):
    """Validate a complete REST snapshot and sort executable price levels."""
    sides = []
    for name in ("bids", "asks"):
        levels = [(float(p), float(q)) for p, q in payload[name]]
        if not levels or any(
            not math.isfinite(p) or not math.isfinite(q) or p <= 0 or q <= 0
            for p, q in levels
        ):
            raise ValueError(f"Invalid {name} depth")
        if len({p for p, _ in levels}) != len(levels):
            raise ValueError(f"Duplicate {name} prices")
        sides.append(sorted(levels, reverse=name == "bids"))
    bids, asks = sides
    if bids[0][0] >= asks[0][0]:
        raise ValueError("Crossed or locked order book")
    return bids, asks


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("Must be positive and finite")
    return result


