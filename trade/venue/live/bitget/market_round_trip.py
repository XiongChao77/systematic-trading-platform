"""One small futures market entry followed immediately by a market close."""

import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
import json
import logging
import os
from pathlib import Path

from trade.venue.live.bitget.bitget_venue import BitgetVenue


def order_summary(order):
    return (
        {
            key: order.get(key)
            for key in (
                "orderId",
                "clientOid",
                "state",
                "baseVolume",
                "priceAvg",
                "fee",
                "feeDetail",
                "profit",
                "cTime",
                "uTime",
            )
        }
        if isinstance(order, dict)
        else None
    )


def error_chain(venue, error):
    errors, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        errors.append(
            {
                "type": type(error).__name__,
                "code": getattr(error, "code", None),
                "message": venue._redact(error),
            }
        )
        error = error.__cause__ or error.__context__
    return errors


def round_trip(venue, emit, *, notional=Decimal("10"), execute=False):
    """Open at most once, with bounded cleanup of a position created by this test."""
    if not notional.is_finite() or not Decimal("0") < notional <= Decimal("10"):
        raise ValueError("Test notional must be positive and at most 10 USDT")
    if venue._position() is not None:
        raise RuntimeError("Test refused: existing position")
    venue._assert_no_open_orders()
    bid, ask = venue.get_bid_ask()
    ask = Decimal(str(ask))
    step = venue.quantity_step
    quantity = (notional / ask / step).to_integral_value(rounding=ROUND_DOWN) * step
    if (
        quantity < venue.minimum_quantity
        or quantity > venue.maximum_market_quantity
        or quantity * ask < venue.minimum_notional
    ):
        raise ValueError("10 USDT budget cannot satisfy current contract limits")
    emit(
        "preflight",
        {
            "flat": True,
            "pending_orders": False,
            "side": "buy",
            "quantity": str(quantity),
            "ask": str(ask),
            "bid": str(bid),
            "estimated_notional_usdt": str(quantity * ask),
            "notional_budget_usdt": str(notional),
            "margin_mode": venue.margin_mode,
            "hedge_mode": venue.hedge_mode,
            "note": "Market fill notional can differ from the quote-time estimate.",
        },
    )
    if not execute:
        return {"status": "preflight_only", "flat": True}

    venue.read_only = False
    failure = None
    flat = False
    try:
        emit("entry_attempt", {"quantity": str(quantity)})
        result = venue.submit_order(float(quantity), True)
        emit("entry_result", order_summary(result))
    except Exception as exc:
        failure = error_chain(venue, exc)
        emit("entry_error", {"errors": failure})
    finally:
        # Never submit another entry. Every retry below can only reduce a position.
        for attempt in range(1, 4):
            try:
                position = venue._position()
                if position is None:
                    venue._assert_no_open_orders()
                    flat = True
                    break
                size = Decimal(str(position["total"]))
                if position["holdSide"] != "long" or size > quantity:
                    emit(
                        "unexpected_position",
                        {"side": position["holdSide"], "quantity": str(size)},
                    )
                    break
                emit("close_attempt", {"attempt": attempt, "quantity": str(size)})
                result = venue.close_position(size=float(size))
                emit("close_result", order_summary(result))
            except Exception as exc:
                emit(
                    "cleanup_error",
                    {"attempt": attempt, "errors": error_chain(venue, exc)},
                )
        if not flat:
            try:
                flat = venue._position() is None
                venue._assert_no_open_orders()
            except Exception as exc:
                flat = False
                emit("verification_error", {"errors": error_chain(venue, exc)})
    return {
        "status": (
            "completed"
            if flat and failure is None
            else "entry_failed" if flat else "cleanup_unconfirmed"
        ),
        "flat": flat,
        "entry_error": failure,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-path", required=True)
    parser.add_argument("--strategy-id", required=True)
    parser.add_argument("--symbol", default="DOGEUSDT")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually place one entry and close its fills",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(
        output, "x", opener=lambda path, flags: os.open(path, flags, 0o600)
    ) as handle:

        def emit(event, data):
            row = {
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "strategy_id": args.strategy_id,
                "symbol": args.symbol,
                "event": event,
                "data": data,
            }
            encoded = json.dumps(row)
            handle.write(encoded + "\n")
            handle.flush()
            print(encoded, flush=True)

        venue = None
        try:
            venue = BitgetVenue(
                args.key_path,
                args.symbol,
                "manual-market-round-trip",
                logger=logging.getLogger("bitget.market_test"),
                read_only=True,
                enable_user_stream=False,
            )
            result = round_trip(venue, emit, execute=args.execute)
        except Exception as exc:
            result = {
                "status": "failed",
                "error": venue._redact(exc) if venue else type(exc).__name__,
            }
        finally:
            if venue is not None:
                venue.shutdown()
        emit("summary", result)
        return 0 if result["status"] in {"completed", "preflight_only"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
