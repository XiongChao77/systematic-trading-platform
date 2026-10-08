"""Read-only Bitget connectivity check; never submits or cancels an order."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json

from trade.venue.live.bitget.bitget_venue import BitgetVenue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-path", required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--skip-websocket", action="store_true")
    args = parser.parse_args()
    venue = None
    try:
        venue = BitgetVenue(
            args.key_path,
            args.symbol,
            "bitget-connectivity-check",
            read_only=True,
            enable_user_stream=not args.skip_websocket,
        )
        balance = venue.get_dashboard_balance()
        position = venue.get_current_state()
        bid, ask = venue.get_bid_ask()
        fills = venue.reconcile_execution_events(
            datetime.now(timezone.utc) - timedelta(hours=24)
        )
        stream_ready = None if args.skip_websocket else venue.wait_for_user_stream(20)
        print(
            json.dumps(
                {
                    "status": "ok" if stream_ready is not False else "websocket_failed",
                    "read_only": True,
                    "symbol": venue.symbol,
                    "position_mode": (
                        "hedge_mode" if venue.hedge_mode else "one_way_mode"
                    ),
                    "margin_mode": venue.margin_mode,
                    "quantity_step": str(venue.quantity_step),
                    "price_tick": str(venue.price_tick),
                    "balance": asdict(balance),
                    "position_size": position.size,
                    "bid": bid,
                    "ask": ask,
                    "owned_fills_reconciled": fills,
                    "websocket_ready": stream_ready,
                },
                indent=2,
            )
        )
        return 0 if stream_ready is not False else 1
    except Exception as exc:
        message = venue._redact(exc) if venue is not None else str(exc)
        print(json.dumps({"status": "failed", "error": message}))
        return 1
    finally:
        if venue is not None:
            venue.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
