# Live dashboard collection

The publisher sends the latest cached state every `publish_interval_seconds`
(one second in `LiveTrading/live_config.json`). Each strategy has one background
collector, with at most one refresh in progress. A blocked collector cannot
hold up another strategy or the publisher. Trading never reads this dashboard
cache.

Every strategy carries `dashboard_age_seconds`, measured from the beginning of
its collection with a monotonic clock. The backend adds elapsed time since
receipt and expires data after five seconds. Repeated heartbeats do not renew
old account values. Missing or stale data affects only that strategy. Runner
heartbeats retain their separate five-second expiry.

## Exchange requests

- cTrader submits trader, unrealized PnL and reconciliation requests together.
  Balance, used margin, position, PnL and position opening time reuse those three
  responses. Quotes still come from the existing quote stream. The transport
  checks its queues every 20 ms and sends trading requests immediately when
  capacity permits. Dashboard work has lower priority and leaves one slot per
  second for trading. A rolling window preserves the installed SDK's five
  messages per second per connection; this remains below cTrader's documented
  50 non-historical requests per second and within its five historical requests
  per second limit. Heartbeats retain the SDK's immediate handling.
- Binance fetches account and position data concurrently through two dedicated,
  reusable dashboard sessions. These reads, including dashboard position-history
  reconstruction, do not acquire the trading HTTP lock. Existing stream-derived
  protective-order prices are reused. Account and position responses remain
  necessary because account data alone lacks fields used by the position card.
  An IP-weight budget shared by venue instances in this process reads limits from
  `exchangeInfo` and observes response weight headers. Dashboard reads stop at
  80% of that budget and respect rate-limit retry delays; trading retains its
  existing request behavior. Separate processes on the same IP are visible only
  through response headers, so their in-flight usage cannot be reserved locally.
- Other venues use the shared dashboard interface and independent collection;
  their exchange-specific request implementations retain their existing behavior.

## Deployment and validation

Restart the backend and runner together for the snapshot freshness field, and
rebuild the frontend for the per-strategy unavailable badge. No running trading
process is restarted by these code changes.

`Dashboard collection timing` reports each collector's duration. `Live monitoring
timing` reports cached payload assembly and upload time. Binance and cTrader API
timings retain endpoint/request names, queue or lock waits, and response duration.
Tests use simulated exchange responses; compare the new session's timings to the
previous eight-second collection cycle after deployment.

Sources: [cTrader limits](https://help.ctrader.com/open-api/),
[Binance request weights](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/general-info).
