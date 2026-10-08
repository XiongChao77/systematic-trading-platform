# Page-driven dashboard collection

The strategy list and strategy detail already have separate read endpoints.
Exchange polling now follows those page reads instead of collecting full details
for every strategy unconditionally.

Each mounted page sends an independent `viewer_id` with its existing GET request.
The backend renews a ten-second lease for either the overview or one strategy.
Multiple tabs combine their demands; detail takes precedence for its strategy.
Hidden, unmounted and closing pages release their lease through
`DELETE /api/live/viewers/{viewer_id}`. A lost release expires automatically.

The runner keeps a dedicated long poll on
`GET /internal/live/demand/{runner_id}?version=...`. A new page or scope change
wakes this request immediately, independently of snapshot publication. The runner
then wakes the affected collection workers; the first read does not wait for the
normal collection interval. Long polls also return after three seconds to renew
active leases. Ordinary page refreshes renew leases without retriggering collection.
A single demand reader applies changes in order, separately from snapshot POSTs.

Each lease has its own remaining duration, so an open detail page cannot keep an
abandoned overview alive. If the backend becomes unreachable, runner leases
expire too. The publisher sends periodic heartbeats even when no venue reads are
requested. Already running reads may finish. Trading and execution reconciliation
remain independent of monitoring leases.

- Overview: balances, margin utilization and strategy unrealized PnL. No holding
  history/timing queries. Bitget also skips protective-order queries, using just
  account and position reads. The API response retains list metrics and totals.
- Detail: full dashboard plus holding timing, only for requested strategies.
  The UI indicates loading until a detail snapshot has arrived.
- No visible pages: metadata and heartbeat publications only; no new dashboard
  exchange reads after demand expires or is released and processed.

cTrader overview still requires Trader, PositionUnrealizedPnL and Reconcile:
margin utilization needs positions and the account margin rule, and strategy PnL
requires matching account-wide PnL entries to owned positions. Existing response
caching and in-flight reuse remain in place. This change reduces the number of
accounts polled on a detail-only page but does not remove those three required
reads when the overview is open.

Deploy the frontend and restart the UI backend and runner together. The backend
remains single-process and in-memory.

## cTrader latency diagnostics

`cTrader API timing` is emitted at INFO for every completed/failed request,
including successful dashboard reads. Correlate it with `Dashboard read timing`
and `cTrader dashboard timing` using `request_id`, account and environment.
Dashboard deadline expiry does not cancel an unfinished SDK call; the same ID
can therefore appear in a timeout log followed by a successful request log.
Cached and in-flight reads retain the original ID rather than a new unused ID.

Request stages (milliseconds):

- `connection_wait_ms`: wait for a connected client.
- `dispatch_ms`: message preparation and scheduling onto the reactor.
- `sdk_handoff_ms`: SDK submission until the protocol queues the message.
- `send_queue_ms`: protocol queueing until transport submission; includes local
  rate-limit and priority waiting. If unsent at failure, elapsed queue wait.
- `response_wait_ms`: transport submission until the SDK's message callback
  observes the response. Includes transport buffering, network, broker handling,
  reactor scheduling and envelope parsing; it is not a pure network RTT.
- `callback_ms`: message handling and response extraction until the caller is
  signalled. Includes synchronous execution-event listeners where applicable.
- `caller_resume_ms`: signalling to resumption and validation on the caller thread.

`phase` identifies the last observed stage on failure; `na` means the stage was
not reached. Queue counts reflect the latest queue observation for that request.
`window_used` is the number of slots already consumed before its send, and
`send_limit` is the local SDK connection limit. `sdk_round_trip_ms` is retained
as an aggregate, not an extra stage to add. Authentication waits happen before
these stages and produce `cTrader session wait` when at least 50 ms or failed.

`Dashboard read timing` logs unsuccessful reads at INFO and successful reads at
DEBUG. It includes cache/new/in-flight source, worker queue time, work duration,
cache age, actual blocking time and the shared batch deadline. Snapshot batches
have a 1250 ms deadline; cTrader's inner three-request batch has 1000 ms. These
nested times overlap and must not be added together. Slow (at least 500 ms) or
partial snapshots also emit `cTrader dashboard timing` at INFO.

The connection follows the [official rate limits](https://help.ctrader.com/open-api/):
50 non-historical requests per rolling second and 5 historical requests per
rolling second, each per connection. Dashboard reads may consume at most 49
non-historical slots, retaining one slot for trading. Historical requests use a
separate queue and budget, so exhausted history capacity cannot block account
reads or orders. Timing logs include `rate_class` and `historical_pending`.

Previously the SDK default of five messages per second limited dashboards to
four, causing six simultaneous overview requests to span multiple windows. The
new ordinary budget removes that local bottleneck without changing cache or
response deadlines. Actual response delays remain visible in the timing logs.

Snapshot publication is paced by `publish_interval_seconds` (currently one
second in the live configuration). Collector notifications do not trigger extra
POSTs: each tick uploads one latest combined snapshot for the runner. Slow uploads
skip missed ticks without catch-up bursts. Startup sends an immediate heartbeat;
shutdown can send a final stopped snapshot outside the normal cadence.
`Live monitoring timing` includes `publish_gap_ms` for cadence diagnostics.
The backend's `Live API timing` and Uvicorn access line describe the same HTTP
request, not two requests. Browser list/detail GET polling is separate from the
runner's POST publication schedule.


Collection and publication now both run once per second while a page is active.
Venue reads still reuse successful results for up to three seconds and reuse
in-flight requests. A newly collected result appears in the next one-second
publication; the frontend continues its own one-second polling.

Latest Signal is populated immediately after inference and before account queue
execution, with a pending decision. Execution later supplies the actual decision,
or a skipped/failed status. An older task cannot overwrite a newer candle's
signal. The signal timestamp is the candle time, not HTTP publication time.
Signals are session-local: startup does not fabricate a prediction or place an
order to populate the UI. Until the first closed-candle prediction, the detail
page explicitly displays its waiting state.
