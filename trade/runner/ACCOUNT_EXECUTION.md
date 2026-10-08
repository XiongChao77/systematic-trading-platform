# Live account execution

The runner computes features and model predictions serially. It submits a copied
market view and prediction to a bounded execution queue for each account. Models
and feature generators never run in these account workers.

Each account has one worker, shared by all its strategies and symbols. Different
accounts may query venues, make strategy decisions and execute orders concurrently.
Account identity is the venue plus account ID, or trader login for cTrader. MT5
execution is serialized across its process-wide terminal connection. Existing
venue locks and shared connection limits still apply.

cTrader connections are shared by canonical credential directory and environment
(`live` or `demo`). A single runner can use multiple credential directories;
each directory resolves and authenticates only accounts granted by its own token.
Strategies using the same directory and environment reuse one connection. All
cached connections are closed during runner shutdown.

Each queue holds at most 64 waiting tasks. A full queue rejects the incoming task
and logs its strategy identity; it never blocks inference or retries an order.
A failed execution logs context and attempts notification through the strategy's
existing notifier. It does not stop other accounts or retry the failed candle.

The existing 30-second market-age limit starts when the feed callback receives
the event, including time spent computing and waiting in either queue. The worker
checks age before querying the account, before strategy processing, and before
order execution. cTrader checks again after margin prechecks. An already submitted
order continues to completion even if the signal subsequently expires.

Received candle progress is tracked separately from calculation progress so a
queued candle is not reported as missing. Duplicate candle events are ignored.
Prediction trace rows remain aggregated and written by the calculation path;
execution reports use the existing recorder queue. `wait_for_execution()` is
available to offline callers that need to await submitted strategy tasks.

Shutdown stops accepting work, cancels waiting account tasks and waits for active
tasks before finalizing strategies or closing venues and recorders. The running
process must be restarted to load these changes.

## Stage timing

INFO logs with `Runner stage timing` correlate work using `identity`,
`candle_open_time_utc`, and `thread`. All durations use a monotonic clock in
milliseconds. `outcome` includes completed, skipped, stale, invalid, or failed;
failed calls retain their elapsed time and exceptions keep their existing behavior.
Only stages actually attempted appear in the log.

- `event`: receipt age at entry includes event queue and calculation lock wait;
  total includes feed snapshot, all serial calculations, submissions and prediction
  trace recording, but does not wait for account workers.
- `calculation`: prepare, features, inference, optional prediction callback, and
  market view conversion for one strategy. Receipt age at entry also includes
  earlier strategies' serial calculation time.
- `account_execution`: account_queue_ms measures submission-to-worker wait;
  total_ms measures worker time, including dispatch, live_record and any failure
  notification. Receipt age at exit is the elapsed time since feed receipt.
- `dispatch`: position, equity, daily_reset, decision, execute_intent, and
  execution_record. execute_intent includes prechecks, venue locks, requests,
  order completion waits and any retries performed by the existing venue code;
  it is not pure network latency. Existing cTrader API timing logs further break
  down individual requests, including SDK queueing in sdk_round_trip_ms.

These phases are nested and can overlap across accounts. Do not sum event,
calculation, dispatch and account_execution totals. Skipped work has no timings
for unattempted stages; shutdown-cancelled queued tasks never enter a worker and
therefore do not produce account_execution timings.

For cTrader execution inputs, the account worker concurrently requests Reconcile,
Trader and PositionUnrealizedPnL using three short-lived I/O workers. All responses
must succeed before constructing the observation; results are fresh and do not
come from dashboard caches. A failed query prevents strategy decision/order
execution. The receipt-age check still runs after the complete batch. Requests
are independent reads, not an atomic exchange snapshot. Model inference remains
serial, and order execution stays on the existing account worker.

The dispatch timing field `account_snapshot_ms` measures the whole cTrader query
batch, including worker setup and result conversion. Individual request timings
remain available in `cTrader API timing`; their durations overlap and must not be
summed to estimate batch latency. Other venues retain position/equity stage timers.

## cTrader connection pauses

A disconnected cTrader session gates feature generation, inference, queued
execution and order submission for its associated strategies. Strategy `enable`
settings are preserved. Shared market-data feeds continue receiving and caching
candles, including while every cTrader strategy on a feed is paused.

After three recovery failures, the connection stops SDK reconnects, heartbeats
and account/dashboard reads while retaining account references and subscriptions.
One background worker waits one hour, then attempts connection and authentication
once. Failed hourly attempts wait another hour; closing the runner cancels this
wait. Recovery restores application and account authentication, reads current
positions/orders and trader state, and restores spot subscriptions before trading
can resume. The next fresh strategy observation also synchronizes holding time.
Signals received before recovery are discarded rather than replayed.

Telegram receives one pause message after the three failures and one recovery
message after a successful hourly attempt, grouped by connection and configured
recipient. Hourly failures only write local logs. A subsequent outage starts a
new notification episode. Manually or risk-disabled strategies remain disabled.
The frontend displays a paused connection and its next retry time without
triggering exchange reads; the market-data feed is not paused or restarted.
