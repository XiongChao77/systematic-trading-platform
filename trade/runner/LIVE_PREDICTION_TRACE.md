# Live prediction trace

Enable persistent CSV traces in `LiveTrading/live_config.json`:

```json
{
  "prediction_trace": true,
  "execution_trace": true
}
```

Each runner uses a stable output directory across starts:

```text
quant_output/live_runner/runner-main/
  initial.json
  order_owners.json
  logs/session.log
  execution_traces/executions.csv
  execution_traces/orders.csv
  execution_traces/fills.csv
  execution_traces/events.csv
  prediction_traces/<data_source>_<trading_type>_<symbol>_<interval>.csv
```

The log and execution journals append across starts. Execution rows retain the
process `run_id`; each startup also writes its run ID to the log. Exact execution
fact replays ignore run ID, and detailed fills are deduplicated by venue/account,
symbol, order and deal (using `trader_login` for cTrader). Status transitions remain separate observations. Raw
execution summaries are not final lifecycle totals: use
`python -m trade.recording.merge_execution_traces PATH/TO/execution_traces`
to reconcile events, link recovered orders and remove aggregate fills superseded
by detailed deals before computing execution statistics.

Execution files have separate responsibilities:

- `executions.csv` contains submission-response reports. Its `completed_at_utc`
  is local report creation time, not proof that the order has finished.
- `orders.csv` contains child-order response snapshots. An accepted response
  acknowledges acceptance only; an empty response does not establish acceptance.
- `events.csv` contains venue lifecycle observations only. Reports no longer
  fabricate `:initial:` events, and reconciliation excludes previously recorded
  initial events from lifecycle reduction.
- `fills.csv` contains actual deals or explicitly marked aggregate fills.

Reconciliation reduces each child order before calculating execution status.
Terminal order states cannot regress to accepted because of a delayed response.
Cancellation-request rejection is not order rejection. A filled child plus a
rejected child is partial execution; a rejected child plus an accepted child
still has an active order. Protection orders are separated from entry orders,
even if the venue echoes the entry client ID. Actual order/deal direction takes
precedence over sparse historical event fields. Reconciled `completed_at_utc`
is populated from terminal venue observations only when all known children have
finished; response-only terminal reports may have no venue completion timestamp.

Prediction files use one row per candle open time. Repeated warm-up rows are
ignored; live rows replace warm-up market data and fill empty prediction cells.
Existing nonempty predictions are preserved. Adding a model extends the header
without removing previous model columns. New candles append normally; enriching
an existing candle, inserting older data, or extending the header atomically
rewrites the same path. Prediction files are therefore persistent tables, not
strictly append-only journals.

Startup imports historical timestamped directories before opening the live
writers. Source checkpoints avoid reimporting unchanged files; changed CSVs are
replayed through deduplication, and logs import only their new complete lines.
Original session directories remain untouched. Open-file sources preserved by
the previous identity migration are included automatically, with ID conversion.
Stop the old runner before starting the new one; its final tail is imported at
startup. The new writer lock prevents two updated runners sharing one output
folder, but cannot lock an old executable that does not implement this protocol.
`initial.json` and `order_owners.json` remain directly in the runner directory.

To import newly appended historical tails without starting trading:

```bash
python -m trade.recording.persistent_output ../quant_output/live_runner/runner-main
```

Use the actual runner directory for your working directory. The maintenance
command acquires the same writer lock as the runner and reports imported source
bytes, recovered snapshots, and pending incomplete log bytes. Unfinished log
lines remain pending until a newline arrives; they are never silently discarded.
Additional `.log` files and numbered `.log.N` files under historical log folders
are included. Preserved backup sources do not require an existing session copy.
If a removed migration backup leaves an older transformed snapshot, its offset
is reset only after verifying that the snapshot already exists byte-for-byte in
the merged log. Unknown replacement content fails explicitly.

Market columns use the canonical raw candle schema. Predictions use wide
strategy-specific columns:

```text
<instance_id>__pred
<instance_id>__pred_prob
<instance_id>__net_score
```

The trace remains valid raw market data for batch inference. Extra trace and
prediction columns are ignored by feature generation.

To replay a single-feed trace through the live inference path and compare it
with the recorded predictions:

```bash
python -m trade.runner.live_prediction_replay \
  --config LiveTrading/live_config.json \
  --market-trace /path/to/feed_trace.csv \
  --output /tmp/replay_predictions.csv \
  --backtest-predictions /path/to/feed_trace.csv \
  --instance-id INSTANCE_ID
```

For exact session replay, use the preserved per-session historical trace.
A consolidated trace combines multiple warm-up windows and process states, so
its `is_warmup` prefix cannot reconstruct every original startup. Prediction class
values must match exactly; probability and net-score values use the comparator's
numeric tolerances.
