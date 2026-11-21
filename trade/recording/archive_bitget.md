# Bitget live archive

Run these commands from the repository root. Both commands use the existing Python environment.

```bash
archive_dir=/root/work/quant_output/live_runner/runner-main/bitget_performance

.venv/bin/python -m trade.recording.archive_bitget \
  --config /root/work/LiveTrading/live_config.json \
  --db "$archive_dir/history.sqlite3" \
  --trace-dir /root/work/quant_output/live_runner/runner-main/execution_traces

.venv/bin/python -m trade.recording.report_bitget \
  --db "$archive_dir/history.sqlite3" \
  --csv "$archive_dir/daily_realized_nav.csv"
```

The collector selects enabled live Bitget Classic USDT futures accounts in the
configuration. The first import requests the exchange's rolling 90-day window
with a small margin for request timing. Later imports resume one millisecond
after the last successfully queried window per account and feed. Set
`--overlap-days 7` to refresh recent records that arrived after their window
was queried.
Records are keyed by account, feed, and exchange record ID. SQLite commits each
completed window with its checkpoint. A failed window is retried on the next
run, and a gap beyond the exchange retention window is recorded in
`coverage_gaps`. Job output counts fetched rows, including overlap.

The local trace import re-reads the source CSVs and inserts only new rows.
Keep execution tracing enabled to capture decision-time prices, local submission
and acknowledgement times, and errors. Exchange history has exchange-side
timestamps and fill prices. Decision-based slippage needs a local price reference.

| Table or view | Contents |
| --- | --- |
| `archive` | Raw exchange payloads for bills, fills, orders, trigger-order history, equity and positions |
| `local_trace` | Raw execution, order, fill and event rows from the runner |
| `fill_analysis` | One row per exchange fill, with order creation time, fill time, liquidity role, fees and realized PnL |
| `execution_analysis` | Strategy execution timing, prices, quotes and existing slippage metrics |
| `fill_execution_links` | Exact account and exchange order ID matches between exchange fills and executions |
| `equity_analysis` | Timestamped account equity, unrealized PnL and available balance |
| `checkpoints`, `coverage_gaps` | Requested history windows and retention gaps |

For regular collection, run the full command hourly and use the same command
with `--snapshots-only` every five minutes if finer account equity and position
sampling is useful. Snapshot history starts with the first successful request.
For coverage across runner restarts, provide the persistent runner trace
directory. The SQLite database contains private account data. Use SQLite's
backup API or the CLI `.backup` command for a consistent copy of a live WAL database.

The report uses complete local days and defaults to Asia/Shanghai and currently
configured accounts. Use `--start`, `--end`, and repeatable `--account` to
select a period or subset. It adjusts transfers and calculates realized NAV,
annualized Sharpe with zero risk-free rate, Calmar, and both daily and bill-event
drawdown. It rejects incomplete coverage and unclassified bill types. The
equity and position snapshots support separate mark-to-market analysis.

To extend the analysis, collect decision and submission-time order book depth,
mark and index prices, and contract specification changes. These enable depth
and market-impact analysis, historical exposure checks, and validation of
quantity and price rounding. Signal context and exchange rejection reasons can
help explain changes in fill rates alongside the existing execution trace.
