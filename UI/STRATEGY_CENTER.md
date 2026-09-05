# Strategy Center architecture

Strategy Center is one deployable application with a single frontend and a
single FastAPI backend. Business code remains separated by vertical module:

```text
UI/quant-ui                         one Vite SPA
UI/backend/main.py                  one FastAPI application
UI/backend/modules/experiments      experiment router and analysis service
UI/backend/modules/backtests        saved-backtest router and detail service
UI/backend/modules/labels           label router and preparation service
UI/backend/modules/live             in-memory live snapshot ingestion and query
```

The browser uses same-origin `/api` requests. During development Vite proxies
those requests to port 8000. After `npm run build`, FastAPI serves the SPA from
`UI/quant-ui/dist` as part of the same deployment unit.

## Modules

- **Labels** loads the current `output/data`
  configuration and forward split, allows editing the complete `BaseDefine`
  configuration, and invokes `preparation.main` only after Generate is chosen.
  Generation writes to a staging directory and publishes the complete artifact
  set only after it has been read and validated successfully.
- **Backtests** reads the fixed standalone payload published by
  `trade/runner/backtest_runner.py`. It can also assemble the same detail DTO
  from a selected experiment's saved report and preparation artifacts without
  running inference or a backtest.
- **Experiments** recursively loads `reports.jsonl` from one or more selected
  folders, then provides generic filtering, grouping, comparison, equity, and
  a hand-off to Backtests.
- **Live** is the default route. Live runners publish one complete current
  snapshot per second to `/internal/live/snapshots`; the backend keeps only the
  latest snapshot in process memory and serves the Strategy List and detail
  views from `/api/live`. No live monitoring data is persisted.

All user-selected experiment paths and derived artifacts must remain below
`REPORTS_ROOT`. Dataset registrations are disposable in-memory references; a
backend restart requires loading the selected folders again.

Run one backend worker. Live snapshots, experiment dataset registrations, and
the Labels generation lock are process-local by design; the documented Uvicorn
command uses one worker.

## Live runner monitoring

Only strategies with `run_live: true` are loaded and published. Configure the
runner-level destination once in the live configuration:

```json
{
  "runner_id": "runner-main",
  "monitoring": {
    "enable": true,
    "publish_url": "http://127.0.0.1:8000/internal/live/snapshots",
    "publish_interval_seconds": 1,
    "request_timeout_seconds": 0.5
  }
}
```

`runner_id` belongs only at the configuration root, not inside `monitoring`.
It identifies the runner for output directories, execution records, and monitoring.
The resolution order is `--runner-id`, `LIVE_RUNNER_ID`, the top-level
`runner_id`, then the default `live-runner`.

`--publish-url`, `--runner-id`, `LIVE_MONITORING_PUBLISH_URL`, and
`LIVE_RUNNER_ID` can override deployment-specific values. Set
`monitoring.enable` to `false` to disable dashboard collection and publishing,
even when a publish URL is supplied through the CLI or environment. The value
must be a JSON boolean and defaults to `true`; without a publish URL, monitoring
is disabled. Monitoring runs in a separate daemon thread. Dashboard or HTTP
failures are caught, and failed snapshots are discarded rather than retried.
Dashboard collection can still contend with trading for venue request locks.

For multiple machines, give every logical runner a stable, unique `runner_id`
and point all runners at the same backend URL. Every process start also receives
an automatic `runner_instance_id`; sequence numbers reject out-of-order updates,
and a newer instance replaces the previous instance of the same runner. Strategy
IDs must be globally unique across runners. A live duplicate is rejected with
HTTP 409, and a runner with no accepted snapshot for five seconds is shown as
unavailable.

## Run

```bash
uvicorn UI.backend.main:app --host 0.0.0.0 --port 8000
cd UI/quant-ui
npm run dev
```

For a single-process deployment, build the frontend first and open port 8000:

```bash
cd UI/quant-ui
npm run build
cd ../..
uvicorn UI.backend.main:app --host 0.0.0.0 --port 8000
```
