# Trading main-flow checks

Run from the repository root:

```bash
.venv/bin/python -m pytest -q
```

Default discovery is limited to `test/regression/`. It contains four small suites:

- `test_initial_state.py`: initialize the account baseline during startup.
- `test_live_runner.py`: receive a closed candle, calculate its signal, dispatch
  through the account worker and confirm a simulated exchange fill.
- `test_bitget_venue.py`: configuration/venue construction, entry with protection,
  closing, and execution-report reconciliation.
- `test_ctrader_trading.py`: submit market/limit buy/sell orders with protection.

Exchange I/O and model output are simulated. These tests check the application
flow; they do not prove current broker connectivity or execute real trades.
Bug-specific tests for dashboard collection, timing, locks, page leases,
persistence and other isolated cases have been removed. Fix verification should
normally be temporary; extend an existing main-flow test only when the flow changes.

## Explicit validation tools

Manual exchange checks and offline model/data validators remain available under
`test/manual/` and `test/offline/`. They are excluded from default pytest discovery.
They require explicit invocation and, where applicable, credentials or model data.

```bash
.venv/bin/python -m test.manual.test_bitget_connectivity --help
.venv/bin/python -m test.manual.test_bitget_account_diagnostics --help
.venv/bin/python -m test.manual.test_bitget_market_round_trip --help
.venv/bin/python -m test.offline.test_verify_prediction_traces --help
.venv/bin/python -m test.offline.test_strategy_validation --help
.venv/bin/python -m test.offline.test_random_prediction_validation --help
```

The market round-trip tool places real orders only with its explicit execution
option. Normal pytest runs never invoke these tools.

For consolidated runner traces containing multiple warm-up segments and archived
model identities, run the offline parity audit explicitly:

```bash
.venv/bin/python -m test.offline.test_runner_main_prediction_parity \
  --trace-dir ../quant_output/live_runner/runner-main/prediction_traces \
  --output-dir ../quant_output/analysis/live_prediction_parity
```

The audit compares every recorded prediction with batch inference, samples eight
causal rolling-window predictions per instance through `LiveRunner._predict`,
and recomputes all strict mismatches with single-sample inference. It saves input
snapshots, artifact hashes, comparison CSVs, and `summary.json`. Missing archived
models are reported as unresolved. Exit status is 1 for strict mismatches, 2 for
unresolved models without mismatches, and 0 only for complete parity. This checks
model outputs using preserved artifacts; it does not reconstruct every original
process startup or simulate execution.

## Order failure notifications

All live venues use the root `telegram_token` credential path from the live
configuration. Failure notifications do not depend on `execution_trace`.
`event=order_execution_failed` includes strategy identity, execution/order IDs,
status, quantity, and the available failure reason. It covers precheck rejection,
submission exceptions or unknown outcomes, rejected child orders, and subsequent
exchange rejection/expiry events. Entry cancellations are reported; routine exit
protection cancellations are not treated as failed submissions.

Delivery runs outside the trading and exchange callback threads, with up to three
attempts. Duplicate order/status failures are suppressed within the runner process.
An exhausted delivery is logged with the full event for investigation; this is not
a persistent delivery queue and it cannot guarantee receipt during an outage.
