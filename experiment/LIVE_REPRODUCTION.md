# Reproduce strategies configured for live trading

Each strategy in `live_config.json` must provide `config_path`, `hash`, and
`model_path`. Paths are relative to the live configuration file. The runner
loads parameters from that strategy's JSONL report; there is no global report
source. Strategies in different files can use the same hash independently.

Check report selection, model files, model types, and feature configurations:

```bash
.venv/bin/python experiment/strategy_validation.py \
  --live-config LiveTrading/live_config.json --dry-run
```

Replay both `long` and `forward` with the model artifacts specified in the live
configuration, without retraining or connecting to any live venue:

```bash
.venv/bin/python experiment/strategy_validation.py \
  --live-config LiveTrading/live_config.json \
  --output-dir /tmp/live-reproduction
```

By default, `--settings original` retains the report's broker and compound
settings so the comparison measures historical reproducibility. To measure the
effect of the configured live initial equity, fees, leverage, and compound
override, use `--settings live`. The outputs list the configuration differences;
different settings may legitimately produce different returns. This is an
offline backtest with the existing simulator, not a simulation of every
exchange-specific live execution rule.

Use repeated `--strategy-id ID` arguments to select individual strategies and
`--periods forward` or `--periods long` to limit replay. Without explicit IDs,
all `run_live=true` entries are selected, including disabled execution entries.
An explicit ID may select a `run_live=false` entry for offline testing.

The command regenerates prepared data from the original market configuration,
checks the market CSV SHA-256 against the original report, and generates new
prediction caches inside the output directory. It never writes to the archived
model directory. Identical replay jobs shared by multiple live strategies are
executed once and reported for every strategy ID.

Outputs:

- `live_reproduction_comparisons.jsonl`: status, configuration differences,
  per-period original and reproduced metrics, signed deltas, and errors.
- `live_reproduction_metrics.csv`: flat metric comparison table.
- `live_reproduction_summary.json`: counts of matches, mismatches, and errors.
- `backtests/<execution-id>/report.json` and `report_details.json`: regenerated
  reports and daily account details.

Compared metrics are gross return, CAGR, ending equity, Sharpe, Calmar, and
maximum drawdown. Period start/end must also match. Defaults are `--rtol 1e-6`
and `--atol 1e-8`; missing or nonfinite metrics do not count as a successful
reproduction. A completed run exits with status 1 if any comparison mismatches
or fails, while retaining its diagnostic outputs. Preparation/model/data errors
are distinguished from metric mismatches.
