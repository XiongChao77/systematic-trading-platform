# Setup and Running

This repository is primarily a research and engineering project rather than a turnkey trading application. The commands below assume that repository-specific data paths, experiment tasks, and artifact locations have been configured. Run all Python commands from the repository root.

## Environment

The Python project requires Python 3.12 or later. Install the research, modeling, backtesting, and UI dependencies with pip:

```bash
python -m pip install numpy scipy pandas scikit-learn matplotlib seaborn plotly \
  notebook jupyterlab ipykernel statsmodels xgboost lightgbm tqdm joblib \
  requests beautifulsoup4 torch pytorch-ignite colorlog backtrader pyarrow \
  numba GitPython service-identity fastapi uvicorn websocket-client

python -m pip install pybit ctrader-open-api  # Live trading venues
```

The Strategy Center frontend also requires Node.js and npm:

```bash
cd UI/quant-ui
npm install
```

Live venues may require platform-specific broker software, API credentials, and account configuration in addition to the Python dependencies.

## Running

### 1. Data Preparation

Download Binance market data, then clean, transform, label, and export the training dataset:

```bash
python -m data_process.download.binance.download_from_zip \
  --symbols BTCUSDT ETHUSDT \
  --intervals 15m 30m 1h
python -m data_process.preparation
```

Preparation behavior is selected through the active configuration in `data_process/config.py`.

### 2. Model Training

Train the configured model and write its checkpoints and evaluation artifacts:

```bash
python -m model.train
```

### 3. Backtesting

Evaluate the trained model through the current backtest runner:

```bash
python -m trade.runner.backtest_runner
```

### 4. Batch Experiments and Validation

Run the combined preparation, training, and simulation workflow for the configured experiment tasks:

```bash
python -m experiment.batch_experiments
```

Reproduce selected experiments with both a fresh training run and an independent
backtest that loads the archived original model without training:

```bash
python -m experiment.batch_experiments --valid
```

Training-reproduction reports are written directly under `valid_train_out`.
Original-model backtest reports and their comparison are written under
`valid_train_out/backtest_reproduction`. This reproduction uses CPU inference so
changing the available GPU does not change the validation execution path.

Generate the configured offline experiment visualizations:

```bash
python -m experiment.trigger_direction_report_view
```

Run cross-period and cross-asset validation for selected candidates:

```bash
python -m test.offline.test_strategy_validation \
  --selected-configs path/to/selected_configs.jsonl
```

Cross-test model artifacts are loaded from the original model archive beside
`selected_configs.jsonl`; retrained validation artifacts are never promoted into
the cross-test output. `model_archive_manifest.jsonl` records the source path and
SHA-256 checksum for each copied model. The backtest reproduction report and its
comparison are copied into the same cross-test directory.

The batch runner creates a source snapshot and requires a clean Git worktree by default. Experiment definitions are loaded from `experiment/task_constructors.py` when present, with `experiment/task_constructors_example.py` serving as the repository example.

### 5. Strategy Center

Strategy Center combines Labels, Backtests, Experiments, and Live Monitoring in one Vite frontend and one FastAPI backend. Start the development services in separate terminals:

```bash
uvicorn UI.backend.main:app --host 0.0.0.0 --port 8000
```

```bash
cd UI/quant-ui
npm run dev
```

Open `http://localhost:5173`. Run a single backend worker because live snapshots, loaded experiment datasets, and label-generation coordination are stored in process memory.

### 6. Live Trading

A validated strategy can be run through the unified live runner. Create a deployment-specific configuration for strategy artifacts, venues, credentials, symbol mappings, monitoring, and risk limits; `trade/runner/live_config_example.json` shows the expected structure.

For cTrader risk alerts, the root-level `telegram_token` path may point to a file containing `chat_id : bot_token`, a JSON file containing `bot_token` and `chat_id`, or a directory containing separate `bot_token` and `chat_id` files.

`run_live` controls whether a strategy is constructed at startup. `enable` controls its initial runtime state; a constructed strategy can also be changed without stopping shared feeds by calling `runner.set_strategy_enabled(strategy_id, enable)`.

```bash
python -m trade.runner.live_runner \
  --config path/to/live_config.json
```

Use a demo or paper-trading account before committing capital. Passing historical validation reduces overfitting risk but does not guarantee future performance. Live monitoring and risk limits remain mandatory.

### 7. Merge Execution Traces

Each live-run process writes immutable execution traces inside its own run directory. Merge every run below a runner directory with:

```bash
python -m trade.recording.merge_execution_traces \
  /path/to/quant_output/live_runner/runner-main
```

The command creates `merged_execution_traces` below the input directory with finalized executions, child orders, fills, lifecycle events, and a merge report. Re-running the command is idempotent. To write elsewhere, pass `--output-dir /path/to/output`.
