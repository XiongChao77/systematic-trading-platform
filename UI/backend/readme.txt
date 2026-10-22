Strategy Center uses one FastAPI application and one port.
Run exactly one backend worker because experiment datasets and label generation
coordination are maintained in process memory.

Development backend:
uvicorn UI.backend.main:app --host 0.0.0.0 --port 8000

Development frontend (install dependencies from the lockfile first):
cd UI/quant-ui
npm ci
npm run dev

Production-style local run:
cd UI/quant-ui
npm run build
cd ../..
uvicorn UI.backend.main:app --host 0.0.0.0 --port 8000

The FastAPI application serves the built frontend and all five API groups:
- /api/live
- /api/factors
- /api/experiments
- /api/backtests
- /api/labels

Live runners publish ephemeral snapshots to:
- /internal/live/snapshots

Live state is held in process memory, so run exactly one Uvicorn worker.

Optional path overrides:
- REPORTS_ROOT=/path/to/quant_output/batch_experiments
- BACKTEST_FRONTEND_REPORT_PATH=/path/to/latest_backtest_report.json
- LABEL_OUTPUT_DIR=/path/to/preparation/output
- FACTOR_DATA_ROOT=/path/to/root/containing/prepared/datasets

The Factors page discovers directories containing both train_data.feather and
test_data.feather under FACTOR_DATA_ROOT. By default this is the repository's
output directory. Analysis runs in a background thread and writes reports to
<dataset>/factor_analysis.
