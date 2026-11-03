import logging
import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
from data_process import common
from trade.runner.backtest_entry import main
from trade.runner.analyze_backtest_report import plot_equity_curves
from trade.runner.config import BrokerConfig, ExperimentContext, ModelDataConfig, RunnerConfig
from trade.strategy.strategy_bbm import BbmStrategyConfig

model_dir = "output/train/rolling_doge_aave"
preparation = common.load_pre_params_from_dir(model_dir)

config = RunnerConfig(
    strategy_config=BbmStrategyConfig(
        fixed_hold_bars=preparation.predict_num,
    ),
    broker_config=BrokerConfig(),
    save_dir="output/expanding_backtest",
    experiment_context=ExperimentContext(git_commit=common.git_revision()),
    data_config=ModelDataConfig(
        prep_output_dir=common.DATA_OUT_DIR,
        train_output_dir=model_dir,
        device="auto",
        symbol= "DOGEUSDT" ,
    ),
)

log_dir = os.path.abspath(config.save_dir)
os.makedirs(log_dir, exist_ok=True)
logger, log_path = common.setup_session_logger(
    log_file_path=os.path.join(log_dir, "experiment.log"),
    console_level=logging.INFO,
    file_level=logging.INFO,
)

try:
    result = main(
        logger,
        config,
        period="long",
        evaluate_stages=False,
    )
    equity_path = plot_equity_curves(
        result,
        log_dir,
        file_name="equity_curve.png",
        normalize_equity=False,
        include_oos=False,
        logger=logger,
    )
    if equity_path is None:
        raise RuntimeError("Backtest returned no equity curve to plot")
    logger.info("Equity curve saved to: %s", os.path.abspath(equity_path))
except Exception:
    logger.exception("Backtest failed")
    raise
finally:
    logger.info("Backtest log file: %s", log_path)