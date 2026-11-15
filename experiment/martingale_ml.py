#!/usr/bin/env python3
"""Train rolling martingale regression or survival binary controls."""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from data_process import common, preparation
from model.train import train_windows
from model.train_config import (
    RidgeRegressionConfig,
    RollingConfig,
    TrainConfig,
    TrainTask,
    TransformerV1RegressionConfig,
    XGBoostConfig,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("regression", "binary-x2", "binary-x3", "binary-x4", "binary-x5", "binary-x6"), default="regression")
    parser.add_argument("--model", choices=("xgboost", "linear", "transformer"), default="xgboost")
    parser.add_argument("--symbol", default="DOGEUSDT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--side", choices=("long", "short"), default="long")
    parser.add_argument("--step-pct", type=float, default=0.01)
    parser.add_argument("--initial-take-profit-pct", type=float, default=0.01)
    parser.add_argument("--intrabar-interval", default="1m")
    parser.add_argument("--mode", choices=("expanding", "sliding"), default="expanding")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--validation-months", type=int, default=2)
    parser.add_argument("--oos-months", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--class-weight-power", type=float, default=0.85,
                        help="Training-fold class balance exponent for binary classification (0 to 1)")
    parser.add_argument("--split-time", help="UTC Forward boundary; defaults to the final three months")
    parser.add_argument("--start-time")
    parser.add_argument("--end-time")
    parser.add_argument("--include-forward", action="store_true")
    parser.add_argument("--prep-output-dir", help="Reuse an existing matching preparation artifact")
    parser.add_argument("--output-dir", help="New output directory; existing directories are not overwritten")
    args = parser.parse_args()

    params = common.MartingaleDefine(
        symbols=(args.symbol,), interval=args.interval,
        side=1 if args.side == "long" else -1, step_pct=args.step_pct,
        initial_take_profit_pct=args.initial_take_profit_pct,
        intrabar_interval=args.intrabar_interval,
    )
    binary = args.experiment != "regression"
    if binary and args.model != "xgboost":
        parser.error("Only XGBoost is supported for binary experiments")
    if not 0 <= args.class_weight_power <= 1:
        parser.error("--class-weight-power must be between 0 and 1")
    if args.model == "xgboost":
        model_config = XGBoostConfig(
            seq_len=args.seq_len, xgb_depth=args.depth, xgb_estimators=args.estimators,
            objective=None if binary else "reg:squarederror",
            eval_metric="logloss" if binary else "rmse", use_scaler=False,
            class_weight_power=args.class_weight_power if binary else 0.0,
        )
    elif args.model == "linear":
        model_config = RidgeRegressionConfig(seq_len=args.seq_len, alpha=args.ridge_alpha)
    else:
        model_config = TransformerV1RegressionConfig(seq_len=args.seq_len)
    config = TrainConfig(
        model_cfg=model_config,
        train_task=TrainTask.MARTINGALE_BINARY if binary else TrainTask.MARTINGALE_REGRESSION,
        max_layers=int(args.experiment[-1]) if binary else None,
        breach_positive=binary,
        feature_conf_list=None, stride=args.stride, selection_metric="loss" if binary else "mae",
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
    )
    rolling = RollingConfig(
        train_months=args.train_months, validation_months=args.validation_months,
        oos_months=args.oos_months, mode=args.mode,
    )
    if args.prep_output_dir:
        existing = common.load_pre_params_from_dir(args.prep_output_dir)
        if existing != params:
            raise ValueError("Existing preparation parameters do not match the requested martingale experiment")
        if any((args.split_time, args.start_time, args.end_time)):
            raise ValueError("Time boundaries are taken from reused preparation artifacts")
    output_dir = args.output_dir or os.path.join(
        PROJECT_DIR, "output", "martingale_ml", args.symbol + "_" + args.interval,
        args.experiment,
        datetime.now(timezone.utc).strftime("%Y-%m-%d/%H_%M_%S_%f"),
    )
    os.makedirs(output_dir, exist_ok=False)
    logger, _ = common.setup_session_logger(
        log_file_path=os.path.join(output_dir, "experiment.log"),
        console_level=logging.INFO, file_level=logging.INFO,
    )
    data_dir = args.prep_output_dir or os.path.join(output_dir, "data")
    if not args.prep_output_dir:
        preparation.main(
            logger, common.FEATURE_GROUP_LIST, [], para=params, prep_output_dir=data_dir,
            split_time=args.split_time, start_time=args.start_time, end_time=args.end_time,
        )
    manifest = train_windows(
        configs=[config], rolling_config=rolling, prep_output_dir=data_dir,
        save_dir=os.path.join(output_dir, "train"), logger=logger,
        include_forward=args.include_forward,
    )
    logger.info("Martingale %s completed | folds=%d | output=%s", args.experiment, len(manifest["folds"]), output_dir)
    return manifest


if __name__ == "__main__":
    main()
