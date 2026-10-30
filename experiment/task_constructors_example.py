"""Parameter-sweep task constructors used by the batch runners.

This module owns task definitions only. Batch orchestration, persistence,
resuming and worker management remain in ``batch_experiments`` and
``batch_train``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from data_process import common
from data_process import feature
from model import train_config
from dataclasses import replace
from data_process import config

# Local batch runtime configuration. Copy this file to task_constructors.py and
# customize that ignored file without changing tracked source files.
MAX_PREP = 1
MAX_TRAIN = 4  # Maximum concurrent train processes; each train uses its own process.
MAX_SIM = 25
INFERENCE_BATCH_SIZE = 256
SYMBOL: str = "DOGEUSDT"  # ETHUSDT or DOGEUSDT
INTERVAL: str = "15m"

# Training mode switch:
# - DIRECT_3CLASS (or another single-model task) uses prep -> train -> sim.
# - TRIGGER_DIRECTION / LONG_SHORT_OVR sweeps both roles separately, then
#   fuses compatible model pairs for backtesting.
TRAIN_MODE: str = train_config.TrainTask.DIRECT_3CLASS

# -----------------------------------------------------------------------------
# prep -> train -> simulation constructors (experiment/batch_experiments.py)
# -----------------------------------------------------------------------------
def construct_experiment_doge(symbol: str, interval: str, train_mode):
    import model.train as train
    from trade.runner import backtest_runner

    preparation_task: List[Any] = []
    for pn in [16, 24, 32]:
        for vol_multiplier in [1.8]:
            for vol_ewma_span in [80]:
                preparation_task.append(common.BaseDefine(
                    market_category="Cryptocurrency",
                    data_source="binance_public_data",
                    vol_ewma_span=vol_ewma_span,
                    predict_num=pn,
                    vol_multiplier_long=vol_multiplier,
                    stop_multiplier_rate_long=None,
                    vol_multiplier_short=vol_multiplier,
                    stop_multiplier_rate_short=None,
                    symbol=symbol,
                    interval=interval,
                    trading_type="um",
                    label_type="FTHL",
                    version=0,
                ))

    training_task: List[train.TrainConfig] = []
    for false_trade_penalty in [1]:
        for seq_len in [12, 16, 24, 96, 256]:
            for flip_penalty in np.arange(0.8, 1.7, 1).round(1):
                for miss_penalty in np.arange(0.5, 2, 0.1).round(1):
                    for stride in [2]:
                        for bestf1 in [True]:
                            for feature_conf in [train_config.feature_conf_list]:
                                train_conf = train.TrainConfig(
                                    use_cache=False,
                                    epochs=100,
                                    batch_size=256,
                                    feature_conf_list=feature_conf,
                                    flip_penalty=float(flip_penalty),
                                    miss_penalty=float(miss_penalty),
                                    false_trade_penalty=false_trade_penalty,
                                    stride=stride,
                                    patience=8,
                                )
                                train_conf.model_cfg = train_config.LogisticConfig(
                                    seq_len=seq_len,
                                    model_version=1,
                                )
                                train_conf.train_task = train_mode
                                training_task.append(train_conf)

    simulation_task: List[Any] = []
    for fixed_hold_bars in [12, 16, 24]:
        for atr_sl_long_mult, atr_sl_short_mult in [(6, 5), (5, 4)]:
            simulation_task.append(
                backtest_runner.MlStrategyConfig(
                    allow_long=True,
                    allow_short=True,
                    fixed_hold_bars=fixed_hold_bars,
                    prob_thresh=None,
                    atr_sl_long_mult=atr_sl_long_mult,
                    atr_sl_short_mult=atr_sl_short_mult,
                    atr_tp_mult=0.99,
                    risk_per_trade_pct=0.4,
                    max_daily_loss_pct=0.04,
                )
            )
    return preparation_task, training_task, simulation_task


def construct_bbm_doge(symbol: str, interval: str, train_mode):
    import model.train as train
    from trade.runner import backtest_runner

    preparation_task: List[Any] = []
    for pn in [2,4]:
        for vol_ewma_span in [80,96]: #[80,96]
            preparation_task.append(replace(common.DOGE_1m,interval="30m", vol_ewma_span=vol_ewma_span, predict_num=pn, label_type = "BBM"
                    ))

    training_task: List[train.TrainConfig] = []
    for seq_len in [24, 96, 256,512,1024]:
        for stride in [2,4]: #[1, 2, 4]
            for feature_conf in [train_config.feature_conf_list]:
                for model_config_type in [
                    # train_config.TransformerConfig,
                    # train_config.ConvLSTMConfig,
                    # train_config.LSTMConfig,
                    train_config.LogisticConfig,
                ]:
                    model_cfg = model_config_type(seq_len=seq_len)
                    train_conf = train.TrainConfig(
                        use_cache=False,
                        epochs=100,
                        batch_size=256,
                        feature_conf_list=feature_conf,
                        model_cfg=model_cfg,
                        stride=stride,
                        patience=8,
                    )
                    train_conf.train_task = train_mode
                    training_task.append(train_conf)

    simulation_task: List[Any] = []
    for min_expected_move_pct in [0.01,0.015,0.02]:
        simulation_task.append(
            backtest_runner.BbmStrategyConfig(
                allow_long=True,
                allow_short=True,
                min_expected_move_pct=min_expected_move_pct,
                risk_per_trade_pct=0.02,
                max_daily_loss_pct=0.025,
            )
        )
    return preparation_task, training_task, simulation_task

def construct_TBM_DIRECT_3CLASS_doge(symbol: str, interval: str, train_mode):
    import model.train as train
    from trade.runner import backtest_runner

    preparation_task: List[Any] = []
    for pn in [64]:#, 64]:
        for vol_ewma_span in [80,160]: #[40, 80, 160]
            for vol_multiplier in [12]:#, [8,10, 12, 14,16]:
                for stop_multiplier_rate in [0.4]:#,[0.3, 0.4,0.5]:
                    preparation_task.append(replace(config.DOGE_15m_TBM, interval = interval,
                            vol_ewma_span=vol_ewma_span, predict_num=pn,
                            vol_multiplier_long=vol_multiplier, vol_multiplier_short= vol_multiplier,
                            stop_multiplier_rate_long= stop_multiplier_rate, stop_multiplier_rate_short=stop_multiplier_rate))

    training_task: List[train.TrainConfig] = []
    for seq_len in [96,128,256]: #[96, 128,256]
        for stride in [2,4]: #[1, 2, 4]
            for feature_conf in [feature.FEATURE_LIST_CRYPTOCURRENCY]:
                for base_model_cfg in [
                                        # train_config.TransformerConfig(model_version=2),
                                          train_config.ConvLSTMConfig(model_version=1), 
                                          train_config.LSTMConfig(model_version=1),
                                        #   train_config.LSTMConfig(model_version=2),
                                          train_config.LogisticConfig(model_version=1),
                ]:
                    train_conf = train.TrainConfig(
                        feature_conf_list=feature_conf,
                        model_cfg=replace(base_model_cfg,seq_len=seq_len),
                        stride=stride,
                        train_task=train_mode,
                    )
                    training_task.append(train_conf)

    simulation_task: List[Any] = []
    for risk_per_trade_pct,max_daily_loss_pct in [(0.01, 0.015),(0.02, 0.025),]: # (0.02, 0.99) not goos
        for fixed_hold_bars in [32,48,64]: #64>48>32
            simulation_task.append(
                backtest_runner.BbmStrategyConfig(
                    allow_long=True,
                    allow_short=True,
                    risk_per_trade_pct= risk_per_trade_pct,
                    max_daily_loss_pct= max_daily_loss_pct,
                    fixed_hold_bars= fixed_hold_bars,
                )
            )
    return preparation_task, training_task, simulation_task

def construct_experiment_doge_combo(symbol: str, interval: str):
    """Construct independent trigger/direction sweeps for combo-model fusion."""
    import model.train as train
    from trade.runner import backtest_runner

    preparation_task: List[Any] = []
    for pn in [16, 24, 32]:
        for vol_multiplier in [1.8, 1.9]:
            for vol_ewma_span in [80]:
                preparation_task.append(common.BaseDefine(
                    market_category="Cryptocurrency",
                    data_source="binance_public_data",
                    vol_ewma_span=vol_ewma_span,
                    predict_num=pn,
                    vol_multiplier_long=vol_multiplier,
                    stop_multiplier_rate_long=None,
                    vol_multiplier_short=vol_multiplier,
                    stop_multiplier_rate_short=None,
                    symbol=symbol,
                    interval=interval,
                    trading_type="um",
                    label_type="FTHL",
                    version=0,
                ))

    training_task: List[train.TrainConfig] = []
    seq_len = 24
    stride = 2
    for minority_ratio in np.arange(0.2, 0.3, 0.02).round(2):
        for miss_penalty in np.linspace(
            1 / minority_ratio / 3 * 2,
            1 / minority_ratio * 4,
            4,
        ).round(2):
            trigger_config = train.TrainConfig(
                use_cache=False,
                epochs=20,
                batch_size=256,
                model_cfg=train_config.LogisticConfig(model_version=1, seq_len=seq_len),
                minority_sampling_ratio=float(minority_ratio),
                miss_penalty=float(miss_penalty),
                stride=stride,
                patience=5,
            )
            trigger_config.train_task = train_config.TrainTask.TRIGGER
            training_task.append(trigger_config)

    for model_cfg in [train_config.TransformerConfig(model_version=1, seq_len=seq_len)]:
        direction_config = train.TrainConfig(
            use_cache=False,
            epochs=20,
            batch_size=256,
            model_cfg=model_cfg,
            stride=stride,
            patience=5,
        )
        direction_config.train_task = train_config.TrainTask.DIRECTION
        training_task.append(direction_config)

    simulation_task: List[Any] = []
    for fixed_hold_bars in [4, 8, 12, 16, 24]:
        for atr_sl_long_mult, atr_sl_short_mult in [(6, 5), (5, 4)]:
            simulation_task.append(
                backtest_runner.MlStrategyConfig(
                    allow_long=True,
                    allow_short=True,
                    fixed_hold_bars=fixed_hold_bars,
                    prob_thresh=None,
                    atr_sl_long_mult=atr_sl_long_mult,
                    atr_sl_short_mult=atr_sl_short_mult,
                    atr_tp_mult=0.99,
                    risk_per_trade_pct=0.4,
                    max_daily_loss_pct=0.04,
                )
            )
    return preparation_task, training_task, simulation_task


def construct_experiment_eth(symbol: str, interval: str):
    import model.train as train
    from trade.runner import backtest_runner

    preparation_task: List[Any] = []
    for predict_num in [4, 8, 16, 24]:
        for vol_multiplier in [1.8, 1.9, 2, 2.1]:
            for vol_ewma_span in [88]:
                preparation_task.append(common.BaseDefine(
                    vol_ewma_span=vol_ewma_span,
                    predict_num=predict_num,
                    vol_multiplier_long=vol_multiplier,
                    stop_multiplier_rate_long=0.2,
                    vol_multiplier_short=vol_multiplier,
                    stop_multiplier_rate_short=0.2,
                    symbol=symbol,
                    interval=interval,
                    trading_type="um",
                    version=0,
                ))

    training_task: List[train.TrainConfig] = []
    for seq_len in [12, 16]:
        for false_trade_penalty in [1]:
            for flip_penalty in np.arange(0.9, 1.7, 0.1).round(1):
                for miss_penalty in np.arange(0.7, 1.2, 0.1).round(1):
                    for stride in [4, 8]:
                        training_task.append(train.TrainConfig(
                            use_cache=False,
                            epochs=100,
                            batch_size=256,
                            model_cfg=train_config.ConvLSTMConfig(
                                model_version=3,
                                seq_len=seq_len,
                            ),
                            flip_penalty=float(flip_penalty),
                            miss_penalty=float(miss_penalty),
                            false_trade_penalty=false_trade_penalty,
                            stride=stride,
                            patience=8,
                            lambda_main=0.7,
                            lambda_dir=0.7,
                            lambda_cost=0.4,
                        ))

    simulation_task: List[Any] = []
    for fixed_hold_bars in [30, 32, 36, 38, 40, 44]:
        for atr_sl_long_mult, atr_sl_short_mult in [(6, 6), (5, 4)]:
            simulation_task.append(
                backtest_runner.MlStrategyConfig(
                    allow_long=True,
                    allow_short=True,
                    fixed_hold_bars=fixed_hold_bars,
                    prob_thresh=None,
                    atr_sl_long_mult=atr_sl_long_mult,
                    atr_sl_short_mult=atr_sl_short_mult,
                    risk_per_trade_pct=0.1,
                    max_daily_loss_pct=0.025,
                )
            )
    return preparation_task, training_task, simulation_task


def construct_experiment_tasks(symbol: str, interval: str, train_mode):
    """Select the prep/train/simulation constructor for one experiment batch."""
    if train_mode in train_config.COMBO_SUB_TASKS:
        if symbol != "DOGEUSDT":
            raise RuntimeError(f"no combo construct for {symbol} yet")
        return construct_experiment_doge_combo(symbol, interval)
    if symbol == "DOGEUSDT":
        if train_mode == train_config.TrainTask.DIRECTION:
            return construct_bbm_doge(symbol, interval, train_mode)
        elif train_mode == train_config.TrainTask.DIRECT_3CLASS:
            return construct_TBM_DIRECT_3CLASS_doge(symbol, interval, train_mode)
        else:
            return construct_experiment_doge(symbol, interval, train_mode)
    if symbol == "ETHUSDT":
        return construct_experiment_eth(symbol, interval)
    raise RuntimeError(f"no construct for {symbol} yet")


# -----------------------------------------------------------------------------
# prep -> train constructors (experiment/batch_train.py)
# -----------------------------------------------------------------------------
def _new_training_task_map(train) -> Dict[str, List[Any]]:
    return {
        train.TrainTask.DIRECTION: [],
        train.TrainTask.TRIGGER: [],
    }


def construct_training_doge(symbol: str, interval: str):
    import model.train as train

    preparation_task: List[Any] = []
    for predict_num in [20]:
        for vol_multiplier in [2]:
            for vol_ewma_span in [80]:
                preparation_task.append(common.BaseDefine(
                    market_category="Cryptocurrency",
                    data_source="binance_public_data",
                    vol_ewma_span=vol_ewma_span,
                    predict_num=predict_num,
                    vol_multiplier_long=vol_multiplier,
                    stop_multiplier_rate_long=None,
                    vol_multiplier_short=vol_multiplier,
                    stop_multiplier_rate_short=None,
                    symbol=symbol,
                    interval=interval,
                    trading_type="spot",
                    label_type="FTHL",
                    version=0,
                ))

    training_task = _new_training_task_map(train)
    for seq_len in [24]:
        for stride in [2]:
            for feature_conf in [train_config.feature_conf_list]:
                for model_cfg in [train_config.LogisticConfig(model_version=1, seq_len=seq_len)]:
                    for pos_ratio in np.arange(0.15, 0.3, 0.01).round(2):
                        for miss_penalty in np.linspace(
                            1 / pos_ratio / 3 * 2,
                            1 / pos_ratio * 4,
                            15,
                        ).round(2):
                            train_conf = train.TrainConfig(
                                use_cache=False,
                                epochs=20,
                                batch_size=256,
                                feature_conf_list=feature_conf,
                                model_cfg=model_cfg,
                                miss_penalty=float(miss_penalty),
                                flip_penalty=1,
                                minority_sampling_ratio=pos_ratio,
                                stride=stride,
                                patience=5,
                            )
                            training_task[train.TrainTask.TRIGGER].append(train_conf)

                for model_cfg in [
                    train_config.TransformerConfig(model_version=1, seq_len=seq_len),
                ]:
                    train_conf = train.TrainConfig(
                        use_cache=False,
                        epochs=20,
                        batch_size=256,
                        feature_conf_list=feature_conf,
                        model_cfg=model_cfg,
                        flip_penalty=1.0,
                        miss_penalty=1,
                        stride=stride,
                        patience=5,
                    )
                    training_task[train.TrainTask.DIRECTION].append(train_conf)
    return preparation_task, training_task


def construct_training_eth(symbol: str, interval: str):
    import model.train as train

    preparation_task: List[Any] = []
    for predict_num in [4, 8, 16, 24]:
        for vol_multiplier in [1.8, 1.9, 2, 2.1]:
            for vol_ewma_span in [88]:
                preparation_task.append(common.BaseDefine(
                    vol_ewma_span=vol_ewma_span,
                    predict_num=predict_num,
                    vol_multiplier_long=vol_multiplier,
                    stop_multiplier_rate_long=0.2,
                    vol_multiplier_short=vol_multiplier,
                    stop_multiplier_rate_short=0.2,
                    symbol=symbol,
                    interval=interval,
                    trading_type="um",
                    version=0,
                ))

    training_task: List[train.TrainConfig] = []
    for seq_len in [12, 16]:
        for false_trade_penalty in [1]:
            for flip_penalty in np.arange(0.9, 1.7, 0.1).round(1):
                for miss_penalty in np.arange(0.7, 1.2, 0.1).round(1):
                    for stride in [4, 8]:
                        training_task.append(train.TrainConfig(
                            use_cache=False,
                            epochs=100,
                            batch_size=256,
                            model_cfg=train_config.ConvLSTMConfig(
                                model_version=3,
                                seq_len=seq_len,
                            ),
                            flip_penalty=float(flip_penalty),
                            miss_penalty=float(miss_penalty),
                            false_trade_penalty=false_trade_penalty,
                            stride=stride,
                            patience=8,
                            lambda_main=0.7,
                            lambda_dir=0.7,
                            lambda_cost=0.4,
                        ))
    return preparation_task, {train.TrainTask.DIRECT_3CLASS: training_task}


def construct_training_xlm(symbol: str, interval: str):
    import model.train as train

    preparation_task: List[Any] = []
    for predict_num in [20]:
        for vol_multiplier in [1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2]:
            for vol_ewma_span in [80]:
                preparation_task.append(common.BaseDefine(
                    market_category="Cryptocurrency",
                    data_source="binance_public_data",
                    vol_ewma_span=vol_ewma_span,
                    predict_num=predict_num,
                    vol_multiplier_long=vol_multiplier,
                    stop_multiplier_rate_long=None,
                    vol_multiplier_short=vol_multiplier,
                    stop_multiplier_rate_short=None,
                    symbol=symbol,
                    interval=interval,
                    trading_type="um",
                    label_type="FTHL",
                    version=0,
                ))

    training_task = _new_training_task_map(train)
    for seq_len in [24]:
        for stride in [1, 2, 4]:
            for feature_conf in [train_config.feature_conf_list]:
                for model_cfg in [train_config.LogisticConfig(model_version=1, seq_len=seq_len)]:
                    for miss_penalty in np.arange(2, 5, 0.1).round(1):
                        train_conf = train.TrainConfig(
                            use_cache=False,
                            epochs=20,
                            batch_size=256,
                            feature_conf_list=feature_conf,
                            model_cfg=model_cfg,
                            miss_penalty=float(miss_penalty),
                            stride=stride,
                            patience=5,
                        )
                        training_task[train.TrainTask.TRIGGER].append(train_conf)

                for model_cfg in [
                    train_config.LSTMConfig(model_version=1, seq_len=seq_len),
                    train_config.TransformerConfig(model_version=2, seq_len=seq_len),
                ]:
                    train_conf = train.TrainConfig(
                        use_cache=False,
                        epochs=20,
                        batch_size=256,
                        feature_conf_list=feature_conf,
                        model_cfg=model_cfg,
                        flip_penalty=1.0,
                        stride=stride,
                        patience=5,
                    )
                    training_task[train.TrainTask.DIRECTION].append(train_conf)
    return preparation_task, training_task


def construct_training_xau(symbol: str, interval: str):
    import model.train as train

    preparation_task: List[Any] = []
    for predict_num in [20]:
        for vol_multiplier in [1.7, 1.8, 1.9]:
            for vol_ewma_span in [80]:
                preparation_task.append(common.BaseDefine(
                    market_category="Forex",
                    data_source="dukascopy",
                    vol_ewma_span=vol_ewma_span,
                    predict_num=predict_num,
                    vol_multiplier_long=vol_multiplier,
                    stop_multiplier_rate_long=None,
                    vol_multiplier_short=vol_multiplier,
                    stop_multiplier_rate_short=None,
                    symbol=symbol,
                    interval=interval,
                    trading_type="spot",
                    label_type="FTHL",
                    version=0,
                ))

    training_task = _new_training_task_map(train)
    for seq_len in [24]:
        for stride in [2]:
            for feature_conf in [FEATURE_LIST_COMMODITY]:
                for model_cfg in [train_config.LogisticConfig(model_version=1, seq_len=seq_len)]:
                    for miss_penalty in [3.5]:
                        train_conf = train.TrainConfig(
                            use_cache=False,
                            epochs=20,
                            batch_size=256,
                            feature_conf_list=feature_conf,
                            model_cfg=model_cfg,
                            miss_penalty=float(miss_penalty),
                            stride=stride,
                            patience=5,
                        )
                        training_task[train.TrainTask.TRIGGER].append(train_conf)

                for model_cfg in [
                    train_config.LSTMConfig(model_version=1, seq_len=seq_len),
                    train_config.TransformerConfig(model_version=2, seq_len=seq_len),
                ]:
                    train_conf = train.TrainConfig(
                        use_cache=False,
                        epochs=20,
                        batch_size=256,
                        feature_conf_list=feature_conf,
                        model_cfg=model_cfg,
                        flip_penalty=1.0,
                        stride=stride,
                        patience=5,
                    )
                    training_task[train.TrainTask.DIRECTION].append(train_conf)
    return preparation_task, training_task


def construct_training_tasks(symbol: str, interval: str):
    """Select the prep/train constructor for one training-only batch."""
    constructors = {
        "DOGEUSDT": construct_training_doge,
        "ETHUSDT": construct_training_eth,
        "XLMUSDT": construct_training_xlm,
        "XAUUSD": construct_training_xau,
    }
    try:
        constructor = constructors[symbol]
    except KeyError as exc:
        raise RuntimeError(f"no construct for {symbol} yet") from exc
    return constructor(symbol, interval)
