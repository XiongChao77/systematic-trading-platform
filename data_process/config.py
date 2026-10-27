import os, sys
from dataclasses import replace
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir,'..'))
from data_process import common
from data_process import feature

DOGE_15m_TBM =common.BaseDefine(
    market_category="Cryptocurrency", data_source="binance_public_data", symbol="DOGEUSDT", interval="15m", 
    trading_type='um', label_type="TBM", vol_ewma_span=80, predict_num=64, vol_multiplier_long=12,
    stop_multiplier_rate_long=0.4, vol_multiplier_short=12, stop_multiplier_rate_short=0.4,
    tbm_take_profit_price="close", min_expected_move_pct=0, version = 0,
)

DOGE_1m_TBM = replace(
    DOGE_15m_TBM,
    interval="1m",
    vol_ewma_span = 60*24*7,
    predict_num = 60*12,
    vol_multiplier_long = 20,
    stop_multiplier_rate_long = 0.6,
    vol_multiplier_short = 20,
    stop_multiplier_rate_short = 0.6,
    min_expected_move_pct=0.1,
    tbm_take_profit_price = "high_low"
)

DOGE_15m_BBM =common.BaseDefine(
    market_category="Cryptocurrency", data_source="binance_public_data", symbol="DOGEUSDT", interval="15m", 
    trading_type='um', label_type="BBM", vol_ewma_span=20, predict_num=64, vol_multiplier_long=8,
    stop_multiplier_rate_long=0.4, vol_multiplier_short=8, stop_multiplier_rate_short=0.4,
    tbm_take_profit_price="close", min_expected_move_pct=0, version = 0,
)

DOGE_1m_PURE_BINARY = common.PureBinaryDefine(
    market_category="Cryptocurrency",
    data_source="binance_public_data",
    symbol="DOGEUSDT",
    interval="1m",
    trading_type="um",
    pure_binary_threshold_pct=0.005,
)
