from pickle import FALSE
from tkinter import TRUE
import pandas as pd
import numpy as np
import datetime, os, sys, re, math, json, logging
from dataclasses import asdict
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir,'..'))
from data_process import common
from data_process import feature
from data_process import config
from data_process.utils import param_hash
from analyse import volatility_prediction_heatmap


def _time_range(df: pd.DataFrame) -> dict:
    if df.empty:
        raise ValueError("Dataset is empty")
    if "open_time_date_utc" in df.columns:
        times = pd.to_datetime(df["open_time_date_utc"], utc=True)
    elif "open_time_ms_utc" in df.columns:
        times = pd.to_datetime(df["open_time_ms_utc"], unit="ms", utc=True)
    else:
        raise ValueError("Dataset has no supported open-time column")
    return {
        "start": times.min().isoformat(),
        "end": times.max().isoformat(),
        "row_count": int(len(df)),
    }


def _source_descriptor(df: pd.DataFrame, path: str) -> dict:
    return {
        "filename": os.path.basename(path),
        "size_bytes": os.path.getsize(path),
        "sha256": common.sha256_file(path),
        **_time_range(df),
    }

def main(logger:logging.Logger, feature_group_list = common.FEATURE_GROUP_LIST,feature_conf_list=[],para: common.PreparationConfig = common.BaseDefine(), prep_output_dir =common.DATA_OUT_DIR ):
    file = common.market_data_path(para)
    logger.info(f"using file :{file}")
    # 1. Convert interval string to milliseconds
    interval_ms = common.get_interval_ms(para.interval)
    
    # 2. Persist metadata for labeling and downstream model usage
    df = pd.read_csv(file)
    source_descriptor = _source_descriptor(df, file)
    if para.market_category == 'Cryptocurrency':
        common.validate_kline_source(
            df,
            interval_ms,
            source=file,
            logger=logger,
        )
    df = common.clean_data_quality_auto(df,logger)
    is_pure_binary = isinstance(para, common.PureBinaryDefine)
    if is_pure_binary:
        df['open_time_date_utc'] = pd.to_datetime(df['open_time_ms_utc'], unit='ms', utc=True)
        latest_time = df['open_time_date_utc'].max()
        start_ts = latest_time - pd.DateOffset(months=6)
        split_ts = latest_time - pd.DateOffset(months=3)
        df = df.loc[df['open_time_date_utc'] >= start_ts].sort_values('open_time_date_utc').reset_index(drop=True)
        logger.info("PURE_BINARY data window: %s -> %s; backtest starts at %s", start_ts, latest_time, split_ts)
    # 3. Pass interval_ms to label logic so it can adapt its volatility window to the real time span.
    label_col = 'label'
    if para.market_category == 'Forex':
        df = common.attach_attr(df, feature_group_list , feature_conf_list, para)
    else:
        df = common.attach_attr(df, feature_group_list , feature_conf_list, para)
    # common.print_zret_statistics(df)
    # Label the complete selected history before splitting; events may occur in the backtest period.
    df = common.attach_label(df, para=para,label_col = label_col)
    if para.label_type == "TBM" and not is_pure_binary:
        common.log_tbm_reach_time_percentiles(df, logger)
        df.drop(columns=list(common.TBM_REACH_TIME_COLUMNS.values()), inplace=True)
    if "expected_vol" in df.columns:
        plot_path = os.path.join(
            prep_output_dir,
            "label_proportion_by_expected_vol.png",
        )
        common.plot_label_proportions_by_expected_vol(
            df,
            output_path=plot_path,
            label_col=label_col,
            bins=200,
        )
        logger.info("📈 Label proportions by expected_vol saved to: %s", plot_path)
    # common.print_label_performance_stats(df, para)
    # # common.attach_macd_event_lifecycle_label(df, interval_ms=interval_ms)
    # # common.attach_boll_event_lifecycle_label(df, interval_ms=interval_ms)
    # # common.attach_sma_7_25_crossover_label(df, interval_ms=interval_ms)

    # ---------------- Summary statistics ----------------
    # ---------------- Summary statistics ----------------
    if 'open_time_date_utc' not in df.columns:
        df['open_time_date_utc'] = pd.to_datetime(df['open_time_ms_utc'], unit='ms', utc=True)
    start_time = df['open_time_date_utc'].iloc[0]
    end_time = df['open_time_date_utc'].iloc[-1]
    duration = pd.to_datetime(end_time) - pd.to_datetime(start_time)
    logger.info(f"Time span: {start_time} -> {end_time} (total: {duration})")
    counts = df[label_col].value_counts().sort_index()
    proportions = df[label_col].value_counts(normalize=True).sort_index()
    
    logger.info("\n=== Label distribution summary ===")
    if is_pure_binary:
        logger.info("PURE_BINARY inclusive return threshold: %.4f%%", para.pure_binary_threshold_pct * 100)
    else:
        logger.info("Thresholds are saved in columns: 'threshold_long' and 'threshold_short'")
        logger.info(f"Long threshold range: Min={df['threshold_long'].min():.4f}, Max={df['threshold_long'].max():.4f}, Mean={df['threshold_long'].mean():.4f}")
        logger.info(f"Short threshold range: Min={df['threshold_short'].min():.4f}, Max={df['threshold_short'].max():.4f}, Mean={df['threshold_short'].mean():.4f}")
        common.log_threshold_percentiles(df, logger)

    for label_val, cnt in counts.items():
        label_name = {
            common.Signal.INVALID: "INVALID",
            common.Signal.NEGATIVE: "Down",
            common.Signal.NEUTRAL: "Range",
            common.Signal.POSITIVE: "Up",
        }.get(label_val, "UNKNOWN")
        pct_val = proportions[label_val]
        logger.info(f"Label {label_val} ({label_name}): {cnt} rows, ratio {pct_val:.4%}")
    logger.info("==========================\n")
    
    # ---------------------------------------------------------
    # 3. Split and persist datasets
    # ---------------------------------------------------------
    if not is_pure_binary:
        split_ts = pd.to_datetime(df['open_time_date_utc'].iloc[-1], utc=True) - pd.DateOffset(months=8)
    times = pd.to_datetime(df['open_time_date_utc'], utc=True)
    train_df, test_df = df[times < split_ts], df[times >= split_ts]

    # Write to prep_output_dir (default common.DATA_OUT_DIR; independent per batch worker)
    out_dir = prep_output_dir
    os.makedirs(out_dir, exist_ok=True)
    common.save_train_df_to_dir(train_df, out_dir)
    common.save_test_df_to_dir(test_df, out_dir)
    meta_path = common.get_data_config_path_in_dir(out_dir)
    para_dict = asdict(para)
    safe_para = common.json_safe(para_dict)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(safe_para, f, indent=4, ensure_ascii=False)

    manifest_payload = {
        "schema_version": 2,
        "source": source_descriptor,
        "configuration_hash": param_hash(safe_para),
        "features": {
            "groups": [
                {
                    "class": (
                        f"{container.feature.__module__}."
                        f"{container.feature.__qualname__}"
                    ),
                    "parameters": common.json_safe(container.parameters),
                }
                for container in feature_group_list
            ],
            "configured_features": list(feature_conf_list),
        },
        "time": _time_range(df),
        "split_time": pd.Timestamp(split_ts).isoformat(),
    }
    manifest = {
        **manifest_payload,
        "data_id": common.data_manifest_id(manifest_payload),
    }
    manifest_path = common.get_data_manifest_path_in_dir(out_dir)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    logger.info("✅ Data preparation completed.")
    logger.info(f"📍 Interval: {para.interval}")
    logger.info(f"📍 Config written to: {meta_path}")
    logger.info(
        "📍 Data manifest: %s | data_id=%s | time=%s -> %s",
        manifest_path,
        manifest["data_id"],
        manifest["time"]["start"],
        manifest["time"]["end"],
    )
    return manifest


if __name__ == "__main__":
#**********column info: open_time_date_utc,open,high,low,close,volume,close_time_ms_utc,quote_asset_volume,number_of_trades,taker_buy_base_volume,taker_buy_quote_volume,ignore
    logger, _ = common.setup_session_logger(sub_folder='data_process')
    pare_para = config.DOGE_1m_PURE_BINARY
    pare_para = config.DOGE_1m_TBM
    if pare_para.market_category == 'Forex':
        feature_conf_list = feature.FEATURE_LIST_COMMODITY
    else:
        feature_conf_list = []
    main(logger,common.FEATURE_GROUP_LIST, para= pare_para)
    # volatility_prediction_heatmap.main(pare_para=pare_para,logger=logger,)
    # volatility_prediction_heatmap.plot_price_and_volatility(pare_para,[10,20,40,80,160],output_path=None,logger=None)
