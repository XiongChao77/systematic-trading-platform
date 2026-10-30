#!/usr/bin/env python3
import argparse
import json,os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.ndimage import uniform_filter

# --- Configuration ---
EPS = 1e-9

def smart_format(x):
    if abs(x) >= 100000:
        return f"{x/1000:.1f}k"
    elif abs(x) >= 1000:
        return f"{x:.0f}"
    else:
        return f"{x:.3f}"

def recursive_get(data, target_key):
    # 1. If it's a dict, first check current level
    if isinstance(data, dict):
        if target_key in data:
            return data[target_key]
        
        # 2. If not found, recursively search each value
        for k, v in data.items():
            res = recursive_get(v, target_key)
            if res is not None:
                return res
                
    # 3. If it's a list (common for backtest parameter combinations), recurse into each item
    elif isinstance(data, list):
        for item in data:
            res = recursive_get(item, target_key)
            if res is not None:
                return res
                
    return None

def load_and_process(path: Path, period: str = 'forward') -> pd.DataFrame:
    """Data loading function with debug output"""
    try:
        # 1. Check if file exists
        if not path.exists():
            print(f"❌ Error: File not found at {path}")
            return pd.DataFrame()

        df_raw = pd.read_json(path, lines=True)
        
        # 2. Debug: print raw column names
        print(f"\n--- Debug: {period.upper()} Period Parsing ---")
        print(f"Raw Columns Found: {df_raw.columns.tolist()}")

        if "params" not in df_raw.columns or "results" not in df_raw.columns:
            print("❌ Error: Canonical params/results columns are missing!")
            return pd.DataFrame()

        # 3. Debug: sample first row structure
        sample_row = df_raw["results"].iloc[0].get(period)
        if isinstance(sample_row, dict):
            print(f"Sample keys in '{period}': {list(sample_row.keys())}")
        else:
            print(f"⚠️ Warning: Data in '{period}' is not a dictionary, it is a {type(sample_row)}")

        def extract_data(row):
            try:
                period_result = row["results"].get(period)
                p = (
                    {"params": row["params"], **period_result}
                    if isinstance(period_result, dict)
                    else None
                )
                if p is None: return None
                
                res = {
                    "flip_penalty": recursive_get(p, "flip_penalty"),
                    "miss_penalty": recursive_get(p, "miss_penalty"),
                    "fixed_hold_bars": recursive_get(p, "fixed_hold_bars"),
                    "cagr": recursive_get(p, "cagr"),
                    "calmar": recursive_get(p, "calmar"),
                    "long_pnl": recursive_get(p, "long_pnl"),
                    "long_win_rate": recursive_get(p, "long_win_rate"),
                    "short_pnl": recursive_get(p, "short_pnl"),
                    "short_win_rate": recursive_get(p, "short_win_rate"),
                }
                return res
            except Exception as e:
                # Only print when there is a real error to avoid floosing logs
                return None

        # 4. Transform and filter
        processed_list = [item for item in df_raw.apply(extract_data, axis=1) if item]
        df_final = pd.DataFrame(processed_list)
        df_final = df_final[df_final["fixed_hold_bars"] == 24]#[20, 24 ,28, 32,36]


        # 5. Debug: inspect the final DataFrame content
        if not df_final.empty:
            print(f"✅ Success: Loaded {len(df_final)} rows for {period}")
            # Check if key fields are all NaN
            nan_counts = df_final[['flip_penalty', 'miss_penalty', 'cagr']].isna().sum()
            if nan_counts.any():
                print(f"⚠️ Warning: Missing values found:\n{nan_counts}")
            print(f"Data Preview:\n{df_final[['flip_penalty', 'miss_penalty', 'cagr']].head(3)}")
        else:
            print("❌ Error: DataFrame is empty after extraction. Check your key paths in recursive_get.")

        return df_final

    except Exception as e:
        print(f"🔥 Critical Loading Error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def calculate_robustness(pivot_df: pd.DataFrame):
    """
    Compute robustness score: mean of current cell and its neighborhoos.
    Helps identify parameter plateaus and avoid isolated spikes.
    """
    if pivot_df.empty:
        print("⚠️ Warning: Empty DataFrame passed to calculate_robustness.")
        return pd.DataFrame()

    data = pivot_df.values
    # Use a 3x3 window for smoothing
    try:
        smoothed = uniform_filter(data, size=3, mode='constant', cval=np.nanmin(data))
        return pd.DataFrame(smoothed, index=pivot_df.index, columns=pivot_df.columns)
    except ValueError as e:
        print(f"⚠️ Error in calculate_robustness: {e}")
        return pd.DataFrame()

def plot_enhanced_heatmaps(df: pd.DataFrame, metric: str, outdir: Path, period: str = 'forward'):
    """
    Generate three heatmaps for each metric: Mean, Max, Robustness
    """
    # Prepare three aggregated datasets
    p_mean = df.groupby(["flip_penalty", "miss_penalty"])[metric].mean().unstack()
    p_max  = df.groupby(["flip_penalty", "miss_penalty"])[metric].max().unstack()
    
    # If all data is NaN, skip plotting to avoid errors
    if p_mean.isna().all().all():
        print(f"⚠️ Skipping {metric} for {period}: All values are NaN.")
        return

    p_rob  = calculate_robustness(p_mean)

    titles = {
        "mean": f"{metric.upper()} - Mean",
        "max": f"{metric.upper()} - Max",
        "robust": f"{metric.upper()} - Robustness"
    }
    
    n_rows, n_cols = p_mean.shape   # Any one dataset is enough

    cell_size = 0.8  # 👈 Size of each cell (inches)

    fig_width  = n_cols * cell_size * 3   # Three plots horizontally
    fig_height = n_rows * cell_size

    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))

    datasets = [p_mean, p_max, p_rob]
    sub_types = ["mean", "max", "robust"]

    for ax, data, stype in zip(axes, datasets, sub_types):
        # Double-check subset is not entirely empty
        if data.isna().all().all():
            ax.set_title(f"{titles[stype]} (No Data)")
            continue

        annot_data = data.map(smart_format)
        sns.heatmap(data, cmap="RdYlGn", annot=annot_data, fmt="", ax=ax, 
                    linewidths=0.5, cbar_kws={'shrink': 0.8},center=0,)
        ax.set_title(titles[stype], fontsize=14, fontweight='bold')
        
        # --- Core fix: only compute argmax when not all values are NaN ---
        try:
            if not np.isnan(data.values).all():
                max_idx = np.unravel_index(np.nanargmax(data.values), data.shape)
                ax.add_patch(plt.Rectangle((max_idx[1], max_idx[0]), 1, 1, 
                                           fill=False, edgecolor='blue', lw=3, ls='--'))
        except ValueError:
            pass 

    plt.suptitle(f"Detailed Sensitivity Analysis ({period.upper()}): {metric.upper()}", fontsize=20, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"detailed_analysis_{period}_{metric}.png"), dpi=150, bbox_inches="tight")
    plt.close()

def save_detailed_stats(df: pd.DataFrame, outdir: Path, period: str = 'forward'):
    """Generate detailed statistics CSV.

    Args:
        df: data DataFrame
        outdir: output directory
        period: 'forward' or 'long'
    """
    for metric in ["cagr", "calmar"]:
        pivot = df.groupby(["flip_penalty", "miss_penalty"])[metric].mean().unstack()
        
        stats = pd.DataFrame(index=pivot.index)
        stats["mean"] = pivot.mean(axis=1)
        stats["max"]  = pivot.max(axis=1)
        stats["min"]  = pivot.min(axis=1)
        stats["std"]  = pivot.std(axis=1)
        
        # Stability score: inverse of coefficient of variation, higher means more stable
        stats["stability_score"] = (stats["mean"].abs() / (stats["std"] + EPS))
        
        # Find best parameter combination
        stats["best_candle_config"] = pivot.idxmax(axis=1)
        
        stats.sort_values("mean", ascending=False).to_csv(os.path.join(outdir, f"comprehensive_stats_{period}_{metric}.csv"))

def main():
    report_path = r"/home/chao/work/quant_output/batch_experiments/2026-02-14/DOGEUSDT_15m/00_29_07/reports.jsonl"

    path = Path(report_path).resolve()
    outdir = os.path.join(path.parent, 'heatmaps')
    os.makedirs(outdir, exist_ok=True)

    # Process the full forward test set and full-history long periods.
    for period in ['long', 'forward']:
        print(f"\n📊 Processing {period.upper()} period...")
        df = load_and_process(path, period=period)
        if df.empty:
            print(f"⚠️ No data found for {period} period")
            continue

        # 1. Plot enhanced heatmaps
        for m in ["cagr", "calmar", "long_pnl", "long_win_rate", "short_pnl", "short_win_rate"]:
            plot_enhanced_heatmaps(df, m, outdir, period=period)
        
        # 2. Save detailed statistics
        save_detailed_stats(df, outdir, period=period)
        
        print(f"✨ Deep analysis for {period.upper()} period completed!")
    
    print("\n✨ All deep analysis plots have been generated!")
    print(f"📂 Report path: {outdir}")
    print("💡 Suggestion: focus first on the regions outlined with blue dashed boxes in the 'robust' heatmap; they represent true parameter plateaus.")

if __name__ == "__main__":
    main()
