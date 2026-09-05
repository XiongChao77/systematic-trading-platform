from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import operator
import re
import os, sys, time, json, math, shutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))
import copy

# Import project modules
from data_process.common import *
from data_process import common
from trade.runner.analyze_backtest_report import (
    build_continuous_equity_path,
    plot_equity_curves,
)

output_dir = os.path.join(common.PERSISTENCE_DIR, "batch_experiments", "selected_configs")
os.makedirs(output_dir, exist_ok=True)
TOP_K = 50
SKIP_PERCENT = 0  # Percentage of front part to skip; 0 means no skip, select from the very beginning
EQUITY_SCALE = "log"  # Supported values: "linear", "log", or "both".
MAX_LOG_END_VALUE_RATIO = 3.0
RISK_COMPARISON_KEYS = ("risk_per_trade_pct", "max_daily_loss_pct")
KEY_STRATEGY_INDICATORS_FILE = "key_strategy_indicators.png"
KEY_STRATEGY_INDICATORS_ADDITIONAL_INFO = {
    "risk": "risk_per_trade_pct",
    "daily_loss": "max_daily_loss_pct",
    "avg_pct_gross": "forward.avg_pct_gross",
}


def clean_output_dir_except(output_dir_path, preserved_path):
    """Remove output entries, optionally preserving one direct child path."""
    output_dir_path = os.path.abspath(output_dir_path)
    if preserved_path is not None:
        preserved_path = os.path.abspath(preserved_path)
        if os.path.dirname(preserved_path) != output_dir_path:
            raise ValueError(
                "The preserved output must be a direct child of the output directory: " f"output_dir={output_dir_path}, preserved={preserved_path}"
            )

    os.makedirs(output_dir_path, exist_ok=True)
    removed_count = 0
    with os.scandir(output_dir_path) as entries:
        for entry in entries:
            entry_path = os.path.abspath(entry.path)
            if preserved_path is not None and entry_path == preserved_path:
                continue
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path)
            else:
                os.unlink(entry.path)
            removed_count += 1
    return removed_count


COMPARISON_OPERATORS = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}

CRITERION_PATTERN = re.compile(
    r"^(?P<period>[A-Za-z_][A-Za-z0-9_]*)\."
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*(?P<operator>>=|<=|==|!=|>|<)\s*"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"
)


def parse_comparison_criterion(expression):
    """Parse a filter expression such as 'forward.cagr>=0.3'."""
    match = CRITERION_PATTERN.fullmatch(expression.strip())
    if match is None:
        raise ValueError(f"Invalid criterion '{expression}'. " "Expected format such as 'forward.cagr>=0.3'.")

    return (
        match.group("period"),
        match.group("key"),
        match.group("operator"),
        float(match.group("value")),
    )


def _risk_comparison_key(report):
    """Return parameters that must match in a risk-only comparison group."""
    params = copy.deepcopy(report["params"])
    strategy = params.get("strategy", {})
    for key in RISK_COMPARISON_KEYS:
        strategy.pop(key, None)

    # These identifiers are derived from simulation parameters, so they change
    # when either risk parameter changes even though no additional input does.
    params.pop("hash", None)
    identity = params.get("identity", {})
    identity.pop("sim_hash", None)
    identity.pop("full_hash", None)
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def _risk_parameter_values(report):
    strategy = report.get("params", {}).get("strategy", {})
    return tuple(strategy.get(key) for key in RISK_COMPARISON_KEYS)


def find_risk_only_comparison_groups(rows):
    """Group reports whose inputs differ only in the two risk parameters."""
    grouped = {}
    for row in rows:
        report = row["raw"]
        if any(key not in report.get("params", {}).get("strategy", {}) for key in RISK_COMPARISON_KEYS):
            continue
        group = grouped.setdefault(_risk_comparison_key(report), {})
        risk_values = _risk_parameter_values(report)
        group.setdefault(risk_values, row)

    comparisons = [[variants[risk_values] for risk_values in sorted(variants)] for variants in grouped.values() if len(variants) > 1]
    comparisons.sort(key=lambda group: str(group[0]["raw"].get("params", {}).get("hash", "")))
    return comparisons


def filter_comparison_groups(groups, criteria=None):
    """Keep groups containing one strategy variant that satisfies all criteria."""
    candidates = [(group, list(group)) for group in groups]
    expressions = [criteria] if isinstance(criteria, str) else list(criteria or [])

    for expression in expressions:
        period, key, operator_text, threshold = parse_comparison_criterion(expression)
        compare = COMPARISON_OPERATORS[operator_text]

        if not candidates:
            print(f"After screening comparison groups by " f"{period}.{key} {operator_text} {threshold}: 0, 0.00%")
            continue

        candidate_rows = [row for _, matching_rows in candidates for row in matching_rows]
        key_path = _criterion_key_path(candidate_rows, period, key)
        if key_path is None:
            print(f"Warning: key '{key}' not found in {period} reports; " "skipping this filter.")
            continue

        previous_count = len(candidates)
        filtered_candidates = []

        for group, matching_rows in candidates:
            matching_rows = [
                row
                for row in matching_rows
                if _matches_criterion(
                    row,
                    period,
                    key_path,
                    compare,
                    threshold,
                )
            ]
            if matching_rows:
                filtered_candidates.append((group, matching_rows))

        candidates = filtered_candidates
        current_count = len(candidates)
        ratio = current_count / previous_count * 100 if previous_count else 0
        print(f"After screening comparison groups by " f"{period}.{key} {operator_text} {threshold}: " f"{current_count}, {ratio:.2f}%")

    return [group for group, _ in candidates]


def print_comparison_group_metrics(groups):
    """Print long and forward metrics for every strategy about to be plotted."""
    columns = (
        ("Hash", "hash", 12),
        ("per_risk", "risk_per_trade_pct", 12),
        ("MaxDayLoss", "max_daily_loss_pct", 10),
        ("L_CAGR", "l_cagr", 9),
        ("L_Calmar", "l_calmar", 9),
        ("L_Sharpe", "l_sharpe", 9),
        ("L_Freq", "l_daily_freq", 9),
        ("F_CAGR", "f_cagr", 9),
        ("F_Calmar", "f_calmar", 9),
        ("F_Sharpe", "f_sharpe", 9),
        ("F_Freq", "f_daily_freq", 9),
    )
    header = " ".join(f"{label:>{width}}" for label, _, width in columns)

    def format_value(value, width):
        if value is None:
            return f"{'-':>{width}}"
        try:
            return f"{float(value):>{width}.4f}"
        except (TypeError, ValueError):
            return f"{str(value):>{width}}"

    for group_index, group in enumerate(groups, start=1):
        print(f"\nComparison group {group_index}/{len(groups)}")
        print(header)
        print("-" * len(header))
        for row in group:
            values = []
            for _, key, width in columns:
                value = row.get(key)
                if key == "hash":
                    values.append(f"{str(value):>{width}}")
                else:
                    values.append(format_value(value, width))
            print(" ".join(values))


def _risk_curve_payload(row):
    """Build a plot payload whose legend names the compared risk settings."""
    detailed = attach_report_details(row)
    report = copy.deepcopy(detailed["raw"])
    risk_per_trade_pct, max_daily_loss_pct = _risk_parameter_values(report)
    report["params"]["hash"] = f"risk_per_trade_pct={risk_per_trade_pct:g}, " f"max_daily_loss_pct={max_daily_loss_pct:g}"
    return {
        "report": report,
        "report_details": detailed["report_details"],
    }


def plot_risk_only_comparisons(
    rows,
    comparison_output_dir,
    *,
    equity_scale=EQUITY_SCALE,
    max_groups=None,
    criteria=None,
):
    """Plot risk variants when one strategy passes all configured criteria."""
    all_groups = find_risk_only_comparison_groups(rows)
    groups = filter_comparison_groups(all_groups, criteria=criteria)
    if criteria:
        print(f"Comparison group filters: {criteria}; " f"before={len(all_groups)}, after={len(groups)}, " f"removed={len(all_groups) - len(groups)}")
    else:
        print(f"Comparison groups before filtering: {len(all_groups)} (no filter)")

    selected_groups = groups if max_groups is None else groups[:max_groups]
    print_comparison_group_metrics(selected_groups)
    os.makedirs(comparison_output_dir, exist_ok=True)
    manifest_path = os.path.join(comparison_output_dir, "comparison_groups.jsonl")

    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for index, group in enumerate(selected_groups, start=1):
            base_hash = str(group[0]["raw"]["params"].get("hash", "unknown"))[:8]
            filename = f"risk_comparison_{index:04d}_{base_hash}.png"
            plot_equity_curves(
                [_risk_curve_payload(row) for row in group],
                comparison_output_dir,
                filename,
                equity_scale=equity_scale,
                align_curve_maxima=True,
            )
            manifest.write(
                json.dumps(
                    {
                        "file": filename,
                        "curve_alignment": "peak",
                        "strategies": [
                            {
                                "hash": row["raw"]["params"].get("hash"),
                                **dict(
                                    zip(
                                        RISK_COMPARISON_KEYS,
                                        _risk_parameter_values(row["raw"]),
                                    )
                                ),
                            }
                            for row in group
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Risk-only comparison plots saved: {len(selected_groups)}; " f"output: {comparison_output_dir}")
    return selected_groups


def analyze_forward_long_correlation(selected):
    """
    Analyze linear correlation between forward and long periods.
    """

    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    forward_cagr = []
    l_cagr = []

    forward_calmar = []
    l_calmar = []

    for r in selected:
        fc = r.get("cagr")
        lc = r.get("l_cagr")
        f_cal = r.get("calmar")
        l_cal = r.get("l_calmar")

        if fc is not None and lc is not None:
            forward_cagr.append(fc)
            l_cagr.append(lc)

        if f_cal is not None and l_cal is not None:
            forward_calmar.append(f_cal)
            l_calmar.append(l_cal)

    if len(forward_cagr) < 5:
        print("Sample size too small to compute correlation")
        return

    print("\n" + "=" * 100)
    print("Forward vs Long correlation analysis")
    print("=" * 100)

    # CAGR
    pearson_cagr = pearsonr(forward_cagr, l_cagr)
    spearman_cagr = spearmanr(forward_cagr, l_cagr)

    print(f"CAGR Pearson:  r = {pearson_cagr.statistic:.4f} | p = {pearson_cagr.pvalue:.4e}")
    print(f"CAGR Spearman: r = {spearman_cagr.statistic:.4f} | p = {spearman_cagr.pvalue:.4e}")

    # Calmar
    if len(forward_calmar) > 5:
        pearson_calmar = pearsonr(forward_calmar, l_calmar)
        spearman_calmar = spearmanr(forward_calmar, l_calmar)

        print(f"\nCalmar Pearson:  r = {pearson_calmar.statistic:.4f} | p = {pearson_calmar.pvalue:.4e}")
        print(f"Calmar Spearman: r = {spearman_calmar.statistic:.4f} | p = {spearman_calmar.pvalue:.4e}")

    print("=" * 100)

    # Quantile monotonicity test
    print("\nQuantile monotonicity check (bucketed by forward CAGR)")

    pairs = list(zip(forward_cagr, l_cagr))
    pairs.sort(key=lambda x: x[0])

    buckets = np.array_split(pairs, 5)

    for i, b in enumerate(buckets):
        long_vals = [x[1] for x in b]
        print(f"Bucket {i+1}: avg long CAGR = {np.mean(long_vals):.4f}")

    print("=" * 100)


def analyze_model_performance_correlation(all_results):
    """
    Analyze correlation between model metrics (Accuracy, F1, Precision, Recall) and l_cagr.
    """
    from scipy.stats import pearsonr, spearmanr
    import pandas as pd

    # 1. Metrics to analyze (must exist in model_metrics)
    metrics_to_check = ["accuracy", "f1_macro", "f1_weighted", "precision_weighted", "recall_weighted"]

    # 2. Extract data
    data_list = []
    for r in all_results:
        # Fetch target return metric
        l_cagr = r.get("l_cagr")
        # Fetch model metrics dict
        model_metrics = r["long"]["model_metrics"]

        if l_cagr is not None and model_metrics:
            row = {"l_cagr": l_cagr}
            # Only extract the 5 metrics shown in the figure
            for m in metrics_to_check:
                val = model_metrics.get(m)
                if val is not None:
                    row[m] = val
            data_list.append(row)

    if len(data_list) < 10:
        print(f"Warning: Sample size too small ({len(data_list)}) for meaningful correlation analysis")
        return

    df = pd.DataFrame(data_list)

    print("\n" + "=" * 80)
    print(f"Model evaluation metrics vs long CAGR correlation (N={len(df)})")
    print("-" * 80)
    print(f"{'Metric Name':<20} | {'Pearson r':>10} | {'p-value':>12} | {'Spearman r':>10}")
    print("-" * 80)

    # 3. Compute correlation between each metric and l_cagr
    for m in metrics_to_check:
        if m not in df.columns:
            continue

        # Drop NaNs
        sub_df = df[["l_cagr", m]].dropna()
        if len(sub_df) < 5:
            continue

        p_r, p_val = pearsonr(sub_df[m], sub_df["l_cagr"])
        s_r, _ = spearmanr(sub_df[m], sub_df["l_cagr"])

        # Mark statistical significance
        sig = "*" if p_val < 0.05 else ""

        print(f"{m:<20} | {p_r:10.4f}{sig} | {p_val:12.2e} | {s_r:10.4f}")

    print("=" * 80)
    print("Note: Pearson r close to 1 implies strong positive linear correlation; p-value < 0.05 (*) is statistically significant.")


ML_PERFORMANCE_METRICS = {
    "accuracy": "model_metrics.accuracy",
    "directional_accuracy": "model_metrics.signal.directional_accuracy",
    "f1_macro": "model_metrics.f1_macro",
    "f1_weighted": "model_metrics.f1_weighted",
    "precision_weighted": "model_metrics.precision_weighted",
    "recall_weighted": "model_metrics.recall_weighted",
}

TRADING_PERFORMANCE_METRICS = {
    "cagr": "performance.cagr",
    "calmar": "performance.calmar",
    "sharpe": "performance.sharpe",
    "gross_return": "performance.gross_return",
    "max_dd_pct": "drawdown.max_dd_pct",
    "avg_pct_gross": "trades.avg_pct_gross",
    "profit_factor": "trades.profit_factor",
    "win_rate": "trades.win_rate",
    "daily_freq": "trades.daily_freq",
}


def _finite_metric(report, path):
    value = common.recursive_get(report, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _ml_trading_observations(all_results, period, group_by_model):
    rows = []
    metric_paths = {
        **ML_PERFORMANCE_METRICS,
        **TRADING_PERFORMANCE_METRICS,
    }

    for result in all_results:
        period_result = result.get(period) or {}
        params = period_result.get("params") or {}
        identity = params.get("identity") or {}
        prep_hash = identity.get("prep_hash")
        train_hash = identity.get("train_hash")
        if not prep_hash or not train_hash:
            continue

        row = {
            "prep_hash": str(prep_hash),
            "train_hash": str(train_hash),
            "strategy_hash": str(params.get("hash") or result.get("hash") or ""),
        }
        row.update({metric_name: _finite_metric(period_result, metric_path) for metric_name, metric_path in metric_paths.items()})
        rows.append(row)

    strategy_frame = pd.DataFrame(rows)
    if strategy_frame.empty or not group_by_model:
        return strategy_frame, strategy_frame

    metric_columns = list(metric_paths)
    model_frame = strategy_frame.groupby(
        ["prep_hash", "train_hash"],
        as_index=False,
    )[metric_columns].mean()
    model_counts = strategy_frame.groupby(["prep_hash", "train_hash"]).size().rename("strategy_count").reset_index()
    model_frame = model_frame.merge(
        model_counts,
        on=["prep_hash", "train_hash"],
        how="left",
    )
    return strategy_frame, model_frame


def _pair_correlation(frame, first_metric, second_metric, min_samples):
    from scipy.stats import pearsonr, spearmanr

    pair = frame[[first_metric, second_metric]].dropna()
    sample_count = len(pair)
    if sample_count < min_samples or pair[first_metric].nunique() < 2 or pair[second_metric].nunique() < 2:
        return {
            "sample_count": sample_count,
            "pearson_r": None,
            "pearson_p_value": None,
            "spearman_r": None,
            "spearman_p_value": None,
        }

    pearson = pearsonr(pair[first_metric], pair[second_metric])
    spearman = spearmanr(pair[first_metric], pair[second_metric])
    return {
        "sample_count": sample_count,
        "pearson_r": float(pearson.statistic),
        "pearson_p_value": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p_value": float(spearman.pvalue),
    }


def _save_ml_trading_heatmap(correlations, value_column, title, output_path):
    matrix = correlations.pivot(
        index="ml_metric",
        columns="trading_metric",
        values=value_column,
    )
    if matrix.empty or matrix.isna().all().all():
        return

    figure_width = max(10, len(matrix.columns) * 1.4)
    figure_height = max(5, len(matrix.index) * 0.8)
    plt.figure(figsize=(figure_width, figure_height))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
    )
    plt.title(title)
    plt.xlabel("Trading performance")
    plt.ylabel("ML performance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def analyze_ml_trading_correlation(
    all_results,
    period="forward",
    output_dir_path=None,
    group_by_model=True,
    min_samples=5,
):
    """Analyze ML evaluation metrics against trading performance for one period."""
    if period not in {"long", "forward"}:
        raise ValueError("period must be 'long' or 'forward'")
    if min_samples < 3:
        raise ValueError("min_samples must be at least 3")

    strategy_frame, observation_frame = _ml_trading_observations(
        all_results,
        period,
        group_by_model,
    )
    correlation_rows = []
    for ml_metric in ML_PERFORMANCE_METRICS:
        for trading_metric in TRADING_PERFORMANCE_METRICS:
            correlation_rows.append(
                {
                    "ml_metric": ml_metric,
                    "trading_metric": trading_metric,
                    **_pair_correlation(
                        observation_frame,
                        ml_metric,
                        trading_metric,
                        min_samples,
                    ),
                }
            )
    correlations = pd.DataFrame(correlation_rows)

    observation_label = "models" if group_by_model else "strategies"
    print("\n" + "=" * 100)
    print(
        f"{period.title()} ML performance vs trading performance correlation | "
        f"strategies={len(strategy_frame)}, {observation_label}={len(observation_frame)}"
    )
    print("=" * 100)
    ranked = correlations.dropna(subset=["spearman_r"]).copy()
    if ranked.empty:
        print("Not enough non-constant observations to compute correlations")
    else:
        ranked["abs_spearman_r"] = ranked["spearman_r"].abs()
        ranked.sort_values("abs_spearman_r", ascending=False, inplace=True)
        print(
            ranked[
                [
                    "ml_metric",
                    "trading_metric",
                    "sample_count",
                    "pearson_r",
                    "pearson_p_value",
                    "spearman_r",
                    "spearman_p_value",
                ]
            ]
            .head(20)
            .to_string(index=False, float_format=lambda value: f"{value:.4f}")
        )
    print("=" * 100)

    if output_dir_path is not None:
        os.makedirs(output_dir_path, exist_ok=True)
        correlations_path = os.path.join(
            output_dir_path,
            f"{period}_ml_trading_correlations.csv",
        )
        observations_path = os.path.join(
            output_dir_path,
            f"{period}_ml_trading_observations.csv",
        )
        correlations.to_csv(correlations_path, index=False)
        observation_frame.to_csv(observations_path, index=False)
        _save_ml_trading_heatmap(
            correlations,
            "pearson_r",
            f"{period.title()} ML vs Trading Pearson Correlation",
            os.path.join(output_dir_path, f"{period}_ml_trading_pearson.png"),
        )
        _save_ml_trading_heatmap(
            correlations,
            "spearman_r",
            f"{period.title()} ML vs Trading Spearman Correlation",
            os.path.join(output_dir_path, f"{period}_ml_trading_spearman.png"),
        )
        print(f"Correlation outputs saved to: {output_dir_path}")

    return {
        "strategy_observations": strategy_frame,
        "observations": observation_frame,
        "correlations": correlations,
    }


def analyze_model_metrics_by_decile(all_results):
    """
    Bucket analysis: split return metrics (CAGR, Calmar, Sharpe) into 10 quantile buckets
    and observe the average model metrics (Accuracy, F1, etc.) within each bucket.
    """
    import pandas as pd
    import numpy as np

    # 1. Config
    trading_metrics = ["l_cagr", "l_calmar", "l_sharpe"]
    model_keys = ["accuracy", "f1_macro", "f1_weighted", "precision_weighted", "recall_weighted"]

    # 2. Extract data
    data_list = []
    for r in all_results:
        # Fetch trading performance
        row = {
            "l_cagr": r.get("l_cagr"),
            "l_calmar": r.get("l_calmar"),
            "l_sharpe": r.get("long", {}).get("performance", {}).get("sharpe"),  # Some versions may use slightly different key names
        }

        # Fetch model metrics
        model_metrics = r["long"].get("model_metrics", {})
        for mk in model_keys:
            row[mk] = model_metrics.get(mk)

        if row["l_cagr"] is not None:
            data_list.append(row)

    if len(data_list) < 20:
        print("Warning: Not enough data for decile analysis")
        return

    df = pd.DataFrame(data_list)

    # 3. For each trading metric, run bucket analysis
    for t_metric in trading_metrics:
        if t_metric not in df.columns or df[t_metric].isnull().all():
            continue

        print("\n" + "=" * 100)
        print(f"Bucket analysis: model metrics ranked by {t_metric.upper()} (10% quantile buckets)")
        print("=" * 100)

        # Use qcut to divide trading metric into 10 equal-sized buckets (deciles).
        # duplicates='drop' avoids failure when too many identical values exist.
        try:
            df["bucket"] = pd.qcut(df[t_metric], 10, labels=[f"Q{i+1}" for i in range(10)], duplicates="drop")
        except ValueError:
            # If samples are too few or values too concentrated, fall back to 5 buckets
            df["bucket"] = pd.qcut(df[t_metric], 5, labels=[f"Q{i+1}" for i in range(5)], duplicates="drop")
            print(f"Note: due to data distribution, {t_metric} bucket count automatically reduced to 5.")

        # Aggregate and compute mean per bucket
        bucket_stats = df.groupby("bucket", observed=True)[model_keys].mean()

        # Add bucket-wise average trading metric as reference
        bucket_stats[f"avg_{t_metric}"] = df.groupby("bucket", observed=True)[t_metric].mean()

        # Reorder columns to put reference metric first
        cols = [f"avg_{t_metric}"] + model_keys
        bucket_stats = bucket_stats[cols]

        # Print results
        pd.options.display.max_columns = None
        pd.options.display.width = 1000
        print(
            bucket_stats.to_string(
                formatters={f"avg_{t_metric}": "{:,.4f}".format, "accuracy": "{:,.4f}".format, "f1_macro": "{:,.4f}".format, "f1_weighted": "{:,.4f}".format}
            )
        )

        # Simple monotonicity hint
        first_val = bucket_stats[model_keys[0]].iloc[0]
        last_val = bucket_stats[model_keys[0]].iloc[-1]
        trend = "positively monotonic" if last_val > first_val else "non-monotonic or reversed"
        print(f"\nNote: Trend check ({model_keys[0]}): from lowest to highest bucket -> {trend}")
        print("-" * 100)


def merge_selected(records):
    """
    Deduplicate records by hash and return a unique list.
    """
    reslut_set = set()
    uni_results = []
    duplicate_r = []

    for r in records:
        h = r["hash"]
        if h not in reslut_set:
            reslut_set.add(h)
            # print(f" Duplicate record {h}")
            uni_results.append(r)
        else:
            duplicate_r.append(r)
    print(f" Total:{len(records)} Duplicate records {len(duplicate_r)}, uni_results {len(uni_results)}")
    return uni_results


def iter_reports_jsonl(root_list):
    """
    Recursively scan for all reports.jsonl files under given roots.
    """
    for root in root_list:
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname == "reports.jsonl":
                    yield os.path.join(dirpath, fname)


def load_reports(path):
    """
    Read jsonl file line by line and skip malformed lines.
    """
    reports = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                reports.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return reports


def _copy_selected_model_artifacts(selected_rows, output_dir_path):
    """Copy each unique selected model while preserving the train hierarchy."""
    copied_models = set()

    for row in selected_rows:
        raw_data = row.get("raw")
        if not raw_data:
            continue

        params = raw_data.get("params", {})
        identity = params.get("identity", {})
        prep_hash = str(identity.get("prep_hash", ""))
        train_hash = str(identity.get("train_hash", ""))
        train_output_dir = params.get("data", {}).get("train_output_dir")
        report_hash = params.get("hash", "unknown")

        if not prep_hash or not train_hash or not train_output_dir:
            raise ValueError("Selected report is missing model artifact identity or path: " f"report_hash={report_hash}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", prep_hash) or not re.fullmatch(r"[A-Za-z0-9_-]+", train_hash):
            raise ValueError("Selected report contains an unsafe model artifact identity: " f"prep_hash={prep_hash}, train_hash={train_hash}")

        source_dir = os.path.abspath(train_output_dir)
        target_dir = os.path.join(
            os.path.abspath(output_dir_path),
            "train",
            f"pre_{prep_hash}",
            f"train_{train_hash}",
        )
        model_key = os.path.abspath(target_dir)
        if model_key in copied_models:
            continue
        if not os.path.isdir(source_dir):
            raise FileNotFoundError("Selected model artifact directory does not exist: " f"report_hash={report_hash}, path={source_dir}")

        if source_dir != os.path.abspath(target_dir):
            shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
        copied_models.add(model_key)

    print(f"Copied {len(copied_models)} unique model artifact directories to: " f"{os.path.join(output_dir_path, 'train')}")


def save_raw_reports(
    selected_rows,
    exp_dir="",
    output_filename="reports_raw.jsonl",
    copy_models=False,
):
    """
    Save raw reports and optionally copy their trained model artifacts.
    """
    if not selected_rows:
        print("Warning: No data to save")
        return

    out_path = os.path.join(exp_dir, output_filename)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    print(f"Extracting and saving {len(selected_rows)} raw reports to: {out_path}")

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for row in selected_rows:
                # Extract the raw field (the full original dict loaded at the beginning)
                raw_data = row.get("raw")
                if raw_data:
                    f.write(json.dumps(raw_data, ensure_ascii=False) + "\n")
                    source_details = find_report_details_path(
                        raw_data,
                        row["path"],
                    )
                    target_details = report_details_path(raw_data, out_path)
                    if os.path.abspath(source_details) != os.path.abspath(target_details):
                        os.makedirs(os.path.dirname(target_details), exist_ok=True)
                        shutil.copy2(source_details, target_details)

        if copy_models:
            _copy_selected_model_artifacts(selected_rows, exp_dir)

        print(f"Raw data saved successfully!")
    except Exception as e:
        print(f"Failed to save: {str(e)}")
        raise


def report_details_path(report, report_path):
    identity = report["params"]["identity"]
    return os.path.join(
        os.path.dirname(report_path),
        "sim_output",
        identity["full_hash"],
        "report_details.json",
    )


def report_details_candidates(report, report_path):
    """Return sidecar locations for the saved report and its source experiment."""
    candidates = [report_details_path(report, report_path)]
    data_params = report.get("params", {}).get("data", {})
    artifact_paths = (
        (data_params.get("prep_output_dir"), 1),
        (data_params.get("train_output_dir"), 1),
    )

    for artifact_path, levels_to_train_dir in artifact_paths:
        if not artifact_path:
            continue
        train_dir = os.path.abspath(artifact_path)
        for _ in range(levels_to_train_dir):
            train_dir = os.path.dirname(train_dir)
        if os.path.basename(train_dir) not in {"pre_output", "train_output"}:
            continue
        source_report_path = os.path.join(
            os.path.dirname(train_dir),
            "reports.jsonl",
        )
        candidate = report_details_path(report, source_report_path)
        if candidate not in candidates:
            candidates.append(candidate)

    return candidates


def load_report_details(report, report_path):
    path = find_report_details_path(report, report_path)
    with open(path, "r", encoding="utf-8") as source:
        return json.load(source)


def find_report_details_path(report, report_path):
    candidates = report_details_candidates(report, report_path)
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("Report details file was not found in the saved output or source " f"experiment: candidates={candidates}")


def period_report(report, period, report_details=None):
    materialized = {
        "params": report["params"],
        **report["results"][period],
    }
    if report_details is not None:
        materialized.update(report_details["results"][period])
    return materialized


def attach_report_details(row):
    detailed = dict(row)
    details = load_report_details(row["raw"], row["path"])
    detailed["long"] = period_report(row["raw"], "long", details)
    detailed["forward"] = period_report(row["raw"], "forward", details)
    detailed["report_details"] = details
    return detailed


def extract_row(report, src_path):
    """
    Extract key fields from a single report.
    """
    long = period_report(report, "long")
    forward = period_report(report, "forward")
    perf = long.get("performance", {})
    params = report["params"]
    common = params.get("common", {})
    long_perf = long.get("performance", {})
    long_params = long.get("params", {})
    long_common = long_params.get("common", {})
    forward_perf = forward.get("performance", {})
    return {
        "cagr": perf.get("cagr"),
        "calmar": perf.get("calmar"),
        "daily_freq": forward.get("trades", {}).get("daily_freq"),
        "l_cagr": long_perf.get("cagr"),
        "l_calmar": long_perf.get("calmar"),
        "l_daily_freq": long.get("trades", {}).get("daily_freq"),
        "l_win_rate": long.get("trades", {}).get("win_rate"),
        "l_avg_pct_gross": long.get("trades", {}).get("avg_pct_gross"),
        "l_sharpe": long_perf.get("sharpe"),
        "f_cagr": forward_perf.get("cagr"),
        "f_calmar": forward_perf.get("calmar"),
        "f_sharpe": forward_perf.get("sharpe"),
        "f_daily_freq": forward.get("trades", {}).get("daily_freq"),
        "max_daily_loss_pct": (report.get("params", {}).get("strategy", {}).get("max_daily_loss_pct")),
        "risk_per_trade_pct": (report.get("params", {}).get("strategy", {}).get("risk_per_trade_pct")),
        "hash": params.get("hash", 0),
        "path": src_path,
        "long": long,
        "forward": forward,
        "raw": report,
    }


def _long_report_with_daily_account(report):
    """Return the long-period report with its detailed daily account attached."""
    long_report = report.get("long", {})
    daily_account = long_report.get("raw_analyzer", {}).get("customize", {}).get("daily_account")
    if daily_account is not None:
        return long_report

    return period_report(
        report["raw"],
        "long",
        load_report_details(report["raw"], report["path"]),
    )


def _annualized_return_for_region(daily_account, start, end):
    """Calculate CAGR using the first start balance and last end balance."""
    start_time = pd.to_datetime(start, utc=True, errors="coerce")
    end_time = pd.to_datetime(end, utc=True, errors="coerce")
    if pd.isna(start_time) or pd.isna(end_time) or end_time <= start_time:
        return None

    frame = pd.DataFrame(daily_account)
    required_columns = {"date", "start_equity", "end_equity"}
    if frame.empty or not required_columns.issubset(frame.columns):
        return None

    frame = frame[["date", "start_equity", "end_equity"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["start_equity"] = pd.to_numeric(frame["start_equity"], errors="coerce")
    frame["end_equity"] = pd.to_numeric(frame["end_equity"], errors="coerce")
    frame = frame.dropna().sort_values("date")

    start_date = start_time.normalize()
    end_date = end_time.normalize()
    region = frame[(frame["date"] >= start_date) & (frame["date"] <= end_date)]
    if region.empty:
        return None

    start_balance = float(region.iloc[0]["start_equity"])
    end_balance = float(region.iloc[-1]["end_equity"])
    if start_balance <= 0 or end_balance <= 0:
        return None

    elapsed_years = (end_time - start_time).total_seconds() / (365 * 24 * 60 * 60)
    if elapsed_years <= 0:
        return None
    return (end_balance / start_balance) ** (1 / elapsed_years) - 1


def filter_by_train_valid_test_cagr(reports, min_train_cagr, valid_test_ratio):
    """Filter reports by train CAGR and combined valid/test CAGR.

    Train CAGR uses the train region's starting balance. The valid/test CAGR
    treats valid and test as one continuous region and uses the valid region's
    starting balance. A report passes when train CAGR is strictly greater than
    ``min_train_cagr`` and valid/test CAGR is at least train CAGR multiplied
    by ``valid_test_ratio``.
    """
    min_train_cagr = float(min_train_cagr)
    valid_test_ratio = float(valid_test_ratio)
    if not math.isfinite(min_train_cagr):
        raise ValueError("min_train_cagr must be finite")
    if not math.isfinite(valid_test_ratio) or valid_test_ratio < 0:
        raise ValueError("valid_test_ratio must be finite and non-negative")

    passed = []
    failed = []
    train_filtered_count = 0
    valid_test_filtered_count = 0

    for report in reports:
        long_report = _long_report_with_daily_account(report)
        regions = long_report.get("time", {}).get("regions", {})
        daily_account = long_report.get("raw_analyzer", {}).get("customize", {}).get("daily_account", [])
        train = regions.get("train", {})
        valid = regions.get("valid", {})
        test = regions.get("test", {})
        train_cagr = _annualized_return_for_region(
            daily_account,
            train.get("start"),
            train.get("end"),
        )
        valid_test_cagr = _annualized_return_for_region(
            daily_account,
            valid.get("start"),
            test.get("end"),
        )
        valid_test_threshold = train_cagr * valid_test_ratio if train_cagr is not None else None

        passes_train_filter = train_cagr is not None and train_cagr > min_train_cagr
        if not passes_train_filter:
            train_filtered_count += 1
            failed.append(report)
            continue

        passes_valid_test_filter = (
            valid_test_cagr is not None
            and valid_test_threshold is not None
            and (
                valid_test_cagr >= valid_test_threshold
                or math.isclose(
                    valid_test_cagr,
                    valid_test_threshold,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
        )
        if passes_valid_test_filter:
            passed.append(report)
        else:
            valid_test_filtered_count += 1
            failed.append(report)

    total = len(reports)
    train_remaining_count = total - train_filtered_count
    train_filtered_ratio = train_filtered_count / total * 100 if total else 0
    valid_test_filtered_ratio = valid_test_filtered_count / train_remaining_count * 100 if train_remaining_count else 0
    pass_ratio = len(passed) / total * 100 if total else 0
    print("Filtered by min_train_cagr: " f"{train_filtered_count}/{total} ({train_filtered_ratio:.2f}%); " f"train > {min_train_cagr}")
    print(
        "Further filtered by valid_test_ratio: "
        f"{valid_test_filtered_count}/{train_remaining_count} "
        f"({valid_test_filtered_ratio:.2f}%); "
        f"valid+test >= train CAGR * {valid_test_ratio}"
    )
    print(
        "After train/valid-test CAGR filter: "
        f"{len(passed)}/{total} remaining ({pass_ratio:.2f}%); "
        f"train > {min_train_cagr}, "
        f"valid+test >= train CAGR * {valid_test_ratio}"
    )
    return passed, failed


def basic_filter(all_results):
    basic_filter_results, _ = filter_by_criteria(
        all_results,
        criteria=["long.cagr>=0.4", "long.daily_freq>=0.3", "long.rc_pos_ratio>=0.6", "long.max_hwm_duration_days < 300"],
    )
    print(f"After basic_filter: {len(basic_filter_results)}, " f"{len(basic_filter_results) / len(all_results) * 100:.2f}%")
    return basic_filter_results


def filter_and_rank_strategies(data, metric, k=30, final_sort_key="l_cagr"):
    """
    Select top K strategies by a given metric, then re-sort them by final_sort_key.

    :param data: original strategy list (list of dicts)
    :param metric: metric name (str) or custom lambda function
    :param k: how many top strategies to select (int)
    :param final_sort_key: final sort key for presentation, default 'l_cagr'
    :return: list of strategies after two-stage sorting
    """

    # 1. Define sort key (handle lambda or normal key)
    if callable(metric):
        key_func = metric
    else:
        # Use get to handle missing keys gracefully
        key_func = lambda x: x.get(metric, 0) if x.get(metric) is not None else 0

    # 2. Select top K strategies by the given metric (descending)
    top_k = sorted(data, key=key_func, reverse=True)[:k]

    # 3. Re-sort these K results by final benchmark (e.g., CAGR)
    final_sorted = sorted(top_k, key=itemgetter(final_sort_key), reverse=True)

    return final_sorted


_MODEL_TYPE_SHORT_NAMES = {
    "conv_lstm": "ConvLSTM",
    "logistic_regression": "LogReg",
    "logistic_regression_sklearn": "LogRegSK",
    "transformer": "Trans",
    "xgboost": "XGB",
    "lstm": "LSTM",
    "mamba": "Mamba",
    "tcn": "TCN",
    "cnn": "CNN",
    "svc": "SVC",
}


def _short_model_type(model_type):
    """Return a compact, stable model label for console tables."""
    if model_type is None:
        return "-"
    model_type = str(model_type)
    return _MODEL_TYPE_SHORT_NAMES.get(model_type, model_type[:8])


def _analysis_list_hash(value):
    """Return the list hash used by parameter-group analysis output."""
    key_text = ",".join(map(str, sorted(value)))
    return hash(key_text)


def _format_additional_info_value(value, source_key=None):
    """Format an additional performance-table value without losing structure."""
    if value is None:
        return "-"
    source_name = str(source_key).rsplit(".", 1)[-1]
    if source_name == "avg_pct_gross" and isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return f"{float(value):.4f}"
    if source_name == "feature_conf_list" and isinstance(value, list):
        return f"Hash:{str(_analysis_list_hash(value))[:8]}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _resolve_additional_info_value(report, source_key):
    """Resolve exact paths and period-relative metric shorthand."""

    missing = object()
    value = common.recursive_get(report, source_key, default=missing)
    if value is not missing:
        return value

    if not isinstance(source_key, str) or "." not in source_key:
        return None
    period, relative_key = source_key.split(".", 1)
    period_report = report.get(period)
    if not isinstance(period_report, dict):
        return None

    metric_path = {
        **ML_PERFORMANCE_METRICS,
        **TRADING_PERFORMANCE_METRICS,
    }.get(relative_key, relative_key)
    return common.recursive_get(period_report, metric_path)


def _model_group_key(report):
    """Return the identity of the trained model artifact used by a strategy."""
    params = report.get("long", {}).get("params", {})
    identity = params.get("identity", {})
    prep_hash = identity.get("prep_hash")
    train_hash = identity.get("train_hash")
    if prep_hash and train_hash:
        return prep_hash, train_hash

    return json.dumps(
        {
            "common": params.get("common", {}),
            "train": params.get("train", {}),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _model_group_numbers(reports):
    """Assign one display number to strategies sharing a trained model."""
    group_numbers = {}
    numbers = []
    for report in reports:
        group_key = _model_group_key(report)
        if group_key not in group_numbers:
            group_numbers[group_key] = len(group_numbers) + 1
        numbers.append(group_numbers[group_key])
    return numbers


def plot_key_strategy_indicators(
    column_labels,
    table_rows,
    output_dir,
    file_name=KEY_STRATEGY_INDICATORS_FILE,
):
    """Render the console performance table as a portable PNG artifact."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)

    column_widths = []
    for column_index, label in enumerate(column_labels):
        maximum_length = max(
            len(str(label)),
            max(
                (len(str(row[column_index])) for row in table_rows if column_index < len(row)),
                default=0,
            ),
        )
        column_widths.append(maximum_length + 2)

    total_width = sum(column_widths)
    normalized_widths = [width / total_width for width in column_widths]
    figure_width = max(16.0, min(36.0, total_width * 0.115))
    figure_height = max(3.0, 1.3 + (len(table_rows) + 1) * 0.38)

    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    axis.axis("off")
    table = axis.table(
        cellText=table_rows,
        colLabels=column_labels,
        colWidths=normalized_widths,
        cellLoc="right",
        colLoc="center",
        bbox=[0.0, 0.0, 1.0, 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        cell.set_linewidth(0.5)
        if row_index == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold", ha="center")
        else:
            cell.set_facecolor("#f3f4f6" if row_index % 2 == 0 else "white")
            if column_index in (0, 1, 2):
                cell.set_text_props(ha="left")

    figure.suptitle(
        "Key Strategy Indicators",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    print(f"Key strategy indicators image saved: {output_path}")
    return output_path


def show_key_strategy_indicators(
    all_results,
    output_dir,
    addition_info=None,
    strategy_num_start=1,
):
    """Print and render indicators without regenerating equity or correlation plots."""
    strategy_numbers = [strategy_num_start + index for index in range(len(all_results))]
    model_numbers = _model_group_numbers(all_results)
    strategy_labels = [
        f"M{model_number}-S{strategy_number}"
        for model_number, strategy_number in zip(
            model_numbers,
            strategy_numbers,
        )
    ]
    label_width = max(
        len("Num"),
        max((len(label) for label in strategy_labels), default=0),
    )
    additional_columns = list({**KEY_STRATEGY_INDICATORS_ADDITIONAL_INFO, **(addition_info or {})}.items())
    additional_values = [
        [
            _format_additional_info_value(
                _resolve_additional_info_value(report, source_key),
                source_key,
            )
            for _, source_key in additional_columns
        ]
        for report in all_results
    ]
    additional_widths = [
        max(
            len(str(column_name)),
            max(
                (len(values[index]) for values in additional_values),
                default=1,
            ),
        )
        for index, (column_name, _) in enumerate(additional_columns)
    ]

    print("-" * 20 + "Key strategy indicators" + "-" * 20)
    header = (
        f"{'Num':>{label_width}} {'Hash':<12} {'Model':<8} {'Ver':>3} "
        f"{'L_CAGR':>7} {'F_CAGR':>7} {'Sharpe':>7} {'Calmar':>7} {'MaxDD':>7} "
        f"{'Freq':>6} {'Win':>6} {'RCMed':>7} {'RCPos':>6} {'DDDays':>7}"
    )
    header += "".join(f" {str(column_name):<{width}}" for (column_name, _), width in zip(additional_columns, additional_widths))
    print(header)
    print("-" * len(header))

    column_labels = [
        "Num",
        "Hash",
        "Model",
        "Ver",
        "L_CAGR",
        "F_CAGR",
        "Sharpe",
        "Calmar",
        "MaxDD",
        "Freq",
        "Win",
        "RCMed",
        "RCPos",
        "DDDays",
        *(str(column_name) for column_name, _ in additional_columns),
    ]
    table_rows = []

    for i, r in enumerate(all_results):
        strategy_label = strategy_labels[i]
        long_metric = lambda key: common.recursive_get(r["long"], key)
        forward_metric = lambda key: common.recursive_get(r["forward"], key)
        model_cfg = r.get("long", {}).get("params", {}).get("train", {}).get("model_cfg", {})
        model_type = model_cfg.get("model_type", long_metric("model_type"))
        model_version = model_cfg.get("model_version", long_metric("model_version"))
        line = (
            f"{strategy_label:>{label_width}} {str(long_metric('hash')):<12} "
            f"{_short_model_type(model_type):<8} {str(model_version):>3} "
            f"{long_metric('cagr'):7.2f} {forward_metric('cagr'):7.2f} "
            f"{long_metric('sharpe'):7.2f} {long_metric('calmar'):7.2f} "
            f"{long_metric('max_dd_pct'):7.2f} {long_metric('daily_freq'):6.2f} "
            f"{long_metric('win_rate'):6.2f} {long_metric('rc_median'):7.2f} "
            f"{long_metric('rc_pos_ratio'):6.2f} "
            f"{long_metric('max_hwm_duration_days'):7.2f}"
        )
        line += "".join(f" {value:<{width}}" for value, width in zip(additional_values[i], additional_widths))
        print(line)
        table_rows.append(
            [
                strategy_label,
                str(long_metric("hash")),
                _short_model_type(model_type),
                str(model_version),
                f"{long_metric('cagr'):.2f}",
                f"{forward_metric('cagr'):.2f}",
                f"{long_metric('sharpe'):.2f}",
                f"{long_metric('calmar'):.2f}",
                f"{long_metric('max_dd_pct'):.2f}",
                f"{long_metric('daily_freq'):.2f}",
                f"{long_metric('win_rate'):.2f}",
                f"{long_metric('rc_median'):.2f}",
                f"{long_metric('rc_pos_ratio'):.2f}",
                f"{long_metric('max_hwm_duration_days'):.2f}",
                *additional_values[i],
            ]
        )
    return plot_key_strategy_indicators(
        column_labels,
        table_rows,
        output_dir,
    )


def show_performance(
    all_results,
    output_dir,
    batch_size=5,
    equity_scale=EQUITY_SCALE,
    addition_info=None,
    plot_ood=False,
    strategy_num_start=1,
):
    show_key_strategy_indicators(
        all_results,
        output_dir,
        addition_info=addition_info,
        strategy_num_start=strategy_num_start,
    )
    strategy_numbers = [strategy_num_start + index for index in range(len(all_results))]
    model_numbers = _model_group_numbers(all_results)
    strategy_labels = [
        f"M{model_number}-S{strategy_number}"
        for model_number, strategy_number in zip(model_numbers, strategy_numbers)
    ]
    detailed_results = [attach_report_details(row) for row in all_results]
    for model_number, strategy_number, strategy_label, row in zip(
        model_numbers,
        strategy_numbers,
        strategy_labels,
        detailed_results,
    ):
        row["_model_num"] = model_number
        row["_strategy_num"] = strategy_number
        row["_strategy_label"] = strategy_label
    compute_correlation(detailed_results, output_dir)
    plot_in_batches(
        detailed_results,
        output_dir,
        batch_size,
        equity_scale=equity_scale,
        plot_ood=plot_ood,
    )


def sort_by_correlation_diversity(all_results):
    """
    Compute the average correlation of each strategy with all others and sort by independence.
    """
    import pandas as pd

    # 1. Build returns DataFrame (reuse existing build_return_series)
    returns_dict = {}
    for strategy_number, r in enumerate(all_results, start=1):
        # Assume build_return_series is already defined
        ret = build_return_series(r)
        returns_dict[f"S{strategy_number}"] = ret

    df = pd.DataFrame(returns_dict).dropna()

    # 2. Compute correlation matrix
    corr_matrix = df.corr()

    # 3. Compute average correlation of each strategy with others (excluding diagonal 1.0)
    # Formula: (column_sum - 1) / (num_strategies - 1)
    n = len(corr_matrix)
    mean_corr = (corr_matrix.sum() - 1) / (n - 1)

    # 4. Convert result to DataFrame and sort
    diversity_df = mean_corr.to_frame(name="mean_correlation").sort_values(by="mean_correlation")

    print("\n" + "-" * 20 + " Strategy Diversity Ranking " + "-" * 20)
    print(diversity_df)

    # 5. Reorder all_results according to ranking
    sorted_indices = [int(idx.replace("S", "")) - 1 for idx in diversity_df.index]
    sorted_results = [all_results[idx] for idx in sorted_indices]

    return sorted_results


def build_return_series(report):
    daily = report["long"].get("raw_analyzer", {}).get("customize", {}).get("daily_account", [])

    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df.set_index("date", inplace=True)

    # Use end-of-day equity to compute daily returns.
    df["ret"] = df["end_equity"].pct_change()

    return df["ret"].dropna()


def compute_correlation(all_results, output_dir, strategy_numbers=None):
    """
    Dynamically compute figure size for correlation heatmap to keep cell size fixed and text clear.
    """
    save_path = os.path.join(output_dir, "correlation_heatmap_fixed_cell.png")
    if strategy_numbers is None:
        strategy_numbers = list(range(1, len(all_results) + 1))
    else:
        strategy_numbers = list(strategy_numbers)
    if len(strategy_numbers) != len(all_results) or len(set(strategy_numbers)) != len(strategy_numbers):
        raise ValueError("Strategy numbers must be unique and match the report count")

    # ===== 1. Build return series =====
    returns_dict = {}
    for strategy_number, r in zip(strategy_numbers, all_results):
        # Use build_return_series
        ret = build_return_series(r)
        returns_dict[f"S{strategy_number}"] = ret

    df = pd.DataFrame(returns_dict).dropna()
    if df.empty:
        print("Warning: Data is empty, skip heatmap generation")
        return

    corr_matrix = df.corr()
    num_strategies = len(corr_matrix)

    # ===== 2. Dynamically compute figure size =====
    # Target size (inches) for each cell
    cell_size = 0.5
    # Margin for axes ticks and title
    margin = 3.0

    # Dynamic total width and height
    fig_width = num_strategies * cell_size + margin
    fig_height = num_strategies * cell_size + margin

    # Adjust font size based on number of strategies to avoid label overlap
    font_scale = 1.0 if num_strategies < 20 else 0.8 if num_strategies < 50 else 0.5

    plt.figure(figsize=(fig_width, fig_height))
    sns.set_theme(font_scale=font_scale)

    # ===== 3. Draw heatmap =====
    # cbar_pos can be used to tweak color bar width for large figures
    ax = sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 10 if num_strategies < 30 else 7},  # Dynamically adjust font size inside cells
        cbar_kws={"shrink": 0.8},
    )

    plt.title(f"Strategy Correlation Matrix (N={num_strategies})", fontsize=16, pad=20)
    ax.tick_params(axis="x", top=True, bottom=True, labeltop=True, labelbottom=True)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    # ===== 4. Save image =====
    plt.savefig(save_path, dpi=150)  # 150 DPI is enough because the figure is already large
    plt.close()

    print(f"Dynamic-size correlation heatmap saved (Size: {fig_width:.1f}x{fig_height:.1f} in): {save_path}")


def _equity_reports_by_period(row, plot_ood=False):
    """Materialize the same period reports used by the equity plotter."""
    report = row.get("raw", {})
    details = row.get("report_details", {})
    params = report.get("params")
    period_results = report.get("results")
    if not isinstance(params, dict) or not isinstance(period_results, dict):
        return []

    detail_results = details.get("results", {}) if isinstance(details, dict) else {}
    reports_by_period = []
    for period in ("long", "short", "forward", "all"):
        if period == "forward" and not plot_ood:
            continue
        period_result = period_results.get(period)
        if not isinstance(period_result, dict) or not period_result:
            continue
        materialized = {"params": params, **period_result}
        period_details = detail_results.get(period)
        if isinstance(period_details, dict):
            materialized.update(period_details)
        reports_by_period.append((period, materialized))
    return reports_by_period


def _equity_end_value(row, plot_ood=False):
    """Return the final value of the unscaled, visible equity curve."""
    all_reports_by_period = _equity_reports_by_period(row, plot_ood=True)
    reports_by_period = all_reports_by_period if plot_ood else [item for item in all_reports_by_period if item[0] != "forward"]
    equity_path, _ = build_continuous_equity_path(
        reports_by_period,
        normalize_equity=False,
    )
    if not plot_ood:
        ood_start = next(
            (
                pd.to_datetime(
                    report.get("time", {}).get("regions", {}).get("ood", {}).get("start"),
                    utc=True,
                    errors="coerce",
                )
                for _, report in all_reports_by_period
                if report.get("time", {}).get("regions", {}).get("ood", {}).get("start") is not None
            ),
            None,
        )
        if ood_start is not None and not pd.isna(ood_start):
            equity_path = equity_path[equity_path.index < ood_start.tz_convert(None)]
    if equity_path.empty:
        return None
    values = pd.to_numeric(
        equity_path["continuous_equity"],
        errors="coerce",
    )
    values = values[np.isfinite(values)]
    if values.empty:
        return None
    return float(values.iloc[-1])


def _group_equity_plots_by_log_end_value(
    all_results,
    batch_size,
    max_log_ratio=MAX_LOG_END_VALUE_RATIO,
    plot_ood=False,
):
    """Group curves whose positive log(end value) coordinates are comparable.

    The ratio is computed after applying the logarithm. Its value is therefore
    independent of the logarithm base. End values at or below one do not have a
    positive log coordinate and are isolated rather than compared ambiguously.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not math.isfinite(max_log_ratio) or max_log_ratio < 1:
        raise ValueError("max_log_ratio must be finite and at least 1")

    items = []
    for fallback_index, row in enumerate(all_results, start=1):
        strategy_number = row.get("_strategy_num", fallback_index)
        strategy_label = row.get("_strategy_label", f"S{strategy_number}")
        end_value = _equity_end_value(row, plot_ood=plot_ood)
        log_end_value = math.log(end_value) if end_value is not None and end_value > 1 else None
        item = (strategy_label, row, end_value, log_end_value)
        items.append(item)

    groups = []
    current_group = []
    current_log_values = []
    for item in items:
        log_end_value = item[3]
        if log_end_value is None or not math.isfinite(log_end_value):
            if current_group:
                groups.append(current_group)
                current_group = []
                current_log_values = []
            groups.append([item])
            continue

        candidate_log_values = current_log_values + [log_end_value]
        exceeds_batch_size = len(current_group) >= batch_size
        exceeds_log_range = current_log_values and max(candidate_log_values) / min(candidate_log_values) > max_log_ratio
        if current_group and (exceeds_batch_size or exceeds_log_range):
            groups.append(current_group)
            current_group = []
            current_log_values = []
        current_group.append(item)
        current_log_values.append(log_end_value)
    if current_group:
        groups.append(current_group)
    return groups


def _indexed_equity_plot_payload(row, strategy_label):
    """Build a plot payload while preserving the performance-table index."""
    report = dict(row["raw"])
    report["params"] = dict(report.get("params", {}))
    report["params"]["hash"] = strategy_label
    return {
        "report": report,
        "report_details": row["report_details"],
    }


def plot_in_batches(
    all_results,
    output_dir,
    batch_size=5,
    equity_scale=EQUITY_SCALE,
    plot_ood=False,
):
    sns.set_theme(style="white")
    groups = _group_equity_plots_by_log_end_value(
        all_results,
        batch_size,
        plot_ood=plot_ood,
    )
    for figure_index, group in enumerate(groups, start=1):
        plot_payloads = [_indexed_equity_plot_payload(row, strategy_label) for strategy_label, row, _, _ in group]
        filename = f"batch_{figure_index}.png"
        strategy_labels = ", ".join(strategy_label for strategy_label, _, _, _ in group)
        log_values = [item[3] for item in group if item[3] is not None]
        log_ratio = max(log_values) / min(log_values) if len(log_values) == len(group) and log_values else None
        ratio_text = f"{log_ratio:.3f}" if log_ratio is not None else "n/a"
        print(f"Equity plot {figure_index}: {strategy_labels}; " f"log(end value) ratio={ratio_text}")
        plot_equity_curves(
            plot_payloads,
            output_dir,
            filename,
            equity_scale=equity_scale,
            include_ood=plot_ood,
        )


# filter by long first.  forward only use for verify, the less the better
def main():
    exp_dir1 = os.path.join(common.PERSISTENCE_DIR, "batch_experiments", "DOGEUSDT_15m", "2026-08-25", "21_00_17")
    exp_dir5 = os.path.join(common.PERSISTENCE_DIR, "batch_experiments", "AAVEUSDT_15m", "2026-08-23", "12_12_48")
    exp_dir7 = os.path.join(common.PERSISTENCE_DIR, "batch_experiments", "XLMUSDT_15m", "2026-08-24", "13_07_55")
    exp_dir8 = os.path.join(common.PERSISTENCE_DIR, "batch_experiments", "ETHUSDT_15m", "2026-09-02", "22_01_49")
    exp_dir9 = os.path.join(common.PERSISTENCE_DIR, "batch_experiments", "BTCUSDT_15m", "2026-09-03", "20_33_34")

    exp_dir_list = [exp_dir8]
    filter_report = None
    filter_report = os.path.join(output_dir, "filtered_raw_reports.jsonl")
    removed_count = clean_output_dir_except(output_dir, filter_report)
    print(f"Cleaned output directory: removed {removed_count} entries; " f"preserved {filter_report}")
    report_files = []
    rows = []
    records = []
    if filter_report:
        report_files.append(filter_report)
    else:
        for jsonl_path in iter_reports_jsonl(exp_dir_list):
            report_files.append(jsonl_path)
    for report_file in report_files:
        records = load_reports(report_file)
        for r in records:
            row = extract_row(r, report_file)
            rows.append(row)
    symbol = rows[0]["forward"]["params"]["common"]["symbol"]
    interval = rows[0]["forward"]["params"]["common"]["interval"]
    print(f"Total reports loaded: {len(rows)}")
    uin_records = merge_selected(rows)
    uin_records = sorted(uin_records, key=itemgetter("l_cagr"), reverse=True)
    print(f"Total uint reports: {len(uin_records)}")
    if not filter_report:
        # analyze_holdbar(uin_records,target_key="seq_len", period ='forward',metric_key="daily_freq")
        # plot_risk_only_comparisons(
        #     uin_records,
        #     comparison_output_dir=output_dir,
        #     equity_scale='linear',
        #     max_groups=30,
        #     criteria=["forward.cagr>=0.3","long.cagr>0.5","long.rc_pos_ratio>0.5","long.max_hwm_duration_days<180"],
        # )
        # exit()
        analyze_ml_trading_correlation(uin_records, period="long", output_dir_path=os.path.join(output_dir, "ml_trading_correlation"), group_by_model=True)
        analyze_ml_trading_correlation(uin_records, period="forward", output_dir_path=os.path.join(output_dir, "ml_trading_correlation"), group_by_model=True)
        uin_records = basic_filter(uin_records)
        # analyze_holdbar(uin_records,target_key="predict_num", period ='long',metric_key="cagr")
        # analyze_holdbar(uin_records,target_key="predict_num", period ='forward',metric_key="cagr")
        # plot_heatmap(uin_records,var1_key='fixed_hold_bars',var2_key='predict_num', metric_key="l_cagr",save_path=os.path.join(output_dir,f"l_cagr_heatmap_combined.png"))
        # plot_heatmap(uin_records,var1_key='fixed_hold_bars',var2_key='predict_num', metric_key="f_cagr",save_path=os.path.join(output_dir,f"f_cagr_heatmap_combined.png"))
        # stats, f_map, groups = analyze_holdbar(uin_records,target_key="feature_conf_list",period ='long', metric_key="cagr")
        # stats, f_map, groups = analyze_holdbar(uin_records,target_key="feature_conf_list",period ='forward', metric_key="cagr")
        save_raw_reports(uin_records, output_dir, "filtered_raw_reports.jsonl")
        exit()

    # uin_records, _ = filter_by_criteria(uin_records, criteria=["long.cagr>=0"])
    # uin_records, _ = filter_by_criteria(uin_records, criteria=["forward.cagr>=0"])
    # analyze_holdbar(uin_records,target_key="stride",period ='long', metric_key="cagr")
    # analyze_holdbar(uin_records,target_key="fixed_hold_bars",period ='long', metric_key="cagr")
    # analyze_holdbar(uin_records,target_key="seq_len",period ='long', metric_key="cagr")
    # analyze_holdbar(uin_records,target_key="vol_ewma_span", period ='long',metric_key="cagr")
    # analyze_holdbar(uin_records,target_key="predict_num", period ='long',metric_key="cagr")
    # analyze_holdbar(uin_records,target_key="min_expected_move_pct", period ='long',metric_key="cagr")
    # analyze_holdbar(uin_records,target_key="vol_multiplier_long", period ='long',metric_key="cagr")
    # analyze_holdbar(uin_records,target_key="stop_multiplier_rate_long", period ='long',metric_key="cagr")
    # analyze_holdbar(uin_records,target_key="model_type", period ='long',metric_key="cagr")
    # analyze_holdbar(uin_records,target_key="model_version", period ='long',metric_key="cagr")
    # analyze_holdbar(uin_records,target_key="max_daily_loss_pct", period ='long',metric_key="cagr")
    print("*********************************************************************************************")
    # analyze_holdbar(uin_records,target_key="stride",period ='forward', metric_key="cagr")
    # analyze_holdbar(uin_records,target_key="fixed_hold_bars",period ='forward', metric_key="cagr")
    # analyze_holdbar(uin_records,target_key="seq_len",period ='forward', metric_key="cagr")
    # analyze_holdbar(uin_records,target_key="vol_ewma_span", period ='forward',metric_key="cagr")
    # analyze_holdbar(uin_records,target_key="predict_num", period ='forward',metric_key="cagr")
    # analyze_holdbar(uin_records,target_key="min_expected_move_pct", period ='forward',metric_key="cagr")
    # analyze_holdbar(uin_records,target_key="vol_multiplier_long", period ='forward',metric_key="cagr")
    # analyze_holdbar(uin_records,target_key="stop_multiplier_rate_long", period ='forward',metric_key="cagr")
    # analyze_holdbar(uin_records,target_key="model_type", period ='forward',metric_key="cagr")
    # analyze_holdbar(uin_records,target_key="model_version", period ='forward',metric_key="cagr")
    # analyze_holdbar(uin_records,target_key="max_daily_loss_pct", period ='forward',metric_key="cagr")
    # analyze_model_performance_correlation(uin_records)
    # analyze_model_metrics_by_decile(uin_records)
    # exit()
    uin_records, fail = filter_by_train_valid_test_cagr(uin_records, min_train_cagr=0.3, valid_test_ratio=0.05)
    uin_records, _ = filter_by_criteria(uin_records, criteria=["forward.cagr>=0.1", "forward.risk_per_trade_pct<0.04"])
    # uin_records = [record for record in uin_records if common.recursive_get(record, "long.params.train.model_cfg.model_type") == "logistic_regression"]
    start = 10
    show_count = 40
    # analyze_ml_trading_correlation(uin_records, period="long", output_dir_path=os.path.join(output_dir, "ml_trading_correlation"), group_by_model=True)
    # analyze_ml_trading_correlation(uin_records, period="forward", output_dir_path=os.path.join(output_dir, "ml_trading_correlation"), group_by_model=True)
    # exit()
    selected = uin_records
    selected_hash_filter = ["19bcaa57b5cb", "a48b13dc4e7b"]
    candidate = set(selected_hash_filter)
    # selected = [record for record in selected if record.get("hash") in candidate]

    show_performance(
        selected,
        os.path.join(output_dir, "plot"),
        3,
        addition_info={
            "risk": "risk_per_trade_pct",
            "daily_loss": "max_daily_loss_pct",
            "avg_pct_gross": "forward.avg_pct_gross",
            "stride":"stride"
            # "hold_bars": "fixed_hold_bars",
        },
        plot_ood=True,
    )
    # rc_pos_ratio_results, unselected = filter_by_criteria(
    #     stable_selected1, criteria=["long.rc_pos_ratio>=0.7"]
    # )
    # print(f"-------------After rc_pos_ratio: {len(rc_pos_ratio_results)} reports")

    # sorted_l_sharpe = sorted(rc_pos_ratio_results, key=itemgetter("l_sharpe"), reverse=True)
    # sorted_calmar = sorted(rc_pos_ratio_results, key=itemgetter("l_calmar"), reverse=True)
    # # sorted_l_win_rate = sorted(rc_results, key=itemgetter("l_win_rate"), reverse=True)
    # # sorted_l_daily_freq = sorted(rc_results, key=itemgetter("l_daily_freq"), reverse=True)
    # top_k = 40
    # merged_selected = merge_selected_sort(sorted_l_sharpe[:top_k],sorted_calmar[:top_k],rc_pos_ratio_results[:top_k],period ='long', sort_key='cagr')
    save_raw_reports(
        selected,
        output_dir,
        "selected_configs.jsonl",
        copy_models=True,
    )


def merge_selected_sort(*selected_lists, period="forward", sort_key=None, reverse=True):
    """
    Merge multiple selected lists, deduplicate by hash, then sort by sort_key.

    Parameters
    ----------
    *selected_lists : any number of selected lists
    sort_key : str
        Field name used for sorting, e.g. "l_cagr"
    reverse : bool
        True for descending order (default from large to small)

    Returns
    -------
    list
        New list after deduplication and sorting
    """

    merged_dict = {}

    for selected in selected_lists:
        for row in selected:
            h = row.get("hash")
            if h not in merged_dict:
                merged_dict[h] = row

    result = list(merged_dict.values())

    # Sorting
    if sort_key is not None:
        result.sort(key=lambda x: common.recursive_get(x.get(period), sort_key), reverse=reverse)

    return result


def para_evaluation(rows, label1="Vol 1.9", label2="Vol 1.7"):
    """
    Generic parameter evaluation function.
    label1/label2: description of what these two groups represent; shown as Group 1/2 in the table.
    """
    group_1_data = []
    group_2_data = []

    # 1. Flexible grouping logic
    for row in rows:
        # You can customize the conditions here; the function is otherwise generic
        # vol = row["report"]["params"]["common"]["vol_multiplier_long_long"]
        # if vol == 1.9:
        #     group_1_data.append(row)
        # elif vol == 1.7:
        #     group_2_data.append(row)
        fixed_hold_bars = row["report"]["params"]["common"]["fixed_hold_bars"]
        fixed_hold_bars = row["report"]["params"]["strategy"]["fixed_hold_bars"]
        if fixed_hold_bars == 20 and fixed_hold_bars == 20:
            group_1_data.append(row)
        elif fixed_hold_bars == 20 and fixed_hold_bars == 16:
            group_2_data.append(row)

    # 2. Internal metric extractor
    def extract_metrics(group_list):
        if not group_list:
            return None
        return {
            "cagr": [r["report"]["performance"]["cagr"] for r in group_list],
            "calmar": [r["report"]["performance"]["calmar"] for r in group_list],
            "sharpe": [r["report"]["performance"]["sharpe"] for r in group_list],
            "max_dd": [r["report"]["drawdown"]["max_dd_pct"] for r in group_list],
            "count": len(group_list),
        }

    g1_metrics = extract_metrics(group_1_data)
    g2_metrics = extract_metrics(group_2_data)

    # 3. Build comparison table
    summary_rows = []
    for i, (m, label_desc) in enumerate([(g1_metrics, label1), (g2_metrics, label2)]):
        if m:
            summary_rows.append(
                {
                    "Group": f"Group {i+1}",
                    "Desc": label_desc,  # Description so you know what Group 1 represents
                    "Count": m["count"],
                    "Avg CAGR": f"{np.mean(m['cagr']):.2%}",
                    "Max CAGR": f"{np.max(m['cagr']):.2%}",
                    "Min CAGR": f"{np.min(m['cagr']):.2%}",
                    "Std CAGR": f"{np.std(m['cagr']):.4f}",
                    "Avg Calmar": f"{np.mean(m['calmar']):.2f}",
                    "Max Calmar": f"{np.max(m['calmar']):.2%}",
                    "Avg Sharpe": f"{np.mean(m['sharpe']):.2f}",
                    "Avg MaxDD": f"{np.mean(m['max_dd']):.2f}%",  # Corrected drawdown display
                }
            )

    # 4. Nicely aligned output
    if summary_rows:
        pd.set_option("display.max_columns", None)  # Show all columns
        pd.set_option("display.expand_frame_repr", False)  # Do not wrap lines (keep on one line)
        pd.set_option("display.max_colwidth", None)  # No column width limit
        pd.set_option("display.width", 1000)  # Set sufficiently wide display width
        pd.set_option("display.expand_frame_repr", True)
        df_final = pd.DataFrame(summary_rows).set_index("Group")

        print("\n" + "=" * 120)  # Slightly longer separator line
        print(f"Parameter group comparison (Group 1: {label1} | Group 2: {label2})")
        print("=" * 120)

        # Force single-line printing
        print(df_final.to_string(justify="center", index=True, line_width=1000))
        print("=" * 120)

        # 5. Short conclusion
        cagr1 = np.mean(g1_metrics["cagr"])
        cagr2 = np.mean(g2_metrics["cagr"])
        winner = "Group 1" if cagr1 > cagr2 else "Group 2"
        print(f"Note: Preview conclusion: {winner} has better expected return ({max(cagr1, cagr2):.2%})")
    else:
        print("Error: failed to classify valid data; please check parameters in rows input.")
    exit()


def _criterion_key_path(reports, period, key):
    """Find a metric path from the first report containing the requested key."""
    for report in reports:
        key_path = find_key_path(report.get(period, {}), key)
        if key_path is not None:
            return key_path
    return None


def _matches_criterion(report, period, key_path, compare, threshold):
    """Return whether one report satisfies a parsed comparison criterion."""
    value = get_value_by_path(report.get(period, {}), key_path)
    if value is None:
        return False
    try:
        return compare(value, threshold)
    except TypeError:
        try:
            return compare(float(value), threshold)
        except (TypeError, ValueError):
            return False


def filter_by_criteria(reports, criteria=None):
    """Filter reports sequentially using period-aware comparison expressions.

    Criteria use the same format as ``plot_risk_only_comparisons``, for example
    ``forward.cagr>=0.3`` or ``long.max_hwm_duration_days<180``. Supported
    operators are ``>=``, ``<=``, ``>``, ``<``, ``==``, and ``!=``.
    """
    if not reports:
        return [], []
    expressions = [criteria] if isinstance(criteria, str) else list(criteria or [])
    initial_len = len(reports)
    passed = list(reports)

    for expression in expressions:
        period, key, operator_text, threshold = parse_comparison_criterion(expression)
        compare = COMPARISON_OPERATORS[operator_text]

        if not passed:
            print(f"After screening reports by " f"{period}.{key} {operator_text} {threshold}: 0, 0.00%")
            continue

        prev_len = len(passed)
        key_path = _criterion_key_path(passed, period, key)
        if key_path is None:
            print(f"Warning: key '{key}' not found in {period} reports; " "skipping this filter.")
            continue

        passed = [
            report
            for report in passed
            if _matches_criterion(
                report,
                period,
                key_path,
                compare,
                threshold,
            )
        ]
        curr_len = len(passed)
        ratio = (curr_len / prev_len * 100) if prev_len > 0 else 0

        print(f"After screening reports by " f"{period}.{key} {operator_text} {threshold}: " f"{curr_len}, {ratio:.2f}%")

    final_len = len(passed)
    filtered_count = initial_len - final_len
    print(f"TOTAL SUMMARY: {final_len} remaining, {filtered_count} filtered out, " f"{final_len / initial_len * 100:.2f}%")
    passed_ids = {id(r) for r in passed}
    failed = [r for r in reports if id(r) not in passed_ids]
    return passed, failed


def filter_by_performance(reports, period="forward", min_cagr=None, min_calmar=None, min_sharpe=None, min_rc_cagr_median=None, min_rc_cagr_q25=None):
    """
    Filter reports based on performance metrics.
    """

    def meets_criteria(report):
        perf = report.get(period).get("performance", {})
        if min_cagr is not None and perf.get("cagr", 0) < min_cagr:
            return False
        if min_calmar is not None and perf.get("calmar", 0) < min_calmar:
            return False
        if min_sharpe is not None and perf.get("sharpe", 0) < min_sharpe:
            return False
        if min_rc_cagr_median is not None and recursive_get(perf, "rc_cagr_median") < min_rc_cagr_median:
            return False
        if min_rc_cagr_q25 is not None and recursive_get(perf, "rc_cagr_q25") < min_rc_cagr_q25:
            return False
        return True

    passed = []
    failed = []
    for r in reports:
        if meets_criteria(r):
            passed.append(r)
        else:
            failed.append(r)

    return passed, failed


def filter_by_rc_summary(
    reports,
    period="forward",
    # -- Survival / tail risk --
    min_rc_es_05=None,  # e.g. > -0.8
    min_rc_q05=None,  # e.g. > -0.5
    # -- Holdability / continuity --
    max_rc_longest_neg_run=None,  # e.g. < 300 (days/windows)
    max_rc_neg_ratio=None,  # e.g. < 0.5
    # -- Typical return level --
    min_rc_median=None,  # e.g. > 0
    min_rc_q25=None,  # e.g. > 0
    # -- Stability / dispersion --
    max_rc_cv=None,  # e.g. < 3
    max_rc_mad=None,  # optional
):
    """
    Filter reports based on rc_summary metrics.
    Goal: remove structurally unstable or un-holdable strategies.
    """

    def ok(report):
        rc = report.get(period).get("performance", {}).get("rc_summary", {})
        if not rc:
            return False

        # ---------- Survival (tail) ----------
        if min_rc_es_05 is not None:
            if rc.get("rc_es_05", -math.inf) < min_rc_es_05:
                return False

        if min_rc_q05 is not None:
            if rc.get("rc_q05", -math.inf) < min_rc_q05:
                return False

        # ---------- Holdability (long-term pain) ----------
        if max_rc_longest_neg_run is not None:
            if rc.get("rc_longest_neg_run", math.inf) > max_rc_longest_neg_run:
                return False

        if max_rc_neg_ratio is not None:
            if rc.get("rc_neg_ratio", 1.0) > max_rc_neg_ratio:
                return False

        # ---------- Typical return level ----------
        if min_rc_median is not None:
            if rc.get("rc_median", -math.inf) < min_rc_median:
                return False

        if min_rc_q25 is not None:
            if rc.get("rc_q25", -math.inf) < min_rc_q25:
                return False

        # ---------- Stability ----------
        if max_rc_cv is not None:
            rc_cv = rc.get("rc_cv", math.inf)
            if not math.isnan(rc_cv) and rc_cv > max_rc_cv:
                return False

        if max_rc_mad is not None:
            if rc.get("rc_mad", math.inf) > max_rc_mad:
                return False

        return True

    return [r for r in reports if ok(r)]


def filter_by_trades(reports, period="forward", min_win_rate=35, min_daily_freq=None):
    """
    Filter reports based on trade statistics.
    """

    def meets_criteria(report):
        trades = report.get(period).get("trades", {})
        if min_win_rate is not None and trades.get("win_rate", 0) < min_win_rate:
            return False
        if min_daily_freq is not None and trades.get("daily_freq", 0) < min_daily_freq:
            return False
        return True

    passed = []
    failed = []
    for r in reports:
        if meets_criteria(r):
            passed.append(r)
        else:
            failed.append(r)

    return passed, failed


def find_key_path(obj, target_key, path=None):
    """
    Recursively find the path of target_key in a nested object.
    Returns a path list which can be used to directly index the value.

    Example: find_key_path(report, "fixed_hold_bars") returns ["params", "common", "fixed_hold_bars"]
    """
    if path is None:
        path = []

    if isinstance(obj, dict):
        if target_key in obj:
            return path + [target_key]
        for key, value in obj.items():
            result = find_key_path(value, target_key, path + [key])
            if result is not None:
                return result
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            result = find_key_path(item, target_key, path + [i])
            if result is not None:
                return result

    return None


def get_value_by_path(obj, path):
    """
    Get a value from an object using a path list.

    Example: get_value_by_path(report, ["params", "common", "fixed_hold_bars"])
    """
    current = obj
    try:
        for key in path:
            current = current[key]
        return current
    except (KeyError, IndexError, TypeError):
        return None


def analyze_holdbar(records, target_key="fixed_hold_bars", period="forward", metric_key="cagr"):
    """
    Final enhanced version:
    1. Supports list-type target_key (auto sort, join, and hash).
    2. Returns grouped_records to keep original records grouped.

    Returns:
        analysis_results (list): list of aggregated statistics.
        hash_map (dict): mapping from hash to original list.
        grouped_records (dict): {hash_or_value: [original_records...]}.
    """
    from collections import defaultdict
    import numpy as np

    if not records:
        print("Report list is empty")
        return [], {}, {}

    # 1. Locate path for target_key
    key_path = find_key_path(records[0][period], target_key)
    if key_path is None:
        print(f"Could not find any {target_key}")
        return [], {}, {}

    print(f"Located path for {target_key}: {' -> '.join(map(str, key_path))}")

    # 2. Group records according to key
    grouped_records = defaultdict(list)  # Store original record groups
    hash_map = {}  # Map from hash to list value

    for report in records:
        value = get_value_by_path(report[period], key_path)
        if value is None:
            continue

        # Handle list: sort, join, and hash
        if isinstance(value, list):
            current_key = _analysis_list_hash(value)
            if current_key not in hash_map:
                hash_map[current_key] = value
        else:
            current_key = value

        grouped_records[current_key].append(report)

    if not grouped_records:
        print(f"No valid {target_key} found")
        return [], {}, {}

    # 3. Compute performance statistics per group
    analysis_results = []
    total_count = sum(len(v) for v in grouped_records.values())

    # Sort by key for stable output
    for key in sorted(grouped_records.keys(), key=lambda x: str(x)):
        group_items = grouped_records[key]
        count = len(group_items)

        metric_list = []
        calmar_list = []

        for report in group_items:
            # Here report is already from the grouped original records
            p_report = report.get(period, report)
            perf = p_report.get("performance", {})
            metric = recursive_get(p_report, metric_key)
            calmar = perf.get("calmar")

            if metric is not None:
                metric_list.append(metric)
            if calmar is not None:
                calmar_list.append(calmar)

        # Label for display
        display_label = f"Hash:{str(key)[:8]}" if key in hash_map else key

        analysis_results.append(
            {
                "group_key": key,  # Key for fetching from grouped_records
                "original_value": hash_map.get(key, key),  # Original list or scalar value
                "display_key": display_label,
                "count": count,
                "percentage": (count / total_count) * 100,
                f"avg_{metric_key}": np.mean(metric_list) if metric_list else None,
                "avg_calmar": np.mean(calmar_list) if calmar_list else None,
                f"max_calmar": np.max(calmar_list) if calmar_list else None,
                f"med_calmar": np.median(calmar_list) if calmar_list else None,
                f"max_{metric_key}": np.max(metric_list) if metric_list else None,
                f"std_{metric_key}": np.std(metric_list) if len(metric_list) > 1 else 0,
                f"med_{metric_key}": np.median(metric_list) if metric_list else None,
            }
        )

    # 4. Print table
    print("\n" + "=" * 110)
    print(f"{target_key} {period} analysis (total {total_count} reports)")
    print("=" * 110)
    header = f"{'Value/Hash':<15} {'Count':<8} {'%':<6} {f'{metric_key.upper()}':<12} {'':<2}{'AVG':<6}{'Max':<6}{'Std':<6}{'Med':<6} {'Calmar:':<8}{'AVG':<6}{'MAX':<6}{'Med':<6}"
    print(header)
    print("-" * 110)

    for r in analysis_results:
        fmt = lambda v, p: f"{v:.2%}" if v is not None and p else (f"{v:.2f}" if v is not None else "N/A")
        print(
            f"{str(r['display_key']):<15} {r['count']:<8} {r['percentage']:<5.1f}% {metric_key.upper():<12}  {fmt(r[f'avg_{metric_key}'],1):<6} {fmt(r[f'max_{metric_key}'],1):<6} {fmt(r[f'std_{metric_key}'],0):<6} {fmt(r[f'med_{metric_key}'],0):<6} {'':<8}{fmt(r['avg_calmar'],0):<6} {fmt(r['max_calmar'],0):<6} {fmt(r['med_calmar'],0):<6}"
        )

    print("=" * 110)

    return analysis_results, hash_map, grouped_records


def analyze_feature_regimes(records, target_key="predict_num", period="forward", metric_key="cagr"):
    """
    Specifically used to analyze how different feature configuration lists affect performance.
    """
    from collections import defaultdict
    import numpy as np

    # 1. Locate path
    key_path = find_key_path(records[0], target_key)
    if key_path is None:
        print(f"Key not found: {target_key}")
        return

    # 2. Group by feature combinations
    groups = defaultdict(list)
    for report in records:
        value = get_value_by_path(report, key_path)
        if value is not None:
            # Core fix: convert list to string so it can be used as a dict key
            # For example, ['open', 'high'] becomes "open, high"
            key_repr = ", ".join(sorted(value)) if isinstance(value, list) else str(value)
            groups[key_repr].append(report)

    # 3. Aggregation logic (reuse existing statistics code)
    analysis_results = []
    for key_repr, reports in groups.items():
        metrics = [recursive_get(r.get(period, r), metric_key) for r in reports]
        calmars = [recursive_get(r.get(period, r), "calmar") for r in reports]

        analysis_results.append(
            {"feature_set": key_repr, "count": len(reports), "avg_metric": np.mean(metrics) if metrics else 0, "avg_calmar": np.mean(calmars) if calmars else 0}
        )

    # 4. Sort and print
    print(f"\nFeature configuration set ({target_key}) impact analysis - Period: {period}")
    print("-" * 100)
    for res in sorted(analysis_results, key=lambda x: x["avg_metric"], reverse=True):
        print(
            f"Count: {res['count']:<4} | Avg {metric_key.upper()}: {res['avg_metric']:.2%} | Calmar: {res['avg_calmar']:.2f} | Features: {res['feature_set']}"
        )


def plot_heatmap(selected, var1_key, var2_key, metric_key="l_cagr", save_path="heatmap_combined.png"):
    """
    Generate a 2x2 heatmap grid containing: mean, median, standard deviation, and maximum.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 1. Data preparation
    if not selected:
        print("Warning: no reports available for heatmap generation")
        return

    path1 = find_key_path(selected[0], var1_key)
    path2 = find_key_path(selected[0], var2_key)
    if path1 is None or path2 is None:
        missing = var1_key if path1 is None else var2_key
        print(f"Warning: parameter {missing!r} not found; skipping heatmap")
        return

    matrix_data = []
    for report in selected:
        v1 = get_value_by_path(report, path1)
        v2 = get_value_by_path(report, path2)
        metric = recursive_get(report, metric_key)
        if v1 is not None and v2 is not None and metric is not None:
            matrix_data.append({var1_key: v1, var2_key: v2, "val": metric})

    df = pd.DataFrame(matrix_data)
    if df.empty:
        print("Warning: no complete parameter/metric rows; skipping heatmap")
        return

    # 2. Compute four statistical dimensions
    # Use groupby to aggregate all metrics at once
    agg_df = df.groupby([var1_key, var2_key])["val"].agg(["mean", "median", "std", "max"]).reset_index()

    # 3. Create 2x2 canvas
    # Use context manager to avoid affecting global styling
    with sns.axes_style("white"):
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()

        stats_titles = {"mean": "Mean (Expectation)", "median": "Median (Robustness)", "std": "Std Dev (Volatility)", "max": "Max (Potential)"}

        # Loop and plot four subplots
        for i, stat in enumerate(["mean", "median", "std", "max"]):
            # Convert current metric to pivot table
            pivot_df = agg_df.pivot(index=var1_key, columns=var2_key, values=stat)

            sns.heatmap(pivot_df, annot=True, fmt=".1%", cmap="RdYlBu_r", ax=axes[i], cbar_kws={"label": stat.upper()})
            axes[i].set_title(f"{stats_titles[stat]} - {metric_key.upper()}", fontsize=14, fontweight="bold")
            axes[i].set_xlabel(var2_key)
            axes[i].set_ylabel(var1_key)

    plt.suptitle(f"Parameter Sensitivity Analysis: {var1_key} vs {var2_key}", fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for overall title

    save_directory = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(save_directory, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    print(f"Four-in-one heatmap saved to: {save_path}")
    plt.close()  # Release memory promptly


if __name__ == "__main__":
    main()
    # filter_by_short_longs()
