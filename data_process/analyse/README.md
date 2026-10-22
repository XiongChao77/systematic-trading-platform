# Time-series factor analysis

Run from the project root with the project's Python environment:

```powershell
python -m data_process.analyse.factor_analysis --data-dir <preparation-output>
python -m data_process.analyse.factor_analysis --data-dir <preparation-output> --factors RSI_14 MACD_12_26_HIST_PCT --horizons 1 5 30 60 240 1440
python -m data_process.analyse.factor_analysis --data-dir <preparation-output> --factors Smart_
python -m data_process.analyse.factor_analysis --data-dir <preparation-output> --workers 4
```

Omit `--factors` to test every eligible numeric factor. To analyze only the
Smart Money factors in the current `output/data` preparation, use a separate
report directory to preserve the full-factor report:

```powershell
python -m data_process.analyse.factor_analysis --data-dir output/data --output-dir output/data/factor_analysis_smart_money --factors Smart_Money_2_0_30 Smart_Money_2_0_60 Smart_Money_2_0_240 Smart_PriceImpact_30 Smart_PriceImpact_60 Smart_PriceImpact_240 Smart_VolDecay_30 Smart_VolDecay_60 Smart_VolDecay_240 Smart_Minute_Enhanced_30 Smart_Minute_Enhanced_60 Smart_Minute_Enhanced_240
```

By default, `--horizons` maps the 1 minute, 5 minute, 30 minute, 1 hour, 4 hour and 1 day return periods to observed-bar counts using the median input interval. Periods shorter than one input bar are omitted. Pass `--horizons` to override with explicit bar counts.

Both `train_data.feather` and `test_data.feather` are required. The default data directory is `output/data`. Results go to `<data-dir>/factor_analysis`, or `--output-dir`. Reusing an output directory replaces its report files. Inputs are never modified. Dependencies already belong to `pyproject.toml`: pandas, numpy, pyarrow, scipy, statsmodels and scikit-learn.

Automatic selection uses numeric columns, including the raw OHLCV features used by FeatureFactory, excluding the current preparation schema's labels, future-derived fields and metadata. `--factors` accepts exact names, case-insensitive substrings, or glob patterns such as `Smart_` or `Smart_*`; each pattern expands to matching eligible factors, and duplicates are removed. Omit it to analyze every eligible factor. The expanded list is recorded in `metadata.json`. No rows are filtered by label validity: labels are not the prediction targets.

Factor analysis uses a process pool by default (`--workers 4`). Each worker
handles one factor at a time, including all splits and forecast horizons for
that factor. Set `--workers 1` to run serially or choose another positive
worker count. The process pool uses the Windows-safe `spawn` start method.

## Outputs

Progress logs print `START` and `DONE ... elapsed=...s` for each factor, split,
horizon and statistical step, including separate ADF, KPSS, ACF/Ljung-Box,
PACF, IC, Newey-West, MI estimation/shift reference, quantile, rolling IC,
report assembly and output timings. Failed steps log `FAILED`
before propagating the exception. The startup workload line reports the number
of factor/split/horizon jobs; rolling IC logs its window count. The final
`TIMING SUMMARY` is sorted by cumulative time and includes call count, mean and
maximum time. Times are inclusive: parent and child times overlap and must not
be added together. CLI logging is enabled by default. API callers with existing
logging configuration should enable INFO for this module.

Runtime grows with factors x three splits x horizons. Each rolling window
reranks its samples, and the default MI reference performs up to 20 estimates
per prediction job. Use the timings to identify the actual bottleneck before
changing sampling or analysis parameters.

| File | Contents |
| --- | --- |
| diagnostics.csv | Finite counts, missing fraction, ADF/KPSS statistics and p-values, stationarity interpretation, warnings |
| autocorrelation.csv | ACF, PACF and cumulative Ljung-Box statistics/p-values for each lag |
| predictive.csv | Pearson IC, Rank IC, rolling Rank IC mean/std/ICIR, HAC regression, MI and quantile monotonicity |
| quantiles.csv | Counts, mean/median future return and positive fraction per train-defined quantile |
| rolling_ic.csv | Pearson and Spearman correlations computed independently inside each trailing window |
| decay.csv | Correlation and HAC significance across forecast horizons |
| metadata.json | Configuration, factor list, train quantile cuts, time span and interval irregularity flag |

Every table distinguishes `train`, `test`, and `combined`. Combined results are descriptive in-sample results. Quantile thresholds are fitted once on train and frozen on test; the train report also uses these full-train thresholds and is in sample. Duplicate quantile boundaries are collapsed, so discrete factors can have fewer groups. Constant factors and insufficient samples have explicit statuses rather than artificial zero correlations.

## Statistical definitions

- Horizons count **observed bars**, not days: `y[t,h] = close[t+h] / close[t] - 1`. Decay uses cumulative forward returns, not the isolated return at lag h. Each split computes its own returns; training targets cannot cross into test. Gaps in market timestamps remain observed-bar steps and are flagged in metadata. This is a single-asset tool; duplicate timestamps and overlapping/reversed splits are rejected.
- ADF's null is a unit root; KPSS's null is level stationarity. Both use an intercept. Agreement at 5% produces `stationary` or `nonstationary`; other combinations are inconclusive. KPSS interpolation-bound warnings are preserved. No automatic differencing is applied. `--transform diff` runs all tests on the first difference and should use a separate output directory. A fixed z-score does not remove a unit root.
- ADF/KPSS/ACF/PACF/Ljung-Box use the last `--diagnostic-max-samples` observations of the longest finite run (default 10,000). This bounds computation without downsampling or joining across missing values; `diagnostic_n` records the actual count. ADF searches up to `--lags` (default 20). Ljung-Box uses the FFT ACF's cumulative Q statistics.
- Pearson measures linear association and Rank IC measures monotonic association. Both are valid descriptive time-series statistics. Rolling Spearman reranks **inside each window**, not once over all history. ICIR is rolling Rank IC mean divided by its sample standard deviation, without annualization. Overlapping rolling windows are dependent; ICIR is not a significance test. `window_end` is the last factor timestamp, not the time all its future returns became observable.
- Newey-West fits an intercept and a standardized factor on the longest finite paired run; `nw_beta` is return per one factor standard deviation. Bartlett HAC bandwidth is at least h-1 and otherwise uses `floor(4*(n/100)^(2/9))`. Student-t p-values and finite-sample correction are used. `nw_fdr_bh` adjusts all valid factor/horizon regression p-values within each split. Missing-run truncation is recorded as `nw_n`. HAC does not resolve nonstationarity, selection bias or all forms of dependence.
- MI uses sklearn's nearest-neighbor estimator, in nats, on the last `--mi-max-samples` observations of the longest paired finite run (default 3,000). Factors with at most 20 distinct sampled values are treated as discrete. The deterministic circular-shift reference preserves internal serial dependence except at wraparound. `--mi-shifts 19` gives minimum attainable reference p=0.05; increase for finer resolution or use 0 to disable. This is a heuristic stationarity-dependent reference, not an exact general-purpose significance test or a multiple-testing correction.
- High serial correlation can reflect a persistent useful signal. One-bar predictability is not automatically noise; slower decay is not automatically better. Fixed IC or ICIR cutoffs do not establish usefulness across assets and sampling frequencies. Inspect test performance, effective sample size, stability and multiple testing together. No automatic pass/fail factor ranking is imposed.

API usage:

```python
from data_process.analyse.factor_analysis import AnalysisConfig, analyze

tables = analyze(
    "<preparation-output>",
    factors=["RSI_14"],
    config=AnalysisConfig(horizons=(1, 5, 10), mi_shifts=99),
)
```

References: [statsmodels statistical API](https://www.statsmodels.org/stable/api.html), [scikit-learn MI implementation](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/feature_selection/_mutual_info.py).

## Interactive HTML dashboard

After analysis, render its CSV reports as a standalone HTML page:

```powershell
python -m data_process.analyse.factor_dashboard --report-dir <preparation-output>/factor_analysis
python -m data_process.analyse.factor_dashboard --report-dir <preparation-output>/factor_analysis --output <output-path>/factor_dashboard.html
```

The page uses the combined data set only. It starts with a table of all factors
and their statistics by horizon, sorted by the maximum absolute Rank IC across
horizons. Selecting a factor in that table or the chart selector updates the
individual charts below it. The Rank IC return-period selector switches the
Rank IC card and rolling IC curve between 1 minute, 5 minutes, 30 minutes,
1 hour, 4 hours and 1 day, mapped to observed bars using the median input
interval. Other charts show IC decay, quantile returns, Newey-West t-statistics,
ACF/PACF, mutual information and stationarity results.
Plotly JavaScript and report data are
embedded in compressed form, so the page works without a Python server or
internet access. The large `rolling_ic.csv` is streamed in chunks instead of
loaded into memory. Rolling IC curves are evenly sampled to 240 points per
factor/horizon by default; use `--max-rolling-points` to change this.
Only the displayed rolling curve is sampled; summary statistics use the full
report.

## Performance

The analysis reuses split return series and bounds the rolling-statistics
working set, computes rolling Pearson and Rank IC with batched NumPy/SciPy
operations, and calculates quantile summaries with pandas aggregation. With
ETHUSDT 15-minute reports, RSI_14, horizon 1, rolling step 21 and reduced
diagnostic/MI sample caps, the report values matched the earlier implementation
and elapsed time fell from about 99 seconds to about 5 seconds. Runtime varies
with rolling-window count, factor cardinality, machine and configuration.
