import "./backtests.css";

import { MarketChart } from "../../shared/market-chart.js";
import { escapeHtml, formatPercent, formatValue, getByPath } from "../../shared/formatters.js";
import { createToast } from "../../shared/toast.js";
import { loadExperimentBacktest, loadLatestBacktest } from "./backtests-api.js";
import { EquityChart } from "./equity-chart.js";

const SUMMARY_FIELDS = [
  { path: "performance.gross_return", label: "Total return", format: "percent" },
  { path: "performance.cagr", label: "CAGR", format: "percent" },
  { path: "drawdown.max_dd_pct", label: "Maximum drawdown", format: "raw-percent" },
  { path: "performance.sharpe", label: "Sharpe" },
  { path: "performance.calmar", label: "Calmar" },
];

const DETAIL_GROUPS = [
  {
    title: "Account and returns",
    fields: [
      { path: "time.start", label: "Start time", format: "date-time", lead: true },
      { path: "time.end", label: "End time", format: "date-time", lead: true },
      { path: "performance.start_value", label: "Start value", format: "currency", lead: true },
      { path: "performance.end_value", label: "End value", format: "currency", lead: true },
      ...SUMMARY_FIELDS,
      { path: "params.hash", label: "Parameter hash" },
    ],
  },
  {
    title: "Risk and drawdown",
    fields: [
      { path: "drawdown.max_dd_pct", label: "Maximum drawdown", format: "raw-percent" },
      { path: "drawdown.max_dd_amt", label: "Maximum drawdown amount", format: "currency" },
      { path: "drawdown.max_daily_dd", label: "Maximum daily drawdown", format: "percent" },
      { path: "drawdown.max_daily_date", label: "Worst day" },
      { path: "drawdown.robust_max_daily_loss", label: "Robust daily loss (2nd–5th average)", format: "percent" },
      { value: topDailyLosses, label: "Top 10 daily losses", format: "percent-list", wide: true },
      { path: "drawdown.dd_3_pct_days", label: "Days above 3% drawdown" },
      { path: "drawdown.dd_4_pct_days", label: "Days above 4% drawdown" },
      { path: "drawdown.dd_5_pct_days", label: "Days above 5% drawdown" },
      { path: "drawdown.max_hwm_duration_days", label: "Maximum recovery days" },
      { value: minimumEquityDistance, label: "Minimum equity vs start", format: "percent" },
      {
        value: maximumLossStatus,
        label: "FTMO 10% maximum-loss limit",
        tone: (value) => value === "Breached" ? "danger" : "success",
      },
      {
        value: dailyLossStatus,
        label: "FTMO 5% daily-loss limit",
        tone: (value) => value === "Breached" ? "danger" : "success",
      },
    ],
  },
  {
    title: "Exposure",
    fields: [
      { path: "strategy.max_margin_level", label: "Peak margin utilization", format: "percent" },
      { path: "exposure.avg_pos", label: "Average position exposure", format: "percent" },
      { path: "exposure.max_pos", label: "Maximum position exposure", format: "percent" },
      { path: "exposure.p95_pos", label: "P95 position exposure", format: "percent" },
      { path: "exposure.risk_per_trade_pct", label: "Risk per trade", format: "percent" },
    ],
  },
  {
    title: "Trades",
    fields: [
      { path: "trades.total", label: "Total trades" },
      { path: "trades.win_rate", label: "Win rate", format: "percent" },
      { path: "trades.daily_freq", label: "Daily frequency", format: "trades-per-day" },
      { path: "trades.lost_longest", label: "Longest losing streak" },
      { path: "trades.won_longest", label: "Longest winning streak" },
    ],
  },
  {
    title: "Holding period",
    fields: [
      { path: "trades.holding_period_bars.count", label: "Closed trades" },
      { path: "trades.holding_period_bars.mean", label: "Mean", format: "bars" },
      { path: "trades.holding_period_bars.min", label: "Minimum", format: "bars" },
      { path: "trades.holding_period_bars.max", label: "Maximum", format: "bars" },
      { path: "trades.holding_period_bars.p10", label: "P10", format: "bars" },
      { path: "trades.holding_period_bars.p25", label: "P25", format: "bars" },
      { path: "trades.holding_period_bars.p50", label: "P50", format: "bars" },
      { path: "trades.holding_period_bars.p75", label: "P75", format: "bars" },
      { path: "trades.holding_period_bars.p90", label: "P90", format: "bars" },
      { path: "trades.holding_period_bars.p95", label: "P95", format: "bars" },
      { path: "trades.holding_period_bars.p99", label: "P99", format: "bars" },
    ],
  },
  {
    title: "Profit and loss",
    fields: [
      { path: "trades.avg_pnl_gross", label: "Average gross P&L", format: "currency" },
      { path: "trades.avg_pct_gross", label: "Average gross return", format: "raw-percent-3" },
      { path: "trades.avg_pnl_net", label: "Average net P&L", format: "currency" },
      { path: "trades.avg_pct_net", label: "Average net return", format: "raw-percent-3" },
      { path: "trades.avg_cost", label: "Average cost per trade", format: "currency" },
      { path: "trades.long_pnl", label: "Long P&L", format: "currency" },
      { path: "trades.long_win_rate", label: "Long win rate", format: "raw-percent" },
      { path: "trades.short_pnl", label: "Short P&L", format: "currency" },
      { path: "trades.short_win_rate", label: "Short win rate", format: "raw-percent" },
    ],
  },
  {
    title: "Model",
    fields: [
      { path: "model_metrics.accuracy", label: "Accuracy", format: "percent" },
      { path: "model_metrics.f1_macro", label: "Macro F1" },
      { path: "model_metrics.f1_weighted", label: "Weighted F1" },
      { path: "params.git_commit", label: "Git commit" },
    ],
  },
];

export async function mountBacktestsPage(container, context) {
  const abortController = new AbortController();
  const toast = createToast();
  const datasetId = context.params.get("dataset_id");
  const recordId = context.params.get("record_id");
  const period = context.params.get("period") || "long";
  const returnTo = context.params.get("return_to");
  const experimentSource = Boolean(datasetId && recordId);
  let chart = null;
  let equityChart = null;
  let rawCandles = [];
  let rawTrades = [];
  let rawEquity = [];
  let reportData = {};
  let activeTimeframe = "original";

  container.innerHTML = `
    <section class="module-page backtests-page">
      <div data-view="overview" class="backtest-view">
        <header class="page-header backtests-header">
          <div>
            <p class="page-eyebrow">Strategy diagnostics</p>
            <div class="backtests-title-line">
              <h1>Backtests</h1>
              <p class="page-subtitle" data-role="source-label">${experimentSource ? "Loading selected experiment report" : "Loading latest standalone report"}</p>
            </div>
          </div>
          <div class="button-row">
            <button data-action="performance" class="secondary-button" type="button" disabled>Performance</button>
            <button data-action="prediction" class="secondary-button" type="button" disabled>Model Prediction</button>
            ${returnTo === "experiments"
              ? '<button data-action="close" class="secondary-button" type="button">Close</button>'
              : '<button data-action="reload" class="secondary-button" type="button">Reload</button>'}
            <span data-role="status" class="status-pill busy">Loading</span>
          </div>
        </header>
        <div data-role="summary" class="summary-grid backtest-summary"></div>
        <section class="panel backtest-equity-panel">
          <div class="equity-legend" aria-label="Equity chart legend">
            <span><i class="legend-line legend-market"></i>Market Price</span>
            <span><i class="legend-line legend-absolute"></i>Absolute Equity</span>
            <span><i class="legend-line legend-log"></i>Log Equity</span>
          </div>
          <div data-role="equity-state" class="page-state"><span class="spinner"></span>Loading backtest report...</div>
          <div data-role="equity-chart" class="backtest-equity-chart hidden"></div>
        </section>
      </div>

      <div data-view="performance" class="backtest-view backtest-subpage hidden">
        <header class="backtest-subpage-header">
          <button data-action="back" class="back-button" type="button" aria-label="Back to backtest overview">&#8592;</button>
          <div>
            <p class="page-eyebrow">Backtest report</p>
            <h1>Performance</h1>
            <p class="page-subtitle" data-role="performance-source">Strategy performance metrics</p>
          </div>
        </header>
        <section data-role="details" class="panel backtest-details"></section>
      </div>

      <div data-view="prediction" class="backtest-view backtest-subpage hidden">
        <header class="backtest-subpage-header">
          <button data-action="back" class="back-button" type="button" aria-label="Back to backtest overview">&#8592;</button>
          <div>
            <p class="page-eyebrow">Backtest report</p>
            <h1>Model Prediction</h1>
            <p class="page-subtitle" data-role="prediction-source">Market prices, labels, and model predictions</p>
          </div>
        </header>
        <div class="backtest-toolbar prediction-toolbar">
          <div class="timeframe-buttons" role="group" aria-label="Chart timeframe">
            <span>Timeframe</span>
            <button type="button" class="timeframe-button active" data-timeframe="original">Original</button>
            <button type="button" class="timeframe-button" data-timeframe="1h">1 hour</button>
            <button type="button" class="timeframe-button" data-timeframe="4h">4 hours</button>
            <button type="button" class="timeframe-button" data-timeframe="1d">Daily</button>
          </div>
        </div>
        <section class="panel backtest-chart-panel">
          <div data-role="chart-state" class="page-state"><span>Open this page after the report has loaded.</span></div>
          <div data-role="chart" class="market-chart backtest-market-chart hidden"></div>
          <div data-role="scrollbar" class="chart-scrollbar hidden">
            <input type="range" min="0" step="1" value="0" aria-label="Visible model prediction chart position" />
          </div>
        </section>
      </div>
    </section>
  `;

  const query = (selector) => container.querySelector(selector);
  const reloadButton = query('[data-action="reload"]');
  const performanceButton = query('[data-action="performance"]');
  const predictionButton = query('[data-action="prediction"]');

  function setStatus(text, state = "") {
    const status = query('[data-role="status"]');
    status.textContent = text;
    status.className = `status-pill${state ? ` ${state}` : ""}`;
  }

  function renderPayload(payload) {
    const normalized = normalizeBacktestPayload(payload);
    const report = normalized.periodReport;
    const reportDetails = normalized.periodDetails;
    rawCandles = normalized.candles;
    rawTrades = reportDetails?.trade_logs || [];
    rawEquity = getByPath(report, "raw_analyzer.customize.daily_account", []);
    reportData = report || {};
    renderSummary(query('[data-role="summary"]'), report);
    renderDetails(query('[data-role="details"]'), report, reportDetails);

    const source = normalized.source || {};
    const sourceLabel = experimentSource
      ? `Experiment ${source.strategy_number === undefined ? String(recordId).slice(0, 12) : `#${source.strategy_number}`} · ${period.toUpperCase()}`
      : `Latest standalone report${normalized.generated_at ? ` · generated ${new Date(normalized.generated_at).toLocaleString()}` : ""}`;
    query('[data-role="source-label"]').textContent = sourceLabel;
    query('[data-role="performance-source"]').textContent = sourceLabel;
    query('[data-role="prediction-source"]').textContent = sourceLabel;
    performanceButton.disabled = false;
    predictionButton.disabled = false;
    activeTimeframe = "original";
    container.querySelectorAll("[data-timeframe]").forEach((button) => {
      button.classList.toggle("active", button.dataset.timeframe === activeTimeframe);
    });
    renderEquityChart();
    setStatus("Ready", "success");
  }

  function renderChart(timeframe) {
    const chartElement = query('[data-role="chart"]');
    const chartState = query('[data-role="chart-state"]');
    const scrollbar = query('[data-role="scrollbar"]');
    const seconds = { "1h": 3600, "4h": 14400, "1d": 86400 }[timeframe];
    const candles = seconds ? resampleCandles(rawCandles, seconds) : rawCandles;
    const trades = seconds ? resampleTrades(rawTrades, seconds) : rawTrades;
    chart?.destroy();
    chart = null;

    if (!candles.length) {
      chartElement.classList.add("hidden");
      scrollbar.classList.add("hidden");
      chartState.className = "page-state";
      chartState.innerHTML = "<strong>No candle data in this report</strong><span>Summary statistics remain available above.</span>";
      return;
    }
    chartState.classList.add("hidden");
    chartElement.classList.remove("hidden");
    scrollbar.classList.remove("hidden");
    chart = new MarketChart(chartElement, {
      slider: scrollbar.querySelector("input"),
      tooltipMode: "backtests",
      showLabels: true,
      showPredictions: true,
    });
    chart.setData(candles, trades);
  }

  function renderEquityChart() {
    const chartElement = query('[data-role="equity-chart"]');
    const chartState = query('[data-role="equity-state"]');
    equityChart?.destroy();
    equityChart = null;

    if (!rawEquity.length) {
      chartElement.classList.add("hidden");
      chartState.className = "page-state";
      chartState.innerHTML = "<strong>No equity data in this report</strong><span>The performance metrics remain available on the Performance page.</span>";
      return;
    }

    chartState.classList.add("hidden");
    chartElement.classList.remove("hidden");
    equityChart = new EquityChart(chartElement);
    if (!equityChart.setData(rawCandles, rawEquity, reportData)) {
      equityChart.destroy();
      equityChart = null;
      chartElement.classList.add("hidden");
      chartState.className = "page-state";
      chartState.innerHTML = "<strong>No valid equity points in this report</strong><span>The performance metrics remain available on the Performance page.</span>";
    }
  }

  function showView(view) {
    if (!container.querySelector(`[data-view="${view}"]`)) return;
    chart?.destroy();
    chart = null;
    equityChart?.destroy();
    equityChart = null;
    container.querySelectorAll("[data-view]").forEach((element) => {
      element.classList.toggle("hidden", element.dataset.view !== view);
    });

    if (view === "overview") renderEquityChart();
    if (view === "prediction") renderChart(activeTimeframe);
  }

  async function load() {
    if (reloadButton) reloadButton.disabled = true;
    setStatus("Loading", "busy");
    try {
      const payload = experimentSource
        ? await loadExperimentBacktest(datasetId, recordId, period, abortController.signal)
        : await loadLatestBacktest(abortController.signal);
      if (!abortController.signal.aborted) renderPayload(payload);
    } catch (error) {
      if (error.name === "AbortError") return;
      setStatus("Load failed", "error");
      query('[data-role="equity-state"]').innerHTML = `<strong>Unable to load this report</strong><span>${escapeHtml(error.message || String(error))}</span>`;
      toast.show(error.message || String(error), true);
    } finally {
      if (!abortController.signal.aborted && reloadButton) reloadButton.disabled = false;
    }
  }

  reloadButton?.addEventListener("click", load, { signal: abortController.signal });
  query('[data-action="close"]')?.addEventListener("click", () => context.navigate("experiments"), { signal: abortController.signal });
  performanceButton.addEventListener("click", () => showView("performance"), { signal: abortController.signal });
  predictionButton.addEventListener("click", () => showView("prediction"), { signal: abortController.signal });
  container.querySelectorAll('[data-action="back"]').forEach((button) => {
    button.addEventListener("click", () => showView("overview"), { signal: abortController.signal });
  });
  container.querySelectorAll("[data-timeframe]").forEach((button) => {
    button.addEventListener("click", () => {
      container.querySelectorAll("[data-timeframe]").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      activeTimeframe = button.dataset.timeframe;
      renderChart(activeTimeframe);
    }, { signal: abortController.signal });
  });

  load();

  return () => {
    abortController.abort();
    chart?.destroy();
    equityChart?.destroy();
    toast.destroy();
    container.innerHTML = "";
  };
}

function normalizeBacktestPayload(payload) {
  const body = payload?.payload || payload || {};
  const reportEnvelope = body.report || {};
  const detailEnvelope = body.report_details || {};
  const results = reportEnvelope.results || {};
  const detailResults = detailEnvelope.results || {};
  const period = body.source?.period
    || ["all", "development", "forward"].find((name) => results[name]);
  const periodResult = period ? results[period] || {} : {};
  const periodDetails = period ? detailResults[period] || {} : {};
  const report = {
    params: reportEnvelope.params || {},
    ...periodResult,
    ...periodDetails,
  };
  return {
    ...body,
    candles: Array.isArray(body.candles) ? body.candles : [],
    periodReport: report,
    periodDetails,
    source: body.source || {
      hash: getByPath(reportEnvelope, "params.hash"),
    },
  };
}

function renderSummary(container, report) {
  container.innerHTML = SUMMARY_FIELDS.map((field) => {
    const value = getByPath(report, field.path);
    return `<article class="summary-card"><span>${field.label}</span><strong>${formatMetric(value, field.format)}</strong></article>`;
  }).join("");
}

function renderDetails(container, report, reportDetails) {
  const standardGroups = DETAIL_GROUPS.map((group) => {
    const cards = group.fields.map((field) => {
      const value = typeof field.value === "function"
        ? field.value(report, reportDetails)
        : getByPath(report, field.path);
      if (value === null || value === undefined) return "";
      const tone = typeof field.tone === "function" ? field.tone(value) : field.tone;
      const classes = [
        "detail-metric",
        field.lead ? "detail-metric-lead" : "",
        field.wide ? "detail-metric-wide" : "",
        tone ? `detail-metric-${tone}` : "",
      ].filter(Boolean).join(" ");
      return `<article class="${classes}"><span>${escapeHtml(field.label)}</span><strong>${escapeHtml(formatMetric(value, field.format))}</strong></article>`;
    }).join("");
    return cards ? performanceSection(group.title, cards) : "";
  }).join("");

  container.innerHTML = [
    standardGroups,
    renderFtmoChallenge(report),
    renderStrategyMetrics(report),
  ].join("");
}

function formatMetric(value, format) {
  if (format === "percent") return formatPercent(value);
  if (format === "raw-percent") return formatPercent(value, { scale: 1 });
  if (format === "raw-percent-3") return formatPercent(value, { scale: 1, decimals: 3 });
  if (format === "currency") {
    const number = Number(value);
    return Number.isFinite(number) ? number.toLocaleString(undefined, { style: "currency", currency: "USD" }) : "—";
  }
  if (format === "bars") return `${formatValue(value, 2)} bars`;
  if (format === "trades-per-day") return `${formatValue(value, 2)} trades/day`;
  if (format === "date-time") return formatDateTime(value);
  if (format === "percent-list") {
    return Array.isArray(value) ? value.map((item) => formatPercent(item)).join(" · ") : "—";
  }
  return formatValue(value);
}

function performanceSection(title, cards, className = "") {
  return `<section class="performance-section ${className}"><h2>${escapeHtml(title)}</h2><div class="detail-metric-grid">${cards}</div></section>`;
}

function renderFtmoChallenge(report) {
  const challenge = getByPath(report, "ftmo_challenge");
  const periods = challenge?.periods;
  if (!periods || typeof periods !== "object") return "";
  const profitTarget = formatPercent(challenge.profit_target_pct);
  const lossLimit = formatPercent(challenge.loss_limit_pct);
  const periodBlocks = Object.entries(periods).map(([periodName, stats]) => {
    if (!stats || typeof stats !== "object") return "";
    const cards = [
      metricCard("Start dates", stats.start_count),
      metricCard(`Reached +${profitTarget}`, countAndRate(stats.profit_target_count, stats.profit_target_rate), "success"),
      metricCard(`Reached -${lossLimit}`, countAndRate(stats.loss_limit_count, stats.loss_limit_rate), "danger"),
      metricCard("Unresolved", countAndRate(stats.unresolved_count, stats.unresolved_rate)),
      metricCard(`Days to +${profitTarget}`, durationPercentiles(stats.duration_days?.profit_target), "", true),
      metricCard(`Days to -${lossLimit}`, durationPercentiles(stats.duration_days?.loss_limit), "", true),
    ].join("");
    return `<div class="performance-subsection"><h3>${escapeHtml(humanize(periodName))}</h3><div class="detail-metric-grid">${cards}</div></div>`;
  }).join("");
  return periodBlocks
    ? `<section class="performance-section"><h2>FTMO challenge simulation</h2>${periodBlocks}</section>`
    : "";
}

function renderStrategyMetrics(report) {
  const strategy = getByPath(report, "strategy");
  if (!strategy || typeof strategy !== "object") return "";
  const cards = Object.entries(strategy).map(([key, value]) => {
    if (value === null || value === undefined || typeof value === "object") return "";
    return metricCard(humanize(key), formatStrategyMetric(key, value));
  }).join("");
  return cards ? performanceSection("Strategy", cards) : "";
}

function metricCard(label, value, tone = "", wide = false) {
  if (value === null || value === undefined) return "";
  const classes = [
    "detail-metric",
    wide ? "detail-metric-wide" : "",
    tone ? `detail-metric-${tone}` : "",
  ].filter(Boolean).join(" ");
  return `<article class="${classes}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(formatValue(value))}</strong></article>`;
}

function topDailyLosses(report) {
  const rows = getByPath(report, "raw_analyzer.customize.daily_account", []);
  if (!Array.isArray(rows)) return null;
  const losses = rows.map((item) => Number(item?.intraday_drawdown_pct))
    .filter((value) => Number.isFinite(value) && value < 0)
    .sort((left, right) => left - right)
    .slice(0, 10);
  return losses.length ? losses : null;
}

function minimumEquityDistance(report, reportDetails) {
  const rawStartValue = getByPath(report, "performance.start_value");
  const rawMinimumEquity = getByPath(reportDetails, "raw_analyzer.customize.global_min_equity");
  const startValue = rawStartValue === null ? NaN : Number(rawStartValue);
  let minimumEquity = rawMinimumEquity === null ? NaN : Number(rawMinimumEquity);
  if (!Number.isFinite(minimumEquity)) {
    const rows = getByPath(report, "raw_analyzer.customize.daily_account", []);
    const equities = Array.isArray(rows)
      ? rows.map((item) => Number(item?.minimum_equity)).filter(Number.isFinite)
      : [];
    minimumEquity = equities.length ? Math.min(...equities) : NaN;
  }
  return Number.isFinite(startValue) && startValue > 0 && Number.isFinite(minimumEquity)
    ? minimumEquity / startValue - 1
    : null;
}

function maximumLossStatus(report, reportDetails) {
  const distance = minimumEquityDistance(report, reportDetails);
  return distance === null ? null : distance < -0.10 ? "Breached" : "Within limit";
}

function dailyLossStatus(report) {
  const rawWorstDailyLoss = getByPath(report, "drawdown.max_daily_dd");
  const worstDailyLoss = rawWorstDailyLoss === null ? NaN : Number(rawWorstDailyLoss);
  return Number.isFinite(worstDailyLoss)
    ? worstDailyLoss < -0.05 ? "Breached" : "Within limit"
    : null;
}

function countAndRate(count, rate) {
  const countText = formatValue(count);
  const rateText = formatPercent(rate);
  return `${countText} (${rateText})`;
}

function durationPercentiles(summary) {
  if (!summary || !summary.count) return "No resolved starts";
  return ["p10", "p25", "p50", "p75", "p90", "p95", "p99"]
    .filter((key) => summary[key] !== null && summary[key] !== undefined)
    .map((key) => `${key.toUpperCase()} ${formatValue(summary[key], 1)}`)
    .join(" · ");
}

function formatStrategyMetric(key, value) {
  if (key === "max_margin_level" || key === "input_f1"
      || key.endsWith("_consistency") || key === "min_expected_move_pct") {
    return formatPercent(value);
  }
  if (key === "fixed_hold_bars") return `${formatValue(value)} bars`;
  return formatValue(value);
}

function formatDateTime(value) {
  if (value === null || value === undefined || value === "") return "—";
  const text = String(value).replace("T", " ");
  if (/Z$|\+00:00$/.test(text)) return text.replace(/Z$|\+00:00$/, " UTC");
  if (/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(text)) return `${text} UTC`;
  return text;
}

function humanize(value) {
  return String(value).replaceAll("_", " ").replace(/\b\w/g, (character) => character.toUpperCase());
}

function resampleCandles(candles, bucketSeconds) {
  const buckets = new Map();
  candles.forEach((raw) => {
    const time = normalizeTimestamp(raw.time);
    if (!Number.isFinite(time)) return;
    const bucketTime = Math.floor(time / bucketSeconds) * bucketSeconds;
    const existing = buckets.get(bucketTime);
    if (!existing) {
      buckets.set(bucketTime, {
        ...raw,
        time: bucketTime,
        open: Number(raw.open),
        high: Number(raw.high),
        low: Number(raw.low),
        close: Number(raw.close),
        volume: Number(raw.volume || 0),
      });
      return;
    }
    existing.high = Math.max(existing.high, Number(raw.high));
    existing.low = Math.min(existing.low, Number(raw.low));
    existing.close = Number(raw.close);
    existing.volume += Number(raw.volume || 0);
    ["label", "pred", "equity", "threshold_long", "threshold_short", "expected_vol", "trend_strength"].forEach((key) => {
      if (raw[key] !== undefined && raw[key] !== null) existing[key] = raw[key];
    });
  });
  return [...buckets.values()].sort((left, right) => left.time - right.time);
}

function resampleTrades(trades, bucketSeconds) {
  return trades.map((trade) => ({
    ...trade,
    dt: Math.floor(Number(trade.dt ?? trade.time) / bucketSeconds) * bucketSeconds,
  }));
}

function normalizeTimestamp(value) {
  const number = Number(value);
  if (Number.isFinite(number)) return number > 10_000_000_000 ? Math.floor(number / 1000) : Math.floor(number);
  const parsed = new Date(value).getTime();
  return Number.isFinite(parsed) ? Math.floor(parsed / 1000) : NaN;
}
