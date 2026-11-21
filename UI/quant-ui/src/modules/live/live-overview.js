import { createChart, createSeriesMarkers, LineSeries } from "lightweight-charts";
import { formatMoney, pnlTone } from "../../shared/formatters.js";
import { loadLiveOverview, loadLiveStrategies, releaseLiveViewer } from "./live-api.js";

const INITIAL_CAPITAL = 1_700;
const DAY_MS = 86_400_000;
const money = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2, maximumFractionDigits: 2,
});
const ratio = new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 2 });
const format = (value, formatter) => Number.isFinite(value) ? formatter.format(value) : "—";
const amount = value => format(value, money);
const percent = value => format(value, ratio);
const utcDate = value => new Date(value).toISOString().slice(0, 10);

function chartPoints(rows, field) {
  const points = [];
  for (const row of rows) {
    const previous = points.at(-1);
    if (previous && Date.parse(row.date) - Date.parse(previous.time) > DAY_MS) {
      points.push({ time: utcDate(Date.parse(previous.time) + DAY_MS) });
    }
    points.push({ time: row.date, value: row[field] });
  }
  return points;
}

export function mountOverview(container, context) {
  const controller = new AbortController();
  const viewerId = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
  const release = () => { releaseLiveViewer(viewerId).catch(() => {}); };
  let disposed = false;
  let loading = false;
  let liveLoading = false;
  let report;
  let liveItems = [];
  let funds;
  let equityLine;
  let markers;
  let drawdown;
  let drawdownLine;
  let settledPoints = [];
  let displayedDay;

  container.innerHTML = `
    <section class="module-page scroll-page live-page">
      <header class="page-header live-portfolio-header">
        <div>
          <h1>Live Portfolio Overview</h1>
          <p class="page-subtitle" data-role="coverage">Bitget · USDT</p>
        </div>
        <button type="button" class="back-button" data-action="strategies">Strategies →</button>
      </header>
      <section aria-label="Live assets">
        <h2>Live Assets</h2>
        <p role="status" data-role="live-status">Waiting for live data</p>
        <dl class="live-portfolio-metrics" data-role="live-metrics"></dl>
      </section>
      <section aria-label="Settled strategy performance">
        <h2>Settled Strategy Performance</h2>
        <p role="status" data-role="notice">Loading the daily portfolio report…</p>
        <dl class="live-portfolio-metrics" data-role="metrics"></dl>
        <div data-role="portfolio" hidden>
          <section class="panel live-portfolio-chart-panel">
            <h2>Portfolio Value</h2>
            <p>Daily close at 00:00 UTC. The green marker shows today's live equity.</p>
            <div class="live-portfolio-chart" data-role="funds"></div>
          </section>
          <section class="panel live-portfolio-chart-panel">
            <h2>Daily Drawdown</h2>
            <div class="live-portfolio-chart live-portfolio-drawdown" data-role="drawdown"></div>
          </section>
        </div>
      </section>
    </section>`;
  const find = role => container.querySelector(`[data-role="${role}"]`);
  container.querySelector('[data-action="strategies"]').addEventListener("click", () => {
    context.navigate("live", { view: "strategies" });
  });

  function renderCards(role, cards) {
    find(role).innerHTML = cards.map(([label, key]) =>
      `<div class="panel"><dt data-role="${key}-label">${label}</dt><dd data-role="${key}">—</dd></div>`,
    ).join("");
  }

  renderCards("live-metrics", [
    ["Initial Capital", "initial-capital"],
    ["Total Balance", "balance"],
    ["Unrealized PnL", "pnl"],
    ["Total Equity", "equity"],
  ]);
  renderCards("metrics", [
    ["Strategy Start Date", "start-date"],
    ["Return", "return"],
    ["Sharpe", "sharpe"],
    ["Calmar", "calmar"],
    ["Daily Max Drawdown", "max-drawdown"],
    ["Longest Drawdown", "drawdown-duration"],
  ]);

  find("initial-capital").textContent = amount(INITIAL_CAPITAL);

  function renderLive() {
    if (!report) return;
    const today = utcDate(Date.now());
    const expected = new Set(report.account_ids);
    const accounts = new Map();
    for (const item of liveItems) {
      const observed = Date.parse(item.account_updated_at);
      if (item.runner_id !== report.runner_id || item.venue.toLowerCase() !== "bitget"
          || !expected.has(item.account_id) || item.status === "stopped"
          || !Number.isFinite(item.equity) || !Number.isFinite(observed)
          || utcDate(observed) !== today) continue;
      const previous = accounts.get(item.account_id);
      if (!previous || observed > previous.observed) {
        accounts.set(item.account_id, { observed, equity: item.equity, balance: item.balance });
      }
    }

    let balance = null;
    let equity = null;
    if (expected.size && accounts.size === expected.size) {
      balance = 0;
      equity = 0;
      for (const account of accounts.values()) {
        balance += account.balance;
        equity += account.equity;
      }
    }
    const pnl = equity === null ? null : equity - balance;
    find("balance").textContent = amount(balance);
    find("equity").textContent = amount(equity);
    find("pnl").textContent = formatMoney(pnl, true);
    find("pnl").className = pnlTone(pnl);
    find("live-status").textContent = equity === null
      ? `Waiting for live equity · ${accounts.size}/${expected.size} accounts`
      : "";

    const points = [...settledPoints];
    if (equity !== null) {
      const last = points.at(-1);
      if (last && Date.parse(today) - Date.parse(last.time) > DAY_MS) {
        points.push({ time: utcDate(Date.parse(last.time) + DAY_MS) });
      }
      points.push({ time: today, value: equity });
    }
    equityLine.setData(points);
    markers.setMarkers(equity === null ? [] : [
      { time: today, position: "inBar", shape: "circle", color: "#008476" },
    ]);
    const liveDay = equity === null ? null : today;
    if (liveDay !== displayedDay) funds.timeScale().fitContent();
    displayedDay = liveDay;
  }

  async function refreshLive() {
    if (liveLoading || disposed || document.hidden) return;
    liveLoading = true;
    try {
      const payload = await loadLiveStrategies({ viewerId, signal: controller.signal });
      if (disposed) return;
      liveItems = payload.items;
    } catch {
      if (disposed) return;
      liveItems = [];
    } finally {
      liveLoading = false;
    }
    renderLive();
  }

  function render(payload) {
    const { metrics, funds_series: rows } = payload;
    report = payload;
    find("coverage").textContent = `Bitget · ${payload.account_ids.length} accounts · USDT · ${payload.runner_id}`;
    find("start-date").textContent = payload.strategy_start;
    find("notice").textContent = metrics
      ? `UTC valuation dates ${metrics.start} to ${metrics.end}`
        + (metrics.missing_days ? ` · ${metrics.missing_days} missing daily valuations` : "")
        + (rows.length === 1 ? " · Return needs at least two daily valuations" : "")
      : "Waiting for daily portfolio valuations.";
    find("return-label").textContent = metrics?.days >= 365 ? "Return (Annualized)" : "Return (Estimated Annualized)";
    find("return").textContent = Number.isFinite(metrics?.total_return)
      ? `${percent(metrics.total_return)} (${percent(metrics.annualized_return)})` : "—";
    find("sharpe").textContent = amount(metrics?.sharpe_zero_risk_free);
    find("calmar").textContent = amount(metrics?.calmar);
    find("max-drawdown").textContent = percent(metrics?.daily_max_drawdown);
    const duration = metrics?.longest_drawdown;
    find("drawdown-duration").textContent = duration
      ? `${duration.days} days${duration.recovered ? "" : " (unrecovered)"}` : metrics ? "0 days" : "—";
    find("portfolio").hidden = false;

    if (!funds) {
      const options = { autoSize: true, localization: { locale: "en-US" } };
      funds = createChart(find("funds"), options);
      equityLine = funds.addSeries(LineSeries, { color: "#246ca0", pointMarkersVisible: true });
      markers = createSeriesMarkers(equityLine);
      drawdown = createChart(find("drawdown"), options);
      drawdownLine = drawdown.addSeries(LineSeries, {
        color: "#b84d49", priceFormat: { type: "custom", formatter: percent },
      });
    }
    settledPoints = chartPoints(rows, "value");
    drawdownLine.setData(chartPoints(rows, "drawdown"));
    drawdown.timeScale().fitContent();
    renderLive();
    funds.timeScale().fitContent();
  }

  async function refresh() {
    if (loading || disposed) return;
    loading = true;
    try {
      const payload = await loadLiveOverview({ signal: controller.signal });
      if (disposed) return;
      if (!payload.available) {
        find("notice").textContent = "Waiting for the runner to publish its portfolio report.";
      } else if (payload.updated_at !== report?.updated_at || payload.runner_id !== report?.runner_id) {
        render(payload);
      }
    } catch {
      if (!disposed) find("notice").textContent = report
        ? `Refresh failed. Showing the report from ${report.updated_at}.`
        : "The daily portfolio report is temporarily unavailable.";
    } finally {
      loading = false;
    }
  }

  const timer = setInterval(refresh, 60_000);
  const liveTimer = setInterval(refreshLive, 1_000);
  document.addEventListener("visibilitychange", refreshLive);
  window.addEventListener("pagehide", release);
  refreshLive();
  refresh();
  return () => {
    disposed = true;
    controller.abort();
    clearInterval(timer);
    clearInterval(liveTimer);
    document.removeEventListener("visibilitychange", refreshLive);
    window.removeEventListener("pagehide", release);
    release();
    funds?.remove();
    drawdown?.remove();
  };
}
