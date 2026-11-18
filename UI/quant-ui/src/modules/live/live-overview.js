import { createChart, LineSeries } from "lightweight-charts";
import { escapeHtml, formatMoney, pnlTone } from "../../shared/formatters.js";
import { loadLiveOverview, loadLiveStrategies, releaseLiveViewer } from "./live-api.js";

const STRATEGY_START_DATE = "2026-09-07";
const INITIAL_CAPITAL = 1_700;
const money = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2, maximumFractionDigits: 2,
});
const ratio = new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 2 });
const format = (value, formatter) => Number.isFinite(value) ? formatter.format(value) : "—";
const amount = value => format(value, money);
const percent = value => format(value, ratio);

export function mountOverview(container, context) {
  const controller = new AbortController();
  const viewerId = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
  const release = () => { releaseLiveViewer(viewerId).catch(() => {}); };
  const charts = [];
  let disposed = false;
  let loading = false;
  let liveLoading = false;
  let updatedAt;
  container.innerHTML = `
    <section class="module-page scroll-page live-page">
      <header class="page-header live-portfolio-header">
        <div>
          <h1>Live Portfolio Overview</h1>
          <p class="page-subtitle" data-role="coverage">Bitget · USDT</p>
        </div>
        <button type="button" class="back-button" data-action="strategies">Strategies →</button>
      </header>
      <p role="status"><span data-role="notice">Loading the daily portfolio report…</span><span data-role="live-status"></span></p>
      <dl class="live-portfolio-metrics" data-role="metrics"></dl>
      <div data-role="portfolio" hidden>
        <section class="panel live-portfolio-chart-panel">
          <h2>Daily Balance and Equity</h2>
          <p class="live-portfolio-legend"><span>● End-of-day balance</span><span>● Sampled equity</span></p>
          <div class="live-portfolio-chart" data-role="funds"></div>
        </section>
        <section class="panel live-portfolio-chart-panel">
          <h2>End-of-day Drawdown</h2>
          <div class="live-portfolio-chart live-portfolio-drawdown" data-role="drawdown"></div>
        </section>
      </div>
    </section>`;
  const find = role => container.querySelector(`[data-role="${role}"]`);
  container.querySelector('[data-action="strategies"]').addEventListener("click", () => {
    context.navigate("live", { view: "strategies" });
  });

  const cards = [
    ["Strategy Start Date", "start-date", STRATEGY_START_DATE],
    ["Initial Capital", "initial-capital", amount(INITIAL_CAPITAL)],
    ["Total Balance", "balance"],
    ["Unrealized PnL", "pnl"],
    ["Return", "return"],
    ["Sharpe", "sharpe"],
    ["Calmar", "calmar"],
    ["Max Drawdown", "max-drawdown"],
    ["Longest Drawdown", "drawdown-duration"],
  ];
  find("metrics").innerHTML = cards.map(([label, role, value = "—"]) =>
    `<div class="panel"${role === "balance" || role === "pnl" ? ' title="Live crypto strategies"' : ""}>
      <dt data-role="${role}-label">${label}</dt><dd data-role="${role}">${escapeHtml(value)}</dd></div>`
  ).join("");

  function chart(role) {
    const result = createChart(find(role), { autoSize: true, localization: { locale: "en-US" } });
    charts.push(result);
    return result;
  }

  function renderLive(summary, status = "") {
    const balance = find("balance");
    const pnl = find("pnl");
    balance.textContent = amount(summary?.balance?.value);
    pnl.textContent = formatMoney(summary?.unrealized_pnl?.value, true);
    pnl.className = pnlTone(summary?.unrealized_pnl?.value);

    const coverage = [
      [summary?.balance, "accounts"],
      [summary?.unrealized_pnl, "strategies"],
    ].filter(([metric]) => metric?.included < metric?.total)
      .map(([metric, unit]) => `${metric.included}/${metric.total} ${unit}`);
    const message = status || coverage.join(", ");
    find("live-status").textContent = message ? ` · Live crypto data: ${message}` : "";
  }

  async function refreshLive() {
    if (liveLoading || disposed || document.hidden) return;
    liveLoading = true;
    try {
      const payload = await loadLiveStrategies({ viewerId, signal: controller.signal });
      const summary = payload.pnl_summary.crypto;
      if (!disposed) renderLive(summary, summary.unrealized_pnl.total === 0 ? "Waiting for strategies" : "");
    } catch {
      if (!disposed) renderLive(null, "Live data unavailable");
    } finally {
      liveLoading = false;
    }
  }

  function render(payload) {
    const { metrics, snapshot, balance_series, equity_series } = payload;
    const balances = balance_series.filter(row => row.date >= STRATEGY_START_DATE);
    const equities = equity_series.filter(row => row.date >= STRATEGY_START_DATE);
    const duration = metrics.longest_drawdown;
    find("return-label").textContent = metrics.funded_days < 365 ? "Return (Estimated Annualized)" : "Return (Annualized)";
    find("return").textContent = `${percent(metrics.total_return)} (${percent(metrics.annualized_return)})`;
    find("sharpe").textContent = amount(metrics.sharpe_zero_risk_free);
    find("calmar").textContent = amount(metrics.calmar);
    find("max-drawdown").textContent = percent(metrics.bill_event_max_drawdown);
    find("drawdown-duration").textContent = duration
      ? `${duration.days} days${duration.recovered ? "" : " (unrecovered)"}` : "0 days";
    find("coverage").textContent = `Bitget · ${metrics.accounts} accounts · USDT · ${payload.runner_id}`;
    find("notice").textContent = `Historical equity sampled at ${snapshot.as_of} · Risk metrics use realized NAV through ${metrics.end}`;
    find("portfolio").hidden = false;
    charts.splice(0).forEach(item => item.remove());

    const funds = chart("funds");
    funds.addSeries(LineSeries, { color: "#246ca0" }).setData(
      balances.map(row => ({ time: row.date, value: row.balance })),
    );
    const equityByDate = new Map(equities.map(row => [row.date, row.equity]));
    const dates = [...new Set([...balances.map(row => row.date), ...equityByDate.keys()])].sort();
    funds.addSeries(LineSeries, { color: "#008476", pointMarkersVisible: true }).setData(
      dates.map(time => equityByDate.has(time) ? { time, value: equityByDate.get(time) } : { time }),
    );
    funds.timeScale().fitContent();

    const drawdown = chart("drawdown");
    drawdown.addSeries(LineSeries, {
      color: "#b84d49", priceFormat: { type: "custom", formatter: percent },
    }).setData(balances.map(row => ({ time: row.date, value: row.drawdown })));
    drawdown.timeScale().fitContent();
  }

  async function refresh() {
    if (loading || disposed) return;
    loading = true;
    try {
      const payload = await loadLiveOverview({ signal: controller.signal });
      if (disposed) return;
      if (!payload.available) find("notice").textContent = "Waiting for the runner to publish its first daily portfolio report.";
      else if (payload.updated_at !== updatedAt) {
        render(payload);
        updatedAt = payload.updated_at;
      }
    } catch {
      if (!disposed) {
        find("notice").textContent = updatedAt
          ? `Refresh failed. Showing the report from ${updatedAt}.`
          : "The daily portfolio report is temporarily unavailable.";
        updatedAt = undefined;
      }
    } finally {
      loading = false;
    }
  }

  const timer = setInterval(refresh, 60_000);
  const liveTimer = setInterval(refreshLive, 1_000);
  document.addEventListener("visibilitychange", refreshLive);
  window.addEventListener("pagehide", release);
  renderLive(null, "Waiting for live data");
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
    charts.forEach(item => item.remove());
  };
}
