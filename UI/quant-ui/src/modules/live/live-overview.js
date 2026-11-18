import { createChart, LineSeries } from "lightweight-charts";
import { escapeHtml } from "../../shared/formatters.js";
import { loadLiveOverview } from "./live-api.js";

const money = new Intl.NumberFormat("zh-CN", {
  minimumFractionDigits: 2, maximumFractionDigits: 2,
});
const ratio = new Intl.NumberFormat("zh-CN", { style: "percent", maximumFractionDigits: 2 });
const format = (value, formatter) => Number.isFinite(value) ? formatter.format(value) : "—";
const amount = value => format(value, money);
const percent = value => format(value, ratio);

export function mountOverview(container, context) {
  const controller = new AbortController();
  const charts = [];
  let disposed = false;
  let loading = false;
  let updatedAt;
  container.innerHTML = `
    <section class="module-page scroll-page live-page">
      <header class="page-header live-portfolio-header">
        <div>
          <h1>实盘资金总览</h1>
          <p class="page-subtitle" data-role="coverage">Bitget · USDT</p>
        </div>
        <button type="button" class="back-button" data-action="strategies">策略详情 →</button>
      </header>
      <p data-role="notice" role="status">正在读取日度资金报告…</p>
      <div data-role="portfolio" hidden>
        <dl class="live-portfolio-metrics" data-role="metrics"></dl>
        <section class="panel live-portfolio-chart-panel">
          <h2>日度资金曲线</h2>
          <p class="live-portfolio-legend"><span>● Balance（日末）</span><span>● Equity（采样）</span></p>
          <div class="live-portfolio-chart" data-role="funds"></div>
        </section>
        <section class="panel live-portfolio-chart-panel">
          <h2>日末回撤</h2>
          <div class="live-portfolio-chart live-portfolio-drawdown" data-role="drawdown"></div>
        </section>
      </div>
    </section>`;
  const find = role => container.querySelector(`[data-role="${role}"]`);
  container.querySelector('[data-action="strategies"]').addEventListener("click", () => {
    context.navigate("live", { view: "strategies" });
  });

  function chart(role) {
    const result = createChart(find(role), { autoSize: true });
    charts.push(result);
    return result;
  }

  function render(payload) {
    const { metrics, snapshot, balance_series: balances, equity_series: equities } = payload;
    const duration = metrics.longest_drawdown;
    const cards = [
      ["初始金额", amount(metrics.opening_balance)],
      ["净划入", amount(metrics.net_transfers)],
      ["Total Balance", amount(snapshot.balance)],
      ["Total Equity", amount(snapshot.equity)],
      ["未实现 PnL", amount(snapshot.unrealized_pnl)],
      ["累计收益", percent(metrics.total_return)],
      ["年化收益", percent(metrics.annualized_return)],
      ["Sharpe", amount(metrics.sharpe_zero_risk_free)],
      ["Calmar", amount(metrics.calmar)],
      ["最大回撤（日末）", percent(metrics.daily_max_drawdown)],
      ["最大回撤（逐笔）", percent(metrics.bill_event_max_drawdown)],
      ["最长回撤持续时间", duration ? `${duration.days} 天${duration.recovered ? "" : "（未恢复）"}` : "0 天"],
    ];
    find("metrics").innerHTML = cards.map(([label, value]) =>
      `<div class="panel"><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`
    ).join("");
    find("coverage").textContent = `Bitget · ${metrics.accounts} 个账户 · USDT · ${payload.runner_id}`;
    find("notice").textContent = `权益采样 ${snapshot.as_of} · 风险指标基于截至 ${metrics.end} 的已实现净值`;
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
      if (!payload.available) find("notice").textContent = "等待 Runner 发布第一份日度资金报告。";
      else if (payload.updated_at !== updatedAt) {
        render(payload);
        updatedAt = payload.updated_at;
      }
    } catch {
      if (!disposed) {
        find("notice").textContent = updatedAt
          ? `刷新失败，正在显示 ${updatedAt} 的报告。`
          : "暂时无法读取日度资金报告。";
        updatedAt = undefined;
      }
    } finally {
      loading = false;
    }
  }

  const timer = setInterval(refresh, 60_000);
  refresh();
  return () => {
    disposed = true;
    controller.abort();
    clearInterval(timer);
    charts.forEach(item => item.remove());
  };
}
