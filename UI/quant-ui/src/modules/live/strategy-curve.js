import { createChart, createSeriesMarkers, LineSeries } from "lightweight-charts";
import { loadLiveStrategyCurve } from "./live-api.js";

const DAY_MS = 86_400_000;
const TRANSITION_BACKTEST_DAYS = 60;
const BACKTEST_COLOR = "#246ca0";
const LIVE_COLOR = "#008476";
const precisePrice = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
const compactPrice = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });
const priceFormat = {
  type: "custom",
  minMove: 0.01,
  formatter: value => precisePrice.format(value),
  tickmarksFormatter: values => values.map(value => compactPrice.format(value)),
};
const utcDate = time => new Date(time).toISOString().slice(0, 10);

export function mountStrategyCurve(panel, instanceId) {
  const status = panel.querySelector('[data-role="curve-status"]');
  const element = panel.querySelector('[data-role="curve-chart"]');
  const legend = panel.querySelector('[data-role="curve-legend"]');
  const liveKey = panel.querySelector('[data-role="curve-live-key"]');
  const boundary = panel.querySelector('[data-role="curve-boundary"]');
  const ranges = panel.querySelector(".live-curve-ranges");
  const rangeButtons = [...ranges.querySelectorAll("[data-curve-range]")];
  let controller;
  let disposed = false;
  let chart;
  let backtest;
  let live;
  let markers;
  let data;
  let range = "full";

  function showStatus(message) {
    status.textContent = message;
    status.hidden = !message;
  }

  function applyRange() {
    const firstLive = data.live[0];
    const selected = firstLive ? range : "full";
    for (const button of rangeButtons) {
      button.disabled = !firstLive && button.dataset.curveRange !== "full";
      button.setAttribute("aria-pressed", String(button.dataset.curveRange === selected));
    }

    const scale = chart.timeScale();
    if (selected === "full") {
      scale.fitContent();
      return;
    }
    const lastBacktest = Date.parse(data.backtest.at(-1).time);
    const lastLive = data.live.at(-1).time;
    const from = selected === "live"
      ? (firstLive.time === lastLive ? utcDate(Date.parse(firstLive.time) - DAY_MS) : firstLive.time)
      : utcDate(Math.max(Date.parse(data.backtest[0].time), lastBacktest - TRANSITION_BACKTEST_DAYS * DAY_MS));
    scale.setVisibleRange({ from, to: lastLive });
  }

  for (const button of rangeButtons) {
    button.addEventListener("click", () => {
      range = button.dataset.curveRange;
      applyRange();
    });
  }

  async function refresh() {
    if (disposed || controller || document.hidden) return;
    controller = new AbortController();
    try {
      const payload = await loadLiveStrategyCurve(instanceId, { signal: controller.signal });
      if (disposed) return;
      if (!payload.available) {
        showStatus("No strategy equity history is available.");
        return;
      }

      const firstLive = payload.live[0];
      const receivedFirstLive = firstLive && !data?.live.length;
      data = payload;
      showStatus(firstLive ? "" : "Waiting for the first live daily valuation.");
      legend.hidden = false;
      ranges.hidden = false;
      liveKey.hidden = !firstLive;
      boundary.hidden = !firstLive;
      boundary.textContent = firstLive ? `First live fill · ${payload.live_start} UTC` : "";
      const firstRender = !chart;
      if (firstRender) {
        element.hidden = false;
        chart = createChart(element, {
          autoSize: true,
          localization: { locale: "en-US" },
          layout: { textColor: "#52617b" },
          grid: { vertLines: { color: "#eef2f7" }, horzLines: { color: "#e8edf4" } },
          rightPriceScale: { borderColor: "#d5dce9", scaleMargins: { top: 0.12, bottom: 0.12 } },
          timeScale: {
            borderColor: "#d5dce9", rightOffsetPixels: 20, ignoreWhitespaceIndices: false,
          },
        });
        backtest = chart.addSeries(LineSeries, {
          color: BACKTEST_COLOR, lineWidth: 2, priceFormat, lastValueVisible: false, priceLineVisible: false,
        });
        live = chart.addSeries(LineSeries, {
          color: LIVE_COLOR, lineWidth: 2, priceFormat, lastValueVisible: false, priceLineVisible: false,
        });
        markers = createSeriesMarkers(live);
      }

      const history = [...payload.backtest];
      if (firstLive) {
        const lastBacktest = Date.parse(history.at(-1).time);
        const firstDate = Date.parse(firstLive.time);
        for (let gap = lastBacktest + DAY_MS; gap < firstDate; gap += DAY_MS) {
          history.push({ time: utcDate(gap) });
        }
      }
      backtest.setData(history);
      live.setData(payload.live);
      markers.setMarkers(firstLive ? [
        { time: firstLive.time, position: "inBar", shape: "circle", color: LIVE_COLOR },
      ] : []);
      if (firstRender || receivedFirstLive) applyRange();
    } catch (error) {
      if (!disposed && error.name !== "AbortError") {
        showStatus(chart ? "Curve refresh failed." : "Strategy equity history is unavailable.");
      }
    } finally {
      controller = null;
    }
  }

  refresh();
  const timer = window.setInterval(refresh, 60_000);
  document.addEventListener("visibilitychange", refresh);
  return () => {
    disposed = true;
    controller?.abort();
    window.clearInterval(timer);
    document.removeEventListener("visibilitychange", refresh);
    chart?.remove();
  };
}
