import {
  createChart,
  CandlestickSeries,
  HistogramSeries,
} from "lightweight-charts";

import Papa from "papaparse";
import "./style.css";

// CSV must be placed at:
// public/test_data.csv
// 
// Vite will serve it as:
// http://localhost:5173/test_data.csv
const CSV_URL = "/test_data.csv";

let chart;
let candleSeries;
let volumeSeries;
let labelSeries;

let totalBars = 0;
let isUpdatingSlider = false;

let isMeasuring = false;
let startPoint = null;
let rulerRect = null;
let rulerLabel = null;
let candleMap = new Map();
let candleData = [];

const fileInfo = document.getElementById("file-info");
const INITIAL_VISIBLE_BARS = 300;

window.addEventListener("DOMContentLoaded", async () => {
  await loadCsvFromUrl(CSV_URL);
});

async function loadCsvFromUrl(url) {
  fileInfo.textContent = `Loading: ${url}`;

  Papa.parse(url, {
    download: true,
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,

    complete: (result) => {
      const rows = result.data;
      const candles = convertRowsToCandles(rows);

      if (!candles.length) {
        fileInfo.textContent = "No valid OHLC data found.";
        return;
      }
      
      renderChart(candles);
      fileInfo.textContent = `${url} | ${candles.length} rows`;
    },

    error: (err) => {
      console.error(err);
      fileInfo.textContent = "CSV load failed.";
    },
  });
}

function convertRowsToCandles(rows) {
  return rows
    .map((r) => {
      const time = parseTime(r);

      const open = Number(r.open);
      const high = Number(r.high);
      const low = Number(r.low);
      const close = Number(r.close);
      const volume = Number(r.volume ?? 0);

      if (
        !time ||
        !Number.isFinite(open) ||
        !Number.isFinite(high) ||
        !Number.isFinite(low) ||
        !Number.isFinite(close)
      ) {
        return null;
      }

      return {
        time,
        open,
        high,
        low,
        close,
        volume,

        label: Number(r.label),

        thresholdLong: toNumberOrNull(r.threshold_long),
        thresholdShort: toNumberOrNull(r.threshold_short),
        expectedVol: toNumberOrNull(r.expected_vol),
        trendStrength: toNumberOrNull(r.trend_strength),

        raw: r,
      };
    })
    .filter(Boolean)
    .sort((a, b) => a.time - b.time);
}

function parseTime(row) {
  // Your prepared data normally has open_time_ms_utc.
  if (row.open_time_ms_utc !== undefined && row.open_time_ms_utc !== null) {
    const t = Number(row.open_time_ms_utc);
    if (Number.isFinite(t)) {
      return Math.floor(t / 1000);
    }
  }

  // Some raw files may use timestamp.
  if (row.timestamp !== undefined && row.timestamp !== null) {
    const t = Number(row.timestamp);
    if (Number.isFinite(t)) {
      return Math.floor(t / 1000);
    }
  }

  // Your data may also have open_time_date_utc.
  if (row.open_time_date_utc) {
    const t = new Date(row.open_time_date_utc).getTime();
    if (Number.isFinite(t)) {
      return Math.floor(t / 1000);
    }
  }

  return null;
}

function toNumberOrNull(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function initChart() {
  const container = document.getElementById("chart-container");
  container.innerHTML = "";

  chart = createChart(container, {
    width: container.clientWidth,
    height: container.clientHeight,

    layout: {
      background: {
        color: "#ffffff",
      },
      textColor: "#333",
    },

    grid: {
      vertLines: {
        color: "#f0f0f0",
      },
      horzLines: {
        color: "#f0f0f0",
      },
    },

    timeScale: {
      timeVisible: true,
      secondsVisible: false,
      borderColor: "#e1e1e1",
      barSpacing: 6,
      minBarSpacing: 0.01,
    },
  });

  candleSeries = chart.addSeries(CandlestickSeries, {
    upColor: "#26a69a",
    downColor: "#ef5350",
    borderVisible: false,
    wickUpColor: "#26a69a",
    wickDownColor: "#ef5350",
  });

  volumeSeries = chart.addSeries(HistogramSeries, {
    priceFormat: {
      type: "volume",
    },
    priceScaleId: "volume-scale",
  });

  chart.priceScale("volume-scale").applyOptions({
    scaleMargins: {
      top: 0.8,
      bottom: 0,
    },
  });

  labelSeries = chart.addSeries(HistogramSeries, {
    title: "Label",
    priceScaleId: "label-scale",
  });

  chart.priceScale("label-scale").applyOptions({
    scaleMargins: {
      top: 0.02,
      bottom: 0.93,
    },
  });

  labelSeries.applyOptions({
    autoscaleInfoProvider: () => ({
      priceRange: {
        minValue: 0,
        maxValue: 1,
      },
    }),
  });

  setupTooltip(container);
  setupRuler(container);
  setupScrollbar();

  window.addEventListener("resize", () => {
    chart.applyOptions({
      width: container.clientWidth,
      height: container.clientHeight,
    });
  });
}

function renderChart(candles) {
  if (!chart) {
    initChart();
  }

  candleData = candles || [];
  candleMap = new Map();

  for (const c of candles) {
    candleMap.set(String(c.time), c);
  }

  candleSeries.setData(
    candles.map((c) => ({
      time: c.time,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }))
  );

  volumeSeries.setData(
    candles.map((c) => ({
      time: c.time,
      value: c.volume,
      color:
        c.close >= c.open
          ? "rgba(38, 166, 154, 0.45)"
          : "rgba(239, 83, 80, 0.45)",
    }))
  );

  labelSeries.setData(
    candles.map((c) => ({
      time: c.time,
      value: 1,
      color: labelColor(c.label),
    }))
  );

  totalBars = candles.length;

  const slider = document.getElementById("time-scrollbar");
  slider.max = totalBars;
  slider.min = 0;
  slider.value = totalBars;

  setInitialVisibleRange(candles.length);
}

function labelColor(label) {
  if (label === 1) return "#26a69a";
  if (label === -1) return "#ef5350";
  if (label === 0) return "#b2b2b2";
  if (label === -2) return "#fbc02d";

  return "#999999";
}

function labelName(label) {
  if (label === 1) return "POSITIVE / Long";
  if (label === -1) return "NEGATIVE / Short";
  if (label === 0) return "NEUTRAL";
  if (label === -2) return "INVALID";

  return "UNKNOWN";
}

function setupTooltip(container) {
  const tooltip = document.createElement("div");
  tooltip.className = "floating-tooltip";
  container.appendChild(tooltip);

  chart.subscribeCrosshairMove((param) => {
    if (
      param.point === undefined ||
      !param.time ||
      param.point.x < 0 ||
      param.point.x > container.clientWidth ||
      param.point.y < 0 ||
      param.point.y > container.clientHeight
    ) {
      tooltip.style.display = "none";
      return;
    }

    const timeKey = normalizeTimeKey(param.time);
    const data = candleMap.get(timeKey);

    if (!data) {
      tooltip.style.display = "none";
      return;
    }

    const change = data.close - data.open;
    const changePct = (change / data.open) * 100;
    const colorClass = change >= 0 ? "win" : "loss";
    const sign = change >= 0 ? "+" : "";

    tooltip.style.display = "block";

    tooltip.innerHTML = `
      <div class="tooltip-title">
        ${new Date(data.time * 1000).toLocaleString()}
      </div>

      <div class="tooltip-row">Open: ${formatPrice(data.open)}</div>
      <div class="tooltip-row">High: ${formatPrice(data.high)}</div>
      <div class="tooltip-row">Low: ${formatPrice(data.low)}</div>
      <div class="tooltip-row">Close: ${formatPrice(data.close)}</div>

      <div class="tooltip-row ${colorClass}">
        Change: ${sign}${changePct.toFixed(2)}%
      </div>

      <hr />

      <div class="tooltip-row">
        Label:
        <span style="color:${labelColor(data.label)};font-weight:bold;">
          ${data.label} / ${labelName(data.label)}
        </span>
      </div>

      <div class="tooltip-row">
        Threshold Long: ${formatMaybePercent(data.thresholdLong)}
      </div>

      <div class="tooltip-row">
        Threshold Short: ${formatMaybePercent(data.thresholdShort)}
      </div>

      <div class="tooltip-row">
        Expected Vol: ${formatMaybePercent(data.expectedVol)}
      </div>

      <div class="tooltip-row">
        Trend Strength: ${formatMaybe(data.trendStrength)}
      </div>
    `;

    const tooltipWidth = 230;
    const tooltipHeight = 240;

    let left = param.point.x + 20;
    let top = param.point.y + 20;

    if (left > container.clientWidth - tooltipWidth) {
      left = param.point.x - tooltipWidth - 20;
    }

    if (top > container.clientHeight - tooltipHeight) {
      top = param.point.y - tooltipHeight - 20;
    }

    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  });
}

function setupRuler(container) {
  rulerRect = document.createElement("div");
  rulerRect.className = "ruler-rect";
  container.appendChild(rulerRect);

  rulerLabel = document.createElement("div");
  rulerLabel.className = "ruler-label";
  container.appendChild(rulerLabel);

  container.addEventListener("mousedown", (e) => {
    if (!e.shiftKey) {
      clearRuler();
      return;
    }

    const rect = container.getBoundingClientRect();

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const logicalIndex = chart.timeScale().coordinateToLogical(x);
    const price = candleSeries.coordinateToPrice(y);

    if (logicalIndex === null || price === null) {
      return;
    }

    isMeasuring = true;

    startPoint = {
      logicalIndex,
      price,
      x,
      y,
    };

    chart.applyOptions({
      handleScroll: {
        pressedMouseMove: false,
      },
    });

    rulerRect.style.display = "block";
    rulerRect.style.left = `${x}px`;
    rulerRect.style.top = `${y}px`;
    rulerRect.style.width = "0px";
    rulerRect.style.height = "0px";

    rulerLabel.style.display = "block";
    rulerLabel.style.left = `${x + 8}px`;
    rulerLabel.style.top = `${y - 40}px`;
    rulerLabel.innerHTML = "";
  });

  container.addEventListener("mousemove", (e) => {
    if (!isMeasuring || !startPoint) {
      return;
    }

    const rect = container.getBoundingClientRect();

    const curX = e.clientX - rect.left;
    const curY = e.clientY - rect.top;

    const curLogicalIndex = chart.timeScale().coordinateToLogical(curX);
    const curPrice = candleSeries.coordinateToPrice(curY);

    if (curLogicalIndex === null || curPrice === null) {
      return;
    }

    const startXAtCurrentScale =
      chart.timeScale().logicalToCoordinate(startPoint.logicalIndex);

    if (startXAtCurrentScale === null) {
      return;
    }

    const rectLeft = Math.min(startXAtCurrentScale, curX);
    const rectTop = Math.min(startPoint.y, curY);
    const rectWidth = Math.abs(curX - startXAtCurrentScale);
    const rectHeight = Math.abs(curY - startPoint.y);

    rulerRect.style.left = `${rectLeft}px`;
    rulerRect.style.top = `${rectTop}px`;
    rulerRect.style.width = `${rectWidth}px`;
    rulerRect.style.height = `${rectHeight}px`;

    const labelX = rectLeft + rectWidth + 6;
    const labelY = Math.max(4, rectTop - 48);

    rulerLabel.style.left = `${labelX}px`;
    rulerLabel.style.top = `${labelY}px`;

    const barCount =
      Math.abs(Math.floor(curLogicalIndex) - Math.floor(startPoint.logicalIndex)) +
      1;

    const priceDiff = curPrice - startPoint.price;
    const percentChange = (priceDiff / startPoint.price) * 100;

    const isUp = priceDiff >= 0;

    rulerRect.style.backgroundColor = isUp
      ? "rgba(38, 166, 154, 0.2)"
      : "rgba(239, 83, 80, 0.2)";

    rulerRect.style.borderColor = isUp ? "#26a69a" : "#ef5350";

    const timeSpanText = getRulerTimeSpanText(
      startPoint.logicalIndex,
      curLogicalIndex
    );

    rulerLabel.innerHTML = `
      <b style="color:${isUp ? "#26a69a" : "#ef5350"}">
        ${isUp ? "▲" : "▼"} ${percentChange.toFixed(2)}%
      </b><br>
      Price Diff: ${priceDiff.toFixed(6)}<br>
      Bars: ${barCount}<br>
      Time: ${timeSpanText}
    `;
  });

  window.addEventListener("mouseup", () => {
    if (!isMeasuring) {
      return;
    }

    isMeasuring = false;

    chart.applyOptions({
      handleScroll: {
        pressedMouseMove: true,
      },
    });
  });

  chart.subscribeClick(() => {
    clearRuler();
  });

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      clearRuler();
    }
  });
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "N/A";

  const totalMinutes = Math.floor(seconds / 60);

  const days = Math.floor(totalMinutes / (24 * 60));
  const hours = Math.floor((totalMinutes % (24 * 60)) / 60);
  const minutes = totalMinutes % 60;

  if (days > 0) {
    return `${days}d ${hours}h`;
  }

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }

  return `${minutes}m`;
}

function clampIndex(idx, min, max) {
  return Math.max(min, Math.min(max, idx));
}

function getRulerTimeSpanText(startLogical, endLogical) {
  if (!candleData || candleData.length === 0) return "N/A";

  let startIdx = Math.floor(startLogical);
  let endIdx = Math.floor(endLogical);

  startIdx = clampIndex(startIdx, 0, candleData.length - 1);
  endIdx = clampIndex(endIdx, 0, candleData.length - 1);

  const leftIdx = Math.min(startIdx, endIdx);
  const rightIdx = Math.max(startIdx, endIdx);

  const startTime = Number(candleData[leftIdx]?.time);
  const endTime = Number(candleData[rightIdx]?.time);

  if (!Number.isFinite(startTime) || !Number.isFinite(endTime)) {
    return "N/A";
  }

  const spanSeconds = Math.abs(endTime - startTime);

  return formatDuration(spanSeconds);
}

function clearRuler() {
  isMeasuring = false;
  startPoint = null;

  if (rulerRect) {
    rulerRect.style.display = "none";
    rulerRect.style.width = "0px";
    rulerRect.style.height = "0px";
  }

  if (rulerLabel) {
    rulerLabel.style.display = "none";
    rulerLabel.innerHTML = "";
  }

  if (chart) {
    chart.applyOptions({
      handleScroll: {
        pressedMouseMove: true,
      },
    });
  }
}

function setupScrollbar() {
  const slider = document.getElementById("time-scrollbar");

  chart.timeScale().subscribeVisibleLogicalRangeChange(() => {
    if (isUpdatingSlider) return;

    requestAnimationFrame(() => {
      if (!slider || totalBars === 0) return;

      const position = chart.timeScale().scrollPosition();
      const newVal = totalBars - position;

      if (Math.abs(Number(slider.value) - newVal) > 0.5) {
        slider.value = newVal;
      }
    });
  });

  slider.oninput = function () {
    if (!chart || totalBars === 0) return;

    isUpdatingSlider = true;

    const distFromRight = totalBars - Number(this.value);
    chart.timeScale().scrollToPosition(-distFromRight, false);

    isUpdatingSlider = false;
  };
}

function formatPrice(v) {
  if (!Number.isFinite(v)) return "-";

  if (Math.abs(v) >= 1000) return v.toFixed(2);
  if (Math.abs(v) >= 1) return v.toFixed(4);

  return v.toFixed(6);
}

function formatMaybe(v) {
  if (v === null || v === undefined || !Number.isFinite(v)) {
    return "-";
  }

  return Number(v).toFixed(6);
}

function formatMaybePercent(v) {
  if (v === null || v === undefined || !Number.isFinite(v)) {
    return "-";
  }

  return `${(Number(v) * 100).toFixed(4)}%`;
}

function setInitialVisibleRange(totalCount) {
  const visibleBars = Math.min(INITIAL_VISIBLE_BARS, totalCount);

  chart.timeScale().setVisibleLogicalRange({
    from: totalCount - visibleBars,
    to: totalCount,
  });
}

function normalizeTimeKey(time) {
  if (typeof time === "number") {
    return String(time);
  }

  if (
    time &&
    typeof time === "object" &&
    time.year &&
    time.month &&
    time.day
  ) {
    const t = Date.UTC(time.year, time.month - 1, time.day) / 1000;
    return String(t);
  }

  return String(time);
}