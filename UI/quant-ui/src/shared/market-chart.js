import {
  CandlestickSeries,
  createChart,
  createSeriesMarkers,
  CrosshairMode,
  HistogramSeries,
} from "lightweight-charts";

import { escapeHtml, formatDuration, formatPercent, formatPrice } from "./formatters.js";

const SIGNAL_COLORS = {
  "-2": "#d5a126",
  "-1": "#df5962",
  "0": "#aeb5c2",
  "1": "#24987d",
};

export class MarketChart {
  constructor(container, options = {}) {
    this.container = container;
    this.slider = options.slider || null;
    this.showLabels = options.showLabels !== false;
    this.showPredictions = Boolean(options.showPredictions);
    this.tooltipMode = options.tooltipMode || "labels";
    this.abortController = new AbortController();
    this.candles = [];
    this.candlesByTime = new Map();
    this.tradesByTime = new Map();
    this.totalBars = 0;
    this.sliderUpdating = false;
    this.sliderFrame = null;
    this.startPoint = null;
    this.measuring = false;

    this.initialize();
  }

  initialize() {
    const { signal } = this.abortController;
    this.container.innerHTML = "";
    this.chart = createChart(this.container, {
      width: this.container.clientWidth,
      height: this.container.clientHeight,
      layout: { background: { color: "#ffffff" }, textColor: "#38435a" },
      grid: { vertLines: { color: "#f0f2f6" }, horzLines: { color: "#f0f2f6" } },
      crosshair: { mode: CrosshairMode.Normal },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        borderColor: "#e0e4ec",
        barSpacing: 6,
        minBarSpacing: 0.01,
      },
    });

    this.candleSeries = this.chart.addSeries(CandlestickSeries, {
      upColor: "#24987d",
      downColor: "#df5962",
      borderVisible: false,
      wickUpColor: "#24987d",
      wickDownColor: "#df5962",
    });
    this.markers = createSeriesMarkers(this.candleSeries);

    this.volumeSeries = this.chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "volume-scale",
    });
    this.chart.priceScale("volume-scale").applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });

    if (this.showLabels) {
      this.labelSeries = this.createSignalSeries("Label", "label-scale", 0.02, 0.93);
    }
    if (this.showPredictions) {
      this.predictionSeries = this.createSignalSeries("Prediction", "prediction-scale", 0.08, 0.87);
    }
    this.tooltip = document.createElement("div");
    this.tooltip.className = "chart-tooltip";
    this.container.appendChild(this.tooltip);
    this.rulerRect = document.createElement("div");
    this.rulerRect.className = "ruler-rect";
    this.container.appendChild(this.rulerRect);
    this.rulerLabel = document.createElement("div");
    this.rulerLabel.className = "ruler-label";
    this.container.appendChild(this.rulerLabel);

    this.crosshairHandler = (param) => this.handleCrosshair(param);
    this.visibleRangeHandler = () => this.syncSlider();
    this.clickHandler = () => this.clearRuler();
    this.chart.subscribeCrosshairMove(this.crosshairHandler);
    this.chart.timeScale().subscribeVisibleLogicalRangeChange(this.visibleRangeHandler);
    this.chart.subscribeClick(this.clickHandler);

    this.container.addEventListener("mousedown", (event) => this.startRuler(event), { signal });
    this.container.addEventListener("mousemove", (event) => this.moveRuler(event), { signal });
    window.addEventListener("mouseup", () => this.stopRuler(), { signal });
    window.addEventListener("keydown", (event) => {
      if (event.key === "Escape") this.clearRuler();
    }, { signal });

    if (this.slider) {
      this.slider.addEventListener("input", () => {
        if (!this.totalBars) return;
        const range = this.chart.timeScale().getVisibleLogicalRange();
        if (!range) return;
        const visibleBars = Math.max(1, range.to - range.from);
        const maximumStart = Math.max(0, this.totalBars - visibleBars);
        const start = clamp(Number(this.slider.value), 0, maximumStart);
        this.sliderUpdating = true;
        this.chart.timeScale().setVisibleLogicalRange({
          from: start,
          to: start + visibleBars,
        });
        this.sliderUpdating = false;
      }, { signal });
    }

    this.resizeObserver = new ResizeObserver(() => {
      this.chart.applyOptions({
        width: this.container.clientWidth,
        height: this.container.clientHeight,
      });
    });
    this.resizeObserver.observe(this.container);
  }

  createSignalSeries(title, scaleId, top, bottom) {
    const series = this.chart.addSeries(HistogramSeries, { title, priceScaleId: scaleId });
    this.chart.priceScale(scaleId).applyOptions({ scaleMargins: { top, bottom } });
    series.applyOptions({
      autoscaleInfoProvider: () => ({ priceRange: { minValue: 0, maxValue: 1 } }),
    });
    return series;
  }

  setData(rawCandles, trades = []) {
    this.candles = normalizeCandles(rawCandles);
    this.candlesByTime = new Map(this.candles.map((candle) => [String(candle.time), candle]));
    this.tradesByTime = groupTrades(trades);
    this.totalBars = this.candles.length;

    this.candleSeries.setData(this.candles.map(({ time, open, high, low, close }) => ({ time, open, high, low, close })));
    this.volumeSeries.setData(this.candles.map((candle) => ({
      time: candle.time,
      value: candle.volume,
      color: candle.close >= candle.open ? "rgba(36, 152, 125, 0.45)" : "rgba(223, 89, 98, 0.45)",
    })));
    if (this.labelSeries) this.labelSeries.setData(signalData(this.candles, "label"));
    if (this.predictionSeries) this.predictionSeries.setData(signalData(this.candles, "pred"));
    const validTimes = new Set(this.candles.map((candle) => Number(candle.time)));
    this.markers.setMarkers(trades
      .filter((trade) => validTimes.has(Number(trade.dt ?? trade.time)))
      .map((trade) => {
        const isBuy = parseBoolean(trade.is_buy);
        return {
          time: Number(trade.dt ?? trade.time),
          position: isBuy ? "belowBar" : "aboveBar",
          color: isBuy ? "#24987d" : "#df5962",
          shape: isBuy ? "arrowUp" : "arrowDown",
          text: `${isBuy ? "BUY" : "SELL"} ${trade.role || ""}`.trim(),
        };
      })
      .sort((left, right) => Number(left.time) - Number(right.time)));

    if (this.slider) {
      this.slider.min = 0;
      this.slider.max = "0";
      this.slider.value = "0";
      this.slider.disabled = true;
    }
    const visibleBars = Math.min(320, this.totalBars);
    if (visibleBars) {
      this.chart.timeScale().setVisibleLogicalRange({
        from: this.totalBars - visibleBars,
        to: this.totalBars,
      });
      this.syncSlider();
    }
  }

  handleCrosshair(param) {
    if (!param.point || !param.time || param.point.x < 0 || param.point.y < 0
        || param.point.x > this.container.clientWidth || param.point.y > this.container.clientHeight) {
      this.tooltip.style.display = "none";
      return;
    }
    const candle = this.candlesByTime.get(normalizeTimeKey(param.time));
    if (!candle) {
      this.tooltip.style.display = "none";
      return;
    }

    const change = candle.close - candle.open;
    const changePercent = candle.open ? change / candle.open * 100 : 0;
    const trades = this.tradesByTime.get(String(candle.time)) || [];
    this.tooltip.innerHTML = this.tooltipHtml(candle, change, changePercent, trades);
    this.tooltip.style.display = "block";
    const width = 250;
    const height = Math.min(350, this.tooltip.scrollHeight || 240);
    let left = param.point.x + 18;
    let top = param.point.y + 18;
    if (left > this.container.clientWidth - width) left = param.point.x - width - 18;
    if (top > this.container.clientHeight - height) top = param.point.y - height - 18;
    this.tooltip.style.left = `${Math.max(4, left)}px`;
    this.tooltip.style.top = `${Math.max(4, top)}px`;
  }

  tooltipHtml(candle, change, changePercent, trades) {
    const sign = change >= 0 ? "+" : "";
    const signalRows = this.tooltipMode === "labels" ? `
      <div class="tooltip-divider"></div>
      <div class="tooltip-row">Label: <strong style="color:${signalColor(candle.label)}">${signalName(candle.label)}</strong></div>
      <div class="tooltip-row">Long threshold: ${formatPercent(candle.threshold_long, { decimals: 4 })}</div>
      <div class="tooltip-row">Short threshold: ${formatPercent(candle.threshold_short, { decimals: 4 })}</div>
      <div class="tooltip-row">Expected volatility: ${formatPercent(candle.expected_vol, { decimals: 4 })}</div>
      <div class="tooltip-row">Trend strength: ${formatPrice(candle.trend_strength)}</div>
    ` : `
      <div class="tooltip-divider"></div>
      <div class="tooltip-row">Label: ${signalName(candle.label)}</div>
      <div class="tooltip-row">Prediction: ${signalName(candle.pred)}</div>
      ${trades.map((trade) => `
        <div class="tooltip-divider"></div>
        <div class="tooltip-row ${parseBoolean(trade.is_buy) ? "positive" : "negative"}"><strong>${parseBoolean(trade.is_buy) ? "BUY" : "SELL"} ${escapeHtml(trade.role || "")}</strong></div>
        <div class="tooltip-row">Price: ${formatPrice(trade.price)}</div>
        <div class="tooltip-row">Size: ${formatPrice(trade.size, 4)}</div>
        <div class="tooltip-row">Stop loss: ${formatPrice(trade.sl_price)}</div>
        <div class="tooltip-row">Take profit: ${formatPrice(trade.tp_price)}</div>
      `).join("")}
    `;

    return `
      <div class="tooltip-title">${new Date(candle.time * 1000).toLocaleString()}</div>
      <div class="tooltip-row">Open: ${formatPrice(candle.open)}</div>
      <div class="tooltip-row">High: ${formatPrice(candle.high)}</div>
      <div class="tooltip-row">Low: ${formatPrice(candle.low)}</div>
      <div class="tooltip-row">Close: ${formatPrice(candle.close)}</div>
      <div class="tooltip-row ${change >= 0 ? "positive" : "negative"}">Change: ${sign}${changePercent.toFixed(2)}%</div>
      ${signalRows}
    `;
  }

  syncSlider() {
    if (this.sliderUpdating || !this.slider || !this.totalBars) return;
    if (this.sliderFrame !== null) return;
    this.sliderFrame = requestAnimationFrame(() => {
      this.sliderFrame = null;
      if (!this.chart || !this.slider) return;
      const range = this.chart.timeScale().getVisibleLogicalRange();
      if (!range) return;
      const visibleBars = Math.max(1, range.to - range.from);
      const maximumStart = Math.max(0, this.totalBars - visibleBars);
      const nextValue = clamp(range.from, 0, maximumStart);
      this.slider.max = String(maximumStart);
      this.slider.disabled = maximumStart <= 0;
      if (Math.abs(Number(this.slider.value) - nextValue) > 0.01) {
        this.slider.value = String(nextValue);
      }
    });
  }

  startRuler(event) {
    if (!event.shiftKey) {
      this.clearRuler();
      return;
    }
    const bounds = this.container.getBoundingClientRect();
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    const logical = this.chart.timeScale().coordinateToLogical(x);
    const price = this.candleSeries.coordinateToPrice(y);
    if (logical === null || price === null) return;
    this.startPoint = { logical, price, y };
    this.measuring = true;
    this.chart.applyOptions({ handleScroll: { pressedMouseMove: false } });
    this.rulerRect.style.display = "block";
    this.rulerLabel.style.display = "block";
  }

  moveRuler(event) {
    if (!this.measuring || !this.startPoint) return;
    const bounds = this.container.getBoundingClientRect();
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    const logical = this.chart.timeScale().coordinateToLogical(x);
    const price = this.candleSeries.coordinateToPrice(y);
    const startX = this.chart.timeScale().logicalToCoordinate(this.startPoint.logical);
    if (logical === null || price === null || startX === null) return;

    const left = Math.min(startX, x);
    const top = Math.min(this.startPoint.y, y);
    const width = Math.abs(x - startX);
    const height = Math.abs(y - this.startPoint.y);
    const difference = price - this.startPoint.price;
    const percent = this.startPoint.price ? difference / this.startPoint.price * 100 : 0;
    const positive = difference >= 0;
    this.rulerRect.style.left = `${left}px`;
    this.rulerRect.style.top = `${top}px`;
    this.rulerRect.style.width = `${width}px`;
    this.rulerRect.style.height = `${height}px`;
    this.rulerRect.style.borderColor = positive ? "#24987d" : "#df5962";
    this.rulerRect.style.background = positive ? "rgba(36, 152, 125, 0.18)" : "rgba(223, 89, 98, 0.18)";
    this.rulerLabel.style.left = `${Math.min(this.container.clientWidth - 145, left + width + 5)}px`;
    this.rulerLabel.style.top = `${Math.max(4, top - 47)}px`;
    const startIndex = clamp(Math.floor(this.startPoint.logical), 0, this.candles.length - 1);
    const endIndex = clamp(Math.floor(logical), 0, this.candles.length - 1);
    const seconds = Math.abs(Number(this.candles[endIndex]?.time) - Number(this.candles[startIndex]?.time));
    this.rulerLabel.innerHTML = `
      <strong style="color:${positive ? "#57d4b5" : "#ff8c94"}">${positive ? "+" : ""}${percent.toFixed(2)}%</strong><br>
      Bars: ${Math.abs(endIndex - startIndex) + 1}<br>
      Time: ${formatDuration(seconds)}
    `;
  }

  stopRuler() {
    if (!this.measuring) return;
    this.measuring = false;
    this.chart.applyOptions({ handleScroll: { pressedMouseMove: true } });
  }

  clearRuler() {
    this.measuring = false;
    this.startPoint = null;
    if (this.rulerRect) this.rulerRect.style.display = "none";
    if (this.rulerLabel) this.rulerLabel.style.display = "none";
    if (this.chart) this.chart.applyOptions({ handleScroll: { pressedMouseMove: true } });
  }

  destroy() {
    this.abortController.abort();
    this.resizeObserver?.disconnect();
    if (this.sliderFrame !== null) {
      cancelAnimationFrame(this.sliderFrame);
      this.sliderFrame = null;
    }
    if (this.chart) {
      this.chart.unsubscribeCrosshairMove(this.crosshairHandler);
      this.chart.timeScale().unsubscribeVisibleLogicalRangeChange(this.visibleRangeHandler);
      this.chart.unsubscribeClick(this.clickHandler);
      this.chart.remove();
      this.chart = null;
    }
    this.container.innerHTML = "";
  }
}

function normalizeCandles(candles) {
  if (!Array.isArray(candles)) return [];
  return candles.map((raw) => ({
    ...raw,
    time: normalizeTimestamp(raw.time ?? raw.open_time_ms_utc ?? raw.open_time_date_utc),
    open: Number(raw.open),
    high: Number(raw.high),
    low: Number(raw.low),
    close: Number(raw.close),
    volume: Number(raw.volume || 0),
  })).filter((candle) => Number.isFinite(candle.time)
      && Number.isFinite(candle.open) && Number.isFinite(candle.high)
      && Number.isFinite(candle.low) && Number.isFinite(candle.close))
    .sort((left, right) => left.time - right.time);
}

function normalizeTimestamp(value) {
  if (typeof value === "number") return value > 10_000_000_000 ? Math.floor(value / 1000) : Math.floor(value);
  const numeric = Number(value);
  if (value !== "" && Number.isFinite(numeric)) return numeric > 10_000_000_000 ? Math.floor(numeric / 1000) : Math.floor(numeric);
  const timestamp = new Date(value).getTime();
  return Number.isFinite(timestamp) ? Math.floor(timestamp / 1000) : NaN;
}

function normalizeTimeKey(time) {
  if (typeof time === "number") return String(time);
  if (time && typeof time === "object" && time.year && time.month && time.day) {
    return String(Date.UTC(time.year, time.month - 1, time.day) / 1000);
  }
  return String(time);
}

function signalData(candles, key) {
  return candles.filter((candle) => Number.isFinite(Number(candle[key])))
    .map((candle) => ({ time: candle.time, value: 1, color: signalColor(candle[key]) }));
}

function signalColor(value) {
  return SIGNAL_COLORS[String(Math.round(Number(value)))] || "#aeb5c2";
}

function signalName(value) {
  const number = Math.round(Number(value));
  if (number === 1) return "Positive";
  if (number === 0) return "Neutral";
  if (number === -1) return "Negative";
  if (number === -2) return "Invalid";
  return "—";
}

function groupTrades(trades) {
  const result = new Map();
  if (!Array.isArray(trades)) return result;
  for (const trade of trades) {
    const key = String(Number(trade.dt ?? trade.time));
    if (!result.has(key)) result.set(key, []);
    result.get(key).push(trade);
  }
  return result;
}

function parseBoolean(value) {
  if (typeof value === "string") return value.toLowerCase() === "true";
  return Boolean(value);
}

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}
