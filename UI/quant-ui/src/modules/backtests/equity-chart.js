import {
  createChart,
  CrosshairMode,
  LineSeries,
  LineStyle,
  PriceScaleMode,
} from "lightweight-charts";

const EQUITY_COLOR = "#296fa8";
const REGION_COLORS = {
  train: "#2e8b57",
  valid: "#d98c00",
  test: "#2878b5",
  oos: "#c43c39",
};

export class EquityChart {
  constructor(container) {
    this.container = container;
    this.regions = [];
    this.regionFrame = null;
    this.chart = createChart(container, {
      width: container.clientWidth,
      height: container.clientHeight,
      layout: { background: { color: "#ffffff" }, textColor: "#38435a" },
      grid: {
        vertLines: { color: "#edf0f4" },
        horzLines: { color: "#edf0f4" },
      },
      crosshair: { mode: CrosshairMode.Normal },
      leftPriceScale: {
        visible: true,
        autoScale: true,
        borderColor: "#dde2e9",
        scaleMargins: { top: 0.1, bottom: 0.08 },
      },
      rightPriceScale: {
        visible: true,
        autoScale: true,
        borderColor: "#dde2e9",
        scaleMargins: { top: 0.1, bottom: 0.08 },
      },
      timeScale: {
        borderColor: "#dde2e9",
        timeVisible: true,
        secondsVisible: false,
        minBarSpacing: 0.001,
        rightOffset: 1,
      },
    });

    this.marketSeries = this.chart.addSeries(LineSeries, {
      title: "Market Price",
      color: "rgba(20, 25, 32, 0.22)",
      lineWidth: 1,
      priceScaleId: "left",
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
      priceFormat: {
        type: "custom",
        minMove: 0.000001,
        formatter: formatMarketPrice,
      },
    });
    this.absoluteSeries = this.chart.addSeries(LineSeries, {
      title: "Absolute Equity",
      color: EQUITY_COLOR,
      lineWidth: 2,
      lineStyle: LineStyle.Solid,
      priceScaleId: "right",
      priceLineVisible: false,
      lastValueVisible: true,
      priceFormat: {
        type: "custom",
        formatter: formatEquity,
      },
    });
    this.logSeries = this.chart.addSeries(LineSeries, {
      title: "Log Equity",
      color: EQUITY_COLOR,
      lineWidth: 2,
      lineStyle: LineStyle.Dashed,
      priceScaleId: "log-equity-scale",
      priceLineVisible: false,
      lastValueVisible: false,
      priceFormat: {
        type: "custom",
        formatter: formatEquity,
      },
    });
    this.chart.priceScale("log-equity-scale").applyOptions({
      autoScale: true,
      mode: PriceScaleMode.Logarithmic,
      scaleMargins: { top: 0.1, bottom: 0.08 },
    });

    this.regionLayer = document.createElement("div");
    this.regionLayer.className = "equity-region-layer";
    container.appendChild(this.regionLayer);
    this.visibleRangeHandler = () => this.scheduleRegionLayout();
    this.chart.timeScale().subscribeVisibleLogicalRangeChange(this.visibleRangeHandler);
    this.resizeObserver = new ResizeObserver(() => {
      if (!this.chart) return;
      this.chart.applyOptions({
        width: this.container.clientWidth,
        height: this.container.clientHeight,
      });
      this.scheduleRegionLayout();
    });
    this.resizeObserver.observe(container);
  }

  setData(rawCandles, rawEquityPoints, report = {}) {
    const marketPoints = normalizeMarketPoints(rawCandles);
    const equityPoints = normalizeEquityPoints(rawEquityPoints, report);
    this.marketSeries.setData(marketPoints);
    this.absoluteSeries.setData(equityPoints);
    this.logSeries.setData(equityPoints.filter((point) => point.value > 0));
    const allTimes = [...marketPoints, ...equityPoints].map((point) => point.time);
    const timeline = [...new Set(allTimes)].sort((left, right) => left - right);
    const [firstTime, lastTime] = allTimes.reduce(
      ([minimum, maximum], time) => [Math.min(minimum, time), Math.max(maximum, time)],
      [Infinity, -Infinity],
    );
    this.regions = normalizeRegions(report?.time?.regions)
      .filter((region) => (
        !Number.isFinite(firstTime)
        || (region.start <= lastTime && (region.end === null || region.end >= firstTime))
      ))
      .map((region) => alignRegionToTimeline(region, timeline))
      .filter(Boolean);
    this.renderRegionElements();
    if (marketPoints.length || equityPoints.length) {
      this.chart.timeScale().fitContent();
      this.scheduleRegionLayout();
    }
    return equityPoints.length;
  }

  renderRegionElements() {
    this.regionLayer.innerHTML = "";
    this.regionLines = this.regions.slice(1).map((region) => {
      const line = document.createElement("div");
      line.className = "equity-region-line";
      line.style.borderColor = region.color;
      this.regionLayer.appendChild(line);
      return { region, element: line };
    });
    this.regionLabels = this.regions.map((region) => {
      const label = document.createElement("span");
      label.className = "equity-region-label";
      label.style.color = region.color;
      label.textContent = region.name.toUpperCase();
      this.regionLayer.appendChild(label);
      return { region, element: label };
    });
  }

  scheduleRegionLayout() {
    if (this.regionFrame !== null) return;
    this.regionFrame = requestAnimationFrame(() => {
      this.regionFrame = null;
      this.layoutRegions();
    });
  }

  layoutRegions() {
    if (!this.chart || !this.regionLayer || !this.regions.length) return;
    const timeScale = this.chart.timeScale();
    const leftScaleWidth = this.chart.priceScale("left").width();
    const rightScaleWidth = this.chart.priceScale("right").width();
    const width = timeScale.width();
    // Time-scale coordinates are relative to the chart pane, while the HTML
    // overlay is positioned inside the full chart container. Exclude both
    // price scales so the two coordinate systems have the same origin.
    this.regionLayer.style.left = `${leftScaleWidth}px`;
    this.regionLayer.style.right = `${rightScaleWidth}px`;
    const coordinate = (time) => timeScale.timeToCoordinate(time);
    const visibleRange = timeScale.getVisibleRange();
    const visibleStart = normalizeTimestamp(visibleRange?.from);
    const visibleEnd = normalizeTimestamp(visibleRange?.to);

    this.regionLines.forEach(({ region, element }) => {
      const x = coordinate(region.start);
      const visible = Number.isFinite(x)
        && (!Number.isFinite(visibleStart) || region.start >= visibleStart)
        && (!Number.isFinite(visibleEnd) || region.start <= visibleEnd)
        && x >= 0
        && x <= width;
      element.style.display = visible ? "block" : "none";
      if (visible) element.style.left = `${x}px`;
    });

    this.regionLabels.forEach(({ region, element }, index) => {
      const nextRegion = this.regions[index + 1];
      const regionEnd = region.end ?? nextRegion?.start ?? visibleEnd;
      const segmentStart = Number.isFinite(visibleStart)
        ? Math.max(region.start, visibleStart)
        : region.start;
      const segmentEnd = Number.isFinite(visibleEnd) && Number.isFinite(regionEnd)
        ? Math.min(regionEnd, visibleEnd)
        : regionEnd;
      const rawStart = coordinate(segmentStart);
      const rawEnd = coordinate(segmentEnd);
      const start = Number.isFinite(rawStart) ? Math.max(0, rawStart) : 0;
      const end = Number.isFinite(rawEnd) ? Math.min(width, rawEnd) : width;
      const visible = Number.isFinite(segmentEnd) && segmentEnd > segmentStart && end - start >= 48;
      element.style.display = visible ? "block" : "none";
      if (visible) element.style.left = `${start + (end - start) / 2}px`;
    });
  }

  destroy() {
    this.resizeObserver?.disconnect();
    if (this.regionFrame !== null) {
      cancelAnimationFrame(this.regionFrame);
      this.regionFrame = null;
    }
    if (this.chart) {
      this.chart.timeScale().unsubscribeVisibleLogicalRangeChange(this.visibleRangeHandler);
      this.chart.remove();
      this.chart = null;
    }
    this.container.innerHTML = "";
  }
}

function normalizeMarketPoints(rawCandles) {
  const byTime = new Map();
  if (!Array.isArray(rawCandles)) return [];
  rawCandles.forEach((point) => {
    const time = normalizeTimestamp(point?.time ?? point?.open_time_ms_utc ?? point?.open_time_date_utc);
    const value = Number(point?.close);
    if (Number.isFinite(time) && Number.isFinite(value)) {
      byTime.set(time, { time, value });
    }
  });
  return [...byTime.values()].sort((left, right) => left.time - right.time);
}

function normalizeEquityPoints(rawPoints, report) {
  const byTime = new Map();
  if (Array.isArray(rawPoints)) {
    rawPoints.forEach((point) => {
      const time = normalizeTimestamp(point?.time ?? point?.date);
      const value = Number(point?.value ?? point?.end_equity);
      if (Number.isFinite(time) && Number.isFinite(value)) {
        byTime.set(time, { time, value });
      }
    });
  }

  const points = [...byTime.values()].sort((left, right) => left.time - right.time);
  const rawStartValue = report?.performance?.start_value;
  const startValue = Number(rawStartValue);
  if (points.length && rawStartValue !== null && rawStartValue !== undefined && Number.isFinite(startValue)) {
    const reportStart = normalizeTimestamp(report?.time?.start);
    const baselineTime = Number.isFinite(reportStart)
      ? Math.min(reportStart, points[0].time - 1)
      : points[0].time - 1;
    byTime.set(baselineTime, { time: baselineTime, value: startValue });
  }
  return [...byTime.values()].sort((left, right) => left.time - right.time);
}

function normalizeRegions(rawRegions) {
  if (!rawRegions || typeof rawRegions !== "object") return [];
  return ["train", "valid", "test", "oos"].map((name) => {
    const region = rawRegions[name];
    if (!region || typeof region !== "object") return null;
    const start = normalizeTimestamp(region.start);
    const end = normalizeTimestamp(region.end);
    if (!Number.isFinite(start)) return null;
    return {
      name,
      start,
      end: Number.isFinite(end) ? end : null,
      color: REGION_COLORS[name],
    };
  }).filter(Boolean);
}

function alignRegionToTimeline(region, timeline) {
  if (!timeline.length) return region;

  const startIndex = lowerBound(timeline, region.start);
  if (startIndex >= timeline.length) return null;
  const start = timeline[startIndex];

  let end = null;
  if (region.end !== null) {
    const endIndex = upperBound(timeline, region.end) - 1;
    if (endIndex < 0 || timeline[endIndex] < start) return null;
    end = timeline[endIndex];
  }
  return { ...region, start, end };
}

function lowerBound(values, target) {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (values[middle] < target) low = middle + 1;
    else high = middle;
  }
  return low;
}

function upperBound(values, target) {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (values[middle] <= target) low = middle + 1;
    else high = middle;
  }
  return low;
}

function normalizeTimestamp(value) {
  if (value === null || value === undefined || value === "") return NaN;
  const numeric = Number(value);
  if (Number.isFinite(numeric)) {
    return numeric > 10_000_000_000 ? Math.floor(numeric / 1000) : Math.floor(numeric);
  }
  const timestamp = new Date(value).getTime();
  return Number.isFinite(timestamp) ? Math.floor(timestamp / 1000) : NaN;
}

function formatMarketPrice(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  if (Math.abs(number) >= 1000) return number.toLocaleString(undefined, { maximumFractionDigits: 2 });
  if (Math.abs(number) >= 1) return number.toLocaleString(undefined, { maximumFractionDigits: 4 });
  return number.toLocaleString(undefined, { maximumFractionDigits: 6 });
}

function formatEquity(value) {
  const number = Number(value);
  return Number.isFinite(number)
    ? number.toLocaleString(undefined, { maximumFractionDigits: 2 })
    : "—";
}
