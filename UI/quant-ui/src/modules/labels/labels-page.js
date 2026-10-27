import "./labels.css";

import { MarketChart } from "../../shared/market-chart.js";
import { escapeHtml, formatValue } from "../../shared/formatters.js";
import { createToast } from "../../shared/toast.js";
import { generateLabels, loadCurrentLabels, loadLabelSchema } from "./labels-api.js";

const FALLBACK_FIELDS = [
  field("market_category", "Market category", "string", "Cryptocurrency", "Market Data", ["Cryptocurrency", "Forex", "Stock"]),
  field("data_source", "Data source", "string", "binance_public_data", "Market Data"),
  field("symbol", "Symbol", "string", "DOGEUSDT", "Market Data"),
  field("interval", "Interval", "string", "30m", "Market Data", ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]),
  field("trading_type", "Trading type", "string", "um", "Market Data", ["spot", "um", "cm"]),
  field("label_type", "Label type", "string", "FTHL", "Label Definition", ["FTHL", "TBM", "TBM_TREND", "BBM"]),
  field("vol_ewma_span", "Volatility EWMA span", "integer", 20, "Label Definition"),
  field("predict_num", "Prediction bars", "integer", 32, "Label Definition"),
  field("vol_multiplier_long", "Long volatility multiplier", "number", 1.7, "Thresholds"),
  field("stop_multiplier_rate_long", "Long stop multiplier rate", "number", null, "Thresholds", null, true),
  field("vol_multiplier_short", "Short volatility multiplier", "number", 1.7, "Thresholds"),
  field("stop_multiplier_rate_short", "Short stop multiplier rate", "number", null, "Thresholds", null, true),
  field("tbm_take_profit_price", "TBM take-profit price", "string", null, "Thresholds", ["close", "high_low"], true),
  field("min_expected_move_pct", "Minimum expected move", "number", 0.01, "Thresholds"),
  field("version", "Version", "number", 0.1, "Metadata"),
];

export async function mountLabelsPage(container) {
  const abortController = new AbortController();
  const toast = createToast();
  let chart = null;
  let schema = FALLBACK_FIELDS;
  let currentPayload = null;

  container.innerHTML = `
    <section class="module-page labels-page">
      <header class="page-header labels-header">
        <div>
          <p class="page-eyebrow">Data preparation</p>
          <h1>Labels</h1>
          <p class="page-subtitle">Configure BaseDefine, generate labels, and inspect the prepared market data.</p>
        </div>
        <div class="button-row">
          <span data-role="status" class="status-pill busy">Loading</span>
          <button data-action="reload" class="secondary-button" type="button">Reload output</button>
          <button data-action="generate" class="primary-button" type="button">Generate</button>
        </div>
      </header>
      <div class="labels-workspace">
        <aside class="labels-config panel">
          <div class="panel-header">
            <h2>BaseDefine configuration</h2>
            <span class="panel-note">Blank optional fields become null</span>
          </div>
          <form data-role="config-form" class="config-form"></form>
        </aside>
        <main class="labels-results">
          <div data-role="summary" class="summary-grid label-summary"></div>
          <section class="panel label-chart-panel">
            <div class="panel-header">
              <div>
                <h2>Generated labels</h2>
                <span data-role="data-meta" class="panel-note">No prepared output loaded</span>
              </div>
              <div class="label-legend" aria-label="Label legend">
                <span class="legend-negative">Negative -1</span>
                <span class="legend-neutral">Neutral 0</span>
                <span class="legend-positive">Positive 1</span>
                <span class="legend-invalid">Invalid -2</span>
              </div>
            </div>
            <div data-role="chart-state" class="page-state"><span class="spinner"></span>Loading prepared data...</div>
            <div data-role="chart" class="market-chart label-market-chart hidden"></div>
            <div data-role="scrollbar" class="chart-scrollbar hidden">
              <input type="range" min="0" step="1" value="0" aria-label="Visible label chart position" />
            </div>
          </section>
        </main>
      </div>
    </section>
  `;

  const query = (selector) => container.querySelector(selector);
  const form = query('[data-role="config-form"]');
  const generateButton = query('[data-action="generate"]');
  const reloadButton = query('[data-action="reload"]');

  function setStatus(text, state = "") {
    const status = query('[data-role="status"]');
    status.textContent = text;
    status.className = `status-pill${state ? ` ${state}` : ""}`;
  }

  function renderForm(config = {}) {
    form.innerHTML = "";
    const groups = new Map();
    schema.forEach((item) => {
      const group = item.group || "Configuration";
      if (!groups.has(group)) groups.set(group, []);
      groups.get(group).push(item);
    });

    groups.forEach((fields, groupName) => {
      const section = document.createElement("fieldset");
      section.className = "config-section";
      section.innerHTML = `<legend>${escapeHtml(groupName)}</legend>`;
      const grid = document.createElement("div");
      grid.className = "config-grid";
      fields.forEach((item) => grid.appendChild(createFieldControl(item, config[item.name])));
      section.appendChild(grid);
      form.appendChild(section);
    });
    syncConditionalRequirements();
  }

  function syncConditionalRequirements() {
    const labelType = form.elements.namedItem("label_type")?.value;
    const takeProfitPrice = form.elements.namedItem("tbm_take_profit_price");
    if (takeProfitPrice) {
      takeProfitPrice.required = ["TBM", "TBM_TREND"].includes(labelType);
    }
  }

  function readConfig() {
    const result = {};
    schema.forEach((item) => {
      const control = form.elements.namedItem(item.name);
      if (!control) return;
      if (item.type === "boolean") {
        result[item.name] = Boolean(control.checked);
        return;
      }
      const value = String(control.value).trim();
      if (!value && item.nullable) {
        result[item.name] = null;
      } else if (item.type === "number") {
        result[item.name] = Number(value);
      } else if (item.type === "integer") {
        result[item.name] = Number.parseInt(value, 10);
      } else {
        result[item.name] = value;
      }
    });
    return result;
  }

  function renderPayload(payload, updateConfig = true) {
    currentPayload = normalizePayload(payload);
    if (updateConfig) renderForm(currentPayload.config || {});
    renderSummary(query('[data-role="summary"]'), currentPayload);

    const candles = currentPayload.candles;
    const chartElement = query('[data-role="chart"]');
    const chartState = query('[data-role="chart-state"]');
    const scrollbar = query('[data-role="scrollbar"]');
    const slider = scrollbar.querySelector("input");
    chart?.destroy();
    chart = null;

    if (!candles.length) {
      chartElement.classList.add("hidden");
      scrollbar.classList.add("hidden");
      chartState.className = "page-state";
      chartState.innerHTML = "<strong>No prepared candles found</strong><span>Generate labels to create output/data results.</span>";
    } else {
      chartState.classList.add("hidden");
      chartElement.classList.remove("hidden");
      scrollbar.classList.remove("hidden");
      chart = new MarketChart(chartElement, {
        slider,
        tooltipMode: "labels",
        showLabels: true,
      });
      chart.setData(candles);
    }

    const manifest = currentPayload.manifest || {};
    const summary = currentPayload.summary || {};
    const time = summary.start && summary.end ? summary : (manifest.time || currentPayload.time || {});
    query('[data-role="data-meta"]').textContent = [
      summary.split ? `Split: ${summary.split}` : null,
      time.start && time.end ? `${formatDate(time.start)} to ${formatDate(time.end)}` : null,
      candles.length ? `${candles.length.toLocaleString()} candles` : null,
      manifest.data_id ? `Data ID ${String(manifest.data_id).slice(0, 12)}` : null,
    ].filter(Boolean).join(" · ") || (
      currentPayload.available ? "Prepared output loaded" : "No prepared output"
    );
    setStatus(
      currentPayload.available ? "Ready" : "No output",
      currentPayload.available ? "success" : "",
    );
  }

  async function loadInitialData() {
    setStatus("Loading", "busy");
    const [schemaResult, currentResult] = await Promise.allSettled([
      loadLabelSchema(abortController.signal),
      loadCurrentLabels(abortController.signal),
    ]);
    if (abortController.signal.aborted) return;
    if (schemaResult.status === "fulfilled") schema = normalizeSchema(schemaResult.value);
    else toast.show(`Using built-in BaseDefine schema: ${schemaResult.reason.message}`, true);
    if (currentResult.status === "fulfilled") {
      renderPayload(currentResult.value, true);
    } else {
      renderForm(defaultConfig(schema));
      renderSummary(query('[data-role="summary"]'), { config: {}, candles: [] });
      query('[data-role="chart-state"]').innerHTML = `<strong>No prepared output available</strong><span>${escapeHtml(currentResult.reason.message)}</span>`;
      setStatus("No output", "error");
    }
  }

  async function runGeneration() {
    if (!form.reportValidity()) return;
    generateButton.disabled = true;
    reloadButton.disabled = true;
    setStatus("Generating", "busy");
    try {
      const payload = await generateLabels(readConfig(), abortController.signal);
      if (abortController.signal.aborted) return;
      renderPayload(payload, true);
      toast.show("Labels generated successfully");
    } catch (error) {
      if (error.name === "AbortError") return;
      setStatus("Generation failed", "error");
      toast.show(error.message || String(error), true);
    } finally {
      if (!abortController.signal.aborted) {
        generateButton.disabled = false;
        reloadButton.disabled = false;
      }
    }
  }

  generateButton.addEventListener("click", runGeneration, { signal: abortController.signal });
  form.addEventListener("change", (event) => {
    if (event.target?.name === "label_type") syncConditionalRequirements();
  }, { signal: abortController.signal });
  reloadButton.addEventListener("click", async () => {
    reloadButton.disabled = true;
    setStatus("Loading", "busy");
    try {
      renderPayload(await loadCurrentLabels(abortController.signal), true);
      toast.show("Prepared output reloaded");
    } catch (error) {
      if (error.name !== "AbortError") {
        setStatus("Load failed", "error");
        toast.show(error.message || String(error), true);
      }
    } finally {
      if (!abortController.signal.aborted) reloadButton.disabled = false;
    }
  }, { signal: abortController.signal });

  loadInitialData().catch((error) => {
    if (error.name !== "AbortError") {
      setStatus("Load failed", "error");
      toast.show(error.message || String(error), true);
    }
  });

  return () => {
    abortController.abort();
    chart?.destroy();
    toast.destroy();
    container.innerHTML = "";
  };
}

function field(name, label, type, defaultValue, group, options = null, nullable = false) {
  return { name, label, type, default: defaultValue, group, options, nullable };
}

function normalizeSchema(payload) {
  let rawFields = Array.isArray(payload) ? payload : payload?.fields;
  if (!Array.isArray(rawFields) && payload?.config_schema?.properties) {
    const required = new Set(payload.config_schema.required || []);
    rawFields = Object.entries(payload.config_schema.properties).map(([name, property]) => {
      const variants = Array.isArray(property.anyOf) ? property.anyOf : [];
      const valueSchema = variants.find((variant) => variant.type !== "null") || property;
      const nullable = variants.some((variant) => variant.type === "null");
      return {
        name,
        label: property.title,
        description: property.description,
        type: valueSchema.type,
        options: valueSchema.enum || property.enum || null,
        nullable,
        required: required.has(name),
        default: payload.defaults?.[name] ?? property.default ?? null,
        minimum: valueSchema.minimum,
        exclusiveMinimum: valueSchema.exclusiveMinimum,
        maximum: valueSchema.maximum,
        exclusiveMaximum: valueSchema.exclusiveMaximum,
        pattern: valueSchema.pattern,
        minLength: valueSchema.minLength,
        maxLength: valueSchema.maxLength,
      };
    });
  }
  if (!Array.isArray(rawFields) || !rawFields.length) return FALLBACK_FIELDS;
  return rawFields.map((raw) => {
    const name = raw.name || raw.key || raw.field;
    const fallback = FALLBACK_FIELDS.find((item) => item.name === name) || {};
    return {
      ...fallback,
      ...raw,
      name,
      label: raw.label || fallback.label || humanize(name),
      type: normalizeType(raw.type || fallback.type),
      group: raw.group || raw.category || fallback.group || "Configuration",
      options: raw.options || raw.choices || raw.enum || fallback.options || null,
      nullable: Boolean(raw.nullable ?? raw.optional ?? fallback.nullable),
      default: payload?.defaults?.[name] ?? raw.default ?? fallback.default ?? null,
    };
  }).filter((item) => item.name);
}

function normalizeType(type) {
  const value = String(type || "string").toLowerCase();
  if (["int", "integer"].includes(value)) return "integer";
  if (["float", "number"].includes(value)) return "number";
  if (["bool", "boolean"].includes(value)) return "boolean";
  return "string";
}

function defaultConfig(schema) {
  return Object.fromEntries(schema.map((item) => [item.name, item.default]));
}

function createFieldControl(item, configuredValue) {
  const wrapper = document.createElement("label");
  wrapper.className = `config-field${item.type === "boolean" ? " boolean-field" : ""}`;
  const value = configuredValue !== undefined ? configuredValue : item.default;
  const label = document.createElement("span");
  label.className = "field-label";
  label.textContent = item.label;
  wrapper.appendChild(label);

  let control;
  if (item.type === "boolean") {
    control = document.createElement("input");
    control.type = "checkbox";
    control.checked = Boolean(value);
  } else if (Array.isArray(item.options)) {
    control = document.createElement("select");
    if (item.nullable) control.appendChild(new Option("None", ""));
    item.options.forEach((option) => {
      const optionValue = typeof option === "object" ? option.value : option;
      const optionLabel = typeof option === "object" ? option.label : option;
      control.appendChild(new Option(String(optionLabel), String(optionValue)));
    });
    control.value = value === null || value === undefined ? "" : String(value);
  } else {
    control = document.createElement("input");
    control.type = ["number", "integer"].includes(item.type) ? "number" : "text";
    if (item.type === "number") control.step = "any";
    if (item.type === "integer") control.step = "1";
    control.value = value === null || value === undefined ? "" : String(value);
    if (item.nullable) control.placeholder = "None";
  }
  control.name = item.name;
  control.className = "field-control";
  control.required = !item.nullable;
  applyInputConstraints(control, item);
  wrapper.appendChild(control);
  return wrapper;
}

function applyInputConstraints(control, item) {
  if (control instanceof HTMLInputElement && control.type === "number") {
    if (item.minimum !== undefined) control.min = String(item.minimum);
    else if (item.exclusiveMinimum !== undefined) {
      control.min = String(exclusiveBoundary(item.exclusiveMinimum, item.type, 1));
    }
    if (item.maximum !== undefined) control.max = String(item.maximum);
    else if (item.exclusiveMaximum !== undefined) {
      control.max = String(exclusiveBoundary(item.exclusiveMaximum, item.type, -1));
    }
  }
  if (control instanceof HTMLInputElement && control.type === "text") {
    if (item.pattern) control.pattern = item.pattern;
    if (item.minLength !== undefined) control.minLength = Number(item.minLength);
    if (item.maxLength !== undefined) control.maxLength = Number(item.maxLength);
  }
}

function exclusiveBoundary(value, type, direction) {
  const number = Number(value);
  if (type === "integer") return number + direction;
  const delta = Number.EPSILON * Math.max(1, Math.abs(number));
  return number + direction * delta;
}

function normalizePayload(payload) {
  const body = payload?.result || payload || {};
  const candles = body.candles || body.rows || body.data || [];
  return {
    ...body,
    config: body.config || body.data_config || body.parameters || {},
    manifest: body.manifest || body.data_manifest || {},
    candles: Array.isArray(candles) ? candles : [],
  };
}

function renderSummary(container, payload) {
  const candles = payload.candles || [];
  const counts = { "-1": 0, "0": 0, "1": 0, "2": 0 };
  candles.forEach((candle) => {
    const key = String(Math.round(Number(candle.label)));
    if (key in counts) counts[key] += 1;
  });
  const total = candles.length || Number(payload.summary?.row_count) || 0;
  const distribution = payload.summary?.label_ratios
    || payload.label_ratios
    || payload.summary?.label_counts
    || payload.label_counts
    || payload.summary?.label_distribution
    || payload.label_distribution
    || {};
  const distributionTotal = Object.values(distribution)
    .map(Number)
    .filter(Number.isFinite)
    .reduce((sum, value) => sum + value, 0);
  const ratio = (key) => {
    const direct = distribution[key] ?? distribution[Number(key)];
    if (direct !== undefined) {
      const numeric = Number(direct);
      const normalized = distributionTotal > 1.000001 ? numeric / distributionTotal : numeric;
      return `${(normalized * 100).toFixed(2)}%`;
    }
    return total ? `${(counts[key] / total * 100).toFixed(2)}%` : "—";
  };
  container.innerHTML = [
    ["Rows", total ? total.toLocaleString() : "—"],
    ["Negative", ratio("-1")],
    ["Neutral", ratio("0")],
    ["Positive", ratio("1")],
  ].map(([label, value]) => `<article class="summary-card"><span>${label}</span><strong>${value}</strong></article>`).join("");
}

function humanize(value) {
  return String(value || "").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? formatValue(value) : date.toLocaleDateString();
}
