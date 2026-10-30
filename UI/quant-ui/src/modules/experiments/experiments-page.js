import "./experiments.css";

import { createChart, CrosshairMode, LineSeries } from "lightweight-charts";
import { escapeHtml, formatValue, shortFieldName } from "../../shared/formatters.js";
import { createToast } from "../../shared/toast.js";
import { experimentGet, experimentPost } from "./experiments-api.js";

const DEFAULT_GROUP = "params.train.model_cfg.model_type";
const DEFAULT_METRIC = "performance.cagr";
const COLORS = ["#4969d1", "#e07a45", "#27947a", "#a25fc0", "#d54f70", "#697b93", "#d19a31", "#357ba8"];
let preservedExperimentState = null;
const COLUMN_LABELS = {
  "params.hash": "Hash",
  "params.train.model_cfg.model_type": "Model",
  "params.train.model_cfg.model_version": "Version",
  "performance.cagr": "CAGR",
  "performance.sharpe": "Sharpe",
  "performance.calmar": "Calmar",
  "drawdown.max_dd_pct": "Max DD",
  "trades.daily_freq": "Frequency",
  "trades.win_rate": "Win rate",
  "performance.rc_summary.rc_median": "RC median",
  "performance.rc_summary.rc_pos_ratio": "RC positive",
  "drawdown.max_hwm_duration_days": "DD days",
};

export async function mountExperimentsPage(container, context) {
  const abortController = new AbortController();
  const toast = createToast();
  const restoredState = preservedExperimentState;
  const state = {
    currentPath: restoredState?.currentPath || "",
    parentPath: null,
    selectedPaths: new Set(restoredState?.selectedPaths || []),
    datasetId: restoredState?.datasetId || null,
    period: restoredState?.period || "long",
    schema: [],
    schemaMap: new Map(),
    defaultColumns: [...(restoredState?.defaultColumns || [])],
    filters: (restoredState?.filters || []).map((filter) => ({ ...filter })),
    page: restoredState?.page || 1,
    pageSize: restoredState?.pageSize || 50,
    total: 0,
    selectedRecords: new Set(restoredState?.selectedRecords || []),
    analysisRun: Boolean(restoredState?.analysisRun),
    equityPlotted: Boolean(restoredState?.equityPlotted),
    chart: null,
    chartSeries: [],
    resizeObserver: null,
  };

  container.innerHTML = experimentTemplate();
  const root = container.querySelector(".experiments-page");
  const $ = (selector) => root.querySelector(selector);
  const $$ = (selector) => [...root.querySelectorAll(selector)];

  const apiGet = (path) => experimentGet(path, abortController.signal);
  const apiPost = (path, payload) => experimentPost(path, payload, abortController.signal);

  async function checkHealth() {
    try {
      const data = await apiGet("/health");
      setApiStatus("API connected", "success");
      $("[data-role=root-label]").textContent = `Root: ${data.reports_root || data.root || "configured server path"}`;
      await browsePath(state.currentPath);
      if (restoredState?.datasetId) await restoreWorkspace(restoredState);
    } catch (error) {
      if (error.name === "AbortError") return;
      setApiStatus("API unavailable", "error");
      toast.show(error.message || String(error), true);
    }
  }

  function setApiStatus(text, status) {
    const element = $("[data-role=api-status]");
    element.textContent = text;
    element.className = `status-pill ${status}`;
  }

  async function browsePath(path) {
    const data = await apiGet(`/browse?path=${encodeURIComponent(path || "")}`);
    state.currentPath = data.current || "";
    state.parentPath = data.parent;
    $("[data-role=path-input]").value = state.currentPath;
    $("[data-action=parent]").disabled = data.parent === null;
    $("[data-role=report-count]").textContent = `${Number(data.recursive_report_count || 0).toLocaleString()} report file(s)`;
    const list = $("[data-role=directory-list]");
    list.innerHTML = "";
    (data.children || []).forEach((child) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `directory-item${child.has_report ? " has-report" : ""}`;
      button.textContent = child.name;
      button.title = child.path;
      button.addEventListener("click", () => browsePath(child.path).catch(handleError), { signal: abortController.signal });
      list.appendChild(button);
    });
    if (!list.children.length) list.innerHTML = '<span class="panel-note">No subdirectories</span>';
  }

  function addCurrentPath() {
    const path = $("[data-role=path-input]").value.trim();
    state.selectedPaths.add(path);
    renderSelectedPaths();
  }

  function renderSelectedPaths() {
    const containerElement = $("[data-role=selected-paths]");
    containerElement.innerHTML = "";
    state.selectedPaths.forEach((path) => {
      const chip = document.createElement("span");
      chip.className = "path-chip";
      chip.innerHTML = `<span title="${escapeHtml(path || "Reports root")}">${escapeHtml(path || "Reports root")}</span>`;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "×";
      remove.setAttribute("aria-label", `Remove ${path || "reports root"}`);
      remove.addEventListener("click", () => {
        state.selectedPaths.delete(path);
        renderSelectedPaths();
      }, { signal: abortController.signal });
      chip.appendChild(remove);
      containerElement.appendChild(chip);
    });
    if (!state.selectedPaths.size) containerElement.innerHTML = '<span class="panel-note">No folders selected</span>';
    $("[data-action=load]").disabled = !state.selectedPaths.size;
  }

  async function loadSelectedPaths() {
    if (!state.selectedPaths.size) return;
    const data = await apiPost("/load", { paths: [...state.selectedPaths], deduplicate: true });
    state.datasetId = data.dataset_id;
    state.defaultColumns = data.default_columns || [];
    state.filters = [];
    state.page = 1;
    state.selectedRecords.clear();
    state.analysisRun = false;
    state.equityPlotted = false;
    $("[data-role=workspace]").classList.remove("hidden");
    $("[data-role=loaded-count]").textContent = Number(data.record_count || 0).toLocaleString();
    $("[data-role=file-count]").textContent = Number(data.report_files?.length || 0).toLocaleString();
    updateSelectedCount();
    toast.show(`Loaded ${Number(data.record_count || 0).toLocaleString()} strategies from ${data.report_files?.length || 0} report file(s)`);
    await loadSchema();
    await runQuery();
  }

  async function loadSchema() {
    const data = await apiGet(`/schema?dataset_id=${encodeURIComponent(state.datasetId)}&period=${encodeURIComponent(state.period)}`);
    state.schema = data.fields || [];
    state.schemaMap = new Map(state.schema.map((item) => [item.path, item]));
    populateSelectors();
    renderFilters();
  }

  function populateSelectors() {
    const groupable = state.schema.filter((field) => field.groupable);
    const numericMetrics = state.schema.filter((field) => field.type === "number" && field.role === "metric");
    fillFieldSelect($("[data-role=group-primary]"), groupable, DEFAULT_GROUP, false);
    fillFieldSelect($("[data-role=group-secondary]"), groupable, "", true);
    fillFieldSelect($("[data-role=metric]"), numericMetrics, DEFAULT_METRIC, false);
    fillFieldSelect($("[data-role=sort-field]"), state.schema.filter((field) => field.type === "number"), DEFAULT_METRIC, false);
  }

  function fillFieldSelect(select, fields, preferred, allowEmpty) {
    const previous = select.value;
    select.innerHTML = "";
    if (allowEmpty) select.appendChild(new Option("None", ""));
    const groups = new Map();
    fields.forEach((item) => {
      const category = item.role === "parameter" ? "Parameters" : item.category || "Metrics";
      if (!groups.has(category)) groups.set(category, []);
      groups.get(category).push(item);
    });
    groups.forEach((items, category) => {
      const group = document.createElement("optgroup");
      group.label = category;
      items.forEach((item) => group.appendChild(new Option(item.path, item.path)));
      select.appendChild(group);
    });
    const candidate = [previous, preferred].find((value) => value && fields.some((field) => field.path === value));
    if (candidate) select.value = candidate;
  }

  function renderFilters() {
    const filterList = $("[data-role=filter-list]");
    filterList.innerHTML = "";
    state.filters.forEach((filter, index) => {
      const row = document.createElement("div");
      row.className = "filter-row";
      const fieldSelect = document.createElement("select");
      fieldSelect.className = "field-control";
      fillFieldSelect(fieldSelect, state.schema, filter.field, false);
      fieldSelect.value = filter.field || fieldSelect.value;
      const operatorSelect = document.createElement("select");
      operatorSelect.className = "field-control";
      const rebuildOperators = () => {
        const field = state.schemaMap.get(fieldSelect.value);
        const operators = operatorsFor(field?.type);
        operatorSelect.innerHTML = "";
        operators.forEach(([value, label]) => operatorSelect.appendChild(new Option(label, value)));
        if (operators.some(([value]) => value === filter.operator)) operatorSelect.value = filter.operator;
      };
      rebuildOperators();
      const input = document.createElement("input");
      input.className = "field-control";
      input.placeholder = "Value; use commas for lists or ranges";
      input.value = filter.rawValue ?? "";
      input.disabled = ["is_null", "not_null"].includes(operatorSelect.value);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "remove-filter";
      remove.textContent = "×";

      fieldSelect.addEventListener("change", () => {
        filter.field = fieldSelect.value;
        filter.operator = "eq";
        filter.rawValue = "";
        rebuildOperators();
        input.value = "";
      }, { signal: abortController.signal });
      operatorSelect.addEventListener("change", () => {
        filter.operator = operatorSelect.value;
        input.disabled = ["is_null", "not_null"].includes(filter.operator);
      }, { signal: abortController.signal });
      input.addEventListener("input", () => { filter.rawValue = input.value; }, { signal: abortController.signal });
      remove.addEventListener("click", () => {
        state.filters.splice(index, 1);
        renderFilters();
      }, { signal: abortController.signal });
      row.append(fieldSelect, operatorSelect);
      const footer = document.createElement("div");
      footer.className = "filter-row-footer";
      footer.append(input, remove);
      row.appendChild(footer);
      filterList.appendChild(row);
    });
  }

  function serializedFilters() {
    return state.filters.map((filter) => {
      const field = state.schemaMap.get(filter.field);
      return {
        field: filter.field,
        operator: filter.operator,
        value: parseFilterValue(filter.rawValue, field?.type, filter.operator),
      };
    });
  }

  async function runQuery() {
    if (!state.datasetId) return;
    const payload = {
      dataset_id: state.datasetId,
      period: state.period,
      filters: serializedFilters(),
      columns: state.defaultColumns,
      sort_by: $("[data-role=sort-field]").value || DEFAULT_METRIC,
      descending: $("[data-action=sort-direction]").dataset.descending === "true",
      page: state.page,
      page_size: Number($("[data-role=page-size]").value),
    };
    const data = await apiPost("/query", payload);
    state.total = data.total;
    state.pageSize = data.page_size;
    $("[data-role=matched-count]").textContent = Number(data.total || 0).toLocaleString();
    $("[data-role=page-label]").textContent = `Page ${data.page} / ${Math.max(1, Math.ceil(data.total / data.page_size))}`;
    $("[data-action=previous-page]").disabled = data.page <= 1;
    $("[data-action=next-page]").disabled = data.page * data.page_size >= data.total;
    renderStrategyTable(data);
  }

  function renderStrategyTable(data) {
    const head = $("[data-role=strategy-table] thead");
    const body = $("[data-role=strategy-table] tbody");
    head.innerHTML = `<tr><th>Select</th><th>Num</th>${data.columns.map((column) => `<th title="${escapeHtml(column)}">${escapeHtml(COLUMN_LABELS[column] || shortFieldName(column))}</th>`).join("")}</tr>`;
    body.innerHTML = "";
    data.items.forEach((item, offset) => {
      const row = document.createElement("tr");
      const selectCell = document.createElement("td");
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = state.selectedRecords.has(item.record_id);
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) state.selectedRecords.add(item.record_id);
        else state.selectedRecords.delete(item.record_id);
        updateSelectedCount();
      }, { signal: abortController.signal });
      selectCell.appendChild(checkbox);
      const numberCell = document.createElement("td");
      numberCell.textContent = String(
        item.strategy_number ?? ((data.page - 1) * data.page_size + offset),
      );
      row.append(selectCell, numberCell);
      data.columns.forEach((column) => {
        const cell = document.createElement("td");
        const value = item.values[column];
        if (column === "params.hash") {
          const button = document.createElement("button");
          button.type = "button";
          button.className = "hash-button";
          button.textContent = String(value ?? "—").slice(0, 12);
          button.title = `Open strategy equity\n${item.source}`;
          button.addEventListener("click", () => openBacktest(item.record_id), { signal: abortController.signal });
          cell.appendChild(button);
        } else {
          cell.textContent = compactValue(value);
          cell.title = value === null || value === undefined ? "" : String(value);
        }
        row.appendChild(cell);
      });
      body.appendChild(row);
    });
  }

  async function runAnalysis() {
    const groupBy = [$("[data-role=group-primary]").value, $("[data-role=group-secondary]").value].filter(Boolean);
    const metric = $("[data-role=metric]").value;
    const aggregation = $("[data-role=aggregation]").value;
    if (!groupBy.length || !metric) {
      toast.show("Select at least one dimension and a metric", true);
      return;
    }
    const aggregations = [...new Set(["count", aggregation])];
    const data = await apiPost("/aggregate", {
      dataset_id: state.datasetId,
      period: state.period,
      filters: serializedFilters(),
      group_by: groupBy,
      metric,
      aggregations,
    });
    state.analysisRun = true;
    $("[data-role=analysis-summary]").textContent = `${data.matched_records} strategies · ${data.group_count} groups${data.truncated ? " · truncated" : ""}`;
    renderComparison(data, aggregation);
  }

  function renderComparison(data, aggregation) {
    const chart = $("[data-role=comparison-chart]");
    const tableHead = $("[data-role=comparison-table] thead");
    const tableBody = $("[data-role=comparison-table] tbody");
    chart.className = "comparison-chart";
    chart.innerHTML = "";
    tableHead.innerHTML = `<tr>${data.group_by.map((path) => `<th>${escapeHtml(shortFieldName(path))}</th>`).join("")}<th>Count</th>${data.aggregations.filter((name) => name !== "count").map((name) => `<th>${escapeHtml(name)}</th>`).join("")}</tr>`;
    tableBody.innerHTML = "";
    const values = data.rows.map((row) => Number(row[aggregation])).filter(Number.isFinite);
    const maxAbsolute = Math.max(1e-12, ...values.map(Math.abs));
    data.rows.forEach((row) => {
      const label = data.group_by.map((path) => `${shortFieldName(path)}=${compactValue(row.group[path])}`).join(" · ");
      const value = Number(row[aggregation]);
      const bar = document.createElement("div");
      bar.className = "comparison-bar-row";
      bar.innerHTML = `<span class="comparison-label" title="${escapeHtml(label)}">${escapeHtml(label)}</span><div class="comparison-track"><div class="comparison-fill${value < 0 ? " negative-fill" : ""}" style="width:${Number.isFinite(value) ? Math.max(1, Math.abs(value) / maxAbsolute * 100) : 0}%"></div></div><span class="comparison-value">${compactValue(row[aggregation])}</span>`;
      chart.appendChild(bar);
      const tableRow = document.createElement("tr");
      data.group_by.forEach((path) => appendCell(tableRow, row.group[path]));
      appendCell(tableRow, row.count);
      data.aggregations.filter((name) => name !== "count").forEach((name) => appendCell(tableRow, row[name]));
      tableBody.appendChild(tableRow);
    });
    if (!data.rows.length) chart.innerHTML = '<div class="page-state">No aggregatable data matches the filters.</div>';
  }

  function ensureEquityChart() {
    if (state.chart) return;
    const chartContainer = $("[data-role=equity-chart]");
    state.chart = createChart(chartContainer, {
      width: chartContainer.clientWidth,
      height: 400,
      layout: { background: { color: "#ffffff" }, textColor: "#59647c" },
      grid: { vertLines: { color: "#eef1f6" }, horzLines: { color: "#eef1f6" } },
      rightPriceScale: { borderColor: "#dfe4ed" },
      timeScale: { borderColor: "#dfe4ed" },
      crosshair: { mode: CrosshairMode.Normal },
    });
    state.resizeObserver = new ResizeObserver(() => {
      if (state.chart) state.chart.applyOptions({ width: chartContainer.clientWidth });
    });
    state.resizeObserver.observe(chartContainer);
  }

  async function plotEquity() {
    if (!state.selectedRecords.size) {
      toast.show("Select at least one strategy first", true);
      return;
    }
    const data = await apiPost("/equity", {
      dataset_id: state.datasetId,
      period: state.period,
      record_ids: [...state.selectedRecords],
    });
    state.equityPlotted = true;
    ensureEquityChart();
    state.chartSeries.forEach((series) => state.chart.removeSeries(series));
    state.chartSeries = [];
    data.series.forEach((item, index) => {
      const series = state.chart.addSeries(LineSeries, {
        color: COLORS[index % COLORS.length],
        lineWidth: 2,
        title: item.label,
        priceLineVisible: false,
      });
      series.setData(item.points);
      state.chartSeries.push(series);
    });
    state.chart.timeScale().fitContent();
  }

  function openBacktest(recordId) {
    preserveState();
    context.navigate("backtests", {
      dataset_id: state.datasetId,
      record_id: recordId,
      period: "all",
      return_to: "experiments",
    });
  }

  function preserveState() {
    preservedExperimentState = {
      currentPath: state.currentPath,
      selectedPaths: [...state.selectedPaths],
      datasetId: state.datasetId,
      defaultColumns: [...state.defaultColumns],
      period: state.period,
      filters: state.filters.map((filter) => ({ ...filter })),
      page: state.page,
      pageSize: Number($("[data-role=page-size]").value) || state.pageSize,
      selectedRecords: [...state.selectedRecords],
      analysisRun: state.analysisRun,
      equityPlotted: state.equityPlotted,
      loadedCount: $("[data-role=loaded-count]").textContent,
      fileCount: $("[data-role=file-count]").textContent,
      scrollTop: root.scrollTop,
      controls: {
        groupPrimary: $("[data-role=group-primary]").value,
        groupSecondary: $("[data-role=group-secondary]").value,
        metric: $("[data-role=metric]").value,
        aggregation: $("[data-role=aggregation]").value,
        sortField: $("[data-role=sort-field]").value,
        descending: $("[data-action=sort-direction]").dataset.descending,
      },
    };
  }

  async function restoreWorkspace(snapshot) {
    $("[data-role=workspace]").classList.remove("hidden");
    $("[data-role=period]").value = state.period;
    $("[data-role=period-label]").textContent = state.period.toUpperCase();
    $("[data-role=loaded-count]").textContent = snapshot.loadedCount || "0";
    $("[data-role=file-count]").textContent = snapshot.fileCount || "0";
    $("[data-role=page-size]").value = String(state.pageSize);
    $("[data-action=sort-direction]").dataset.descending = snapshot.controls?.descending || "true";
    $("[data-action=sort-direction]").textContent = snapshot.controls?.descending === "false" ? "Ascending" : "Descending";
    await loadSchema();
    setControlValue("[data-role=group-primary]", snapshot.controls?.groupPrimary);
    setControlValue("[data-role=group-secondary]", snapshot.controls?.groupSecondary);
    setControlValue("[data-role=metric]", snapshot.controls?.metric);
    setControlValue("[data-role=aggregation]", snapshot.controls?.aggregation);
    setControlValue("[data-role=sort-field]", snapshot.controls?.sortField);
    updateSelectedCount();
    await runQuery();
    if (snapshot.analysisRun) await runAnalysis();
    if (snapshot.equityPlotted && state.selectedRecords.size) await plotEquity();
    requestAnimationFrame(() => requestAnimationFrame(() => { root.scrollTop = Number(snapshot.scrollTop || 0); }));
  }

  function setControlValue(selector, value) {
    if (value === null || value === undefined) return;
    const control = $(selector);
    if ([...control.options].some((option) => option.value === value)) control.value = value;
  }

  function openInBacktests() {
    if (state.selectedRecords.size !== 1) return;
    openBacktest([...state.selectedRecords][0]);
  }

  function updateSelectedCount() {
    const count = state.selectedRecords.size;
    $("[data-role=selected-count]").textContent = `${count} ${count === 1 ? "strategy" : "strategies"} selected`;
    $("[data-action=open-backtest]").disabled = count !== 1;
  }

  function handleError(error) {
    if (error.name === "AbortError") return;
    console.error(error);
    toast.show(error.message || String(error), true);
  }

  $("[data-action=browse]").addEventListener("click", () => browsePath($("[data-role=path-input]").value.trim()).catch(handleError), { signal: abortController.signal });
  $("[data-action=parent]").addEventListener("click", () => browsePath(state.parentPath || "").catch(handleError), { signal: abortController.signal });
  $("[data-action=add-path]").addEventListener("click", addCurrentPath, { signal: abortController.signal });
  $("[data-action=clear-paths]").addEventListener("click", () => { state.selectedPaths.clear(); renderSelectedPaths(); }, { signal: abortController.signal });
  $("[data-action=load]").addEventListener("click", () => loadSelectedPaths().catch(handleError), { signal: abortController.signal });
  $("[data-role=period]").addEventListener("change", async (event) => {
    state.period = event.target.value;
    state.page = 1;
    state.filters = [];
    state.selectedRecords.clear();
    state.analysisRun = false;
    state.equityPlotted = false;
    $("[data-role=period-label]").textContent = state.period.toUpperCase();
    updateSelectedCount();
    try {
      await loadSchema();
      await runQuery();
    } catch (error) {
      handleError(error);
    }
  }, { signal: abortController.signal });
  $("[data-action=add-filter]").addEventListener("click", () => {
    const defaultField = state.schema.find((item) => item.path === DEFAULT_GROUP) || state.schema[0];
    state.filters.push({ field: defaultField?.path || "", operator: "eq", rawValue: "" });
    renderFilters();
  }, { signal: abortController.signal });
  $("[data-action=reset-filters]").addEventListener("click", () => {
    state.filters = [];
    state.page = 1;
    renderFilters();
    runQuery().catch(handleError);
  }, { signal: abortController.signal });
  $("[data-action=apply-filters]").addEventListener("click", () => { state.page = 1; runQuery().catch(handleError); }, { signal: abortController.signal });
  $("[data-action=run-analysis]").addEventListener("click", () => runAnalysis().catch(handleError), { signal: abortController.signal });
  $("[data-role=sort-field]").addEventListener("change", () => { state.page = 1; runQuery().catch(handleError); }, { signal: abortController.signal });
  $("[data-action=sort-direction]").addEventListener("click", (event) => {
    const descending = event.currentTarget.dataset.descending !== "true";
    event.currentTarget.dataset.descending = String(descending);
    event.currentTarget.textContent = descending ? "Descending" : "Ascending";
    runQuery().catch(handleError);
  }, { signal: abortController.signal });
  $("[data-role=page-size]").addEventListener("change", () => { state.page = 1; runQuery().catch(handleError); }, { signal: abortController.signal });
  $("[data-action=previous-page]").addEventListener("click", () => {
    if (state.page > 1) { state.page -= 1; runQuery().catch(handleError); }
  }, { signal: abortController.signal });
  $("[data-action=next-page]").addEventListener("click", () => {
    if (state.page * state.pageSize < state.total) { state.page += 1; runQuery().catch(handleError); }
  }, { signal: abortController.signal });
  $("[data-action=plot-equity]").addEventListener("click", () => plotEquity().catch(handleError), { signal: abortController.signal });
  $("[data-action=clear-records]").addEventListener("click", () => {
    state.selectedRecords.clear();
    updateSelectedCount();
    $$("[data-role=strategy-table] tbody input[type=checkbox]").forEach((checkbox) => { checkbox.checked = false; });
  }, { signal: abortController.signal });
  $("[data-action=open-backtest]").addEventListener("click", openInBacktests, { signal: abortController.signal });

  renderSelectedPaths();
  updateSelectedCount();
  checkHealth();

  return () => {
    abortController.abort();
    state.resizeObserver?.disconnect();
    state.chart?.remove();
    state.chart = null;
    toast.destroy();
    container.innerHTML = "";
  };
}

function experimentTemplate() {
  return `
    <section class="module-page scroll-page experiments-page">
      <header class="page-header">
        <div>
          <p class="page-eyebrow">Batch research</p>
          <h1>Experiments</h1>
          <p class="page-subtitle">Load one or more report folders, filter strategies, and compare parameters.</p>
        </div>
        <span data-role="api-status" class="status-pill busy">Connecting</span>
      </header>
      <section class="panel experiment-path-panel">
        <div class="panel-header"><h2>Select report folders</h2><span data-role="report-count" class="panel-note"></span></div>
        <div class="path-toolbar">
          <button data-action="parent" class="secondary-button" type="button">Parent</button>
          <input data-role="path-input" class="field-control" type="text" placeholder="Path relative to the reports root" />
          <button data-action="browse" class="secondary-button" type="button">Browse</button>
          <button data-action="add-path" class="secondary-button" type="button">Add folder</button>
        </div>
        <div class="path-meta"><span data-role="root-label"></span></div>
        <div data-role="directory-list" class="directory-list"></div>
        <div class="selected-path-row">
          <div data-role="selected-paths" class="selected-paths"></div>
          <button data-action="clear-paths" class="text-button" type="button">Clear folders</button>
          <button data-action="load" class="primary-button" type="button">Load selected folders</button>
        </div>
      </section>
      <div data-role="workspace" class="experiment-workspace hidden">
        <aside class="experiment-sidebar">
          <section class="panel compact-panel">
            <div class="panel-header"><h2>Filters</h2><button data-action="reset-filters" class="text-button" type="button">Reset</button></div>
            <label class="field-label" for="experiment-period">Period</label>
            <select id="experiment-period" data-role="period" class="field-control"><option value="long">Long / full history</option><option value="forward">Forward / OOS</option></select>
            <div data-role="filter-list" class="filter-list"></div>
            <button data-action="add-filter" class="secondary-button full-button" type="button">Add condition</button>
            <button data-action="apply-filters" class="primary-button full-button" type="button">Apply filters</button>
          </section>
          <section class="panel compact-panel">
            <div class="panel-header"><h2>Parameter comparison</h2></div>
            <label class="field-label">Primary dimension</label><select data-role="group-primary" class="field-control"></select>
            <label class="field-label">Secondary dimension</label><select data-role="group-secondary" class="field-control"></select>
            <label class="field-label">Metric</label><select data-role="metric" class="field-control"></select>
            <label class="field-label">Aggregation</label>
            <select data-role="aggregation" class="field-control"><option value="median">Median</option><option value="mean">Mean</option><option value="p25">P25</option><option value="p75">P75</option><option value="p90">P90</option><option value="count">Count</option></select>
            <button data-action="run-analysis" class="primary-button full-button" type="button">Run comparison</button>
          </section>
        </aside>
        <main class="experiment-content">
          <section class="summary-grid">
            <article class="summary-card"><span>Loaded strategies</span><strong data-role="loaded-count">0</strong></article>
            <article class="summary-card"><span>Matched strategies</span><strong data-role="matched-count">0</strong></article>
            <article class="summary-card"><span>Report files</span><strong data-role="file-count">0</strong></article>
            <article class="summary-card"><span>Current period</span><strong data-role="period-label">LONG</strong></article>
          </section>
          <section class="panel">
            <div class="panel-header table-toolbar">
              <h2>Strategies</h2>
              <div class="button-row">
                <select data-role="sort-field" class="field-control inline-select"></select>
                <button data-action="sort-direction" class="secondary-button" type="button" data-descending="true">Descending</button>
                <select data-role="page-size" class="field-control inline-select"><option value="25">25 / page</option><option value="50" selected>50 / page</option><option value="100">100 / page</option></select>
              </div>
            </div>
            <div class="table-wrap"><table data-role="strategy-table"><thead></thead><tbody></tbody></table></div>
            <div class="pagination"><button data-action="previous-page" class="secondary-button" type="button">Previous</button><span data-role="page-label">Page 1</span><button data-action="next-page" class="secondary-button" type="button">Next</button></div>
          </section>
          <section class="panel">
            <div class="panel-header"><h2>Grouped comparison</h2><span data-role="analysis-summary" class="panel-note">Select dimensions to generate results</span></div>
            <div data-role="comparison-chart" class="comparison-chart page-state">No comparison results</div>
            <div class="table-wrap comparison-table"><table data-role="comparison-table"><thead></thead><tbody></tbody></table></div>
          </section>
          <section class="panel">
            <div class="panel-header">
              <h2>Selected strategies</h2>
              <div class="button-row">
                <span data-role="selected-count" class="panel-note">0 strategies selected</span>
                <button data-action="open-backtest" class="primary-button" type="button" disabled>Open in Backtests</button>
                <button data-action="plot-equity" class="secondary-button" type="button">Plot equity</button>
                <button data-action="clear-records" class="secondary-button" type="button">Clear</button>
              </div>
            </div>
            <div data-role="equity-chart" class="equity-chart"></div>
          </section>
        </main>
      </div>
    </section>
  `;
}

function operatorsFor(type) {
  if (type === "number") return [["eq", "Equals"], ["ne", "Not equal"], ["gte", "Greater than or equal"], ["lte", "Less than or equal"], ["gt", "Greater than"], ["lt", "Less than"], ["between", "Between"], ["in", "In"], ["is_null", "Missing"], ["not_null", "Not missing"]];
  if (type === "boolean") return [["eq", "Equals"], ["ne", "Not equal"], ["is_null", "Missing"], ["not_null", "Not missing"]];
  if (type === "list") return [["contains", "Contains"], ["eq", "Exact match"], ["is_null", "Missing"], ["not_null", "Not missing"]];
  return [["eq", "Equals"], ["ne", "Not equal"], ["in", "In"], ["not_in", "Not in"], ["contains", "Contains text"], ["is_null", "Missing"], ["not_null", "Not missing"]];
}

function parseFilterValue(raw, type, operator) {
  if (["is_null", "not_null"].includes(operator)) return null;
  if (["between", "in", "not_in"].includes(operator)) return String(raw).split(",").map((part) => parseScalar(part.trim(), type));
  if (type === "list" && operator === "eq") {
    try { return JSON.parse(raw); } catch { return String(raw).split(",").map((value) => value.trim()); }
  }
  return parseScalar(raw, type);
}

function parseScalar(value, type) {
  if (type === "number") return Number(value);
  if (type === "boolean") return String(value).toLowerCase() === "true";
  return value;
}

function compactValue(value) {
  const formatted = formatValue(value);
  return formatted.length > 28 ? `${formatted.slice(0, 26)}…` : formatted;
}

function appendCell(row, value) {
  const cell = document.createElement("td");
  cell.textContent = compactValue(value);
  row.appendChild(cell);
}
