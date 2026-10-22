import "./factors.css";

import { escapeHtml } from "../../shared/formatters.js";
import { browseFactorReports, loadFactorDatasets, loadFactorReport, loadFactorStatus, startFactorAnalysis } from "./factors-api.js";

export async function mountFactorsPage(container) {
  const controller = new AbortController();
  const { signal } = controller;
  let datasets = [];
  let selectedId = "";
  let activeReportPath = "";
  let timer = null;
  let openOnCompletion = false;

  container.innerHTML = `
    <section class="module-page scroll-page factors-page">
      <section data-role="source-view">
        <header class="page-header">
          <div>
            <p class="page-eyebrow">Research and diagnostics</p>
            <h1>Factors</h1>
            <p class="page-subtitle">Run factor analysis on prepared train/test data, or load a saved analysis report by path.</p>
          </div>
          <span data-role="status" class="status-pill busy">Loading</span>
        </header>
        <section class="panel factors-controls">
        <label class="field-label" for="factor-dataset">Prepared dataset</label>
        <div class="factors-run-row">
          <select id="factor-dataset" class="field-control" data-role="dataset"></select>
          <button class="secondary-button" data-action="reload" type="button">Reload</button>
          <button class="primary-button" data-action="run" type="button">Run analysis</button>
        </div>
        <label class="field-label" for="factor-list">Factors to analyze <span class="panel-note">Leave empty to analyze all eligible numeric factors</span></label>
        <input id="factor-list" class="field-control" data-role="factor-list" type="text" placeholder="RSI_14, MACD_12_26_HIST_PCT" autocomplete="off" />
        <div class="factors-job" data-role="job" aria-live="polite"></div>
        <label class="field-label" for="factor-report-path">Load existing analysis results</label>
        <div class="factors-run-row">
          <input id="factor-report-path" class="field-control" data-role="report-path" type="text" value="output/data/factor_analysis" placeholder="output/data/factor_analysis" autocomplete="off" />
          <button class="secondary-button" data-action="browse-reports" type="button">Browse folders</button>
          <button class="primary-button" data-action="load-report" type="button">Load results</button>
        </div>
        <section class="report-browser hidden" data-role="report-browser" aria-label="Choose a factor analysis report folder">
          <div class="report-browser-toolbar">
            <button class="secondary-button" data-action="report-parent" type="button">Parent</button>
            <div class="report-browser-location"><span class="panel-note">Current folder</span><strong data-role="browser-current"></strong></div>
            <button class="primary-button" data-action="choose-report-folder" type="button" disabled>Use this folder</button>
          </div>
          <div class="report-browser-note" data-role="browser-note"></div>
          <div class="report-directory-list" data-role="report-directory-list"></div>
        </section>
        <div class="factors-load-status" data-role="load-status" aria-live="polite"></div>
        </section>
      </section>
      <section class="factors-dashboard-view hidden" data-role="dashboard-view">
        <header class="page-header factors-dashboard-header">
          <div>
            <p class="page-eyebrow">Factor research</p>
            <h1>Analysis dashboard</h1>
            <p class="page-subtitle" data-role="report-note"></p>
          </div>
          <button class="secondary-button" data-action="back" type="button">Back to Factors</button>
        </header>
        <iframe class="factors-frame" data-role="frame" title="Interactive factor analysis dashboard" loading="eager"></iframe>
      </section>
    </section>
  `;

  const query = (selector) => container.querySelector(selector);
  const datasetSelect = query('[data-role="dataset"]');
  const status = query('[data-role="status"]');
  const job = query('[data-role="job"]');
  const loadStatus = query('[data-role="load-status"]');
  const sourceView = query('[data-role="source-view"]');
  const dashboardView = query('[data-role="dashboard-view"]');
  const frame = query('[data-role="frame"]');
  const reportNote = query('[data-role="report-note"]');
  const reportPathInput = query('[data-role="report-path"]');
  const reportBrowser = query('[data-role="report-browser"]');
  let browserCurrent = "";

  const setStatus = (text, tone = "busy") => {
    status.textContent = text;
    status.className = `status-pill ${tone}`;
  };

  function currentDataset() {
    return datasets.find((item) => item.id === selectedId) || null;
  }

  function showReportState(title, message) {
    loadStatus.innerHTML = `<strong>${escapeHtml(title)}</strong><span>${escapeHtml(message)}</span>`;
  }

  function showReport(report) {
    activeReportPath = report.report_path;
    sourceView.classList.add("hidden");
    dashboardView.classList.remove("hidden");
    frame.src = `/api/factors/dashboard?report_path=${encodeURIComponent(activeReportPath)}`;
    reportNote.textContent = `${report.factor_count} factors · ${Number(report.row_count).toLocaleString()} rows · ${report.start || ""} to ${report.end || ""} · ${activeReportPath}`;
  }

  async function loadReportFromPath() {
    const path = reportPathInput.value.trim();
    if (!path) return;
    showReportState("Loading saved results", path);
    try {
      const report = await loadFactorReport(path, signal);
      reportPathInput.value = report.report_path;
      showReport(report);
    } catch (error) {
      const message = /not found|404/i.test(error.message)
        ? `The Factors API endpoint is unavailable. Restart the updated FastAPI backend, then try again. (${error.message})`
        : /fetch|network|connection/i.test(error.message)
          ? `Cannot reach the FastAPI backend. Start or restart it, then try again. (${error.message})`
          : error.message;
      showReportState("Unable to load results", message);
      reportNote.textContent = path;
    }
  }

  async function browseReportFolders(path = "") {
    const result = await browseFactorReports(path, signal);
    browserCurrent = result.current || "";
    reportBrowser.classList.remove("hidden");
    query('[data-role="browser-current"]').textContent = browserCurrent || result.root;
    query('[data-action="report-parent"]').disabled = result.parent === null;
    query('[data-action="choose-report-folder"]').disabled = !result.direct_report;
    query('[data-role="browser-note"]').textContent = result.direct_report
      ? "This folder contains a factor analysis report and can be selected."
      : "Open folders until one contains a saved factor analysis report (metadata.json).";

    const list = query('[data-role="report-directory-list"]');
    list.innerHTML = "";
    (result.children || []).forEach((child) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `report-directory${child.has_report ? " has-report" : ""}`;
      button.innerHTML = `<span class="report-directory-icon">📁</span><span>${escapeHtml(child.name)}</span>${child.has_report ? '<span class="report-directory-badge">Report</span>' : ""}`;
      button.title = child.path;
      button.addEventListener("click", () => browseReportFolders(child.path).catch((error) => {
        showReportState("Unable to browse folders", error.message);
      }), { signal });
      list.appendChild(button);
    });
    if (!list.children.length) {
      list.innerHTML = '<span class="panel-note">No subfolders here.</span>';
    }
  }

  async function reloadDatasets() {
    const result = await loadFactorDatasets(signal);
    datasets = result.datasets || [];
    datasetSelect.innerHTML = datasets.length
      ? datasets.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.label)}${item.available ? " · report ready" : ""}</option>`).join("")
      : '<option value="">No prepared train/test datasets found</option>';
    if (!datasets.some((item) => item.id === selectedId)) {
      selectedId = datasets[0]?.id || "";
    }
    datasetSelect.value = selectedId;
    const dataset = currentDataset();
    if (dataset?.report_path && !reportPathInput.value.trim()) {
      reportPathInput.value = dataset.report_path;
    }
    if (!datasets.length) {
      loadStatus.innerHTML = `<strong>No prepared dataset found</strong><span>Expected train_data.feather and test_data.feather under ${escapeHtml(result.data_root)}.</span>`;
    }
  }

  function renderJob(value) {
    if (value.status === "running") {
      const progress = value.progress;
      const detail = progress
        ? `Factor ${progress.completed}/${progress.total}: ${escapeHtml(progress.factor)}`
        : escapeHtml(value.message || "Preparing analysis...");
      job.innerHTML = `<div class="factors-progress"><span class="spinner"></span><div><strong>Analysis running</strong><span>${detail}</span><span>This runs in the background; you can leave this page open.</span></div></div>`;
      setStatus("Running", "busy");
      query('[data-action="run"]').disabled = true;
      return;
    }
    query('[data-action="run"]').disabled = false;
    if (value.status === "failed") {
      job.innerHTML = `<div class="factors-job-error"><strong>Analysis failed</strong><span>${escapeHtml(value.error || value.message || "Unknown error")}</span></div>`;
      setStatus("Failed", "error");
    } else if (value.status === "completed") {
      job.innerHTML = `<div class="factors-job-success">${escapeHtml(value.message || "Analysis complete.")}</div>`;
      setStatus("Ready", "success");
    } else {
      job.innerHTML = '<div class="factors-job-idle">Choose a dataset and run analysis. The default analysis checks all eligible factors.</div>';
      setStatus("Ready", "success");
    }
  }

  function startPolling() {
    if (!timer) {
      timer = window.setInterval(() => refreshStatus().catch((error) => {
        job.textContent = error.message;
      }), 2500);
    }
  }

  async function refreshStatus() {
    const value = await loadFactorStatus(signal);
    renderJob(value);
    if (value.status === "running") startPolling();
    if (value.status === "completed" && openOnCompletion && value.dataset_id === selectedId) {
      openOnCompletion = false;
      await reloadDatasets();
      reportPathInput.value = value.report_path || currentDataset()?.report_path || reportPathInput.value;
      await loadReportFromPath();
    }
    if (value.status === "failed") openOnCompletion = false;
    if (value.status !== "running" && timer) {
      window.clearInterval(timer);
      timer = null;
    }
  }

  query('[data-action="reload"]').addEventListener("click", async () => {
    try {
      await reloadDatasets();
      await refreshStatus();
    } catch (error) {
      setStatus("Unavailable", "error");
      showReportState("Unable to reload datasets", error.message);
    }
  }, { signal });

  datasetSelect.addEventListener("change", () => {
    selectedId = datasetSelect.value;
    const dataset = currentDataset();
    if (dataset?.report_path) reportPathInput.value = dataset.report_path;
    showReportState("Report path changed", "Load results to open the selected report directory.");
  }, { signal });

  query('[data-action="back"]').addEventListener("click", () => {
    dashboardView.classList.add("hidden");
    sourceView.classList.remove("hidden");
  }, { signal });

  query('[data-action="load-report"]').addEventListener("click", loadReportFromPath, { signal });

  query('[data-action="browse-reports"]').addEventListener("click", async () => {
    try {
      const typedPath = reportPathInput.value.trim();
      await browseReportFolders(typedPath);
    } catch (error) {
      try {
        await browseReportFolders("");
      } catch (rootError) {
        showReportState("Unable to browse folders", rootError.message || error.message);
      }
    }
  }, { signal });

  query('[data-action="report-parent"]').addEventListener("click", async () => {
    try {
      const result = await browseFactorReports(browserCurrent, signal);
      if (result.parent !== null) await browseReportFolders(result.parent);
    } catch (error) {
      showReportState("Unable to browse folders", error.message);
    }
  }, { signal });

  query('[data-action="choose-report-folder"]').addEventListener("click", () => {
    reportPathInput.value = browserCurrent;
    reportBrowser.classList.add("hidden");
    showReportState("Folder selected", "Load results to open the report in this folder.");
  }, { signal });

  query('[data-action="run"]').addEventListener("click", async () => {
    if (!selectedId) return;
    const factors = query('[data-role="factor-list"]').value
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean);
    try {
      openOnCompletion = true;
      const started = await startFactorAnalysis({ dataset_id: selectedId, factors: factors.length ? factors : null }, signal);
      renderJob(started);
      if (started.report_path) reportPathInput.value = started.report_path;
      startPolling();
    } catch (error) {
      job.textContent = error.message;
      setStatus("Failed", "error");
    }
  }, { signal });

  try {
    await reloadDatasets();
    await refreshStatus();
  } catch (error) {
    setStatus("Unavailable", "error");
    showReportState("Unable to load factor analysis", error.message);
  }

  return () => {
    controller.abort();
    if (timer) window.clearInterval(timer);
  };
}
