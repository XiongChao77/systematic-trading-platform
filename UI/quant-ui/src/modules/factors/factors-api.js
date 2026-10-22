import { apiRequest, postJson } from "../../shared/api-client.js";

export function loadFactorDatasets(signal) {
  return apiRequest("/api/factors/datasets", { signal });
}

export function browseFactorReports(path = "", signal) {
  const query = path ? `?path=${encodeURIComponent(path)}` : "";
  return apiRequest(`/api/factors/browse${query}`, { signal });
}

export function loadFactorStatus(signal) {
  return apiRequest("/api/factors/status", { signal });
}

export function loadFactorReport(reportPath, signal) {
  return postJson("/api/factors/load", { report_path: reportPath }, { signal });
}

export function startFactorAnalysis(payload, signal) {
  return postJson("/api/factors/analyze", payload, { signal });
}
