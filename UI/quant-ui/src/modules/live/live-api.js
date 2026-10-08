import { apiRequest } from "../../shared/api-client.js";

export function loadLiveStrategies({ viewerId, ...options } = {}) {
  return apiRequest(`/api/live/strategies?viewer_id=${encodeURIComponent(viewerId || "")}`, options);
}

export function loadLiveStrategy(instanceId, { viewerId, ...options } = {}) {
  return apiRequest(
    `/api/live/strategies/${encodeURIComponent(instanceId)}?viewer_id=${encodeURIComponent(viewerId || "")}`,
    options,
  );
}



export function releaseLiveViewer(viewerId) {
  return apiRequest(`/api/live/viewers/${encodeURIComponent(viewerId)}`, {
    method: "DELETE", keepalive: true,
  });
}
