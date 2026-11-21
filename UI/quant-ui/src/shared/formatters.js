export function getByPath(value, path, fallback = null) {
  if (!value || !path) return fallback;
  let current = value;
  for (const key of path.split(".")) {
    if (current === null || current === undefined || current[key] === null || current[key] === undefined) {
      return fallback;
    }
    current = current[key];
  }
  return current;
}

export function formatValue(value, maximumFractionDigits = 4) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "—";
    if (Number.isInteger(value)) return value.toLocaleString();
    return value.toLocaleString(undefined, { maximumFractionDigits });
  }
  if (typeof value === "boolean") return value ? "True" : "False";
  if (Array.isArray(value)) return value.join(", ");
  return String(value);
}

export function formatPrice(value, decimals = 6) {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  if (Math.abs(number) >= 1000) return number.toFixed(2);
  if (Math.abs(number) >= 1) return number.toFixed(Math.min(decimals, 4));
  return number.toFixed(decimals);
}

export function formatPercent(value, { scale = 100, decimals = 2 } = {}) {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return `${(number * scale).toFixed(decimals)}%`;
}

export function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[character]);
}

export function shortFieldName(path) {
  return String(path).split(".").at(-1).replaceAll("_", " ");
}

export function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  const totalMinutes = Math.floor(seconds / 60);
  const days = Math.floor(totalMinutes / 1440);
  const hours = Math.floor((totalMinutes % 1440) / 60);
  const minutes = totalMinutes % 60;
  if (days) return `${days}d ${hours}h`;
  if (hours) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}

export function pnlTone(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number === 0) return "";
  return number > 0 ? "positive" : "negative";
}

export function formatMoney(value, signed = false) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
  const number = Number(value);
  const sign = signed && number > 0 ? "+" : "";
  const negative = number < 0 ? "-" : "";
  return `${negative}${sign}$${Math.abs(number).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}
