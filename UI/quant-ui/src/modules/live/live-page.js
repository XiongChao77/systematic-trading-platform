import "./live.css";

import { escapeHtml, formatPrice } from "../../shared/formatters.js";
import { loadLiveStrategies, loadLiveStrategy } from "./live-api.js";

const REFRESH_INTERVAL_MS = 1_000;

export async function mountLivePage(container, context) {
  const strategyId = context.params.get("strategy_id");
  return strategyId
    ? mountDetail(container, context, strategyId)
    : mountList(container, context);
}

function mountList(container, context) {
  let disposed = false;
  let activeController = null;
  let hasData = false;

  container.innerHTML = `
    <section class="module-page scroll-page live-page">
      <header class="page-header">
        <div>
          <p class="page-eyebrow">Live monitoring</p>
          <h1>Strategy List</h1>
          <p class="page-subtitle">Current state reported by live runners</p>
        </div>
        <span data-role="status" class="status-pill busy">Connecting</span>
      </header>
      <section class="panel live-list-panel">
        <div data-role="state" class="page-state">
          <span class="spinner"></span>
          <span>Waiting for live strategies...</span>
        </div>
        <div data-role="table-wrap" class="table-wrap hidden">
          <table class="live-strategy-table">
            <thead>
              <tr>
                <th>Strategy ID</th>
                <th>Model</th>
                <th>Venue</th>
                <th>Symbol</th>
                <th>Interval</th>
                <th>Balance</th>
                <th title="Used position margin / account equity">Margin utilization</th>
                <th>Unrealized PnL</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </section>
    </section>
  `;

  const state = container.querySelector('[data-role="state"]');
  const tableWrap = container.querySelector('[data-role="table-wrap"]');
  const tableBody = container.querySelector("tbody");
  const status = container.querySelector('[data-role="status"]');

  function render(payload) {
    const items = Array.isArray(payload?.items) ? payload.items : [];
    hasData = items.length > 0;
    status.textContent = payload?.runners?.unavailable
      ? `${payload.runners.unavailable} runner unavailable`
      : "Live";
    status.className = `status-pill ${payload?.runners?.unavailable ? "busy" : "success"}`;

    if (!items.length) {
      tableWrap.classList.add("hidden");
      state.className = "page-state";
      state.innerHTML = "<strong>No live strategies</strong><span>Waiting for a runner snapshot.</span>";
      return;
    }

    state.classList.add("hidden");
    tableWrap.classList.remove("hidden");
    tableWrap.classList.remove("live-request-failed");
    tableBody.innerHTML = items.map((item) => {
      const available = item.available === true;
      const balanceAvailable = available
        && item.balance !== null
        && item.balance !== undefined
        && Number.isFinite(Number(item.balance));
      const pnlAvailable = available
        && item.unrealized_pnl !== null
        && item.unrealized_pnl !== undefined
        && Number.isFinite(Number(item.unrealized_pnl));
      const pnl = pnlAvailable ? formatMoney(item.unrealized_pnl, true) : "—";
      const margin = available ? formatPercent(item.margin_utilization) : "—";
      const tone = pnlAvailable && Number(item.unrealized_pnl) > 0
        ? "positive"
        : pnlAvailable && Number(item.unrealized_pnl) < 0
          ? "negative"
          : "";
      return `
        <tr class="live-strategy-row${available ? "" : " unavailable"}" tabindex="0" data-strategy-id="${escapeHtml(item.strategy_id)}">
          <td class="strategy-id-cell">${escapeHtml(item.strategy_id)}</td>
          <td title="${escapeHtml(item.model_type)}">${escapeHtml(shortModelType(item.model_type))}</td>
          <td>${escapeHtml(formatVenue(item.venue))}</td>
          <td>${escapeHtml(item.symbol)}</td>
          <td>${escapeHtml(item.interval)}</td>
          <td class="${balanceAvailable ? "" : "value-unavailable"}">${balanceAvailable ? formatMoney(item.balance) : "—"}</td>
          <td class="${margin === "—" ? "value-unavailable" : ""}">${margin}</td>
          <td class="${pnlAvailable ? tone : "value-unavailable"}">${pnl}</td>
          <td>${renderStatus(item.status, available)}</td>
        </tr>
      `;
    }).join("");
  }

  function openRow(row) {
    const selected = row?.dataset.strategyId;
    if (selected) context.navigate("live", { strategy_id: selected });
  }

  tableBody.addEventListener("click", (event) => openRow(event.target.closest("tr")));
  tableBody.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openRow(event.target.closest("tr"));
    }
  });

  async function refresh() {
    if (disposed || activeController) return;
    activeController = new AbortController();
    try {
      render(await loadLiveStrategies({ signal: activeController.signal }));
    } catch (error) {
      if (disposed || error.name === "AbortError") return;
      status.textContent = "Unavailable";
      status.className = "status-pill";
      if (hasData) {
        tableWrap.classList.add("live-request-failed");
      } else {
        state.className = "page-state error-state";
        state.innerHTML = `<strong>Live API unavailable</strong><span>${escapeHtml(error.message)}</span>`;
      }
    } finally {
      activeController = null;
    }
  }

  refresh();
  const timer = window.setInterval(refresh, REFRESH_INTERVAL_MS);
  return () => {
    disposed = true;
    window.clearInterval(timer);
    activeController?.abort();
  };
}

function mountDetail(container, context, strategyId) {
  let disposed = false;
  let activeController = null;
  let hasData = false;

  container.innerHTML = `
    <section class="module-page scroll-page live-page live-detail-page">
      <header class="live-detail-header">
        <button data-action="back" class="back-button" type="button" aria-label="Back to strategy list">&#8592;</button>
        <div>
          <p class="page-eyebrow">Live strategy</p>
          <h1>${escapeHtml(strategyId)}</h1>
          <p data-role="subtitle" class="page-subtitle">Loading current state...</p>
        </div>
        <span data-role="status" class="status-pill busy">Connecting</span>
      </header>
      <div data-role="cards" class="live-detail-grid">
        ${detailCard("strategy", "Strategy Status")}
        ${detailCard("account", "Account")}
        ${detailCard("position", "Current Position")}
        ${detailCard("signal", "Latest Signal")}
      </div>
      <div data-role="state" class="page-state hidden"></div>
    </section>
  `;

  container.querySelector('[data-action="back"]').addEventListener("click", () => {
    context.navigate("live");
  });
  const cards = container.querySelector('[data-role="cards"]');
  const state = container.querySelector('[data-role="state"]');
  const status = container.querySelector('[data-role="status"]');
  const subtitle = container.querySelector('[data-role="subtitle"]');

  function render(payload) {
    hasData = true;
    cards.classList.remove("live-request-failed");
    cards.classList.remove("hidden");
    state.classList.add("hidden");
    subtitle.textContent = `${payload.symbol} · ${payload.interval}`;
    status.textContent = payload.available
      ? titleCase(payload.status)
      : "Unavailable";
    status.className = `status-pill ${statusPillTone(payload.status, payload.available)}`;

    renderCard(container, "strategy", true, [
      ["Strategy ID", payload.strategy_id],
      ["Symbol", payload.symbol],
      ["Interval", payload.interval],
      ["Risk per trade", formatPercent(payload.risk_per_trade_pct)],
      ["Max daily loss", formatPercent(payload.max_daily_loss_pct)],
      ["Max holding time", formatDuration(payload.max_holding_seconds)],
      ["Status", payload.available ? titleCase(payload.status) : null, statusValueTone(payload.status, payload.available)],
    ]);

    const accountAvailable = payload.available && payload.availability?.account;
    renderCard(container, "account", accountAvailable, [
      ["Balance", formatMoney(payload.account?.balance)],
      ["Equity", formatMoney(payload.account?.equity)],
      ["Used margin", formatMoney(accountAvailable ? payload.account?.used_margin : null)],
      [
        "Margin utilization",
        formatPercent(accountAvailable ? payload.account?.margin_utilization : null),
        "",
        "Used position margin / account equity",
      ],
    ]);

    const positionAvailable = payload.available && payload.availability?.position;
    const position = payload.position;
    renderCard(container, "position", positionAvailable, [
      ["Side", position ? titleCase(position.side) : "Flat"],
      ["Size", position ? formatPrice(position.quantity, 6) : "—"],
      ["Entry", position ? formatPrice(position.entry_price) : "—"],
      ["Current price", position ? formatPrice(position.mark_price) : "—"],
      ["Stop loss", position ? formatProtectionPrice(position.stop_loss_price, position.entry_price, position.side) : "—"],
      ["Take profit", position ? formatProtectionPrice(position.take_profit_price, position.entry_price, position.side) : "—"],
      ["Opened at", position ? formatDateTime(position.opened_at) : "—"],
      ["Remaining holding time", position ? formatDuration(position.remaining_holding_seconds) : "—"],
      [
        "Unrealized PnL",
        position ? formatMoney(position.unrealized_pnl, true) : "$0.00",
        pnlTone(position?.unrealized_pnl),
      ],
    ]);
    renderPositionComponents(container, positionAvailable, position);

    const signalAvailable = payload.available && payload.availability?.latest_signal;
    const signal = payload.latest_signal;
    renderCard(container, "signal", signalAvailable, [
      ["Model output", signal ? titleCase(signal.model_output) : null],
      ["Probability", formatProbability(signal?.probability)],
      ["Decision", signal ? titleCase(signal.decision) : null],
      ["Updated", formatDateTime(signal?.updated_at)],
    ]);
  }

  async function refresh() {
    if (disposed || activeController) return;
    activeController = new AbortController();
    try {
      render(await loadLiveStrategy(strategyId, { signal: activeController.signal }));
    } catch (error) {
      if (disposed || error.name === "AbortError") return;
      status.textContent = "Unavailable";
      status.className = "status-pill";
      if (hasData) {
        cards.classList.add("live-request-failed");
      } else {
        cards.classList.add("hidden");
        state.className = "page-state error-state";
        state.innerHTML = `<strong>Strategy unavailable</strong><span>${escapeHtml(error.message)}</span>`;
      }
    } finally {
      activeController = null;
    }
  }

  refresh();
  const timer = window.setInterval(refresh, REFRESH_INTERVAL_MS);
  return () => {
    disposed = true;
    window.clearInterval(timer);
    activeController?.abort();
  };
}

function detailCard(name, title) {
  return `
    <section data-card="${name}" class="panel live-detail-card unavailable">
      <div class="panel-header"><h2>${title}</h2></div>
      <dl></dl>
    </section>
  `;
}

function renderCard(container, name, available, fields) {
  const card = container.querySelector(`[data-card="${name}"]`);
  card.classList.toggle("unavailable", !available);
  card.querySelector("dl").innerHTML = fields.map(([label, value, tone = "", description = ""]) => {
    const unavailable = value === null || value === undefined || value === "—";
    const classes = [unavailable ? "value-unavailable" : "", unavailable ? "" : tone]
      .filter(Boolean)
      .join(" ");
    return `
      <div>
        <dt${description ? ` title="${escapeHtml(description)}"` : ""}>${escapeHtml(label)}</dt>
        <dd class="${classes}">${escapeHtml(value ?? "—")}</dd>
      </div>
    `;
  }).join("");
}

function renderPositionComponents(container, available, position) {
  const card = container.querySelector('[data-card="position"]');
  card.querySelector('[data-role="position-components"]')?.remove();
  if (!available || !position) return;

  const components = Array.isArray(position.components) && position.components.length
    ? position.components
    : [position];
  const section = document.createElement("section");
  section.className = "live-position-components";
  section.dataset.role = "position-components";
  section.innerHTML = `
    <h3>Venue positions</h3>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Size</th>
            <th>Entry price</th>
            <th>SL</th>
            <th>TP</th>
          </tr>
        </thead>
        <tbody>
          ${components.map((component) => `
            <tr>
              <td>${escapeHtml(formatPrice(component.quantity, 6))}</td>
              <td>${escapeHtml(formatPrice(component.entry_price))}</td>
              <td>${escapeHtml(formatProtectionPrice(component.stop_loss_price, component.entry_price, position.side))}</td>
              <td>${escapeHtml(formatProtectionPrice(component.take_profit_price, component.entry_price, position.side))}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
  card.append(section);
}

function renderStatus(value, available) {
  if (!available || !value) return '<span class="status-pill">—</span>';
  return `<span class="status-pill ${statusPillTone(value, available)}">${escapeHtml(titleCase(value))}</span>`;
}

function statusPillTone(value, available) {
  if (!available || !value) return "";
  return value === "running" ? "success" : "error";
}

function statusValueTone(value, available) {
  if (!available || !value) return "";
  return value === "running" ? "positive" : "negative";
}

function pnlTone(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number === 0) return "";
  return number > 0 ? "positive" : "negative";
}

function shortModelType(value) {
  const names = {
    conv_lstm: "ConvLSTM",
    logistic_regression: "LogReg",
    logistic_regression_sklearn: "LogRegSK",
    transformer: "Trans",
    xgboost: "XGB",
    lstm: "LSTM",
    mamba: "Mamba",
    tcn: "TCN",
    cnn: "CNN",
    svc: "SVC",
  };
  if (value === null || value === undefined) return "-";
  const modelType = String(value);
  return Object.hasOwn(names, modelType) ? names[modelType] : modelType.slice(0, 8);
}

function formatVenue(value) {
  const normalized = String(value || "").trim().toLowerCase();
  return {
    ctrader: "cTrader",
    mt5: "MT5",
    bybit: "Bybit",
    binance: "Binance",
    bitget: "Bitget",
    mock: "Mock",
  }[normalized] || value || "—";
}

function formatMoney(value, signed = false) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
  const number = Number(value);
  const sign = signed && number > 0 ? "+" : "";
  const negative = number < 0 ? "-" : "";
  return `${negative}${sign}$${Math.abs(number).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function formatProbability(value) {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "—";
}

function formatPercent(value) {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(2)}%` : "—";
}

function formatProtectionPrice(value, entryPrice, side) {
  const formattedPrice = formatPrice(value);
  const price = Number(value);
  const entry = Number(entryPrice);
  const direction = side === "long" ? 1 : side === "short" ? -1 : null;
  if (
    formattedPrice === "—"
    || !Number.isFinite(entry)
    || entry <= 0
    || direction === null
  ) {
    return formattedPrice;
  }
  const returnPct = direction * (price / entry - 1) * 100;
  const sign = returnPct > 0 ? "+" : "";
  return `${formattedPrice} (${sign}${returnPct.toFixed(2)}%)`;
}

function formatDateTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString();
}

function formatDuration(value) {
  if (value === null || value === undefined || value === "") return "—";
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue) || numericValue < 0) return "—";
  let seconds = Math.ceil(numericValue);
  const days = Math.floor(seconds / 86_400);
  seconds %= 86_400;
  const hours = Math.floor(seconds / 3_600);
  seconds %= 3_600;
  const minutes = Math.floor(seconds / 60);
  seconds %= 60;
  const parts = [];
  if (days) parts.push(`${days}d`);
  if (hours || days) parts.push(`${hours}h`);
  if (minutes || hours || days) parts.push(`${minutes}m`);
  parts.push(`${seconds}s`);
  return parts.join(" ");
}

function titleCase(value) {
  if (!value) return "—";
  const text = String(value).replaceAll("_", " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}
