const NAVIGATION = [
  { id: "live", label: "Live", short: "LV" },
  { id: "factors", label: "Factors", short: "FA" },
  { id: "experiments", label: "Experiments", short: "EX" },
  { id: "backtests", label: "Backtests", short: "BT" },
  { id: "labels", label: "Labels", short: "LB" },
  { divider: true },
  { id: "alerts", label: "Alerts", short: "AL", disabled: true },
  { id: "settings", label: "Settings", short: "ST", disabled: true },
];

export function createAppShell(root) {
  root.innerHTML = `
    <div class="app-shell">
      <aside class="app-sidebar" aria-label="Primary navigation">
        <div class="brand">
          <span class="brand-mark" aria-hidden="true">SC</span>
          <div>
            <strong>Strategy Center</strong>
            <span>Research workspace</span>
          </div>
        </div>
        <nav class="app-navigation"></nav>
        <div class="sidebar-footer">
          <span class="connection-dot" aria-hidden="true"></span>
          Unified Application
        </div>
      </aside>
      <main class="app-content">
        <div id="route-outlet" class="route-outlet" aria-live="polite"></div>
      </main>
    </div>
  `;

  const navigation = root.querySelector(".app-navigation");
  const navItems = new Map();

  for (const item of NAVIGATION) {
    if (item.divider) {
      const divider = document.createElement("div");
      divider.className = "nav-divider";
      navigation.appendChild(divider);
      continue;
    }

    const link = document.createElement(item.disabled ? "button" : "a");
    link.className = `nav-item${item.disabled ? " disabled" : ""}`;
    link.innerHTML = `
      <span class="nav-icon" aria-hidden="true">${item.short}</span>
      <span>${item.label}</span>
      ${item.disabled ? '<span class="nav-status">Soon</span>' : ""}
    `;
    if (item.disabled) {
      link.type = "button";
      link.disabled = true;
      link.setAttribute("aria-disabled", "true");
    } else {
      link.href = `#/${item.id}`;
      navItems.set(item.id, link);
    }
    navigation.appendChild(link);
  }

  return {
    outlet: root.querySelector("#route-outlet"),
    navItems,
  };
}
