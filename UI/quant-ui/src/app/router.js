const ROUTE_PATHS = new Map([
  ["live", "/"],
  ["experiments", "/experiments"],
  ["backtests", "/backtest"],
  ["labels", "/labels"],
]);
const PATH_ROUTES = new Map([
  ...Array.from(ROUTE_PATHS, ([route, path]) => [path, route]),
  ["/live", "live"],
  ["/backtests", "backtests"],
]);

export function createRouter({ outlet, navItems, routes, defaultRoute }) {
  let disposeCurrent = null;
  let navigationVersion = 0;

  function parseLocation() {
    const { pathname, search, hash } = window.location;
    if (hash.startsWith("#/")) {
      const legacy = hash.slice(1);
      const queryIndex = legacy.indexOf("?");
      const path = queryIndex === -1 ? legacy : legacy.slice(0, queryIndex);
      const route = PATH_ROUTES.get(path) ?? defaultRoute;
      const params = new URLSearchParams(search);
      if (queryIndex !== -1) {
        const legacyParams = new URLSearchParams(legacy.slice(queryIndex));
        for (const key of new Set(legacyParams.keys())) {
          params.delete(key);
          for (const value of legacyParams.getAll(key)) params.append(key, value);
        }
      }
      window.history.replaceState(null, "", routeUrl(route, params));
      return { route, params };
    }

    const path = pathname.replace(/\/+$/, "") || "/";
    const route = PATH_ROUTES.get(path) ?? defaultRoute;
    if (PATH_ROUTES.has(path) && pathname !== ROUTE_PATHS.get(route)) {
      window.history.replaceState(null, "", ROUTE_PATHS.get(route) + search + hash);
    }
    return { route, params: new URLSearchParams(search) };
  }

  function routeUrl(route, params) {
    const query = params.toString();
    return ROUTE_PATHS.get(route) + (query ? `?${query}` : "");
  }

  function navigate(route, params = {}) {
    if (!ROUTE_PATHS.has(route)) return;
    const search = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== null && value !== undefined && value !== "") {
        search.set(key, String(value));
      }
    });
    const url = routeUrl(route, search);
    const current = window.location.pathname + window.location.search + window.location.hash;
    if (url === current) return;
    window.history.pushState(null, "", url);
    render();
  }

  async function render() {
    const location = parseLocation();

    navigationVersion += 1;
    const version = navigationVersion;
    if (disposeCurrent) {
      disposeCurrent();
      disposeCurrent = null;
    }

    navItems.forEach((element, name) => {
      element.classList.toggle("active", name === location.route);
      if (name === location.route) element.setAttribute("aria-current", "page");
      else element.removeAttribute("aria-current");
    });

    outlet.innerHTML = '<div class="route-loading"><span class="spinner"></span>Loading module...</div>';
    try {
      const dispose = await routes[location.route](outlet, {
        params: location.params,
        navigate,
      });
      if (version !== navigationVersion) {
        if (typeof dispose === "function") dispose();
        return;
      }
      disposeCurrent = typeof dispose === "function" ? dispose : null;
    } catch (error) {
      if (version !== navigationVersion) return;
      console.error(error);
      outlet.innerHTML = `
        <div class="page-state error-state">
          <strong>Unable to open this module</strong>
          <span>${escapeHtml(error.message || String(error))}</span>
        </div>
      `;
    }
  }

  function start() {
    navItems.forEach((element, route) => {
      element.href = ROUTE_PATHS.get(route);
      element.addEventListener("click", event => {
        if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey
            || event.shiftKey || event.altKey) return;
        event.preventDefault();
        navigate(route);
      });
    });
    window.addEventListener("popstate", render);
    window.addEventListener("hashchange", () => {
      if (window.location.hash.startsWith("#/")) render();
    });
    render();
  }

  return { start, navigate };
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[character]);
}
