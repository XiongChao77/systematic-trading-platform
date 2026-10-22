import "./styles/tokens.css";
import "./styles/shell.css";
import "./styles/components.css";

import { createAppShell } from "./app/app-shell.js";
import { createRouter } from "./app/router.js";
import { mountLabelsPage } from "./modules/labels/labels-page.js";
import { mountBacktestsPage } from "./modules/backtests/backtests-page.js";
import { mountExperimentsPage } from "./modules/experiments/experiments-page.js";
import { mountLivePage } from "./modules/live/live-page.js";
import { mountFactorsPage } from "./modules/factors/factors-page.js";

const root = document.querySelector("#app");
const shell = createAppShell(root);

const router = createRouter({
  outlet: shell.outlet,
  navItems: shell.navItems,
  routes: {
    live: mountLivePage,
    factors: mountFactorsPage,
    labels: mountLabelsPage,
    backtests: mountBacktestsPage,
    experiments: mountExperimentsPage,
  },
  defaultRoute: "live",
});

router.start();
