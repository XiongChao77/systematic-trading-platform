"""Self-contained long martingale simulator with a worst-of-two-paths bar model.

No backtrader, no venue, no broker adapter: this module owns the whole ledger
(fees, leverage, initial margin, maintenance margin, forced liquidation) and
consumes plain OHLC floats.

Why two paths per bar
---------------------
Every safety order changes the latest layer entry and therefore resets both
layer boundaries. So ``Open -> Low -> High -> Close`` is *not* automatically the
worst intrabar route: reaching Low first can lower the dynamically recalculated take-profit price
enough for the later High to close the cycle, while reaching High first may
miss the old take profit and then leave a larger position open after the fall.
Which one hurts more depends on the numbers, so both are replayed from the same
snapshot and the worse outcome is kept:

    path A: Open -> Low  -> High -> Close
    path B: Open -> High -> Low  -> Close

"Worse" is ordered as (1) liquidation, (2) lower mark-to-market equity at
Close, and (3) larger position notional carried into the next bar. Ruin is a
backtest-level capital rule; the engine only records realized grid breaks,
liquidations and mark-to-market extrema. Intrabar equity and margin extrema are
tracked on the selected path rather than only at the bar close.

When the caller supplies finer bars for the same period, the two paths are
only used where they disagree: the finer sequence then replays the true order
and replaces the pessimistic guess.

Within one leg the price moves monotonically, so events fire in strict price
order: liquidation, then full-grid break, then the next safety order on the way
down; the take profit on the way up.  Every event re-prices the ones after it.

Layer boundaries
----------------
At layer ``i`` both price boundaries use ``grid_deviation_pcts[i]`` from the
explicit grid list. The favourable boundary takes profit; the adverse boundary
enters the next layer, or stops the cycle once the list is exhausted.
``take_profit_pct`` is only the fixed cash target as a share of the cycle's
starting balance. Before a base order is opened, the simulator prices the whole
planned grid and rejects the cycle if any scheduled layer would breach the
margin cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, List, Optional, Sequence


# A cycle's side.  Every price rule is written as "adverse" / "favourable"
# relative to this sign, so long and short share one implementation.
LONG = 1
SHORT = -1
SIDE_NAME = {LONG: "long", SHORT: "short"}


# ============================================================
# Parameters
# ============================================================
@dataclass(frozen=True)
class MartingaleParams:
    """Decision parameters of one martingale cycle."""

    grid_deviation_pcts: Sequence[float] = (
        0.01, 0.015, 0.0225, 0.03375, 0.050625,
        0.0759375, 0.11390625, 0.170859375, 0.2562890625, 0.38443359375,
    )
    take_profit_pct: float = 0.01         # fixed net profit / cycle start balance

    # --- full-grid break once the ladder can no longer average down ---
    stop_at_full_layers: bool = True      # exit at the next grid price

    # --- direction ---
    initial_direction: str = "long"          # "long" or "short"
    # Flip the side of the next cycle every time one breaks.  A long
    # martingale only loses in a downtrend, so the question this answers is
    # whether following the break is better than starting the same ladder again.
    reverse_after_grid_break: bool = False

    # --- pause after a losing cycle ---
    # Bars to sit out after a loss.  0 keeps the default behaviour: the cycle
    # that closed cannot re-open inside its own bar, so the next base order
    # fills at the very next bar's open.  N pushes that N bars further out.
    loss_cooldown_bars: int = 0
    # Optional maximum age of one cycle, in bars.  The backtest layer converts
    # the user-facing day value to bars after it knows the series frequency.
    max_cycle_bars: Optional[int] = None

    def __post_init__(self):
        deviations = tuple(float(item) for item in self.grid_deviation_pcts)
        if not deviations:
            raise ValueError("grid_deviation_pcts must not be empty")
        if any(not 0.0 < item < 1.0 for item in deviations):
            raise ValueError("every grid_deviation_pcts item must be in (0, 1)")
        object.__setattr__(self, "grid_deviation_pcts", deviations)
        if not 0.0 < self.take_profit_pct < 1.0:
            raise ValueError("take_profit_pct must be in (0, 1)")
        if self.loss_cooldown_bars < 0:
            raise ValueError("loss_cooldown_bars must be >= 0")
        if self.max_cycle_bars is not None and self.max_cycle_bars < 1:
            raise ValueError("max_cycle_bars must be >= 1 or None")
        if self.initial_direction not in ("long", "short"):
            raise ValueError("initial_direction must be 'long' or 'short'")

    @property
    def initial_side(self) -> int:
        return LONG if self.initial_direction == "long" else SHORT

    @property
    def layer_count(self) -> int:
        return len(self.grid_deviation_pcts)


@dataclass(frozen=True)
class AccountParams:
    """Broker side: fees, leverage and the margin model.

    All ``*_pct`` fee/margin fields are percent, i.e. ``0.05`` means 0.05%.
    Ladder and take-profit orders rest on the book (maker); the base order,
    grid break and forced liquidation cross the spread (taker).
    """

    initial_equity: float = 10_000.0
    leverage: float = 10.0
    taker_fee_pct: float = 0.05
    maker_fee_pct: float = 0.02
    maintenance_margin_pct: float = 0.5   # of position notional
    margin_usage_cap_pct: float = 0.80    # refuse orders past this share of equity
    liquidation_penalty_pct: float = 0.0  # extra fee charged on a forced close
    ruin_equity_pct: float = 0.10         # ruin when balance <= peak balance * this

    def __post_init__(self):
        if self.initial_equity <= 0.0:
            raise ValueError("initial_equity must be positive")
        if self.leverage < 1.0:
            raise ValueError("leverage must be >= 1")
        if not 0.0 < self.margin_usage_cap_pct <= 1.0:
            raise ValueError("margin_usage_cap_pct must be in (0, 1]")
        if not 0.0 <= self.maintenance_margin_pct < 100.0:
            raise ValueError("maintenance_margin_pct must be in [0, 100)")
        for name in ("taker_fee_pct", "maker_fee_pct", "liquidation_penalty_pct"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0")

    @property
    def taker_rate(self) -> float:
        return self.taker_fee_pct / 100.0

    @property
    def maker_rate(self) -> float:
        return self.maker_fee_pct / 100.0

    @property
    def maintenance_rate(self) -> float:
        return self.maintenance_margin_pct / 100.0

    @property
    def liquidation_penalty_rate(self) -> float:
        return self.liquidation_penalty_pct / 100.0


# ============================================================
# Mutable state
# ============================================================
@dataclass
class Cycle:
    """The open martingale cycle: one averaged long position."""

    side: int = LONG             # LONG or SHORT, fixed for the whole cycle
    layers: int = 0
    qty: float = 0.0             # always positive: the side carries the sign
    cost: float = 0.0            # sum of qty * fill price, the entry notional
    entry_fees: float = 0.0      # already deducted from AccountState.equity
    last_entry_price: float = 0.0
    base_notional: float = 0.0
    start_balance: float = 0.0   # account balance when the base order filled
    start_bar: int = -1
    start_time: Any = None
    fills_in_bar: int = 0
    max_fills_in_bar: int = 0
    planned_notionals: List[float] = field(default_factory=list)
    planned_layers: List[dict] = field(default_factory=list)
    fills: List[dict] = field(default_factory=list)  # one record per layer filled
    min_price_seen: float = float("inf")             # worst excursion while open
    max_price_seen: float = 0.0

    def clone(self) -> "Cycle":
        # ``fills`` must be copied: the two candidate paths of one bar branch
        # from the same cycle and would otherwise append into a shared list.
        clone = replace(self)
        clone.fills = [dict(row) for row in self.fills]
        clone.planned_notionals = list(self.planned_notionals)
        clone.planned_layers = [dict(row) for row in self.planned_layers]
        return clone


@dataclass
class AccountState:
    """Wallet balance plus the open cycle. Everything a path needs to branch on."""

    equity: float                # realized balance, all paid fees included
    cycle: Cycle = field(default_factory=Cycle)
    bankrupt: bool = False
    trades: List[dict] = field(default_factory=list)
    events: int = 0              # fills + closes, used to detect a quiet bar
    unfilled_layers: int = 0     # layers a margin cap refused
    up_first_worse_bars: int = 0
    dual_path_bars: int = 0
    ruin_threshold_hit: bool = False
    grid_infeasible: bool = False
    min_equity_seen: float = float("inf")
    max_equity_seen: float = 0.0
    max_balance_seen: float = 0.0
    max_margin_usage_seen: float = 0.0
    max_drawdown_seen: float = 0.0
    failure_equity: Optional[float] = None
    failure_price: Optional[float] = None
    # No base order on or before this bar: set when a cycle closes at a loss.
    cooldown_until_bar: int = -1
    cooldown_blocked_bars: int = 0
    # Bars where the base order was allowed by the cooldown but refused by the
    # caller's volatility gate.
    volatility_blocked_bars: int = 0
    next_side: int = LONG        # side the next base order will take
    reversals: int = 0

    @property
    def failed(self) -> bool:
        return self.bankrupt or self.ruin_threshold_hit or self.grid_infeasible

    def clone(self) -> "AccountState":
        return AccountState(
            equity=self.equity,
            cycle=self.cycle.clone(),
            bankrupt=self.bankrupt,
            trades=list(self.trades),   # records are never mutated after append
            events=self.events,
            unfilled_layers=self.unfilled_layers,
            up_first_worse_bars=self.up_first_worse_bars,
            dual_path_bars=self.dual_path_bars,
            ruin_threshold_hit=self.ruin_threshold_hit,
            grid_infeasible=self.grid_infeasible,
            min_equity_seen=self.min_equity_seen,
            max_equity_seen=self.max_equity_seen,
            max_balance_seen=self.max_balance_seen,
            max_margin_usage_seen=self.max_margin_usage_seen,
            max_drawdown_seen=self.max_drawdown_seen,
            failure_equity=self.failure_equity,
            failure_price=self.failure_price,
            cooldown_until_bar=self.cooldown_until_bar,
            cooldown_blocked_bars=self.cooldown_blocked_bars,
            volatility_blocked_bars=self.volatility_blocked_bars,
            next_side=self.next_side,
            reversals=self.reversals,
        )


# ============================================================
# Simulator
# ============================================================
class MartingaleSimulator:
    """Replays completed OHLC bars against the martingale ledger."""

    def __init__(self, params: MartingaleParams, account: AccountParams):
        self.params = params
        self.account = account
        maker = account.maker_rate
        taker = account.taker_rate
        boundary_deviations = params.grid_deviation_pcts
        minimum_edge = min(
            deviation - (1.0 + side * deviation) * maker - taker
            for deviation in boundary_deviations
            for side in (LONG, SHORT)
        )
        if minimum_edge <= 0.0:
            raise ValueError(
                "every grid_deviation_pcts item must exceed round-trip fees for "
                "target-derived sizing"
            )
        self._bar_index = -1
        self._bar_time = None

    # ---------- construction ----------
    def new_state(self) -> AccountState:
        initial = self.account.initial_equity
        return AccountState(
            equity=initial,
            min_equity_seen=initial,
            max_equity_seen=initial,
            max_balance_seen=initial,
            next_side=self.params.initial_side,
            cycle=Cycle(side=self.params.initial_side),
        )

    # ---------- read-only views ----------
    def equity_at(self, state: AccountState, price: float) -> float:
        """Mark-to-market equity: balance plus unrealized PnL."""
        cycle = state.cycle
        if cycle.qty <= 0.0:
            return state.equity
        return state.equity + cycle.side * (cycle.qty * price - cycle.cost)

    def margin_usage(self, state: AccountState, price: float) -> float:
        """Initial margin locked by the position, as a share of equity."""
        equity = self.equity_at(state, price)
        if equity <= 0.0 or state.cycle.qty <= 0.0:
            return 0.0
        return (state.cycle.cost / self.account.leverage) / equity

    def liquidation_price(self, state: AccountState) -> Optional[float]:
        """Price where mark-to-market equity falls to the maintenance margin."""
        cycle = state.cycle
        if cycle.qty <= 0.0:
            return None
        maintenance = self.account.maintenance_rate
        if cycle.side == LONG:
            denominator = cycle.qty * (1.0 - maintenance)
            if denominator <= 0.0:
                return None
            price = (cycle.cost - state.equity) / denominator
        else:
            # A short is liquidated on the way up, and can never be squeezed
            # out at a non-positive price.
            price = (cycle.cost + state.equity) / (cycle.qty * (1.0 + maintenance))
        return price if price > 0.0 else None


    def _observe(self, state: AccountState, price: float) -> float:
        """Record intrabar MTM equity/margin.

        Ruin is deliberately not checked on floating loss. It is checked only
        after a full-grid break has been realized into account equity.
        """
        cycle = state.cycle
        if cycle.qty > 0.0:
            cycle.min_price_seen = min(cycle.min_price_seen, price)
            cycle.max_price_seen = max(cycle.max_price_seen, price)
        equity = self.equity_at(state, price)
        state.min_equity_seen = min(state.min_equity_seen, equity)
        state.max_equity_seen = max(state.max_equity_seen, equity)
        state.max_balance_seen = max(state.max_balance_seen, state.equity)
        if state.max_equity_seen > 0.0:
            state.max_drawdown_seen = max(
                state.max_drawdown_seen,
                (state.max_equity_seen - equity) / state.max_equity_seen,
            )
        state.max_margin_usage_seen = max(
            state.max_margin_usage_seen, self.margin_usage(state, price)
        )
        return equity

    # ---------- ladder geometry ----------
    def _layer_deviation(self, state: AccountState) -> Optional[float]:
        """Current layer adverse distance from the latest fill."""
        cycle = state.cycle
        if cycle.layers <= 0 or cycle.last_entry_price <= 0.0:
            return None
        deviation = self.params.grid_deviation_pcts[cycle.layers - 1]
        return deviation if deviation < 1.0 else None

    def _grid_price(self, state: AccountState) -> Optional[float]:
        """Current layer adverse boundary around the latest fill."""
        cycle = state.cycle
        if 0 < cycle.layers <= len(cycle.planned_layers):
            row = cycle.planned_layers[cycle.layers - 1]
            return row.get("execution_next_adverse_price", row["next_adverse_price"])
        deviation = self._layer_deviation(state)
        if deviation is None:
            return None
        return cycle.last_entry_price * (1.0 - cycle.side * deviation)

    def next_layer_price(self, state: AccountState) -> Optional[float]:
        cycle = state.cycle
        if cycle.layers == 0:
            return None
        if cycle.layers >= self.params.layer_count:
            return None
        return self._grid_price(state)

    def _full_layer_stop_armed(self, state: AccountState) -> bool:
        """True once the cycle can no longer average down."""
        if not self.params.stop_at_full_layers:
            return False
        return state.cycle.layers >= self.params.layer_count

    def grid_break_price(self, state: AccountState) -> Optional[float]:
        """Full-layer adverse boundary."""
        if state.cycle.qty <= 0.0 or not self._full_layer_stop_armed(state):
            return None
        return self._grid_price(state)

    def take_profit_price(self, state: AccountState) -> Optional[float]:
        """Current layer favourable boundary around the latest fill."""
        cycle = state.cycle
        if cycle.qty <= 0.0:
            return None
        if 0 < cycle.layers <= len(cycle.planned_layers):
            row = cycle.planned_layers[cycle.layers - 1]
            return row.get("execution_take_profit_price", row["take_profit_price"])
        deviation = self._layer_deviation(state)
        if deviation is None:
            return None
        return cycle.last_entry_price * (1.0 + cycle.side * deviation)

    # ---------- target-derived sizing ----------
    def profit_target(self, state: AccountState) -> float:
        """Fixed cash target for the cycle: start balance times take-profit pct."""
        cycle = state.cycle
        basis = cycle.start_balance if cycle.qty > 0.0 else state.equity
        return basis * self.params.take_profit_pct

    def _target_notional_from_position(
        self,
        *,
        side: int,
        price: float,
        layer_index: int,
        target_cash: float,
        qty: float,
        cost: float,
        entry_fees: float,
        is_base: bool,
    ) -> float:
        """Additional notional needed to net target_cash at this layer TP."""
        deviation = self.params.grid_deviation_pcts[layer_index]
        if deviation >= 1.0:
            return 0.0

        exit_ratio = 1.0 + side * deviation
        exit_price = price * exit_ratio
        maker = self.account.maker_rate
        entry_rate = self.account.taker_rate if is_base else maker
        incremental_return = (
            side * (exit_ratio - 1.0)
            - exit_ratio * maker
            - entry_rate
        )
        if incremental_return <= 0.0:
            return 0.0

        existing_net_at_exit = (
            side * (qty * exit_price - cost)
            - entry_fees
            - qty * exit_price * maker
        )
        required = (target_cash - existing_net_at_exit) / incremental_return
        return max(0.0, required)

    def _affordable_position(
        self,
        *,
        equity: float,
        side: int,
        qty: float,
        cost: float,
        notional: float,
        price: float,
        is_base: bool,
    ) -> bool:
        """Reject an order that would push initial margin past the cap."""
        if notional <= 0.0 or price <= 0.0:
            return False
        fee = notional * (self.account.taker_rate if is_base else self.account.maker_rate)
        equity_mtm = equity + side * (qty * price - cost)
        equity_after = equity_mtm - fee
        if equity_after <= 0.0:
            return False
        initial_margin = (cost + notional) / self.account.leverage
        return initial_margin <= equity_after * self.account.margin_usage_cap_pct + 1e-12

    def _plan_cycle(self, state: AccountState, price: float) -> Optional[List[dict]]:
        """Plan and validate every scheduled layer before the base order opens."""
        side = state.next_side
        equity = state.equity
        target_cash = equity * self.params.take_profit_pct
        qty = 0.0
        cost = 0.0
        entry_fees = 0.0
        layer_price = price
        base_price = price
        rows: List[dict] = []

        for layer_index, deviation in enumerate(self.params.grid_deviation_pcts):
            if layer_index > 0:
                adverse = self.params.grid_deviation_pcts[layer_index - 1]
                if adverse >= 1.0:
                    return None
                layer_price = layer_price * (1.0 - side * adverse)
                if layer_price <= 0.0:
                    return None

            is_base = layer_index == 0
            notional = self._target_notional_from_position(
                side=side,
                price=layer_price,
                layer_index=layer_index,
                target_cash=target_cash,
                qty=qty,
                cost=cost,
                entry_fees=entry_fees,
                is_base=is_base,
            )
            if notional <= 0.0 or not self._affordable_position(
                equity=equity,
                side=side,
                qty=qty,
                cost=cost,
                notional=notional,
                price=layer_price,
                is_base=is_base,
            ):
                return None

            fee = notional * (
                self.account.taker_rate if is_base else self.account.maker_rate
            )
            layer_qty = notional / layer_price
            take_profit_price = layer_price * (1.0 + side * deviation)
            next_adverse_price = layer_price * (1.0 - side * deviation)
            position_qty_after = qty + layer_qty
            cumulative_cost_after = cost + notional
            cumulative_entry_fees_after = entry_fees + fee
            exit_notional_at_tp = position_qty_after * take_profit_price
            gross_pnl_at_tp = side * (exit_notional_at_tp - cumulative_cost_after)
            exit_fee_at_tp = exit_notional_at_tp * self.account.maker_rate
            net_pnl_at_tp = (
                gross_pnl_at_tp
                - cumulative_entry_fees_after
                - exit_fee_at_tp
            )
            layer_exit_notional_at_tp = layer_qty * take_profit_price
            layer_gross_pnl_at_tp = side * (
                layer_exit_notional_at_tp - notional
            )
            layer_exit_fee_at_tp = layer_exit_notional_at_tp * self.account.maker_rate
            layer_net_pnl_at_tp = (
                layer_gross_pnl_at_tp - fee - layer_exit_fee_at_tp
            )
            rows.append({
                "layer": layer_index + 1,
                "role": "base" if is_base else "safety",
                "entry_price": layer_price,
                "entry_price_pct_from_base": side * (layer_price / base_price - 1.0),
                "take_profit_price": take_profit_price,
                "take_profit_pct_from_entry": side * (
                    take_profit_price / layer_price - 1.0
                ),
                "next_adverse_price": next_adverse_price,
                "next_adverse_pct_from_entry": side * (
                    next_adverse_price / layer_price - 1.0
                ),
                "grid_break_price": (
                    next_adverse_price
                    if self.params.stop_at_full_layers
                    and layer_index + 1 >= self.params.layer_count
                    else None
                ),
                "notional": notional,
                "qty": layer_qty,
                "fee": fee,
                "cumulative_qty_after": position_qty_after,
                "cumulative_notional_after": cumulative_cost_after,
                "cumulative_entry_fees_after": cumulative_entry_fees_after,
                "avg_entry_price_after": cumulative_cost_after / position_qty_after,
                "cycle_gross_pnl_at_tp": gross_pnl_at_tp,
                "cycle_exit_fee_at_tp": exit_fee_at_tp,
                "cycle_net_pnl_at_tp": net_pnl_at_tp,
                "layer_gross_pnl_at_tp": layer_gross_pnl_at_tp,
                "layer_entry_fee": fee,
                "layer_exit_fee_at_tp": layer_exit_fee_at_tp,
                "layer_net_pnl_at_tp": layer_net_pnl_at_tp,
            })
            equity -= fee
            qty = position_qty_after
            cost = cumulative_cost_after
            entry_fees = cumulative_entry_fees_after

        return rows

    def target_notional(self, state: AccountState, price: float) -> float:
        """Additional notional needed to net the cycle target at the new layer TP."""
        cycle = state.cycle
        side = cycle.side if cycle.qty > 0.0 else state.next_side
        return self._target_notional_from_position(
            side=side,
            price=price,
            layer_index=cycle.layers,
            target_cash=self.profit_target(state),
            qty=cycle.qty,
            cost=cycle.cost,
            entry_fees=cycle.entry_fees,
            is_base=cycle.qty <= 0.0,
        )

    def grid_diagnostics(self, price: float) -> dict[str, dict]:
        """Explain every full-grid preflight decision for long and short.

        This intentionally mirrors ``_plan_cycle`` but retains the intermediate
        values that method discards, so configuration errors can identify the
        exact layer and constraint instead of returning a generic ``None``.
        """
        if price <= 0.0:
            raise ValueError("startup grid diagnostic price must be positive")

        results: dict[str, dict] = {}
        for side in (LONG, SHORT):
            equity = self.account.initial_equity
            target_cash = equity * self.params.take_profit_pct
            qty = 0.0
            cost = 0.0
            entry_fees = 0.0
            layer_price = price
            rows = []
            failure_reason = None

            for layer_index, deviation in enumerate(self.params.grid_deviation_pcts):
                layer = layer_index + 1
                adverse = None
                if layer_index > 0:
                    adverse = self.params.grid_deviation_pcts[layer_index - 1]
                    if adverse >= 1.0:
                        failure_reason = "adverse_step_ge_100pct"
                        rows.append({
                            "layer": layer,
                            "status": "failed",
                            "failure_reason": failure_reason,
                            "adverse_step_pct": adverse,
                        })
                        break
                    layer_price = layer_price * (1.0 - side * adverse)
                    if layer_price <= 0.0:
                        failure_reason = "non_positive_layer_price"
                        rows.append({
                            "layer": layer,
                            "status": "failed",
                            "failure_reason": failure_reason,
                            "adverse_step_pct": adverse,
                            "entry_price": layer_price,
                        })
                        break

                is_base = layer_index == 0
                entry_rate = (
                    self.account.taker_rate if is_base else self.account.maker_rate
                )
                row = {
                    "layer": layer,
                    "status": "ok",
                    "failure_reason": None,
                    "adverse_step_pct": adverse,
                    "take_profit_deviation_pct": deviation,
                    "entry_price": layer_price,
                    "target_cash": target_cash,
                    "equity_ledger_before": equity,
                    "position_qty_before": qty,
                    "cumulative_notional_before": cost,
                    "entry_fees_before": entry_fees,
                }

                if deviation >= 1.0:
                    failure_reason = "take_profit_deviation_ge_100pct"
                    row.update(status="failed", failure_reason=failure_reason)
                    rows.append(row)
                    break

                exit_ratio = 1.0 + side * deviation
                exit_price = layer_price * exit_ratio
                incremental_return = (
                    side * (exit_ratio - 1.0)
                    - exit_ratio * self.account.maker_rate
                    - entry_rate
                )
                existing_net_at_tp = (
                    side * (qty * exit_price - cost)
                    - entry_fees
                    - qty * exit_price * self.account.maker_rate
                )
                raw_required_notional = (
                    (target_cash - existing_net_at_tp) / incremental_return
                    if incremental_return > 0.0 else None
                )
                required_notional = (
                    max(0.0, raw_required_notional)
                    if raw_required_notional is not None else 0.0
                )
                order_fee = required_notional * entry_rate
                equity_mtm_before_order = equity + side * (qty * layer_price - cost)
                equity_after_order = equity_mtm_before_order - order_fee
                qty_after_order = qty + required_notional / layer_price
                cost_after_order = cost + required_notional
                entry_fees_after_order = entry_fees + order_fee
                initial_margin_after = (
                    cost_after_order / self.account.leverage
                )
                margin_cap_amount = (
                    equity_after_order * self.account.margin_usage_cap_pct
                )
                next_adverse_price = layer_price * (1.0 - side * deviation)
                grid_break_price = (
                    next_adverse_price
                    if self.params.stop_at_full_layers
                    and layer >= self.params.layer_count
                    else None
                )
                grid_break_exit_notional = (
                    qty_after_order * grid_break_price
                    if grid_break_price is not None else None
                )
                grid_break_exit_fee = (
                    grid_break_exit_notional * self.account.taker_rate
                    if grid_break_exit_notional is not None else None
                )
                grid_break_gross_pnl = (
                    side * (grid_break_exit_notional - cost_after_order)
                    if grid_break_exit_notional is not None else None
                )
                grid_break_net_pnl = (
                    grid_break_gross_pnl
                    - entry_fees_after_order
                    - grid_break_exit_fee
                    if grid_break_gross_pnl is not None else None
                )
                adverse_gross_pnl = side * (
                    qty_after_order * next_adverse_price - cost_after_order
                )
                adverse_equity_mtm = (
                    self.account.initial_equity
                    + adverse_gross_pnl
                    - entry_fees_after_order
                )
                adverse_loss_pct = (
                    (self.account.initial_equity - adverse_equity_mtm)
                    / self.account.initial_equity
                )
                row.update({
                    "take_profit_price": exit_price,
                    "next_adverse_price": next_adverse_price,
                    "grid_break_price": grid_break_price,
                    "incremental_return": incremental_return,
                    "existing_net_at_take_profit": existing_net_at_tp,
                    "raw_required_notional": raw_required_notional,
                    "required_notional": required_notional,
                    "order_fee": order_fee,
                    "equity_mtm_before_order": equity_mtm_before_order,
                    "equity_after_order": equity_after_order,
                    "cumulative_qty_after": qty_after_order,
                    "cumulative_notional_after": cost_after_order,
                    "cumulative_entry_fees_after": entry_fees_after_order,
                    "initial_margin_after": initial_margin_after,
                    "margin_cap_amount": margin_cap_amount,
                    "grid_break_exit_notional": grid_break_exit_notional,
                    "grid_break_exit_fee": grid_break_exit_fee,
                    "grid_break_gross_pnl": grid_break_gross_pnl,
                    "grid_break_net_pnl": grid_break_net_pnl,
                    "adverse_equity_mtm": adverse_equity_mtm,
                    "adverse_loss_pct": adverse_loss_pct,
                    "margin_usage_pct": (
                        initial_margin_after / equity_after_order
                        if equity_after_order > 0.0 else None
                    ),
                })

                if incremental_return <= 0.0:
                    failure_reason = "non_positive_incremental_return"
                elif required_notional <= 0.0:
                    failure_reason = (
                        "target_already_met_without_new_order"
                        if existing_net_at_tp >= target_cash
                        else "non_positive_required_notional"
                    )
                elif equity_after_order <= 0.0:
                    failure_reason = "non_positive_equity_after_order"
                elif initial_margin_after > margin_cap_amount + 1e-12:
                    failure_reason = "margin_cap_exceeded"
                elif (
                    grid_break_net_pnl is not None
                    and self.account.initial_equity + grid_break_net_pnl <= 0.0
                ):
                    failure_reason = "grid_break_equity_non_positive"

                if failure_reason is not None:
                    row.update(status="failed", failure_reason=failure_reason)
                    rows.append(row)
                    break

                rows.append(row)
                equity -= order_fee
                qty = qty_after_order
                cost = cost_after_order
                entry_fees = entry_fees_after_order

            name = SIDE_NAME[side]
            results[name] = {
                "side": name,
                "executable": failure_reason is None and len(rows) == self.params.layer_count,
                "failure_layer": (
                    None if failure_reason is None else rows[-1]["layer"]
                ),
                "failure_reason": failure_reason,
                "target_cash": target_cash,
                "layers": rows,
            }
        return results

    def validate_initial_grid(
        self, price: float, diagnostics: Optional[dict[str, dict]] = None
    ):
        """Fail fast if the configured full grid cannot be executed at startup."""
        if price <= 0.0:
            raise ValueError("startup grid validation price must be positive")
        diagnostics = (
            diagnostics if diagnostics is not None else self.grid_diagnostics(price)
        )
        failed = [result for result in diagnostics.values() if not result["executable"]]
        if failed:
            details = "; ".join(
                f"{result['side']} layer {result['failure_layer']}: "
                f"{result['failure_reason']}"
                for result in failed
            )
            raise ValueError(
                "configured martingale grid is not executable at startup "
                f"at price {price:.12g}; {details}. See GRID PREFLIGHT log lines "
                "for prices, notionals, target PnL, fees and margin constraints"
            )

    # ---------- fills ----------
    def _mark_layer_boundary(
        self,
        state: AccountState,
        *,
        kind: str,
        price: float,
        next_layer: Optional[int] = None,
    ):
        """Record the first boundary reached after the currently active layer.

        This is intentionally separate from the cycle's final close.  For layer
        ``i`` the first boundary is either its own take-profit or the adverse
        boundary that opens ``i + 1``.  Only the final layer's adverse boundary is
        the cycle grid break.
        """
        if not state.cycle.fills:
            return
        fill = state.cycle.fills[-1]
        if fill.get("boundary_kind") is not None:
            return
        fill.update({
            "boundary_kind": kind,
            "boundary_bar": self._bar_index,
            "boundary_time": self._bar_time,
            "boundary_price": price,
            "boundary_bars_held": self._bar_index - fill["bar"],
            "boundary_next_layer": next_layer,
        })

    def _fill(
        self,
        state: AccountState,
        price: float,
        notional: float,
        is_base: bool,
        planned: Optional[dict] = None,
    ):
        cycle = state.cycle
        fee = notional * (
            self.account.taker_rate if is_base else self.account.maker_rate
        )
        if is_base:
            cycle.side = state.next_side
            cycle.base_notional = notional
            cycle.start_balance = state.equity
            cycle.start_bar = self._bar_index
            cycle.start_time = self._bar_time
            cycle.max_fills_in_bar = 0
        qty = notional / price
        cycle.qty += qty
        cycle.cost += notional
        cycle.entry_fees += fee
        cycle.last_entry_price = price
        cycle.layers += 1
        layer_index = cycle.layers - 1
        deviation = (
            self.params.grid_deviation_pcts[layer_index]
        )
        execution_take_profit_price = price * (1.0 + cycle.side * deviation)
        execution_next_adverse_price = price * (1.0 - cycle.side * deviation)
        execution_exit_notional_at_tp = (
            cycle.qty * execution_take_profit_price
        )
        execution_gross_pnl_at_tp = cycle.side * (
            execution_exit_notional_at_tp - cycle.cost
        )
        execution_exit_fee_at_tp = (
            execution_exit_notional_at_tp * self.account.maker_rate
        )
        execution_net_pnl_at_tp = (
            execution_gross_pnl_at_tp
            - cycle.entry_fees
            - execution_exit_fee_at_tp
        )
        layer_exit_notional_at_tp = qty * execution_take_profit_price
        execution_layer_gross_pnl_at_tp = cycle.side * (
            layer_exit_notional_at_tp - notional
        )
        execution_layer_exit_fee_at_tp = (
            layer_exit_notional_at_tp * self.account.maker_rate
        )
        execution_layer_net_pnl_at_tp = (
            execution_layer_gross_pnl_at_tp
            - fee
            - execution_layer_exit_fee_at_tp
        )
        if layer_index < len(cycle.planned_layers):
            cycle.planned_layers[layer_index].update({
                "execution_entry_price": price,
                "execution_take_profit_price": execution_take_profit_price,
                "execution_next_adverse_price": execution_next_adverse_price,
                "execution_grid_break_price": (
                    execution_next_adverse_price
                    if self.params.stop_at_full_layers
                    and cycle.layers >= self.params.layer_count
                    else None
                ),
                "execution_cycle_gross_pnl_at_tp": execution_gross_pnl_at_tp,
                "execution_cycle_exit_fee_at_tp": execution_exit_fee_at_tp,
                "execution_cycle_net_pnl_at_tp": execution_net_pnl_at_tp,
                "execution_layer_gross_pnl_at_tp": (
                    execution_layer_gross_pnl_at_tp
                ),
                "execution_layer_exit_fee_at_tp": execution_layer_exit_fee_at_tp,
                "execution_layer_net_pnl_at_tp": execution_layer_net_pnl_at_tp,
            })
        cycle.fills_in_bar += 1
        cycle.max_fills_in_bar = max(cycle.max_fills_in_bar, cycle.fills_in_bar)
        state.equity -= fee
        state.events += 1
        fill = {
            "layer": cycle.layers,
            "bar": self._bar_index,
            "time": self._bar_time,
            "price": price,
            "qty": qty,
            "notional": notional,
            "fee": fee,
            "role": "base" if is_base else "safety",
            "avg_entry_price_after": cycle.cost / cycle.qty,
            "position_qty_after": cycle.qty,
            "balance_after": state.equity,
        }
        if planned:
            fill.update({
                "planned_entry_price": planned["entry_price"],
                "planned_entry_price_pct_from_base": planned[
                    "entry_price_pct_from_base"
                ],
                "planned_take_profit_price": planned["take_profit_price"],
                "planned_take_profit_pct_from_entry": planned[
                    "take_profit_pct_from_entry"
                ],
                "planned_next_adverse_price": planned["next_adverse_price"],
                "planned_next_adverse_pct_from_entry": planned[
                    "next_adverse_pct_from_entry"
                ],
                "planned_grid_break_price": planned["grid_break_price"],
                "planned_layer_gross_pnl_at_tp": planned[
                    "layer_gross_pnl_at_tp"
                ],
                "planned_layer_entry_fee": planned["layer_entry_fee"],
                "planned_layer_exit_fee_at_tp": planned[
                    "layer_exit_fee_at_tp"
                ],
                "planned_layer_net_pnl_at_tp": planned["layer_net_pnl_at_tp"],
                "planned_cycle_gross_pnl_at_tp": planned[
                    "cycle_gross_pnl_at_tp"
                ],
                "planned_cycle_exit_fee_at_tp": planned["cycle_exit_fee_at_tp"],
                "planned_cycle_net_pnl_at_tp": planned["cycle_net_pnl_at_tp"],
            })
        fill.update({
            "execution_take_profit_price": execution_take_profit_price,
            "execution_next_adverse_price": execution_next_adverse_price,
            "execution_grid_break_price": (
                execution_next_adverse_price
                if self.params.stop_at_full_layers
                and cycle.layers >= self.params.layer_count
                else None
            ),
            "execution_layer_gross_pnl_at_tp": execution_layer_gross_pnl_at_tp,
            "execution_layer_exit_fee_at_tp": execution_layer_exit_fee_at_tp,
            "execution_layer_net_pnl_at_tp": execution_layer_net_pnl_at_tp,
            "execution_cycle_gross_pnl_at_tp": execution_gross_pnl_at_tp,
            "execution_cycle_exit_fee_at_tp": execution_exit_fee_at_tp,
            "execution_cycle_net_pnl_at_tp": execution_net_pnl_at_tp,
        })
        cycle.fills.append(fill)
        self._observe(state, price)

    def _mark_grid_infeasible(self, state: AccountState, price: float, layers: int):
        self._mark_layer_boundary(state, kind="grid_infeasible", price=price)
        state.grid_infeasible = True
        state.unfilled_layers += max(1, layers)
        state.failure_equity = self.equity_at(state, price)
        state.failure_price = price

    def _open_base(self, state: AccountState, price: float) -> bool:
        planned = self._plan_cycle(state, price)
        if not planned:
            self._mark_grid_infeasible(state, price, self.params.layer_count)
            return False
        state.cycle.planned_layers = planned
        state.cycle.planned_notionals = [row["notional"] for row in planned]
        self._fill(state, price, planned[0]["notional"], is_base=True, planned=planned[0])
        return True

    def _add_layer(self, state: AccountState, price: float) -> bool:
        cycle = state.cycle
        if cycle.layers >= len(cycle.planned_layers):
            raise RuntimeError(
                "planned grid exhausted before full-grid break; "
                "this indicates inconsistent ladder state"
            )
        planned = cycle.planned_layers[cycle.layers]
        notional = planned["notional"]
        self._mark_layer_boundary(
            state,
            kind="adverse",
            price=price,
            next_layer=cycle.layers + 1,
        )
        self._fill(state, price, notional, is_base=False, planned=planned)
        return True

    # ---------- exits ----------
    def _close(self, state: AccountState, price: float, reason: str, fee_rate: float):
        cycle = state.cycle
        exit_notional = cycle.qty * price
        exit_fee = exit_notional * fee_rate
        gross_pnl = cycle.side * (exit_notional - cycle.cost)
        avg_entry_price = cycle.cost / cycle.qty
        balance_before = state.equity
        state.equity += gross_pnl - exit_fee
        shortfall = 0.0
        if state.equity < 0.0:
            # The exchange absorbs the gap; a wallet cannot go negative.
            shortfall = -state.equity
            state.equity = 0.0
        net_pnl = gross_pnl - exit_fee - cycle.entry_fees
        for fill in cycle.fills:
            layer_exit_notional = fill["qty"] * price
            layer_gross_pnl = cycle.side * (
                layer_exit_notional - fill["notional"]
            )
            layer_exit_fee = layer_exit_notional * fee_rate
            fill.update({
                "exit_bar": self._bar_index,
                "exit_time": self._bar_time,
                "exit_price": price,
                "exit_reason": reason,
                "exit_fee_rate": fee_rate,
                # Holding time in completed bars. A layer filled and closed on
                # the same candle has zero elapsed bars, which is the least
                # surprising definition for duration statistics.
                "layer_bars_held": self._bar_index - fill["bar"],
                "layer_exit_notional": layer_exit_notional,
                "layer_gross_pnl": layer_gross_pnl,
                "layer_entry_fee": fill["fee"],
                "layer_exit_fee": layer_exit_fee,
                "layer_fees": fill["fee"] + layer_exit_fee,
                "layer_net_pnl": layer_gross_pnl - fill["fee"] - layer_exit_fee,
                "layer_pnl_pct": (
                    (layer_gross_pnl - fill["fee"] - layer_exit_fee)
                    / fill["notional"]
                    if fill["notional"] else 0.0
                ),
                "exit_pct_from_layer_entry": cycle.side * (
                    price / fill["price"] - 1.0
                ),
            })
        self._mark_layer_boundary(state, kind=reason, price=price)
        # Adverse is down for a long and up for a short.
        if cycle.side == LONG:
            worst_price = cycle.min_price_seen if cycle.fills else price
            best_price = cycle.max_price_seen
        else:
            worst_price = cycle.max_price_seen if cycle.fills else price
            best_price = cycle.min_price_seen

        state.trades.append(
            {
                "index": len(state.trades) + 1,
                "direction": SIDE_NAME[cycle.side],
                # --- time ---
                "opened_at": cycle.start_time,
                "closed_at": self._bar_time,
                "entry_bar": cycle.start_bar,
                "exit_bar": self._bar_index,
                "bars_held": self._bar_index - cycle.start_bar + 1,
                "same_bar": cycle.start_bar == self._bar_index,
                # --- price ---
                "first_entry_price": cycle.fills[0]["price"] if cycle.fills else price,
                "last_entry_price": cycle.last_entry_price,
                "avg_entry_price": avg_entry_price,
                "exit_price": price,
                "worst_price": worst_price,
                "best_price": best_price,
                # Signed by side, so negative always means "against the trade".
                "mae_pct": cycle.side * (worst_price / avg_entry_price - 1.0),
                "mfe_pct": cycle.side * (best_price / avg_entry_price - 1.0),
                "exit_pct_from_avg": cycle.side * (price / avg_entry_price - 1.0),
                # --- size ---
                "layers": cycle.layers,
                "qty": cycle.qty,
                "entry_notional": cycle.cost,
                "exit_notional": exit_notional,
                "max_fills_in_one_bar": cycle.max_fills_in_bar,
                "multiple_fills_same_bar": cycle.max_fills_in_bar > 1,
                "full_layers": cycle.layers >= self.params.layer_count,
                # --- pnl ---
                "gross_pnl": gross_pnl,
                "entry_fees": cycle.entry_fees,
                "exit_fee": exit_fee,
                "fees": cycle.entry_fees + exit_fee,
                "net_pnl": net_pnl,
                "profit_pct": net_pnl / cycle.cost if cycle.cost else 0.0,
                "profit_vs_base": (
                    net_pnl / cycle.base_notional if cycle.base_notional else 0.0
                ),
                "start_balance": cycle.start_balance,
                "profit_target": self.profit_target(state),
                "profit_vs_start_balance": (
                    net_pnl / cycle.start_balance if cycle.start_balance else 0.0
                ),
                "balance_before": balance_before,
                "balance_after": state.equity,
                "return_on_balance": (
                    net_pnl / balance_before if balance_before > 0.0 else 0.0
                ),
                "shortfall": shortfall,
                "reason": reason,
                # --- per layer detail ---
                "fills": cycle.fills,
            }
        )
        state.events += 1
        if reason == "grid_break" and self.params.reverse_after_grid_break:
            state.next_side = -cycle.side
            state.reversals += 1
        if net_pnl < 0.0:
            # Sit out the next ``loss_cooldown_bars`` bars.  At the default 0
            # this still bans re-entry inside the closing bar and no more.
            state.cooldown_until_bar = self._bar_index + self.params.loss_cooldown_bars
        if reason == "liquidation":
            state.bankrupt = True
            state.failure_equity = state.equity
            state.failure_price = price
        # A closed cycle never re-opens inside the same bar: the next base order
        # is decided on this close and can only fill at the next bar's open.
        # The reset must happen before observing, otherwise the position that
        # was just realized into the balance is counted a second time as
        # unrealized PnL and the equity extrema come out at double the loss.
        state.cycle = Cycle(side=state.next_side, fills_in_bar=cycle.fills_in_bar)
        self._observe(state, price)

    def _liquidate(self, state: AccountState, price: float):
        self._close(
            state,
            price,
            "liquidation",
            self.account.taker_rate + self.account.liquidation_penalty_rate,
        )

    # ---------- price legs ----------
    @staticmethod
    def _reached(price: float, price_now: float, to_price: float) -> bool:
        """Is ``price`` swept by a monotone move from price_now to to_price?"""
        low, high = (price_now, to_price) if price_now <= to_price else (to_price, price_now)
        tolerance = max(abs(price), abs(price_now), abs(to_price)) * 1e-12
        return low - tolerance <= price <= high + tolerance

    @staticmethod
    def _crossed_at_open(side: int, open_: float, trigger: Optional[float], *, adverse: bool) -> bool:
        """Whether the bar opens already beyond a trigger."""
        if trigger is None:
            return False
        tolerance = max(abs(open_), abs(trigger)) * 1e-12
        distance = side * (open_ - trigger)
        return distance <= tolerance if adverse else distance >= -tolerance

    def _settle_open_position(self, state: AccountState, open_: float):
        """Settle every trigger already crossed by the opening gap.

        The open has no intrabar order ambiguity. It only tells us that a
        resting order was crossed; fills still occur at their trigger/limit
        prices under the strict limit-order model.
        """
        while not state.failed and state.cycle.qty > 0.0:
            side = state.cycle.side
            liquidation = self.liquidation_price(state)
            if self._crossed_at_open(side, open_, liquidation, adverse=True):
                self._liquidate(state, liquidation)
                return

            grid_break = self.grid_break_price(state)
            if self._crossed_at_open(side, open_, grid_break, adverse=True):
                self._close(state, grid_break, "grid_break", self.account.taker_rate)
                return

            layer = self.next_layer_price(state)
            if self._crossed_at_open(side, open_, layer, adverse=True):
                if not self._add_layer(state, layer):
                    return
                continue

            take_profit = self.take_profit_price(state)
            if self._crossed_at_open(side, open_, take_profit, adverse=False):
                self._close(state, take_profit, "take_profit", self.account.maker_rate)
                return

            self._observe(state, open_)
            return

    def _walk_adverse(self, state: AccountState, from_price: float, to_price: float):
        """Move against the position: layers, stops and liquidation fire here.

        Down for a long, up for a short.  Events are handled in the order the
        price actually reaches them, and every fill re-prices the rest.
        """
        price_now = from_price
        while not state.failed and state.cycle.qty > 0.0:
            side = state.cycle.side
            liquidation = self.liquidation_price(state)
            grid_break = self.grid_break_price(state)
            layer = self.next_layer_price(state)

            # ``-side * price`` sorts by "reached first": the highest price for
            # a long, the lowest for a short.  On a tie the most adverse
            # terminal event wins over a stop, and a stop over a new layer.
            candidates = []
            for price, rank, kind in (
                (liquidation, 0, "liquidation"),
                (grid_break, 1, "grid_break"),
                (layer, 3, "layer"),
            ):
                if price is not None and self._reached(price, price_now, to_price):
                    candidates.append((-side * price, rank, price, kind))
            if not candidates:
                self._observe(state, to_price)
                return

            _, _, price, kind = min(candidates)
            # A gap never fills better than where the price already is.
            if side * (price - price_now) > 0.0:
                price = price_now
            self._observe(state, price)
            if kind == "liquidation":
                self._liquidate(state, price)
                return
            if kind == "grid_break":
                self._close(state, price, "grid_break", self.account.taker_rate)
                return
            if not self._add_layer(state, price):
                return
            price_now = price

        if not state.failed:
            self._observe(state, to_price)

    def _walk_favourable(self, state: AccountState, from_price: float, to_price: float):
        """Move in the position's favour: only the repriced take profit fires."""
        if state.failed or state.cycle.qty <= 0.0:
            return
        side = state.cycle.side
        take_profit = self.take_profit_price(state)
        tolerance = (
            0.0 if take_profit is None
            else max(abs(to_price), abs(take_profit)) * 1e-12
        )
        if take_profit is None or side * (to_price - take_profit) < -tolerance:
            self._observe(state, to_price)
            return
        fill_price = take_profit
        self._observe(state, fill_price)
        self._close(state, fill_price, "take_profit", self.account.maker_rate)

    def _replay(self, state: AccountState, start: float, waypoints) -> AccountState:
        price_now = start
        self._observe(state, start)
        for price in waypoints:
            if state.failed or state.cycle.qty <= 0.0:
                break
            travel = 1 if price > price_now else (-1 if price < price_now else 0)
            if travel == 0:
                self._observe(state, price)
            elif travel == -state.cycle.side:
                self._walk_adverse(state, price_now, price)
            else:
                self._walk_favourable(state, price_now, price)
            price_now = price
        return state

    def _close_if_cycle_timeout(self, state: AccountState, close: float) -> AccountState:
        """Force-close a still-open cycle at bar close once its age limit is hit."""
        limit = self.params.max_cycle_bars
        if (
            limit is None
            or state.failed
            or state.cycle.qty <= 0.0
            or state.cycle.start_bar < 0
        ):
            return state
        bars_held = self._bar_index - state.cycle.start_bar + 1
        if bars_held >= limit:
            self._observe(state, close)
            self._close(state, close, "time_limit", self.account.taker_rate)
        return state

    # ---------- path choice ----------
    @staticmethod
    def _outcome_signature(state: AccountState):
        """Everything a path can change; equal signatures mean no ambiguity."""
        return (
            state.failed,
            state.bankrupt,
            state.ruin_threshold_hit,
            round(state.equity, 9),
            round(state.cycle.qty, 12),
            state.cycle.layers,
            len(state.trades),
        )

    def _paths_disagree(self, first: AccountState, second: AccountState) -> bool:
        return self._outcome_signature(first) != self._outcome_signature(second)

    def _replay_intrabar(
        self, state: AccountState, intrabar: Sequence[Sequence[float]], trade_resolver=None
    ) -> Optional[AccountState]:
        """Replay the finer bars in their real order to settle a bar ambiguity.

        Returns ``None`` when a row is unusable, so the caller keeps the
        pessimistic two-path result.  A cycle closed inside the bar does not
        reopen before the next bar, matching the bar-level restart rule.
        """
        for fine_index, row in enumerate(intrabar):
            open_, high, low, close = (float(value) for value in row)
            # NaN compares unequal to itself; unusable rows fall back.
            if not all(value > 0.0 and value == value for value in (open_, high, low, close)):
                return None
            if high < max(open_, low, close) or low > min(open_, high, close):
                return None
            if state.failed or state.cycle.qty <= 0.0:
                break
            self._settle_open_position(state, open_)
            if state.failed or state.cycle.qty <= 0.0:
                break
            baseline_events = state.events
            down_first = self._replay(state.clone(), open_, (low, high, close))
            up_first = self._replay(state.clone(), open_, (high, low, close))
            if trade_resolver is not None and self._paths_disagree(down_first, up_first):
                prices = trade_resolver(fine_index, (open_, high, low, close))
                if prices is None:
                    raise ValueError(
                        f"Unresolved intrabar path at finer bar {fine_index}: "
                        "aggTrades data is missing or does not match the candle OHLC"
                    )
                state = self._replay(
                    state.clone(), float(prices[0]), tuple(float(price) for price in prices[1:])
                )
                continue
            if (
                down_first.events == baseline_events
                and up_first.events == baseline_events
                and not down_first.failed
                and not up_first.failed
            ):
                state = down_first
                continue
            if self._path_key(up_first, close) < self._path_key(down_first, close):
                up_first.up_first_worse_bars += 1
                state = up_first
            else:
                state = down_first
        return state

    def _path_key(self, state: AccountState, close: float):
        """Ascending order = worse first."""
        close_equity = (
            state.min_equity_seen if state.failed else self.equity_at(state, close)
        )
        return (
            0 if state.bankrupt else 1,
            0 if state.ruin_threshold_hit else 1,
            close_equity,
            -state.cycle.qty * close,
        )

    # ---------- public entry point ----------
    def process_bar(
        self,
        state: AccountState,
        *,
        index: int,
        time: Any,
        open_: float,
        high: float,
        low: float,
        close: float,
        intrabar: Optional[Sequence[Sequence[float]]] = None,
        trade_resolver=None,
        allow_open: bool = True,
    ) -> AccountState:
        """Replay one completed bar and return the pessimistic valid path.

        ``intrabar`` optionally supplies the finer bars that compose this bar
        (rows of ``open, high, low, close`` in chronological order).  The
        pessimistic two-path choice is exact only when the bar's internals are
        unknown; when the two candidates disagree and the finer series is
        available, the actual order resolves them instead of the worse guess.

        ``allow_open`` gates only the base order of a flat account.  An open
        cycle is always managed: a caller-supplied filter must never strand a
        position without its take profit or stop.  The caller is responsible
        for making that flag causal (no data from this bar or later).
        """
        if state.failed:
            return state
        if min(open_, high, low, close) <= 0.0:
            raise ValueError("OHLC prices must be positive")
        if high < max(open_, low, close) or low > min(open_, high, close):
            raise ValueError("inconsistent OHLC bar")

        self._bar_index = index
        self._bar_time = time
        state.cycle.fills_in_bar = 0

        # What the open alone settles is identical on both candidate paths.
        if state.cycle.qty > 0.0:
            self._settle_open_position(state, open_)
        elif index > state.cooldown_until_bar and allow_open:
            self._open_base(state, open_)
            self._observe(state, open_)
        else:
            if index > state.cooldown_until_bar:
                state.volatility_blocked_bars += 1
            else:
                state.cooldown_blocked_bars += 1
            self._observe(state, open_)

        # A cycle closed at the open cannot restart inside this bar.
        if state.failed or state.cycle.qty <= 0.0:
            return state

        baseline_events = state.events
        down_first = self._replay(state.clone(), open_, (low, high, close))
        up_first = self._replay(state.clone(), open_, (high, low, close))

        # Keep path counters only when at least one path had a trading/risk event.
        if down_first.events == baseline_events and up_first.events == baseline_events \
                and not down_first.failed and not up_first.failed:
            return self._close_if_cycle_timeout(down_first, close)

        down_first = self._close_if_cycle_timeout(down_first, close)
        up_first = self._close_if_cycle_timeout(up_first, close)

        dual_count = state.dual_path_bars + 1
        if intrabar is not None and self._paths_disagree(down_first, up_first):
            resolved = self._replay_intrabar(state.clone(), intrabar, trade_resolver)
            if resolved is not None:
                resolved = self._close_if_cycle_timeout(resolved, close)
                resolved.dual_path_bars = dual_count
                return resolved
        if trade_resolver is not None and self._paths_disagree(down_first, up_first):
            raise ValueError(
                f"Unresolved candle path at {time}: finer bars are missing or invalid"
            )
        if self._path_key(up_first, close) < self._path_key(down_first, close):
            up_first.up_first_worse_bars = state.up_first_worse_bars + 1
            up_first.dual_path_bars = dual_count
            return up_first
        down_first.up_first_worse_bars = state.up_first_worse_bars
        down_first.dual_path_bars = dual_count
        return down_first
