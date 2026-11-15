"""Financial projections of shared close-entry martingale price labels.

Rates are decimal fractions. Funding is not modeled. Resting additions and take
profits fill at their limit; market entries and grid stops use adverse slippage.
Minimal label events do not provide exact full-path accounting risk extrema.
"""

from __future__ import annotations

import multiprocessing
from dataclasses import asdict, dataclass
from typing import Optional, Protocol

import numpy as np

from data_process import common

LONG, SHORT = 1, -1
TAKE_PROFIT, GRID_STOP, UNRESOLVED = 0, 1, 2


@dataclass(frozen=True)
class ScanParams:
    initial_equity: float = 10000.0
    target_profit_pct: float = 0.01
    base_notional: Optional[float] = None
    max_nominal_leverage: float = 1000.0
    maker_fee_rate: float = 0.0
    taker_fee_rate: float = 0.0
    base_slippage_rate: float = 0.0
    stop_slippage_rate: float = 0.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("initial_equity", "target_profit_pct", "max_nominal_leverage"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.base_notional is not None and self.base_notional <= 0:
            raise ValueError("base_notional must be positive")
        for name in (
            "maker_fee_rate",
            "taker_fee_rate",
            "base_slippage_rate",
            "stop_slippage_rate",
        ):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")

    @property
    def goal(self) -> float:
        return self.initial_equity * self.target_profit_pct


class GridPolicy(Protocol):
    max_layers: int
    side: int
    params: ScanParams

    @property
    def price_params(self) -> common.MartingaleLabelParams: ...

    def layer_price(self, entry_price: float, addition: int) -> float: ...
    def stop_price(self, entry_price: float) -> float: ...
    def take_profit_price(self, entry_price: float, state: Position) -> float: ...
    def order_size(self, entry_price: float, state: Position) -> float: ...


@dataclass(frozen=True)
class FibonacciGrid:
    """K additions plus a base order; Fibonacci numbers are absolute offsets."""

    max_layers: int = 6
    step_pct: float = 0.01
    side: int = LONG
    params: ScanParams = ScanParams()
    initial_take_profit_pct: float = 0.01

    def __post_init__(self):
        if not isinstance(self.max_layers, int) or not 0 <= self.max_layers <= 8:
            raise ValueError("max_layers must be an integer between 0 and 8; addition 9 is the label cap")
        if self.stop_price(1.0) <= 0:
            raise ValueError("The K+1 Fibonacci stop must have a positive price")
        state = _base_position(1.0, self)
        if self.take_profit_price(1.0, state) <= 0:
            raise ValueError("Base take profit must be positive")
        for addition in range(self.max_layers + 1):
            price = self.layer_price(1.0, addition)
            if addition:
                if price <= 0:
                    raise ValueError("Fibonacci addition prices must remain positive")
                if self.take_profit_price(1.0, state) <= 0:
                    raise ValueError("Fibonacci take-profit prices must remain positive")
                qty = self.order_size(1.0, state)
                state.qty += qty
                state.cost += qty * price
                state.fees += qty * price * self.params.maker_fee_rate
                state.filled_layers += 1
            required = state.qty * price / self.params.initial_equity
            limit = self.params.max_nominal_leverage
            if required > limit and not np.isclose(required, limit, rtol=1e-10, atol=1e-12):
                side = "long" if self.side == LONG else "short"
                raise ValueError(
                    f"Planned nominal leverage exceeds limit: side={side}, "
                    f"max_layers={self.max_layers}, addition={addition}, "
                    f"required={required:.12g}, limit={limit:.12g}"
                )

    @property
    def price_params(self) -> common.MartingaleLabelParams:
        return common.MartingaleLabelParams(
            self.side, self.step_pct, self.initial_take_profit_pct
        )

    def layer_price(self, entry_price: float, addition: int) -> float:
        return self.price_params.layer_price(entry_price, addition)

    def stop_price(self, entry_price: float) -> float:
        return self.layer_price(entry_price, self.max_layers + 1)

    def take_profit_price(self, entry_price: float, state: Position) -> float:
        return self.price_params.target_price(entry_price, state.filled_layers - 1)

    def order_size(self, entry_price: float, state: Position) -> float:
        p = self.params
        price = self.layer_price(entry_price, state.filled_layers)
        target = self.price_params.target_price(entry_price, state.filled_layers)
        denominator = self.side * (target - price) - p.maker_fee_rate * (price + target)
        if denominator <= 0:
            raise ValueError("Addition profit denominator is nonpositive after fees")
        numerator = (
            p.goal
            - self.side * (state.qty * target - state.cost)
            + state.fees
            + p.maker_fee_rate * state.qty * target
        )
        qty = numerator / denominator
        if not np.isfinite(qty) or qty <= 0:
            raise ValueError("Required addition quantity must be finite and positive")
        return qty


@dataclass
class Position:
    qty: float
    cost: float
    fees: float
    filled_layers: int = 1
    terminal: int = -1
    exit_reason: str = "unresolved"
    exit_price: float = 0.0
    max_adverse_loss_pct: float = 0.0
    max_nominal_leverage: float = 0.0


def _base_position(entry: float, policy: GridPolicy) -> Position:
    p = policy.params
    fill = entry * (1 + policy.side * p.base_slippage_rate)
    qty = (p.initial_equity if p.base_notional is None else p.base_notional) / fill
    return Position(qty, qty * fill, qty * fill * p.taker_fee_rate)


def _pnl(state: Position, price: float, policy: GridPolicy, fee: float) -> float:
    return (
        policy.side * (state.qty * price - state.cost)
        - state.fees
        - state.qty * price * fee
    )


def _observe(state: Position, price: float, policy: GridPolicy) -> None:
    p = policy.params
    state.max_adverse_loss_pct = max(
        state.max_adverse_loss_pct,
        -_pnl(state, price, policy, p.taker_fee_rate) / p.initial_equity,
    )
    state.max_nominal_leverage = max(
        state.max_nominal_leverage, state.qty * price / p.initial_equity
    )


def _close(state: Position, price: float, reason: str, policy: GridPolicy) -> None:
    state.terminal = TAKE_PROFIT if reason == "take_profit" else GRID_STOP
    state.exit_reason = reason
    state.exit_price = (
        price
        if reason == "take_profit"
        else price * (1 - policy.side * policy.params.stop_slippage_rate)
    )
    _observe(state, state.exit_price, policy)


@dataclass
class CycleOutcome:
    start_index: int
    start_time: object
    entry_price: float
    side: int
    max_layers: int
    label: Optional[int]
    label_status: str
    label_max_additions: int
    path_source: str
    terminal: int
    exit_reason: str
    filled_layers: int
    exit_index: int
    exit_time: object
    exit_price: float
    avg_price: float
    bars_held: int
    breach_index: Optional[int]
    breach_time: object
    breach_price: Optional[float]
    breach_drawdown_pct: Optional[float]
    initial_equity: float
    quantity: float
    entry_cost: float
    entry_fees: float
    exit_fee: float
    net_pnl: float
    net_return_pct: float
    max_adverse_loss_pct: float
    max_nominal_leverage: float

    def as_row(self) -> dict:
        return asdict(self)


def project_label(
    bars, label: common.MartingaleLabel, policy: GridPolicy
) -> Optional[CycleOutcome]:
    """Project financial fills without replaying or changing the price label."""
    if not 0 <= policy.max_layers <= 8:
        raise ValueError("max_layers must be between 0 and 8; addition 9 is the label cap")
    if label.status == "invalid":
        return None
    entry = label.entry_price
    state = _base_position(entry, policy)
    exit_index = len(bars.close) - 1
    exit_time = bars.time[exit_index]
    for hit in label.hits:
        if hit.addition == policy.max_layers + 1:
            _close(state, hit.price, "grid_stop", policy)
            exit_index, exit_time = hit.index, hit.time
            break
        level = policy.layer_price(entry, hit.addition)
        qty = policy.order_size(entry, state)
        state.qty += qty
        state.cost += qty * level
        state.fees += qty * level * policy.params.maker_fee_rate
        state.filled_layers += 1
    if state.terminal < 0 and label.status == "take_profit" and label.max_additions <= policy.max_layers:
        _close(state, policy.price_params.target_price(entry, label.max_additions), "take_profit", policy)
        exit_index, exit_time = label.exit_index, label.exit_time
    terminal = UNRESOLVED if state.terminal < 0 else state.terminal
    price = float(bars.close[exit_index]) if terminal == UNRESOLVED else state.exit_price
    fee_rate = policy.params.maker_fee_rate if terminal == TAKE_PROFIT else policy.params.taker_fee_rate
    net = _pnl(state, price, policy, fee_rate)
    breach = terminal == GRID_STOP
    return CycleOutcome(
        start_index=label.start_index,
        start_time=label.start_time,
        entry_price=entry,
        side=policy.side,
        max_layers=policy.max_layers,
        label=label.label,
        label_status=label.status,
        label_max_additions=label.max_additions,
        path_source=label.path_source,
        terminal=terminal,
        exit_reason=state.exit_reason,
        filled_layers=state.filled_layers,
        exit_index=exit_index,
        exit_time=exit_time,
        exit_price=price,
        avg_price=state.cost / state.qty,
        bars_held=exit_index - label.start_index,
        breach_index=exit_index if breach else None,
        breach_time=exit_time if breach else None,
        breach_price=price if breach else None,
        breach_drawdown_pct=net / policy.params.initial_equity if breach else None,
        initial_equity=policy.params.initial_equity,
        quantity=state.qty,
        entry_cost=state.cost,
        entry_fees=state.fees,
        exit_fee=state.qty * price * fee_rate,
        net_pnl=net,
        net_return_pct=net / policy.params.initial_equity,
        max_adverse_loss_pct=float("nan"),
        max_nominal_leverage=float("nan"),
    )


def simulate_cycle(
    bars, start_index: int, policy: GridPolicy
) -> Optional[CycleOutcome]:
    label = common.martingale_label(bars, start_index, policy.price_params)
    return project_label(bars, label, policy)


_GLOBAL_BARS = None
_GLOBAL_POLICIES: tuple[GridPolicy, ...] = ()
_GLOBAL_GATE = None


def _init_worker(bars, policies, gate=None) -> None:
    global _GLOBAL_BARS, _GLOBAL_POLICIES, _GLOBAL_GATE
    _GLOBAL_BARS, _GLOBAL_POLICIES, _GLOBAL_GATE = bars, tuple(policies), gate


def _scan_chunk(bounds: tuple[int, int]) -> dict[int, list[dict]]:
    rows = {policy.max_layers: [] for policy in _GLOBAL_POLICIES}
    price_params = _GLOBAL_POLICIES[0].price_params
    for index in range(*bounds):
        if _GLOBAL_GATE is not None and not _GLOBAL_GATE[index]:
            continue
        label = common.martingale_label(_GLOBAL_BARS, index, price_params)
        for policy in _GLOBAL_POLICIES:
            outcome = project_label(_GLOBAL_BARS, label, policy)
            if outcome is not None:
                rows[policy.max_layers].append(outcome.as_row())
    return rows


def resolved_workers(workers: int, chunks: int) -> int:
    return max(
        1,
        min(
            workers if workers > 0 else max(1, (multiprocessing.cpu_count() or 1) - 1),
            max(1, chunks),
        ),
    )


def scan_starts(
    bars,
    policy: GridPolicy,
    *,
    workers: int = 0,
    chunk_size: int = 256,
    limit: Optional[int] = None,
    gate=None,
) -> list[dict]:
    return scan_policies(
        bars,
        (policy,),
        workers=workers,
        chunk_size=chunk_size,
        limit=limit,
        gate=gate,
    )[policy.max_layers]


def scan_policies(
    bars,
    policies,
    *,
    workers: int = 0,
    chunk_size: int = 256,
    limit: Optional[int] = None,
    gate=None,
) -> dict[int, list[dict]]:
    """Compute one common label per start for policies sharing price parameters.

    ``gate`` is an optional boolean mask aligned to bars; ``False`` skips that
    start, which models an entry filter that only blocks new ladders.
    """
    policies = tuple(policies)
    rows = {policy.max_layers: [] for policy in policies}
    if not policies:
        return rows
    if len(rows) != len(policies):
        raise ValueError("Policies must have distinct max_layers")
    for policy in policies:
        if not 0 <= policy.max_layers <= 8:
            raise ValueError("max_layers must be between 0 and 8; addition 9 is the label cap")
        if policy.price_params != policies[0].price_params:
            raise ValueError("Policies must share side and all price parameters")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if gate is not None:
        gate = np.asarray(gate, dtype=bool).reshape(-1)
        if len(gate) != len(bars.close):
            raise ValueError("gate must align with bars")
    total = len(bars.close) if limit is None else min(limit, len(bars.close))
    bounds = [
        (start, min(start + chunk_size, total)) for start in range(0, total, chunk_size)
    ]
    if not bounds:
        return rows
    resolved = resolved_workers(workers, len(bounds))
    _init_worker(bars, policies, gate)
    try:
        if resolved == 1:
            pieces = map(_scan_chunk, bounds)
            for piece in pieces:
                for layers in rows:
                    rows[layers].extend(piece[layers])
        else:
            with multiprocessing.get_context("fork").Pool(resolved) as pool:
                for piece in pool.imap(_scan_chunk, bounds):
                    for layers in rows:
                        rows[layers].extend(piece[layers])
        return rows
    finally:
        _init_worker(None, ())

