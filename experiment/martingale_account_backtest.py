"""Standalone sequential account backtest for the martingale ladder.

The script does not touch the production backtest pipeline. It replays one
ladder at a time on the same price labels used by the scan, charges maker fees
on base orders, additions and take profits, charges the taker fee only on grid
stops, and compounds the balance on every closed cycle.

A cycle opens at the close of an allowed bar, targets a 1% price move, and sizes
the base order so that move earns the profit goal (0.2% of the account by
default); additions are sized to keep that goal after averaging down. Trading
stops permanently when a grid stop leaves the balance below a share of the peak
equity. The default run is long-only and first prints an analytic grid snapshot
(layer prices, stop price, worst floating loss, leverage) per layer count.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import sys
import time
from dataclasses import dataclass, asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_process import common
from trade.sim import martingale_backtest as mb
from trade.sim import martingale_filters as mf
from trade.sim import martingale_scan as ms
from experiment.martingale_layer_scan import _data_params

LONG = 1
SHORT = -1
FORWARD_START = "2026-07-27T23:45:00+00:00"


class WarningAggTradeResolver:
    """Track failed candle resolutions and warn once per candle."""

    def __init__(self, resolver):
        self.resolver = resolver
        self.failures = []
        self.warned = set()

    def __call__(self, moment, ohlc, is_fine):
        prices = self.resolver(moment, ohlc, is_fine)
        if prices is None:
            self.failures.append(moment)
            key = (str(moment), is_fine)
            if key not in self.warned:
                self.warned.add(key)
                logging.getLogger("account").warning(
                    "Unresolved candle path at %s: aggTrades %s; continuing with OHLC fallback",
                    moment, getattr(self.resolver, "last_error", None) or "unavailable",
                )
        return prices


def fallback_stats(affected, total):
    return dict(
        path_fallback_trades=int(affected),
        path_fallback_total_trades=int(total),
        path_fallback_pct=affected / total * 100 if total else 0.0,
    )


@dataclass
class RunResult:
    side: str
    layers: int
    cycles: int
    take_profits: int
    grid_stops: int
    unresolved: int
    start_balance: float
    final_balance: float
    return_pct: float
    peak_balance: float
    max_drawdown_pct: float
    min_balance: float
    stopped: bool
    stop_time: object
    end_time: object
    years: float
    cagr_pct: float
    maker_fees: float
    taker_fees: float
    mean_cycle_return_pct: float
    mean_bars_held: float
    max_nominal_leverage: float
    path_fallback_trades: int = 0
    path_fallback_pct: float = 0.0

    def as_row(self) -> dict:
        return asdict(self)


def _policy(side: int, layers: int, balance: float, args) -> ms.FibonacciGrid:
    base_notional = (
        balance * args.base_notional_pct
        if args.base_notional_pct is not None
        else balance * args.goal_pct / args.take_profit_pct
    )
    return ms.FibonacciGrid(
        max_layers=layers,
        step_pct=args.step_pct,
        side=side,
        params=ms.ScanParams(
            initial_equity=balance,
            target_profit_pct=args.goal_pct,
            base_notional=base_notional,
            max_nominal_leverage=args.max_nominal_leverage,
            maker_fee_rate=args.maker_fee,
            taker_fee_rate=args.taker_fee,
            base_slippage_rate=0.0,
            stop_slippage_rate=0.0,
        ),
        initial_take_profit_pct=args.take_profit_pct,
    )


def run_cycle(bars, index: int, side: str, layers: int, balance: float, args, policy=None, label=None):
    """Simulate one ladder cycle; returns realized cash flows or ``None``."""
    direction = LONG if side == "long" else SHORT
    price_params = common.MartingaleLabelParams(
        side=direction, step_pct=args.step_pct, initial_take_profit_pct=args.take_profit_pct
    )
    resolver = getattr(bars, "agg_trade_prices", None)
    if label is None:
        if hasattr(resolver, "failures"):
            resolver.failures.clear()
        label = common.martingale_label(
            bars, index, price_params, path_policy=getattr(args, "path_policy", "fixed")
        )
    if label.status in ("invalid", "ambiguous"):
        return None
    policy = policy if policy is not None else _policy(direction, layers, balance, args)
    entry = float(label.entry_price)
    base_notional = policy.params.base_notional
    state = ms.Position(
        qty=base_notional / entry,
        cost=base_notional,
        fees=base_notional * args.maker_fee,
        filled_layers=1,
    )
    leverage = state.qty * entry / balance
    last = len(bars.close) - 1
    exit_index = last
    exit_price = None
    reason = None
    for hit in label.hits:
        if hit.addition == layers + 1:
            exit_index = hit.index
            exit_price = hit.price * (1 - direction * args.stop_slippage)
            reason = "grid_stop"
            break
        level = policy.layer_price(entry, hit.addition)
        quantity = policy.order_size(entry, state)
        state.qty += quantity
        state.cost += quantity * level
        state.fees += quantity * level * args.maker_fee
        state.filled_layers += 1
        leverage = max(leverage, state.qty * level / balance)
    if reason is None:
        if label.status == "take_profit" and label.max_additions <= layers:
            exit_index = label.exit_index
            exit_price = price_params.target_price(entry, label.max_additions)
            reason = "take_profit"
        else:
            exit_index = last
            exit_price = float(bars.close[last])
            reason = "unresolved"
    exit_fee_rate = args.maker_fee if reason == "take_profit" else args.taker_fee
    exit_fee = state.qty * exit_price * exit_fee_rate
    gross = direction * (state.qty * exit_price - state.cost)
    net = gross - state.fees - exit_fee
    leverage = max(leverage, state.qty * exit_price / balance)
    return dict(
        start_index=index,
        start_time=str(bars.time[index]),
        exit_index=exit_index,
        reason=reason,
        gross=gross,
        maker_fee=state.fees + (exit_fee if reason == "take_profit" else 0.0),
        taker_fee=0.0 if reason == "take_profit" else exit_fee,
        net=net,
        leverage=leverage,
        bars_held=exit_index - index,
        path_fallback=any(
            pd.Timestamp(moment) <= pd.Timestamp(bars.time[exit_index]) + pd.Timedelta(
                milliseconds=getattr(getattr(resolver, "resolver", None), "main_interval_ms", 0)
            ) - pd.Timedelta(nanoseconds=1)
            for moment in getattr(resolver, "failures", [])
        ),
    )


def simulate(bars, gate, side: str, layers: int, args) -> RunResult:
    side_name = side
    balance = float(args.initial_balance)
    start_balance = balance
    peak = balance
    lowest = balance
    max_drawdown = 0.0
    maker_fees = taker_fees = 0.0
    cycles = take_profits = grid_stops = unresolved = affected = 0
    returns = []
    bars_held = []
    stopped = False
    stop_time = None
    end_time = bars.time[-1]
    max_leverage = 0.0
    index = 0
    last = len(bars.close) - 1
    while index <= last:
        if not gate[index]:
            index += 1
            continue
        result = run_cycle(bars, index, side, layers, balance, args)
        if result is None:
            index += 1
            continue
        maker_fees += result["maker_fee"]
        taker_fees += result["taker_fee"]
        max_leverage = max(max_leverage, result["leverage"])
        net = result["net"]
        cycle_return = net / balance
        wiped = balance + net <= 0
        balance = 0.0 if wiped else balance + net
        returns.append(cycle_return)
        bars_held.append(result["bars_held"])
        cycles += 1
        affected += int(result.get("path_fallback", False))
        if result["reason"] == "take_profit":
            take_profits += 1
        elif result["reason"] == "grid_stop":
            grid_stops += 1
        else:
            unresolved += 1
        peak = max(peak, balance)
        lowest = min(lowest, balance)
        max_drawdown = max(max_drawdown, 1 - balance / peak)
        if wiped or (result["reason"] == "grid_stop" and balance < args.stop_peak_pct * peak):
            stopped = True
            stop_time = bars.time[result["exit_index"]]
            end_time = stop_time
            break
        if result["reason"] == "unresolved":
            end_time = bars.time[result["exit_index"]]
            break
        index = result["exit_index"] + 1
    final = balance
    days = (pd.Timestamp(end_time) - pd.Timestamp(bars.time[0])).total_seconds() / 86400
    years = max(days / 365.25, 1e-9)
    cagr = ((final / start_balance) ** (1 / years) - 1) * 100 if final > 0 else -100.0
    return RunResult(
        side=side_name,
        layers=layers,
        cycles=cycles,
        take_profits=take_profits,
        grid_stops=grid_stops,
        unresolved=unresolved,
        start_balance=start_balance,
        final_balance=final,
        return_pct=(final / start_balance - 1) * 100,
        peak_balance=peak,
        max_drawdown_pct=max_drawdown * 100,
        min_balance=lowest,
        stopped=stopped,
        stop_time=stop_time,
        end_time=end_time,
        years=years,
        cagr_pct=cagr,
        maker_fees=maker_fees,
        taker_fees=taker_fees,
        mean_cycle_return_pct=float(np.mean(returns) * 100) if returns else 0.0,
        mean_bars_held=float(np.mean(bars_held)) if bars_held else 0.0,
        max_nominal_leverage=max_leverage,
        path_fallback_trades=affected,
        path_fallback_pct=affected / cycles * 100 if cycles else 0.0,
    )


def monte_carlo(bars, gate, side: str, layers: int, args) -> tuple[dict, list[dict]]:
    """Random-start single-cycle trials with a fixed account balance."""
    rng = np.random.default_rng(args.seed)
    candidates = np.flatnonzero(gate)
    picks = rng.choice(candidates, size=args.runs, replace=True)
    policy = _policy(LONG if side == "long" else SHORT, layers, args.initial_balance, args)
    records = []
    for index in picks:
        result = run_cycle(bars, int(index), side, layers, args.initial_balance, args, policy)
        if result is not None:
            records.append(result)
    if not records:
        return dict(runs=0), records
    gross = np.array([row["gross"] for row in records])
    net = np.array([row["net"] for row in records])
    maker = np.array([row["maker_fee"] for row in records])
    taker = np.array([row["taker_fee"] for row in records])
    fees = maker + taker
    reasons = np.array([row["reason"] for row in records])
    wins = net > 0
    fee_ratio = np.divide(fees, gross, out=np.full_like(fees, np.nan), where=gross > 0)
    positive = gross > 0
    summary = dict(
        side=side,
        layers=layers,
        runs=len(records),
        **fallback_stats(sum(row.get("path_fallback", False) for row in records), len(records)),
        take_profits=int((reasons == "take_profit").sum()),
        grid_stops=int((reasons == "grid_stop").sum()),
        unresolved=int((reasons == "unresolved").sum()),
        win_rate_pct=float(wins.mean() * 100),
        mean_gross=float(gross.mean()),
        mean_net=float(net.mean()),
        median_net=float(np.median(net)),
        p05_net=float(np.percentile(net, 5)),
        p95_net=float(np.percentile(net, 95)),
        worst_net=float(net.min()),
        best_net=float(net.max()) if len(net) else 0.0,
        mean_maker_fee=float(maker.mean()),
        mean_taker_fee=float(taker.mean()),
        mean_total_fee=float(fees.mean()),
        total_gross=float(gross.sum()),
        total_fees=float(fees.sum()),
        total_net=float(net.sum()),
        fee_to_gross_pct=float(fees.sum() / gross.sum() * 100) if gross.sum() > 0 else float("nan"),
        mean_fee_to_gross_pct=float(np.nanmean(fee_ratio) * 100) if positive.any() else float("nan"),
        fee_exceeds_gross_pct=float((fee_ratio >= 1).mean() * 100) if positive.any() else float("nan"),
        mean_bars_held=float(np.mean([row["bars_held"] for row in records])),
        max_leverage=float(max(row["leverage"] for row in records)),
    )
    return summary, records


def print_monte_carlo(summaries: list[dict]) -> None:
    print()
    print(
        "side  K   runs  win%   tp%  stop%  mean_gross  mean_fee  mean_net  median_net"
        "      p05      p95  fee/win%  fee>gross%  sum_net"
    )
    for row in summaries:
        runs = row["runs"]
        if not runs:
            print("%-5s %d %6d  no sample" % (row["side"], row["layers"], runs))
            continue
        print(
            "%-5s %d %6d %5.1f %5.1f %5.1f %11.2f %9.2f %9.2f %11.2f %8.2f %8.2f %11.2f %11.2f %10.2f"
            % (
                row["side"],
                row["layers"],
                runs,
                row["win_rate_pct"],
                row["take_profits"] / runs * 100,
                row["grid_stops"] / runs * 100,
                row["mean_gross"],
                row["mean_total_fee"],
                row["mean_net"],
                row["median_net"],
                row["p05_net"],
                row["p95_net"],
                row["mean_fee_to_gross_pct"],
                row["fee_exceeds_gross_pct"],
                row["total_net"],
            )
        )


def precompute_cycles(bars, gate, side: str, args):
    """Cache per-start cycle outcomes for every layer count (scale invariant)."""
    direction = LONG if side == "long" else SHORT
    price_params = common.MartingaleLabelParams(
        side=direction, step_pct=args.step_pct, initial_take_profit_pct=args.take_profit_pct
    )
    allowed = np.flatnonzero(gate)
    count = len(allowed)
    data = {
        layers: dict(
            ret=np.full(count, np.nan),
            exit=np.full(count, -1, dtype=np.int64),
            reason=np.zeros(count, dtype=np.int8),
            maker=np.zeros(count),
            taker=np.zeros(count),
            path_fallback=np.zeros(count, dtype=bool),
        )
        for layers in args.layers
    }
    policies = {layers: _policy(direction, layers, args.initial_balance, args) for layers in args.layers}
    report_every = max(count // 10, 1)
    logging.getLogger("account").info("Precomputing %d start points for %s", count, side)

    def store(pos, layers, result):
        slot = data[layers]
        slot["ret"][pos] = result["net"] / args.initial_balance
        slot["exit"][pos] = result["exit_index"]
        slot["reason"][pos] = (
            0 if result["reason"] == "take_profit" else 1 if result["reason"] == "grid_stop" else 2
        )
        slot["maker"][pos] = result["maker_fee"] / args.initial_balance
        slot["taker"][pos] = result["taker_fee"] / args.initial_balance
        slot["path_fallback"][pos] = result.get("path_fallback", False)

    for pos in range(count):
        index = int(allowed[pos])
        if getattr(args, "path_policy", "fixed") == "dp":
            for layers in args.layers:
                result = dp_cycle(bars, index, layers, args, side, policies[layers])
                if result is not None:
                    store(pos, layers, result)
            if pos % report_every == 0:
                logging.getLogger("account").info("  %s %d/%d", side, pos, count)
            continue
        if hasattr(getattr(bars, "agg_trade_prices", None), "failures"):
            bars.agg_trade_prices.failures.clear()
        label = common.martingale_label(
            bars, index, price_params, path_policy=getattr(args, "path_policy", "fixed")
        )
        if label.status in ("invalid", "ambiguous"):
            continue
        for layers in args.layers:
            result = run_cycle(
                bars, index, side, layers, args.initial_balance, args, policies[layers], label
            )
            if result is not None:
                store(pos, layers, result)
        if pos % report_every == 0:
            logging.getLogger("account").info("  %s %d/%d", side, pos, count)
    return allowed, data


def path_simulation(
    allowed,
    slot,
    bars,
    side: str,
    layers: int,
    args,
    save_pct: float = 0.0,
    withdraw_multiple: float = 0.0,
    trigger_pct: float = 0.0,
    share_pct: float = 0.0,
) -> list[dict]:
    rng = np.random.default_rng(args.seed)
    count = len(allowed)
    starts = rng.integers(0, count, size=args.path_runs)
    initial = args.initial_balance
    last = len(bars.close) - 1
    exit_list = slot["exit"].tolist()
    ret_list = slot["ret"].tolist()
    maker_list = slot["maker"].tolist()
    taker_list = slot["taker"].tolist()
    reason_list = slot["reason"].tolist()
    fallback_list = slot["path_fallback"].tolist()
    nxt = np.searchsorted(allowed, slot["exit"] + 1).tolist()
    trigger_mult = 1.0 + trigger_pct / 100.0
    share = share_pct / 100.0
    rows = []
    for run_id, start_pos in enumerate(starts):
        balance = initial
        base = initial
        vault = 0.0
        withdrawals = 0
        peak = balance
        equity_peak = balance
        max_drawdown = 0.0
        fees = 0.0
        cycles = affected = 0
        halted = wiped = unfinished = False
        doubled = False
        double_index = last
        double_cycles = 0
        end_index = last
        pos = int(start_pos)
        while pos < count:
            ratio = ret_list[pos]
            if not np.isfinite(ratio):
                pos += 1
                continue
            fees += (maker_list[pos] + taker_list[pos]) * balance
            net = balance * ratio
            if net > 0 and save_pct > 0:
                saved = net * save_pct
                vault += saved
                balance += net - saved
            else:
                balance += net
            wiped = balance <= 0
            balance = 0.0 if wiped else balance
            peak = max(peak, balance)
            if withdraw_multiple > 0 and balance >= withdraw_multiple * initial:
                vault += balance - initial
                balance = initial
                withdrawals += 1
            if trigger_pct > 0 and balance >= trigger_mult * base:
                profit = balance - base
                saved = profit * share
                vault += saved
                balance -= saved
                base = balance
                withdrawals += 1
            cycles += 1
            affected += int(fallback_list[pos])
            equity = balance + vault
            equity_peak = max(equity_peak, equity)
            max_drawdown = max(max_drawdown, 1 - equity / equity_peak)
            exit_index = exit_list[pos]
            end_index = exit_index
            if not doubled and balance >= 2 * initial:
                doubled = True
                double_index = exit_index
                double_cycles = cycles
            reason = reason_list[pos]
            if wiped or (reason == 1 and balance < args.stop_peak_pct * peak):
                halted = True
                break
            if reason == 2:
                unfinished = True
                break
            pos = nxt[pos]
        if not halted:
            unfinished = True
        total = balance + vault
        days_to_double = None
        if doubled:
            start_ns = bars.time[int(allowed[start_pos])]
            double_ns = bars.time[double_index]
            days_to_double = float((double_ns - start_ns) / np.timedelta64(1, "D"))
        rows.append(
            dict(
                run=run_id,
                side=side,
                layers=layers,
                goal_pct=args.goal_pct * 100,
                stop_peak_pct=args.stop_peak_pct,
                profit_saved_pct=save_pct * 100,
                withdraw_multiple=withdraw_multiple,
                trigger_pct=trigger_pct,
                share_pct=share_pct,
                withdrawals=withdrawals,
                start_time=str(bars.time[int(allowed[start_pos])]),
                end_time=str(bars.time[end_index]),
                cycles=cycles,
                **fallback_stats(affected, cycles),
                doubled=doubled,
                cycles_to_double=double_cycles,
                days_to_double=days_to_double,
                final_balance=balance,
                vault=vault,
                total=total,
                return_pct=(balance / initial - 1) * 100,
                total_return_pct=(total / initial - 1) * 100,
                max_drawdown_pct=max_drawdown * 100,
                halted=halted,
                wiped=wiped,
                unfinished=unfinished,
                fees=fees,
            )
        )
    return rows


_WORKER_STATE: dict = {}


def _variant(args, **overrides):
    """Copy an argparse namespace with a few fields replaced."""
    values = dict(vars(args))
    values.update(overrides)
    return argparse.Namespace(**values)


def _path_task(task):
    """Worker entry point for one (layers, policy) configuration."""
    layers, save_pct, withdraw_multiple, trigger_pct, share_pct, side, args = task
    state = _WORKER_STATE
    return path_simulation(
        state["allowed"],
        state["slots"][layers],
        state["bars"],
        side,
        layers,
        args,
        save_pct,
        withdraw_multiple,
        trigger_pct,
        share_pct,
    )


def summarize_paths(rows: list[dict], side: str, layers: int, args) -> dict:
    final = np.array([row["final_balance"] for row in rows])
    vault = np.array([row["vault"] for row in rows])
    total = np.array([row["total"] for row in rows])
    total_returns = np.array([row["total_return_pct"] for row in rows])
    returns = np.array([row["return_pct"] for row in rows])
    drawdowns = np.array([row["max_drawdown_pct"] for row in rows])
    cycles = np.array([row["cycles"] for row in rows])
    fees = np.array([row["fees"] for row in rows])
    halted = np.array([row["halted"] for row in rows])
    wiped = np.array([row["wiped"] for row in rows])
    unfinished = np.array([row["unfinished"] for row in rows])
    initial = args.initial_balance
    gross = final - initial + fees
    positive = gross > 0
    rescued = (final < initial) & (total >= initial)
    doubled = np.array([row["doubled"] for row in rows])
    double_days = np.array([row["days_to_double"] for row in rows if row["doubled"]])
    return dict(
        side=side,
        layers=layers,
        goal_pct=args.goal_pct * 100,
        stop_peak_pct=args.stop_peak_pct,
        save_pct=rows[0]["profit_saved_pct"],
        withdraw_multiple=rows[0]["withdraw_multiple"],
        trigger_pct=rows[0]["trigger_pct"],
        share_pct=rows[0]["share_pct"],
        median_withdrawals=float(np.median([row["withdrawals"] for row in rows])),
        runs=len(rows),
        **fallback_stats(
            sum(row["path_fallback_trades"] for row in rows), int(cycles.sum()),
        ),
        doubled_pct=float(doubled.mean() * 100),
        median_days_to_double=float(np.median(double_days)) if len(double_days) else float("nan"),
        q25_days_to_double=float(np.percentile(double_days, 25)) if len(double_days) else float("nan"),
        q75_days_to_double=float(np.percentile(double_days, 75)) if len(double_days) else float("nan"),
        doubled_runs=int(doubled.sum()),
        halted_pct=float(halted.mean() * 100),
        wiped_pct=float(wiped.mean() * 100),
        unfinished_pct=float(unfinished.mean() * 100),
        mean_return_pct=float(returns.mean()),
        median_return_pct=float(np.median(returns)),
        p05_return_pct=float(np.percentile(returns, 5)),
        p95_return_pct=float(np.percentile(returns, 95)),
        mean_final=float(final.mean()),
        median_final=float(np.median(final)),
        median_vault=float(np.median(vault)),
        mean_vault=float(vault.mean()),
        median_total=float(np.median(total)),
        mean_total=float(total.mean()),
        median_total_return_pct=float(np.median(total_returns)),
        p05_total_return_pct=float(np.percentile(total_returns, 5)),
        p95_total_return_pct=float(np.percentile(total_returns, 95)),
        total_below_initial_pct=float((total < initial).mean() * 100),
        rescued_pct=float(rescued.mean() * 100),
        median_drawdown_pct=float(np.median(drawdowns)),
        median_cycles=float(np.median(cycles)),
        mean_fees=float(fees.mean()),
        total_fees=float(fees.sum()),
        fee_to_gross_pct=float(fees.sum() / gross.sum() * 100) if gross.sum() > 0 else float("nan"),
        mean_fee_to_gross_pct=float(np.mean(fees[positive] / gross[positive]) * 100)
        if positive.any()
        else float("nan"),
    )


def print_path_summary(summaries: list[dict]) -> None:
    print()
    print(
        "side  K  goal%  DD%  save%   k   X%    y%    runs  halt%  doubled%  days_med"
        "  median_total  median_total%  total<init%  med_draw"
    )
    for row in summaries:
        print(
            "%-5s %d %6.2f %4.0f %6.1f %5.2f %5.1f %5.1f %6d %5.1f %9.1f %9.1f %13.2f %14.2f %12.1f %10.0f"
            % (
                row["side"],
                row["layers"],
                row["goal_pct"],
                (1 - row["stop_peak_pct"]) * 100,
                row["save_pct"],
                row["withdraw_multiple"],
                row["trigger_pct"],
                row["share_pct"],
                row["runs"],
                row["halted_pct"],
                row["doubled_pct"],
                row["median_days_to_double"],
                row["median_total"],
                row["median_total_return_pct"],
                row["total_below_initial_pct"],
                row["median_withdrawals"],
            )
        )


def static_grid(layers: int, args) -> dict:
    """Analytic long-side grid snapshot independent of history."""
    policy = _policy(LONG, layers, args.initial_balance, args)
    entry = 100.0
    base = policy.params.base_notional
    state = ms.Position(
        qty=base / entry,
        cost=base,
        fees=base * args.maker_fee,
        filled_layers=1,
    )
    rows = []
    max_leverage = state.qty * entry / args.initial_balance
    for addition in range(layers + 1):
        price = policy.layer_price(entry, addition)
        if addition:
            quantity = policy.order_size(entry, state)
            state.qty += quantity
            state.cost += quantity * price
            state.fees += quantity * price * args.maker_fee
            state.filled_layers += 1
        else:
            quantity = state.qty
        target = policy.price_params.target_price(entry, addition)
        target_net = (
            state.qty * target - state.cost - state.fees - state.qty * target * args.maker_fee
        )
        leverage = state.qty * price / args.initial_balance
        max_leverage = max(max_leverage, leverage)
        rows.append(
            dict(
                layer=addition,
                offset_pct=(price / entry - 1) * 100,
                price=price,
                add_qty=quantity,
                cumulative_qty=state.qty,
                notional=state.qty * price,
                leverage=leverage,
                target_price=target,
                target_net=target_net,
                target_net_pct=target_net / args.initial_balance * 100,
            )
        )
    stop_price = policy.stop_price(entry)
    unrealized = state.qty * stop_price - state.cost
    exit_fee = state.qty * stop_price * args.taker_fee
    net_stop = unrealized - state.fees - exit_fee
    stop_leverage = state.qty * stop_price / args.initial_balance
    return dict(
        layers=layers,
        entry=entry,
        rows=rows,
        stop_price=stop_price,
        stop_offset_pct=(stop_price / entry - 1) * 100,
        stop_qty=state.qty,
        stop_notional=state.qty * stop_price,
        stop_leverage=stop_leverage,
        max_leverage=max(max_leverage, stop_leverage),
        unrealized_loss=unrealized,
        unrealized_loss_pct=unrealized / args.initial_balance * 100,
        exit_fee=exit_fee,
        maker_fees=state.fees,
        net_stop_loss=net_stop,
        net_stop_loss_pct=net_stop / args.initial_balance * 100,
        remaining_balance=args.initial_balance + net_stop,
        halt_if_first_stop=args.initial_balance + net_stop < args.stop_peak_pct * args.initial_balance,
    )


def print_static_grids(args) -> list[dict]:
    grids = []
    print()
    print(
        "Static long grid | entry 100.0000 | balance %.2f | base notional %.2f (%.1f%%)"
        " | maker %.4f%% | taker %.4f%%"
        % (
            args.initial_balance,
            args.initial_balance * args.goal_pct / args.take_profit_pct,
            args.goal_pct / args.take_profit_pct * 100,
            args.maker_fee * 100,
            args.taker_fee * 100,
        )
    )
    print(
        "  goal %.2f%% of balance | take profit %.2f%% | step %.2f%% | %s"
        % (
            args.goal_pct * 100,
            args.take_profit_pct * 100,
            args.step_pct * 100,
            "entry filter disabled"
            if args.no_entry_filter
            else "block: volatility<=50%, ADX<=10, |WR|<=10, clustering group 1, Sunday",
        )
    )
    deepest = max(args.layers)
    for layers in args.layers:
        grid = static_grid(layers, args)
        grids.append(grid)
        if layers != deepest:
            continue
        print()
        print("K=%d (deepest, includes all shallower layers)" % layers)
        print(
            "layer  offset%    price      add_qty  cum_qty   notional  leverage  tp_price"
            "   tp_net  tp_net%"
        )
        for row in grid["rows"]:
            print(
                "%5d %+8.2f %9.4f %10.4f %9.4f %10.2f %9.4f %10.4f %8.2f %8.4f"
                % (
                    row["layer"],
                    row["offset_pct"],
                    row["price"],
                    row["add_qty"],
                    row["cumulative_qty"],
                    row["notional"],
                    row["leverage"],
                    row["target_price"],
                    row["target_net"],
                    row["target_net_pct"],
                )
            )
        print(
            "stop  %+8.2f %9.4f %10s %9.4f %10.2f %9.4f"
            % (
                grid["stop_offset_pct"],
                grid["stop_price"],
                "-",
                grid["stop_qty"],
                grid["stop_notional"],
                grid["stop_leverage"],
            )
        )
        print(
            "  worst unrealized loss %.2f (%.2f%% of balance) | exit taker %.2f"
            " | net after fees %.2f (%.2f%%) | remaining balance %.2f (%.2f%%)"
            % (
                grid["unrealized_loss"],
                grid["unrealized_loss_pct"],
                grid["exit_fee"],
                grid["net_stop_loss"],
                grid["net_stop_loss_pct"],
                grid["remaining_balance"],
                grid["remaining_balance"] / args.initial_balance * 100,
            )
        )
        print(
            "  max leverage %.4fx at stop | maker fees to stop %.2f | taker fees to stop %.2f"
            " | halt on first stop: %s"
            % (
                grid["max_leverage"],
                grid["maker_fees"],
                grid["exit_fee"],
                "YES" if grid["halt_if_first_stop"] else "no",
            )
        )
    return grids


def _dp_positions(policy, entry: float, limit: int, args) -> list:
    """Position snapshot after each addition count, matching run_cycle sizing."""
    base = policy.params.base_notional
    qty, cost, fees = base / entry, base, base * args.maker_fee
    positions = [ms.Position(qty=qty, cost=cost, fees=fees, filled_layers=1)]
    for addition in range(1, limit + 1):
        level = policy.layer_price(entry, addition)
        quantity = policy.order_size(entry, positions[-1])
        qty += quantity
        cost += quantity * level
        fees += quantity * level * args.maker_fee
        positions.append(ms.Position(qty=qty, cost=cost, fees=fees, filled_layers=addition + 1))
    return positions


def dp_cycle(bars, start_index: int, layers: int, args, side: str, policy=None, horizon: int = 5_000):
    """Worst-case cycle value solved as a dynamic program over ladder states.

    The addition count fully describes the open ladder, so the pessimistic
    continuation is the minimum over the two orderings of every finest bar.
    Unlike the greedy per-bar pick, ``worst`` recurses on the resulting state,
    which makes the reported cash a lower bound over the reachable orderings.
    """
    if hasattr(getattr(bars, "agg_trade_prices", None), "failures"):
        bars.agg_trade_prices.failures.clear()
    direction = LONG if side == "long" else SHORT
    entry = float(bars.close[start_index])
    if not np.isfinite(entry) or entry <= 0:
        return None
    policy = policy or _policy(direction, layers, args.initial_balance, args)
    price_params = common.MartingaleLabelParams(
        side=direction, step_pct=args.step_pct, initial_take_profit_pct=args.take_profit_pct
    )
    limit = min(layers + 1, 9)
    positions = _dp_positions(policy, entry, limit, args)
    last = len(bars.close) - 1
    memo = {}
    fine_cache = {}

    def settle(a, price, taker, slippage=False):
        position = positions[min(a, limit)]
        exit_price = price * (1 - direction * args.stop_slippage) if slippage else price
        exit_fee = position.qty * exit_price * (args.taker_fee if taker else args.maker_fee)
        gross = direction * (position.qty * exit_price - position.cost)
        return gross - position.fees - exit_fee, exit_fee

    def advance(a, price):
        target = price_params.target_price(entry, a)
        if direction * (price - target) >= 0:
            return "take_profit", a, target
        while a < limit and direction * (price - policy.layer_price(entry, a + 1)) <= 0:
            a += 1
            if a == limit:
                return "grid_stop", a, policy.layer_price(entry, a)
        return None, a, price

    def walk(row, a, alternate):
        o, h, l, c = row
        path = (o, l, h, c) if ((direction == 1) != alternate) else (o, h, l, c)
        kind, a, price = advance(a, o)
        if kind:
            return kind, a, price
        previous = o
        for endpoint in path[1:]:
            while previous != endpoint:
                target = price_params.target_price(entry, a)
                level = policy.layer_price(entry, a + 1) if a < limit else target
                crossed = [x for x in (target, level)
                           if previous < x <= endpoint or endpoint <= x < previous]
                if not crossed:
                    break
                point = min(crossed, key=lambda x: abs(x - previous))
                kind, a, price = advance(a, point)
                if kind:
                    return kind, a, price
                previous = point
            previous = endpoint
        return None, a, c

    def fine_rows(t):
        cached = fine_cache.get(t)
        if cached is not None:
            return cached
        finer = None if getattr(bars, "intrabar", None) is None else bars.intrabar[t]
        valid = False
        if finer is not None and len(finer):
            finer = np.asarray(finer)
            row = (bars.open[t], bars.high[t], bars.low[t], bars.close[t])
            valid = (all(common._martingale_valid_row(r) for r in finer)
                     and np.allclose([finer[0, 0], finer[:, 1].max(), finer[:, 2].min(), finer[-1, 3]], row,
                                     rtol=1e-9, atol=1e-12))
        if valid:
            rows = [tuple(map(float, r[:4])) for r in finer]
        else:
            rows = [(float(bars.open[t]), float(bars.high[t]), float(bars.low[t]), float(bars.close[t]))]
        fine_cache[t] = rows
        return rows

    def worst(a, t, k):
        key = (a, t, k)
        if key in memo:
            return memo[key]
        while True:
            if t > last or t - start_index > horizon:
                index = min(t, last)
                net, fee = settle(a, float(bars.close[index]), taker=True)
                memo[key] = (net, index, "unresolved", a, fee, False)
                return memo[key]
            if k == 0:
                target = price_params.target_price(entry, a)
                level = policy.layer_price(entry, a + 1) if a < limit else target
                high, low = float(bars.high[t]), float(bars.low[t])
                if direction * (high - target) < 0 and direction * (low - level) > 0:
                    t += 1
                    continue
            rows = fine_rows(t)
            if k >= len(rows):
                t += 1
                k = 0
                continue
            primary = walk(rows[k], a, False)
            alternate = walk(rows[k], a, True)
            routes = (primary, alternate)
            used_fallback = False
            if primary != alternate:
                resolver = getattr(bars, "agg_trade_prices", None)
                if resolver is not None:
                    is_fine = len(rows) > 1
                    moment = bars.time[t]
                    if is_fine:
                        moment += np.timedelta64(k * bars.intrabar_step_ms, "ms")
                    prices = resolver(moment, rows[k], is_fine)
                    if prices is None:
                        used_fallback = True
                    if prices is not None:
                        current_a = a
                        resolved = (None, a, float(prices[-1]))
                        for trade_price in prices:
                            resolved = advance(current_a, float(trade_price))
                            kind, current_a, _ = resolved
                            if kind:
                                break
                        routes = (resolved,)
                else:
                    used_fallback = True
            best = None
            for kind, new_a, price in routes:
                if kind == "take_profit":
                    net, fee = settle(new_a, price, taker=False)
                    candidate = (net, t, kind, new_a, 0.0, used_fallback)
                elif kind == "grid_stop":
                    net, fee = settle(new_a, price, taker=True, slippage=True)
                    candidate = (net, t, kind, new_a, fee, used_fallback)
                else:
                    candidate = worst(new_a, t, k + 1)
                    candidate = (*candidate[:5], used_fallback or candidate[5])
                if best is None or candidate[0] < best[0]:
                    best = candidate
            memo[key] = best
            return best

    net, exit_index, reason, final_a, taker_fee, path_fallback = worst(0, start_index + 1, 0)
    return dict(
        net=net,
        exit_index=exit_index,
        reason=reason,
        maker_fee=positions[final_a].fees,
        taker_fee=taker_fee,
        path_fallback=path_fallback,
    )


def count_ambiguous(bars, gate, args) -> None:
    """Report take-profit and grid-stop counts without running the backtest.

    Every start whose label hits a bar the two orderings resolve differently is
    discarded, so the reported split describes only the trades that are fully
    determined at the finest available granularity.
    """
    rng = np.random.default_rng(args.seed)
    candidates = np.flatnonzero(gate)
    picks = rng.choice(candidates, size=args.runs, replace=True)
    rows = []
    for side in args.sides:
        direction = LONG if side == "long" else SHORT
        price_params = common.MartingaleLabelParams(
            side=direction, step_pct=args.step_pct, initial_take_profit_pct=args.take_profit_pct
        )
        counters = {
            layers: dict(tp=0, stop=0, ambiguous=0, invalid=0, unresolved=0) for layers in args.layers
        }
        for index in picks:
            label = common.martingale_label(bars, int(index), price_params, path_policy="strict")
            if label.status in ("ambiguous", "invalid"):
                for count in counters.values():
                    count[label.status] += 1
                continue
            for layers, count in counters.items():
                if label.status == "take_profit" and label.max_additions <= layers:
                    count["tp"] += 1
                elif label.max_additions >= layers + 1:
                    count["stop"] += 1
                else:
                    count["unresolved"] += 1
        for layers, count in counters.items():
            decided = count["tp"] + count["stop"]
            rows.append(
                dict(
                    side=side,
                    layers=layers,
                    starts=len(picks),
                    tp=count["tp"],
                    stop=count["stop"],
                    ambiguous=count["ambiguous"],
                    invalid=count["invalid"],
                    unresolved=count["unresolved"],
                    tp_share_pct=count["tp"] / decided * 100 if decided else 0.0,
                    stop_share_pct=count["stop"] / decided * 100 if decided else 0.0,
                    ambiguous_share_pct=count["ambiguous"] / len(picks) * 100 if len(picks) else 0.0,
                )
            )
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    folder = args.out / datetime.now().strftime("%Y-%m-%d/%H_%M_%S_%f")
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "ambiguous_counts.csv", index=False)
    logging.getLogger("account").info("Saved label counts to %s", folder / "ambiguous_counts.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sides", nargs="+", default=["long"], choices=["long", "short"])
    parser.add_argument("--layers", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--goal-pct", type=float, nargs="+", default=[0.002])
    parser.add_argument(
        "--base-notional-pct",
        type=float,
        default=None,
        help="Base notional as a share of balance; defaults to goal_pct / take_profit_pct",
    )
    parser.add_argument("--max-nominal-leverage", type=float, default=10.0)
    parser.add_argument("--take-profit-pct", type=float, default=0.01)
    parser.add_argument("--step-pct", type=float, default=0.01)
    parser.add_argument("--maker-fee", type=float, default=0.0002)
    parser.add_argument("--taker-fee", type=float, default=0.0005)
    parser.add_argument("--stop-slippage", type=float, default=0.0)
    parser.add_argument("--stop-peak-pct", type=float, nargs="+", default=[0.4])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--intrabar-interval", default="1m")
    parser.add_argument("--no-intrabar", action="store_true")
    parser.add_argument(
        "--path-policy",
        choices=["fixed", "pessimistic", "strict", "dp"],
        default="fixed",
        help="Intrabar ambiguity handling: fixed keeps the historical ordering, "
        "pessimistic replays both orderings and keeps the worse outcome, "
        "strict reports label counts, "
        "dp solves the worst case as a dynamic program over ladder states; "
        "unresolved aggTrades emit warnings and use the selected OHLC fallback",
    )
    parser.add_argument("--no-entry-filter", action="store_true")
    parser.add_argument(
        "--gate-drop",
        nargs="+",
        choices=["vol", "adx", "wr", "clustering", "weekday"],
        default=[],
        help="Disable individual entry-filter conditions for ablation runs",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10_000,
        help="Random-start trials per configuration (random mode)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--profit-save",
        type=float,
        nargs="+",
        default=[0.0],
        help="Fractions of each positive cycle profit moved to an untraded vault (e.g. 0.1 0.2)",
    )
    parser.add_argument(
        "--withdraw-multiple",
        type=float,
        nargs="+",
        default=[],
        help="Withdraw all profit above one base whenever balance reaches k times the base (e.g. 1.2 1.5 2.0)",
    )
    parser.add_argument(
        "--withdraw-trigger",
        type=float,
        nargs="+",
        default=[],
        help="Profit growth thresholds in percent that trigger a withdrawal (e.g. 10 20 50)",
    )
    parser.add_argument(
        "--withdraw-share",
        type=float,
        nargs="+",
        default=[],
        help="Share of accumulated profit to withdraw at each trigger, in percent (e.g. 30 50)",
    )
    parser.add_argument(
        "--path-runs",
        type=int,
        default=0,
        help="Random-start compounding paths per configuration (0 disables path mode)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes for path mode; configurations are distributed across them",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Replay one continuous account path instead of random-start trials",
    )
    parser.add_argument("--out", type=Path, default=Path("output/martingale_account_backtest"))
    args = parser.parse_args()

    goal_list = list(args.goal_pct)
    stop_list = list(args.stop_peak_pct)
    args.goal_pct = goal_list[0]
    args.stop_peak_pct = stop_list[0]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("account")
    started = time.perf_counter()
    bars = mb.load_bars(
        logger,
        _data_params(),
        intrabar_interval=None if args.no_intrabar else args.intrabar_interval,
    )
    logger.info(
        "Loaded %s bars; intrabar rows=%s",
        len(bars.close),
        len(bars.intrabar) if bars.intrabar is not None else 0,
    )
    if args.limit is not None and args.limit < len(bars.close):
        bars = replace(
            bars,
            time=bars.time[: args.limit],
            open=bars.open[: args.limit],
            high=bars.high[: args.limit],
            low=bars.low[: args.limit],
            close=bars.close[: args.limit],
            intrabar=None if bars.intrabar is None else bars.intrabar[: args.limit],
        )
    main_interval_ms = int(np.median(np.diff(bars.time)) / np.timedelta64(1, "ms"))
    fine_interval_ms = common.get_interval_ms(args.intrabar_interval)
    bars.intrabar_step_ms = fine_interval_ms
    bars.agg_trade_prices = WarningAggTradeResolver(
        common.AggTradeResolver(main_interval_ms, fine_interval_ms)
    )
    bars.require_resolved_path = False
    condition_fields = {
        "vol": "max_annualized_vol_pct",
        "adx": "max_adx",
        "wr": "max_abs_wr",
        "clustering": "max_clustering",
        "weekday": "weekdays",
    }
    filter_params = replace(
        mf.EntryFilterParams(),
        **{condition_fields[name]: None for name in args.gate_drop},
    )
    gate = (
        np.ones(len(bars.close), dtype=bool)
        if args.no_entry_filter
        else mf.build_entry_gate(bars, filter_params)
    )
    logger.info(
        "History %s -> %s (development + forward from %s)",
        bars.time[0],
        bars.time[-1],
        FORWARD_START,
    )
    logger.info("Bars %d, entry filter allows %d (%.2f%%)", len(gate), gate.sum(), gate.mean() * 100)

    if args.path_policy == "strict":
        count_ambiguous(bars, gate, args)
        return

    grids = print_static_grids(args)
    results = []
    summaries = []
    records = []
    path_summaries = []
    path_records = []
    precomputed_fallback = []
    if args.sequential:
        for side in args.sides:
            for layers in args.layers:
                result = simulate(bars, gate, side, layers, args)
                results.append(result)
                logger.info(
                    "%s K%d cycles=%d tp=%d stops=%d final=%.2f (%.2f%%) maxdd=%.2f%%%s",
                    side,
                    layers,
                    result.cycles,
                    result.take_profits,
                    result.grid_stops,
                    result.final_balance,
                    result.return_pct,
                    result.max_drawdown_pct,
                    " STOPPED" if result.stopped else "",
                )
    elif args.path_runs > 0:
        policies = []
        if args.withdraw_trigger and args.withdraw_share:
            for trigger in args.withdraw_trigger:
                for share in args.withdraw_share:
                    policies.append((0.0, 0.0, trigger, share))
        else:
            for withdraw_multiple in args.withdraw_multiple or [0.0]:
                for save_pct in args.profit_save:
                    policies.append((save_pct, withdraw_multiple, 0.0, 0.0))
        workers = max(1, args.workers)
        context = multiprocessing.get_context("fork")
        for side in args.sides:
            for goal in goal_list:
                goal_args = _variant(args, goal_pct=goal)
                allowed, data = precompute_cycles(bars, gate, side, goal_args)
                for layers, slot in data.items():
                    valid = np.isfinite(slot["ret"])
                    precomputed_fallback.append(dict(
                        side=side, layers=layers, goal_pct=goal,
                        **fallback_stats(int(slot["path_fallback"][valid].sum()), int(valid.sum())),
                    ))
                tasks = [
                    (
                        layers,
                        save_pct,
                        withdraw_multiple,
                        trigger_pct,
                        share_pct,
                        side,
                        _variant(goal_args, stop_peak_pct=stop),
                    )
                    for layers in args.layers
                    for save_pct, withdraw_multiple, trigger_pct, share_pct in policies
                    for stop in stop_list
                ]
                _WORKER_STATE.clear()
                _WORKER_STATE.update(allowed=allowed, slots=data, bars=bars)
                if workers > 1 and len(tasks) > 1:
                    logger.info(
                        "Running %d configurations on %d workers", len(tasks), min(workers, len(tasks))
                    )
                    with context.Pool(min(workers, len(tasks))) as pool:
                        results_by_task = pool.map(_path_task, tasks)
                else:
                    results_by_task = [_path_task(task) for task in tasks]
                _WORKER_STATE.clear()
                for (layers, save_pct, withdraw_multiple, trigger_pct, share_pct, _side, cfg), rows in zip(
                    tasks, results_by_task
                ):
                    summary = summarize_paths(rows, side, layers, cfg)
                    path_summaries.append(summary)
                    path_records.extend(rows)
                    logger.info(
                        "%s K%d goal=%.2f%% stop=%.0f%% save=%.0f%% k=%.2f X=%.0f%% y=%.0f%%"
                        " paths=%d median_total=%.2f halt=%.1f%%",
                        side,
                        layers,
                        goal * 100,
                        cfg.stop_peak_pct * 100,
                        save_pct * 100,
                        withdraw_multiple,
                        trigger_pct,
                        share_pct,
                        len(rows),
                        summary["median_total"],
                        summary["halted_pct"],
                    )
    else:
        for side in args.sides:
            for layers in args.layers:
                summary, rows = monte_carlo(bars, gate, side, layers, args)
                summaries.append(summary)
                for row in rows:
                    row["side"] = side
                    row["layers"] = layers
                records.extend(rows)
                logger.info(
                    "%s K%d runs=%s win=%.1f%% mean_net=%.2f mean_fee=%.2f",
                    side,
                    layers,
                    summary.get("runs"),
                    summary.get("win_rate_pct", float("nan")),
                    summary.get("mean_net", float("nan")),
                    summary.get("mean_total_fee", float("nan")),
                )
    elapsed = time.perf_counter() - started
    fallback_rows = (
        [dict(side=result.side, layers=result.layers, **fallback_stats(
            result.path_fallback_trades, result.cycles,
        )) for result in results]
        if args.sequential else path_summaries if args.path_runs > 0 else summaries
    )
    affected = sum(row.get("path_fallback_trades", 0) for row in fallback_rows)
    total_trades = sum(row.get("path_fallback_total_trades", 0) for row in fallback_rows)
    for row in fallback_rows:
        logger.info(
            "PATH FALLBACK %s K%d: %d/%d trades (%.4f%%)", row["side"], row["layers"],
            row.get("path_fallback_trades", 0), row.get("path_fallback_total_trades", 0),
            row.get("path_fallback_pct", 0.0),
        )
    fallback_summary = fallback_stats(affected, total_trades)
    logger.info(
        "PATH FALLBACK TOTAL: %d/%d executed trades (%.4f%%); %d unique unresolved candles",
        affected, total_trades, fallback_summary["path_fallback_pct"],
        len(bars.agg_trade_prices.warned),
    )

    if args.sequential:
        print()
        print(
            "side  K cycles   tp stops halt_date           final_bal  return%    cagr%   maxDD%"
            "  max_lev stopped  maker_fee  taker_fee  avg_cycle%  bars"
        )
        for result in results:
            halt = str(result.end_time)[:10]
            print(
                "%-5s %d %6d %4d %5d %s %11.2f %8.2f %8.2f %8.2f %8.2f %7s %10.2f %10.2f %10.4f %5.1f"
                % (
                    result.side,
                    result.layers,
                    result.cycles,
                    result.take_profits,
                    result.grid_stops,
                    str(halt),
                    result.final_balance,
                    result.return_pct,
                    result.cagr_pct,
                    result.max_drawdown_pct,
                    result.max_nominal_leverage,
                    "YES" if result.stopped else "no",
                    result.maker_fees,
                    result.taker_fees,
                    result.mean_cycle_return_pct,
                    result.mean_bars_held,
                )
            )
    else:
        if args.path_runs > 0:
            print_path_summary(path_summaries)
        else:
            print_monte_carlo(summaries)

    stamp = datetime.now().strftime("%Y-%m-%d/%H_%M_%S_%f")
    out_dir = args.out / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = None
    if records:
        frame = pd.DataFrame(records)
        frame["total_fee"] = frame["maker_fee"] + frame["taker_fee"]
        frame["fee_to_gross"] = frame.apply(
            lambda row: row["total_fee"] / row["gross"] if row["gross"] > 0 else float("nan"),
            axis=1,
        )
        frame["fee_pct_of_balance"] = frame["total_fee"] / args.initial_balance * 100
        frame["net_pct_of_balance"] = frame["net"] / args.initial_balance * 100
        runs_csv = out_dir / "random_runs.csv"
        frame.to_csv(runs_csv, index=False)
    paths_csv = None
    if path_records:
        paths_frame = pd.DataFrame(path_records)
        paths_csv = out_dir / "random_paths.csv"
        paths_frame.to_csv(paths_csv, index=False)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "sequential" if args.sequential else ("paths" if args.path_runs > 0 else "random"),
        "bars": len(bars.close),
        "start": str(bars.time[0]),
        "end": str(bars.time[-1]),
        "forward_start": FORWARD_START,
        "params": {
            key: value
            for key, value in vars(args).items()
            if key != "out"
        }
        | {"goal_pct_sweep": goal_list, "stop_peak_pct_sweep": stop_list},
        "allowed_bars": int(gate.sum()),
        "elapsed_seconds": elapsed,
        "static_grids": grids,
        "monte_carlo": summaries,
        "paths": path_summaries,
        "path_fallback": fallback_summary,
        "precomputed_path_fallback": precomputed_fallback,
        "results": [result.as_row() for result in results],
        "runs_csv": str(runs_csv) if runs_csv else None,
        "paths_csv": str(paths_csv) if paths_csv else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, default=str, indent=2))
    logger.info("Summary saved to %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()
