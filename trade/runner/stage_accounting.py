"""Slice a continuous account without replaying orders or resetting positions."""

import numpy as np
import pandas as pd

STAGE_ACCOUNTING_VERSION = "continuous_account_v1"
ACCOUNT_COLUMNS = ("time", "previous_equity", "equity", "previous_position", "position", "exposure")


def slice_account(history, closed_trades, region, interval_ms):
    """Use before-first-bar equity and mark-to-market PnL, including carried positions.

    Closed trades are attributed by exit time, and retain their lifetime PnL.
    That descriptive PnL does not reconcile to segment mark-to-market returns
    when trades cross boundaries. Daily Sharpe uses zero risk-free rate, sample
    standard deviation and 365 days/year; partial first/last days are included.
    """
    account = pd.DataFrame(history, columns=ACCOUNT_COLUMNS)
    times = pd.to_datetime(account.time, utc=True)
    if times.isna().any() or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError("Account history timestamps must be unique and chronological")
    start, end = pd.to_datetime(region["start"], utc=True), pd.to_datetime(region["end"], utc=True)
    before_end = times < end if region.get("exclusive_end") else times <= end
    rows = account.loc[(times >= start) & before_end].copy()
    if rows.empty:
        return None, None
    rows["time"] = pd.to_datetime(rows.time, utc=True)
    baseline = float(rows.previous_equity.iloc[0])
    final = float(rows.equity.iloc[-1])
    values = np.r_[baseline, rows.equity.to_numpy(float)]
    if not np.isfinite(values).all() or baseline <= 0:
        raise ValueError("Segment accounting requires finite equity and a positive initial equity")
    highs = np.maximum.accumulate(values)
    drawdowns = np.where(highs > 0, (highs - values) / highs, 0.)
    duration_days = (rows.time.iloc[-1] - rows.time.iloc[0]).total_seconds() / 86400 + interval_ms / 86400000
    gross_return = final / baseline - 1
    annual_log_return = np.log(final / baseline) * 365 / duration_days if final > 0 else -np.inf
    cagr = float(np.expm1(annual_log_return)) if annual_log_return < 709 else None
    daily = []
    for date, day in rows.groupby(rows.time.dt.date, sort=True):
        day_values = np.r_[day.previous_equity.iloc[0], day.equity.to_numpy(float)]
        daily.append(dict(date=str(date), start_equity=float(day_values[0]),
                          minimum_equity=float(day_values.min()), end_equity=float(day_values[-1])))
    daily_returns = np.array([day["end_equity"] / day["start_equity"] - 1
                              if day["start_equity"] > 0 else np.nan for day in daily])
    finite = daily_returns[np.isfinite(daily_returns)]
    deviation = finite.std(ddof=1) if len(finite) > 1 else 0.
    sharpe = float(finite.mean() / deviation * np.sqrt(365)) if deviation > 0 else None
    selected = []
    for trade in closed_trades:
        exit_time = pd.to_datetime(trade["exit_time"], utc=True)
        if exit_time >= rows.time.iloc[0] and (exit_time < end if region.get("exclusive_end") else exit_time <= end):
            selected.append({**trade, "crosses_start": bool(pd.to_datetime(trade["entry_time"], utc=True) < rows.time.iloc[0])})
    pnls = np.array([trade["pnl_net"] for trade in selected], dtype=float)
    won = pnls[pnls > 0].sum()
    lost = -pnls[pnls < 0].sum()
    max_dd = float(drawdowns.max() * 100)
    worst = min(daily, key=lambda day: day["minimum_equity"] / day["start_equity"] if day["start_equity"] > 0 else 0)
    daily_dd = [day["minimum_equity"] / day["start_equity"] - 1 if day["start_equity"] > 0 else 0 for day in daily]
    holds = np.array([trade["hold_bars"] for trade in selected], dtype=float)
    holding_stats = {"count": len(holds), "mean": float(holds.mean()) if len(holds) else 0.,
                     "min": float(holds.min()) if len(holds) else 0., "max": float(holds.max()) if len(holds) else 0.,
                     **{f"p{p}": float(np.percentile(holds, p)) if len(holds) else 0.
                                           for p in (10, 25, 50, 75, 90, 95, 99)}}
    streak, longest = {"won": 0, "lost": 0}, {"won": 0, "lost": 0}
    for pnl in pnls:
        for name, positive in (("won", pnl > 0), ("lost", pnl < 0)):
            streak[name] = streak[name] + 1 if positive else 0
            longest[name] = max(longest[name], streak[name])
    result = {
        "accounting": {"version": STAGE_ACCOUNTING_VERSION, "mode": "continuous_account_slice",
                       "equity_baseline": "immediately_before_first_included_bar",
                       "carry_in_position": float(rows.previous_position.iloc[0]),
                       "carry_out_position": float(rows.position.iloc[-1]),
                       "trade_attribution": "exit notification time when reflected in account; PnL includes the full trade lifetime",
                       "trade_pnl_reconciles_to_segment_return": False,
                       "sharpe_definition": "daily simple returns, rf=0, ddof=1, annualization=365, partial days included"},
        "time": {"start": rows.time.iloc[0].isoformat(), "end": rows.time.iloc[-1].isoformat(),
                 "sample_count": len(rows), "regions": region},
        "performance": {"gross_return": gross_return, "cagr": cagr, "sharpe": sharpe,
                        "calmar": float(cagr * 100 / max_dd) if max_dd > 0 and cagr is not None else None,
                        "start_value": baseline, "end_value": final},
        "drawdown": {"max_dd_pct": max_dd, "max_dd_amt": float((highs - values).max()),
                     "max_daily_dd": min(daily_dd), "max_daily_date": worst["date"],
                     **{f"dd_{p}_pct_days": sum(v < -p / 100 for v in daily_dd) for p in (3, 4, 5)}},
        "exposure": {"avg_pos": float(rows.exposure.mean()), "max_pos": float(rows.exposure.max()),
                     "p95_pos": float(rows.exposure.quantile(.95))},
        "trades": {"total": len(selected), "daily_freq": len(selected) / duration_days,
                   "win_rate": float((pnls > 0).mean()) if len(pnls) else 0.,
                   "profit_factor": float(won / lost) if lost > 0 else None,
                   "avg_pnl_net": float(pnls.mean()) if len(pnls) else 0.,
                   "avg_pnl_gross": float(np.mean([t["pnl"] for t in selected])) if selected else 0.,
                   "avg_cost": float(np.mean([t["commission"] for t in selected])) if selected else 0.,
                   "lost_longest": longest["lost"], "won_longest": longest["won"],
                   "holding_period_bars": holding_stats,
                   "cross_boundary_count": sum(t["crosses_start"] for t in selected),
                   **{f"{side}_pnl": sum(t["pnl_net"] for t in selected if t["long"] == flag)
                      for side, flag in (("long", True), ("short", False))},
                   **{f"{side}_win_rate": float(np.mean([t["pnl_net"] > 0 for t in selected if t["long"] == flag]) * 100)
                      if any(t["long"] == flag for t in selected) else 0.
                      for side, flag in (("long", True), ("short", False))}},
        "strategy": {},
    }
    details = {"accounting": result["accounting"], "raw_analyzer": {"customize": {"daily_account": daily}},
               "account_history_source": "full-period account_history; filter by stage time bounds",
               "closed_trades": selected}
    return result, details
