import numpy as np
import backtrader as bt
import math


def daily_drawdown_pct(item) -> float:
    """Calculate intraday drawdown from persisted daily account facts."""
    start_equity = float(item.get("start_equity", 0.0))
    minimum_equity = float(item.get("minimum_equity", start_equity))
    if start_equity <= 0:
        return 0.0
    return minimum_equity / start_equity - 1.0


def _count_daily_drawdown_breaches(daily_stats, threshold: float) -> int:
    """Count days whose drawdown magnitude is strictly greater than threshold."""
    return sum(
        1
        for item in daily_stats
        if daily_drawdown_pct(item) < -threshold
        and not math.isclose(
            daily_drawdown_pct(item),
            -threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )

class CusAnalyzer(bt.Analyzer):
    """
    Combined risk analyzer (high cohesion version)
    Responsibilities:
    1. Track position exposure
    2. Track the intraday max drawdown (FTMO standard)
    3. Flat (no position) distribution  <-- added
    """
    def start(self):
        # --- 1. position exposure state ---
        self._ratios = []
        self.account_history = []
        self.closed_trades = []
        self._previous_equity = float(self.strategy.broker.getvalue())
        self._previous_position = 0.0
        self._max_exposure = 0.0

        # --- 2. intraday drawdown state ---
        self._daily_stats = []
        self._curr_date = None
        self._day_start_equity = self.strategy.broker.getvalue()
        self._day_min_equity = self._day_start_equity
        self._day_end_equity = self._day_start_equity

        # --- 3. global minimum equity state ---
        self._global_min_equity = self.strategy.broker.getvalue()

        # --- FLAT DIST: flat distribution state (added) ---
        self._last_exit_dt = None
        self._prev_has_pos = False
        self._flat_days_round = 2 
        self._flat_periods_days = []        # length of every flat stretch (in days)

        # --- added: HWM state ---
        self._hwm = self._global_min_equity
        self._hwm_dt = self.strategy.data.datetime.datetime(0)
        self._max_hwm_duration = 0
    def next(self):
        """Called at the end of every bar, dispatches the logic"""
        self._track_exposure()
        self._track_daily_drawdown()
        self._track_global_min()

        # --- FLAT DIST: flat distribution tracking (added) ---
        self._track_flat_distribution()
        position = float(self.strategy.getposition(self.strategy.data).size)
        equity = float(self.strategy.broker.getvalue())
        self.account_history.append((
            self.strategy.data.datetime.datetime(0).isoformat(),
            self._previous_equity, equity, self._previous_position, position,
            abs(position * float(self.strategy.data.close[0])) / equity if equity > 0 else 0.0,
        ))
        self._previous_equity = equity
        self._previous_position = position

    def notify_trade(self, trade):
        if trade.isclosed:
            self.closed_trades.append({
                "entry_time": bt.num2date(trade.dtopen).isoformat(),
                "exit_time": self.strategy.data.datetime.datetime(0).isoformat(),
                "execution_exit_time": bt.num2date(trade.dtclose).isoformat(),
                "pnl": float(trade.pnl), "pnl_net": float(trade.pnlcomm),
                "commission": float(trade.commission), "long": bool(trade.long),
                "hold_bars": int(trade.barlen),
            })

    def stop(self):
        """Backtest finished, run the final aggregation"""
        self._record_day(self._curr_date)

        exposure_metrics = self._finalize_exposure()
        drawdown_metrics = self._finalize_drawdown()

        global_metrics = {
                    'global_min_equity': self._global_min_equity,
                    'max_hwm_duration_days': self._max_hwm_duration  # plain integer
        }

        # --- FLAT DIST: wrap up at the end of the backtest (added) ---
        flat_metrics = self._finalize_flat_distribution()

        self.rets = {**exposure_metrics, **drawdown_metrics, **global_metrics, **flat_metrics}

    def get_analysis(self):
        return self.rets

    # =========================================================
    # Global minimum equity
    # =========================================================
    def _track_global_min(self):
        current_equity = self.strategy.broker.getvalue()
        if current_equity < self._global_min_equity:
            self._global_min_equity = current_equity
        if current_equity >= self._hwm:
            self._hwm = current_equity
            self._hwm_dt = self.strategy.data.datetime.datetime(0)
        else:
            # .days returns the whole number of days between the two timestamps
            duration = (self.strategy.data.datetime.datetime(0) - self._hwm_dt).days
            if duration > self._max_hwm_duration:
                self._max_hwm_duration = duration
    # =========================================================
    # Position exposure
    # =========================================================
    def _track_exposure(self):
        equity = float(self.strategy.broker.getvalue())
        if equity <= 0:
            return

        gross_value = 0.0
        for d in self.strategy.datas:
            pos = self.strategy.getposition(d)
            if pos.size:
                gross_value += abs(pos.size) * float(d.close[0])

        ratio = gross_value / equity
        self._ratios.append(ratio)
        self._max_exposure = max(self._max_exposure, ratio)

    def _finalize_exposure(self):
        if not self._ratios:
            return {'avg_pos_ratio': 0.0, 'std_pos_ratio': 0.0, 'p95_pos_ratio': 0.0, 'max_pos_ratio': 0.0}

        arr = np.asarray(self._ratios, dtype=float)
        return {
            'avg_pos_ratio': float(arr.mean()),
            'std_pos_ratio': float(arr.std(ddof=0)),
            'p95_pos_ratio': float(np.quantile(arr, 0.95)),
            'max_pos_ratio': float(self._max_exposure),
        }

    # =========================================================
    # Intraday drawdown
    # =========================================================
    def _track_daily_drawdown(self):
        dt = self.strategy.data.datetime.date(0)
        current_equity = self.strategy.broker.getvalue()

        if self._curr_date is None:
            self._curr_date = dt

        if self._curr_date != dt:
            self._record_day(self._curr_date)
            self._curr_date = dt
            self._day_start_equity = current_equity
            self._day_min_equity = current_equity
            self._day_end_equity = current_equity
        else:
            if current_equity < self._day_min_equity:
                self._day_min_equity = current_equity
            self._day_end_equity = current_equity

    def _record_day(self, date_obj):
        if date_obj is None:
            return

        self._daily_stats.append({
            "date": str(date_obj),
            "start_equity": self._day_start_equity,
            "minimum_equity": self._day_min_equity,
            "end_equity": self._day_end_equity,
        })

    def _finalize_drawdown(self):
        if not self._daily_stats:
            return {
                'max_daily_dd': 0.0,
                'max_daily_dd_date': None,
                'daily_dd_violation_days': 0,
                'daily_account': [],
            }

        worst_day = min(
            self._daily_stats,
            key=daily_drawdown_pct,
        )
        violation_count_3 = _count_daily_drawdown_breaches(self._daily_stats, 0.03)
        violation_count_4 = _count_daily_drawdown_breaches(self._daily_stats, 0.04)
        violation_count_5 = _count_daily_drawdown_breaches(self._daily_stats, 0.05)

        return {
            'max_daily_dd': daily_drawdown_pct(worst_day),
            'max_daily_dd_date': worst_day['date'],
            'daily_dd_violation_days': violation_count_4,
            'daily_dd_max_violation_days': violation_count_5,
            'daily_dd_max_3_violation_days': violation_count_3,
            'daily_account': self._daily_stats,
        }

    # =========================================================
    # FLAT DIST: flat distribution (added)
    # =========================================================
    def _has_any_position(self) -> bool:
        """Multi-data compatible: any data holding a position counts as not flat"""
        for d in self.strategy.datas:
            pos = self.strategy.getposition(d)
            if pos.size:
                return True
        return False

    def _track_flat_distribution(self):
        cur_has_pos = self._has_any_position()
        prev_has_pos = self._prev_has_pos

        if cur_has_pos == prev_has_pos:
            return

        dt = self.strategy.data.datetime.datetime(0)

        # In position -> flat: record when the flat stretch started
        if prev_has_pos and (not cur_has_pos):
            self._last_exit_dt = dt

        # Flat -> in position: settle the previous flat stretch
        elif (not prev_has_pos) and cur_has_pos:
            if self._last_exit_dt is not None:
                flat_days = (dt - self._last_exit_dt).total_seconds() / 86400.0
                if flat_days > 0:
                    self._flat_periods_days.append(flat_days)
            self._last_exit_dt = None

        self._prev_has_pos = cur_has_pos

            
    def _finalize_flat_distribution(self):
        if (not self._prev_has_pos) and (self._last_exit_dt is not None):
            dt_end = self.strategy.data.datetime.datetime(0)
            flat_days = (dt_end - self._last_exit_dt).total_seconds() / 86400.0
            if flat_days > 0:
                self._flat_periods_days.append(flat_days)

        if not self._flat_periods_days:
            return {
                'flat_count': 0,
                'flat_max_days': 0,
                'flat_mean_days': 0.0,
                'flat_p50': 0, 'flat_p75': 0, 'flat_p90': 0, 'flat_p95': 0, 'flat_p99': 0,
                'flat_tail_mean_10pct': 0.0,
                'flat_tail_mean_5pct': 0.0,
                'flat_ge_7d': 0, 'flat_ge_14d': 0, 'flat_ge_30d': 0,
                'flat_bucket_pct': {},
                'flat_periods_raw': [],
            }

        arr = np.asarray(self._flat_periods_days, dtype=float)
        r = self._flat_days_round

        p50 = round(float(np.percentile(arr, 50)), r)
        p75 = round(float(np.percentile(arr, 75)), r)
        p90 = round(float(np.percentile(arr, 90)), r)
        p95 = round(float(np.percentile(arr, 95)), r)
        p99 = round(float(np.percentile(arr, 99)), r)

        flat_max = round(float(arr.max()), r)
        flat_mean = round(float(arr.mean()), r)

        q90 = np.percentile(arr, 90)
        q95 = np.percentile(arr, 95)
        tail10 = round(float(arr[arr >= q90].mean()), r) if (arr >= q90).any() else 0.0
        tail5  = round(float(arr[arr >= q95].mean()), r) if (arr >= q95).any() else 0.0

        return {
            'flat_count': int(len(arr)),
            'flat_max_days': flat_max,
            'flat_mean_days': flat_mean,
            'flat_p50': p50, 'flat_p75': p75, 'flat_p90': p90, 'flat_p95': p95, 'flat_p99': p99,
            'flat_tail_mean_10pct': tail10,
            'flat_tail_mean_5pct': tail5,
            'flat_ge_7d': int((arr >= 7).sum()),
            'flat_ge_14d': int((arr >= 14).sum()),
            'flat_ge_30d': int((arr >= 30).sum()),
        }

    @property
    def day_start_equity(self):
        return self._day_start_equity
