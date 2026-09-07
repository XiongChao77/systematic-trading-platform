from abc import ABC, abstractmethod
import logging, math, os
import pandas as pd
import numpy as np
from numba import njit, float64, int64

EPS = 1e-9  # avoid division by zero
_logger = logging.getLogger()


# All features should be based on this
class FeatureBase(ABC):
    def __init__(self, factory, **kwargs):
        self.params = kwargs
        self.features: list[str] = []
        self.factory: FeatureFactory = factory
        self.kline_interval_ms: int = kwargs.get("kline_interval_ms", None)

    # Note: using variance vs. mean as denominator scales small vs. large moves differently.

    def _apply_squashing(self, vals, scale=1.0, method=None):
        """Apply optional fixed-scale long-tail compression to standardized values."""
        if scale is None:
            raise ValueError("scale must not be None")
        if not np.isscalar(scale) or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"scale must be a positive finite scalar, got {scale!r}")
        scale = float(scale)

        if method is None:
            return vals
        if method == "tanh":
            return np.tanh(vals / scale)
        if method == "log":
            return np.sign(vals) * np.log1p(np.abs(vals / scale))
        raise ValueError(f"Unsupported squashing method: {method!r}")

    def _get_target_indices(self, feature_cols: list[str], target_feature_cols: list[str]):
        """
        Core helper for selecting target indices.
        1. Check that feature names exist in factory._feature_index.
        2. Check that feature names are present in the current feature_cols list.
        Returns: (valid index list, valid feature-name list).
        """
        # Use a set for faster membership checks
        cols_set = set(feature_cols)
        valid_indices = []
        valid_names = []

        for f in target_feature_cols:
            # Require existence both in global index and current feature set
            if f in self.factory._feature_index and f in cols_set:
                valid_indices.append(self.factory._feature_index[f])
                valid_names.append(f)

        return valid_indices, valid_names

    def _normalize_z_score(self, X, feature_cols, target_feature_cols, feature_base, factory, scale=1.0, method=None):
        target_indices, _ = self._get_target_indices(feature_cols, target_feature_cols)
        if not target_indices:
            return
        mu_base, sigma_base = factory.get_base_stats(feature_base)
        mu_base_3d = mu_base[:, :, np.newaxis] if mu_base.ndim == 2 else mu_base.reshape(-1, 1, 1)
        denom = sigma_base[:, :, np.newaxis]
        standardized = np.where(denom > EPS, (X[:, :, target_indices] - mu_base_3d) / denom, 0.0)
        # 2. Apply scaling and long-tail squashing
        X[:, :, target_indices] = self._apply_squashing(standardized, scale, method)

    def _normalize_z_score_group(self, X, feature_cols, target_feature_cols, factory, scale=1.0, method=None):
        """
        Group Z-Score normalization.
        Treat a feature group as a whole and compute a shared mean/std across time + feature axes.
        Advantage: preserves relative distances within the group (e.g. Upper vs Lower bands).
        """
        target_indices, _ = self._get_target_indices(feature_cols, target_feature_cols)
        if not target_indices:
            return

        # 2. Extract group data: shape (samples, time, group_features)
        vals = X[:, :, target_indices]

        # 3. Core: pool over time (axis=1) and feature (axis=2) axes
        #    Each batch sample gets a [1, 1, 1] stat tensor.
        group_mu = np.nanmean(vals, axis=(1, 2), keepdims=True)
        group_sigma = np.nanstd(vals, axis=(1, 2), keepdims=True)

        # 4. Standardize, with EPS guard. Zero-variance groups map to 0.
        standardized = np.where(group_sigma > EPS, (vals - group_mu) / group_sigma, 0.0)
        X[:, :, target_indices] = self._apply_squashing(standardized, scale, method)

    # method 'tanh'/'log'
    def _normalize_signal_group(self, X, feature_cols, target_feature_cols, factory, scale=1.0, method=None):
        """
        Zero-anchored group scaling:
        - Center around 0 and apply symmetric squashing.
        - method: 'tanh' (map to [-1, 1]) or 'log' (symmetric log1p).
        """
        target_indices, _ = self._get_target_indices(feature_cols, target_feature_cols)
        if not target_indices:
            return

        # 1. Extract group data and compute RMS to preserve zero axis and relative scaling.
        vals = X[:, :, target_indices]
        rms_group = np.sqrt(np.nanmean(vals**2, axis=(1, 2), keepdims=True))

        # 2. Dimensionless base scaling
        standardized = np.where(rms_group > EPS, vals / (rms_group), 0.0)

        X[:, :, target_indices] = self._apply_squashing(standardized, scale, method)

    def _normalize_z_score_rel(self, X, feature_cols, target_feature_cols, feature_base, factory, scale=1.0, method=None):
        target_indices, _ = self._get_target_indices(feature_cols, target_feature_cols)
        if not target_indices:
            return
        mu_base, sigma_base = factory.get_base_stats(feature_base)
        denom = sigma_base[:, :, np.newaxis]
        mu_self = np.nanmean(X[:, :, target_indices], axis=1, keepdims=True)
        standardized = np.where(denom > EPS, (X[:, :, target_indices] - mu_self) / denom, 0.0)
        X[:, :, target_indices] = self._apply_squashing(standardized, scale, method)

    def _normalize_signal(self, X, feature_cols, target_feature_cols, feature_base, factory):
        """
        For rate-of-change / direction / relative-deviation features.
        0 is neutral and the distribution is symmetric.
        sigma=0 -> 0
        """
        target_indices, _ = self._get_target_indices(feature_cols, target_feature_cols)
        if not target_indices:
            return

        mu, sigma = self.factory.get_base_stats(feature_base)

        X[:, :, target_indices] = np.where(sigma[:, :, None] > 0, (X[:, :, target_indices] - mu[:, :, None]) / sigma[:, :, None], 0.0)

    def _normalize_volume_rlc(self, X, feature_cols, target_feature_cols, feature_base):
        """
        Relative Log-Compression (RLC) for long-tailed volume-like features.
        Combines non-dimensionalization with strong tail suppression.
        :param feature_base: base feature for scaling (e.g. 'volume' or 'quote_asset_volume')
        """
        target_indices, _ = self._get_target_indices(feature_cols, target_feature_cols)
        if not target_indices:
            return

        # 2. Fetch base statistics: use mean as scaling reference.
        mu_base, _ = self.factory.get_base_stats(feature_base)
        mu_base = mu_base[:, :, np.newaxis]  # add dim for broadcasting (M, 1, 1)

        # Strict check: zero-mean base means corrupted/degenerate data.
        if np.any(mu_base == 0):
            raise ValueError(f"RLC Error: Base feature '{feature_base}' has zero mean. Data invalid.")

        # 3. Extract data
        X_target = X[:, :, target_indices]

        # 4. Non-dimensionalize and apply log1p compression:
        #    - 0 maps to 0
        #    - X = mean -> log(2) ≈ 0.69
        #    - X = 100 * mean -> log(101) ≈ 4.6
        X[:, :, target_indices] = np.log1p(np.maximum(X_target / mu_base, 0.0))

    def _normalize_scs(self, X, feature_cols, target_feature_cols, feature_base):
        """
        Structural-Consistency Scaling (SCS).

        Logic:
        1. Index lookup: locate the target feature group and base feature (e.g. Close) in X.
        2. Non-dimensionalize: divide target features by the base feature to get ratios.
        3. Pool stats: compute global min/max over time (T) and feature (F_sub) axes per sample.
        4. Map: min-max scale the whole ratio group to [0, 1] while preserving relative geometry.
        """
        # 1. Get indices
        target_indices, _ = self._get_target_indices(feature_cols, target_feature_cols)
        base_idx = self.factory._feature_index.get(feature_base)

        if not target_indices or base_idx is None:
            return

        # 2. Extract target data and base (M, T, F_sub) and (M, T, 1)
        X_target = X[:, :, target_indices]
        X_base = X[:, :, [base_idx]]
        # --- Strict check A: base feature contains zeros ---
        # e.g. Close price being 0 indicates corrupted data
        if np.any(X_base == 0):
            # Find the sample index of the anomaly for debugging data sources
            sample_idx = np.where(X_base == 0)[0][0]
            raise ValueError(
                f"SCS Error: Base feature '{feature_base}' contains zero at sample index {sample_idx}. " f"This indicates data corruption or price zeroing."
            )
        # 3. Compute ratios (Ratio = P / Base)
        ratios = X_target / X_base

        # 4. Compute group-level global stats
        group_min = np.nanmin(ratios, axis=(1, 2), keepdims=True)
        group_max = np.nanmax(ratios, axis=(1, 2), keepdims=True)
        group_diff = group_max - group_min

        # --- Strict check B: group has no variation ---
        # If max == min, the group is constant within the window (e.g. flatline/constant values)
        # This makes the Min-Max denominator 0
        if np.any(group_diff == 0):
            raise ZeroDivisionError(
                f"SCS Error: Feature group {target_feature_cols} has zero variance " f"within the window. Check if these features are constant."
            )

        # 5. Apply scaling
        X[:, :, target_indices] = (ratios - group_min) / group_diff

    @abstractmethod
    def generate(self, df: pd.DataFrame, kline_interval_ms) -> None: ...
    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        pass

    @abstractmethod
    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        pass

    def min_history_request(self, kline_interval_ms: int = None) -> int:
        min_request = self._min_history_request(kline_interval_ms)
        _logger.debug(f"{self.__class__.__name__} min_history_request {min_request}")
        return min_request


"""
Add MACD and multiple moving averages (SMA and EMA) to raw data.
Generated columns:
    - MACD_DIF, MACD_DEA, MACD
    - MA_{w}  (simple moving average)
    - EMA_{w} (exponential moving average; strict: NaN until window is full)
"""


class FeatureMACD(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fast = kwargs.get("fast", 12)
        self.slow = kwargs.get("slow", 16)
        self.signal = kwargs.get("signal", 9)

        prefix = f"MACD_{self.fast}_{self.slow}"

        # 1. Absolute-value feature group (price units)
        self.macd_abs_group = [f"{prefix}_DIF", f"{prefix}_DEA", f"{prefix}_HIST"]

        # 2. Percentage/momentum feature group
        self.macd_pct_group = [f"{prefix}_DIF_PCT", f"{prefix}_HIST_PCT", f"{prefix}_HIST_ACCEL"]

        # 3. Standalone feature (ratio)
        self.sig_dist = [f"{prefix}_SIG_DIST"]

        # Combine all features
        self.features = self.macd_abs_group + self.macd_pct_group + self.sig_dist

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        close = df["close"].astype(float)
        prefix = f"MACD_{self.fast}_{self.slow}"
        res = {}

        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()

        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.signal, adjust=False).mean()
        hist = dif - dea

        res[f"{prefix}_DIF"] = dif
        res[f"{prefix}_DEA"] = dea
        res[f"{prefix}_HIST"] = hist

        dif_pct = np.where(ema_slow != 0, (ema_fast - ema_slow) / ema_slow, np.nan)
        dif_pct_s = pd.Series(dif_pct, index=df.index)
        dea_pct = dif_pct_s.ewm(span=self.signal, adjust=False).mean()
        hist_pct = dif_pct_s - dea_pct

        res[f"{prefix}_DIF_PCT"] = dif_pct_s
        res[f"{prefix}_HIST_PCT"] = hist_pct
        res[f"{prefix}_HIST_ACCEL"] = hist_pct.diff().fillna(0)
        res[f"{prefix}_SIG_DIST"] = np.where(dea_pct != 0, (dif_pct_s - dea_pct) / np.abs(dea_pct), np.nan)

        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        Layered normalization strategy.
        """
        # 1. Absolute-value group: price-dimension features.
        # Use _normalize_z_score_rel anchored to a price base (e.g. 'close') to remove symbol price scale differences.
        self._normalize_z_score_rel(X, feature_cols, self.macd_abs_group, feature_base="close", factory=factory, method="log")

        # 2. Momentum percentage group (DIF_PCT, HIST_PCT, ACCEL)
        # Use zero-anchored group scaling
        self._normalize_signal_group(X, feature_cols, self.macd_pct_group, factory=factory, scale=1.0, method="log")

        # 3. SIG_DIST (scale separately; ratio-of-ratio)
        self._normalize_signal_group(X, feature_cols, self.sig_dist, factory=factory, scale=1.0, method="log")

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        return int((self.slow + self.signal) * 4)


class FeatureMAStructure(FeatureBase):
    """
    Multi-timescale MA structure features (keep relative relationships only).

    Output features (default):
    - BAR_S_L   = log(MA_bar_short / MA_bar_long)
    - BAR_M_L   = log(MA_bar_mid   / MA_bar_long)
    - DAY_S_L   = log(MA_day_short / MA_day_long)
    - WEEK_M_L  = log(MA_week_mid  / MA_week_long)

    Optional:
    - Δ_BAR_S_L
    - Δ_DAY_S_L
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ===== Configuration =====
        self.bar_windows = kwargs.get("bar_windows", (7, 21, 63))  # short, mid, long
        self.day_windows = kwargs.get("day_windows", (5, 20))  # short, long
        self.week_windows = kwargs.get("week_windows", (7, 25))  # mid, long
        self.slope_window = kwargs.get("slope_window", 5)

        self.add_delta = kwargs.get("add_delta", False)
        self.method = kwargs.get("method", "sma").lower()  # 'sma' or 'ema'
        self.strict = kwargs.get("strict", True)

        # ===== Register features =====
        self.features = [
            "MA_BAR_S_L",
            "MA_BAR_M_L",
            "MA_DAY_S_L",
            "MA_WEEK_M_L",
            "MA_WEEK_L_SLOPE",
        ]

        if self.add_delta:
            self.features += [
                "D_MA_BAR_S_L",
                "D_MA_DAY_S_L",
            ]

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _ma(self, series: pd.Series, window: int) -> pd.Series:
        if self.method == "ema":
            ma = series.ewm(span=window, adjust=False).mean()
            if self.strict:
                ma = ma.where(series.expanding().count() >= window, np.nan)
            return ma
        else:
            return series.rolling(window=window, min_periods=window if self.strict else 1).mean()

    def _calc_klines_per_day_week(self, kline_interval_ms: int) -> tuple[int, int]:
        one_day_ms = 24 * 60 * 60 * 1000
        one_week_ms = 7 * one_day_ms
        kpd = max(int(round(one_day_ms / kline_interval_ms)), 1)
        kpw = max(int(round(one_week_ms / kline_interval_ms)), 1)
        return kpd, kpw

    # ------------------------------------------------------------------
    # Feature generation
    # ------------------------------------------------------------------
    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        close = df["close"].astype(float)
        kpd, kpw = self._calc_klines_per_day_week(kline_interval_ms)
        res = {}

        # === BAR level ===
        b_s, b_m, b_l = self.bar_windows
        ma_bs = self._ma(close, b_s)
        ma_bm = self._ma(close, b_m)
        ma_bl = self._ma(close, b_l)
        res["MA_BAR_S_L"] = np.log(ma_bs / ma_bl)
        res["MA_BAR_M_L"] = np.log(ma_bm / ma_bl)

        # === DAY / WEEK level ===
        d_s, d_l = self.day_windows
        ma_ds = self._ma(close, d_s * kpd)
        ma_dl = self._ma(close, d_l * kpd)
        res["MA_DAY_S_L"] = np.log(ma_ds / ma_dl)

        w_m, w_l = self.week_windows
        ma_wm = self._ma(close, w_m * kpw)
        ma_wl = self._ma(close, w_l * kpw)
        res["MA_WEEK_M_L"] = np.log(ma_wm / ma_wl)
        res["MA_WEEK_L_SLOPE"] = np.log(ma_wl / ma_wl.shift(self.slope_window))

        # === Δ ===
        if self.add_delta:
            res["D_MA_BAR_S_L"] = pd.Series(res["MA_BAR_S_L"]).diff().fillna(0.0)
            res["D_MA_DAY_S_L"] = pd.Series(res["MA_DAY_S_L"]).diff().fillna(0.0)

        return pd.DataFrame(res, index=df.index)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        All features are:
        - log-ratio
        - 0-centered (neutral at 0)
        -> self z-score + symmetric log squashing is sufficient
        """
        for f in self.features:
            if f in feature_cols:
                self._normalize_z_score_rel(
                    X,
                    feature_cols,
                    [f],
                    feature_base=f,
                    factory=factory,
                    method="log",
                )

    # ------------------------------------------------------------------
    # Minimal history requirement
    # ------------------------------------------------------------------
    def _min_history_request(self, kline_interval_ms: int) -> int:
        # Slope requires looking back slope_window steps
        kpd, kpw = self._calc_klines_per_day_week(self.kline_interval_ms)
        max_week = max(self.week_windows) * kpw

        base = max(max(self.bar_windows), max(self.day_windows) * kpd, max_week)

        # Add extra buffer for slope
        base += self.slope_window

        if self.method == "ema":
            base = int(base * 3.5)
        return int(base + 2)


# Dimensionless
class FeatureRsi(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.period: int = kwargs.get("period", 14)
        self.price_col = kwargs.get("price_col", "close")
        self.strict = kwargs.get("strict", True)  # strict: NaN until full window; relaxed: produce values early
        self.prefix = kwargs.get("prefix", "RSI")
        self.features: list[str] = [f"{self.prefix}_{self.period}"]

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        close = df[self.price_col].astype(float)
        res = {}
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        avg_gain = gain.ewm(alpha=1 / float(self.period), adjust=False, min_periods=self.period if self.strict else 1).mean()
        avg_loss = loss.ewm(alpha=1 / float(self.period), adjust=False, min_periods=self.period if self.strict else 1).mean()

        rs = avg_gain / (avg_loss + EPS)
        rsi_values = np.where(avg_loss == 0, 100.0, 100.0 - (100.0 / (1.0 + rs)))
        rsi_values = np.where((avg_gain == 0) & (avg_loss == 0), 50.0, rsi_values)

        valid = close.expanding().count() >= self.period
        res[f"{self.prefix}_{self.period}"] = np.where(valid, rsi_values, np.nan)
        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        RSI range: [0, 100]
        Transform: (RSI / 100) - 0.5
        Output range: [-0.5, 0.5]
        0.0 corresponds to RSI 50 (neutral)
        """
        target_indices, _ = self._get_target_indices(feature_cols, self.features)
        if not target_indices:
            return

        # Simple scaling, preserves absolute position information
        X[:, :, target_indices] = (X[:, :, target_indices] / 100.0) - 0.5

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        """
        Minimal history needed for RSI.
        RSI uses Wilder smoothing (alpha=1/period), effectively an EMA.
        For convergence similar to standard implementations, provide ~5–10x period of history.
        """
        # Base period (e.g. 14)
        base_period = self.period
        # For live accuracy, use ~6x period as warm-up
        # e.g. 14 * 6 = 84 bars
        # This ensures EMA initial weights decay enough for stable values
        warmup_factor = 6
        return int(base_period * warmup_factor)


"""
Classic KDJ:
    RSV = (C - LLV(n)) / (HHV(n) - LLV(n)) * 100
    K = EMA(RSV, alpha=1/m1)
    D = EMA(K,   alpha=1/m2)
    J = 3*K - 2*D
Output columns: {prefix}_K, {prefix}_D, {prefix}_J
"""


# Dimensionless
class FeatureKdj(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features: list[str] = []
        self.n: int = kwargs.get("n", 9)
        self.m1 = kwargs.get("m1", 3)
        self.m2 = kwargs.get("m2", 3)
        self.high_col = kwargs.get("high_col", "high")
        self.low_col = kwargs.get("low_col", "low")
        self.close_col = kwargs.get("close_col", "close")
        self.strict = kwargs.get("strict", True)  # strict: NaN until full window; relaxed: produce values early
        self.prefix = kwargs.get("prefix", "KDJ")
        k_col, d_col, j_col = f"{self.prefix}_K", f"{self.prefix}_D", f"{self.prefix}_J"
        self.features = [k_col, d_col, j_col]

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        high, low, close = df[self.high_col].astype(float), df[self.low_col].astype(float), df[self.close_col].astype(float)
        res = {}
        llv = low.rolling(window=self.n, min_periods=self.n if self.strict else 1).min()
        hhv = high.rolling(window=self.n, min_periods=self.n if self.strict else 1).max()
        diff = hhv - llv
        rsv = np.where(diff == 0, 50.0, (close - llv) / (diff + EPS) * 100.0)

        rsv_s = pd.Series(rsv, index=df.index)
        K = rsv_s.ewm(alpha=1 / float(self.m1), adjust=False, min_periods=self.m1 if self.strict else 1).mean()
        D = K.ewm(alpha=1 / float(self.m2), adjust=False, min_periods=self.m2 if self.strict else 1).mean()
        J = 3 * K - 2 * D

        valid = close.expanding().count() >= self.n
        res[f"{self.prefix}_K"] = K.where(valid, np.nan)
        res[f"{self.prefix}_D"] = D.where(valid, np.nan)
        res[f"{self.prefix}_J"] = J.where(valid, np.nan)
        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        # KDJ is a 0–100 indicator; use simple scaling
        target_indices, _ = self._get_target_indices(feature_cols, self.features)
        if not target_indices:
            return

        # Map to [-0.5, 0.5]
        X[:, :, target_indices] = (X[:, :, target_indices] / 100.0) - 0.5

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        """
        Minimal history needed for KDJ.
        1. Base window n: RSV needs n bars.
        2. Smoothing windows m1, m2: K and D use recursive smoothing with alpha=1/m.
        For stable EWM convergence, use >=4x smoothing buffer.
        """
        # 1. Base RSV window
        base_n = self.n

        # 2. Smoothing convergence requirement (nested smoothing)
        # Following EMA convergence heuristics, (m1 + m2) * 4 is a safe buffer
        warmup_buffer = int((self.m1 + self.m2) * 4)

        # Total = base window + warm-up buffer
        # e.g. default (9, 3, 3) -> 9 + (3+3)*4 = 33 bars
        return base_n + warmup_buffer


class FeatureContainer:
    def __init__(self, feature: type[FeatureBase], **kwargs):
        # self.input_require =
        self.feature = feature
        self.parameters = kwargs


class FeatureATRRegime(FeatureBase):
    """
    Multi-horizon volatility regime analyzer.
    1. atr_{w}: percentage volatility (NATR-style) for modeling/trading.
    2. vol_regime: relative position of short-term vol vs. long-term backdrop (used to filter non-trending markets).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Support multiple ATR windows via 'windows' parameter, e.g. [14, 100, 1000]
        self.windows = kwargs.get("windows", [14, 100, 1000])
        self.short_w = self.windows[0]  # use first window as "short-term" baseline
        self.long_w = self.windows[-1]  # use last window as "long-term" backdrop

        self.features = []
        # Dynamically register all ATR features
        for w in self.windows:
            self.features.append(f"atr_{w}")

        # Regime reference features
        self.features.append(f"vol_regime_{self.long_w}")
        self.features.append("Vol_Trend")

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        close, h, l = df["close"].astype(float), df["high"], df["low"]
        pc = close.shift(1)
        res = {}
        tr = np.maximum(h - l, np.maximum((h - pc).abs(), (l - pc).abs()))

        atr_series_map = {}
        for w in self.windows:
            atr_w = tr.rolling(w).mean()
            atr_series_map[w] = atr_w
            res[f"atr_{w}"] = np.where(close > 0, atr_w / close, 0.0)

        short_atr = atr_series_map[self.short_w]
        long_atr_ref = short_atr.rolling(self.long_w).mean()
        res[f"vol_regime_{self.long_w}"] = np.where(long_atr_ref > 0, short_atr / long_atr_ref, 1.0)
        res["Vol_Trend"] = short_atr.diff(5) / (short_atr.shift(5) + EPS)
        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        # 1. Process all ATR features
        # Use per-feature z-score + tanh squashing so different volatility scales become comparable
        for w in self.windows:
            col = f"atr_{w}"
            if col in feature_cols:
                self._normalize_z_score(X, feature_cols, [col], feature_base=col, factory=factory, method="tanh")

        # 2. Regime reference feature
        regime_col = f"vol_regime_{self.long_w}"
        if regime_col in feature_cols:
            self._normalize_z_score(X, feature_cols, [regime_col], feature_base=regime_col, factory=factory, method="log")

        # 3. Vol_Trend (rate-of-change)
        if "Vol_Trend" in feature_cols:
            idx = feature_cols.index("Vol_Trend")
            X[:, :, idx] = self._apply_squashing(X[:, :, idx], scale=1.0, method="tanh")

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        # Must satisfy the largest window requirement
        return int(max(self.windows) * 1.2)


class FeatureVolMa(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vol_ma_windows: list = kwargs.get("vol_ma_windows", (5, 10, 20))
        self.ma_features = []
        self.ma_features_ratio = []
        for w in self.vol_ma_windows:
            self.ma_features.append(f"VOL_MA_{w}")
            self.ma_features_ratio.append(f"VOL_ratio_{w}")
            self.features.extend([f"VOL_MA_{w}", f"VOL_ratio_{w}"])

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        res = {}
        vol = df["volume"]
        for w in self.vol_ma_windows:
            vol_ma = vol.rolling(w).mean()
            res[f"VOL_MA_{w}"] = vol_ma
            res[f"VOL_ratio_{w}"] = np.where(vol_ma > EPS, vol / vol_ma, 0.0)
        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        self._normalize_volume_rlc(X, feature_cols, self.ma_features, "volume")
        self._normalize_z_score_group(X, feature_cols, self.ma_features, factory=factory, method="log", scale=3)
        self._normalize_z_score_group(X, feature_cols, self.ma_features_ratio, factory=factory, method="log")

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        """
        Compute minimal history length for volume MA features.
        Use the max configured window plus a small buffer to stabilize normalization.
        """
        if not self.vol_ma_windows:
            return 0
        # 1. Max window (e.g. 20)
        max_window = max(self.vol_ma_windows)
        # 2. Add buffer (1.5x) so normalization has enough samples.
        return int(max_window * 1.5)


class FeatureQavMa(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 1. Load configuration
        self.windows = kwargs.get("windows", [7, 25, 99])  # multiple window lengths
        self.add_surge = kwargs.get("add_surge", True)  # add surge ratios (intensity)
        self.add_slope = kwargs.get("add_slope", True)  # add trend slope (consistency)
        self.add_bias = kwargs.get("add_bias", True)  # add VWAP bias (cost)
        self.slope_step = kwargs.get("slope_step", 3)  # slope estimation span

        self.surge_features = []
        self.slope_features = []
        self.bias_features = []

        # 2. Pre-register feature names
        for w in self.windows:
            if self.add_surge:
                col = f"QAV_SURGE_{w}"
                self.features.append(col)
                self.surge_features.append(col)

            if self.add_slope:
                col = f"QAV_SLOPE_{w}"
                self.features.append(col)
                self.slope_features.append(col)

        if self.add_bias:
            # VWAP bias usually needs only one dimension (deviation vs traded VWAP)
            col = "VWAP_BIAS"
            self.features.append(col)
            self.bias_features.append(col)

    def _slope_reg_vectorized(self, series: pd.Series, steps: int) -> pd.Series:
        """
        Vectorized least-squares regression slope (without log transform).
        Fit over the full window to reduce endpoint noise.
        """
        if steps <= 1:
            return pd.Series(np.nan, index=series.index)

        n = float(steps)
        # x axis: [0, 1, 2, ..., n-1]
        x_mean = (n - 1) / 2.0
        var_x = (n * (n**2 - 1)) / 12.0  # variance of x

        # Vectorized y statistics
        y_filled = series.fillna(0)
        s1 = y_filled.cumsum()
        s2 = s1.cumsum()

        sum_y = s1 - s1.shift(steps)
        shift_s1 = s1.shift(steps)
        shift_s2 = s2.shift(steps)

        # Compute sum(x*y) via cumulative sums
        weighted_sum_rev = (s2 - shift_s2) - steps * shift_s1
        sum_xy = (steps * sum_y) - weighted_sum_rev

        # Slope formula: beta = Cov(x, y) / Var(x)
        y_mean = sum_y / n
        slope = (sum_xy - n * x_mean * y_mean) / var_x

        # Normalize slope by mean so relative changes are comparable across price/volume levels.
        return slope / (y_mean + EPS)

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        qav = df["quote_asset_volume"].astype(float)
        vol = df["volume"].astype(float)
        close = df["close"].astype(float)
        res = {}

        for w in self.windows:
            ma_qav = qav.rolling(w).mean()
            if self.add_surge:
                res[f"QAV_SURGE_{w}"] = np.where(ma_qav > EPS, qav / ma_qav, 1.0)
            if self.add_slope:
                res[f"QAV_SLOPE_{w}"] = self._slope_reg_vectorized(ma_qav, self.slope_step)

        if self.add_bias:
            vwap = np.where(vol > EPS, qav / vol, 0.0)
            res["VWAP_BIAS"] = np.where(vwap > EPS, (close / vwap) - 1.0, 0.0)

        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        # 1. Surge ratio group: heavy-tailed and volatile; use log squashing
        if self.surge_features:
            for f in self.surge_features:
                self._normalize_z_score_rel(X, feature_cols, [f], factory=factory, feature_base=f, method="log")

        # 2. Trend slope group: after log1p, slopes are stable; regular normalization works
        if self.slope_features:
            # Slopes can be group-normalized together since they share the same scale (percentage change)
            self._normalize_z_score_rel(X, feature_cols, self.slope_features, feature_base=self.slope_features[0], factory=factory, method="log")

        # 3. VWAP bias group: percentage-like; normalize separately (typically within [-1, 1])
        if self.bias_features:
            self._normalize_z_score_rel(X, feature_cols, self.bias_features, feature_base="VWAP_BIAS", factory=factory, method="log")

        # 4. Optional physical guardrails (either here or in FeatureFactory)
        # X[:, :, indices] = np.clip(X[:, :, indices], -5.0, 5.0)

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        # Need max window + slope step as warm-up
        max_w = max(self.windows) if self.windows else 0
        return int((max_w + self.slope_step) * 1.5)


# ==== 4. OBV ====
class FeatureOBV(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = ["OBV"]

    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        close = df["close"]
        vol = df["volume"].astype(float)

        # 1. Basic direction logic
        sign = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
        obv_raw = (sign * vol).cumsum()

        # 2. Non-dimensionalization should be handled in normalize with the Factory context.
        # We keep raw values here and normalize later for consistency.
        df["OBV"] = obv_raw

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        self._normalize_z_score_rel(X, feature_cols, self.features, feature_base=self.features[0], factory=factory, method="log")  # Self-Normalization

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        """
        Minimal history length for OBV.
        OBV is cumulative; to stabilize normalization (mean/std), provide at least 2x the model window of history.
        """
        model_window = getattr(self, "window", 100)

        # Enough history helps the cumulative series stabilize for normalization
        return int(model_window * 2)


# ==== PVT ====
class FeaturePVT(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = ["PVT"]

    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        # 1. Price pct_change
        # Note: df['close'] must be float to avoid dtype issues
        close = df["close"].astype(float)
        volume = df["quote_asset_volume"].astype(float)  # prefer QAV over raw volume

        pct = close.pct_change().fillna(0)

        # 2. Increment: return * volume proxy
        incremental_pvt = pct * volume

        # 3. Cumulative sum to get standard PVT (a long-memory series)
        pvt_raw = incremental_pvt.cumsum()

        # 4. Store to DataFrame
        df["PVT"] = pvt_raw

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        # Finally apply symmetric scaling to roughly [-0.5, 0.5].
        # After log1p, useful information tends to spread more evenly within this range.
        self._normalize_z_score_rel(X, feature_cols, self.features, feature_base="quote_asset_volume", factory=factory)  # Self-Normalization

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        """
        Minimal history length for PVT.
        PVT is cumulative; sufficient samples are needed for stable normalization statistics.
        """
        # Use training window size as reference (often ~100). If not available, use a conservative default.
        model_window = getattr(self, "window", 100)
        # 2x window ensures enough history before the first inference window to form initial stats.
        return int(model_window * 2)


# ==== VWAP (rolling) ====
class FeatureWAP(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Window config: defaults to a 7/25/99 style (or user-specified like 20/48/96)
        self.vwap_windows: list = kwargs.get("vwap_windows", (20, 48, 96))
        self.add_bias: bool = kwargs.get("add_bias", True)  # whether to add bias features

        self.absolute_features = []  # price-dimension features
        self.ratio_features = []  # bias percentage-like features

        for w in self.vwap_windows:
            # 1. Register raw VWAP
            vwap_col = f"VWAP_{w}"
            self.features.append(vwap_col)
            self.absolute_features.append(vwap_col)

            # 2. Register VWAP bias
            if self.add_bias:
                bias_col = f"VWAP_Bias_{w}"
                self.features.append(bias_col)
                self.ratio_features.append(bias_col)

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        qav, vol, close = df["quote_asset_volume"].astype(float), df["volume"].astype(float), df["close"].astype(float)
        res = {}
        for w in self.vwap_windows:
            rolling_qav = qav.rolling(w).sum()
            rolling_vol = vol.rolling(w).sum()
            vwap_series = np.where(rolling_vol > EPS, rolling_qav / rolling_vol, close)
            res[f"VWAP_{w}"] = vwap_series
            if self.add_bias:
                res[f"VWAP_Bias_{w}"] = np.where(vwap_series > EPS, (close / vwap_series) - 1.0, 0.0)
        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        Normalize with physical meaning alignment.
        """
        # 1. Raw VWAP: price-dimension; must be anchored to close
        if self.absolute_features:
            self._normalize_z_score_rel(X, feature_cols, self.absolute_features, feature_base="close", factory=factory, method="log")

        # 2. VWAP bias: percentage-like and stable; normalize separately
        if self.ratio_features:
            self._normalize_z_score_rel(X, feature_cols, self.ratio_features, feature_base=self.ratio_features[-1], factory=factory, method="log")
            # for f in self.ratio_features:
            #     # Bias is usually within [-0.1, 0.1]; self-scaling often suffices
            #     self._normalize_z_score_rel(X, feature_cols, [f], f, factory=factory)

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        """
        Minimal history for rolling VWAP.
        Use the max configured window and add a buffer to stabilize early live normalization.
        """
        if not self.vwap_windows:
            return 0

        # 1. Max window (e.g. 96)
        max_window = max(self.vwap_windows)

        # 2. Add buffer (suggested 1.5x) so rolling sums are complete and normalization has enough background samples
        return int(max_window * 1.5)


# ==== CMF ====
# If price closes near the high with increasing volume, it may indicate accumulation; the opposite may indicate distribution.
class FeatureCFM(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Supports multiple windows, e.g. (10, 20, 60)
        self.cmf_windows: list = kwargs.get("cmf_windows", [10, 20, 60])

        self.ratio_features = []
        for w in self.cmf_windows:
            col = f"CMF_{w}"
            self.features.append(col)
            self.ratio_features.append(col)

    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)  # for more advanced use, consider quote_asset_volume

        range_hl = high - low

        # 1. Money Flow Multiplier (MFM)
        # Logic: ((close-low) - (high-close)) / (high-low)
        # This measures the close position within the candle range.
        mfm = np.where(range_hl > EPS, ((close - low) - (high - close)) / range_hl, 0.0)

        # 2. Money Flow Volume (MFV)
        mfv = mfm * volume

        # 3. Compute CMF across windows
        for w in self.cmf_windows:
            mfv_sum = mfv.rolling(w).sum()
            vol_sum = volume.rolling(w).sum()

            # CMF = sum(MFV) / sum(volume) over the window, in [-1, 1]
            df[f"CMF_{w}"] = np.where(vol_sum > EPS, mfv_sum / vol_sum, 0.0)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        CMF normalization notes:
        CMF is already dimensionless ([-1.0, 1.0]) and typically centered near 0.
        Self-scaling normalization helps capture oscillatory overbought/oversold signals.
        """
        # self._normalize_z_score_rel(...)  # based on tests, per-feature normalization may work better
        if self.ratio_features:
            for f in self.ratio_features:
                # Oscillators do not need anchoring to close; per-feature z-score is sufficient
                self._normalize_z_score_rel(X, feature_cols, [f], feature_base=f, factory=factory, method="log")

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        # Use the largest window plus a buffer
        max_w = max(self.cmf_windows) if self.cmf_windows else 20
        return int(max_w * 1.5)


# ==== MFI: measures money inflow/outflow and its strength ====
class FeatureMFI(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Multi-window configuration
        self.mfi_windows: list = kwargs.get("mfi_windows", [14, 25, 99])

        self.ratio_features = []
        for w in self.mfi_windows:
            col = f"MFI_{w}"
            self.features.append(col)
            self.ratio_features.append(col)

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        high, low, close = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
        volume = df["quote_asset_volume"].astype(float)
        res = {}

        tp = (high + low + close) / 3.0
        mf = tp * volume
        tp_diff = tp.diff()

        pos_mf = pd.Series(np.where(tp_diff > 0, mf, 0.0), index=df.index)
        neg_mf = pd.Series(np.where(tp_diff < 0, mf, 0.0), index=df.index)

        for w in self.mfi_windows:
            p_sum = pos_mf.rolling(w).sum()
            n_sum = neg_mf.rolling(w).sum()
            total_mf = p_sum + n_sum
            res[f"MFI_{w}"] = np.divide(100.0 * p_sum, total_mf, out=np.full_like(p_sum, 50.0), where=total_mf > 0)

        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        Zero-mean symmetric scaling: map [0, 100] to [-0.5, 0.5]
        Helps the model distinguish inflow-dominant vs outflow-dominant regimes.
        """
        for f in self.ratio_features:
            if f in feature_cols:
                idx = feature_cols.index(f)
                # Linear mapping: (val / 100) - 0.5
                X[:, :, idx] = (X[:, :, idx] / 100.0) - 0.5

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        max_w = max(self.mfi_windows) if self.mfi_windows else 14
        return int(max_w * 2.1)


class FeatureATS(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = ["ATS"]

    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        # ==== 2. Average trade size (ATS) ====
        df["ATS"] = np.where(df["number_of_trades"] > EPS, df["volume"] / df["number_of_trades"], 0.0)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        ATS represents per-trade activity/intensity.
        """
        self._normalize_z_score_rel(
            X=X, feature_cols=feature_cols, target_feature_cols=self.features, feature_base="ATS", factory=factory
        )  # must be self-scaled

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        """
        Minimal history needed for ATS.
        Although computed per-bar, z-score normalization needs enough history (at least as long as the model window).
        """
        # Use training window size as reference (often ~100)
        model_window = getattr(self, "window", 100)

        # Provide 1x window as the baseline sample size for normalization
        return int(model_window)


class FeatureAdvancedVol(FeatureBase):
    """
    Advanced volatility estimators (often more efficient than ATR).
    Includes: Parkinson and Garman-Klass.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.windows = kwargs.get("windows", [14, 100])
        self.features = []
        for w in self.windows:
            self.features.append(f"vol_parkinson_{w}")
            self.features.append(f"vol_gk_{w}")

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        o, h, l, c = df["open"].astype(float), df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
        res = {}
        log_hl = np.log(h / (l + EPS)) ** 2
        log_co = np.log(c / (o + EPS)) ** 2
        for w in self.windows:
            const_p = 1.0 / (4.0 * np.log(2.0))
            res[f"vol_parkinson_{w}"] = np.sqrt(const_p * log_hl.rolling(w).mean())
            const_gk = 2.0 * np.log(2.0) - 1.0
            gk_term = 0.5 * log_hl - const_gk * log_co
            res[f"vol_gk_{w}"] = np.sqrt(gk_term.rolling(w).mean())
        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        # Volatility features often have heavy tails
        # Use z-score + log squashing so the model can better detect volatility spikes
        for f in self.features:
            if f in feature_cols:
                self._normalize_z_score(X, feature_cols, [f], feature_base=f, factory=factory, method="log")

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        return int(max(self.windows) * 1.5)


class FeatureFractalPersistence(FeatureBase):
    """
    Fractal/persistence indicators: quantify trend purity and memory.
    Includes: Hurst exponent (variance-time scaling) and Efficiency Ratio (ER / Kaufman).

    - ER: net displacement / total path length in [0, 1]. 1 => pure trend, 0 => choppy/range.
    - Hurst: rolling estimate via variance-time scaling. H>0.5 trend persistence, H<0.5 mean reversion, H=0.5 random walk.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Suggested windows: 14 (short-term sensitivity), 126 (swing/longer-term baseline)
        self.windows = kwargs.get("windows", [14, 126])
        self.features = []
        for w in self.windows:
            self.features.append(f"hurst_{w}")
            self.features.append(f"er_{w}")

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        close = df["close"].astype(float)
        diff = close.diff().abs()
        res = {}
        for w in self.windows:
            # ER
            net_change = (close - close.shift(w)).abs()
            total_path = diff.rolling(w).sum()
            res[f"er_{w}"] = np.where(total_path > EPS, net_change / total_path, 0.0)
            # Hurst
            log_close = np.log(close + EPS)
            std_1 = log_close.diff().rolling(w).std()
            std_w = log_close.diff(w).rolling(w).std()
            ratio = np.where(std_1 > EPS, std_w / (std_1 + EPS), 1.0)
            res[f"hurst_{w}"] = np.clip(np.log(np.maximum(ratio, EPS)) / np.log(max(w, 2)), 0.0, 1.0)
        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        # ER is already in [0, 1]; shift it to [-0.5, 0.5]
        er_indices, _ = self._get_target_indices(feature_cols, [f for f in self.features if "er_" in f])
        if er_indices:
            X[:, :, er_indices] = X[:, :, er_indices] - 0.5

        # Hurst is typically in [0, 1] with a center around 0.5
        hurst_indices, _ = self._get_target_indices(feature_cols, [f for f in self.features if "hurst_" in f])
        if hurst_indices:
            # Shift to [-0.5, 0.5] so 0 corresponds to random walk
            X[:, :, hurst_indices] = X[:, :, hurst_indices] - 0.5

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        return int(max(self.windows) * 2)


class FeatureOrderFlow(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.windows = kwargs.get("windows", [14, 49])
        self.poc_bias_step = kwargs.get("poc_bias_step", [7, 25, 99, 200, 400, 900])

        self.features = []
        for w in self.windows:
            self.features.extend([f"imbalance_{w}", f"vpin_{w}", f"trade_density_{w}"])
        for w in self.poc_bias_step:
            self.features.append(f"poc_bias_{w}")

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        # 1. Base data
        vol = df["volume"].astype(float)
        taker_vol = df["taker_buy_base_volume"].astype(float)
        trades = df["number_of_trades"].astype(float)
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        # Precompute typical price and PV (price * volume) for VWAP
        tp = (high + low + close) / 3.0
        pv = tp * vol

        # Output container
        res = {}

        # 2. Instant buy/sell ratio (vectorized)
        raw_imbalance_series = pd.Series(np.divide(taker_vol, vol, out=np.full_like(vol, 0.5), where=vol > EPS), index=df.index)

        # 3. Compute window features
        for w in self.windows:
            rolling_imb = raw_imbalance_series.rolling(w)
            res[f"imbalance_{w}"] = rolling_imb.mean() - 0.5
            res[f"vpin_{w}"] = rolling_imb.std()

            trades_ma = trades.rolling(w).mean()
            res[f"trade_density_{w}"] = np.divide(trades, trades_ma, out=np.full_like(trades, 1.0), where=trades_ma > EPS)

        # 4. Vectorized POC bias (core optimization)
        for w in self.poc_bias_step:
            # Use vectorized rolling sums instead of apply
            rolling_pv_sum = pv.rolling(w).sum()
            rolling_vol_sum = vol.rolling(w).sum()

            # Rolling VWAP
            vwap = np.divide(rolling_pv_sum, rolling_vol_sum, out=np.full_like(close, np.nan), where=rolling_vol_sum > EPS)

            # Log deviation
            res[f"poc_bias_{w}"] = np.log(close / (vwap + EPS))

        # 5. Convert to DataFrame in one shot to avoid fragmentation warnings
        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        # 1. Imbalance: in [-0.5, 0.5] and centered; apply z-score
        imbalance_feats = [f for f in self.features if "imbalance_" in f]
        self._normalize_z_score_group(X, feature_cols, imbalance_feats, factory, method="tanh")

        # 2. VPIN: non-negative order-flow dispersion; normalize the windows
        # together so their relative magnitudes are retained.
        vpin_feats = [f for f in self.features if "vpin_" in f]
        self._normalize_z_score_group(X, feature_cols, vpin_feats, factory, method="tanh")

        # 3. Trade density is a ratio whose neutral value is 1. Use a bounded
        # log-ratio so 1 -> 0, reciprocal deviations are symmetric, and a zero
        # trade count cannot create a very large negative input.
        density_features = [f for f in self.features if "trade_density_" in f]
        for f in density_features:
            if f in feature_cols:
                idx = feature_cols.index(f)
                log_ratio = np.log(np.maximum(X[:, :, idx], EPS))
                X[:, :, idx] = np.tanh(log_ratio)

        # 4. POC bias is already log(close / VWAP) when generated. Preserve its
        # zero anchor and only scale the group; applying log again would compress
        # the signal twice.
        poc_feats = [f for f in self.features if "poc_bias_" in f]
        self._normalize_signal_group(X, feature_cols, poc_feats, factory, method=None)

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        return int(int(max(max(self.windows), max(self.poc_bias_step))) * 1.5)


class FeatureClassicFactors(FeatureBase):
    """
    Classic factor features: statistical moments, information dispersion, and extremum position.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.windows = kwargs.get("windows", [20, 100])
        self.features = []
        for w in self.windows:
            self.features.extend([f"skew_{w}", f"kurt_{w}", f"id_factor_{w}", f"dist_to_high_{w}"])

    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        close = df["close"].astype(float)
        returns = close.pct_change()

        for w in self.windows:
            # 1. Moments (skew & kurt)
            # Capture asymmetry and fat-tail properties of returns
            df[f"skew_{w}"] = returns.rolling(w).skew()
            df[f"kurt_{w}"] = returns.rolling(w).kurt()

            # 2. Information dispersion (ID factor)
            # Difference between counts of up vs down bars
            pos_bars = (returns > 0).rolling(w).sum()
            neg_bars = (returns < 0).rolling(w).sum()
            # Normalize by window length
            df[f"id_factor_{w}"] = (pos_bars - neg_bars) / w

            # 3. Extremum position (distance to high)
            # Use log distance to keep scale consistent
            rolling_high = df["high"].rolling(w).max()
            df[f"dist_to_high_{w}"] = np.log(rolling_high / (close + EPS))

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        # 1. Skew/kurt: typically around [-3, 3]; z-score is sufficient
        skew_kurt = [f for f in self.features if "skew" in f or "kurt" in f]
        self._normalize_z_score_group(X, feature_cols, skew_kurt, factory, method="tanh")

        # 2. ID factor: naturally in [-1, 1] with 0 as balance
        id_feats = [f for f in self.features if "id_factor" in f]
        # No shift needed; z-score helps enhance signal strength
        self._normalize_signal_group(X, feature_cols, id_feats, factory, method=None)

        # 3. Dist to high: positive and long-tailed; apply log compression
        dist_feats = [f for f in self.features if "dist_to_high" in f]
        self._normalize_signal_group(X, feature_cols, dist_feats, factory, method="log")

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        # Moments need enough samples (recommend >= 20) for stability
        return int(max(self.windows) * 1.5)


class FeatureMomentum(FeatureBase):
    """
    Classic factor-style momentum features.

    Features:
      1) MOM_h              = log(close / close.shift(h))
      2) MOM_h_SKIPk        = log(close.shift(k) / close.shift(h+k))   (default k=1)
      3) MOM_h_RV{v}        = MOM_h / (RV_v + EPS), RV_v = rolling std of log returns

    Notes:
      - All are causal (<= t).
      - Designed as factor-type features: directly tied to realized returns.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # --- config ---
        self.horizons: list[int] = kwargs.get("horizons", [5, 10, 20, 60])
        self.include_skip: bool = kwargs.get("include_skip", True)
        self.skip_k: int = kwargs.get("skip_k", 1)
        self.skip_horizon: int | None = kwargs.get("skip_horizon", 20)  # only build skip for this horizon; None => for all

        self.include_vol_adj: bool = kwargs.get("include_vol_adj", True)
        self.vol_window: int = kwargs.get("vol_window", 20)
        self.vol_adj_horizon: int | None = kwargs.get("vol_adj_horizon", 20)  # only build vol-adj for this horizon; None => for all

        # --- feature names ---
        self.mom_cols = [f"MOM_{h}" for h in self.horizons]

        self.skip_cols = []
        if self.include_skip:
            if self.skip_horizon is None:
                self.skip_cols = [f"MOM_{h}_SKIP{self.skip_k}" for h in self.horizons]
            else:
                self.skip_cols = [f"MOM_{self.skip_horizon}_SKIP{self.skip_k}"]

        self.vol_adj_cols = []
        if self.include_vol_adj:
            if self.vol_adj_horizon is None:
                self.vol_adj_cols = [f"MOM_{h}_RV{self.vol_window}" for h in self.horizons]
            else:
                self.vol_adj_cols = [f"MOM_{self.vol_adj_horizon}_RV{self.vol_window}"]

        self.features = self.mom_cols + self.skip_cols + self.vol_adj_cols

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        close = df["close"].astype(float)
        log_close = np.log(close + EPS)
        res = {}
        for h in self.horizons:
            res[f"MOM_{h}"] = log_close - log_close.shift(h)
        if self.include_skip:
            k = int(self.skip_k)
            for h in (self.horizons if self.skip_horizon is None else [self.skip_horizon]):
                res[f"MOM_{h}_SKIP{k}"] = log_close.shift(k) - log_close.shift(h + k)
        if self.include_vol_adj:
            rv = log_close.diff().rolling(window=self.vol_window).std()
            for h in (self.horizons if self.vol_adj_horizon is None else [self.vol_adj_horizon]):
                res[f"MOM_{h}_RV{self.vol_window}"] = np.where(rv > EPS, res[f"MOM_{h}"] / (rv + EPS), np.nan)
        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        Momentum factors are 0-centered and symmetric (log-returns / ratios).
        Recommended: RMS-based zero-anchored group scaling + symmetric log squashing,
        preserving relative strength across horizons.
        """
        # 1) plain momentum as one group
        self._normalize_signal_group(X, feature_cols, self.mom_cols, factory=factory, scale=1.0, method="log")

        # 2) skip momentum (if any) - keep separate group (slightly different distribution)
        if self.skip_cols:
            self._normalize_signal_group(X, feature_cols, self.skip_cols, factory=factory, scale=1.0, method="log")

        # 3) vol-adjusted momentum (if any) - separate group
        if self.vol_adj_cols:
            self._normalize_signal_group(X, feature_cols, self.vol_adj_cols, factory=factory, scale=1.0, method="log")

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        """
        Minimal history length:
          - plain MOM: max(horizons)
          - skip MOM: max(h+skip_k)
          - vol adj: needs vol_window + 1 (for diff) and horizon itself
        """
        max_h = max(self.horizons) if self.horizons else 1

        need = max_h

        if self.include_skip:
            k = int(self.skip_k)
            if self.skip_horizon is None:
                need = max(need, max_h + k)
            else:
                need = max(need, int(self.skip_horizon) + k)

        if self.include_vol_adj:
            # rv needs vol_window bars of log_ret, log_ret needs 1 extra
            need = max(need, int(self.vol_window) + 1)
            if self.vol_adj_horizon is None:
                need = max(need, max_h)
            else:
                need = max(need, int(self.vol_adj_horizon))

        # buffer
        return int(need + 2)


@njit(float64[:](float64[:], int64, int64), cache=True)
def calc_rolling_top_k_reference(data, window, k):
    n = len(data)
    out = np.full(n, np.nan)

    # Force k as int to avoid Numba type inference issues
    k_int = int(k)

    for i in range(window, n):
        # 1. Slice window data explicitly
        window_data = data[i - window : i]

        # 2. Use partition to get top-k elements
        # Assign to a variable to help Numba lock types
        top_k_elements = np.partition(window_data, -k_int)[-k_int:]

        # 3. Manual accumulation for mean to avoid potential issues with top_k_elements.mean() in Numba
        accumulator = 0.0
        for val in top_k_elements:
            accumulator += val

        out[i] = accumulator / k_int

    return out


class FeatureVolumeEvent(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.windows = kwargs.get("windows", [5000, 1500, 500])
        self.top_k = kwargs.get("top_k", 3)
        # Dynamically register all features
        self.features = []
        for w in self.windows:
            self.features.append(f"vol_event_flag_{w}")

    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        # Ensure input is a float64 numpy array
        vol_values = df["volume"].values.astype(np.float64)

        for w in self.windows:
            flag_col = f"vol_event_flag_{w}"

            # Call the JIT function optimized with a manual loop
            v_ref_raw_values = calc_rolling_top_k_reference(vol_values, w, self.top_k)

            # Convert to Series for subsequent EWM smoothing
            v_ref_raw = pd.Series(v_ref_raw_values, index=df.index)
            # Slightly larger alpha makes the reference decay faster, improving responsiveness
            v_ref = v_ref_raw.ewm(alpha=1 / (w * 2), adjust=False).mean()

            # This computation is done in pandas/numpy and is stable
            ratio_col = np.where(v_ref > EPS, df["volume"] / v_ref, np.nan)
            df[flag_col] = (ratio_col >= 1.0).astype(int)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        # Event-level feature.
        # Do NOT normalize / z-score / center.
        pass

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        # Must return the largest window to ensure enough initialization history
        return int(max(self.windows) * 1.2)


class FeatureCandle(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 1. Magnitude features: price-dependent and heavily long-tailed
        self.feat_magnitude = ["body", "upper_wick", "lower_wick", "max_range", "body_mom"]
        # 2. Ratio features: naturally in [0, 1]
        self.feat_ratio = ["body_pct", "upper_wick_pct", "lower_wick_pct", "close_pos", "doji_score"]
        # 3. Score features: directional, typically in [-1, 1]
        self.feat_score = ["wick_bias"]

        self.features = self.feat_magnitude + self.feat_ratio + self.feat_score

    def generate(self, df: pd.DataFrame, kline_interval_ms: int = None) -> pd.DataFrame:
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        res = {}

        # Magnitudes
        res["body"] = np.abs(c - o)
        res["upper_wick"] = h - np.maximum(o, c)
        res["lower_wick"] = np.minimum(o, c) - l
        res["max_range"] = h - l
        res["body_mom"] = pd.Series(res["body"]).diff().fillna(0)

        # Ratios
        rng = res["max_range"]
        res["body_pct"] = np.where(rng > 0, res["body"] / rng, 0.0)
        res["upper_wick_pct"] = np.where(rng > 0, res["upper_wick"] / rng, 0.0)
        res["lower_wick_pct"] = np.where(rng > 0, res["lower_wick"] / rng, 0.0)
        res["close_pos"] = np.where(rng > 0, (c - l) / rng, 0.5)
        res["doji_score"] = 1.0 - res["body_pct"]
        res["wick_bias"] = res["upper_wick_pct"] - res["lower_wick_pct"]

        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        Layered compression strategy:
        - Magnitude: price-normalize -> group z-score -> log1p compression
        - Ratio: center to [-0.5, 0.5] -> optional log enhancement
        """
        self._normalize_z_score_group(X, feature_cols, self.feat_magnitude, factory, method="log")

        # 2. Ratio features
        # self._normalize_signal_group(X,feature_cols,self.feat_ratio, factory, method = 'log')
        # 2. Ratio features: shift to [-0.5, 0.5]
        # Column indices for ratio features
        ratio_indices = [feature_cols.index(f) for f in self.feat_ratio if f in feature_cols]

        if ratio_indices:
            # Shift: [0, 1] -> [-0.5, 0.5]
            X[:, :, ratio_indices] = X[:, :, ratio_indices] - 0.5
        # 3. Score features (e.g. wick_bias)

        self._normalize_z_score(X, feature_cols, self.feat_score, self.feat_score[0], factory)

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        """
        Minimal history needed for candlestick-shape features.
        Although most are per-bar, gap/body_mom depend on previous bar,
        and z-score normalization needs enough background samples.
        """
        # 1. Base dependency: diff/shift needs at least 2 bars
        base_dependency = 2

        # 2. Normalization stability: use model window as reference. If absent, use a conservative default.
        model_window = getattr(self, "window", 100)

        # Return model window size so z-score has enough samples
        return max(base_dependency, int(model_window))


class FeatureDonchian(FeatureBase):
    """
    Multi-period Donchian channels:
    accept a list of periods and generate both price skeleton and derived ratio/shape features.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Accept list input; default is [20]
        self.periods = kwargs.get("periods", [20])

        self.price_features = []
        self.ratio_features = []
        self.pos_features = []  # track POS features for centering

        # Build feature list dynamically
        for p in self.periods:
            # 1. Price features
            self.price_features.extend([f"DONCHIAN_UPPER_{p}", f"DONCHIAN_LOWER_{p}", f"DONCHIAN_MIDDLE_{p}"])
            # 2. Ratio/shape features
            p_pos = f"DONCHIAN_POS_{p}"
            self.pos_features.append(p_pos)
            self.ratio_features.extend([p_pos, f"DONCHIAN_BW_{p}", f"DONCHIAN_DIST_U_{p}", f"DONCHIAN_DIST_L_{p}"])

        self.features = self.price_features + self.ratio_features

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        high, low, close = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
        res = {}
        for p in self.periods:
            upper, lower = high.rolling(p).max(), low.rolling(p).min()
            middle = (upper + lower) / 2
            range_hl = upper - lower
            res[f"DONCHIAN_UPPER_{p}"], res[f"DONCHIAN_LOWER_{p}"], res[f"DONCHIAN_MIDDLE_{p}"] = upper, lower, middle
            res[f"DONCHIAN_POS_{p}"] = np.where(range_hl > EPS, (close - lower) / range_hl, 0.5)
            res[f"DONCHIAN_BW_{p}"] = np.where(middle > EPS, range_hl / middle, 0.0)
            res[f"DONCHIAN_DIST_U_{p}"] = (upper - close) / (close + EPS)
            res[f"DONCHIAN_DIST_L_{p}"] = (close - lower) / (close + EPS)
        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        Use Factory group-normalization to process all periods in one pass.
        """
        # 1. Price channels: anchor Upper/Lower/Middle to close for relative normalization
        if self.price_features:
            self._normalize_z_score_rel(X, feature_cols, self.price_features, feature_base="close", factory=factory, method="log")

        # 2. POS: center to [-0.5, 0.5]
        # 0 => middle band, positive => upper half, negative => lower half
        pos_indices = [factory._feature_index[f] for f in self.pos_features if f in factory._feature_index]
        if pos_indices:
            X[:, :, pos_indices] = X[:, :, pos_indices] - 0.5

        # 3. Other ratios (BW, DIST): signal-group normalization + log compression
        # This makes squeeze intensity comparable across periods
        other_ratios = [f for f in self.ratio_features if f not in self.pos_features]
        if other_ratios:
            self._normalize_signal_group(X, feature_cols, other_ratios, factory=factory, method="log")

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        # Use a multiple of the max period to stabilize normalization stats
        return int(max(self.periods) * 1.5)


class FeatureKeltner(FeatureBase):
    """
    Keltner channels:
    Middle Band = EMA(Close, N)
    Upper Band = Middle + Multiplier * ATR(N)
    Lower Band = Middle - Multiplier * ATR(N)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.period = kwargs.get("period", 20)
        self.multiplier = kwargs.get("multiplier", 2.0)
        self.features = [f"KELTNER_UPPER_{self.period}", f"KELTNER_LOWER_{self.period}", f"KELTNER_MIDDLE_{self.period}"]

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        close, high, low = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
        res = {}
        middle = close.ewm(span=self.period, adjust=False).mean()
        tr = np.maximum((high - low), np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = tr.rolling(window=self.period).mean()
        res[f"KELTNER_UPPER_{self.period}"] = middle + (self.multiplier * atr)
        res[f"KELTNER_LOWER_{self.period}"] = middle - (self.multiplier * atr)
        res[f"KELTNER_MIDDLE_{self.period}"] = middle
        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        # Use relative normalization to preserve channel geometry
        # self._normalize_scs(X, feature_cols, self.features, "close")
        self._normalize_z_score_rel(X, feature_cols, self.features, feature_base="close", factory=factory, method="log")

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        # EMA needs ~4x period for warm-up convergence
        return int(self.period * 4)


class FeatureBoll(FeatureBase):
    """
    Bollinger Bands:
    Middle Band = SMA(Close, N)
    Upper Band = Middle + Multiplier * StdDev(Close, N)
    Lower Band = Middle - Multiplier * StdDev(Close, N)
    Bandwidth = (Upper - Lower) / Middle  (volatility / squeeze proxy)
    %B = (Close - Lower) / (Upper - Lower) (price position within the channel)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.period = kwargs.get("period", 20)
        self.multiplier = kwargs.get("multiplier", 2.0)

        # Price channel features
        self.price_features = [f"BOLL_UPPER_{self.period}", f"BOLL_LOWER_{self.period}", f"BOLL_MIDDLE_{self.period}"]
        # Dimensionless derived features
        self.ratio_features = [f"BOLL_BW_{self.period}", f"BOLL_PB_{self.period}"]  # bandwidth: squeeze intensity proxy  # %B: relative price position
        self.features = self.price_features + self.ratio_features

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        close = df["close"].astype(float)
        res = {}

        middle = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        upper = middle + (self.multiplier * std)
        lower = middle - (self.multiplier * std)

        res[f"BOLL_UPPER_{self.period}"] = upper
        res[f"BOLL_LOWER_{self.period}"] = lower
        res[f"BOLL_MIDDLE_{self.period}"] = middle
        res[f"BOLL_BW_{self.period}"] = (upper - lower) / (middle + EPS)
        res[f"BOLL_PB_{self.period}"] = (close - lower) / (upper - lower + EPS)

        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        # A. Channel prices: align to close with relative normalization
        self._normalize_z_score_rel(X, feature_cols, self.price_features, feature_base="close", factory=factory, method="log")

        # B. %B position: roughly in [0, 1]; center it
        pb_idx = [factory._feature_index[f] for f in [self.ratio_features[1]] if f in factory._feature_index]
        if pb_idx:
            X[:, :, pb_idx] = X[:, :, pb_idx] - 0.5

        # C. Bandwidth: positive and long-tailed; apply log1p compression
        bw_idx = [factory._feature_index[f] for f in [self.ratio_features[0]] if f in factory._feature_index]
        if bw_idx:
            X[:, :, bw_idx] = np.log1p(X[:, :, bw_idx])

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        # SMA base window
        return self.period


class FeatureOrigin(FeatureBase):  # add taker_buy_base_volume/volume and taker_buy_quote_volume/quote_asset_volume ratios
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.price_base_features = [
            "open",
            "high",
            "low",
            "close",
        ]  # []
        self.volume_base_features = ["taker_buy_base_volume", "volume"]  # [] # feature used as basic must be the last!!!
        self.quote_base_features = ["taker_buy_quote_volume", "quote_asset_volume"]  # []   #the basic is quote_asset
        self.self_based_features = ["number_of_trades"]  # []
        self.features = self.price_base_features + self.volume_base_features + self.quote_base_features + self.self_based_features

    def generate(self, df: pd.DataFrame, kline_interval_ms: int = None):
        # for f in self.factory.base_features:
        #     df[f'base_{f}'] = df[f]
        pass

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        self._normalize_z_score_rel(X, feature_cols, self.price_base_features, feature_base=self.price_base_features[-1], factory=factory, method="log")
        self._normalize_z_score_rel(X, feature_cols, self.volume_base_features, feature_base=self.volume_base_features[-1], factory=factory, method="log")
        self._normalize_z_score_rel(X, feature_cols, self.quote_base_features, feature_base=self.quote_base_features[-1], factory=factory, method="log")
        # self._normalize_z_score_group(X, feature_cols , self.price_base_features , factory= factory, method= 'log')
        # self._normalize_z_score_group(X, feature_cols , self.volume_base_features , factory= factory, method= 'log')
        # self._normalize_z_score_group(X, feature_cols , self.quote_base_features , factory= factory, method= 'log')#_normalize_z_score_group
        for f in self.self_based_features:
            self._normalize_z_score(X, feature_cols, [f], feature_base=f, factory=factory, method="log")

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        return 1


# MA / momentum / structure
# --- 1. Price trend & indicator features ---
FCVolumeEvent = FeatureContainer(FeatureVolumeEvent, **{"windows": [5000, 1500, 1000, 500, 200, 100], "top_k": 3})
FCMACD = FeatureContainer(FeatureMACD, **{"fast": 12, "slow": 26, "signal": 9})
FCMA = FeatureContainer(
    FeatureMAStructure,
    bar_windows=(7, 21, 63),
    day_windows=(5, 20),
    week_windows=(7, 25),
    add_delta=True,
    method="sma",
    strict=True,
)
FCRSI = FeatureContainer(FeatureRsi, **{"period": 14, "price_col": "close", "strict": True, "prefix": "RSI"})
FCKDJ = FeatureContainer(FeatureKdj, **{"n": 9, "m1": 3, "m2": 3, "strict": True, "prefix": "KDJ"})

# --- 2. Price channel features ---
FCDonchian = FeatureContainer(FeatureDonchian, **{"periods": [7, 25, 99]})
FCKeltner = FeatureContainer(FeatureKeltner, **{"period": 14, "multiplier": 2})
FCBoll = FeatureContainer(FeatureBoll, **{"period": 25})

# --- 3. Volume & activity features ---
FCVolMa = FeatureContainer(FeatureVolMa, **{"vol_ma_windows": [7, 14, 25, 99]})
FCQavMa = FeatureContainer(FeatureQavMa, **{"windows": [7, 25, 99]})
FCOBV = FeatureContainer(FeatureOBV, **{})
FCPVT = FeatureContainer(FeaturePVT, **{})
FCWAP = FeatureContainer(FeatureWAP, **{"vwap_windows": [7, 25, 99], "add_bias": True})
FCCFM = FeatureContainer(FeatureCFM, **{"cmf_windows": [7, 25, 99]})
FCMFI = FeatureContainer(FeatureMFI, **{"mfi_windows": [7, 25, 99, 200, 500, 999]})
FCATS = FeatureContainer(FeatureATS, **{})

FCAdvancedVol = FeatureContainer(FeatureAdvancedVol, windows=[14, 100])
FCFractalPersistence = FeatureContainer(FeatureFractalPersistence, windows=[14, 126])
FCOrderFlow = FeatureContainer(FeatureOrderFlow, windows=[14, 49], poc_bias_step=[7, 25, 99, 200, 300, 400, 500, 600, 700, 800, 900], include_poc_bias=True)
FCOrderClassicFactors = FeatureContainer(FeatureClassicFactors, windows=[20, 100])
FCOrderMomentum = FeatureContainer(
    FeatureMomentum, horizons=[10, 20, 60], include_skip=True, skip_horizon=20, include_vol_adj=True, vol_adj_horizon=20, vol_window=20
)


FCATR = FeatureContainer(FeatureATRRegime, windows=[14])  # 14, 16, 1000 , 2000, 5000

# --- 4. Candlestick shapes & raw fields ---
FCCandle = FeatureContainer(FeatureCandle, **{})
FCOrigin = FeatureContainer(FeatureOrigin, **{})

# ==============================================================================
# 2. Final feature group list (FEATURE_GROUP_LIST)
# ==============================================================================

FEATURE_GROUP_LIST = [
    # 1. Custom volume event features
    FCVolumeEvent,
    # 2. Price trend & indicators
    FCMACD,  # (12,26,9), (6,13,5), or (10,20,7)
    FCMA,  # pair with slope features
    FCRSI,
    FCKDJ,
    # Price channels (pick your set)
    FCDonchian,
    FCKeltner,
    FCBoll,
    # 3. Volume & activity features
    FCVolMa,
    FCQavMa,
    # FCOBV,    # similar to FeaturePVT but loses magnitude; prefer PVT
    FCPVT,  # cumulative; often less useful for short-horizon prediction than momentum
    FCWAP,
    FCCFM,
    FCMFI,
    # FCATS,  # negative impact in some tests
    FCAdvancedVol,
    FCFractalPersistence,
    FCOrderFlow,
    FCOrderClassicFactors,
    FCOrderMomentum,
    FCATR,
    # 4. Candlestick shape features
    FCCandle,
    FCOrigin,
]


class FeatureFactory:
    def __init__(self, kline_interval_ms: int, feature_group_list: list[FeatureContainer] = FEATURE_GROUP_LIST, feature_conf_list=[]):
        self.all_feature_list = []  # feature names
        self.selected_feature_list = []  # feature names
        self.price_features = {}
        self._kline_interval_ms = kline_interval_ms
        self._X = None
        self._feature_index = None
        self._base_stats_pool = None
        self.feature_group_list: list[FeatureBase] = []
        self.base_features = [
            "open",
            "high",
            "low",
            "close",
            "taker_buy_base_volume",
            "volume",
            "taker_buy_quote_volume",
            "quote_asset_volume",
            "number_of_trades",
        ]
        for container in feature_group_list:
            instance = container.feature(factory=self, kline_interval_ms=kline_interval_ms, **container.parameters)
            self.feature_group_list.append(instance)
            self.all_feature_list.extend(instance.features)
        if feature_conf_list:
            filtered_groups = []
            for group in self.feature_group_list:
                for f in feature_conf_list:
                    if f in group.features:
                        filtered_groups.append(group)
                        break
            self.feature_group_list = filtered_groups

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        # Collect DataFrame blocks generated by each feature group
        feature_blocks = [df]

        for f in self.feature_group_list:
            # Get generated feature block for this group
            # Note: each feature generator should return a DataFrame (or dict-like that becomes a DataFrame)
            res_block = f.generate(df, self._kline_interval_ms)
            if res_block is not None and not res_block.empty:
                feature_blocks.append(res_block)
        # Core: concatenate horizontally in one shot (fastest for high-dimensional features)
        combined_df = pd.concat(feature_blocks, axis=1)
        # If you worry about memory fragmentation affecting later normalization, keep the copy()
        return combined_df.copy()

    def _prepare_normalize_context(self, X, feature_cols):
        self._X = X
        self._feature_cols = tuple(feature_cols)
        self._feature_index = {f: i for i, f in enumerate(feature_cols)}
        self._base_stats_pool = {}

    def get_base_stats(self, base_feature):
        if self._X is None or self._feature_index is None:
            raise RuntimeError("prepare_normalize_context() must be called first")

        base_idx = self._feature_index[base_feature] if isinstance(base_feature, str) else base_feature

        if base_idx not in self._base_stats_pool:
            base = self._X[:, :, base_idx]
            mu = np.nanmean(base, axis=1, keepdims=True)
            sigma = np.nanstd(base, axis=1, keepdims=True)
            denom = sigma  # + 0.1 * np.abs(mu) + EPS
            self._base_stats_pool[base_idx] = (mu, denom)

        return self._base_stats_pool[base_idx]

    def fit_group_magnitude(self, df_list, target_cols, base_col):
        """
        Precompute group statistics on the training set.
        df_list: list of training DataFrames
        target_cols: ['body', 'upper_wick', 'lower_wick', ...]
        base_col: 'close'
        """
        all_ratios = []
        for df in df_list:
            # 1. Compute ratios: X / close
            # You can also use a rolling mean as base, depending on your generate logic.
            base_price = df[base_col].replace(0, np.nan)
            for col in target_cols:
                ratio = (df[col] / base_price).dropna()
                all_ratios.append(ratio.values)

        # 2. Pool all ratio values to compute global stats
        combined_ratios = np.concatenate(all_ratios)
        group_mu = np.mean(combined_ratios)
        group_std = np.std(combined_ratios)

        # 3. Store results
        group_key = "_".join(sorted(target_cols))
        self.group_stats[group_key] = (group_mu, group_std)
        return group_mu, group_std

    def normalize(self, X: np.ndarray, feature_cols: list[str]):
        self._prepare_normalize_context(X, feature_cols)
        for group in self.feature_group_list:
            if any(f in feature_cols for f in group.features):
                group.normalize(X, feature_cols, self)

    def get_global_min_history(self) -> int:
        """Return the maximum min-history requirement across all registered features."""
        return max([f.min_history_request(self._kline_interval_ms) for f in self.feature_group_list])


FEATURE_LIST_COMMODITY = [
    # =========================
    # Raw Market State
    # =========================
    "open",
    "high",
    "low",
    "close",
    "volume",
    # =========================
    # 1) Trend / directional persistence: whether price has continuation
    # =========================
    # "MA_WEEK_M_L",        # Long-term regime direction (core)
    # "PVT",                # Price-volume enhanced momentum
    "dist_to_high_100",  # Breakout-style trend structure
    "id_factor_100",
    "id_factor_20",
    # "MFI_999",            # Extreme money flow
    # "MFI_99",
    # =========================
    # 2) Volatility regime: amplitude and risk environment
    # =========================
    "vol_gk_100",  # Long-term volatility
    "vol_gk_14",  # Short-term volatility
    "skew_100",
    "kurt_100",  # Tail structure (extreme risk)
    # "BOLL_BW_25",         # Decide after uplift testing
    "RSI_14",  # Relative Strength Index (momentum/overheat; indirectly reflects volatility)
    # =========================
    # 3) Efficiency / market structure: trending vs ranging
    # =========================
    "er_126",  # Trend efficiency ratio (high-quality structure factor)
    # =========================
    # 4) Participation / liquidity: market activity
    # =========================
    # "trade_density_14",    # Continuous participation intensity
    # "vol_event_flag_500",  # Extreme volume event (regime trigger)
    # =========================
    # 5) Order flow / imbalance: buy-vs-sell dominance
    # =========================
    # "vpin_49",             # Mid-term order-flow imbalance
    # "vpin_14",             # Needs uplift testing
    # =========================
    # 6) Spatial / price position: where price sits within ranges/cost
    # =========================
    # "poc_bias_600",        # Deviation from high-volume node (strong structural anchor)
    # "poc_bias_99",
    # "close_pos",           # Relative position within range
    # =========================
    # 7) Candlestick / path microstructure
    # =========================
    "upper_wick_pct",
    "lower_wick_pct",
]

FEATURE_LIST_CRYPTOCURRENCY = [
    # =========================
    # Raw Market State
    # =========================
    "open",
    "high",
    "low",
    "close",
    "volume",
    "number_of_trades",  # CRYPTO_ONLY: not enabled in FEATURE_LIST_COMMODITY
    "quote_asset_volume",  # CRYPTO_ONLY: not enabled in FEATURE_LIST_COMMODITY
    "taker_buy_base_volume",  # CRYPTO_ONLY: not enabled in FEATURE_LIST_COMMODITY
    "taker_buy_quote_volume",  # CRYPTO_ONLY: not enabled in FEATURE_LIST_COMMODITY
    # =========================
    # 1) Trend / directional persistence: whether price has continuation
    # =========================
    # "MA_WEEK_M_L",        # Long-term regime direction (core)
    "PVT",  # CRYPTO_ONLY: not enabled in FEATURE_LIST_COMMODITY
    "dist_to_high_100",  # Breakout-style trend structure
    "id_factor_100",
    "id_factor_20",
    "MFI_999",  # CRYPTO_ONLY: not enabled in FEATURE_LIST_COMMODITY
    "MFI_99",  # CRYPTO_ONLY: not enabled in FEATURE_LIST_COMMODITY
    # =========================
    # 2) Volatility regime: amplitude and risk environment
    # =========================
    "vol_gk_100",  # Long-term volatility
    "vol_gk_14",  # Short-term volatility
    "skew_100",
    "kurt_100",  # Tail structure (extreme risk)
    "BOLL_BW_25",  # Decide after uplift testing
    "RSI_14",  # Relative Strength Index (momentum/overheat; indirectly reflects volatility)
    # =========================
    # 3) Efficiency / market structure: trending vs ranging
    # =========================
    "er_126",  # Trend efficiency ratio (high-quality structure factor)
    # =========================
    # 4) Participation / liquidity: market activity
    # =========================
    "trade_density_14",  # CRYPTO_ONLY: not enabled in FEATURE_LIST_COMMODITY
    "vol_event_flag_500",  # CRYPTO_ONLY: not enabled in FEATURE_LIST_COMMODITY
    # =========================
    # 5) Order flow / imbalance: buy-vs-sell dominance
    # =========================
    "vpin_49",  # CRYPTO_ONLY: not enabled in FEATURE_LIST_COMMODITY
    "vpin_14",  # CRYPTO_ONLY: not enabled in FEATURE_LIST_COMMODITY
    # =========================
    # 6) Spatial / price position: where price sits within ranges/cost
    # =========================
    "poc_bias_600",  # CRYPTO_ONLY: not enabled in FEATURE_LIST_COMMODITY
    "poc_bias_99",  # CRYPTO_ONLY: not enabled in FEATURE_LIST_COMMODITY
    "close_pos",  # CRYPTO_ONLY: not enabled in FEATURE_LIST_COMMODITY
    # =========================
    # 7) Candlestick / path microstructure
    # =========================
    "upper_wick_pct",
    "lower_wick_pct",
]
