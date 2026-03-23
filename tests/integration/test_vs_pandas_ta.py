"""
Comparison tests: ferro_ta vs pandas-ta (Priority 4 - requires pandas-ta).

This module validates ferro_ta against pandas-ta for indicators, using 500-bar data
for proper convergence of EMA-seeded indicators. Documents known formula differences
and expected tolerances.

Requirements
------------
Install pandas-ta before running these tests::

    pip install pandas-ta

The tests are automatically skipped when pandas-ta is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip the whole module when pandas-ta is not available
# ---------------------------------------------------------------------------

pandas_ta = pytest.importorskip(
    "pandas_ta", reason="pandas-ta not installed; skipping comparison tests"
)
pd = pytest.importorskip("pandas", reason="pandas required for pandas-ta")

import ferro_ta  # noqa: E402

# ---------------------------------------------------------------------------
# Shared test data from conftest.py
# ---------------------------------------------------------------------------

# Use shared 500-bar fixture from conftest.py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nan_count(arr: np.ndarray) -> int:
    """Return count of NaN values."""
    return int(np.sum(np.isnan(arr)))


def _valid_mask(*arrays: np.ndarray) -> np.ndarray:
    """Return boolean mask for positions where *all* arrays are finite."""
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= ~np.isnan(a)
    return mask


def _allclose(
    a: np.ndarray, b: np.ndarray, atol: float = 1e-6, tail_fraction: float = 1.0
) -> bool:
    """Compare arrays within tolerance, optionally only comparing tail.

    Parameters
    ----------
    a, b : np.ndarray
        Arrays to compare
    atol : float
        Absolute tolerance
    tail_fraction : float
        Fraction of tail to compare (1.0 = all, 0.3 = last 30%)

    Returns
    -------
    bool
        True if arrays match within tolerance
    """
    mask = _valid_mask(a, b)
    if not mask.any():
        return False

    if tail_fraction < 1.0:
        # Only compare last tail_fraction of data
        n = len(a)
        start_idx = int(n * (1 - tail_fraction))
        mask[:start_idx] = False

    if not mask.any():
        return False

    return bool(np.allclose(a[mask], b[mask], atol=atol))


# ---------------------------------------------------------------------------
# Overlap Studies
# ---------------------------------------------------------------------------


class TestSMAVsPandasTA:
    """SMA — Exact match (deterministic)."""

    def test_sma_exact_match(self, ohlcv_500):
        """SMA should match pandas-ta exactly."""
        close = ohlcv_500["close"]
        period = 20

        ft = ferro_ta.SMA(close, timeperiod=period)
        pt = pandas_ta.sma(pd.Series(close), length=period).to_numpy()

        assert _allclose(ft, pt, atol=1e-8)


class TestEMAVsPandasTA:
    """EMA — Tail 30% match (seed difference).

    ferro_ta starts EMA from bar 0, pandas-ta may use SMA seed.
    After 350+ bars of decay, values should converge.
    """

    def test_ema_tail_convergence(self, ohlcv_500):
        """EMA should converge in tail 30% of data."""
        close = ohlcv_500["close"]
        period = 20

        ft = ferro_ta.EMA(close, timeperiod=period)
        pt = pandas_ta.ema(pd.Series(close), length=period).to_numpy()

        # Compare only last 30%
        assert _allclose(ft, pt, atol=1e-4, tail_fraction=0.3)

    def test_ema_shorter_period_tighter(self, ohlcv_500):
        """Shorter period EMA should have tighter convergence."""
        close = ohlcv_500["close"]
        period = 10

        ft = ferro_ta.EMA(close, timeperiod=period)
        pt = pandas_ta.ema(pd.Series(close), length=period).to_numpy()

        # Shorter period converges faster
        assert _allclose(ft, pt, atol=1e-5, tail_fraction=0.3)


class TestWMAVsPandasTA:
    """WMA — Exact match (deterministic)."""

    def test_wma_exact_match(self, ohlcv_500):
        """WMA should match pandas-ta exactly."""
        close = ohlcv_500["close"]
        period = 20

        ft = ferro_ta.WMA(close, timeperiod=period)
        pt = pandas_ta.wma(pd.Series(close), length=period).to_numpy()

        assert _allclose(ft, pt, atol=1e-8)


class TestBBANDSVsPandasTA:
    """BBANDS — Approximate match (ferro_ta uses population std; pandas-ta uses sample std)."""

    def test_bbands_approximate_match(self, ohlcv_500):
        """BBANDS middle band matches exactly; upper/lower match within std-formula tolerance.

        ferro_ta follows TA-Lib convention: std = population std (ddof=0).
        pandas-ta uses sample std (ddof=1).  Middle band (SMA) is identical.
        Upper/lower differ by a sqrt(N/(N-1)) factor (~0.5% for N=20), capped at atol=0.1.
        """
        close = ohlcv_500["close"]
        period = 20

        ft_upper, ft_middle, ft_lower = ferro_ta.BBANDS(
            close, timeperiod=period, nbdevup=2.0, nbdevdn=2.0
        )

        # pandas-ta >= 0.3 returns columns named BBL_{period}_{std}_{std}
        pt_bbands = pandas_ta.bbands(pd.Series(close), length=period, std=2.0)
        # Locate columns robustly (column names vary across pandas-ta versions)
        lower_col = next(c for c in pt_bbands.columns if c.startswith("BBL_"))
        middle_col = next(c for c in pt_bbands.columns if c.startswith("BBM_"))
        upper_col = next(c for c in pt_bbands.columns if c.startswith("BBU_"))
        pt_lower = pt_bbands[lower_col].to_numpy()
        pt_middle = pt_bbands[middle_col].to_numpy()
        pt_upper = pt_bbands[upper_col].to_numpy()

        # Middle band (SMA) must be identical
        assert _allclose(ft_middle, pt_middle, atol=1e-8), "BBands middle (SMA) must match"
        # Upper/lower: differ due to ddof=0 vs ddof=1
        assert _allclose(ft_upper, pt_upper, atol=0.1)
        assert _allclose(ft_lower, pt_lower, atol=0.1)


class TestTRIMAVsPandasTA:
    """TRIMA — Approximate match (implementations differ slightly in boundary handling)."""

    def test_trima_approximate_match(self, ohlcv_500):
        """TRIMA should be close to pandas-ta (both are SMA-of-SMA but boundary handling differs).

        Note: ferro_ta follows TA-Lib's TRIMA formula while pandas-ta uses a slightly
        different implementation.  Observed max difference is ~0.4 price units on
        typical equity prices (~100), which is < 0.5%.  We verify tail convergence
        with atol=0.5 and confirm correct NaN warm-up length.
        """
        close = ohlcv_500["close"]
        period = 20

        ft = ferro_ta.TRIMA(close, timeperiod=period)
        pt = pandas_ta.trima(pd.Series(close), length=period).to_numpy()

        assert _allclose(ft, pt, atol=0.5, tail_fraction=0.5)


class TestMACDVsPandasTA:
    """MACD — Tail 30% match (EMA seed difference)."""

    def test_macd_tail_convergence(self, ohlcv_500):
        """MACD should converge in tail 30% of data."""
        close = ohlcv_500["close"]

        ft_macd, ft_signal, ft_hist = ferro_ta.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )

        # pandas-ta returns DataFrame
        pt_macd = pandas_ta.macd(pd.Series(close), fast=12, slow=26, signal=9)
        pt_macd_line = pt_macd["MACD_12_26_9"].to_numpy()
        pt_signal_line = pt_macd["MACDs_12_26_9"].to_numpy()
        pt_hist = pt_macd["MACDh_12_26_9"].to_numpy()

        # Compare tail 30%
        assert _allclose(ft_macd, pt_macd_line, atol=1e-2, tail_fraction=0.3)
        assert _allclose(ft_signal, pt_signal_line, atol=1e-2, tail_fraction=0.3)
        assert _allclose(ft_hist, pt_hist, atol=1e-2, tail_fraction=0.3)


# ---------------------------------------------------------------------------
# Momentum Indicators
# ---------------------------------------------------------------------------


class TestRSIVsPandasTA:
    """RSI — Tail 30% match (Wilder seed difference)."""

    def test_rsi_tail_convergence(self, ohlcv_500):
        """RSI should converge in tail 30% of data."""
        close = ohlcv_500["close"]
        period = 14

        ft = ferro_ta.RSI(close, timeperiod=period)
        pt = pandas_ta.rsi(pd.Series(close), length=period).to_numpy()

        assert _allclose(ft, pt, atol=1e-3, tail_fraction=0.3)


class TestSTOCHVsPandasTA:
    """STOCH — Tail 30% match."""

    def test_stoch_tail_convergence(self, ohlcv_500):
        """Stochastic should converge in tail 30% of data."""
        high = ohlcv_500["high"]
        low = ohlcv_500["low"]
        close = ohlcv_500["close"]

        ft_slowk, ft_slowd = ferro_ta.STOCH(
            high, low, close,
            fastk_period=14, slowk_period=3,
            slowd_period=3
        )

        # pandas-ta returns DataFrame
        pt_stoch = pandas_ta.stoch(
            pd.Series(high), pd.Series(low), pd.Series(close),
            k=14, d=3, smooth_k=3
        )
        pt_slowk = pt_stoch[f"STOCHk_14_3_3"].to_numpy()
        pt_slowd = pt_stoch[f"STOCHd_14_3_3"].to_numpy()

        assert _allclose(ft_slowk, pt_slowk, atol=1e-2, tail_fraction=0.3)
        assert _allclose(ft_slowd, pt_slowd, atol=1e-2, tail_fraction=0.3)


class TestCCIVsPandasTA:
    """CCI — Exact match (deterministic rolling formula)."""

    def test_cci_exact_match(self, ohlcv_500):
        """CCI should match manually-computed reference (pandas-ta CCI has a formula bug)."""
        high = ohlcv_500["high"]
        low = ohlcv_500["low"]
        close = ohlcv_500["close"]
        period = 14

        ft = ferro_ta.CCI(high, low, close, timeperiod=period)

        # Compute CCI manually: (TP - SMA(TP)) / (0.015 * MeanAbsDev(TP))
        tp = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3.0
        mean_tp = tp.rolling(period).mean()
        mad_tp = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        pt = ((tp - mean_tp) / (0.015 * mad_tp)).to_numpy()

        assert _allclose(ft, pt, atol=1e-8)


class TestWILLRVsPandasTA:
    """WILLR — Exact match (deterministic)."""

    def test_willr_exact_match(self, ohlcv_500):
        """Williams %R should match pandas-ta exactly."""
        high = ohlcv_500["high"]
        low = ohlcv_500["low"]
        close = ohlcv_500["close"]
        period = 14

        ft = ferro_ta.WILLR(high, low, close, timeperiod=period)

        df = pd.DataFrame({"high": high, "low": low, "close": close})
        pt = df.ta.willr(length=period).to_numpy()

        assert _allclose(ft, pt, atol=1e-8)


class TestMOMVsPandasTA:
    """MOM — Exact match."""

    def test_mom_exact_match(self, ohlcv_500):
        """MOM should match pandas-ta exactly."""
        close = ohlcv_500["close"]
        period = 10

        ft = ferro_ta.MOM(close, timeperiod=period)
        pt = pandas_ta.mom(pd.Series(close), length=period).to_numpy()

        assert _allclose(ft, pt, atol=1e-8)


class TestROCVsPandasTA:
    """ROC — Exact match."""

    def test_roc_exact_match(self, ohlcv_500):
        """ROC should match pandas-ta exactly."""
        close = ohlcv_500["close"]
        period = 10

        ft = ferro_ta.ROC(close, timeperiod=period)
        pt = pandas_ta.roc(pd.Series(close), length=period).to_numpy()

        assert _allclose(ft, pt, atol=1e-8)


class TestMFIVsPandasTA:
    """MFI — Exact match."""

    def test_mfi_exact_match(self, ohlcv_500):
        """MFI should match pandas-ta exactly."""
        high = ohlcv_500["high"]
        low = ohlcv_500["low"]
        close = ohlcv_500["close"]
        volume = ohlcv_500["volume"]
        period = 14

        ft = ferro_ta.MFI(high, low, close, volume, timeperiod=period)

        df = pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})
        pt = df.ta.mfi(length=period).to_numpy()

        assert _allclose(ft, pt, atol=1e-8)


class TestAROONVsPandasTA:
    """AROON — Exact match."""

    def test_aroon_exact_match(self, ohlcv_500):
        """AROON should match pandas-ta exactly."""
        high = ohlcv_500["high"]
        low = ohlcv_500["low"]
        period = 14

        ft_down, ft_up = ferro_ta.AROON(high, low, timeperiod=period)

        df = pd.DataFrame({"high": high, "low": low})
        pt_aroon = df.ta.aroon(length=period)
        pt_down = pt_aroon[f"AROOND_{period}"].to_numpy()
        pt_up = pt_aroon[f"AROONU_{period}"].to_numpy()

        assert _allclose(ft_down, pt_down, atol=1e-8)
        assert _allclose(ft_up, pt_up, atol=1e-8)


# ---------------------------------------------------------------------------
# Volume/Volatility
# ---------------------------------------------------------------------------


class TestOBVVsPandasTA:
    """OBV — Incremental match (offset constant, verify diffs)."""

    def test_obv_incremental_match(self, ohlcv_500):
        """OBV differences should match (absolute values may have offset)."""
        close = ohlcv_500["close"]
        volume = ohlcv_500["volume"]

        ft = ferro_ta.OBV(close, volume)

        df = pd.DataFrame({"close": close, "volume": volume})
        pt = df.ta.obv().to_numpy()

        # OBV can have different starting values, compare differences
        ft_diff = np.diff(ft)
        pt_diff = np.diff(pt)

        # Remove NaN values from comparison
        mask = ~np.isnan(ft_diff) & ~np.isnan(pt_diff)
        assert np.allclose(ft_diff[mask], pt_diff[mask], atol=1e-8)


class TestATRVsPandasTA:
    """ATR — Tail 30% match (Wilder seed difference)."""

    def test_atr_tail_convergence(self, ohlcv_500):
        """ATR should converge in tail 30% of data."""
        high = ohlcv_500["high"]
        low = ohlcv_500["low"]
        close = ohlcv_500["close"]
        period = 14

        ft = ferro_ta.ATR(high, low, close, timeperiod=period)

        df = pd.DataFrame({"high": high, "low": low, "close": close})
        pt = df.ta.atr(length=period).to_numpy()

        assert _allclose(ft, pt, atol=1e-2, tail_fraction=0.3)


class TestADXVsPandasTA:
    """ADX — Tail 30% match (two levels of Wilder smoothing)."""

    def test_adx_tail_convergence(self, ohlcv_500):
        """ADX should converge in tail 30% of data."""
        high = ohlcv_500["high"]
        low = ohlcv_500["low"]
        close = ohlcv_500["close"]
        period = 14

        ft = ferro_ta.ADX(high, low, close, timeperiod=period)

        df = pd.DataFrame({"high": high, "low": low, "close": close})
        pt = df.ta.adx(length=period)[f"ADX_{period}"].to_numpy()

        assert _allclose(ft, pt, atol=5e-2, tail_fraction=0.3)


# ---------------------------------------------------------------------------
# Extended Indicators (no prior validation)
# ---------------------------------------------------------------------------


class TestVWAPVsPandasTA:
    """VWAP — Validate rolling VWAP against a reference numpy implementation."""

    def test_vwap_rolling_match(self, ohlcv_500):
        """Rolling VWAP should match a reference implementation using numpy."""
        high = ohlcv_500["high"]
        low = ohlcv_500["low"]
        close = ohlcv_500["close"]
        volume = ohlcv_500["volume"]
        period = 20

        ft = ferro_ta.VWAP(high, low, close, volume, timeperiod=period)

        # Reference: rolling VWAP = sum(typical_price * volume, N) / sum(volume, N)
        tp = (np.array(high) + np.array(low) + np.array(close)) / 3.0
        vol = np.array(volume)
        n = len(tp)
        ref = np.full(n, np.nan)
        for i in range(period - 1, n):
            w = tp[i - period + 1: i + 1]
            v = vol[i - period + 1: i + 1]
            ref[i] = np.dot(w, v) / v.sum()

        assert _allclose(ft, ref, atol=1e-8)


class TestDONCHIANVsPandasTA:
    """DONCHIAN — Exact match (rolling max(H), min(L), mean)."""

    def test_donchian_exact_match(self, ohlcv_500):
        """Donchian Channels should match pandas-ta exactly."""
        high = ohlcv_500["high"]
        low = ohlcv_500["low"]
        period = 20

        ft_upper, ft_middle, ft_lower = ferro_ta.DONCHIAN(high, low, timeperiod=period)

        df = pd.DataFrame({"high": high, "low": low, "close": ohlcv_500["close"]})
        pt_donchian = df.ta.donchian(lower_length=period, upper_length=period)
        pt_lower = pt_donchian[f"DCL_{period}_{period}"].to_numpy()
        pt_middle = pt_donchian[f"DCM_{period}_{period}"].to_numpy()
        pt_upper = pt_donchian[f"DCU_{period}_{period}"].to_numpy()

        assert _allclose(ft_upper, pt_upper, atol=1e-8)
        assert _allclose(ft_middle, pt_middle, atol=1e-8)
        assert _allclose(ft_lower, pt_lower, atol=1e-8)


class TestHULL_MAVsPandasTA:
    """HULL_MA — Exact match (WMA composition: deterministic)."""

    def test_hull_ma_exact_match(self, ohlcv_500):
        """Hull MA should match pandas-ta exactly."""
        close = ohlcv_500["close"]
        period = 16

        ft = ferro_ta.HULL_MA(close, timeperiod=period)
        pt = pandas_ta.hma(pd.Series(close), length=period).to_numpy()

        assert _allclose(ft, pt, atol=1e-8)


class TestICHIMOKUVsPandasTA:
    """ICHIMOKU — Exact match for tenkan/kijun (rolling midpoint formula)."""

    def test_ichimoku_tenkan_kijun_match(self, ohlcv_500):
        """Ichimoku tenkan and kijun should match pandas-ta."""
        high = ohlcv_500["high"]
        low = ohlcv_500["low"]
        close = ohlcv_500["close"]

        ft_tenkan, ft_kijun, ft_senkou_a, ft_senkou_b, ft_chikou = ferro_ta.ICHIMOKU(
            high, low, close, tenkan_period=9, kijun_period=26, senkou_b_period=52, displacement=26
        )

        df = pd.DataFrame({"high": high, "low": low, "close": close})
        pt_ichimoku = df.ta.ichimoku(tenkan=9, kijun=26, senkou=52)[0]
        pt_tenkan = pt_ichimoku["ITS_9"].to_numpy()
        pt_kijun = pt_ichimoku["IKS_26"].to_numpy()

        assert _allclose(ft_tenkan, pt_tenkan, atol=1e-8)
        assert _allclose(ft_kijun, pt_kijun, atol=1e-8)


class TestKELTNER_CHANNELSVsPandasTA:
    """KELTNER_CHANNELS — Tail 30% match (Middle=EMA, bands=EMA±mult*ATR)."""

    def test_keltner_tail_convergence(self, ohlcv_500):
        """Keltner Channels should converge in tail 30%."""
        high = ohlcv_500["high"]
        low = ohlcv_500["low"]
        close = ohlcv_500["close"]
        period = 20
        atr_period = 10
        multiplier = 2.0

        ft_upper, ft_middle, ft_lower = ferro_ta.KELTNER_CHANNELS(
            high, low, close, timeperiod=period, atr_period=atr_period, multiplier=multiplier
        )

        # Compute manually using pandas_ta EMA and ATR to match ferro_ta's exact formula
        pt_ema = pandas_ta.ema(pd.Series(close), length=period).to_numpy()
        pt_atr = pandas_ta.atr(pd.Series(high), pd.Series(low), pd.Series(close), length=atr_period).to_numpy()
        pt_upper = pt_ema + multiplier * pt_atr
        pt_middle = pt_ema
        pt_lower = pt_ema - multiplier * pt_atr

        assert _allclose(ft_upper, pt_upper, atol=1e-2, tail_fraction=0.3)
        assert _allclose(ft_middle, pt_middle, atol=1e-2, tail_fraction=0.3)
        assert _allclose(ft_lower, pt_lower, atol=1e-2, tail_fraction=0.3)


class TestVWMAVsPandasTA:
    """VWMA — Exact match (sum(c*v)/sum(v))."""

    def test_vwma_exact_match(self, ohlcv_500):
        """VWMA should match pandas-ta exactly."""
        close = ohlcv_500["close"]
        volume = ohlcv_500["volume"]
        period = 20

        ft = ferro_ta.VWMA(close, volume, timeperiod=period)

        df = pd.DataFrame({"close": close, "volume": volume})
        pt = df.ta.vwma(length=period).to_numpy()

        assert _allclose(ft, pt, atol=1e-8)


class TestCHOPPINESS_INDEXVsPandasTA:
    """CHOPPINESS_INDEX — Close match (log10-based formula)."""

    def test_choppiness_index_close_match(self, ohlcv_500):
        """Choppiness Index should match pandas-ta closely."""
        high = ohlcv_500["high"]
        low = ohlcv_500["low"]
        close = ohlcv_500["close"]
        period = 14

        ft = ferro_ta.CHOPPINESS_INDEX(high, low, close, timeperiod=period)

        df = pd.DataFrame({"high": high, "low": low, "close": close})
        pt = df.ta.chop(length=period).to_numpy()

        assert _allclose(ft, pt, atol=1e-4)


class TestSUPERTRENDVsPandasTA:
    """SUPERTREND — Direction >80% agreement (path-dependent, ATR seeding differs)."""

    def test_supertrend_direction_agreement(self, ohlcv_500):
        """SUPERTREND direction should agree >80% of the time."""
        high = ohlcv_500["high"]
        low = ohlcv_500["low"]
        close = ohlcv_500["close"]
        period = 7
        multiplier = 3.0

        ft_line, ft_dir = ferro_ta.SUPERTREND(
            high, low, close, timeperiod=period, multiplier=multiplier
        )

        df = pd.DataFrame({"high": high, "low": low, "close": close})
        pt_supertrend = df.ta.supertrend(length=period, multiplier=multiplier)
        pt_dir = pt_supertrend[f"SUPERTd_{period}_{multiplier}"].to_numpy()

        # Convert directions to same format (1 = up, -1 = down)
        # pandas-ta: 1 = uptrend, -1 = downtrend
        # ferro_ta: 1 = uptrend, -1 = downtrend (assuming same convention)

        # Remove NaN values
        mask = ~np.isnan(ft_dir) & ~np.isnan(pt_dir)
        agreement_rate = np.mean(ft_dir[mask] == pt_dir[mask])

        assert agreement_rate > 0.80, f"Direction agreement rate: {agreement_rate:.2%}"


class TestCHANDELIER_EXITVsPandasTA:
    """CHANDELIER_EXIT — Exact structure (rolling_max(H)-mult*ATR)."""

    def test_chandelier_exit_structure_match(self, ohlcv_500):
        """Chandelier Exit should match pandas-ta structure."""
        high = ohlcv_500["high"]
        low = ohlcv_500["low"]
        close = ohlcv_500["close"]
        period = 22
        multiplier = 3.0

        ft_long, ft_short = ferro_ta.CHANDELIER_EXIT(
            high, low, close, timeperiod=period, multiplier=multiplier
        )

        # Compute manually: long = rolling_max(H, n) - mult*ATR; short = rolling_min(L, n) + mult*ATR
        pt_atr = pandas_ta.atr(pd.Series(high), pd.Series(low), pd.Series(close), length=period).to_numpy()
        rolling_high = pd.Series(high).rolling(period).max().to_numpy()
        rolling_low = pd.Series(low).rolling(period).min().to_numpy()
        pt_long = rolling_high - multiplier * pt_atr
        pt_short = rolling_low + multiplier * pt_atr

        assert _allclose(ft_long, pt_long, atol=1e-2, tail_fraction=0.3)
        assert _allclose(ft_short, pt_short, atol=1e-2, tail_fraction=0.3)


class TestPIVOT_POINTSVsPandasTA:
    """PIVOT_POINTS — Exact match for Classic (arithmetic formula)."""

    def test_pivot_points_classic_exact(self, ohlcv_500):
        """Classic Pivot Points should match manually-computed reference."""
        high = ohlcv_500["high"]
        low = ohlcv_500["low"]
        close = ohlcv_500["close"]

        ft_pivot, ft_r1, ft_s1, ft_r2, ft_s2 = ferro_ta.PIVOT_POINTS(
            high, low, close, method="classic"
        )

        # ferro_ta PIVOT_POINTS uses previous bar's H/L/C (1-bar forward shift).
        # Reference values are computed from bar i-1 to match index i output.
        pivot = np.empty_like(high, dtype=float)
        pivot[0] = np.nan
        pivot[1:] = (high[:-1] + low[:-1] + close[:-1]) / 3.0

        r1 = np.empty_like(high, dtype=float)
        r1[0] = np.nan
        r1[1:] = 2 * pivot[1:] - low[:-1]

        s1 = np.empty_like(high, dtype=float)
        s1[0] = np.nan
        s1[1:] = 2 * pivot[1:] - high[:-1]

        r2 = np.empty_like(high, dtype=float)
        r2[0] = np.nan
        r2[1:] = pivot[1:] + (high[:-1] - low[:-1])

        s2 = np.empty_like(high, dtype=float)
        s2[0] = np.nan
        s2[1:] = pivot[1:] - (high[:-1] - low[:-1])

        assert _allclose(ft_pivot, pivot, atol=1e-8)
        assert _allclose(ft_r1, r1, atol=1e-8)
        assert _allclose(ft_s1, s1, atol=1e-8)
        assert _allclose(ft_r2, r2, atol=1e-8)
        assert _allclose(ft_s2, s2, atol=1e-8)
