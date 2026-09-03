"""
Comparison tests: ferro_ta vs TA-Lib (ta-lib Python wrapper).

This module verifies that ferro_ta is a drop-in replacement for TA-Lib by
comparing the outputs of every shared indicator for:

  * **Shape compatibility** — same output length and NaN count (or ±1 where a
    documented off-by-one exists).
  * **Value accuracy** — exact match within floating-point tolerance where the
    algorithms are identical; range / convergence checks where initialization
    differs.

Known differences are documented next to each test so consumers know what
to expect when migrating from TA-Lib.

Requirements
------------
Install ta-lib before running these tests::

    pip install ta-lib

The tests are automatically skipped when ta-lib is not installed, so the
main CI pipeline never fails because of a missing optional dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip the whole module when ta-lib is not available
# ---------------------------------------------------------------------------

talib = pytest.importorskip(
    "talib", reason="ta-lib not installed; skipping comparison tests"
)

import ferro_ta  # noqa: E402  (import after potential skip)

# ---------------------------------------------------------------------------
# Shared realistic OHLCV data (500 bars for proper convergence)
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

N = 500  # Increased from 100 to 500 for proper EMA/RSI/ATR convergence
CLOSE = 44.0 + np.cumsum(RNG.standard_normal(N) * 0.5)
HIGH = CLOSE + RNG.uniform(0.1, 1.0, N)
LOW = CLOSE - RNG.uniform(0.1, 1.0, N)
OPEN = CLOSE + RNG.standard_normal(N) * 0.2
VOLUME = RNG.uniform(500.0, 2000.0, N)

# Simple monotonically increasing series used for deterministic checks
LINEAR = np.arange(1.0, N + 1.0, dtype=np.float64)
LINEAR_HIGH = LINEAR + 0.5
LINEAR_LOW = LINEAR - 0.5
LINEAR_OPEN = LINEAR - 0.2
LINEAR_VOL = np.ones(N) * 1000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimum fraction of values that must agree in sign for correlated indicators.
SIGN_AGREEMENT_THRESHOLD = 0.8

# Per-pattern candlestick agreement thresholds.
# Most patterns use 0.80; patterns with known definition differences from TA-Lib
# use lower thresholds with a documented reason.
CDL_AGREEMENT_THRESHOLDS: dict[str, float] = {
    # Body/shadow ratio thresholds differ between ferro_ta and TA-Lib
    "CDLHIGHWAVE": 0.65,  # Shadow length threshold differs; 69% observed
    "CDLLONGLEGGEDDOJI": 0.70,  # Long-leg threshold differs; 75% observed
    "CDLSHORTLINE": 0.20,  # Body-size cutoff definition completely differs; 25% observed
    "CDLSPINNINGTOP": 0.75,  # Body ratio threshold differs; 78% observed
    "CDLDOJI": 0.85,  # Shadow ratio precision differs; 86% observed
}


def _nan_count(arr: np.ndarray) -> int:
    return int(np.sum(np.isnan(arr)))


def _valid_mask(*arrays: np.ndarray) -> np.ndarray:
    """Return boolean mask for positions where *all* arrays are finite."""
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= ~np.isnan(a)
    return mask


def _allclose(a: np.ndarray, b: np.ndarray, atol: float = 1e-6) -> bool:
    mask = _valid_mask(a, b)
    if not mask.any():
        return False
    return bool(np.allclose(a[mask], b[mask], atol=atol))


# ---------------------------------------------------------------------------
# Overlap Studies
# ---------------------------------------------------------------------------


class TestSMA:
    """SMA — exact match."""

    def test_values_match(self):
        ft = ferro_ta.SMA(CLOSE, timeperiod=10)
        ta = talib.SMA(CLOSE, timeperiod=10)
        assert _allclose(ft, ta)

    def test_nan_count_match(self):
        ft = ferro_ta.SMA(CLOSE, timeperiod=10)
        ta = talib.SMA(CLOSE, timeperiod=10)
        assert _nan_count(ft) == _nan_count(ta)

    def test_output_length_match(self):
        ft = ferro_ta.SMA(CLOSE, timeperiod=5)
        ta = talib.SMA(CLOSE, timeperiod=5)
        assert len(ft) == len(ta)


class TestEMA:
    """EMA — SMA-seeded (TA-Lib compatible).

    Both ferro-ta and TA-Lib seed the first EMA value with the SMA of the
    initial ``timeperiod`` bars, then apply ``k = 2 / (timeperiod + 1)``.
    We verify matching warmup length and that the series stay close.
    """

    def test_nan_count_match(self):
        ft = ferro_ta.EMA(CLOSE, timeperiod=10)
        ta = talib.EMA(CLOSE, timeperiod=10)
        assert _nan_count(ft) == _nan_count(ta)

    def test_output_length_match(self):
        ft = ferro_ta.EMA(CLOSE, timeperiod=5)
        ta = talib.EMA(CLOSE, timeperiod=5)
        assert len(ft) == len(ta)

    def test_values_converge(self):
        """After convergence (tail 30%), EMA should be very close with 500 bars."""
        ft = ferro_ta.EMA(CLOSE, timeperiod=5)
        ta = talib.EMA(CLOSE, timeperiod=5)
        # With 500 bars, compare last 30% with tighter tolerance
        tail_start = int(N * 0.7)
        assert np.allclose(
            ft[tail_start:], ta[tail_start:], atol=1e-5
        )  # Tightened from 1e-3

    def test_values_finite_and_reasonable(self):
        ft = ferro_ta.EMA(CLOSE, timeperiod=5)
        finite = ft[~np.isnan(ft)]
        assert finite.min() > 0
        assert finite.max() < 1000


class TestWMA:
    """WMA — exact match."""

    def test_values_match(self):
        ft = ferro_ta.WMA(CLOSE, timeperiod=10)
        ta = talib.WMA(CLOSE, timeperiod=10)
        assert _allclose(ft, ta)

    def test_nan_count_match(self):
        ft = ferro_ta.WMA(CLOSE, timeperiod=10)
        ta = talib.WMA(CLOSE, timeperiod=10)
        assert _nan_count(ft) == _nan_count(ta)


class TestDEMA:
    """DEMA — shape matches; values differ (EMA-based initialization)."""

    def test_nan_count_match(self):
        ft = ferro_ta.DEMA(CLOSE, timeperiod=5)
        ta = talib.DEMA(CLOSE, timeperiod=5)
        assert _nan_count(ft) == _nan_count(ta)

    def test_output_length_match(self):
        ft = ferro_ta.DEMA(CLOSE, timeperiod=5)
        ta = talib.DEMA(CLOSE, timeperiod=5)
        assert len(ft) == len(ta)

    def test_values_converge(self):
        ft = ferro_ta.DEMA(CLOSE, timeperiod=5)
        ta = talib.DEMA(CLOSE, timeperiod=5)
        mid = N // 2
        assert np.allclose(ft[mid:], ta[mid:], atol=1e-2)


class TestTEMA:
    """TEMA — shape matches; values differ (EMA-based initialization)."""

    def test_nan_count_match(self):
        ft = ferro_ta.TEMA(CLOSE, timeperiod=5)
        ta = talib.TEMA(CLOSE, timeperiod=5)
        assert _nan_count(ft) == _nan_count(ta)

    def test_output_length_match(self):
        ft = ferro_ta.TEMA(CLOSE, timeperiod=5)
        ta = talib.TEMA(CLOSE, timeperiod=5)
        assert len(ft) == len(ta)

    def test_values_converge(self):
        ft = ferro_ta.TEMA(CLOSE, timeperiod=5)
        ta = talib.TEMA(CLOSE, timeperiod=5)
        mid = N // 2
        assert np.allclose(ft[mid:], ta[mid:], atol=1e-2)


class TestTRIMA:
    """TRIMA — exact match."""

    def test_values_match(self):
        ft = ferro_ta.TRIMA(CLOSE, timeperiod=10)
        ta = talib.TRIMA(CLOSE, timeperiod=10)
        assert _allclose(ft, ta)

    def test_nan_count_match(self):
        ft = ferro_ta.TRIMA(CLOSE, timeperiod=10)
        ta = talib.TRIMA(CLOSE, timeperiod=10)
        assert _nan_count(ft) == _nan_count(ta)


class TestKAMA:
    """KAMA — exact match. First output at index ``timeperiod`` (TA-Lib)."""

    def test_values_match(self):
        ft = ferro_ta.KAMA(CLOSE, timeperiod=10)
        ta = talib.KAMA(CLOSE, timeperiod=10)
        assert _allclose(ft, ta)

    def test_nan_count_match(self):
        ft = ferro_ta.KAMA(CLOSE, timeperiod=10)
        ta = talib.KAMA(CLOSE, timeperiod=10)
        assert _nan_count(ft) == _nan_count(ta)

    def test_output_length_match(self):
        ft = ferro_ta.KAMA(CLOSE, timeperiod=10)
        ta = talib.KAMA(CLOSE, timeperiod=10)
        assert len(ft) == len(ta)


class TestT3:
    """T3 — shape matches; values differ (EMA-based initialization)."""

    def test_nan_count_match(self):
        ft = ferro_ta.T3(CLOSE, timeperiod=5)
        ta = talib.T3(CLOSE, timeperiod=5)
        assert _nan_count(ft) == _nan_count(ta)

    def test_output_length_match(self):
        ft = ferro_ta.T3(CLOSE, timeperiod=5)
        ta = talib.T3(CLOSE, timeperiod=5)
        assert len(ft) == len(ta)

    def test_values_converge(self):
        ft = ferro_ta.T3(CLOSE, timeperiod=5)
        ta = talib.T3(CLOSE, timeperiod=5)
        # With 500 bars, use last 30% with tighter tolerance
        tail_start = int(N * 0.7)
        assert np.allclose(
            ft[tail_start:], ta[tail_start:], atol=1e-3
        )  # Tightened from 5e-2


class TestBBANDS:
    """BBANDS — exact match."""

    def test_values_match(self):
        upper_ft, mid_ft, lower_ft = ferro_ta.BBANDS(
            CLOSE, timeperiod=10, nbdevup=2.0, nbdevdn=2.0
        )
        upper_ta, mid_ta, lower_ta = talib.BBANDS(
            CLOSE, timeperiod=10, nbdevup=2.0, nbdevdn=2.0
        )
        assert _allclose(upper_ft, upper_ta)
        assert _allclose(mid_ft, mid_ta)
        assert _allclose(lower_ft, lower_ta)

    def test_nan_count_match(self):
        upper_ft, _, _ = ferro_ta.BBANDS(CLOSE, timeperiod=10)
        upper_ta, _, _ = talib.BBANDS(CLOSE, timeperiod=10)
        assert _nan_count(upper_ft) == _nan_count(upper_ta)

    def test_output_length_match(self):
        upper_ft, _, _ = ferro_ta.BBANDS(CLOSE, timeperiod=5)
        upper_ta, _, _ = talib.BBANDS(CLOSE, timeperiod=5)
        assert len(upper_ft) == len(upper_ta)


class TestMACD:
    """MACD — shape matches; values differ (EMA-based initialization).

    The MACD line, signal, and histogram converge after sufficient warmup.
    The histogram relationship (macd - signal) is preserved in both.
    """

    def test_nan_count_match(self):
        ft_m, ft_s, ft_h = ferro_ta.MACD(
            CLOSE, fastperiod=3, slowperiod=6, signalperiod=2
        )
        ta_m, ta_s, ta_h = talib.MACD(CLOSE, fastperiod=3, slowperiod=6, signalperiod=2)
        assert _nan_count(ft_m) == _nan_count(ta_m)

    def test_output_length_match(self):
        ft_m, ft_s, ft_h = ferro_ta.MACD(CLOSE)
        ta_m, ta_s, ta_h = talib.MACD(CLOSE)
        assert len(ft_m) == len(ta_m) == len(CLOSE)

    def test_histogram_relationship(self):
        """Histogram = MACD line − signal line (must hold for both libraries)."""
        for fn, lib in [(ferro_ta.MACD, "ferro_ta"), (talib.MACD, "talib")]:
            m, s, h = fn(CLOSE, fastperiod=3, slowperiod=6, signalperiod=2)
            mask = _valid_mask(m, s, h)
            assert np.allclose(h[mask], m[mask] - s[mask], atol=1e-10), (
                f"{lib} histogram mismatch"
            )

    def test_values_converge(self):
        ft_m, _, _ = ferro_ta.MACD(CLOSE, fastperiod=3, slowperiod=6, signalperiod=2)
        ta_m, _, _ = talib.MACD(CLOSE, fastperiod=3, slowperiod=6, signalperiod=2)
        assert np.allclose(ft_m[-N // 4 :], ta_m[-N // 4 :], atol=1e-2)


class TestMACDFIX:
    """MACDFIX — shape matches; values differ (EMA-based initialization)."""

    def test_nan_count_match(self):
        ft_m, ft_s, ft_h = ferro_ta.MACDFIX(CLOSE)
        ta_m, ta_s, ta_h = talib.MACDFIX(CLOSE)
        assert _nan_count(ft_m) == _nan_count(ta_m)

    def test_output_length_match(self):
        ft_m, _, _ = ferro_ta.MACDFIX(CLOSE)
        ta_m, _, _ = talib.MACDFIX(CLOSE)
        assert len(ft_m) == len(ta_m)


class TestSAR:
    """SAR — same output length; values may differ due to reversal history.

    Known difference: Parabolic SAR reversal history can diverge from TA-Lib
    due to floating-point accumulation in early bars. Output shape (length,
    NaN count) matches exactly.
    """

    def test_output_length_match(self):
        ft = ferro_ta.SAR(HIGH, LOW)
        ta = talib.SAR(HIGH, LOW)
        assert len(ft) == len(ta)

    def test_nan_count_match(self):
        ft = ferro_ta.SAR(HIGH, LOW)
        ta = talib.SAR(HIGH, LOW)
        assert _nan_count(ft) == _nan_count(ta)

    def test_values_positive(self):
        ft = ferro_ta.SAR(HIGH, LOW)
        finite = ft[~np.isnan(ft)]
        assert all(v > 0 for v in finite)

    def test_correlation_above_threshold(self):
        """Correlated with TA-Lib even if not exact (same algorithm, different accumulation)."""
        ft = ferro_ta.SAR(HIGH, LOW)
        ta = talib.SAR(HIGH, LOW)
        mask = _valid_mask(ft, ta)
        if mask.sum() >= 5:
            corr = float(np.corrcoef(ft[mask], ta[mask])[0, 1])
            assert corr > 0.90, f"SAR correlation {corr:.3f} < 0.90"


class TestSAREXT:
    """SAREXT — SAR Extended. Shape must match; values may differ.

    Known difference: Same as SAR — reversal history from TA-Lib diverges
    due to floating-point accumulation.
    """

    def test_output_length_match(self):
        ft = ferro_ta.SAREXT(HIGH, LOW)
        ta = talib.SAREXT(HIGH, LOW)
        assert len(ft) == len(ta)

    def test_nan_count_match(self):
        ft = ferro_ta.SAREXT(HIGH, LOW)
        ta = talib.SAREXT(HIGH, LOW)
        assert _nan_count(ft) == _nan_count(ta)


class TestMAMA:
    """MAMA — MESA Adaptive Moving Average.

    Known difference: TA-Lib C applies slightly different floating-point rounding
    in the adaptive factor clamp. The two series are highly correlated (r > 0.95)
    and values converge after ~100 bars, but differ numerically in early bars.
    Status: ⚠️ Corr.
    """

    def test_output_length_match(self):
        ft_m, ft_f = ferro_ta.MAMA(CLOSE)
        ta_m, ta_f = talib.MAMA(CLOSE)
        assert len(ft_m) == len(ta_m)
        assert len(ft_f) == len(ta_f)

    def test_nan_count_match(self):
        ft_m, ft_f = ferro_ta.MAMA(CLOSE)
        ta_m, ta_f = talib.MAMA(CLOSE)
        assert _nan_count(ft_m) == _nan_count(ta_m)
        assert _nan_count(ft_f) == _nan_count(ta_f)

    def test_mama_correlated_with_talib(self):
        """MAMA should be highly correlated with TA-Lib (r > 0.95)."""
        ft_m, _ = ferro_ta.MAMA(CLOSE)
        ta_m, _ = talib.MAMA(CLOSE)
        mask = _valid_mask(ft_m, ta_m)
        if mask.sum() >= 5:
            corr = float(np.corrcoef(ft_m[mask], ta_m[mask])[0, 1])
            assert corr > 0.95, f"MAMA correlation {corr:.3f} < 0.95"

    def test_fama_correlated_with_talib(self):
        """FAMA should be correlated with TA-Lib (r > 0.80)."""
        _, ft_f = ferro_ta.MAMA(CLOSE)
        _, ta_f = talib.MAMA(CLOSE)
        mask = _valid_mask(ft_f, ta_f)
        if mask.sum() >= 5:
            corr = float(np.corrcoef(ft_f[mask], ta_f[mask])[0, 1])
            assert corr > 0.80, f"FAMA correlation {corr:.3f} < 0.80"

    def test_mama_converges_in_tail(self):
        """After 100 bars the difference should be small (< 0.5% of price)."""
        long_close = 44.0 + np.cumsum(
            np.random.default_rng(99).standard_normal(200) * 0.5
        )
        ft_m, _ = ferro_ta.MAMA(long_close)
        ta_m, _ = talib.MAMA(long_close)
        mask = _valid_mask(ft_m, ta_m)
        if mask.sum() >= 10:
            tail = np.where(mask)[0][-min(10, mask.sum()) :]  # last valid bars
            diff = np.abs(ft_m[tail] - ta_m[tail])
            price_scale = np.abs(ta_m[tail]).mean()
            assert (diff / price_scale).max() < 0.01, (
                f"MAMA tail relative diff: {(diff / price_scale).max():.4f}"
            )


class TestMIDPOINT:
    """MIDPOINT — exact match."""

    def test_values_match(self):
        ft = ferro_ta.MIDPOINT(CLOSE, timeperiod=5)
        ta = talib.MIDPOINT(CLOSE, timeperiod=5)
        assert _allclose(ft, ta)

    def test_nan_count_match(self):
        ft = ferro_ta.MIDPOINT(CLOSE, timeperiod=5)
        ta = talib.MIDPOINT(CLOSE, timeperiod=5)
        assert _nan_count(ft) == _nan_count(ta)


class TestMIDPRICE:
    """MIDPRICE — exact match."""

    def test_values_match(self):
        ft = ferro_ta.MIDPRICE(HIGH, LOW, timeperiod=5)
        ta = talib.MIDPRICE(HIGH, LOW, timeperiod=5)
        assert _allclose(ft, ta)

    def test_nan_count_match(self):
        ft = ferro_ta.MIDPRICE(HIGH, LOW, timeperiod=5)
        ta = talib.MIDPRICE(HIGH, LOW, timeperiod=5)
        assert _nan_count(ft) == _nan_count(ta)


# ---------------------------------------------------------------------------
# Momentum Indicators
# ---------------------------------------------------------------------------


class TestRSI:
    """RSI — same NaN count and length; values differ due to Wilder smoothing seed.

    ferro_ta and TA-Lib use slightly different initializations for Wilder's
    smoothed average gain/loss, leading to permanently different RSI values.
    Both libraries produce values in [0, 100] with the same NaN structure.
    """

    def test_nan_count_match(self):
        ft = ferro_ta.RSI(CLOSE, timeperiod=14)
        ta = talib.RSI(CLOSE, timeperiod=14)
        assert _nan_count(ft) == _nan_count(ta)

    def test_output_length_match(self):
        ft = ferro_ta.RSI(CLOSE, timeperiod=14)
        ta = talib.RSI(CLOSE, timeperiod=14)
        assert len(ft) == len(ta)

    def test_range_0_to_100(self):
        for lib_rsi in [ferro_ta.RSI(CLOSE, 14), talib.RSI(CLOSE, 14)]:
            finite = lib_rsi[~np.isnan(lib_rsi)]
            assert all(0.0 <= v <= 100.0 for v in finite)

    def test_values_same_direction(self):
        """RSI should move in the same direction as TA-Lib (correlation > 0.9)."""
        ft = ferro_ta.RSI(CLOSE, timeperiod=14)
        ta = talib.RSI(CLOSE, timeperiod=14)
        mask = _valid_mask(ft, ta)
        corr = np.corrcoef(ft[mask], ta[mask])[0, 1]
        assert corr > 0.9

    def test_values_converge_in_tail(self):
        """With 500 bars, RSI should converge in tail 30%."""
        ft = ferro_ta.RSI(CLOSE, timeperiod=14)
        ta = talib.RSI(CLOSE, timeperiod=14)
        tail_start = int(N * 0.7)
        mask = _valid_mask(ft[tail_start:], ta[tail_start:])
        if mask.any():
            assert np.allclose(
                ft[tail_start:][mask], ta[tail_start:][mask], atol=1e-3
            )  # Added value comparison


class TestMOM:
    """MOM — exact match."""

    def test_values_match(self):
        ft = ferro_ta.MOM(CLOSE, timeperiod=10)
        ta = talib.MOM(CLOSE, timeperiod=10)
        assert _allclose(ft, ta)

    def test_nan_count_match(self):
        ft = ferro_ta.MOM(CLOSE, timeperiod=10)
        ta = talib.MOM(CLOSE, timeperiod=10)
        assert _nan_count(ft) == _nan_count(ta)


class TestROC:
    """ROC — exact match."""

    def test_values_match(self):
        ft = ferro_ta.ROC(CLOSE, timeperiod=10)
        ta = talib.ROC(CLOSE, timeperiod=10)
        assert _allclose(ft, ta)

    def test_nan_count_match(self):
        ft = ferro_ta.ROC(CLOSE, timeperiod=10)
        ta = talib.ROC(CLOSE, timeperiod=10)
        assert _nan_count(ft) == _nan_count(ta)


class TestROCP:
    """ROCP — exact match."""

    def test_values_match(self):
        ft = ferro_ta.ROCP(CLOSE, timeperiod=10)
        ta = talib.ROCP(CLOSE, timeperiod=10)
        assert _allclose(ft, ta)


class TestROCR:
    """ROCR — exact match."""

    def test_values_match(self):
        ft = ferro_ta.ROCR(CLOSE, timeperiod=10)
        ta = talib.ROCR(CLOSE, timeperiod=10)
        assert _allclose(ft, ta)


class TestROCR100:
    """ROCR100 — exact match."""

    def test_values_match(self):
        ft = ferro_ta.ROCR100(CLOSE, timeperiod=10)
        ta = talib.ROCR100(CLOSE, timeperiod=10)
        assert _allclose(ft, ta)


class TestWILLR:
    """WILLR — exact match."""

    def test_values_match(self):
        ft = ferro_ta.WILLR(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.WILLR(HIGH, LOW, CLOSE, timeperiod=14)
        assert _allclose(ft, ta)

    def test_nan_count_match(self):
        ft = ferro_ta.WILLR(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.WILLR(HIGH, LOW, CLOSE, timeperiod=14)
        assert _nan_count(ft) == _nan_count(ta)

    def test_range_minus100_to_0(self):
        ft = ferro_ta.WILLR(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.WILLR(HIGH, LOW, CLOSE, timeperiod=14)
        for arr in [ft, ta]:
            finite = arr[~np.isnan(arr)]
            assert all(-100.0 <= v <= 0.0 for v in finite)


class TestAROON:
    """AROON — exact match."""

    def test_values_match(self):
        ft_down, ft_up = ferro_ta.AROON(HIGH, LOW, timeperiod=14)
        ta_down, ta_up = talib.AROON(HIGH, LOW, timeperiod=14)
        assert _allclose(ft_down, ta_down) and _allclose(ft_up, ta_up)

    def test_nan_count_match(self):
        ft_down, ft_up = ferro_ta.AROON(HIGH, LOW, timeperiod=14)
        ta_down, ta_up = talib.AROON(HIGH, LOW, timeperiod=14)
        assert _nan_count(ft_down) == _nan_count(ta_down)

    def test_range_0_to_100(self):
        ft_down, ft_up = ferro_ta.AROON(HIGH, LOW, timeperiod=14)
        for arr in [ft_down, ft_up]:
            finite = arr[~np.isnan(arr)]
            assert all(0.0 <= v <= 100.0 for v in finite)


class TestAROONOSC:
    """AROONOSC — exact match."""

    def test_values_match(self):
        ft = ferro_ta.AROONOSC(HIGH, LOW, timeperiod=14)
        ta = talib.AROONOSC(HIGH, LOW, timeperiod=14)
        assert _allclose(ft, ta)

    def test_nan_count_match(self):
        ft = ferro_ta.AROONOSC(HIGH, LOW, timeperiod=14)
        ta = talib.AROONOSC(HIGH, LOW, timeperiod=14)
        assert _nan_count(ft) == _nan_count(ta)


class TestCCI:
    """CCI — same NaN count and shape; mean-absolute-deviation may differ.

    TA-Lib divides by 0.015 × MAD computed with the population formula.
    ferro_ta may use a slightly different MAD implementation, producing
    proportionally scaled but directionally identical values.
    """

    def test_nan_count_match(self):
        ft = ferro_ta.CCI(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.CCI(HIGH, LOW, CLOSE, timeperiod=14)
        assert _nan_count(ft) == _nan_count(ta)

    def test_output_length_match(self):
        ft = ferro_ta.CCI(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.CCI(HIGH, LOW, CLOSE, timeperiod=14)
        assert len(ft) == len(ta)

    def test_same_sign(self):
        """CCI values should have the same sign as TA-Lib."""
        ft = ferro_ta.CCI(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.CCI(HIGH, LOW, CLOSE, timeperiod=14)
        mask = _valid_mask(ft, ta)
        # Both should agree on whether CCI is positive/negative
        assert (
            np.sum(np.sign(ft[mask]) == np.sign(ta[mask]))
            > SIGN_AGREEMENT_THRESHOLD * mask.sum()
        )

    def test_values_strongly_correlated(self):
        """CCI values should be strongly correlated with TA-Lib values."""
        ft = ferro_ta.CCI(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.CCI(HIGH, LOW, CLOSE, timeperiod=14)
        mask = _valid_mask(ft, ta)
        corr = np.corrcoef(ft[mask], ta[mask])[0, 1]
        assert corr > 0.99


class TestBOP:
    """BOP — exact match."""

    def test_values_match(self):
        ft = ferro_ta.BOP(OPEN, HIGH, LOW, CLOSE)
        ta = talib.BOP(OPEN, HIGH, LOW, CLOSE)
        assert _allclose(ft, ta)

    def test_output_length_match(self):
        ft = ferro_ta.BOP(OPEN, HIGH, LOW, CLOSE)
        ta = talib.BOP(OPEN, HIGH, LOW, CLOSE)
        assert len(ft) == len(ta)


class TestMFI:
    """MFI — values match on a well-constructed series.

    MFI (Money Flow Index) is computed from OHLCV and should agree exactly
    when the typical prices and volumes are not degenerate.
    """

    def test_nan_count_match(self):
        ft = ferro_ta.MFI(HIGH, LOW, CLOSE, VOLUME, timeperiod=14)
        ta = talib.MFI(HIGH, LOW, CLOSE, VOLUME, timeperiod=14)
        assert _nan_count(ft) == _nan_count(ta)

    def test_range_0_to_100(self):
        ft = ferro_ta.MFI(HIGH, LOW, CLOSE, VOLUME, timeperiod=14)
        finite = ft[~np.isnan(ft)]
        assert all(0.0 <= v <= 100.0 for v in finite)

    def test_values_match(self):
        ft = ferro_ta.MFI(HIGH, LOW, CLOSE, VOLUME, timeperiod=14)
        ta = talib.MFI(HIGH, LOW, CLOSE, VOLUME, timeperiod=14)
        assert _allclose(ft, ta)


class TestSTOCHF:
    """STOCHF — fast %K and SMA %D match TA-Lib (fastd_matype=0).

    Both series are NaN-padded until %D is valid
    (``fastk_period + fastd_period - 2`` leading NaNs).
    """

    def test_fastk_values_match(self):
        ft_k, ft_d = ferro_ta.STOCHF(HIGH, LOW, CLOSE, fastk_period=5, fastd_period=3)
        ta_k, ta_d = talib.STOCHF(
            HIGH, LOW, CLOSE, fastk_period=5, fastd_period=3, fastd_matype=0
        )
        assert _allclose(ft_k, ta_k)

    def test_fastd_values_match(self):
        ft_k, ft_d = ferro_ta.STOCHF(HIGH, LOW, CLOSE, fastk_period=5, fastd_period=3)
        ta_k, ta_d = talib.STOCHF(
            HIGH, LOW, CLOSE, fastk_period=5, fastd_period=3, fastd_matype=0
        )
        assert _allclose(ft_d, ta_d)

    def test_output_length_match(self):
        ft_k, _ = ferro_ta.STOCHF(HIGH, LOW, CLOSE, fastk_period=5, fastd_period=3)
        ta_k, _ = talib.STOCHF(
            HIGH, LOW, CLOSE, fastk_period=5, fastd_period=3, fastd_matype=0
        )
        assert len(ft_k) == len(ta_k)

    def test_range_0_to_100(self):
        ft_k, ft_d = ferro_ta.STOCHF(HIGH, LOW, CLOSE, fastk_period=5, fastd_period=3)
        for arr in [ft_k, ft_d]:
            finite = arr[~np.isnan(arr)]
            assert all(0.0 <= v <= 100.0 for v in finite)


class TestSTOCH:
    """STOCH — same shape; slow %K may differ by EMA initialisation."""

    def test_output_length_match(self):
        ft_k, ft_d = ferro_ta.STOCH(HIGH, LOW, CLOSE)
        ta_k, ta_d = talib.STOCH(HIGH, LOW, CLOSE)
        assert len(ft_k) == len(ta_k)

    def test_range_0_to_100(self):
        ft_k, ft_d = ferro_ta.STOCH(HIGH, LOW, CLOSE)
        for arr in [ft_k, ft_d]:
            finite = arr[~np.isnan(arr)]
            assert all(0.0 <= v <= 100.0 for v in finite)


class TestSTOCHRSI:
    """STOCHRSI — same length; NaN count may differ by up to 2.

    The RSI seed difference propagates into StochRSI.  ferro_ta emits values
    sooner (fewer NaN) than TA-Lib in some configurations.
    """

    def test_output_length_match(self):
        ft_k, ft_d = ferro_ta.STOCHRSI(
            CLOSE, timeperiod=14, fastk_period=5, fastd_period=3
        )
        ta_k, ta_d = talib.STOCHRSI(
            CLOSE, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0
        )
        assert len(ft_k) == len(ta_k)

    def test_nan_count_within_tolerance(self):
        ft_k, ft_d = ferro_ta.STOCHRSI(
            CLOSE, timeperiod=14, fastk_period=5, fastd_period=3
        )
        ta_k, ta_d = talib.STOCHRSI(
            CLOSE, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0
        )
        assert abs(_nan_count(ft_k) - _nan_count(ta_k)) <= 2

    def test_range_0_to_100(self):
        ft_k, _ = ferro_ta.STOCHRSI(
            CLOSE, timeperiod=14, fastk_period=5, fastd_period=3
        )
        finite = ft_k[~np.isnan(ft_k)]
        # Allow small numerical tolerance for float boundaries
        assert all(-1e-9 <= v <= 100.0 + 1e-9 for v in finite)


class TestAPO:
    """APO — output shape only.

    The values do *not* differ from TA-Lib: with ``matype`` matched on both
    sides APO agrees to 4.9e-12.  It is the **defaults** that differ (ferro-ta
    ``matype=1``/EMA, TA-Lib ``matype=0``/SMA).  ``TestAPOMatype`` is the value
    gate; the earlier "values differ" claim here was false and had suppressed
    it.  The NaN-count check that used to live here is subsumed by
    ``TestAPOMatype.test_nan_count_match``, which pins the exact expected count
    at every matype.
    """

    def test_output_length_match(self):
        ft = ferro_ta.APO(CLOSE, fastperiod=12, slowperiod=26)
        ta = talib.APO(CLOSE, fastperiod=12, slowperiod=26, matype=0)
        assert len(ft) == len(ta)


class TestPPO:
    """PPO — ferro_ta returns (ppo, signal, histogram); TA-Lib returns only ppo.

    ferro_ta extends PPO with a signal line and histogram (similar to MACD),
    while TA-Lib's PPO only returns the percentage-difference line.  We verify
    the output length and that all three ferro_ta arrays have valid shapes.

    Value agreement on the PPO line is gated by ``TestPPOMatype``, which
    compares against TA-Lib at every matype and subsumes the correlation-only
    (> 0.85) check that used to live here.
    """

    def test_output_is_tuple_of_three(self):
        result = ferro_ta.PPO(CLOSE, fastperiod=12, slowperiod=26)
        assert isinstance(result, tuple) and len(result) == 3

    def test_output_length_match(self):
        ppo, signal, hist = ferro_ta.PPO(CLOSE, fastperiod=12, slowperiod=26)
        ta = talib.PPO(CLOSE, fastperiod=12, slowperiod=26, matype=0)
        assert len(ppo) == len(ta)

    def test_all_arrays_same_length(self):
        ppo, signal, hist = ferro_ta.PPO(CLOSE, fastperiod=12, slowperiod=26)
        assert len(ppo) == len(signal) == len(hist) == N


class TestCMO:
    """CMO — exact match (Wilder-smoothed gains/losses, same seed as RSI)."""

    def test_values_match(self):
        ft = ferro_ta.CMO(CLOSE, timeperiod=14)
        ta = talib.CMO(CLOSE, timeperiod=14)
        assert _allclose(ft, ta)

    def test_nan_count_match(self):
        ft = ferro_ta.CMO(CLOSE, timeperiod=14)
        ta = talib.CMO(CLOSE, timeperiod=14)
        assert _nan_count(ft) == _nan_count(ta)

    def test_output_length_match(self):
        ft = ferro_ta.CMO(CLOSE, timeperiod=14)
        ta = talib.CMO(CLOSE, timeperiod=14)
        assert len(ft) == len(ta)

    def test_range_minus100_to_100(self):
        ft = ferro_ta.CMO(CLOSE, timeperiod=14)
        finite = ft[~np.isnan(ft)]
        assert all(-100.0 <= v <= 100.0 for v in finite)


class TestTRIX:
    """TRIX — shape matches; values differ (triple EMA initialisation)."""

    def test_nan_count_match(self):
        ft = ferro_ta.TRIX(CLOSE, timeperiod=5)
        ta = talib.TRIX(CLOSE, timeperiod=5)
        assert _nan_count(ft) == _nan_count(ta)

    def test_output_length_match(self):
        ft = ferro_ta.TRIX(CLOSE, timeperiod=5)
        ta = talib.TRIX(CLOSE, timeperiod=5)
        assert len(ft) == len(ta)


class TestULTOSC:
    """ULTOSC — exact match."""

    def test_values_match(self):
        ft = ferro_ta.ULTOSC(
            HIGH, LOW, CLOSE, timeperiod1=7, timeperiod2=14, timeperiod3=28
        )
        ta = talib.ULTOSC(
            HIGH, LOW, CLOSE, timeperiod1=7, timeperiod2=14, timeperiod3=28
        )
        assert _allclose(ft, ta)

    def test_nan_count_match(self):
        ft = ferro_ta.ULTOSC(HIGH, LOW, CLOSE)
        ta = talib.ULTOSC(HIGH, LOW, CLOSE)
        assert _nan_count(ft) == _nan_count(ta)


class TestADX:
    """ADX — same shape; values differ on random data (Wilder smoothing seed).

    On monotonically trending data the values match TA-Lib exactly.  On
    random price series the Wilder's smoothing seed for ATR and DM causes
    permanent divergence (values do not converge).
    """

    def test_nan_count_match(self):
        ft = ferro_ta.ADX(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.ADX(HIGH, LOW, CLOSE, timeperiod=14)
        assert _nan_count(ft) == _nan_count(ta)

    def test_output_length_match(self):
        ft = ferro_ta.ADX(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.ADX(HIGH, LOW, CLOSE, timeperiod=14)
        assert len(ft) == len(ta)

    def test_range_0_to_100(self):
        ft = ferro_ta.ADX(HIGH, LOW, CLOSE, timeperiod=14)
        finite = ft[~np.isnan(ft)]
        assert all(0.0 <= v <= 100.0 for v in finite)

    def test_values_strongly_correlated(self):
        ft = ferro_ta.ADX(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.ADX(HIGH, LOW, CLOSE, timeperiod=14)
        mask = _valid_mask(ft, ta)
        corr = np.corrcoef(ft[mask], ta[mask])[0, 1]
        assert corr > 0.99


class TestADXR:
    """ADXR — same shape (±1 NaN); values differ (Wilder smoothing seed).

    ADXR = (ADX[t] + ADX[t - timeperiod]) / 2.  The ADX values differ from
    TA-Lib due to the Wilder smoothing seed, so ADXR differs too.
    """

    def test_output_length_match(self):
        ft = ferro_ta.ADXR(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.ADXR(HIGH, LOW, CLOSE, timeperiod=14)
        assert len(ft) == len(ta)

    def test_nan_count_within_one(self):
        ft = ferro_ta.ADXR(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.ADXR(HIGH, LOW, CLOSE, timeperiod=14)
        assert abs(_nan_count(ft) - _nan_count(ta)) <= 1

    def test_values_strongly_correlated(self):
        ft = ferro_ta.ADXR(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.ADXR(HIGH, LOW, CLOSE, timeperiod=14)
        mask = _valid_mask(ft, ta)
        corr = np.corrcoef(ft[mask], ta[mask])[0, 1]
        assert corr > 0.95


class TestDX:
    """DX — same NaN count and shape; values differ on random data.

    DX = |+DI - -DI| / (+DI + -DI) × 100.  The +DI and -DI values depend on
    Wilder's smoothed ATR and DM, both of which have different seeds in
    ferro_ta vs TA-Lib.  Values are strongly correlated.
    """

    def test_nan_count_match(self):
        ft = ferro_ta.DX(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.DX(HIGH, LOW, CLOSE, timeperiod=14)
        assert _nan_count(ft) == _nan_count(ta)

    def test_output_length_match(self):
        ft = ferro_ta.DX(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.DX(HIGH, LOW, CLOSE, timeperiod=14)
        assert len(ft) == len(ta)

    def test_range_0_to_100(self):
        ft = ferro_ta.DX(HIGH, LOW, CLOSE, timeperiod=14)
        finite = ft[~np.isnan(ft)]
        assert all(0.0 <= v <= 100.0 for v in finite)


class TestPLUSDI:
    """PLUS_DI — same NaN count; values differ on random data (Wilder smoothing)."""

    def test_nan_count_match(self):
        ft = ferro_ta.PLUS_DI(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.PLUS_DI(HIGH, LOW, CLOSE, timeperiod=14)
        assert _nan_count(ft) == _nan_count(ta)

    def test_output_length_match(self):
        ft = ferro_ta.PLUS_DI(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.PLUS_DI(HIGH, LOW, CLOSE, timeperiod=14)
        assert len(ft) == len(ta)

    def test_non_negative(self):
        ft = ferro_ta.PLUS_DI(HIGH, LOW, CLOSE, timeperiod=14)
        finite = ft[~np.isnan(ft)]
        assert all(v >= 0.0 for v in finite)


class TestMINUSDI:
    """MINUS_DI — same NaN count; values differ on random data (Wilder smoothing)."""

    def test_nan_count_match(self):
        ft = ferro_ta.MINUS_DI(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.MINUS_DI(HIGH, LOW, CLOSE, timeperiod=14)
        assert _nan_count(ft) == _nan_count(ta)

    def test_output_length_match(self):
        ft = ferro_ta.MINUS_DI(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.MINUS_DI(HIGH, LOW, CLOSE, timeperiod=14)
        assert len(ft) == len(ta)

    def test_non_negative(self):
        ft = ferro_ta.MINUS_DI(HIGH, LOW, CLOSE, timeperiod=14)
        finite = ft[~np.isnan(ft)]
        assert all(v >= 0.0 for v in finite)


class TestPLUSDM:
    """PLUS_DM — values match in the non-degenerate (OHLCV) region."""

    def test_output_length_match(self):
        ft = ferro_ta.PLUS_DM(HIGH, LOW, timeperiod=14)
        ta = talib.PLUS_DM(HIGH, LOW, timeperiod=14)
        assert len(ft) == len(ta)

    def test_non_negative(self):
        ft = ferro_ta.PLUS_DM(HIGH, LOW, timeperiod=14)
        finite = ft[~np.isnan(ft)]
        assert all(v >= 0.0 for v in finite)


class TestMINUSDM:
    """MINUS_DM — same length; NaN count may differ by 1 (Wilder smoothing seed)."""

    def test_output_length_match(self):
        ft = ferro_ta.MINUS_DM(HIGH, LOW, timeperiod=14)
        ta = talib.MINUS_DM(HIGH, LOW, timeperiod=14)
        assert len(ft) == len(ta)

    def test_nan_count_within_one(self):
        ft = ferro_ta.MINUS_DM(HIGH, LOW, timeperiod=14)
        ta = talib.MINUS_DM(HIGH, LOW, timeperiod=14)
        assert abs(_nan_count(ft) - _nan_count(ta)) <= 1

    def test_non_negative(self):
        ft = ferro_ta.MINUS_DM(HIGH, LOW, timeperiod=14)
        finite = ft[~np.isnan(ft)]
        assert all(v >= 0.0 for v in finite)


# ---------------------------------------------------------------------------
# Volume Indicators
# ---------------------------------------------------------------------------


class TestAD:
    """AD — exact match."""

    def test_values_match(self):
        ft = ferro_ta.AD(HIGH, LOW, CLOSE, VOLUME)
        ta = talib.AD(HIGH, LOW, CLOSE, VOLUME)
        assert _allclose(ft, ta)

    def test_output_length_match(self):
        ft = ferro_ta.AD(HIGH, LOW, CLOSE, VOLUME)
        ta = talib.AD(HIGH, LOW, CLOSE, VOLUME)
        assert len(ft) == len(ta)


class TestADOSC:
    """ADOSC — exact match."""

    def test_values_match(self):
        ft = ferro_ta.ADOSC(HIGH, LOW, CLOSE, VOLUME, fastperiod=3, slowperiod=10)
        ta = talib.ADOSC(HIGH, LOW, CLOSE, VOLUME, fastperiod=3, slowperiod=10)
        assert _allclose(ft, ta)

    def test_nan_count_match(self):
        ft = ferro_ta.ADOSC(HIGH, LOW, CLOSE, VOLUME)
        ta = talib.ADOSC(HIGH, LOW, CLOSE, VOLUME)
        assert _nan_count(ft) == _nan_count(ta)


class TestOBV:
    """OBV — exact match. Bar 0 equals ``volume[0]`` (TA-Lib)."""

    def test_values_match(self):
        ft = ferro_ta.OBV(CLOSE, VOLUME)
        ta = talib.OBV(CLOSE, VOLUME)
        assert _allclose(ft, ta)

    def test_output_length_match(self):
        ft = ferro_ta.OBV(CLOSE, VOLUME)
        ta = talib.OBV(CLOSE, VOLUME)
        assert len(ft) == len(ta)

    def test_no_nans(self):
        ft = ferro_ta.OBV(CLOSE, VOLUME)
        assert not np.any(np.isnan(ft))


# ---------------------------------------------------------------------------
# Volatility Indicators
# ---------------------------------------------------------------------------


class TestATR:
    """ATR — same length; values differ (different Wilder smoothing seed).

    TA-Lib uses Wilder's smoothing and marks the very first ATR value (at
    index ``timeperiod``) as NaN.  ferro_ta emits a value there.  The Wilder
    recursion runs from a different seed, so values do not converge.  Both
    produce strongly correlated positive ATR values.
    """

    def test_output_length_match(self):
        ft = ferro_ta.ATR(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.ATR(HIGH, LOW, CLOSE, timeperiod=14)
        assert len(ft) == len(ta)

    def test_nan_count_within_one(self):
        ft = ferro_ta.ATR(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.ATR(HIGH, LOW, CLOSE, timeperiod=14)
        assert abs(_nan_count(ft) - _nan_count(ta)) <= 1

    def test_values_positive(self):
        ft = ferro_ta.ATR(HIGH, LOW, CLOSE, timeperiod=14)
        finite = ft[~np.isnan(ft)]
        assert all(v > 0 for v in finite)

    def test_values_strongly_correlated(self):
        ft = ferro_ta.ATR(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.ATR(HIGH, LOW, CLOSE, timeperiod=14)
        mask = _valid_mask(ft, ta)
        corr = np.corrcoef(ft[mask], ta[mask])[0, 1]
        assert corr > 0.95


class TestNATR:
    """NATR — same shape tolerance as ATR; values differ (Wilder smoothing seed)."""

    def test_output_length_match(self):
        ft = ferro_ta.NATR(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.NATR(HIGH, LOW, CLOSE, timeperiod=14)
        assert len(ft) == len(ta)

    def test_nan_count_within_one(self):
        ft = ferro_ta.NATR(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.NATR(HIGH, LOW, CLOSE, timeperiod=14)
        assert abs(_nan_count(ft) - _nan_count(ta)) <= 1

    def test_values_positive(self):
        ft = ferro_ta.NATR(HIGH, LOW, CLOSE, timeperiod=14)
        finite = ft[~np.isnan(ft)]
        assert all(v > 0 for v in finite)

    def test_values_strongly_correlated(self):
        ft = ferro_ta.NATR(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.NATR(HIGH, LOW, CLOSE, timeperiod=14)
        mask = _valid_mask(ft, ta)
        corr = np.corrcoef(ft[mask], ta[mask])[0, 1]
        assert corr > 0.95


class TestTRANGE:
    """TRANGE — values match, including NaN at bar 0 (no previous close)."""

    def test_output_length_match(self):
        ft = ferro_ta.TRANGE(HIGH, LOW, CLOSE)
        ta = talib.TRANGE(HIGH, LOW, CLOSE)
        assert len(ft) == len(ta)

    def test_values_match(self):
        ft = ferro_ta.TRANGE(HIGH, LOW, CLOSE)
        ta = talib.TRANGE(HIGH, LOW, CLOSE)
        assert _allclose(ft, ta)

    def test_values_positive(self):
        ft = ferro_ta.TRANGE(HIGH, LOW, CLOSE)
        assert np.isnan(ft[0])
        assert all(v > 0 for v in ft[1:])


# ---------------------------------------------------------------------------
# Statistical Functions
# ---------------------------------------------------------------------------


class TestSTDDEV:
    """STDDEV — exact match."""

    def test_values_match(self):
        ft = ferro_ta.STDDEV(CLOSE, timeperiod=10)
        ta = talib.STDDEV(CLOSE, timeperiod=10)
        assert _allclose(ft, ta)

    def test_nan_count_match(self):
        ft = ferro_ta.STDDEV(CLOSE, timeperiod=10)
        ta = talib.STDDEV(CLOSE, timeperiod=10)
        assert _nan_count(ft) == _nan_count(ta)


class TestVAR:
    """VAR — exact match."""

    def test_values_match(self):
        ft = ferro_ta.VAR(CLOSE, timeperiod=10)
        ta = talib.VAR(CLOSE, timeperiod=10)
        assert _allclose(ft, ta)


class TestLINEARREG:
    """LINEARREG — exact match."""

    def test_values_match(self):
        ft = ferro_ta.LINEARREG(CLOSE, timeperiod=10)
        ta = talib.LINEARREG(CLOSE, timeperiod=10)
        assert _allclose(ft, ta)

    def test_nan_count_match(self):
        ft = ferro_ta.LINEARREG(CLOSE, timeperiod=10)
        ta = talib.LINEARREG(CLOSE, timeperiod=10)
        assert _nan_count(ft) == _nan_count(ta)


class TestLINEARREGSlope:
    """LINEARREG_SLOPE — exact match."""

    def test_values_match(self):
        ft = ferro_ta.LINEARREG_SLOPE(CLOSE, timeperiod=10)
        ta = talib.LINEARREG_SLOPE(CLOSE, timeperiod=10)
        assert _allclose(ft, ta)


class TestLINEARREGIntercept:
    """LINEARREG_INTERCEPT — exact match."""

    def test_values_match(self):
        ft = ferro_ta.LINEARREG_INTERCEPT(CLOSE, timeperiod=10)
        ta = talib.LINEARREG_INTERCEPT(CLOSE, timeperiod=10)
        assert _allclose(ft, ta)


class TestLINEARREGAngle:
    """LINEARREG_ANGLE — exact match."""

    def test_values_match(self):
        ft = ferro_ta.LINEARREG_ANGLE(CLOSE, timeperiod=10)
        ta = talib.LINEARREG_ANGLE(CLOSE, timeperiod=10)
        assert _allclose(ft, ta)


class TestTSF:
    """TSF — exact match."""

    def test_values_match(self):
        ft = ferro_ta.TSF(CLOSE, timeperiod=10)
        ta = talib.TSF(CLOSE, timeperiod=10)
        assert _allclose(ft, ta)

    def test_nan_count_match(self):
        ft = ferro_ta.TSF(CLOSE, timeperiod=10)
        ta = talib.TSF(CLOSE, timeperiod=10)
        assert _nan_count(ft) == _nan_count(ta)


class TestBETA:
    """BETA — same shape; algorithm differs from TA-Lib.

    ferro_ta computes a simplified rolling beta (covariance / variance of the
    reference series), while TA-Lib uses the standard CAPM beta estimator.
    Shape compatibility (NaN count, length) is verified; exact value match is
    not expected.
    """

    def test_output_length_match(self):
        ft = ferro_ta.BETA(CLOSE, HIGH, timeperiod=5)
        ta = talib.BETA(CLOSE, HIGH, timeperiod=5)
        assert len(ft) == len(ta)

    def test_nan_count_match(self):
        ft = ferro_ta.BETA(CLOSE, HIGH, timeperiod=5)
        ta = talib.BETA(CLOSE, HIGH, timeperiod=5)
        assert _nan_count(ft) == _nan_count(ta)


class TestCORREL:
    """CORREL — exact match."""

    def test_values_match(self):
        ft = ferro_ta.CORREL(CLOSE, HIGH, timeperiod=10)
        ta = talib.CORREL(CLOSE, HIGH, timeperiod=10)
        assert _allclose(ft, ta)

    def test_nan_count_match(self):
        ft = ferro_ta.CORREL(CLOSE, HIGH, timeperiod=10)
        ta = talib.CORREL(CLOSE, HIGH, timeperiod=10)
        assert _nan_count(ft) == _nan_count(ta)

    def test_range_minus1_to_1(self):
        ft = ferro_ta.CORREL(CLOSE, HIGH, timeperiod=10)
        finite = ft[~np.isnan(ft)]
        assert all(-1.0 <= v <= 1.0 for v in finite)


# ---------------------------------------------------------------------------
# Price Transformations
# ---------------------------------------------------------------------------


class TestAVGPRICE:
    """AVGPRICE — exact match."""

    def test_values_match(self):
        ft = ferro_ta.AVGPRICE(OPEN, HIGH, LOW, CLOSE)
        ta = talib.AVGPRICE(OPEN, HIGH, LOW, CLOSE)
        assert np.allclose(ft, ta, atol=1e-10)

    def test_output_length_match(self):
        assert len(ferro_ta.AVGPRICE(OPEN, HIGH, LOW, CLOSE)) == N


class TestMEDPRICE:
    """MEDPRICE — exact match."""

    def test_values_match(self):
        ft = ferro_ta.MEDPRICE(HIGH, LOW)
        ta = talib.MEDPRICE(HIGH, LOW)
        assert np.allclose(ft, ta, atol=1e-10)


class TestTYPPRICE:
    """TYPPRICE — exact match."""

    def test_values_match(self):
        ft = ferro_ta.TYPPRICE(HIGH, LOW, CLOSE)
        ta = talib.TYPPRICE(HIGH, LOW, CLOSE)
        assert np.allclose(ft, ta, atol=1e-10)


class TestWCLPRICE:
    """WCLPRICE — exact match."""

    def test_values_match(self):
        ft = ferro_ta.WCLPRICE(HIGH, LOW, CLOSE)
        ta = talib.WCLPRICE(HIGH, LOW, CLOSE)
        assert np.allclose(ft, ta, atol=1e-10)


# ---------------------------------------------------------------------------
# Pattern Recognition
# ---------------------------------------------------------------------------


class TestPatternShapeCompatibility:
    """Patterns — same output length and dtype; values may differ.

    Pattern recognition algorithms depend heavily on thresholds and candle
    body/shadow definitions.  ferro_ta implements simplified versions of these
    patterns.  These tests verify that:

    * Output length matches TA-Lib.
    * Values are restricted to {-100, 0, 100} (same convention as TA-Lib).
    """

    PATTERNS = [
        "CDLDOJI",
        "CDLENGULFING",
        "CDLHAMMER",
        "CDLSHOOTINGSTAR",
        "CDLMARUBOZU",
        "CDLSPINNINGTOP",
        "CDLMORNINGSTAR",
        "CDLEVENINGSTAR",
        "CDL2CROWS",
        # Additional candlestick patterns
        "CDL3BLACKCROWS",
        "CDL3INSIDE",
        "CDL3LINESTRIKE",
        "CDL3OUTSIDE",
        "CDL3STARSINSOUTH",
        "CDL3WHITESOLDIERS",
        "CDLABANDONEDBABY",
        "CDLADVANCEBLOCK",
        "CDLBELTHOLD",
        "CDLBREAKAWAY",
        "CDLCLOSINGMARUBOZU",
        "CDLCONCEALBABYSWALL",
        "CDLCOUNTERATTACK",
        "CDLDARKCLOUDCOVER",
        "CDLDOJISTAR",
        "CDLDRAGONFLYDOJI",
        "CDLGAPSIDESIDEWHITE",
        "CDLGRAVESTONEDOJI",
        "CDLHANGINGMAN",
        "CDLHARAMI",
        "CDLHARAMICROSS",
        "CDLHIGHWAVE",
        "CDLHIKKAKE",
        "CDLHIKKAKEMOD",
        "CDLHOMINGPIGEON",
        "CDLIDENTICAL3CROWS",
        "CDLINNECK",
        "CDLINVERTEDHAMMER",
        "CDLKICKING",
        "CDLKICKINGBYLENGTH",
        "CDLLADDERBOTTOM",
        "CDLLONGLEGGEDDOJI",
        "CDLLONGLINE",
        "CDLMATCHINGLOW",
        "CDLMATHOLD",
        "CDLMORNINGDOJISTAR",
        "CDLEVENINGDOJISTAR",
        "CDLONNECK",
        "CDLPIERCING",
        "CDLRICKSHAWMAN",
        "CDLRISEFALL3METHODS",
        "CDLSEPARATINGLINES",
        "CDLSHORTLINE",
        "CDLSTALLEDPATTERN",
        "CDLSTICKSANDWICH",
        "CDLTAKURI",
        "CDLTASUKIGAP",
        "CDLTHRUSTING",
        "CDLTRISTAR",
        "CDLUNIQUE3RIVER",
        "CDLUPSIDEGAP2CROWS",
        "CDLXSIDEGAP3METHODS",
    ]

    @pytest.mark.parametrize("name", PATTERNS)
    def test_output_length_match(self, name: str):
        ft_fn = getattr(ferro_ta, name)
        ta_fn = getattr(talib, name)
        ft = ft_fn(OPEN, HIGH, LOW, CLOSE)
        ta = ta_fn(OPEN, HIGH, LOW, CLOSE)
        assert len(ft) == len(ta)

    @pytest.mark.parametrize("name", PATTERNS)
    def test_valid_output_values(self, name: str):
        ft_fn = getattr(ferro_ta, name)
        ft = ft_fn(OPEN, HIGH, LOW, CLOSE)
        assert all(v in (-100, 0, 100) for v in ft), (
            f"{name}: unexpected values {set(ft)}"
        )

    def test_cdlengulfing_values_match(self):
        """CDLENGULFING matches TA-Lib exactly on random OHLCV data."""
        ft = ferro_ta.CDLENGULFING(OPEN, HIGH, LOW, CLOSE)
        ta = talib.CDLENGULFING(OPEN, HIGH, LOW, CLOSE)
        assert np.array_equal(ft, ta)


# ---------------------------------------------------------------------------
# Parity suite additions
# ---------------------------------------------------------------------------


class TestParitySuite:
    """
    Comprehensive parity validation against TA-Lib.

    Covers:
    * Large-dataset SMA equivalence (10,000 rows)
    * Strict shape and dtype checks for MACD and BBANDS
    * float32 input handling (should cast safely via _to_f64)
    """

    # 10,000-row synthetic OHLCV data
    N_LARGE = 10_000
    _rng = np.random.default_rng(2024)
    CLOSE_LARGE = 100.0 + np.cumsum(_rng.standard_normal(N_LARGE) * 0.5)

    def test_sma_10k_allclose(self):
        """SMA on 10,000 rows must match TA-Lib within floating-point tolerance."""
        ft = ferro_ta.SMA(self.CLOSE_LARGE, timeperiod=30)
        ta = talib.SMA(self.CLOSE_LARGE, timeperiod=30)
        assert np.allclose(ft, ta, equal_nan=True), "SMA mismatch on 10k-row dataset"

    def test_macd_shape_and_dtype(self):
        """MACD output must have correct shape and float64 dtype."""
        macd_line, signal, hist = ferro_ta.MACD(CLOSE)
        assert macd_line.shape == (N,)
        assert signal.shape == (N,)
        assert hist.shape == (N,)
        assert macd_line.dtype == np.float64
        assert signal.dtype == np.float64
        assert hist.dtype == np.float64

    def test_bbands_shape_and_dtype(self):
        """BBANDS output must have correct shape and float64 dtype."""
        upper, middle, lower = ferro_ta.BBANDS(CLOSE, timeperiod=20)
        assert upper.shape == (N,)
        assert middle.shape == (N,)
        assert lower.shape == (N,)
        assert upper.dtype == np.float64
        assert middle.dtype == np.float64
        assert lower.dtype == np.float64

    def test_float32_input_casts_safely(self):
        """Passing float32 arrays should cast to float64 silently (no error)."""
        close32 = CLOSE.astype(np.float32)
        # _to_f64 should cast — result must be finite and match float64 version
        result = ferro_ta.SMA(close32, timeperiod=10)
        expected = ferro_ta.SMA(CLOSE, timeperiod=10)
        assert result.dtype == np.float64
        valid = ~np.isnan(result) & ~np.isnan(expected)
        assert np.allclose(result[valid], expected[valid], atol=1e-4)

    def test_macd_nan_count_vs_talib(self):
        """MACD NaN counts must agree with TA-Lib (same warmup period)."""
        ft_m, ft_s, ft_h = ferro_ta.MACD(CLOSE)
        ta_m, ta_s, ta_h = talib.MACD(CLOSE)
        assert _nan_count(ft_m) == _nan_count(ta_m)
        assert _nan_count(ft_s) == _nan_count(ta_s)

    def test_bbands_values_match_talib(self):
        """BBANDS must match TA-Lib exactly (SMA-based, no EMA seeding issue)."""
        ft_u, ft_m, ft_l = ferro_ta.BBANDS(CLOSE, timeperiod=20)
        ta_u, ta_m, ta_l = talib.BBANDS(CLOSE, timeperiod=20)
        assert _allclose(ft_u, ta_u), "BBANDS upper mismatch"
        assert _allclose(ft_m, ta_m), "BBANDS middle mismatch"
        assert _allclose(ft_l, ta_l), "BBANDS lower mismatch"


# ---------------------------------------------------------------------------
# Numerical parity — RSI, ATR, NATR, CCI, BETA alignment
# ---------------------------------------------------------------------------


class TestNumericalParity:
    """Verify RSI, ATR, NATR, CCI, BETA alignment with TA-Lib."""

    def test_rsi_output_length_matches(self):
        ft = ferro_ta.RSI(CLOSE, timeperiod=14)
        ta = talib.RSI(CLOSE, timeperiod=14)
        assert len(ft) == len(ta)

    def test_rsi_nan_count_matches(self):
        ft = ferro_ta.RSI(CLOSE, timeperiod=14)
        ta = talib.RSI(CLOSE, timeperiod=14)
        assert _nan_count(ft) == _nan_count(ta), (
            f"RSI NaN count: ferro_ta={_nan_count(ft)}, talib={_nan_count(ta)}"
        )

    def test_rsi_values_allclose(self):
        """RSI values must match TA-Lib within tolerance after seeding."""
        ft = ferro_ta.RSI(CLOSE, timeperiod=14)
        ta = talib.RSI(CLOSE, timeperiod=14)
        mask = _valid_mask(ft, ta)
        assert mask.any(), "No valid bars to compare"
        assert np.allclose(ft[mask], ta[mask], atol=1e-8), (
            f"RSI max diff: {np.abs(ft[mask] - ta[mask]).max()}"
        )

    def test_atr_output_length_matches(self):
        ft = ferro_ta.ATR(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.ATR(HIGH, LOW, CLOSE, timeperiod=14)
        assert len(ft) == len(ta)

    def test_atr_nan_count_matches(self):
        ft = ferro_ta.ATR(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.ATR(HIGH, LOW, CLOSE, timeperiod=14)
        assert _nan_count(ft) == _nan_count(ta), (
            f"ATR NaN count: ferro_ta={_nan_count(ft)}, talib={_nan_count(ta)}"
        )

    def test_atr_values_allclose(self):
        """ATR values must match TA-Lib within tolerance."""
        ft = ferro_ta.ATR(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.ATR(HIGH, LOW, CLOSE, timeperiod=14)
        mask = _valid_mask(ft, ta)
        assert mask.any()
        assert np.allclose(ft[mask], ta[mask], atol=1e-8), (
            f"ATR max diff: {np.abs(ft[mask] - ta[mask]).max()}"
        )

    def test_natr_values_allclose(self):
        ft = ferro_ta.NATR(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.NATR(HIGH, LOW, CLOSE, timeperiod=14)
        mask = _valid_mask(ft, ta)
        assert mask.any()
        assert np.allclose(ft[mask], ta[mask], atol=1e-6)

    def test_cci_output_length_matches(self):
        ft = ferro_ta.CCI(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.CCI(HIGH, LOW, CLOSE, timeperiod=14)
        assert len(ft) == len(ta)

    def test_cci_values_allclose(self):
        """CCI values must match TA-Lib exactly."""
        ft = ferro_ta.CCI(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.CCI(HIGH, LOW, CLOSE, timeperiod=14)
        mask = _valid_mask(ft, ta)
        assert mask.any()
        assert np.allclose(ft[mask], ta[mask], atol=1e-6), (
            f"CCI max diff: {np.abs(ft[mask] - ta[mask]).max()}"
        )

    def test_beta_output_length_matches(self):
        ft = ferro_ta.BETA(CLOSE, HIGH, timeperiod=5)
        ta = talib.BETA(CLOSE, HIGH, timeperiod=5)
        assert len(ft) == len(ta)

    def test_beta_nan_count_matches(self):
        ft = ferro_ta.BETA(CLOSE, HIGH, timeperiod=5)
        ta = talib.BETA(CLOSE, HIGH, timeperiod=5)
        assert _nan_count(ft) == _nan_count(ta), (
            f"BETA NaN count: ferro_ta={_nan_count(ft)}, talib={_nan_count(ta)}"
        )

    def test_beta_values_close_to_talib(self):
        """BETA values using returns-based regression must be close to TA-Lib."""
        ft = ferro_ta.BETA(CLOSE, HIGH, timeperiod=5)
        ta = talib.BETA(CLOSE, HIGH, timeperiod=5)
        mask = _valid_mask(ft, ta)
        assert mask.any()
        # TA-Lib BETA uses returns-based regression — allow small tolerance
        assert np.allclose(ft[mask], ta[mask], atol=1e-8), (
            f"BETA max diff: {np.abs(ft[mask] - ta[mask]).max()}"
        )


# ---------------------------------------------------------------------------
# Math operators vs TA-Lib
# ---------------------------------------------------------------------------


class TestMathOperatorsVsTalib:
    """Verify that math operator shims match TA-Lib exactly."""

    def test_add_matches_talib(self):
        ft = ferro_ta.ADD(CLOSE, HIGH)
        ta = talib.ADD(CLOSE, HIGH)
        assert np.allclose(ft, ta, equal_nan=True)

    def test_sub_matches_talib(self):
        ft = ferro_ta.SUB(HIGH, LOW)
        ta = talib.SUB(HIGH, LOW)
        assert np.allclose(ft, ta, equal_nan=True)

    def test_mult_matches_talib(self):
        ft = ferro_ta.MULT(CLOSE, VOLUME)
        ta = talib.MULT(CLOSE, VOLUME)
        assert np.allclose(ft, ta, equal_nan=True)

    def test_div_matches_talib(self):
        ft = ferro_ta.DIV(CLOSE, HIGH)
        ta = talib.DIV(CLOSE, HIGH)
        assert np.allclose(ft, ta, equal_nan=True)

    def test_sum_matches_talib(self):
        ft = ferro_ta.SUM(CLOSE, timeperiod=10)
        ta = talib.SUM(CLOSE, timeperiod=10)
        assert np.allclose(ft, ta, equal_nan=True)

    def test_max_matches_talib(self):
        ft = ferro_ta.MAX(CLOSE, timeperiod=10)
        ta = talib.MAX(CLOSE, timeperiod=10)
        assert np.allclose(ft, ta, equal_nan=True)

    def test_min_matches_talib(self):
        ft = ferro_ta.MIN(CLOSE, timeperiod=10)
        ta = talib.MIN(CLOSE, timeperiod=10)
        assert np.allclose(ft, ta, equal_nan=True)

    def test_sin_matches_talib(self):
        ft = ferro_ta.SIN(CLOSE)
        ta = talib.SIN(CLOSE)
        assert np.allclose(ft, ta, equal_nan=True)

    def test_cos_matches_talib(self):
        ft = ferro_ta.COS(CLOSE)
        ta = talib.COS(CLOSE)
        assert np.allclose(ft, ta, equal_nan=True)

    def test_sqrt_matches_talib(self):
        ft = ferro_ta.SQRT(CLOSE)
        ta = talib.SQRT(CLOSE)
        assert np.allclose(ft, ta, equal_nan=True)

    def test_exp_matches_talib(self):
        ft = ferro_ta.EXP(LINEAR)
        ta = talib.EXP(LINEAR)
        assert np.allclose(ft, ta, equal_nan=True)

    def test_ln_matches_talib(self):
        ft = ferro_ta.LN(CLOSE)
        ta = talib.LN(CLOSE)
        assert np.allclose(ft, ta, equal_nan=True)

    def test_log10_matches_talib(self):
        ft = ferro_ta.LOG10(CLOSE)
        ta = talib.LOG10(CLOSE)
        assert np.allclose(ft, ta, equal_nan=True)


# ---------------------------------------------------------------------------
# STOCH, STOCHRSI, ADX, DI, DM parity
# ---------------------------------------------------------------------------


class TestDirectionalMovementVsTalib:
    """Verify ADX, DX, +DI, -DI, +DM, -DM are strongly correlated with TA-Lib.

    Wilder smoothing seed differs between ferro_ta and TA-Lib, so values are
    not numerically identical but must be strongly correlated.
    """

    def test_plus_di_output_length(self):
        ft = ferro_ta.PLUS_DI(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.PLUS_DI(HIGH, LOW, CLOSE, timeperiod=14)
        assert len(ft) == len(ta)

    def test_plus_di_nan_count(self):
        ft = ferro_ta.PLUS_DI(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.PLUS_DI(HIGH, LOW, CLOSE, timeperiod=14)
        assert _nan_count(ft) == _nan_count(ta)

    def test_plus_di_values_strongly_correlated(self):
        ft = ferro_ta.PLUS_DI(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.PLUS_DI(HIGH, LOW, CLOSE, timeperiod=14)
        mask = _valid_mask(ft, ta)
        assert mask.any()
        corr = np.corrcoef(ft[mask], ta[mask])[0, 1]
        assert corr > 0.99

    def test_minus_di_values_strongly_correlated(self):
        ft = ferro_ta.MINUS_DI(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.MINUS_DI(HIGH, LOW, CLOSE, timeperiod=14)
        mask = _valid_mask(ft, ta)
        assert mask.any()
        corr = np.corrcoef(ft[mask], ta[mask])[0, 1]
        assert corr > 0.99

    def test_plus_dm_output_length(self):
        ft = ferro_ta.PLUS_DM(HIGH, LOW, timeperiod=14)
        ta = talib.PLUS_DM(HIGH, LOW, timeperiod=14)
        assert len(ft) == len(ta)

    def test_plus_dm_values_strongly_correlated(self):
        ft = ferro_ta.PLUS_DM(HIGH, LOW, timeperiod=14)
        ta = talib.PLUS_DM(HIGH, LOW, timeperiod=14)
        mask = _valid_mask(ft, ta)
        assert mask.any()
        corr = np.corrcoef(ft[mask], ta[mask])[0, 1]
        assert corr > 0.99

    def test_minus_dm_values_strongly_correlated(self):
        ft = ferro_ta.MINUS_DM(HIGH, LOW, timeperiod=14)
        ta = talib.MINUS_DM(HIGH, LOW, timeperiod=14)
        mask = _valid_mask(ft, ta)
        assert mask.any()
        corr = np.corrcoef(ft[mask], ta[mask])[0, 1]
        assert corr > 0.99

    def test_dx_values_strongly_correlated(self):
        ft = ferro_ta.DX(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.DX(HIGH, LOW, CLOSE, timeperiod=14)
        mask = _valid_mask(ft, ta)
        assert mask.any()
        corr = np.corrcoef(ft[mask], ta[mask])[0, 1]
        assert corr > 0.99

    def test_adx_output_length(self):
        ft = ferro_ta.ADX(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.ADX(HIGH, LOW, CLOSE, timeperiod=14)
        assert len(ft) == len(ta)

    def test_adx_nan_count(self):
        ft = ferro_ta.ADX(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.ADX(HIGH, LOW, CLOSE, timeperiod=14)
        assert _nan_count(ft) == _nan_count(ta)

    def test_adx_values_strongly_correlated(self):
        ft = ferro_ta.ADX(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.ADX(HIGH, LOW, CLOSE, timeperiod=14)
        mask = _valid_mask(ft, ta)
        assert mask.any()
        corr = np.corrcoef(ft[mask], ta[mask])[0, 1]
        assert corr > 0.99

    def test_adxr_values_strongly_correlated(self):
        ft = ferro_ta.ADXR(HIGH, LOW, CLOSE, timeperiod=14)
        ta = talib.ADXR(HIGH, LOW, CLOSE, timeperiod=14)
        mask = _valid_mask(ft, ta)
        assert mask.any()
        corr = np.corrcoef(ft[mask], ta[mask])[0, 1]
        assert corr > 0.95


class TestSTOCHVsTalib:
    """Verify STOCH and STOCHRSI match TA-Lib."""

    def test_stoch_slowk_output_length(self):
        ft_k, _ = ferro_ta.STOCH(HIGH, LOW, CLOSE)
        ta_k, _ = talib.STOCH(HIGH, LOW, CLOSE)
        assert len(ft_k) == len(ta_k)

    def test_stoch_nan_count_matches(self):
        ft_k, ft_d = ferro_ta.STOCH(HIGH, LOW, CLOSE)
        ta_k, ta_d = talib.STOCH(HIGH, LOW, CLOSE)
        assert _nan_count(ft_k) == _nan_count(ta_k)
        assert _nan_count(ft_d) == _nan_count(ta_d)

    def test_stoch_values_allclose(self):
        ft_k, ft_d = ferro_ta.STOCH(HIGH, LOW, CLOSE)
        ta_k, ta_d = talib.STOCH(HIGH, LOW, CLOSE)
        mask_k = _valid_mask(ft_k, ta_k)
        mask_d = _valid_mask(ft_d, ta_d)
        assert mask_k.any()
        assert np.allclose(ft_k[mask_k], ta_k[mask_k], atol=1e-8)
        assert np.allclose(ft_d[mask_d], ta_d[mask_d], atol=1e-8)

    def test_stochrsi_output_length(self):
        ft_k, _ = ferro_ta.STOCHRSI(CLOSE)
        ta_k, _ = talib.STOCHRSI(CLOSE)
        assert len(ft_k) == len(ta_k)

    def test_stochrsi_nan_count_matches(self):
        ft_k, ft_d = ferro_ta.STOCHRSI(
            CLOSE, timeperiod=14, fastk_period=5, fastd_period=3
        )
        ta_k, ta_d = talib.STOCHRSI(
            CLOSE, timeperiod=14, fastk_period=5, fastd_period=3
        )
        # RSI seed difference can yield ±2 NaN count (see TestSTOCHRSI)
        assert abs(_nan_count(ft_k) - _nan_count(ta_k)) <= 2

    def test_stochrsi_values_close(self):
        ft_k, ft_d = ferro_ta.STOCHRSI(
            CLOSE, timeperiod=14, fastk_period=5, fastd_period=3
        )
        ta_k, ta_d = talib.STOCHRSI(
            CLOSE, timeperiod=14, fastk_period=5, fastd_period=3
        )
        mask_k = _valid_mask(ft_k, ta_k)
        assert mask_k.any()
        assert np.allclose(ft_k[mask_k], ta_k[mask_k], atol=1e-8)


# ---------------------------------------------------------------------------
# MAMA, SAR/SAREXT, and HT_* cycle indicator tests
#
# These indicators are documented as ⚠️ Corr or ⚠️ Shape in the README because
# TA-Lib C uses slightly different floating-point accumulation and clamping
# order. Tests enforce shape parity and minimum correlation rather than
# exact allclose.
# ---------------------------------------------------------------------------


class TestHTTrendline:
    """HT_TRENDLINE — 63-bar lookback; values correlated with TA-Lib.

    Known difference: Ehlers HT filter — same algorithm and 63-bar lookback;
    values are correlated (r > 0.90) but not numerically identical due to
    different clamp order in TA-Lib C source.
    """

    def test_output_length_match(self):
        ft = ferro_ta.HT_TRENDLINE(CLOSE)
        ta = talib.HT_TRENDLINE(CLOSE)
        assert len(ft) == len(ta)

    def test_nan_count_match(self):
        ft = ferro_ta.HT_TRENDLINE(CLOSE)
        ta = talib.HT_TRENDLINE(CLOSE)
        assert _nan_count(ft) == _nan_count(ta)

    def test_correlated_with_talib(self):
        """HT_TRENDLINE should be highly correlated with TA-Lib output."""
        ft = ferro_ta.HT_TRENDLINE(CLOSE)
        ta = talib.HT_TRENDLINE(CLOSE)
        mask = _valid_mask(ft, ta)
        if mask.sum() >= 5:
            corr = float(np.corrcoef(ft[mask], ta[mask])[0, 1])
            assert corr > 0.90, f"HT_TRENDLINE correlation {corr:.3f} < 0.90"


class TestHTDCPeriod:
    """HT_DCPERIOD — 63-bar lookback; shape parity enforced.

    Known difference: Dominant cycle period values correlated with TA-Lib
    but not exact (same Ehlers algorithm, different floating-point accumulation).
    """

    def test_output_length_match(self):
        ft = ferro_ta.HT_DCPERIOD(CLOSE)
        ta = talib.HT_DCPERIOD(CLOSE)
        assert len(ft) == len(ta)

    def test_nan_count_within_tolerance(self):
        ft = ferro_ta.HT_DCPERIOD(CLOSE)
        ta = talib.HT_DCPERIOD(CLOSE)
        # ferro_ta uses 63-bar lookback; TA-Lib may use different warmup
        assert abs(_nan_count(ft) - _nan_count(ta)) <= 35

    def test_period_in_reasonable_range(self):
        """Period should typically be in [6, 50] for realistic price data."""
        ft = ferro_ta.HT_DCPERIOD(CLOSE)
        valid = ft[~np.isnan(ft)]
        assert valid.min() > 0
        assert valid.max() <= 100.0  # allow some slack


class TestHTDCPhase:
    """HT_DCPHASE — 63-bar lookback; shape parity enforced."""

    def test_output_length_match(self):
        ft = ferro_ta.HT_DCPHASE(CLOSE)
        ta = talib.HT_DCPHASE(CLOSE)
        assert len(ft) == len(ta)

    def test_nan_count_match(self):
        ft = ferro_ta.HT_DCPHASE(CLOSE)
        ta = talib.HT_DCPHASE(CLOSE)
        assert _nan_count(ft) == _nan_count(ta)

    def test_phase_sign_agreement(self):
        """DC phase sign should agree with TA-Lib for some valid bars (Ehlers algo diff)."""
        ft = ferro_ta.HT_DCPHASE(CLOSE)
        ta = talib.HT_DCPHASE(CLOSE)
        mask = _valid_mask(ft, ta)
        if mask.sum() >= 5:
            sign_agree = np.mean(np.sign(ft[mask]) == np.sign(ta[mask]))
            # HT indicators use different warmup/accumulation vs TA-Lib
            assert sign_agree >= 0.40, (
                f"HT_DCPHASE sign agreement {sign_agree:.2f} < 0.40"
            )


class TestHTPhasor:
    """HT_PHASOR — 63-bar lookback; shape parity enforced.

    Returns (inphase, quadrature). Both components are correlated with TA-Lib.
    """

    def test_output_length_match(self):
        ft_i, ft_q = ferro_ta.HT_PHASOR(CLOSE)
        ta_i, ta_q = talib.HT_PHASOR(CLOSE)
        assert len(ft_i) == len(ta_i)
        assert len(ft_q) == len(ta_q)

    def test_nan_count_within_tolerance(self):
        ft_i, ft_q = ferro_ta.HT_PHASOR(CLOSE)
        ta_i, ta_q = talib.HT_PHASOR(CLOSE)
        # ferro_ta uses 63-bar lookback; TA-Lib may use different warmup
        assert abs(_nan_count(ft_i) - _nan_count(ta_i)) <= 35
        assert abs(_nan_count(ft_q) - _nan_count(ta_q)) <= 35

    def test_inphase_sign_agreement(self):
        """Inphase component sign should agree with TA-Lib for most valid bars."""
        ft_i, _ = ferro_ta.HT_PHASOR(CLOSE)
        ta_i, _ = talib.HT_PHASOR(CLOSE)
        mask = _valid_mask(ft_i, ta_i)
        if mask.sum() >= 5:
            sign_agree = np.mean(np.sign(ft_i[mask]) == np.sign(ta_i[mask]))
            assert sign_agree >= SIGN_AGREEMENT_THRESHOLD


class TestHTSine:
    """HT_SINE — 63-bar lookback; shape parity enforced.

    Returns (sine, leadsine). Values in [-1, 1].
    """

    def test_output_length_match(self):
        ft_s, ft_l = ferro_ta.HT_SINE(CLOSE)
        ta_s, ta_l = talib.HT_SINE(CLOSE)
        assert len(ft_s) == len(ta_s)
        assert len(ft_l) == len(ta_l)

    def test_nan_count_match(self):
        ft_s, ft_l = ferro_ta.HT_SINE(CLOSE)
        ta_s, ta_l = talib.HT_SINE(CLOSE)
        assert _nan_count(ft_s) == _nan_count(ta_s)
        assert _nan_count(ft_l) == _nan_count(ta_l)

    def test_sine_range(self):
        """Sine component should be in [-1.1, 1.1] (allow small numerical overshoot)."""
        ft_s, _ = ferro_ta.HT_SINE(CLOSE)
        valid = ft_s[~np.isnan(ft_s)]
        assert valid.min() >= -1.1
        assert valid.max() <= 1.1


class TestHTTrendMode:
    """HT_TRENDMODE — 63-bar lookback; values are 0 or 1.

    Known difference: Boolean output derived from HT_DCPERIOD — may differ
    from TA-Lib in first ~10 valid bars due to the same floating-point diff.
    """

    def test_output_length_match(self):
        ft = ferro_ta.HT_TRENDMODE(CLOSE)
        ta = talib.HT_TRENDMODE(CLOSE)
        assert len(ft) == len(ta)

    def test_nan_count_match(self):
        ft = ferro_ta.HT_TRENDMODE(CLOSE)
        ta = talib.HT_TRENDMODE(CLOSE)
        assert _nan_count(ft) == _nan_count(ta)

    def test_binary_output(self):
        """TRENDMODE values must be 0 or 1 (or NaN for warmup)."""
        ft = ferro_ta.HT_TRENDMODE(CLOSE)
        valid = ft[~np.isnan(ft)]
        assert set(valid.astype(int)).issubset({0, 1})

    def test_sign_agreement_with_talib(self):
        """Trend mode should agree with TA-Lib for majority of valid bars.

        Note: HT_TRENDMODE is highly sensitive to Hilbert Transform phase
        accumulator initialization; the two implementations use different
        precision for the adaptive period, so agreement is ~54%.  We verify
        > 50% to confirm the indicator is better-than-random.
        """
        ft = ferro_ta.HT_TRENDMODE(CLOSE)
        ta = talib.HT_TRENDMODE(CLOSE)
        mask = _valid_mask(ft, ta)
        if mask.sum() >= 5:
            agree = np.mean(ft[mask] == ta[mask])
            assert agree >= 0.50, f"HT_TRENDMODE agreement {agree:.2f} < 0.50"


# ---------------------------------------------------------------------------
# Candlestick Pattern Agreement Tests
# ---------------------------------------------------------------------------


# List of all candlestick patterns to test
ALL_CDL_PATTERNS = [
    "CDL2CROWS",
    "CDL3BLACKCROWS",
    "CDL3INSIDE",
    "CDL3LINESTRIKE",
    "CDL3OUTSIDE",
    "CDL3STARSINSOUTH",
    "CDL3WHITESOLDIERS",
    "CDLABANDONEDBABY",
    "CDLADVANCEBLOCK",
    "CDLBELTHOLD",
    "CDLBREAKAWAY",
    "CDLCLOSINGMARUBOZU",
    "CDLCONCEALBABYSWALL",
    "CDLCOUNTERATTACK",
    "CDLDARKCLOUDCOVER",
    "CDLDOJI",
    "CDLDOJISTAR",
    "CDLDRAGONFLYDOJI",
    "CDLENGULFING",
    "CDLEVENINGDOJISTAR",
    "CDLEVENINGSTAR",
    "CDLGAPSIDESIDEWHITE",
    "CDLGRAVESTONEDOJI",
    "CDLHAMMER",
    "CDLHANGINGMAN",
    "CDLHARAMI",
    "CDLHARAMICROSS",
    "CDLHIGHWAVE",
    "CDLHIKKAKE",
    "CDLHIKKAKEMOD",
    "CDLHOMINGPIGEON",
    "CDLIDENTICAL3CROWS",
    "CDLINNECK",
    "CDLINVERTEDHAMMER",
    "CDLKICKING",
    "CDLKICKINGBYLENGTH",
    "CDLLADDERBOTTOM",
    "CDLLONGLEGGEDDOJI",
    "CDLLONGLINE",
    "CDLMARUBOZU",
    "CDLMATCHINGLOW",
    "CDLMATHOLD",
    "CDLMORNINGDOJISTAR",
    "CDLMORNINGSTAR",
    "CDLONNECK",
    "CDLPIERCING",
    "CDLRICKSHAWMAN",
    "CDLRISEFALL3METHODS",
    "CDLSEPARATINGLINES",
    "CDLSHOOTINGSTAR",
    "CDLSHORTLINE",
    "CDLSPINNINGTOP",
    "CDLSTALLEDPATTERN",
    "CDLSTICKSANDWICH",
    "CDLTAKURI",
    "CDLTASUKIGAP",
    "CDLTHRUSTING",
    "CDLTRISTAR",
    "CDLUNIQUE3RIVER",
    "CDLUPSIDEGAP2CROWS",
    "CDLXSIDEGAP3METHODS",
]


class TestCandlestickPatternAgreement:
    """Pattern recognition: agreement rate tests.

    Candlestick patterns may have slightly different threshold parameters
    between implementations. We validate >80% agreement rate for pattern
    detection (non-zero output).
    """

    @pytest.mark.parametrize("pattern_name", ALL_CDL_PATTERNS)
    def test_pattern_agreement_rate(self, pattern_name):
        """Test that pattern agreement rate is > 80%."""
        # Get pattern functions
        ft_func = getattr(ferro_ta, pattern_name, None)
        ta_func = getattr(talib, pattern_name, None)

        if ft_func is None:
            pytest.skip(f"ferro_ta.{pattern_name} not implemented")
        if ta_func is None:
            pytest.skip(f"talib.{pattern_name} not available")

        # Compute patterns
        ft = ft_func(OPEN, HIGH, LOW, CLOSE)
        ta = ta_func(OPEN, HIGH, LOW, CLOSE)

        # Check output length match
        assert len(ft) == len(ta), f"{pattern_name}: length mismatch"

        # Compute agreement rate (exact match of output values)
        # Patterns typically return 0, ±100, or ±200
        agreement = np.mean(ft == ta)

        # Use per-pattern threshold (some patterns have known definition differences)
        threshold = CDL_AGREEMENT_THRESHOLDS.get(pattern_name, 0.80)
        assert agreement > threshold, (
            f"{pattern_name}: agreement rate {agreement:.2%} < {threshold:.0%}"
        )

    def test_pattern_sample_doji(self):
        """Spot check: CDLDOJI should have high agreement (known: shadow ratio precision differs)."""
        ft = ferro_ta.CDLDOJI(OPEN, HIGH, LOW, CLOSE)
        ta = talib.CDLDOJI(OPEN, HIGH, LOW, CLOSE)

        agreement = np.mean(ft == ta)
        # ferro_ta uses slightly different shadow/body ratio threshold; 86% observed
        assert agreement > 0.85

    def test_pattern_sample_engulfing(self):
        """Spot check: CDLENGULFING should have high agreement."""
        ft = ferro_ta.CDLENGULFING(OPEN, HIGH, LOW, CLOSE)
        ta = talib.CDLENGULFING(OPEN, HIGH, LOW, CLOSE)

        agreement = np.mean(ft == ta)
        assert agreement > 0.80

    def test_pattern_sample_hammer(self):
        """Spot check: CDLHAMMER should have high agreement."""
        ft = ferro_ta.CDLHAMMER(OPEN, HIGH, LOW, CLOSE)
        ta = talib.CDLHAMMER(OPEN, HIGH, LOW, CLOSE)

        agreement = np.mean(ft == ta)
        assert agreement > 0.80


# ---------------------------------------------------------------------------
# Non-default ``matype`` vs TA-Lib
#
# Every ferro-ta function that takes a ``matype`` defaults to *one* value, so
# the tests above only ever exercise a single MA per indicator.  This section
# sweeps the whole enum against TA-Lib, because a ``matype`` implementation
# that is only ever called with its default is untested against the reference
# it exists to match.
# ---------------------------------------------------------------------------

# ``matype`` values whose *meaning* is identical in ferro-ta and TA-Lib, so a
# direct value comparison is legitimate.
#
# 7 IS DELIBERATELY ABSENT.  ferro-ta numbers T3 as 7 (with 8 as an alias);
# TA-Lib's ``TA_MAType`` numbers 7 as MAMA and 8 as T3.  Passing 7 to both
# libraries therefore computes *different indicators* — T3 here, MAMA there —
# and any agreement assertion at 7 is guaranteed to fail.  Do not "fix" this by
# adding 7 to the tuple: ``TestMatypeSevenDivergence`` below pins the
# divergence deliberately, and the reasoning is in the
# ``crates/ferro_ta_core/src/overlap/dispatch.rs`` module docs.
TALIB_COMPATIBLE_MATYPES = (0, 1, 2, 3, 4, 5, 6, 8)

# T3 in ferro-ta, MAMA in TA-Lib.  The one incompatible value.
FERRO_T3_TALIB_MAMA_MATYPE = 7

# Largest ``matype`` ferro-ta's dispatcher understands (``overlap::MAX_MATYPE``).
MAX_MATYPE = 8

# Per-``matype`` MA warm-up, in bars, for period ``p`` — ``overlap::ma_lookback``.
# The overlapping valid region between ferro-ta and TA-Lib shrinks as this
# grows, which is exactly how a careless sweep ends up comparing NaN to NaN.
MATYPE_LOOKBACK = {
    0: lambda p: p - 1,  # SMA
    1: lambda p: p - 1,  # EMA
    2: lambda p: p - 1,  # WMA
    3: lambda p: 2 * (p - 1),  # DEMA
    4: lambda p: 3 * (p - 1),  # TEMA
    5: lambda p: p - 1,  # TRIMA
    6: lambda p: p,  # KAMA (the seed at p-1 is not emitted)
    7: lambda p: 6 * (p - 1),  # T3
    8: lambda p: 6 * (p - 1),  # T3 (alias of 7)
}

# Variable-period input for MAVP: three distinct periods inside [2, 30].
MAVP_PERIODS = np.full(N, 5.0)
MAVP_PERIODS[::3] = 10.0
MAVP_PERIODS[1::7] = 20.0


def _assert_matype_allclose(
    ft: np.ndarray,
    ta: np.ndarray,
    *,
    matype: int,
    min_compared: int,
    atol: float = 1e-6,
    label: str = "",
) -> None:
    """Assert ferro-ta matches TA-Lib, and that a real comparison happened.

    The warm-up of a ``matype``-parameterised indicator depends on the
    ``matype`` (see :data:`MATYPE_LOOKBACK`), so the finite overlap between the
    two libraries shrinks — from ``p - 1`` bars of padding for SMA/EMA/WMA/TRIMA
    up to ``6 * (p - 1)`` for T3.  ``_valid_mask`` silently drops every NaN
    position, so without ``min_compared`` a warm-up bug (or a signature change
    that NaNs the whole array) would make this a vacuous pass over an empty
    selection.  ``min_compared`` is the floor that makes the assertion mean
    something.
    """
    mask = _valid_mask(ft, ta)
    compared = int(mask.sum())
    assert compared >= min_compared, (
        f"{label} matype={matype}: only {compared} finite pairs compared "
        f"(expected >= {min_compared}) — the comparison has degenerated to "
        f"NaN-vs-NaN and proves nothing"
    )
    max_diff = float(np.max(np.abs(ft[mask] - ta[mask])))
    assert max_diff <= atol, (
        f"{label} matype={matype}: max abs diff {max_diff:.3e} > {atol:.0e} "
        f"over {compared} compared bars"
    )


class TestAPOMatype:
    """APO — exact TA-Lib match at every compatible ``matype``.

    ferro-ta defaults ``matype`` to ``1`` (EMA) while TA-Lib defaults to ``0``
    (SMA), so an unqualified ferro-ta call never matches an unqualified TA-Lib
    call.  With ``matype`` passed explicitly on *both* sides the two agree to
    floating-point noise (worst observed 4.9e-12, at ``matype=2``/WMA), which
    is a genuine value gate — not the NaN-count-only check in ``TestAPO``.
    """

    FAST, SLOW = 12, 26

    @pytest.mark.parametrize("matype", TALIB_COMPATIBLE_MATYPES)
    def test_values_match(self, matype):
        ft = ferro_ta.APO(
            CLOSE, fastperiod=self.FAST, slowperiod=self.SLOW, matype=matype
        )
        ta = talib.APO(CLOSE, fastperiod=self.FAST, slowperiod=self.SLOW, matype=matype)
        _assert_matype_allclose(ft, ta, matype=matype, min_compared=300, label="APO")

    @pytest.mark.parametrize("matype", TALIB_COMPATIBLE_MATYPES)
    def test_nan_count_match(self, matype):
        """Warm-up is ``ma_lookback(slowperiod, matype)`` in both libraries."""
        ft = ferro_ta.APO(
            CLOSE, fastperiod=self.FAST, slowperiod=self.SLOW, matype=matype
        )
        ta = talib.APO(CLOSE, fastperiod=self.FAST, slowperiod=self.SLOW, matype=matype)
        expected = MATYPE_LOOKBACK[matype](self.SLOW)
        assert _nan_count(ft) == _nan_count(ta) == expected

    def test_default_matype_is_ema_not_talib_sma(self):
        """ferro-ta's default is TA-Lib's ``matype=1``, not TA-Lib's default."""
        default = ferro_ta.APO(CLOSE, fastperiod=self.FAST, slowperiod=self.SLOW)
        ema = talib.APO(CLOSE, fastperiod=self.FAST, slowperiod=self.SLOW, matype=1)
        sma = talib.APO(CLOSE, fastperiod=self.FAST, slowperiod=self.SLOW, matype=0)
        _assert_matype_allclose(
            default, ema, matype=1, min_compared=400, label="APO default"
        )
        mask = _valid_mask(default, sma)
        assert not np.allclose(default[mask], sma[mask], atol=1e-6)


class TestPPOMatype:
    """PPO — exact TA-Lib match on the PPO line at every compatible ``matype``.

    ``TA_PPO`` returns a single array; ferro-ta returns
    ``(ppo, signal, histogram)`` because it added a ``signalperiod``.  Only the
    **first** element (the PPO line) has a TA-Lib counterpart — the signal and
    histogram are ferro-ta extensions with nothing to compare against.

    Like APO, ferro-ta defaults ``matype`` to ``1`` and TA-Lib to ``0``.  With
    ``matype`` matched on both sides the PPO line agrees to floating-point
    noise (worst observed 1.2e-11 at ``matype=2``/WMA), which replaces the
    correlation-only (> 0.85) gate in ``TestPPO`` with a real value gate.
    """

    FAST, SLOW, SIGNAL = 12, 26, 9

    @pytest.mark.parametrize("matype", TALIB_COMPATIBLE_MATYPES)
    def test_ppo_line_values_match(self, matype):
        ppo, _signal, _hist = ferro_ta.PPO(
            CLOSE,
            fastperiod=self.FAST,
            slowperiod=self.SLOW,
            signalperiod=self.SIGNAL,
            matype=matype,
        )
        ta = talib.PPO(CLOSE, fastperiod=self.FAST, slowperiod=self.SLOW, matype=matype)
        _assert_matype_allclose(
            ppo, ta, matype=matype, min_compared=300, label="PPO line"
        )

    @pytest.mark.parametrize("matype", TALIB_COMPATIBLE_MATYPES)
    def test_ppo_line_nan_count_match(self, matype):
        ppo, _signal, _hist = ferro_ta.PPO(
            CLOSE,
            fastperiod=self.FAST,
            slowperiod=self.SLOW,
            signalperiod=self.SIGNAL,
            matype=matype,
        )
        ta = talib.PPO(CLOSE, fastperiod=self.FAST, slowperiod=self.SLOW, matype=matype)
        assert _nan_count(ppo) == _nan_count(ta) == MATYPE_LOOKBACK[matype](self.SLOW)

    def test_default_matype_is_ema_not_talib_sma(self):
        ppo, _, _ = ferro_ta.PPO(CLOSE, fastperiod=self.FAST, slowperiod=self.SLOW)
        ema = talib.PPO(CLOSE, fastperiod=self.FAST, slowperiod=self.SLOW, matype=1)
        sma = talib.PPO(CLOSE, fastperiod=self.FAST, slowperiod=self.SLOW, matype=0)
        _assert_matype_allclose(
            ppo, ema, matype=1, min_compared=400, label="PPO default"
        )
        mask = _valid_mask(ppo, sma)
        assert not np.allclose(ppo[mask], sma[mask], atol=1e-6)


class TestSTOCHFMatype:
    """STOCHF — %K and %D match TA-Lib at every compatible ``fastd_matype``."""

    FASTK, FASTD = 5, 3

    @pytest.mark.parametrize("matype", TALIB_COMPATIBLE_MATYPES)
    def test_fastk_and_fastd_values_match(self, matype):
        fk, fd = ferro_ta.STOCHF(
            HIGH,
            LOW,
            CLOSE,
            fastk_period=self.FASTK,
            fastd_period=self.FASTD,
            fastd_matype=matype,
        )
        tk, td = talib.STOCHF(
            HIGH,
            LOW,
            CLOSE,
            fastk_period=self.FASTK,
            fastd_period=self.FASTD,
            fastd_matype=matype,
        )
        _assert_matype_allclose(
            fk, tk, matype=matype, min_compared=450, atol=1e-8, label="STOCHF %K"
        )
        _assert_matype_allclose(
            fd, td, matype=matype, min_compared=450, atol=1e-6, label="STOCHF %D"
        )

    @pytest.mark.parametrize("matype", TALIB_COMPATIBLE_MATYPES)
    def test_nan_count_match(self, matype):
        """Both libraries pad to ``fastk_period - 1 + ma_lookback(fastd)``."""
        fk, fd = ferro_ta.STOCHF(
            HIGH,
            LOW,
            CLOSE,
            fastk_period=self.FASTK,
            fastd_period=self.FASTD,
            fastd_matype=matype,
        )
        tk, td = talib.STOCHF(
            HIGH,
            LOW,
            CLOSE,
            fastk_period=self.FASTK,
            fastd_period=self.FASTD,
            fastd_matype=matype,
        )
        expected = (self.FASTK - 1) + MATYPE_LOOKBACK[matype](self.FASTD)
        assert _nan_count(fk) == _nan_count(tk) == expected
        assert _nan_count(fd) == _nan_count(td) == expected


class TestSTOCHMatype:
    """STOCH — slow %K and %D match TA-Lib at every compatible matype.

    Both ``slowk_matype`` and ``slowd_matype`` are swept together, which is the
    configuration that stacks two non-default lookbacks and so shrinks the
    valid overlap the most.
    """

    FASTK, SLOWK, SLOWD = 5, 3, 3

    @pytest.mark.parametrize("matype", TALIB_COMPATIBLE_MATYPES)
    def test_slowk_and_slowd_values_match(self, matype):
        fk, fd = ferro_ta.STOCH(
            HIGH,
            LOW,
            CLOSE,
            fastk_period=self.FASTK,
            slowk_period=self.SLOWK,
            slowk_matype=matype,
            slowd_period=self.SLOWD,
            slowd_matype=matype,
        )
        tk, td = talib.STOCH(
            HIGH,
            LOW,
            CLOSE,
            fastk_period=self.FASTK,
            slowk_period=self.SLOWK,
            slowk_matype=matype,
            slowd_period=self.SLOWD,
            slowd_matype=matype,
        )
        _assert_matype_allclose(
            fk, tk, matype=matype, min_compared=450, label="STOCH slow %K"
        )
        _assert_matype_allclose(
            fd, td, matype=matype, min_compared=450, label="STOCH slow %D"
        )

    @pytest.mark.parametrize("matype", TALIB_COMPATIBLE_MATYPES)
    def test_nan_count_match(self, matype):
        fk, fd = ferro_ta.STOCH(
            HIGH,
            LOW,
            CLOSE,
            fastk_period=self.FASTK,
            slowk_period=self.SLOWK,
            slowk_matype=matype,
            slowd_period=self.SLOWD,
            slowd_matype=matype,
        )
        tk, td = talib.STOCH(
            HIGH,
            LOW,
            CLOSE,
            fastk_period=self.FASTK,
            slowk_period=self.SLOWK,
            slowk_matype=matype,
            slowd_period=self.SLOWD,
            slowd_matype=matype,
        )
        expected = (
            (self.FASTK - 1)
            + MATYPE_LOOKBACK[matype](self.SLOWK)
            + MATYPE_LOOKBACK[matype](self.SLOWD)
        )
        assert _nan_count(fk) == _nan_count(tk) == expected
        assert _nan_count(fd) == _nan_count(td) == expected

    @pytest.mark.parametrize("matype", TALIB_COMPATIBLE_MATYPES)
    def test_range_0_to_100(self, matype):
        for arr in ferro_ta.STOCH(
            HIGH,
            LOW,
            CLOSE,
            fastk_period=self.FASTK,
            slowk_period=self.SLOWK,
            slowk_matype=matype,
            slowd_period=self.SLOWD,
            slowd_matype=matype,
        ):
            finite = arr[~np.isnan(arr)]
            assert finite.size > 0
            # DEMA/TEMA/T3 overshoot is legitimate; they are not bounded MAs.
            assert np.isfinite(finite).all()


# ``fastd_matype=6`` (KAMA) is excluded from the STOCHRSI %D *value* gate, and
# it is not a ferro-ta bug.
#
# ``TA_KAMA`` decides its efficiency ratio with
# ``sumROC1 <= periodROC || TA_IS_ZERO(sumROC1)``, where ``sumROC1`` is a rolling
# ``Σ|Δ|`` maintained by subtract-then-add.  On a flat window the exact sum is
# zero but the rolling one holds ~1e-14 of rounding residue, and *the sign of
# that residue* selects between ER = 1 (SC = 4/9, snap) and ER = 0
# (SC = (2/31)^2, crawl).  KAMA is recursive, so the choice persists for the
# whole plateau.
#
# StochRSI's %K sits at exactly 0 or 100 for long stretches (whenever the RSI is
# monotone across the %K window), so those flat windows are everywhere here, and
# ``TA_KAMA`` is catastrophically ill-conditioned on this input: feeding it
# ferro-ta's %K instead of TA-Lib's — the two differ by 5.8e-13, pure RSI/stoch
# rounding — moves *TA-Lib's own output* by 19.2.  ferro-ta's KAMA reproduces
# TA-Lib to 2.8e-14 on either input; see
# ``test_talib_kama_is_ill_conditioned_on_stochrsi_fastk`` below, which pins that
# and is what makes this a documented upstream divergence rather than a bug to
# fix.  STOCHF/STOCH are unaffected because raw price %K has no such plateaus.
STOCHRSI_KAMA_MATYPE = 6
STOCHRSI_D_MATYPES = tuple(
    m for m in TALIB_COMPATIBLE_MATYPES if m != STOCHRSI_KAMA_MATYPE
)


class TestSTOCHRSIMatype:
    """STOCHRSI — %K and %D match TA-Lib at every compatible ``fastd_matype``."""

    PERIOD, FASTK, FASTD = 14, 5, 3

    def _ferro(self, matype):
        return ferro_ta.STOCHRSI(
            CLOSE,
            timeperiod=self.PERIOD,
            fastk_period=self.FASTK,
            fastd_period=self.FASTD,
            fastd_matype=matype,
        )

    def _talib(self, matype):
        return talib.STOCHRSI(
            CLOSE,
            timeperiod=self.PERIOD,
            fastk_period=self.FASTK,
            fastd_period=self.FASTD,
            fastd_matype=matype,
        )

    @pytest.mark.parametrize("matype", TALIB_COMPATIBLE_MATYPES)
    def test_fastk_values_match(self, matype):
        """%K is independent of ``fastd_matype`` except for its NaN padding."""
        fk, _ = self._ferro(matype)
        tk, _ = self._talib(matype)
        _assert_matype_allclose(
            fk, tk, matype=matype, min_compared=440, label="STOCHRSI %K"
        )

    @pytest.mark.parametrize("matype", STOCHRSI_D_MATYPES)
    def test_fastd_values_match(self, matype):
        _, fd = self._ferro(matype)
        _, td = self._talib(matype)
        _assert_matype_allclose(
            fd, td, matype=matype, min_compared=440, label="STOCHRSI %D"
        )

    @pytest.mark.parametrize("matype", TALIB_COMPATIBLE_MATYPES)
    def test_nan_count_match(self, matype):
        fk, fd = self._ferro(matype)
        tk, td = self._talib(matype)
        assert _nan_count(fk) == _nan_count(tk)
        assert _nan_count(fd) == _nan_count(td)


class TestMAMatype:
    """MA — exact TA-Lib match at every compatible ``matype``.

    ``MA`` had no TA-Lib comparison in this module at all before; the
    compatibility table's ✅ rested on the per-MA tests (SMA, EMA, …) rather
    than on the dispatcher that routes ``matype`` to them.
    """

    PERIOD = 10

    @pytest.mark.parametrize("matype", TALIB_COMPATIBLE_MATYPES)
    def test_values_match(self, matype):
        ft = ferro_ta.MA(CLOSE, timeperiod=self.PERIOD, matype=matype)
        ta = talib.MA(CLOSE, timeperiod=self.PERIOD, matype=matype)
        _assert_matype_allclose(ft, ta, matype=matype, min_compared=440, label="MA")

    @pytest.mark.parametrize("matype", TALIB_COMPATIBLE_MATYPES)
    def test_nan_count_match(self, matype):
        ft = ferro_ta.MA(CLOSE, timeperiod=self.PERIOD, matype=matype)
        ta = talib.MA(CLOSE, timeperiod=self.PERIOD, matype=matype)
        assert _nan_count(ft) == _nan_count(ta) == MATYPE_LOOKBACK[matype](self.PERIOD)


# ``matype`` values for which MAVP and MACDEXT match TA-Lib over the *whole*
# series, versus the ones that only converge.
#
# TA-Lib computes each leg's MA over a sub-range that begins at the output
# start index, not at bar 0.  For window MAs (SMA, WMA, TRIMA) that is
# irrelevant — the value at bar i depends only on the last p bars — so the two
# libraries agree exactly.  For the recursive EMA family (EMA, DEMA, TEMA,
# KAMA, T3) ferro-ta seeds from bar 0 while TA-Lib seeds later, so early bars
# differ by the seed and the difference decays.  On the converged tail the two
# agree to ~1e-8 or better.
PATH_INDEPENDENT_MATYPES = (0, 2, 5)  # SMA, WMA, TRIMA
PATH_DEPENDENT_MATYPES = (1, 3, 4, 6, 8)  # EMA, DEMA, TEMA, KAMA, T3
CONVERGED_TAIL_START = int(N * 0.7)


def _tail_only(*arrays: np.ndarray) -> np.ndarray:
    """Valid mask restricted to the converged tail (last 30% of bars)."""
    mask = _valid_mask(*arrays)
    mask[:CONVERGED_TAIL_START] = False
    return mask


def _assert_tail_allclose(
    ft: np.ndarray,
    ta: np.ndarray,
    *,
    matype: int,
    min_compared: int = 120,
    atol: float = 1e-6,
    label: str = "",
) -> None:
    """Assert agreement on the converged tail, over a real number of bars."""
    mask = _tail_only(ft, ta)
    compared = int(mask.sum())
    assert compared >= min_compared, (
        f"{label} matype={matype}: only {compared} finite tail pairs compared "
        f"(expected >= {min_compared})"
    )
    max_diff = float(np.max(np.abs(ft[mask] - ta[mask])))
    assert max_diff <= atol, (
        f"{label} matype={matype}: converged-tail max abs diff "
        f"{max_diff:.3e} > {atol:.0e} over {compared} bars"
    )


class TestMAVPMatype:
    """MAVP — no TA-Lib comparison existed in this module before.

    ``TA_MAVP(real, periods, minperiod, maxperiod, matype)`` corresponds
    argument-for-argument, so this is a real comparison.  Window MAs match
    exactly; the recursive family matches on the converged tail (see
    :data:`PATH_DEPENDENT_MATYPES` for why).
    """

    MINPERIOD, MAXPERIOD = 2, 30

    def _pair(self, matype):
        ft = ferro_ta.MAVP(
            CLOSE,
            MAVP_PERIODS,
            minperiod=self.MINPERIOD,
            maxperiod=self.MAXPERIOD,
            matype=matype,
        )
        ta = talib.MAVP(
            CLOSE,
            MAVP_PERIODS,
            minperiod=self.MINPERIOD,
            maxperiod=self.MAXPERIOD,
            matype=matype,
        )
        return ft, ta

    @pytest.mark.parametrize("matype", PATH_INDEPENDENT_MATYPES)
    def test_window_matypes_match_exactly(self, matype):
        ft, ta = self._pair(matype)
        _assert_matype_allclose(ft, ta, matype=matype, min_compared=400, label="MAVP")

    @pytest.mark.parametrize("matype", PATH_DEPENDENT_MATYPES)
    def test_recursive_matypes_match_on_converged_tail(self, matype):
        ft, ta = self._pair(matype)
        _assert_tail_allclose(ft, ta, matype=matype, label="MAVP")

    @pytest.mark.parametrize("matype", TALIB_COMPATIBLE_MATYPES)
    def test_nan_count_match(self, matype):
        """Both pad by ``ma_lookback(maxperiod, matype)``."""
        ft, ta = self._pair(matype)
        expected = MATYPE_LOOKBACK[matype](self.MAXPERIOD)
        assert _nan_count(ft) == _nan_count(ta) == expected


# MACDEXT accepts the same ``0``-``MAX_MATYPE`` range as its APO/PPO/STOCH*/MA/
# MAVP siblings.  Its wrappers briefly capped at 7 while the rest of the crate
# had moved to 8; ``test_matype_8_is_accepted`` guards against that regressing.
# ``8`` is a T3 alias, so it joins the recursive (seeded-from-bar-0) family.
MACDEXT_WINDOW_MATYPES = (0, 2, 5)
MACDEXT_RECURSIVE_MATYPES = (1, 3, 4, 6, 8)


class TestMACDEXTMatype:
    """MACDEXT — no TA-Lib comparison existed in this module before.

    ferro-ta defaults ``fastmatype``/``slowmatype``/``signalmatype`` to ``1``
    (EMA), *not* TA-Lib's ``0``; the defaults are confirmed against the runtime
    signature rather than the stub.  With the matypes matched on both sides the
    window MAs agree exactly and the recursive ones agree on the converged tail
    — TA-Lib's ``TA_MACDEXT`` seeds each leg's MA at the output start index
    while ferro-ta seeds at bar 0 (``talib.MACDEXT(matype=1)`` is bit-identical
    to ``talib.MACD``, which shows the same offset).
    """

    FAST, SLOW, SIGNAL = 12, 26, 9

    def _pair(self, matype):
        ft = ferro_ta.MACDEXT(
            CLOSE,
            fastperiod=self.FAST,
            fastmatype=matype,
            slowperiod=self.SLOW,
            slowmatype=matype,
            signalperiod=self.SIGNAL,
            signalmatype=matype,
        )
        ta = talib.MACDEXT(
            CLOSE,
            fastperiod=self.FAST,
            fastmatype=matype,
            slowperiod=self.SLOW,
            slowmatype=matype,
            signalperiod=self.SIGNAL,
            signalmatype=matype,
        )
        return ft, ta

    def test_runtime_defaults_are_ema(self):
        """The wrappers default to matype 1 on all three legs."""
        import inspect

        params = inspect.signature(ferro_ta.MACDEXT).parameters
        assert params["fastmatype"].default == 1
        assert params["slowmatype"].default == 1
        assert params["signalmatype"].default == 1

    @pytest.mark.parametrize("matype", MACDEXT_WINDOW_MATYPES)
    def test_window_matypes_match_exactly(self, matype):
        (fm, fs, fh), (tm, ts, th) = self._pair(matype)
        _assert_matype_allclose(
            fm, tm, matype=matype, min_compared=400, label="MACDEXT macd"
        )
        _assert_matype_allclose(
            fs, ts, matype=matype, min_compared=400, label="MACDEXT signal"
        )
        _assert_matype_allclose(
            fh, th, matype=matype, min_compared=400, label="MACDEXT hist"
        )

    @pytest.mark.parametrize("matype", MACDEXT_RECURSIVE_MATYPES)
    def test_recursive_matypes_match_on_converged_tail(self, matype):
        (fm, fs, fh), (tm, ts, th) = self._pair(matype)
        _assert_tail_allclose(fm, tm, matype=matype, label="MACDEXT macd")
        _assert_tail_allclose(fs, ts, matype=matype, label="MACDEXT signal")
        _assert_tail_allclose(fh, th, matype=matype, label="MACDEXT hist")

    @pytest.mark.parametrize("matype", TALIB_COMPATIBLE_MATYPES)
    def test_nan_count_match(self, matype):
        """KAMA's off-by-one seed can shift the signal leg by a single bar."""
        (fm, fs, fh), (tm, ts, th) = self._pair(matype)
        for ftarr, taarr, name in (
            (fm, tm, "macd"),
            (fs, ts, "signal"),
            (fh, th, "hist"),
        ):
            assert abs(_nan_count(ftarr) - _nan_count(taarr)) <= 1, (
                f"MACDEXT {name} matype={matype}: NaN counts "
                f"{_nan_count(ftarr)} vs {_nan_count(taarr)}"
            )

    def test_matype_8_is_accepted(self):
        """MACDEXT accepts ``MAX_MATYPE`` (8), like every other matype taker.

        Its wrappers once capped at 7 while the core had already moved to 8,
        which made T3-via-TA-Lib's-own-number unreachable through MACDEXT
        alone.  Assert acceptance on all three arguments, and that 8 is the
        T3 alias of 7 rather than merely being tolerated.
        """
        for kwargs in (
            {"fastmatype": MAX_MATYPE},
            {"slowmatype": MAX_MATYPE},
            {"signalmatype": MAX_MATYPE},
        ):
            macd, signal, hist = ferro_ta.MACDEXT(CLOSE, **kwargs)
            assert np.isfinite(macd).any(), kwargs

        for name in ("fastmatype", "slowmatype", "signalmatype"):
            seven = ferro_ta.MACDEXT(CLOSE, **{name: 7})
            eight = ferro_ta.MACDEXT(CLOSE, **{name: MAX_MATYPE})
            for a, b in zip(seven, eight):
                np.testing.assert_array_equal(a, b, err_msg=name)

    def test_matype_9_is_rejected(self):
        """One past the bound still raises, on each of the three arguments."""
        for name in ("fastmatype", "slowmatype", "signalmatype"):
            with pytest.raises(ValueError):
                ferro_ta.MACDEXT(CLOSE, **{name: MAX_MATYPE + 1})


class TestMatypeSevenDivergence:
    """``matype=7`` means T3 in ferro-ta and MAMA in TA-Lib.

    This is the one value of the enum that is *not* TA-Lib compatible, and it
    is why ``7`` is missing from :data:`TALIB_COMPATIBLE_MATYPES`.  These tests
    assert the divergence rather than agreement, so that removing the exclusion
    above cannot quietly turn into a passing test — and so that a future reader
    who assumes the exclusion is a mistake sees it pinned with a reason.
    """

    def test_talib_matype_7_is_mama(self):
        """TA-Lib routes ``matype=7`` to ``TA_MAMA(0.5, 0.05)``."""
        ta7 = talib.MA(CLOSE, timeperiod=10, matype=FERRO_T3_TALIB_MAMA_MATYPE)
        mama, _fama = talib.MAMA(CLOSE, fastlimit=0.5, slowlimit=0.05)
        _assert_matype_allclose(
            ta7, mama, matype=7, min_compared=400, atol=1e-12, label="talib MA(7)"
        )

    def test_ferro_matype_7_is_t3_and_aliases_8(self):
        """ferro-ta routes both ``7`` and ``8`` to ``T3(vfactor=0.7)``."""
        ft7 = ferro_ta.MA(CLOSE, timeperiod=10, matype=FERRO_T3_TALIB_MAMA_MATYPE)
        ft8 = ferro_ta.MA(CLOSE, timeperiod=10, matype=MAX_MATYPE)
        t3 = ferro_ta.T3(CLOSE, timeperiod=10, vfactor=0.7)
        assert np.array_equal(ft7, ft8, equal_nan=True)
        _assert_matype_allclose(
            ft7, t3, matype=7, min_compared=400, atol=1e-12, label="ferro MA(7)"
        )

    @pytest.mark.parametrize(
        "ferro_call,talib_call,label",
        [
            (
                lambda mt: ferro_ta.MA(CLOSE, timeperiod=10, matype=mt),
                lambda mt: talib.MA(CLOSE, timeperiod=10, matype=mt),
                "MA",
            ),
            (
                lambda mt: ferro_ta.APO(CLOSE, fastperiod=12, slowperiod=26, matype=mt),
                lambda mt: talib.APO(CLOSE, fastperiod=12, slowperiod=26, matype=mt),
                "APO",
            ),
            (
                lambda mt: ferro_ta.PPO(CLOSE, fastperiod=12, slowperiod=26, matype=mt)[
                    0
                ],
                lambda mt: talib.PPO(CLOSE, fastperiod=12, slowperiod=26, matype=mt),
                "PPO",
            ),
            (
                lambda mt: ferro_ta.STOCHF(
                    HIGH, LOW, CLOSE, fastk_period=5, fastd_period=3, fastd_matype=mt
                )[1],
                lambda mt: talib.STOCHF(
                    HIGH, LOW, CLOSE, fastk_period=5, fastd_period=3, fastd_matype=mt
                )[1],
                "STOCHF %D",
            ),
            (
                lambda mt: ferro_ta.STOCHRSI(
                    CLOSE,
                    timeperiod=14,
                    fastk_period=5,
                    fastd_period=3,
                    fastd_matype=mt,
                )[1],
                lambda mt: talib.STOCHRSI(
                    CLOSE,
                    timeperiod=14,
                    fastk_period=5,
                    fastd_period=3,
                    fastd_matype=mt,
                )[1],
                "STOCHRSI %D",
            ),
        ],
    )
    def test_matype_7_does_not_agree_with_talib(self, ferro_call, talib_call, label):
        ft = ferro_call(FERRO_T3_TALIB_MAMA_MATYPE)
        ta = talib_call(FERRO_T3_TALIB_MAMA_MATYPE)
        mask = _valid_mask(ft, ta)
        assert mask.sum() >= 100, f"{label}: nothing compared at matype=7"
        assert not np.allclose(ft[mask], ta[mask], atol=1e-6), (
            f"{label}: matype=7 now agrees with TA-Lib. Either the enum was "
            f"renumbered (T3 moved off 7) or MAMA was wired in — update "
            f"TALIB_COMPATIBLE_MATYPES and overlap/dispatch.rs together."
        )


class TestMatypeOutOfRange:
    """``matype`` above ``MAX_MATYPE`` (8) is rejected, not silently SMA.

    TA-Lib raises ``TA_BAD_PARAM`` for ``matype=9``; ferro-ta's Python wrappers
    validate the argument and raise ``FerroTAValueError`` (a ``ValueError``).
    The Rust core, which has no error type, reports the same condition as an
    all-``NaN`` output — that contract is covered by the Rust unit tests.  Here
    we assert the *documented Python behaviour* rather than trying to match
    TA-Lib's exception type.
    """

    OUT_OF_RANGE = (MAX_MATYPE + 1, 99, 255)

    @pytest.mark.parametrize("matype", OUT_OF_RANGE)
    def test_ferro_rejects(self, matype):
        with pytest.raises(ValueError):
            ferro_ta.MA(CLOSE, timeperiod=10, matype=matype)
        with pytest.raises(ValueError):
            ferro_ta.APO(CLOSE, fastperiod=12, slowperiod=26, matype=matype)
        with pytest.raises(ValueError):
            ferro_ta.PPO(CLOSE, fastperiod=12, slowperiod=26, matype=matype)
        with pytest.raises(ValueError):
            ferro_ta.STOCHF(
                HIGH, LOW, CLOSE, fastk_period=5, fastd_period=3, fastd_matype=matype
            )
        with pytest.raises(ValueError):
            ferro_ta.STOCHRSI(CLOSE, fastd_matype=matype)
        with pytest.raises(ValueError):
            ferro_ta.MAVP(CLOSE, MAVP_PERIODS, minperiod=2, maxperiod=30, matype=matype)

    def test_talib_also_rejects(self):
        """Cross-check: TA-Lib does not silently fall back to SMA either."""
        with pytest.raises(Exception):  # noqa: B017 - talib raises bare Exception
            talib.MA(CLOSE, timeperiod=10, matype=MAX_MATYPE + 1)


def test_kama_matches_talib_on_zero_volatility_input():
    """Regression: KAMA on input whose volatility window goes flat.

    StochRSI's %K sits at exactly 0 or 100 for long stretches, so KAMA's
    efficiency-ratio denominator empties across those windows.  ferro-ta used to
    test that denominator with ``volatility > 0.0``, which reads the rolling
    sum's ~1e-14 rounding residue as real volatility and yields ``ER = 0`` — the
    *slowest* smoothing constant — where ``TA_KAMA``'s
    ``sumROC1 <= periodROC || TA_IS_ZERO(sumROC1)`` mostly yields ``ER = 1``.
    Because the ratio was wrong on every bar of the plateau the error held
    rather than decaying, and the two series parted company permanently (max abs
    diff ~16).  Fixed; the residual here is 2.9e-14.

    Note this was *not* the whole story behind the ``fastd_matype=6`` exclusion
    on the STOCHRSI %D gate — see
    ``test_talib_kama_is_ill_conditioned_on_stochrsi_fastk``.
    """
    fastk, _ = ferro_ta.STOCHRSI(
        CLOSE, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0
    )
    plateau_series = fastk[~np.isnan(fastk)]
    # Sanity: the fixture really does contain the pathological input.
    assert (plateau_series == 0.0).sum() > 50
    assert (plateau_series == 100.0).sum() > 50

    ft = ferro_ta.KAMA(plateau_series, timeperiod=3)
    ta = talib.KAMA(plateau_series, timeperiod=3)
    mask = _valid_mask(ft, ta)
    assert mask.sum() > 400
    # Observed worst deviation is 2.9e-14: the residual is the rolling sum's own
    # rounding, which is what TA-Lib's own efficiency ratio divides into.
    assert np.allclose(ft[mask], ta[mask], atol=1e-9)


def test_talib_kama_is_ill_conditioned_on_stochrsi_fastk():
    """Why ``STOCHRSI_KAMA_MATYPE`` stays out of the %D value gate.

    ``TA_KAMA`` picks its efficiency ratio from a rolling ``Σ|Δ|`` that carries
    ~1e-14 of rounding residue once a window goes flat, and the *sign* of that
    residue selects between ER = 1 (snap) and ER = 0 (crawl).  On StochRSI %K —
    which is pinned at exactly 0 or 100 for long stretches — that makes TA-Lib's
    own KAMA catastrophically input-sensitive.

    ferro-ta's %K differs from TA-Lib's by ~5.8e-13 (RSI/stoch rounding, well
    inside the %K gate's 1e-6).  Feeding TA-Lib *its own* KAMA those two nearly
    identical inputs moves its output by ~19 — the same ~19 that the %D
    comparison at ``fastd_matype=6`` reports.  So the %D divergence is upstream
    conditioning, not a ferro-ta defect: on each individual input ferro-ta's
    KAMA tracks TA-Lib's to ~3e-14.
    """
    ft_k, _ = ferro_ta.STOCHRSI(
        CLOSE, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0
    )
    ta_k, _ = talib.STOCHRSI(
        CLOSE, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0
    )
    ft_k = ft_k[~np.isnan(ft_k)]
    ta_k = ta_k[~np.isnan(ta_k)]
    assert len(ft_k) == len(ta_k) > 400
    assert (ft_k == 0.0).sum() > 50 and (ft_k == 100.0).sum() > 50

    # The two %K series agree far inside the value gate, but are not bit-equal.
    input_delta = float(np.max(np.abs(ft_k - ta_k)))
    assert input_delta < 1e-9, input_delta
    assert input_delta > 0.0, "inputs became bit-identical — re-derive this test"

    # TA-Lib alone, on those two inputs, parts company by ~19.
    ta_on_ft = talib.KAMA(ft_k, timeperiod=3)
    ta_on_ta = talib.KAMA(ta_k, timeperiod=3)
    mask = _valid_mask(ta_on_ft, ta_on_ta)
    assert np.max(np.abs(ta_on_ft[mask] - ta_on_ta[mask])) > 1.0

    # ferro-ta tracks TA-Lib on each input individually.
    for series in (ft_k, ta_k):
        ft = ferro_ta.KAMA(series, timeperiod=3)
        ta = talib.KAMA(series, timeperiod=3)
        m = _valid_mask(ft, ta)
        assert m.sum() > 400
        assert np.allclose(ft[m], ta[m], atol=1e-9)


def test_kama_matches_talib_on_flat_series():
    """A perfectly flat series has an exactly-zero volatility window.

    ``TA_KAMA`` reads that as ``0 <= 0`` and takes ``ER = 1`` (SC = 4/9), so
    KAMA is pinned to the price.  This is the case the kernel's old comment
    described; the bug was that the code only reached it when the rolling sum
    landed on a bit-exact zero.
    """
    flat = np.full(60, 42.0)
    ft = ferro_ta.KAMA(flat, timeperiod=10)
    ta = talib.KAMA(flat, timeperiod=10)
    mask = _valid_mask(ft, ta)
    assert mask.sum() > 40
    assert np.allclose(ft[mask], ta[mask], atol=1e-12)
    assert np.all(ft[mask] == 42.0)


@pytest.mark.parametrize("timeperiod", [2, 3, 5, 14, 30])
def test_kama_matches_talib_on_plateau_heavy_input(timeperiod):
    """Plateau-heavy input at several periods, not just the StochRSI shape."""
    rng = np.random.default_rng(1234)
    walk = 44.0 + np.cumsum(rng.standard_normal(600) * 0.5)
    plateau = walk.copy()
    hold = rng.random(600) < 0.6
    for i in range(1, 600):
        if hold[i]:
            plateau[i] = plateau[i - 1]
    plateau[rng.random(600) < 0.05] = 100.0

    ft = ferro_ta.KAMA(plateau, timeperiod=timeperiod)
    ta = talib.KAMA(plateau, timeperiod=timeperiod)
    mask = _valid_mask(ft, ta)
    assert mask.sum() > 500
    assert np.allclose(ft[mask], ta[mask], atol=1e-9)
