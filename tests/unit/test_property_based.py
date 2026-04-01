"""Property-based tests (Hypothesis) for ferro-ta."""

import numpy as np
import pytest

from ferro_ta import ATR, BBANDS, CDLDOJI, EMA, MACD, OBV, RSI, SMA, WMA

try:
    from hypothesis import given, settings
    from hypothesis.strategies import floats, integers, lists

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

if HAS_HYPOTHESIS:
    # Strategy: finite floats, reasonable length
    finite_floats = floats(
        min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False
    )
    price_arrays = lists(finite_floats, min_size=2, max_size=500).map(np.array)
    periods = integers(min_value=1, max_value=100)

    @given(price_arrays, periods)
    @settings(max_examples=50, deadline=5000)
    def test_sma_output_length_matches_input(close, timeperiod):
        if len(close) < timeperiod:
            timeperiod = min(timeperiod, len(close))
            if timeperiod < 1:
                timeperiod = 1
        result = SMA(close, timeperiod=timeperiod)
        assert len(result) == len(close)

    @given(price_arrays, periods)
    @settings(max_examples=50, deadline=5000)
    def test_ema_output_length_matches_input(close, timeperiod):
        if len(close) < timeperiod:
            timeperiod = min(timeperiod, len(close))
            if timeperiod < 1:
                timeperiod = 1
        result = EMA(close, timeperiod=timeperiod)
        assert len(result) == len(close)

    @given(price_arrays, periods)
    @settings(max_examples=50, deadline=5000)
    def test_rsi_output_length_matches_input(close, timeperiod):
        if len(close) < timeperiod:
            timeperiod = min(timeperiod, len(close))
            if timeperiod < 1:
                timeperiod = 1
        result = RSI(close, timeperiod=timeperiod)
        assert len(result) == len(close)

    @given(price_arrays, periods)
    @settings(max_examples=30, deadline=5000)
    def test_bbands_three_outputs_same_length(close, timeperiod):
        if len(close) < timeperiod:
            timeperiod = min(timeperiod, len(close))
            if timeperiod < 1:
                timeperiod = 1
        upper, middle, lower = BBANDS(close, timeperiod=timeperiod)
        assert len(upper) == len(close)
        assert len(middle) == len(close)
        assert len(lower) == len(close)

    @given(
        lists(finite_floats, min_size=3, max_size=100).map(np.array),
        lists(finite_floats, min_size=3, max_size=100).map(np.array),
        lists(finite_floats, min_size=3, max_size=100).map(np.array),
        lists(finite_floats, min_size=3, max_size=100).map(np.array),
    )
    @settings(max_examples=20, deadline=5000)
    def test_cdl_pattern_output_values_in_set(open_, high, low, close):
        n = min(len(open_), len(high), len(low), len(close))
        open_ = open_[:n]
        high = high[:n]
        low = low[:n]
        close = close[:n]
        result = CDLDOJI(open_, high, low, close)
        assert len(result) == n
        assert all(v in (-100, 0, 100) for v in result)

    # ------------------------------------------------------------------
    # EMA extended properties
    # ------------------------------------------------------------------

    @given(price_arrays, integers(min_value=2, max_value=50))
    @settings(max_examples=50, deadline=5000)
    def test_ema_values_finite_when_input_finite(close, timeperiod):
        if len(close) < timeperiod:
            timeperiod = min(timeperiod, len(close))
            if timeperiod < 1:
                timeperiod = 1
        result = EMA(close, timeperiod=timeperiod)
        assert np.all(np.isfinite(result) | np.isnan(result))
        # All non-NaN values must be finite
        valid = result[~np.isnan(result)]
        assert np.all(np.isfinite(valid))

    @given(price_arrays)
    @settings(max_examples=50, deadline=5000)
    def test_ema_period_1_equals_input(close):
        result = EMA(close, timeperiod=1)
        assert len(result) == len(close)
        # EMA with period=1 should reproduce the input exactly
        np.testing.assert_allclose(result, close, rtol=1e-10)

    # ------------------------------------------------------------------
    # BBANDS extended properties
    # ------------------------------------------------------------------

    @given(price_arrays, integers(min_value=2, max_value=50))
    @settings(max_examples=30, deadline=5000)
    def test_bbands_upper_ge_middle_ge_lower(close, timeperiod):
        if len(close) < timeperiod:
            timeperiod = min(timeperiod, len(close))
            if timeperiod < 1:
                timeperiod = 1
        upper, middle, lower = BBANDS(close, timeperiod=timeperiod)
        # Where all three are finite, upper >= middle >= lower
        mask = np.isfinite(upper) & np.isfinite(middle) & np.isfinite(lower)
        assert np.all(upper[mask] >= middle[mask] - 1e-10)
        assert np.all(middle[mask] >= lower[mask] - 1e-10)

    @given(price_arrays, integers(min_value=2, max_value=50))
    @settings(max_examples=30, deadline=5000)
    def test_bbands_middle_equals_sma(close, timeperiod):
        if len(close) < timeperiod:
            timeperiod = min(timeperiod, len(close))
            if timeperiod < 1:
                timeperiod = 1
        _, middle, _ = BBANDS(close, timeperiod=timeperiod)
        sma = SMA(close, timeperiod=timeperiod)
        mask = np.isfinite(middle) & np.isfinite(sma)
        np.testing.assert_allclose(middle[mask], sma[mask], rtol=1e-10)

    # ------------------------------------------------------------------
    # MACD properties
    # ------------------------------------------------------------------

    @given(
        lists(finite_floats, min_size=40, max_size=500).map(np.array),
    )
    @settings(max_examples=50, deadline=5000)
    def test_macd_output_lengths(close):
        macd, signal, hist = MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        assert len(macd) == len(close)
        assert len(signal) == len(close)
        assert len(hist) == len(close)

    @given(
        lists(finite_floats, min_size=40, max_size=500).map(np.array),
    )
    @settings(max_examples=50, deadline=5000)
    def test_macd_histogram_equals_macd_minus_signal(close):
        macd, signal, hist = MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        mask = np.isfinite(macd) & np.isfinite(signal) & np.isfinite(hist)
        if np.any(mask):
            np.testing.assert_allclose(
                hist[mask], macd[mask] - signal[mask], atol=1e-10
            )

    # ------------------------------------------------------------------
    # ATR properties
    # ------------------------------------------------------------------

    @given(
        lists(finite_floats, min_size=20, max_size=500).map(np.array),
        integers(min_value=2, max_value=50),
    )
    @settings(max_examples=50, deadline=5000)
    def test_atr_output_length(prices, timeperiod):
        # Build high/low/close from prices with valid OHLC relationships
        close = prices
        high = prices * 1.01
        low = prices * 0.99
        if len(close) < timeperiod:
            timeperiod = min(timeperiod, len(close))
            if timeperiod < 1:
                timeperiod = 1
        result = ATR(high, low, close, timeperiod=timeperiod)
        assert len(result) == len(close)

    @given(
        lists(finite_floats, min_size=20, max_size=500).map(np.array),
        integers(min_value=2, max_value=50),
    )
    @settings(max_examples=50, deadline=5000)
    def test_atr_non_negative(prices, timeperiod):
        close = prices
        high = prices * 1.01
        low = prices * 0.99
        if len(close) < timeperiod:
            timeperiod = min(timeperiod, len(close))
            if timeperiod < 1:
                timeperiod = 1
        result = ATR(high, low, close, timeperiod=timeperiod)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)

    # ------------------------------------------------------------------
    # WMA properties
    # ------------------------------------------------------------------

    @given(price_arrays, integers(min_value=2, max_value=50))
    @settings(max_examples=50, deadline=5000)
    def test_wma_output_length(close, timeperiod):
        if len(close) < timeperiod:
            timeperiod = min(timeperiod, len(close))
            if timeperiod < 1:
                timeperiod = 1
        result = WMA(close, timeperiod=timeperiod)
        assert len(result) == len(close)

    @given(
        lists(finite_floats, min_size=20, max_size=500).map(np.array),
        integers(min_value=2, max_value=50),
    )
    @settings(max_examples=50, deadline=5000)
    def test_wma_leading_nans(close, timeperiod):
        if len(close) < timeperiod:
            timeperiod = min(timeperiod, len(close))
            if timeperiod < 2:
                timeperiod = 2
        result = WMA(close, timeperiod=timeperiod)
        # First (timeperiod - 1) values should be NaN
        assert np.all(np.isnan(result[: timeperiod - 1]))

    # ------------------------------------------------------------------
    # OBV properties
    # ------------------------------------------------------------------

    @given(
        lists(finite_floats, min_size=20, max_size=500).map(np.array),
        lists(finite_floats, min_size=20, max_size=500).map(np.array),
    )
    @settings(max_examples=50, deadline=5000)
    def test_obv_output_length(close, volume):
        n = min(len(close), len(volume))
        close = close[:n]
        volume = volume[:n]
        result = OBV(close, volume)
        assert len(result) == n

    @given(
        lists(finite_floats, min_size=20, max_size=500).map(np.array),
        lists(finite_floats, min_size=20, max_size=500).map(np.array),
    )
    @settings(max_examples=50, deadline=5000)
    def test_obv_all_finite(close, volume):
        n = min(len(close), len(volume))
        close = close[:n]
        volume = volume[:n]
        result = OBV(close, volume)
        assert np.all(np.isfinite(result))


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestPropertyBased:
    """Placeholder for running property-based tests as a class."""

    def test_import(self):
        assert HAS_HYPOTHESIS
