"""Property-based tests (Hypothesis) for ferro-ta."""

import numpy as np
import pytest

from ferro_ta import BBANDS, CDLDOJI, EMA, RSI, SMA

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


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestPropertyBased:
    """Placeholder for running property-based tests as a class."""

    def test_import(self):
        assert HAS_HYPOTHESIS
