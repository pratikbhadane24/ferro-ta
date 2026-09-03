"""Unit tests for ferro_ta.indicators.utils"""

import numpy as np
import pytest

from ferro_ta.indicators.math_ops import MAX, MIN
from ferro_ta.indicators.utils import (
    CHANGE,
    CROSS,
    CROSSOVER,
    CROSSUNDER,
    EXREM,
    FALLING,
    FLIP,
    HIGHEST,
    LOWEST,
    RISING,
    VALUEWHEN,
)

LINEAR10 = np.arange(1.0, 11.0)
SMALL = np.array([3.0, 1.0, 4.0, 1.0, 5.0])


class TestCROSSOVER:
    def test_known_values(self):
        result = CROSSOVER(np.array([1.0, 2.0, 5.0]), np.array([3.0, 3.0, 3.0]))
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], rtol=1e-10)

    def test_no_cross_when_equal(self):
        a = np.array([1.0, 2.0, 2.0])
        b = np.array([2.0, 2.0, 2.0])
        np.testing.assert_allclose(CROSSOVER(a, b), [0.0, 0.0, 0.0], rtol=1e-10)

    def test_length(self):
        assert len(CROSSOVER(LINEAR10, LINEAR10 + 1.0)) == 10


class TestCROSSUNDER:
    def test_known_values(self):
        result = CROSSUNDER(np.array([5.0, 4.0, 1.0]), np.array([3.0, 3.0, 3.0]))
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], rtol=1e-10)


class TestCROSS:
    def test_either_direction(self):
        a = np.array([1.0, 4.0, 1.0])
        b = np.array([2.0, 2.0, 2.0])
        np.testing.assert_allclose(CROSS(a, b), [0.0, 1.0, 1.0], rtol=1e-10)
        np.testing.assert_allclose(CROSSOVER(a, b), [0.0, 1.0, 0.0], rtol=1e-10)
        np.testing.assert_allclose(CROSSUNDER(a, b), [0.0, 0.0, 1.0], rtol=1e-10)


class TestHIGHESTLOWEST:
    def test_highest_matches_max(self):
        np.testing.assert_allclose(
            HIGHEST(SMALL, timeperiod=3), MAX(SMALL, timeperiod=3), equal_nan=True
        )

    def test_lowest_matches_min(self):
        np.testing.assert_allclose(
            LOWEST(SMALL, timeperiod=3), MIN(SMALL, timeperiod=3), equal_nan=True
        )

    def test_warmup(self):
        result = HIGHEST(SMALL, timeperiod=3)
        assert np.all(np.isnan(result[:2]))
        assert result[2] == pytest.approx(4.0)


class TestCHANGE:
    def test_known_values(self):
        result = CHANGE(LINEAR10, timeperiod=2)
        assert np.all(np.isnan(result[:2]))
        np.testing.assert_allclose(result[2:], np.full(8, 2.0), rtol=1e-10)

    def test_default_period_one(self):
        result = CHANGE(np.array([1.0, 3.0, 2.0]))
        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1:], [2.0, -1.0], rtol=1e-10)


class TestRISINGFALLING:
    def test_vs_n_bars_ago(self):
        x = np.array([1.0, 2.0, 1.5, 3.0])
        np.testing.assert_allclose(RISING(x, timeperiod=2)[2:], [1.0, 1.0], rtol=1e-10)
        np.testing.assert_allclose(FALLING(x, timeperiod=2)[2:], [0.0, 0.0], rtol=1e-10)

    def test_matches_change_sign(self):
        ch = CHANGE(LINEAR10, timeperiod=3)
        up = RISING(LINEAR10, timeperiod=3)
        down = FALLING(LINEAR10, timeperiod=3)
        finite = np.isfinite(ch)
        np.testing.assert_array_equal(up[finite] == 1.0, ch[finite] > 0)
        np.testing.assert_array_equal(down[finite] == 1.0, ch[finite] < 0)


class TestEXREM:
    def test_keeps_first_until_reset(self):
        primary = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
        secondary = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(
            EXREM(primary, secondary), [1.0, 0.0, 0.0, 1.0, 0.0], rtol=1e-10
        )


class TestFLIP:
    def test_holds_until_off(self):
        on = np.array([1.0, 0.0, 0.0, 0.0])
        off = np.array([0.0, 0.0, 1.0, 0.0])
        np.testing.assert_allclose(FLIP(on, off), [1.0, 1.0, 0.0, 0.0], rtol=1e-10)


class TestVALUEWHEN:
    def test_occurrence_one_is_most_recent(self):
        cond = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        src = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        np.testing.assert_allclose(
            VALUEWHEN(cond, src, occurrence=1),
            [np.nan, 20.0, 20.0, 40.0, 40.0],
            equal_nan=True,
        )
        np.testing.assert_allclose(
            VALUEWHEN(cond, src, occurrence=2),
            [np.nan, np.nan, np.nan, 20.0, 20.0],
            equal_nan=True,
        )
