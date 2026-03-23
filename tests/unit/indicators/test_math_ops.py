"""Unit tests for ferro_ta.indicators.math_ops"""

import numpy as np

from ferro_ta.indicators.math_ops import (
    ACOS,
    ADD,
    ASIN,
    ATAN,
    CEIL,
    COS,
    COSH,
    DIV,
    EXP,
    FLOOR,
    LN,
    LOG10,
    MAX,
    MAXINDEX,
    MIN,
    MININDEX,
    MULT,
    SIN,
    SINH,
    SQRT,
    SUB,
    SUM,
    TAN,
    TANH,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

A3 = np.array([1.0, 2.0, 3.0])
B3 = np.array([4.0, 5.0, 6.0])
TRIG = np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])
UNIT = np.array([0.0, 0.25, 0.5, 0.75, 1.0])  # values in [0,1] for ASIN/ACOS

RNG = np.random.default_rng(17)
N = 100
_ARR = 1.0 + RNG.random(N) * 9.0  # positive values in (1, 10]


# ---------------------------------------------------------------------------
# ADD
# ---------------------------------------------------------------------------


class TestADD:
    def test_known_values(self):
        result = ADD(A3, B3)
        np.testing.assert_allclose(result, [5.0, 7.0, 9.0], rtol=1e-10)

    def test_commutative(self):
        np.testing.assert_allclose(ADD(A3, B3), ADD(B3, A3), rtol=1e-10)

    def test_length(self):
        assert len(ADD(_ARR, _ARR)) == N


# ---------------------------------------------------------------------------
# SUB
# ---------------------------------------------------------------------------


class TestSUB:
    def test_known_values(self):
        result = SUB(B3, A3)
        np.testing.assert_allclose(result, [3.0, 3.0, 3.0], rtol=1e-10)

    def test_length(self):
        assert len(SUB(_ARR, _ARR)) == N


# ---------------------------------------------------------------------------
# MULT
# ---------------------------------------------------------------------------


class TestMULT:
    def test_known_values(self):
        result = MULT(A3, B3)
        np.testing.assert_allclose(result, [4.0, 10.0, 18.0], rtol=1e-10)

    def test_commutative(self):
        np.testing.assert_allclose(MULT(A3, B3), MULT(B3, A3), rtol=1e-10)

    def test_length(self):
        assert len(MULT(_ARR, _ARR)) == N


# ---------------------------------------------------------------------------
# DIV
# ---------------------------------------------------------------------------


class TestDIV:
    def test_known_values(self):
        result = DIV(B3, A3)
        np.testing.assert_allclose(result, [4.0, 2.5, 2.0], rtol=1e-10)

    def test_self_division_is_one(self):
        np.testing.assert_allclose(DIV(_ARR, _ARR), np.ones(N), rtol=1e-10)

    def test_length(self):
        assert len(DIV(_ARR, _ARR)) == N


# ---------------------------------------------------------------------------
# SUM
# ---------------------------------------------------------------------------


class TestSUM:
    def test_known_values(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = SUM(arr, timeperiod=3)
        assert np.isnan(result[0]) and np.isnan(result[1])
        np.testing.assert_allclose(result[2], 6.0, rtol=1e-10)
        np.testing.assert_allclose(result[4], 12.0, rtol=1e-10)

    def test_nan_warmup(self):
        result = SUM(_ARR, timeperiod=5)
        assert np.all(np.isnan(result[:4]))

    def test_length(self):
        assert len(SUM(_ARR, 5)) == N


# ---------------------------------------------------------------------------
# MAX
# ---------------------------------------------------------------------------


class TestMAX:
    def test_known_values(self):
        arr = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        result = MAX(arr, timeperiod=3)
        assert np.isnan(result[0]) and np.isnan(result[1])
        np.testing.assert_allclose(result[2], 3.0, rtol=1e-10)
        np.testing.assert_allclose(result[3], 5.0, rtol=1e-10)
        np.testing.assert_allclose(result[4], 5.0, rtol=1e-10)

    def test_nan_warmup(self):
        result = MAX(_ARR, timeperiod=5)
        assert np.all(np.isnan(result[:4]))

    def test_length(self):
        assert len(MAX(_ARR, 5)) == N


# ---------------------------------------------------------------------------
# MIN
# ---------------------------------------------------------------------------


class TestMIN:
    def test_known_values(self):
        arr = np.array([5.0, 3.0, 4.0, 1.0, 2.0])
        result = MIN(arr, timeperiod=3)
        assert np.isnan(result[0]) and np.isnan(result[1])
        np.testing.assert_allclose(result[2], 3.0, rtol=1e-10)
        np.testing.assert_allclose(result[3], 1.0, rtol=1e-10)

    def test_length(self):
        assert len(MIN(_ARR, 5)) == N


# ---------------------------------------------------------------------------
# MAXINDEX
# ---------------------------------------------------------------------------


class TestMAXINDEX:
    def test_known_values(self):
        arr = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        result = MAXINDEX(arr, timeperiod=3)
        # warmup entries are -1 (sentinel for "no data")
        assert result[0] < 0 and result[1] < 0
        # window[0:3] = [1,5,3] → max at local index 1 → absolute index 1
        np.testing.assert_allclose(result[2], 1.0, rtol=1e-10)
        # window[2:5] = [3,2,4] → max at local index 2 → absolute index 4
        np.testing.assert_allclose(result[4], 4.0, rtol=1e-10)

    def test_length(self):
        assert len(MAXINDEX(_ARR, 5)) == N


# ---------------------------------------------------------------------------
# MININDEX
# ---------------------------------------------------------------------------


class TestMININDEX:
    def test_known_values(self):
        arr = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        result = MININDEX(arr, timeperiod=3)
        # warmup entries are -1 (sentinel for "no data")
        assert result[0] < 0 and result[1] < 0
        # window[0:3] = [5,1,3] → min at local index 1 → absolute index 1
        np.testing.assert_allclose(result[2], 1.0, rtol=1e-10)
        # window[2:5] = [3,2,4] → min at local index 1 → absolute index 3
        np.testing.assert_allclose(result[4], 3.0, rtol=1e-10)

    def test_length(self):
        assert len(MININDEX(_ARR, 5)) == N


# ---------------------------------------------------------------------------
# Trig functions
# ---------------------------------------------------------------------------


class TestSIN:
    def test_known_values(self):
        angles = np.array([0.0, np.pi / 2, np.pi])
        result = SIN(angles)
        np.testing.assert_allclose(result, np.sin(angles), atol=1e-10)

    def test_matches_numpy(self):
        np.testing.assert_allclose(SIN(TRIG), np.sin(TRIG), rtol=1e-10)


class TestCOS:
    def test_matches_numpy(self):
        np.testing.assert_allclose(COS(TRIG), np.cos(TRIG), rtol=1e-10)


class TestTAN:
    def test_matches_numpy(self):
        safe = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(TAN(safe), np.tan(safe), rtol=1e-10)


class TestASIN:
    def test_matches_numpy(self):
        np.testing.assert_allclose(ASIN(UNIT), np.arcsin(UNIT), rtol=1e-10)


class TestACOS:
    def test_matches_numpy(self):
        np.testing.assert_allclose(ACOS(UNIT), np.arccos(UNIT), rtol=1e-10)


class TestATAN:
    def test_matches_numpy(self):
        np.testing.assert_allclose(ATAN(TRIG), np.arctan(TRIG), rtol=1e-10)


class TestSINH:
    def test_matches_numpy(self):
        np.testing.assert_allclose(SINH(A3), np.sinh(A3), rtol=1e-10)


class TestCOSH:
    def test_matches_numpy(self):
        np.testing.assert_allclose(COSH(A3), np.cosh(A3), rtol=1e-10)


class TestTANH:
    def test_matches_numpy(self):
        np.testing.assert_allclose(TANH(UNIT), np.tanh(UNIT), rtol=1e-10)


# ---------------------------------------------------------------------------
# Rounding/exponential
# ---------------------------------------------------------------------------


class TestCEIL:
    def test_known_values(self):
        arr = np.array([1.1, 2.5, 3.9, -0.5])
        np.testing.assert_allclose(CEIL(arr), np.ceil(arr), rtol=1e-10)


class TestFLOOR:
    def test_known_values(self):
        arr = np.array([1.1, 2.5, 3.9, -0.5])
        np.testing.assert_allclose(FLOOR(arr), np.floor(arr), rtol=1e-10)


class TestEXP:
    def test_matches_numpy(self):
        np.testing.assert_allclose(EXP(A3), np.exp(A3), rtol=1e-10)

    def test_exp_zero_is_one(self):
        np.testing.assert_allclose(EXP(np.array([0.0])), [1.0], rtol=1e-10)


class TestLN:
    def test_matches_numpy(self):
        np.testing.assert_allclose(LN(_ARR), np.log(_ARR), rtol=1e-10)

    def test_ln_exp_inverse(self):
        np.testing.assert_allclose(LN(EXP(A3)), A3, rtol=1e-10)


class TestLOG10:
    def test_matches_numpy(self):
        np.testing.assert_allclose(LOG10(_ARR), np.log10(_ARR), rtol=1e-10)

    def test_log10_of_100_is_2(self):
        np.testing.assert_allclose(LOG10(np.array([100.0])), [2.0], rtol=1e-10)


class TestSQRT:
    def test_matches_numpy(self):
        np.testing.assert_allclose(SQRT(_ARR), np.sqrt(_ARR), rtol=1e-10)

    def test_sqrt_of_4_is_2(self):
        np.testing.assert_allclose(SQRT(np.array([4.0])), [2.0], rtol=1e-10)
