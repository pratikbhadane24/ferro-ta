"""
Comparison tests: ferro_ta.math_ops vs NumPy (Priority 1 - no optional deps).

Math operators should be exact numpy wrappers. Zero tolerance for deviation.

This module validates that all math operators and transforms in ferro_ta.math_ops
produce identical results to their NumPy equivalents within strict tolerances:
  - Element-wise transforms: atol=1e-14 (direct numpy calls)
  - Binary operators: atol=1e-14 (direct numpy calls)
  - Rolling operators: atol=1e-12 (float sum reordering)
  - Index operators: exact index matching

All tests use NO optional dependencies - they run in every CI environment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ferro_ta.indicators import math_ops

# ---------------------------------------------------------------------------
# Test Data (seeded for reproducibility)
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 100

# Standard test data
CLOSE = 44.0 + np.cumsum(RNG.standard_normal(N) * 0.5)
CLOSE_POSITIVE = np.abs(CLOSE) + 1.0  # For SQRT, LN, LOG10
CLOSE_NORMALIZED = CLOSE / np.max(np.abs(CLOSE))  # For ASIN, ACOS (range [-1, 1])


# ---------------------------------------------------------------------------
# Element-wise Transform Tests
# ---------------------------------------------------------------------------


class TestElementWiseTransforms:
    """Test all 15 unary math transforms against NumPy equivalents.

    Expected tolerance: atol=1e-14 (direct numpy calls)
    """

    def test_sin_exact_match(self):
        """SIN should match np.sin exactly."""
        result = math_ops.SIN(CLOSE)
        expected = np.sin(CLOSE)
        assert np.allclose(result, expected, atol=1e-14)

    def test_cos_exact_match(self):
        """COS should match np.cos exactly."""
        result = math_ops.COS(CLOSE)
        expected = np.cos(CLOSE)
        assert np.allclose(result, expected, atol=1e-14)

    def test_tan_exact_match(self):
        """TAN should match np.tan exactly."""
        result = math_ops.TAN(CLOSE)
        expected = np.tan(CLOSE)
        assert np.allclose(result, expected, atol=1e-14)

    def test_sinh_exact_match(self):
        """SINH should match np.sinh exactly."""
        result = math_ops.SINH(CLOSE)
        expected = np.sinh(CLOSE)
        assert np.allclose(result, expected, atol=1e-14)

    def test_cosh_exact_match(self):
        """COSH should match np.cosh exactly."""
        result = math_ops.COSH(CLOSE)
        expected = np.cosh(CLOSE)
        assert np.allclose(result, expected, atol=1e-14)

    def test_tanh_exact_match(self):
        """TANH should match np.tanh exactly."""
        result = math_ops.TANH(CLOSE)
        expected = np.tanh(CLOSE)
        assert np.allclose(result, expected, atol=1e-14)

    def test_asin_exact_match(self):
        """ASIN should match np.arcsin exactly."""
        result = math_ops.ASIN(CLOSE_NORMALIZED)
        expected = np.arcsin(CLOSE_NORMALIZED)
        assert np.allclose(result, expected, atol=1e-14)

    def test_acos_exact_match(self):
        """ACOS should match np.arccos exactly."""
        result = math_ops.ACOS(CLOSE_NORMALIZED)
        expected = np.arccos(CLOSE_NORMALIZED)
        assert np.allclose(result, expected, atol=1e-14)

    def test_atan_exact_match(self):
        """ATAN should match np.arctan exactly."""
        result = math_ops.ATAN(CLOSE)
        expected = np.arctan(CLOSE)
        assert np.allclose(result, expected, atol=1e-14)

    def test_exp_exact_match(self):
        """EXP should match np.exp exactly."""
        # Use smaller values to avoid overflow
        small_values = CLOSE / 10.0
        result = math_ops.EXP(small_values)
        expected = np.exp(small_values)
        assert np.allclose(result, expected, atol=1e-14)

    def test_ln_exact_match(self):
        """LN should match np.log exactly."""
        result = math_ops.LN(CLOSE_POSITIVE)
        expected = np.log(CLOSE_POSITIVE)
        assert np.allclose(result, expected, atol=1e-14)

    def test_log10_exact_match(self):
        """LOG10 should match np.log10 exactly."""
        result = math_ops.LOG10(CLOSE_POSITIVE)
        expected = np.log10(CLOSE_POSITIVE)
        assert np.allclose(result, expected, atol=1e-14)

    def test_sqrt_exact_match(self):
        """SQRT should match np.sqrt exactly."""
        result = math_ops.SQRT(CLOSE_POSITIVE)
        expected = np.sqrt(CLOSE_POSITIVE)
        assert np.allclose(result, expected, atol=1e-14)

    def test_ceil_exact_match(self):
        """CEIL should match np.ceil exactly."""
        result = math_ops.CEIL(CLOSE)
        expected = np.ceil(CLOSE)
        assert np.allclose(result, expected, atol=1e-14)

    def test_floor_exact_match(self):
        """FLOOR should match np.floor exactly."""
        result = math_ops.FLOOR(CLOSE)
        expected = np.floor(CLOSE)
        assert np.allclose(result, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# Binary Operator Tests
# ---------------------------------------------------------------------------


class TestBinaryOps:
    """Test binary operators against NumPy equivalents.

    Expected tolerance: atol=1e-14 (direct numpy calls)
    """

    def test_add_exact_match(self):
        """ADD should match np.add exactly."""
        other = RNG.standard_normal(N)
        result = math_ops.ADD(CLOSE, other)
        expected = np.add(CLOSE, other)
        assert np.allclose(result, expected, atol=1e-14)

    def test_sub_exact_match(self):
        """SUB should match np.subtract exactly."""
        other = RNG.standard_normal(N)
        result = math_ops.SUB(CLOSE, other)
        expected = np.subtract(CLOSE, other)
        assert np.allclose(result, expected, atol=1e-14)

    def test_mult_exact_match(self):
        """MULT should match np.multiply exactly."""
        other = RNG.standard_normal(N)
        result = math_ops.MULT(CLOSE, other)
        expected = np.multiply(CLOSE, other)
        assert np.allclose(result, expected, atol=1e-14)

    def test_div_exact_match(self):
        """DIV should match np.divide exactly."""
        other = RNG.uniform(0.5, 2.0, N)  # Avoid division by zero
        result = math_ops.DIV(CLOSE, other)
        expected = np.divide(CLOSE, other)
        assert np.allclose(result, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# Rolling Operator Tests
# ---------------------------------------------------------------------------


class TestRollingOps:
    """Test rolling operators against pandas equivalents.

    Expected tolerance: atol=1e-12 (float sum reordering)
    """

    @pytest.mark.parametrize("period", [5, 10, 20, 30])
    def test_sum_matches_pandas_rolling(self, period):
        """SUM should match pd.Series.rolling(p).sum()."""
        result = math_ops.SUM(CLOSE, timeperiod=period)
        expected = pd.Series(CLOSE).rolling(period).sum().to_numpy()

        # Check NaN positions match
        assert np.sum(np.isnan(result)) == np.sum(np.isnan(expected))

        # Check values match where both are finite
        mask = ~np.isnan(result) & ~np.isnan(expected)
        assert np.allclose(result[mask], expected[mask], atol=1e-12)

    @pytest.mark.parametrize("period", [5, 10, 20, 30])
    def test_max_matches_pandas_rolling(self, period):
        """MAX should match pd.Series.rolling(p).max()."""
        result = math_ops.MAX(CLOSE, timeperiod=period)
        expected = pd.Series(CLOSE).rolling(period).max().to_numpy()

        # Check NaN positions match
        assert np.sum(np.isnan(result)) == np.sum(np.isnan(expected))

        # Check values match where both are finite
        mask = ~np.isnan(result) & ~np.isnan(expected)
        assert np.allclose(result[mask], expected[mask], atol=1e-12)

    @pytest.mark.parametrize("period", [5, 10, 20, 30])
    def test_min_matches_pandas_rolling(self, period):
        """MIN should match pd.Series.rolling(p).min()."""
        result = math_ops.MIN(CLOSE, timeperiod=period)
        expected = pd.Series(CLOSE).rolling(period).min().to_numpy()

        # Check NaN positions match
        assert np.sum(np.isnan(result)) == np.sum(np.isnan(expected))

        # Check values match where both are finite
        mask = ~np.isnan(result) & ~np.isnan(expected)
        assert np.allclose(result[mask], expected[mask], atol=1e-12)


# ---------------------------------------------------------------------------
# Index Operator Tests
# ---------------------------------------------------------------------------


class TestIndexOps:
    """Test MAXINDEX and MININDEX point to correct argmax/argmin in window."""

    @pytest.mark.parametrize("period", [5, 10, 20])
    def test_maxindex_points_to_max(self, period):
        """MAXINDEX should point to the index of the rolling maximum."""
        result_idx = math_ops.MAXINDEX(CLOSE, timeperiod=period)
        result_max = math_ops.MAX(CLOSE, timeperiod=period)

        # Skip warmup period
        for i in range(period - 1, N):
            idx = result_idx[i]
            max_val = result_max[i]

            # During warmup, index is -1
            if idx == -1:
                assert np.isnan(max_val)
            else:
                # Index should point to the actual maximum in the window
                assert CLOSE[idx] == max_val, (
                    f"At position {i}, MAXINDEX={idx} but CLOSE[{idx}]={CLOSE[idx]} "
                    f"!= MAX={max_val}"
                )

    @pytest.mark.parametrize("period", [5, 10, 20])
    def test_minindex_points_to_min(self, period):
        """MININDEX should point to the index of the rolling minimum."""
        result_idx = math_ops.MININDEX(CLOSE, timeperiod=period)
        result_min = math_ops.MIN(CLOSE, timeperiod=period)

        # Skip warmup period
        for i in range(period - 1, N):
            idx = result_idx[i]
            min_val = result_min[i]

            # During warmup, index is -1
            if idx == -1:
                assert np.isnan(min_val)
            else:
                # Index should point to the actual minimum in the window
                assert CLOSE[idx] == min_val, (
                    f"At position {i}, MININDEX={idx} but CLOSE[{idx}]={CLOSE[idx]} "
                    f"!= MIN={min_val}"
                )

    def test_maxindex_warmup_returns_minus_one(self):
        """MAXINDEX should return -1 during warmup period."""
        period = 10
        result = math_ops.MAXINDEX(CLOSE, timeperiod=period)

        # First period-1 values should be -1
        for i in range(period - 1):
            assert result[i] == -1, f"Expected -1 at index {i}, got {result[i]}"

    def test_minindex_warmup_returns_minus_one(self):
        """MININDEX should return -1 during warmup period."""
        period = 10
        result = math_ops.MININDEX(CLOSE, timeperiod=period)

        # First period-1 values should be -1
        for i in range(period - 1):
            assert result[i] == -1, f"Expected -1 at index {i}, got {result[i]}"


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and document behavior.

    Documents behavior for:
    - LN(negative) → NaN
    - SQRT(negative) → NaN
    - DIV(by zero) → inf
    - ACOS(>1) → NaN
    """

    def test_ln_negative_returns_nan(self):
        """LN of negative values should return NaN."""
        negative = np.array([-1.0, -2.0, -3.0])
        result = math_ops.LN(negative)
        assert np.all(np.isnan(result)), "LN(negative) should return NaN"

    def test_sqrt_negative_returns_nan(self):
        """SQRT of negative values should return NaN."""
        negative = np.array([-1.0, -4.0, -9.0])
        result = math_ops.SQRT(negative)
        assert np.all(np.isnan(result)), "SQRT(negative) should return NaN"

    def test_div_by_zero_returns_inf(self):
        """DIV by zero should return inf (NumPy behavior)."""
        numerator = np.array([1.0, 2.0, 3.0])
        denominator = np.array([0.0, 0.0, 0.0])
        result = math_ops.DIV(numerator, denominator)
        assert np.all(np.isinf(result)), "DIV(by zero) should return inf"

    def test_acos_out_of_range_returns_nan(self):
        """ACOS of values outside [-1, 1] should return NaN."""
        out_of_range = np.array([1.5, 2.0, -1.5])
        result = math_ops.ACOS(out_of_range)
        assert np.all(np.isnan(result)), "ACOS(>1 or <-1) should return NaN"

    def test_asin_out_of_range_returns_nan(self):
        """ASIN of values outside [-1, 1] should return NaN."""
        out_of_range = np.array([1.5, 2.0, -1.5])
        result = math_ops.ASIN(out_of_range)
        assert np.all(np.isnan(result)), "ASIN(>1 or <-1) should return NaN"

    def test_log10_zero_returns_negative_inf(self):
        """LOG10(0) should return -inf."""
        zero = np.array([0.0])
        result = math_ops.LOG10(zero)
        assert np.isinf(result[0]) and result[0] < 0, "LOG10(0) should return -inf"

    def test_ln_zero_returns_negative_inf(self):
        """LN(0) should return -inf."""
        zero = np.array([0.0])
        result = math_ops.LN(zero)
        assert np.isinf(result[0]) and result[0] < 0, "LN(0) should return -inf"
