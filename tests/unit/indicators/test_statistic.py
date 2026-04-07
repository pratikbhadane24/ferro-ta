"""Unit tests for ferro_ta.indicators.statistic"""

import numpy as np
import pytest

from ferro_ta.indicators.statistic import (
    BATCH_DTW,
    BETA,
    CORREL,
    DTW,
    DTW_DISTANCE,
    LINEARREG,
    LINEARREG_ANGLE,
    LINEARREG_INTERCEPT,
    LINEARREG_SLOPE,
    STDDEV,
    TSF,
    VAR,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(11)
N = 100
_A = 100 + np.cumsum(RNG.normal(0, 0.5, N))
_B = 100 + np.cumsum(RNG.normal(0, 0.5, N))

LINDATA = np.arange(1.0, 6.0)  # [1,2,3,4,5]
CONSTDATA = np.ones(10)  # all 1.0


def _naive_linreg_window(window: np.ndarray) -> tuple[float, float]:
    x = np.arange(len(window), dtype=np.float64)
    sum_x = float(np.sum(x))
    sum_y = float(np.sum(window))
    sum_xy = float(np.sum(x * window))
    sum_x2 = float(np.sum(x * x))
    n = float(len(window))
    denom = n * sum_x2 - sum_x * sum_x
    slope = (n * sum_xy - sum_x * sum_y) / denom if denom != 0.0 else 0.0
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


def _naive_linearreg(series: np.ndarray, timeperiod: int, x_value: float) -> np.ndarray:
    out = np.full(len(series), np.nan, dtype=np.float64)
    for end in range(timeperiod - 1, len(series)):
        slope, intercept = _naive_linreg_window(series[end + 1 - timeperiod : end + 1])
        out[end] = intercept + slope * x_value
    return out


def _naive_correl(x: np.ndarray, y: np.ndarray, timeperiod: int) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=np.float64)
    for end in range(timeperiod - 1, len(x)):
        x_window = x[end + 1 - timeperiod : end + 1]
        y_window = y[end + 1 - timeperiod : end + 1]
        mean_x = float(np.sum(x_window)) / timeperiod
        mean_y = float(np.sum(y_window)) / timeperiod
        cov = float(np.sum((x_window - mean_x) * (y_window - mean_y)))
        std_x = float(np.sqrt(np.sum((x_window - mean_x) ** 2)))
        std_y = float(np.sqrt(np.sum((y_window - mean_y) ** 2)))
        denom = std_x * std_y
        out[end] = cov / denom if denom != 0.0 else np.nan
    return out


def _naive_beta(x: np.ndarray, y: np.ndarray, timeperiod: int) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=np.float64)
    for end in range(timeperiod, len(x)):
        start = end - timeperiod
        rx = np.array(
            [
                x[idx + 1] / x[idx] - 1.0 if x[idx] != 0.0 else np.nan
                for idx in range(start, end)
            ],
            dtype=np.float64,
        )
        ry = np.array(
            [
                y[idx + 1] / y[idx] - 1.0 if y[idx] != 0.0 else np.nan
                for idx in range(start, end)
            ],
            dtype=np.float64,
        )
        mean_x = float(np.sum(rx)) / timeperiod
        mean_y = float(np.sum(ry)) / timeperiod
        cov = float(np.sum((rx - mean_x) * (ry - mean_y))) / timeperiod
        var_x = float(np.sum((rx - mean_x) ** 2)) / timeperiod
        out[end] = cov / var_x if var_x != 0.0 else np.nan
    return out


# ---------------------------------------------------------------------------
# STDDEV
# ---------------------------------------------------------------------------


class TestSTDDEV:
    def test_constant_is_zero(self):
        result = STDDEV(CONSTDATA, timeperiod=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_known_values(self):
        # std([1,2,3,4,5], ddof=0) = sqrt(2)
        result = STDDEV(LINDATA, timeperiod=5)
        np.testing.assert_allclose(result[4], np.sqrt(2.0), rtol=1e-6)

    def test_nan_warmup(self):
        result = STDDEV(_A, timeperiod=5)
        assert np.all(np.isnan(result[:4]))

    def test_length(self):
        assert len(STDDEV(_A, 5)) == N

    def test_positive(self):
        result = STDDEV(_A, 5)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)


# ---------------------------------------------------------------------------
# VAR
# ---------------------------------------------------------------------------


class TestVAR:
    def test_constant_is_zero(self):
        result = VAR(CONSTDATA, timeperiod=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_known_values(self):
        # var([1,2,3,4,5], ddof=0) = 2.0
        result = VAR(LINDATA, timeperiod=5)
        np.testing.assert_allclose(result[4], 2.0, rtol=1e-6)

    def test_equals_stddev_squared(self):
        std = STDDEV(_A, timeperiod=10)
        var = VAR(_A, timeperiod=10)
        valid = ~np.isnan(std) & ~np.isnan(var)
        np.testing.assert_allclose(var[valid], std[valid] ** 2, rtol=1e-6)

    def test_length(self):
        assert len(VAR(_A, 5)) == N


# ---------------------------------------------------------------------------
# LINEARREG
# ---------------------------------------------------------------------------


class TestLINEARREG:
    def test_perfect_line(self):
        # For [1,2,3,4,5] over window 5, forecast = 5.0
        result = LINEARREG(LINDATA, timeperiod=5)
        np.testing.assert_allclose(result[4], 5.0, rtol=1e-10)

    def test_nan_warmup(self):
        result = LINEARREG(_A, timeperiod=14)
        assert np.all(np.isnan(result[:13]))

    def test_length(self):
        assert len(LINEARREG(_A, 14)) == N

    def test_matches_naive_regression(self):
        expected = _naive_linearreg(_A, timeperiod=14, x_value=13.0)
        result = LINEARREG(_A, timeperiod=14)
        np.testing.assert_allclose(result, expected, equal_nan=True)


# ---------------------------------------------------------------------------
# LINEARREG_SLOPE
# ---------------------------------------------------------------------------


class TestLINEARREG_SLOPE:
    def test_perfect_line_slope_one(self):
        result = LINEARREG_SLOPE(LINDATA, timeperiod=5)
        np.testing.assert_allclose(result[4], 1.0, rtol=1e-10)

    def test_constant_slope_zero(self):
        result = LINEARREG_SLOPE(CONSTDATA, timeperiod=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_length(self):
        assert len(LINEARREG_SLOPE(_A, 14)) == N


# ---------------------------------------------------------------------------
# LINEARREG_INTERCEPT
# ---------------------------------------------------------------------------


class TestLINEARREG_INTERCEPT:
    def test_perfect_line_intercept_one(self):
        # y = [1,2,3,4,5] with x=[0,1,2,3,4] → y = 1 + 1*x → intercept = 1.0
        result = LINEARREG_INTERCEPT(LINDATA, timeperiod=5)
        np.testing.assert_allclose(result[4], 1.0, atol=1e-10)

    def test_length(self):
        assert len(LINEARREG_INTERCEPT(_A, 14)) == N


# ---------------------------------------------------------------------------
# LINEARREG_ANGLE
# ---------------------------------------------------------------------------


class TestLINEARREG_ANGLE:
    def test_slope_one_gives_45_degrees(self):
        result = LINEARREG_ANGLE(LINDATA, timeperiod=5)
        # arctan(1) * 180/pi = 45
        np.testing.assert_allclose(result[4], 45.0, rtol=1e-6)

    def test_constant_gives_zero_degrees(self):
        result = LINEARREG_ANGLE(CONSTDATA, timeperiod=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-8)

    def test_length(self):
        assert len(LINEARREG_ANGLE(_A, 14)) == N


# ---------------------------------------------------------------------------
# BETA
# ---------------------------------------------------------------------------


class TestBETA:
    def test_nan_warmup(self):
        result = BETA(_A, _B, timeperiod=5)
        assert np.all(np.isnan(result[:4]))

    def test_length(self):
        assert len(BETA(_A, _B, 5)) == N

    def test_same_series(self):
        # Beta of x vs x = 1.0 (regression of itself)
        result = BETA(_A, _A, timeperiod=5)
        valid = result[~np.isnan(result)]
        assert np.all(np.isfinite(valid))

    def test_finite_after_warmup(self):
        result = BETA(_A, _B, timeperiod=5)
        valid = result[~np.isnan(result)]
        assert np.all(np.isfinite(valid))

    def test_matches_naive_beta(self):
        expected = _naive_beta(_A, _B, timeperiod=5)
        result = BETA(_A, _B, timeperiod=5)
        np.testing.assert_allclose(result, expected, equal_nan=True)


# ---------------------------------------------------------------------------
# CORREL
# ---------------------------------------------------------------------------


class TestCOREL:
    def test_self_correlation_is_one(self):
        result = CORREL(_A, _A, timeperiod=10)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 1.0, atol=1e-10)

    def test_opposite_correlation_is_minus_one(self):
        arr = np.arange(1.0, 11.0)
        result = CORREL(arr, arr[::-1], timeperiod=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, -1.0, atol=1e-10)

    def test_range(self):
        result = CORREL(_A, _B, timeperiod=10)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= -1 - 1e-10) and np.all(valid <= 1 + 1e-10)

    def test_length(self):
        assert len(CORREL(_A, _B, 10)) == N

    def test_matches_naive_correlation(self):
        expected = _naive_correl(_A, _B, timeperiod=10)
        result = CORREL(_A, _B, timeperiod=10)
        np.testing.assert_allclose(result, expected, equal_nan=True)


# ---------------------------------------------------------------------------
# TSF
# ---------------------------------------------------------------------------


class TestTSF:
    def test_perfect_line(self):
        arr = np.arange(1.0, 10.0)
        result = TSF(arr, timeperiod=3)
        # TSF(3) on [1,2,...] = linear forecast one period ahead
        # Over window [1,2,3]: slope=1, intercept=0 → forecast at bar 2+1=3 → TSF[2]=4
        np.testing.assert_allclose(result[2], 4.0, rtol=1e-10)
        np.testing.assert_allclose(result[3], 5.0, rtol=1e-10)

    def test_nan_warmup(self):
        result = TSF(_A, timeperiod=14)
        assert np.all(np.isnan(result[:13]))

    def test_length(self):
        assert len(TSF(_A, 14)) == N

    def test_matches_naive_tsf(self):
        expected = _naive_linearreg(_A, timeperiod=14, x_value=14.0)
        result = TSF(_A, timeperiod=14)
        np.testing.assert_allclose(result, expected, equal_nan=True)


# ---------------------------------------------------------------------------
# DTW — Dynamic Time Warping
# ---------------------------------------------------------------------------

dtai = pytest.importorskip("dtaidistance", reason="dtaidistance not installed")

_DTW_RNG = np.random.default_rng(42)


class TestDTW:
    # --- Validation against dtaidistance (SOTA reference) ---

    def test_distance_matches_dtaidistance_random(self):
        """Core correctness: our distance == dtaidistance on 20 random pairs."""
        for _ in range(20):
            n = int(_DTW_RNG.integers(5, 50))
            a = _DTW_RNG.random(n)
            b = _DTW_RNG.random(n)
            expected = dtai.dtw.distance(a, b)
            actual = DTW_DISTANCE(a, b)
            np.testing.assert_allclose(
                actual, expected, rtol=1e-9, err_msg=f"Mismatch on series length {n}"
            )

    def test_distance_matches_dtaidistance_unequal_length(self):
        """Handles unequal-length series correctly."""
        for _ in range(10):
            a = _DTW_RNG.random(int(_DTW_RNG.integers(5, 30)))
            b = _DTW_RNG.random(int(_DTW_RNG.integers(5, 30)))
            expected = dtai.dtw.distance(a, b)
            actual = DTW_DISTANCE(a, b)
            np.testing.assert_allclose(actual, expected, rtol=1e-9)

    def test_path_distance_matches_dtaidistance(self):
        """DTW() path variant: returned distance matches dtaidistance."""
        a = _DTW_RNG.random(20)
        b = _DTW_RNG.random(25)
        expected = dtai.dtw.distance(a, b)
        dist, _ = DTW(a, b)
        np.testing.assert_allclose(dist, expected, rtol=1e-9)

    def test_path_matches_dtaidistance_warping_path(self):
        """Warping path matches dtaidistance.dtw.warping_path() on same-length series."""
        for _ in range(10):
            n = int(_DTW_RNG.integers(5, 20))
            a = _DTW_RNG.random(n)
            b = _DTW_RNG.random(n)
            expected_path = dtai.dtw.warping_path(a, b)
            _, actual_path = DTW(a, b)
            actual_pairs = [tuple(int(x) for x in row) for row in actual_path]
            assert actual_pairs == expected_path, (
                f"Path mismatch for n={n}:\n  ours={actual_pairs}\n  dtai={expected_path}"
            )

    def test_window_constrained_matches_dtaidistance(self):
        """Sakoe-Chiba window matches dtaidistance window parameter."""
        a = _DTW_RNG.random(30)
        b = _DTW_RNG.random(30)
        for w in [3, 8, 15]:
            expected = dtai.dtw.distance(a, b, window=w)
            actual = DTW_DISTANCE(a, b, window=w)
            np.testing.assert_allclose(
                actual, expected, rtol=1e-9, err_msg=f"Mismatch at window={w}"
            )

    def test_batch_matches_dtaidistance(self):
        """BATCH_DTW matches calling dtaidistance per-row."""
        ref = _DTW_RNG.random(20)
        matrix = _DTW_RNG.random((8, 20))
        batch_result = BATCH_DTW(matrix, ref)
        for i in range(8):
            expected = dtai.dtw.distance(matrix[i], ref)
            np.testing.assert_allclose(
                batch_result[i],
                expected,
                rtol=1e-9,
                err_msg=f"Batch mismatch at row {i}",
            )

    # --- Mathematical properties ---

    def test_identical_distance_is_zero(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist, _ = DTW(a, a)
        assert dist == pytest.approx(0.0, abs=1e-10)

    def test_symmetry(self):
        a, b = _DTW_RNG.random(20), _DTW_RNG.random(20)
        assert DTW_DISTANCE(a, b) == pytest.approx(DTW_DISTANCE(b, a), rel=1e-10)

    def test_triangle_inequality(self):
        a, b, c = _DTW_RNG.random(15), _DTW_RNG.random(15), _DTW_RNG.random(15)
        assert DTW_DISTANCE(a, c) <= DTW_DISTANCE(a, b) + DTW_DISTANCE(b, c) + 1e-9

    # --- Known hardcoded values ---

    def test_known_shifted_series(self):
        # [0,1,2] vs [1,2,3]: optimal path (0,0)→(1,0)→(2,1)→(2,2)
        # Squared costs: 1+0+0+1=2, sqrt(2). Verified against dtaidistance.
        a = np.array([0.0, 1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(DTW_DISTANCE(a, b), np.sqrt(2.0), rtol=1e-9)

    def test_known_single_element(self):
        # sqrt((3-7)^2) = sqrt(16) = 4.0
        np.testing.assert_allclose(
            DTW_DISTANCE(np.array([3.0]), np.array([7.0])), 4.0, rtol=1e-9
        )

    def test_known_constant_series(self):
        assert DTW_DISTANCE(np.full(10, 5.0), np.full(10, 5.0)) == pytest.approx(
            0.0, abs=1e-12
        )

    # --- Path structural guarantees ---

    def test_path_starts_at_origin(self):
        _, path = DTW(_DTW_RNG.random(10), _DTW_RNG.random(10))
        assert tuple(int(x) for x in path[0]) == (0, 0)

    def test_path_ends_at_corner(self):
        _, path = DTW(_DTW_RNG.random(7), _DTW_RNG.random(9))
        assert tuple(int(x) for x in path[-1]) == (6, 8)

    def test_path_is_monotone(self):
        _, path = DTW(_DTW_RNG.random(20), _DTW_RNG.random(20))
        for k in range(1, len(path)):
            assert path[k][0] >= path[k - 1][0]
            assert path[k][1] >= path[k - 1][1]

    def test_path_steps_unit_size(self):
        _, path = DTW(_DTW_RNG.random(15), _DTW_RNG.random(12))
        for k in range(1, len(path)):
            di = int(path[k][0]) - int(path[k - 1][0])
            dj = int(path[k][1]) - int(path[k - 1][1])
            assert di in (0, 1) and dj in (0, 1)
            assert not (di == 0 and dj == 0)

    # --- DTW_DISTANCE == DTW distance ---

    def test_distance_only_matches_full(self):
        a, b = _DTW_RNG.random(25), _DTW_RNG.random(25)
        d_full, _ = DTW(a, b)
        np.testing.assert_allclose(DTW_DISTANCE(a, b), d_full, rtol=1e-10)

    # --- Batch ---

    def test_batch_single_row(self):
        ref = np.array([1.0, 2.0, 3.0])
        result = BATCH_DTW(np.array([[1.0, 2.0, 3.0]]), ref)
        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_batch_matches_single_calls(self):
        ref = _DTW_RNG.random(20)
        matrix = _DTW_RNG.random((8, 20))
        batch = BATCH_DTW(matrix, ref)
        for i in range(8):
            np.testing.assert_allclose(
                batch[i], DTW_DISTANCE(matrix[i], ref), rtol=1e-10
            )

    # --- Edge cases ---

    def test_empty_series_raises(self):
        with pytest.raises((ValueError, Exception)):
            DTW(np.array([]), np.array([1.0, 2.0]))

    def test_window_constrained_ge_unconstrained(self):
        a, b = _DTW_RNG.random(20), _DTW_RNG.random(20)
        d_full = DTW_DISTANCE(a, b)
        d_narrow = DTW_DISTANCE(a, b, window=2)
        assert d_narrow >= d_full - 1e-9
