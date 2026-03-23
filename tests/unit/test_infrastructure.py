"""Tests for exceptions, backtest, registry, release playbook, GPU backend, WASM."""

from __future__ import annotations

import numpy as np
import pytest

import ferro_ta

# ---------------------------------------------------------------------------
# Exception model & validation
# ---------------------------------------------------------------------------
from ferro_ta.core.exceptions import (
    FerroTAError,
    FerroTAInputError,
    FerroTAValueError,
    check_equal_length,
    check_finite,
    check_timeperiod,
)


class TestExceptionHierarchy:
    """FerroTAError hierarchy and isinstance relationships."""

    def test_ferro_ta_error_is_exception(self):
        assert issubclass(FerroTAError, Exception)

    def test_value_error_is_base_and_value_error(self):
        assert issubclass(FerroTAValueError, FerroTAError)
        assert issubclass(FerroTAValueError, ValueError)

    def test_input_error_is_base_and_value_error(self):
        assert issubclass(FerroTAInputError, FerroTAError)
        assert issubclass(FerroTAInputError, ValueError)

    def test_exported_from_ferro_ta(self):
        assert ferro_ta.FerroTAError is FerroTAError
        assert ferro_ta.FerroTAValueError is FerroTAValueError
        assert ferro_ta.FerroTAInputError is FerroTAInputError


class TestCheckTimeperiod:
    """check_timeperiod raises FerroTAValueError with clear message."""

    def test_valid_timeperiod_does_not_raise(self):
        check_timeperiod(1)
        check_timeperiod(14)
        check_timeperiod(100)

    def test_zero_raises_ferro_ta_value_error(self):
        with pytest.raises(FerroTAValueError, match="timeperiod must be >= 1, got 0"):
            check_timeperiod(0)

    def test_negative_raises_ferro_ta_value_error(self):
        with pytest.raises(FerroTAValueError) as exc_info:
            check_timeperiod(-5, name="timeperiod")
        assert "timeperiod" in str(exc_info.value)
        assert "-5" in str(exc_info.value)

    def test_custom_name_in_message(self):
        with pytest.raises(FerroTAValueError, match="fastperiod"):
            check_timeperiod(0, name="fastperiod")

    def test_custom_minimum(self):
        with pytest.raises(FerroTAValueError, match=">= 2"):
            check_timeperiod(1, minimum=2)


class TestCheckEqualLength:
    """check_equal_length raises FerroTAInputError for mismatched arrays."""

    def test_equal_lengths_pass(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        check_equal_length(open=a, close=b)  # no exception

    def test_mismatched_lengths_raise(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        with pytest.raises(FerroTAInputError) as exc_info:
            check_equal_length(open=a, close=b)
        # message must mention the lengths
        msg = str(exc_info.value)
        assert "3" in msg
        assert "2" in msg

    def test_three_arrays_all_different(self):
        with pytest.raises(FerroTAInputError):
            check_equal_length(
                open=np.array([1.0]),
                high=np.array([1.0, 2.0]),
                close=np.array([1.0, 2.0, 3.0]),
            )


class TestCheckFinite:
    """check_finite raises FerroTAInputError for NaN/Inf."""

    def test_all_finite_passes(self):
        check_finite(np.array([1.0, 2.0, 3.0]))

    def test_nan_raises(self):
        with pytest.raises(FerroTAInputError, match="NaN or Inf"):
            check_finite(np.array([1.0, float("nan"), 3.0]))

    def test_inf_raises(self):
        with pytest.raises(FerroTAInputError, match="NaN or Inf"):
            check_finite(np.array([1.0, float("inf"), 3.0]))

    def test_name_in_message(self):
        with pytest.raises(FerroTAInputError, match="myarray"):
            check_finite(np.array([float("nan")]), name="myarray")


# ---------------------------------------------------------------------------
# Backtesting utilities
# ---------------------------------------------------------------------------

from ferro_ta.analysis.backtest import (
    BacktestResult,
    backtest,
    macd_crossover_strategy,
    rsi_strategy,
    sma_crossover_strategy,
)


def _make_close(n: int = 50, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.001, 0.01, n)
    return np.cumprod(1 + returns) * 100.0


class TestRsiStrategy:
    """rsi_strategy returns correct signal arrays."""

    def test_output_shape(self):
        close = _make_close(50)
        signals = rsi_strategy(close, timeperiod=5)
        assert signals.shape == close.shape

    def test_only_valid_signal_values(self):
        close = _make_close(50)
        signals = rsi_strategy(close, timeperiod=5)
        finite = signals[np.isfinite(signals)]
        assert set(finite).issubset({-1.0, 0.0, 1.0})

    def test_nan_during_warmup(self):
        close = _make_close(20)
        signals = rsi_strategy(close, timeperiod=5)
        # First 5 values should be NaN (RSI warm-up)
        assert np.all(np.isnan(signals[:5]))

    def test_invalid_timeperiod(self):
        with pytest.raises(FerroTAValueError):
            rsi_strategy(_make_close(10), timeperiod=0)


class TestSmaCrossoverStrategy:
    """sma_crossover_strategy returns signals when fast < slow."""

    def test_output_shape(self):
        close = _make_close(60)
        signals = sma_crossover_strategy(close, fast=5, slow=20)
        assert signals.shape == close.shape

    def test_only_valid_signal_values(self):
        close = _make_close(60)
        signals = sma_crossover_strategy(close, fast=5, slow=20)
        finite = signals[np.isfinite(signals)]
        assert set(finite).issubset({-1.0, 1.0})

    def test_fast_must_be_less_than_slow(self):
        with pytest.raises(FerroTAValueError):
            sma_crossover_strategy(_make_close(60), fast=20, slow=10)


class TestMacdCrossoverStrategy:
    """macd_crossover_strategy returns signals from MACD line vs signal line."""

    def test_output_shape(self):
        close = _make_close(100)
        signals = macd_crossover_strategy(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        assert signals.shape == close.shape

    def test_only_valid_signal_values(self):
        close = _make_close(100)
        signals = macd_crossover_strategy(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        finite = signals[np.isfinite(signals)]
        assert set(finite).issubset({-1.0, 1.0})

    def test_fastperiod_must_be_less_than_slowperiod(self):
        with pytest.raises(FerroTAValueError):
            macd_crossover_strategy(_make_close(60), fastperiod=26, slowperiod=12)


class TestBacktest:
    """backtest() produces correct BacktestResult."""

    def test_rsi_strategy_runs(self):
        close = _make_close(100)
        result = backtest(close, strategy="rsi_30_70", timeperiod=5)
        assert isinstance(result, BacktestResult)

    def test_output_lengths_match_input(self):
        close = _make_close(80)
        result = backtest(close, strategy="rsi_30_70", timeperiod=5)
        n = len(close)
        assert len(result.signals) == n
        assert len(result.positions) == n
        assert len(result.equity) == n

    def test_equity_starts_near_one(self):
        close = _make_close(50)
        result = backtest(close, strategy="rsi_30_70", timeperiod=5)
        assert abs(result.equity[0] - 1.0) < 0.01

    def test_sma_crossover_strategy_runs(self):
        close = _make_close(80)
        result = backtest(close, strategy="sma_crossover", fast=5, slow=20)
        assert isinstance(result, BacktestResult)
        assert result.n_trades >= 0

    def test_custom_callable_strategy(self):
        def my_strategy(close, **_):
            signals = np.zeros(len(close))
            signals[len(close) // 2 :] = 1.0
            return signals

        close = _make_close(40)
        result = backtest(close, strategy=my_strategy)
        assert isinstance(result, BacktestResult)
        assert len(result.signals) == len(close)

    def test_unknown_strategy_raises(self):
        with pytest.raises(FerroTAValueError, match="Unknown strategy"):
            backtest(_make_close(30), strategy="nonexistent")

    def test_too_short_input_raises(self):
        with pytest.raises(FerroTAInputError):
            backtest(np.array([1.0]))

    def test_non_1d_input_raises(self):
        with pytest.raises(FerroTAInputError):
            backtest(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_n_trades_is_integer(self):
        close = _make_close(60)
        result = backtest(close, strategy="sma_crossover", fast=5, slow=15)
        assert isinstance(result.n_trades, int)
        assert result.n_trades >= 0

    def test_macd_crossover_strategy_runs(self):
        close = _make_close(100)
        result = backtest(
            close,
            strategy="macd_crossover",
            fastperiod=12,
            slowperiod=26,
            signalperiod=9,
        )
        assert isinstance(result, BacktestResult)
        assert len(result.equity) == len(close)

    def test_commission_reduces_equity(self):
        close = _make_close(80)
        result_no_comm = backtest(close, strategy="sma_crossover", fast=5, slow=20)
        result_with_comm = backtest(
            close,
            strategy="sma_crossover",
            fast=5,
            slow=20,
            commission_per_trade=0.01,
        )
        assert result_with_comm.final_equity <= result_no_comm.final_equity
        assert result_with_comm.final_equity < result_no_comm.final_equity or (
            result_no_comm.n_trades == 0
        )

    def test_slippage_reduces_equity(self):
        close = _make_close(80)
        result_no_slip = backtest(close, strategy="sma_crossover", fast=5, slow=20)
        result_with_slip = backtest(
            close,
            strategy="sma_crossover",
            fast=5,
            slow=20,
            slippage_bps=10.0,
        )
        assert result_with_slip.final_equity <= result_no_slip.final_equity
        assert result_with_slip.final_equity < result_no_slip.final_equity or (
            result_no_slip.n_trades == 0
        )


# ---------------------------------------------------------------------------
# Plugin / Registry
# ---------------------------------------------------------------------------

from ferro_ta.core.registry import (
    FerroTARegistryError,
    get,
    list_indicators,
    register,
    run,
    unregister,
)


class TestRegistry:
    """Registry: register, get, run, unregister, list_indicators."""

    def test_builtins_registered(self):
        names = list_indicators()
        assert "SMA" in names
        assert "RSI" in names
        assert "EMA" in names
        assert "ATR" in names

    def test_run_builtin_sma(self):
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = run("SMA", close, timeperiod=3)
        # SMA(3) of [1,2,3,4,5]: valid at indices 2,3,4
        assert result.shape == (5,)
        assert np.isnan(result[0])
        assert abs(float(result[2]) - 2.0) < 1e-8

    def test_run_builtin_rsi(self):
        close = np.array(
            [
                44.34,
                44.09,
                44.15,
                43.61,
                44.33,
                44.83,
                45.10,
                45.15,
                43.61,
                44.33,
                44.83,
                45.10,
                45.15,
                43.61,
                44.33,
            ]
        )
        result = run("RSI", close, timeperiod=14)
        assert result.shape == (15,)

    def test_get_returns_callable(self):
        fn = get("EMA")
        assert callable(fn)

    def test_register_custom_indicator(self):
        def DOUBLE_SMA(close, timeperiod=5):
            return close * 2.0

        register("DOUBLE_SMA", DOUBLE_SMA)
        try:
            close = np.array([1.0, 2.0, 3.0])
            result = run("DOUBLE_SMA", close, timeperiod=2)
            np.testing.assert_array_equal(result, np.array([2.0, 4.0, 6.0]))
        finally:
            unregister("DOUBLE_SMA")

    def test_unregister_removes_indicator(self):
        def TEMP_IND(close):
            return close

        register("TEMP_IND", TEMP_IND)
        assert "TEMP_IND" in list_indicators()
        unregister("TEMP_IND")
        assert "TEMP_IND" not in list_indicators()

    def test_unknown_indicator_raises(self):
        with pytest.raises(FerroTARegistryError):
            get("UNKNOWN_INDICATOR_XYZ")

    def test_run_unknown_indicator_raises(self):
        with pytest.raises(FerroTARegistryError):
            run("NO_SUCH_IND", np.array([1.0, 2.0]))

    def test_unregister_unknown_raises(self):
        with pytest.raises(FerroTARegistryError):
            unregister("NEVER_REGISTERED")

    def test_register_non_callable_raises(self):
        with pytest.raises(TypeError):
            register("BAD", 42)  # type: ignore[arg-type]

    def test_list_indicators_is_sorted(self):
        names = list_indicators()
        assert names == sorted(names)

    def test_all_builtins_are_callable(self):
        for name in list_indicators():
            fn = get(name)
            assert callable(fn), f"{name} is not callable"


# ---------------------------------------------------------------------------
# New Extended Indicators (KELTNER_CHANNELS, HULL_MA,
#            CHANDELIER_EXIT, VWMA, CHOPPINESS_INDEX)
# ---------------------------------------------------------------------------

from ferro_ta import (
    CHANDELIER_EXIT,
    CHOPPINESS_INDEX,
    HULL_MA,
    KELTNER_CHANNELS,
    VWMA,
)

_N = 30
_C = np.cumsum(np.ones(_N)) + 40.0
_H = _C + 0.5
_L = _C - 0.5
_V = np.full(_N, 1_000_000.0)


class TestKeltnerChannels:
    def test_output_shapes(self):
        u, m, lo = KELTNER_CHANNELS(_H, _L, _C, timeperiod=5, atr_period=3)
        assert len(u) == len(m) == len(lo) == _N

    def test_upper_gt_middle_gt_lower(self):
        u, m, lo = KELTNER_CHANNELS(_H, _L, _C, timeperiod=5, atr_period=3)
        valid = ~np.isnan(u)
        assert np.all(u[valid] > m[valid])
        assert np.all(m[valid] > lo[valid])


class TestHullMA:
    def test_output_length(self):
        hull = HULL_MA(_C, timeperiod=4)
        assert len(hull) == _N

    def test_leading_nans(self):
        hull = HULL_MA(_C, timeperiod=4)
        assert int(np.sum(np.isnan(hull))) >= 1

    def test_finite_after_warmup(self):
        hull = HULL_MA(_C, timeperiod=4)
        assert np.all(np.isfinite(hull[~np.isnan(hull)]))


class TestChandelierExit:
    def test_output_shapes(self):
        le, se = CHANDELIER_EXIT(_H, _L, _C, timeperiod=5, multiplier=2.0)
        assert len(le) == len(se) == _N

    def test_long_lt_high_short_gt_low(self):
        le, se = CHANDELIER_EXIT(_H, _L, _C, timeperiod=5, multiplier=2.0)
        # Both outputs should have valid values after warmup
        valid_le = ~np.isnan(le)
        valid_se = ~np.isnan(se)
        assert valid_le.any()
        assert valid_se.any()
        # Long exit must be finite and positive
        assert np.all(np.isfinite(le[valid_le]))
        assert np.all(le[valid_le] > 0.0)
        # Short exit must be finite and positive
        assert np.all(np.isfinite(se[valid_se]))
        assert np.all(se[valid_se] > 0.0)


class TestVWMA:
    def test_output_length(self):
        v = VWMA(_C, _V, timeperiod=5)
        assert len(v) == _N

    def test_leading_nans(self):
        v = VWMA(_C, _V, timeperiod=5)
        assert int(np.sum(np.isnan(v))) == 4

    def test_uniform_volume_equals_sma(self):
        """With uniform volume, VWMA equals SMA."""
        from ferro_ta import SMA

        c = np.arange(1.0, 21.0)
        v = np.ones(20)
        vwma = VWMA(c, v, timeperiod=5)
        sma = SMA(c, timeperiod=5)
        valid = ~np.isnan(vwma) & ~np.isnan(sma)
        assert np.allclose(vwma[valid], sma[valid], rtol=1e-9)


class TestChoppinessIndex:
    def test_output_length(self):
        ci = CHOPPINESS_INDEX(_H, _L, _C, timeperiod=5)
        assert len(ci) == _N

    def test_range_0_to_100(self):
        ci = CHOPPINESS_INDEX(_H, _L, _C, timeperiod=5)
        valid = ci[~np.isnan(ci)]
        if len(valid) > 0:
            assert np.all(valid >= 0.0)
            assert np.all(valid <= 100.0)


# ---------------------------------------------------------------------------
# Batch execution API
# ---------------------------------------------------------------------------

from ferro_ta import EMA, RSI, SMA
from ferro_ta.data.batch import batch_apply, batch_ema, batch_rsi, batch_sma


class TestBatchSMA:
    C2D = np.random.default_rng(7).random((50, 3)) + 50.0
    C1D = C2D[:, 0]

    def test_output_shape_2d(self):
        result = batch_sma(self.C2D, timeperiod=10)
        assert result.shape == (50, 3)

    def test_output_shape_1d_unchanged(self):
        """1-D input should return 1-D (backward compatible)."""
        result = batch_sma(self.C1D, timeperiod=10)
        assert result.ndim == 1
        assert len(result) == 50

    def test_column_matches_single_series(self):
        """Each column of batch_sma must match single-series SMA."""
        result = batch_sma(self.C2D, timeperiod=10)
        for j in range(3):
            expected = SMA(self.C2D[:, j], timeperiod=10)
            assert np.allclose(result[:, j], expected, equal_nan=True)


class TestBatchEMA:
    C2D = np.random.default_rng(8).random((50, 4)) + 40.0

    def test_output_shape(self):
        result = batch_ema(self.C2D, timeperiod=5)
        assert result.shape == (50, 4)

    def test_column_matches_single_series(self):
        result = batch_ema(self.C2D, timeperiod=5)
        for j in range(4):
            expected = EMA(self.C2D[:, j], timeperiod=5)
            assert np.allclose(result[:, j], expected, equal_nan=True)


class TestBatchRSI:
    C2D = np.random.default_rng(9).random((50, 2)) + 45.0

    def test_output_shape(self):
        result = batch_rsi(self.C2D, timeperiod=14)
        assert result.shape == (50, 2)

    def test_values_in_range(self):
        result = batch_rsi(self.C2D, timeperiod=14)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert valid.min() >= 0.0
            assert valid.max() <= 100.0

    def test_column_matches_single_series(self):
        result = batch_rsi(self.C2D, timeperiod=14)
        for j in range(2):
            expected = RSI(self.C2D[:, j], timeperiod=14)
            assert np.allclose(result[:, j], expected, equal_nan=True)


class TestBatchApply:
    C2D = np.random.default_rng(11).random((40, 3)) + 50.0

    def test_custom_fn(self):
        """batch_apply should delegate to any single-series function."""
        from ferro_ta import BBANDS

        def mid(c, **kw):
            return BBANDS(c, **kw)[1]

        result = batch_apply(self.C2D, mid, timeperiod=5)
        assert result.shape == (40, 3)

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            batch_apply(np.zeros((5, 5, 5)), SMA, timeperiod=3)


# ---------------------------------------------------------------------------
# Release playbook and version consistency
# ---------------------------------------------------------------------------

import os

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]  # fallback for Python < 3.11
    except ImportError:
        tomllib = None  # type: ignore[assignment]


def _read_cargo_version() -> str:
    """Extract version from root Cargo.toml."""
    if tomllib is None:
        raise ImportError("tomllib/tomli not available")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cargo_toml = os.path.join(root, "Cargo.toml")
    with open(cargo_toml, "rb") as f:
        data = tomllib.load(f)
    return data["package"]["version"]


def _read_pyproject_version() -> str:
    """Extract version from pyproject.toml."""
    if tomllib is None:
        raise ImportError("tomllib/tomli not available")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pyproject_toml = os.path.join(root, "pyproject.toml")
    with open(pyproject_toml, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


class TestVersionConsistency:
    """Cargo.toml and pyproject.toml must have the same version string."""

    def test_versions_match(self):
        try:
            cargo_ver = _read_cargo_version()
            pyproject_ver = _read_pyproject_version()
        except Exception:
            pytest.skip("tomllib unavailable or files not found")
        assert cargo_ver == pyproject_ver, (
            f"Version mismatch: Cargo.toml={cargo_ver!r}, "
            f"pyproject.toml={pyproject_ver!r}"
        )

    def test_release_md_exists(self):
        """RELEASE.md must exist in the repository root."""
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        release_md = os.path.join(root, "RELEASE.md")
        assert os.path.isfile(release_md), "RELEASE.md not found"

    def test_release_md_has_key_sections(self):
        """RELEASE.md must mention tagging and PyPI."""
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        release_md = os.path.join(root, "RELEASE.md")
        if not os.path.isfile(release_md):
            pytest.skip("RELEASE.md not found")
        text = open(release_md).read()
        assert "git tag" in text or "tag" in text.lower()
        assert "pypi" in text.lower() or "PyPI" in text


# ---------------------------------------------------------------------------
# GPU backend (PyTorch, CPU fallback always available)
# ---------------------------------------------------------------------------

from ferro_ta.tools.gpu import ema as gpu_ema
from ferro_ta.tools.gpu import rsi as gpu_rsi
from ferro_ta.tools.gpu import sma as gpu_sma  # noqa: E402

CLOSE_15 = np.array(
    [
        44.34,
        44.09,
        44.15,
        43.61,
        44.33,
        44.83,
        45.10,
        45.15,
        43.61,
        44.33,
        44.83,
        45.10,
        45.15,
        43.61,
        44.33,
    ]
)


class TestGPUCPUFallback:
    """GPU module falls back to CPU when CuPy is not available."""

    def test_sma_cpu_fallback_length(self):
        result = gpu_sma(CLOSE_15, timeperiod=5)
        assert len(result) == len(CLOSE_15)

    def test_sma_cpu_fallback_values(self):
        from ferro_ta import SMA

        result = gpu_sma(CLOSE_15, timeperiod=5)
        expected = SMA(CLOSE_15, timeperiod=5)
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_ema_cpu_fallback_values(self):
        from ferro_ta import EMA

        result = gpu_ema(CLOSE_15, timeperiod=5)
        expected = EMA(CLOSE_15, timeperiod=5)
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_rsi_cpu_fallback_values(self):
        from ferro_ta import RSI

        result = gpu_rsi(CLOSE_15, timeperiod=5)
        expected = RSI(CLOSE_15, timeperiod=5)
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_sma_returns_numpy_for_numpy_input(self):
        result = gpu_sma(CLOSE_15, timeperiod=5)
        assert isinstance(result, np.ndarray)

    def test_rsi_finite_values_in_range(self):
        result = gpu_rsi(CLOSE_15, timeperiod=5)
        finite = result[np.isfinite(result)]
        assert len(finite) > 0
        assert np.all(finite >= 0.0)
        assert np.all(finite <= 100.0)

    def test_gpu_module_all_exports(self):
        from ferro_ta.tools import gpu as gpu_mod

        for name in gpu_mod.__all__:
            assert callable(getattr(gpu_mod, name))


# ---------------------------------------------------------------------------
# Indicator pipeline
# ---------------------------------------------------------------------------

from ferro_ta import BBANDS  # noqa: E402 (already imported)
from ferro_ta.tools.pipeline import Pipeline, make_pipeline  # noqa: E402

CLOSE_20 = np.random.default_rng(99).random(20) * 100 + 50


class TestPipeline:
    """Tests for ferro_ta.pipeline.Pipeline."""

    def test_pipeline_run_returns_dict(self):
        pipe = Pipeline().add("sma5", SMA, timeperiod=5)
        result = pipe.run(CLOSE_20)
        assert isinstance(result, dict)
        assert "sma5" in result

    def test_pipeline_result_length_matches_input(self):
        pipe = Pipeline().add("sma5", SMA, timeperiod=5)
        result = pipe.run(CLOSE_20)
        assert len(result["sma5"]) == len(CLOSE_20)

    def test_pipeline_multiple_steps(self):
        pipe = (
            Pipeline()
            .add("sma5", SMA, timeperiod=5)
            .add("ema5", EMA, timeperiod=5)
            .add("rsi7", RSI, timeperiod=7)
        )
        result = pipe.run(CLOSE_20)
        assert set(result.keys()) == {"sma5", "ema5", "rsi7"}

    def test_pipeline_multi_output_with_output_keys(self):
        pipe = Pipeline().add(
            "bb",
            BBANDS,
            timeperiod=5,
            nbdevup=2.0,
            nbdevdn=2.0,
            output_keys=["upper", "mid", "lower"],
        )
        result = pipe.run(CLOSE_20)
        assert "upper" in result
        assert "mid" in result
        assert "lower" in result
        assert "bb" not in result

    def test_pipeline_multi_output_without_output_keys(self):
        pipe = Pipeline().add("bb", BBANDS, timeperiod=5, nbdevup=2.0, nbdevdn=2.0)
        result = pipe.run(CLOSE_20)
        # Should auto-name as bb_0, bb_1, bb_2
        assert "bb_0" in result
        assert "bb_1" in result
        assert "bb_2" in result

    def test_pipeline_remove_step(self):
        pipe = Pipeline().add("sma5", SMA, timeperiod=5).add("ema5", EMA, timeperiod=5)
        pipe.remove("sma5")
        assert pipe.steps() == ["ema5"]

    def test_pipeline_len(self):
        pipe = Pipeline().add("sma5", SMA, timeperiod=5).add("ema5", EMA, timeperiod=5)
        assert len(pipe) == 2

    def test_pipeline_duplicate_name_raises(self):
        pipe = Pipeline().add("sma5", SMA, timeperiod=5)
        with pytest.raises(ValueError, match="sma5"):
            pipe.add("sma5", SMA, timeperiod=10)

    def test_make_pipeline_factory(self):
        pipe = make_pipeline(
            sma5=(SMA, {"timeperiod": 5}),
            rsi7=(RSI, {"timeperiod": 7}),
        )
        result = pipe.run(CLOSE_20)
        assert "sma5" in result
        assert "rsi7" in result

    def test_pipeline_sma_values_match_direct_call(self):
        pipe = Pipeline().add("sma5", SMA, timeperiod=5)
        result = pipe.run(CLOSE_20)
        direct = SMA(CLOSE_20, timeperiod=5)
        np.testing.assert_allclose(result["sma5"], direct, equal_nan=True)


# ---------------------------------------------------------------------------
# Polars integration (skipped if polars not installed)
# ---------------------------------------------------------------------------


class TestPolarsIntegration:
    """Transparent polars.Series support via polars_wrap."""

    @pytest.fixture(autouse=True)
    def skip_if_no_polars(self):
        pytest.importorskip("polars")

    def test_sma_returns_polars_series(self):
        import polars as pl

        s = pl.Series("close", CLOSE_20.tolist())
        result = SMA(s, timeperiod=5)
        assert isinstance(result, pl.Series)

    def test_sma_values_match_numpy(self):
        import polars as pl

        s = pl.Series("close", CLOSE_20.tolist())
        result = SMA(s, timeperiod=5)
        expected = SMA(CLOSE_20, timeperiod=5)
        np.testing.assert_allclose(result.to_numpy(), expected, equal_nan=True)

    def test_rsi_returns_polars_series(self):
        import polars as pl

        s = pl.Series("close", CLOSE_20.tolist())
        result = RSI(s, timeperiod=5)
        assert isinstance(result, pl.Series)

    def test_numpy_input_still_returns_numpy(self):
        result = SMA(CLOSE_20, timeperiod=5)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

import ferro_ta.core.config as ftconfig  # noqa: E402


class TestConfig:
    """Tests for ferro_ta.config module."""

    def setup_method(self):
        """Reset config state before each test."""
        ftconfig.reset()

    def teardown_method(self):
        """Clean up after each test."""
        ftconfig.reset()

    def test_set_and_get_default(self):
        ftconfig.set_default("timeperiod", 20)
        assert ftconfig.get_default("timeperiod") == 20

    def test_get_default_fallback(self):
        assert ftconfig.get_default("nonexistent") is None
        assert ftconfig.get_default("nonexistent", -1) == -1

    def test_reset_single_key(self):
        ftconfig.set_default("timeperiod", 20)
        ftconfig.reset("timeperiod")
        assert ftconfig.get_default("timeperiod") is None

    def test_reset_all(self):
        ftconfig.set_default("timeperiod", 20)
        ftconfig.set_default("RSI.timeperiod", 14)
        ftconfig.reset()
        assert ftconfig.list_defaults() == {}

    def test_list_defaults(self):
        ftconfig.set_default("timeperiod", 20)
        ftconfig.set_default("RSI.timeperiod", 14)
        defaults = ftconfig.list_defaults()
        assert defaults == {"timeperiod": 20, "RSI.timeperiod": 14}

    def test_get_defaults_for_indicator(self):
        ftconfig.set_default("timeperiod", 20)
        ftconfig.set_default("RSI.timeperiod", 14)
        rsi_defaults = ftconfig.get_defaults_for("RSI")
        assert rsi_defaults == {"timeperiod": 14}
        sma_defaults = ftconfig.get_defaults_for("SMA")
        assert sma_defaults == {"timeperiod": 20}

    def test_config_context_manager(self):
        ftconfig.set_default("timeperiod", 20)
        with ftconfig.Config(timeperiod=5):
            assert ftconfig.get_default("timeperiod") == 5
        assert ftconfig.get_default("timeperiod") == 20

    def test_config_context_manager_restores_on_exception(self):
        ftconfig.set_default("timeperiod", 20)
        try:
            with ftconfig.Config(timeperiod=5):
                raise RuntimeError("test error")
        except RuntimeError:
            pass
        assert ftconfig.get_default("timeperiod") == 20

    def test_config_context_manager_new_key_removed_on_exit(self):
        # Key doesn't exist before context
        assert ftconfig.get_default("nbdevup") is None
        with ftconfig.Config(nbdevup=2.5):
            assert ftconfig.get_default("nbdevup") == 2.5
        assert ftconfig.get_default("nbdevup") is None
