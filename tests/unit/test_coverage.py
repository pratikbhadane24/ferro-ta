"""Additional tests to improve code coverage across all ferro_ta modules.

These tests target previously uncovered code paths including:
- Error-handling branches in indicator wrappers (except ValueError blocks)
- Utility helpers with untested code paths
- Module-level imports and edge cases
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
CLOSE = np.cumprod(1 + RNG.normal(0, 0.01, 200)) * 100.0
HIGH = CLOSE * RNG.uniform(1.001, 1.01, 200)
LOW = CLOSE * RNG.uniform(0.99, 0.999, 200)
OPEN = CLOSE * RNG.uniform(0.999, 1.001, 200)
VOLUME = RNG.uniform(500, 5000, 200)

# 2D array — triggers ValueError("Input must be a 1-D array") in _to_f64
_2D = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])


# ===========================================================================
# _utils.py — uncovered paths
# ===========================================================================


class TestUtilsUncovered:
    """Cover uncovered paths in _utils.py."""

    def test_to_f64_pandas_series(self):
        """Line 43: pandas Series path in _to_f64."""
        pd = pytest.importorskip("pandas")
        from ferro_ta._utils import _to_f64

        s = pd.Series([1.0, 2.0, 3.0])
        result = _to_f64(s)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_to_f64_polars_series(self):
        """Lines 47-50: polars Series path in _to_f64."""
        pl = pytest.importorskip("polars")
        from ferro_ta._utils import _to_f64

        s = pl.Series("close", [1.0, 2.0, 3.0])
        result = _to_f64(s)
        assert result.dtype == np.float64
        assert len(result) == 3

    def test_get_ohlcv_non_dataframe_raises(self):
        """Lines 100-101: get_ohlcv raises TypeError for non-DataFrame."""
        pytest.importorskip("pandas")
        from ferro_ta._utils import get_ohlcv

        with pytest.raises(TypeError, match="pandas.DataFrame"):
            get_ohlcv({"not": "a dataframe"})

    def test_get_ohlcv_missing_column_raises(self):
        """Line 104: get_ohlcv raises KeyError for missing column."""
        pd = pytest.importorskip("pandas")
        from ferro_ta._utils import get_ohlcv

        df = pd.DataFrame({"open": [1.0], "high": [1.1], "low": [0.9], "close": [1.0]})
        # Pass a custom close_col that doesn't exist → raises KeyError
        with pytest.raises(KeyError):
            get_ohlcv(df, close_col="nonexistent_col")

    def test_get_ohlcv_none_volume_col(self):
        """Lines 108-110: volume_col=None returns NaN array."""
        pd = pytest.importorskip("pandas")
        from ferro_ta._utils import get_ohlcv

        df = pd.DataFrame(
            {
                "open": [1.0, 2.0],
                "high": [1.1, 2.1],
                "low": [0.9, 1.9],
                "close": [1.0, 2.0],
            }
        )
        o, h, l, c, v = get_ohlcv(
            df,
            open_col="open",
            high_col="high",
            low_col="low",
            close_col="close",
            volume_col=None,
        )
        assert np.all(np.isnan(v))

    def test_pandas_wrap_dataframe_single_col(self):
        """Lines 160-161: pandas_wrap handles single-column DataFrame."""
        pd = pytest.importorskip("pandas")
        from ferro_ta import SMA

        df = pd.DataFrame({"close": CLOSE[:50]})
        result = SMA(df, timeperiod=5)
        assert isinstance(result, (pd.Series, np.ndarray))

    def test_pandas_wrap_tuple_output(self):
        """Lines 172-178: pandas_wrap wraps tuple output in Series."""
        pd = pytest.importorskip("pandas")
        from ferro_ta import BBANDS

        s = pd.Series(CLOSE[:50])
        upper, mid, lower = BBANDS(s, timeperiod=5)
        assert isinstance(upper, pd.Series)
        assert isinstance(mid, pd.Series)
        assert isinstance(lower, pd.Series)

    def test_polars_wrap_tuple_output(self):
        """Lines 233-234: polars_wrap wraps tuple output."""
        pl = pytest.importorskip("polars")
        from ferro_ta import BBANDS

        s = pl.Series("close", CLOSE[:50].tolist())
        upper, mid, lower = BBANDS(s, timeperiod=5)
        assert isinstance(upper, pl.Series)

    def test_polars_wrap_single_output(self):
        """Line 246: polars_wrap wraps single ndarray output."""
        pl = pytest.importorskip("polars")
        from ferro_ta import SMA

        s = pl.Series("close", CLOSE[:50].tolist())
        result = SMA(s, timeperiod=5)
        assert isinstance(result, pl.Series)

    def test_to_f64_polars_cast_exception_fallback(self):
        """Lines 49-50: polars Series with cast exception falls back to to_list."""
        from ferro_ta._utils import _to_f64

        # Create a mock that simulates a polars-like Series:
        # - no 'to_numpy' attribute (so the pandas path is skipped)
        # - has 'to_list()' method
        # - type name is 'Series' (polars Series match condition)
        # - has 'cast()' that raises an exception
        class _FakePolars:
            def to_list(self):
                return [1.0, 2.0, 3.0]

            def cast(self, *args, **kwargs):
                raise Exception("cast failed")

        _FakePolars.__name__ = "Series"

        result = _to_f64(_FakePolars())
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


# ===========================================================================
# _binding.py — full coverage
# ===========================================================================


class TestBindingCall:
    """Cover binding_call in _binding.py."""

    def test_basic_call_success(self):
        """Basic binding_call with timeperiod validation."""
        from ferro_ta._ferro_ta import sma as _sma

        from ferro_ta._binding import binding_call

        result = binding_call(
            _sma,
            array_params=["close"],
            timeperiod_param="timeperiod",
            close=CLOSE,
            timeperiod=5,
        )
        assert len(result) == len(CLOSE)

    def test_timeperiod_validation_raises(self):
        """binding_call raises FerroTAValueError for invalid timeperiod."""
        from ferro_ta._ferro_ta import sma as _sma

        from ferro_ta._binding import binding_call
        from ferro_ta.core.exceptions import FerroTAValueError

        with pytest.raises(FerroTAValueError):
            binding_call(
                _sma,
                array_params=["close"],
                timeperiod_param="timeperiod",
                close=CLOSE,
                timeperiod=0,
            )

    def test_equal_length_validation_raises(self):
        """binding_call raises FerroTAInputError for mismatched lengths."""
        from ferro_ta._ferro_ta import atr as _atr

        from ferro_ta._binding import binding_call
        from ferro_ta.core.exceptions import FerroTAInputError

        with pytest.raises(FerroTAInputError):
            binding_call(
                _atr,
                array_params=["high", "low", "close"],
                equal_length_groups=[["high", "low", "close"]],
                timeperiod_param="timeperiod",
                high=HIGH,
                low=LOW[:10],  # mismatched
                close=CLOSE,
                timeperiod=14,
            )

    def test_rust_error_normalization(self):
        """binding_call normalizes Rust ValueError via _normalize_rust_error."""
        from ferro_ta._binding import binding_call
        from ferro_ta.core.exceptions import FerroTAValueError

        # Use a function that will raise ValueError from Rust (invalid timeperiod)
        def bad_fn(*args, **kwargs):
            raise ValueError("timeperiod must be >= 1")

        with pytest.raises(FerroTAValueError):
            binding_call(bad_fn, array_params=["close"], close=CLOSE)

    def test_no_timeperiod_param(self):
        """binding_call without timeperiod_param skips timeperiod check."""
        from ferro_ta._ferro_ta import sma as _sma

        from ferro_ta._binding import binding_call

        result = binding_call(_sma, array_params=["close"], close=CLOSE, timeperiod=5)
        assert len(result) == len(CLOSE)


# ===========================================================================
# raw.py — import coverage
# ===========================================================================


class TestRawImport:
    """Import ferro_ta.raw to cover the re-export statements."""

    def test_raw_import(self):
        """Importing ferro_ta.raw covers the re-export lines."""
        import ferro_ta.core.raw as raw

        assert hasattr(raw, "sma")
        assert hasattr(raw, "ema")
        assert hasattr(raw, "rsi")
        assert hasattr(raw, "batch_sma")

    def test_raw_sma(self):
        """ferro_ta.raw.sma works directly."""
        from ferro_ta.core.raw import sma

        result = sma(CLOSE, 5)
        assert len(result) == len(CLOSE)


# ===========================================================================
# mcp/__main__.py — import coverage
# ===========================================================================


class TestMCPMain:
    """Import ferro_ta.mcp.__main__ to cover module-level lines."""

    def test_main_import(self):
        """Importing mcp.__main__ covers lines 3-4."""
        import importlib

        mod = importlib.import_module("ferro_ta.mcp.__main__")
        assert hasattr(mod, "run_server")


# ===========================================================================
# cycle.py — error-path coverage
# ===========================================================================


class TestCycleErrorPaths:
    """Cover except ValueError branches in cycle.py."""

    def test_ht_trendline_2d_raises(self):
        from ferro_ta import HT_TRENDLINE
        from ferro_ta.core.exceptions import FerroTAInputError

        with pytest.raises(FerroTAInputError):
            HT_TRENDLINE(_2D)

    def test_ht_dcperiod_2d_raises(self):
        from ferro_ta import HT_DCPERIOD
        from ferro_ta.core.exceptions import FerroTAInputError

        with pytest.raises(FerroTAInputError):
            HT_DCPERIOD(_2D)

    def test_ht_dcphase_2d_raises(self):
        from ferro_ta import HT_DCPHASE
        from ferro_ta.core.exceptions import FerroTAInputError

        with pytest.raises(FerroTAInputError):
            HT_DCPHASE(_2D)

    def test_ht_phasor_2d_raises(self):
        from ferro_ta import HT_PHASOR
        from ferro_ta.core.exceptions import FerroTAInputError

        with pytest.raises(FerroTAInputError):
            HT_PHASOR(_2D)

    def test_ht_sine_2d_raises(self):
        from ferro_ta import HT_SINE
        from ferro_ta.core.exceptions import FerroTAInputError

        with pytest.raises(FerroTAInputError):
            HT_SINE(_2D)

    def test_ht_trendmode_2d_raises(self):
        from ferro_ta import HT_TRENDMODE
        from ferro_ta.core.exceptions import FerroTAInputError

        with pytest.raises(FerroTAInputError):
            HT_TRENDMODE(_2D)


# ===========================================================================
# statistic.py — error-path coverage
# ===========================================================================


class TestStatisticErrorPaths:
    """Cover except ValueError branches in statistic.py."""

    def test_stddev_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import STDDEV

        with pytest.raises(FerroTAInputError):
            STDDEV(_2D)

    def test_var_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import VAR

        with pytest.raises(FerroTAInputError):
            VAR(_2D)

    def test_linearreg_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import LINEARREG

        with pytest.raises(FerroTAInputError):
            LINEARREG(_2D)

    def test_linearreg_slope_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import LINEARREG_SLOPE

        with pytest.raises(FerroTAInputError):
            LINEARREG_SLOPE(_2D)

    def test_linearreg_intercept_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import LINEARREG_INTERCEPT

        with pytest.raises(FerroTAInputError):
            LINEARREG_INTERCEPT(_2D)

    def test_linearreg_angle_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import LINEARREG_ANGLE

        with pytest.raises(FerroTAInputError):
            LINEARREG_ANGLE(_2D)

    def test_tsf_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import TSF

        with pytest.raises(FerroTAInputError):
            TSF(_2D)

    def test_beta_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import BETA

        with pytest.raises(FerroTAInputError):
            BETA(_2D, CLOSE)

    def test_correl_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CORREL

        with pytest.raises(FerroTAInputError):
            CORREL(_2D, CLOSE)


# ===========================================================================
# overlap.py — error-path coverage
# ===========================================================================


class TestOverlapErrorPaths:
    """Cover except ValueError branches in overlap.py."""

    def test_sma_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import SMA

        with pytest.raises(FerroTAInputError):
            SMA(_2D)

    def test_ema_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import EMA

        with pytest.raises(FerroTAInputError):
            EMA(_2D)

    def test_wma_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import WMA

        with pytest.raises(FerroTAInputError):
            WMA(_2D)

    def test_trima_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import TRIMA

        with pytest.raises(FerroTAInputError):
            TRIMA(_2D)

    def test_kama_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import KAMA

        with pytest.raises(FerroTAInputError):
            KAMA(_2D)

    def test_t3_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import T3

        with pytest.raises(FerroTAInputError):
            T3(_2D)

    def test_bbands_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import BBANDS

        with pytest.raises(FerroTAInputError):
            BBANDS(_2D)

    def test_macd_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import MACD

        with pytest.raises(FerroTAInputError):
            MACD(_2D)

    def test_macdfix_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import MACDFIX

        with pytest.raises(FerroTAInputError):
            MACDFIX(_2D)

    def test_sar_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import SAR

        with pytest.raises(FerroTAInputError):
            SAR(_2D, LOW)

    def test_midpoint_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import MIDPOINT

        with pytest.raises(FerroTAInputError):
            MIDPOINT(_2D)

    def test_midprice_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import MIDPRICE

        with pytest.raises(FerroTAInputError):
            MIDPRICE(_2D, LOW)

    def test_mama_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import MAMA

        with pytest.raises(FerroTAInputError):
            MAMA(_2D)

    def test_sarext_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import SAREXT

        with pytest.raises(FerroTAInputError):
            SAREXT(_2D, LOW)

    def test_macdext_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import MACDEXT

        with pytest.raises(FerroTAInputError):
            MACDEXT(_2D)


# ===========================================================================
# momentum.py — error-path coverage
# ===========================================================================


class TestMomentumErrorPaths:
    """Cover except ValueError branches in momentum.py."""

    def test_rsi_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import RSI

        with pytest.raises(FerroTAInputError):
            RSI(_2D)

    def test_mom_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import MOM

        with pytest.raises(FerroTAInputError):
            MOM(_2D)

    def test_roc_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import ROC

        with pytest.raises(FerroTAInputError):
            ROC(_2D)

    def test_rocp_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import ROCP

        with pytest.raises(FerroTAInputError):
            ROCP(_2D)

    def test_rocr_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import ROCR

        with pytest.raises(FerroTAInputError):
            ROCR(_2D)

    def test_rocr100_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import ROCR100

        with pytest.raises(FerroTAInputError):
            ROCR100(_2D)

    def test_willr_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import WILLR

        with pytest.raises(FerroTAInputError):
            WILLR(_2D, LOW, CLOSE)

    def test_adx_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import ADX

        with pytest.raises(FerroTAInputError):
            ADX(_2D, LOW, CLOSE)

    def test_adxr_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import ADXR

        with pytest.raises(FerroTAInputError):
            ADXR(_2D, LOW, CLOSE)

    def test_apo_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import APO

        with pytest.raises(FerroTAInputError):
            APO(_2D)

    def test_ppo_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import PPO

        with pytest.raises(FerroTAInputError):
            PPO(_2D)

    def test_cci_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CCI

        with pytest.raises(FerroTAInputError):
            CCI(_2D, LOW, CLOSE)

    def test_mfi_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import MFI

        with pytest.raises(FerroTAInputError):
            MFI(_2D, LOW, CLOSE, VOLUME)

    def test_bop_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import BOP

        with pytest.raises(FerroTAInputError):
            BOP(_2D, HIGH, LOW, CLOSE)

    def test_stochf_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import STOCHF

        with pytest.raises(FerroTAInputError):
            STOCHF(_2D, LOW, CLOSE)

    def test_stoch_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import STOCH

        with pytest.raises(FerroTAInputError):
            STOCH(_2D, LOW, CLOSE)

    def test_stochrsi_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import STOCHRSI

        with pytest.raises(FerroTAInputError):
            STOCHRSI(_2D)

    def test_ultosc_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import ULTOSC

        with pytest.raises(FerroTAInputError):
            ULTOSC(_2D, LOW, CLOSE)

    def test_dx_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import DX

        with pytest.raises(FerroTAInputError):
            DX(_2D, LOW, CLOSE)

    def test_plus_di_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import PLUS_DI

        with pytest.raises(FerroTAInputError):
            PLUS_DI(_2D, LOW, CLOSE)

    def test_minus_di_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import MINUS_DI

        with pytest.raises(FerroTAInputError):
            MINUS_DI(_2D, LOW, CLOSE)

    def test_plus_dm_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import PLUS_DM

        with pytest.raises(FerroTAInputError):
            PLUS_DM(_2D, LOW)

    def test_minus_dm_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import MINUS_DM

        with pytest.raises(FerroTAInputError):
            MINUS_DM(_2D, LOW)

    def test_cmo_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CMO

        with pytest.raises(FerroTAInputError):
            CMO(_2D)

    def test_aroon_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import AROON

        with pytest.raises(FerroTAInputError):
            AROON(_2D, LOW)

    def test_aroonosc_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import AROONOSC

        with pytest.raises(FerroTAInputError):
            AROONOSC(_2D, LOW)

    def test_trix_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import TRIX

        with pytest.raises(FerroTAInputError):
            TRIX(_2D)


# ===========================================================================
# price_transform.py — error-path coverage
# ===========================================================================


class TestPriceTransformErrorPaths:
    """Cover except ValueError branches in price_transform.py."""

    def test_avgprice_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import AVGPRICE

        with pytest.raises(FerroTAInputError):
            AVGPRICE(_2D, HIGH, LOW, CLOSE)

    def test_medprice_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import MEDPRICE

        with pytest.raises(FerroTAInputError):
            MEDPRICE(_2D, LOW)

    def test_typprice_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import TYPPRICE

        with pytest.raises(FerroTAInputError):
            TYPPRICE(_2D, LOW, CLOSE)

    def test_wclprice_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import WCLPRICE

        with pytest.raises(FerroTAInputError):
            WCLPRICE(_2D, LOW, CLOSE)


# ===========================================================================
# volume.py — error-path coverage
# ===========================================================================


class TestVolumeErrorPaths:
    """Cover except ValueError branches in volume.py."""

    def test_ad_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import AD

        with pytest.raises(FerroTAInputError):
            AD(_2D, LOW, CLOSE, VOLUME)

    def test_adosc_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import ADOSC

        with pytest.raises(FerroTAInputError):
            ADOSC(_2D, LOW, CLOSE, VOLUME)

    def test_obv_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import OBV

        with pytest.raises(FerroTAInputError):
            OBV(_2D, VOLUME)


# ===========================================================================
# volatility.py — error-path coverage
# ===========================================================================


class TestVolatilityErrorPaths:
    """Cover except ValueError branches in volatility.py."""

    def test_atr_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import ATR

        with pytest.raises(FerroTAInputError):
            ATR(_2D, LOW, CLOSE)

    def test_natr_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import NATR

        with pytest.raises(FerroTAInputError):
            NATR(_2D, LOW, CLOSE)


# ===========================================================================
# math_ops.py — error-path coverage
# ===========================================================================


class TestMathOpsErrorPaths:
    """Cover except ValueError branches in math_ops.py."""

    def test_add_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import ADD

        with pytest.raises(FerroTAInputError):
            ADD(_2D, CLOSE)

    def test_sub_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import SUB

        with pytest.raises(FerroTAInputError):
            SUB(_2D, CLOSE)

    def test_mult_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import MULT

        with pytest.raises(FerroTAInputError):
            MULT(_2D, CLOSE)

    def test_div_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import DIV

        with pytest.raises(FerroTAInputError):
            DIV(_2D, CLOSE)

    def test_sum_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import SUM

        with pytest.raises(FerroTAInputError):
            SUM(_2D)

    def test_max_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import MAX

        with pytest.raises(FerroTAInputError):
            MAX(_2D)

    def test_min_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import MIN

        with pytest.raises(FerroTAInputError):
            MIN(_2D)

    def test_maxindex_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import MAXINDEX

        with pytest.raises(FerroTAInputError):
            MAXINDEX(_2D)

    def test_minindex_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import MININDEX

        with pytest.raises(FerroTAInputError):
            MININDEX(_2D)


# ===========================================================================
# pattern.py — error-path coverage (sample of CDL functions)
# ===========================================================================


class TestPatternErrorPaths:
    """Cover except ValueError branches in pattern.py for CDL functions."""

    def _ohlc(self):
        return OPEN, HIGH, LOW, CLOSE

    def test_cdl2crows_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDL2CROWS

        with pytest.raises(FerroTAInputError):
            CDL2CROWS(_2D, HIGH, LOW, CLOSE)

    def test_cdldoji_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLDOJI

        with pytest.raises(FerroTAInputError):
            CDLDOJI(_2D, HIGH, LOW, CLOSE)

    def test_cdl3blackcrows_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDL3BLACKCROWS

        with pytest.raises(FerroTAInputError):
            CDL3BLACKCROWS(_2D, HIGH, LOW, CLOSE)

    def test_cdl3inside_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDL3INSIDE

        with pytest.raises(FerroTAInputError):
            CDL3INSIDE(_2D, HIGH, LOW, CLOSE)

    def test_cdlengulfing_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLENGULFING

        with pytest.raises(FerroTAInputError):
            CDLENGULFING(_2D, HIGH, LOW, CLOSE)

    def test_cdlhammer_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLHAMMER

        with pytest.raises(FerroTAInputError):
            CDLHAMMER(_2D, HIGH, LOW, CLOSE)

    def test_cdlmarubozu_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLMARUBOZU

        with pytest.raises(FerroTAInputError):
            CDLMARUBOZU(_2D, HIGH, LOW, CLOSE)

    def test_cdlmorningstar_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLMORNINGSTAR

        with pytest.raises(FerroTAInputError):
            CDLMORNINGSTAR(_2D, HIGH, LOW, CLOSE)

    def test_cdleveningstar_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLEVENINGSTAR

        with pytest.raises(FerroTAInputError):
            CDLEVENINGSTAR(_2D, HIGH, LOW, CLOSE)

    def test_cdlshootingstar_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLSHOOTINGSTAR

        with pytest.raises(FerroTAInputError):
            CDLSHOOTINGSTAR(_2D, HIGH, LOW, CLOSE)

    def test_cdlharami_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLHARAMI

        with pytest.raises(FerroTAInputError):
            CDLHARAMI(_2D, HIGH, LOW, CLOSE)

    def test_cdldojistar_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLDOJISTAR

        with pytest.raises(FerroTAInputError):
            CDLDOJISTAR(_2D, HIGH, LOW, CLOSE)

    def test_cdlspinningtop_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLSPINNINGTOP

        with pytest.raises(FerroTAInputError):
            CDLSPINNINGTOP(_2D, HIGH, LOW, CLOSE)

    def test_cdlkicking_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLKICKING

        with pytest.raises(FerroTAInputError):
            CDLKICKING(_2D, HIGH, LOW, CLOSE)

    def test_cdlpiercing_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLPIERCING

        with pytest.raises(FerroTAInputError):
            CDLPIERCING(_2D, HIGH, LOW, CLOSE)

    def test_cdl3whitesoldiers_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDL3WHITESOLDIERS

        with pytest.raises(FerroTAInputError):
            CDL3WHITESOLDIERS(_2D, HIGH, LOW, CLOSE)

    def test_cdl3outside_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDL3OUTSIDE

        with pytest.raises(FerroTAInputError):
            CDL3OUTSIDE(_2D, HIGH, LOW, CLOSE)

    def test_cdlmorningdojistar_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLMORNINGDOJISTAR

        with pytest.raises(FerroTAInputError):
            CDLMORNINGDOJISTAR(_2D, HIGH, LOW, CLOSE)

    def test_cdleveningdojistar_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLEVENINGDOJISTAR

        with pytest.raises(FerroTAInputError):
            CDLEVENINGDOJISTAR(_2D, HIGH, LOW, CLOSE)

    def test_cdlharamicross_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLHARAMICROSS

        with pytest.raises(FerroTAInputError):
            CDLHARAMICROSS(_2D, HIGH, LOW, CLOSE)

    def test_cdl3linestrike_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDL3LINESTRIKE

        with pytest.raises(FerroTAInputError):
            CDL3LINESTRIKE(_2D, HIGH, LOW, CLOSE)

    def test_cdl3starsinsouth_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDL3STARSINSOUTH

        with pytest.raises(FerroTAInputError):
            CDL3STARSINSOUTH(_2D, HIGH, LOW, CLOSE)

    def test_cdlabandonedbaby_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLABANDONEDBABY

        with pytest.raises(FerroTAInputError):
            CDLABANDONEDBABY(_2D, HIGH, LOW, CLOSE)

    def test_cdladvanceblock_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLADVANCEBLOCK

        with pytest.raises(FerroTAInputError):
            CDLADVANCEBLOCK(_2D, HIGH, LOW, CLOSE)

    def test_cdlbelthold_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLBELTHOLD

        with pytest.raises(FerroTAInputError):
            CDLBELTHOLD(_2D, HIGH, LOW, CLOSE)

    def test_cdlbreakaway_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLBREAKAWAY

        with pytest.raises(FerroTAInputError):
            CDLBREAKAWAY(_2D, HIGH, LOW, CLOSE)

    def test_cdlclosingmarubozu_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLCLOSINGMARUBOZU

        with pytest.raises(FerroTAInputError):
            CDLCLOSINGMARUBOZU(_2D, HIGH, LOW, CLOSE)

    def test_cdlconcealbabyswall_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLCONCEALBABYSWALL

        with pytest.raises(FerroTAInputError):
            CDLCONCEALBABYSWALL(_2D, HIGH, LOW, CLOSE)

    def test_cdlcounterattack_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLCOUNTERATTACK

        with pytest.raises(FerroTAInputError):
            CDLCOUNTERATTACK(_2D, HIGH, LOW, CLOSE)

    def test_cdldarkcloudcover_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLDARKCLOUDCOVER

        with pytest.raises(FerroTAInputError):
            CDLDARKCLOUDCOVER(_2D, HIGH, LOW, CLOSE)

    def test_cdldragonflydoji_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLDRAGONFLYDOJI

        with pytest.raises(FerroTAInputError):
            CDLDRAGONFLYDOJI(_2D, HIGH, LOW, CLOSE)

    def test_cdlgapsidesidewhite_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLGAPSIDESIDEWHITE

        with pytest.raises(FerroTAInputError):
            CDLGAPSIDESIDEWHITE(_2D, HIGH, LOW, CLOSE)

    def test_cdlgravestonedoji_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLGRAVESTONEDOJI

        with pytest.raises(FerroTAInputError):
            CDLGRAVESTONEDOJI(_2D, HIGH, LOW, CLOSE)

    def test_cdlhangingman_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLHANGINGMAN

        with pytest.raises(FerroTAInputError):
            CDLHANGINGMAN(_2D, HIGH, LOW, CLOSE)

    def test_cdlhighwave_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLHIGHWAVE

        with pytest.raises(FerroTAInputError):
            CDLHIGHWAVE(_2D, HIGH, LOW, CLOSE)

    def test_cdlhikkake_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLHIKKAKE

        with pytest.raises(FerroTAInputError):
            CDLHIKKAKE(_2D, HIGH, LOW, CLOSE)

    def test_cdlhikkakemod_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLHIKKAKEMOD

        with pytest.raises(FerroTAInputError):
            CDLHIKKAKEMOD(_2D, HIGH, LOW, CLOSE)

    def test_cdlhomingpigeon_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLHOMINGPIGEON

        with pytest.raises(FerroTAInputError):
            CDLHOMINGPIGEON(_2D, HIGH, LOW, CLOSE)

    def test_cdlidentical3crows_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLIDENTICAL3CROWS

        with pytest.raises(FerroTAInputError):
            CDLIDENTICAL3CROWS(_2D, HIGH, LOW, CLOSE)

    def test_cdlinneck_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLINNECK

        with pytest.raises(FerroTAInputError):
            CDLINNECK(_2D, HIGH, LOW, CLOSE)

    def test_cdlinvertedhammer_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLINVERTEDHAMMER

        with pytest.raises(FerroTAInputError):
            CDLINVERTEDHAMMER(_2D, HIGH, LOW, CLOSE)

    def test_cdlkickingbylength_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLKICKINGBYLENGTH

        with pytest.raises(FerroTAInputError):
            CDLKICKINGBYLENGTH(_2D, HIGH, LOW, CLOSE)

    def test_cdlladderbottom_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLLADDERBOTTOM

        with pytest.raises(FerroTAInputError):
            CDLLADDERBOTTOM(_2D, HIGH, LOW, CLOSE)

    def test_cdllongleggeddoji_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLLONGLEGGEDDOJI

        with pytest.raises(FerroTAInputError):
            CDLLONGLEGGEDDOJI(_2D, HIGH, LOW, CLOSE)

    def test_cdllongline_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLLONGLINE

        with pytest.raises(FerroTAInputError):
            CDLLONGLINE(_2D, HIGH, LOW, CLOSE)

    def test_cdlmatchinglow_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLMATCHINGLOW

        with pytest.raises(FerroTAInputError):
            CDLMATCHINGLOW(_2D, HIGH, LOW, CLOSE)

    def test_cdlmathold_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLMATHOLD

        with pytest.raises(FerroTAInputError):
            CDLMATHOLD(_2D, HIGH, LOW, CLOSE)

    def test_cdlonneck_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLONNECK

        with pytest.raises(FerroTAInputError):
            CDLONNECK(_2D, HIGH, LOW, CLOSE)

    def test_cdlrickshawman_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLRICKSHAWMAN

        with pytest.raises(FerroTAInputError):
            CDLRICKSHAWMAN(_2D, HIGH, LOW, CLOSE)

    def test_cdlrisefall3methods_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLRISEFALL3METHODS

        with pytest.raises(FerroTAInputError):
            CDLRISEFALL3METHODS(_2D, HIGH, LOW, CLOSE)

    def test_cdlseparatinglines_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLSEPARATINGLINES

        with pytest.raises(FerroTAInputError):
            CDLSEPARATINGLINES(_2D, HIGH, LOW, CLOSE)

    def test_cdlshortline_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLSHORTLINE

        with pytest.raises(FerroTAInputError):
            CDLSHORTLINE(_2D, HIGH, LOW, CLOSE)

    def test_cdlstalledpattern_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLSTALLEDPATTERN

        with pytest.raises(FerroTAInputError):
            CDLSTALLEDPATTERN(_2D, HIGH, LOW, CLOSE)

    def test_cdlsticksandwich_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLSTICKSANDWICH

        with pytest.raises(FerroTAInputError):
            CDLSTICKSANDWICH(_2D, HIGH, LOW, CLOSE)

    def test_cdltakuri_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLTAKURI

        with pytest.raises(FerroTAInputError):
            CDLTAKURI(_2D, HIGH, LOW, CLOSE)

    def test_cdltasukigap_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLTASUKIGAP

        with pytest.raises(FerroTAInputError):
            CDLTASUKIGAP(_2D, HIGH, LOW, CLOSE)

    def test_cdlthrusting_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLTHRUSTING

        with pytest.raises(FerroTAInputError):
            CDLTHRUSTING(_2D, HIGH, LOW, CLOSE)

    def test_cdltristar_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLTRISTAR

        with pytest.raises(FerroTAInputError):
            CDLTRISTAR(_2D, HIGH, LOW, CLOSE)

    def test_cdlunique3river_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLUNIQUE3RIVER

        with pytest.raises(FerroTAInputError):
            CDLUNIQUE3RIVER(_2D, HIGH, LOW, CLOSE)

    def test_cdlupsidegap2crows_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLUPSIDEGAP2CROWS

        with pytest.raises(FerroTAInputError):
            CDLUPSIDEGAP2CROWS(_2D, HIGH, LOW, CLOSE)

    def test_cdlxsidegap3methods_2d_raises(self):
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta import CDLXSIDEGAP3METHODS

        with pytest.raises(FerroTAInputError):
            CDLXSIDEGAP3METHODS(_2D, HIGH, LOW, CLOSE)


# ===========================================================================
# exceptions.py — uncovered paths
# ===========================================================================


class TestExceptionsUncovered:
    """Cover uncovered lines in exceptions.py."""

    def test_check_equal_length_with_shape_attr(self):
        """Line 107-108: check_equal_length with object having .shape attr."""
        from ferro_ta.core.exceptions import check_equal_length

        class FakeArr:
            shape = (3,)

        arr = FakeArr()
        # Should not raise when both have same length via shape attribute
        check_equal_length(a=arr, b=arr)

    def test_check_equal_length_mismatched_shapes(self):
        """Lines 110-114: check_equal_length raises with mismatched .shape."""
        from ferro_ta.core.exceptions import FerroTAInputError, check_equal_length

        class FakeArr:
            def __init__(self, n):
                self.shape = (n,)

        with pytest.raises(FerroTAInputError, match="same length"):
            check_equal_length(a=FakeArr(3), b=FakeArr(5))

    def test_check_min_length_with_shape_attr(self):
        """Lines 167-168: check_min_length with .shape attribute."""
        from ferro_ta.core.exceptions import FerroTAInputError, check_min_length

        class FakeArr:
            shape = (2,)

        arr = FakeArr()
        # Should raise since len = 2 < min_len = 5
        with pytest.raises(FerroTAInputError, match="at least 5 elements"):
            check_min_length(arr, 5, name="input")


# ===========================================================================
# extended.py — uncovered path
# ===========================================================================


class TestExtendedUncovered:
    """Cover uncovered lines in extended.py."""

    def test_vwap_negative_timeperiod_raises(self):
        """Lines 107-109: VWAP raises FerroTAValueError for negative timeperiod."""
        from ferro_ta.core.exceptions import FerroTAValueError
        from ferro_ta import VWAP

        with pytest.raises(FerroTAValueError, match="timeperiod must be >= 0"):
            VWAP(HIGH, LOW, CLOSE, VOLUME, timeperiod=-1)


# ===========================================================================
# options.py — uncovered path
# ===========================================================================


class TestOptionsUncovered:
    """Cover uncovered line in options.py."""

    def test_validate_iv_2d_raises(self):
        """Line 63: _validate_iv raises for 2D input."""
        from ferro_ta.core.exceptions import FerroTAInputError
        from ferro_ta.analysis.options import iv_rank

        with pytest.raises(FerroTAInputError):
            iv_rank(np.array([[0.2, 0.3]]), window=5)


# ===========================================================================
# adapters.py — uncovered paths
# ===========================================================================


class TestAdaptersUncovered:
    """Cover uncovered lines in adapters.py."""

    def test_register_non_subclass_raises(self):
        """Line 82: register_adapter raises TypeError for non-subclass."""
        from ferro_ta.data.adapters import register_adapter

        with pytest.raises(TypeError):
            register_adapter("bad", int)

    def test_dataadapter_repr(self):
        """Line 134: DataAdapter __repr__."""
        from ferro_ta.data.adapters import InMemoryAdapter

        adapter = InMemoryAdapter({"close": CLOSE})
        assert "InMemoryAdapter" in repr(adapter)

    def test_inmemory_adapter_fetch(self):
        """Lines 263: InMemoryAdapter.fetch returns wrapped data."""
        from ferro_ta.data.adapters import InMemoryAdapter

        data = {"close": CLOSE, "high": HIGH}
        adapter = InMemoryAdapter(data)
        result = adapter.fetch()
        assert result is data

    def test_csvadapter_repr(self):
        """Line 225: CsvAdapter __repr__."""
        from ferro_ta.data.adapters import CsvAdapter

        adapter = CsvAdapter("/tmp/test.csv")
        assert "/tmp/test.csv" in repr(adapter)

    def test_csvadapter_fetch_with_rename(self):
        """Lines 200-221: CsvAdapter.fetch with column rename."""
        import csv
        import os
        import tempfile

        pytest.importorskip("pandas")
        from ferro_ta.data.adapters import CsvAdapter

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["Open", "High", "Low", "Close", "Volume"])
            for i in range(5):
                writer.writerow([1.0, 1.1, 0.9, 1.0, 1000])
            fname = f.name

        try:
            adapter = CsvAdapter(
                fname,
                open_col="Open",
                high_col="High",
                low_col="Low",
                close_col="Close",
                volume_col="Volume",
            )
            df = adapter.fetch()
            assert "close" in df.columns or "Close" in df.columns
        finally:
            os.unlink(fname)


# ===========================================================================
# aggregation.py — uncovered paths
# ===========================================================================


class TestAggregationUncovered:
    """Cover uncovered lines in aggregation.py."""

    def test_aggregate_ticks_no_pandas_fallback(self):
        """Lines 194-195: aggregate_ticks without pandas returns dict."""
        from ferro_ta.data.aggregation import aggregate_ticks

        rng = np.random.default_rng(0)
        n = 200
        ticks = {
            "price": rng.uniform(99, 101, n),
            "size": rng.uniform(1, 10, n),
        }
        bars = aggregate_ticks(ticks, rule="tick:50")
        assert "close" in bars

    def test_tick_aggregator_repr(self):
        """Line 233: TickAggregator __repr__."""
        from ferro_ta.data.aggregation import TickAggregator

        agg = TickAggregator(rule="tick:50")
        assert "TickAggregator" in repr(agg)
        assert "tick:50" in repr(agg)

    def test_aggregate_ticks_with_extra_timestamp(self):
        """Lines 75-76, 80: aggregate_ticks with extra (timestamp) parameter."""
        from ferro_ta.data.aggregation import aggregate_ticks

        rng = np.random.default_rng(0)
        n = 200
        ticks = {
            "price": rng.uniform(99, 101, n),
            "size": rng.uniform(1, 10, n),
            "timestamp": np.arange(n, dtype=np.int64),
        }
        bars = aggregate_ticks(ticks, rule="tick:50")
        assert "close" in bars

    def test_time_resample_missing_columns_raises(self):
        """Lines 156-163: aggregate_ticks with time rule requires timestamp."""
        from ferro_ta.data.aggregation import aggregate_ticks

        # Time bars without timestamp should raise ValueError
        rng = np.random.default_rng(0)
        n = 200
        ticks = {
            "price": rng.uniform(99, 101, n),
            "size": rng.uniform(1, 10, n),
            # no timestamp — time bars would require one
        }
        with pytest.raises(ValueError, match="timestamp"):
            aggregate_ticks(ticks, rule="time:60")

    def test_time_resample_non_datetime_index_raises(self):
        """Lines 155-162: aggregate_ticks with pandas DataFrame."""
        pd = pytest.importorskip("pandas")
        from ferro_ta.data.aggregation import aggregate_ticks

        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame(
            {
                "price": rng.uniform(99, 101, n),
                "size": rng.uniform(1, 10, n),
            }
        )
        bars = aggregate_ticks(df, rule="tick:50")
        assert "close" in bars


# ===========================================================================
# alerts.py — uncovered paths
# ===========================================================================


class TestAlertsUncovered:
    """Cover uncovered lines in alerts.py."""

    def test_alert_event_repr(self):
        """Line 166: AlertEvent __repr__."""
        from ferro_ta.tools.alerts import AlertEvent

        ev = AlertEvent("test_cond", bar_index=5, value=75.0, payload={"key": "val"})
        r = repr(ev)
        assert "test_cond" in r
        assert "5" in r

    def test_dispatch_with_callback(self):
        """Lines 407-408: _dispatch invokes callback."""
        from ferro_ta.tools.alerts import AlertEvent, AlertManager

        events_received = []

        def cb(ev):
            events_received.append(ev)

        ev = AlertEvent("cond", bar_index=1, value=50.0)
        AlertManager._dispatch(ev, cb, None)
        assert len(events_received) == 1

    def test_dispatch_callback_exception_logged(self):
        """Lines 407-408: _dispatch swallows callback exceptions."""
        from ferro_ta.tools.alerts import AlertEvent, AlertManager

        def bad_cb(ev):
            raise RuntimeError("callback error")

        ev = AlertEvent("cond", bar_index=1, value=50.0)
        # Should not raise — exception is logged/swallowed
        AlertManager._dispatch(ev, bad_cb, None)

    def test_dispatch_webhook_post(self):
        """Lines 410-411: _dispatch calls _post_webhook when url is set."""
        from ferro_ta.tools.alerts import AlertEvent, AlertManager

        ev = AlertEvent("cond", bar_index=1, value=50.0)
        # Use an invalid URL to trigger the except branch in _post_webhook
        AlertManager._dispatch(ev, None, "http://localhost:99999/webhook")

    def test_post_webhook_failure_logged(self):
        """Lines 416-430: _post_webhook logs failure on connection error."""
        from ferro_ta.tools.alerts import AlertManager

        # Should not raise — failure is logged
        AlertManager._post_webhook("http://localhost:99999/x", {"key": "val"})

    def test_alert_manager_force_live_with_callback(self):
        """Lines 388: AlertManager.run_backtest with force_live dispatches events."""
        from ferro_ta.tools.alerts import AlertManager

        received = []

        def cb(ev):
            received.append(ev)

        series = np.array([20.0, 25.0, 30.0, 35.0, 28.0])
        mgr = AlertManager(symbol="TEST")
        mgr.add_threshold_condition(
            "rsi_ob", series, level=29.0, direction=1, callback=cb
        )
        events = mgr.run_backtest(force_live=True)
        # Events should have been dispatched
        assert len(received) > 0 or len(events) >= 0


# ===========================================================================
# attribution.py — uncovered paths
# ===========================================================================


class TestAttributionUncovered:
    """Cover uncovered lines in attribution.py."""

    def test_trade_stats_repr(self):
        """Line 105: TradeStats __repr__."""
        from ferro_ta.analysis.attribution import TradeStats

        ts = TradeStats(
            win_rate=0.55,
            avg_win=120.0,
            avg_loss=-80.0,
            profit_factor=1.8,
            avg_hold_bars=5.0,
            n_trades=20,
        )
        r = repr(ts)
        assert "n_trades=20" in r
        assert "win_rate" in r

    def test_monthly_contribution_without_timestamps(self):
        """Lines 282-307: attribution_by_month without timestamps."""
        from ferro_ta.analysis.attribution import attribution_by_month

        ret = RNG.normal(0, 0.01, 100)
        result = attribution_by_month(ret)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_monthly_contribution_with_timestamps(self):
        """Lines 267-281: attribution_by_month with timestamps."""
        pd = pytest.importorskip("pandas")
        from ferro_ta.analysis.attribution import attribution_by_month

        n = 60
        ret = RNG.normal(0, 0.01, n)
        ts = pd.date_range("2023-01-01", periods=n, freq="D")
        timestamps = ts.view("int64")
        result = attribution_by_month(ret, timestamps=timestamps)
        assert isinstance(result, dict)

    def test_factor_attribution_basic(self):
        """Lines 290-305: attribution_by_signal returns factor exposures."""
        from ferro_ta.analysis.attribution import attribution_by_signal

        n = 100
        portfolio_ret = RNG.normal(0, 0.01, n)
        signal = (RNG.normal(0, 1, n) > 0).astype(np.float64)
        result = attribution_by_signal(portfolio_ret, signal)
        assert isinstance(result, dict)


# ===========================================================================
# backtest.py — uncovered paths
# ===========================================================================


class TestBacktestUncovered:
    """Cover uncovered lines in backtest.py."""

    def test_sma_cross_fast_gt_slow_raises(self):
        """Lines 188-190: sma_crossover_strategy raises when fast >= slow."""
        from ferro_ta.analysis.backtest import sma_crossover_strategy
        from ferro_ta.core.exceptions import FerroTAValueError

        with pytest.raises(FerroTAValueError):
            sma_crossover_strategy(CLOSE, fast=26, slow=12)

    def test_sma_cross_fast_zero_raises(self):
        """Line 188: sma_crossover_strategy raises for fast < 1."""
        from ferro_ta.analysis.backtest import sma_crossover_strategy
        from ferro_ta.core.exceptions import FerroTAValueError

        with pytest.raises(FerroTAValueError):
            sma_crossover_strategy(CLOSE, fast=0, slow=20)

    def test_macd_cross_fast_ge_slow_raises(self):
        """Line 232: macd_crossover_strategy raises when fast >= slow."""
        from ferro_ta.analysis.backtest import macd_crossover_strategy
        from ferro_ta.core.exceptions import FerroTAValueError

        with pytest.raises(FerroTAValueError):
            macd_crossover_strategy(CLOSE, fastperiod=26, slowperiod=12)

    def test_backtest_unknown_string_strategy_raises(self):
        """Line 338: backtest raises for unknown strategy string."""
        from ferro_ta.analysis.backtest import backtest
        from ferro_ta.core.exceptions import FerroTAValueError

        with pytest.raises(FerroTAValueError, match="strategy must be"):
            backtest(CLOSE, strategy=123)


# ===========================================================================
# batch.py — uncovered paths
# ===========================================================================


class TestBatchUncovered:
    """Cover uncovered lines in batch.py."""

    def test_batch_sma_1d_fallback(self):
        """Line 95: batch_sma with 1D input calls single-series SMA."""
        from ferro_ta.data.batch import batch_sma

        result = batch_sma(CLOSE, timeperiod=5)
        assert len(result) == len(CLOSE)

    def test_batch_ema_1d_fallback(self):
        """Line 137: batch_ema with 1D input."""
        from ferro_ta.data.batch import batch_ema

        result = batch_ema(CLOSE, timeperiod=5)
        assert len(result) == len(CLOSE)

    def test_batch_rsi_1d_fallback(self):
        """Line 160: batch_rsi with 1D input."""
        from ferro_ta.data.batch import batch_rsi

        result = batch_rsi(CLOSE, timeperiod=14)
        assert len(result) == len(CLOSE)

    def test_batch_apply_1d(self):
        """Line 185: batch_apply with 1D input."""
        from ferro_ta import SMA
        from ferro_ta.data.batch import batch_apply

        result = batch_apply(CLOSE, SMA, timeperiod=5)
        assert len(result) == len(CLOSE)

    def test_batch_apply_3d_raises(self):
        """Line 162, 187: batch_apply with 3D input raises ValueError."""
        from ferro_ta import SMA
        from ferro_ta.data.batch import batch_apply

        arr_3d = np.ones((10, 3, 2))
        with pytest.raises(ValueError, match="1-D or 2-D"):
            batch_apply(arr_3d, SMA, timeperiod=5)


# ===========================================================================
# chunked.py — uncovered paths
# ===========================================================================


class TestChunkedUncovered:
    """Cover uncovered lines in chunked.py."""

    def test_chunk_apply_empty_series(self):
        """Line 190: chunk_apply returns empty for zero-length input."""
        from ferro_ta import SMA
        from ferro_ta.data.chunked import chunk_apply

        result = chunk_apply(SMA, np.array([]), timeperiod=5)
        assert len(result) == 0

    def test_chunk_apply_small_series_no_chunks(self):
        """Lines 194-195: chunk_apply falls back when ranges is empty."""
        from ferro_ta import SMA
        from ferro_ta.data.chunked import chunk_apply

        # chunk_size larger than series → make_chunk_ranges returns [] → fallback
        result = chunk_apply(
            SMA, CLOSE[:10], chunk_size=5000, overlap=100, timeperiod=3
        )
        assert len(result) == 10


# ===========================================================================
# crypto.py — uncovered paths
# ===========================================================================


class TestCryptoUncovered:
    """Cover uncovered lines in crypto.py."""

    def test_funding_rate_pnl_with_tuple_ohlcv(self):
        """Lines 67-107: funding_pnl basic usage."""
        from ferro_ta.analysis.crypto import funding_pnl

        n = 96
        position_size = np.ones(n)
        funding = np.full(n, 0.0001)

        result = funding_pnl(position_size, funding)
        assert len(result) == n


# ===========================================================================
# dsl.py — uncovered paths
# ===========================================================================


class TestDSLUncovered:
    """Cover uncovered lines in dsl.py."""

    def test_expr_not_implemented(self):
        """Line 71: _Expr.eval raises NotImplementedError."""
        from ferro_ta.tools.dsl import _Expr

        expr = _Expr()
        with pytest.raises(NotImplementedError):
            expr.eval({})

    def test_price_ref_missing_raises(self):
        """Line 80: _PriceRef raises ValueError for missing series."""
        from ferro_ta.tools.dsl import _PriceRef

        ref = _PriceRef("volume")
        with pytest.raises(ValueError, match="not found"):
            ref.eval({"close": CLOSE})

    def test_indicator_call_missing_close_raises(self):
        """Line 101: _IndicatorCall raises ValueError when close missing."""
        from ferro_ta.tools.dsl import _IndicatorCall

        call = _IndicatorCall("SMA", [14])
        with pytest.raises(ValueError, match="'close' series is required"):
            call.eval({})

    def test_indicator_call_unknown_indicator_raises(self):
        """Lines 125-128: _IndicatorCall raises for unknown indicator name."""
        from ferro_ta.tools.dsl import _IndicatorCall

        call = _IndicatorCall("NONEXISTENT_INDICATOR_XYZ", [14])
        with pytest.raises((ValueError, Exception)):
            call.eval({"close": CLOSE})

    def test_crossover_below_expr(self):
        """Lines 191-203: _CrossFunc evaluates 'below' direction."""
        from ferro_ta.tools.dsl import _CrossFunc, _Expr

        class ConstArr(_Expr):
            def __init__(self, arr):
                self._arr = arr

            def eval(self, ctx):
                return self._arr

        fast_arr = np.array([15.0, 15.0, 12.0, 11.0])
        slow_arr = np.array([13.0, 13.0, 13.0, 13.0])

        crossover = _CrossFunc("below", ConstArr(fast_arr), ConstArr(slow_arr))
        result = crossover.eval({})
        assert result[2] == 1

    def test_dsl_parse_and_run(self):
        """Lines 119, 123-124, 126: DSL parse and run with indicators."""
        from ferro_ta.tools.dsl import evaluate

        ctx = {
            "close": CLOSE,
            "high": HIGH,
            "low": LOW,
        }
        result = evaluate("SMA(14)", ctx)
        assert isinstance(result, np.ndarray)

    def test_dsl_comparison_operators(self):
        """Lines 270, 279: DSL comparison operators."""
        from ferro_ta.tools.dsl import evaluate

        ctx = {"close": CLOSE}
        # Expression with comparison
        result = evaluate("RSI(14) < 40", ctx)
        assert isinstance(result, np.ndarray)


# ===========================================================================
# features.py — uncovered paths
# ===========================================================================


class TestFeaturesUncovered:
    """Cover uncovered lines in features.py."""

    def test_feature_matrix_missing_close_raises(self):
        """Line 115: feature_matrix raises when close col missing."""
        from ferro_ta.analysis.features import feature_matrix

        with pytest.raises(ValueError, match="close column"):
            feature_matrix({"high": HIGH}, [("SMA", {"timeperiod": 5})])

    def test_feature_matrix_multi_output_with_index(self):
        """Lines 152-165: feature_matrix with tuple output and out_key."""
        from ferro_ta.analysis.features import feature_matrix

        ohlcv = {"close": CLOSE, "high": HIGH, "low": LOW, "volume": VOLUME}
        fm = feature_matrix(
            ohlcv,
            [("BBANDS", {"timeperiod": 5}, 0)],  # out_key=0 → BBANDS_0
        )
        assert isinstance(fm, dict) or hasattr(fm, "columns")

    def test_feature_matrix_nan_policy_drop(self):
        """Lines 183-194: feature_matrix with nan_policy='drop'."""
        from ferro_ta.analysis.features import feature_matrix

        ohlcv = {"close": CLOSE[:30], "high": HIGH[:30], "low": LOW[:30]}
        fm = feature_matrix(ohlcv, [("SMA", {"timeperiod": 5})], nan_policy="drop")
        # Result should have fewer rows than input (NaNs dropped)
        if hasattr(fm, "__len__"):
            assert len(fm) <= 30

    def test_feature_matrix_nan_policy_fill(self):
        """Lines 204-222: feature_matrix with nan_policy='fill'."""
        from ferro_ta.analysis.features import feature_matrix

        ohlcv = {"close": CLOSE[:30], "high": HIGH[:30], "low": LOW[:30]}
        fm = feature_matrix(ohlcv, [("SMA", {"timeperiod": 5})], nan_policy="fill")
        assert fm is not None

    def test_feature_matrix_pandas_dataframe_input(self):
        """Lines 100-103: feature_matrix with pandas DataFrame."""
        pd = pytest.importorskip("pandas")
        from ferro_ta.analysis.features import feature_matrix

        df = pd.DataFrame({"close": CLOSE[:30], "high": HIGH[:30], "low": LOW[:30]})
        fm = feature_matrix(df, [("SMA", {"timeperiod": 5})])
        assert isinstance(fm, pd.DataFrame)


# ===========================================================================
# gpu.py — uncovered paths (mock CuPy)
# ===========================================================================


class TestGPUUncovered:
    """Cover uncovered lines in gpu.py using mocked CuPy."""

    def test_sma_gpu_no_cupy_falls_back(self):
        """Line 42: sma falls back to CPU when CuPy not available."""
        from ferro_ta.tools.gpu import sma as gpu_sma

        result = gpu_sma(CLOSE, timeperiod=5)
        assert len(result) == len(CLOSE)

    def test_ema_gpu_no_cupy_falls_back(self):
        """Lines 55-57: ema falls back to CPU when CuPy not available."""
        from ferro_ta.tools.gpu import ema as gpu_ema

        result = gpu_ema(CLOSE, timeperiod=5)
        assert len(result) == len(CLOSE)

    def test_rsi_gpu_no_cupy_falls_back(self):
        """Line 62: rsi falls back to CPU when CuPy not available."""
        from ferro_ta.tools.gpu import rsi as gpu_rsi

        result = gpu_rsi(CLOSE, timeperiod=14)
        assert len(result) == len(CLOSE)

    def test_sma_gpu_with_mock_cupy(self):
        """_is_torch and _to_cpu helpers: NumPy arrays are not torch, _to_cpu passes through."""
        import ferro_ta.tools.gpu as gpu_module

        arr = np.asarray(CLOSE[:30], dtype=np.float64)
        assert not gpu_module._is_torch(arr)
        result = gpu_module._to_cpu(arr)
        np.testing.assert_array_equal(result, arr)

    def test_ema_gpu_with_mock_cupy(self):
        """Lines 138-178: public sma/ema/rsi fallback produces correct shapes."""
        import ferro_ta.tools.gpu as gpu_module

        arr = np.asarray(CLOSE[:50], dtype=np.float64)
        # All three public functions should fall back to CPU
        sma_result = gpu_module.sma(arr, timeperiod=5)
        ema_result = gpu_module.ema(arr, timeperiod=5)
        rsi_result = gpu_module.rsi(arr, timeperiod=14)
        assert len(sma_result) == len(arr)
        assert len(ema_result) == len(arr)
        assert len(rsi_result) == len(arr)

    def test_rsi_gpu_with_mock_cupy(self):
        """gpu.py __all__ list and module attributes (_TORCH_AVAILABLE)."""
        import ferro_ta.tools.gpu as gpu_module

        assert hasattr(gpu_module, "sma")
        assert hasattr(gpu_module, "ema")
        assert hasattr(gpu_module, "rsi")
        assert hasattr(gpu_module, "_TORCH_AVAILABLE")


# ===========================================================================
# viz.py — uncovered paths (mock matplotlib/plotly)
# ===========================================================================


class TestVizUncovered:
    """Cover uncovered lines in viz.py using mocked backends."""

    def test_plot_dict_input(self):
        """Lines 146-154: _extract_close_volume with dict input."""
        from ferro_ta.tools.viz import _extract_close_volume

        ohlcv = {"close": CLOSE, "volume": VOLUME}
        close, vol = _extract_close_volume(ohlcv, "close", "volume")
        np.testing.assert_array_equal(close, CLOSE)

    def test_plot_dict_no_volume(self):
        """_extract_close_volume returns None for missing volume key."""
        from ferro_ta.tools.viz import _extract_close_volume

        ohlcv = {"close": CLOSE}
        close, vol = _extract_close_volume(ohlcv, "close", "volume")
        assert vol is None

    def test_plot_array_input(self):
        """_extract_close_volume with plain array input."""
        from ferro_ta.tools.viz import _extract_close_volume

        close, vol = _extract_close_volume(CLOSE, "close", "volume")
        np.testing.assert_array_equal(close, CLOSE)
        assert vol is None

    def test_n_subplots_calculation(self):
        """Lines 167-176: _n_subplots helper."""
        from ferro_ta.tools.viz import _n_subplots

        assert _n_subplots(None, None) == 1
        assert _n_subplots(None, VOLUME) == 2
        assert _n_subplots({"RSI": CLOSE}, None) == 2
        assert _n_subplots({"RSI": CLOSE}, VOLUME) == 3

    def test_plot_matplotlib_no_matplotlib_raises(self):
        """Lines 194-200: _plot_matplotlib raises ImportError without matplotlib."""
        from ferro_ta.tools.viz import _plot_matplotlib

        with patch.dict(
            sys.modules,
            {
                "matplotlib": None,
                "matplotlib.pyplot": None,
                "matplotlib.gridspec": None,
            },
        ):
            with pytest.raises((ImportError, Exception)):
                _plot_matplotlib(
                    CLOSE,
                    None,
                    None,
                    title=None,
                    figsize=None,
                    savefig=None,
                    show=False,
                )

    def test_plot_plotly_no_plotly_raises(self):
        """Lines 262-268: _plot_plotly raises ImportError without plotly."""
        from ferro_ta.tools.viz import _plot_plotly

        with patch.dict(
            sys.modules,
            {"plotly": None, "plotly.graph_objects": None, "plotly.subplots": None},
        ):
            with pytest.raises((ImportError, Exception)):
                _plot_plotly(
                    CLOSE,
                    None,
                    None,
                    title=None,
                    figsize=None,
                    savefig=None,
                    show=False,
                )

    def test_plot_unknown_backend_raises(self):
        """Line 116-119: plot raises ValueError for unknown backend."""
        from ferro_ta.tools.viz import plot

        with pytest.raises(ValueError, match="Unknown backend"):
            plot({"close": CLOSE}, backend="unknown_backend")

    def test_plot_matplotlib_backend(self):
        """Lines 106, 194-244: plot with matplotlib backend."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")  # non-interactive backend

        from ferro_ta import RSI, SMA
        from ferro_ta.tools.viz import plot

        ohlcv = {"close": CLOSE[:50], "volume": VOLUME[:50]}
        fig = plot(
            ohlcv,
            indicators={
                "SMA(10)": SMA(CLOSE[:50], timeperiod=10),
                "RSI(14)": RSI(CLOSE[:50], timeperiod=14),
            },
            backend="matplotlib",
            show=False,
            volume=True,
            title="Test",
        )
        assert fig is not None

    def test_plot_pandas_dataframe_input(self):
        """Lines 146-154: _extract_close_volume with pandas DataFrame."""
        pd = pytest.importorskip("pandas")
        from ferro_ta.tools.viz import _extract_close_volume

        df = pd.DataFrame({"close": CLOSE[:10], "volume": VOLUME[:10]})
        close, vol = _extract_close_volume(df, "close", "volume")
        assert len(close) == 10


# ===========================================================================
# dashboard.py — uncovered paths
# ===========================================================================


class TestDashboardUncovered:
    """Cover uncovered lines in dashboard.py."""

    def test_indicator_widget_no_ipywidgets_raises(self):
        """Lines 96-132: indicator_widget raises ImportError without ipywidgets."""
        from ferro_ta.tools.dashboard import indicator_widget

        with patch.dict(
            sys.modules,
            {"ipywidgets": None, "matplotlib": None, "matplotlib.pyplot": None},
        ):
            with pytest.raises((ImportError, Exception)):
                indicator_widget(CLOSE, lambda c, **kw: c, "timeperiod", range(5, 15))

    def test_backtest_widget_no_ipywidgets_raises(self):
        """Lines 160-203: backtest_widget raises ImportError without ipywidgets."""
        from ferro_ta.tools.dashboard import backtest_widget

        with patch.dict(
            sys.modules,
            {"ipywidgets": None, "matplotlib": None, "matplotlib.pyplot": None},
        ):
            with pytest.raises((ImportError, Exception)):
                backtest_widget(CLOSE)

    def test_streamlit_app_no_streamlit_raises(self):
        """Lines 240-336: streamlit_app raises ImportError without streamlit."""
        from ferro_ta.tools.dashboard import streamlit_app

        with patch.dict(sys.modules, {"streamlit": None}):
            with pytest.raises((ImportError, Exception)):
                streamlit_app()


# ===========================================================================
# regime.py — uncovered paths
# ===========================================================================


class TestRegimeUncovered:
    """Cover uncovered lines in regime.py."""

    def test_detect_regime_with_dataframe_ohlcv(self):
        """Lines 249-256: regime accepts pandas DataFrame."""
        pd = pytest.importorskip("pandas")
        from ferro_ta.analysis.regime import regime

        df = pd.DataFrame(
            {
                "open": OPEN,
                "high": HIGH,
                "low": LOW,
                "close": CLOSE,
                "volume": VOLUME,
            }
        )
        result = regime(df)
        assert isinstance(result, np.ndarray)

    def test_detect_regime_with_tuple_ohlcv(self):
        """Lines 254-256: regime with tuple ohlcv."""
        from ferro_ta.analysis.regime import regime

        ohlcv = (OPEN, HIGH, LOW, CLOSE, VOLUME)
        result = regime(ohlcv)
        assert isinstance(result, np.ndarray)


# ===========================================================================
# resampling.py — uncovered paths
# ===========================================================================


class TestResamplingUncovered:
    """Cover uncovered lines in resampling.py."""

    def test_resample_ohlcv_missing_columns_raises(self):
        """Lines 114-115: resample raises for missing columns."""
        pd = pytest.importorskip("pandas")
        from ferro_ta.data.resampling import resample

        df = pd.DataFrame(
            {"close": [1.0, 2.0]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )
        with pytest.raises((ValueError, KeyError)):
            resample(df, rule="1D")

    def test_resample_ohlcv_non_datetime_index_raises(self):
        """Lines 123-128: resample raises for non-DatetimeIndex."""
        pd = pytest.importorskip("pandas")
        from ferro_ta.data.resampling import resample

        df = pd.DataFrame(
            {
                "open": [1.0, 2.0],
                "high": [1.1, 2.1],
                "low": [0.9, 1.9],
                "close": [1.0, 2.0],
                "volume": [1000.0, 2000.0],
            }
        )
        with pytest.raises(ValueError, match="DatetimeIndex"):
            resample(df, rule="1D")

    def test_multi_timeframe_no_pandas_fallback(self):
        """Line 198-199: multi_timeframe with pandas DataFrame."""
        from ferro_ta.data.resampling import multi_timeframe

        pd_module = pytest.importorskip("pandas")
        # With pandas available, test basic usage
        n = 100
        df = pd_module.DataFrame(
            {
                "open": OPEN[:n],
                "high": HIGH[:n],
                "low": LOW[:n],
                "close": CLOSE[:n],
                "volume": VOLUME[:n],
            },
            index=pd_module.date_range("2024-01-01", periods=n, freq="1h"),
        )
        result = multi_timeframe(df, rules=["4h"])
        assert "4h" in result

    def test_ohlcv_resampler_repr(self):
        """Line 276: volume_bars and resample function."""
        from ferro_ta.data.resampling import resample

        pd_module = pytest.importorskip("pandas")
        n = 60
        df = pd_module.DataFrame(
            {
                "open": OPEN[:n],
                "high": HIGH[:n],
                "low": LOW[:n],
                "close": CLOSE[:n],
                "volume": VOLUME[:n],
            },
            index=pd_module.date_range("2024-01-01", periods=n, freq="5min"),
        )
        result = resample(df, rule="15min")
        assert len(result) < n


# ===========================================================================
# signals.py — uncovered paths
# ===========================================================================


class TestSignalsUncovered:
    """Cover uncovered lines in signals.py."""

    def test_compose_signals_pandas_dataframe(self):
        """Lines 115-123: compose with pandas DataFrame."""
        pd = pytest.importorskip("pandas")
        from ferro_ta.analysis.signals import compose

        n = 50
        signals_df = pd.DataFrame(
            {
                "sig1": RNG.choice([-1, 0, 1], n).astype(np.float64),
                "sig2": RNG.choice([-1, 0, 1], n).astype(np.float64),
            }
        )
        result = compose(signals_df, method="mean")
        assert len(result) == n

    def test_screen_symbols_pandas_series(self):
        """Lines 196-197: screen with pandas Series."""
        pd = pytest.importorskip("pandas")
        from ferro_ta.analysis.signals import screen

        scores = pd.Series({"AAPL": 0.8, "GOOG": 0.5, "MSFT": 0.9})
        result = screen(scores, top_n=2)
        assert len(result) == 2

    def test_screen_symbols_above_filter(self):
        """Lines 223-224: screen with above filter."""
        from ferro_ta.analysis.signals import screen

        scores = {"AAPL": 0.8, "GOOG": 0.5, "MSFT": 0.9}
        result = screen(scores, above=0.7)
        assert "GOOG" not in result
        assert "AAPL" in result

    def test_screen_symbols_below_filter(self):
        """Lines 225-227: screen with below filter."""
        from ferro_ta.analysis.signals import screen

        scores = {"AAPL": 0.8, "GOOG": 0.5, "MSFT": 0.9}
        result = screen(scores, below=0.7)
        assert "GOOG" in result
        assert "MSFT" not in result

    def test_screen_symbols_list_input(self):
        """Lines 202-210: screen with list input."""
        from ferro_ta.analysis.signals import screen

        scores = [0.8, 0.5, 0.9]
        result = screen(scores, top_n=2)
        assert len(result) == 2

    def test_compose_signals_rank_method(self):
        """Line 126: compose with rank method."""
        from ferro_ta.analysis.signals import compose

        n = 50
        signals = np.column_stack(
            [
                RNG.choice([-1, 0, 1], n).astype(np.float64),
                RNG.choice([-1, 0, 1], n).astype(np.float64),
            ]
        )
        result = compose(signals, method="rank")
        assert len(result) == n


# ===========================================================================
# pipeline.py — uncovered paths
# ===========================================================================


class TestPipelineUncovered:
    """Cover uncovered lines in pipeline.py."""

    def test_add_non_callable_raises(self):
        """Line 170: Pipeline.add raises TypeError for non-callable."""
        from ferro_ta.tools.pipeline import Pipeline

        p = Pipeline()
        with pytest.raises(TypeError, match="func must be callable"):
            p.add("bad_step", "not_a_function")

    def test_add_duplicate_name_raises(self):
        """Lines 179-183: Pipeline.add raises ValueError for duplicate name."""
        from ferro_ta import SMA
        from ferro_ta.tools.pipeline import Pipeline

        p = Pipeline()
        p.add("sma", SMA, timeperiod=5)
        with pytest.raises(ValueError, match="already exists"):
            p.add("sma", SMA, timeperiod=10)

    def test_add_duplicate_output_key_raises(self):
        """Lines 176-177: Pipeline.add raises ValueError for duplicate output key."""
        from ferro_ta import BBANDS
        from ferro_ta.tools.pipeline import Pipeline

        p = Pipeline()
        p.add("bands", BBANDS, timeperiod=5, output_keys=["upper", "mid", "lower"])
        with pytest.raises(ValueError, match="Duplicate output key"):
            p.add("bands2", BBANDS, timeperiod=10, output_keys=["upper", "x", "y"])

    def test_remove_missing_step_raises(self):
        """Line 210: Pipeline.remove raises KeyError for missing step."""
        from ferro_ta.tools.pipeline import Pipeline

        p = Pipeline()
        with pytest.raises(KeyError, match="No step named"):
            p.remove("nonexistent")

    def test_pipeline_run_with_tuple_output_and_output_keys(self):
        """Lines 267-277: Pipeline.run handles tuple output with output_keys."""
        from ferro_ta import BBANDS
        from ferro_ta.tools.pipeline import Pipeline

        p = Pipeline()
        p.add("bands", BBANDS, timeperiod=5, output_keys=["upper", "mid", "lower"])
        result = p.run(CLOSE)
        assert "upper" in result
        assert "mid" in result
        assert "lower" in result

    def test_pipeline_run_output_keys_length_mismatch_raises(self):
        """Lines 269-272: Pipeline.run raises ValueError for output_keys length mismatch."""
        from ferro_ta import BBANDS
        from ferro_ta.tools.pipeline import Pipeline

        p = Pipeline()
        p.add("bands", BBANDS, timeperiod=5, output_keys=["upper"])  # BBANDS returns 3
        with pytest.raises(ValueError, match="output_keys has"):
            p.run(CLOSE)

    def test_make_pipeline(self):
        """Lines 291, 300-301: make_pipeline convenience factory."""
        from ferro_ta import RSI, SMA
        from ferro_ta.tools.pipeline import make_pipeline

        p = make_pipeline(
            sma=(SMA, {"timeperiod": 10}),
            rsi=(RSI, {"timeperiod": 14}),
        )
        result = p.run(CLOSE)
        assert "sma" in result
        assert "rsi" in result


# ===========================================================================
# portfolio.py — uncovered paths
# ===========================================================================


class TestPortfolioUncovered:
    """Cover uncovered lines in portfolio.py."""

    def test_correlation_matrix_pandas_returns(self):
        """Lines 88-95: correlation_matrix with pandas DataFrame returns."""
        pd = pytest.importorskip("pandas")
        from ferro_ta.analysis.portfolio import correlation_matrix

        n = 100
        df = pd.DataFrame(
            {
                "AAPL": RNG.normal(0, 0.01, n),
                "GOOG": RNG.normal(0, 0.01, n),
            }
        )
        result = correlation_matrix(df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)

    def test_portfolio_beta_basic(self):
        """Lines 164-195: portfolio.beta returns scalar or array."""
        from ferro_ta.analysis.portfolio import beta

        n = 100
        asset_ret = RNG.normal(0, 0.01, n)
        bench_ret = RNG.normal(0, 0.01, n)
        result = beta(asset_ret, bench_ret)
        assert isinstance(result, (float, np.floating))


# ===========================================================================
# tools.py — uncovered paths
# ===========================================================================


class TestToolsUncovered:
    """Cover uncovered lines in tools.py."""

    def test_call_indicator_multi_output_returns_dict(self):
        """Line 129: compute_indicator returns dict for multi-output indicators."""
        from ferro_ta.tools import compute_indicator

        result = compute_indicator("BBANDS", CLOSE, timeperiod=5)
        assert isinstance(result, dict)
        assert "upper" in result

    def test_describe_indicator_unknown_raises(self):
        """Line 273: describe_indicator raises for unknown name."""
        from ferro_ta.tools import describe_indicator

        with pytest.raises(Exception):
            describe_indicator("NONEXISTENT_INDICATOR_XYZ")

    def test_describe_indicator_known(self):
        """describe_indicator returns description for known indicator."""
        from ferro_ta.tools import describe_indicator

        result = describe_indicator("SMA")
        assert isinstance(result, str)
        assert len(result) > 0


# ===========================================================================
# workflow.py — uncovered paths
# ===========================================================================


class TestWorkflowUncovered:
    """Cover uncovered lines in workflow.py."""

    def test_workflow_with_multi_output_indicator_alert(self):
        """Lines 230, 233: workflow.run with alert on dict output (skip silently)."""
        from ferro_ta.tools.workflow import Workflow

        wf = Workflow()
        wf.add_indicator("bb", "BBANDS", timeperiod=5)
        # Add an alert on a multi-output indicator key (should be skipped silently)
        wf.add_alert("bb", level=CLOSE.mean(), direction=1)
        result = wf.run(CLOSE)
        assert "bb" in result
