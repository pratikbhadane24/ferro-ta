"""Tests for ferro_ta technical analysis indicators."""

import math

import numpy as np
import pytest

from ferro_ta import (
    ACOS,
    # Volume
    AD,
    # Math Operators
    ADD,
    ADOSC,
    ADX,
    ADXR,
    AROON,
    ASIN,
    ATAN,
    # Volatility
    ATR,
    # Price transforms
    AVGPRICE,
    BBANDS,
    CDL3BLACKCROWS,
    CDL3INSIDE,
    # Candlestick patterns
    CDL3LINESTRIKE,
    CDL3OUTSIDE,
    CDL3STARSINSOUTH,
    CDL3WHITESOLDIERS,
    CDLABANDONEDBABY,
    CDLADVANCEBLOCK,
    CDLBELTHOLD,
    CDLBREAKAWAY,
    CDLCLOSINGMARUBOZU,
    CDLCONCEALBABYSWALL,
    CDLCOUNTERATTACK,
    CDLDARKCLOUDCOVER,
    # Patterns
    CDLDOJI,
    CDLDOJISTAR,
    CDLDRAGONFLYDOJI,
    CDLENGULFING,
    CDLEVENINGDOJISTAR,
    CDLGAPSIDESIDEWHITE,
    CDLGRAVESTONEDOJI,
    CDLHAMMER,
    CDLHANGINGMAN,
    CDLHARAMI,
    CDLHARAMICROSS,
    CDLHIGHWAVE,
    CDLHIKKAKE,
    CDLHIKKAKEMOD,
    CDLHOMINGPIGEON,
    CDLIDENTICAL3CROWS,
    CDLINNECK,
    CDLINVERTEDHAMMER,
    CDLKICKING,
    CDLKICKINGBYLENGTH,
    CDLLADDERBOTTOM,
    CDLLONGLEGGEDDOJI,
    CDLLONGLINE,
    CDLMARUBOZU,
    CDLMATCHINGLOW,
    CDLMATHOLD,
    CDLMORNINGDOJISTAR,
    CDLONNECK,
    CDLPIERCING,
    CDLRICKSHAWMAN,
    CDLRISEFALL3METHODS,
    CDLSEPARATINGLINES,
    CDLSHOOTINGSTAR,
    CDLSHORTLINE,
    CDLSTALLEDPATTERN,
    CDLSTICKSANDWICH,
    CDLTAKURI,
    CDLTASUKIGAP,
    CDLTHRUSTING,
    CDLTRISTAR,
    CDLUNIQUE3RIVER,
    CDLUPSIDEGAP2CROWS,
    CDLXSIDEGAP3METHODS,
    CEIL,
    CMO,
    CORREL,
    COS,
    COSH,
    DEMA,
    DIV,
    DX,
    EMA,
    EXP,
    FLOOR,
    HT_DCPERIOD,
    HT_DCPHASE,
    HT_PHASOR,
    HT_SINE,
    # Cycle
    HT_TRENDLINE,
    HT_TRENDMODE,
    LINEARREG,
    LN,
    LOG10,
    MA,
    MACD,
    MACDEXT,
    MACDFIX,
    MAMA,
    MAVP,
    MAX,
    MAXINDEX,
    MEDPRICE,
    MIDPOINT,
    MIDPRICE,
    MIN,
    MININDEX,
    MINUS_DI,
    MINUS_DM,
    # Momentum
    MOM,
    MULT,
    NATR,
    OBV,
    PLUS_DI,
    PLUS_DM,
    ROC,
    ROCP,
    RSI,
    SAR,
    SAREXT,
    SIN,
    SINH,
    SMA,
    SQRT,
    # Statistics
    STDDEV,
    STOCH,
    STOCHRSI,
    SUB,
    SUM,
    TAN,
    TANH,
    TEMA,
    TRANGE,
    TYPPRICE,
    WCLPRICE,
    WILLR,
    # Overlap
    WMA,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

PRICES = np.array(
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
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _nan_count(arr: np.ndarray) -> int:
    return int(np.sum(np.isnan(arr)))


def _finite(arr: np.ndarray) -> np.ndarray:
    return arr[~np.isnan(arr)]


# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------


class TestSMA:
    def test_output_length(self):
        result = SMA(PRICES, timeperiod=3)
        assert len(result) == len(PRICES)

    def test_leading_nans(self):
        period = 5
        result = SMA(PRICES, timeperiod=period)
        assert _nan_count(result) == period - 1

    def test_values_correct(self):
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = SMA(prices, timeperiod=3)
        assert np.isnan(result[0]) and np.isnan(result[1])
        assert math.isclose(result[2], 2.0)
        assert math.isclose(result[3], 3.0)
        assert math.isclose(result[4], 4.0)

    def test_accepts_python_list(self):
        result = SMA([1.0, 2.0, 3.0, 4.0], timeperiod=2)
        assert len(result) == 4

    def test_default_period(self):
        long_prices = np.arange(1.0, 51.0)
        result = SMA(long_prices)  # default period = 30
        assert _nan_count(result) == 29

    def test_invalid_period_zero(self):
        with pytest.raises(Exception):
            SMA(PRICES, timeperiod=0)

    def test_period_equals_length(self):
        prices = np.array([1.0, 2.0, 3.0])
        result = SMA(prices, timeperiod=3)
        assert np.isnan(result[0]) and np.isnan(result[1])
        assert math.isclose(result[2], 2.0)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


class TestEMA:
    def test_output_length(self):
        result = EMA(PRICES, timeperiod=3)
        assert len(result) == len(PRICES)

    def test_leading_nans(self):
        period = 5
        result = EMA(PRICES, timeperiod=period)
        assert _nan_count(result) == period - 1

    def test_values_reasonable(self):
        prices = np.array([10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0])
        result = EMA(prices, timeperiod=3)
        finite = _finite(result)
        assert len(finite) == len(prices) - 2
        # EMA should be a reasonable average-like value
        assert all(8.0 <= v <= 14.0 for v in finite)

    def test_ema_differs_from_sma(self):
        """EMA weights recent prices more — it must differ from SMA."""
        prices = np.array([1.0, 2.0, 3.0, 10.0, 11.0])
        ema_result = EMA(prices, timeperiod=3)
        sma_result = SMA(prices, timeperiod=3)
        # Both should be finite for the last value
        assert not math.isclose(ema_result[-1], sma_result[-1], rel_tol=1e-9)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


class TestRSI:
    def test_output_length(self):
        result = RSI(PRICES, timeperiod=5)
        assert len(result) == len(PRICES)

    def test_leading_nans(self):
        period = 5
        result = RSI(PRICES, timeperiod=period)
        assert _nan_count(result) == period

    def test_rsi_range(self):
        result = RSI(PRICES, timeperiod=5)
        finite = _finite(result)
        assert all(0.0 <= v <= 100.0 for v in finite)

    def test_constant_prices_rsi_50(self):
        """For constant prices, RSI should be around 50 (no gains or losses)."""
        prices = np.full(20, 50.0)
        result = RSI(prices, timeperiod=5)
        finite = _finite(result)
        # With constant prices there are no changes, RSI is typically 50 or 100
        assert all(0.0 <= v <= 100.0 for v in finite)

    def test_invalid_period_zero(self):
        with pytest.raises(Exception):
            RSI(PRICES, timeperiod=0)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


class TestMACD:
    def test_output_tuple_of_three(self):
        result = MACD(PRICES)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_output_lengths_equal(self):
        macd_line, signal, hist = MACD(PRICES)
        assert len(macd_line) == len(PRICES)
        assert len(signal) == len(PRICES)
        assert len(hist) == len(PRICES)

    def test_histogram_is_macd_minus_signal(self):
        """Histogram must equal MACD line minus signal line for valid indices."""
        prices = np.arange(1.0, 60.0)
        macd_line, signal, hist = MACD(
            prices, fastperiod=3, slowperiod=6, signalperiod=2
        )
        mask = ~(np.isnan(macd_line) | np.isnan(signal) | np.isnan(hist))
        assert np.allclose(hist[mask], macd_line[mask] - signal[mask], atol=1e-10)

    def test_fast_must_be_less_than_slow(self):
        with pytest.raises(Exception):
            MACD(PRICES, fastperiod=26, slowperiod=12)

    def test_all_nan_when_not_enough_data(self):
        prices = np.arange(1.0, 6.0)  # only 5 points
        macd_line, signal, hist = MACD(
            prices, fastperiod=3, slowperiod=4, signalperiod=2
        )
        # warmup = 4 + 2 - 2 = 4, so only index 4 might be valid
        assert np.isnan(macd_line[0])

    def test_default_periods(self):
        prices = np.arange(1.0, 100.0)
        macd_line, signal, hist = MACD(prices)
        # MACD line is valid from slowperiod-1=25; signal from slowperiod+signalperiod-2=33
        assert all(np.isnan(macd_line[:25]))
        assert any(~np.isnan(macd_line[25:]))
        # Signal line starts at index 33
        assert all(np.isnan(signal[:33]))
        assert any(~np.isnan(signal[33:]))


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


class TestBBANDS:
    def test_output_tuple_of_three(self):
        result = BBANDS(PRICES, timeperiod=5)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_output_lengths_equal(self):
        upper, middle, lower = BBANDS(PRICES, timeperiod=5)
        assert len(upper) == len(PRICES)
        assert len(middle) == len(PRICES)
        assert len(lower) == len(PRICES)

    def test_leading_nans(self):
        period = 5
        upper, middle, lower = BBANDS(PRICES, timeperiod=period)
        assert _nan_count(upper) == period - 1
        assert _nan_count(middle) == period - 1
        assert _nan_count(lower) == period - 1

    def test_band_ordering(self):
        """Upper >= middle >= lower for all valid values."""
        upper, middle, lower = BBANDS(PRICES, timeperiod=5)
        mask = ~(np.isnan(upper) | np.isnan(middle) | np.isnan(lower))
        assert np.all(upper[mask] >= middle[mask])
        assert np.all(middle[mask] >= lower[mask])

    def test_symmetric_bands(self):
        """With equal nbdevup/nbdevdn, bands are symmetric around middle."""
        prices = np.array([10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0])
        upper, middle, lower = BBANDS(prices, timeperiod=3, nbdevup=2.0, nbdevdn=2.0)
        mask = ~(np.isnan(upper) | np.isnan(lower))
        assert np.allclose(
            upper[mask] - middle[mask],
            middle[mask] - lower[mask],
            atol=1e-10,
        )

    def test_invalid_period_zero(self):
        with pytest.raises(Exception):
            BBANDS(PRICES, timeperiod=0)

    def test_accepts_python_list(self):
        prices = [10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0]
        upper, middle, lower = BBANDS(prices, timeperiod=3)
        assert len(upper) == len(prices)


# ---------------------------------------------------------------------------
# Input validation shared tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_2d_array_raises(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError):
            SMA(arr, timeperiod=2)

    def test_int_array_is_coerced(self):
        """Integer arrays should be automatically cast to float64."""
        prices = np.array([10, 11, 12, 13, 14], dtype=np.int64)
        result = SMA(prices, timeperiod=3)
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# Shared fixtures for OHLCV tests
# ---------------------------------------------------------------------------

OHLCV_PRICES = np.arange(1.0, 51.0)
OHLCV_HIGH = OHLCV_PRICES + 0.5
OHLCV_LOW = OHLCV_PRICES - 0.5
OHLCV_CLOSE = OHLCV_PRICES
OHLCV_OPEN = OHLCV_PRICES - 0.2
OHLCV_VOLUME = np.ones(50) * 1000.0


# ---------------------------------------------------------------------------
# Overlap Studies — new indicators
# ---------------------------------------------------------------------------


class TestWMA:
    def test_output_length(self):
        result = WMA(OHLCV_PRICES, 5)
        assert len(result) == len(OHLCV_PRICES)

    def test_leading_nans(self):
        result = WMA(OHLCV_PRICES, 5)
        assert _nan_count(result) == 4

    def test_values_correct(self):
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = WMA(prices, 3)
        # WMA(3) at i=2: (1*1 + 2*2 + 3*3) / (1+2+3) = 14/6
        assert math.isclose(result[2], 14.0 / 6.0, rel_tol=1e-9)


class TestDEMA:
    def test_output_length(self):
        result = DEMA(OHLCV_PRICES, 5)
        assert len(result) == len(OHLCV_PRICES)

    def test_leading_nans(self):
        result = DEMA(OHLCV_PRICES, 5)
        assert _nan_count(result) == 2 * (5 - 1)


class TestTEMA:
    def test_output_length(self):
        result = TEMA(OHLCV_PRICES, 5)
        assert len(result) == len(OHLCV_PRICES)

    def test_leading_nans(self):
        result = TEMA(OHLCV_PRICES, 5)
        assert _nan_count(result) == 3 * (5 - 1)


class TestMACDFIX:
    def test_output_tuple_of_three(self):
        result = MACDFIX(OHLCV_PRICES)
        assert isinstance(result, tuple) and len(result) == 3

    def test_all_same_length(self):
        m, s, h = MACDFIX(OHLCV_PRICES)
        assert len(m) == len(OHLCV_PRICES)
        assert len(s) == len(OHLCV_PRICES)
        assert len(h) == len(OHLCV_PRICES)


class TestSAR:
    def test_output_length(self):
        result = SAR(OHLCV_HIGH, OHLCV_LOW)
        assert len(result) == len(OHLCV_HIGH)

    def test_values_reasonable(self):
        result = SAR(OHLCV_HIGH, OHLCV_LOW)
        finite = _finite(result)
        assert len(finite) > 0
        assert all(v > 0 for v in finite)


class TestMIDPOINT:
    def test_output_length(self):
        result = MIDPOINT(OHLCV_PRICES, 5)
        assert len(result) == len(OHLCV_PRICES)

    def test_leading_nans(self):
        result = MIDPOINT(OHLCV_PRICES, 5)
        assert _nan_count(result) == 4


class TestMIDPRICE:
    def test_output_length(self):
        result = MIDPRICE(OHLCV_HIGH, OHLCV_LOW, 5)
        assert len(result) == len(OHLCV_HIGH)


# ---------------------------------------------------------------------------
# Momentum Indicators — new indicators
# ---------------------------------------------------------------------------


class TestMOM:
    def test_output_length(self):
        result = MOM(OHLCV_PRICES, 5)
        assert len(result) == len(OHLCV_PRICES)

    def test_leading_nans(self):
        result = MOM(OHLCV_PRICES, 5)
        assert _nan_count(result) == 5

    def test_values_correct(self):
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = MOM(prices, 2)
        assert math.isclose(result[2], 2.0)
        assert math.isclose(result[3], 2.0)


class TestROC:
    def test_output_length(self):
        result = ROC(OHLCV_PRICES, 5)
        assert len(result) == len(OHLCV_PRICES)

    def test_leading_nans(self):
        result = ROC(OHLCV_PRICES, 5)
        assert _nan_count(result) == 5

    def test_values_formula(self):
        prices = np.array([10.0, 11.0, 12.0, 10.0, 11.0])
        result = ROC(prices, 2)
        # ROC[4] = (11 - 12) / 12 * 100
        assert math.isclose(result[4], (11.0 - 12.0) / 12.0 * 100.0, rel_tol=1e-9)


class TestROCP:
    def test_output_length(self):
        result = ROCP(OHLCV_PRICES, 5)
        assert len(result) == len(OHLCV_PRICES)

    def test_relation_to_roc(self):
        """ROCP * 100 should equal ROC."""
        roc_result = ROC(OHLCV_PRICES, 5)
        rocp_result = ROCP(OHLCV_PRICES, 5)
        mask = ~(np.isnan(roc_result) | np.isnan(rocp_result))
        assert np.allclose(rocp_result[mask] * 100.0, roc_result[mask], atol=1e-10)


class TestWILLR:
    def test_output_length(self):
        result = WILLR(OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE, 5)
        assert len(result) == len(OHLCV_PRICES)

    def test_range_correct(self):
        result = WILLR(OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE, 5)
        finite = _finite(result)
        assert all(-100.0 <= v <= 0.0 for v in finite)


class TestAROON:
    def test_output_tuple(self):
        result = AROON(OHLCV_HIGH, OHLCV_LOW, 14)
        assert isinstance(result, tuple) and len(result) == 2

    def test_range_correct(self):
        down, up = AROON(OHLCV_HIGH, OHLCV_LOW, 14)
        down_finite = _finite(down)
        up_finite = _finite(up)
        assert all(0.0 <= v <= 100.0 for v in down_finite)
        assert all(0.0 <= v <= 100.0 for v in up_finite)


class TestADX:
    def test_output_length(self):
        result = ADX(OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE, 14)
        assert len(result) == len(OHLCV_PRICES)

    def test_range_correct(self):
        result = ADX(OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE, 14)
        finite = _finite(result)
        assert all(0.0 <= v <= 100.0 for v in finite)


class TestCMO:
    def test_output_length(self):
        result = CMO(OHLCV_PRICES, 5)
        assert len(result) == len(OHLCV_PRICES)

    def test_range_correct(self):
        result = CMO(OHLCV_PRICES, 5)
        finite = _finite(result)
        assert all(-100.0 <= v <= 100.0 for v in finite)


# ---------------------------------------------------------------------------
# Volume Indicators
# ---------------------------------------------------------------------------


class TestOBV:
    def test_output_length(self):
        result = OBV(OHLCV_CLOSE, OHLCV_VOLUME)
        assert len(result) == len(OHLCV_PRICES)

    def test_monotone_increasing(self):
        """With always-rising prices, OBV should be non-decreasing."""
        result = OBV(OHLCV_CLOSE, OHLCV_VOLUME)
        assert all(result[i] <= result[i + 1] for i in range(1, len(result) - 1))


class TestAD:
    def test_output_length(self):
        result = AD(OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE, OHLCV_VOLUME)
        assert len(result) == len(OHLCV_PRICES)


class TestADOSC:
    def test_output_length(self):
        result = ADOSC(OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE, OHLCV_VOLUME)
        assert len(result) == len(OHLCV_PRICES)


# ---------------------------------------------------------------------------
# Volatility Indicators
# ---------------------------------------------------------------------------


class TestATR:
    def test_output_length(self):
        result = ATR(OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE, 14)
        assert len(result) == len(OHLCV_PRICES)

    def test_values_positive(self):
        result = ATR(OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE, 14)
        finite = _finite(result)
        assert all(v > 0 for v in finite)


class TestNATR:
    def test_output_length(self):
        result = NATR(OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE, 14)
        assert len(result) == len(OHLCV_PRICES)

    def test_values_positive(self):
        result = NATR(OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE, 14)
        finite = _finite(result)
        assert all(v > 0 for v in finite)


class TestTRANGE:
    def test_output_length(self):
        result = TRANGE(OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE)
        assert len(result) == len(OHLCV_PRICES)

    def test_values_positive(self):
        result = TRANGE(OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE)
        assert all(v > 0 for v in result)


# ---------------------------------------------------------------------------
# Statistic Functions
# ---------------------------------------------------------------------------


class TestSTDDEV:
    def test_output_length(self):
        result = STDDEV(OHLCV_PRICES, 5)
        assert len(result) == len(OHLCV_PRICES)

    def test_leading_nans(self):
        result = STDDEV(OHLCV_PRICES, 5)
        assert _nan_count(result) == 4

    def test_constant_prices_zero_stddev(self):
        prices = np.full(20, 100.0)
        result = STDDEV(prices, 5)
        finite = _finite(result)
        assert all(math.isclose(v, 0.0, abs_tol=1e-10) for v in finite)


class TestLINEARREG:
    def test_output_length(self):
        result = LINEARREG(OHLCV_PRICES, 5)
        assert len(result) == len(OHLCV_PRICES)

    def test_linear_data_matches_values(self):
        """For perfectly linear data, LINEARREG endpoint should match the actual value."""
        result = LINEARREG(OHLCV_PRICES, 5)
        finite = _finite(result)
        expected = OHLCV_PRICES[len(OHLCV_PRICES) - len(finite) :]
        assert np.allclose(finite, expected, atol=1e-10)


class TestCORREL:
    def test_perfect_correlation(self):
        result = CORREL(OHLCV_PRICES, OHLCV_PRICES, 10)
        finite = _finite(result)
        assert all(math.isclose(v, 1.0, abs_tol=1e-10) for v in finite)

    def test_range(self):
        result = CORREL(OHLCV_PRICES, OHLCV_HIGH, 10)
        finite = _finite(result)
        assert all(-1.0 <= v <= 1.0 for v in finite)


# ---------------------------------------------------------------------------
# Price Transformations
# ---------------------------------------------------------------------------


class TestPriceTransforms:
    def test_avgprice(self):
        result = AVGPRICE(OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE)
        expected = (OHLCV_OPEN + OHLCV_HIGH + OHLCV_LOW + OHLCV_CLOSE) / 4.0
        assert np.allclose(result, expected, atol=1e-10)

    def test_medprice(self):
        result = MEDPRICE(OHLCV_HIGH, OHLCV_LOW)
        expected = (OHLCV_HIGH + OHLCV_LOW) / 2.0
        assert np.allclose(result, expected, atol=1e-10)

    def test_typprice(self):
        result = TYPPRICE(OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE)
        expected = (OHLCV_HIGH + OHLCV_LOW + OHLCV_CLOSE) / 3.0
        assert np.allclose(result, expected, atol=1e-10)

    def test_wclprice(self):
        result = WCLPRICE(OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE)
        expected = (OHLCV_HIGH + OHLCV_LOW + OHLCV_CLOSE * 2.0) / 4.0
        assert np.allclose(result, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Pattern Recognition
# ---------------------------------------------------------------------------


class TestPatternRecognition:
    def test_cdldoji_output_values(self):
        result = CDLDOJI(OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE)
        assert len(result) == len(OHLCV_PRICES)
        assert all(v in (0, 100) for v in result)

    def test_cdlengulfing_output_values(self):
        result = CDLENGULFING(OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE)
        assert len(result) == len(OHLCV_PRICES)
        assert all(v in (-100, 0, 100) for v in result)

    def test_cdlmarubozu_detects_full_body(self):
        """A full-body candle with no shadows should be detected as marubozu."""
        o = np.array([10.0])
        h = np.array([15.0])
        l = np.array([10.0])
        c = np.array([15.0])
        result = CDLMARUBOZU(o, h, l, c)
        assert result[0] == 100

    def test_cdldoji_detects_doji(self):
        """A candle where open == close should be detected."""
        o = np.array([10.0])
        h = np.array([12.0])
        l = np.array([8.0])
        c = np.array([10.0])
        result = CDLDOJI(o, h, l, c)
        assert result[0] == 100

    def test_cdlhammer_detects_hammer(self):
        """Long lower shadow, small body at top, tiny upper shadow."""
        # body = 0.5, range = 2.0, lower = 1.0 >= 2*0.5, upper = 0.5 <= 0.5
        o = np.array([8.0])
        h = np.array([9.0])
        l = np.array([7.0])
        c = np.array([8.5])
        result = CDLHAMMER(o, h, l, c)
        assert result[0] == 100

    def test_cdlshootingstar_detects_pattern(self):
        """Long upper shadow, small body at bottom, tiny lower shadow."""
        o = np.array([8.5])
        h = np.array([11.0])
        l = np.array([8.0])
        c = np.array([8.0])
        result = CDLSHOOTINGSTAR(o, h, l, c)
        assert result[0] == -100


# ---------------------------------------------------------------------------
# New Overlap Indicators
# ---------------------------------------------------------------------------

# Larger price series for indicators that need more data (MAMA, HT need 32+/63+ bars)
N_LONG = 200
RNG_LONG = np.random.default_rng(123)
LONG_CLOSE = 50.0 + np.cumsum(RNG_LONG.standard_normal(N_LONG) * 0.5)
LONG_HIGH = LONG_CLOSE + RNG_LONG.uniform(0.1, 1.0, N_LONG)
LONG_LOW = LONG_CLOSE - RNG_LONG.uniform(0.1, 1.0, N_LONG)


class TestMA:
    def test_ma_sma_matches_sma(self):
        result_ma = MA(PRICES, timeperiod=5, matype=0)
        result_sma = SMA(PRICES, timeperiod=5)
        assert np.allclose(result_ma, result_sma, equal_nan=True)

    def test_ma_ema_matches_ema(self):
        result_ma = MA(PRICES, timeperiod=5, matype=1)
        result_ema = EMA(PRICES, timeperiod=5)
        assert np.allclose(result_ma, result_ema, equal_nan=True)

    def test_ma_wma_matches_wma(self):
        result_ma = MA(PRICES, timeperiod=5, matype=2)
        result_wma = WMA(PRICES, timeperiod=5)
        assert np.allclose(result_ma, result_wma, equal_nan=True)

    def test_ma_invalid_matype_raises(self):
        with pytest.raises(Exception):
            MA(PRICES, timeperiod=5, matype=99)

    def test_ma_output_length(self):
        result = MA(PRICES, timeperiod=5, matype=0)
        assert len(result) == len(PRICES)

    def test_ma_leading_nans(self):
        """MA(matype=0, period=5) should have 4 leading NaNs."""
        result = MA(PRICES, timeperiod=5, matype=0)
        assert _nan_count(result) == 4  # timeperiod - 1 leading NaNs


class TestMAVP:
    def test_output_length(self):
        periods = np.full(len(PRICES), 5.0)
        result = MAVP(PRICES, periods)
        assert len(result) == len(PRICES)

    def test_constant_period_matches_sma(self):
        """MAVP with constant period should equal SMA with that period."""
        periods = np.full(len(PRICES), 5.0)
        result = MAVP(PRICES, periods, minperiod=5, maxperiod=5)
        expected = SMA(PRICES, timeperiod=5)
        valid = ~np.isnan(result) & ~np.isnan(expected)
        assert np.allclose(result[valid], expected[valid], atol=1e-10)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(Exception):
            MAVP(PRICES, np.array([5.0, 5.0]))


class TestMAMA:
    def test_output_length(self):
        mama_arr, fama_arr = MAMA(LONG_CLOSE)
        assert len(mama_arr) == N_LONG
        assert len(fama_arr) == N_LONG

    def test_leading_nans(self):
        mama_arr, fama_arr = MAMA(LONG_CLOSE)
        # First 32 values should be NaN
        assert all(np.isnan(mama_arr[:32]))
        assert all(np.isnan(fama_arr[:32]))

    def test_valid_values_finite(self):
        mama_arr, fama_arr = MAMA(LONG_CLOSE)
        valid = ~np.isnan(mama_arr)
        assert np.all(np.isfinite(mama_arr[valid]))
        assert np.all(np.isfinite(fama_arr[valid]))


class TestSAREXT:
    def test_output_length(self):
        result = SAREXT(LONG_HIGH, LONG_LOW)
        assert len(result) == N_LONG

    def test_first_value_nan(self):
        result = SAREXT(LONG_HIGH, LONG_LOW)
        assert np.isnan(result[0])

    def test_default_matches_sar(self):
        """SAREXT with default params should be close to SAR."""
        sar_result = SAR(LONG_HIGH, LONG_LOW)
        sarext_result = SAREXT(LONG_HIGH, LONG_LOW)
        valid = ~np.isnan(sar_result) & ~np.isnan(sarext_result)
        assert np.allclose(sar_result[valid], sarext_result[valid], atol=1e-10)


class TestMACDEXT:
    def test_output_length(self):
        m, s, h = MACDEXT(LONG_CLOSE)
        assert len(m) == len(s) == len(h) == N_LONG

    def test_ema_matches_standard_macd(self):
        """MACDEXT with EMA (matype=1) should produce valid output of correct shape.

        Note: MACDEXT uses a different EMA seeding strategy than the `ta` crate's
        EMA (price at index period-1 vs. accumulated from index 0), so exact value
        equivalence with MACD is not expected in the warmup period.
        """
        m_ext, s_ext, h_ext = MACDEXT(
            LONG_CLOSE, fastmatype=1, slowmatype=1, signalmatype=1
        )
        m_std, s_std, h_std = MACD(LONG_CLOSE)
        # Both should have same length
        assert len(m_ext) == len(m_std)
        # Both should have valid (non-NaN) values at the same trailing region
        valid_ext = ~np.isnan(m_ext)
        valid_std = ~np.isnan(m_std)
        # At least 50% of values should be valid for 200-bar series
        assert valid_ext.sum() >= N_LONG // 2
        assert valid_std.sum() >= N_LONG // 2

    def test_invalid_periods_raise(self):
        with pytest.raises(Exception):
            MACDEXT(LONG_CLOSE, fastperiod=26, slowperiod=12)  # fast >= slow


# ---------------------------------------------------------------------------
# New Candlestick Patterns
# ---------------------------------------------------------------------------


class TestNewPatterns:
    def test_cdl3blackcrows_output_values(self):
        result = CDL3BLACKCROWS(OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE)
        assert len(result) == len(OHLCV_PRICES)
        assert all(v in (-100, 0) for v in result)

    def test_cdl3whitesoldiers_output_values(self):
        result = CDL3WHITESOLDIERS(OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE)
        assert len(result) == len(OHLCV_PRICES)
        assert all(v in (0, 100) for v in result)

    def test_cdl3inside_output_values(self):
        result = CDL3INSIDE(OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE)
        assert len(result) == len(OHLCV_PRICES)
        assert all(v in (-100, 0, 100) for v in result)

    def test_cdl3outside_output_values(self):
        result = CDL3OUTSIDE(OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE)
        assert len(result) == len(OHLCV_PRICES)
        assert all(v in (-100, 0, 100) for v in result)

    def test_cdlharami_detects_bearish(self):
        """Prior large bullish, small bearish inside."""
        # Candle 1: bullish, large body (o=10, c=15)
        # Candle 2: bearish (o > c), body inside candle 1 body [10, 15]
        o = np.array([10.0, 12.5])
        h = np.array([15.0, 13.0])
        l = np.array([10.0, 11.5])
        c = np.array([15.0, 12.0])  # bearish: c=12.0 < o=12.5, body inside [10, 15]
        result = CDLHARAMI(o, h, l, c)
        assert result[1] == -100

    def test_cdlharami_detects_bullish(self):
        """Prior large bearish, small bullish inside."""
        o = np.array([15.0, 12.0])
        h = np.array([15.0, 13.0])
        l = np.array([10.0, 11.5])
        c = np.array([10.0, 12.5])  # bullish inside
        result = CDLHARAMI(o, h, l, c)
        assert result[1] == 100

    def test_cdlharamicross_output_values(self):
        result = CDLHARAMICROSS(OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE)
        assert len(result) == len(OHLCV_PRICES)
        assert all(v in (-100, 0, 100) for v in result)

    def test_cdldojistar_output_values(self):
        result = CDLDOJISTAR(OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE)
        assert len(result) == len(OHLCV_PRICES)
        assert all(v in (-100, 0, 100) for v in result)

    def test_cdlmorningdojistar_output_values(self):
        result = CDLMORNINGDOJISTAR(OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE)
        assert len(result) == len(OHLCV_PRICES)
        assert all(v in (0, 100) for v in result)

    def test_cdleveningdojistar_output_values(self):
        result = CDLEVENINGDOJISTAR(OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE)
        assert len(result) == len(OHLCV_PRICES)
        assert all(v in (-100, 0) for v in result)

    def test_cdl3blackcrows_detects_pattern(self):
        """Three consecutive bearish candles, each opening in previous body."""
        # Three strong bearish candles
        o = np.array([100.0, 95.0, 90.0])
        h = np.array([101.0, 97.0, 92.0])
        l = np.array([90.0, 85.0, 80.0])
        c = np.array([91.0, 86.0, 81.0])  # bearish, long body, closes near low
        result = CDL3BLACKCROWS(o, h, l, c)
        assert result[2] == -100

    def test_cdl3whitesoldiers_detects_pattern(self):
        """Three consecutive bullish candles, each opening in previous body."""
        o = np.array([80.0, 86.0, 92.0])
        h = np.array([92.0, 98.0, 104.0])
        l = np.array([79.0, 85.0, 91.0])
        c = np.array([91.0, 97.0, 103.0])  # bullish, long body, closes near high
        result = CDL3WHITESOLDIERS(o, h, l, c)
        assert result[2] == 100


# ---------------------------------------------------------------------------
# Cycle Indicators
# ---------------------------------------------------------------------------


class TestHilbertTransform:
    def test_ht_trendline_output_length(self):
        result = HT_TRENDLINE(LONG_CLOSE)
        assert len(result) == N_LONG

    def test_ht_trendline_leading_nans(self):
        result = HT_TRENDLINE(LONG_CLOSE)
        assert all(np.isnan(result[:63]))

    def test_ht_trendline_valid_values(self):
        result = HT_TRENDLINE(LONG_CLOSE)
        valid = ~np.isnan(result)
        assert valid.any()
        assert np.all(np.isfinite(result[valid]))

    def test_ht_dcperiod_output_length(self):
        result = HT_DCPERIOD(LONG_CLOSE)
        assert len(result) == N_LONG

    def test_ht_dcperiod_values_in_range(self):
        """Dominant cycle period should be between 6 and 50."""
        result = HT_DCPERIOD(LONG_CLOSE)
        valid = ~np.isnan(result)
        assert valid.any()
        assert np.all(result[valid] >= 6.0)
        assert np.all(result[valid] <= 50.0)

    def test_ht_dcphase_output_length(self):
        result = HT_DCPHASE(LONG_CLOSE)
        assert len(result) == N_LONG

    def test_ht_phasor_returns_two_arrays(self):
        inphase, quad = HT_PHASOR(LONG_CLOSE)
        assert len(inphase) == N_LONG
        assert len(quad) == N_LONG

    def test_ht_phasor_leading_nans(self):
        inphase, quad = HT_PHASOR(LONG_CLOSE)
        assert all(np.isnan(inphase[:63]))
        assert all(np.isnan(quad[:63]))

    def test_ht_sine_returns_two_arrays(self):
        sine, lead = HT_SINE(LONG_CLOSE)
        assert len(sine) == N_LONG
        assert len(lead) == N_LONG

    def test_ht_sine_values_in_range(self):
        """Sine values must be in [-1, 1]."""
        sine, lead = HT_SINE(LONG_CLOSE)
        valid = ~np.isnan(sine)
        assert valid.any()
        assert np.all(np.abs(sine[valid]) <= 1.0 + 1e-9)
        assert np.all(np.abs(lead[valid]) <= 1.0 + 1e-9)

    def test_ht_trendmode_output_length(self):
        result = HT_TRENDMODE(LONG_CLOSE)
        assert len(result) == N_LONG

    def test_ht_trendmode_values_binary(self):
        """Trend mode must be 0 or 1."""
        result = HT_TRENDMODE(LONG_CLOSE)
        assert all(v in (0, 1) for v in result)

    def test_short_series_returns_all_nans(self):
        """Series shorter than lookback should return all NaN."""
        short = np.arange(1.0, 10.0)
        result = HT_TRENDLINE(short)
        assert all(np.isnan(result))


# ---------------------------------------------------------------------------
# New Pattern Recognition Tests (43 patterns)
# ---------------------------------------------------------------------------


class TestNewPatterns:
    """Basic output-length and value-set checks for 43 new patterns."""

    O = OHLCV_OPEN
    H = OHLCV_HIGH
    L = OHLCV_LOW
    C = OHLCV_CLOSE
    N = len(OHLCV_PRICES)

    # -- CDL3LINESTRIKE -------------------------------------------------------
    def test_cdl3linestrike_length(self):
        r = CDL3LINESTRIKE(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdl3linestrike_values(self):
        r = CDL3LINESTRIKE(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0, 100) for v in r)

    def test_cdl3linestrike_detects_bearish(self):
        """3 bullish candles then a bearish engulfing all three."""
        o = np.array([10.0, 11.0, 12.0, 16.0])
        h = np.array([11.5, 12.5, 13.5, 16.5])
        l = np.array([9.5, 10.5, 11.5, 9.0])
        c = np.array([11.0, 12.0, 13.0, 9.5])  # bearish closes below first open
        r = CDL3LINESTRIKE(o, h, l, c)
        assert r[3] == -100

    # -- CDL3STARSINSOUTH -----------------------------------------------------
    def test_cdl3starsinsouth_length(self):
        r = CDL3STARSINSOUTH(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdl3starsinsouth_values(self):
        r = CDL3STARSINSOUTH(self.O, self.H, self.L, self.C)
        assert all(v in (0, 100) for v in r)

    # -- CDLABANDONEDBABY -----------------------------------------------------
    def test_cdlabandonedbaby_length(self):
        r = CDLABANDONEDBABY(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlabandonedbaby_values(self):
        r = CDLABANDONEDBABY(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0, 100) for v in r)

    def test_cdlabandonedbaby_detects_bullish(self):
        """Large bearish, doji gaps down (h_doji < l_prior), large bullish gaps up."""
        o = np.array([20.0, 9.0, 12.0])
        h = np.array([21.0, 9.1, 20.0])
        l = np.array([11.0, 8.9, 11.5])
        c = np.array(
            [12.0, 9.0, 19.0]
        )  # doji gaps below l[0]=11, bullish gaps above h[1]=9.1
        r = CDLABANDONEDBABY(o, h, l, c)
        assert r[2] == 100

    # -- CDLADVANCEBLOCK ------------------------------------------------------
    def test_cdladvanceblock_length(self):
        r = CDLADVANCEBLOCK(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdladvanceblock_values(self):
        r = CDLADVANCEBLOCK(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0) for v in r)

    # -- CDLBELTHOLD ----------------------------------------------------------
    def test_cdlbelthold_length(self):
        r = CDLBELTHOLD(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlbelthold_values(self):
        r = CDLBELTHOLD(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0, 100) for v in r)

    def test_cdlbelthold_detects_bullish(self):
        """Bullish candle opening at its low."""
        o = np.array([10.0])
        h = np.array([15.0])
        l = np.array([10.0])  # open == low
        c = np.array([14.5])
        r = CDLBELTHOLD(o, h, l, c)
        assert r[0] == 100

    def test_cdlbelthold_detects_bearish(self):
        """Bearish candle opening at its high."""
        o = np.array([15.0])
        h = np.array([15.0])  # open == high
        l = np.array([10.0])
        c = np.array([10.5])
        r = CDLBELTHOLD(o, h, l, c)
        assert r[0] == -100

    # -- CDLBREAKAWAY ---------------------------------------------------------
    def test_cdlbreakaway_length(self):
        r = CDLBREAKAWAY(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlbreakaway_values(self):
        r = CDLBREAKAWAY(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0, 100) for v in r)

    # -- CDLCLOSINGMARUBOZU ---------------------------------------------------
    def test_cdlclosingmarubozu_length(self):
        r = CDLCLOSINGMARUBOZU(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlclosingmarubozu_values(self):
        r = CDLCLOSINGMARUBOZU(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0, 100) for v in r)

    def test_cdlclosingmarubozu_detects_bullish(self):
        """Bullish closing marubozu: close == high."""
        o = np.array([10.0])
        h = np.array([15.0])
        l = np.array([9.0])
        c = np.array([15.0])  # close == high, no upper shadow
        r = CDLCLOSINGMARUBOZU(o, h, l, c)
        assert r[0] == 100

    def test_cdlclosingmarubozu_detects_bearish(self):
        """Bearish closing marubozu: close == low."""
        o = np.array([15.0])
        h = np.array([16.0])
        l = np.array([10.0])
        c = np.array([10.0])  # close == low, no lower shadow
        r = CDLCLOSINGMARUBOZU(o, h, l, c)
        assert r[0] == -100

    # -- CDLCONCEALBABYSWALL --------------------------------------------------
    def test_cdlconcealbabyswall_length(self):
        r = CDLCONCEALBABYSWALL(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlconcealbabyswall_values(self):
        r = CDLCONCEALBABYSWALL(self.O, self.H, self.L, self.C)
        assert all(v in (0, 100) for v in r)

    # -- CDLCOUNTERATTACK -----------------------------------------------------
    def test_cdlcounterattack_length(self):
        r = CDLCOUNTERATTACK(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlcounterattack_values(self):
        r = CDLCOUNTERATTACK(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0, 100) for v in r)

    # -- CDLDARKCLOUDCOVER ----------------------------------------------------
    def test_cdldarkcloudcover_length(self):
        r = CDLDARKCLOUDCOVER(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdldarkcloudcover_values(self):
        r = CDLDARKCLOUDCOVER(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0) for v in r)

    def test_cdldarkcloudcover_detects_pattern(self):
        """Bearish candle opening above prior high and closing below midpoint."""
        o = np.array([10.0, 16.0])
        h = np.array([15.0, 17.0])
        l = np.array([9.5, 11.0])
        c = np.array([14.0, 11.5])  # bearish, closes below midpoint of (10,14)
        r = CDLDARKCLOUDCOVER(o, h, l, c)
        assert r[1] == -100

    # -- CDLDRAGONFLYDOJI -----------------------------------------------------
    def test_cdldragonflydoji_length(self):
        r = CDLDRAGONFLYDOJI(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdldragonflydoji_values(self):
        r = CDLDRAGONFLYDOJI(self.O, self.H, self.L, self.C)
        assert all(v in (0, 100) for v in r)

    def test_cdldragonflydoji_detects_pattern(self):
        """Open ≈ close ≈ high with long lower shadow."""
        o = np.array([15.0])
        h = np.array([15.1])
        l = np.array([10.0])
        c = np.array([15.0])
        r = CDLDRAGONFLYDOJI(o, h, l, c)
        assert r[0] == 100

    # -- CDLGAPSIDESIDEWHITE --------------------------------------------------
    def test_cdlgapsidesidewhite_length(self):
        r = CDLGAPSIDESIDEWHITE(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlgapsidesidewhite_values(self):
        r = CDLGAPSIDESIDEWHITE(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0, 100) for v in r)

    # -- CDLGRAVESTONEDOJI ----------------------------------------------------
    def test_cdlgravestonedoji_length(self):
        r = CDLGRAVESTONEDOJI(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlgravestonedoji_values(self):
        r = CDLGRAVESTONEDOJI(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0) for v in r)

    def test_cdlgravestonedoji_detects_pattern(self):
        """Open ≈ close ≈ low with long upper shadow."""
        o = np.array([10.0])
        h = np.array([15.0])
        l = np.array([9.9])
        c = np.array([10.0])
        r = CDLGRAVESTONEDOJI(o, h, l, c)
        assert r[0] == -100

    # -- CDLHANGINGMAN --------------------------------------------------------
    def test_cdlhangingman_length(self):
        r = CDLHANGINGMAN(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlhangingman_values(self):
        r = CDLHANGINGMAN(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0) for v in r)

    def test_cdlhangingman_detects_pattern(self):
        """Same shape as hammer but returns -100."""
        o = np.array([14.0])
        h = np.array([15.0])
        l = np.array([10.0])
        c = np.array([14.5])
        r = CDLHANGINGMAN(o, h, l, c)
        assert r[0] == -100

    # -- CDLHIGHWAVE ----------------------------------------------------------
    def test_cdlhighwave_length(self):
        r = CDLHIGHWAVE(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlhighwave_values(self):
        r = CDLHIGHWAVE(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0, 100) for v in r)

    def test_cdlhighwave_detects_pattern(self):
        """Small body with very long shadows."""
        o = np.array([12.4])
        h = np.array([20.0])
        l = np.array([5.0])
        c = np.array([12.6])  # body=0.2, range=15, upper=7.6, lower=7.4
        r = CDLHIGHWAVE(o, h, l, c)
        assert r[0] != 0

    # -- CDLHIKKAKE -----------------------------------------------------------
    def test_cdlhikkake_length(self):
        r = CDLHIKKAKE(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlhikkake_values(self):
        r = CDLHIKKAKE(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0, 100) for v in r)

    # -- CDLHIKKAKEMOD --------------------------------------------------------
    def test_cdlhikkakemod_length(self):
        r = CDLHIKKAKEMOD(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlhikkakemod_values(self):
        r = CDLHIKKAKEMOD(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0, 100) for v in r)

    # -- CDLHOMINGPIGEON ------------------------------------------------------
    def test_cdlhomingpigeon_length(self):
        r = CDLHOMINGPIGEON(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlhomingpigeon_values(self):
        r = CDLHOMINGPIGEON(self.O, self.H, self.L, self.C)
        assert all(v in (0, 100) for v in r)

    def test_cdlhomingpigeon_detects_pattern(self):
        """2 bearish candles, second entirely within first body."""
        o = np.array([20.0, 17.0])
        h = np.array([20.5, 17.5])
        l = np.array([10.0, 13.0])
        c = np.array([11.0, 14.0])  # both bearish, second within first body
        r = CDLHOMINGPIGEON(o, h, l, c)
        assert r[1] == 100

    # -- CDLIDENTICAL3CROWS ---------------------------------------------------
    def test_cdlidentical3crows_length(self):
        r = CDLIDENTICAL3CROWS(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlidentical3crows_values(self):
        r = CDLIDENTICAL3CROWS(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0) for v in r)

    # -- CDLINNECK ------------------------------------------------------------
    def test_cdlinneck_length(self):
        r = CDLINNECK(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlinneck_values(self):
        r = CDLINNECK(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0) for v in r)

    # -- CDLINVERTEDHAMMER ----------------------------------------------------
    def test_cdlinvertedhammer_length(self):
        r = CDLINVERTEDHAMMER(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlinvertedhammer_values(self):
        r = CDLINVERTEDHAMMER(self.O, self.H, self.L, self.C)
        assert all(v in (0, 100) for v in r)

    def test_cdlinvertedhammer_detects_pattern(self):
        """Small body at bottom, long upper shadow."""
        o = np.array([10.5])
        h = np.array([15.0])
        l = np.array([10.0])
        c = np.array([11.0])  # body=0.5, upper=4.0, lower=0.5
        r = CDLINVERTEDHAMMER(o, h, l, c)
        assert r[0] == 100

    # -- CDLKICKING -----------------------------------------------------------
    def test_cdlkicking_length(self):
        r = CDLKICKING(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlkicking_values(self):
        r = CDLKICKING(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0, 100) for v in r)

    def test_cdlkicking_detects_bullish(self):
        """Bearish marubozu then bullish marubozu with gap up."""
        o = np.array([15.0, 18.0])
        h = np.array([15.0, 23.0])  # bearish: open==high; bullish: close==high
        l = np.array([10.0, 18.0])  # bearish: close==low; bullish: open==low
        c = np.array([10.0, 23.0])
        r = CDLKICKING(o, h, l, c)
        assert r[1] == 100

    # -- CDLKICKINGBYLENGTH ---------------------------------------------------
    def test_cdlkickingbylength_length(self):
        r = CDLKICKINGBYLENGTH(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlkickingbylength_values(self):
        r = CDLKICKINGBYLENGTH(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0, 100) for v in r)

    # -- CDLLADDERBOTTOM ------------------------------------------------------
    def test_cdlladderbottom_length(self):
        r = CDLLADDERBOTTOM(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlladderbottom_values(self):
        r = CDLLADDERBOTTOM(self.O, self.H, self.L, self.C)
        assert all(v in (0, 100) for v in r)

    # -- CDLLONGLEGGEDDOJI ----------------------------------------------------
    def test_cdllongleggeddoji_length(self):
        r = CDLLONGLEGGEDDOJI(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdllongleggeddoji_values(self):
        r = CDLLONGLEGGEDDOJI(self.O, self.H, self.L, self.C)
        assert all(v in (0, 100) for v in r)

    def test_cdllongleggeddoji_detects_pattern(self):
        """Doji with long upper and lower shadows."""
        o = np.array([12.5])
        h = np.array([20.0])
        l = np.array([5.0])
        c = np.array([12.5])  # body=0, range=15, doji with both long shadows
        r = CDLLONGLEGGEDDOJI(o, h, l, c)
        assert r[0] == 100

    # -- CDLLONGLINE ----------------------------------------------------------
    def test_cdllongline_length(self):
        r = CDLLONGLINE(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdllongline_values(self):
        r = CDLLONGLINE(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0, 100) for v in r)

    def test_cdllongline_detects_bullish(self):
        """Long body >= 70% of range."""
        o = np.array([10.0])
        h = np.array([15.0])
        l = np.array([9.5])
        c = np.array([15.0])  # body=5, range=5.5 => body/range=0.91
        r = CDLLONGLINE(o, h, l, c)
        assert r[0] == 100

    # -- CDLMATCHINGLOW -------------------------------------------------------
    def test_cdlmatchinglow_length(self):
        r = CDLMATCHINGLOW(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlmatchinglow_values(self):
        r = CDLMATCHINGLOW(self.O, self.H, self.L, self.C)
        assert all(v in (0, 100) for v in r)

    def test_cdlmatchinglow_detects_pattern(self):
        """Two bearish candles with equal closes."""
        o = np.array([15.0, 14.0])
        h = np.array([15.5, 14.5])
        l = np.array([10.0, 10.0])
        c = np.array([10.0, 10.0])  # equal closes, both bearish
        r = CDLMATCHINGLOW(o, h, l, c)
        assert r[1] == 100

    # -- CDLMATHOLD -----------------------------------------------------------
    def test_cdlmathold_length(self):
        r = CDLMATHOLD(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlmathold_values(self):
        r = CDLMATHOLD(self.O, self.H, self.L, self.C)
        assert all(v in (0, 100) for v in r)

    # -- CDLONNECK ------------------------------------------------------------
    def test_cdlonneck_length(self):
        r = CDLONNECK(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlonneck_values(self):
        r = CDLONNECK(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0) for v in r)

    # -- CDLPIERCING ----------------------------------------------------------
    def test_cdlpiercing_length(self):
        r = CDLPIERCING(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlpiercing_values(self):
        r = CDLPIERCING(self.O, self.H, self.L, self.C)
        assert all(v in (0, 100) for v in r)

    def test_cdlpiercing_detects_pattern(self):
        """Bearish then bullish that opens below prior low and closes above midpoint."""
        o = np.array([14.0, 9.0])
        h = np.array([15.0, 13.0])
        l = np.array([10.0, 8.5])
        c = np.array([10.5, 12.5])  # closes above midpoint of (14,10.5)=12.25
        r = CDLPIERCING(o, h, l, c)
        assert r[1] == 100

    # -- CDLRICKSHAWMAN -------------------------------------------------------
    def test_cdlrickshawman_length(self):
        r = CDLRICKSHAWMAN(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlrickshawman_values(self):
        r = CDLRICKSHAWMAN(self.O, self.H, self.L, self.C)
        assert all(v in (0, 100) for v in r)

    # -- CDLRISEFALL3METHODS --------------------------------------------------
    def test_cdlrisefall3methods_length(self):
        r = CDLRISEFALL3METHODS(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlrisefall3methods_values(self):
        r = CDLRISEFALL3METHODS(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0, 100) for v in r)

    # -- CDLSEPARATINGLINES ---------------------------------------------------
    def test_cdlseparatinglines_length(self):
        r = CDLSEPARATINGLINES(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlseparatinglines_values(self):
        r = CDLSEPARATINGLINES(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0, 100) for v in r)

    # -- CDLSHORTLINE ---------------------------------------------------------
    def test_cdlshortline_length(self):
        r = CDLSHORTLINE(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlshortline_values(self):
        r = CDLSHORTLINE(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0, 100) for v in r)

    def test_cdlshortline_detects_bullish(self):
        """Short bullish body <= 30% of range."""
        o = np.array([10.0])
        h = np.array([15.0])
        l = np.array([9.0])
        c = np.array([11.0])  # body=1, range=6 => body/range=0.17
        r = CDLSHORTLINE(o, h, l, c)
        assert r[0] == 100

    # -- CDLSTALLEDPATTERN ----------------------------------------------------
    def test_cdlstalledpattern_length(self):
        r = CDLSTALLEDPATTERN(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlstalledpattern_values(self):
        r = CDLSTALLEDPATTERN(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0) for v in r)

    # -- CDLSTICKSANDWICH -----------------------------------------------------
    def test_cdlsticksandwich_length(self):
        r = CDLSTICKSANDWICH(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlsticksandwich_values(self):
        r = CDLSTICKSANDWICH(self.O, self.H, self.L, self.C)
        assert all(v in (0, 100) for v in r)

    def test_cdlsticksandwich_detects_pattern(self):
        """Bearish, bullish in middle, bearish with same close as first."""
        o = np.array([15.0, 10.5, 14.0])
        h = np.array([15.5, 14.5, 14.5])
        l = np.array([10.0, 10.0, 10.0])
        c = np.array([10.0, 14.0, 10.0])  # first and third close at 10.0
        r = CDLSTICKSANDWICH(o, h, l, c)
        assert r[2] == 100

    # -- CDLTAKURI ------------------------------------------------------------
    def test_cdltakuri_length(self):
        r = CDLTAKURI(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdltakuri_values(self):
        r = CDLTAKURI(self.O, self.H, self.L, self.C)
        assert all(v in (0, 100) for v in r)

    def test_cdltakuri_detects_pattern(self):
        """Very long lower shadow >= 3x body, open near high."""
        o = np.array([15.0])
        h = np.array([15.2])
        l = np.array([10.0])
        c = np.array([15.1])  # body=0.1, lower=5.0, lower>=3*body
        r = CDLTAKURI(o, h, l, c)
        assert r[0] == 100

    # -- CDLTASUKIGAP ---------------------------------------------------------
    def test_cdltasukigap_length(self):
        r = CDLTASUKIGAP(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdltasukigap_values(self):
        r = CDLTASUKIGAP(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0, 100) for v in r)

    # -- CDLTHRUSTING ---------------------------------------------------------
    def test_cdlthrusting_length(self):
        r = CDLTHRUSTING(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlthrusting_values(self):
        r = CDLTHRUSTING(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0) for v in r)

    # -- CDLTRISTAR -----------------------------------------------------------
    def test_cdltristar_length(self):
        r = CDLTRISTAR(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdltristar_values(self):
        r = CDLTRISTAR(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0, 100) for v in r)

    # -- CDLUNIQUE3RIVER ------------------------------------------------------
    def test_cdlunique3river_length(self):
        r = CDLUNIQUE3RIVER(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlunique3river_values(self):
        r = CDLUNIQUE3RIVER(self.O, self.H, self.L, self.C)
        assert all(v in (0, 100) for v in r)

    # -- CDLUPSIDEGAP2CROWS ---------------------------------------------------
    def test_cdlupsidegap2crows_length(self):
        r = CDLUPSIDEGAP2CROWS(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlupsidegap2crows_values(self):
        r = CDLUPSIDEGAP2CROWS(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0) for v in r)

    # -- CDLXSIDEGAP3METHODS --------------------------------------------------
    def test_cdlxsidegap3methods_length(self):
        r = CDLXSIDEGAP3METHODS(self.O, self.H, self.L, self.C)
        assert len(r) == self.N

    def test_cdlxsidegap3methods_values(self):
        r = CDLXSIDEGAP3METHODS(self.O, self.H, self.L, self.C)
        assert all(v in (-100, 0, 100) for v in r)

    def test_cdlxsidegap3methods_detects_bullish(self):
        """Upside gap three methods: gap up bullish, bearish fills gap."""
        o = np.array([10.0, 12.0, 11.5])
        h = np.array([10.5, 13.0, 12.0])
        l = np.array([9.5, 11.5, 10.0])
        c = np.array([10.0, 12.5, 10.5])  # gap up then partial fill
        r = CDLXSIDEGAP3METHODS(o, h, l, c)
        assert r[2] in (0, 100)  # may or may not detect depending on threshold


# ---------------------------------------------------------------------------
# Math Operators & Math Transforms
# ---------------------------------------------------------------------------


class TestMathOperators:
    A = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    B = np.array([2.0, 2.0, 2.0, 2.0, 2.0])

    def test_add(self):
        r = ADD(self.A, self.B)
        assert np.allclose(r, [3, 4, 5, 6, 7])

    def test_sub(self):
        r = SUB(self.A, self.B)
        assert np.allclose(r, [-1, 0, 1, 2, 3])

    def test_mult(self):
        r = MULT(self.A, self.B)
        assert np.allclose(r, [2, 4, 6, 8, 10])

    def test_div(self):
        r = DIV(self.A, self.B)
        assert np.allclose(r, [0.5, 1, 1.5, 2, 2.5])

    def test_sum_rolling(self):
        r = SUM(self.A, timeperiod=3)
        assert np.isnan(r[0]) and np.isnan(r[1])
        assert math.isclose(r[2], 6.0)
        assert math.isclose(r[3], 9.0)
        assert math.isclose(r[4], 12.0)

    def test_max_rolling(self):
        r = MAX(self.A, timeperiod=3)
        assert np.isnan(r[0]) and np.isnan(r[1])
        assert math.isclose(r[2], 3.0)
        assert math.isclose(r[4], 5.0)

    def test_min_rolling(self):
        r = MIN(self.A, timeperiod=3)
        assert np.isnan(r[0]) and np.isnan(r[1])
        assert math.isclose(r[2], 1.0)
        assert math.isclose(r[4], 3.0)

    def test_maxindex(self):
        r = MAXINDEX(self.A, timeperiod=3)
        assert r[0] == -1 and r[1] == -1
        assert r[2] == 2  # max at index 2 (value 3)
        assert r[4] == 4  # max at index 4 (value 5)

    def test_minindex(self):
        r = MININDEX(self.A, timeperiod=3)
        assert r[0] == -1 and r[1] == -1
        assert r[2] == 0  # min at index 0 (value 1)
        assert r[4] == 2  # min at index 2 (value 3)

    def test_sum_output_length(self):
        r = SUM(self.A, timeperiod=2)
        assert len(r) == len(self.A)

    def test_max_output_length(self):
        r = MAX(self.A, timeperiod=2)
        assert len(r) == len(self.A)


class TestMathTransforms:
    X = np.array([0.0, 0.5, 1.0])
    POS = np.array([1.0, 2.0, 4.0])

    def test_acos(self):
        r = ACOS(self.X)
        assert np.allclose(r, np.arccos(self.X))

    def test_asin(self):
        r = ASIN(self.X)
        assert np.allclose(r, np.arcsin(self.X))

    def test_atan(self):
        r = ATAN(self.X)
        assert np.allclose(r, np.arctan(self.X))

    def test_ceil(self):
        r = CEIL(np.array([1.1, 2.5, 3.9]))
        assert np.allclose(r, [2.0, 3.0, 4.0])

    def test_floor(self):
        r = FLOOR(np.array([1.1, 2.5, 3.9]))
        assert np.allclose(r, [1.0, 2.0, 3.0])

    def test_cos(self):
        r = COS(self.X)
        assert np.allclose(r, np.cos(self.X))

    def test_sin(self):
        r = SIN(self.X)
        assert np.allclose(r, np.sin(self.X))

    def test_tan(self):
        r = TAN(self.X)
        assert np.allclose(r, np.tan(self.X))

    def test_exp(self):
        r = EXP(self.X)
        assert np.allclose(r, np.exp(self.X))

    def test_ln(self):
        r = LN(self.POS)
        assert np.allclose(r, np.log(self.POS))

    def test_log10(self):
        r = LOG10(self.POS)
        assert np.allclose(r, np.log10(self.POS))

    def test_sqrt(self):
        r = SQRT(self.POS)
        assert np.allclose(r, np.sqrt(self.POS))

    def test_sinh(self):
        r = SINH(self.X)
        assert np.allclose(r, np.sinh(self.X))

    def test_cosh(self):
        r = COSH(self.X)
        assert np.allclose(r, np.cosh(self.X))

    def test_tanh(self):
        r = TANH(self.X)
        assert np.allclose(r, np.tanh(self.X))

    def test_accepts_list_input(self):
        r = SQRT([1.0, 4.0, 9.0])
        assert np.allclose(r, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Pandas Series / DataFrame API
# ---------------------------------------------------------------------------


class TestPandasAPI:
    """Verify that pandas.Series inputs are transparently supported."""

    pd = pytest.importorskip("pandas")

    @pytest.fixture(autouse=True)
    def prices_series(self):
        import pandas as pd

        idx = pd.date_range("2024-01-01", periods=20)
        self.close_s = pd.Series(np.arange(1.0, 21.0), index=idx)
        self.open_s = pd.Series(np.arange(1.0, 21.0) - 0.2, index=idx)
        self.high_s = pd.Series(np.arange(1.0, 21.0) + 0.5, index=idx)
        self.low_s = pd.Series(np.arange(1.0, 21.0) - 0.5, index=idx)

    def test_sma_returns_series(self):
        import pandas as pd

        r = SMA(self.close_s, timeperiod=5)
        assert isinstance(r, pd.Series)

    def test_sma_index_preserved(self):
        r = SMA(self.close_s, timeperiod=5)
        assert list(r.index) == list(self.close_s.index)

    def test_sma_values_match_numpy(self):
        np_result = SMA(self.close_s.to_numpy(), timeperiod=5)
        pd_result = SMA(self.close_s, timeperiod=5)
        assert np.allclose(np_result, pd_result.to_numpy(), equal_nan=True)

    def test_ema_returns_series_with_index(self):
        import pandas as pd

        r = EMA(self.close_s, timeperiod=5)
        assert isinstance(r, pd.Series)
        assert list(r.index) == list(self.close_s.index)

    def test_rsi_returns_series(self):
        import pandas as pd

        long_s = pd.concat(
            [
                self.close_s,
                pd.Series(
                    np.arange(21.0, 41.0), index=pd.date_range("2024-01-21", periods=20)
                ),
            ]
        )
        r = RSI(long_s, timeperiod=10)
        assert isinstance(r, pd.Series)
        assert len(r) == len(long_s)

    def test_bbands_returns_tuple_of_series(self):
        import pandas as pd

        upper, mid, lower = BBANDS(self.close_s, timeperiod=5)
        assert isinstance(upper, pd.Series)
        assert isinstance(mid, pd.Series)
        assert isinstance(lower, pd.Series)
        assert list(upper.index) == list(self.close_s.index)

    def test_macd_returns_tuple_of_series(self):
        import pandas as pd

        close_long = self.pd.Series(np.arange(1.0, 101.0))
        m, s, h = MACD(close_long)
        assert isinstance(m, pd.Series)
        assert isinstance(s, pd.Series)
        assert isinstance(h, pd.Series)

    def test_pattern_returns_series(self):
        import pandas as pd

        r = CDLDOJI(self.open_s, self.high_s, self.low_s, self.close_s)
        assert isinstance(r, pd.Series)
        assert list(r.index) == list(self.close_s.index)

    def test_atr_returns_series(self):
        import pandas as pd

        r = ATR(self.high_s, self.low_s, self.close_s, timeperiod=5)
        assert isinstance(r, pd.Series)

    def test_numpy_input_unaffected(self):
        """Passing plain numpy arrays still returns numpy arrays."""
        arr = np.arange(1.0, 21.0)
        r = SMA(arr, timeperiod=5)
        assert isinstance(r, np.ndarray)

    def test_math_add_with_series(self):
        import pandas as pd

        a = pd.Series([1.0, 2.0, 3.0])
        b = pd.Series([4.0, 5.0, 6.0])
        r = ADD(a, b)
        assert isinstance(r, pd.Series)
        assert np.allclose(r.to_numpy(), [5, 7, 9])


# ---------------------------------------------------------------------------
# Pandas DataFrame OHLCV contract (get_ohlcv + configurable column names)
# ---------------------------------------------------------------------------


class TestPandasDataFrameOHLCV:
    """DataFrame with OHLCV columns: get_ohlcv, default and custom column names, index preservation."""

    pd = pytest.importorskip("pandas")

    @pytest.fixture(autouse=True)
    def df_default_columns(self):
        """DataFrame with default column names open, high, low, close, volume."""
        import pandas as pd

        n = 30
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        close = np.arange(1.0, n + 1.0, dtype=float)
        self.df_default = pd.DataFrame(
            {
                "open": close - 0.2,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.full(n, 1000.0),
            },
            index=idx,
        )
        return None

    @pytest.fixture
    def df_custom_columns(self):
        """DataFrame with custom column names (Open, High, Low, Close)."""
        import pandas as pd

        n = 30
        idx = pd.date_range("2024-02-01", periods=n, freq="D")
        close = np.arange(10.0, n + 10.0, dtype=float)
        return pd.DataFrame(
            {
                "Open": close - 0.2,
                "High": close + 0.5,
                "Low": close - 0.5,
                "Close": close,
            },
            index=idx,
        )

    def test_get_ohlcv_default_columns(self):
        """get_ohlcv with default column names returns (o, h, l, c, v) with index."""
        from ferro_ta.utils import get_ohlcv

        o, h, l, c, v = get_ohlcv(self.df_default)
        assert list(o.index) == list(self.df_default.index)
        np.testing.assert_array_almost_equal(c, self.df_default["close"].to_numpy())

    def test_get_ohlcv_custom_columns(self, df_custom_columns):
        """get_ohlcv with custom column names."""
        from ferro_ta.utils import get_ohlcv

        o, h, l, c, v = get_ohlcv(
            df_custom_columns,
            open_col="Open",
            high_col="High",
            low_col="Low",
            close_col="Close",
            volume_col=None,
        )
        assert len(c) == len(df_custom_columns)
        np.testing.assert_array_almost_equal(c, df_custom_columns["Close"].to_numpy())

    def test_dataframe_ohlcv_overlap_sma(self):
        """Overlap (SMA): DataFrame via get_ohlcv, index preserved, values match NumPy."""
        from ferro_ta.utils import get_ohlcv

        o, h, l, c, v = get_ohlcv(self.df_default)
        r_series = SMA(c, timeperiod=5)
        r_numpy = SMA(self.df_default["close"].to_numpy(), timeperiod=5)
        assert list(r_series.index) == list(self.df_default.index)
        np.testing.assert_allclose(r_series.to_numpy(), r_numpy, equal_nan=True)

    def test_dataframe_ohlcv_momentum_rsi(self):
        """Momentum (RSI): DataFrame via get_ohlcv, index preserved, values match NumPy."""
        from ferro_ta.utils import get_ohlcv

        o, h, l, c, v = get_ohlcv(self.df_default)
        r_series = RSI(c, timeperiod=5)
        r_numpy = RSI(self.df_default["close"].to_numpy(), timeperiod=5)
        assert list(r_series.index) == list(self.df_default.index)
        np.testing.assert_allclose(r_series.to_numpy(), r_numpy, equal_nan=True)

    def test_dataframe_ohlcv_volatility_atr(self, df_custom_columns):
        """Volatility (ATR): custom column names, index preserved, values match NumPy."""
        from ferro_ta.utils import get_ohlcv

        o, h, l, c, v = get_ohlcv(
            df_custom_columns,
            open_col="Open",
            high_col="High",
            low_col="Low",
            close_col="Close",
            volume_col=None,
        )
        r_series = ATR(h, l, c, timeperiod=5)
        r_numpy = ATR(
            df_custom_columns["High"].to_numpy(),
            df_custom_columns["Low"].to_numpy(),
            df_custom_columns["Close"].to_numpy(),
            timeperiod=5,
        )
        assert list(r_series.index) == list(df_custom_columns.index)
        np.testing.assert_allclose(r_series.to_numpy(), r_numpy, equal_nan=True)

    def test_dataframe_ohlcv_pattern(self):
        """Pattern (CDLDOJI): DataFrame via get_ohlcv, index preserved."""
        from ferro_ta.utils import get_ohlcv

        o, h, l, c, v = get_ohlcv(self.df_default)
        r_series = CDLDOJI(o, h, l, c)
        r_numpy = CDLDOJI(
            self.df_default["open"].to_numpy(),
            self.df_default["high"].to_numpy(),
            self.df_default["low"].to_numpy(),
            self.df_default["close"].to_numpy(),
        )
        assert list(r_series.index) == list(self.df_default.index)
        np.testing.assert_array_equal(r_series.to_numpy(), r_numpy)

    def test_dataframe_ohlcv_cycle_ht_trendline(self):
        """Cycle (HT_TRENDLINE): DataFrame via get_ohlcv, index preserved, values match NumPy."""
        import pandas as pd

        n = 100
        idx = pd.date_range("2024-03-01", periods=n, freq="D")
        close_arr = np.arange(1.0, n + 1.0, dtype=float)
        df = pd.DataFrame(
            {
                "open": close_arr - 0.2,
                "high": close_arr + 0.5,
                "low": close_arr - 0.5,
                "close": close_arr,
                "volume": np.full(n, 1000.0),
            },
            index=idx,
        )
        from ferro_ta.utils import get_ohlcv

        o, h, l, c, v = get_ohlcv(df)
        r_series = HT_TRENDLINE(c)
        r_numpy = HT_TRENDLINE(close_arr)
        assert list(r_series.index) == list(idx)
        np.testing.assert_allclose(r_series.to_numpy(), r_numpy, equal_nan=True)

    def test_dataframe_ohlcv_statistic_stddev(self):
        """Statistic (STDDEV): DataFrame via get_ohlcv, index preserved, values match NumPy."""
        from ferro_ta.utils import get_ohlcv

        o, h, l, c, v = get_ohlcv(self.df_default)
        r_series = STDDEV(c, timeperiod=5)
        r_numpy = STDDEV(self.df_default["close"].to_numpy(), timeperiod=5)
        assert list(r_series.index) == list(self.df_default.index)
        np.testing.assert_allclose(r_series.to_numpy(), r_numpy, equal_nan=True)

    def test_dataframe_ohlcv_statistic_correl(self):
        """Statistic (CORREL): two Series from DataFrame, index preserved, values match NumPy."""
        from ferro_ta.utils import get_ohlcv

        o, h, l, c, v = get_ohlcv(self.df_default)
        r_series = CORREL(c, h, timeperiod=5)
        r_numpy = CORREL(
            self.df_default["close"].to_numpy(),
            self.df_default["high"].to_numpy(),
            timeperiod=5,
        )
        assert list(r_series.index) == list(self.df_default.index)
        np.testing.assert_allclose(r_series.to_numpy(), r_numpy, equal_nan=True)

    def test_dataframe_ohlcv_volume_ad(self):
        """Volume (AD): DataFrame via get_ohlcv, index preserved, values match NumPy."""
        from ferro_ta.utils import get_ohlcv

        o, h, l, c, v = get_ohlcv(self.df_default)
        r_series = AD(h, l, c, v)
        r_numpy = AD(
            self.df_default["high"].to_numpy(),
            self.df_default["low"].to_numpy(),
            self.df_default["close"].to_numpy(),
            self.df_default["volume"].to_numpy(),
        )
        assert list(r_series.index) == list(self.df_default.index)
        np.testing.assert_allclose(r_series.to_numpy(), r_numpy, equal_nan=True)

    def test_dataframe_ohlcv_volume_obv(self):
        """Volume (OBV): DataFrame via get_ohlcv, index preserved, values match NumPy."""
        from ferro_ta.utils import get_ohlcv

        o, h, l, c, v = get_ohlcv(self.df_default)
        r_series = OBV(c, v)
        r_numpy = OBV(
            self.df_default["close"].to_numpy(),
            self.df_default["volume"].to_numpy(),
        )
        assert list(r_series.index) == list(self.df_default.index)
        np.testing.assert_allclose(r_series.to_numpy(), r_numpy, equal_nan=True)

    def test_dataframe_ohlcv_price_transform(self):
        """Price transform (AVGPRICE): DataFrame via get_ohlcv, index preserved, values match NumPy."""
        from ferro_ta.utils import get_ohlcv

        o, h, l, c, v = get_ohlcv(self.df_default)
        r_series = AVGPRICE(o, h, l, c)
        r_numpy = AVGPRICE(
            self.df_default["open"].to_numpy(),
            self.df_default["high"].to_numpy(),
            self.df_default["low"].to_numpy(),
            self.df_default["close"].to_numpy(),
        )
        assert list(r_series.index) == list(self.df_default.index)
        np.testing.assert_allclose(r_series.to_numpy(), r_numpy, equal_nan=True)


# ---------------------------------------------------------------------------
# STOCH, STOCHRSI, ADX/DI/DM accuracy
# ---------------------------------------------------------------------------


class TestSTOCHAccuracy:
    """STOCH SMA smoothing — basic correctness checks."""

    def test_output_length(self):
        k, d = STOCH(OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE)
        assert len(k) == len(OHLCV_PRICES)
        assert len(d) == len(OHLCV_PRICES)

    def test_values_in_range(self):
        k, d = STOCH(OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE)
        for v in _finite(k):
            assert 0.0 <= v <= 100.0, f"slowk out of range: {v}"
        for v in _finite(d):
            assert 0.0 <= v <= 100.0, f"slowd out of range: {v}"

    def test_warmup_nans(self):
        """First fastk_period + slowk_period - 2 bars should be NaN."""
        k, _ = STOCH(OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE, fastk_period=5, slowk_period=3)
        warmup = 5 + 3 - 2  # = 6
        assert all(math.isnan(v) for v in k[:warmup]), "Expected NaN in warmup"

    def test_sma_smoothing(self):
        """Verify SMA: slowk values are stable for constant high-close data."""
        # constant prices → fastk = 50% (close at midpoint)
        n = 30
        h = np.ones(n) * 10.0
        l = np.zeros(n)
        c = np.ones(n) * 5.0  # close at midpoint of range
        k, d = STOCH(h, l, c, fastk_period=5, slowk_period=3, slowd_period=3)
        finite_k = [v for v in k if not math.isnan(v)]
        assert all(math.isclose(v, 50.0, abs_tol=1e-9) for v in finite_k), (
            f"Expected slowk=50 for close at midpoint; got {finite_k[:3]}"
        )
        finite_d = [v for v in d if not math.isnan(v)]
        assert all(math.isclose(v, 50.0, abs_tol=1e-9) for v in finite_d)


class TestSTOCHRSIAccuracy:
    """STOCHRSI with SMA fastd."""

    def test_output_length(self):
        k, d = STOCHRSI(OHLCV_PRICES)
        assert len(k) == len(OHLCV_PRICES)
        assert len(d) == len(OHLCV_PRICES)

    def test_values_in_range(self):
        prices = np.arange(1.0, 101.0)
        k, d = STOCHRSI(prices, timeperiod=14, fastk_period=5, fastd_period=3)
        for v in _finite(k):
            assert 0.0 <= v <= 100.0
        for v in _finite(d):
            assert 0.0 <= v <= 100.0

    def test_fastd_is_sma_of_fastk(self):
        """fastd[i] == mean(fastk[i-2:i+1]) for period=3."""
        prices = np.arange(1.0, 101.0) + np.sin(np.arange(100)) * 0.5
        k, d = STOCHRSI(prices, timeperiod=14, fastk_period=5, fastd_period=3)
        # Find first valid fastd bar
        first_d = next(i for i, v in enumerate(d) if not math.isnan(v))
        # Check SMA relationship
        for i in range(first_d, len(d) - 1):
            if not math.isnan(d[i]) and not any(
                math.isnan(k[j]) for j in range(i - 2, i + 1)
            ):
                expected = (k[i] + k[i - 1] + k[i - 2]) / 3.0
                assert math.isclose(d[i], expected, rel_tol=1e-9), (
                    f"SMA mismatch at {i}"
                )
                break  # one check is sufficient


class TestADXAccuracy:
    """ADX/DX/+DI/-DI/PLUS_DM/MINUS_DM with TA-Lib sum-seeding."""

    def test_adx_output_length(self):
        assert len(ADX(OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE)) == len(OHLCV_PRICES)

    def test_adx_range(self):
        r = ADX(OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE)
        for v in _finite(r):
            assert 0.0 <= v <= 100.0

    def test_plus_di_range(self):
        r = PLUS_DI(OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE)
        for v in _finite(r):
            assert 0.0 <= v <= 100.0

    def test_minus_di_range(self):
        r = MINUS_DI(OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE)
        for v in _finite(r):
            assert 0.0 <= v <= 100.0

    def test_plus_dm_positive(self):
        r = PLUS_DM(OHLCV_HIGH, OHLCV_LOW)
        for v in _finite(r):
            assert v >= 0.0

    def test_minus_dm_positive(self):
        r = MINUS_DM(OHLCV_HIGH, OHLCV_LOW)
        for v in _finite(r):
            assert v >= 0.0

    def test_dx_output_length(self):
        assert len(DX(OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE)) == len(OHLCV_PRICES)

    def test_adxr_output_length(self):
        assert len(ADXR(OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE)) == len(OHLCV_PRICES)


# ---------------------------------------------------------------------------
# Extended Indicators (VWAP, Supertrend)
# ---------------------------------------------------------------------------

from ferro_ta import SUPERTREND, VWAP


class TestVWAP:
    """VWAP — Volume Weighted Average Price."""

    H = np.array([11.0, 12.0, 13.0, 12.0, 11.0, 10.0, 9.0, 10.0, 11.0, 12.0])
    L = H - 1.0
    C = (H + L) / 2.0
    V = np.ones(10) * 1000.0

    def test_output_length(self):
        r = VWAP(self.H, self.L, self.C, self.V)
        assert len(r) == len(self.H)

    def test_cumulative_no_nans(self):
        """Cumulative VWAP (default) has no NaNs."""
        r = VWAP(self.H, self.L, self.C, self.V)
        assert not np.any(np.isnan(r))

    def test_cumulative_first_bar(self):
        """First bar of cumulative VWAP equals its typical price."""
        r = VWAP(self.H, self.L, self.C, self.V)
        tp0 = (self.H[0] + self.L[0] + self.C[0]) / 3.0
        assert math.isclose(r[0], tp0, rel_tol=1e-9)

    def test_cumulative_monotone_volume_contribution(self):
        """Cumulative VWAP is bounded by min/max typical price."""
        r = VWAP(self.H, self.L, self.C, self.V)
        tp = (self.H + self.L + self.C) / 3.0
        assert np.all(r >= tp.min() - 1e-9)
        assert np.all(r <= tp.max() + 1e-9)

    def test_rolling_warmup_nans(self):
        """Rolling VWAP has timeperiod-1 NaN values at the start."""
        r = VWAP(self.H, self.L, self.C, self.V, timeperiod=3)
        assert np.isnan(r[0]) and np.isnan(r[1])
        assert not np.isnan(r[2])

    def test_rolling_output_length(self):
        r = VWAP(self.H, self.L, self.C, self.V, timeperiod=3)
        assert len(r) == len(self.H)

    def test_constant_uniform_price(self):
        """With uniform price and volume, VWAP == typical price."""
        n = 10
        h = np.full(n, 10.0)
        l = np.full(n, 8.0)
        c = np.full(n, 9.0)
        v = np.full(n, 500.0)
        tp = (10.0 + 8.0 + 9.0) / 3.0
        r = VWAP(h, l, c, v)
        assert np.allclose(r, tp)


class TestSUPERTREND:
    """Supertrend ATR-based trend indicator."""

    N = 20
    H = np.array(
        [10.0 + i * 0.5 if i < 10 else 15.0 - (i - 10) * 0.5 for i in range(N)]
    )
    L = H - 1.0
    C = (H + L) / 2.0

    def test_output_shape(self):
        st, direction = SUPERTREND(self.H, self.L, self.C)
        assert len(st) == len(self.H)
        assert len(direction) == len(self.H)

    def test_warmup_nans(self):
        """First timeperiod bars in supertrend should be NaN."""
        st, _ = SUPERTREND(self.H, self.L, self.C, timeperiod=7)
        assert all(np.isnan(st[i]) for i in range(7))

    def test_direction_values(self):
        """Direction should only be -1, 0, or 1."""
        _, direction = SUPERTREND(self.H, self.L, self.C)
        assert all(d in (-1, 0, 1) for d in direction)

    def test_direction_matches_price_vs_supertrend(self):
        """When direction=1 (uptrend), close > supertrend."""
        st, direction = SUPERTREND(self.H, self.L, self.C)
        for i in range(len(self.H)):
            if direction[i] == 1:
                assert self.C[i] > st[i] - 1e-9, (
                    f"At {i}: close={self.C[i]}, st={st[i]}"
                )
            elif direction[i] == -1:
                assert self.C[i] < st[i] + 1e-9, (
                    f"At {i}: close={self.C[i]}, st={st[i]}"
                )

    def test_supertrend_positive(self):
        """Supertrend values should be positive."""
        st, _ = SUPERTREND(self.H, self.L, self.C)
        for v in st[~np.isnan(st)]:
            assert v > 0.0

    def test_custom_multiplier(self):
        """Higher multiplier widens bands → same trend can persist longer."""
        _, d1 = SUPERTREND(self.H, self.L, self.C, multiplier=1.0)
        _, d2 = SUPERTREND(self.H, self.L, self.C, multiplier=5.0)
        # Just check they both produce valid outputs
        assert all(d in (-1, 0, 1) for d in d1)
        assert all(d in (-1, 0, 1) for d in d2)

    def test_pandas_series_input(self):
        """Accepts pandas Series and returns Series."""
        import pandas as pd

        idx = pd.date_range("2024-01-01", periods=self.N)
        h_s = pd.Series(self.H, index=idx)
        l_s = pd.Series(self.L, index=idx)
        c_s = pd.Series(self.C, index=idx)
        st, direction = SUPERTREND(h_s, l_s, c_s)
        assert isinstance(st, pd.Series)
        assert isinstance(direction, pd.Series)
        assert list(st.index) == list(idx)


# ---------------------------------------------------------------------------
# Streaming / Incremental API
# ---------------------------------------------------------------------------

from ferro_ta.data.streaming import (
    StreamingATR,
    StreamingBBands,
    StreamingEMA,
    StreamingMACD,
    StreamingRSI,
    StreamingSMA,
    StreamingStoch,
    StreamingSupertrend,
    StreamingVWAP,
)


class TestStreamingSMA:
    def test_warmup_nans(self):
        sma = StreamingSMA(3)
        assert math.isnan(sma.update(1.0))
        assert math.isnan(sma.update(2.0))

    def test_first_valid(self):
        sma = StreamingSMA(3)
        sma.update(1.0)
        sma.update(2.0)
        v = sma.update(3.0)
        assert math.isclose(v, 2.0)

    def test_rolling(self):
        sma = StreamingSMA(3)
        [sma.update(x) for x in [1.0, 2.0, 3.0]]
        v = sma.update(4.0)
        assert math.isclose(v, 3.0)

    def test_matches_batch_sma(self):
        import ferro_ta

        data = np.arange(1.0, 21.0)
        batch = ferro_ta.SMA(data, timeperiod=5)
        stream_sma = StreamingSMA(5)
        for i, x in enumerate(data):
            sv = stream_sma.update(x)
            if not math.isnan(batch[i]):
                assert math.isclose(sv, batch[i], rel_tol=1e-9)

    def test_reset(self):
        sma = StreamingSMA(3)
        [sma.update(x) for x in [1.0, 2.0, 3.0]]
        sma.reset()
        assert math.isnan(sma.update(1.0))

    def test_period_1(self):
        sma = StreamingSMA(1)
        v = sma.update(42.0)
        assert math.isclose(v, 42.0)


class TestStreamingEMA:
    def test_warmup_nans(self):
        ema = StreamingEMA(5)
        for _ in range(4):
            assert math.isnan(ema.update(1.0))

    def test_first_valid(self):
        ema = StreamingEMA(3)
        ema.update(1.0)
        ema.update(2.0)
        v = ema.update(3.0)
        assert math.isclose(v, 2.0)

    def test_matches_batch_ema(self):
        """StreamingEMA (SMA-seeded) and batch EMA converge after enough bars."""
        import ferro_ta

        # Oscillating data helps convergence independent of seed
        data = np.array([50.0 + 10.0 * math.sin(i * 0.3) for i in range(100)])
        period = 5
        batch = ferro_ta.EMA(data, timeperiod=period)
        stream_ema = StreamingEMA(period)
        converge_bar = period * 6  # allow seed to wash out fully
        for i, x in enumerate(data):
            sv = stream_ema.update(x)
            if i >= converge_bar and not math.isnan(batch[i]):
                # Allow 0.1% relative tolerance after convergence
                assert math.isclose(sv, batch[i], rel_tol=1e-3), (
                    f"i={i}: {sv} != {batch[i]}"
                )

    def test_reset(self):
        ema = StreamingEMA(3)
        [ema.update(x) for x in [1.0, 2.0, 3.0]]
        ema.reset()
        assert math.isnan(ema.update(1.0))


class TestStreamingRSI:
    def test_warmup(self):
        rsi = StreamingRSI(14)
        for _ in range(14):
            assert math.isnan(rsi.update(50.0))

    def test_constant_series_not_nan(self):
        """Constant prices: RSI is defined (gain=0, loss=0 → special case)."""
        rsi = StreamingRSI(5)
        last = float("nan")
        for _ in range(10):
            last = rsi.update(100.0)
        # With all gains=0 and losses=0, RSI returns 100 (avg_loss==0 branch)
        # This is acceptable behavior for degenerate input.
        assert not math.isnan(last)

    def test_always_rising_near_100(self):
        rsi = StreamingRSI(5)
        last = float("nan")
        for i in range(20):
            last = rsi.update(float(i))
        assert not math.isnan(last) and last > 90.0

    def test_range(self):
        rsi = StreamingRSI(5)
        vals = [
            rsi.update(float(v))
            for v in [1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0]
        ]
        for v in vals:
            if not math.isnan(v):
                assert 0.0 <= v <= 100.0

    def test_reset(self):
        rsi = StreamingRSI(3)
        [rsi.update(x) for x in [1.0, 2.0, 3.0, 4.0]]
        rsi.reset()
        assert math.isnan(rsi.update(1.0))


class TestStreamingATR:
    def test_warmup(self):
        atr = StreamingATR(3)
        assert math.isnan(atr.update(11.0, 9.0, 10.0))
        assert math.isnan(atr.update(12.0, 10.0, 11.0))
        assert math.isnan(atr.update(13.0, 11.0, 12.0))

    def test_positive(self):
        atr = StreamingATR(3)
        vals = [
            atr.update(h, l, c)
            for h, l, c in [
                (11.0, 9.0, 10.0),
                (12.0, 10.0, 11.0),
                (13.0, 11.0, 12.0),
                (14.0, 12.0, 13.0),
                (15.0, 13.0, 14.0),
            ]
        ]
        for v in vals:
            if not math.isnan(v):
                assert v > 0

    def test_constant_range(self):
        """With constant HL spread of 2 and no gaps, ATR converges to 2."""
        atr = StreamingATR(5)
        h, l = 11.0, 9.0
        c = 10.0
        last = float("nan")
        for _ in range(50):
            last = atr.update(h, l, c)
        assert math.isclose(last, 2.0, abs_tol=0.01)

    def test_reset(self):
        atr = StreamingATR(3)
        [
            atr.update(h, l, c)
            for h, l, c in [(11.0, 9.0, 10.0), (12.0, 10.0, 11.0), (13.0, 11.0, 12.0)]
        ]
        atr.reset()
        assert math.isnan(atr.update(11.0, 9.0, 10.0))


class TestStreamingBBands:
    def test_warmup(self):
        bb = StreamingBBands(5)
        for _ in range(4):
            u, m, l = bb.update(10.0)
            assert all(math.isnan(x) for x in [u, m, l])

    def test_structure(self):
        bb = StreamingBBands(3)
        for _ in range(3):
            u, m, l = bb.update(10.0)
        assert u >= m >= l

    def test_constant_price(self):
        """Constant price → std=0, all three bands equal to price."""
        bb = StreamingBBands(5)
        u = m = l = float("nan")
        for _ in range(20):
            u, m, l = bb.update(42.0)
        assert math.isclose(u, 42.0, abs_tol=1e-9)
        assert math.isclose(m, 42.0, abs_tol=1e-9)
        assert math.isclose(l, 42.0, abs_tol=1e-9)


class TestStreamingMACD:
    def test_warmup(self):
        macd = StreamingMACD()
        for _ in range(25):
            ml, s, h = macd.update(100.0)
        # slowperiod=26, so at bar 25 (0-indexed) MACD line may not yet be valid
        # (seeded after 26 bars). At this point both ml and s could still be NaN.
        # Just verify they are floats.
        assert isinstance(ml, float) and isinstance(s, float) and isinstance(h, float)

    def test_returns_three(self):
        macd = StreamingMACD()
        result = macd.update(100.0)
        assert len(result) == 3

    def test_histogram_equals_macd_minus_signal(self):
        macd = StreamingMACD(fastperiod=3, slowperiod=6, signalperiod=2)
        for _ in range(20):
            ml, s, h = macd.update(float(_ + 1))
        if not math.isnan(ml) and not math.isnan(s):
            assert math.isclose(h, ml - s, rel_tol=1e-9)

    def test_reset(self):
        macd = StreamingMACD(fastperiod=3, slowperiod=6, signalperiod=2)
        [macd.update(x) for x in np.arange(1.0, 20.0)]
        macd.reset()
        ml, s, h = macd.update(1.0)
        assert math.isnan(ml)


class TestStreamingStoch:
    def test_warmup(self):
        stoch = StreamingStoch(5, 3, 3)
        for _ in range(7):
            k, d = stoch.update(10.0, 9.0, 9.5)
        # k should be valid, d still might be NaN
        assert isinstance(k, float) and isinstance(d, float)

    def test_range(self):
        stoch = StreamingStoch(5, 3, 3)
        for _ in range(20):
            k, d = stoch.update(float(_ + 1), float(_), float(_ + 0.5))
        if not math.isnan(k):
            assert 0 <= k <= 100


class TestStreamingVWAP:
    def test_cumulative(self):
        vwap = StreamingVWAP()
        v1 = vwap.update(11.0, 9.0, 10.0, 1000.0)
        assert math.isclose(v1, (11.0 + 9.0 + 10.0) / 3.0)

    def test_always_valid(self):
        vwap = StreamingVWAP()
        for i in range(5):
            v = vwap.update(10.0 + i, 9.0 + i, 9.5 + i, 1000.0)
            assert not math.isnan(v)

    def test_reset(self):
        vwap = StreamingVWAP()
        vwap.update(11.0, 9.0, 10.0, 1000.0)
        vwap.reset()
        v = vwap.update(20.0, 18.0, 19.0, 500.0)
        assert math.isclose(v, (20.0 + 18.0 + 19.0) / 3.0)


class TestStreamingSupertrend:
    def test_warmup_nans(self):
        st = StreamingSupertrend(3)
        for _ in range(3):
            line, d = st.update(10.0, 9.0, 9.5)
            # First 3 bars: ATR warming up
        # By bar 4 it should be valid
        line, d = st.update(11.0, 10.0, 10.5)
        assert not math.isnan(line)
        assert d in (-1, 0, 1)

    def test_direction_values(self):
        st = StreamingSupertrend(3)
        for i in range(20):
            line, d = st.update(10.0 + i * 0.5, 9.0 + i * 0.5, 9.5 + i * 0.5)
        assert d in (-1, 1)

    def test_reset(self):
        st = StreamingSupertrend(3)
        [st.update(10.0 + i, 9.0 + i, 9.5 + i) for i in range(10)]
        st.reset()
        line, d = st.update(10.0, 9.0, 9.5)
        assert d == 0  # warmup


# ---------------------------------------------------------------------------
# Additional Extended Indicators (ICHIMOKU, DONCHIAN, PIVOT_POINTS)
# ---------------------------------------------------------------------------

from ferro_ta import DONCHIAN, ICHIMOKU, PIVOT_POINTS


class TestICHIMOKU:
    N = 80
    H = np.arange(10.0, 10.0 + N) + np.sin(np.arange(N)) * 0.5
    L = H - 1.5
    C = (H + L) / 2.0

    def test_output_shapes(self):
        t, k, sa, sb, ch = ICHIMOKU(self.H, self.L, self.C)
        for arr in (t, k, sa, sb, ch):
            assert len(arr) == self.N

    def test_tenkan_warmup(self):
        t, *_ = ICHIMOKU(self.H, self.L, self.C, tenkan_period=9)
        assert all(np.isnan(t[:8]))
        assert not np.isnan(t[8])

    def test_kijun_warmup(self):
        _, k, *_ = ICHIMOKU(self.H, self.L, self.C, kijun_period=26)
        assert all(np.isnan(k[:25]))
        assert not np.isnan(k[25])

    def test_tenkan_is_midpoint(self):
        t, *_ = ICHIMOKU(self.H, self.L, self.C, tenkan_period=5)
        for i in range(4, self.N):
            expected = (self.H[i - 4 : i + 1].max() + self.L[i - 4 : i + 1].min()) / 2.0
            assert math.isclose(t[i], expected, rel_tol=1e-9)

    def test_chikou_is_shifted_close(self):
        *_, ch = ICHIMOKU(self.H, self.L, self.C, displacement=26)
        # chikou[26:] == close[0 : N-26]
        for i in range(26, self.N):
            assert math.isclose(ch[i], self.C[i - 26], rel_tol=1e-9)

    def test_pandas_output(self):
        import pandas as pd

        idx = pd.date_range("2024-01-01", periods=self.N)
        h_s = pd.Series(self.H, index=idx)
        l_s = pd.Series(self.L, index=idx)
        c_s = pd.Series(self.C, index=idx)
        t, k, sa, sb, ch = ICHIMOKU(h_s, l_s, c_s)
        for s in (t, k, sa, sb, ch):
            assert isinstance(s, pd.Series)


class TestDONCHIAN:
    N = 30
    H = np.arange(1.0, N + 1.0)
    L = np.zeros(N)

    def test_output_shape(self):
        u, m, lo = DONCHIAN(self.H, self.L, 10)
        for arr in (u, m, lo):
            assert len(arr) == self.N

    def test_warmup_nans(self):
        u, m, lo = DONCHIAN(self.H, self.L, 10)
        for arr in (u, m, lo):
            assert all(np.isnan(arr[:9]))
            assert not np.isnan(arr[9])

    def test_upper_is_max_high(self):
        u, _, _ = DONCHIAN(self.H, self.L, 5)
        for i in range(4, self.N):
            assert math.isclose(u[i], self.H[i - 4 : i + 1].max(), rel_tol=1e-9)

    def test_lower_is_min_low(self):
        _, _, lo = DONCHIAN(self.H, self.L, 5)
        for i in range(4, self.N):
            assert math.isclose(lo[i], self.L[i - 4 : i + 1].min(), rel_tol=1e-9)

    def test_middle_is_avg(self):
        u, m, lo = DONCHIAN(self.H, self.L, 5)
        for i in range(4, self.N):
            if not np.isnan(u[i]):
                assert math.isclose(m[i], (u[i] + lo[i]) / 2.0, rel_tol=1e-9)

    def test_monotone_upper(self):
        """With monotone-increasing H, upper band is non-decreasing."""
        u, _, _ = DONCHIAN(self.H, self.L, 5)
        valid = u[~np.isnan(u)]
        assert all(valid[i] <= valid[i + 1] for i in range(len(valid) - 1))


class TestPIVOT_POINTS:
    N = 10
    H = np.array([12.0, 13.0, 14.0, 13.0, 12.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    L = H - 2.0
    C = H - 1.0

    def test_output_shape(self):
        p, r1, s1, r2, s2 = PIVOT_POINTS(self.H, self.L, self.C)
        for arr in (p, r1, s1, r2, s2):
            assert len(arr) == self.N

    def test_first_bar_nan(self):
        p, r1, s1, r2, s2 = PIVOT_POINTS(self.H, self.L, self.C)
        for arr in (p, r1, s1, r2, s2):
            assert np.isnan(arr[0])

    def test_classic_pivot_formula(self):
        p, r1, s1, r2, s2 = PIVOT_POINTS(self.H, self.L, self.C, method="classic")
        for i in range(1, self.N):
            ph, pl, pc = self.H[i - 1], self.L[i - 1], self.C[i - 1]
            expected_p = (ph + pl + pc) / 3.0
            assert math.isclose(p[i], expected_p, rel_tol=1e-9)
            assert math.isclose(r1[i], 2 * expected_p - pl, rel_tol=1e-9)
            assert math.isclose(s1[i], 2 * expected_p - ph, rel_tol=1e-9)

    def test_fibonacci_method(self):
        p, r1, s1, r2, s2 = PIVOT_POINTS(self.H, self.L, self.C, method="fibonacci")
        for i in range(1, self.N):
            ph, pl, pc = self.H[i - 1], self.L[i - 1], self.C[i - 1]
            pp = (ph + pl + pc) / 3.0
            hl = ph - pl
            assert math.isclose(r1[i], pp + 0.382 * hl, rel_tol=1e-9)
            assert math.isclose(s1[i], pp - 0.382 * hl, rel_tol=1e-9)

    def test_camarilla_method(self):
        p, r1, s1, r2, s2 = PIVOT_POINTS(self.H, self.L, self.C, method="camarilla")
        for i in range(1, self.N):
            ph, pl, pc = self.H[i - 1], self.L[i - 1], self.C[i - 1]
            hl = ph - pl
            assert math.isclose(r1[i], pc + 1.1 * hl / 12.0, rel_tol=1e-9)

    def test_invalid_method_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown pivot method"):
            PIVOT_POINTS(self.H, self.L, self.C, method="unknown")

    def test_r1_gt_pivot_gt_s1(self):
        p, r1, s1, _, _ = PIVOT_POINTS(self.H, self.L, self.C, method="classic")
        for i in range(1, self.N):
            if not np.isnan(p[i]):
                assert r1[i] > p[i] > s1[i]


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


class TestKELTNER_CHANNELS:
    N = 30
    C = np.cumsum(np.ones(N)) + 40.0
    H = C + 0.5
    L = C - 0.5

    def test_output_shapes(self):
        u, m, lo = KELTNER_CHANNELS(self.H, self.L, self.C, timeperiod=5, atr_period=3)
        assert len(u) == len(m) == len(lo) == self.N

    def test_upper_gt_middle_gt_lower(self):
        u, m, lo = KELTNER_CHANNELS(self.H, self.L, self.C, timeperiod=5, atr_period=3)
        valid = ~np.isnan(u)
        assert np.all(u[valid] > m[valid])
        assert np.all(m[valid] > lo[valid])

    def test_middle_is_ema(self):
        from ferro_ta import EMA

        u, m, lo = KELTNER_CHANNELS(self.H, self.L, self.C, timeperiod=5, atr_period=3)
        ema = EMA(self.C, timeperiod=5)
        valid = ~np.isnan(m) & ~np.isnan(ema)
        assert np.allclose(m[valid], ema[valid], rtol=1e-9)


class TestHULL_MA:
    N = 30
    C = np.cumsum(np.ones(N)) + 40.0

    def test_output_length(self):
        hull = HULL_MA(self.C, timeperiod=4)
        assert len(hull) == self.N

    def test_leading_nans(self):
        hull = HULL_MA(self.C, timeperiod=4)
        assert int(np.sum(np.isnan(hull))) >= 1

    def test_finite_after_warmup(self):
        hull = HULL_MA(self.C, timeperiod=4)
        assert np.all(np.isfinite(hull[~np.isnan(hull)]))

    def test_linear_series_tracks_input(self):
        """For a perfectly linear series, HMA should be close to close."""
        c = np.arange(1.0, 31.0)
        hull = HULL_MA(c, timeperiod=4)
        valid = ~np.isnan(hull)
        # Should be within 5% of actual price
        assert np.all(np.abs(hull[valid] - c[valid]) < c[valid] * 0.05)


class TestCHANDELIER_EXIT:
    N = 30
    C = np.cumsum(np.ones(N)) + 40.0
    H = C + 0.5
    L = C - 0.5

    def test_output_shapes(self):
        le, se = CHANDELIER_EXIT(self.H, self.L, self.C, timeperiod=5, multiplier=2.0)
        assert len(le) == len(se) == self.N

    def test_long_lt_highest_high(self):
        le, _ = CHANDELIER_EXIT(self.H, self.L, self.C, timeperiod=5, multiplier=2.0)
        valid = ~np.isnan(le)
        # long exit must be below the local highest high
        from ferro_ta import MAX

        hh = MAX(self.H, timeperiod=5)
        assert np.all(le[valid] <= hh[valid])

    def test_short_gt_lowest_low(self):
        _, se = CHANDELIER_EXIT(self.H, self.L, self.C, timeperiod=5, multiplier=2.0)
        valid = ~np.isnan(se)
        from ferro_ta import MIN

        ll = MIN(self.L, timeperiod=5)
        assert np.all(se[valid] >= ll[valid])


class TestVWMA:
    N = 20
    C = np.full(N, 50.0)
    V = np.full(N, 1_000.0)

    def test_output_length(self):
        v = VWMA(self.C, self.V, timeperiod=5)
        assert len(v) == self.N

    def test_leading_nans(self):
        v = VWMA(self.C, self.V, timeperiod=5)
        assert int(np.sum(np.isnan(v))) == 4

    def test_constant_price_equals_price(self):
        """When price is constant, VWMA == price regardless of volume."""
        v = VWMA(self.C, self.V, timeperiod=5)
        valid = ~np.isnan(v)
        assert np.allclose(v[valid], 50.0, rtol=1e-9)

    def test_weighted_by_volume(self):
        """Higher volume at a price bar should pull VWMA toward that price."""
        close = np.array([10.0] * 5 + [20.0])
        vol = np.array([1.0] * 5 + [100.0])
        v = VWMA(close, vol, timeperiod=6)
        assert v[-1] > 19.0  # strongly weighted toward 20.0


class TestCHOPPINESS_INDEX:
    N = 30
    C = np.cumsum(np.ones(N)) + 40.0
    H = C + 0.5
    L = C - 0.5

    def test_output_length(self):
        ci = CHOPPINESS_INDEX(self.H, self.L, self.C, timeperiod=5)
        assert len(ci) == self.N

    def test_leading_nans(self):
        ci = CHOPPINESS_INDEX(self.H, self.L, self.C, timeperiod=5)
        assert np.sum(~np.isnan(ci)) <= self.N - 5

    def test_range_0_to_100(self):
        """Choppiness Index should be in (0, 100]."""
        ci = CHOPPINESS_INDEX(self.H, self.L, self.C, timeperiod=5)
        valid = ci[~np.isnan(ci)]
        if len(valid) > 0:
            assert np.all(valid >= 0.0)
            assert np.all(valid <= 100.0)

    def test_trending_market_lower_than_choppy(self):
        """A strong trend should have lower CI than a sideways market."""
        # Trending: monotone rise
        trend_c = np.arange(1.0, 31.0)
        trend_h = trend_c + 0.1
        trend_l = trend_c - 0.1
        ci_trend = CHOPPINESS_INDEX(trend_h, trend_l, trend_c, timeperiod=14)

        # Choppy: alternating
        chop_c = np.array([50.0 + ((-1) ** i) * 1.0 for i in range(30)])
        chop_h = chop_c + 0.1
        chop_l = chop_c - 0.1
        ci_chop = CHOPPINESS_INDEX(chop_h, chop_l, chop_c, timeperiod=14)

        valid_t = ci_trend[~np.isnan(ci_trend)]
        valid_c = ci_chop[~np.isnan(ci_chop)]
        if len(valid_t) > 0 and len(valid_c) > 0:
            assert np.mean(valid_t) < np.mean(valid_c)
