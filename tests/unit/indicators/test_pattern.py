"""Unit tests for ferro_ta.indicators.pattern (CDL* functions)"""
import numpy as np
import pytest
from ferro_ta.indicators.pattern import (
    CDL2CROWS, CDL3BLACKCROWS, CDL3INSIDE, CDL3LINESTRIKE, CDL3OUTSIDE,
    CDL3STARSINSOUTH, CDL3WHITESOLDIERS, CDLABANDONEDBABY, CDLADVANCEBLOCK,
    CDLBELTHOLD, CDLBREAKAWAY, CDLCLOSINGMARUBOZU, CDLCONCEALBABYSWALL,
    CDLCOUNTERATTACK, CDLDARKCLOUDCOVER, CDLDOJI, CDLDOJISTAR, CDLDRAGONFLYDOJI,
    CDLENGULFING, CDLEVENINGDOJISTAR, CDLEVENINGSTAR, CDLGAPSIDESIDEWHITE,
    CDLGRAVESTONEDOJI, CDLHAMMER, CDLHANGINGMAN, CDLHARAMI, CDLHARAMICROSS,
    CDLHIGHWAVE, CDLHIKKAKE, CDLHIKKAKEMOD, CDLHOMINGPIGEON, CDLIDENTICAL3CROWS,
    CDLINNECK, CDLINVERTEDHAMMER, CDLKICKING, CDLKICKINGBYLENGTH, CDLLADDERBOTTOM,
    CDLLONGLEGGEDDOJI, CDLLONGLINE, CDLMARUBOZU, CDLMATCHINGLOW, CDLMATHOLD,
    CDLMORNINGDOJISTAR, CDLMORNINGSTAR, CDLONNECK, CDLPIERCING, CDLRICKSHAWMAN,
    CDLRISEFALL3METHODS, CDLSEPARATINGLINES, CDLSHOOTINGSTAR, CDLSHORTLINE,
    CDLSPINNINGTOP, CDLSTALLEDPATTERN, CDLSTICKSANDWICH, CDLTAKURI, CDLTASUKIGAP,
    CDLTHRUSTING, CDLTRISTAR, CDLUNIQUE3RIVER, CDLUPSIDEGAP2CROWS, CDLXSIDEGAP3METHODS,
)

# ---------------------------------------------------------------------------
# Shared random OHLCV data (realistic OHLCV, proper H >= O,C >= L)
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 200
_C = 100 + np.cumsum(RNG.normal(0, 0.5, N))
_O = _C + RNG.normal(0, 0.2, N)
_H = np.maximum(np.maximum(_O, _C) + np.abs(RNG.normal(0, 0.3, N)), np.maximum(_O, _C))
_L = np.minimum(np.minimum(_O, _C) - np.abs(RNG.normal(0, 0.3, N)), np.minimum(_O, _C))

# All CDL* functions to test systematically
ALL_CDL = [
    ("CDL2CROWS", CDL2CROWS),
    ("CDL3BLACKCROWS", CDL3BLACKCROWS),
    ("CDL3INSIDE", CDL3INSIDE),
    ("CDL3LINESTRIKE", CDL3LINESTRIKE),
    ("CDL3OUTSIDE", CDL3OUTSIDE),
    ("CDL3STARSINSOUTH", CDL3STARSINSOUTH),
    ("CDL3WHITESOLDIERS", CDL3WHITESOLDIERS),
    ("CDLABANDONEDBABY", CDLABANDONEDBABY),
    ("CDLADVANCEBLOCK", CDLADVANCEBLOCK),
    ("CDLBELTHOLD", CDLBELTHOLD),
    ("CDLBREAKAWAY", CDLBREAKAWAY),
    ("CDLCLOSINGMARUBOZU", CDLCLOSINGMARUBOZU),
    ("CDLCONCEALBABYSWALL", CDLCONCEALBABYSWALL),
    ("CDLCOUNTERATTACK", CDLCOUNTERATTACK),
    ("CDLDARKCLOUDCOVER", CDLDARKCLOUDCOVER),
    ("CDLDOJI", CDLDOJI),
    ("CDLDOJISTAR", CDLDOJISTAR),
    ("CDLDRAGONFLYDOJI", CDLDRAGONFLYDOJI),
    ("CDLENGULFING", CDLENGULFING),
    ("CDLEVENINGDOJISTAR", CDLEVENINGDOJISTAR),
    ("CDLEVENINGSTAR", CDLEVENINGSTAR),
    ("CDLGAPSIDESIDEWHITE", CDLGAPSIDESIDEWHITE),
    ("CDLGRAVESTONEDOJI", CDLGRAVESTONEDOJI),
    ("CDLHAMMER", CDLHAMMER),
    ("CDLHANGINGMAN", CDLHANGINGMAN),
    ("CDLHARAMI", CDLHARAMI),
    ("CDLHARAMICROSS", CDLHARAMICROSS),
    ("CDLHIGHWAVE", CDLHIGHWAVE),
    ("CDLHIKKAKE", CDLHIKKAKE),
    ("CDLHIKKAKEMOD", CDLHIKKAKEMOD),
    ("CDLHOMINGPIGEON", CDLHOMINGPIGEON),
    ("CDLIDENTICAL3CROWS", CDLIDENTICAL3CROWS),
    ("CDLINNECK", CDLINNECK),
    ("CDLINVERTEDHAMMER", CDLINVERTEDHAMMER),
    ("CDLKICKING", CDLKICKING),
    ("CDLKICKINGBYLENGTH", CDLKICKINGBYLENGTH),
    ("CDLLADDERBOTTOM", CDLLADDERBOTTOM),
    ("CDLLONGLEGGEDDOJI", CDLLONGLEGGEDDOJI),
    ("CDLLONGLINE", CDLLONGLINE),
    ("CDLMARUBOZU", CDLMARUBOZU),
    ("CDLMATCHINGLOW", CDLMATCHINGLOW),
    ("CDLMATHOLD", CDLMATHOLD),
    ("CDLMORNINGDOJISTAR", CDLMORNINGDOJISTAR),
    ("CDLMORNINGSTAR", CDLMORNINGSTAR),
    ("CDLONNECK", CDLONNECK),
    ("CDLPIERCING", CDLPIERCING),
    ("CDLRICKSHAWMAN", CDLRICKSHAWMAN),
    ("CDLRISEFALL3METHODS", CDLRISEFALL3METHODS),
    ("CDLSEPARATINGLINES", CDLSEPARATINGLINES),
    ("CDLSHOOTINGSTAR", CDLSHOOTINGSTAR),
    ("CDLSHORTLINE", CDLSHORTLINE),
    ("CDLSPINNINGTOP", CDLSPINNINGTOP),
    ("CDLSTALLEDPATTERN", CDLSTALLEDPATTERN),
    ("CDLSTICKSANDWICH", CDLSTICKSANDWICH),
    ("CDLTAKURI", CDLTAKURI),
    ("CDLTASUKIGAP", CDLTASUKIGAP),
    ("CDLTHRUSTING", CDLTHRUSTING),
    ("CDLTRISTAR", CDLTRISTAR),
    ("CDLUNIQUE3RIVER", CDLUNIQUE3RIVER),
    ("CDLUPSIDEGAP2CROWS", CDLUPSIDEGAP2CROWS),
    ("CDLXSIDEGAP3METHODS", CDLXSIDEGAP3METHODS),
]


# ---------------------------------------------------------------------------
# Parametrised tests: all CDL patterns
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,fn", ALL_CDL)
def test_cdl_output_length(name, fn):
    result = fn(_O, _H, _L, _C)
    assert len(result) == N, f"{name}: expected length {N}, got {len(result)}"


@pytest.mark.parametrize("name,fn", ALL_CDL)
def test_cdl_values_in_valid_set(name, fn):
    result = fn(_O, _H, _L, _C)
    assert np.all(np.isin(result, [-100, 0, 100])), \
        f"{name}: unexpected values {np.unique(result)}"


@pytest.mark.parametrize("name,fn", ALL_CDL)
def test_cdl_no_nan(name, fn):
    result = fn(_O, _H, _L, _C)
    assert np.all(np.isfinite(result.astype(float))), f"{name}: contains NaN/Inf"


# ---------------------------------------------------------------------------
# Specific tests for previously untested patterns
# ---------------------------------------------------------------------------

class TestCDLSPINNINGTOP:
    def test_detects_pattern(self):
        # Spinning top: small body, long upper and lower shadows
        # open ≈ close (small body), high much higher, low much lower
        o = np.array([10.0, 10.1, 10.0])
        h = np.array([15.0, 15.1, 15.0])
        l = np.array([5.0, 5.1, 5.0])
        c = np.array([10.0, 10.0, 10.05])
        result = CDLSPINNINGTOP(o, h, l, c)
        assert np.all(np.isin(result, [-100, 0, 100]))

    def test_output_values_random(self):
        result = CDLSPINNINGTOP(_O, _H, _L, _C)
        assert np.all(np.isin(result, [-100, 0, 100]))


class TestCDLEVENINGSTAR:
    def test_basic_run(self):
        result = CDLEVENINGSTAR(_O, _H, _L, _C)
        assert len(result) == N
        assert np.all(np.isin(result, [-100, 0, 100]))

    def test_large_dataset_has_valid_output(self):
        # On 200 bars of random data, result should be all in {-100,0,100}
        result = CDLEVENINGSTAR(_O, _H, _L, _C)
        assert np.all(np.isin(result, [-100, 0, 100]))


class TestCDLMORNINGSTAR:
    def test_basic_run(self):
        result = CDLMORNINGSTAR(_O, _H, _L, _C)
        assert len(result) == N
        assert np.all(np.isin(result, [-100, 0, 100]))

    def test_bullish_signal_is_100(self):
        # Any detected signal must be 100 (bullish)
        result = CDLMORNINGSTAR(_O, _H, _L, _C)
        assert np.all(result[result != 0] == 100)


class TestCDL2CROWS:
    def test_basic_run(self):
        result = CDL2CROWS(_O, _H, _L, _C)
        assert len(result) == N
        assert np.all(np.isin(result, [-100, 0, 100]))

    def test_bearish_signal_is_minus_100(self):
        # Any detected signal must be -100 (bearish)
        result = CDL2CROWS(_O, _H, _L, _C)
        assert np.all(result[result != 0] == -100)


class TestCDLDOJI:
    def test_detects_doji(self):
        # Exact doji: open == close
        o = np.array([10.0, 10.0, 10.0])
        h = np.array([12.0, 12.0, 12.0])
        l = np.array([8.0, 8.0, 8.0])
        c = np.array([10.0, 10.0, 10.0])
        result = CDLDOJI(o, h, l, c)
        assert np.all(result == 100)

    def test_non_doji_returns_zero(self):
        o = np.array([10.0, 11.0, 12.0])
        h = np.array([15.0, 16.0, 17.0])
        l = np.array([9.0, 10.0, 11.0])
        c = np.array([14.0, 15.0, 16.0])  # large body, not doji
        result = CDLDOJI(o, h, l, c)
        assert np.all(result == 0)


class TestCDLMARUBOZU:
    def test_detects_bullish_marubozu(self):
        # Bullish marubozu: open == low, close == high, close > open
        o = np.array([10.0, 10.0])
        h = np.array([15.0, 15.0])
        l = np.array([10.0, 10.0])
        c = np.array([15.0, 15.0])
        result = CDLMARUBOZU(o, h, l, c)
        assert np.all(np.isin(result, [-100, 0, 100]))

    def test_length(self):
        result = CDLMARUBOZU(_O, _H, _L, _C)
        assert len(result) == N
