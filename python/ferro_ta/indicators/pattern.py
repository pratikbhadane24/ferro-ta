"""
Pattern Recognition — Candlestick pattern detection.

All functions return an integer array where:
  100  = bullish signal
  -100 = bearish signal
  0    = no pattern detected

Functions
---------
CDL2CROWS          — Two Crows (bearish)
CDL3BLACKCROWS     — Three Black Crows (bearish)
CDL3WHITESOLDIERS  — Three White Soldiers (bullish)
CDL3INSIDE         — Three Inside Up/Down
CDL3OUTSIDE        — Three Outside Up/Down
CDLDOJI            — Doji
CDLDOJISTAR        — Doji Star
CDLENGULFING       — Engulfing Pattern
CDLHAMMER          — Hammer (bullish)
CDLHARAMI          — Harami Pattern
CDLHARAMICROSS     — Harami Cross Pattern
CDLMARUBOZU        — Marubozu
CDLMORNINGSTAR     — Morning Star (bullish, 3-candle)
CDLMORNINGDOJISTAR — Morning Doji Star (bullish, 3-candle)
CDLEVENINGSTAR     — Evening Star (bearish, 3-candle)
CDLEVENINGDOJISTAR — Evening Doji Star (bearish, 3-candle)
CDLSHOOTINGSTAR    — Shooting Star (bearish)
CDLSPINNINGTOP     — Spinning Top
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ferro_ta._ferro_ta import (
    cdl2crows as _cdl2crows,
)
from ferro_ta._ferro_ta import (
    cdl3blackcrows as _cdl3blackcrows,
)
from ferro_ta._ferro_ta import (
    cdl3inside as _cdl3inside,
)
from ferro_ta._ferro_ta import (
    cdl3linestrike as _cdl3linestrike,
)
from ferro_ta._ferro_ta import (
    cdl3outside as _cdl3outside,
)
from ferro_ta._ferro_ta import (
    cdl3starsinsouth as _cdl3starsinsouth,
)
from ferro_ta._ferro_ta import (
    cdl3whitesoldiers as _cdl3whitesoldiers,
)
from ferro_ta._ferro_ta import (
    cdlabandonedbaby as _cdlabandonedbaby,
)
from ferro_ta._ferro_ta import (
    cdladvanceblock as _cdladvanceblock,
)
from ferro_ta._ferro_ta import (
    cdlbelthold as _cdlbelthold,
)
from ferro_ta._ferro_ta import (
    cdlbreakaway as _cdlbreakaway,
)
from ferro_ta._ferro_ta import (
    cdlclosingmarubozu as _cdlclosingmarubozu,
)
from ferro_ta._ferro_ta import (
    cdlconcealbabyswall as _cdlconcealbabyswall,
)
from ferro_ta._ferro_ta import (
    cdlcounterattack as _cdlcounterattack,
)
from ferro_ta._ferro_ta import (
    cdldarkcloudcover as _cdldarkcloudcover,
)
from ferro_ta._ferro_ta import (
    cdldoji as _cdldoji,
)
from ferro_ta._ferro_ta import (
    cdldojistar as _cdldojistar,
)
from ferro_ta._ferro_ta import (
    cdldragonflydoji as _cdldragonflydoji,
)
from ferro_ta._ferro_ta import (
    cdlengulfing as _cdlengulfing,
)
from ferro_ta._ferro_ta import (
    cdleveningdojistar as _cdleveningdojistar,
)
from ferro_ta._ferro_ta import (
    cdleveningstar as _cdleveningstar,
)
from ferro_ta._ferro_ta import (
    cdlgapsidesidewhite as _cdlgapsidesidewhite,
)
from ferro_ta._ferro_ta import (
    cdlgravestonedoji as _cdlgravestonedoji,
)
from ferro_ta._ferro_ta import (
    cdlhammer as _cdlhammer,
)
from ferro_ta._ferro_ta import (
    cdlhangingman as _cdlhangingman,
)
from ferro_ta._ferro_ta import (
    cdlharami as _cdlharami,
)
from ferro_ta._ferro_ta import (
    cdlharamicross as _cdlharamicross,
)
from ferro_ta._ferro_ta import (
    cdlhighwave as _cdlhighwave,
)
from ferro_ta._ferro_ta import (
    cdlhikkake as _cdlhikkake,
)
from ferro_ta._ferro_ta import (
    cdlhikkakemod as _cdlhikkakemod,
)
from ferro_ta._ferro_ta import (
    cdlhomingpigeon as _cdlhomingpigeon,
)
from ferro_ta._ferro_ta import (
    cdlidentical3crows as _cdlidentical3crows,
)
from ferro_ta._ferro_ta import (
    cdlinneck as _cdlinneck,
)
from ferro_ta._ferro_ta import (
    cdlinvertedhammer as _cdlinvertedhammer,
)
from ferro_ta._ferro_ta import (
    cdlkicking as _cdlkicking,
)
from ferro_ta._ferro_ta import (
    cdlkickingbylength as _cdlkickingbylength,
)
from ferro_ta._ferro_ta import (
    cdlladderbottom as _cdlladderbottom,
)
from ferro_ta._ferro_ta import (
    cdllongleggeddoji as _cdllongleggeddoji,
)
from ferro_ta._ferro_ta import (
    cdllongline as _cdllongline,
)
from ferro_ta._ferro_ta import (
    cdlmarubozu as _cdlmarubozu,
)
from ferro_ta._ferro_ta import (
    cdlmatchinglow as _cdlmatchinglow,
)
from ferro_ta._ferro_ta import (
    cdlmathold as _cdlmathold,
)
from ferro_ta._ferro_ta import (
    cdlmorningdojistar as _cdlmorningdojistar,
)
from ferro_ta._ferro_ta import (
    cdlmorningstar as _cdlmorningstar,
)
from ferro_ta._ferro_ta import (
    cdlonneck as _cdlonneck,
)
from ferro_ta._ferro_ta import (
    cdlpiercing as _cdlpiercing,
)
from ferro_ta._ferro_ta import (
    cdlrickshawman as _cdlrickshawman,
)
from ferro_ta._ferro_ta import (
    cdlrisefall3methods as _cdlrisefall3methods,
)
from ferro_ta._ferro_ta import (
    cdlseparatinglines as _cdlseparatinglines,
)
from ferro_ta._ferro_ta import (
    cdlshootingstar as _cdlshootingstar,
)
from ferro_ta._ferro_ta import (
    cdlshortline as _cdlshortline,
)
from ferro_ta._ferro_ta import (
    cdlspinningtop as _cdlspinningtop,
)
from ferro_ta._ferro_ta import (
    cdlstalledpattern as _cdlstalledpattern,
)
from ferro_ta._ferro_ta import (
    cdlsticksandwich as _cdlsticksandwich,
)
from ferro_ta._ferro_ta import (
    cdltakuri as _cdltakuri,
)
from ferro_ta._ferro_ta import (
    cdltasukigap as _cdltasukigap,
)
from ferro_ta._ferro_ta import (
    cdlthrusting as _cdlthrusting,
)
from ferro_ta._ferro_ta import (
    cdltristar as _cdltristar,
)
from ferro_ta._ferro_ta import (
    cdlunique3river as _cdlunique3river,
)
from ferro_ta._ferro_ta import (
    cdlupsidegap2crows as _cdlupsidegap2crows,
)
from ferro_ta._ferro_ta import (
    cdlxsidegap3methods as _cdlxsidegap3methods,
)
from ferro_ta._utils import _to_f64
from ferro_ta.core.exceptions import _normalize_rust_error


def CDL2CROWS(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Two Crows — bearish 3-candle reversal pattern.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        -100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdl2crows(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CDLDOJI(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Doji — open ≈ close, reflecting market indecision.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdldoji(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CDLENGULFING(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Engulfing Pattern.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish), -100 (bearish), or 0.
    """
    try:
        return _cdlengulfing(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CDLHAMMER(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Hammer — small body at top, long lower shadow.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdlhammer(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CDLSHOOTINGSTAR(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Shooting Star — small body at bottom, long upper shadow.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        -100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdlshootingstar(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLMORNINGSTAR(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Morning Star — 3-candle bullish reversal pattern.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdlmorningstar(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLEVENINGSTAR(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Evening Star — 3-candle bearish reversal pattern.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        -100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdleveningstar(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLMARUBOZU(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Marubozu — full body candle with no or minimal shadows.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish), -100 (bearish), or 0.
    """
    try:
        return _cdlmarubozu(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CDLSPINNINGTOP(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Spinning Top — small body with shadows longer than the body.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish), -100 (bearish), or 0.
    """
    try:
        return _cdlspinningtop(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDL3BLACKCROWS(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Three Black Crows — bearish 3-candle reversal.

    Three consecutive long bearish candles, each opening within the prior body
    and closing near its low.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        -100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdl3blackcrows(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDL3WHITESOLDIERS(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Three White Soldiers — bullish 3-candle reversal.

    Three consecutive long bullish candles, each opening within the prior body
    and closing near its high.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdl3whitesoldiers(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDL3INSIDE(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Three Inside Up/Down — harami followed by confirmation candle.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish Three Inside Up), -100 (bearish Three Inside Down), or 0.
    """
    try:
        return _cdl3inside(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CDL3OUTSIDE(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Three Outside Up/Down — engulfing followed by confirmation candle.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish Three Outside Up), -100 (bearish Three Outside Down), or 0.
    """
    try:
        return _cdl3outside(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CDLDOJISTAR(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Doji Star — doji that gaps away from the prior large candle.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish), -100 (bearish), or 0.
    """
    try:
        return _cdldojistar(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CDLMORNINGDOJISTAR(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Morning Doji Star — 3-candle bullish reversal with doji star.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdlmorningdojistar(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLEVENINGDOJISTAR(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Evening Doji Star — 3-candle bearish reversal with doji star.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        -100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdleveningdojistar(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLHARAMI(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Harami Pattern — small candle inside the prior large candle's body.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish), -100 (bearish), or 0.
    """
    try:
        return _cdlharami(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CDLHARAMICROSS(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Harami Cross — doji inside the prior large candle's body.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish), -100 (bearish), or 0.
    """
    try:
        return _cdlharamicross(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDL3LINESTRIKE(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Three-Line Strike — 4-candle reversal pattern.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish), -100 (bearish), or 0.
    """
    try:
        return _cdl3linestrike(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDL3STARSINSOUTH(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Three Stars In The South — 3-candle bullish reversal.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdl3starsinsouth(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLABANDONEDBABY(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Abandoned Baby — 3-candle reversal with gapping doji in the middle.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish), -100 (bearish), or 0.
    """
    try:
        return _cdlabandonedbaby(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLADVANCEBLOCK(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Advance Block — 3 bullish candles with weakening momentum, bearish warning.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        -100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdladvanceblock(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLBELTHOLD(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Belt-hold — single candle opening at extreme with long body.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish), -100 (bearish), or 0.
    """
    try:
        return _cdlbelthold(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CDLBREAKAWAY(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Breakaway — 5-candle reversal pattern.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish), -100 (bearish), or 0.
    """
    try:
        return _cdlbreakaway(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CDLCLOSINGMARUBOZU(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Closing Marubozu — candle with no shadow on the closing side.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish), -100 (bearish), or 0.
    """
    try:
        return _cdlclosingmarubozu(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLCONCEALBABYSWALL(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Concealing Baby Swallow — 4-candle bullish reversal.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdlconcealbabyswall(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLCOUNTERATTACK(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Counterattack Lines — 2-candle pattern with opposite candles closing at same price.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish), -100 (bearish), or 0.
    """
    try:
        return _cdlcounterattack(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLDARKCLOUDCOVER(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Dark Cloud Cover — 2-candle bearish reversal.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        -100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdldarkcloudcover(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLDRAGONFLYDOJI(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Dragonfly Doji — doji with long lower shadow.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdldragonflydoji(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLGAPSIDESIDEWHITE(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Up/Down-Gap Side-by-Side White Lines — 3-candle continuation.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (upside gap), -100 (downside gap), or 0.
    """
    try:
        return _cdlgapsidesidewhite(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLGRAVESTONEDOJI(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Gravestone Doji — doji with long upper shadow.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        -100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdlgravestonedoji(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLHANGINGMAN(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Hanging Man — same shape as hammer but bearish warning.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        -100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdlhangingman(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLHIGHWAVE(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """High-Wave Candle — small body with very long upper and lower shadows.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish), -100 (bearish), or 0.
    """
    try:
        return _cdlhighwave(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CDLHIKKAKE(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Hikkake Pattern — inside bar followed by false breakout then reversal.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish), -100 (bearish), or 0.
    """
    try:
        return _cdlhikkake(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CDLHIKKAKEMOD(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Modified Hikkake Pattern — hikkake with delayed confirmation.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish), -100 (bearish), or 0.
    """
    try:
        return _cdlhikkakemod(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLHOMINGPIGEON(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Homing Pigeon — 2 bearish candles, second inside the first body.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdlhomingpigeon(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLIDENTICAL3CROWS(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Identical Three Crows — 3 bearish candles each opening at prior close.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        -100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdlidentical3crows(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLINNECK(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """In-Neck Pattern — bearish then bullish closing near prior close, bearish continuation.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        -100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdlinneck(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CDLINVERTEDHAMMER(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Inverted Hammer — small body at bottom, long upper shadow.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdlinvertedhammer(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLKICKING(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Kicking — two opposite marubozu candles with a gap.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish), -100 (bearish), or 0.
    """
    try:
        return _cdlkicking(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CDLKICKINGBYLENGTH(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Kicking by the Longer Marubozu — direction determined by longer marubozu.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish), -100 (bearish), or 0.
    """
    try:
        return _cdlkickingbylength(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLLADDERBOTTOM(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Ladder Bottom — 5-candle bullish reversal.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdlladderbottom(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLLONGLEGGEDDOJI(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Long Legged Doji — doji with long upper and lower shadows.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdllongleggeddoji(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLLONGLINE(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Long Line Candle — long body candle (body >= 70% of range).

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish), -100 (bearish), or 0.
    """
    try:
        return _cdllongline(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CDLMATCHINGLOW(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Matching Low — 2 bearish candles with equal closes, bullish reversal.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdlmatchinglow(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLMATHOLD(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Mat Hold — 5-candle bullish continuation pattern.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdlmathold(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CDLONNECK(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """On-Neck Pattern — bearish then bullish reaching only prior low, bearish continuation.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        -100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdlonneck(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CDLPIERCING(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Piercing Pattern — bearish then bullish piercing past midpoint, bullish reversal.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdlpiercing(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CDLRICKSHAWMAN(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Rickshaw Man — doji with long shadows and body near center.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdlrickshawman(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLRISEFALL3METHODS(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Rising/Falling Three Methods — 5-candle continuation pattern.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish), -100 (bearish), or 0.
    """
    try:
        return _cdlrisefall3methods(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLSEPARATINGLINES(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Separating Lines — 2-candle continuation with same open, opposite direction.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish), -100 (bearish), or 0.
    """
    try:
        return _cdlseparatinglines(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLSHORTLINE(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Short Line Candle — small body (body <= 30% of range).

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish), -100 (bearish), or 0.
    """
    try:
        return _cdlshortline(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CDLSTALLEDPATTERN(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Stalled Pattern — 3 bullish candles with stalling on the third, bearish warning.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        -100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdlstalledpattern(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLSTICKSANDWICH(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Stick Sandwich — 2 bearish candles surrounding a bullish, same close.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdlsticksandwich(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLTAKURI(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Takuri — Dragonfly Doji with very long lower shadow (>= 3x body).

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdltakuri(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CDLTASUKIGAP(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Tasuki Gap — 3-candle gap continuation with partial fill.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish), -100 (bearish), or 0.
    """
    try:
        return _cdltasukigap(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CDLTHRUSTING(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Thrusting Pattern — bearish then bullish reaching below midpoint, bearish continuation.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        -100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdlthrusting(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CDLTRISTAR(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Tristar Pattern — 3 dojis with reversal implication.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish), -100 (bearish), or 0.
    """
    try:
        return _cdltristar(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CDLUNIQUE3RIVER(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Unique 3 River — 3-candle bullish reversal.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdlunique3river(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLUPSIDEGAP2CROWS(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Upside Gap Two Crows — 3-candle bearish reversal.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        -100 where pattern is detected, 0 otherwise.
    """
    try:
        return _cdlupsidegap2crows(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CDLXSIDEGAP3METHODS(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Upside/Downside Gap Three Methods — 3-candle gap fill continuation.

    Parameters
    ----------
    open, high, low, close : array-like
        OHLC price arrays.

    Returns
    -------
    numpy.ndarray[int32]
        100 (bullish), -100 (bearish), or 0.
    """
    try:
        return _cdlxsidegap3methods(
            _to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close)
        )
    except ValueError as e:
        _normalize_rust_error(e)


__all__ = [
    "CDL2CROWS",
    "CDL3BLACKCROWS",
    "CDL3INSIDE",
    "CDL3LINESTRIKE",
    "CDL3OUTSIDE",
    "CDL3STARSINSOUTH",
    "CDL3WHITESOLDIERS",
    "CDLABANDONEDBABY",
    "CDLADVANCEBLOCK",
    "CDLBELTHOLD",
    "CDLBREAKAWAY",
    "CDLCLOSINGMARUBOZU",
    "CDLCONCEALBABYSWALL",
    "CDLCOUNTERATTACK",
    "CDLDARKCLOUDCOVER",
    "CDLDOJI",
    "CDLDOJISTAR",
    "CDLDRAGONFLYDOJI",
    "CDLENGULFING",
    "CDLEVENINGDOJISTAR",
    "CDLEVENINGSTAR",
    "CDLGAPSIDESIDEWHITE",
    "CDLGRAVESTONEDOJI",
    "CDLHAMMER",
    "CDLHANGINGMAN",
    "CDLHARAMI",
    "CDLHARAMICROSS",
    "CDLHIGHWAVE",
    "CDLHIKKAKE",
    "CDLHIKKAKEMOD",
    "CDLHOMINGPIGEON",
    "CDLIDENTICAL3CROWS",
    "CDLINNECK",
    "CDLINVERTEDHAMMER",
    "CDLKICKING",
    "CDLKICKINGBYLENGTH",
    "CDLLADDERBOTTOM",
    "CDLLONGLEGGEDDOJI",
    "CDLLONGLINE",
    "CDLMARUBOZU",
    "CDLMATCHINGLOW",
    "CDLMATHOLD",
    "CDLMORNINGDOJISTAR",
    "CDLMORNINGSTAR",
    "CDLONNECK",
    "CDLPIERCING",
    "CDLRICKSHAWMAN",
    "CDLRISEFALL3METHODS",
    "CDLSEPARATINGLINES",
    "CDLSHOOTINGSTAR",
    "CDLSHORTLINE",
    "CDLSPINNINGTOP",
    "CDLSTALLEDPATTERN",
    "CDLSTICKSANDWICH",
    "CDLTAKURI",
    "CDLTASUKIGAP",
    "CDLTHRUSTING",
    "CDLTRISTAR",
    "CDLUNIQUE3RIVER",
    "CDLUPSIDEGAP2CROWS",
    "CDLXSIDEGAP3METHODS",
]
