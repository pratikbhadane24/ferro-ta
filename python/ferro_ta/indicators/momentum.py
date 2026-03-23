"""
Momentum Indicators — Oscillators measuring speed and change of price movements.

Functions
---------
RSI       — Relative Strength Index
MOM       — Momentum
ROC       — Rate of Change: ((price/prevPrice)-1)*100
ROCP      — Rate of Change Percentage: (price-prevPrice)/prevPrice
ROCR      — Rate of Change Ratio: price/prevPrice
ROCR100   — Rate of Change Ratio 100 scale: (price/prevPrice)*100
WILLR     — Williams' %R
AROON     — Aroon (returns aroon_down, aroon_up)
AROONOSC  — Aroon Oscillator
CCI       — Commodity Channel Index
MFI       — Money Flow Index
BOP       — Balance Of Power
STOCHF    — Stochastic Fast
STOCH     — Stochastic
STOCHRSI  — Stochastic Relative Strength Index
APO       — Absolute Price Oscillator
PPO       — Percentage Price Oscillator
CMO       — Chande Momentum Oscillator
PLUS_DM   — Plus Directional Movement
MINUS_DM  — Minus Directional Movement
PLUS_DI   — Plus Directional Indicator
MINUS_DI  — Minus Directional Indicator
DX        — Directional Movement Index
ADX       — Average Directional Movement Index
ADXR      — Average Directional Movement Index Rating
TRIX      — 1-day Rate-Of-Change of Triple Smooth EMA
ULTOSC    — Ultimate Oscillator
TRANGE    — True Range (also in volatility)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ferro_ta._ferro_ta import (
    adx as _adx,
)
from ferro_ta._ferro_ta import (
    adxr as _adxr,
)
from ferro_ta._ferro_ta import (
    apo as _apo,
)
from ferro_ta._ferro_ta import (
    aroon as _aroon,
)
from ferro_ta._ferro_ta import (
    aroonosc as _aroonosc,
)
from ferro_ta._ferro_ta import (
    bop as _bop,
)
from ferro_ta._ferro_ta import (
    cci as _cci,
)
from ferro_ta._ferro_ta import (
    cmo as _cmo,
)
from ferro_ta._ferro_ta import (
    dx as _dx,
)
from ferro_ta._ferro_ta import (
    mfi as _mfi,
)
from ferro_ta._ferro_ta import (
    minus_di as _minus_di,
)
from ferro_ta._ferro_ta import (
    minus_dm as _minus_dm,
)
from ferro_ta._ferro_ta import (
    mom as _mom,
)
from ferro_ta._ferro_ta import (
    plus_di as _plus_di,
)
from ferro_ta._ferro_ta import (
    plus_dm as _plus_dm,
)
from ferro_ta._ferro_ta import (
    ppo as _ppo,
)
from ferro_ta._ferro_ta import (
    roc as _roc,
)
from ferro_ta._ferro_ta import (
    rocp as _rocp,
)
from ferro_ta._ferro_ta import (
    rocr as _rocr,
)
from ferro_ta._ferro_ta import (
    rocr100 as _rocr100,
)
from ferro_ta._ferro_ta import (
    rsi as _rsi,
)
from ferro_ta._ferro_ta import (
    stoch as _stoch,
)
from ferro_ta._ferro_ta import (
    stochf as _stochf,
)
from ferro_ta._ferro_ta import (
    stochrsi as _stochrsi,
)
from ferro_ta._ferro_ta import (
    trix as _trix,
)
from ferro_ta._ferro_ta import (
    ultosc as _ultosc,
)
from ferro_ta._ferro_ta import (
    willr as _willr,
)
from ferro_ta._utils import _to_f64
from ferro_ta.core.exceptions import _normalize_rust_error
from ferro_ta.indicators.volatility import TRANGE


def RSI(close: ArrayLike, timeperiod: int = 14) -> np.ndarray:
    """Relative Strength Index.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Number of periods (default 14).

    Returns
    -------
    numpy.ndarray
        Array of RSI values (0–100); leading ``timeperiod`` entries are ``NaN``.
    """
    try:
        return _rsi(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def MOM(close: ArrayLike, timeperiod: int = 10) -> np.ndarray:
    """Momentum.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Number of periods (default 10).

    Returns
    -------
    numpy.ndarray
        Array of MOM values; leading ``timeperiod`` entries are ``NaN``.
    """
    try:
        return _mom(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def ROC(close: ArrayLike, timeperiod: int = 10) -> np.ndarray:
    """Rate of Change: ((price/prevPrice)-1)*100.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Number of periods (default 10).

    Returns
    -------
    numpy.ndarray
        Array of ROC values; leading ``timeperiod`` entries are ``NaN``.
    """
    try:
        return _roc(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def ROCP(close: ArrayLike, timeperiod: int = 10) -> np.ndarray:
    """Rate of Change Percentage: (price-prevPrice)/prevPrice.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Number of periods (default 10).

    Returns
    -------
    numpy.ndarray
        Array of ROCP values; leading ``timeperiod`` entries are ``NaN``.
    """
    try:
        return _rocp(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def ROCR(close: ArrayLike, timeperiod: int = 10) -> np.ndarray:
    """Rate of Change Ratio: price/prevPrice.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Number of periods (default 10).

    Returns
    -------
    numpy.ndarray
        Array of ROCR values; leading ``timeperiod`` entries are ``NaN``.
    """
    try:
        return _rocr(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def ROCR100(close: ArrayLike, timeperiod: int = 10) -> np.ndarray:
    """Rate of Change Ratio 100 scale: (price/prevPrice)*100.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Number of periods (default 10).

    Returns
    -------
    numpy.ndarray
        Array of ROCR100 values; leading ``timeperiod`` entries are ``NaN``.
    """
    try:
        return _rocr100(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def WILLR(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
) -> np.ndarray:
    """Williams' %R.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Number of periods (default 14).

    Returns
    -------
    numpy.ndarray
        Array of WILLR values (-100 to 0); leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _willr(_to_f64(high), _to_f64(low), _to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def AROON(
    high: ArrayLike,
    low: ArrayLike,
    timeperiod: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    """Aroon.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    timeperiod : int, optional
        Number of periods (default 14).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(aroondown, aroonup)`` — two arrays of equal length.
        Leading ``timeperiod`` entries are ``NaN``.
    """
    try:
        return _aroon(_to_f64(high), _to_f64(low), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def AROONOSC(
    high: ArrayLike,
    low: ArrayLike,
    timeperiod: int = 14,
) -> np.ndarray:
    """Aroon Oscillator.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    timeperiod : int, optional
        Number of periods (default 14).

    Returns
    -------
    numpy.ndarray
        Array of AROONOSC values; leading ``timeperiod`` entries are ``NaN``.
    """
    try:
        return _aroonosc(_to_f64(high), _to_f64(low), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def CCI(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
) -> np.ndarray:
    """Commodity Channel Index.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Number of periods (default 14).

    Returns
    -------
    numpy.ndarray
        Array of CCI values; leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _cci(_to_f64(high), _to_f64(low), _to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def MFI(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike,
    timeperiod: int = 14,
) -> np.ndarray:
    """Money Flow Index.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    close : array-like
        Sequence of closing prices.
    volume : array-like
        Sequence of volume values.
    timeperiod : int, optional
        Number of periods (default 14).

    Returns
    -------
    numpy.ndarray
        Array of MFI values (0–100); leading ``timeperiod`` entries are ``NaN``.
    """
    try:
        return _mfi(
            _to_f64(high), _to_f64(low), _to_f64(close), _to_f64(volume), timeperiod
        )
    except ValueError as e:
        _normalize_rust_error(e)


def BOP(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """Balance Of Power.

    Parameters
    ----------
    open : array-like
        Sequence of open prices.
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    close : array-like
        Sequence of closing prices.

    Returns
    -------
    numpy.ndarray
        Array of BOP values (-1 to 1).
    """
    try:
        return _bop(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def STOCHF(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    fastk_period: int = 5,
    fastd_period: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Stochastic Fast.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    close : array-like
        Sequence of closing prices.
    fastk_period : int, optional
        %K period (default 5).
    fastd_period : int, optional
        %D smoothing period (default 3).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(fastk, fastd)`` — two arrays of equal length.
    """
    try:
        return _stochf(
            _to_f64(high), _to_f64(low), _to_f64(close), fastk_period, fastd_period
        )
    except ValueError as e:
        _normalize_rust_error(e)


def STOCH(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    fastk_period: int = 5,
    slowk_period: int = 3,
    slowd_period: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Stochastic.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    close : array-like
        Sequence of closing prices.
    fastk_period : int, optional
        Fast %K period (default 5).
    slowk_period : int, optional
        Slow %K smoothing period (default 3).
    slowd_period : int, optional
        Slow %D smoothing period (default 3).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(slowk, slowd)`` — two arrays of equal length.
    """
    try:
        return _stoch(
            _to_f64(high),
            _to_f64(low),
            _to_f64(close),
            fastk_period,
            slowk_period,
            slowd_period,
        )
    except ValueError as e:
        _normalize_rust_error(e)


def STOCHRSI(
    close: ArrayLike,
    timeperiod: int = 14,
    fastk_period: int = 5,
    fastd_period: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Stochastic Relative Strength Index.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        RSI period (default 14).
    fastk_period : int, optional
        Stochastic %K period (default 5).
    fastd_period : int, optional
        Stochastic %D smoothing period (default 3).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(fastk, fastd)`` — two arrays of equal length.
    """
    try:
        return _stochrsi(_to_f64(close), timeperiod, fastk_period, fastd_period)
    except ValueError as e:
        _normalize_rust_error(e)


def APO(
    close: ArrayLike,
    fastperiod: int = 12,
    slowperiod: int = 26,
) -> np.ndarray:
    """Absolute Price Oscillator.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    fastperiod : int, optional
        Fast EMA period (default 12).
    slowperiod : int, optional
        Slow EMA period (default 26).

    Returns
    -------
    numpy.ndarray
        Array of APO values; leading ``slowperiod - 1`` entries are ``NaN``.
    """
    try:
        return _apo(_to_f64(close), fastperiod, slowperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def PPO(
    close: ArrayLike,
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentage Price Oscillator.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    fastperiod : int, optional
        Fast EMA period (default 12).
    slowperiod : int, optional
        Slow EMA period (default 26).
    signalperiod : int, optional
        Signal EMA period (default 9).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(ppo, signal, histogram)`` — three arrays of equal length.
    """
    try:
        return _ppo(_to_f64(close), fastperiod, slowperiod, signalperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def CMO(close: ArrayLike, timeperiod: int = 14) -> np.ndarray:
    """Chande Momentum Oscillator.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Number of periods (default 14).

    Returns
    -------
    numpy.ndarray
        Array of CMO values (-100 to 100); leading ``timeperiod`` entries are ``NaN``.
    """
    try:
        return _cmo(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def PLUS_DM(high: ArrayLike, low: ArrayLike, timeperiod: int = 14) -> np.ndarray:
    """Plus Directional Movement.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    timeperiod : int, optional
        Smoothing period (default 14).

    Returns
    -------
    numpy.ndarray
        Array of +DM values.
    """
    try:
        return _plus_dm(_to_f64(high), _to_f64(low), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def MINUS_DM(high: ArrayLike, low: ArrayLike, timeperiod: int = 14) -> np.ndarray:
    """Minus Directional Movement.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    timeperiod : int, optional
        Smoothing period (default 14).

    Returns
    -------
    numpy.ndarray
        Array of -DM values.
    """
    try:
        return _minus_dm(_to_f64(high), _to_f64(low), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def PLUS_DI(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
) -> np.ndarray:
    """Plus Directional Indicator.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Smoothing period (default 14).

    Returns
    -------
    numpy.ndarray
        Array of +DI values.
    """
    try:
        return _plus_di(_to_f64(high), _to_f64(low), _to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def MINUS_DI(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
) -> np.ndarray:
    """Minus Directional Indicator.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Smoothing period (default 14).

    Returns
    -------
    numpy.ndarray
        Array of -DI values.
    """
    try:
        return _minus_di(_to_f64(high), _to_f64(low), _to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def DX(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
) -> np.ndarray:
    """Directional Movement Index.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Smoothing period (default 14).

    Returns
    -------
    numpy.ndarray
        Array of DX values (0–100).
    """
    try:
        return _dx(_to_f64(high), _to_f64(low), _to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def ADX(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
) -> np.ndarray:
    """Average Directional Movement Index.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Smoothing period (default 14).

    Returns
    -------
    numpy.ndarray
        Array of ADX values (0–100).
    """
    try:
        return _adx(_to_f64(high), _to_f64(low), _to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def ADXR(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
) -> np.ndarray:
    """Average Directional Movement Index Rating.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Smoothing period (default 14).

    Returns
    -------
    numpy.ndarray
        Array of ADXR values (0–100).
    """
    try:
        return _adxr(_to_f64(high), _to_f64(low), _to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def TRIX(close: ArrayLike, timeperiod: int = 30) -> np.ndarray:
    """1-day Rate-Of-Change of a Triple Smooth EMA.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        EMA period (default 30).

    Returns
    -------
    numpy.ndarray
        Array of TRIX values.
    """
    try:
        return _trix(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def ULTOSC(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod1: int = 7,
    timeperiod2: int = 14,
    timeperiod3: int = 28,
) -> np.ndarray:
    """Ultimate Oscillator.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    close : array-like
        Sequence of closing prices.
    timeperiod1 : int, optional
        First period (default 7).
    timeperiod2 : int, optional
        Second period (default 14).
    timeperiod3 : int, optional
        Third period (default 28).

    Returns
    -------
    numpy.ndarray
        Array of ULTOSC values (0–100).
    """
    try:
        return _ultosc(
            _to_f64(high),
            _to_f64(low),
            _to_f64(close),
            timeperiod1,
            timeperiod2,
            timeperiod3,
        )
    except ValueError as e:
        _normalize_rust_error(e)


__all__ = [
    "RSI",
    "MOM",
    "ROC",
    "ROCP",
    "ROCR",
    "ROCR100",
    "WILLR",
    "AROON",
    "AROONOSC",
    "CCI",
    "MFI",
    "BOP",
    "STOCHF",
    "STOCH",
    "STOCHRSI",
    "APO",
    "PPO",
    "CMO",
    "PLUS_DM",
    "MINUS_DM",
    "PLUS_DI",
    "MINUS_DI",
    "DX",
    "ADX",
    "ADXR",
    "TRIX",
    "ULTOSC",
    "TRANGE",
]
