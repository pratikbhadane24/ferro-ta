"""
Volatility Indicators — Measure the magnitude of price fluctuations.

Functions
---------
ATR                    — Average True Range
NATR                   — Normalized Average True Range
TRANGE                 — True Range
CHAIKIN_VOL            — Chaikin Volatility
MASS                   — Mass Index
BBPERCENT              — Bollinger %B
BBWIDTH                — Bollinger Bandwidth
HISTORICAL_VOLATILITY  — Annualized close-to-close volatility
ULCER_INDEX            — Ulcer Index
STARC                  — Stoller Average Range Channels
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ferro_ta._ferro_ta import (
    atr as _atr,
)
from ferro_ta._ferro_ta import (
    bbpercent as _bbpercent,
)
from ferro_ta._ferro_ta import (
    bbwidth as _bbwidth,
)
from ferro_ta._ferro_ta import (
    chaikin_vol as _chaikin_vol,
)
from ferro_ta._ferro_ta import (
    historical_volatility as _historical_volatility,
)
from ferro_ta._ferro_ta import (
    mass as _mass,
)
from ferro_ta._ferro_ta import (
    natr as _natr,
)
from ferro_ta._ferro_ta import (
    starc as _starc,
)
from ferro_ta._ferro_ta import (
    trange as _trange,
)
from ferro_ta._ferro_ta import (
    ulcer_index as _ulcer_index,
)
from ferro_ta._utils import _to_f64
from ferro_ta.core.exceptions import _normalize_rust_error


def ATR(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
) -> np.ndarray:
    """Average True Range.

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
        Array of ATR values; leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _atr(_to_f64(high), _to_f64(low), _to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def NATR(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
) -> np.ndarray:
    """Normalized Average True Range.

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
        Array of NATR values (percentage); leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _natr(_to_f64(high), _to_f64(low), _to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def TRANGE(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> np.ndarray:
    """True Range.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    close : array-like
        Sequence of closing prices.

    Returns
    -------
    numpy.ndarray
        Array of True Range values.
    """
    try:
        return _trange(_to_f64(high), _to_f64(low), _to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def CHAIKIN_VOL(
    high: ArrayLike,
    low: ArrayLike,
    timeperiod: int = 10,
    rocperiod: int = 10,
) -> np.ndarray:
    """Chaikin Volatility — ROC of an EMA of the high–low range.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    timeperiod : int, optional
        EMA period of the range (default 10).
    rocperiod : int, optional
        Rate-of-change lookback of that EMA (default 10).

    Returns
    -------
    numpy.ndarray
        Array of Chaikin Volatility values (percent).
    """
    try:
        return _chaikin_vol(_to_f64(high), _to_f64(low), timeperiod, rocperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def MASS(
    high: ArrayLike,
    low: ArrayLike,
    timeperiod: int = 9,
    sumperiod: int = 25,
) -> np.ndarray:
    """Mass Index — rolling sum of the single/double EMA ratio of the range.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    timeperiod : int, optional
        EMA period (default 9).
    sumperiod : int, optional
        Rolling-sum length of the EMA ratio (default 25).

    Returns
    -------
    numpy.ndarray
        Array of Mass Index values.
    """
    try:
        return _mass(_to_f64(high), _to_f64(low), timeperiod, sumperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def BBPERCENT(
    close: ArrayLike,
    timeperiod: int = 5,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
) -> np.ndarray:
    """Bollinger %B — ``(close - lower) / (upper - lower)``.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Bollinger window (default 5).
    nbdevup : float, optional
        Upper-band standard deviations (default 2.0).
    nbdevdn : float, optional
        Lower-band standard deviations (default 2.0).

    Returns
    -------
    numpy.ndarray
        Array of %B values. Leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _bbpercent(_to_f64(close), timeperiod, nbdevup, nbdevdn)
    except ValueError as e:
        _normalize_rust_error(e)


def BBWIDTH(
    close: ArrayLike,
    timeperiod: int = 5,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
) -> np.ndarray:
    """Bollinger Bandwidth — ``(upper - lower) / middle``.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Bollinger window (default 5).
    nbdevup : float, optional
        Upper-band standard deviations (default 2.0).
    nbdevdn : float, optional
        Lower-band standard deviations (default 2.0).

    Returns
    -------
    numpy.ndarray
        Array of bandwidth values. Leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _bbwidth(_to_f64(close), timeperiod, nbdevup, nbdevdn)
    except ValueError as e:
        _normalize_rust_error(e)


def HISTORICAL_VOLATILITY(
    close: ArrayLike,
    timeperiod: int = 20,
    annual: float = 252.0,
) -> np.ndarray:
    """Close-to-close historical volatility, annualized and in percent.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Log-return window (default 20).
    annual : float, optional
        Annualization factor (default 252).

    Returns
    -------
    numpy.ndarray
        ``stddev(ln returns) * sqrt(annual) * 100``. First valid value is
        at index ``timeperiod``.
    """
    try:
        return _historical_volatility(_to_f64(close), timeperiod, annual)
    except ValueError as e:
        _normalize_rust_error(e)


def ULCER_INDEX(close: ArrayLike, timeperiod: int = 14) -> np.ndarray:
    """Ulcer Index — RMS of percent drawdowns versus the rolling peak.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Peak and RMS lookback (default 14).

    Returns
    -------
    numpy.ndarray
        Array of Ulcer Index values.
    """
    try:
        return _ulcer_index(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def STARC(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 15,
    atr_period: int = 15,
    multiplier: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stoller Average Range Channels — SMA(close) ± ``multiplier * ATR``.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        SMA period of close (default 15).
    atr_period : int, optional
        ATR period (default 15).
    multiplier : float, optional
        ATR multiple (default 2.0).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(upper, middle, lower)``.
    """
    try:
        return _starc(
            _to_f64(high),
            _to_f64(low),
            _to_f64(close),
            timeperiod,
            atr_period,
            multiplier,
        )
    except ValueError as e:
        _normalize_rust_error(e)


__all__ = [
    "ATR",
    "NATR",
    "TRANGE",
    "CHAIKIN_VOL",
    "MASS",
    "BBPERCENT",
    "BBWIDTH",
    "HISTORICAL_VOLATILITY",
    "ULCER_INDEX",
    "STARC",
]
