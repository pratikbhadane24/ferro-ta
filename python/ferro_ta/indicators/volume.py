"""
Volume Indicators — Require volume data to measure buying and selling pressure.

Functions
---------
AD             — Chaikin A/D Line
ADOSC          — Chaikin A/D Oscillator
OBV            — On Balance Volume
OBV_SMOOTHED   — OBV followed by SMA or EMA
CMF            — Chaikin Money Flow
EMV            — Ease of Movement
FORCE_INDEX    — Elder's Force Index
NVI            — Negative Volume Index
NVI_WITH_EMA   — NVI plus EMA signal
PVI            — Positive Volume Index
PVI_WITH_SIGNAL — PVI plus moving-average signal
VOLOSC         — Percentage volume oscillator
VROC           — Volume rate of change
KVO            — Klinger Volume Oscillator
PVT            — Price-Volume Trend
RVOL           — Relative volume
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ferro_ta._ferro_ta import (
    ad as _ad,
)
from ferro_ta._ferro_ta import (
    adosc as _adosc,
)
from ferro_ta._ferro_ta import (
    cmf as _cmf,
)
from ferro_ta._ferro_ta import (
    emv as _emv,
)
from ferro_ta._ferro_ta import (
    force_index as _force_index,
)
from ferro_ta._ferro_ta import (
    kvo as _kvo,
)
from ferro_ta._ferro_ta import (
    nvi as _nvi,
)
from ferro_ta._ferro_ta import (
    nvi_with_ema as _nvi_with_ema,
)
from ferro_ta._ferro_ta import (
    obv as _obv,
)
from ferro_ta._ferro_ta import (
    obv_smoothed as _obv_smoothed,
)
from ferro_ta._ferro_ta import (
    pvi as _pvi,
)
from ferro_ta._ferro_ta import (
    pvi_with_signal as _pvi_with_signal,
)
from ferro_ta._ferro_ta import (
    pvt as _pvt,
)
from ferro_ta._ferro_ta import (
    rvol as _rvol,
)
from ferro_ta._ferro_ta import (
    volosc as _volosc,
)
from ferro_ta._ferro_ta import (
    vroc as _vroc,
)
from ferro_ta._utils import _to_f64
from ferro_ta.core.exceptions import _normalize_rust_error


def AD(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike,
) -> np.ndarray:
    """Chaikin A/D Line.

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

    Returns
    -------
    numpy.ndarray
        Cumulative A/D Line values.
    """
    try:
        return _ad(_to_f64(high), _to_f64(low), _to_f64(close), _to_f64(volume))
    except ValueError as e:
        _normalize_rust_error(e)


def ADOSC(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike,
    fastperiod: int = 3,
    slowperiod: int = 10,
) -> np.ndarray:
    """Chaikin A/D Oscillator.

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
    fastperiod : int, optional
        Fast EMA period (default 3).
    slowperiod : int, optional
        Slow EMA period (default 10).

    Returns
    -------
    numpy.ndarray
        Array of ADOSC values; leading ``slowperiod - 1`` entries are ``NaN``.
    """
    try:
        return _adosc(
            _to_f64(high),
            _to_f64(low),
            _to_f64(close),
            _to_f64(volume),
            fastperiod,
            slowperiod,
        )
    except ValueError as e:
        _normalize_rust_error(e)


def OBV(close: ArrayLike, volume: ArrayLike) -> np.ndarray:
    """On Balance Volume.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    volume : array-like
        Sequence of volume values.

    Returns
    -------
    numpy.ndarray
        Cumulative OBV values.
    """
    try:
        return _obv(_to_f64(close), _to_f64(volume))
    except ValueError as e:
        _normalize_rust_error(e)


def OBV_SMOOTHED(
    close: ArrayLike,
    volume: ArrayLike,
    timeperiod: int = 20,
    matype: int = 1,
) -> np.ndarray:
    """On-Balance Volume smoothed with a moving average.

    Parameters
    ----------
    close, volume : array-like
    timeperiod : int, optional
        Smoothing period (default 20).
    matype : int, optional
        ``0`` = SMA, ``1`` = EMA (default), matching ``MA``.
    """
    try:
        return _obv_smoothed(_to_f64(close), _to_f64(volume), timeperiod, matype)
    except ValueError as e:
        _normalize_rust_error(e)


def CMF(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike,
    timeperiod: int = 20,
) -> np.ndarray:
    """Chaikin Money Flow.

    Rolling sum of CLV × volume divided by rolling sum of volume.
    """
    try:
        return _cmf(
            _to_f64(high),
            _to_f64(low),
            _to_f64(close),
            _to_f64(volume),
            timeperiod,
        )
    except ValueError as e:
        _normalize_rust_error(e)


def EMV(
    high: ArrayLike,
    low: ArrayLike,
    volume: ArrayLike,
    timeperiod: int = 14,
    divisor: float = 10000.0,
) -> np.ndarray:
    """Ease of Movement (Arms)."""
    try:
        return _emv(_to_f64(high), _to_f64(low), _to_f64(volume), timeperiod, divisor)
    except ValueError as e:
        _normalize_rust_error(e)


def FORCE_INDEX(
    close: ArrayLike,
    volume: ArrayLike,
    timeperiod: int = 13,
) -> np.ndarray:
    """Elder's Force Index.

    ``timeperiod=1`` returns the raw ``(close - prev) * volume`` series.
    """
    try:
        return _force_index(_to_f64(close), _to_f64(volume), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def NVI(close: ArrayLike, volume: ArrayLike) -> np.ndarray:
    """Negative Volume Index, seeded at 1000."""
    try:
        return _nvi(_to_f64(close), _to_f64(volume))
    except ValueError as e:
        _normalize_rust_error(e)


def NVI_WITH_EMA(
    close: ArrayLike,
    volume: ArrayLike,
    timeperiod: int = 255,
) -> tuple[np.ndarray, np.ndarray]:
    """Negative Volume Index plus its EMA signal."""
    try:
        return _nvi_with_ema(_to_f64(close), _to_f64(volume), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def PVI(close: ArrayLike, volume: ArrayLike) -> np.ndarray:
    """Positive Volume Index, seeded at 1000."""
    try:
        return _pvi(_to_f64(close), _to_f64(volume))
    except ValueError as e:
        _normalize_rust_error(e)


def PVI_WITH_SIGNAL(
    close: ArrayLike,
    volume: ArrayLike,
    timeperiod: int = 255,
    matype: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Positive Volume Index plus a moving-average signal."""
    try:
        return _pvi_with_signal(_to_f64(close), _to_f64(volume), timeperiod, matype)
    except ValueError as e:
        _normalize_rust_error(e)


def VOLOSC(
    volume: ArrayLike,
    fastperiod: int = 5,
    slowperiod: int = 10,
) -> np.ndarray:
    """Percentage volume oscillator: ``100 * (fast SMA - slow SMA) / slow SMA``."""
    try:
        return _volosc(_to_f64(volume), fastperiod, slowperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def VROC(volume: ArrayLike, timeperiod: int = 25) -> np.ndarray:
    """Volume rate of change."""
    try:
        return _vroc(_to_f64(volume), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def KVO(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike,
    fastperiod: int = 34,
    slowperiod: int = 55,
    signalperiod: int = 13,
) -> tuple[np.ndarray, np.ndarray]:
    """Klinger Volume Oscillator and signal line."""
    try:
        return _kvo(
            _to_f64(high),
            _to_f64(low),
            _to_f64(close),
            _to_f64(volume),
            fastperiod,
            slowperiod,
            signalperiod,
        )
    except ValueError as e:
        _normalize_rust_error(e)


def PVT(close: ArrayLike, volume: ArrayLike) -> np.ndarray:
    """Price-Volume Trend."""
    try:
        return _pvt(_to_f64(close), _to_f64(volume))
    except ValueError as e:
        _normalize_rust_error(e)


def RVOL(volume: ArrayLike, timeperiod: int = 20) -> np.ndarray:
    """Relative volume: ``volume / SMA(volume)``."""
    try:
        return _rvol(_to_f64(volume), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


__all__ = [
    "AD",
    "ADOSC",
    "OBV",
    "OBV_SMOOTHED",
    "CMF",
    "EMV",
    "FORCE_INDEX",
    "NVI",
    "NVI_WITH_EMA",
    "PVI",
    "PVI_WITH_SIGNAL",
    "VOLOSC",
    "VROC",
    "KVO",
    "PVT",
    "RVOL",
]
