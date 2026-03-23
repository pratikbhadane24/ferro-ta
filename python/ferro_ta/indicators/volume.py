"""
Volume Indicators — Require volume data to measure buying and selling pressure.

Functions
---------
AD    — Chaikin A/D Line
ADOSC — Chaikin A/D Oscillator
OBV   — On Balance Volume
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
    obv as _obv,
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


__all__ = ["AD", "ADOSC", "OBV"]
