"""
Statistic Functions — Standard statistical math applied to rolling windows of price data.

Functions
---------
STDDEV           — Standard Deviation
VAR              — Variance
LINEARREG        — Linear Regression
LINEARREG_SLOPE  — Linear Regression Slope
LINEARREG_INTERCEPT — Linear Regression Intercept
LINEARREG_ANGLE  — Linear Regression Angle (degrees)
TSF              — Time Series Forecast
BETA             — Beta
CORREL           — Pearson's Correlation Coefficient (r)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ferro_ta._ferro_ta import (
    beta as _beta,
)
from ferro_ta._ferro_ta import (
    correl as _correl,
)
from ferro_ta._ferro_ta import (
    linearreg as _linearreg,
)
from ferro_ta._ferro_ta import (
    linearreg_angle as _linearreg_angle,
)
from ferro_ta._ferro_ta import (
    linearreg_intercept as _linearreg_intercept,
)
from ferro_ta._ferro_ta import (
    linearreg_slope as _linearreg_slope,
)
from ferro_ta._ferro_ta import (
    stddev as _stddev,
)
from ferro_ta._ferro_ta import (
    tsf as _tsf,
)
from ferro_ta._ferro_ta import (
    var as _var,
)
from ferro_ta._utils import _to_f64
from ferro_ta.core.exceptions import _normalize_rust_error


def STDDEV(close: ArrayLike, timeperiod: int = 5, nbdev: float = 1.0) -> np.ndarray:
    """Standard Deviation.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Rolling window size (default 5).
    nbdev : float, optional
        Number of standard deviations (default 1.0).

    Returns
    -------
    numpy.ndarray
        Array of STDDEV values; leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _stddev(_to_f64(close), timeperiod, nbdev)
    except ValueError as e:
        _normalize_rust_error(e)


def VAR(close: ArrayLike, timeperiod: int = 5, nbdev: float = 1.0) -> np.ndarray:
    """Variance.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Rolling window size (default 5).
    nbdev : float, optional
        Number of deviations (default 1.0).

    Returns
    -------
    numpy.ndarray
        Array of VAR values; leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _var(_to_f64(close), timeperiod, nbdev)
    except ValueError as e:
        _normalize_rust_error(e)


def LINEARREG(close: ArrayLike, timeperiod: int = 14) -> np.ndarray:
    """Linear Regression.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Regression window (default 14).

    Returns
    -------
    numpy.ndarray
        Array of linear regression end-point values; leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _linearreg(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def LINEARREG_SLOPE(close: ArrayLike, timeperiod: int = 14) -> np.ndarray:
    """Linear Regression Slope.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Regression window (default 14).

    Returns
    -------
    numpy.ndarray
        Array of slope values; leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _linearreg_slope(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def LINEARREG_INTERCEPT(close: ArrayLike, timeperiod: int = 14) -> np.ndarray:
    """Linear Regression Intercept.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Regression window (default 14).

    Returns
    -------
    numpy.ndarray
        Array of intercept values; leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _linearreg_intercept(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def LINEARREG_ANGLE(close: ArrayLike, timeperiod: int = 14) -> np.ndarray:
    """Linear Regression Angle (in degrees).

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Regression window (default 14).

    Returns
    -------
    numpy.ndarray
        Array of angle values in degrees; leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _linearreg_angle(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def TSF(close: ArrayLike, timeperiod: int = 14) -> np.ndarray:
    """Time Series Forecast — linear regression extrapolated one period ahead.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Regression window (default 14).

    Returns
    -------
    numpy.ndarray
        Array of TSF values; leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _tsf(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def BETA(real0: ArrayLike, real1: ArrayLike, timeperiod: int = 5) -> np.ndarray:
    """Beta — regression slope of real0 relative to real1.

    Parameters
    ----------
    real0 : array-like
        Sequence of prices for asset 0 (dependent variable).
    real1 : array-like
        Sequence of prices for asset 1 (independent variable).
    timeperiod : int, optional
        Rolling window (default 5).

    Returns
    -------
    numpy.ndarray
        Array of BETA values; leading ``timeperiod`` entries are ``NaN``.
    """
    try:
        return _beta(_to_f64(real0), _to_f64(real1), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def CORREL(real0: ArrayLike, real1: ArrayLike, timeperiod: int = 30) -> np.ndarray:
    """Pearson's Correlation Coefficient (r).

    Parameters
    ----------
    real0 : array-like
        First data series.
    real1 : array-like
        Second data series.
    timeperiod : int, optional
        Rolling window (default 30).

    Returns
    -------
    numpy.ndarray
        Array of CORREL values (-1 to 1); leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _correl(_to_f64(real0), _to_f64(real1), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


__all__ = [
    "STDDEV",
    "VAR",
    "LINEARREG",
    "LINEARREG_SLOPE",
    "LINEARREG_INTERCEPT",
    "LINEARREG_ANGLE",
    "TSF",
    "BETA",
    "CORREL",
]
