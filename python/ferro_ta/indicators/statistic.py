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
DTW              — Dynamic Time Warping (distance + warping path)
DTW_DISTANCE     — Dynamic Time Warping distance only (faster)
BATCH_DTW        — Batch DTW: N series vs 1 reference, in parallel
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from ferro_ta._ferro_ta import (
    batch_dtw as _batch_dtw,
)
from ferro_ta._ferro_ta import (
    beta as _beta,
)
from ferro_ta._ferro_ta import (
    correl as _correl,
)
from ferro_ta._ferro_ta import (
    dtw as _dtw,
)
from ferro_ta._ferro_ta import (
    dtw_distance as _dtw_distance,
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


def DTW(
    series1: ArrayLike,
    series2: ArrayLike,
    window: Optional[int] = None,
) -> tuple[float, np.ndarray]:
    """Dynamic Time Warping — distance and optimal warping path.

    Parameters
    ----------
    series1 : array-like
        First time series.
    series2 : array-like
        Second time series (may differ in length from series1).
    window : int, optional
        Sakoe-Chiba band width. ``None`` (default) = unconstrained.

    Returns
    -------
    distance : float
        DTW distance (accumulated Euclidean cost along the optimal path).
    path : numpy.ndarray, shape (N, 2)
        Warping path as ``(i, j)`` index pairs from ``(0, 0)`` to
        ``(len(series1)-1, len(series2)-1)``.
    """
    try:
        return _dtw(_to_f64(series1), _to_f64(series2), window)
    except ValueError as e:
        _normalize_rust_error(e)


def DTW_DISTANCE(
    series1: ArrayLike,
    series2: ArrayLike,
    window: Optional[int] = None,
) -> float:
    """Dynamic Time Warping distance only (faster — no path reconstruction).

    Parameters
    ----------
    series1 : array-like
        First time series.
    series2 : array-like
        Second time series (may differ in length from series1).
    window : int, optional
        Sakoe-Chiba band width. ``None`` (default) = unconstrained.

    Returns
    -------
    float
        DTW distance (accumulated Euclidean cost along the optimal path).
    """
    try:
        return _dtw_distance(_to_f64(series1), _to_f64(series2), window)
    except ValueError as e:
        _normalize_rust_error(e)


def BATCH_DTW(
    matrix: ArrayLike,
    reference: ArrayLike,
    window: Optional[int] = None,
) -> np.ndarray:
    """Batch Dynamic Time Warping — N series vs 1 reference, computed in parallel.

    Parameters
    ----------
    matrix : array-like, shape (N, L)
        N time series of length L. Each row is compared against ``reference``.
    reference : array-like, shape (L,)
        The reference series.
    window : int, optional
        Sakoe-Chiba band width. ``None`` (default) = unconstrained.

    Returns
    -------
    numpy.ndarray, shape (N,)
        DTW distance from each row of ``matrix`` to ``reference``.
    """
    try:
        mat = np.ascontiguousarray(matrix, dtype=np.float64)
        if mat.ndim != 2:
            from ferro_ta.core.exceptions import FerroTAInputError

            raise FerroTAInputError(
                f"matrix must be a 2-D array, got {mat.ndim}-D.",
                suggestion="Pass a 2-D NumPy array of shape (N, L).",
            )
        return _batch_dtw(mat, _to_f64(reference), window)
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
    "DTW",
    "DTW_DISTANCE",
    "BATCH_DTW",
]
