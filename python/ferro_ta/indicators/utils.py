"""
Signal utilities — crossovers, rolling extremes, change, and state helpers.

Functions
---------
CROSSOVER   — 1.0 where *real0* crosses strictly above *real1*
CROSSUNDER  — 1.0 where *real0* crosses strictly below *real1*
CROSS       — 1.0 where *real0* crosses *real1* in either direction
HIGHEST     — Rolling maximum over *timeperiod* bars (same math as MAX)
LOWEST      — Rolling minimum over *timeperiod* bars (same math as MIN)
CHANGE      — ``real[i] - real[i - timeperiod]``
RISING      — 1.0 when *real* is strictly greater than *timeperiod* bars ago
FALLING     — 1.0 when *real* is strictly less than *timeperiod* bars ago
EXREM       — Keep the first *primary* signal until a *secondary* reset
FLIP        — Hold 1.0 from *primary* until *secondary* clears it
VALUEWHEN   — Value of *real* at the *occurrence*-th most recent condition
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ferro_ta._ferro_ta import (
    change as _change,
)
from ferro_ta._ferro_ta import (
    cross as _cross,
)
from ferro_ta._ferro_ta import (
    crossover as _crossover,
)
from ferro_ta._ferro_ta import (
    crossunder as _crossunder,
)
from ferro_ta._ferro_ta import (
    exrem as _exrem,
)
from ferro_ta._ferro_ta import (
    falling as _falling,
)
from ferro_ta._ferro_ta import (
    flip as _flip,
)
from ferro_ta._ferro_ta import (
    highest as _highest,
)
from ferro_ta._ferro_ta import (
    lowest as _lowest,
)
from ferro_ta._ferro_ta import (
    rising as _rising,
)
from ferro_ta._ferro_ta import (
    valuewhen as _valuewhen,
)
from ferro_ta._utils import _to_f64
from ferro_ta.core.exceptions import _normalize_rust_error


def CROSSOVER(real0: ArrayLike, real1: ArrayLike) -> np.ndarray:
    """1.0 on the bar where *real0* crosses strictly above *real1*.

    A cross is ``real0[i-1] <= real1[i-1]`` and ``real0[i] > real1[i]``.
    The first bar is ``0.0``.

    Parameters
    ----------
    real0, real1 : array-like
        Input series (same length).

    Returns
    -------
    numpy.ndarray[float64]
    """
    try:
        return _crossover(_to_f64(real0), _to_f64(real1))
    except ValueError as e:
        _normalize_rust_error(e)


def CROSSUNDER(real0: ArrayLike, real1: ArrayLike) -> np.ndarray:
    """1.0 on the bar where *real0* crosses strictly below *real1*.

    A cross is ``real0[i-1] >= real1[i-1]`` and ``real0[i] < real1[i]``.
    The first bar is ``0.0``.

    Parameters
    ----------
    real0, real1 : array-like
        Input series (same length).

    Returns
    -------
    numpy.ndarray[float64]
    """
    try:
        return _crossunder(_to_f64(real0), _to_f64(real1))
    except ValueError as e:
        _normalize_rust_error(e)


def CROSS(real0: ArrayLike, real1: ArrayLike) -> np.ndarray:
    """1.0 on any bar where *real0* crosses *real1* in either direction.

    Parameters
    ----------
    real0, real1 : array-like
        Input series (same length).

    Returns
    -------
    numpy.ndarray[float64]
    """
    try:
        return _cross(_to_f64(real0), _to_f64(real1))
    except ValueError as e:
        _normalize_rust_error(e)


def HIGHEST(real: ArrayLike, timeperiod: int = 30) -> np.ndarray:
    """Rolling highest value over *timeperiod* bars.

    Same math as :func:`ferro_ta.MAX`. Leading ``timeperiod - 1`` values
    are ``NaN``.

    Parameters
    ----------
    real : array-like
    timeperiod : int, optional
        Window length (default 30).

    Returns
    -------
    numpy.ndarray[float64]
    """
    try:
        return _highest(_to_f64(real), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def LOWEST(real: ArrayLike, timeperiod: int = 30) -> np.ndarray:
    """Rolling lowest value over *timeperiod* bars.

    Same math as :func:`ferro_ta.MIN`. Leading ``timeperiod - 1`` values
    are ``NaN``.

    Parameters
    ----------
    real : array-like
    timeperiod : int, optional
        Window length (default 30).

    Returns
    -------
    numpy.ndarray[float64]
    """
    try:
        return _lowest(_to_f64(real), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def CHANGE(real: ArrayLike, timeperiod: int = 1) -> np.ndarray:
    """Lookback difference: ``real[i] - real[i - timeperiod]``.

    Leading ``timeperiod`` values are ``NaN``.

    Parameters
    ----------
    real : array-like
    timeperiod : int, optional
        Bars to look back (default 1).

    Returns
    -------
    numpy.ndarray[float64]
    """
    try:
        return _change(_to_f64(real), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def RISING(real: ArrayLike, timeperiod: int = 1) -> np.ndarray:
    """1.0 when *real* is strictly greater than it was *timeperiod* bars ago.

    Leading ``timeperiod`` values are ``NaN``.

    Parameters
    ----------
    real : array-like
    timeperiod : int, optional
        Bars to look back (default 1).

    Returns
    -------
    numpy.ndarray[float64]
    """
    try:
        return _rising(_to_f64(real), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def FALLING(real: ArrayLike, timeperiod: int = 1) -> np.ndarray:
    """1.0 when *real* is strictly less than it was *timeperiod* bars ago.

    Leading ``timeperiod`` values are ``NaN``.

    Parameters
    ----------
    real : array-like
    timeperiod : int, optional
        Bars to look back (default 1).

    Returns
    -------
    numpy.ndarray[float64]
    """
    try:
        return _falling(_to_f64(real), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def EXREM(primary: ArrayLike, secondary: ArrayLike) -> np.ndarray:
    """Keep the first *primary* signal and suppress further primaries until
    a *secondary* signal occurs.

    Finite non-zero values are treated as true. Same-bar primary and
    secondary emits the primary (if not latched) and then resets.

    Parameters
    ----------
    primary, secondary : array-like
        Signal series (same length).

    Returns
    -------
    numpy.ndarray[float64]
    """
    try:
        return _exrem(_to_f64(primary), _to_f64(secondary))
    except ValueError as e:
        _normalize_rust_error(e)


def FLIP(primary: ArrayLike, secondary: ArrayLike) -> np.ndarray:
    """Hold 1.0 from a *primary* signal until a *secondary* signal clears it.

    Finite non-zero values are treated as true. Same-bar primary and
    secondary leaves the state off.

    Parameters
    ----------
    primary, secondary : array-like
        Signal series (same length).

    Returns
    -------
    numpy.ndarray[float64]
    """
    try:
        return _flip(_to_f64(primary), _to_f64(secondary))
    except ValueError as e:
        _normalize_rust_error(e)


def VALUEWHEN(condition: ArrayLike, real: ArrayLike, occurrence: int = 1) -> np.ndarray:
    """Value of *real* at the *occurrence*-th most recent true *condition*.

    ``occurrence=1`` is the most recent hit, including the current bar.
    Bars that have not yet seen that many hits are ``NaN``.

    Parameters
    ----------
    condition : array-like
        Finite non-zero values are treated as true.
    real : array-like
        Source series (same length as *condition*).
    occurrence : int, optional
        1 = most recent, 2 = previous hit, … (default 1).

    Returns
    -------
    numpy.ndarray[float64]
    """
    try:
        return _valuewhen(_to_f64(condition), _to_f64(real), occurrence)
    except ValueError as e:
        _normalize_rust_error(e)


__all__ = [
    "CROSSOVER",
    "CROSSUNDER",
    "CROSS",
    "HIGHEST",
    "LOWEST",
    "CHANGE",
    "RISING",
    "FALLING",
    "EXREM",
    "FLIP",
    "VALUEWHEN",
]
