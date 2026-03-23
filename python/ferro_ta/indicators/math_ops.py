"""
Math Operators & Math Transforms — TA-Lib compatibility shims.

Rolling functions (SUM, MAX, MIN, MAXINDEX, MININDEX) are implemented in Rust
using O(n) monotonic deque / prefix-sum algorithms.  All other functions are
thin NumPy wrappers (element-wise operations).

Functions
---------
Math Operators:
  ADD        — Element-wise addition
  SUB        — Element-wise subtraction
  MULT       — Element-wise multiplication
  DIV        — Element-wise division
  SUM        — Rolling sum over *timeperiod* bars  (Rust)
  MAX        — Rolling maximum over *timeperiod* bars  (Rust)
  MIN        — Rolling minimum over *timeperiod* bars  (Rust)
  MAXINDEX   — Index of rolling maximum over *timeperiod* bars  (Rust)
  MININDEX   — Index of rolling minimum over *timeperiod* bars  (Rust)

Math Transforms (element-wise):
  ACOS ASIN ATAN CEIL COS COSH EXP FLOOR LN LOG10 SIN SINH SQRT TAN TANH

Rust backend
------------
Rolling operators delegate to::

    from ferro_ta._ferro_ta import rolling_sum, rolling_max, rolling_min, ...
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

# ---------------------------------------------------------------------------
# Import Rust rolling operators
# ---------------------------------------------------------------------------
from ferro_ta._ferro_ta import (
    rolling_max as _rust_rolling_max,
)
from ferro_ta._ferro_ta import (
    rolling_maxindex as _rust_rolling_maxindex,
)
from ferro_ta._ferro_ta import (
    rolling_min as _rust_rolling_min,
)
from ferro_ta._ferro_ta import (
    rolling_minindex as _rust_rolling_minindex,
)
from ferro_ta._ferro_ta import (
    rolling_sum as _rust_rolling_sum,
)
from ferro_ta._utils import _to_f64
from ferro_ta.core.exceptions import _normalize_rust_error

# ---------------------------------------------------------------------------
# Math Operators
# ---------------------------------------------------------------------------


def ADD(real0: ArrayLike, real1: ArrayLike) -> np.ndarray:
    """Element-wise addition: real0 + real1.

    Parameters
    ----------
    real0, real1 : array-like
        Input arrays (same length).

    Returns
    -------
    numpy.ndarray[float64]
    """
    try:
        return np.add(_to_f64(real0), _to_f64(real1))
    except ValueError as e:
        _normalize_rust_error(e)


def SUB(real0: ArrayLike, real1: ArrayLike) -> np.ndarray:
    """Element-wise subtraction: real0 - real1.

    Parameters
    ----------
    real0, real1 : array-like
        Input arrays (same length).

    Returns
    -------
    numpy.ndarray[float64]
    """
    try:
        return np.subtract(_to_f64(real0), _to_f64(real1))
    except ValueError as e:
        _normalize_rust_error(e)


def MULT(real0: ArrayLike, real1: ArrayLike) -> np.ndarray:
    """Element-wise multiplication: real0 * real1.

    Parameters
    ----------
    real0, real1 : array-like
        Input arrays (same length).

    Returns
    -------
    numpy.ndarray[float64]
    """
    try:
        return np.multiply(_to_f64(real0), _to_f64(real1))
    except ValueError as e:
        _normalize_rust_error(e)


def DIV(real0: ArrayLike, real1: ArrayLike) -> np.ndarray:
    """Element-wise division: real0 / real1.

    Parameters
    ----------
    real0, real1 : array-like
        Input arrays (same length).

    Returns
    -------
    numpy.ndarray[float64]
    """
    try:
        # Suppress divide-by-zero warnings while preserving inf/NaN outputs.
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.divide(_to_f64(real0), _to_f64(real1))
    except ValueError as e:
        _normalize_rust_error(e)


def SUM(real: ArrayLike, timeperiod: int = 30) -> np.ndarray:
    """Rolling sum over *timeperiod* bars.

    Parameters
    ----------
    real : array-like
    timeperiod : int, default 30

    Returns
    -------
    numpy.ndarray[float64]
        NaN for the first ``timeperiod - 1`` bars.

    Notes
    -----
    Implemented in Rust using O(n) prefix-sum algorithm.
    """
    try:
        arr = _to_f64(real)
        return np.asarray(_rust_rolling_sum(arr, timeperiod))
    except ValueError as e:
        _normalize_rust_error(e)


def MAX(real: ArrayLike, timeperiod: int = 30) -> np.ndarray:
    """Rolling maximum over *timeperiod* bars.

    Parameters
    ----------
    real : array-like
    timeperiod : int, default 30

    Returns
    -------
    numpy.ndarray[float64]
        NaN for the first ``timeperiod - 1`` bars.

    Notes
    -----
    Implemented in Rust using O(n) monotonic deque algorithm.
    """
    try:
        arr = _to_f64(real)
        return np.asarray(_rust_rolling_max(arr, timeperiod))
    except ValueError as e:
        _normalize_rust_error(e)


def MIN(real: ArrayLike, timeperiod: int = 30) -> np.ndarray:
    """Rolling minimum over *timeperiod* bars.

    Parameters
    ----------
    real : array-like
    timeperiod : int, default 30

    Returns
    -------
    numpy.ndarray[float64]
        NaN for the first ``timeperiod - 1`` bars.

    Notes
    -----
    Implemented in Rust using O(n) monotonic deque algorithm.
    """
    try:
        arr = _to_f64(real)
        return np.asarray(_rust_rolling_min(arr, timeperiod))
    except ValueError as e:
        _normalize_rust_error(e)


def MAXINDEX(real: ArrayLike, timeperiod: int = 30) -> np.ndarray:
    """Index of the rolling maximum over *timeperiod* bars.

    The index is the absolute position in the input array.

    Parameters
    ----------
    real : array-like
    timeperiod : int, default 30

    Returns
    -------
    numpy.ndarray[int64]
        -1 for the first ``timeperiod - 1`` bars (warmup period).

    Notes
    -----
    Implemented in Rust using O(n) monotonic deque algorithm.
    """
    try:
        arr = _to_f64(real)
        return np.asarray(_rust_rolling_maxindex(arr, timeperiod))
    except ValueError as e:
        _normalize_rust_error(e)


def MININDEX(real: ArrayLike, timeperiod: int = 30) -> np.ndarray:
    """Index of the rolling minimum over *timeperiod* bars.

    The index is the absolute position in the input array.

    Parameters
    ----------
    real : array-like
    timeperiod : int, default 30

    Returns
    -------
    numpy.ndarray[int64]
        -1 for the first ``timeperiod - 1`` bars (warmup period).

    Notes
    -----
    Implemented in Rust using O(n) monotonic deque algorithm.
    """
    try:
        arr = _to_f64(real)
        return np.asarray(_rust_rolling_minindex(arr, timeperiod))
    except ValueError as e:
        _normalize_rust_error(e)


# ---------------------------------------------------------------------------
# Math Transforms (element-wise)
# ---------------------------------------------------------------------------


def ACOS(real: ArrayLike) -> np.ndarray:
    """Arc cosine (element-wise). Returns NaN outside [-1, 1]."""
    with np.errstate(invalid="ignore"):
        return np.arccos(_to_f64(real))


def ASIN(real: ArrayLike) -> np.ndarray:
    """Arc sine (element-wise). Returns NaN outside [-1, 1]."""
    with np.errstate(invalid="ignore"):
        return np.arcsin(_to_f64(real))


def ATAN(real: ArrayLike) -> np.ndarray:
    """Arc tangent (element-wise)."""
    return np.arctan(_to_f64(real))


def CEIL(real: ArrayLike) -> np.ndarray:
    """Ceiling (element-wise)."""
    return np.ceil(_to_f64(real))


def COS(real: ArrayLike) -> np.ndarray:
    """Cosine (element-wise)."""
    return np.cos(_to_f64(real))


def COSH(real: ArrayLike) -> np.ndarray:
    """Hyperbolic cosine (element-wise)."""
    return np.cosh(_to_f64(real))


def EXP(real: ArrayLike) -> np.ndarray:
    """Exponential (element-wise)."""
    return np.exp(_to_f64(real))


def FLOOR(real: ArrayLike) -> np.ndarray:
    """Floor (element-wise)."""
    return np.floor(_to_f64(real))


def LN(real: ArrayLike) -> np.ndarray:
    """Natural logarithm (element-wise). Returns NaN for non-positive inputs."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(_to_f64(real))


def LOG10(real: ArrayLike) -> np.ndarray:
    """Base-10 logarithm (element-wise). Returns NaN for non-positive inputs."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(_to_f64(real))


def SIN(real: ArrayLike) -> np.ndarray:
    """Sine (element-wise)."""
    return np.sin(_to_f64(real))


def SINH(real: ArrayLike) -> np.ndarray:
    """Hyperbolic sine (element-wise)."""
    return np.sinh(_to_f64(real))


def SQRT(real: ArrayLike) -> np.ndarray:
    """Square root (element-wise). Returns NaN for negative inputs."""
    with np.errstate(invalid="ignore"):
        return np.sqrt(_to_f64(real))


def TAN(real: ArrayLike) -> np.ndarray:
    """Tangent (element-wise)."""
    return np.tan(_to_f64(real))


def TANH(real: ArrayLike) -> np.ndarray:
    """Hyperbolic tangent (element-wise)."""
    return np.tanh(_to_f64(real))


__all__ = [
    # Math Operators
    "ADD",
    "SUB",
    "MULT",
    "DIV",
    "SUM",
    "MAX",
    "MIN",
    "MAXINDEX",
    "MININDEX",
    # Math Transforms
    "ACOS",
    "ASIN",
    "ATAN",
    "CEIL",
    "COS",
    "COSH",
    "EXP",
    "FLOOR",
    "LN",
    "LOG10",
    "SIN",
    "SINH",
    "SQRT",
    "TAN",
    "TANH",
]
