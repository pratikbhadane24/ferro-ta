"""
Cycle Indicators — Hilbert Transform-based cycle analysis.

All functions use a 63-bar lookback period (first 63 values are NaN).

Functions
---------
HT_TRENDLINE  — Hilbert Transform - Instantaneous Trendline
HT_DCPERIOD   — Hilbert Transform - Dominant Cycle Period
HT_DCPHASE    — Hilbert Transform - Dominant Cycle Phase
HT_PHASOR     — Hilbert Transform - Phasor Components (returns inphase, quadrature)
HT_SINE       — Hilbert Transform - SineWave (returns sine, leadsine)
HT_TRENDMODE  — Hilbert Transform - Trend vs Cycle Mode (1=trend, 0=cycle)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ferro_ta._ferro_ta import (
    ht_dcperiod as _ht_dcperiod,
)
from ferro_ta._ferro_ta import (
    ht_dcphase as _ht_dcphase,
)
from ferro_ta._ferro_ta import (
    ht_phasor as _ht_phasor,
)
from ferro_ta._ferro_ta import (
    ht_sine as _ht_sine,
)
from ferro_ta._ferro_ta import (
    ht_trendline as _ht_trendline,
)
from ferro_ta._ferro_ta import (
    ht_trendmode as _ht_trendmode,
)
from ferro_ta._utils import _to_f64
from ferro_ta.core.exceptions import _normalize_rust_error


def HT_TRENDLINE(close: ArrayLike) -> np.ndarray:
    """Hilbert Transform - Instantaneous Trendline.

    Computes the underlying trend of the price series using the Hilbert
    Transform.  The trendline is the dominant-cycle-period average of the
    smoothed price.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.

    Returns
    -------
    numpy.ndarray
        Trendline values; first 63 entries are ``NaN``.
    """
    try:
        return _ht_trendline(_to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def HT_DCPERIOD(close: ArrayLike) -> np.ndarray:
    """Hilbert Transform - Dominant Cycle Period.

    Estimates the current dominant cycle period in bars using the Hilbert
    Transform.  Values are smoothed and clamped to [6, 50].

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.

    Returns
    -------
    numpy.ndarray
        Dominant cycle period values; first 63 entries are ``NaN``.
    """
    try:
        return _ht_dcperiod(_to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def HT_DCPHASE(close: ArrayLike) -> np.ndarray:
    """Hilbert Transform - Dominant Cycle Phase.

    Returns the instantaneous phase (in degrees) of the dominant cycle.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.

    Returns
    -------
    numpy.ndarray
        Phase values in degrees; first 63 entries are ``NaN``.
    """
    try:
        return _ht_dcphase(_to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def HT_PHASOR(
    close: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Hilbert Transform - Phasor Components.

    Returns the In-Phase (I) and Quadrature (Q) components of the Hilbert
    Transform.  These represent the real and imaginary parts of the analytic
    signal derived from the price series.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(inphase, quadrature)`` — two arrays; first 63 entries are ``NaN``.
    """
    try:
        return _ht_phasor(_to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def HT_SINE(
    close: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Hilbert Transform - SineWave.

    Returns the sine and lead-sine (45-degree lead) of the dominant cycle
    phase.  Used to detect cycle turning points.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(sine, leadsine)`` — two arrays; first 63 entries are ``NaN``.
    """
    try:
        return _ht_sine(_to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


def HT_TRENDMODE(close: ArrayLike) -> np.ndarray:
    """Hilbert Transform - Trend vs Cycle Mode.

    Returns 1 when the market is in a trending mode (dominant cycle period
    below 20 bars) and 0 when in a cycling mode.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.

    Returns
    -------
    numpy.ndarray[int32]
        Array of 1 (trending) or 0 (cycling).
    """
    try:
        return _ht_trendmode(_to_f64(close))
    except ValueError as e:
        _normalize_rust_error(e)


__all__ = [
    "HT_TRENDLINE",
    "HT_DCPERIOD",
    "HT_DCPHASE",
    "HT_PHASOR",
    "HT_SINE",
    "HT_TRENDMODE",
]
