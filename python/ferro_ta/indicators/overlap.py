"""
Overlap Studies — Moving averages and bands that overlay directly on the price chart.

Functions
---------
SMA      — Simple Moving Average
EMA      — Exponential Moving Average
WMA      — Weighted Moving Average
DEMA     — Double Exponential Moving Average
TEMA     — Triple Exponential Moving Average
TRIMA    — Triangular Moving Average
KAMA     — Kaufman Adaptive Moving Average
T3       — Triple Exponential Moving Average (Tillson T3)
BBANDS   — Bollinger Bands
MACD     — Moving Average Convergence/Divergence
MACDFIX  — MACD with fixed 12/26 periods
MACDEXT  — MACD with controllable MA types
SAR      — Parabolic SAR
SAREXT   — Parabolic SAR Extended
MA       — Generic Moving Average (dispatches on matype)
MAVP     — Moving Average with Variable Period
MAMA     — MESA Adaptive Moving Average
MIDPOINT — MidPoint over period
MIDPRICE — MidPrice over period (High/Low)
ALMA     — Arnaud Legoux Moving Average
ZLEMA    — Zero-Lag EMA
FRAMA    — Fractal Adaptive Moving Average
MCGINLEY — McGinley Dynamic
VIDYA    — Variable Index Dynamic Average
ALLIGATOR — Bill Williams Alligator (jaw / teeth / lips)
MA_ENVELOPES — Moving-average envelopes
CHANDE_KROLL_STOP — Chande Kroll Stop
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ferro_ta._ferro_ta import (
    alligator as _alligator,
)
from ferro_ta._ferro_ta import (
    alma as _alma,
)
from ferro_ta._ferro_ta import (
    bbands as _bbands,
)
from ferro_ta._ferro_ta import (
    chande_kroll_stop as _chande_kroll_stop,
)
from ferro_ta._ferro_ta import (
    dema as _dema,
)
from ferro_ta._ferro_ta import (
    ema as _ema,
)
from ferro_ta._ferro_ta import (
    frama as _frama,
)
from ferro_ta._ferro_ta import (
    kama as _kama,
)
from ferro_ta._ferro_ta import (
    ma as _ma,
)
from ferro_ta._ferro_ta import (
    ma_envelopes as _ma_envelopes,
)
from ferro_ta._ferro_ta import (
    macd as _macd,
)
from ferro_ta._ferro_ta import (
    macdext as _macdext,
)
from ferro_ta._ferro_ta import (
    macdfix as _macdfix,
)
from ferro_ta._ferro_ta import (
    mama as _mama,
)
from ferro_ta._ferro_ta import (
    mavp as _mavp,
)
from ferro_ta._ferro_ta import (
    mcginley as _mcginley,
)
from ferro_ta._ferro_ta import (
    midpoint as _midpoint,
)
from ferro_ta._ferro_ta import (
    midprice as _midprice,
)
from ferro_ta._ferro_ta import (
    sar as _sar,
)
from ferro_ta._ferro_ta import (
    sarext as _sarext,
)
from ferro_ta._ferro_ta import (
    sma as _sma,
)
from ferro_ta._ferro_ta import (
    t3 as _t3,
)
from ferro_ta._ferro_ta import (
    tema as _tema,
)
from ferro_ta._ferro_ta import (
    trima as _trima,
)
from ferro_ta._ferro_ta import (
    vidya as _vidya,
)
from ferro_ta._ferro_ta import (
    wma as _wma,
)
from ferro_ta._ferro_ta import (
    zlema as _zlema,
)
from ferro_ta._utils import _to_f64
from ferro_ta.core.exceptions import _normalize_rust_error


def SMA(close: ArrayLike, timeperiod: int = 30) -> np.ndarray:
    """Simple Moving Average.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Number of periods (default 30).

    Returns
    -------
    numpy.ndarray
        Array of SMA values; leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _sma(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def EMA(close: ArrayLike, timeperiod: int = 30) -> np.ndarray:
    """Exponential Moving Average.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Number of periods (default 30).

    Returns
    -------
    numpy.ndarray
        Array of EMA values; leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _ema(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def WMA(close: ArrayLike, timeperiod: int = 30) -> np.ndarray:
    """Weighted Moving Average.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Number of periods (default 30).

    Returns
    -------
    numpy.ndarray
        Array of WMA values; leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _wma(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def DEMA(close: ArrayLike, timeperiod: int = 30) -> np.ndarray:
    """Double Exponential Moving Average.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Number of periods (default 30).

    Returns
    -------
    numpy.ndarray
        Array of DEMA values; leading ``2 * (timeperiod - 1)`` entries are ``NaN``.
    """
    try:
        return _dema(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def TEMA(close: ArrayLike, timeperiod: int = 30) -> np.ndarray:
    """Triple Exponential Moving Average.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Number of periods (default 30).

    Returns
    -------
    numpy.ndarray
        Array of TEMA values; leading ``3 * (timeperiod - 1)`` entries are ``NaN``.
    """
    try:
        return _tema(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def TRIMA(close: ArrayLike, timeperiod: int = 30) -> np.ndarray:
    """Triangular Moving Average.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Number of periods (default 30).

    Returns
    -------
    numpy.ndarray
        Array of TRIMA values; leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _trima(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def KAMA(close: ArrayLike, timeperiod: int = 30) -> np.ndarray:
    """Kaufman Adaptive Moving Average.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Efficiency Ratio lookback period (default 30).

    Returns
    -------
    numpy.ndarray
        Array of KAMA values; leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _kama(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def T3(close: ArrayLike, timeperiod: int = 5, vfactor: float = 0.7) -> np.ndarray:
    """Triple Exponential Moving Average (Tillson T3).

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Number of periods (default 5).
    vfactor : float, optional
        Volume factor (default 0.7).

    Returns
    -------
    numpy.ndarray
        Array of T3 values.
    """
    try:
        return _t3(_to_f64(close), timeperiod, vfactor)
    except ValueError as e:
        _normalize_rust_error(e)


def BBANDS(
    close: ArrayLike,
    timeperiod: int = 5,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Moving average window (default 5).
    nbdevup : float, optional
        Number of standard deviations above the middle band (default 2.0).
    nbdevdn : float, optional
        Number of standard deviations below the middle band (default 2.0).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(upperband, middleband, lowerband)`` — three arrays of equal length.
        Leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _bbands(_to_f64(close), timeperiod, nbdevup, nbdevdn)
    except ValueError as e:
        _normalize_rust_error(e)


def MACD(
    close: ArrayLike,
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moving Average Convergence/Divergence.

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
        ``(macd, signal, histogram)`` — three arrays of equal length.
        Leading values that cannot be computed are ``NaN``.
    """
    try:
        return _macd(_to_f64(close), fastperiod, slowperiod, signalperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def MACDFIX(
    close: ArrayLike,
    signalperiod: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moving Average Convergence/Divergence Fix 12/26.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    signalperiod : int, optional
        Signal EMA period (default 9).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(macd, signal, histogram)`` — three arrays of equal length.
    """
    try:
        return _macdfix(_to_f64(close), signalperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def SAR(
    high: ArrayLike,
    low: ArrayLike,
    acceleration: float = 0.02,
    maximum: float = 0.2,
) -> np.ndarray:
    """Parabolic SAR.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    acceleration : float, optional
        Acceleration factor step (default 0.02).
    maximum : float, optional
        Maximum acceleration factor (default 0.2).

    Returns
    -------
    numpy.ndarray
        Array of SAR values; first entry is ``NaN``.
    """
    try:
        return _sar(_to_f64(high), _to_f64(low), acceleration, maximum)
    except ValueError as e:
        _normalize_rust_error(e)


def MIDPOINT(close: ArrayLike, timeperiod: int = 14) -> np.ndarray:
    """MidPoint over period — (max + min) / 2 of close.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Number of periods (default 14).

    Returns
    -------
    numpy.ndarray
        Array of MIDPOINT values; leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _midpoint(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def MIDPRICE(high: ArrayLike, low: ArrayLike, timeperiod: int = 14) -> np.ndarray:
    """MidPrice over period — (highest high + lowest low) / 2.

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
        Array of MIDPRICE values; leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _midprice(_to_f64(high), _to_f64(low), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def MA(close: ArrayLike, timeperiod: int = 30, matype: int = 0) -> np.ndarray:
    """Generic Moving Average.

    Dispatches to the appropriate MA implementation based on *matype*.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Number of periods (default 30).
    matype : int, optional
        Moving average type (default 0):

        * 0 = SMA (Simple)
        * 1 = EMA (Exponential)
        * 2 = WMA (Weighted)
        * 3 = DEMA (Double EMA)
        * 4 = TEMA (Triple EMA)
        * 5 = TRIMA (Triangular)
        * 6 = KAMA (Kaufman Adaptive)
        * 7 = T3 (Tillson)

    Returns
    -------
    numpy.ndarray
        Array of MA values.
    """
    try:
        return _ma(_to_f64(close), timeperiod, matype)
    except ValueError as e:
        _normalize_rust_error(e)


def MAVP(
    close: ArrayLike,
    periods: ArrayLike,
    minperiod: int = 2,
    maxperiod: int = 30,
) -> np.ndarray:
    """Moving Average with Variable Period.

    Computes a simple moving average at each bar using the period given by the
    corresponding element of *periods*.  Periods are clamped to
    ``[minperiod, maxperiod]``.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    periods : array-like
        Sequence of period values (one per bar, same length as *close*).
    minperiod : int, optional
        Minimum allowed period (default 2).
    maxperiod : int, optional
        Maximum allowed period (default 30).

    Returns
    -------
    numpy.ndarray
        Array of variable-period MA values.
    """
    try:
        return _mavp(_to_f64(close), _to_f64(periods), minperiod, maxperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def MAMA(
    close: ArrayLike,
    fastlimit: float = 0.5,
    slowlimit: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """MESA Adaptive Moving Average.

    Returns the MAMA and FAMA (Following Adaptive MA) lines.  The adaptive
    alpha is derived from the rate of phase change of the Hilbert Transform.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    fastlimit : float, optional
        Upper bound on the adaptive smoothing factor (default 0.5).
    slowlimit : float, optional
        Lower bound on the adaptive smoothing factor (default 0.05).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(mama, fama)`` — two arrays; first 32 entries are ``NaN``.
    """
    try:
        return _mama(_to_f64(close), fastlimit, slowlimit)
    except ValueError as e:
        _normalize_rust_error(e)


def SAREXT(
    high: ArrayLike,
    low: ArrayLike,
    startvalue: float = 0.0,
    offsetonreverse: float = 0.0,
    accelerationinitlong: float = 0.02,
    accelerationlong: float = 0.02,
    accelerationmaxlong: float = 0.2,
    accelerationinitshort: float = 0.02,
    accelerationshort: float = 0.02,
    accelerationmaxshort: float = 0.2,
) -> np.ndarray:
    """Parabolic SAR Extended.

    An extended version of the Parabolic SAR that allows independent
    acceleration parameters for long and short positions, plus an optional
    fixed start value and a gap-on-reverse offset.

    Parameters
    ----------
    high : array-like
        Sequence of high prices.
    low : array-like
        Sequence of low prices.
    startvalue : float, optional
        Fixed initial SAR value (0 = auto-detect, default 0.0).
    offsetonreverse : float, optional
        Multiplier applied to the SAR on trend reversal (default 0.0).
    accelerationinitlong : float, optional
        Initial acceleration factor for long positions (default 0.02).
    accelerationlong : float, optional
        Acceleration step for long positions (default 0.02).
    accelerationmaxlong : float, optional
        Maximum acceleration for long positions (default 0.2).
    accelerationinitshort : float, optional
        Initial acceleration factor for short positions (default 0.02).
    accelerationshort : float, optional
        Acceleration step for short positions (default 0.02).
    accelerationmaxshort : float, optional
        Maximum acceleration for short positions (default 0.2).

    Returns
    -------
    numpy.ndarray
        Array of SAREXT values; first entry is ``NaN``.
    """
    try:
        return _sarext(
            _to_f64(high),
            _to_f64(low),
            startvalue,
            offsetonreverse,
            accelerationinitlong,
            accelerationlong,
            accelerationmaxlong,
            accelerationinitshort,
            accelerationshort,
            accelerationmaxshort,
        )
    except ValueError as e:
        _normalize_rust_error(e)


def MACDEXT(
    close: ArrayLike,
    fastperiod: int = 12,
    fastmatype: int = 1,
    slowperiod: int = 26,
    slowmatype: int = 1,
    signalperiod: int = 9,
    signalmatype: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD with Controllable MA Types.

    Like :func:`MACD` but allows specifying the moving average type for each
    of the fast, slow, and signal lines independently.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    fastperiod : int, optional
        Fast MA period (default 12).
    fastmatype : int, optional
        MA type for the fast line (default 1 = EMA).
    slowperiod : int, optional
        Slow MA period (default 26).
    slowmatype : int, optional
        MA type for the slow line (default 1 = EMA).
    signalperiod : int, optional
        Signal MA period (default 9).
    signalmatype : int, optional
        MA type for the signal line (default 1 = EMA).

    MA type codes: 0=SMA, 1=EMA, 2=WMA.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(macd, signal, histogram)`` — three arrays of equal length.
    """
    try:
        return _macdext(
            _to_f64(close),
            fastperiod,
            fastmatype,
            slowperiod,
            slowmatype,
            signalperiod,
            signalmatype,
        )
    except ValueError as e:
        _normalize_rust_error(e)


def ALMA(
    close: ArrayLike,
    timeperiod: int = 21,
    offset: float = 0.85,
    sigma: float = 6.0,
) -> np.ndarray:
    """Arnaud Legoux Moving Average.

    Gaussian-weighted moving average. The peak sits at
    ``floor(offset * (timeperiod - 1))`` from the oldest bar in the window;
    the width is ``timeperiod / sigma``. Leading ``timeperiod - 1`` values
    are ``NaN``.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Window length (default 21).
    offset : float, optional
        Gaussian peak location in ``[0, 1]`` (default 0.85).
    sigma : float, optional
        Gaussian width control (default 6.0). Must be ``> 0``.

    Returns
    -------
    numpy.ndarray
        ALMA values.
    """
    try:
        return _alma(_to_f64(close), timeperiod, offset, sigma)
    except ValueError as e:
        _normalize_rust_error(e)


def ZLEMA(close: ArrayLike, timeperiod: int = 14) -> np.ndarray:
    """Zero-Lag Exponential Moving Average.

    Applies an EMA to the error-corrected series
    ``close + (close - close[lag])`` where ``lag = floor((timeperiod - 1) / 2)``.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        EMA period (default 14).

    Returns
    -------
    numpy.ndarray
        ZLEMA values.
    """
    try:
        return _zlema(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def FRAMA(close: ArrayLike, timeperiod: int = 16) -> np.ndarray:
    """Fractal Adaptive Moving Average.

    Adaptive EMA whose smoothing follows the fractal dimension of ``close``
    over ``timeperiod`` (two half-windows versus the full window).
    ``alpha = exp(-4.6 * (dim - 1))`` clipped to ``[0.01, 1]``.
    Prefer an even ``timeperiod``.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Fractal lookback (default 16). Must be ``>= 2``.

    Returns
    -------
    numpy.ndarray
        FRAMA values; leading ``timeperiod - 1`` entries are ``NaN``.
    """
    try:
        return _frama(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def MCGINLEY(close: ArrayLike, timeperiod: int = 14) -> np.ndarray:
    """McGinley Dynamic.

    ``MD = MD[1] + (close - MD[1]) / (timeperiod * (close / MD[1]) ** 4)``,
    seeded with the SMA of the first ``timeperiod`` bars.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Smoothing period (default 14).

    Returns
    -------
    numpy.ndarray
        McGinley Dynamic values.
    """
    try:
        return _mcginley(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def VIDYA(
    close: ArrayLike,
    timeperiod: int = 14,
    cmo_period: int = 9,
) -> np.ndarray:
    """Variable Index Dynamic Average.

    EMA whose smoothing constant is scaled by ``|CMO(cmo_period)| / 100``.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Base EMA period (default 14).
    cmo_period : int, optional
        Chande Momentum Oscillator lookback (default 9).

    Returns
    -------
    numpy.ndarray
        VIDYA values.
    """
    try:
        return _vidya(_to_f64(close), timeperiod, cmo_period)
    except ValueError as e:
        _normalize_rust_error(e)


def ALLIGATOR(
    high: ArrayLike,
    low: ArrayLike,
    jaw_period: int = 13,
    jaw_shift: int = 8,
    teeth_period: int = 8,
    teeth_shift: int = 5,
    lips_period: int = 5,
    lips_shift: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bill Williams Alligator (jaw, teeth, lips).

    Each line is an SMMA of the median price ``(high + low) / 2``, shifted
    forward by the given offset.

    Parameters
    ----------
    high, low : array-like
        High and low price series.
    jaw_period, teeth_period, lips_period : int, optional
        SMMA periods (defaults 13, 8, 5).
    jaw_shift, teeth_shift, lips_shift : int, optional
        Forward displacements (defaults 8, 5, 3).

    Returns
    -------
    jaw, teeth, lips : numpy.ndarray
    """
    try:
        return _alligator(
            _to_f64(high),
            _to_f64(low),
            jaw_period,
            jaw_shift,
            teeth_period,
            teeth_shift,
            lips_period,
            lips_shift,
        )
    except ValueError as e:
        _normalize_rust_error(e)


def MA_ENVELOPES(
    close: ArrayLike,
    timeperiod: int = 20,
    percent: float = 2.5,
    matype: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moving-average envelopes.

    ``upper = MA * (1 + percent / 100)``, ``lower = MA * (1 - percent / 100)``.

    Parameters
    ----------
    close : array-like
        Sequence of closing prices.
    timeperiod : int, optional
        Moving-average window (default 20).
    percent : float, optional
        Envelope width in percent (default 2.5).
    matype : int, optional
        Moving-average type, same as :func:`MA` (default 0 = SMA).

    Returns
    -------
    upper, middle, lower : numpy.ndarray
    """
    try:
        return _ma_envelopes(_to_f64(close), timeperiod, percent, matype)
    except ValueError as e:
        _normalize_rust_error(e)


def CHANDE_KROLL_STOP(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 10,
    multiplier: float = 1.0,
    stop_period: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """Chande Kroll Stop.

    First stops are ``HH(high, p) - x * ATR(p)`` and ``LL(low, p) + x * ATR(p)``.
    The published stops are the ``stop_period``-bar highest first-long and
    lowest first-short.

    Parameters
    ----------
    high, low, close : array-like
        OHLC series of equal length.
    timeperiod : int, optional
        ``p`` — HH/LL and ATR lookback (default 10).
    multiplier : float, optional
        ``x`` — ATR multiplier (default 1.0).
    stop_period : int, optional
        ``q`` — stop confirmation window (default 9).

    Returns
    -------
    long_stop, short_stop : numpy.ndarray
    """
    try:
        return _chande_kroll_stop(
            _to_f64(high),
            _to_f64(low),
            _to_f64(close),
            timeperiod,
            multiplier,
            stop_period,
        )
    except ValueError as e:
        _normalize_rust_error(e)


__all__ = [
    "SMA",
    "EMA",
    "WMA",
    "DEMA",
    "TEMA",
    "TRIMA",
    "KAMA",
    "T3",
    "BBANDS",
    "MACD",
    "MACDFIX",
    "MACDEXT",
    "SAR",
    "SAREXT",
    "MA",
    "MAVP",
    "MAMA",
    "MIDPOINT",
    "MIDPRICE",
    "ALMA",
    "ZLEMA",
    "FRAMA",
    "MCGINLEY",
    "VIDYA",
    "ALLIGATOR",
    "MA_ENVELOPES",
    "CHANDE_KROLL_STOP",
]
