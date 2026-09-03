"""
Oscillator Indicators — AO, AC, KST, TSI, and related.

Functions
---------
AO      — Awesome Oscillator
AC      — Accelerator Oscillator
PO      — Price Oscillator (SMA)
DPO     — Detrended Price Oscillator
RVI     — Relative Vigor Index
CHO     — Chaikin Oscillator (same math as ADOSC)
KST     — Know Sure Thing
TSI     — True Strength Index
VORTEX  — Vortex Indicator
STC     — Schaff Trend Cycle
GATOR   — Gator Oscillator
COPPOCK — Coppock Curve
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ferro_ta._ferro_ta import (
    ac as _ac,
)
from ferro_ta._ferro_ta import (
    ao as _ao,
)
from ferro_ta._ferro_ta import (
    cho as _cho,
)
from ferro_ta._ferro_ta import (
    coppock as _coppock,
)
from ferro_ta._ferro_ta import (
    dpo as _dpo,
)
from ferro_ta._ferro_ta import (
    gator as _gator,
)
from ferro_ta._ferro_ta import (
    kst as _kst,
)
from ferro_ta._ferro_ta import (
    po as _po,
)
from ferro_ta._ferro_ta import (
    rvi as _rvi,
)
from ferro_ta._ferro_ta import (
    stc as _stc,
)
from ferro_ta._ferro_ta import (
    tsi as _tsi,
)
from ferro_ta._ferro_ta import (
    vortex as _vortex,
)
from ferro_ta._utils import _to_f64
from ferro_ta.core.exceptions import _normalize_rust_error


def AO(
    high: ArrayLike,
    low: ArrayLike,
    fastperiod: int = 5,
    slowperiod: int = 34,
) -> np.ndarray:
    """Awesome Oscillator: SMA(median, 5) − SMA(median, 34)."""
    try:
        return _ao(_to_f64(high), _to_f64(low), fastperiod, slowperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def AC(
    high: ArrayLike,
    low: ArrayLike,
    fastperiod: int = 5,
    slowperiod: int = 34,
    smoothperiod: int = 5,
) -> np.ndarray:
    """Accelerator Oscillator: AO − SMA(AO, 5)."""
    try:
        return _ac(_to_f64(high), _to_f64(low), fastperiod, slowperiod, smoothperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def PO(
    close: ArrayLike,
    fastperiod: int = 10,
    slowperiod: int = 21,
) -> np.ndarray:
    """Price Oscillator: SMA(close, fast) − SMA(close, slow)."""
    try:
        return _po(_to_f64(close), fastperiod, slowperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def DPO(close: ArrayLike, timeperiod: int = 20) -> np.ndarray:
    """Detrended Price Oscillator."""
    try:
        return _dpo(_to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def RVI(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Relative Vigor Index (Ehlers) and 4-bar SWMA signal."""
    try:
        return _rvi(
            _to_f64(open),
            _to_f64(high),
            _to_f64(low),
            _to_f64(close),
            timeperiod,
        )
    except ValueError as e:
        _normalize_rust_error(e)


def CHO(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike,
    fastperiod: int = 3,
    slowperiod: int = 10,
) -> np.ndarray:
    """Chaikin Oscillator — same math as ``ADOSC``."""
    try:
        return _cho(
            _to_f64(high),
            _to_f64(low),
            _to_f64(close),
            _to_f64(volume),
            fastperiod,
            slowperiod,
        )
    except ValueError as e:
        _normalize_rust_error(e)


def KST(
    close: ArrayLike,
    roc1: int = 10,
    roc2: int = 15,
    roc3: int = 20,
    roc4: int = 30,
    sma1: int = 10,
    sma2: int = 10,
    sma3: int = 10,
    sma4: int = 15,
    signalperiod: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """Know Sure Thing and signal line."""
    try:
        return _kst(
            _to_f64(close),
            roc1,
            roc2,
            roc3,
            roc4,
            sma1,
            sma2,
            sma3,
            sma4,
            signalperiod,
        )
    except ValueError as e:
        _normalize_rust_error(e)


def TSI(
    close: ArrayLike,
    longperiod: int = 25,
    shortperiod: int = 13,
    signalperiod: int = 13,
) -> tuple[np.ndarray, np.ndarray]:
    """True Strength Index and signal line."""
    try:
        return _tsi(_to_f64(close), longperiod, shortperiod, signalperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def VORTEX(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    """Vortex Indicator: ``(+VI, −VI)``."""
    try:
        return _vortex(_to_f64(high), _to_f64(low), _to_f64(close), timeperiod)
    except ValueError as e:
        _normalize_rust_error(e)


def STC(
    close: ArrayLike,
    fastperiod: int = 23,
    slowperiod: int = 50,
    cycleperiod: int = 10,
    d1: int = 3,
    d2: int = 3,
) -> np.ndarray:
    """Schaff Trend Cycle."""
    try:
        return _stc(_to_f64(close), fastperiod, slowperiod, cycleperiod, d1, d2)
    except ValueError as e:
        _normalize_rust_error(e)


def GATOR(
    high: ArrayLike,
    low: ArrayLike,
    jaw_period: int = 13,
    jaw_shift: int = 8,
    teeth_period: int = 8,
    teeth_shift: int = 5,
    lips_period: int = 5,
    lips_shift: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Gator Oscillator from Alligator SMMA lines.

    Returns ``(upper, lower)`` where upper is ``|jaw − teeth|`` and lower is
    ``−|teeth − lips|``.
    """
    try:
        return _gator(
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


def COPPOCK(
    close: ArrayLike,
    wma_period: int = 10,
    roc1_period: int = 14,
    roc2_period: int = 11,
) -> np.ndarray:
    """Coppock Curve: WMA of ROC(14) + ROC(11)."""
    try:
        return _coppock(_to_f64(close), wma_period, roc1_period, roc2_period)
    except ValueError as e:
        _normalize_rust_error(e)


__all__ = [
    "AO",
    "AC",
    "PO",
    "DPO",
    "RVI",
    "CHO",
    "KST",
    "TSI",
    "VORTEX",
    "STC",
    "GATOR",
    "COPPOCK",
]
