"""
type stubs for ferro_ta.
Generated for IDE auto-completion and static type checking.
"""

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

_F = TypeVar("_F", bound=Callable[..., Any])

__version__: str

# ---------------------------------------------------------------------------
# Overlap Studies
# ---------------------------------------------------------------------------

def SMA(real: ArrayLike, timeperiod: int = 30) -> NDArray[np.float64]: ...
def EMA(real: ArrayLike, timeperiod: int = 30) -> NDArray[np.float64]: ...
def WMA(real: ArrayLike, timeperiod: int = 30) -> NDArray[np.float64]: ...
def DEMA(real: ArrayLike, timeperiod: int = 30) -> NDArray[np.float64]: ...
def TEMA(real: ArrayLike, timeperiod: int = 30) -> NDArray[np.float64]: ...
def TRIMA(real: ArrayLike, timeperiod: int = 30) -> NDArray[np.float64]: ...
def KAMA(real: ArrayLike, timeperiod: int = 30) -> NDArray[np.float64]: ...
def T3(
    real: ArrayLike, timeperiod: int = 5, vfactor: float = 0.7
) -> NDArray[np.float64]: ...
def BBANDS(
    real: ArrayLike,
    timeperiod: int = 5,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
    matype: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
def MACD(
    real: ArrayLike,
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
def MACDFIX(
    real: ArrayLike,
    signalperiod: int = 9,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
def MACDEXT(
    real: ArrayLike,
    fastperiod: int = 12,
    fastmatype: int = 0,
    slowperiod: int = 26,
    slowmatype: int = 0,
    signalperiod: int = 9,
    signalmatype: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
def SAR(
    high: ArrayLike,
    low: ArrayLike,
    acceleration: float = 0.02,
    maximum: float = 0.2,
) -> NDArray[np.float64]: ...
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
) -> NDArray[np.float64]: ...
def MA(
    real: ArrayLike, timeperiod: int = 30, matype: int = 0
) -> NDArray[np.float64]: ...
def MAVP(
    real: ArrayLike,
    periods: ArrayLike,
    minperiod: int = 2,
    maxperiod: int = 30,
    matype: int = 0,
) -> NDArray[np.float64]: ...
def MAMA(
    real: ArrayLike,
    fastlimit: float = 0.5,
    slowlimit: float = 0.05,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def MIDPOINT(real: ArrayLike, timeperiod: int = 14) -> NDArray[np.float64]: ...
def MIDPRICE(
    high: ArrayLike,
    low: ArrayLike,
    timeperiod: int = 14,
) -> NDArray[np.float64]: ...

# ---------------------------------------------------------------------------
# Momentum Indicators
# ---------------------------------------------------------------------------

def RSI(real: ArrayLike, timeperiod: int = 14) -> NDArray[np.float64]: ...
def MOM(real: ArrayLike, timeperiod: int = 10) -> NDArray[np.float64]: ...
def ROC(real: ArrayLike, timeperiod: int = 10) -> NDArray[np.float64]: ...
def ROCP(real: ArrayLike, timeperiod: int = 10) -> NDArray[np.float64]: ...
def ROCR(real: ArrayLike, timeperiod: int = 10) -> NDArray[np.float64]: ...
def ROCR100(real: ArrayLike, timeperiod: int = 10) -> NDArray[np.float64]: ...
def WILLR(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
) -> NDArray[np.float64]: ...
def AROON(
    high: ArrayLike,
    low: ArrayLike,
    timeperiod: int = 14,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def AROONOSC(
    high: ArrayLike,
    low: ArrayLike,
    timeperiod: int = 14,
) -> NDArray[np.float64]: ...
def CCI(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
) -> NDArray[np.float64]: ...
def MFI(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike,
    timeperiod: int = 14,
) -> NDArray[np.float64]: ...
def BOP(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> NDArray[np.float64]: ...
def STOCHF(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_matype: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def STOCH(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    fastk_period: int = 5,
    slowk_period: int = 3,
    slowk_matype: int = 0,
    slowd_period: int = 3,
    slowd_matype: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def STOCHRSI(
    real: ArrayLike,
    timeperiod: int = 14,
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_matype: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def APO(
    real: ArrayLike,
    fastperiod: int = 12,
    slowperiod: int = 26,
    matype: int = 0,
) -> NDArray[np.float64]: ...
def PPO(
    real: ArrayLike,
    fastperiod: int = 12,
    slowperiod: int = 26,
    matype: int = 0,
) -> NDArray[np.float64]: ...
def CMO(real: ArrayLike, timeperiod: int = 14) -> NDArray[np.float64]: ...
def PLUS_DM(
    high: ArrayLike,
    low: ArrayLike,
    timeperiod: int = 14,
) -> NDArray[np.float64]: ...
def MINUS_DM(
    high: ArrayLike,
    low: ArrayLike,
    timeperiod: int = 14,
) -> NDArray[np.float64]: ...
def PLUS_DI(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
) -> NDArray[np.float64]: ...
def MINUS_DI(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
) -> NDArray[np.float64]: ...
def DX(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
) -> NDArray[np.float64]: ...
def ADX(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
) -> NDArray[np.float64]: ...
def ADXR(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
) -> NDArray[np.float64]: ...
def TRIX(real: ArrayLike, timeperiod: int = 30) -> NDArray[np.float64]: ...
def ULTOSC(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod1: int = 7,
    timeperiod2: int = 14,
    timeperiod3: int = 28,
) -> NDArray[np.float64]: ...
def TRANGE(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> NDArray[np.float64]: ...

# ---------------------------------------------------------------------------
# Volume Indicators
# ---------------------------------------------------------------------------

def AD(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike,
) -> NDArray[np.float64]: ...
def ADOSC(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike,
    fastperiod: int = 3,
    slowperiod: int = 10,
) -> NDArray[np.float64]: ...
def OBV(
    real: ArrayLike,
    volume: ArrayLike,
) -> NDArray[np.float64]: ...

# ---------------------------------------------------------------------------
# Volatility Indicators
# ---------------------------------------------------------------------------

def ATR(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
) -> NDArray[np.float64]: ...
def NATR(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
) -> NDArray[np.float64]: ...

# ---------------------------------------------------------------------------
# Statistic Functions
# ---------------------------------------------------------------------------

def STDDEV(
    real: ArrayLike,
    timeperiod: int = 5,
    nbdev: float = 1.0,
) -> NDArray[np.float64]: ...
def VAR(
    real: ArrayLike,
    timeperiod: int = 5,
    nbdev: float = 1.0,
) -> NDArray[np.float64]: ...
def LINEARREG(real: ArrayLike, timeperiod: int = 14) -> NDArray[np.float64]: ...
def LINEARREG_SLOPE(real: ArrayLike, timeperiod: int = 14) -> NDArray[np.float64]: ...
def LINEARREG_INTERCEPT(
    real: ArrayLike, timeperiod: int = 14
) -> NDArray[np.float64]: ...
def LINEARREG_ANGLE(real: ArrayLike, timeperiod: int = 14) -> NDArray[np.float64]: ...
def TSF(real: ArrayLike, timeperiod: int = 14) -> NDArray[np.float64]: ...
def BETA(
    real0: ArrayLike,
    real1: ArrayLike,
    timeperiod: int = 5,
) -> NDArray[np.float64]: ...
def CORREL(
    real0: ArrayLike,
    real1: ArrayLike,
    timeperiod: int = 30,
) -> NDArray[np.float64]: ...

# ---------------------------------------------------------------------------
# Price Transforms
# ---------------------------------------------------------------------------

def AVGPRICE(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> NDArray[np.float64]: ...
def MEDPRICE(high: ArrayLike, low: ArrayLike) -> NDArray[np.float64]: ...
def TYPPRICE(
    high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.float64]: ...
def WCLPRICE(
    high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.float64]: ...

# ---------------------------------------------------------------------------
# Cycle Indicators
# ---------------------------------------------------------------------------

def HT_TRENDLINE(real: ArrayLike) -> NDArray[np.float64]: ...
def HT_DCPERIOD(real: ArrayLike) -> NDArray[np.float64]: ...
def HT_DCPHASE(real: ArrayLike) -> NDArray[np.float64]: ...
def HT_PHASOR(real: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def HT_SINE(real: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def HT_TRENDMODE(real: ArrayLike) -> NDArray[np.int32]: ...

# ---------------------------------------------------------------------------
# Math Operators
# ---------------------------------------------------------------------------

def ADD(real0: ArrayLike, real1: ArrayLike) -> NDArray[np.float64]: ...
def SUB(real0: ArrayLike, real1: ArrayLike) -> NDArray[np.float64]: ...
def MULT(real0: ArrayLike, real1: ArrayLike) -> NDArray[np.float64]: ...
def DIV(real0: ArrayLike, real1: ArrayLike) -> NDArray[np.float64]: ...
def SUM(real: ArrayLike, timeperiod: int = 30) -> NDArray[np.float64]: ...
def MAX(real: ArrayLike, timeperiod: int = 30) -> NDArray[np.float64]: ...
def MIN(real: ArrayLike, timeperiod: int = 30) -> NDArray[np.float64]: ...
def MAXINDEX(real: ArrayLike, timeperiod: int = 30) -> NDArray[np.int32]: ...
def MININDEX(real: ArrayLike, timeperiod: int = 30) -> NDArray[np.int32]: ...

# Math Transforms
def ACOS(real: ArrayLike) -> NDArray[np.float64]: ...
def ASIN(real: ArrayLike) -> NDArray[np.float64]: ...
def ATAN(real: ArrayLike) -> NDArray[np.float64]: ...
def CEIL(real: ArrayLike) -> NDArray[np.float64]: ...
def COS(real: ArrayLike) -> NDArray[np.float64]: ...
def COSH(real: ArrayLike) -> NDArray[np.float64]: ...
def EXP(real: ArrayLike) -> NDArray[np.float64]: ...
def FLOOR(real: ArrayLike) -> NDArray[np.float64]: ...
def LN(real: ArrayLike) -> NDArray[np.float64]: ...
def LOG10(real: ArrayLike) -> NDArray[np.float64]: ...
def SIN(real: ArrayLike) -> NDArray[np.float64]: ...
def SINH(real: ArrayLike) -> NDArray[np.float64]: ...
def SQRT(real: ArrayLike) -> NDArray[np.float64]: ...
def TAN(real: ArrayLike) -> NDArray[np.float64]: ...
def TANH(real: ArrayLike) -> NDArray[np.float64]: ...

# ---------------------------------------------------------------------------
# Extended Indicators (Phase 8 + 9)
# ---------------------------------------------------------------------------

def VWAP(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike,
    timeperiod: int = 0,
) -> NDArray[np.float64]: ...
def SUPERTREND(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 7,
    multiplier: float = 3.0,
) -> tuple[NDArray[np.float64], NDArray[np.int8]]: ...
def ICHIMOKU(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]: ...
def DONCHIAN(
    high: ArrayLike,
    low: ArrayLike,
    timeperiod: int = 20,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
def PIVOT_POINTS(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    method: str = "classic",
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]: ...
def KELTNER_CHANNELS(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
def HULL_MA(
    close: ArrayLike,
    timeperiod: int = 16,
) -> NDArray[np.float64]: ...
def CHANDELIER_EXIT(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 22,
    multiplier: float = 3.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def VWMA(
    close: ArrayLike,
    volume: ArrayLike,
    timeperiod: int = 20,
) -> NDArray[np.float64]: ...
def CHOPPINESS_INDEX(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    timeperiod: int = 14,
) -> NDArray[np.float64]: ...

# ---------------------------------------------------------------------------
# Streaming / Incremental API (Phase 3)
# ---------------------------------------------------------------------------

class StreamingSMA:
    period: int
    def __init__(self, period: int) -> None: ...
    def update(self, value: float) -> float: ...
    def reset(self) -> None: ...

class StreamingEMA:
    period: int
    def __init__(self, period: int) -> None: ...
    def update(self, value: float) -> float: ...
    def reset(self) -> None: ...

class StreamingRSI:
    period: int
    def __init__(self, period: int = 14) -> None: ...
    def update(self, value: float) -> float: ...
    def reset(self) -> None: ...

class StreamingATR:
    period: int
    def __init__(self, period: int = 14) -> None: ...
    def update(self, high: float, low: float, close: float) -> float: ...
    def reset(self) -> None: ...

class StreamingBBands:
    period: int
    def __init__(
        self,
        period: int = 20,
        nbdevup: float = 2.0,
        nbdevdn: float = 2.0,
    ) -> None: ...
    def update(self, value: float) -> tuple[float, float, float]: ...
    def reset(self) -> None: ...

class StreamingMACD:
    def __init__(
        self,
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9,
    ) -> None: ...
    def update(self, value: float) -> tuple[float, float, float]: ...
    def reset(self) -> None: ...

class StreamingStoch:
    def __init__(
        self,
        fastk_period: int = 5,
        slowk_period: int = 3,
        slowd_period: int = 3,
    ) -> None: ...
    def update(self, high: float, low: float, close: float) -> tuple[float, float]: ...
    def reset(self) -> None: ...

class StreamingVWAP:
    def __init__(self) -> None: ...
    def update(self, high: float, low: float, close: float, volume: float) -> float: ...
    def reset(self) -> None: ...

class StreamingSupertrend:
    period: int
    def __init__(self, period: int = 7, multiplier: float = 3.0) -> None: ...
    def update(self, high: float, low: float, close: float) -> tuple[float, int]: ...
    def reset(self) -> None: ...

# ---------------------------------------------------------------------------
# Candlestick Patterns
# ---------------------------------------------------------------------------

def CDL2CROWS(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDL3BLACKCROWS(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDL3INSIDE(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDL3LINESTRIKE(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDL3OUTSIDE(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDL3STARSINSOUTH(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDL3WHITESOLDIERS(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLABANDONEDBABY(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLADVANCEBLOCK(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLBELTHOLD(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLBREAKAWAY(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLCLOSINGMARUBOZU(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLCONCEALBABYSWALL(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLCOUNTERATTACK(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLDARKCLOUDCOVER(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLDOJI(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLDOJISTAR(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLDRAGONFLYDOJI(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLENGULFING(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLEVENINGDOJISTAR(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLEVENINGSTAR(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLGAPSIDESIDEWHITE(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLGRAVESTONEDOJI(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLHAMMER(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLHANGINGMAN(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLHARAMI(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLHARAMICROSS(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLHIGHWAVE(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLHIKKAKE(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLHIKKAKEMOD(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLHOMINGPIGEON(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLIDENTICAL3CROWS(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLINNECK(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLINVERTEDHAMMER(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLKICKING(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLKICKINGBYLENGTH(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLLADDERBOTTOM(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLLONGLEGGEDDOJI(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLLONGLINE(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLMARUBOZU(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLMATCHINGLOW(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLMATHOLD(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLMORNINGDOJISTAR(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLMORNINGSTAR(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLONNECK(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLPIERCING(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLRICKSHAWMAN(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLRISEFALL3METHODS(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLSEPARATINGLINES(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLSHOOTINGSTAR(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLSHORTLINE(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLSPINNINGTOP(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLSTALLEDPATTERN(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLSTICKSANDWICH(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLTAKURI(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLTASUKIGAP(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLTHRUSTING(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLTRISTAR(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLUNIQUE3RIVER(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLUPSIDEGAP2CROWS(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...
def CDLXSIDEGAP3METHODS(
    open: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.int32]: ...

# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------

from ferro_ta.batch import batch_apply as batch_apply
from ferro_ta.batch import batch_ema as batch_ema
from ferro_ta.batch import batch_rsi as batch_rsi
from ferro_ta.batch import batch_sma as batch_sma
from ferro_ta.batch import compute_many as compute_many

# ---------------------------------------------------------------------------
# Exception hierarchy (re-exported from ferro_ta.exceptions)
# ---------------------------------------------------------------------------

class FerroTAError(Exception):
    code: str
    suggestion: str | None
    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        suggestion: str | None = None,
    ) -> None: ...

class FerroTAValueError(FerroTAError, ValueError):
    code: str
    suggestion: str | None

class FerroTAInputError(FerroTAError, ValueError):
    code: str
    suggestion: str | None

# ---------------------------------------------------------------------------
# API discovery (ferro_ta.api_info)
# ---------------------------------------------------------------------------

def about() -> dict[str, Any]: ...
def indicators(category: str | None = None) -> list[dict[str, Any]]: ...
def info(func_or_name: Callable[..., Any] | str) -> dict[str, Any]: ...
def methods(category: str | None = None) -> list[dict[str, Any]]: ...

# ---------------------------------------------------------------------------
# Logging utilities (ferro_ta.logging_utils)
# ---------------------------------------------------------------------------

def get_logger() -> logging.Logger: ...
def enable_debug(fmt: str = ...) -> None: ...
def disable_debug() -> None: ...
def debug_mode(fmt: str = ...) -> AbstractContextManager[logging.Logger]: ...
def log_call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...
def benchmark(
    func: Callable[..., Any],
    *args: Any,
    n: int = 100,
    warmup: int = 5,
    **kwargs: Any,
) -> dict[str, float]: ...
def traced(func: _F) -> _F: ...
