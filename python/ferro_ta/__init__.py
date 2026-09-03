"""
ferro_ta — A fast Technical Analysis library powered by Rust and PyO3.

Drop-in alternative to TA-Lib with pre-compiled wheels for all platforms.

Indicators are organized into sub-modules matching TA-Lib's category structure,
and are also importable directly from this top-level package for convenience.

Sub-packages
------------
* :mod:`ferro_ta.indicators` — All indicator functions (overlap, momentum, volume, volatility, statistic, cycle, pattern, price_transform, math_ops, extended)
* :mod:`ferro_ta.core`       — Core utilities (exceptions, config, logging, registry, raw)
* :mod:`ferro_ta.data`       — Data utilities (streaming, batch, chunked, resampling, aggregation, adapters)
* :mod:`ferro_ta.analysis`   — Analysis tools (portfolio, backtest, regime, cross_asset, attribution, signals, features, crypto, options, futures, derivatives payoff)
* :mod:`ferro_ta.tools`      — Developer tools (tools, viz, dashboard, alerts, dsl, pipeline, workflow, api_info, gpu)

Sub-modules (also accessible via sub-packages above)
-----------------------------------------------------
* :mod:`ferro_ta.indicators.overlap`         — Overlap Studies (SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, T3, MACD, BBANDS, SAR, MA, MAVP, MAMA, SAREXT, MACDEXT, …)
* :mod:`ferro_ta.indicators.momentum`        — Momentum Indicators (RSI, STOCH, ADX, CCI, WILLR, AROON, MFI, …)
* :mod:`ferro_ta.indicators.volume`          — Volume Indicators (AD, ADOSC, OBV)
* :mod:`ferro_ta.indicators.volatility`      — Volatility Indicators (ATR, NATR, TRANGE)
* :mod:`ferro_ta.indicators.statistic`       — Statistic Functions (STDDEV, VAR, LINEARREG, BETA, CORREL, …)
* :mod:`ferro_ta.indicators.price_transform` — Price Transformations (AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE)
* :mod:`ferro_ta.indicators.pattern`         — Pattern Recognition (CDLDOJI, CDLENGULFING, CDLHAMMER, …)
* :mod:`ferro_ta.indicators.cycle`           — Cycle Indicators (HT_TRENDLINE, HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDMODE)
* :mod:`ferro_ta.indicators.math_ops`        — Math Operators/Transforms (ADD, SUB, MULT, DIV, SUM, MAX, MIN, ACOS, SIN, …)
* :mod:`ferro_ta.indicators.extended`        — Extended Indicators (VWAP, SUPERTREND, ICHIMOKU, DONCHIAN, PIVOT_POINTS, KELTNER_CHANNELS, HULL_MA, CHANDELIER_EXIT, VWMA, CHOPPINESS_INDEX)
* :mod:`ferro_ta.data.streaming`             — Streaming / Incremental API (bar-by-bar stateful classes for live trading)
* :mod:`ferro_ta.data.batch`                 — Batch Execution API (run SMA/EMA/RSI on 2-D arrays of multiple series)
* :mod:`ferro_ta.data.resampling`            — OHLCV resampling and multi-timeframe API
* :mod:`ferro_ta.data.aggregation`           — Tick/trade aggregation pipeline
* :mod:`ferro_ta.tools.dsl`                  — Strategy expression DSL
* :mod:`ferro_ta.analysis.signals`           — Signal composition and screening
* :mod:`ferro_ta.analysis.portfolio`         — Portfolio and multi-asset analytics
* :mod:`ferro_ta.analysis.cross_asset`       — Cross-asset and relative strength
* :mod:`ferro_ta.analysis.features`          — Feature matrix and ML readiness
* :mod:`ferro_ta.analysis.options`           — Options pricing, Greeks, IV, smile, and chain analytics
* :mod:`ferro_ta.analysis.futures`           — Futures basis, carry, roll, and curve analytics
* :mod:`ferro_ta.tools.viz`                  — Charting and visualisation API
* :mod:`ferro_ta.data.adapters`              — Market data adapters

Usage
-----
>>> import numpy as np
>>> from ferro_ta import SMA, EMA, RSI, MACD, BBANDS
>>> close = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 13.5, 12.5])
>>> SMA(close, timeperiod=3)
array([ nan,  nan, 11. , 12. , 13. , 13.5, 13.33...])

>>> # Or import from sub-packages:
>>> from ferro_ta.indicators.overlap import SMA, BBANDS
>>> from ferro_ta.indicators.momentum import RSI, ADX
>>> from ferro_ta.indicators.volatility import ATR
>>> from ferro_ta.indicators.cycle import HT_TRENDLINE, HT_DCPERIOD
>>> # Backward-compat flat imports still work:
>>> from ferro_ta.overlap import SMA  # noqa: F401 (stub)
"""

from __future__ import annotations

import re as _re
import sys as _sys
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path as _Path

try:
    import tomllib as _tomllib
except ImportError:  # pragma: no cover
    try:
        import tomli as _tomllib  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover
        _tomllib = None  # type: ignore[assignment]


def _detect_version() -> str:
    try:
        return _dist_version("ferro-ta")
    except _PackageNotFoundError:
        pass

    if _tomllib is not None:
        pyproject_toml = _Path(__file__).resolve().parents[2] / "pyproject.toml"
        if pyproject_toml.is_file():
            try:
                with pyproject_toml.open("rb") as handle:
                    data = _tomllib.load(handle)
                return data.get("project", {}).get("version", "0+unknown")
            except Exception:
                pass

    pyproject_toml = _Path(__file__).resolve().parents[2] / "pyproject.toml"
    if pyproject_toml.is_file():
        try:
            text = pyproject_toml.read_text(encoding="utf-8")
            match = _re.search(r'^version\s*=\s*"([^"]+)"', text, _re.MULTILINE)
            if match:
                return match.group(1)
        except Exception:
            pass

    return "0+unknown"


__version__ = _detect_version()

# ---------------------------------------------------------------------------
# Exceptions — exported at the top level for convenient catching
# ---------------------------------------------------------------------------
from ferro_ta.core.exceptions import (  # noqa: F401
    FerroTAError,
    FerroTaError,
    FerroTAInputError,
    FerroTAValueError,
    InsufficientDataError,
    InvalidInputError,
    InvalidPeriodError,
    LengthMismatchError,
    NumericConvergenceError,
)

# ---------------------------------------------------------------------------
# Cycle Indicators
# ---------------------------------------------------------------------------
from ferro_ta.indicators.cycle import (  # noqa: F401
    HT_DCPERIOD,
    HT_DCPHASE,
    HT_PHASOR,
    HT_SINE,
    HT_TRENDLINE,
    HT_TRENDMODE,
)

# ---------------------------------------------------------------------------
# Math Operators & Math Transforms
# ---------------------------------------------------------------------------
from ferro_ta.indicators.math_ops import (  # noqa: F401
    ACOS,
    ADD,
    ASIN,
    ATAN,
    CEIL,
    COS,
    COSH,
    DIV,
    EXP,
    FLOOR,
    LN,
    LOG10,
    MAX,
    MAXINDEX,
    MIN,
    MININDEX,
    MULT,
    SIN,
    SINH,
    SQRT,
    SUB,
    SUM,
    TAN,
    TANH,
)

# ---------------------------------------------------------------------------
# Momentum Indicators
# ---------------------------------------------------------------------------
from ferro_ta.indicators.momentum import (  # noqa: F401
    ADX,
    ADXR,
    APO,
    AROON,
    AROONOSC,
    BOP,
    CCI,
    CMO,
    CRSI,
    DMI,
    DX,
    ELDER_RAY,
    FISHER,
    MFI,
    MINUS_DI,
    MINUS_DM,
    MOM,
    PLUS_DI,
    PLUS_DM,
    PPO,
    ROC,
    ROCP,
    ROCR,
    ROCR100,
    RSI,
    STOCH,
    STOCHF,
    STOCHRSI,
    TRANGE,
    TRIX,
    ULTOSC,
    WILLR,
)

# ---------------------------------------------------------------------------
# Volume Indicators
# ---------------------------------------------------------------------------
from ferro_ta.indicators.oscillators import (  # noqa: F401
    AC,
    AO,
    CHO,
    COPPOCK,
    DPO,
    GATOR,
    KST,
    PO,
    RVI,
    STC,
    TSI,
    VORTEX,
)

# ---------------------------------------------------------------------------
# Overlap Studies
# ---------------------------------------------------------------------------
from ferro_ta.indicators.overlap import (  # noqa: F401
    ALLIGATOR,
    ALMA,
    BBANDS,
    CHANDE_KROLL_STOP,
    DEMA,
    EMA,
    FRAMA,
    KAMA,
    MA,
    MA_ENVELOPES,
    MACD,
    MACDEXT,
    MACDFIX,
    MAMA,
    MAVP,
    MCGINLEY,
    MIDPOINT,
    MIDPRICE,
    SAR,
    SAREXT,
    SMA,
    T3,
    TEMA,
    TRIMA,
    VIDYA,
    WMA,
    ZLEMA,
)

# ---------------------------------------------------------------------------
# Pattern Recognition
# ---------------------------------------------------------------------------
from ferro_ta.indicators.pattern import (  # noqa: F401
    CDL2CROWS,
    CDL3BLACKCROWS,
    CDL3INSIDE,
    CDL3LINESTRIKE,
    CDL3OUTSIDE,
    CDL3STARSINSOUTH,
    CDL3WHITESOLDIERS,
    CDLABANDONEDBABY,
    CDLADVANCEBLOCK,
    CDLBELTHOLD,
    CDLBREAKAWAY,
    CDLCLOSINGMARUBOZU,
    CDLCONCEALBABYSWALL,
    CDLCOUNTERATTACK,
    CDLDARKCLOUDCOVER,
    CDLDOJI,
    CDLDOJISTAR,
    CDLDRAGONFLYDOJI,
    CDLENGULFING,
    CDLEVENINGDOJISTAR,
    CDLEVENINGSTAR,
    CDLGAPSIDESIDEWHITE,
    CDLGRAVESTONEDOJI,
    CDLHAMMER,
    CDLHANGINGMAN,
    CDLHARAMI,
    CDLHARAMICROSS,
    CDLHIGHWAVE,
    CDLHIKKAKE,
    CDLHIKKAKEMOD,
    CDLHOMINGPIGEON,
    CDLIDENTICAL3CROWS,
    CDLINNECK,
    CDLINVERTEDHAMMER,
    CDLKICKING,
    CDLKICKINGBYLENGTH,
    CDLLADDERBOTTOM,
    CDLLONGLEGGEDDOJI,
    CDLLONGLINE,
    CDLMARUBOZU,
    CDLMATCHINGLOW,
    CDLMATHOLD,
    CDLMORNINGDOJISTAR,
    CDLMORNINGSTAR,
    CDLONNECK,
    CDLPIERCING,
    CDLRICKSHAWMAN,
    CDLRISEFALL3METHODS,
    CDLSEPARATINGLINES,
    CDLSHOOTINGSTAR,
    CDLSHORTLINE,
    CDLSPINNINGTOP,
    CDLSTALLEDPATTERN,
    CDLSTICKSANDWICH,
    CDLTAKURI,
    CDLTASUKIGAP,
    CDLTHRUSTING,
    CDLTRISTAR,
    CDLUNIQUE3RIVER,
    CDLUPSIDEGAP2CROWS,
    CDLXSIDEGAP3METHODS,
)

# ---------------------------------------------------------------------------
# Price Transformations
# ---------------------------------------------------------------------------
from ferro_ta.indicators.price_transform import (  # noqa: F401
    AVGPRICE,
    MEDPRICE,
    TYPPRICE,
    WCLPRICE,
)

# ---------------------------------------------------------------------------
# Statistic Functions
# ---------------------------------------------------------------------------
from ferro_ta.indicators.statistic import (  # noqa: F401
    BETA,
    CORREL,
    LINEARREG,
    LINEARREG_ANGLE,
    LINEARREG_INTERCEPT,
    LINEARREG_SLOPE,
    MEDIAN,
    MEDIAN_BANDS,
    MODE,
    STDDEV,
    TSF,
    VAR,
)

# ---------------------------------------------------------------------------
# Signal utilities
# ---------------------------------------------------------------------------
from ferro_ta.indicators.utils import (  # noqa: F401
    CHANGE,
    CROSS,
    CROSSOVER,
    CROSSUNDER,
    EXREM,
    FALLING,
    FLIP,
    HIGHEST,
    LOWEST,
    RISING,
    VALUEWHEN,
)

# ---------------------------------------------------------------------------
# Volatility Indicators
# ---------------------------------------------------------------------------
from ferro_ta.indicators.volatility import (  # noqa: F401
    ATR,
    BBPERCENT,
    BBWIDTH,
    CHAIKIN_VOL,
    HISTORICAL_VOLATILITY,
    MASS,
    NATR,
    STARC,
    ULCER_INDEX,
)
from ferro_ta.indicators.volume import (  # noqa: F401
    AD,
    ADOSC,
    CMF,
    EMV,
    FORCE_INDEX,
    KVO,
    NVI,
    NVI_WITH_EMA,
    OBV,
    OBV_SMOOTHED,
    PVI,
    PVI_WITH_SIGNAL,
    PVT,
    RVOL,
    VOLOSC,
    VROC,
)

__all__ = [
    "__version__",
    # Overlap Studies
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
    # Momentum
    "RSI",
    "MOM",
    "ROC",
    "ROCP",
    "ROCR",
    "ROCR100",
    "WILLR",
    "AROON",
    "AROONOSC",
    "CCI",
    "MFI",
    "BOP",
    "STOCHF",
    "STOCH",
    "STOCHRSI",
    "APO",
    "PPO",
    "CMO",
    "PLUS_DM",
    "MINUS_DM",
    "PLUS_DI",
    "MINUS_DI",
    "DX",
    "ADX",
    "DMI",
    "ADXR",
    "TRIX",
    "ULTOSC",
    "TRANGE",
    "ELDER_RAY",
    "FISHER",
    "CRSI",
    # Volume
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
    # Oscillators
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
    # Volatility
    "ATR",
    "NATR",
    "CHAIKIN_VOL",
    "MASS",
    "BBPERCENT",
    "BBWIDTH",
    "HISTORICAL_VOLATILITY",
    "ULCER_INDEX",
    "STARC",
    # Statistics
    "STDDEV",
    "VAR",
    "LINEARREG",
    "LINEARREG_SLOPE",
    "LINEARREG_INTERCEPT",
    "LINEARREG_ANGLE",
    "TSF",
    "BETA",
    "CORREL",
    "MEDIAN",
    "MEDIAN_BANDS",
    "MODE",
    # Price transforms
    "AVGPRICE",
    "MEDPRICE",
    "TYPPRICE",
    "WCLPRICE",
    # Patterns
    "CDL2CROWS",
    "CDL3BLACKCROWS",
    "CDL3INSIDE",
    "CDL3LINESTRIKE",
    "CDL3OUTSIDE",
    "CDL3STARSINSOUTH",
    "CDL3WHITESOLDIERS",
    "CDLABANDONEDBABY",
    "CDLADVANCEBLOCK",
    "CDLBELTHOLD",
    "CDLBREAKAWAY",
    "CDLCLOSINGMARUBOZU",
    "CDLCONCEALBABYSWALL",
    "CDLCOUNTERATTACK",
    "CDLDARKCLOUDCOVER",
    "CDLDOJI",
    "CDLDOJISTAR",
    "CDLDRAGONFLYDOJI",
    "CDLENGULFING",
    "CDLEVENINGDOJISTAR",
    "CDLEVENINGSTAR",
    "CDLGAPSIDESIDEWHITE",
    "CDLGRAVESTONEDOJI",
    "CDLHAMMER",
    "CDLHANGINGMAN",
    "CDLHARAMI",
    "CDLHARAMICROSS",
    "CDLHIGHWAVE",
    "CDLHIKKAKE",
    "CDLHIKKAKEMOD",
    "CDLHOMINGPIGEON",
    "CDLIDENTICAL3CROWS",
    "CDLINNECK",
    "CDLINVERTEDHAMMER",
    "CDLKICKING",
    "CDLKICKINGBYLENGTH",
    "CDLLADDERBOTTOM",
    "CDLLONGLEGGEDDOJI",
    "CDLLONGLINE",
    "CDLMARUBOZU",
    "CDLMATCHINGLOW",
    "CDLMATHOLD",
    "CDLMORNINGDOJISTAR",
    "CDLMORNINGSTAR",
    "CDLONNECK",
    "CDLPIERCING",
    "CDLRICKSHAWMAN",
    "CDLRISEFALL3METHODS",
    "CDLSEPARATINGLINES",
    "CDLSHOOTINGSTAR",
    "CDLSHORTLINE",
    "CDLSPINNINGTOP",
    "CDLSTALLEDPATTERN",
    "CDLSTICKSANDWICH",
    "CDLTAKURI",
    "CDLTASUKIGAP",
    "CDLTHRUSTING",
    "CDLTRISTAR",
    "CDLUNIQUE3RIVER",
    "CDLUPSIDEGAP2CROWS",
    "CDLXSIDEGAP3METHODS",
    # Cycle
    "HT_TRENDLINE",
    "HT_DCPERIOD",
    "HT_DCPHASE",
    "HT_PHASOR",
    "HT_SINE",
    "HT_TRENDMODE",
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
    # Extended Indicators
    "VWAP",
    "SUPERTREND",
    "ICHIMOKU",
    "DONCHIAN",
    "PIVOT_POINTS",
    "KELTNER_CHANNELS",
    "HULL_MA",
    "CHANDELIER_EXIT",
    "VWMA",
    "CHOPPINESS_INDEX",
    "WILLIAMS_FRACTALS",
    "RWI",
    # Signal utilities
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
    # API discovery
    "about",
    "indicators",
    "methods",
    "info",
    # Logging utilities
    "enable_debug",
    "disable_debug",
    "debug_mode",
    "get_logger",
    "log_call",
    "benchmark",
    "traced",
]

# ---------------------------------------------------------------------------
# Extended Indicators
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Pandas API — apply transparent pandas.Series / DataFrame support to every
# public indicator function exported from this module.
# ---------------------------------------------------------------------------
from ferro_ta._utils import pandas_wrap as _pandas_wrap  # noqa: E402
from ferro_ta._utils import polars_wrap as _polars_wrap  # noqa: E402
from ferro_ta.analysis.attribution import (  # noqa: F401, E402
    TradeStats,
    attribution_by_month,
    attribution_by_signal,
    from_backtest,
    trade_stats,
)
from ferro_ta.analysis.crypto import (  # noqa: F401, E402
    continuous_bar_labels,
    funding_pnl,
    resample_continuous,
    session_boundaries,
)
from ferro_ta.analysis.regime import (  # noqa: F401, E402
    detect_breaks_cusum,
    regime,
    regime_adx,
    regime_combined,
    rolling_variance_break,
    structural_breaks,
)
from ferro_ta.core import exceptions as exceptions  # noqa: F401, E402

# ---------------------------------------------------------------------------
# Logging utilities — ferro_ta.enable_debug() / ferro_ta.benchmark()
# ---------------------------------------------------------------------------
from ferro_ta.core.logging_utils import (  # noqa: F401, E402
    benchmark,
    debug_mode,
    disable_debug,
    enable_debug,
    get_logger,
    log_call,
    traced,
)
from ferro_ta.data import batch as batch  # noqa: F401, E402
from ferro_ta.data import streaming as streaming  # noqa: F401, E402

# ---------------------------------------------------------------------------
# Batch API (not in __all__ — use directly from ferro_ta.batch)
# Import: from ferro_ta.batch import batch_sma, batch_ema, batch_rsi
# ---------------------------------------------------------------------------
from ferro_ta.data.batch import (  # noqa: F401, E402
    batch_apply,
    batch_ema,
    batch_rsi,
    batch_sma,
    compute_many,
)
from ferro_ta.data.chunked import (  # noqa: F401, E402
    chunk_apply,
    make_chunk_ranges,
    stitch_chunks,
    trim_overlap,
)

# ---------------------------------------------------------------------------
# Streaming / Incremental API  (not in __all__ — these are classes, not funcs)
# Import directly: from ferro_ta.streaming import StreamingSMA, ...
# ---------------------------------------------------------------------------
from ferro_ta.data.streaming import (  # noqa: F401, E402  # type: ignore[assignment]
    StreamingATR,  # type: ignore[attr-defined]
    StreamingBBands,  # type: ignore[attr-defined]
    StreamingEMA,  # type: ignore[attr-defined]
    StreamingMACD,  # type: ignore[attr-defined]
    StreamingRSI,  # type: ignore[attr-defined]
    StreamingSMA,  # type: ignore[attr-defined]
    StreamingStoch,  # type: ignore[attr-defined]
    StreamingSupertrend,  # type: ignore[attr-defined]
    StreamingVWAP,  # type: ignore[attr-defined]
)
from ferro_ta.indicators import cycle as cycle  # noqa: F401, E402
from ferro_ta.indicators import extended as extended  # noqa: F401, E402
from ferro_ta.indicators import math_ops as math_ops  # noqa: F401, E402
from ferro_ta.indicators import momentum as momentum  # noqa: F401, E402
from ferro_ta.indicators import oscillators as oscillators  # noqa: F401, E402
from ferro_ta.indicators import overlap as overlap  # noqa: F401, E402
from ferro_ta.indicators import pattern as pattern  # noqa: F401, E402
from ferro_ta.indicators import price_transform as price_transform  # noqa: F401, E402
from ferro_ta.indicators import statistic as statistic  # noqa: F401, E402
from ferro_ta.indicators import volatility as volatility  # noqa: F401, E402
from ferro_ta.indicators import volume as volume  # noqa: F401, E402
from ferro_ta.indicators.extended import (  # noqa: F401, E402
    CHANDELIER_EXIT,
    CHOPPINESS_INDEX,
    DONCHIAN,
    HULL_MA,
    ICHIMOKU,
    KELTNER_CHANNELS,
    PIVOT_POINTS,
    RWI,
    SUPERTREND,
    VWAP,
    VWMA,
    WILLIAMS_FRACTALS,
)

# ---------------------------------------------------------------------------
# Additional modules (not in __all__ — access via submodule)
# ---------------------------------------------------------------------------
from ferro_ta.tools.alerts import (  # noqa: F401, E402
    AlertManager,
    check_cross,
    check_threshold,
    collect_alert_bars,
)

# ---------------------------------------------------------------------------
# API discovery helpers — ferro_ta.about(), ferro_ta.methods(),
# ferro_ta.indicators(), and ferro_ta.info()
# ---------------------------------------------------------------------------
from ferro_ta.tools.api_info import about, indicators, info, methods  # noqa: F401, E402

_ALIASED_SUBMODULES = {
    "batch": batch,
    "cycle": cycle,
    "exceptions": exceptions,
    "extended": extended,
    "math_ops": math_ops,
    "momentum": momentum,
    "overlap": overlap,
    "pattern": pattern,
    "price_transform": price_transform,
    "statistic": statistic,
    "streaming": streaming,
    "oscillators": oscillators,
    "volatility": volatility,
    "volume": volume,
}

for _module_name, _module in _ALIASED_SUBMODULES.items():
    setattr(_sys.modules[__name__], _module_name, _module)
    _sys.modules[f"{__name__}.{_module_name}"] = _module

_g = globals()
for _name in __all__:
    _fn = _g.get(_name)
    if callable(_fn) and not getattr(_fn, "_pandas_wrapped", False):
        _g[_name] = _pandas_wrap(_fn)
    _fn = _g.get(_name)
    if callable(_fn) and not getattr(_fn, "_polars_wrapped", False):
        _g[_name] = _polars_wrap(_fn)
del _ALIASED_SUBMODULES, _g, _module, _module_name, _name, _fn, _sys
