Migration from TA-Lib
=====================

ferro-ta is designed as a drop-in replacement for `ta-lib` (the Python
`talib` package) for the most-commonly used indicators. This guide explains
the differences so you can migrate existing code with confidence.

.. contents::
   :local:
   :depth: 2


Import changes
--------------

TA-Lib uses a single flat namespace::

    import talib
    result = talib.SMA(close, timeperiod=14)

ferro-ta exposes the same names at the top level **and** in sub-modules::

    # Option A — top-level (most concise, mirrors talib)
    from ferro_ta import SMA, EMA, RSI
    result = SMA(close, timeperiod=14)

    # Option B — sub-modules
    from ferro_ta.overlap import SMA
    from ferro_ta.momentum import RSI

Multi-output functions return a **tuple** in both libraries::

    # talib
    upper, middle, lower = talib.BBANDS(close)

    # ferro_ta
    upper, middle, lower = ferro_ta.BBANDS(close)


Input / output conventions
--------------------------

Both libraries accept NumPy ``float64`` arrays. ferro-ta also accepts any
array-like (Python list, ``float32``, pandas Series) and converts
automatically.

- **Leading NaN values** — both libraries emit ``NaN`` for the "warm-up"
  period at the start of an array. The number of ``NaN`` values is identical
  for all indicators marked **Exact** or **Close** in the accuracy table.
- **Output length** — always equal to input length, matching TA-Lib.
- **Pandas Series** — ferro-ta transparently preserves the original index when
  a ``pd.Series`` is passed as input.


Accuracy levels
---------------

.. list-table::
   :header-rows: 1

   * - Symbol
     - Meaning
   * - ✅ **Exact**
     - Values match TA-Lib to floating-point precision.
   * - ✅ **Close**
     - Values converge to TA-Lib after the warm-up window (EMA-seed
       differences resolve within ~50 bars for typical periods).
   * - ⚠️ **Corr**
     - Strong correlation (> 0.95) but not numerically identical (e.g.
       MAMA uses the same algorithm but slightly different initialization).
   * - ⚠️ **Shape**
     - Same output shape and NaN structure; absolute values differ (e.g. SAR
       reversal history can diverge due to floating-point accumulation).

All overlap, momentum, volume, volatility, statistic, and price-transform
functions are **Exact** or **Close**. The only remaining **Corr / Shape**
functions are MAMA, SAR, SAREXT, and the six HT_* cycle indicators — see
the roadmap for details.


Known behavioural differences
------------------------------

EMA / DEMA / TEMA / T3 / MACD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TA-Lib seeds the first EMA value with a simple moving average. ferro-ta uses
the same seeding, so values converge after the warm-up period. For a 14-period
EMA on typical market data, convergence is complete by bar ~60.

RSI
~~~

ferro-ta uses the same Wilder smoothing seed as TA-Lib (SMA seed for the first
``timeperiod`` bars) and produces **Exact** results.

SAR / SAREXT
~~~~~~~~~~~~

Parabolic SAR reversal history can diverge in rare edge-cases due to
floating-point accumulation differences. Output shapes (NaN count, length)
match exactly.

HT_* cycle indicators
~~~~~~~~~~~~~~~~~~~~~

The Hilbert Transform cycle indicators (``HT_DCPERIOD``, ``HT_DCPHASE``,
``HT_PHASOR``, ``HT_SINE``, ``HT_TRENDLINE``, ``HT_TRENDMODE``) use the
same Ehlers algorithm as TA-Lib but may differ slightly in floating-point
accumulation. All six share a 63-bar lookback matching TA-Lib.

OBV
~~~

ferro-ta OBV starts accumulation from zero at bar 0 (same as TA-Lib for most
data sets). If your TA-Lib OBV shows an offset this is usually due to a
starting volume difference in the input data.


Before / after example
-----------------------

.. code-block:: python

    # --- Before (ta-lib) ---
    import numpy as np
    import talib

    close = np.random.rand(200).cumsum() + 100.0
    high  = close + 0.5
    low   = close - 0.5

    sma   = talib.SMA(close, timeperiod=14)
    ema   = talib.EMA(close, timeperiod=14)
    rsi   = talib.RSI(close, timeperiod=14)
    upper, mid, lower = talib.BBANDS(close, timeperiod=20)
    macd, signal, hist = talib.MACD(close)
    atr   = talib.ATR(high, low, close, timeperiod=14)

    # --- After (ferro_ta) ---
    import numpy as np
    from ferro_ta import SMA, EMA, RSI, BBANDS, MACD, ATR

    close = np.random.rand(200).cumsum() + 100.0
    high  = close + 0.5
    low   = close - 0.5

    sma   = SMA(close, timeperiod=14)
    ema   = EMA(close, timeperiod=14)
    rsi   = RSI(close, timeperiod=14)
    upper, mid, lower = BBANDS(close, timeperiod=20)
    macd, signal, hist = MACD(close)
    atr   = ATR(high, low, close, timeperiod=14)

Only the import line changes for the most common indicators.


Extended (non-TA-Lib) indicators
---------------------------------

ferro-ta additionally provides indicators not in TA-Lib::

    from ferro_ta import (
        VWAP, SUPERTREND, ICHIMOKU, DONCHIAN, PIVOT_POINTS,
        KELTNER_CHANNELS, HULL_MA, CHANDELIER_EXIT, VWMA, CHOPPINESS_INDEX,
    )

See :doc:`extended` for full API documentation.
