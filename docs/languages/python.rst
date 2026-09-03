Python
======

Python bindings for ferro-ta. The package wraps ``ferro_ta_core`` via PyO3 and
adds NumPy conversion, optional pandas/polars wrappers, and Sphinx autodoc.

Installation
------------

.. code-block:: bash

   pip install ferro-ta

   # For Pandas support:
   pip install ferro-ta pandas

   # For benchmarks:
   pip install ferro-ta pytest-benchmark

Basic Usage
-----------

All functions accept NumPy arrays and return NumPy arrays. SMA, RSI, MACD, and
BBANDS are the shared examples used on every language page.

.. code-block:: python

   import numpy as np
   from ferro_ta import SMA, EMA, RSI, MACD, BBANDS, ATR

   close = np.linspace(44.0, 48.0, 40)
   high  = close + 0.5
   low   = close - 0.5

   sma  = SMA(close, timeperiod=5)
   ema  = EMA(close, timeperiod=5)
   rsi  = RSI(close, timeperiod=14)
   atr  = ATR(high, low, close, timeperiod=5)

   upper, middle, lower = BBANDS(close, timeperiod=5)
   macd_line, signal, histogram = MACD(close)

Pandas Integration
------------------

All functions transparently accept ``pandas.Series`` and preserve the index:

.. code-block:: python

   import pandas as pd
   from ferro_ta import SMA, BBANDS

   idx   = pd.date_range("2024-01-01", periods=40, freq="D")
   close = pd.Series(list(range(40)), index=idx, dtype="float64") + 44.0

   sma = SMA(close, timeperiod=3)           # → pd.Series, same index
   upper, mid, lower = BBANDS(close, timeperiod=3)  # → tuple of pd.Series

Streaming / Live Trading
------------------------

Use the :mod:`ferro_ta.streaming` module for bar-by-bar processing:

.. code-block:: python

   from ferro_ta.streaming import StreamingSMA, StreamingRSI, StreamingATR

   sma = StreamingSMA(period=5)
   rsi = StreamingRSI(period=14)
   atr = StreamingATR(period=14)

   for bar in live_feed:
       current_sma = sma.update(bar.close)
       current_rsi = rsi.update(bar.close)
       current_atr = atr.update(bar.high, bar.low, bar.close)

Extended Indicators
-------------------

.. code-block:: python

   from ferro_ta import VWAP, SUPERTREND, ICHIMOKU, DONCHIAN, PIVOT_POINTS
   import numpy as np

   high  = np.array([...])
   low   = np.array([...])
   close = np.array([...])
   vol   = np.array([...])

   vwap = VWAP(high, low, close, vol)
   rolling_vwap = VWAP(high, low, close, vol, timeperiod=14)

   st_line, direction = SUPERTREND(high, low, close, timeperiod=7, multiplier=3.0)
   tenkan, kijun, senkou_a, senkou_b, chikou = ICHIMOKU(high, low, close)
   dc_upper, dc_mid, dc_lower = DONCHIAN(high, low, timeperiod=20)
   pivot, r1, s1, r2, s2 = PIVOT_POINTS(high, low, close, method="classic")

Derivatives Analytics
---------------------

.. code-block:: python

   from ferro_ta.analysis.options import greeks, option_price
   from ferro_ta.analysis.futures import basis

   call_price = option_price(100.0, 100.0, 0.05, 1.0, 0.20, option_type="call")
   call_greeks = greeks(100.0, 100.0, 0.05, 1.0, 0.20, option_type="call")
   front_basis = basis(100.0, 103.0)

See :doc:`/derivatives` for the full analytics surface, including implied
volatility inversion, smile metrics, strike selection, futures curve tools,
strategy schemas, and multi-leg payoff helpers.

API map
-------

Python keeps TA-Lib-style ``UPPER_CASE`` names. The generated reference lives
under :doc:`/api/index`:

.. list-table::
   :header-rows: 1
   :widths: 24 40 36

   * - Category
     - Module
     - Notes
   * - Overlap
     - ``ferro_ta.overlap`` / ``ferro_ta.SMA``
     - SMA, EMA, MACD, BBANDS, …
   * - Momentum
     - ``ferro_ta.momentum``
     - RSI, STOCH, ADX, ``PLUS_DI``, …
   * - Pattern
     - ``ferro_ta.pattern``
     - 61 ``CDL*`` candlestick functions
   * - Streaming
     - ``ferro_ta.streaming``
     - ``StreamingSMA``, ``StreamingRSI``, …
   * - Batch
     - ``ferro_ta.batch``
     - 2-D multi-series helpers

Cross-language coverage (including names that exist only on Python) is
generated in :doc:`coverage`.
