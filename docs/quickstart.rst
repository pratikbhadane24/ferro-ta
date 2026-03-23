Quick Start
===========

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

All functions accept NumPy arrays and return NumPy arrays:

.. code-block:: python

   import numpy as np
   from ferro_ta import SMA, EMA, RSI, MACD, BBANDS, ATR

   close = np.array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.15, 43.61, 44.33])
   high  = close + 0.5
   low   = close - 0.5

   # Single output
   sma  = SMA(close, timeperiod=5)
   ema  = EMA(close, timeperiod=5)
   rsi  = RSI(close, timeperiod=5)
   atr  = ATR(high, low, close, timeperiod=5)

   # Multi output
   upper, middle, lower = BBANDS(close, timeperiod=5)
   macd_line, signal, histogram = MACD(close)

Pandas Integration
------------------

All functions transparently accept ``pandas.Series`` and preserve the index:

.. code-block:: python

   import pandas as pd
   from ferro_ta import SMA, BBANDS

   idx   = pd.date_range("2024-01-01", periods=10, freq="D")
   close = pd.Series([44.34, 44.09, 44.15, 43.61, 44.33,
                      44.83, 45.10, 45.15, 43.61, 44.33], index=idx)

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

   # VWAP
   vwap = VWAP(high, low, close, vol)
   rolling_vwap = VWAP(high, low, close, vol, timeperiod=14)

   # Supertrend
   st_line, direction = SUPERTREND(high, low, close, timeperiod=7, multiplier=3.0)

   # Ichimoku Cloud
   tenkan, kijun, senkou_a, senkou_b, chikou = ICHIMOKU(high, low, close)

   # Donchian Channels
   dc_upper, dc_mid, dc_lower = DONCHIAN(high, low, timeperiod=20)

   # Pivot Points
   pivot, r1, s1, r2, s2 = PIVOT_POINTS(high, low, close, method="classic")
