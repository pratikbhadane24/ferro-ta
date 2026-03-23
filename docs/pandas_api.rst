Pandas API contract
===================

**Contract**

- All indicators accept ``pandas.Series`` (or 1-D DataFrame columns) and return
  ``pandas.Series`` — or a **tuple of Series** for multi-output functions (e.g. MACD, BBANDS)
  — with the **original index preserved**.
- Default OHLCV column names for DataFrames are ``open``, ``high``, ``low``, ``close``, ``volume``.
- To use different column names, use :func:`ferro_ta.utils.get_ohlcv` to extract arrays/Series
  with configurable column names, then call the indicator.

**Single Series**

.. code-block:: python

   import pandas as pd
   from ferro_ta import SMA, RSI
   close = pd.Series([44.34, 44.09, 44.15], index=pd.date_range("2024-01-01", periods=3))
   sma = SMA(close, timeperiod=2)   # returns pd.Series with same index

**DataFrame with OHLCV (configurable column names)**

.. code-block:: python

   import pandas as pd
   from ferro_ta import ATR, RSI
   from ferro_ta.utils import get_ohlcv

   df = pd.DataFrame({
       "Open": [1, 2, 3], "High": [1.1, 2.1, 3.1],
       "Low": [0.9, 1.9, 2.9], "Close": [1.05, 2.05, 3.05],
   }, index=pd.date_range("2024-01-01", periods=3, freq="D"))

   o, h, l, c, v = get_ohlcv(df, open_col="Open", high_col="High",
                             low_col="Low", close_col="Close", volume_col=None)
   atr = ATR(h, l, c, timeperiod=2)   # index preserved
   rsi = RSI(c, timeperiod=2)         # index preserved

**Multi-output**

Functions like ``MACD`` and ``BBANDS`` return a tuple of ``pandas.Series``, all with the same index as the input.

**See also**

- :mod:`ferro_ta.utils` — :func:`get_ohlcv` for DataFrame OHLCV extraction.
