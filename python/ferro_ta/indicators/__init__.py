"""
ferro_ta.indicators — Technical indicator functions.

Sub-modules
-----------
* :mod:`ferro_ta.indicators.momentum`        — Momentum Indicators (RSI, STOCH, ADX, CCI, …)
* :mod:`ferro_ta.indicators.overlap`         — Overlap Studies (SMA, EMA, BBANDS, MACD, …)
* :mod:`ferro_ta.indicators.volatility`      — Volatility Indicators (ATR, NATR, TRANGE)
* :mod:`ferro_ta.indicators.volume`          — Volume Indicators (AD, ADOSC, OBV)
* :mod:`ferro_ta.indicators.statistic`       — Statistic Functions (STDDEV, VAR, LINEARREG, …)
* :mod:`ferro_ta.indicators.price_transform` — Price Transforms (AVGPRICE, MEDPRICE, …)
* :mod:`ferro_ta.indicators.pattern`         — Candlestick Pattern Recognition (CDL*)
* :mod:`ferro_ta.indicators.cycle`           — Cycle Indicators (HT_TRENDLINE, HT_DCPERIOD, …)
* :mod:`ferro_ta.indicators.math_ops`        — Math Operators/Transforms (ADD, SUB, SUM, …)
* :mod:`ferro_ta.indicators.extended`        — Extended Indicators (VWAP, SUPERTREND, ICHIMOKU, …)

All indicators are also importable directly from :mod:`ferro_ta`::

    import ferro_ta
    result = ferro_ta.RSI(close, timeperiod=14)

    # or directly from the sub-module:
    from ferro_ta.indicators.momentum import RSI
    result = RSI(close, timeperiod=14)
"""
