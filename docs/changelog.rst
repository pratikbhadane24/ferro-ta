Changelog
=========

0.1.0 (2024)
------------

**Candlestick Pattern Parity (61/61)**

- All 61 TA-Lib candlestick patterns implemented in Rust
- ``{-100, 0, 100}`` convention, consistent with TA-Lib

**Numerical Parity**

- RSI, ATR/NATR, CCI, BETA, STOCH, STOCHRSI, ADX/DX/DI/DM all rewritten to match TA-Lib seeding
- Removed dependency on ``ta`` crate for these indicators

**Streaming / Incremental API**

- New :mod:`ferro_ta.streaming` module with bar-by-bar stateful classes
- ``StreamingSMA``, ``StreamingEMA``, ``StreamingRSI``, ``StreamingATR``, ``StreamingBBands``, ``StreamingMACD``, ``StreamingStoch``, ``StreamingVWAP``, ``StreamingSupertrend``

**Pandas Integration**

- All indicators transparently accept ``pandas.Series`` and return ``Series`` with original index preserved
- Multi-output functions return tuples of ``Series``

**Math Operators / Transforms**

- 24 functions: arithmetic (ADD/SUB/MULT/DIV), rolling (SUM/MAX/MIN/MAXINDEX/MININDEX), element-wise math transforms
- SUM uses vectorized cumsum (220× faster than a naive loop)

**Documentation**

- Sphinx documentation setup with API reference, quickstart guide, and benchmarks page

**Benchmarking Suite**

- ``benchmarks/test_speed.py`` for authoritative ``pytest-benchmark`` speed runs
- ``benchmarks/bench_vs_talib.py`` for TA-Lib head-to-head comparisons

**Extended Indicators**

- ``VWAP`` — cumulative or rolling window
- ``SUPERTREND`` — ATR-based trend signal

**Additional Extended Indicators**

- ``ICHIMOKU`` — Ichimoku Cloud (Tenkan, Kijun, Senkou A/B, Chikou)
- ``DONCHIAN`` — Donchian Channels (upper, middle, lower)
- ``PIVOT_POINTS`` — Classic, Fibonacci, and Camarilla pivot points

**Type Stubs & Packaging**

- ``python/ferro_ta/__init__.pyi`` type stub for IDE auto-completion
- ``pyproject.toml``: added optional extras (benchmark, pandas, docs, all), project URLs, Python 3.10–3.13 classifiers
