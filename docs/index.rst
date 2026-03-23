ferro-ta Documentation
======================

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   migration_talib
   pandas_api
   error_handling
   api/index
   streaming
   extended
   batch
   derivatives
   benchmarks
   plugins
   changelog
   contributing

Overview
--------

**ferro-ta** is a fast Technical Analysis library — a drop-in alternative to TA-Lib
powered by Rust and PyO3.

Features:

- 160+ indicators covering all TA-Lib categories
- 10 extended indicators not in TA-Lib (VWAP, Supertrend, Ichimoku Cloud, …)
- Batch execution API — run indicators on 2-D arrays of multiple series
- Pure Rust core library (``crates/ferro_ta_core``) — no PyO3 / numpy dependency
- Streaming / bar-by-bar API for live trading
- Transparent pandas.Series support
- Math operators and transforms
- Type stubs (.pyi) for IDE auto-completion
- WASM binding for browser/Node.js use
- Options/IV helpers and derivatives analytics — see :doc:`derivatives`
- Agentic workflow and LangChain tool wrappers — see `Agentic guide <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/agentic.md>`_
- MCP server for Cursor/Claude integration — see `MCP guide <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/mcp.md>`_
- Sphinx documentation

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install ferro-ta

Quick Start
~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from ferro_ta import SMA, EMA, RSI, MACD, BBANDS

   close = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 13.5, 12.5])
   print(SMA(close, timeperiod=3))

   # Batch: run SMA on 5 symbols at once
   from ferro_ta.batch import batch_sma
   data = np.random.rand(100, 5)
   result = batch_sma(data, timeperiod=10)

Further Reading
~~~~~~~~~~~~~~~

- `Architecture <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/architecture.md>`_ — Rust/Python layout, two-crate design, binding flow.
- `Performance Guide <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/performance.md>`_ — when to use raw numpy vs pandas/polars, batch notes, tips.
- `API Stability <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/stability.md>`_ — stability tiers, versioning, and deprecation policy.
- `Rust-First Policy <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/rust_first.md>`_ — all compute logic belongs in Rust; how to add new indicators.
- `Out-of-Core Execution <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/out-of-core.md>`_ — chunked processing and Dask integration.
- :doc:`derivatives` — IV helpers, options pricing/Greeks/IV, futures analytics, strategy schemas, and payoff helpers.
- `Agentic Workflow <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/agentic.md>`_ — tools.py, workflow.py, LangChain integration.
- `MCP Server <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/mcp.md>`_ — run ferro-ta as an MCP server in Cursor/Claude.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
