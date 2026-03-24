ferro-ta Documentation
======================

.. toctree::
   :maxdepth: 2
   :caption: Core Library

   quickstart
   migration_talib
   support_matrix
   pandas_api
   error_handling
   api/index
   streaming
   batch
   extended

.. toctree::
   :maxdepth: 2
   :caption: Evidence and Releases

   benchmarks
   changelog

.. toctree::
   :maxdepth: 2
   :caption: Adjacent and Experimental

   derivatives
   adjacent_tooling
   plugins
   contributing

Overview
--------

**ferro-ta** is a Rust-powered Python technical analysis library focused on a
TA-Lib-compatible API for NumPy-centered workloads.

.. important::

   Performance varies by indicator, array layout, warmup, build flags, and
   machine. ferro-ta is often faster on selected indicators, not universally
   faster. See :doc:`benchmarks` for the reproducible workflow, methodology
   notes, and the indicators where TA-Lib still wins or ties in the current
   checked-in artifact.

Core library:

- 160+ indicators covering all TA-Lib categories
- TA-Lib-style imports such as ``ferro_ta.SMA(close, timeperiod=20)``
- Pre-built wheels for the supported Python/OS matrix
- Pure Rust core library (``crates/ferro_ta_core``) — no PyO3 / numpy dependency
- Batch execution API — run indicators on 2-D arrays of multiple series
- Streaming / bar-by-bar API for live trading
- Transparent pandas.Series support
- Type stubs (.pyi) for IDE auto-completion
- 10 extended indicators not in TA-Lib (VWAP, Supertrend, Ichimoku Cloud, ...)

Adjacent and experimental tooling:

- Derivatives analytics — see :doc:`derivatives`
- Agentic workflow and LangChain tool wrappers — see `Agentic guide <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/agentic.md>`_
- MCP server for Cursor/Claude integration — see `MCP guide <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/mcp.md>`_
- WASM, plugins, and other optional surfaces — see :doc:`adjacent_tooling`

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
- :doc:`support_matrix` — parity status, tested wheel targets, supported Python versions, and experimental modules.
- `Rust-First Policy <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/rust_first.md>`_ — all compute logic belongs in Rust; how to add new indicators.
- `Out-of-Core Execution <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/out-of-core.md>`_ — chunked processing and Dask integration.
- :doc:`derivatives` — IV helpers, options pricing/Greeks/IV, futures analytics, strategy schemas, and payoff helpers.
- :doc:`adjacent_tooling` — optional surfaces such as derivatives, MCP, WASM, GPU, plugins, and agent-oriented integrations.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
