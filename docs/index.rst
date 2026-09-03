ferro-ta Documentation
======================

.. toctree::
   :maxdepth: 2
   :caption: Languages

   languages/index
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Python API

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
   :caption: Adjacent tooling

   derivatives
   adjacent_tooling
   plugins
   contributing

Overview
--------

**ferro-ta** is a Rust-core technical analysis library with first-class
bindings for Python, Rust, JavaScript (WASM), and Flutter.

Every language wraps ``ferro_ta_core``. Python is the most complete ergonomic
surface (TA-Lib names, pandas/polars, Sphinx autodoc). It is not the only
product. New languages may only wrap the Rust core — see
:doc:`languages/adding`.

.. important::

   Performance varies by indicator, array layout, warmup, build flags, and
   machine. ferro-ta is often faster on selected indicators, not universally
   faster. See :doc:`benchmarks` for the reproducible workflow, methodology
   notes, and the indicators where TA-Lib still wins or ties in the current
   checked-in artifact.

Shared core:

- 160+ indicators covering all TA-Lib categories
- Pure Rust compute in ``crates/ferro_ta_core`` — no Python, NumPy, or JS dependency
- First-class packages: PyPI ``ferro-ta``, crates.io ``ferro_ta_core``, npm
  ``ferro-ta-wasm``, pub.dev ``ferro_ta``
- Batch execution and streaming / bar-by-bar APIs on the surfaces that expose them
- 10 extended indicators not in TA-Lib (VWAP, Supertrend, Ichimoku Cloud, ...)

Python-only ergonomics:

- TA-Lib-style imports such as ``ferro_ta.SMA(close, timeperiod=20)``
- Pre-built wheels for the supported Python/OS matrix
- Transparent ``pandas.Series`` / ``polars.Series`` support
- Type stubs (``.pyi``) and Sphinx autodoc under :doc:`api/index`

Adjacent tooling (not language bindings):

- **Backtesting engine** — OHLCV fill, 23 metrics, Monte Carlo, walk-forward, multi-asset — see :doc:`adjacent_tooling`
- Derivatives analytics — see :doc:`derivatives`
- Agentic workflow and LangChain tool wrappers — see `Agentic guide <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/agentic.md>`_
- MCP server for MCP-compatible clients — see `MCP guide <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/mcp.md>`_
- GPU helpers and plugins — see :doc:`adjacent_tooling`

Installation
~~~~~~~~~~~~

Pick a language. Full examples live on the language pages.

.. code-block:: bash

   pip install ferro-ta
   cargo add ferro_ta_core
   npm install ferro-ta-wasm
   flutter pub add ferro_ta

Quick Start
~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from ferro_ta import SMA, RSI, MACD, BBANDS

   close = np.linspace(44.0, 48.0, 40)
   print(SMA(close, timeperiod=5))
   print(RSI(close, timeperiod=14))
   macd_line, signal, hist = MACD(close)
   upper, middle, lower = BBANDS(close, timeperiod=5)

See :doc:`quickstart` for Rust, JavaScript, and Flutter snippets, or jump
straight to :doc:`languages/python`, :doc:`languages/rust`,
:doc:`languages/wasm`, or :doc:`languages/flutter`.

Further Reading
~~~~~~~~~~~~~~~

- `Architecture <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/architecture.md>`_ — core vs bindings, marshalling flow.
- `Performance Guide <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/performance.md>`_ — when to use raw numpy vs pandas/polars, batch notes, tips.
- `API Stability <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/stability.md>`_ — stability tiers, versioning, and deprecation policy.
- :doc:`support_matrix` — parity status, language coverage, tested wheel targets, and supported Python versions.
- `Core-first binding policy <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/rust_first.md>`_ — compute lives in ``ferro_ta_core``; every language is a thin wrapper.
- :doc:`languages/adding` — checklist for adding a new language binding.
- `Out-of-Core Execution <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/out-of-core.md>`_ — chunked processing and Dask integration.
- :doc:`derivatives` — IV helpers, options pricing/Greeks/IV, futures analytics, strategy schemas, and payoff helpers.
- :doc:`adjacent_tooling` — optional surfaces such as derivatives, MCP, GPU, plugins, and agent-oriented integrations.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
