Support Matrix
==============

The primary product is the Python technical analysis library: TA-Lib-style
indicator calls backed by a Rust implementation.

Indicator compatibility
-----------------------

.. list-table::
   :header-rows: 1

   * - Status
     - Scope
     - Notes
   * - Exact parity
     - Common TA-Lib-compatible indicators such as ``SMA``, ``WMA``,
       ``BBANDS``, ``RSI``, ``ATR``, ``NATR``, ``CCI``, ``STOCH``,
       ``STOCHRSI``, and most candlestick patterns
     - Matches TA-Lib numerically within floating-point tolerance in the
       current comparison suite.
   * - Approximate parity
     - EMA-family indicators (``EMA``, ``DEMA``, ``TEMA``, ``T3``, ``MACD``),
       ``MAMA`` / ``FAMA``, ``SAR`` / ``SAREXT``, and ``HT_*`` cycle
       indicators
     - Same API and intended use, with convergence-window or floating-point
       differences documented in the migration guide.
   * - Intentionally different
     - ferro-ta-only indicators such as ``VWAP``, ``SUPERTREND``,
       ``ICHIMOKU``, ``DONCHIAN``, ``KELTNER_CHANNELS``, ``HULL_MA``,
       ``CHANDELIER_EXIT``, ``VWMA``, and ``CHOPPINESS_INDEX``
     - These extend the library beyond TA-Lib and are not parity claims.

For migration details and known indicator-specific differences, see
:doc:`migration_talib`.

Module status
-------------

.. list-table::
   :header-rows: 1

   * - Surface
     - Status
     - Notes
   * - Top-level indicators and category submodules
     - Stable core
     - This is the main supported surface of the project.
   * - ``ferro_ta.batch``
     - Supported
     - Public API is supported; internal dispatch may evolve.
   * - ``ferro_ta.streaming``
     - Supported, still evolving
     - Suitable for live workflows; some API details are still marked
       experimental in the stability policy.
   * - ``ferro_ta.extended``
     - Supported extension
     - Useful indicators beyond TA-Lib, but not part of drop-in parity claims.
   * - ``ferro_ta.analysis.*``
     - Adjacent tooling
     - Useful analytics helpers, but not the primary product story.
   * - MCP, WASM, GPU, plugin, and agent-oriented tooling
     - Experimental or adjacent
     - Evaluate these independently from the core indicator library.

Supported Python versions
-------------------------

.. list-table::
   :header-rows: 1

   * - Python
     - Status
   * - 3.13
     - Supported and tested in CI
   * - 3.12
     - Supported and tested in CI
   * - 3.11
     - Supported and tested in CI
   * - 3.10
     - Supported and tested in CI
   * - < 3.10
     - Not supported

Tested wheel targets
--------------------

.. list-table::
   :header-rows: 1

   * - OS
     - Architecture
     - Wheel status
   * - Linux
     - ``x86_64`` (manylinux2014 / ``manylinux_2_17``)
     - Tested wheel target
   * - macOS
     - ``universal2``
     - Tested wheel target for Intel and Apple Silicon
   * - Windows
     - ``x86_64``
     - Tested wheel target

For source builds, packaging details, and platform notes, see
`PLATFORMS.md <https://github.com/pratikbhadane24/ferro-ta/blob/main/PLATFORMS.md>`_.

Release status
--------------

These docs track package version ``1.0.6``.

- Release notes by version: :doc:`changelog`
- Canonical project changelog: `CHANGELOG.md <https://github.com/pratikbhadane24/ferro-ta/blob/main/CHANGELOG.md>`_
- Stability policy: `docs/stability.md <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/stability.md>`_

If the package version, docs version, or support matrix disagree, treat that as
a documentation bug.
