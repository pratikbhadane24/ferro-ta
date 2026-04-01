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
   * - ``ferro_ta.analysis.resample``
     - Supported (v1.1.0)
     - ``resample_ohlcv()``, ``align_to_coarse()``, ``resample_ohlcv_labels()`` — pure-NumPy
       OHLCV bar aggregation across timeframes.
   * - ``ferro_ta.analysis.multitf``
     - Supported (v1.1.0)
     - ``MultiTimeframeEngine`` — multi-timeframe signal generation with automatic alignment.
   * - ``ferro_ta.analysis.adjust``
     - Supported (v1.1.0)
     - ``adjust_ohlcv()``, ``adjust_for_splits()``, ``adjust_for_dividends()`` — backward-adjusted
       price series for equity/index strategies.
   * - ``ferro_ta.analysis.plot``
     - Supported (v1.1.0)
     - ``plot_backtest()`` — interactive Plotly backtest visualization (requires plotly).
   * - ``ferro_ta.analysis.regime``
     - Supported (v1.1.0)
     - ``detect_volatility_regime()``, ``detect_trend_regime()``, ``detect_combined_regime()``,
       ``RegimeFilter`` — pure-NumPy 6-state market regime labeling; no ML dependencies.
   * - ``ferro_ta.analysis.optimize``
     - Supported (v1.1.0)
     - ``PortfolioOptimizer``, ``mean_variance_optimize()``, ``risk_parity_optimize()``,
       ``max_sharpe_optimize()`` — portfolio optimization via SLSQP (requires scipy).
   * - ``ferro_ta.analysis.live``
     - Supported (v1.1.0)
     - ``PaperTrader`` — event-driven paper trading bridge matching backtest logic exactly.
   * - MCP, WASM, GPU, plugin, and agent-oriented tooling
     - Experimental or adjacent
     - Evaluate these independently from the core indicator library.

Backtesting engine features
---------------------------

.. list-table::
   :header-rows: 1

   * - Feature
     - Status
     - Notes
   * - Flat/proportional commission
     - Supported
     - Via ``CommissionModel`` presets and ``BacktestEngine.with_commission_model()``.
   * - Bid-ask spread model (``spread_bps``)
     - Supported (v1.1.0)
     - New ``CommissionModel.spread_bps`` field; half-spread deducted per leg.
   * - Short borrow cost (``short_borrow_rate_annual``)
     - Supported (v1.1.0)
     - New ``CommissionModel.short_borrow_rate_annual`` field; accrued per bar for short positions.
   * - Trailing stop loss
     - Supported
     - ``BacktestEngine.with_trailing_stop(pct)`` — intrabar high-water mark tracking.
   * - Breakeven stop (``breakeven_pct``)
     - Supported (v1.1.0)
     - ``BacktestEngine.with_breakeven_stop(pct)`` — moves stop to entry once profit reaches ``pct``.
   * - Bracket order priority
     - Supported (v1.1.0)
     - When both SL and TP are breached on the same bar, the level closer to open fires first.
   * - Leverage / margin modeling
     - Supported (v1.1.0)
     - ``BacktestEngine.with_leverage(margin_ratio, margin_call_pct)`` — tracks margin and
       triggers force-close on margin call.
   * - Loss circuit breakers
     - Supported (v1.1.0)
     - ``BacktestEngine.with_loss_limits(daily, total)`` — halts trading on drawdown breach.
   * - Portfolio constraints
     - Supported (v1.1.0)
     - ``BacktestEngine.with_portfolio_constraints(max_asset_weight, max_gross_exposure,
       max_net_exposure)`` for multi-asset backtests.
   * - Volatility-target position sizing
     - Supported
     - ``BacktestEngine.with_position_sizing("volatility_target", ...)``.
   * - Walk-forward / Monte Carlo
     - Supported
     - Available via ``BacktestEngine`` higher-level methods.
   * - Benchmark comparison
     - Supported
     - ``BacktestEngine.with_benchmark(close_array)`` — alpha, beta, information ratio.

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

These docs track package version ``1.1.1``.

- Release notes by version: :doc:`changelog`
- Canonical project changelog: `CHANGELOG.md <https://github.com/pratikbhadane24/ferro-ta/blob/main/CHANGELOG.md>`_
- Stability policy: `docs/stability.md <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/stability.md>`_

If the package version, docs version, or support matrix disagree, treat that as
a documentation bug.
