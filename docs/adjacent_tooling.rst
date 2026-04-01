Adjacent Tooling
================

These modules are useful, but they are secondary to ferro-ta's core identity as
a Python technical analysis library.

.. list-table::
   :header-rows: 1

   * - Area
     - Status
     - What it is
   * - Backtesting engine
     - Adjacent
     - Vectorized Rust backtester: OHLCV fill, stop-loss/TP, 23 performance
       metrics, trade extraction, parallel Monte Carlo, walk-forward analysis,
       and multi-asset portfolio simulation. See :ref:`backtesting-engine`.
   * - Derivatives analytics
     - Adjacent
     - Options pricing, Greeks, implied volatility helpers, futures basis,
       curve, and roll utilities. See :doc:`derivatives`.
   * - Agent workflow wrappers
     - Adjacent
     - Tool and workflow helpers for agent-style integrations. See
       `docs/agentic.md <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/agentic.md>`_.
   * - MCP server
     - Experimental or adjacent
     - FastMCP-based server exposing selected ferro-ta capabilities to
       MCP-compatible clients. See
       `docs/mcp.md <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/mcp.md>`_.
   * - WASM package
     - Experimental
     - Browser and Node.js package with a smaller indicator subset. See
       `wasm/README.md <https://github.com/pratikbhadane24/ferro-ta/blob/main/wasm/README.md>`_.
   * - GPU backend
     - Experimental
     - Optional PyTorch-backed acceleration for a limited subset of indicators.
       See `docs/gpu-backend.md <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/gpu-backend.md>`_.
   * - Plugin system
     - Experimental
     - Registry and plugin packaging model for custom indicators. See
       :doc:`plugins`.

.. _backtesting-engine:

Backtesting Engine
------------------

``ferro_ta.analysis.backtest`` ships a production-grade backtesting engine
backed entirely by Rust hot-path functions.

**Core API:**

.. code-block:: python

    from ferro_ta.analysis.backtest import BacktestEngine, monte_carlo, walk_forward

    result = (
        BacktestEngine()
        .with_commission(0.001)
        .with_slippage(5.0)                       # basis points
        .with_ohlcv(high=high, low=low, open_=open_)
        .with_stop_loss(0.02)
        .with_take_profit(0.04)
        .run(close, "sma_crossover")
    )

    print(result.metrics["sharpe"])               # one of 23 metrics
    print(result.trades)                          # pandas DataFrame
    print(result.drawdown_series.min())           # max drawdown

    mc = monte_carlo(result, n_sims=1000)         # parallel bootstrap
    wf = walk_forward(close, "rsi", param_grid=[{"timeperiod": t} for t in [10,14,20]],
                      train_bars=500, test_bars=100)

**Available Rust primitives** (``ferro_ta._ferro_ta``):

- ``backtest_core`` — close-only, vectorized, commission + slippage
- ``backtest_ohlcv_core`` — fill at open, intrabar stop-loss / take-profit
- ``compute_performance_metrics`` — 23 metrics in one pass (Sharpe, Sortino,
  Calmar, CAGR, Omega, Ulcer, win rate, profit factor, tail ratio, etc.)
- ``extract_trades_ohlcv`` — 9 parallel arrays (entry/exit bar, MAE, MFE, …)
- ``backtest_multi_asset_core`` — N-asset parallel backtest via Rayon
- ``monte_carlo_bootstrap`` — parallel block bootstrap, returns (n_sims, n_bars)
- ``walk_forward_indices`` — anchored/rolling fold index generator
- ``kelly_fraction`` / ``half_kelly_fraction``

**Speed vs competitors** (100k bars, SMA crossover, Apple M-series):

.. list-table::
   :header-rows: 1

   * - Library
     - Time
     - vs ferro-ta
   * - ferro-ta ``backtest_core``
     - 0.29 ms
     - —
   * - NumPy vectorized
     - 0.46 ms
     - 1.6× slower
   * - vectorbt
     - 2.9 ms
     - 10× slower
   * - backtesting.py
     - 320 ms
     - 1,100× slower
   * - backtrader
     - ~520 ms (10k bars)
     - >15,000× slower

How to read the project
-----------------------

When evaluating ferro-ta:

- Start with the core library docs, migration guide, support matrix, and benchmarks.
- Treat adjacent tooling as opt-in layers, not as proof that the core indicator
  library is broader or more stable than it is.
- Check the release notes and stability policy before depending on experimental
  surfaces in production.
