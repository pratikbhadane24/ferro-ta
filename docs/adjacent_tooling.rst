Adjacent Tooling
================

These modules are useful, but they are secondary to ferro-ta's core identity as
a Rust-core technical analysis library. Language bindings (Python, Rust,
JavaScript/WASM, Flutter) are first-class products — they are documented under
:doc:`languages/index`, not here.

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

``ferro_ta.analysis.backtest`` ships a vectorized backtesting engine whose
hot paths run entirely in Rust.

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
- ``backtest_sweep_core`` — one close series × N signal columns, Rayon-parallel
- ``probabilistic_sharpe_ratio`` / ``deflated_sharpe_ratio`` / ``expected_max_sharpe``
- ``pbo_cscv`` — Probability of Backtest Overfitting via CSCV
- ``purged_kfold_indices`` — k-fold boundaries with purge/embargo gaps

.. _honest-optimization:

Parameter sweeps and overfitting diagnostics
--------------------------------------------

Search enough parameter combinations and one will look brilliant by chance.
``ferro_ta.analysis.sweep`` runs the whole grid in parallel Rust and keeps
every combination's returns, so ``ferro_ta.analysis.overfit`` can tell you how
much of the winner's edge survives the search:

.. code-block:: python

    from ferro_ta.analysis.sweep import sweep

    res = sweep(close, "sma_crossover",
                {"fast": [5, 10, 20, 30], "slow": [50, 100, 150, 200]},
                commission_per_trade=1.0, slippage_bps=2)

    res.best_params("sharpe")        # {'fast': 10, 'slow': 100}
    res.best_params("sharpe", radius=1)   # best *neighbourhood*, not best spike
    res.to_frame()                   # pandas: params + 24 metrics per combo
    rows, cols, grid = res.pivot("sharpe", "fast", "slow")   # heatmap

    res.deflated_sharpe()            # DSR of the winner vs all 16 trials
    res.pbo(n_splits=16).pbo         # P(IS winner is below OOS median)

How to read the numbers:

- **Deflated Sharpe Ratio** (Bailey & López de Prado, 2014) is the probability
  that the selected strategy's true Sharpe beats the best Sharpe you would
  expect from the same number of *zero-skill* trials. Above ~0.95 suggests a
  real edge; around 0.5 means the winner is indistinguishable from luck.
- **PBO** (Bailey, Borwein, López de Prado & Zhu, 2017) splits the history into
  ``n_splits`` blocks and, for every half/half split, checks where the
  in-sample winner ranks out of sample. Pure noise gives ~0.5; a genuinely
  dominant configuration gives ~0. ``PBOResult.is_sharpe`` / ``oos_sharpe``
  expose the performance degradation, ``prob_oos_loss`` the chance the
  winner loses money out of sample.
- **Purged k-fold** (``purged_kfold``) drops ``purge_bars`` before and
  ``embargo_bars`` after each test fold, for labels or features that span
  multiple bars.

All Sharpe ratios in ``overfit`` are per-period (not annualised). Sweep
columns use exactly the same engine as ``backtest()``, so each column matches
a single run bit-for-bit.

**Performance shape.** The signal → equity loop, commission and slippage
application, all 23 metrics, trade extraction, the Monte Carlo bootstrap and the
multi-asset run happen in Rust with the GIL released; ``backtest_core`` and
``backtest_ohlcv_core`` are single O(n) passes over the bars, and
``walk_forward_indices`` is O(number of folds) rather than O(bars). It is the
right tool for signal-array backtests, parameter sweeps, bootstrap confidence
intervals and array-expressible multi-asset runs — and the wrong tool for
path-dependent strategies that must decide inside the bar loop in Python, or
for anything needing a broker event model.

No timings are reproduced here. Read
``benchmarks/artifacts/latest/bench_backtest_results.json``, which records
per-size timings for every path above alongside the machine, commit, Python
version and Rust toolchain that produced them, and regenerate it with:

.. code-block:: bash

   python benchmarks/bench_backtest.py --sizes 10000 100000 \
       --json /tmp/bench_backtest_results.json

See :doc:`benchmarks` for what the backtester buys you structurally, and
``docs/performance.md`` for the wider performance guide.

How to read the project
-----------------------

When evaluating ferro-ta:

- Start with the :doc:`languages/index` hub, the Python migration guide, the
  support matrix, and benchmarks.
- Treat adjacent tooling as opt-in layers, not as proof that the core indicator
  library is broader or more stable than it is.
- Check the release notes and stability policy before depending on experimental
  surfaces in production.
