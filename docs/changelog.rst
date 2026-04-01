Release Notes
=============

These docs track package version ``1.1.0``.

1.1.0-audit (2026-03-28)
------------------------

**Comprehensive audit: 90 findings addressed**

*Code quality & correctness*

- **Welford's algorithm for BBANDS**: replaced naive ``sum_sq/N - mean^2`` variance
  with numerically stable Welford's rolling algorithm in both batch and streaming BBANDS.
  Fixes catastrophic cancellation for large-valued series (e.g., prices near 1e12).
- **FFI boundary safety**: ``transpose_to_series_major()`` in ``batch/mod.rs`` now
  returns ``PyResult`` instead of using ``expect()``. Remaining ``as_slice().expect()``
  calls in ``allow_threads`` closures are documented with SAFETY comments (structurally
  infallible after C-contiguous transpose).
- **Clippy clean**: resolved all clippy warnings — complex type in ``adx_all`` extracted
  to ``AdxAllResult`` type alias; ``welford_step`` helper annotated with
  ``#[allow(clippy::too_many_arguments)]``.

*Performance*

- **``target-cpu=native``**: new ``.cargo/config.toml`` enables native CPU instruction
  set (AVX2, NEON, etc.) for all non-WASM targets. CI can override via ``RUSTFLAGS``.

*Testing*

- **Streaming unit tests**: 37 new tests in ``tests/unit/streaming/test_streaming.py``
  covering ``StreamingSMA``, ``StreamingEMA``, ``StreamingRSI`` — batch parity, warmup
  NaN behavior, reset, edge cases, and large dataset numerical stability.
- **Edge case tests**: 31 new tests in ``tests/unit/test_edge_cases.py`` — empty arrays,
  single elements, all-NaN input, NaN propagation, extreme values (1e300, 1e-300),
  constant series, period boundary conditions, OHLCV edge cases, and dtype coercion
  (float32, int64).
- **Property-based tests**: expanded Hypothesis tests for EMA, BBANDS, MACD, ATR, WMA,
  and OBV with algebraic invariants (upper >= middle >= lower, histogram == macd - signal,
  ATR non-negative, etc.).
- **Pandas/polars integration tests**: new ``test_dataframe_integration.py`` verifying
  transparent ``pd.Series`` and ``polars.Series`` support across SMA, EMA, RSI, BBANDS,
  MACD, and end-to-end DataFrame workflows.
- **Fuzzing**: expanded from 2 to 9 fuzz targets — added EMA, BBANDS, MACD, ATR, STOCH,
  MFI, and WMA with output invariant assertions.
- **Test helpers**: new ``tests/unit/helpers.py`` consolidating duplicated assertion
  patterns (``nan_count``, ``finite``, ``assert_nan_warmup``, ``assert_output_length``,
  ``assert_range``, ``make_ohlcv``).

*Documentation*

- **README benchmarks**: updated to match actual artifact data — MFI 3.25x, WMA 2.20x,
  BBANDS 1.97x, SMA 1.93x; corrected win count from 6 to 7 at 100k bars.
- **Rust doc comments**: added comprehensive ``///`` documentation to all public functions
  in ``ferro_ta_core`` — overlap (SMA, EMA, WMA, BBANDS, MACD), momentum (RSI, STOCH,
  ADX family), volatility (ATR, TRANGE), volume (OBV, MFI), statistic (STDDEV), and math
  (sum, max, min, sliding_max, sliding_min).

*Linting*

- **Ruff clean**: fixed import sorting, unused imports, trailing whitespace, and
  formatting across all Python files.
- **cargo fmt**: all Rust code formatted.

1.1.0 (2026-03-28)
------------------

**Phase 1 — Simulation fidelity**

- **Bid-ask spread model**: new ``CommissionModel.spread_bps`` field (basis points).
  Half-spread is deducted per leg (entry and exit), modelling real market microstructure costs.
- **Breakeven stop**: new ``backtest_ohlcv_core`` parameter ``breakeven_pct`` and
  ``BacktestEngine.with_breakeven_stop(pct)``.  Once profit reaches ``pct``, the
  effective stop-loss is moved to the entry price, guaranteeing at worst a breakeven exit.
- **Bracket order priority**: when both stop-loss and take-profit are breached on the
  same bar, the level closer to the bar's open price fires first (previously SL always won).

**Phase 2 — Portfolio & risk**

- **Short borrow cost**: new ``CommissionModel.short_borrow_rate_annual`` field.
  Accrued per bar for short positions at the specified annualised rate.
- **Leverage / margin modeling**: new ``BacktestEngine.with_leverage(margin_ratio, margin_call_pct)``.
  Tracks margin usage and triggers a margin-call force-close when equity falls below
  ``margin_call_pct × initial_margin``.
- **Loss circuit breakers**: new ``BacktestEngine.with_loss_limits(daily, total)``.
  Halts all trading when a per-bar loss or total drawdown threshold is breached.
- **Portfolio constraints**: new ``BacktestEngine.with_portfolio_constraints(max_asset_weight,
  max_gross_exposure, max_net_exposure)`` for multi-asset backtests.

**Phase 3 — Data & UX**

- **Bar aggregation** (``ferro_ta.analysis.resample``): ``resample_ohlcv()``, ``align_to_coarse()``,
  ``resample_ohlcv_labels()`` — pure-NumPy OHLCV resampling from any fine TF to any coarser TF.
- **Multi-timeframe engine** (``ferro_ta.analysis.multitf``): ``MultiTimeframeEngine`` — compute
  strategy signals on coarser bars and execute on finer bars, with automatic signal alignment.
- **Dividend/split adjustment** (``ferro_ta.analysis.adjust``): ``adjust_ohlcv()``,
  ``adjust_for_splits()``, ``adjust_for_dividends()`` — backward-adjusted price series for
  equity/index strategies.
- **Visualization** (``ferro_ta.analysis.plot``): ``plot_backtest()`` — interactive Plotly chart
  with equity curve, drawdown panel, position panel, trade markers, and optional benchmark overlay.

**Phase 4 — Differentiation**

- **Regime detection** (``ferro_ta.analysis.regime``): ``detect_volatility_regime()``,
  ``detect_trend_regime()``, ``detect_combined_regime()``, ``RegimeFilter`` — pure-NumPy
  6-state market regime labeling and signal filtering; no external ML dependencies.
- **Portfolio optimization** (``ferro_ta.analysis.optimize``): ``PortfolioOptimizer``,
  ``mean_variance_optimize()``, ``risk_parity_optimize()``, ``max_sharpe_optimize()`` —
  minimum-variance, risk-parity, and maximum-Sharpe portfolios via SLSQP (requires scipy).
- **Paper trading bridge** (``ferro_ta.analysis.live``): ``PaperTrader`` — event-driven
  bar-by-bar simulator matching ``backtest_ohlcv_core`` logic exactly; supports streaming
  data, live state inspection, and seamless strategy migration from backtesting to live.

1.1.0 (2026-03-27)
------------------

**Advanced commission and fee model (Indian market support)**

- New ``CommissionModel`` class (pure Rust in ``ferro_ta_core``, exposed via
  PyO3 and WASM) replaces the broken flat ``commission_per_trade`` scalar.  The
  old code subtracted an absolute currency amount from a 1.0-normalised equity
  curve — equivalent to a 2 000 % error on a ₹1 lakh account.  The new model
  correctly converts every charge to a fraction of ``initial_capital`` before
  deducting it from the equity curve.
- ``CommissionModel`` supports: proportional brokerage (``rate_of_value``),
  flat per-order fee (``flat_per_order``), per-lot fee (``per_lot``), brokerage
  cap (``max_brokerage``), Securities Transaction Tax (``stt_rate`` with
  configurable buy/sell sides), exchange transaction charges, SEBI regulatory
  charges, 18 % GST on brokerage + exchange + regulatory levies, and stamp duty
  on buy leg only.
- Built-in presets: ``CommissionModel.equity_delivery_india()``,
  ``CommissionModel.equity_intraday_india()``,
  ``CommissionModel.futures_india()``, ``CommissionModel.options_india()``,
  ``CommissionModel.proportional(rate)``, ``CommissionModel.zero()``.
- JSON persistence: ``model.to_json()`` / ``CommissionModel.from_json(s)``,
  ``model.save(path)`` / ``CommissionModel.load(path)``.
- ``BacktestEngine.with_commission_model(model)`` — pass a full
  ``CommissionModel``; old ``with_commission(rate)`` kept as a shim.
- New ``initial_capital`` parameter (default ₹1,00,000) on both
  ``backtest_core`` and ``backtest_ohlcv_core``; also exposed as
  ``BacktestEngine.with_initial_capital(capital)``.

**Currency system — INR default with lakh/crore formatting**

- New ``Currency`` immutable descriptor in the Python layer with constants
  ``INR``, ``USD``, ``EUR``, ``GBP``, ``JPY``, ``USDT``.
- ``INR`` is the default currency for ``BacktestEngine``; change via
  ``engine.with_currency("USD")`` or ``engine.with_currency(EUR)``.
- ``currency.format(amount)`` produces Indian lakh/crore grouping for INR
  (e.g. ``₹1,23,45,678.00``) and standard Western grouping for other
  currencies.
- Module-level helper ``format_currency(amount, currency=INR)``.
- ``AdvancedBacktestResult`` gains ``currency``, ``initial_capital``, and
  ``equity_abs`` (absolute currency equity curve) slots.
- ``summary()`` now includes ``initial_capital``, ``final_capital``,
  ``absolute_pnl``, and ``currency`` keys.
- ``AdvancedBacktestResult.__repr__`` shows the final capital in the correct
  currency symbol (e.g. ``final=₹1,23,450.00``).
- Trade log gains a ``pnl_abs`` column (PnL in absolute currency units).
- ``to_equity_dataframe()`` now includes an ``equity_abs`` column.

**Trailing stop loss**

- ``backtest_ohlcv_core`` (and ``BacktestEngine.with_trailing_stop(pct)``)
  now supports a trailing stop implemented intrabar in Rust: the high-water
  mark is updated each bar; the position is exited at
  ``trail_high × (1 − pct)`` when ``low[i]`` crosses below it (long trades),
  or ``trail_low × (1 + pct)`` for short trades.

**Benchmark comparison metrics**

- ``compute_performance_metrics`` accepts an optional ``benchmark_returns``
  array.  When provided, ``summary()`` includes: ``benchmark_total_return``,
  ``benchmark_cagr``, ``benchmark_annualized_vol``, ``benchmark_sharpe``,
  ``alpha`` (active return), ``beta``, ``tracking_error``, and
  ``information_ratio``.
- ``BacktestEngine.with_benchmark(close_array)`` — pass benchmark close prices.

**Volatility-target position sizing**

- New ``"volatility_target"`` method for ``with_position_sizing()``:
  ``engine.with_position_sizing("volatility_target", target_vol=0.15, vol_window=20)``.
  Signals are pre-scaled in Python by ``clip(target_vol / rolling_annualised_vol, 0, 3)``
  before the Rust core call, keeping the hot loop unchanged.

**Backtesting engine v2 — full feature set**

- ``BacktestEngine`` now supports true two-pass Kelly / half-Kelly position
  sizing: a unit-signal pass computes win statistics, then the core engine
  re-runs with signals scaled by the Kelly fraction.
- Added ``fixed_fractional`` position sizing method:
  ``engine.with_position_sizing("fixed_fractional", fraction=0.5)``.
- New ``StreamingBacktest`` Rust class for bar-by-bar incremental backtesting
  (no bulk arrays needed); exposes ``.on_bar()``, ``.summary()``, ``.reset()``.
- ``AdvancedBacktestResult.to_equity_dataframe(freq)`` — returns equity,
  returns, and drawdown as a ``pd.DataFrame`` with a synthetic DatetimeIndex.
- ``AdvancedBacktestResult.summary()`` — concise dict of the 9 most commonly
  cited metrics plus ``n_trades``.

**Core indicator speedup**

- ADX-family indicators (``adx_all`` public API): all six series (PDM, MDM,
  +DI, -DI, DX, ADX) can now be computed from a single TR/PDM/MDM pass via
  ``ferro_ta.adx_all()``, eliminating the 6× redundant computation that
  occurred when callers fetched each series independently.
- ``adxr`` now reuses a single ``adx_inner`` call internally (was calling
  ``adx()`` which re-ran the inner loop).

1.0.6 (2026-03-24)
------------------

- Added a repo-managed pre-push gate so the core Rust, Python, docs, and WASM
  checks can be run locally before release.
- Expanded Rust-backed analysis/data helpers, broadened the WASM exports, and
  added cross-surface API manifest verification plus Node conformance checks.
- Refreshed benchmark coverage and perf artifacts, aligned Python CI with the
  local tooling flow, and updated the locked security fixes needed for a clean
  release pass.

1.0.4 (2026-03-24)
------------------

- Expanded the optional MCP server from a small hand-written subset to the
  broader public ferro-ta callable surface, including stateful class support
  through stored-instance management tools.
- Split the root documentation so the full TA-Lib compatibility matrix lives in
  ``TA_LIB_COMPATIBILITY.md`` while the README stays product-first and shorter.
- Refreshed MCP docs/tests and updated locked low-risk Python dependencies as
  part of the release cleanup pass.
- Stopped tracking the stray ``.coverage`` artifact and aligned ignore rules
  for local coverage outputs.

1.0.3 (2026-03-24)
------------------

- Added top-level package metadata helpers such as ``ferro_ta.__version__``,
  ``ferro_ta.about()``, and ``ferro_ta.methods()``.
- Added a standalone derivatives benchmark artifact for selected options
  pricing, IV, Greeks, and Black-76 comparisons.
- Simplified release version bumps with a single script and updated release
  guidance.
- Fixed Python CI/type-stub gaps around the new metadata API and corrected the
  tag-driven GitHub Release workflow trigger used for publish automation.

1.0.2 (2026-03-24)
------------------

- Improved rolling statistical kernels and several Python analysis hotspots.
- Added reproducible perf-contract artifacts, TA-Lib regression guards, and
  updated benchmark tooling.
- Tightened the public benchmark documentation so claims, caveats, and evidence
  live closer together.

1.0.1 (2026-03-24)
------------------

- Improved release automation for PyPI, crates.io, and npm.
- Fixed CI workflow issues that caused otherwise healthy release jobs to fail.
- Ensured the published WASM package includes its built ``pkg/`` artifacts.

1.0.0 (2026-03-23)
------------------

- First stable release of the Rust-backed Python technical analysis library.
- Shipped broad TA-Lib coverage, streaming APIs, extended indicators, and the
  initial Sphinx documentation set.
- Added the benchmark suite, release playbook, and compatibility/testing
  scaffolding for stable releases.

For the canonical project changelog, including the full per-version details,
see `CHANGELOG.md <https://github.com/pratikbhadane24/ferro-ta/blob/main/CHANGELOG.md>`_.
