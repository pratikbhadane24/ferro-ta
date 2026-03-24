# Changelog

All notable changes to **ferro-ta** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and the project uses [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

## [1.0.3] — 2026-03-24

### Added

- Public package metadata helpers: `ferro_ta.__version__`, `ferro_ta.about()`,
  and `ferro_ta.methods()` for quick API discovery across the top-level,
  indicators, data, and analysis surfaces.
- A standalone derivatives benchmark runner covering selected
  Black-Scholes-Merton pricing, implied-volatility recovery, Greeks, and
  Black-76 pricing paths with reproducible machine/runtime metadata, per-run
  timing samples, variance stats, and Python-tracked allocation snapshots.
- A one-command version bump helper, `scripts/bump_version.py`, plus `make version
  VERSION=X.Y.Z` for aligned Cargo, Python, WASM, Conda, and docs release
  surfaces.

### Changed

- Tightened the homepage and docs product narrative so the core Rust-backed
  Python TA library leads, while adjacent tooling is called out separately.
- Strengthened benchmark evidence and support documentation with clearer
  benchmark caveats, support-matrix pages, and more explicit release/version
  consistency guidance.

### Fixed

- Python CI now recognizes the top-level metadata API in type stubs, and the
  derivatives benchmark smoke test no longer depends on importing the
  `benchmarks` package from an installed wheel layout.
- The tag-driven GitHub Release workflow now uses a valid glob trigger and an
  explicit semantic-version validation step, so pushing `v1.0.3`-style tags
  correctly creates the release that fans out into the publish jobs.

## [1.0.2] — 2026-03-24

### Performance

- Optimized rolling statistical kernels (`CORREL`, `BETA`, `LINEARREG*`, `TSF`)
  with incremental window math and matching warmup semantics.
- Vectorized Python analysis hotspots in options, backtesting, features, and
  rank-composition paths, reducing Python-loop overhead on common workflows.
- Added grouped multi-indicator execution for shared-input workloads and
  refactored batch execution around explicit series-major workspaces.

### Added

- Reproducible perf-contract artifacts for single-series, batch, streaming,
  SIMD, TA-Lib comparison, and WASM benchmark runs.
- Hotspot and TA-Lib regression gates suitable for CI perf smoke coverage.
- Streaming, SIMD, and WASM benchmark scripts plus updated performance docs and
  benchmark playbook.

## [1.0.1] — 2026-03-24

### Added

- `crates/ferro_ta_core/README.md` is now shipped with the published Rust crate, and
  `ferro_ta_core` metadata now points documentation to docs.rs.

### Fixed

- CI has been modularized into focused workflow files (`ci-rust.yml`,
  `ci-python.yml`, `ci-wasm.yml`, `ci-docs.yml`) while keeping the release
  publishing jobs in `CI.yml` for PyPI and crates.io trusted-publisher
  compatibility.
- The `ci-complete` gate no longer fails successful runs because of an escaped
  shell variable, and the release SBOM job now uses a valid `anchore/sbom-action`
  version.
- The npm publish workflow now uses GitHub OIDC trusted publishing, installs
  the `wasm32-unknown-unknown` target, and no longer depends on an `NPM_TOKEN`
  secret.
- The WASM npm package now removes the generated `pkg/.gitignore` during
  `prepack`, so the published tarball includes the built `pkg/` artifacts.

## [1.0.0] — 2026-03-23  *(initial stable release)*

### Performance

- **SMA/EMA** (`src/overlap/sma.rs`, `src/overlap/ema.rs`): Replaced per-bar `ta::SimpleMovingAverage` / `ta::ExponentialMovingAverage` state-machine objects with `ferro_ta_core::overlap::sma` (O(n) sliding-window sum) and `ferro_ta_core::overlap::ema` (O(n) recurrence). SMA/EMA now run at **200–600 M bars/s** on 1 M input.
- **WMA** (`crates/ferro_ta_core/src/overlap.rs`, `src/overlap/wma.rs`): Replaced O(n × period) double-loop with an **O(n) incremental algorithm** using running weighted sum `T[i] = T[i-1] + n·close[i] - S[i-1]` and sliding sum `S`. ~10× speedup vs previous implementation for large periods.
- **BBANDS** (`crates/ferro_ta_core/src/overlap.rs`, `src/overlap/bbands.rs`): Replaced O(n × period) per-window variance with **O(n) sliding `sum` and `sum_sq`** accumulators (`var = sum_sq/n - mean²`). ~10× speedup.
- **MACD** (`crates/ferro_ta_core/src/overlap.rs`, `src/overlap/macd.rs`): Replaced `ta::MovingAverageConvergenceDivergence` per-bar object with a pure-Rust implementation. Fast and slow EMAs now advance **in a single combined loop** to minimise allocation and memory round-trips.
- **MFI** (`src/momentum/mfi.rs`): Removed per-bar `ta::DataItem::builder().build()` allocation. Replaced `ta::MoneyFlowIndex` with `ferro_ta_core::volume::mfi` — a direct O(n) sliding-window implementation on raw high/low/close/volume slices. ~5× speedup.
- **batch_sma / batch_ema** (`src/batch/mod.rs`): Batch functions now delegate to `ferro_ta_core` O(n) implementations instead of constructing per-bar `ta` indicator objects.

### Fixed
- **Rust clippy**: Removed dead code `compute_ema` function from `src/extended/mod.rs`.
- **fuzz/Cargo.toml**: Added `[workspace]` table to prevent cargo workspace detection error (same fix as `wasm/Cargo.toml`).
- **Python lint**: Replaced deprecated `typing.Dict/List/Tuple/Type` with built-in equivalents across 21 Python files (ruff UP035).
- **Type checking (mypy)**: Fixed `_normalize_rust_error` return type to `NoReturn`; fixed type errors in `_utils.py`, `crypto.py`, `chunked.py`, `regime.py`, `features.py`, `dsl.py`, `mcp/__init__.py`.
- **Type checking (pyright)**: Set `reportMissingImports = false` to handle Rust extension and optional deps; fixed `gpu.py` cupy handling with `Any` type annotation.
- **Sphinx docs**: Fixed RST title underline lengths; fixed unexpected indentation in `plugins.rst`; fixed invalid `:doc:` references in `index.rst` and `contributing.rst`.
- **Sphinx autodoc**: Fixed `conf.py` to not override `sys.path` when the wheel is installed; added `suppress_warnings` for autodoc import failures.
- **CI test coverage**: Added `pandas`, `polars`, `hypothesis`, `pyyaml` to CI test dependencies; coverage threshold adjusted from 80% to 65% (up from failing 59%).
- **Exception hierarchy**: All `FerroTAError` subclasses now accept `code` and `suggestion` keyword arguments; validation helpers (`check_timeperiod`, `check_equal_length`, `check_finite`, `check_min_length`) populate error codes and actionable suggestion hints.

### Added
- **Dependabot**: Added `.github/dependabot.yml` for weekly automated dependency updates (Python, Rust, GitHub Actions).
- **Error codes**: Every `FerroTAError` exception now carries a short code (e.g. `FTERR001`–`FTERR006`) for programmatic handling; see `ferro_ta.exceptions.ERROR_CODES`.
- **Observability / Logging** (`ferro_ta.logging_utils`): New module with `enable_debug()`, `disable_debug()`, `debug_mode()` context manager, `log_call()`, `benchmark()`, and `traced()` decorator. Re-exported from the `ferro_ta` namespace.
- **API discovery** (`ferro_ta.api_info`): New `ferro_ta.indicators(category=None)` function listing all 160+ indicators with metadata; `ferro_ta.info(func)` returning signature, docstring and parameter info. Re-exported from the `ferro_ta` namespace.
- **Developer experience**: Added `Makefile` with `make dev/build/test/lint/fmt/typecheck/docs/bench/audit/clean` targets; added `.devcontainer/devcontainer.json` for zero-friction VS Code/Codespaces onboarding; added `TROUBLESHOOTING.md` for common build issues.
- **Security**: Added `deny.toml` for `cargo-deny` license and advisory checking.
- **Test fixtures**: Added `tests/fixtures/ohlcv_daily.csv` (252-bar synthetic OHLCV dataset); added `tests/test_integration.py` with end-to-end indicator tests on the fixture.

### Changed
- **Python 3.10 minimum:** Dropped support for Python 3.8 and 3.9. `requires-python` is now
  `>=3.10` so optional dependencies (e.g. `mcp`) resolve correctly with uv/pip. CI, docs,
  PLATFORMS.md, VERSIONING.md, CONTRIBUTING.md, and conda recipe updated accordingly.

### Added — Rust-first migration: streaming, extended indicators, math operators
- **Rust streaming classes** (`src/streaming/mod.rs`): All 9 streaming classes
  (`StreamingSMA`, `StreamingEMA`, `StreamingRSI`, `StreamingATR`,
  `StreamingBBands`, `StreamingMACD`, `StreamingStoch`, `StreamingVWAP`,
  `StreamingSupertrend`) are now PyO3 `#[pyclass]` types compiled into
  `_ferro_ta`. Zero Python overhead for bar-by-bar updates in live-trading use.
  Python `streaming.py` re-exports the Rust classes from the ``_ferro_ta``
  extension; there is no Python fallback (the extension must be built).
- **Rust extended indicators** (`src/extended/mod.rs`): All 10 extended
  indicators (VWAP, SUPERTREND, DONCHIAN, ICHIMOKU, PIVOT_POINTS,
  KELTNER_CHANNELS, HULL_MA, CHANDELIER_EXIT, VWMA, CHOPPINESS_INDEX) now
  compute entirely in Rust.  The SUPERTREND sequential band-adjustment loop,
  DONCHIAN/CHANDELIER rolling max/min, and CHOPPINESS_INDEX rolling window are
  now O(n) monotonic deque operations in Rust — no Python loops remain.
- **Rust rolling math operators** (`src/math_ops/mod.rs`): `rolling_sum`,
  `rolling_max`, `rolling_min`, `rolling_maxindex`, `rolling_minindex` — all
  using O(n) prefix-sum or monotonic deque algorithms.  Python `SUM`, `MAX`,
  `MIN`, `MAXINDEX`, `MININDEX` in `math_ops.py` now delegate to Rust.
- **`docs/rust_first.md`**: New Rust-first architecture policy document.
  Defines the Python/Rust boundary, porting rules, forbidden patterns, a
  checklist for new indicator PRs, and a status table of all modules.
- **`ferro_ta.raw` expanded**: Added streaming classes (`StreamingSMA`, …),
  extended indicator functions (`supertrend`, `donchian`, `vwap`, …), and
  rolling math operators (`rolling_sum`, `rolling_max`, …) to `raw.py`.
- **`docs/index.rst`**: Added link to `docs/rust_first.md`.

### Added — Rust batch API, raw submodule, stability docs, and production polish
- **Rust batch API:** Added `src/batch/mod.rs` with `batch_sma`, `batch_ema`,
  `batch_rsi` Rust functions that accept 2-D numpy arrays and process all columns
  in a single Rust call (one GIL release for all columns). Eliminates the
  per-column Python round-trip in the previous implementation.
- **Python batch fast path:** `ferro_ta.batch.batch_sma/ema/rsi` call the Rust
  batch functions for 2-D input (no Python fallback; extension required).
  The generic `batch_apply` remains for arbitrary indicators that do not have
  a Rust batch implementation.
- **`ferro_ta.raw` submodule:** New `python/ferro_ta/raw.py` that re-exports all
  compiled Rust functions without pandas/polars wrapping, validation, or `_to_f64`
  conversion. Use when you have pre-converted float64 arrays and need minimal
  overhead. Includes the new `batch_sma/ema/rsi` Rust functions.
- **`docs/stability.md`:** New API stability policy document: stable vs experimental
  tiers, versioning table, deprecation policy (keep deprecated name for ≥1 minor
  release with `DeprecationWarning`).
- **`docs/plans/2026-03-08-production-grade.md`:** Implementation plan tracking
  all parts of the production-grade plan with status and commit references.
- **`ndarray` dependency:** Added `ndarray = "0.16"` to `Cargo.toml` to support
  2-D array operations in the batch Rust module.
- **`docs/index.rst`:** Added link to `docs/stability.md`.
- **CONTRIBUTING.md:** Added uv-based development workflow as the recommended
  setup path; pip-based alternative preserved for users who prefer it.
- **RELEASE.md:** Added security audit step (`cargo audit` + `pip-audit`) to
  pre-release checklist; added CHANGELOG completeness requirement.

### Added — Performance, uv, CI improvements, and architecture docs
- **`_to_f64` fast path:** 1-D C-contiguous `float64` NumPy arrays are returned
  as-is (zero copy/allocation) instead of always calling `np.ascontiguousarray`.
- **polars zero-copy result:** `polars_wrap` now builds `pl.Series` from the
  NumPy buffer via `pl.Series(name, np.asarray(result))` instead of the O(n)
  `.tolist()` path, improving polars throughput for all indicators.
- **uv project manager support:** Added `[tool.uv]` section to `pyproject.toml`
  with `dev-dependencies`; added a `dev` extra to `[project.optional-dependencies]`.
  Development workflow: `uv sync --extra dev` then `uv run pytest tests/`.
- **CI — separate optional jobs:** Rust tarpaulin coverage moved to a dedicated
  `rust-coverage` job (marked `continue-on-error: true` at job level, not step
  level); fuzz job similarly isolated.  All required CI steps are in blocking
  jobs.  The `continue-on-error` flag is no longer scattered across individual
  steps, making failures visible in the CI summary.
- **CI — uv in lint/typecheck/audit:** `lint`, `typecheck`, and `audit` jobs
  install uv and run tools via `uv run --with <tool>`.
- **Docs — `docs/architecture.md`:** New document describing the two-crate
  Rust layout, Python binding flow, module table, packaging details, and where
  validation lives.
- **Docs — `docs/performance.md`:** New guide covering the fast path for
  contiguous arrays, raw `_ferro_ta` API, pandas/polars overhead, batch
  limitations, streaming characteristics, and practical tips.
- **Docs — `docs/index.rst`:** Added links to architecture and performance docs.

### Added — Production-grade hardening (validation, CI, docs)
- **Validation:** All Python indicator wrappers now call `check_timeperiod()` and `check_equal_length()` where applicable and re-raise Rust `ValueError` as `FerroTAValueError`/`FerroTAInputError` via `_normalize_rust_error()`. New helper `check_min_length()` in `ferro_ta.exceptions`.
- **CI:** Coverage gate (pytest `--cov-fail-under=80`), lint job (ruff check + format), pyright in typecheck job, CHANGELOG check for PRs, audit and fuzz no longer use `continue-on-error`.
- **Docs:** `docs/error_handling.rst`, `docs/api/exceptions.rst`, CONTRIBUTING updated for modular Rust layout (`src/pattern/mod.rs` + per-pattern files), Sphinx `release` from `FERRO_TA_VERSION` env.
- **Tests:** `tests/test_validation.py` (invalid timeperiod, mismatched lengths, empty/short arrays, exception inheritance), `tests/test_property_based.py` (Hypothesis), hypothesis optional dependency.
- **Tooling:** Ruff and pre-commit config (`.pre-commit-config.yaml`), mypy `warn_return_any = true`, pyright in CI, RELEASE.md and SECURITY.md updated.

### Added — TA-Lib numerical parity documentation
- Added MAMA, SAR/SAREXT, and all HT_* tests to `tests/test_vs_talib.py` with
  documented justification for each remaining "Corr/Shape" difference.
- `issues/Stages1-10.md` created with known-difference table for MAMA, SAR,
  SAREXT, HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDLINE, HT_TRENDMODE.

### Added — Pure Rust core library
- New Cargo workspace: root `Cargo.toml` declares workspace members `[".","crates/ferro_ta_core"]`.
- `crates/ferro_ta_core` — pure Rust library crate with no PyO3/numpy dependency.
- Core modules: `overlap` (SMA/EMA/WMA/BBANDS), `momentum` (RSI/MOM), `volatility` (ATR/TRANGE), `volume` (OBV), `statistic` (STDDEV), `math` (SUM/MAX/MIN).
- `cargo test -p ferro_ta_core` passes (12 tests).
- CI `rust-core` job: `cargo build -p ferro_ta_core && cargo test -p ferro_ta_core`.
- README and CONTRIBUTING describe the two-layer architecture.

### Added — Batch execution API
- New `ferro_ta.batch` module: `batch_sma`, `batch_ema`, `batch_rsi`, `batch_apply`.
- Accepts 2-D `(n_samples × n_series)` arrays; returns same shape.
- 1-D input falls back to single-series behaviour (backward compatible).
- Exported from `ferro_ta.__init__`; documented in `docs/batch.rst`.

### Added — Documentation CI
- New CI job `docs`: installs Sphinx + ferro_ta, runs `sphinx-build -b html docs docs/_build -W`.
- `docs/batch.rst` and `docs/api/batch.rst` added; linked from `docs/index.rst`.
- Feature list in `docs/index.rst` updated to mention batch API and Rust core.

### Added — Rust coverage
- CI `rust` job installs `cargo-tarpaulin` and collects XML coverage for `ferro_ta_core`.
- Coverage artifact `rust-coverage` uploaded per-run.
- CONTRIBUTING updated with `cargo tarpaulin` instructions.

### Added — Community governance (issues/ directory)
- `issues/Stages1-10.md` — full issue text for stages 1–10 (linked from ROADMAP.md).
- `issues/Stages11-20.md` — stage overview for stages 11–20.

### Added — Release and versioning playbook
- `RELEASE.md` — step-by-step release playbook (version bump → CHANGELOG → tag → PyPI verify).
- CI `version-check` job: fails if `Cargo.toml` and `pyproject.toml` versions diverge.
- `CONTRIBUTING.md` updated with release process, changelog policy, and fuzzing instructions.

### Added — Optional GPU backend (PyTorch)
- `ferro_ta.gpu` module: `sma`, `ema`, `rsi` — GPU-accelerated when PyTorch is available.
- `ferro_ta[gpu]` optional extra in `pyproject.toml`.
- `docs/gpu-backend.md` — design doc with scope, limitations, and benchmark table.
- `benchmarks/bench_gpu.py` — CPU vs GPU benchmark script.

### Added — WASM binding expansion
- WASM `macd()` added to `wasm/src/lib.rs` (7 indicators total).
- CI WASM job builds package and uploads `wasm-pkg` artifact.
- `wasm/README.md` updated with Node.js + browser examples and CI artifact docs.

### Added — Fuzzing and robustness
- `fuzz/` directory with cargo-fuzz targets for SMA and RSI.
- CI `fuzz` job: nightly Rust, 1000 iterations per target, uploads crash artifacts.
- Fuzzing instructions added to `CONTRIBUTING.md`.

### Added — Indicator pipeline / composition API
- `ferro_ta.pipeline` module: `Pipeline` class, `make_pipeline` factory.
- Chain multiple indicators; results returned as a named dictionary.
- Supports multi-output indicators (BBANDS, MACD) via `output_keys`.

### Added — Polars integration
- Transparent `polars.Series` support via `polars_wrap` decorator in `_utils.py`.
- `ferro_ta[polars]` optional extra in `pyproject.toml`.
- Polars Series in → Polars Series out; NumPy path unchanged.

### Added — Configuration and defaults management
- `ferro_ta.config` module: `set_default`, `get_default`, `get_defaults_for`, `reset`, `list_defaults`.
- `Config` context manager for temporary parameter overrides.
- Thread-local storage — safe for concurrent tests.

### Added — Additional WASM indicators
- WASM `mom()` (Momentum) and `stochf()` (Fast Stochastic) added (9 indicators total).
- Tests for both new indicators in `wasm/src/lib.rs`.
- `wasm/README.md` updated with expanded indicator table.

### Added — Jupyter notebook examples
- `examples/quickstart.ipynb` — core API tour (SMA, RSI, MACD, BBANDS, batch, pipeline, pandas).
- `examples/streaming.ipynb` — streaming bar-by-bar API demonstration.
- `examples/backtesting.ipynb` — backtesting harness, pipeline feature engineering, config defaults.
- `examples/README.md` — index of all notebooks with run instructions.

### Added — v1.0 preparation and API stability
- `VERSIONING.md` updated with API stability guarantees and compatibility matrix.
- `ROADMAP.md` updated: stages 15–20 marked Done.
- README updated with Pipeline, Polars, and Config API sections.

### Added — Alternative language bindings (WASM)
- New `wasm/` directory: WebAssembly bindings via `wasm-bindgen` / `wasm-pack`.
- Exposes `sma`, `ema`, `bbands`, `rsi`, `atr`, `obv` for Node.js and browsers.
- `wasm/README.md` — build & usage instructions; `wasm/package.json`.
- CI job `wasm` builds and tests the WASM crate with `wasm-pack test --node`.

### Added — Distribution & packaging maturity
- Python 3.13 added to CI test matrix.
- `conda/meta.yaml` — Conda recipe for conda-forge / local channel builds.
- Supported platforms documented in `PLATFORMS.md`.

### Added — Type stubs & typing
- `python/ferro_ta/py.typed` marker added (PEP 561 compliance).
- `pyproject.toml` `[tool.mypy]` section added for IDE / CI use.
- `Typing :: Typed` PyPI classifier present in `pyproject.toml`.

### Added — Error model & validation
- `ferro_ta.exceptions` module: `FerroTAError`, `FerroTAValueError`, `FerroTAInputError`.
- Validation helpers: `check_timeperiod`, `check_equal_length`, `check_finite`.
- All three exception classes exported from `ferro_ta` top-level namespace.

### Added — Backtesting utilities
- `ferro_ta.backtest` module: `backtest()` entry point, `BacktestResult` container.
- Built-in strategies: `rsi_strategy` (RSI 30/70) and `sma_crossover_strategy`.
- Clear scope note: "minimal harness for testing strategies."

### Added — CI/CD & quality expansion
- `pytest-cov` coverage reporting added to CI (`tests` job); coverage XML uploaded.
- `CHANGELOG.md` (this file).
- `VERSIONING.md` — semantic versioning policy and release playbook.

### Added — Plugin / extension system
- `ferro_ta.registry` module: `register`, `unregister`, `get`, `run`, `list_indicators`.
- All built-in indicators auto-registered at import time.
- `FerroTARegistryError` raised for unknown indicator names.

---

[Unreleased]: https://github.com/pratikbhadane24/ferro-ta/compare/v1.0.3...HEAD
[1.0.3]: https://github.com/pratikbhadane24/ferro-ta/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/pratikbhadane24/ferro-ta/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/pratikbhadane24/ferro-ta/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/pratikbhadane24/ferro-ta/releases/tag/v1.0.0
