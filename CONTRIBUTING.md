# Contributing to ferro-ta

Thank you for your interest in contributing to **ferro-ta**! This guide explains how to add new candlestick patterns and other indicators.

## Prerequisites

- Rust toolchain (stable, ≥ 1.70)
- **Python 3.10–3.13** (PyO3 supports up to 3.13; for 3.14+ use a separate venv with an older interpreter)
- [maturin](https://www.maturin.rs/) (`pip install maturin`)
- numpy (`pip install numpy`)
- pytest (`pip install pytest`)

## Recommended: set up with uv (fast, reproducible)

[uv](https://docs.astral.sh/uv/) is the recommended development tool for ferro-ta.
It handles virtual environments, dependency locking, and running commands:

```bash
# Install uv (once)
pip install uv          # or: curl -Lsf https://astral.sh/uv/install.sh | sh

# Sync dev environment (creates .venv and installs all dev deps)
uv sync --extra dev

# Build the Rust extension and install in the current env
uv run maturin build --release --out dist
pip install dist/*.whl

# Run tests
uv run pytest tests/unit/ tests/integration/

# Run linter
uv run ruff check python/ tests/

# Run type checker
uv run mypy python/ferro_ta --ignore-missing-imports
```

## Git hooks and pre-push checks

Install the repo-managed git hooks after syncing your environment:

```bash
make hooks
```

That installs both the existing `pre-commit` hook and a `pre-push` hook that
runs the local CI gate before anything is pushed.

To run the same gate manually:

```bash
make prepush
```

To run only part of it while iterating:

```bash
make prepush CHECKS="version changelog python_lint"
```

The pre-push runner covers the basic required CI categories we can execute
locally: version/changelog checks, Rust fmt/clippy/core checks, Python
lint/typecheck/tests, docs, and WASM. It intentionally skips the multi-version
matrix, audit, and benchmark-regression jobs.

## Alternative: set up with plain pip

```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install maturin numpy pytest
maturin develop --release        # Build and install in editable mode
```

---

## Adding a New Candlestick Pattern

All candlestick patterns live in **`src/pattern/`** (Rust: `mod.rs` plus one `.rs` file per pattern) and **`python/ferro_ta/indicators/pattern.py`** (Python wrapper).

### Step 1 — Implement the Rust function

Add a new file `src/pattern/cdl_mypattern.rs` with your `#[pyfunction]`, or add the function to an existing pattern file. Register it in **`src/pattern/mod.rs`**. Open `src/pattern/mod.rs` to see how other patterns are declared and registered (e.g. `mod cdl_doji;` and `self::cdl_doji::cdl_doji` in `register()`). Then implement the logic in your new file (e.g. open `src/pattern/cdl_doji.rs` as a template) and add a new `#[pyfunction]` using the shared helper functions already available at the top of the file:

| Helper | Description |
|---|---|
| `body_size(open, close)` | Absolute body size |
| `upper_shadow(open, high, close)` | Upper shadow length |
| `lower_shadow(open, low, close)` | Lower shadow length |
| `candle_range(high, low)` | Full candle range (high − low) |
| `is_bullish(open, close)` | `true` when close ≥ open |
| `is_bearish(open, close)` | `true` when close < open |

**Template for a single-candle pattern** (save as `src/pattern/cdl_mypattern.rs` and add `mod cdl_mypattern;` plus the register call in `src/pattern/mod.rs`):

```rust
#[pyfunction]
pub fn cdl_mypattern<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let opens = open.as_slice()?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    let n = opens.len();
    if n != highs.len() || n != lows.len() || n != closes.len() {
        return Err(PyValueError::new_err("arrays must have the same length"));
    }
    let mut result = vec![0i32; n];
    for i in 0..n {
        let body  = body_size(opens[i], closes[i]);
        let range = candle_range(highs[i], lows[i]);
        let lower = lower_shadow(opens[i], lows[i], closes[i]);
        let upper = upper_shadow(opens[i], highs[i], closes[i]);

        // TODO: replace with your pattern conditions
        if range > 0.0 && /* pattern conditions */ {
            result[i] = 100;  // bullish  (use -100 for bearish)
        }
    }
    Ok(result.into_pyarray(py))
}
```

**Template for a multi-candle pattern** (adjust `i in K..n` for K-candle lookback):

```rust
#[pyfunction]
pub fn cdl_mypattern<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let opens  = open.as_slice()?;
    let highs  = high.as_slice()?;
    let lows   = low.as_slice()?;
    let closes = close.as_slice()?;
    let n = opens.len();
    if n != highs.len() || n != lows.len() || n != closes.len() {
        return Err(PyValueError::new_err("arrays must have the same length"));
    }
    let mut result = vec![0i32; n];
    for i in 2..n {  // 3-candle: use 2..n; 2-candle: use 1..n
        let (o1, h1, l1, c1) = (opens[i-2], highs[i-2], lows[i-2], closes[i-2]);
        let (o2, h2, l2, c2) = (opens[i-1], highs[i-1], lows[i-1], closes[i-1]);
        let (o3, h3, l3, c3) = (opens[i],   highs[i],   lows[i],   closes[i]  );

        // TODO: add your multi-candle conditions here
        if /* conditions */ {
            result[i] = 100;  // or -100
        }
    }
    Ok(result.into_pyarray(py))
}
```

### Step 2 — Register the function

In **`src/pattern/mod.rs`**, add `mod cdl_mypattern;` at the top with the other pattern modules, and in the `register()` function add `self::cdl_mypattern::cdl_mypattern` to the list of registered functions.

### Step 3 — Add the Python wrapper

Open `python/ferro_ta/indicators/pattern.py` and:

1. Import the Rust function at the top:
   ```python
   from ferro_ta._ferro_ta import cdl_mypattern as _cdl_mypattern
   ```

2. Add a typed Python wrapper:
   ```python
   def CDL_MYPATTERN(
       open: ArrayLike,
       high: ArrayLike,
       low: ArrayLike,
       close: ArrayLike,
   ) -> np.ndarray:
       """One-line summary.

       Parameters
       ----------
       open, high, low, close : array-like
           OHLC price arrays.

       Returns
       -------
       numpy.ndarray[int32]
           100 (bullish), -100 (bearish), or 0.
       """
       return _cdl_mypattern(_to_f64(open), _to_f64(high), _to_f64(low), _to_f64(close))
   ```

3. Add `"CDL_MYPATTERN"` to the `__all__` list.

### Step 4 — Export from the top-level package

Open `python/ferro_ta/__init__.py` and add an import (the canonical source is
`ferro_ta.indicators.pattern`; old flat path `ferro_ta.pattern` still works via
backward-compat stub):

```python
from ferro_ta.indicators.pattern import (  # noqa: F401
    # ... existing imports ...
    CDL_MYPATTERN,
)
```

Also add `"CDL_MYPATTERN"` to `__all__`.

### Step 5 — Write a test

Add a test class to `tests/unit/test_ferro_ta.py`:

```python
class TestCDLMyPattern:
    def test_output_values(self):
        result = CDL_MYPATTERN(OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE)
        assert len(result) == len(OHLCV_PRICES)
        assert all(v in (-100, 0, 100) for v in result)

    def test_detects_pattern(self):
        """Craft minimal OHLC data that must match the pattern."""
        o = np.array([...])
        h = np.array([...])
        l = np.array([...])
        c = np.array([...])
        result = CDL_MYPATTERN(o, h, l, c)
        assert result[-1] in (100, -100)
```

### Step 6 — Build and verify

```bash
maturin develop --release
pytest tests/unit/test_ferro_ta.py -v -k mypattern
```

---

## Adding Other Indicators

- **Overlap Studies** (MAs, bands): `src/overlap/` (e.g. `mod.rs`, `sma.rs`) + `python/ferro_ta/indicators/overlap.py`
- **Momentum Indicators**: `src/momentum/` + `python/ferro_ta/indicators/momentum.py`
- **Cycle Indicators**: `src/cycle/` + `python/ferro_ta/indicators/cycle.py`
- **Volatility / Volume / Statistics**: corresponding `src/*/` directories + `python/ferro_ta/indicators/*.py` files

Each module has a `pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()>` at the bottom — add your `wrap_pyfunction!` call there.

---

## Code Style

- Rust: follow `rustfmt` defaults (`cargo fmt`).
- Python: follow PEP 8; use Ruff for lint and format (`ruff check`, `ruff format`).
- Every public Rust function needs a docstring above the `#[pyfunction]` attribute.
- Every Python wrapper must have a NumPy-style docstring with `Parameters` and `Returns` sections.

## Validation and tests for new indicators

All new indicators **must**:

- Use the validation helpers in `ferro_ta.exceptions`: call `check_timeperiod()` for every period parameter and `check_equal_length()` for multi-array inputs (OHLCV) before calling the Rust extension. Wrap the Rust call in `try/except ValueError` and re-raise with `_normalize_rust_error(e)`.
- Have tests in `tests/unit/` (including at least one test for invalid parameters or edge cases where applicable).
- Update docstrings and type stubs (`python/ferro_ta/__init__.pyi`) when adding or changing the public API.

## Running the Full Test Suite

```bash
pytest tests/unit/ tests/integration/ -v
```

CI runs on Python 3.10–3.13 across Linux, macOS, and Windows.  Please make sure your change passes on all targets locally before opening a pull request.

## Pull Request Checklist

- [ ] Rust code compiles without warnings (`cargo build --release`)
- [ ] All existing tests still pass
- [ ] New test(s) cover the added function(s)
- [ ] Python wrapper and `__all__` updated
- [ ] `__init__.py` re-exports updated
- [ ] Docstrings present in both Rust and Python
- [ ] No vulnerable dependencies introduced (CI runs `cargo audit` and `pip-audit`; critical/high should be addressed)

---

## Architecture: Two-Layer Rust/Python Design

ferro-ta uses a **workspace** with two Rust crates:

| Crate | Path | Purpose |
|-------|------|---------|
| `ferro_ta` | `.` (root) | PyO3 `#[pyfunction]` wrappers — converts numpy ↔ `&[f64]`; builds the Python `.whl` |
| `ferro_ta_core` | `crates/ferro_ta_core/` | Pure Rust indicators — no PyO3 / numpy dependency |

When adding a new indicator:

1. Implement the algorithm in `crates/ferro_ta_core/src/<module>.rs` with a unit test.
2. Add a thin `#[pyfunction]` wrapper in `src/<module>.rs` (or the appropriate submodule under `src/<module>/`) that calls into the core.
3. Add the Python wrapper in `python/ferro_ta/indicators/<module>.py` (or the appropriate
   sub-package: `python/ferro_ta/data/`, `python/ferro_ta/analysis/`, `python/ferro_ta/tools/`).

```bash
# Build and test only the core (no Python required)
cargo build -p ferro_ta_core
cargo test -p ferro_ta_core
```

### Python sub-package layout

The `python/ferro_ta/` package is organized into sub-packages by concern.
Backward-compat stubs at the old flat paths (e.g. `ferro_ta.momentum`) re-export
from the new locations so existing code continues to work without changes.

```
python/ferro_ta/
├── __init__.py        # top-level re-exports and public API
├── core/              # Exceptions, configuration, registry, logging, raw FFI bindings
├── indicators/        # Technical indicators (momentum, overlap, volatility, volume,
│                      #   statistic, cycle, pattern, price_transform, math_ops, extended)
├── data/              # Streaming, batch, chunked, resampling, aggregation, adapters
├── analysis/          # Portfolio, backtest, regime, cross_asset, attribution,
│                      #   signals, features, crypto, options
├── tools/             # Visualisation, alerting, DSL, pipeline, workflow,
│                      #   api_info, GPU support
└── mcp/               # Model Context Protocol server
```

### Test directory layout

```
tests/
├── conftest.py        # shared fixtures (inherited by all sub-directories)
├── unit/              # pure unit tests and property-based tests
│   ├── test_ferro_ta.py
│   ├── test_coverage.py
│   ├── test_validation.py
│   ├── test_known_values.py
│   ├── test_property_based.py
│   ├── test_stages_*.py
│   └── test_math_ops_vs_numpy.py
├── integration/       # integration and comparison tests (vs TA-Lib, pandas-ta, ta)
│   ├── test_integration.py
│   ├── test_streaming_accuracy.py
│   ├── test_vs_talib.py
│   ├── test_vs_pandas_ta.py
│   └── test_vs_ta.py
└── benchmarks/        # benchmark tests are in top-level benchmarks/
```



The root crate (`src/`) is organized by TA-Lib category:

| Module | Path | Contents |
|--------|------|----------|
| `overlap` | `src/overlap/` | Overlap studies: SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, T3, BBANDS, MACD, SAR, MAMA, SAREXT, MACDEXT, MIDPOINT, MIDPRICE, MA, MAVP |
| `momentum` | `src/momentum/` | Momentum: RSI, MOM, ROC, WILLR, AROON, CCI, MFI, STOCH, ADX, TRIX, etc. |
| `pattern` | `src/pattern/` | Candlestick patterns: CDLDOJI, CDLENGULFING, CDLHAMMER, … |
| `cycle` | `src/cycle/` | Cycle: HT_TRENDLINE, HT_DCPERIOD, HT_PHASOR, HT_SINE, HT_TRENDMODE |
| `volatility` | `src/volatility/` | ATR, NATR, TRANGE |
| `volume` | `src/volume/` | AD, ADOSC, OBV |
| `statistic` | `src/statistic/` | STDDEV, VAR, LINEARREG, BETA, CORREL, … |
| `price_transform` | `src/price_transform/` | AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE |

Each module is a directory with `mod.rs` and one or more `.rs` files (e.g. `src/overlap/mod.rs`, `src/overlap/sma.rs`).

**Modular layout:** Pattern recognition is split into `src/pattern/mod.rs` plus one file per pattern (e.g. `src/pattern/cdl_doji.rs`, `src/pattern/cdl_engulfing.rs`). Overlap and momentum use a similar directory layout. To add a new pattern: add a new `src/pattern/cdl_*.rs` file with your `#[pyfunction]` and register it in `src/pattern/mod.rs`.

---

## Batch API

The `ferro_ta.batch` module provides `batch_sma`, `batch_ema`, and `batch_rsi`
(Rust 2-D implementations) plus the generic `batch_apply(data, fn, **kwargs)`.
For a new indicator that does not have a dedicated Rust batch function, use
`batch_apply(data, YOUR_INDICATOR)`.

---

## Running Rust Benchmarks

```bash
# Compile benchmarks only (fast, used in CI)
cargo bench --no-run

# Run benchmarks and get timings
cargo bench
```

Benchmarks are in `benches/indicators.rs` using [Criterion](https://github.com/bheisler/criterion.rs).

---

## Rust Coverage

```bash
# Install cargo-tarpaulin (one-time)
cargo install cargo-tarpaulin

# Collect coverage for the core crate
cargo tarpaulin -p ferro_ta_core --out Html

# Open htmlcov/index.html
```

---

## Type Checking (mypy)

```bash
# Install mypy (one-time)
pip install mypy numpy

# Run type checking
mypy python/ferro_ta --ignore-missing-imports

# No errors should be reported.
```

Type stubs live in `python/ferro_ta/__init__.pyi`. Update them whenever you add a new
public function.

---

## Release Process

See [RELEASE.md](RELEASE.md) for the full step-by-step release playbook and
[PACKAGING.md](PACKAGING.md) for conda-forge submission and feedstock maintenance. (version bump →
changelog → tag → CI builds wheels → publish to PyPI).

See [VERSIONING.md](VERSIONING.md) for the versioning policy (MAJOR/MINOR/PATCH rules,
supported Python version policy, and changelog maintenance requirements).

### Changelog requirement

Every PR that touches `src/`, `python/`, or `wasm/` **must** add an entry to the
`[Unreleased]` section of [CHANGELOG.md](CHANGELOG.md). Use the
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format
(`Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`).

### Version consistency

`Cargo.toml` and `pyproject.toml` must always carry the same `version` string.
CI enforces this with the `version-check` job — a PR that changes one but not the
other will fail CI.

### Dependency audits

CI runs **cargo audit** (Rust) and **pip-audit** (Python) in the `audit` job. PRs must
not introduce critical or high-severity vulnerabilities. If a dependency cannot be
updated immediately, document the accepted risk in the PR or in SECURITY.md. See
[SECURITY.md](SECURITY.md) for the full policy.

### Fuzzing (robustness)

Fuzz targets live in `fuzz/` (cargo-fuzz). To run fuzzing locally:

```bash
# Install cargo-fuzz (one-time)
cargo install cargo-fuzz

# Run the SMA fuzz target for 60 seconds
cargo fuzz run fuzz_sma -- -max_total_time=60

# Run the RSI fuzz target
cargo fuzz run fuzz_rsi -- -max_total_time=60
```

Any crash found by the fuzzer is saved to `fuzz/artifacts/<target>/`. Open a bug report
with the reproducing input and the panic message.


---

## Getting Help

If you have a question, found a bug, or want to suggest a new indicator:

- **GitHub Discussions** — For questions, ideas, and general discussion, use our [Discussions](https://github.com/pratikbhadane24/ferro-ta/discussions) space:
  - **Q&A** — Ask usage or API questions
  - **Ideas** — Propose new features or indicators
  - **Show & Tell** — Share strategies and projects built with ferro-ta
  - **Announcements** — Follow for release notes and important updates
- **GitHub Issues** — For confirmed bugs and actionable feature requests, open an [issue](https://github.com/pratikbhadane24/ferro-ta/issues).
- **Security issues** — See [SECURITY.md](SECURITY.md) for responsible disclosure instructions.
