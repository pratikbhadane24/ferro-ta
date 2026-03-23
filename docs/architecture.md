# Architecture

This document describes the internal layout of **ferro-ta** — how the Rust and
Python layers are organised, how they communicate, and what each component is
responsible for.

---

## Repository Layout

```
ferro-ta/
├── src/                         # Root PyO3 crate (Python extension, _ferro_ta)
│   ├── lib.rs                   # Module registration — assembles all sub-modules
│   ├── overlap/                 # SMA, EMA, WMA, DEMA, TEMA, KAMA, BBANDS, …
│   ├── momentum/                # RSI, STOCH, ADX, CCI, AROON, WILLR, MFI, …
│   ├── volatility/              # ATR, NATR, TRANGE
│   ├── volume/                  # AD, ADOSC, OBV
│   ├── statistic/               # STDDEV, VAR, LINEARREG, BETA, CORREL, …
│   ├── price_transform/         # AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE
│   ├── pattern/                 # 61 CDL candlestick patterns
│   ├── cycle/                   # HT_TRENDLINE, HT_DCPERIOD, HT_DCPHASE, …
│   └── common.rs                # Shared helpers (Wilder smoothing, etc.)
│
├── crates/
│   └── ferro_ta_core/            # Pure-Rust library (no PyO3 / numpy)
│       └── src/                 # Used by fuzz targets and WASM binding
│
├── python/
│   └── ferro_ta/                 # Python package
│       ├── __init__.py          # Public API — re-exports + pandas/polars wraps
│       ├── _utils.py            # _to_f64, pandas_wrap, polars_wrap, get_ohlcv
│       ├── overlap.py           # Thin wrappers around _ferro_ta overlap functions
│       ├── momentum.py          # … momentum
│       ├── volatility.py        # … volatility
│       ├── volume.py            # … volume
│       ├── statistic.py         # … statistic
│       ├── price_transform.py   # … price_transform
│       ├── pattern.py           # … pattern (61 CDL functions)
│       ├── cycle.py             # … cycle
│       ├── math_ops.py          # ADD, SUB, MULT, DIV, SUM, MAX, MIN, math transforms
│       ├── extended.py          # Extended indicators (VWAP, SUPERTREND, ICHIMOKU, …)
│       ├── streaming.py         # Stateful streaming classes (StreamingSMA, …)
│       ├── batch.py             # Batch execution API (batch_sma, batch_ema, …)
│       ├── pipeline.py          # Pipeline / make_pipeline
│       ├── config.py            # set_default / Config
│       ├── registry.py          # Indicator registry (list_indicators, run)
│       ├── backtest.py          # Simple backtest helpers
│       ├── gpu.py               # CuPy-backed GPU PoC (SMA, EMA, RSI)
│       ├── exceptions.py        # FerroTAError, FerroTAValueError, FerroTAInputError
│       ├── utils.py             # Public re-export of get_ohlcv
│       └── py.typed             # PEP 561 marker
│
├── fuzz/                        # cargo-fuzz targets (fuzz_sma, fuzz_rsi, …)
├── wasm/                        # wasm-pack / wasm-bindgen binding (uses ferro_ta_core)
├── benches/                     # Rust criterion benchmarks
├── benchmarks/                  # Python pytest-benchmark benchmarks
├── docs/                        # Sphinx documentation source
└── tests/                       # Python pytest test suite
```

---

## Two Rust Crates

ferro-ta has **two** Rust crates that serve different purposes:

### 1. Root crate (`src/`) — Python extension (`_ferro_ta`)

| Property       | Value                                             |
|----------------|---------------------------------------------------|
| Crate type     | `cdylib` (compiled to a `.so` / `.pyd` file)     |
| PyO3 / numpy   | Yes — depends on `pyo3` and `numpy`               |
| Depends on     | `ta` crate (provides TA-Lib-compatible algorithms)|
| Used by        | Python extension (`ferro_ta._ferro_ta`)             |

Each category module (`src/overlap/`, `src/momentum/`, …) registers
`#[pyfunction]`s that accept `numpy` arrays (via `PyReadonlyArray1<f64>`)
and return `Vec<f64>` which PyO3 converts to a Python list/ndarray.

### 2. `crates/ferro_ta_core/` — Pure Rust library

| Property       | Value                                                             |
|----------------|-------------------------------------------------------------------|
| Crate type     | `lib` (not a Python extension)                                    |
| PyO3 / numpy   | No — pure Rust, no Python dependency                              |
| Depends on     | Nothing outside `std`                                             |
| Used by        | `fuzz/` targets and `wasm/` binding                               |

`ferro_ta_core` provides the same indicator categories with a `&[f64]` API,
making it usable from WASM and fuzz targets without pulling in PyO3 or numpy.

> **Note:** The root crate and `ferro_ta_core` are *independent* implementations.
> They are not merged by design — merging them would require careful testing of
> both the Python and WASM/fuzz surfaces.  If you want to share code, the
> recommended path is to make the root crate depend on `ferro_ta_core` and wrap
> its `&[f64]` API with PyO3 `#[pyfunction]`s; that is a future refactor.

---

## Python Binding Flow

```
User code
  │
  ├── from ferro_ta import SMA            # __init__.py re-export
  │         │
  │         └── python/ferro_ta/overlap.py::SMA
  │                   │
  │                   ├── _utils._to_f64(close)      # convert to float64 ndarray
  │                   ├── check_timeperiod(n)         # validate parameters
  │                   └── _ferro_ta.sma(arr, n)        # call Rust extension
  │                             │
  │                             └── src/overlap/sma.rs  # pure Rust computation
  │
  ├── SMA(pd.Series(...))                # pandas_wrap intercepts first
  │         │
  │         ├── extracts .to_numpy(dtype=float64)
  │         ├── calls SMA(ndarray)
  │         └── wraps result in pd.Series(result, index=original_index)
  │
  └── SMA(pl.Series(...))                # polars_wrap intercepts first
            │
            ├── extracts .cast(Float64).to_numpy()
            ├── calls SMA(ndarray)
            └── wraps result in pl.Series(name, np.asarray(result))
```

Both `pandas_wrap` and `polars_wrap` are applied to every public name in
`__init__.py` so the same function transparently handles numpy arrays,
pandas Series, and polars Series.

---

## Extended Indicators, Streaming, and Batch

| Module        | Implementation              | Notes                                                       |
|---------------|-----------------------------|-------------------------------------------------------------|
| `extended.py` | Rust (`src/extended/`)      | VWAP, SUPERTREND, ICHIMOKU, DONCHIAN, PIVOT_POINTS, …       |
| `streaming.py`| Rust re-export              | Stateful classes (StreamingSMA, StreamingEMA, …) from `_ferro_ta`; no Python fallback |
| `batch.py`    | Rust for 2-D SMA/EMA/RSI    | `batch_sma`, `batch_ema`, `batch_rsi` call Rust batch functions; `batch_apply` is a Python loop for other indicators |

Streaming and batch 2-D paths are implemented in Rust for maximum performance.
The generic `batch_apply` remains for indicators that do not have a dedicated
Rust batch implementation (see `docs/performance.md`).

---

## Packaging and Build

- **Build backend:** [maturin](https://www.maturin.rs/) — compiles the root
  crate and packages it alongside the Python source into a wheel.
- **`python-source = "python"`** in `pyproject.toml` tells maturin where the
  Python package lives.
- **`module-name = "ferro_ta._ferro_ta"`** tells maturin to place the compiled
  `.so` at `ferro_ta/_ferro_ta.so` inside the wheel.
- Wheels are built for Linux (manylinux), Windows, and macOS via CI on release.

---

## Where Validation Lives

Currently most validation (array length checks, `timeperiod` range checks) is
done in Python wrappers before the Rust call.  A future improvement is to move
these checks into the `#[pyfunction]`s so that callers using the raw
`_ferro_ta` extension directly also get clear errors.

---

## Related Documents

- [`docs/performance.md`](performance.md) — when to use raw numpy vs pandas/polars,
  how to avoid unnecessary conversion, batch performance notes.
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — development workflow, running tests,
  adding a new indicator.
- [`CHANGELOG.md`](../CHANGELOG.md) — version history.
