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
│   └── validation.rs            # Shared parameter validation helpers
│
├── crates/
│   └── ferro_ta_core/            # Pure-Rust library (no PyO3 / numpy)
│       ├── src/                 # Shared by the root crate, fuzz targets, and WASM binding
│       └── benches/             # Rust criterion benchmarks
│
├── python/
│   └── ferro_ta/                 # Python package
│       ├── __init__.py          # Public API — re-exports + pandas/polars wraps
│       ├── _utils.py            # _to_f64, pandas_wrap, polars_wrap, get_ohlcv
│       ├── indicators/          # Thin wrappers around _ferro_ta functions:
│       │                        #   overlap.py, momentum.py, volatility.py, volume.py,
│       │                        #   statistic.py, price_transform.py, pattern.py (61 CDL),
│       │                        #   cycle.py, math_ops.py, extended.py
│       ├── data/                # streaming.py, batch.py, chunked.py, resampling.py,
│       │                        #   aggregation.py, adapters.py
│       ├── core/                # config.py, registry.py, exceptions.py, raw.py,
│       │                        #   logging_utils.py
│       ├── analysis/            # backtest.py, signals.py, options.py, futures.py, …
│       ├── tools/               # pipeline.py, gpu.py, alerts.py, dsl.py, viz.py,
│       │                        #   workflow.py, tools.py, api_info.py, dashboard.py
│       ├── mcp/                 # Optional MCP server (python -m ferro_ta.mcp)
│       ├── utils.py             # Public re-export of get_ohlcv
│       └── py.typed             # PEP 561 marker
│
├── fuzz/                        # cargo-fuzz targets (fuzz_sma, fuzz_rsi, …)
├── wasm/                        # wasm-pack / wasm-bindgen binding (uses ferro_ta_core)
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
| Depends on     | `ferro_ta_core` (shared algorithms) and the `ta` crate |
| Used by        | Python extension (`ferro_ta._ferro_ta`)             |

Each category module (`src/overlap/`, `src/momentum/`, …) registers
`#[pyfunction]`s that accept `numpy` arrays (via `PyReadonlyArray1<f64>`)
and return `PyArray1<f64>` NumPy arrays.

### 2. `crates/ferro_ta_core/` — Pure Rust library

| Property       | Value                                                             |
|----------------|-------------------------------------------------------------------|
| Crate type     | `lib` (not a Python extension)                                    |
| PyO3 / numpy   | No — pure Rust, no Python dependency                              |
| Depends on     | Only optional crates: `multiversion` (`simd` feature, default-on) and `serde`/`serde_json` (`serde` feature) |
| Used by        | Root crate (PyO3 wrappers), `fuzz/` targets, and `wasm/` binding  |

`ferro_ta_core` provides the indicator categories with a `&[f64]` API,
making it usable from WASM and fuzz targets without pulling in PyO3 or numpy.

> **Note:** The root crate depends on `ferro_ta_core` and wraps its `&[f64]`
> API with PyO3 `#[pyfunction]`s, so the Python, WASM, and fuzz surfaces all
> share the same core implementations.

---

## Python Binding Flow

```
User code
  │
  ├── from ferro_ta import SMA            # __init__.py re-export
  │         │
  │         └── python/ferro_ta/indicators/overlap.py::SMA
  │                   │
  │                   ├── _utils._to_f64(close)      # convert to float64 ndarray
  │                   └── _ferro_ta.sma(arr, n)        # call Rust extension
  │                             │
  │                             └── src/overlap/sma.rs  # validates timeperiod, calls ferro_ta_core
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

Parameter validation (`timeperiod` range checks, equal-length checks) is done
in Rust inside the `#[pyfunction]`s via `src/validation.rs`, so callers using
the raw `_ferro_ta` extension directly also get clear errors.  The Python
wrappers handle array conversion (`_to_f64`) and normalise Rust errors into
`FerroTAError` subclasses.

---

## Related Documents

- [`docs/performance.md`](performance.md) — when to use raw numpy vs pandas/polars,
  how to avoid unnecessary conversion, batch performance notes.
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — development workflow, running tests,
  adding a new indicator.
- [`CHANGELOG.md`](../CHANGELOG.md) — version history.
