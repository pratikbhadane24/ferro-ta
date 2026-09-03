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
│   ├── overlap/                 # Thin wrappers: SMA, EMA, WMA, DEMA, TEMA, …
│   ├── momentum/                # Thin wrappers: RSI, STOCH, ADX, CCI, …
│   ├── volatility/              # Thin wrappers: ATR, NATR, TRANGE
│   ├── volume/                  # Thin wrappers: AD, ADOSC, OBV
│   ├── statistic/               # Thin wrappers: STDDEV, VAR, LINEARREG, …
│   ├── price_transform/         # Thin wrappers: AVGPRICE, MEDPRICE, …
│   ├── pattern/                 # 61 CDL candlestick pattern heuristics
│   ├── cycle/                   # Hilbert-transform cycle heuristics
│   └── validation.rs            # Shared parameter validation helpers
│
├── crates/
│   └── ferro_ta_core/            # Single compute engine (pure Rust, no PyO3)
│       ├── src/                 # Indicators, streaming, options, backtest
│       └── benches/             # Rust criterion benchmarks
│
├── python/
│   └── ferro_ta/                 # Python package
│       ├── __init__.py          # Public API — re-exports + pandas/polars wraps
│       ├── _utils.py            # _to_f64, pandas_wrap, polars_wrap, get_ohlcv
│       ├── indicators/          # Thin wrappers around _ferro_ta functions
│       ├── analysis/            # Backtest, portfolio, options, signals
│       ├── data/                # Streaming, batch, resampling
│       ├── core/                # Config, registry, raw extension access
│       ├── tools/               # pipeline, GPU, alerts, DSL, viz, MCP
│       └── py.typed             # PEP 561 marker
│
├── fuzz/                        # cargo-fuzz targets (fuzz_sma, fuzz_rsi, …)
├── wasm/                        # wasm-pack / wasm-bindgen binding (uses ferro_ta_core)
├── benchmarks/                  # Python pytest-benchmark benchmarks
├── docs/                        # Sphinx documentation source
└── tests/                       # Python pytest test suite
```

---

## Single compute engine

ferro-ta has **two** Rust crates, but only **one** formula implementation:

```
Python ferro_ta / _ferro_ta          WASM / Flutter / fuzz
        │                                      │
        └── src/  (thin PyO3 wrappers)         │
                │                              │
                └── ferro_ta_core  ◄───────────┘
```

The third-party `ta` crate has been **removed**. Indicator math lives in
`ferro_ta_core`. The root crate converts numpy arrays to `&[f64]`, calls core,
and converts the result back. Pattern and Hilbert-cycle helpers stay in
`src/pattern/` and `src/cycle/` as local heuristics; they are not a second
formula engine for the overlap / momentum / volatility / statistic set.

### 1. Root crate (`src/`) — Python extension (`_ferro_ta`)

| Property       | Value                                             |
|----------------|---------------------------------------------------|
| Crate type     | `cdylib` (compiled to a `.so` / `.pyd` file)     |
| PyO3 / numpy   | Yes — depends on `pyo3` and `numpy`               |
| Depends on     | `ferro_ta_core` (no `ta` crate)                   |
| Used by        | Python extension (`ferro_ta._ferro_ta`)             |

Each category module (`src/overlap/`, `src/momentum/`, …) registers
`#[pyfunction]`s that accept `numpy` arrays (via `PyReadonlyArray1<f64>`),
delegate to `ferro_ta_core`, and return `Vec<f64>` which PyO3 converts to
an ndarray.

### 2. `crates/ferro_ta_core/` — Pure Rust library

| Property       | Value                                                             |
|----------------|-------------------------------------------------------------------|
| Crate type     | `lib` (not a Python extension)                                    |
| PyO3 / numpy   | No — pure Rust, no Python dependency                              |
| Depends on     | Optional `multiversion` (`simd`, default-on) and `serde`/`serde_json` |
| Used by        | Root PyO3 crate, `fuzz/` targets, `wasm/`, Flutter bridge         |

`ferro_ta_core` is the `&[f64]` API used by every binding. Numerical
parity is locked by core unit tests and
`tests/integration/test_core_py_parity.py` (Python public API vs those
goldens).

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
  │                             └── src/overlap/sma.rs  # thin PyO3 wrapper
  │                                       │
  │                                       └── ferro_ta_core::overlap::sma
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
| `extended.py` | Rust (`src/extended/` → core) | VWAP, SUPERTREND, ICHIMOKU, DONCHIAN, PIVOT_POINTS, …     |
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
