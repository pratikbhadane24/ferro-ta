# Performance Guide

This document explains the performance characteristics of **ferro-ta** and gives
practical advice on how to get the best speed from the library.

---

## Quick Summary

| Use case                              | Recommended API                        | Notes                                  |
|---------------------------------------|----------------------------------------|----------------------------------------|
| Fast path — NumPy arrays              | Pass `np.ndarray` (float64, C-order)   | Zero overhead; no conversion needed   |
| pandas users                          | Pass `pd.Series`; result is `pd.Series`| Small overhead for index wrapping      |
| polars users                          | Pass `pl.Series`; result is `pl.Series`| Small overhead for type conversion     |
| Raw Rust access (expert)              | `from ferro_ta._ferro_ta import sma`     | Bypasses all Python wrappers           |
| Multiple series at once               | `batch_sma`, `batch_ema`, `batch_rsi`  | One Python call for all columns        |

**Recorded baseline and roadmap:** Performance roadmap and trade-offs are tracked
in [PERFORMANCE_ROADMAP.md](../PERFORMANCE_ROADMAP.md). For reproducible benchmark
inputs/results and methodology, use [benchmarks/README.md](../benchmarks/README.md)
and regenerate with `python benchmarks/bench_vs_talib.py --json benchmark_vs_talib.json`.

---

## The Rust Core Is Fast; Overhead Is in Python

The Rust extension (`_ferro_ta`) is compiled with full optimisations and is very
fast.  The bottlenecks for most users are in the Python wrapping layer:

1. **Array conversion** — `_to_f64` converts any array-like to a contiguous
   `float64` NumPy array.  If your input is already a C-contiguous `float64`
   ndarray the fast path returns it without any copy or allocation.

2. **pandas wrapping** — `pandas_wrap` extracts the NumPy array from a
   `pd.Series`, calls the Rust function, and wraps the result back into a
   `pd.Series` with the original index.  The wrapping itself is cheap but adds
   a small constant overhead per call.

3. **polars wrapping** — `polars_wrap` converts a `pl.Series` to NumPy and back.
   The result is now built from the NumPy buffer directly (`pl.Series(name,
   np.asarray(result))`), which avoids the O(n) `.tolist()` conversion of
   earlier versions.

4. **Batch** — `batch_sma`/`batch_ema`/`batch_rsi` use Rust-side batch functions
   for 2-D input (single GIL release for all columns).  The generic
   `batch_apply` runs any indicator in a Python loop over columns; use the
   dedicated batch functions when available.

---

## The Fast Path: Pass Contiguous float64 NumPy Arrays

The cheapest way to call any indicator is to pass a C-contiguous `float64`
NumPy array.  `_to_f64` detects this case and returns the array as-is:

```python
import numpy as np
from ferro_ta import SMA

# Already float64 and C-contiguous — _to_f64 is a no-op (zero copy)
close = np.random.rand(10_000).astype(np.float64)
result = SMA(close, timeperiod=20)
```

If your array is in a different dtype or order, `_to_f64` will create a new
array.  You can force the fast path once and reuse the result:

```python
close_f64 = np.ascontiguousarray(close, dtype=np.float64)  # one-time conversion
result = SMA(close_f64, timeperiod=20)  # no copy inside _to_f64
```

---

## Raw Numpy-Only API (No Wrapper Overhead)

If you want zero Python overhead — no pandas/polars wrapping, no validation —
you can import functions directly from the compiled extension:

```python
from ferro_ta._ferro_ta import sma, ema, rsi  # raw Rust functions

import numpy as np
close = np.random.rand(10_000).astype(np.float64)
result = sma(close, 20)   # returns a NumPy array (PyArray1<f64> from PyO3)
```

> **Warning:** The raw `_ferro_ta` API is internal and may change between
> versions.  It does *not* validate inputs — passing an empty array or a wrong
> type will raise an obscure error from PyO3.  Use it only if you have
> profiled a bottleneck and need the absolute minimum overhead.

For a stable raw API with the same functions, use the `ferro_ta.raw` submodule
(no pandas/polars wrapping or validation).

---

## pandas Series

```python
import pandas as pd
from ferro_ta import SMA

s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=pd.date_range("2024-01-01", periods=5))
result = SMA(s, timeperiod=3)
# result is a pd.Series with the same DatetimeIndex
```

Overhead compared to a raw numpy call: one `pd.Series.to_numpy()` call (cheap)
plus one `pd.Series(result, index=...)` call (cheap).  For large arrays this
is negligible; for very tight loops (millions of calls per second) prefer numpy.

---

## polars Series

```python
import polars as pl
from ferro_ta import SMA

s = pl.Series("close", [1.0, 2.0, 3.0, 4.0, 5.0])
result = SMA(s, timeperiod=3)
# result is a pl.Series named "close"
```

Overhead: one `.cast(Float64).to_numpy()` call plus one `pl.Series(name,
np.asarray(result))` call.  The result is built from the numpy buffer
(zero-copy where polars allows it) rather than going through `.tolist()`.

---

## Batch Execution

Use the batch API when you have many series (e.g., one column per symbol):

```python
import numpy as np
from ferro_ta.batch import batch_sma, batch_ema, batch_rsi, batch_apply

data = np.random.rand(252, 500).astype(np.float64)   # 252 bars × 500 symbols
sma_out = batch_sma(data, timeperiod=20)              # shape (252, 500)
rsi_out = batch_rsi(data, timeperiod=14)
```

`batch_apply` lets you run any indicator on a 2-D array:

```python
from ferro_ta import ATR
from ferro_ta.batch import batch_apply

ohlcv = np.random.rand(252, 100, 3).astype(np.float64)  # not directly supported
# For indicators that take multiple arrays use a manual loop instead
```

For 2-D input, `batch_sma`/`batch_ema`/`batch_rsi` use Rust-side batch
functions (single GIL release for all columns).  Use `batch_apply` for other
indicators that do not have a dedicated Rust batch implementation.

---

## Streaming (Bar-by-Bar)

```python
from ferro_ta.streaming import StreamingSMA

sma = StreamingSMA(period=20)
for bar in live_feed:
    value = sma.update(bar.close)
    if value is not None:
        print(f"SMA(20) = {value:.4f}")
```

The streaming classes are implemented in Rust (PyO3 `#[pyclass]` in
`_ferro_ta`) and re-exported from `ferro_ta.streaming`.  They are suitable for
live trading at typical bar rates with minimal Python overhead.

---

## Extended Indicators

`VWAP`, `SUPERTREND`, `ICHIMOKU`, `DONCHIAN`, `PIVOT_POINTS`, `KELTNER_CHANNELS`,
`HULL_MA`, `CHANDELIER_EXIT`, `VWMA`, and `CHOPPINESS_INDEX` are implemented in
Rust (`src/extended/mod.rs`).  The Python module `ferro_ta/extended.py` is a thin
wrapper with validation and `_to_f64`; all computation runs in the extension.

---

## Tips for Best Performance

1. **Pre-convert once.** If you call multiple indicators on the same array,
   convert it to `float64` + C-contiguous once:
   ```python
   close = np.ascontiguousarray(raw_close, dtype=np.float64)
   ```

2. **Avoid repeated dtype conversions.** Passing a `float32` or `int` array
   triggers a copy every call.

3. **Use batch functions for multiple symbols.** For SMA, EMA, and RSI use
   `batch_sma`/`batch_ema`/`batch_rsi` (Rust-side loop, single GIL release).
   The generic `batch_apply` runs a Python loop over columns; use it only for
   indicators that do not have a dedicated Rust batch.

4. **Avoid wrapping in very tight loops.** If you call an indicator millions
   of times per second (e.g., in a simulation) use the raw `_ferro_ta` API
   and manage conversion yourself.

5. **Profile before optimising.** Use `cProfile` or `py-spy` to find the
   actual bottleneck before assuming a particular layer is slow.

---

## Performance Improvements (implemented)

The following improvements are already in place. See
[docs/plans/2026-03-08-production-grade.md](plans/2026-03-08-production-grade.md)
for history and commits.

| Area        | Improvement                                                    | Where |
|-------------|----------------------------------------------------------------|-------|
| **Utils**   | `_to_f64` fast path: no copy for 1-D C-contiguous float64      | `python/ferro_ta/_utils.py` (lines 34–39) |
| **Utils**   | Polars result: `pl.Series(name, result)` from NumPy buffer (no `.tolist()`) | `python/ferro_ta/_utils.py` (e.g. 254–258) |
| **Raw API** | `ferro_ta.raw` — bypass pandas/polars and validation             | `python/ferro_ta/raw.py` |
| **Batch**   | Rust batch for SMA/EMA/RSI — single GIL release for 2-D        | `src/batch/mod.rs`, `python/ferro_ta/batch.py` |
| **Streaming** | All streaming classes in Rust (PyO3)                          | `src/streaming/mod.rs` |
| **Extended** | All extended indicators (incl. SUPERTREND) in Rust            | `src/extended/mod.rs`, `python/ferro_ta/extended.py` wraps Rust |

---

## Known Bottlenecks and Possible Improvements

Maintainer-facing list of slower paths and optional improvements. Update as
bottlenecks are fixed or deferred.

**Backtest** (`python/ferro_ta/backtest.py`):
- Equity with commission uses an O(n) Python loop (lines 374–380). Could
  vectorize (e.g. cumsum of commission events) or move to a small Rust helper.
- When both slippage and commission are used, `position_changed` is computed
  twice; compute once and reuse.
- Built-in strategies do redundant `np.asarray(..., dtype=np.float64)` if
  callers already pass contiguous float64; minor.

**Batch** (`python/ferro_ta/batch.py`):
- `batch_apply` runs a Python loop over columns (one Python call per column).
  Use `batch_sma`/`batch_ema`/`batch_rsi` when possible.
- No fast path for already 2-D C-contiguous float64 in batch_sma/ema/rsi
  (unlike `_to_f64` for 1-D); could avoid a potential copy.

**Options** (`python/ferro_ta/options.py`):
- `iv_rank`, `iv_percentile`, `iv_zscore` use Python loops over windows
  (O(n) iterations with per-window NumPy). Could move to Rust or vectorize.
  See also `docs/options-volatility.md`.

**Features** (`python/ferro_ta/features.py`):
- With `nan_policy="fill"` and no pandas, a Python loop fills NaN per column.
- Indicators are run in a Python loop (one call per indicator); no bulk API.

**Signals** (`python/ferro_ta/signals.py`):
- `compose(..., method="rank")` uses a list comprehension over columns (one
  Python round-trip per column). Could add a Rust batch rank for 2-D input.

**Other**:
- **dsl.py**: Some code paths use Python loops over bars.
- **gpu.py**: Fallback SMA/EMA/RSI use Python loops when GPU is not used.
- **tools.py / viz.py**: `.tolist()` for JSON/Plotly; acceptable for I/O.
- **Validation**: `check_equal_length`, `check_timeperiod` run in Python;
  cost is small; moving to Rust is deferred (see production-grade plan).
- **pandas_wrap / polars_wrap**: Per-call overhead; use `ferro_ta.raw` when
  minimising overhead.

---

## Benchmarking and comparison

For cross-library speed, run:
`pytest benchmarks/test_speed.py --benchmark-only --benchmark-json=benchmarks/results.json`.

To convert benchmark JSON into a markdown table:
`python benchmarks/benchmark_table.py`.

For focused TA-Lib comparison on the same data/parameters, run
`python benchmarks/bench_vs_talib.py` (requires `pip install ta-lib`).
Results are reported as speedup = TA-Lib time / ferro_ta time (values &gt; 1 mean
ferro_ta is faster). Speedup depends on indicator and data size.

---

## Related Documents

- [`docs/architecture.md`](architecture.md) — how the Rust/Python layers are
  organised and how they communicate.
- [`benchmarks/test_speed.py`](../benchmarks/test_speed.py) —
  Authoritative cross-library speed benchmarks (pytest-benchmark).
- [`benchmarks/benchmark_table.py`](../benchmarks/benchmark_table.py) —
  Render speed tables from `benchmarks/results.json`.
- [`crates/ferro_ta_core/benches/indicators.rs`](../crates/ferro_ta_core/benches/indicators.rs) —
  Rust Criterion benchmarks for the pure core (run with `cargo bench -p ferro_ta_core`).
- [`benchmarks/bench_vs_talib.py`](../benchmarks/bench_vs_talib.py) — speed comparison vs
  TA-Lib (same data and parameters); run with `python benchmarks/bench_vs_talib.py` (requires
  `ta-lib`). See README “Performance vs TA-Lib” for methodology and a comparison table.
- [`benchmarks/check_vs_talib_regression.py`](../benchmarks/check_vs_talib_regression.py) —
  CI guardrail script for detecting severe benchmark regressions from JSON artifacts.
