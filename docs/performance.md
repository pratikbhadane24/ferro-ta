# Performance Guide

This document explains the performance characteristics of **ferro-ta** and gives
practical advice on how to get the best speed from the library.

---

## How to read the numbers in this document

This guide deliberately contains almost no absolute timings. Absolute
milliseconds depend on the machine, the CPython build, the Rust toolchain and
the exact commit, and a pasted table goes stale the first time any of those
change — so this document states *shapes* (what is fast, what is not, and why)
and points at the machine-readable artifact that carries the measurement.

The sources of truth, in order of authority:

| Source | What it is |
|---|---|
| [`perf-contract/*.json`](../perf-contract) | The gate *inputs*: `manifest`, `simd`, `batch`, `streaming`, `indicator_latency`, `runtime_hotspots`. Each embeds git commit, CPU model, Python/rustc versions and dataset parameters. The **committed** copies are a snapshot from one developer machine and are *not* what gates a PR: `.github/workflows/ci-python.yml` regenerates this directory on the runner and checks the fresh output, so the checked-in numbers can (and do) fail the same check they pass in CI. Read them as a dated reference, not as thresholds. |
| [`benchmarks/artifacts/latest/*.json`](../benchmarks/artifacts) | Checked-in dated baselines, including the cross-library comparison artifacts. |
| [`benchmarks/README.md`](../benchmarks/README.md) | Methodology of record — datasets, harness settings, how to regenerate. |
| [`PERFORMANCE_ROADMAP.md`](../PERFORMANCE_ROADMAP.md) | Which optimizations have landed, which are open, and which were rejected. |

Where a number appears below it is dated and its artifact is named. If you need
a current figure, regenerate the artifact rather than trusting this page.

---

## Quick Summary

| Use case                              | Recommended API                        | Notes                                  |
|---------------------------------------|----------------------------------------|----------------------------------------|
| Fast path — NumPy arrays              | Pass `np.ndarray` (float64, C-order)   | Zero-copy input; no conversion needed  |
| pandas users                          | Pass `pd.Series`; result is `pd.Series`| Small constant overhead for index wrapping |
| polars users                          | Pass `pl.Series`; result is `pl.Series`| Small constant overhead for type conversion |
| Minimum wrapper overhead (expert)     | `from ferro_ta.core.raw import sma`    | Skips pandas/polars wrapping and validation |
| Many series, one indicator            | `batch_sma`, `batch_ema`, `batch_rsi`, `batch_atr`, `batch_adx` | One Python call for all columns — **measure before assuming it is faster**, see [Batch Execution](#batch-execution) |
| Many indicators, same arrays          | `compute_many`                         | One Rust crossing per input-shape family — wins on HLC bundles, near parity on close-only bundles |
| One bar at a time (live feed)         | `ferro_ta.streaming` classes           | Optimizes **per-update latency and O(1) state**, not total throughput |

---

## The Rust Core Is Fast; Overhead Is in Python

The Rust extension (`_ferro_ta`) is compiled with full optimisations
(`lto = true`, `codegen-units = 1` — see `[profile.release]` in
[`Cargo.toml`](../Cargo.toml)) and is very fast. For most users the remaining
cost is in the Python wrapping layer:

1. **Array conversion** — `_to_f64` converts any array-like to a contiguous
   `float64` NumPy array. If your input is already a 1-D C-contiguous
   `float64` ndarray the fast path returns it unchanged, with no copy
   (`python/ferro_ta/_utils.py`).

2. **pandas wrapping** — `pandas_wrap` extracts the NumPy array from a
   `pd.Series`, calls the Rust function, and wraps the result back into a
   `pd.Series` with the original index. Cheap, but a constant cost per call.

3. **polars wrapping** — `polars_wrap` converts a `pl.Series` via
   `.cast(pl.Float64).to_numpy()` and builds the result with
   `pl.Series(name, result)` directly from the NumPy buffer, avoiding the O(n)
   `.to_list()` conversion of earlier versions (that path survives only as an
   exception fallback).

4. **Boundary crossings** — every Python call into Rust has a fixed cost. On
   cheap kernels at small-to-medium sizes that fixed cost, not the arithmetic,
   is what you are measuring. This is why the batch and grouped APIs do not
   automatically win: see the two sections below.

---

## The Fast Path: Pass Contiguous float64 NumPy Arrays

The cheapest way to call any indicator is to pass a C-contiguous `float64`
NumPy array. `_to_f64` detects this case and returns the array as-is:

```python
import numpy as np
from ferro_ta import SMA

# Already float64 and C-contiguous — _to_f64 is a no-op (zero copy)
close = np.random.rand(10_000).astype(np.float64)
result = SMA(close, timeperiod=20)
```

If your array is in a different dtype or order, `_to_f64` will create a new
array. Force the fast path once and reuse the result:

```python
close_f64 = np.ascontiguousarray(close, dtype=np.float64)  # one-time conversion
result = SMA(close_f64, timeperiod=20)  # no copy inside _to_f64
```

---

## Raw Numpy-Only API (No Wrapper Overhead)

If you want minimum Python overhead — no pandas/polars wrapping, no
validation — use the `ferro_ta.core.raw` submodule:

```python
import numpy as np
from ferro_ta.core.raw import sma, ema, rsi

close = np.random.rand(10_000).astype(np.float64)
result = sma(close, 20)
```

The same functions are also importable straight from the compiled extension:

```python
from ferro_ta._ferro_ta import sma, ema, rsi  # internal
```

> **Warning:** `_ferro_ta` is an internal module and may change between
> versions. It does *not* validate inputs — passing an empty array or a wrong
> type raises an obscure error from PyO3. Prefer `ferro_ta.core.raw`, and use
> either only after profiling shows the wrapper is your bottleneck.

---

## pandas Series

```python
import pandas as pd
from ferro_ta import SMA

s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=pd.date_range("2024-01-01", periods=5))
result = SMA(s, timeperiod=3)
# result is a pd.Series with the same DatetimeIndex
```

Overhead compared to a raw numpy call: one `pd.Series.to_numpy()` plus one
`pd.Series(result, index=...)`. For large arrays this is negligible; for very
tight loops prefer numpy.

---

## polars Series

```python
import polars as pl
from ferro_ta import SMA

s = pl.Series("close", [1.0, 2.0, 3.0, 4.0, 5.0])
result = SMA(s, timeperiod=3)
# result is a pl.Series named "close"
```

Overhead: one `.cast(pl.Float64).to_numpy()` plus one `pl.Series(name, result)`
built from the NumPy buffer.

---

## Batch Execution

The batch API applies one indicator to every column of a 2-D
(`n_samples × n_series`) array in a single Python call, with the GIL released
for the whole traversal:

```python
import numpy as np
from ferro_ta.batch import batch_sma, batch_ema, batch_rsi, batch_apply

data = np.random.rand(252, 500).astype(np.float64)   # 252 bars × 500 symbols
sma_out = batch_sma(data, timeperiod=20)              # shape (252, 500)
rsi_out = batch_rsi(data, timeperiod=14)
```

`batch_sma`/`batch_ema`/`batch_rsi` take a `parallel: bool = True` argument
that spreads columns across Rayon threads. `ferro_ta.batch` additionally
exposes `batch_atr`, `batch_adx` and `batch_stoch` for HLC inputs; those three
are importable and benchmarked (`benchmarks/bench_batch.py`) but are not yet
listed in the module's `__all__`, so treat their signatures as less settled.

`batch_apply` runs any single-series indicator over the columns of a 2-D
array. It dispatches to the Rust batch kernels when the callable is `SMA`,
`EMA` or `RSI` and the only keyword is `timeperiod`; otherwise it falls back to
a Python per-column loop.

### Batching is not automatically a speedup

**This is the most commonly misread part of the library.** A batch call is one
Python crossing instead of *n*, but the per-column work is unchanged, and the
plain Python loop it replaces is already calling straight into Rust. At the
sizes the perf contract measures, batching can easily lose.

In the checked-in gate `perf-contract/batch.json` (32 series × 20 000 bars,
Apple M3 Max, generated 2026-03-24), `sequential_speedup_vs_loop` is between
**0.41 and 0.78** for SMA, RSI, ATR and ADX — i.e. the sequential batch call
was 1.3–2.4× *slower* than a Python `for` loop over the columns. The grouped
`close_bundle_3` case scored 0.43. A fresh run of
`benchmarks/bench_batch.py` on a later build moved several of those figures
above 1.0 for the *parallel* path while SMA and every *sequential* path stayed
below it, so the sign of the result depends on the build, the indicator, the
column count and whether Rayon is enabled.

Practical guidance:

- Use the batch API for the **API shape** — one call, one array in, one array
  out, no Python-level loop to maintain — not on the assumption that it is
  faster.
- If you use it for speed, **measure your own shape** with
  `python benchmarks/bench_batch.py --samples <n> --series <k> --json out.json`
  and compare `parallel_ms` / `sequential_ms` against `loop_ms` in the output.
- `parallel=True` needs enough per-column work to pay for thread hand-off. It
  helps most on the more expensive kernels (RSI, ATR, ADX) and least on the
  cheapest (SMA).

### Grouped multi-indicator calls

When you need several indicators over the same 1-D arrays, `compute_many`
groups them into one Rust crossing per input-shape family and returns a
**list** of results in the order requested:

```python
from ferro_ta.batch import compute_many

sma, ema, rsi = compute_many(
    [
        ("SMA", {"timeperiod": 10}),
        ("EMA", {"timeperiod": 12}),
        ("RSI", {"timeperiod": 14}),
    ],
    close=close,
)
```

Grouping applies to two families, defined in `python/ferro_ta/data/batch.py`:

- **close-only** — `SMA`, `EMA`, `RSI`, `STDDEV`, `VAR`, `LINEARREG`,
  `LINEARREG_SLOPE`, `LINEARREG_INTERCEPT`, `LINEARREG_ANGLE`, `TSF`
- **high/low/close** — `ATR`, `NATR`, `ADX`, `ADXR`, `CCI`, `WILLR`

Any spec outside those families, or carrying keywords other than
`timeperiod`, falls back to the normal registry path automatically, so
correctness never depends on grouping.

The measured effect is modest and family-dependent: the HLC bundle is the
consistent win, while the close-only bundle sits near parity or below it —
`perf-contract/runtime_hotspots.json` records `compute_many_close` at
**0.77** against its per-call reference (2026-03-24), and
`benchmarks/check_hotspot_regression.py` enforces a floor of only 0.85 for
that case. Group HLC bundles freely; do not expect close-only grouping to be a
speed lever.

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

The streaming classes are implemented in Rust
(`crates/ferro_ta_core/src/streaming.rs`, exposed as PyO3 `#[pyclass]`es in
`src/streaming/mod.rs`) and re-exported from `ferro_ta.streaming`.

### Streaming buys latency, not throughput

Streaming a whole series bar by bar is **slower in total** than one batch call
over the same series — substantially so. `perf-contract/streaming.json`
(20 000 bars, Apple M3 Max, generated 2026-03-24) reports
`stream_over_batch_ratio` **greater than 1 for every measured class**, ranging
from about **10× to 107×**, and a fresh run on a later build reproduces the
same range. Every `update()` call pays a Python→Rust crossing that the batch
path pays once.

What streaming actually gives you:

- **Per-update latency** — tens to low hundreds of nanoseconds per `update()`
  in the same artifact, which is what matters when the next bar has not
  arrived yet.
- **O(1) state** — no growing buffer, no recomputation over history.
- **Incremental correctness** — the value after *k* updates matches the batch
  value at index *k*.

So: use streaming for live feeds and for memory-bounded processing of
unbounded data. Do **not** use it to speed up work on a series you already
hold in memory — call the batch indicator instead.

Nine streaming classes exist; `perf-contract/streaming.json` currently
measures five of them (SMA, EMA, RSI, ATR, VWAP). See
`PERFORMANCE_ROADMAP.md` for the unmeasured remainder.

---

## Extended Indicators

The extended indicator set (`VWAP`, `SUPERTREND`, `ICHIMOKU`, `DONCHIAN`,
`KELTNER_CHANNELS`, `CHANDELIER_EXIT`, `CHOPPINESS_INDEX` and others) is
implemented in Rust under `crates/ferro_ta_core/src/extended/` and wrapped by
`src/extended/mod.rs`. The Python module
`python/ferro_ta/indicators/extended.py` is a thin layer of validation plus
`_to_f64`; all computation runs in the extension. For the authoritative,
current list read that module's `__all__` — the set is actively growing.

---

## Tips for Best Performance

1. **Pre-convert once.** If you call multiple indicators on the same array,
   convert it to `float64` + C-contiguous once:
   ```python
   close = np.ascontiguousarray(raw_close, dtype=np.float64)
   ```

2. **Avoid repeated dtype conversions.** Passing a `float32` or `int` array
   triggers a copy on every call.

3. **Don't reach for batch or grouped APIs expecting free speed.** Read
   [Batch Execution](#batch-execution) first, then measure your own shape.

4. **Don't stream data you already have.** Streaming is a latency and
   memory-footprint tool, not a throughput tool.

5. **Avoid wrapping in very tight loops.** If you call an indicator millions
   of times per second, use `ferro_ta.core.raw` and manage conversion
   yourself.

6. **Profile before optimising.** Use `cProfile` or `py-spy` to find the
   actual bottleneck before assuming a particular layer is slow.

7. **Use the perf-contract scripts for evidence.**
   `benchmarks/run_perf_contract.py` and
   `benchmarks/profile_runtime_hotspots.py` record timings with git, runtime
   and build metadata, so you can compare like with like across machines and
   commits.

---

## Backtesting Performance

ferro-ta's backtester is designed for the vectorized case: you already have a
signal array (or an OHLCV frame plus signals), and you want the equity curve,
the trade list and the summary statistics.

What that buys you, structurally:

- **The signal → equity loop is entirely in Rust.** `backtest_core` and
  `backtest_ohlcv_core` are single O(n) passes over the bars with the GIL
  released; there is no Python on the hot path. Commission and slippage are
  applied inside the same pass, not as a follow-up Python step
  (`python/ferro_ta/analysis/backtest.py`, `src/backtest/mod.rs`).
- **All 23 performance metrics come from one Rust call.**
  `compute_performance_metrics` computes the whole set in a single traversal,
  so asking for twenty-three statistics costs about what asking for one costs.
  (Note the corollary: against a NumPy snippet that computes only Sharpe and
  max drawdown, the full call is *slower* — the committed artifact records
  that ratio explicitly. You are buying the other twenty-one metrics.)
- **Monte Carlo and multi-asset runs are Rayon-parallel** with deterministic
  LCG seeding, again with the GIL released, so a bootstrap over many
  simulations scales across cores instead of serialising on the interpreter.
- **Walk-forward index generation is O(number of folds)**, not O(bars) — it
  produces slice boundaries, not copies.

When it is the right tool: signal-array backtests, parameter sweeps, bootstrap
confidence intervals, and multi-asset portfolio runs where the per-bar logic is
expressible as arrays. When it is not: path-dependent strategies that must
make a decision inside the bar loop in Python, or anything needing a broker
event model — for that, use a dedicated event-driven engine
(`python/ferro_ta/analysis/backtest.py` says as much in its module docstring).

**For numbers, read the artifact, not this page.**
`benchmarks/artifacts/latest/bench_backtest_results.json` carries per-size
timings for every path above, together with the machine, commit, Python and
Rust versions that produced them. Regenerate it with:

```bash
python benchmarks/bench_backtest.py --sizes 10000 100000 \
    --json /tmp/bench_backtest_results.json
```

Add `--skip-competitors` to run only the ferro-ta paths, and `--assets` /
`--sims` to change the multi-asset and Monte Carlo shapes. The absolute
figures move a lot between builds and machines — a fresh local run of the same
script disagreed with the committed artifact by more than 2× on several rows,
including flipping the sign of the multi-asset parallel-vs-loop comparison —
which is exactly why they are not reproduced here.

> **Removed on purpose:** earlier revisions of this page carried a table
> ranking ferro-ta's backtester against named third-party libraries. Most of
> those numbers were not reproducible from anything in this repository —
> `benchmarks/bench_backtest.py` has never contained an event-driven-library
> comparison, and its remaining optional competitor path covers a single
> vectorized library. Comparative numbers belong in the benchmark artifacts,
> where they carry their own provenance and get regenerated.

---

## Cross-Library Comparison

ferro-ta is fast, and on a clear majority of indicators it is the faster of the
two libraries the comparison harness measures — but it does not win
everywhere, and this page will not claim otherwise.

The dated, checked-in measurement is
`benchmarks/artifacts/latest/benchmark_vs_openalgo.json`. Its own `summary`
block, generated **2026-09-03** on an Apple M3 Max with CPython 3.13.5 over
synthetic OHLCV data, reads at **100 000 bars**:

- 69 indicator rows compared
- **49 wins, 7 ties, 13 losses** (71% win rate, 81% non-loss rate)
- **median speedup 1.36×**, ranging from 0.19× to 20.7×

At 10 000 bars the same artifact records 49 wins, 2 ties and 18 losses, median
1.26×. The artifact also lists, per size, exactly which indicators lose or tie
(`openalgo_wins_or_ties`) — consult that list rather than assuming.

Scope of that claim, stated plainly: **one comparison library, one machine, one
synthetic dataset, one commit.** It is not a ranking of the Python ecosystem,
and no measurement in this repository supports one.

The separate TA-Lib comparison
(`benchmarks/artifacts/latest/benchmark_vs_talib.json`) is likewise mixed;
`benchmarks/README.md` summarises it as *"often faster on selected
indicators," not "faster everywhere"* and that remains the accurate reading.

Regenerate either comparison with:

```bash
# TA-Lib head-to-head (requires `pip install ta-lib`)
python benchmarks/bench_vs_talib.py --sizes 10000 100000 \
    --json /tmp/benchmark_vs_talib.json

# Cross-library head-to-head
python benchmarks/bench_vs_openalgo.py --sizes 10000 100000 \
    --json /tmp/benchmark_vs_openalgo.json
```

> ⚠️ `bench_vs_openalgo.py` defaults `--json` to the **committed baseline**
> `benchmarks/artifacts/latest/benchmark_vs_openalgo.json`. Always pass an
> explicit `--json` (or `--no-json`) unless you intend to replace the
> baseline.

Speedups are reported as `reference_time / ferro_ta_time`, so values above 1
mean ferro-ta is faster. Results depend on the indicator and the data size.

---

## Benchmark Tooling

Machine-readable scripts, beyond the pytest benchmark table:

```bash
python benchmarks/bench_batch.py --samples 100000 --series 100 --json batch_benchmark.json
python benchmarks/bench_streaming.py --bars 100000 --json streaming_benchmark.json
python benchmarks/bench_backtest.py --json bench_backtest_results.json
python benchmarks/profile_runtime_hotspots.py --json runtime_hotspots.json
python benchmarks/check_hotspot_regression.py --input runtime_hotspots.json
python benchmarks/run_perf_contract.py --output-dir benchmarks/artifacts/latest --skip-simd --skip-talib
```

> ⚠️ **`benchmarks/bench_simd.py` rebuilds the extension.** It runs
> `maturin develop --release` twice — once with `--no-default-features` and
> once with the default features — so it replaces whatever build you currently
> have installed. `run_perf_contract.py` invokes it too unless you pass
> `--skip-simd`. Pass `--skip-simd` (and `--skip-talib`, which needs `ta-lib`
> installed) unless you specifically want those suites.

The full pytest benchmark suite needs the optional `benchmark` extra
(`pytest-benchmark` is not a default dependency):

```bash
pip install "ferro-ta[benchmark]"    # or: uv sync --extra benchmark
pytest benchmarks/test_speed.py --benchmark-only \
    --benchmark-json=benchmarks/results.json
python benchmarks/benchmark_table.py     # renders results.json as markdown
```

The WASM bindings ship with a Node benchmark:

```bash
cd wasm && npm run build:node   # wasm-pack build --target nodejs --out-dir node
node bench.js --json ../wasm_benchmark.json
```

---

## SIMD And Build Flags

**SIMD is on by default.** `simd` is a default feature of both the `ferro_ta`
extension crate and `ferro_ta_core` (see `[features]` in
[`Cargo.toml`](../Cargo.toml) and
[`crates/ferro_ta_core/Cargo.toml`](../crates/ferro_ta_core/Cargo.toml)), so
published wheels and a plain `maturin develop --release` already get the
vectorized reductions. There is nothing to opt into; passing
`--features simd` is a no-op.

The implementation uses `multiversion` to compile each reduction in
`crates/ferro_ta_core/src/simd.rs` into several CPU-feature-specific variants
and pick the widest one the running CPU supports, via CPUID. That is why the
crate does **not** ship a static `-C target-cpu`: one binary runs on any CPU
of the target architecture without illegal-instruction (SIGILL) crashes on
older chips. See [`docs/guides/simd.md`](guides/simd.md).

The levers that do exist:

```bash
# Default build — runtime-dispatched SIMD, portable
uv run maturin develop --release

# Pure-scalar baseline (opt OUT of SIMD)
uv run maturin develop --release --no-default-features

# Maximum tuning for the current machine only — NOT portable
RUSTFLAGS="-C target-cpu=native" uv run maturin develop --release
```

### Expect SIMD to be worth very little

`perf-contract/simd.json` (macOS arm64, CPython 3.12.11, generated
2026-03-23 on branch `feat/performace-1.0.2` — an orphan run, not refreshed by
CI, which passes `--skip-simd`) compares a `--no-default-features` build
against the default build across nine cases.
Every `speedup_simd_vs_portable` sits between **0.96 and 1.06** — i.e. inside
noise. This is structural, not a bug: SIMD only reaches *fixed-window
reductions*. The O(n) streaming recurrences (`window_sum += new - old`) are
serial dependency chains, and branchy kernels (SAR, candlestick patterns) do
not vectorize at all. Do not expect the `simd` feature to close a kernel gap,
and do not disable it either — it costs nothing to keep.

Shipping policy:

- Ship portable wheels on the default release profile (`lto = true`,
  `codegen-units = 1`, default features).
- Never put `target-cpu=native` in the committed cargo config; reserve it for
  developer workstations and private deploys, because those binaries are not
  portable across CPU families. `PERFORMANCE_ROADMAP.md` records this as an
  explicitly rejected approach.

---

## Performance Improvements (implemented)

The following are already in place. See [CHANGELOG.md](../CHANGELOG.md) for
history and commits, and `PERFORMANCE_ROADMAP.md` for what is still open.

| Area          | Improvement                                                    | Where |
|---------------|----------------------------------------------------------------|-------|
| **Utils**     | `_to_f64` fast path: no copy for 1-D C-contiguous float64      | `python/ferro_ta/_utils.py` |
| **Utils**     | Polars result built from the NumPy buffer, no `.to_list()`     | `python/ferro_ta/_utils.py` |
| **Raw API**   | `ferro_ta.core.raw` — bypasses pandas/polars wrapping and validation | `python/ferro_ta/core/raw.py` |
| **Batch**     | Rust 2-D kernels for SMA/EMA/RSI/ATR/ADX/STOCH, one GIL release per call, optional Rayon | `crates/ferro_ta_core/src/batch.rs`, `python/ferro_ta/data/batch.py` |
| **Grouping**  | `compute_many` — one Rust crossing per input-shape family      | `python/ferro_ta/data/batch.py` |
| **Streaming** | All streaming classes in Rust (PyO3)                           | `crates/ferro_ta_core/src/streaming.rs`, `src/streaming/mod.rs` |
| **Extended**  | All extended indicators in Rust                                | `crates/ferro_ta_core/src/extended/`, wrapped by `src/extended/mod.rs` |
| **Backtest**  | Signal→equity loop, commission/slippage and all 23 metrics in Rust; Rayon Monte Carlo | `crates/ferro_ta_core/src/backtest.rs`, `src/backtest/mod.rs` |
| **Options**   | `iv_rank` / `iv_percentile` / `iv_zscore` delegate to Rust     | `python/ferro_ta/analysis/options.py` |
| **SIMD**      | Runtime CPU-feature dispatch via `multiversion`, default-on    | `crates/ferro_ta_core/src/simd.rs` |

---

## Known Bottlenecks and Possible Improvements

Maintainer-facing list of slower paths. Update as items are fixed or deferred;
`PERFORMANCE_ROADMAP.md` is the fuller treatment.

**Batch and grouping** (`python/ferro_ta/data/batch.py`):
- Sequential 2-D batch calls are measurably slower than a Python loop over
  columns at contract sizes (`perf-contract/batch.json`,
  `sequential_speedup_vs_loop` 0.41–0.78 as of 2026-03-24). The parallel path
  is better on the expensive kernels but still loses on SMA.
- `compute_many` close-only grouping is below parity
  (`runtime_hotspots.json`, `compute_many_close` 0.77 as of 2026-03-24). The
  HLC bundle is the one grouped path that reliably wins.
- `batch_apply` falls back to a Python per-column loop for anything other than
  `SMA`/`EMA`/`RSI` with a bare `timeperiod`.

**Derivatives analytics** (`python/ferro_ta/analysis/options.py`):
- `iv_rank`, `iv_percentile` and `iv_zscore` delegate to Rust; the Python layer
  only broadcasts and shapes results.
- Model-based implied-volatility inversion is much faster than it was but
  remains more expensive than direct pricing or Greeks, because of the
  root-finding.

**Features** (`python/ferro_ta/analysis/features.py`):
- `nan_policy="fill"` is vectorized.
- `feature_matrix(...)` routes through `compute_many(...)`, and inherits its
  grouping economics — `runtime_hotspots.json` records it at 0.93 against the
  ungrouped reference (2026-03-24), i.e. near parity.

**Signals** (`python/ferro_ta/analysis/signals.py`):
- `compose(..., method="rank")` uses a one-call Rust rank-composition path
  (`compose_rank`), but the gain is moderate rather than dramatic. Measure
  before treating it as a major lever.

**Other**:
- `python/ferro_ta/tools/dsl.py` — some paths use Python loops over bars.
- `python/ferro_ta/tools/gpu.py` — CPU fallbacks for the EMA/RSI recurrences
  are Python loops when no GPU backend is in use. See
  [`docs/gpu-backend.md`](gpu-backend.md).
- `python/ferro_ta/tools/tools.py`, `python/ferro_ta/tools/viz.py` —
  `.tolist()` for JSON/Plotly output; acceptable for I/O.
- Validation (`check_equal_length`, `check_timeperiod` in
  `python/ferro_ta/core/exceptions.py`) runs in Python; cost is small and
  constant.
- `pandas_wrap` / `polars_wrap` — constant per-call overhead; use
  `ferro_ta.core.raw` when minimising it.

---

## Related Documents

- [`docs/architecture.md`](architecture.md) — how the Rust and Python layers
  are organised and how they communicate.
- [`docs/guides/simd.md`](guides/simd.md) — why runtime dispatch instead of
  `target-cpu`, and what SIMD does and does not reach.
- [`benchmarks/README.md`](../benchmarks/README.md) — methodology of record.
- [`PERFORMANCE_ROADMAP.md`](../PERFORMANCE_ROADMAP.md) — landed, open and
  rejected optimizations; the enforcement policy for each gate.
- [`benchmarks/test_speed.py`](../benchmarks/test_speed.py) — cross-library
  pytest-benchmark suite (needs the `benchmark` extra).
- [`benchmarks/benchmark_table.py`](../benchmarks/benchmark_table.py) — render
  speed tables from `benchmarks/results.json`.
- [`crates/ferro_ta_core/benches/indicators.rs`](../crates/ferro_ta_core/benches/indicators.rs)
  — Rust Criterion benchmarks for the pure core
  (`cargo bench -p ferro_ta_core`).
- [`benchmarks/check_hotspot_regression.py`](../benchmarks/check_hotspot_regression.py)
  and
  [`benchmarks/check_vs_talib_regression.py`](../benchmarks/check_vs_talib_regression.py)
  — CI guardrails that read the JSON artifacts.
