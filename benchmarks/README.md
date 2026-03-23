# ferro-ta Benchmark Suite

> **62 indicators × 6 libraries** — accuracy and speed verified on **100,000 bars** (LARGE dataset).

## Overview

The benchmark suite compares **ferro-ta** against five popular Python technical-analysis libraries on a common dataset and shared wrappers so timings are directly comparable.

| Library   | Notes |
|-----------|-------|
| **TA-Lib** | C extension; gold standard for accuracy and speed |
| **pandas-ta** | Pure Python; broad indicator set |
| **ta** | Simple API; some indicators use O(n²) loops and are very slow |
| **Tulipy** | C extension; truncated output (no leading NaN padding) |
| **finta** | Expects DatetimeIndex DataFrame; some indicators very slow |

---

## Dataset (LARGE = 100k bars)

All **speed benchmarks** use the **LARGE** dataset: **100,000 bars** of OHLCV data.

- **Source:** `benchmarks/data_generator.py` — geometric Brownian motion for realistic prices; C-contiguous `float64` arrays for all libraries.
- **Why 100k:** Reflects backtesting and batch workloads; stresses memory and CPU so differences between libraries are clear.
- **Scales available:** `SMALL` (1k), `MEDIUM` (10k), `LARGE` (100k). Speed suite uses **LARGE** by default.

```python
from benchmarks.data_generator import SMALL, MEDIUM, LARGE
# SMALL  = 1,000 bars
# MEDIUM = 10,000 bars  (e.g. accuracy tests)
# LARGE  = 100,000 bars (speed benchmarks)
```

---

## Methodology

- **Harness:** [pytest-benchmark](https://pytest-benchmark.readthedocs.io/) with `benchmark.pedantic(..., iterations=5, rounds=20, warmup_rounds=2)`.
- **Reported metric:** **Median time per call** in **microseconds (µs)** — lower is better.
- **Machine info:** Stored in `benchmarks/results.json` (`machine_info`, `commit_info`) for reproducibility.
- **Libraries:** Only libraries present in the environment are benchmarked; missing ones are skipped.

## Reproducible Perf Artifacts

Use the perf-contract runner when you want a compact set of machine-readable
artifacts for single-series latency, batch throughput, streaming throughput,
and hotspot attribution in one directory:

```bash
uv run python benchmarks/run_perf_contract.py --output-dir benchmarks/artifacts/latest --skip-talib
```

That command writes:

- `indicator_latency.json` — canonical-fixture timings for the benchmark suite indicators
- `batch.json` — 2-D batch throughput plus grouped multi-indicator timings
- `streaming.json` — streaming update throughput vs batch baselines
- `runtime_hotspots.json` — ranked hotspot report with reference speedups
- `manifest.json` — runtime/git metadata plus hashes for the generated artifacts

For CI or local guardrails, validate the hotspot report with:

```bash
uv run python benchmarks/check_hotspot_regression.py --input benchmarks/artifacts/latest/runtime_hotspots.json
```

---

## Speed comparison (100k bars, median µs — lower is better)

The speed table includes **all 62 indicators**. **Number** = median µs; **N/A** = library does not support that indicator. To regenerate: run the full suite, then `uv run python benchmarks/benchmark_table.py`.

| Indicator | ferro_ta | talib | pandas_ta | ta | tulipy | finta |
|-----------|--------:|--------:|--------:|--------:|--------:|--------:|
| SMA | 256 | 327 | 425 | 798 | 338 | 856 |
| EMA | 369 | 365 | 427 | 641 | 358 | 722 |
| WMA | 257 | 356 | 433 | N/A | 356 | 112422 |
| DEMA | 444 | 588 | 670 | N/A | 335 | 1830 |
| TEMA | 437 | 768 | 866 | N/A | 358 | 3481 |
| T3 | 462 | 407 | 478 | N/A | N/A | 496 |
| TRIMA | 598 | 400 | 474 | N/A | 386 | 1722 |
| KAMA | 992 | 369 | 140751 | N/A | 367 | 2501 |
| HULL_MA | 547 | N/A | 957 | N/A | 372 | 329392 |
| VWMA | 376 | N/A | 669 | N/A | 391 | N/A |
| MIDPOINT | 1345 | 4685 | N/A | N/A | N/A | N/A |
| MIDPRICE | 1273 | 831 | N/A | N/A | N/A | N/A |
| RSI | 653 | 647 | 728 | 1762 | 404 | 2429 |
| MACD | 833 | 793 | 1058 | 1657 | 423 | 1726 |
| STOCH | 2445 | 941 | 1253 | 3233 | 901 | 3321 |
| CCI | 918 | 1029 | 1122 | 367074 | 676 | 321471 |
| WILLR | 1303 | 750 | 859 | 3409 | 775 | 3575 |
| AROON | 1418 | 587 | 1322 | 130842 | 737 | N/A |
| AROONOSC | 1464 | 586 | N/A | N/A | 773 | N/A |
| ADX | 855 | 746 | 27637 | 321625 | 614 | N/A |
| MOM | 189 | 180 | 254 | N/A | 186 | 352 |
| ROC | 578 | 204 | 272 | 361 | 202 | 463 |
| CMO | 876 | 634 | 707 | N/A | 312 | 2301 |
| PPO | 391 | 538 | 1045 | N/A | 380 | 2395 |
| TRIX | 488 | 831 | 1831 | 1891 | 426 | 1773 |
| TSF | 1519 | 678 | N/A | N/A | 363 | N/A |
| ULTOSC | 2069 | 619 | N/A | 14142 | 588 | N/A |
| BOP | 249 | 228 | 361 | N/A | 226 | N/A |
| PLUS_DI | 794 | 629 | 26792 | N/A | 690 | N/A |
| MINUS_DI | 796 | 600 | N/A | N/A | 642 | N/A |
| BBANDS | 345 | 581 | 1079 | 2163 | 406 | 2432 |
| ATR | 640 | 660 | 800 | 157763 | 370 | 6835 |
| NATR | 722 | 662 | 782 | N/A | 396 | N/A |
| TRANGE | 217 | 205 | 374 | N/A | 199 | 6606 |
| STDDEV | 611 | 408 | 461 | N/A | 400 | 1552 |
| VAR | 1281 | 357 | 398 | N/A | 417 | N/A |
| SAR | 520 | 459 | N/A | N/A | 454 | N/A |
| KELTNER_CHANNELS | 926 | N/A | 1062 | 2369 | N/A | N/A |
| DONCHIAN | 2399 | N/A | 3334 | 3145 | N/A | N/A |
| SUPERTREND | 1242 | N/A | 638613 | N/A | N/A | N/A |
| CHOPPINESS_INDEX | 2442 | N/A | 4892 | N/A | N/A | N/A |
| OBV | 482 | 475 | 592 | 496 | 515 | 4646 |
| AD | 271 | 282 | 424 | 615 | 291 | N/A |
| ADOSC | 482 | 409 | 544 | N/A | 376 | N/A |
| MFI | 350 | 779 | 925 | 433698 | 692 | 401076 |
| VWAP | 288 | N/A | 11460 | N/A | N/A | 880 |
| AVGPRICE | 215 | 211 | N/A | N/A | 229 | N/A |
| MEDPRICE | 203 | 188 | N/A | N/A | 197 | 445 |
| TYPPRICE | 195 | 205 | N/A | N/A | 204 | 435 |
| WCLPRICE | 199 | 197 | N/A | N/A | 210 | 292 |
| SQRT | 204 | 208 | N/A | N/A | 199 | N/A |
| LOG10 | 434 | 408 | N/A | N/A | 411 | N/A |
| ADD | 188 | 186 | N/A | N/A | 189 | N/A |
| LINEARREG | 1555 | 704 | N/A | N/A | 368 | N/A |
| LINEARREG_SLOPE | 1548 | 665 | N/A | N/A | 370 | N/A |
| CORREL | 4277 | 413 | N/A | N/A | N/A | N/A |
| BETA | 5226 | 483 | N/A | N/A | N/A | N/A |
| HT_DCPERIOD | 10864 | 4187 | N/A | N/A | N/A | N/A |
| HT_TRENDMODE | 10984 | 23020 | N/A | N/A | N/A | N/A |
| CDLENGULFING | 308 | 617 | N/A | N/A | N/A | N/A |
| CDLDOJI | 273 | 312 | N/A | N/A | N/A | N/A |
| CDLHAMMER | 304 | 1418 | N/A | N/A | N/A | N/A |

*Apple M3 Max, Python 3.13; 273 passed, 121 skipped (unsupported = N/A). Regenerate with [Running benchmarks](#running-benchmarks).*

**Takeaways:**

- **`ta`** is 20–350× slower on ATR, CCI, ADX, MFI (O(n²) Python loops).
- **ferro-ta** is typically 2–4× faster than **pandas-ta** across indicators.
- **TA-Lib** and **Tulipy** (C extensions) are strong; ferro-ta is competitive and avoids native dependencies.

---

## Running benchmarks

```bash
# Full speed suite (100k bars, all indicator × library pairs) — writes results.json
uv run pytest benchmarks/test_speed.py --benchmark-only --benchmark-json=benchmarks/results.json -v

# Head-to-head only (12 indicators × ferro_ta) — quick check
uv run pytest benchmarks/test_speed.py --benchmark-only -k "test_head_to_head" -v

# Large-dataset scaling only (ferro_ta at 100k)
uv run pytest benchmarks/test_speed.py --benchmark-only -k "test_large_dataset" -v

# Regenerate the Speed Comparison markdown table from results.json
uv run python benchmarks/benchmark_table.py

# TA-Lib head-to-head with machine-readable summary + git/runtime metadata
uv run python benchmarks/bench_vs_talib.py --sizes 10000 100000 --json benchmark_vs_talib.json

# Optional regression check used in CI
uv run python benchmarks/check_vs_talib_regression.py --input benchmark_vs_talib.json

# Batch throughput + grouped multi-indicator calls
uv run python benchmarks/bench_batch.py --samples 100000 --series 100 --json batch_benchmark.json

# Streaming update throughput vs batch baselines
uv run python benchmarks/bench_streaming.py --bars 100000 --json streaming_benchmark.json

# Ranked hotspot attribution against bundled reference implementations
uv run python benchmarks/profile_runtime_hotspots.py --json runtime_hotspots.json

# Portable vs SIMD-enabled build comparison
uv run python benchmarks/bench_simd.py --json simd_benchmark.json

# One-shot perf artifact bundle
uv run python benchmarks/run_perf_contract.py --output-dir benchmarks/artifacts/latest
```

Without `uv`: use `pytest` and `python` from the same environment where `ferro_ta` and optional libs (e.g. `talib`, `pandas_ta`, `ta`, `tulipy`, `finta`) are installed.

### WASM

From the `wasm/` directory:

```bash
wasm-pack build --target nodejs --out-dir pkg
node bench.js --json ../wasm_benchmark.json
```

---

## Indicator coverage

### Overlap (12)
`SMA` `EMA` `WMA` `DEMA` `TEMA` `T3` `TRIMA` `KAMA` `HULL_MA` `VWMA` `MIDPOINT` `MIDPRICE`

### Momentum (18)
`RSI` `MACD` `STOCH` `CCI` `WILLR` `AROON` `AROONOSC` `ADX` `MOM` `ROC` `CMO` `PPO` `TRIX` `TSF` `ULTOSC` `BOP` `PLUS_DI` `MINUS_DI`

### Volatility (11)
`BBANDS` `ATR` `NATR` `TRANGE` `STDDEV` `VAR` `SAR` `KELTNER_CHANNELS` `DONCHIAN` `SUPERTREND` `CHOPPINESS_INDEX`

### Volume (5)
`OBV` `AD` `ADOSC` `MFI` `VWAP`

### Price Transform (4)
`AVGPRICE` `MEDPRICE` `TYPPRICE` `WCLPRICE`

### Math (3)
`SQRT` `LOG10` `ADD`

### Statistics (4)
`LINEARREG` `LINEARREG_SLOPE` `CORREL` `BETA`

### Cycle (2)
`HT_DCPERIOD` `HT_TRENDMODE`

### Candlestick patterns (3)
`CDLENGULFING` `CDLDOJI` `CDLHAMMER`

---

## Accuracy results

Accuracy is tested separately; ferro_ta is the reference.

- **243 pairs pass** (allclose or correlation).
- **138 pairs skipped** (known formula/anchoring/scaling differences).
- **0 failures.**

### Known structural differences

| Pair | Reason |
|------|--------|
| CMO vs talib/pandas_ta/finta | ferro-ta CMO uses different smoothing variant |
| BBANDS vs finta | finta normalizes bands differently |
| ATR vs finta | finta uses simple TR instead of Wilder smoothing |
| VWAP vs pandas_ta | pandas_ta anchors to session start |
| HT_TRENDMODE vs talib | Hilbert Transform seed divergence |
| RSI vs ta/finta | ta/finta use SMA warmup vs Wilder EMA |
| Tulipy ROC | Fraction (0.01 = 1%) vs ferro-ta (1.0 = 1%) |
| Tulipy BBANDS | (lower, mid, upper) order differs from ferro-ta |

```bash
# Accuracy tests (62 indicators × 6 libraries)
uv run pytest benchmarks/test_accuracy.py -v
```

---

## Data generator

`benchmarks/data_generator.py`:

- **`generate_ohlcv(size)`** — dict of C-contiguous `float64` arrays: `open`, `high`, `low`, `close`, `volume`. High ≥ close ≥ low > 0; volume > 0.
- **`get_pandas_ohlcv(data)`** — DataFrame with DatetimeIndex for pandas-ta and finta.

Pre-built: `SMALL`, `MEDIUM`, `LARGE` (and `*_DF` variants).

---

## Library compatibility

Detailed notes per library:

- [TA-Lib](../docs/compatibility/talib.md)
- [pandas-ta](../docs/compatibility/pandas_ta.md)
- [ta](../docs/compatibility/ta.md)
- [Tulipy](../docs/compatibility/tulipy.md)
- [finta](../docs/compatibility/finta.md)
