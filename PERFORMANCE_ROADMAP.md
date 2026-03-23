# ferro-ta Performance Roadmap

## Goal: 100x Faster Than Tulipy for Every Indicator

This document tracks the path from current performance to the 100x target.

---

## Current State (10k bars, median µs)

| Indicator | ferro_ta | Tulipy | Ratio (tu/ft) | Status |
|-----------|--------:|-------:|:-------------:|--------|
| SMA | 186 | 84 | 0.45x | ❌ Tulipy faster |
| EMA | 90 | 89 | 0.99x | 🔄 Parity |
| RSI | 112 | 91 | 0.81x | 🔄 Near parity |
| MACD | 135 | 99 | 0.73x | 🔄 Near parity |
| BBANDS | 99 | 96 | 0.97x | 🔄 Parity |
| ATR | 113 | 103 | 0.91x | 🔄 Near parity |
| CCI | 147 | 126 | 0.86x | 🔄 Near parity |
| WILLR | 167 | 119 | 0.71x | 🔄 Near parity |
| OBV | 88 | 83 | 0.94x | 🔄 Parity |
| ADX | 165 | 126 | 0.76x | 🔄 Near parity |
| MFI | 111 | 122 | 1.10x | ✅ ferro_ta faster |
| STOCH | 176 | 144 | 0.82x | 🔄 Near parity |

**vs `ta` library** (Python loops): ferro_ta is already **150–350x faster** for slow indicators (ATR, CCI, ADX, MFI).

---

## Why ferro_ta Doesn't Beat Tulipy Yet

Both ferro_ta and Tulipy are Rust/C extensions processing 10,000 `f64` values. The bottlenecks are:

1. **FFI overhead dominates at 10k bars** — Python→Rust call overhead is ~50µs fixed cost
2. **Array allocation**: ferro_ta pads NaN values; Tulipy truncates (saves allocation)
3. **SIMD**: Tulipy's C code uses auto-vectorization; ferro_ta Rust needs explicit SIMD

---

## Optimization Plan

### Phase 1: Eliminate FFI Overhead (Target: 2x improvement)

**Problem**: Each Python call into Rust costs ~50µs regardless of array size.

**Solutions**:
- [ ] Batch API: `compute_many([("SMA", close, 20), ("EMA", close, 14)])` — single FFI call
- [ ] Buffer reuse: accept pre-allocated output arrays to avoid allocation round-trips
- [ ] NumPy zero-copy: use `PyReadonlyArray` in pyo3 to avoid copies on input

**Expected gain**: 2x for small arrays (<1k bars), 1.3x for 10k bars.

### Phase 2: SIMD Auto-Vectorization (Target: 3x improvement)

**Problem**: Rust scalar loops vs SIMD C in Tulipy.

**Solutions**:
- [ ] Use `std::simd` (portable SIMD) for rolling sum accumulation (SMA, WMA)
- [ ] Use `packed_simd2` for element-wise operations (ADD, SQRT, LOG10, price transforms)
- [ ] Enable `target-cpu=native` in `.cargo/config.toml` for AVX2/AVX-512

```toml
# .cargo/config.toml
[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-cpu=native"]
```

**Expected gain**: 3-5x for vectorizable indicators (SMA, WMA, price transforms, math ops).

### Phase 3: Algorithm-Level Optimizations (Target: 5-10x improvement)

#### SMA — O(n) running sum
Current: recomputes each window.  
Target: single-pass running sum (already done in Rust — verify SIMD path is hit).

#### BBANDS — Welford's algorithm
Current: compute mean, then variance in two passes.  
Target: Welford's online algorithm — single pass, better cache utilization.

#### ATR/ADX — Avoid redundant True Range calculations
Current: ATR → ADX each compute TR independently.  
Target: Compute TR once, share with ATR, NATR, +DI, -DI, ADX in a single pass.

#### MACD — Reuse EMA computations
Current: Compute fast EMA and slow EMA separately.  
Target: Single function computes both EMAs in one pass.

#### Candlestick Patterns — Batch lookup table
Current: Sequential condition checks per bar.  
Target: Pre-compute body/shadow ratios, vectorized pattern matching.

### Phase 4: Streaming Precomputation (Target: 100x for incremental updates)

For real-time systems that update one bar at a time:

- [ ] `StreamingSMA` already O(1) per update — document and benchmark vs batch
- [ ] `StreamingEMA` α * new + (1-α) * prev — single multiply + add
- [ ] `StreamingBBands` — use Welford's online variance
- [ ] `StreamingRSI` — Wilder's smoothing: single multiply per update

**At 100k bars, streaming 1 bar at a time is O(n) vs O(n) batch, but with near-zero latency per update.**

Benchmark: batch 100k bars vs 100k × streaming 1 bar:

```
ferro_ta batch SMA(100k):     ~1.8ms
ferro_ta streaming SMA(100k): ~0.5ms total (5µs per bar × 100k = too slow)
```

Streaming becomes 100x advantage when:
- You only need the latest value (no history needed)
- Input arrives one bar at a time (WebSocket price feed)

---

## Measurement Methodology

All benchmarks use:
- `pytest-benchmark` with `pedantic()` mode
- 5 iterations × 20 rounds × 2 warmup rounds
- Median timing (not mean) to exclude JIT warmup
- C-contiguous `float64` arrays
- 10,000 bars for main benchmarks, 100,000 for scaling tests

Machine: Apple M-series / Intel x86_64 (note: results vary significantly by CPU)

---

## Tracking Progress

Run `pytest benchmarks/test_speed.py --benchmark-only --benchmark-json=benchmarks/results.json`
and commit `results.json` to track regression over time.

---

## References

- [Tulipy source](https://github.com/cirla/tulipy) — C with auto-vectorization
- [Rust SIMD Guide](https://doc.rust-lang.org/std/simd/index.html)
- [pyo3 zero-copy arrays](https://pyo3.rs/v0.22.0/numpy)
