# SIMD acceleration

ferro-ta accelerates hot reductions with **runtime CPU-feature dispatch**
via the [`multiversion`](https://crates.io/crates/multiversion) crate. Each
dispatched function is compiled into several variants — baseline, SSE,
AVX2/FMA, AVX-512 on x86_64; NEON on aarch64 — and the fastest one the
**current** CPU supports is chosen at load time via CPUID.

## Why dispatch instead of `-C target-cpu`

A static `RUSTFLAGS=-C target-cpu=x86-64-v3` build *requires* AVX2 on the
running CPU; on an older chip it crashes with an illegal instruction
(SIGILL). Runtime dispatch instead ships every code path in one binary and
picks at runtime, so a single artifact:

- runs on **any** CPU of the target architecture (no SIGILL on pre-AVX2
  hardware), and
- still uses wide vector units where the hardware has them.

That property is what lets the **same** wheel / Docker image / crate run
across a heterogeneous fleet.

## When it helps

SIMD helps indicators whose inner loop is a reduction over contiguous
`f64` data — e.g. the initial window sum that seeds SMA, the `(T, S)` seed
for WMA, and similar fixed-window reductions. It does **not** help:

- The O(n) streaming recurrences (`window_sum += new - old`): each step
  depends on the previous one, so they are inherently sequential.
- Branchy inner loops (SAR, candlestick patterns).
- Streaming classes (a single-bar update is one or two ops).

The shared primitives live in `crates/ferro_ta_core/src/simd.rs`
(`sum`, `wma_seed`). They accumulate into independent lanes before a final
horizontal combine — that lane independence is what allows the optimizer to
vectorize each CPU-feature variant. A consequence is that results differ
from a strict left-to-right sum by a few ULPs, well inside every
indicator's documented tolerance.

## The `simd` feature

Dispatch is gated behind the `simd` Cargo feature, which is **on by
default**:

```bash
# default build — runtime dispatch enabled
cargo build -p ferro_ta_core --release

# pure-scalar build (debugging / baseline benchmarking)
cargo build -p ferro_ta_core --release --no-default-features
```

For Python, wheels published to PyPI are built with the default features,
so `pip install ferro-ta` ships the dispatched fast path with no action on
your part. To build a pure-scalar extension from source:

```bash
maturin develop --release --no-default-features
```

## Measured speedups

The nightly `benchmarks/bench_simd.py` job (see
`.github/workflows/nightly-bench.yml`) builds the extension twice — once
with `--no-default-features` (pure scalar) and once with `--features simd`
(dispatch) — and reports the per-indicator delta. Numbers are regenerated
on every run and vary with hardware; treat any table in a PR as a snapshot,
not a contract. The dispatched kernels here target correctness-preserving
reductions, so gains are modest on the sliding-window indicators and larger
on full-array reductions.

## Adding a SIMD-optimized indicator

1. Write and test the **scalar** implementation first — it is the ground
   truth.
2. If the hot path is a contiguous `f64` reduction, route it through a
   `crate::simd` primitive, or wrap a new helper in
   `#[multiversion::multiversion(targets = "simd")]` with the loop body
   accumulating into independent lanes.
3. Add a parity test comparing the dispatched result against the strict
   scalar reference within tolerance (see `simd.rs` tests for the pattern).
4. Benchmark scalar vs dispatch via `bench_simd.py`. Only keep the SIMD
   path if it wins — alignment and tail-handling overhead can make a naive
   vectorization *lose* to scalar.

## See also

- `crates/ferro_ta_core/src/simd.rs` — dispatched primitives and tests.
- `crates/ferro_ta_core/benches/indicators.rs` — criterion suite.
- `crates/ferro_ta_core/Cargo.toml` `[features] simd = ["dep:multiversion"]`
  — the gate (default-on).
