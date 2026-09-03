# ferro-ta Performance Roadmap

Maintainer-facing. This file records **what we are optimising for, what has
landed, what is still open, and which approaches were rejected and why**.

It deliberately holds **no measurement tables**. Numbers live in the generated
artifacts and in the maintained performance guide; duplicating them here is how
the previous version of this document went stale and started misreporting the
state of the codebase.

| Where | What it owns |
|---|---|
| [`docs/performance.md`](docs/performance.md) | User-facing guide: fast paths, which API to use, implemented improvements, known slow paths. |
| [`benchmarks/README.md`](benchmarks/README.md) | Methodology of record: datasets, harness, cross-library tables, how to regenerate. |
| `benchmarks/artifacts/latest/*.json`, `perf-contract/*.json` | The measurements themselves, with git/runtime/build metadata. |
| **This file** | Goal and acceptance rule, phase status, rejected approaches, open work, structural lessons. |

---

## Goal and acceptance rule

The concrete, machine-checkable target is the per-kernel acceptance gate in
[`benchmarks/check_vs_openalgo_regression.py`](benchmarks/check_vs_openalgo_regression.py):

> Run with `--require-all-wins`: **every** `(indicator, size)` row must be a
> strict win — `speedup > 1.05` — and every size's `openalgo_wins_or_ties`
> list must be empty.

The tie band is `0.95 … 1.05` (`TIE_EPSILON = 0.05` in both
`benchmarks/bench_vs_openalgo.py` and the checker), and **a tie counts as a
loss** for this gate. `speedup` is defined in the artifact metadata as
`comparison_median_us / ferro_ta_median_us` at 10k and 100k bars on
C-contiguous `float64`.

Without `--require-all-wins` the same script is a regression differ: it joins a
baseline artifact against a candidate on `(indicator, size)` and fails on
per-row slowdown beyond `--max-slowdown-pct` (default 25), on win→tie/loss
outcome transitions, and on rows disappearing from the bench.

The previous stated goal — "100x faster than every competitor for every
indicator" — was never measured, never gated, and is not pursued. It is gone.

**Current standing:** read `benchmarks/artifacts/latest/benchmark_vs_openalgo.json`
(`summary.by_size[]` has the win/tie/loss counts and the per-size
`openalgo_wins_or_ties` list). The checked-in artifact is a **dated
pre-optimization snapshot** taken on `feat/extended-indicators`; a large kernel
optimization pass is landing now and will move many rows. Do not quote it as
current truth — regenerate.

---

## What is enforced, and where

| Gate | Enforced | Where |
|---|---|---|
| Hotspot attribution policy | **Hard fail in CI** | `.github/workflows/ci-python.yml` regenerates `perf-contract/` via `benchmarks/run_perf_contract.py`, then runs `benchmarks/check_hotspot_regression.py` |
| Hotspot + TA-Lib regression | Advisory (`continue-on-error`) | `.github/workflows/nightly-bench.yml` |
| Per-kernel win rule | **Manual / release-time only** | `benchmarks/check_vs_openalgo_regression.py --require-all-wins` — not yet wired into any workflow |

Wiring the win rule into CI is open work (see below).

---

## Phase status

### Phase 1 — FFI overhead — **landed** (the win at small sizes is not)

- **Batch API** — `crates/ferro_ta_core/src/batch.rs` (`batch_sma`, `batch_ema`,
  `batch_rsi`, `batch_atr`, `batch_stoch`, `batch_adx`, plus the grouped
  `run_close_indicators` / `run_hlc_indicators`), exposed as PyO3 functions in
  `src/batch/mod.rs`, documented in `docs/batch.rst`, gated by
  `perf-contract/batch.json`.
- **Zero-copy NumPy input** — `PyReadonlyArray1` is the standard idiom across
  the binding layer (149 files under `src/` use it).
- **Still open:** the batch and grouped paths are *not currently faster than a
  Python loop* at the sizes the contract measures — see the
  `*_speedup_vs_loop` and `speedup_vs_separate` fields in
  `perf-contract/batch.json` and the `ffi_grouping` rows in
  `perf-contract/runtime_hotspots.json`. The API shipped; the payoff has not.
  Either find the crossover size and document it, or fix the grouped path.
- **Still open:** caller-supplied output buffers. The core has a `*_into`
  convention (`overlap::sma_into`, `rolling::sliding_min_max_into`) but it is
  crate-internal — there is no `PyReadwriteArray` anywhere in `src/`, so
  Python callers cannot reuse an output array across calls.

### Phase 2 — SIMD — **landed**

- **Runtime CPU-feature dispatch** via the `multiversion` crate in
  `crates/ferro_ta_core/src/simd.rs`, documented in `docs/guides/simd.md`.
  Behind the `simd` Cargo feature, which is **on by default**.
- **Benchmarked, not gated.** SIMD has a benchmark (`benchmarks/bench_simd.py`)
  but no enforced threshold anywhere: the nightly workflow writes
  `benchmarks/artifacts/nightly_simd.json` and asserts nothing against it, and
  no checker script reads `perf-contract/simd.json`. Treat both as recorded
  observations. Note also that `perf-contract/simd.json` is an **orphan** —
  generated 2026-03-23 on CPython 3.12.11 from branch `feat/performace-1.0.2`,
  while the other four artifacts in that directory are 2026-03-24 / CPython
  3.13.5 / `main`. CI regenerates `perf-contract/` with `--skip-simd`, so it is
  never refreshed and is legitimately absent from the manifest. Do not read it
  as current.
- The primitives it actually provides are five fixed-window reductions:
  `sum`, `wma_seed`, `abs_dev_sum` (CCI's mean-absolute-deviation numerator),
  `abs_diff_sum` (KAMA's volatility term), and `count_le` (percent-rank inner
  loop). Each has a scalar fallback under `#[cfg(not(feature = "simd"))]`.
- **Scope limit, by construction:** SIMD only reaches *fixed-window
  reductions*. The O(n) streaming recurrences (`window_sum += new - old`) are
  serial dependency chains, and branchy kernels (SAR, candlestick patterns)
  do not vectorize. `docs/guides/simd.md` states this; the measured effect in
  `perf-contract/simd.json` is correspondingly small. Do not expect SIMD to
  close a kernel gap on its own.

### Phase 3 — Algorithmic — partly landed

| Item | Status |
|---|---|
| SMA single-pass running sum | **Done** — `overlap::sma_into` is an incremental rolling sum seeded by `simd::sum`, with a 2-wide unrolled steady state. |
| BBANDS rolling Welford | **Done** — `overlap::bbands` uses a documented constant-window Welford update with an `m2 < 0.0` drift clamp. |
| MACD — both EMAs in one pass | **Done** — `overlap::macd` advances the fast and slow EMA in a single loop. |
| Shared true range across the ADX family | **Done for ADX** — `momentum::adx_inner` computes TR / +DM / −DM once and returns all six series (`adx_all` exposes it). |
| Shared true range across **ATR/NATR and ADX** | **Open** — `volatility::atr` still computes TR independently of `adx_inner`, and `volatility::natr` is a post-pass over `atr`. A caller wanting ATR + ADX pays for TR twice. |
| Candlestick patterns — precomputed body/shadow ratios | **Open** — `pattern.rs` recomputes `body_size` / `upper_shadow` / `lower_shadow` per bar inside every pattern. |

### Phase 4 — Streaming — **landed; the old framing was wrong**

Nine streaming classes exist in `crates/ferro_ta_core/src/streaming.rs`, are
exposed as PyO3 `#[pyclass]`es in `src/streaming/mod.rs`, and are re-exported
from `python/ferro_ta/data/streaming.py`: `StreamingSMA`, `StreamingEMA`,
`StreamingRSI`, `StreamingATR`, `StreamingBBands`, `StreamingMACD`,
`StreamingStoch`, `StreamingVWAP`, `StreamingSupertrend`.

`perf-contract/streaming.json` benchmarks five of them (SMA, EMA, RSI, ATR,
VWAP) and reports `stream_ns_per_update` alongside a batch baseline.

The old document claimed streaming was a "100x" win. It is not, and the
contract artifact says so: `stream_over_batch_ratio` is **greater than 1** for
every measured class — feeding n bars one at a time costs *more* total time
than one batch call over the same n. Streaming's value is **per-update
latency and O(1) state** for live feeds, not throughput. Benchmarking the four
unmeasured classes is open work.

---

## Rejected approaches

**`-C target-cpu=native` in `.cargo/config.toml` — rejected. Do not
reintroduce.** A static `target-cpu` *requires* the assumed features on the
running CPU; on an older chip the binary dies with an illegal instruction
(SIGILL). This is the explicit rationale in the module doc of
`crates/ferro_ta_core/src/simd.rs` and in `docs/guides/simd.md`: the crate
ships one artifact that runs on any CPU of the target architecture and picks
the widest supported variant at runtime via CPUID. `docs/performance.md`
records the shipping policy — portable wheels on the default release profile,
`target-cpu=native` reserved for developer workstations and private deploys.
An earlier version of this roadmap recommended adding it to the committed cargo
config, i.e. to shipped artifacts. That was a known-harmful recommendation.

**`packed_simd2` — unavailable.** Unmaintained; not a dependency and should not
become one.

**`std::simd` — unavailable.** Still unstable, and the crate builds on stable
Rust. `multiversion` plus `slice::as_chunks` lane-local accumulation gives the
same vectorization on stable — that is exactly what `simd.rs` does.

---

## Structural lessons from the current optimization pass

Non-obvious, cost us measurement time, and easy to get backwards:

1. **Asymptotics can lose to constants on the index gather.** A monotonic-deque
   sliding min/max built on `VecDeque<usize>` is O(n) but was measurably
   *slower per pass* than the naive O(n·p) window scan it replaced at typical
   periods: every front/back access pays ring-buffer mask arithmetic, and the
   pop predicate has to *gather* `real[j]` to recover the value behind a stored
   index. The fix was not to go back to the scan but to change the layout —
   `rolling.rs` keeps **parallel `Vec<f64>` values and `Vec<u32>` indices** with
   a `head` cursor, so the hot predicate reads `vals[len - 1]`: sequential, in
   L1, no gather. See the "Why not `VecDeque`" note in
   `crates/ferro_ta_core/src/rolling.rs`.

2. **`vec![f64::NAN; n]` is a real store pass; `vec![0.0; n]` is nearly free.**
   A NaN fill cannot be a lazily-mapped zero page, so it writes the whole
   array — `momentum::adx_inner` notes 4.8 MB of stores at n = 100 000 across
   its six outputs — which the kernel then overwrites almost entirely. Building
   with `Vec::with_capacity` + `resize(warmup, NAN)` + `push` removes the double
   write. **This applies to NaN warmups only.** `vec![0.0; n]` goes through
   `alloc_zeroed`, so applying the same "optimization" to a zero-seeded
   cumulative kernel replaces a free allocation with a real one — a regression.

3. **A divide inside a loop-carried recurrence is not hoistable by LLVM.**
   Wilder smoothing written as `(avg * (p - 1.0) + x) / p` keeps a ~14-cycle
   divide *in the serial dependency chain*; `p` is loop-invariant but
   `x / p → x * (1/p)` is a value-changing transform LLVM may not perform
   without fast-math. Hoisting it by hand (`inv_p = 1.0 / p`) takes the step
   from ~22 to ~12 cycles per bar and dominated RSI's runtime. The correctness
   argument — the recurrence is a contraction, so per-step rounding decays
   instead of accumulating — is in the module doc of
   `crates/ferro_ta_core/src/momentum.rs`, with a ulp test
   (`wilder_reciprocal_hoist_stays_within_one_ulp_per_step`).

---

## Tracking progress

The successor to the old "commit `benchmarks/results.json`" workflow is the
**perf-contract**: six machine-readable artifacts regenerated from a single
runner, plus checker scripts. `perf-contract/manifest.json` records
runtime/git/build metadata and SHA-256 hashes of the other five
(`indicator_latency`, `batch`, `streaming`, `runtime_hotspots`, and — when not
skipped — `simd`).

```bash
# Regenerate the committed contract artifacts (this is what CI runs)
uv run python benchmarks/run_perf_contract.py --output-dir perf-contract \
  --skip-talib --skip-simd \
  --batch-samples 20000 --batch-series 32 --streaming-bars 20000 \
  --price-bars 20000 --iv-bars 50000

# Hard gate: hotspot attribution policy
uv run python benchmarks/check_hotspot_regression.py \
  --input perf-contract/runtime_hotspots.json

# Acceptance gate: every benchmarked row a strict win at every size
uv run python benchmarks/check_vs_openalgo_regression.py \
  --baseline before.json --candidate after.json --require-all-wins
```

To produce the `candidate` artifact, run `benchmarks/bench_vs_openalgo.py`.

> **Careful:** that script's default `--json` path **is** the committed
> baseline at `benchmarks/artifacts/latest/benchmark_vs_openalgo.json`. Always
> pass an explicit `--json /tmp/…` (or `--no-json`) unless you intend to
> replace the baseline.

`benchmarks/results.json` and the cross-library table it feeds
(`benchmarks/benchmark_table.py`) are still committed and still regenerated
from `pytest benchmarks/test_speed.py --benchmark-only`, but they are the
broad survey, not the gate. See `benchmarks/README.md` for the full command
list, including `bench_batch.py`, `bench_streaming.py`, `bench_simd.py`, and
`profile_runtime_hotspots.py`.

Rust-level kernel work should also be checked with the Criterion benches in
`crates/ferro_ta_core/benches/` (`cargo bench -p ferro_ta_core`), which measure
the core without any Python or FFI in the way.

---

## Open work, collected

1. Wire `check_vs_openalgo_regression.py --require-all-wins` into a workflow.
2. Close the rows still outside the win band — see the per-size
   `openalgo_wins_or_ties` lists in a freshly regenerated artifact.
3. Make the batch and grouped-`compute_many` paths actually beat a Python loop,
   or document the crossover size honestly.
4. Expose caller-supplied output buffers (extend the `*_into` convention to the
   binding layer).
5. Share true range between `volatility::atr`/`natr` and `momentum::adx_inner`.
6. Precompute body/shadow ratios once per bar for the candlestick patterns.
7. Benchmark the four streaming classes the contract does not cover
   (`StreamingBBands`, `StreamingMACD`, `StreamingStoch`, `StreamingSupertrend`).
