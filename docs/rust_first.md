# Core-First Binding Policy

> **Rule:** All non-trivial computation lives in `ferro_ta_core`. Every
> language is a thin wrapper (FFI, wasm-bindgen, flutter_rust_bridge, …).
> Reimplementing indicators in a new language is out of scope.

Python is one interface layer — validation, type dispatch, pandas/polars
wrapping — not *the* interface layer. WASM and Flutter are peer bindings over
the same crate.

See [docs/languages/adding.rst](languages/adding.rst) for the new-language
checklist.

---

## Rationale

ferro-ta is built on the insight that host languages are excellent as glue
(validation, type dispatch, dataframe wrapping) but poor as a compute engine
(GIL, interpreter overhead, per-call allocation, JS↔WASM copies). Every host
loop over data is a performance regression *and* a second algorithm to keep
in sync.

This policy formalises what the codebase already does: algorithms in
`crates/ferro_ta_core`, marshalling in each binding.

---

## The Boundary

```
Language bindings (thin)                   ferro_ta_core (thick)
─────────────────────────────────          ────────────────────────────────
python/ferro_ta/*.py  + src/*.rs  ────▶   crates/ferro_ta_core/src/*.rs
wasm/src/lib.rs                   ────▶   same functions, &[f64] API
flutter/rust/src/api/*.rs         ────▶   generated wrappers, still core
```

**Binding responsibilities (ONLY):**
- Input validation and idiomatic errors
- Buffer conversion (`_to_f64()`, `Float64Array`, `Vec<f64>`, …)
- Host-library wrapping (pandas/polars, Dart `Float64List`)
- Re-exporting and documentation

**Core responsibilities (EVERYTHING ELSE):**
- All loops over data
- All rolling window computations
- All stateful streaming state machines
- All mathematical transformations applied bar-by-bar
- All batch operations

---

## Implementation Rules

### Rule 1: New indicators go in `ferro_ta_core` first

When adding a new indicator:

1. Implement the algorithm in `crates/ferro_ta_core/src/<category>.rs` (or a
   new module) with a unit test. Public API is `&[f64]` / `Vec<f64>` (or
   tuples), NaN warmup.
2. Add a thin PyO3 wrapper in `src/<category>/` that converts numpy slices
   and calls the core function. Register it in `src/lib.rs`.
3. Write a thin Python wrapper in `python/ferro_ta/<category>.py` that:
   - Validates inputs
   - Calls `_to_f64()` on array arguments
   - Calls the Rust function
   - Wraps the result for pandas/polars if the output is a `np.ndarray`
4. Export from `python/ferro_ta/__init__.py` via the usual `__all__` +
   `pandas_wrap` / `polars_wrap` pattern.
5. Expose the same symbol on WASM (`wasm/src/lib.rs`) and regenerate Flutter
   (`python3 scripts/build_flutter_bridge.py`) when the indicator is in the
   shared core set. Refresh `python3 scripts/build_api_manifest.py`.

**Do not write the algorithm in Python (or JS, or Dart) first and port it
later.** Porting is expensive; getting it right in core first is cheaper.

### Rule 2: Porting Python algorithms to Rust

If you find a Python loop that iterates over data (e.g., `for i in range(n):`)
or a pure-Python rolling window computation, it is a porting candidate.
Priority order:
1. Hot paths called from batch or streaming contexts.
2. Any loop where `n` can be 10,000+.
3. Loops inside extended indicators.

When porting:
- The algorithm lands in `ferro_ta_core`.
- The Python function becomes a thin wrapper that calls the Rust function.
- There is no Python fallback; the extension must be built. If the Rust call
  fails, the function is allowed to fail (no silent fallback to Python).

### Rule 3: No raw NumPy loops in indicator logic

The following patterns are **forbidden** in indicator implementation code:

```python
# ❌ Forbidden: Python loop over data
for i in range(n):
    result[i] = compute(data[i - period : i])

# ❌ Forbidden: nested Python loop in rolling window
for i in range(timeperiod - 1, n):
    result[i] = data[i + 1 - timeperiod : i + 1].max()
```

The following are **allowed** in Python wrappers only:
```python
# ✓ Allowed: vectorised NumPy (no loop)
result = np.cumsum(data)

# ✓ Allowed: scalar operations (no loop over n)
tp = (high + low + close) / 3.0
```

### Rule 4: Streaming classes are Rust types

Streaming (bar-by-bar stateful) classes **must** be implemented in
`ferro_ta_core` and exposed through each binding. Python uses `#[pyclass]`
wrappers in `src/streaming/mod.rs`. WASM uses `WasmStreaming*` types. Do not
re-implement the state machine in the host language.

Template for a new Python streaming class (after the core type exists):
```rust
#[pyclass(module = "ferro_ta._ferro_ta")]
pub struct StreamingMyIndicator {
    inner: ferro_ta_core::streaming::StreamingMyIndicator,
}

#[pymethods]
impl StreamingMyIndicator {
    #[new]
    pub fn new(period: usize) -> PyResult<Self> { ... }
    pub fn update(&mut self, value: f64) -> f64 { ... }
    pub fn reset(&mut self) { ... }
    #[getter]
    pub fn period(&self) -> usize { self.period }
}
```

Then in `src/streaming/mod.rs::register()`:
```rust
m.add_class::<StreamingMyIndicator>()?;
```

And in `python/ferro_ta/streaming.py`:
```python
from ferro_ta._ferro_ta import StreamingMyIndicator  # noqa: F401
```

### Rule 5: Batch operations are Rust functions

Batch functions that process multiple time-series at once must be implemented
in `ferro_ta_core` and wrapped from `src/batch/mod.rs`. They accept 2-D numpy
arrays and loop over columns entirely in Rust (one GIL release covers all
columns).

### Rule 6: Document the Rust location

Every Python wrapper docstring must note that the algorithm is in Rust:

```python
def MY_INDICATOR(close, timeperiod=14):
    """My Indicator.
    ...
    Notes
    -----
    Implemented in Rust — see ``crates/ferro_ta_core/src/my_category.rs``.
    """
```

---

## What Belongs in Python Only

Some things are **intentionally** in Python and should stay there:

| Thing | Why it stays in Python |
|---|---|
| `pandas_wrap` / `polars_wrap` decorators | Pandas/polars are Python libraries; zero-copy Rust wrappers are not practical here |
| `_to_f64` fast path check | One Python branch beats a PyO3 round-trip for the already-valid case |
| `check_equal_length`, `check_timeperiod` | Negligible overhead vs indicator computation; keeps Rust functions focused |
| `Pipeline`, `Config` | Orchestration logic — Python is appropriate |
| `gpu.py` (CuPy PoC) | CuPy is Python-native; Rust cannot talk to GPU without CUDA bindings |
| `backtest.py` helpers | High-level orchestration |

---

## Current Status (as of 2026-03-08)

| Module | Logic location |
|---|---|
| `overlap.py` | ✅ Rust (`ferro_ta_core` via `src/overlap/`) |
| `momentum.py` | ✅ Rust (`ferro_ta_core` via `src/momentum/`) |
| `volatility.py` | ✅ Rust (`src/volatility/`) |
| `statistic.py` | ✅ Rust (`src/statistic/`) |
| `volume.py` | ✅ Rust (`src/volume/`) |
| `price_transform.py` | ✅ Rust (`src/price_transform/`) |
| `pattern.py` | ✅ Rust (`src/pattern/`) |
| `cycle.py` | ✅ Rust (`src/cycle/`) |
| `batch.py` | ✅ Rust (`src/batch/`) |
| `streaming.py` | ✅ Rust (`src/streaming/`) — all 9 classes |
| `extended.py` | ✅ Rust (`src/extended/`) — all 10 indicators |
| `math_ops.py` (rolling) | ✅ Rust (`src/math_ops/`) — SUM/MAX/MIN/MAXINDEX/MININDEX |
| `math_ops.py` (element-wise) | ✅ NumPy wrappers (no loops — vectorised by NumPy's C core) |
| `gpu.py` | ⚠️ CuPy (Python/CUDA — intentional, see above) |
| `pipeline.py` | ✅ Orchestration only (no indicator loops) |
| `config.py` | ✅ Configuration only |
| `backtest.py` | ✅ Orchestration only |

---

## Checklist for New Indicator PRs

- [ ] Algorithm implemented in `crates/ferro_ta_core/src/<category>.rs`
- [ ] Thin PyO3 wrapper in `src/<category>/` calls core (no second algorithm)
- [ ] `cargo fmt --check` passes
- [ ] `cargo clippy --release -- -D warnings` passes
- [ ] Python wrapper is **thin** (validation + `_to_f64` + Rust call)
- [ ] No Python loops over data
- [ ] Docstring notes "Implemented in Rust" and points at core
- [ ] Registered in `src/lib.rs` and exported from `__init__.py`
- [ ] WASM export added (or documented skip) in `wasm/src/lib.rs`
- [ ] Flutter wrappers regenerated (`python3 scripts/build_flutter_bridge.py`) or skip listed
- [ ] `python3 scripts/build_api_manifest.py` refreshed
- [ ] Tests added in `tests/` (and core unit tests)
