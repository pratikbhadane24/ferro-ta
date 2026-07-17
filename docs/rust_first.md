# Rust-First Architecture Policy

> **Rule:** All non-trivial computation and processing logic MUST be
> implemented in Rust and exposed to Python via PyO3.  Python is the
> **interface layer** only.

---

## Rationale

ferro-ta is built on the insight that Python is excellent as a glue layer
(validation, type dispatch, pandas/polars wrapping) but poor as a compute
engine (GIL, interpreter overhead, per-call allocation).  Every Python loop
over data is a performance regression.

This policy formalises what the codebase already does for standard TA-Lib
indicators and extends it to all new and existing indicators.

---

## The Boundary

There are three layers, not two. `src/` is a *binding* layer: it validates and
converts, then delegates to `ferro_ta_core`, which owns every algorithm and is
shared with the WASM and Flutter bindings. Implementing maths in `src/` forks it
away from those bindings.

```
Python layer (thin)               PyO3 bindings (thin)        Algorithms (thick)
────────────────────────────      ─────────────────────       ───────────────────────────────
ferro_ta/indicators/overlap.py ─▶ src/overlap/*.rs      ────▶ crates/ferro_ta_core/src/overlap.rs
ferro_ta/indicators/momentum.py ─▶ src/momentum/*.rs    ────▶ crates/ferro_ta_core/src/momentum.rs
ferro_ta/data/streaming.py     ─▶ src/streaming/mod.rs  ────▶ crates/ferro_ta_core/src/streaming.rs
ferro_ta/indicators/extended.py ─▶ src/extended/mod.rs  ────▶ crates/ferro_ta_core/src/extended.rs
ferro_ta/indicators/math_ops.py ─▶ src/math_ops/mod.rs  ────▶ crates/ferro_ta_core/src/math_ops.rs
ferro_ta/data/batch.py         ─▶ src/batch/mod.rs      ────▶ crates/ferro_ta_core/src/batch.rs
ferro_ta/indicators/pattern.py ─▶ src/pattern/*.rs      ────▶ crates/ferro_ta_core/src/pattern.rs
...                            ─▶ ...                   ────▶ ...

                                  wasm/src/lib.rs       ────▶ (same core)
                                  flutter/rust/src/...  ────▶ (same core)
```

**Python layer responsibilities (ONLY):**
- Input validation (`check_equal_length`, `check_timeperiod`)
- `_to_f64()` conversion (already has fast path for contiguous float64)
- pandas/polars wrapping (via `pandas_wrap` / `polars_wrap` decorators)
- Re-exporting and documentation

**Rust layer responsibilities (EVERYTHING ELSE):**
- All loops over data
- All rolling window computations
- All stateful streaming state machines
- All mathematical transformations applied bar-by-bar
- All batch operations

---

## Implementation Rules

### Rule 1: New indicators go in Rust first

When adding a new indicator:

1. Implement the algorithm in `crates/ferro_ta_core/src/<category>.rs` as a
   pure function over slices — never in `src/`. Core is shared with the WASM
   and Flutter bindings, so an algorithm written in the PyO3 layer is invisible
   to them and will drift.
2. Add a thin PyO3 wrapper in `src/<category>/<name>.rs` that validates inputs
   and delegates to core, then register it in `src/lib.rs` via
   `<category>::register(m)?`.
3. Write a thin Python wrapper in `python/ferro_ta/indicators/<category>.py` that:
   - Validates inputs
   - Calls `_to_f64()` on array arguments
   - Calls the Rust function
   - Wraps the result for pandas/polars if the output is a `np.ndarray`
4. Export from `python/ferro_ta/__init__.py` via the usual `__all__` +
   `pandas_wrap` / `polars_wrap` pattern.

**Do not write the algorithm in Python first and port it later.**  Porting is
expensive; getting it right in Rust first is cheaper.

### Rule 2: Porting Python algorithms to Rust

If you find a Python loop that iterates over data (e.g., `for i in range(n):`)
or a pure-Python rolling window computation, it is a porting candidate.
Priority order:
1. Hot paths called from batch or streaming contexts.
2. Any loop where `n` can be 10,000+.
3. Loops inside extended indicators.

When porting:
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

### Rule 4: Streaming classes are Rust PyO3 classes

Streaming (bar-by-bar stateful) classes **must** be `#[pyclass]` types
implemented in `src/streaming/mod.rs`.  Python should import and re-export
them — never re-implement them.

Template for a new streaming class:
```rust
#[pyclass(module = "ferro_ta._ferro_ta")]
pub struct StreamingMyIndicator {
    period: usize,
    // ... state fields
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

And in `python/ferro_ta/data/streaming.py`:
```python
from ferro_ta._ferro_ta import StreamingMyIndicator  # noqa: F401
```

### Rule 5: Batch operations are Rust functions

Batch functions that process multiple time-series at once must be implemented
in `src/batch/mod.rs`.  They accept 2-D numpy arrays and loop over columns
entirely in Rust (one GIL release covers all columns).

### Rule 6: Document the Rust location

Every Python wrapper docstring must note that the algorithm is in Rust:

```python
def MY_INDICATOR(close, timeperiod=14):
    """My Indicator.
    ...
    Notes
    -----
    Implemented in Rust — see ``src/my_category/my_indicator.rs``.
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
| `gpu.py` (PyTorch backend) | PyTorch is Python-native; Rust cannot talk to GPU without CUDA bindings |
| `backtest.py` helpers | High-level orchestration |

---

## Current Status (as of 2026-03-08)

| Module | Logic location |
|---|---|
| `overlap.py` | ✅ Rust (`src/overlap/`) |
| `momentum.py` | ✅ Rust (`src/momentum/`) |
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
| `gpu.py` | ⚠️ PyTorch (Python/CUDA/MPS — intentional, see above) |
| `pipeline.py` | ✅ Orchestration only (no indicator loops) |
| `config.py` | ✅ Configuration only |
| `backtest.py` | ✅ Orchestration only |

---

## Checklist for New Indicator PRs

- [ ] Algorithm implemented in `src/<category>/mod.rs`
- [ ] `cargo fmt --check` passes
- [ ] `cargo clippy --release -- -D warnings` passes
- [ ] Python wrapper is **thin** (validation + `_to_f64` + Rust call)
- [ ] No Python loops over data
- [ ] Docstring notes "Implemented in Rust"
- [ ] Registered in `src/lib.rs` and exported from `__init__.py`
- [ ] Tests added in `tests/`
