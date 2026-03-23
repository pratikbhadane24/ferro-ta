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

```
Python layer (thin)                  Rust layer (thick)
─────────────────────────────        ────────────────────────────────────
ferro_ta/overlap.py          ────▶   src/overlap/mod.rs
ferro_ta/momentum.py         ────▶   src/momentum/mod.rs
ferro_ta/streaming.py        ────▶   src/streaming/mod.rs  (PyO3 classes)
ferro_ta/extended.py         ────▶   src/extended/mod.rs
ferro_ta/math_ops.py         ────▶   src/math_ops/mod.rs
ferro_ta/batch.py            ────▶   src/batch/mod.rs
ferro_ta/pattern.py          ────▶   src/pattern/mod.rs
...                         ────▶   ...
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

1. Implement the algorithm in `src/<category>/mod.rs` (or a new category
   module if the category does not exist).
2. Register the function in `src/lib.rs` via `<category>::register(m)?`.
3. Write a thin Python wrapper in `python/ferro_ta/<category>.py` that:
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

And in `python/ferro_ta/streaming.py`:
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
| `gpu.py` (CuPy PoC) | CuPy is Python-native; Rust cannot talk to GPU without CUDA bindings |
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
| `gpu.py` | ⚠️ CuPy (Python/CUDA — intentional, see above) |
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
