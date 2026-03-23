/*!
ferro_ta_core — Pure Rust indicator library.

This crate contains all indicator implementations as pure functions operating
on `&[f64]` slices and returning `Vec<f64>`. It has **no dependency on PyO3
or numpy** so it can be used from any Rust project, or compiled to WASM /
Node.js via napi-rs without dragging in Python bindings.

The Python wheel (`ferro_ta` PyPI package) is built from a thin binding crate
that calls into this core and converts NumPy arrays to/from Rust slices.

# Two-layer architecture

The root crate (`ferro_ta`) contains PyO3 `#[pyfunction]` wrappers that convert
numpy arrays to `&[f64]` and delegate to this core crate.

# Usage (Rust)

```rust
use ferro_ta_core::overlap;

let close = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
let sma = overlap::sma(&close, 3);
assert!(sma[0].is_nan());
assert!((sma[2] - 2.0).abs() < 1e-10);
```
*/

pub mod futures;
pub mod math;
pub mod momentum;
pub mod options;
pub mod overlap;
pub mod statistic;
pub mod volatility;
pub mod volume;
