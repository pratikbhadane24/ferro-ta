# ferro_ta_core

`ferro_ta_core` is the pure Rust indicator engine behind [`ferro-ta`](https://github.com/pratikbhadane24/ferro-ta).

It provides allocation-friendly indicator functions over `&[f64]` slices without any
PyO3, NumPy, or Python runtime dependency, which makes it a good fit for:

- Rust-native technical analysis workloads
- custom services and backtesting engines
- future non-Python bindings such as WASM and other FFI layers

## Installation

```toml
[dependencies]
ferro_ta_core = "1.2.0"
```

## Design

- Pure functions over Rust slices
- No Python or NumPy dependency
- Shared core for the Python package and WASM bindings
- Output shape matches TA-Lib-style full-length series with `NaN` warm-up values where applicable

## Modules

- `overlap` - moving averages, MACD, Bollinger Bands
- `momentum` - RSI, MOM
- `volatility` - ATR, TRANGE
- `volume` - OBV
- `statistic` - STDDEV
- `math` - rolling SUM/MAX/MIN helpers

## Example

```rust
use ferro_ta_core::overlap;

fn main() {
    let close = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let sma = overlap::sma(&close, 3);

    assert!(sma[0].is_nan());
    assert!(sma[1].is_nan());
    assert!((sma[2] - 2.0).abs() < 1e-10);
}
```

## Relationship To `ferro-ta`

The published Python package:

- crate: `ferro_ta`
- PyPI package: `ferro-ta`

wraps this crate with PyO3 bindings and adds:

- NumPy conversion
- pandas/polars wrappers
- streaming classes
- batch helpers
- higher-level Python tooling

If you only need Rust indicator functions, use `ferro_ta_core` directly.

## Development

From the repository root:

```bash
cargo build -p ferro_ta_core
cargo test -p ferro_ta_core
cargo bench -p ferro_ta_core --no-run
```

## License

MIT
