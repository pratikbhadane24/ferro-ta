# ferro_ta_core

`ferro_ta_core` is the pure Rust indicator engine behind [`ferro-ta`](https://github.com/pratikbhadane24/ferro-ta).

It provides allocation-friendly indicator functions over `&[f64]` slices without any
PyO3, NumPy, or Python runtime dependency, which makes it a good fit for:

- Rust-native technical analysis workloads
- custom services and backtesting engines
- non-Python bindings (WASM, FFI)

## Installation

```toml
[dependencies]
ferro_ta_core = "1.1.4"
```

## Design

- Pure functions over Rust slices
- No Python or NumPy dependency
- Shared core for the Python package and WASM bindings
- Output shape matches TA-Lib-style full-length series with `NaN` warm-up values where applicable

## Modules

| Module | Functions | Highlights |
|--------|-----------|------------|
| `overlap` | 20 | SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, T3, BBANDS, MACD, MACDFIX, MACDEXT, SAR, SAREXT, MAMA, MIDPOINT, MIDPRICE, MA, MAVP, Hull MA |
| `momentum` | 26 | RSI, MOM, STOCH, STOCHF, ADX, ADXR, DX, +DI, -DI, +DM, -DM, ROC, WILLR, AROON, CCI, BOP, STOCHRSI, APO, PPO, CMO, TRIX, ULTOSC |
| `volatility` | 3 | ATR, NATR, TRANGE |
| `volume` | 4 | OBV, MFI, AD, ADOSC |
| `pattern` | 61 | All TA-Lib candlestick patterns (CDL2CROWS through CDLXSIDEGAP3METHODS) |
| `statistic` | 9 | STDDEV, VAR, LINEARREG, LINEARREG_SLOPE/INTERCEPT/ANGLE, TSF, BETA, CORREL |
| `math` | 24 | Rolling SUM/MAX/MIN/MAXINDEX/MININDEX, element-wise ADD/SUB/MULT/DIV, 15 transforms (trig, exp, log, sqrt, ceil, floor) |
| `price_transform` | 4 | AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE |
| `cycle` | 7 | Hilbert Transform: TRENDLINE, DCPERIOD, DCPHASE, PHASOR, SINE, TRENDMODE |
| `extended` | 10 | VWAP, VWMA, Supertrend, Donchian, Keltner, Ichimoku, Pivot Points, Hull MA, Chandelier Exit, Choppiness Index |
| `streaming` | 9 | Stateful bar-by-bar: SMA, EMA, RSI, ATR, BBands, MACD, Stoch, VWAP, Supertrend |
| `batch` | 8 | Vectorized multi-column: batch_sma/ema/rsi/atr/stoch/adx, run_close/hlc_indicators |
| `backtest` | 19 | Signal generators, close-only and OHLCV engines, walk-forward, Monte Carlo, performance metrics |
| `options` | 18 | Black-Scholes/Black-76 pricing, Greeks, implied volatility, IV rank/percentile/zscore, smile metrics, chain analytics |
| `futures` | 14 | Basis, annualized basis, carry, roll (weighted/back-adjusted/ratio), curve analysis, synthetic forward/spot |
| `portfolio` | 10 | Beta, correlation matrix, drawdown, relative strength, spread, ratio, z-score, portfolio volatility |
| `signals` | 4 | Rank values, compose rank, top/bottom N indices |
| `alerts` | 3 | Threshold crossings, cross detection, alert bar collection |
| `regime` | 4 | ADX regime, combined regime, CUSUM breaks, variance breaks |
| `aggregation` | 3 | Tick bars, volume bars, time bars from trade data |
| `resampling` | 2 | Volume bars, OHLCV aggregation by label |
| `chunked` | 4 | Trim overlap, stitch chunks, make chunk ranges, forward fill NaN |
| `crypto` | 3 | Funding cumulative PnL, continuous bar labels, session boundaries |

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

The published Python package (`ferro-ta` on PyPI) wraps this crate with PyO3 bindings and adds NumPy conversion, pandas/polars wrappers, and higher-level Python tooling. The WASM package (`ferro-ta-wasm` on npm) also wraps this crate with full feature parity.

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
