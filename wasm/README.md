# ferro-ta WASM

WebAssembly bindings for the [ferro-ta](https://github.com/pratikbhadane24/ferro-ta) technical analysis library. Full feature parity with the Python and Rust core packages.

## Install from npm

```bash
npm install ferro-ta-wasm
```

```javascript
const ferro = require('ferro-ta-wasm');

const close = new Float64Array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10]);
console.log('SMA:', Array.from(ferro.sma(close, 3)));
console.log('RSI:', Array.from(ferro.rsi(close, 14)));
```

## Available Indicators (200+ exports)

| Category | Functions | Examples |
|----------|-----------|----------|
| Overlap Studies (20) | Moving averages, bands, SAR | `sma`, `ema`, `wma`, `dema`, `tema`, `trima`, `kama`, `t3`, `bbands`, `macd`, `macdfix`, `macdext`, `sar`, `sarext`, `mama`, `midpoint`, `midprice`, `ma`, `mavp`, `hull_ma` |
| Momentum (26) | Oscillators, directional movement | `rsi`, `mom`, `stoch`, `stochf`, `adx`, `adxr`, `dx`, `plus_di`, `minus_di`, `roc`, `willr`, `aroon`, `aroonosc`, `cci`, `bop`, `stochrsi`, `apo`, `ppo`, `cmo`, `trix_indicator`, `ultosc` |
| Candlestick Patterns (61) | All TA-Lib patterns | `cdlhammer`, `cdlengulfing`, `cdldoji`, `cdlmorningstar`, `cdlshootingstar`, ... (all 61) |
| Volatility (3) | True range, ATR | `atr`, `natr`, `trange` |
| Volume (6) | On-balance volume, accumulation | `obv`, `mfi`, `vwap`, `vwma`, `ad`, `adosc` |
| Price Transforms (4) | Synthetic prices | `avgprice`, `medprice`, `typprice`, `wclprice` |
| Cycle / Hilbert (6) | Hilbert Transform suite | `ht_trendline`, `ht_dcperiod`, `ht_dcphase`, `ht_phasor`, `ht_sine`, `ht_trendmode` |
| Statistics (10) | Regression, correlation | `stddev`, `var`, `linearreg`, `linearreg_slope`, `linearreg_intercept`, `linearreg_angle`, `tsf`, `beta_rolling`, `correl` |
| Math (19) | Operators and transforms | `math_add`, `math_sub`, `math_mult`, `math_div`, `transform_sin`, `transform_cos`, `transform_exp`, `transform_sqrt`, ... |
| Extended (10) | Supertrend, channels, Ichimoku | `supertrend`, `donchian`, `keltner_channels`, `ichimoku`, `pivot_points`, `chandelier_exit`, `choppiness_index` |
| Streaming API (9 classes) | Bar-by-bar stateful | `WasmStreamingSMA`, `WasmStreamingEMA`, `WasmStreamingRSI`, `WasmStreamingATR`, `WasmStreamingBBands`, `WasmStreamingMACD`, `WasmStreamingStoch`, `WasmStreamingVWAP`, `WasmStreamingSupertrend` |
| Options (14) | Pricing, Greeks, IV | `black_scholes_price`, `black_76_price`, `black_scholes_greeks`, `implied_volatility`, `iv_rank`, `smile_metrics`, ... |
| Futures (12) | Basis, roll, curve | `futures_basis`, `annualized_basis`, `roll_yield`, `weighted_continuous`, `calendar_spreads`, `curve_summary`, ... |
| Backtesting (9) | Signal generation, engines | `backtest_core`, `backtest_ohlcv`, `rsi_threshold_signals`, `macd_crossover_signals`, `walk_forward_indices`, `monte_carlo_bootstrap`, ... |
| Alerts & Regime (7) | Signals and regime detection | `check_threshold`, `check_cross`, `regime_adx`, `regime_combined`, `detect_breaks_cusum` |
| Batch & Portfolio (9) | Multi-asset analytics | `batch_sma`, `batch_ema`, `batch_rsi`, `correlation_matrix`, `portfolio_volatility`, `drawdown_series` |
| Aggregation (8) | Tick/volume/time bars | `aggregate_tick_bars`, `aggregate_volume_bars_ticks`, `volume_bars`, `ohlcv_agg` |

## Prerequisites

```bash
# Install Rust (if not already present)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

## Build

```bash
cd wasm/

# Build both Node.js and web targets
npm run build

# Or build individually:
npm run build:node   # → node/
npm run build:web    # → web/
```

This produces two directories:
- `node/` -- CommonJS glue for Node.js (`require()`)
- `web/` -- ESM glue for browsers and web workers (`import`)

Both contain `ferro_ta_wasm.js`, `ferro_ta_wasm_bg.wasm`, and `ferro_ta_wasm.d.ts`.

## Usage (Node.js)

```javascript
const {
  sma, ema, rsi, bbands, macd, atr, adx, obv, mfi,
  cdlhammer, cdlengulfing,
  WasmStreamingSMA,
} = require('ferro-ta-wasm');

const close = new Float64Array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10]);
const high  = new Float64Array([45.0, 46.0, 47.0, 46.0, 45.0, 44.0, 45.0]);
const low   = new Float64Array([43.0, 44.0, 45.0, 44.0, 43.0, 42.0, 43.0]);

// Indicators
console.log('SMA:', Array.from(sma(close, 3)));
console.log('RSI:', Array.from(rsi(close, 5)));

// Multi-output
const [upper, middle, lower] = bbands(close, 5, 2.0, 2.0);
const [macdLine, signal, hist] = macd(close, 3, 5, 2);

// Streaming (bar-by-bar)
const stream = new WasmStreamingSMA(3);
for (const price of close) {
  console.log('streaming SMA:', stream.update(price));
}
```

## Usage (Browser)

```html
<script type="module">
  import init, { sma, rsi, macd } from './pkg-web/ferro_ta_wasm.js';
  await init();

  const close = new Float64Array([44.34, 44.09, 44.15, 43.61, 44.33]);
  console.log('SMA:', Array.from(sma(close, 3)));
</script>
```

## Run Tests

```bash
cd wasm/
wasm-pack test --node
```

## Limitations

- Large arrays (> 10M bars) may be slow due to JS-WASM memory copies. For high-throughput use cases prefer the Python (PyO3) binding.
- WASM does not support multi-threading natively in browsers (SharedArrayBuffer requires COOP/COEP headers).
- The npm package ships both Node.js (`require`) and browser/web worker (`import`) builds. Conditional exports in `package.json` select the right one automatically.

## License

MIT
