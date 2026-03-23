# ferro-ta WASM

WebAssembly bindings for the [ferro-ta](https://github.com/pratikbhadane24/ferro-ta) technical analysis library.

## Install from npm

Once published, install the Node.js build from npm:

```bash
npm install ferro-ta-wasm
```

```javascript
const { sma, ema, rsi, bbands, atr, obv, macd } = require('ferro-ta-wasm');

const close = new Float64Array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10]);
const smaOut = sma(close, 3);
console.log('SMA:', Array.from(smaOut));
```

> **Decision**: We chose WebAssembly (wasm-bindgen / wasm-pack) as the second binding because it runs in
> browsers *and* Node.js without any native addons, and shares zero unsafe FFI surface with the Python
> build.  Node.js users get a pure-JS entry point; browser users get the same `.wasm` file.

## Available Indicators

| Category   | Function      | Parameters                                         | Returns |
|------------|---------------|----------------------------------------------------|---------|
| Overlap    | `sma`         | `close: Float64Array, timeperiod: number`          | `Float64Array` |
| Overlap    | `ema`         | `close: Float64Array, timeperiod: number`          | `Float64Array` |
| Overlap    | `bbands`      | `close, timeperiod, nbdevup, nbdevdn`              | `Array[upper, middle, lower]` |
| Momentum   | `rsi`         | `close: Float64Array, timeperiod: number`          | `Float64Array` |
| Momentum   | `macd`        | `close, fastperiod, slowperiod, signalperiod`      | `Array[macd, signal, hist]` |
| Momentum   | `mom`         | `close: Float64Array, timeperiod: number`          | `Float64Array` |
| Momentum   | `stochf`      | `high, low, close, fastk_period, fastd_period`     | `Array[fastk, fastd]` |
| Volatility | `atr`         | `high, low, close: Float64Array, timeperiod`       | `Float64Array` |
| Volume     | `obv`         | `close: Float64Array, volume: Float64Array`        | `Float64Array` |

### Adding more indicators

All implementations are self-contained in `src/lib.rs` — no external crate dependency needed.
To add a new indicator:

1. Implement the algorithm in a `#[wasm_bindgen]` function in `src/lib.rs`.
2. Add at least two `#[wasm_bindgen_test]` tests covering output length and a known value.
3. Update this README table.
4. Run `wasm-pack test --node` to verify.

## Prerequisites

```bash
# Install Rust (if not already present)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
# OR via cargo:
cargo install wasm-pack
```

## Build

```bash
cd wasm/
wasm-pack build --target nodejs --out-dir pkg
```

This produces a `pkg/` directory containing:
- `ferro_ta_wasm.js` — JavaScript glue code
- `ferro_ta_wasm_bg.wasm` — compiled WebAssembly binary
- `ferro_ta_wasm.d.ts` — TypeScript declarations

For a browser build:

```bash
wasm-pack build --target web --out-dir pkg-web
```

## Usage (Node.js)

```javascript
const { sma, ema, rsi, bbands, atr, obv, macd } = require('./pkg/ferro_ta_wasm.js');

const close = new Float64Array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10]);

// Simple Moving Average (period 3)
const smaOut = sma(close, 3);
console.log('SMA:', Array.from(smaOut));  // [ NaN, NaN, 44.193, ... ]

// RSI (period 5)
const rsiOut = rsi(close, 5);
console.log('RSI:', Array.from(rsiOut));

// Bollinger Bands (period 5, ±2σ) — returns [upper, middle, lower]
const [upper, middle, lower] = bbands(close, 5, 2.0, 2.0);
console.log('BBANDS upper:', Array.from(upper));

// MACD (fast=3, slow=5, signal=2) — returns [macd_line, signal_line, histogram]
const [macdLine, signalLine, histogram] = macd(close, 3, 5, 2);
console.log('MACD:', Array.from(macdLine));
console.log('Signal:', Array.from(signalLine));
console.log('Histogram:', Array.from(histogram));

// ATR (period 3)
const high   = new Float64Array([45.0, 46.0, 47.0, 46.0, 45.0, 44.0, 45.0]);
const low    = new Float64Array([43.0, 44.0, 45.0, 44.0, 43.0, 42.0, 43.0]);
const atrOut = atr(high, low, close, 3);
console.log('ATR:', Array.from(atrOut));

// OBV
const volume  = new Float64Array([1000, 1200, 900, 1500, 800, 600, 700]);
const obvOut  = obv(close, volume);
console.log('OBV:', Array.from(obvOut));
```

## Usage (Browser)

```html
<script type="module">
  import init, { sma, macd } from './pkg-web/ferro_ta_wasm.js';
  await init();  // loads the .wasm binary

  const close = new Float64Array([44.34, 44.09, 44.15, 43.61, 44.33]);
  const smaOut = sma(close, 3);
  console.log('SMA:', Array.from(smaOut));

  // MACD
  const [macdLine, signal, hist] = macd(close, 3, 5, 2);
  console.log('MACD line:', Array.from(macdLine));
</script>
```

## Run Tests

```bash
cd wasm/
wasm-pack test --node
```

## CI Artifact

Every CI run on `main` builds the WASM package and uploads it as a GitHub Actions
artifact named `wasm-pkg`.  To download the latest pre-built package without building
from source:

1. Go to the [Actions tab](https://github.com/pratikbhadane24/ferro-ta/actions).
2. Open the latest successful CI run.
3. Download the `wasm-pkg` artifact from the **Artifacts** section.
4. Unzip and use `pkg/ferro_ta_wasm.js` directly in your project.

## Limitations

- Only 9 indicators are currently exposed (SMA, EMA, BBANDS, RSI, MACD, MOM, STOCHF, ATR, OBV).
  Additional indicators will be added following the same pattern in `src/lib.rs`.
- Large arrays (> 10M bars) may be slow due to JS↔WASM memory copies.  For high-throughput
  use cases prefer the Python (PyO3) binding.
- WASM does not support multi-threading natively in browsers (SharedArrayBuffer requires
  COOP/COEP headers).
