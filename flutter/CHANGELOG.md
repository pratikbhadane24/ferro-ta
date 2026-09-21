# Changelog

All notable changes to the `ferro_ta` Flutter package are documented here. The
package version tracks the ferro-ta release version.

## 1.3.0

- Bound the full extended catalog: 51 additional indicators (adaptive moving
  averages and stops, momentum/volatility studies, volume indicators,
  oscillators, statistic/hybrid helpers) and signal utilities (`CROSSOVER`,
  `CROSSUNDER`, `HIGHEST`, `LOWEST`, `VALUEWHEN`, ...). Multi-output indicators
  return Dart tuples in the documented output order.
- Numerical fixes from the single-engine unification onto `ferro_ta_core`
  (CMO Wilder smoothing, no-lookahead Ichimoku Senkou, `OBV[0] = volume[0]`,
  `TRANGE[0] = NaN`, KAMA warmup). See the main CHANGELOG for details.

## 1.2.0

- Initial Flutter binding for ferro-ta via flutter_rust_bridge.
- 130+ technical-analysis indicators exposed over the pure-Rust `ferro_ta_core`
  crate (moving averages, RSI, MACD, Bollinger Bands, ADX, ATR, OBV/MFI,
  candlestick patterns, Hilbert-transform cycle functions, and more).
- Prebuilt native libraries bundled for Android (arm64-v8a, armeabi-v7a,
  x86_64), iOS, macOS (universal), Windows (x64), and Linux (x64).
- Web support reuses the published `ferro-ta-wasm` package via JS interop
  (`package:ferro_ta/ferro_ta_web.dart`).
