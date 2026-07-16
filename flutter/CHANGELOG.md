# Changelog

All notable changes to the `ferro_ta` Flutter package are documented here. The
package version tracks the ferro-ta release version.

## 1.2.0

- Initial Flutter binding for ferro-ta via flutter_rust_bridge.
- 130+ technical-analysis indicators exposed over the pure-Rust `ferro_ta_core`
  crate (moving averages, RSI, MACD, Bollinger Bands, ADX, ATR, OBV/MFI,
  candlestick patterns, Hilbert-transform cycle functions, and more).
- Prebuilt native libraries bundled for Android (arm64-v8a, armeabi-v7a,
  x86_64), iOS, macOS (universal), Windows (x64), and Linux (x64).
- Web support reuses the published `ferro-ta-wasm` package via JS interop
  (`package:ferro_ta/ferro_ta_web.dart`).
