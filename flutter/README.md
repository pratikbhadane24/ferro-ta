# ferro_ta (Flutter)

Fast technical-analysis indicators for Flutter, powered by the Rust
[`ferro_ta_core`](https://crates.io/crates/ferro_ta_core) crate through
[flutter_rust_bridge](https://github.com/fzyzcjy/flutter_rust_bridge).

Native platforms load a prebuilt Rust library (no toolchain required by app
developers); Flutter web reuses the published `ferro-ta-wasm` package.

## Install

```yaml
dependencies:
  ferro_ta: ^1.2.0
```

## Usage (native: Android / iOS / macOS / Windows / Linux)

```dart
import 'dart:typed_data';
import 'package:ferro_ta/ferro_ta.dart';

Future<void> main() async {
  await FerroTa.init(); // load the native library once

  final close = Float64List.fromList([1, 2, 3, 4, 5, 6, 7]);
  final sma3 = await sma(close: close, timeperiod: 3);
  final (upper, middle, lower) =
      await bbands(close: close, timeperiod: 5, nbdevup: 2, nbdevdn: 2);
}
```

## Usage (web)

```dart
import 'dart:typed_data';
import 'package:ferro_ta/ferro_ta_web.dart';

// Reuses the ferro-ta-wasm module exposed on globalThis.ferroTaWasm.
final close = Float64List.fromList([1, 2, 3, 4, 5, 6, 7]);
final sma3 = smaWeb(close, 3);
```

## Supported platforms

| Platform | Architectures | Path |
|---|---|---|
| Android | arm64-v8a, armeabi-v7a, x86_64 | native FFI |
| iOS | arm64 (device + simulator), x86_64 (sim) | native FFI |
| macOS | universal (arm64 + x86_64) | native FFI |
| Windows | x64 | native FFI |
| Linux | x64 | native FFI |
| Web | — | `ferro-ta-wasm` (JS interop) |

## Indicator coverage

130+ indicators are generated automatically from the same source of truth as
the WASM binding (`scripts/build_flutter_bridge.py`), keeping the Python, WASM,
and Flutter surfaces in lockstep. A small set of struct-returning surfaces
(options greeks/pricing, backtest engines, crossover-signal indices, batch
array-of-array ops) require hand-written bridge wrappers and are on the
follow-up list — see `MANUAL_EXCLUDE` in the generator.

## Development

```sh
make flutter-gen   # regenerate api wrappers + flutter_rust_bridge glue
make flutter       # verify wrappers are fresh and compile against core
```

This package is developed in the [ferro-ta monorepo](https://github.com/pratikbhadane24/ferro-ta).
