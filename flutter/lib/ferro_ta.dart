/// ferro-ta — fast technical-analysis indicators for Flutter.
///
/// Native platforms (Android, iOS, macOS, Windows, Linux) are powered by the
/// Rust `ferro_ta_core` crate through flutter_rust_bridge. Call [FerroTa.init]
/// once at startup, then use the generated indicator functions.
///
/// ```dart
/// import 'package:ferro_ta/ferro_ta.dart';
///
/// Future<void> main() async {
///   await FerroTa.init();
///   final out = await sma(close: myCloses, timeperiod: 30);
/// }
/// ```
///
/// For Flutter **web**, import `package:ferro_ta/ferro_ta_web.dart`, which
/// reuses the published `ferro-ta-wasm` package instead of native FFI.
library;

// Re-export the flutter_rust_bridge-generated indicator API. These files are
// produced by `flutter_rust_bridge_codegen generate` (`make flutter-gen`) and
// are not checked in until codegen has run.
export 'src/rust/api/indicators.dart';
export 'src/rust/frb_generated.dart' show RustLib;

import 'src/rust/frb_generated.dart';

/// Entry point for initializing the native ferro-ta runtime.
abstract final class FerroTa {
  /// Loads the native library and initializes the bridge. Call once before
  /// using any indicator function. Safe to await multiple times.
  static Future<void> init() => RustLib.init();
}
