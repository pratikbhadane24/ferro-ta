/// Flutter **web** entry point for ferro-ta.
///
/// Flutter web cannot use the native FFI path, so this surface reuses the
/// already-published [`ferro-ta-wasm`](https://www.npmjs.com/package/ferro-ta-wasm)
/// binding via JS interop rather than shipping a second WebAssembly artifact.
///
/// ## Setup
/// Add the wasm glue as a web asset and load it before use. In `web/index.html`
/// (or via a bundler), expose the module on `globalThis.ferroTaWasm`, then:
///
/// ```dart
/// import 'package:ferro_ta/ferro_ta_web.dart';
///
/// final out = smaWeb(closes, 30); // Float64List -> Float64List
/// ```
///
/// The functions below mirror the `ferro-ta-wasm` export names 1:1. They cover
/// the common indicators; extend as needed — every wasm export follows the same
/// `Float64Array -> Float64Array` (or array-of-arrays) shape.
library ferro_ta_web;

import 'dart:js_interop';
import 'dart:typed_data';

/// The `ferro-ta-wasm` module, expected on `globalThis.ferroTaWasm`.
@JS('globalThis.ferroTaWasm')
external _FerroTaWasm get _wasm;

extension type _FerroTaWasm(JSObject _) implements JSObject {
  external JSFloat64Array sma(JSFloat64Array close, int timeperiod);
  external JSFloat64Array ema(JSFloat64Array close, int timeperiod);
  external JSFloat64Array wma(JSFloat64Array close, int timeperiod);
  external JSFloat64Array rsi(JSFloat64Array close, int timeperiod);
  external JSArray<JSFloat64Array> macd(
      JSFloat64Array close, int fast, int slow, int signal);
  external JSArray<JSFloat64Array> bbands(
      JSFloat64Array close, int timeperiod, double up, double down);
  external JSFloat64Array atr(JSFloat64Array high, JSFloat64Array low,
      JSFloat64Array close, int timeperiod);
  external JSFloat64Array obv(JSFloat64Array close, JSFloat64Array volume);
}

Float64List _out(JSFloat64Array a) => a.toDart;
JSFloat64Array _in(Float64List a) => a.toJS;

/// Simple Moving Average (web).
Float64List smaWeb(Float64List close, int timeperiod) =>
    _out(_wasm.sma(_in(close), timeperiod));

/// Exponential Moving Average (web).
Float64List emaWeb(Float64List close, int timeperiod) =>
    _out(_wasm.ema(_in(close), timeperiod));

/// Weighted Moving Average (web).
Float64List wmaWeb(Float64List close, int timeperiod) =>
    _out(_wasm.wma(_in(close), timeperiod));

/// Relative Strength Index (web).
Float64List rsiWeb(Float64List close, int timeperiod) =>
    _out(_wasm.rsi(_in(close), timeperiod));

/// MACD -> (macd, signal, hist) (web).
(Float64List, Float64List, Float64List) macdWeb(
    Float64List close, int fast, int slow, int signal) {
  final r = _wasm.macd(_in(close), fast, slow, signal).toDart;
  return (_out(r[0]), _out(r[1]), _out(r[2]));
}

/// Bollinger Bands -> (upper, middle, lower) (web).
(Float64List, Float64List, Float64List) bbandsWeb(
    Float64List close, int timeperiod, double up, double down) {
  final r = _wasm.bbands(_in(close), timeperiod, up, down).toDart;
  return (_out(r[0]), _out(r[1]), _out(r[2]));
}

/// Average True Range (web).
Float64List atrWeb(
        Float64List high, Float64List low, Float64List close, int timeperiod) =>
    _out(_wasm.atr(_in(high), _in(low), _in(close), timeperiod));

/// On-Balance Volume (web).
Float64List obvWeb(Float64List close, Float64List volume) =>
    _out(_wasm.obv(_in(close), _in(volume)));
