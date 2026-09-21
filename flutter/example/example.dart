// Minimal ferro_ta usage on a native platform (Android, iOS, macOS, Windows,
// Linux). For Flutter web, see `package:ferro_ta/ferro_ta_web.dart`.
import 'dart:typed_data';

import 'package:ferro_ta/ferro_ta.dart';

Future<void> main() async {
  // Load the native library once, before calling any indicator.
  await FerroTa.init();

  final close = Float64List.fromList(
    [44.3, 44.1, 44.2, 43.6, 44.3, 44.8, 45.1, 45.4, 45.8, 46.1, 45.9, 46.3],
  );

  // Period arguments map from Rust `usize`, so they are `BigInt` in Dart.
  final sma5 = await sma(close: close, timeperiod: BigInt.from(5));
  final rsi5 = await rsi(close: close, timeperiod: BigInt.from(5));

  // Multi-output indicators return a Dart record in the documented order.
  final (upper, middle, lower) = await bbands(
    close: close,
    timeperiod: BigInt.from(5),
    nbdevup: 2,
    nbdevdn: 2,
    matype: 0, // 0 = SMA
  );

  // Warmup positions are NaN, matching TA-Lib.
  print('SMA(5):   $sma5');
  print('RSI(5):   $rsi5');
  print('BBANDS upper/middle/lower (last): '
      '${upper.last}, ${middle.last}, ${lower.last}');
}
