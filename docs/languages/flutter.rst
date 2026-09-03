Flutter / Dart
==============

pub.dev package ``ferro_ta`` exposes ``ferro_ta_core`` through
flutter_rust_bridge. Native apps load a **prebuilt** Rust library (no Rust
toolchain for app developers). Flutter **web** reuses npm ``ferro-ta-wasm``.

Wrappers are generated from the WASM signatures
(``scripts/build_flutter_bridge.py``), so Flutter stays in lockstep with the
JS API rather than reimplementing indicators.

Installation
------------

.. code-block:: yaml

   dependencies:
     ferro_ta: ^1.2.0

Or ``flutter pub add ferro_ta``.

Native (Android / iOS / macOS / Windows / Linux)
------------------------------------------------

Call ``FerroTa.init()`` once, then use the generated functions.

.. code-block:: dart

   import 'dart:typed_data';
   import 'package:ferro_ta/ferro_ta.dart';

   Future<void> main() async {
     await FerroTa.init();

     final close = Float64List.fromList([
       for (var i = 0; i < 40; i++) 44.0 + i * 0.1,
     ]);

     final smaOut = await sma(close: close, timeperiod: 5);
     final rsiOut = await rsi(close: close, timeperiod: 14);
     final (macdLine, signal, hist) = await macd(
       close: close,
       fastperiod: 12,
       slowperiod: 26,
       signalperiod: 9,
     );
     final (upper, middle, lower) = await bbands(
       close: close,
       timeperiod: 5,
       nbdevup: 2,
       nbdevdn: 2,
     );
   }

Web
---

.. code-block:: dart

   import 'package:ferro_ta/ferro_ta_web.dart';
   // Reuses ferro-ta-wasm on globalThis.ferroTaWasm.
   final sma3 = smaWeb(close, 3);

API map
-------

Generated names follow the WASM ``snake_case`` exports (``sma``, ``rsi``,
``macd``, ``bbands``). Coverage is a **subset** of WASM: some signatures need
hand-written bridge wrappers and are listed in ``MANUAL_EXCLUDE``.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Surface
     - Status
   * - Core indicators (SMA, RSI, MACD, BBANDS, ATR, …)
     - Generated and shipped
   * - Most overlap / momentum / volume / volatility
     - Generated
   * - Options greeks / pricing, backtest engines
     - ``MANUAL_EXCLUDE`` — not exposed yet
   * - Crossover-signal indices, some batch array-of-array ops
     - ``MANUAL_EXCLUDE``
   * - ``ht_trendmode``, ``stochf``, ``supertrend``, ``dtw_distance``
     - ``MANUAL_EXCLUDE``
   * - Candlestick ``cdl*`` (``Int32`` series)
     - Not in the current generated module; see :doc:`coverage`
   * - Streaming classes (``StreamingSMA``, …)
     - Not generated; absent on :doc:`coverage` (not ``MANUAL_EXCLUDE``)

The checked-in coverage table marks Flutter gaps as ``excluded`` when they
appear in ``MANUAL_EXCLUDE``. Do not claim parity the table does not show.

Platforms
---------

See :doc:`/support_matrix` and
`PLATFORMS.md <https://github.com/pratikbhadane24/ferro-ta/blob/main/PLATFORMS.md>`_
for native architectures. Package README:
`flutter/README.md <https://github.com/pratikbhadane24/ferro-ta/blob/main/flutter/README.md>`_.
