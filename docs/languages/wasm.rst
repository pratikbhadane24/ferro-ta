JavaScript / WASM
=================

npm package ``ferro-ta-wasm`` wraps ``ferro_ta_core`` with wasm-bindgen.
Node.js (``require``) and browsers (``import`` + ``init()``) share the same
exports.

Installation
------------

.. code-block:: bash

   npm install ferro-ta-wasm

Node.js
-------

.. code-block:: javascript

   const { sma, rsi, macd, bbands } = require('ferro-ta-wasm');

   const close = Float64Array.from({ length: 40 }, (_, i) => 44 + i * 0.1);

   const smaOut = sma(close, 5);
   const rsiOut = rsi(close, 14);
   const [macdLine, signal, hist] = macd(close, 12, 26, 9);
   const [upper, middle, lower] = bbands(close, 5, 2.0, 2.0);

Browser
-------

.. code-block:: html

   <script type="module">
     import init, { sma, rsi, macd, bbands } from 'ferro-ta-wasm';
     await init();

     const close = Float64Array.from({ length: 40 }, (_, i) => 44 + i * 0.1);
     console.log(sma(close, 5));
     console.log(rsi(close, 14));
     const [macdLine, signal, hist] = macd(close, 12, 26, 9);
     const [upper, middle, lower] = bbands(close, 5, 2.0, 2.0);
   </script>

Streaming
---------

Stateful classes are prefixed with ``Wasm``:

.. code-block:: javascript

   const { WasmStreamingSMA, WasmStreamingRSI } = require('ferro-ta-wasm');

   const stream = new WasmStreamingSMA(5);
   for (const price of close) {
     console.log(stream.update(price));
   }

API map
-------

Export names are ``snake_case``. A few names differ from TA-Lib / Python
(``trix_indicator``, ``math_add``, ``transform_sin``); the generated
:doc:`coverage` table normalizes them.

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Category
     - Exports
     - Examples
   * - Overlap
     - 20
     - ``sma``, ``ema``, ``bbands``, ``macd``, ``sar``
   * - Momentum
     - 26
     - ``rsi``, ``stoch``, ``plus_di``, ``trix_indicator``
   * - Candlestick
     - 61
     - ``cdldoji``, ``cdlengulfing``, ``cdlhammer``
   * - Volatility / volume
     - 9
     - ``atr``, ``obv``, ``mfi``, ``vwap``
   * - Math
     - 19+
     - ``math_add``, ``transform_sin``, ``transform_sqrt``
   * - Streaming
     - 9 classes
     - ``WasmStreamingSMA``, ``WasmStreamingRSI``, …
   * - Options / futures / backtest
     - 30+
     - ``black_scholes_price``, ``futures_basis``, ``backtest_core``

The full npm README is
`wasm/README.md <https://github.com/pratikbhadane24/ferro-ta/blob/main/wasm/README.md>`_.

Limitations
-----------

Large arrays (tens of millions of bars) pay a JS↔WASM copy. For that shape of
workload, call ``ferro_ta_core`` from Rust or use the Python binding.
