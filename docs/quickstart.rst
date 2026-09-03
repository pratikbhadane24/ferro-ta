Quick Start
===========

ferro-ta is a Rust-core technical analysis library with first-class bindings
for Python, Rust, JavaScript (WASM), and Flutter. Pick a language — each page
uses the same four examples (SMA, RSI, MACD, BBANDS).

.. list-table::
   :header-rows: 1
   :widths: 18 28 54

   * - Language
     - Install
     - Guide
   * - Python
     - ``pip install ferro-ta``
     - :doc:`languages/python`
   * - Rust
     - ``cargo add ferro_ta_core``
     - :doc:`languages/rust`
   * - JavaScript (WASM)
     - ``npm install ferro-ta-wasm``
     - :doc:`languages/wasm`
   * - Flutter / Dart
     - ``flutter pub add ferro_ta``
     - :doc:`languages/flutter`

Coverage across languages is generated from the API manifest — see
:doc:`languages/coverage`. New bindings must wrap ``ferro_ta_core``:
:doc:`languages/adding`.
