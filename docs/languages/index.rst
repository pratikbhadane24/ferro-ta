Languages
=========

ferro-ta is one library: ``ferro_ta_core`` holds every indicator algorithm.
Rust exposes that crate directly; Python, JavaScript (WASM), and Flutter wrap
it.

.. list-table::
   :header-rows: 1
   :widths: 20 24 28 28

   * - Language
     - Package
     - Compute
     - Guide
   * - Python
     - PyPI ``ferro-ta``
     - PyO3 → ``ferro_ta_core``
     - :doc:`python`
   * - Rust
     - crates.io ``ferro_ta_core``
     - Direct ``&[f64]`` API
     - :doc:`rust`
   * - JavaScript
     - npm ``ferro-ta-wasm``
     - wasm-bindgen → core
     - :doc:`wasm`
   * - Flutter / Dart
     - pub.dev ``ferro_ta``
     - flutter_rust_bridge → core; web reuses WASM
     - :doc:`flutter`

Python keeps the richest *ergonomic* surface (TA-Lib names, pandas/polars,
Sphinx autodoc). Indicator coverage is not identical on every binding — check
:doc:`coverage` before claiming parity.

Hard rule
---------

A new language may only wrap ``ferro_ta_core`` (FFI, wasm-bindgen, UniFFI,
flutter_rust_bridge, napi-rs, and similar). Reimplementing indicators in the
new language is out of scope. See :doc:`adding`.

.. toctree::
   :maxdepth: 1

   python
   rust
   wasm
   flutter
   coverage
   adding
