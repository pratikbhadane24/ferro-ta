Languages
=========

ferro-ta is one library: ``ferro_ta_core`` holds every indicator algorithm.
Rust exposes that crate directly; Python, JavaScript (WASM), Flutter, and Go
wrap it. Go and C/C++ share one C ABI crate, ``ferro_ta_ffi``, which future
C-ABI languages (C#, JVM, Ruby, ...) build on.

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
   * - Go
     - ``github.com/pratikbhadane24/ferro-ta/bindings/go``
     - cgo → ``ferro_ta_ffi`` C ABI → core
     - :doc:`go`
   * - C / C++
     - GitHub Release assets
     - ``ferro_ta.h`` → ``ferro_ta_ffi`` → core
     - :doc:`c`

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
   go
   c
   coverage
   adding
