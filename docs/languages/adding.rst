Adding a language binding
=========================

ferro-ta is a Rust-core technical analysis library. **A new language may only
wrap** ``ferro_ta_core``. Reimplementing indicators in the new language is out
of scope: if the language cannot call Rust, the binding is rejected.

Follow this list in order. The pull-request template repeats it as checkboxes.

1. Core only
------------

Every compute call is ``ferro_ta_core::…``. No ported loops, no second
algorithm, no "temporary" Python/JS/Dart reimplementation. Marshalling,
validation, and idiomatic naming belong in the binding. Arithmetic over bars
does not.

2. Choose a binding style
-------------------------

Document which style you chose, and why:

- Direct FFI to ``ferro_ta_core`` (PyO3, UniFFI, cbindgen, napi-rs, …)
- WASM interop (reuse ``ferro-ta-wasm``, like Flutter web)
- Generated from an existing binding's signatures (like
  ``scripts/build_flutter_bridge.py`` from WASM) — still must call core, not
  reimplement

3. Repo layout
--------------

Use ``bindings/<lang>/`` or an existing top-level directory (``python/``,
``wasm/``, ``flutter/``). Own README, license, and package manifest.

4. API shape
------------

Document, in the language page and README:

- Naming (TA-Lib ``SMA`` vs snake_case ``sma``)
- Input types (owned vs borrowed buffers)
- NaN warmup
- Multi-output (tuples vs structs)
- Errors

5. Coverage plan
----------------

Start from core public functions. Check the generated table in
:doc:`coverage`. List ``MANUAL_EXCLUDE``-style skips with reasons. Do not
claim parity the table does not show.

6. Numeric tests
----------------

Use the same fixtures / known values as core (or Python
``tests/unit/test_known_values.py`` vectors). At least SMA, EMA, RSI, MACD,
BBANDS, ATR, and one CDL pattern.

7. CI
-----

Add ``.github/workflows/ci-<lang>.yml`` and wire it into
``.github/workflows/CI.yml``. If wrappers are generated, add a ``--check``
job that wrappers stay fresh.

8. Publish
----------

Registry, trusted publishing / tokens, a row in ``RELEASE.md``'s publish
matrix, and a version carrier in ``scripts/bump_version.py``.

9. Docs
-------

Language page under ``docs/languages/``, README install row, support-matrix
row, changelog, and ``python3 scripts/build_api_manifest.py`` so
:doc:`coverage` updates.

10. Policy updates
------------------

Update `docs/rust_first.md <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/rust_first.md>`_
(core-first policy) and CONTRIBUTING "Adding other indicators" so the new
binding is listed next to Python / WASM / Flutter when the indicator is in
the shared core set.
