Cross-language coverage
=======================

This page is generated from the cross-surface API manifest. Do not edit the
table by hand.

Refresh after adding or wrapping an indicator:

.. code-block:: bash

   python3 scripts/build_api_manifest.py
   python3 scripts/check_api_manifest.py

CI fails if ``docs/api_manifest.json`` or the include below drift from source
exports (Python ``__all__``, ``ferro_ta_core`` ``pub fn``\s, WASM
``#[wasm_bindgen]`` items, Flutter wrappers, and ``MANUAL_EXCLUDE``).

Names are normalized so ``SMA``, ``sma``, ``PLUS_DI`` / ``plus_di``, and
``TRIX`` / ``trix_indicator`` count as the same row. Flutter cells marked
``excluded`` are in ``scripts/build_flutter_bridge.py``'s ``MANUAL_EXCLUDE``
set (hand-written bridge still required).

.. include:: _coverage.inc.rst
