Contributing
============

Thank you for your interest in contributing to ferro-ta!

This page summarises how to get started. The full details are in
`CONTRIBUTING.md <https://github.com/pratikbhadane24/ferro-ta/blob/main/CONTRIBUTING.md>`_
at the repository root.

.. contents::
   :local:
   :depth: 2


Development setup
-----------------

Prerequisites: Rust stable toolchain, Python 3.10+, and ``maturin``.

.. code-block:: bash

   git clone https://github.com/pratikbhadane24/ferro-ta.git
   cd ferro-ta
   pip install maturin numpy pytest pytest-cov
   maturin develop --release
   pytest tests/


Git hooks and pre-push checks
-----------------------------

Install the repository-managed hooks after setting up the environment:

.. code-block:: bash

   make hooks

Run the same push gate manually with:

.. code-block:: bash

   make prepush

You can scope it to selected checks while iterating:

.. code-block:: bash

   make prepush CHECKS="version changelog python_lint"


Adding a new indicator
-----------------------

1. **Core** — implement the algorithm in ``crates/ferro_ta_core/src/<module>.rs``
   with a unit test. Slice inputs, ``Vec<f64>`` output, leading NaN warmup.

2. **Python wrapper** — thin PyO3 function in ``src/<module>/`` that calls
   core, plus a ``python/ferro_ta/*.py`` wrapper using ``_to_f64``. Export it
   in ``__all__``.

3. **Re-export** — add the function to ``python/ferro_ta/__init__.py``'s
   ``__all__`` list and import block.

4. **Type stub** — add a type annotation to ``python/ferro_ta/__init__.pyi``.

5. **Other bindings** — add the WASM export and regenerate Flutter when the
   indicator is in the shared core set. Refresh
   ``python3 scripts/build_api_manifest.py``.

6. **Tests** — add at least one test class in ``tests/test_ferro_ta.py``
   covering output length, NaN count, and a known-value check.

7. **README** — add a row to the appropriate accuracy table.

Language bindings
-----------------

Rust exposes ``ferro_ta_core`` directly. Python, JavaScript (WASM), and
Flutter are peer wrappers over that crate. New languages may only wrap the
core. See :doc:`languages/adding` and
`CONTRIBUTING.md <https://github.com/pratikbhadane24/ferro-ta/blob/main/CONTRIBUTING.md>`_.


Code style
----------

- Rust: ``cargo fmt`` (enforced in CI) and ``cargo clippy -- -D warnings``
- Python: PEP 8; function names in UPPER_CASE to match TA-Lib convention.
- All public Python functions should have NumPy-style docstrings.


Running tests
-------------

.. code-block:: bash

   # Python tests
   pytest tests/ -v

   # Rust format check
   cargo fmt --check

   # Rust lints
   cargo clippy --release -- -D warnings

   # Optional: TA-Lib comparison tests (requires ta-lib installed)
   pytest tests/test_vs_talib.py -v


Type checking
-------------

The package is typed (PEP 561). To run mypy::

   pip install mypy numpy
   mypy python/ferro_ta --ignore-missing-imports


Questions
---------

Open a GitHub Issue or Discussion. For security vulnerabilities see
`SECURITY.md <https://github.com/pratikbhadane24/ferro-ta/blob/main/SECURITY.md>`_.
