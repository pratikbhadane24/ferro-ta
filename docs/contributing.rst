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

1. **Rust** — implement the function in the appropriate ``src/<module>/``
   directory (e.g. ``src/overlap/mod.rs`` and ``src/overlap/sma.rs``).  Follow
   the existing patterns: slice inputs, ``Vec<f64>`` output, leading NaN for
   warm-up bars, a ``#[pyfunction]`` decorator, and registration in the
   module's ``register(m)`` function.

2. **Python** — add a thin wrapper in the matching ``python/ferro_ta/*.py``
   module using the ``_to_f64`` helper.  Export it in ``__all__``.

3. **Re-export** — add the function to ``python/ferro_ta/__init__.py``'s
   ``__all__`` list and import block.

4. **Type stub** — add a type annotation to ``python/ferro_ta/__init__.pyi``.

5. **Tests** — add at least one test class in ``tests/test_ferro_ta.py``
   covering output length, NaN count, and a known-value check.

6. **README** — add a row to the appropriate accuracy table.


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
