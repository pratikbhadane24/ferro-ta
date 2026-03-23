# API Stability Policy

This document describes which parts of **ferro-ta** are considered stable, which
are experimental, and what the deprecation process is.

---

## Stability Tiers

### Stable

The following are considered **stable** and will not change in incompatible ways
without a major version bump (i.e., following [Semantic Versioning 2.0.0]):

- All indicator functions exported from `ferro_ta.*` by name (e.g. `ferro_ta.SMA`,
  `ferro_ta.RSI`, `ferro_ta.BBANDS`).
- Sub-module imports: `from ferro_ta.overlap import SMA` etc.
- Function signatures: positional array arguments and `timeperiod` / other keyword
  arguments documented in the docstrings.
- Return types: single `np.ndarray` or tuple of `np.ndarray` as documented.
- Exception classes: `FerroTAError`, `FerroTAValueError`, `FerroTAInputError`.
- Utility helpers: `ferro_ta.utils.get_ohlcv`, `ferro_ta._utils.get_ohlcv`.
- `pandas_wrap` / `polars_wrap` behaviour: `pd.Series` in → `pd.Series` out;
  `pl.Series` in → `pl.Series` out.
- Registry API: `ferro_ta.registry.register`, `run`, `get`, `list_indicators`.
- Pipeline API: `ferro_ta.pipeline.Pipeline`, `make_pipeline`.
- Config API: `ferro_ta.config.set_default`, `ferro_ta.config.Config`.

### Experimental

The following are **experimental** and may change in minor releases:

- **`ferro_ta.raw`** — direct access to the compiled Rust extension; function
  signatures follow the Rust layer and may change when the Rust layer changes.
- **`ferro_ta.batch`** internals — the Python↔Rust dispatch logic may change as
  the Rust batch API evolves.
- **`ferro_ta.streaming`** — the streaming class API (especially the `reset()`
  method and internal buffer access) may evolve; the `update()` method signature
  is stable.
- **`ferro_ta.extended`** — extended indicators (VWAP, SUPERTREND, etc.) are
  considered stable in return shape and semantics, but implementation details
  (e.g. whether computation is in Python or Rust) may change.
- **`ferro_ta.backtest`** — the backtest helpers are convenience utilities and
  may be refactored.
- **`ferro_ta.gpu`** — the CuPy GPU backend is an experimental proof-of-concept.

### Internal / Private

Names prefixed with `_` (e.g. `_ferro_ta`, `_utils`, `_to_f64`) are internal
and may change at any time without notice.  Do not rely on them in user code.

---

## Versioning

ferro-ta follows [Semantic Versioning 2.0.0]:

| Change type                            | Version bump |
|----------------------------------------|--------------|
| Breaking API change (removed indicator, renamed parameter, changed return type) | **MAJOR** |
| New indicators, new sub-modules, new features (backward-compatible) | **MINOR** |
| Bug fixes, performance improvements, docs, dependency bumps | **PATCH** |

The current version (`1.x`) is stable. Breaking changes to stable APIs are
reserved for future **major** releases.

---

## Deprecation Policy

Before removing or renaming any **stable** API:

1. The deprecated name/function is kept until the next **major release** after
   the deprecation notice.
2. A `DeprecationWarning` is raised when the deprecated API is used.
3. The deprecation and removal are documented in `CHANGELOG.md` under
   `### Deprecated` and `### Removed`.

Example timeline:

- `1.1.0` — `OLD_NAME` deprecated, `DeprecationWarning` added; `NEW_NAME` available.
- `2.0.0` — `OLD_NAME` removed.

---

## What is NOT covered

- The Rust ABI of the compiled extension (`_ferro_ta.so` / `_ferro_ta.pyd`).
  Only the Python-level API is covered by this policy.
- Numerical precision beyond what is documented (exact TA-Lib matches for listed
  indicators, "correlated" for Wilder-seeded indicators).
- Performance characteristics — we may change the implementation to be faster
  (e.g. moving a Python loop to Rust) without a version bump.

---

## Requesting Stability Guarantees

If you depend on an experimental API and would like it promoted to stable, please
open an issue on GitHub explaining your use case.  We will consider promoting
experimental APIs to stable when they have been in use long enough to be confident
in their design.

[Semantic Versioning 2.0.0]: https://semver.org/
