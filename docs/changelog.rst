Release Notes
=============

These docs track package version ``1.0.6``.

1.0.6 (2026-03-24)
------------------

- Added a repo-managed pre-push gate so the core Rust, Python, docs, and WASM
  checks can be run locally before release.
- Expanded Rust-backed analysis/data helpers, broadened the WASM exports, and
  added cross-surface API manifest verification plus Node conformance checks.
- Refreshed benchmark coverage and perf artifacts, aligned Python CI with the
  local tooling flow, and updated the locked security fixes needed for a clean
  release pass.

1.0.4 (2026-03-24)
------------------

- Expanded the optional MCP server from a small hand-written subset to the
  broader public ferro-ta callable surface, including stateful class support
  through stored-instance management tools.
- Split the root documentation so the full TA-Lib compatibility matrix lives in
  ``TA_LIB_COMPATIBILITY.md`` while the README stays product-first and shorter.
- Refreshed MCP docs/tests and updated locked low-risk Python dependencies as
  part of the release cleanup pass.
- Stopped tracking the stray ``.coverage`` artifact and aligned ignore rules
  for local coverage outputs.

1.0.3 (2026-03-24)
------------------

- Added top-level package metadata helpers such as ``ferro_ta.__version__``,
  ``ferro_ta.about()``, and ``ferro_ta.methods()``.
- Added a standalone derivatives benchmark artifact for selected options
  pricing, IV, Greeks, and Black-76 comparisons.
- Simplified release version bumps with a single script and updated release
  guidance.
- Fixed Python CI/type-stub gaps around the new metadata API and corrected the
  tag-driven GitHub Release workflow trigger used for publish automation.

1.0.2 (2026-03-24)
------------------

- Improved rolling statistical kernels and several Python analysis hotspots.
- Added reproducible perf-contract artifacts, TA-Lib regression guards, and
  updated benchmark tooling.
- Tightened the public benchmark documentation so claims, caveats, and evidence
  live closer together.

1.0.1 (2026-03-24)
------------------

- Improved release automation for PyPI, crates.io, and npm.
- Fixed CI workflow issues that caused otherwise healthy release jobs to fail.
- Ensured the published WASM package includes its built ``pkg/`` artifacts.

1.0.0 (2026-03-23)
------------------

- First stable release of the Rust-backed Python technical analysis library.
- Shipped broad TA-Lib coverage, streaming APIs, extended indicators, and the
  initial Sphinx documentation set.
- Added the benchmark suite, release playbook, and compatibility/testing
  scaffolding for stable releases.

For the canonical project changelog, including the full per-version details,
see `CHANGELOG.md <https://github.com/pratikbhadane24/ferro-ta/blob/main/CHANGELOG.md>`_.
