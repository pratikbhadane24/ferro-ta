# Versioning & Release Policy

**ferro-ta** uses [Semantic Versioning 2.0.0](https://semver.org/).

## Version numbers: `MAJOR.MINOR.PATCH`

| Component | Increment when… |
|-----------|-----------------|
| **MAJOR** | A breaking API change is introduced (indicator removed, parameter renamed, return-type changed). |
| **MINOR** | New indicators, features, or bindings are added in a backward-compatible way. |
| **PATCH** | Bug fixes, performance improvements, documentation-only changes, or dependency bumps that do not change the public API. |

## Supported Python versions

We support the **three most recent stable Python minor releases** at the time
of a MINOR or MAJOR release.  Python versions that have reached end-of-life
(EOL) per the [Python release calendar](https://devguide.python.org/versions/)
are removed in the next MINOR release; this counts as a non-breaking change.

Currently supported: **3.10, 3.11, 3.12, 3.13** (see `pyproject.toml`).

## Release playbook

### Fast path

1. **Bump tracked version files with one command**:
   ```bash
   python3 scripts/bump_version.py 1.0.3
   ```
   or:
   ```bash
   make version VERSION=1.0.3
   ```
2. **Verify everything matches**:
   ```bash
   python3 scripts/bump_version.py --check
   ```
3. **Update `CHANGELOG.md`**: move the `[Unreleased]` block to a new dated section
   `[1.0.1] — YYYY-MM-DD` and open a fresh `[Unreleased]` block.
4. **Commit** the version bump and changelog update with message
   `chore: release v1.0.1`.
5. **Create a tag**: `git tag v1.0.1 && git push origin v1.0.1`.
6. **Create a GitHub Release** for tag `v1.0.1` — the CI `build-wheels` and
   `publish` jobs trigger automatically on `release: published`.

The bump script updates the tracked release-version carriers that are easy to
miss manually: root Cargo, Python packaging, the core crate, the core crate
README install snippet, the WASM package, the Conda recipe, and the docs pages
that show the current released version.

## Breaking-change policy

- Removing an indicator or changing its signature is a **MAJOR** change.
- Changing a default parameter value is a **MINOR** change (with a deprecation
  notice in the changelog).
- Fixing a numeric output to match TA-Lib more closely is a **PATCH** change
  (but noted clearly in the changelog).

## Changelog maintenance

Every PR that changes user-visible behaviour must add an entry to the
`[Unreleased]` section of `CHANGELOG.md`.  CI enforces this for PRs that
touch `src/`, `python/`, or `wasm/`.

---

## API Stability Guarantees

The following modules are considered **stable API** as of `1.0.0` and will not
have breaking changes in minor releases:

| Module | Stability |
|---|---|
| `ferro_ta` (top-level) — all `__all__` names | Stable |
| `ferro_ta.overlap`, `ferro_ta.momentum`, etc. | Stable |
| `ferro_ta.batch` | Stable |
| `ferro_ta.streaming` | Stable |
| `ferro_ta.extended` | Stable |
| `ferro_ta.exceptions` | Stable |
| `ferro_ta.registry` | Stable |
| `ferro_ta.backtest` | Stable |
| `ferro_ta.pipeline` | Stable |
| `ferro_ta.config` | Stable |
| `ferro_ta.gpu` (optional) | Beta — API may evolve |
| `ferro_ta._utils` (private) | Not stable — do not import directly |

### v1.0 readiness checklist

- [x] All 155+ TA-Lib indicators implemented and tested
- [x] Type stubs (`.pyi`) for all public functions
- [x] Sphinx documentation for all modules
- [x] Streaming bar-by-bar API (9 classes)
- [x] Batch execution API
- [x] Extended indicators (10 beyond TA-Lib)
- [x] WASM bindings (9 indicators)
- [x] Pandas integration (transparent)
- [x] Polars integration (transparent)
- [x] Backtesting harness
- [x] Plugin registry
- [x] Error model and input validation
- [x] Release playbook (RELEASE.md)
- [x] Changelog (CHANGELOG.md)
- [x] Version consistency CI check
- [x] Fuzzing (cargo-fuzz, SMA + RSI)
- [x] Optional GPU backend (CuPy)
- [x] Indicator pipeline API
- [x] Configuration defaults API
- [x] Jupyter notebook examples

### Post-1.0 notes

With `1.0.0` released:
1. The package now uses the `Development Status :: 5 - Production/Stable` classifier.
2. `CHANGELOG.md` now contains the `[1.0.0]` release section.
3. This file now reflects the stable-series SemVer contract.

### Compatibility matrix

| Python | Platform | Status |
|---|---|---|
| 3.10–3.13 | Linux x86_64 (manylinux) | ✅ Supported |
| 3.10–3.13 | macOS x86_64 | ✅ Supported |
| 3.10–3.13 | macOS aarch64 (Apple Silicon) | ✅ Supported |
| 3.10–3.13 | Windows x86_64 | ✅ Supported |
