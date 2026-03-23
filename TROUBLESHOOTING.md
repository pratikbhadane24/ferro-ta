# ferro-ta Troubleshooting Guide

Common build and runtime issues and how to fix them.

---

## Table of Contents

1. [maturin build fails](#maturin-build-fails)
2. [PyO3 version mismatches](#pyo3-version-mismatches)
3. [ImportError: cannot import name '_ferro_ta'](#importerror-cannot-import-name-_ferro_ta)
4. [Rust toolchain not found](#rust-toolchain-not-found)
5. [tests fail with 'ferro_ta not installed'](#tests-fail-with-ferro_ta-not-installed)
6. [mypy / pyright type errors after install](#mypy--pyright-type-errors-after-install)
7. [WASM build fails](#wasm-build-fails)
8. [GPU / CuPy errors](#gpu--cupy-errors)
9. [Coverage below threshold](#coverage-below-threshold)
10. [Common Rust compilation errors](#common-rust-compilation-errors)

---

## maturin build fails

**Symptom:** `maturin develop` or `maturin build` exits with a Rust compilation error.

**Fixes:**
- Ensure you have the **stable** Rust toolchain installed:
  ```bash
  rustup toolchain install stable
  rustup default stable
  ```
- Ensure `rustfmt` and `clippy` components are installed:
  ```bash
  rustup component add rustfmt clippy
  ```
- Make sure Python headers are available.  On Debian/Ubuntu:
  ```bash
  sudo apt-get install python3-dev
  ```
- If you changed `Cargo.toml`, run `cargo check` first to isolate Rust errors from maturin wrapping issues.

---

## PyO3 version mismatches

**Symptom:** `pyo3` version conflict between your Python interpreter and the version pinned in `Cargo.toml`.

**Fix:**  ferro-ta uses PyO3 with the `abi3` feature flag which supports Python 3.10+.  If you need a specific version:
```toml
# Cargo.toml
[dependencies]
pyo3 = { version = "0.22", features = ["extension-module", "abi3-py310"] }
```
Run `cargo update -p pyo3` to pull the latest compatible version.

---

## ImportError: cannot import name '_ferro_ta'

**Symptom:**
```
ImportError: cannot import name '_ferro_ta' from 'ferro_ta'
```

**Causes and fixes:**
1. The Rust extension has not been compiled yet — run `maturin develop --release` or `make build`.
2. The `.so` file was compiled for a different Python version — rebuild with the current interpreter.
3. You are running `python` from a different virtualenv — activate the correct environment.

Check that the compiled extension is present:
```bash
python -c "import ferro_ta._ferro_ta; print('OK')"
```

---

## Rust toolchain not found

**Symptom:** `cargo: command not found` or `rustup: command not found`.

**Fix:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

---

## tests fail with 'ferro_ta not installed'

**Symptom:** pytest reports import errors for `ferro_ta`.

**Fix:** Build and install the development wheel first:
```bash
maturin develop --release
# or
make build
```
Then re-run tests:
```bash
pytest tests/
```

---

## mypy / pyright type errors after install

**Symptom:** mypy or pyright reports errors for optional dependencies (cupy, polars, etc.).

**Fix:** ferro-ta ships a `pyrightconfig.json` that sets `reportMissingImports = false` for optional deps.  For mypy, pass `--ignore-missing-imports`:
```bash
mypy python/ferro_ta --ignore-missing-imports
```
The CI uses this flag by default.

---

## WASM build fails

**Symptom:** `wasm-pack build` fails inside `wasm/`.

**Fix:**
1. Install `wasm-pack`:
   ```bash
   curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
   ```
2. Add the WASM target:
   ```bash
   rustup target add wasm32-unknown-unknown
   ```
3. Run from the `wasm/` subdirectory (it has its own `[workspace]` table):
   ```bash
   cd wasm && wasm-pack build --target nodejs
   ```

---

## GPU / CuPy errors

**Symptom:** `ImportError: No module named 'cupy'` or CUDA errors in `ferro_ta.gpu`.

**Fix:** The GPU module is **optional**.  Install CuPy matching your CUDA version:
```bash
pip install cupy-cuda12x   # for CUDA 12.x
```
If no GPU is available, all ferro_ta functions fall back silently to CPU (NumPy) computation.

---

## Coverage below threshold

**Symptom:** `pytest --cov-fail-under=65` fails with a coverage percentage below 65 %.

**Fix:**
- Run `pytest tests/ --cov=ferro_ta --cov-report=term-missing` to see which lines are uncovered.
- The threshold in CI is 65 %.  Local runs may vary if optional dependencies (pandas, polars) are not installed.
- Install all test extras:
  ```bash
  pip install pandas polars hypothesis pyyaml
  ```

---

## Common Rust compilation errors

### `error[E0425]: cannot find function 'compute_ema'`

Dead code was removed in a refactor.  Run `cargo clean` then `cargo build --release`.

### `error: current package believes it's in a workspace when it's not`

This happens in `fuzz/` or `wasm/` sub-crates.  Both have `[workspace]` in their `Cargo.toml` to opt out of the root workspace.  If you create a new sub-crate, add `[workspace]` to its `Cargo.toml`.

### Linker errors on macOS (Apple Silicon)

```bash
export MACOSX_DEPLOYMENT_TARGET=11.0
maturin develop --release
```

---

## Getting help

- Open an issue: <https://github.com/pratikbhadane24/ferro-ta/issues>
- See `CONTRIBUTING.md` for development guidelines.
