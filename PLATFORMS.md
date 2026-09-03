# Supported Platforms and Packages

ferro-ta publishes first-class packages for Python, Rust, JavaScript (WASM),
and Flutter. All of them wrap [`ferro_ta_core`](crates/ferro_ta_core).

## Published packages

| Language | Package | Registry | Notes |
|---|---|---|---|
| Python | `ferro-ta` | PyPI / conda recipe | abi3 wheels; see below |
| Rust | `ferro_ta_core` | crates.io | `cargo add ferro_ta_core` |
| JavaScript | `ferro-ta-wasm` | npm | Node + browser builds |
| Flutter / Dart | `ferro_ta` | pub.dev | Prebuilt natives; web reuses npm |

## Python versions

| Python | Status |
|--------|--------|
| 3.13   | ✅ Supported (tested in CI) |
| 3.12   | ✅ Supported (tested in CI) |
| 3.11   | ✅ Supported (tested in CI) |
| 3.10   | ✅ Supported (tested in CI) |
| < 3.10 | ❌ Not supported |

We follow the [NEP 29](https://numpy.org/neps/nep-0029-deprecation-policy.html)
deprecation schedule: Python versions that have reached end-of-life are dropped
in the next minor release of ferro-ta.

## Python wheels (OS & architecture)

Pre-compiled wheels are published to PyPI for the following targets:

| OS      | Architecture    | Notes |
|---------|-----------------|-------|
| Linux   | x86_64, aarch64 (manylinux2014 / `manylinux_2_17`) | Pre-compiled wheel |
| Linux (musl) | x86_64, aarch64 (`musllinux_1_2`) | Pre-compiled wheel |
| macOS   | universal2      | One wheel covers Intel + Apple Silicon |
| Windows | x64, arm64      | |

Wheel releases are `cp310-abi3` stable-ABI wheels: one wheel per target covers
CPython 3.10+. A source distribution is also published so other compatible
environments can build from source.

> **Note:** Python 3.14+ is not yet tested.  Set
> `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` to attempt a build on a newer
> interpreter and report any issues.

## Flutter natives (pub.dev package `ferro_ta`)

The Flutter binding ships **prebuilt native libraries inside the package**, so
app developers need no Rust toolchain:

| Platform | Architectures | Path |
|---|---|---|
| Android | arm64-v8a, armeabi-v7a, x86_64 | native FFI |
| iOS     | arm64 (device), arm64 + x86_64 (simulator) | native FFI |
| macOS   | universal (arm64 + x86_64) | native FFI |
| Windows | x64 | native FFI |
| Linux   | x64 | native FFI |
| Web     | — | reuses the npm `ferro-ta-wasm` package via JS interop |

See [`flutter/README.md`](flutter/README.md) and
[`docs/languages/flutter.rst`](docs/languages/flutter.rst).

## Installation

### Python (pip)

```bash
pip install ferro-ta
```

No C-compiler required on the wheel targets listed above.

### Rust

```bash
cargo add ferro_ta_core
```

### JavaScript

```bash
npm install ferro-ta-wasm
```

### Flutter

```bash
flutter pub add ferro_ta
```

### conda / conda-forge

A Conda recipe is available in `conda/meta.yaml`.  To build locally, see
[PACKAGING.md](PACKAGING.md). Quick start:

```bash
conda install conda-build
conda build conda/
conda install --use-local ferro-ta
```

Once submitted to conda-forge the package will be installable via:

```bash
conda install -c conda-forge ferro-ta
```

## Source build (Python)

If no wheel is available for your platform, pip will attempt a source build:

```bash
# Requires Rust (stable toolchain) and maturin
pip install maturin
pip install ferro-ta --no-binary ferro-ta
```

## Known limitations

- WASM binding: 200+ exports including TA-Lib indicators, candlestick patterns,
  streaming API, options, futures, and backtesting (see `wasm/README.md`).
  Large arrays copy through JS↔WASM memory.
- Flutter binding: generated from the WASM signatures. A subset (options
  greeks/pricing, backtest engines, crossover-signal indices, and some batch
  ops) needs hand-written bridge wrappers — see `MANUAL_EXCLUDE` in
  `scripts/build_flutter_bridge.py` and [docs/languages/coverage.rst](docs/languages/coverage.rst).
- Python 3.14+: untested; may work with `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`.
- 32-bit platforms: not officially supported; source builds may succeed.
