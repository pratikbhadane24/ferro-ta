# Supported Platforms & Python Versions

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

## Operating systems & architectures

Pre-compiled wheels are published to PyPI for the following targets:

| OS      | Architecture    | Notes |
|---------|-----------------|-------|
| Linux   | x86_64 (manylinux2014 / `manylinux_2_17`) | Default CI runner |
| Linux   | aarch64         | Built via maturin cross-compilation |
| macOS   | x86_64          | Intel |
| macOS   | arm64           | Apple Silicon |
| macOS   | universal2      | Intel + Apple Silicon fat binary |
| Windows | x86_64          | |

> **Note:** Python 3.14+ is not yet tested.  Set
> `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` to attempt a build on a newer
> interpreter and report any issues.

## Installation

### pip (recommended)

```bash
pip install ferro-ta
```

No C-compiler required — pre-compiled wheels are available for all platforms
listed above.

### conda / conda-forge

A Conda recipe is available in `conda/meta.yaml`.  To build locally, see
[PACKAGING.md](PACKAGING.md). Quick start:

```bash
conda install conda-build
conda build conda/
conda install --use-local ferro_ta
```

Once submitted to conda-forge the package will be installable via:

```bash
conda install -c conda-forge ferro_ta
```

## Source build

If no wheel is available for your platform, pip will attempt a source build:

```bash
# Requires Rust (stable toolchain) and maturin
pip install maturin
pip install ferro-ta --no-binary ferro-ta
```

## Known limitations

- WASM binding: only 6 indicators exposed (see `wasm/README.md`).
- Python 3.14+: untested; may work with `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`.
- 32-bit platforms: not officially supported; source builds may succeed.
