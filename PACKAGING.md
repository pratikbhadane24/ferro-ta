# Packaging and distribution

This document describes how ferro-ta is packaged and published.

## PyPI (pip)

Wheels are built by CI on release (see [RELEASE.md](RELEASE.md)).
Release publishing currently targets CPython 3.10, 3.11, 3.12, and 3.13 on:

- Linux x86_64 (`manylinux_2_17` / `manylinux2014`)
- macOS universal2 (covers Intel and Apple Silicon)
- Windows x86_64

Each release also publishes a source distribution (`sdist`) so compatible
environments outside the wheel matrix can still build from source.

Publishing uses PyPI Trusted Publishing via GitHub OIDC; no long-lived PyPI API
token is required.

Supported platforms and Python versions are documented in [PLATFORMS.md](PLATFORMS.md).

## npm (WASM)

The Node.js / browser WASM package is published to npm by the **wasm-publish** workflow on release.

## crates.io (Rust)

The pure-Rust library `ferro_ta_core` is published to crates.io by the CI job **publish-cratesio** on release.
