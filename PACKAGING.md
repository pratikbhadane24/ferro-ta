# Packaging and distribution

This document describes how ferro-ta is packaged and published.

## PyPI (pip)

Wheels are built by CI on release (see [RELEASE.md](RELEASE.md)).

Supported platforms and Python versions are documented in [PLATFORMS.md](PLATFORMS.md).

## npm (WASM)

The Node.js / browser WASM package is published to npm by the **wasm-publish** workflow on release.

## crates.io (Rust)

The pure-Rust library `ferro_ta_core` is published to crates.io by the CI job **publish-cratesio** on release.
