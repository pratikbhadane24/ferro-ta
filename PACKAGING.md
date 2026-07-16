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

## pub.dev (Flutter)

The Flutter package `ferro_ta` (source in `flutter/`) is published to pub.dev by
the **flutter-publish** workflow on release. Unlike the other channels, the
native libraries are **bundled inside the published package**, so app developers
need no Rust toolchain:

| Platform | Architectures | Artifact |
|---|---|---|
| Android | arm64-v8a, armeabi-v7a, x86_64 | `.so` in `android/src/main/jniLibs/` |
| iOS | arm64 device, arm64 + x86_64 simulator | `ferro_ta_flutter.xcframework` |
| macOS | universal (arm64 + x86_64) | `.dylib` |
| Windows | x64 | `.dll` |
| Linux | x64 | `.so` |
| Web | — | reuses the npm `ferro-ta-wasm` package |

The workflow builds each platform on its own runner, uploads the libraries as
artifacts, then a single job assembles them into the plugin tree and runs
`flutter pub publish`. Publishing uses **pub.dev automated publishing via GitHub
OIDC**; no long-lived pub.dev credential is stored.

> One-time setup: the package must exist on pub.dev and have *Automated
> publishing* enabled (bound to this repo and the `v{{version}}` tag pattern)
> before the workflow can publish. The first publish is manual.

Binaries are never committed to git — they are produced in CI and published
straight from the working tree.
