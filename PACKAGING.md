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
the **flutter-publish** workflow, which is triggered by the **`vX.Y.Z` tag push**
— *not* by `release: published` like the other channels.

> This difference is mandatory, not stylistic. pub.dev
> [rejects automated publishing](https://dart.dev/tools/pub/automated-publishing)
> from any GitHub Actions run that was not triggered by a tag push, so this
> workflow cannot hang off the GitHub Release event the way PyPI and npm do.
> A manual `workflow_dispatch` can therefore only ever dry-run.

Unlike the other channels, the native libraries are **bundled inside the
published package**, so app developers need no Rust toolchain:

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
`dart pub publish --force`. Publishing uses **pub.dev automated publishing via
GitHub OIDC** (`id-token: write`, scoped to the publish job); no long-lived
pub.dev credential is stored.

> **One-time setup, required before automated publishing works:**
> pub.dev [only automates publishing of *existing* packages](https://dart.dev/tools/pub/automated-publishing)
> — *"To create a new package, you must publish the first version using
> `dart pub publish`."* So: publish `ferro_ta` once by hand from `flutter/`,
> then enable **Automated publishing** in the pub.dev package admin, bound to
> this repository with the `v{{version}}` tag pattern.
>
> Note also that pub cannot publish a *new* package directly to a verified
> publisher; publish to a Google Account first, then transfer the package.

### What actually gets published

`dart pub publish` includes everything under the package root **except** hidden
files (names starting with `.`) and anything matched by `.pubignore`/`.gitignore`.
`.pubignore` **overrules `.gitignore` per-directory**: because `flutter/` holds
both, `flutter/.gitignore` is ignored entirely by pub. That is deliberate and
load-bearing — `.gitignore` excludes the native libraries and the generated
bindings (they are build artifacts), and without `.pubignore` pub would strip
them and ship a broken package. The publish job asserts both are present in the
`--dry-run` file list before publishing.

Package size limits: **< 100 MB gzipped and < 256 MB uncompressed.**

Binaries are never committed to git — they are produced in CI and published
straight from the working tree.
