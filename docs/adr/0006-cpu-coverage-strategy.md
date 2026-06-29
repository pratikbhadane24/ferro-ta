# ADR 0006: Runtime CPU dispatch for broad hardware coverage, not a pinned `target-cpu`

**Status:** Accepted
**Date:** 2026-06-29
**Related:** ADR 0003 (CPU path), `docs/guides/simd.md`

## Context

ferro-ta ships through three channels that all run on hardware we do not
control:

1. **PyPI** — Python wheels (`maturin`), installed onto arbitrary user CPUs.
2. **Container / compute nodes** — `api/Dockerfile` installs the wheel.
3. **crates.io** — `ferro_ta_core`, compiled by downstream Rust users.

A suggestion was raised to rebuild from sdist with
`RUSTFLAGS="-C target-cpu=x86-64-v3"` (AVX2+FMA+BMI) to match a specific
node fleet. That conflates two independent coverage axes:

- **Microarchitecture** (baseline vs AVX2 vs AVX-512, within one arch).
- **Platform/architecture** (linux/macOS/windows × x86_64/aarch64/musl).

A static `target-cpu` is the wrong tool for both. It raises the
*minimum* required instruction set: a `v3` binary executes an illegal
instruction (SIGILL) and **crashes** on any pre-Haswell / pre-Zen CPU, and
the flag is meaningless on aarch64. It narrows coverage to buy speed — the
opposite of the goal.

## Decision

**Axis 1 — microarchitecture: runtime dispatch, never a pinned
`target-cpu`.** Hot reductions are multiversioned with the
[`multiversion`](https://crates.io/crates/multiversion) crate
(`#[multiversion(targets = "simd")]`). Every variant (baseline … AVX-512,
NEON) is compiled into one binary and selected at load time via CPUID. The
`simd` feature is **on by default** in `ferro_ta_core` and forwarded as the
default in the `ferro_ta` extension crate, so all three channels get
adaptive SIMD with no flags and no crashes. `--no-default-features` yields a
pure-scalar build.

**Axis 2 — platform/architecture: an explicit build matrix.** The release
workflow builds one **abi3** (`cp310-abi3`) wheel per target:

| OS | Arch / libc |
|---|---|
| linux (manylinux_2_17) | x86_64, aarch64 |
| linux (musllinux_1_2) | x86_64, aarch64 |
| macOS | universal2 (x86_64 + arm64) |
| windows | x64, arm64 |

`abi3-py310` collapses the Python-version axis to a single wheel per
platform that also covers future CPython (3.11–3.14+). Because
`extension-module` + `abi3` links no `libpython`, the linux aarch64/musl
wheels cross-compile from the x86_64 runners.

## Rationale

- **One mechanism fixes all three channels.** Dispatch lives in
  `ferro_ta_core`, so the wheel, the Docker image, and crates.io consumers
  all inherit it — microarch coverage is solved once, not per channel.
- **`target-cpu` is a floor; CPUID dispatch is a branch.** The first
  excludes hardware at build time; the second adapts to it at run time.
  "Runs anywhere, fast where possible" requires the branch.
- **abi3 widens coverage while shrinking the matrix** — fewer artifacts,
  automatic forward-compatibility with new interpreters.
- **Pre-existing bug fixed.** Released wheels previously enabled no SIMD
  feature at all and shipped no aarch64 linux wheel (so arm64 containers
  failed to install). Both are corrected here.

## Trade-offs accepted

- **Larger binary / longer compile.** Multiversioning emits several clones
  per dispatched function. Acceptable for the reduction kernels involved.
- **Modest dispatch overhead** (one CPUID-backed branch per call) vs a
  statically specialized build. Negligible against per-call work; a uniform
  fleet that truly wants a pinned build can still set `RUSTFLAGS` locally.
- **Not bit-identical to scalar.** Lane-parallel reductions reorder
  floating-point adds (~ULPs), within documented tolerances.

## Explicitly out of scope (no silent caps)

These targets are deliberately **not** built and would be additive later:
32-bit x86, `armv7`, `s390x`, `ppc64le`, and free-threaded CPython
(`3.13t`/`3.14t`, which cannot use abi3 and needs a separate per-version,
`gil_used = false` build). Platforms without a wheel fall back to an sdist
build, which requires a Rust toolchain at install time.

## Verification

- `crates/ferro_ta_core/src/simd.rs` — parity tests assert dispatched
  `sum` / `wma_seed` match a strict scalar reference within tolerance,
  across empty / sub-lane / exact-multiple / remainder inputs.
- Full core suite (279 tests) passes under both `--features simd` (default)
  and `--no-default-features`.
- `cargo tree` confirms `multiversion` is present in root-default and
  crates.io-default builds and absent under `--no-default-features`.
- A local `cp310-abi3` wheel built and imported on CPython 3.14, running the
  dispatched SMA/WMA kernels.
