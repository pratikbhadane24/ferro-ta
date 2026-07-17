# Release Playbook

This document describes the step-by-step process for cutting a new **ferro-ta** release.
Follow every step in order to produce a consistent, reproducible release.

For the packaging and release overview, see [PACKAGING.md](PACKAGING.md).

---

## Publish matrix (all automatic on release)

| Artifact    | How |
|-------------|-----|
| **PyPI**    | CI job `publish` using PyPI Trusted Publishing (OIDC) |
| **npm (WASM)** | Workflow `wasm-publish`|
| **crates.io** | CI job `publish-cratesio` |
| **pub.dev (Flutter)** | Workflow `flutter-publish` using pub.dev automated publishing (OIDC). **Triggered by the `vX.Y.Z` tag push, not by the GitHub Release** — pub.dev rejects publishes that are not tag-triggered. |

PyPI releases are expected to include:

- One `cp310-abi3` wheel per platform/architecture (each covers CPython 3.10+)
- Linux x86_64 and aarch64 (`manylinux_2_17` and `musllinux_1_2`)
- macOS universal2
- Windows x64 and arm64
- One source distribution (`sdist`)

---

## Pre-release checklist

Before starting a release:

- [ ] All CI checks are green on `main`: Rust (fmt, clippy), tests (with coverage gate),
      lint (ruff), typecheck (mypy, pyright), docs (Sphinx), WASM, audit (cargo-deny
      advisories, pip-audit), fuzz (no crashes).
- [ ] **Security audit clean:** Run `cargo audit` and `pip-audit` locally and confirm
      no high/critical vulnerabilities. Address any findings before tagging.
      ```bash
      cargo audit
      pip-audit  # or: uv run --with pip-audit pip-audit
      ```
- [ ] No open blocking issues or PRs that must land first.
- [ ] `CHANGELOG.md` has a `## [X.Y.Z]` section (not `[Unreleased]`) with all
      changes since the last release documented under `### Added`, `### Changed`,
      `### Fixed`, `### Removed` headings.
- [ ] Public docs match the release: `docs/conf.py`, `docs/changelog.rst`, and
      `docs/support_matrix.rst` reflect the version and current support status.

---

## Step 1 — Decide the version number

Follow [Semantic Versioning 2.0.0](https://semver.org/) and the policy in
[VERSIONING.md](VERSIONING.md):

| Change type | Version component to bump |
|---|---|
| Breaking API change (indicator removed, parameter renamed, return type changed) | **MAJOR** |
| New indicators, features, or bindings (backward-compatible) | **MINOR** |
| Bug fixes, performance, docs-only, dependency bumps | **PATCH** |

Example: current version is `0.1.0` and you are adding new indicators → new version is `0.2.0`.

---

## Step 2 — Sync version everywhere

These files must carry **the same version string** (e.g. `0.2.0`). The easiest
way to do that is:

```bash
python3 scripts/bump_version.py 0.2.0
python3 scripts/bump_version.py --check
```

That script updates the tracked release-version carriers for you.

Files covered by the bump script:

| File | Location |
|------|----------|
| `Cargo.toml` | Root (source of truth) |
| `crates/ferro_ta_core/Cargo.toml` | Same version for crates.io publish |
| `crates/ferro_ta_core/README.md` | Installation snippet should show the current crate version |
| `pyproject.toml` | Root |
| `wasm/Cargo.toml` | WASM crate version |
| `wasm/package.json` | Package version |
| `flutter/pubspec.yaml` | pub.dev package version |
| `flutter/rust/Cargo.toml` | Flutter bridge crate version |
| `conda/meta.yaml` | Conda recipe version |
| `docs/changelog.rst` | Tracked-version note on the docs changelog page |
| `docs/support_matrix.rst` | Tracked-version note on the support matrix page |

**`Cargo.toml`** (root):
```toml
[package]
name = "ferro_ta"
version = "X.Y.Z"   # ← or use scripts/bump_version.py X.Y.Z
```

**`pyproject.toml`**:
```toml
[project]
version = "X.Y.Z"   # ← must match Cargo.toml exactly
```

> **Rule:** `Cargo.toml` is the source of truth. Sync the others to match before tagging.

---

## Step 3 — Update CHANGELOG.md

1. Open `CHANGELOG.md`.
2. Rename the `[Unreleased]` section to `[X.Y.Z] — YYYY-MM-DD` (today's date).
3. Add a fresh empty `[Unreleased]` section at the top.
4. Update the comparison links at the bottom:

```markdown
[Unreleased]: https://github.com/pratikbhadane24/ferro-ta/compare/vX.Y.Z...HEAD
[X.Y.Z]: https://github.com/pratikbhadane24/ferro-ta/compare/vPREVIOUS...vX.Y.Z
```

Follow the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format:
`Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`.

Also update the docs-facing release surfaces for the same version:

- `docs/changelog.rst` with a concise release-notes entry
- `docs/support_matrix.rst` if supported versions, tested wheels, or module
  stability changed

---

## Step 4 — Commit the version bump

```bash
git add Cargo.toml Cargo.lock pyproject.toml CHANGELOG.md \
    crates/ferro_ta_core/Cargo.toml crates/ferro_ta_core/README.md \
    wasm/Cargo.toml wasm/package.json flutter/pubspec.yaml flutter/rust/Cargo.toml \
    conda/meta.yaml docs/changelog.rst docs/support_matrix.rst
git commit -m "chore: release v0.2.0"
git push origin main
```

Wait for CI to pass on this commit before proceeding.

---

## Step 5 — Create and push the tag

```bash
git tag v0.2.0
git push origin v0.2.0
```

> Tags must be in the form `vMAJOR.MINOR.PATCH` (e.g. `v0.2.0`).

Pushing the tag immediately starts **two** workflows:

- `release.yml` — creates the GitHub Release (which then triggers the PyPI, npm,
  and crates.io publishes).
- `flutter-publish.yml` — builds the native libraries and publishes to pub.dev.
  This one hangs off the **tag push itself**, because pub.dev rejects automated
  publishing from a run that was not tag-triggered. The tag must match the
  `v{{version}}` pattern configured on pub.dev and the `version:` in
  `flutter/pubspec.yaml`.

---

## Step 6 — Verify the GitHub Release

The `release.yml` workflow creates and publishes the GitHub Release
automatically from the tag: it extracts the `[X.Y.Z]` section from
`CHANGELOG.md` and uses it as the release notes. Verify the release appears
under **Releases** with the right title (`v0.2.0`) and notes. (Only if the
workflow failed should you draft the release manually in the GitHub UI.)

Publishing the release triggers the CI wheel build jobs, `build-sdist`, and `publish`
automatically (the workflow responds to `release: published`). The PyPI upload
uses Trusted Publishing via GitHub OIDC, so no `PYPI_API_TOKEN` secret is used.

---

## Step 7 — Monitor CI and verify PyPI

1. Watch the **Actions** tab: the release wheel jobs, `build-sdist`, `publish` (PyPI), `publish-cratesio` (crates.io), the **wasm-publish** workflow (npm), and the **flutter-publish** workflow (pub.dev).
2. After the `publish` job succeeds, verify the package is live:

```bash
pip install ferro-ta==0.2.0
python -c "import ferro_ta; print(ferro_ta.__version__ if hasattr(ferro_ta,'__version__') else 'ok')"
```

For version-specific verification, also check at least one install on each
supported Python line, for example:

```bash
uv venv --python 3.13 .venv-313
. .venv-313/bin/activate
uv pip install ferro-ta==0.2.0
python -c "import ferro_ta; print(ferro_ta.SMA([1.0, 2.0, 3.0], 2))"
```

3. If anything fails: fix the issue, bump to a patch version (`0.2.1`), and repeat.

---


---

## Hotfix releases

For urgent bug fixes on a released version:

1. Branch from the release tag: `git checkout -b hotfix/0.1.1 v0.1.0`
2. Apply the fix, bump to `0.1.1`, update CHANGELOG.
3. Merge the branch into `main`.
4. Tag and release as above.

---

> **Note:** `ferro_ta_core` is published to crates.io automatically by the CI job `publish-cratesio` when you publish a release (requires `CARGO_REGISTRY_TOKEN` secret).

> **Note (Flutter / pub.dev):** the `flutter-publish` workflow builds the native
> libraries for every platform, bundles them into the `ferro_ta` package, and
> runs `dart pub publish --force` using GitHub OIDC — no stored credential.
>
> It is triggered by the **`vX.Y.Z` tag push**, not by the GitHub Release:
> pub.dev "rejects publishing from GitHub Actions triggered without a tag"
> ([docs](https://dart.dev/tools/pub/automated-publishing)). A manual
> `workflow_dispatch` can only `--dry-run`; it can never publish.
>
> **One-time setup, required first:** pub.dev can only automate publishing of
> *existing* packages — "To create a new package, you must publish the first
> version using `dart pub publish`." So publish `ferro_ta` once by hand from
> `flutter/`, then enable **Automated publishing** in the pub.dev package admin,
> bound to this repository with the `v{{version}}` tag pattern. (A new package
> also cannot be published directly to a verified publisher — publish to a
> Google Account first, then transfer.)

---

## See also

- [VERSIONING.md](VERSIONING.md) — versioning policy and breaking-change rules
- [CHANGELOG.md](CHANGELOG.md) — changelog history
- [CONTRIBUTING.md](CONTRIBUTING.md) — development setup and PR guidelines
