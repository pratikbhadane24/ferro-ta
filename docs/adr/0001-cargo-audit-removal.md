# ADR 0001: Remove standalone cargo-audit, rely on cargo-deny advisories

**Status:** Accepted
**Date:** 2026-04-14
**Deciders:** ferro-ta maintainers
**Commit:** `388dc05` (CI: remove cargo-audit step from CI workflows)

## Context

Until commit `388dc05`, CI ran both `cargo audit` and `cargo deny check` on
every push. `cargo audit` consults the RustSec advisory database directly;
`cargo deny` can do the same thing via its `[advisories]` table (since
`cargo-deny` 0.14). Running both produced duplicate advisory scans on every
PR and doubled the CI time for security checks.

## Decision

Remove the standalone `cargo audit` job and consolidate advisory scanning
into `cargo deny check`, which is already configured in `deny.toml` with a
`[advisories]` section pointing at the RustSec database:

```toml
[advisories]
db-urls = ["https://github.com/rustsec/advisory-db"]
version = 2
ignore = []
```

`cargo deny check` (as invoked by `EmbarkStudios/cargo-deny-action@v2` in
`.github/workflows/ci-rust.yml`) runs **all** checks by default: `bans`,
`licenses`, `advisories`, and `sources`. Advisory scanning is therefore still
active on every PR — the coverage did not regress, only the job count.

## Consequences

**Positive:**
- Single tool for supply-chain checks; simpler CI.
- License, bans, sources, and advisories all fail the same job, so contributors
  see all issues in one place.

**Negative:**
- Consumers looking for a `cargo audit` step by name in the workflow won't
  find one. This ADR exists to document where advisory scanning now lives.

**Mitigations:**
- The `ci-rust.yml` `cargo-deny` job is listed in the `ci-complete` required
  gate, so an advisory failure still blocks merge.
- The CHANGELOG `[Unreleased]` section calls out the consolidation under
  "Security" for user visibility.
