# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for
ferro-ta, documenting design choices that future contributors might
otherwise have to reverse-engineer from git history.

Format: [Michael Nygard's ADR template](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions).
Each ADR has `Context`, `Decision`, and `Consequences` sections.

## Index

| # | Title | Status |
|---|---|---|
| [0001](0001-cargo-audit-removal.md) | Remove standalone cargo-audit, rely on cargo-deny advisories | Accepted |
| [0002](0002-mcp-optional-feature.md) | MCP support is an optional extra, not a core dependency | Accepted |
| [0003](0003-gpu-backend-optional.md) | GPU backend (torch) is an optional extra | Accepted |
| [0004](0004-dtw-algorithm-choice.md) | Classic DP dynamic time warping, not FastDTW | Accepted |
| [0005](0005-ta-lib-compatibility-classes.md) | TA-Lib parity is graded, not binary | Accepted |
| [0006](0006-cpu-coverage-strategy.md) | Runtime CPU dispatch for broad coverage, not a pinned target-cpu | Accepted |

## When to write an ADR

Write one when a decision:

- Constrains future work in a non-obvious way ("we can't just switch to X
  because...").
- Would otherwise be re-litigated every time a new contributor shows up.
- Involves a trade-off where the "wrong" side has real advocates.

Don't write ADRs for routine refactors, naming conventions, or anything
already covered by CONTRIBUTING.md.
