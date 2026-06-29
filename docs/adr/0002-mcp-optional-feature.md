# ADR 0002: MCP support is an optional extra, not a core dependency

**Status:** Accepted
**Date:** 2026-04-14

## Context

ferro-ta ships a Model Context Protocol (MCP) integration that exposes
indicators as tools for LLM agents. MCP requires the `mcp>=1.0` Python
package, which pulls in `pydantic`, `jsonschema`, and transport libraries
that most ferro-ta users (quants, backtesters, data pipelines) do not need.

## Decision

MCP lives behind an optional extra: `pip install "ferro-ta[mcp]"`. The
default `pip install ferro-ta` does not install the MCP dependency tree.
Imports from `ferro_ta.mcp` raise a helpful `ModuleNotFoundError` that
points users to the extra.

## Consequences

**Positive:**
- Default install footprint stays small (numpy + the compiled wheel).
- Non-MCP users are not exposed to MCP's transitive dependencies
  (pydantic v2, jsonschema, anyio).
- MCP package can evolve its API without forcing every ferro-ta release.

**Negative:**
- Users who want MCP have to know about the extra. Mitigation: the
  `docs/guides/mcp.md` guide is the first link in the MCP README section.
- CI must run a separate install step to exercise MCP code paths.

## Non-goals

- ferro-ta does not aim to re-implement MCP primitives. We depend on the
  official `mcp` package and expose indicators as MCP tools — not the
  other way around.
