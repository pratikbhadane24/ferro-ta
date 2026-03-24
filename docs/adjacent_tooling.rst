Adjacent Tooling
================

These modules are useful, but they are secondary to ferro-ta's core identity as
a Python technical analysis library.

.. list-table::
   :header-rows: 1

   * - Area
     - Status
     - What it is
   * - Derivatives analytics
     - Adjacent
     - Options pricing, Greeks, implied volatility helpers, futures basis,
       curve, and roll utilities. See :doc:`derivatives`.
   * - Agent workflow wrappers
     - Adjacent
     - Tool and workflow helpers for agent-style integrations. See
       `docs/agentic.md <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/agentic.md>`_.
   * - MCP server
     - Experimental or adjacent
     - FastMCP-based server exposing selected ferro-ta capabilities to
       MCP-compatible clients. See
       `docs/mcp.md <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/mcp.md>`_.
   * - WASM package
     - Experimental
     - Browser and Node.js package with a smaller indicator subset. See
       `wasm/README.md <https://github.com/pratikbhadane24/ferro-ta/blob/main/wasm/README.md>`_.
   * - GPU backend
     - Experimental
     - Optional PyTorch-backed acceleration for a limited subset of indicators.
       See `docs/gpu-backend.md <https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/gpu-backend.md>`_.
   * - Plugin system
     - Experimental
     - Registry and plugin packaging model for custom indicators. See
       :doc:`plugins`.

How to read the project
-----------------------

When evaluating ferro-ta:

- Start with the core library docs, migration guide, support matrix, and benchmarks.
- Treat adjacent tooling as opt-in layers, not as proof that the core indicator
  library is broader or more stable than it is.
- Check the release notes and stability policy before depending on experimental
  surfaces in production.
