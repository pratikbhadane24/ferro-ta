"""Entry point so the MCP server can be run as ``python -m ferro_ta.mcp``."""

from ferro_ta.mcp import run_server

if __name__ == "__main__":
    run_server()  # pragma: no cover
