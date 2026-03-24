"""
ferro_ta.mcp — Model Context Protocol (MCP) Server
==================================================

An MCP server that exposes ferro_ta indicators and backtest tools to
AI agents (e.g. Claude in Cursor, LangChain, OpenAI function calling).

Running the server
------------------
Start the server directly::

    python -m ferro_ta.mcp

Or with ``uvicorn`` / ``mcp`` runner if the official MCP SDK is installed::

    uvicorn ferro_ta.mcp:app --port 8765

Cursor integration
------------------
Add the following to your Cursor MCP settings
(``~/.cursor/mcp.json`` or workspace ``.cursor/mcp.json``)::

    {
      "mcpServers": {
        "ferro-ta": {
          "command": "python",
          "args": ["-m", "ferro_ta.mcp"],
          "description": "ferro_ta technical analysis tools"
        }
      }
    }

After reloading Cursor, you can ask the AI assistant things like:

* "Compute SMA(14) on this price series: [100, 102, ...]"
* "Run a backtest with RSI 30/70 strategy on this data"
* "list all available indicators"

See ``docs/mcp.md`` for the full guide.

Install optional dependency
---------------------------
The MCP server requires the ``mcp`` SDK::

    pip install ferro-ta[mcp]

or::

    pip install "mcp>=1.0"

Tools exposed
-------------
* ``sma``            — Simple Moving Average
* ``ema``            — Exponential Moving Average
* ``rsi``            — Relative Strength Index
* ``macd``           — MACD line, signal, histogram
* ``backtest``       — Run a vectorized backtest
* ``list_indicators``— list all registered indicators
* ``describe_indicator`` — Describe an indicator
"""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np

import ferro_ta as ft
from ferro_ta.tools import (
    compute_indicator,
    describe_indicator,
    list_indicators,
    run_backtest,
)

__all__ = ["run_server", "handle_list_tools", "handle_call_tool"]

# ---------------------------------------------------------------------------
# Tool definitions (JSON-schema style)
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "sma",
        "description": "Compute the Simple Moving Average (SMA) of a price series.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "close": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Close price series.",
                },
                "timeperiod": {
                    "type": "integer",
                    "description": "Look-back period (default 14).",
                    "default": 14,
                },
            },
            "required": ["close"],
        },
    },
    {
        "name": "ema",
        "description": "Compute the Exponential Moving Average (EMA) of a price series.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "close": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Close price series.",
                },
                "timeperiod": {
                    "type": "integer",
                    "description": "Look-back period (default 14).",
                    "default": 14,
                },
            },
            "required": ["close"],
        },
    },
    {
        "name": "rsi",
        "description": "Compute the Relative Strength Index (RSI) of a price series.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "close": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Close price series.",
                },
                "timeperiod": {
                    "type": "integer",
                    "description": "Look-back period (default 14).",
                    "default": 14,
                },
            },
            "required": ["close"],
        },
    },
    {
        "name": "macd",
        "description": (
            "Compute MACD (Moving Average Convergence/Divergence). "
            "Returns macd line, signal line, and histogram."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "close": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Close price series.",
                },
                "fastperiod": {
                    "type": "integer",
                    "description": "Fast EMA period (default 12).",
                    "default": 12,
                },
                "slowperiod": {
                    "type": "integer",
                    "description": "Slow EMA period (default 26).",
                    "default": 26,
                },
                "signalperiod": {
                    "type": "integer",
                    "description": "Signal EMA period (default 9).",
                    "default": 9,
                },
            },
            "required": ["close"],
        },
    },
    {
        "name": "backtest",
        "description": (
            "Run a vectorized backtest on close prices using a named strategy. "
            "Returns final equity, number of trades, and the equity curve."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "close": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Close price series (at least 2 bars).",
                },
                "strategy": {
                    "type": "string",
                    "description": (
                        "Strategy name: 'rsi_30_70', 'sma_crossover', or 'macd_crossover'."
                    ),
                    "default": "rsi_30_70",
                },
                "commission_per_trade": {
                    "type": "number",
                    "description": "Fixed commission per trade (default 0).",
                    "default": 0.0,
                },
                "slippage_bps": {
                    "type": "number",
                    "description": "Slippage in basis points (default 0).",
                    "default": 0.0,
                },
            },
            "required": ["close"],
        },
    },
    {
        "name": "list_indicators",
        "description": "list all available indicator names registered in ferro_ta.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "describe_indicator",
        "description": "Return a description of a named ferro_ta indicator.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Indicator name (e.g. 'SMA', 'RSI', 'BBANDS').",
                }
            },
            "required": ["name"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def handle_list_tools() -> dict[str, Any]:
    """Return the ListTools response."""
    return {"tools": _TOOLS}


def handle_call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a CallTool request and return the result.

    Parameters
    ----------
    name : str
        Tool name (one of the ``_TOOLS`` entries).
    arguments : dict
        Tool arguments as provided by the MCP client.

    Returns
    -------
    dict
        MCP content response with type ``"text"`` containing the JSON result.
    """
    try:
        if name in ("sma", "ema", "rsi"):
            close = np.asarray(arguments["close"], dtype=np.float64)
            timeperiod = int(arguments.get("timeperiod", 14))
            result = compute_indicator(name.upper(), close, timeperiod=timeperiod)
            # Replace NaN with None for JSON serialisation
            payload = [None if np.isnan(v) else float(v) for v in result]
            return {"content": [{"type": "text", "text": json.dumps(payload)}]}

        elif name == "macd":
            close = np.asarray(arguments["close"], dtype=np.float64)
            kwargs = {
                "fastperiod": int(arguments.get("fastperiod", 12)),
                "slowperiod": int(arguments.get("slowperiod", 26)),
                "signalperiod": int(arguments.get("signalperiod", 9)),
            }
            result = compute_indicator("MACD", close, **kwargs)
            assert isinstance(result, dict)
            macd_payload = {
                k: [None if np.isnan(v) else float(v) for v in arr]
                for k, arr in result.items()
            }
            return {"content": [{"type": "text", "text": json.dumps(macd_payload)}]}

        elif name == "backtest":
            close = np.asarray(arguments["close"], dtype=np.float64)
            strategy = str(arguments.get("strategy", "rsi_30_70"))
            commission = float(arguments.get("commission_per_trade", 0.0))
            slippage = float(arguments.get("slippage_bps", 0.0))
            summary = run_backtest(
                strategy,
                close,
                commission_per_trade=commission,
                slippage_bps=slippage,
            )
            # JSON-serialise (equity is already a list)
            return {"content": [{"type": "text", "text": json.dumps(summary)}]}

        elif name == "list_indicators":
            return {
                "content": [{"type": "text", "text": json.dumps(list_indicators())}]
            }

        elif name == "describe_indicator":
            ind_name = str(arguments["name"])
            description = describe_indicator(ind_name)
            return {"content": [{"type": "text", "text": description}]}

        else:
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Unknown tool: {name!r}"}],
            }

    except Exception as exc:
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Error: {exc}"}],
        }


# ---------------------------------------------------------------------------
# Stdio MCP server (JSON-RPC over stdin/stdout)
# ---------------------------------------------------------------------------


def run_server() -> None:  # pragma: no cover
    """Run the MCP server over stdin/stdout (JSON-RPC 2.0 protocol).

    This implements a minimal MCP server that handles ``initialize``,
    ``tools/list``, and ``tools/call`` messages.  It is compatible with the
    MCP client built into Cursor (as of early 2025) and with the official
    `mcp` Python SDK client.

    The server reads one JSON-RPC message per line from stdin and writes
    one response per line to stdout.
    """
    # Try to use official mcp SDK if available
    try:
        _run_with_sdk()
    except ImportError:
        _run_stdio_fallback()


def _run_with_sdk() -> None:  # pragma: no cover
    """Run using the official MCP Python SDK."""
    import mcp  # type: ignore[import]
    import mcp.server.stdio  # type: ignore[import]
    from mcp.server import Server  # type: ignore[import]
    from mcp.types import (  # type: ignore[import]
        CallToolRequest,
        ListToolsRequest,
    )

    app = Server("ferro-ta")

    @app.list_tools()
    async def _list_tools(_req: ListToolsRequest):
        return handle_list_tools()["tools"]

    @app.call_tool()
    async def _call_tool(req: CallToolRequest):
        return handle_call_tool(req.params.name, req.params.arguments or {})

    import asyncio

    asyncio.run(mcp.server.stdio.stdio_server(app))


def _run_stdio_fallback() -> None:  # pragma: no cover
    """Minimal stdin/stdout JSON-RPC MCP implementation (no SDK required)."""
    import json as _json

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            msg = _json.loads(raw_line)
        except _json.JSONDecodeError:
            continue

        msg_id = msg.get("id")
        method = msg.get("method", "")

        if method == "initialize":
            resp = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "ferro-ta", "version": ft.__version__},
                },
            }
        elif method == "tools/list":
            resp = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": handle_list_tools(),
            }
        elif method == "tools/call":
            params = msg.get("params", {})
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            resp = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": handle_call_tool(tool_name, arguments),
            }
        else:
            resp = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method!r}"},
            }

        sys.stdout.write(_json.dumps(resp) + "\n")
        sys.stdout.flush()
