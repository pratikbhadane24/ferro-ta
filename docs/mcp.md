# MCP Server — Connect ferro-ta in Cursor

ferro-ta ships an MCP (Model Context Protocol) server that exposes
indicators and backtest tools to AI agents.  This guide shows how to run
the server and connect it to Cursor or any MCP-compatible client.

---

## Installation

The MCP server requires no additional dependencies beyond ferro_ta itself.
For the full MCP SDK integration (recommended), install the optional extra:

```bash
pip install "ferro-ta[mcp]"
```

or install the `mcp` package separately:

```bash
pip install "mcp>=1.0"
```

---

## Running the server

```bash
python -m ferro_ta.mcp
```

The server listens on stdin/stdout using JSON-RPC 2.0 (the MCP protocol).

---

## Connect in Cursor

1. Open Cursor settings (Command Palette → "Open User Settings (JSON)").
2. Find or create the `mcpServers` section:

```json
{
  "mcpServers": {
    "ferro-ta": {
      "command": "python",
      "args": ["-m", "ferro_ta.mcp"],
      "description": "ferro_ta — Technical Analysis MCP server"
    }
  }
}
```

3. Reload Cursor (Command Palette → "Developer: Reload Window").
4. The ferro-ta tools will appear in the Tools panel.

### Workspace-level config

You can also add the config to your project's `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "ferro-ta": {
      "command": "python",
      "args": ["-m", "ferro_ta.mcp"]
    }
  }
}
```

---

## Example prompts

Once connected, you can ask Claude (or any MCP-enabled AI) things like:

> "Compute RSI(14) on this price series: [100, 102, 101, 105, 108, 104, 107]"

> "Run a backtest with the rsi_30_70 strategy on [100, 101, 99, 103, 106, 102, 108, 105, 109, 112, 108, 111]"

> "List all available ferro_ta indicators"

> "What does the SMA indicator do?"

---

## Available tools

| Tool | Description |
|------|-------------|
| `sma` | Simple Moving Average |
| `ema` | Exponential Moving Average |
| `rsi` | Relative Strength Index |
| `macd` | MACD line, signal, histogram |
| `backtest` | Vectorized backtest (rsi_30_70, sma_crossover, macd_crossover) |
| `list_indicators` | List all registered indicators |
| `describe_indicator` | Describe a named indicator |

---

## Programmatic use (Python client)

You can also use the MCP handlers directly in Python without the server:

```python
from ferro_ta.mcp import handle_list_tools, handle_call_tool
import numpy as np

# List tools
tools = handle_list_tools()
print([t["name"] for t in tools["tools"]])

# Call RSI
close = list(np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, 50)) * 100)
result = handle_call_tool("rsi", {"close": close, "timeperiod": 14})
print(result)
```

---

## See also

- `ferro_ta.mcp` — module source.
- `ferro_ta.tools` — underlying tool functions.
- `docs/agentic.md` — LangChain and workflow integration.
