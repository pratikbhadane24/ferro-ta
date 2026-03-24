# MCP Server

ferro-ta ships an optional MCP (Model Context Protocol) server built on the
official Python SDK's FastMCP layer. The server now exposes the broad public
ferro-ta callable surface instead of a tiny hand-picked subset.

That means MCP clients can use:

- Exact top-level ferro-ta exports such as `SMA`, `RSI`, `MACD`, `about`,
  `methods`, `info`, `benchmark`, and `traced`
- Non-top-level public tools such as `compute_indicator`, `run_backtest`,
  `check_cross`, `aggregate_ticks`, `TickAggregator`, and `AlertManager`
- Legacy lowercase convenience aliases: `sma`, `ema`, `rsi`, `macd`,
  and `backtest`
- Generic instance tools for stateful classes and stored callables:
  `list_instances`, `describe_instance`, `call_instance_method`,
  `call_stored_callable`, and `delete_instance`

---

## Installation

Install the optional MCP extra:

```bash
pip install "ferro-ta[mcp]"
```

If you are working from this repository, you can install the same extra into
the project environment with:

```bash
uv sync --extra mcp
```

---

## Running the server

Run the server over stdio:

```bash
python -m ferro_ta.mcp
```

The command exits immediately with an install hint if the optional `mcp`
dependency is missing.

---

## Connect in Cursor

Add the server to Cursor's MCP settings:

```json
{
  "mcpServers": {
    "ferro-ta": {
      "command": "python",
      "args": ["-m", "ferro_ta.mcp"],
      "description": "ferro-ta technical analysis tools"
    }
  }
}
```

You can place this in your user settings JSON or in a workspace-level
`.cursor/mcp.json`.

---

## Tool naming

The MCP server prefers the real ferro-ta API names.

- Use exact public names when possible, for example `SMA`, `MACD`,
  `compute_indicator`, `trade_stats`, `TickAggregator`, or `AlertManager`
- Use the legacy lowercase aliases only when you want the old MCP-friendly
  shortcuts and result shapes
- Use `about`, `methods`, `indicators`, and `info` to discover what is
  available from inside an MCP client

---

## Stateful classes and object references

Class tools return stored object references instead of plain text placeholders.
For example, calling `TickAggregator` or `AlertManager` returns a payload like:

```json
{
  "instance_id": "tickaggregator-0001",
  "type": "ferro_ta.data.aggregation.TickAggregator",
  "repr": "TickAggregator(rule='tick:2')"
}
```

Use that `instance_id` with:

- `describe_instance` to inspect the stored object and list public methods
- `call_instance_method` to call methods like `aggregate`, `update`,
  `run_backtest`, or `to_dict`
- `delete_instance` to remove stored objects when you are done

If a tool returns a stored callable, use `call_stored_callable`.

---

## Callable references

Some ferro-ta APIs accept other callables, for example `benchmark`,
`log_call`, `traced`, or `multi_timeframe(indicator=...)`.

Pass public ferro-ta callables using:

```json
{"callable": "SMA"}
```

Pass stored objects using:

```json
{"instance_id": "function-0001"}
```

---

## Example prompts

Once connected, you can ask an MCP-compatible client things like:

> "Run `SMA` with `close=[100, 101, 102, 103, 104]` and `timeperiod=3`."

> "Use `compute_indicator` to calculate `MACD` for this close series."

> "Call `about` and summarize the current ferro-ta API surface."

> "Create a `TickAggregator` with `rule='tick:50'`, aggregate this tick data,
> then delete the instance."

> "Benchmark `SMA` over this price series using a callable reference."

---

## Programmatic use

Use the server entrypoint:

```python
from ferro_ta.mcp import create_server

server = create_server()
# server.run(transport="stdio")
```

Or call the handlers directly without starting the server:

```python
from ferro_ta.mcp import handle_call_tool, handle_list_tools
import json

tools = handle_list_tools()
print(len(tools["tools"]))

close = [100, 101, 102, 103, 104]
result = handle_call_tool("SMA", {"close": close, "timeperiod": 3})
print(json.loads(result["content"][0]["text"]))

aggregator = json.loads(
    handle_call_tool("TickAggregator", {"rule": "tick:2"})["content"][0]["text"]
)
bars = handle_call_tool(
    "call_instance_method",
    {
        "instance_id": aggregator["instance_id"],
        "method": "aggregate",
        "args": [{"price": [1, 2, 3, 4], "size": [1, 1, 1, 1]}],
    },
)
print(json.loads(bars["content"][0]["text"]))
```

---

## See also

- `python -m ferro_ta.mcp` - stdio MCP entrypoint
- `ferro_ta.mcp.create_server()` - FastMCP server factory
- `ferro_ta.tools.api_info` - API discovery helpers used by the MCP catalog
- `ferro_ta.tools` - stable wrappers such as `compute_indicator`
- `docs/agentic.md` - workflow and agent integration notes
