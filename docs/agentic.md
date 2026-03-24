# Agentic Workflow and Tools

ferro-ta provides stable tool wrappers and a workflow orchestrator that make
it easy to integrate with AI agents, LangChain, LlamaIndex, or any
framework that supports function calling.

---

## Overview

The agentic API consists of two modules:

| Module | Purpose |
|--------|---------|
| `ferro_ta.tools` | Stable, documented functions for agent wrapping |
| `ferro_ta.workflow` | End-to-end pipeline: indicators → strategy → alerts |

---

## `ferro_ta.tools` — Tool wrappers

```python
from ferro_ta.tools import compute_indicator, run_backtest, list_indicators, describe_indicator
import numpy as np

close = np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, 200)) * 100

# Compute any indicator by name
sma = compute_indicator("SMA", close, timeperiod=20)
rsi = compute_indicator("RSI", close, timeperiod=14)
bb  = compute_indicator("BBANDS", close, timeperiod=20)  # returns dict

# Run a backtest
summary = run_backtest("rsi_30_70", close)
print(f"Final equity: {summary['final_equity']:.4f}")
print(f"Trades: {summary['n_trades']}")

# List all indicators
names = list_indicators()   # sorted list of strings

# Describe an indicator (returns first paragraph of docstring)
desc = describe_indicator("RSI")
```

### Function signatures

```python
def compute_indicator(name: str, *args, **kwargs) -> ndarray | dict:
    ...

def run_backtest(strategy: str, close, commission_per_trade=0.0, slippage_bps=0.0, **kwargs) -> dict:
    ...

def list_indicators() -> list[str]:
    ...

def describe_indicator(name: str) -> str:
    ...
```

---

## `ferro_ta.workflow` — End-to-end pipeline

```python
from ferro_ta.workflow import Workflow
import numpy as np

rng = np.random.default_rng(42)
close = np.cumprod(1 + rng.normal(0, 0.01, 200)) * 100

result = (
    Workflow()
    .add_indicator("sma_20", "SMA", timeperiod=20)
    .add_indicator("rsi_14", "RSI", timeperiod=14)
    .add_strategy("rsi_30_70")
    .add_alert("rsi_14", level=30.0, direction=-1)  # alert when RSI crosses below 30
    .run(close)
)

print(result.keys())
# dict_keys(['sma_20', 'rsi_14', 'backtest', 'alert_rsi_14_30_-1'])
```

### Functional interface

```python
from ferro_ta.workflow import run_pipeline

result = run_pipeline(
    close,
    indicators={
        "sma_20": {"name": "SMA", "timeperiod": 20},
        "rsi_14": {"name": "RSI", "timeperiod": 14},
    },
    strategy="rsi_30_70",
    alert_indicator="rsi_14",
    alert_level=30.0,
    alert_direction=-1,
)
```

---

## LangChain integration

Wrap the tools as LangChain `Tool` objects:

```python
from langchain.tools import Tool
from ferro_ta.tools import compute_indicator, run_backtest, list_indicators
import numpy as np
import json

def _compute_tool(input_str: str) -> str:
    """Parse JSON input and compute an indicator."""
    args = json.loads(input_str)
    name = args.pop("name")
    close = np.asarray(args.pop("close"), dtype=np.float64)
    result = compute_indicator(name, close, **args)
    if isinstance(result, dict):
        return json.dumps({k: v.tolist() for k, v in result.items()})
    return json.dumps(result.tolist())

def _backtest_tool(input_str: str) -> str:
    args = json.loads(input_str)
    close = np.asarray(args.pop("close"), dtype=np.float64)
    strategy = args.pop("strategy", "rsi_30_70")
    summary = run_backtest(strategy, close, **args)
    return json.dumps(summary)

tools = [
    Tool(
        name="compute_indicator",
        func=_compute_tool,
        description=(
            'Compute a technical indicator. Input JSON: {"name": "SMA", '
            '"close": [...], "timeperiod": 14}'
        ),
    ),
    Tool(
        name="run_backtest",
        func=_backtest_tool,
        description=(
            'Run a backtest. Input JSON: {"strategy": "rsi_30_70", '
            '"close": [...]}'
        ),
    ),
    Tool(
        name="list_indicators",
        func=lambda _: json.dumps(list_indicators()),
        description="List all available indicator names. No input required.",
    ),
]
```

---

## Scheduling

### Run once

```python
python examples/run_workflow.py
```

### Run every N minutes (cron)

Add to your crontab:

```
*/15 * * * * /usr/bin/python /path/to/examples/run_workflow.py >> /var/log/ferro_ta.log 2>&1
```

### Run on a schedule with `schedule` library

```python
import schedule
import time

def job():
    import numpy as np
    from ferro_ta.workflow import run_pipeline
    # fetch latest prices here ...
    close = np.ones(100)  # replace with real data
    result = run_pipeline(close, indicators={"rsi": {"name": "RSI", "timeperiod": 14}})
    print(result)

schedule.every(15).minutes.do(job)
while True:
    schedule.run_pending()
    time.sleep(1)
```

---

## See also

- `ferro_ta.tools` — module source.
- `ferro_ta.workflow` — module source.
- `docs/mcp.md` — MCP server for MCP-compatible clients.
- `ferro_ta.backtest` — backtest harness.
- `ferro_ta.registry` — indicator registry.
