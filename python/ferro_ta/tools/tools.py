"""
ferro_ta.tools — Stable Tool Wrappers for Agent / LLM Integration
=================================================================

Provides stable, well-documented functions that are easy to wrap as
LangChain/LlamaIndex/OpenAI Function tools or to call from automated agents.

All functions have clear signatures, descriptive docstrings, and return
JSON-serializable types so that agent frameworks can inspect and call them
without special handling.

See ``docs/agentic.md`` for the full agentic workflow guide, LangChain
integration examples, and scheduling instructions.

Quick start
-----------
>>> import numpy as np
>>> from ferro_ta.tools import compute_indicator, run_backtest, list_indicators
>>>
>>> close = np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, 100)) * 100
>>>
>>> # Compute a single indicator by name
>>> result = compute_indicator("SMA", close, timeperiod=14)
>>>
>>> # Run a backtest
>>> summary = run_backtest("rsi_30_70", close)
>>> print(summary["final_equity"])

API
---
compute_indicator(name, *args, **kwargs) → array or dict
    Compute a built-in or registered indicator by name.

run_backtest(strategy, close, **kwargs) → dict
    Run a backtest and return a summary dict.

list_indicators() → list[str]
    list all registered indicator names.

describe_indicator(name) → str
    Return the docstring of a registered indicator (or a summary).
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "compute_indicator",
    "run_backtest",
    "list_indicators",
    "describe_indicator",
]


def compute_indicator(
    name: str,
    *args: ArrayLike,
    **kwargs: Any,
) -> Union[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """Compute a named indicator and return the result.

    Delegates to the ferro_ta registry so that both built-in and custom
    indicators can be called by name.

    Parameters
    ----------
    name : str
        Indicator name (e.g. ``"SMA"``, ``"RSI"``, ``"BBANDS"``).
        Case-sensitive; use :func:`list_indicators` to see all names.
    *args : array-like
        Positional data arrays forwarded to the indicator (e.g. close, high).
    **kwargs
        Parameter keyword arguments forwarded to the indicator
        (e.g. ``timeperiod=14``).

    Returns
    -------
    ndarray or dict of str → ndarray
        For single-output indicators, returns a 1-D ``numpy.ndarray``.
        For multi-output indicators (e.g. BBANDS, MACD), returns a dict
        mapping output names to arrays.  The dict keys follow TA-Lib
        conventions where known (``"upper"``/``"middle"``/``"lower"`` for
        BBANDS; ``"macd"``/``"signal"``/``"hist"`` for MACD; etc.).

    Raises
    ------
    ferro_ta.registry.FerroTARegistryError
        If *name* is not a known indicator.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.tools import compute_indicator
    >>> close = np.linspace(100, 110, 20)
    >>> result = compute_indicator("SMA", close, timeperiod=5)
    >>> result.shape
    (20,)
    >>> bb = compute_indicator("BBANDS", close, timeperiod=5)
    >>> sorted(bb.keys())
    ['lower', 'middle', 'upper']
    """
    from ferro_ta.core.registry import run as _registry_run

    raw = _registry_run(name, *args, **kwargs)

    if isinstance(raw, tuple):
        # Multi-output: try to map to named keys for well-known indicators
        _multi_keys: dict[str, list[str]] = {
            "BBANDS": ["upper", "middle", "lower"],
            "MACD": ["macd", "signal", "hist"],
            "MACDEXT": ["macd", "signal", "hist"],
            "MACDFIX": ["macd", "signal", "hist"],
            "STOCH": ["slowk", "slowd"],
            "STOCHF": ["fastk", "fastd"],
            "STOCHRSI": ["fastk", "fastd"],
            "AROON": ["aroondown", "aroonup"],
            "HT_PHASOR": ["inphase", "quadrature"],
            "HT_SINE": ["sine", "leadsine"],
            "MAMA": ["mama", "fama"],
        }
        keys = _multi_keys.get(name.upper())
        if keys and len(keys) == len(raw):
            return {k: np.asarray(v, dtype=np.float64) for k, v in zip(keys, raw)}
        # Fallback: use integer keys
        return {str(i): np.asarray(v, dtype=np.float64) for i, v in enumerate(raw)}

    return np.asarray(raw, dtype=np.float64)


def run_backtest(
    strategy: str,
    close: ArrayLike,
    commission_per_trade: float = 0.0,
    slippage_bps: float = 0.0,
    **strategy_kwargs: Any,
) -> dict[str, Any]:
    """Run a named backtest strategy and return a summary dictionary.

    This is a convenience wrapper around :func:`ferro_ta.backtest.backtest`
    that returns a JSON-serializable summary dict rather than a
    ``BacktestResult`` object, making it easy to use from agent tools.

    Parameters
    ----------
    strategy : str
        Name of the built-in strategy: ``"rsi_30_70"``, ``"sma_crossover"``,
        or ``"macd_crossover"``.
    close : array-like
        Close prices (1-D, at least 2 bars).
    commission_per_trade : float
        Fixed commission deducted from equity on each position change.
    slippage_bps : float
        Slippage in basis points applied on position-change bars.
    **strategy_kwargs
        Extra kwargs forwarded to the strategy function
        (e.g. ``timeperiod=14``, ``oversold=25``).

    Returns
    -------
    dict
        Summary with the following keys:

        * ``"strategy"`` — the strategy name used.
        * ``"n_bars"`` — number of price bars.
        * ``"n_trades"`` — number of position changes.
        * ``"final_equity"`` — terminal equity value (start = 1.0).
        * ``"max_drawdown"`` — maximum drawdown fraction (0–1, positive value
          represents the magnitude of loss).
        * ``"equity"`` — equity curve as a Python list of floats.
        * ``"signals"`` — signal array as a Python list.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.tools import run_backtest
    >>> close = np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, 100)) * 100
    >>> summary = run_backtest("rsi_30_70", close)
    >>> isinstance(summary["final_equity"], float)
    True
    """
    from ferro_ta.analysis.backtest import backtest as _backtest

    result = _backtest(
        close,
        strategy=strategy,
        commission_per_trade=commission_per_trade,
        slippage_bps=slippage_bps,
        **strategy_kwargs,
    )

    equity = np.asarray(result.equity, dtype=np.float64)
    # Compute max drawdown
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / np.where(running_max > 0, running_max, 1.0)
    max_dd = float(np.nanmax(drawdowns)) if len(drawdowns) > 0 else 0.0

    return {
        "strategy": strategy,
        "n_bars": len(result.signals),
        "n_trades": result.n_trades,
        "final_equity": result.final_equity,
        "max_drawdown": max_dd,
        "equity": equity.tolist(),
        "signals": np.asarray(result.signals, dtype=np.float64).tolist(),
    }


def list_indicators() -> list[str]:
    """Return a sorted list of all registered indicator names.

    Includes both built-in ferro_ta indicators and any custom indicators
    registered via :func:`ferro_ta.registry.register`.

    Returns
    -------
    list of str
        Sorted list of indicator names (e.g. ``["AD", "ADOSC", "ADX", …]``).

    Examples
    --------
    >>> from ferro_ta.tools import list_indicators
    >>> names = list_indicators()
    >>> "SMA" in names
    True
    >>> "RSI" in names
    True
    """
    from ferro_ta.core.registry import list_indicators as _list

    return _list()


def describe_indicator(name: str) -> str:
    """Return a human-readable description of a registered indicator.

    Looks up the indicator's docstring and returns the first paragraph (up to
    the first blank line) so it can be used in agent prompts or tool
    descriptions.

    Parameters
    ----------
    name : str
        Indicator name (case-sensitive).  Use :func:`list_indicators` to get
        valid names.

    Returns
    -------
    str
        The first paragraph of the indicator's docstring, or a fallback
        message if no docstring is available.

    Raises
    ------
    ferro_ta.registry.FerroTARegistryError
        If *name* is not a known indicator.

    Examples
    --------
    >>> from ferro_ta.tools import describe_indicator
    >>> desc = describe_indicator("SMA")
    >>> isinstance(desc, str) and len(desc) > 0
    True
    """
    from ferro_ta.core.registry import get as _get

    func = _get(name)
    doc = getattr(func, "__doc__", None) or ""
    if not doc.strip():
        return f"{name}: no description available."

    # Return only the first paragraph (before the first blank line)
    lines = doc.strip().splitlines()
    para: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "" and para:
            break
        para.append(stripped)

    return " ".join(para).strip() or f"{name}: no description available."
