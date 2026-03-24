"""
ferro_ta.api_info — API discovery helpers.

Provides :func:`indicators`, :func:`methods`, :func:`about`, and :func:`info`
for exploring the ferro_ta public API without reading source code.

Usage
-----
>>> import ferro_ta
>>> ferro_ta.indicators()                      # all indicators, sorted
>>> ferro_ta.indicators(category="momentum")   # filter by category
>>> ferro_ta.methods()                         # public callables across modules
>>> ferro_ta.about()["version"]               # package metadata summary
>>> ferro_ta.info(ferro_ta.SMA)                 # parameter docs for SMA

API
---
indicators(category=None)   — Return list of dicts describing every indicator.
methods(category=None)      — Return list of public callables across modules.
about()                     — Return package/version/module summary metadata.
info(func_or_name)          — Return a dict with full signature/docstring info.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

__all__ = ["indicators", "methods", "about", "info"]

# ---------------------------------------------------------------------------
# Category → module mapping used by indicators()
# ---------------------------------------------------------------------------

_CATEGORY_MODULES: dict[str, str] = {
    "overlap": "ferro_ta.indicators.overlap",
    "momentum": "ferro_ta.indicators.momentum",
    "volume": "ferro_ta.indicators.volume",
    "volatility": "ferro_ta.indicators.volatility",
    "statistic": "ferro_ta.indicators.statistic",
    "price_transform": "ferro_ta.indicators.price_transform",
    "pattern": "ferro_ta.indicators.pattern",
    "cycle": "ferro_ta.indicators.cycle",
    "math_ops": "ferro_ta.indicators.math_ops",
    "extended": "ferro_ta.indicators.extended",
    "batch": "ferro_ta.data.batch",
    "streaming": "ferro_ta.data.streaming",
    "resampling": "ferro_ta.data.resampling",
    "aggregation": "ferro_ta.data.aggregation",
    "signals": "ferro_ta.analysis.signals",
    "portfolio": "ferro_ta.analysis.portfolio",
    "features": "ferro_ta.analysis.features",
    "alerts": "ferro_ta.tools.alerts",
    "crypto": "ferro_ta.analysis.crypto",
    "regime": "ferro_ta.analysis.regime",
}

_METHOD_MODULES: dict[str, str] = {
    "top_level": "ferro_ta",
    **_CATEGORY_MODULES,
    "options": "ferro_ta.analysis.options",
    "futures": "ferro_ta.analysis.futures",
    "backtest": "ferro_ta.analysis.backtest",
    "options_strategy": "ferro_ta.analysis.options_strategy",
    "derivatives_payoff": "ferro_ta.analysis.derivatives_payoff",
    "attribution": "ferro_ta.analysis.attribution",
    "cross_asset": "ferro_ta.analysis.cross_asset",
    "tools": "ferro_ta.tools.tools",
    "viz": "ferro_ta.tools.viz",
}


def _iter_module_callables(
    module_name: str,
) -> list[tuple[str, Any]]:
    """Import *module_name* and return its ``__all__`` callables."""
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return []

    names = getattr(mod, "__all__", [])
    result = []
    for name in names:
        obj = getattr(mod, name, None)
        if callable(obj):
            result.append((name, obj))
    return result


def indicators(category: str | None = None) -> list[dict[str, Any]]:
    """Return a list of all ferro_ta indicators with metadata.

    Each entry is a dict with the following keys:

    - ``"name"`` (str): The indicator name, e.g. ``"SMA"``.
    - ``"category"`` (str): The category / sub-module, e.g. ``"overlap"``.
    - ``"module"`` (str): The fully qualified module name.
    - ``"doc"`` (str): First line of the docstring, or ``""`` if absent.
    - ``"params"`` (list[str]): Names of the function's parameters.

    Parameters
    ----------
    category : str | None
        If given, only return indicators from that category.  Must be one of
        the keys in :data:`ferro_ta.api_info._CATEGORY_MODULES`.

    Returns
    -------
    list[dict[str, Any]]
        Sorted alphabetically by ``"name"``.

    Examples
    --------
    >>> import ferro_ta
    >>> all_inds = ferro_ta.indicators()
    >>> len(all_inds) > 50
    True
    >>> overlap_inds = ferro_ta.indicators(category="overlap")
    >>> any(d["name"] == "SMA" for d in overlap_inds)
    True
    """
    cats: dict[str, str] = (
        {category: _CATEGORY_MODULES[category]}
        if category is not None
        else _CATEGORY_MODULES
    )
    result: list[dict[str, Any]] = []
    seen: set[str] = set()

    for cat, mod_name in cats.items():
        for name, func in _iter_module_callables(mod_name):
            if name in seen:
                continue
            seen.add(name)
            doc = inspect.getdoc(func) or ""
            first_line = doc.splitlines()[0] if doc else ""
            try:
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
            except (ValueError, TypeError):
                params = []
            result.append(
                {
                    "name": name,
                    "category": cat,
                    "module": mod_name,
                    "doc": first_line,
                    "params": params,
                }
            )

    result.sort(key=lambda d: d["name"])
    return result


def methods(category: str | None = None) -> list[dict[str, Any]]:
    """Return public callables across ferro_ta modules.

    Parameters
    ----------
    category : str | None
        Optional key from :data:`_METHOD_MODULES`, such as ``"top_level"``,
        ``"options"``, ``"futures"``, or ``"batch"``.
    """
    cats: dict[str, str] = (
        {category: _METHOD_MODULES[category]}
        if category is not None
        else _METHOD_MODULES
    )

    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for cat, mod_name in cats.items():
        for name, func in _iter_module_callables(mod_name):
            key = (mod_name, name)
            if key in seen:
                continue
            seen.add(key)
            doc = inspect.getdoc(func) or ""
            first_line = doc.splitlines()[0] if doc else ""
            try:
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
            except (ValueError, TypeError):
                params = []
            result.append(
                {
                    "name": name,
                    "category": cat,
                    "module": mod_name,
                    "doc": first_line,
                    "params": params,
                }
            )

    result.sort(key=lambda d: (d["category"], d["name"]))
    return result


def about() -> dict[str, Any]:
    """Return a small metadata summary for the installed ferro_ta package."""
    import ferro_ta  # noqa: PLC0415

    top_level_exports = sorted(getattr(ferro_ta, "__all__", []))
    return {
        "name": "ferro-ta",
        "version": getattr(ferro_ta, "__version__", "0+unknown"),
        "top_level_export_count": len(top_level_exports),
        "indicator_count": len(indicators()),
        "method_count": len(methods()),
        "categories": sorted(_METHOD_MODULES.keys()),
        "top_level_exports": top_level_exports,
    }


def info(func_or_name: Any) -> dict[str, Any]:
    """Return detailed information about an indicator function.

    Parameters
    ----------
    func_or_name : callable | str
        The indicator function (e.g. ``ferro_ta.SMA``) or its name as a
        string (e.g. ``"SMA"``).

    Returns
    -------
    dict[str, Any]
        Dictionary with the following keys:

        - ``"name"`` (str)
        - ``"module"`` (str)
        - ``"signature"`` (str): Full ``inspect.signature`` string.
        - ``"doc"`` (str): Full docstring.
        - ``"params"`` (dict[str, dict]): Mapping of parameter name →
          ``{"default": ..., "kind": str}`` for each parameter.

    Raises
    ------
    ValueError
        If *func_or_name* is a string that does not match any indicator.

    Examples
    --------
    >>> import ferro_ta
    >>> d = ferro_ta.info(ferro_ta.SMA)
    >>> d["name"]
    'SMA'
    >>> "close" in d["params"]
    True
    """
    if isinstance(func_or_name, str):
        import ferro_ta  # noqa: PLC0415

        func = getattr(ferro_ta, func_or_name, None)
        if func is None:
            raise ValueError(
                f"No indicator named {func_or_name!r} found in ferro_ta. "
                "Use ferro_ta.indicators() to list all available indicators."
            )
    else:
        func = func_or_name

    name = getattr(func, "__name__", repr(func))
    module = getattr(func, "__module__", "")
    doc = inspect.getdoc(func) or ""

    try:
        sig = inspect.signature(func)
        sig_str = str(sig)
        params = {}
        for pname, param in sig.parameters.items():
            kind_map = {
                inspect.Parameter.POSITIONAL_ONLY: "positional_only",
                inspect.Parameter.POSITIONAL_OR_KEYWORD: "positional_or_keyword",
                inspect.Parameter.VAR_POSITIONAL: "var_positional",
                inspect.Parameter.KEYWORD_ONLY: "keyword_only",
                inspect.Parameter.VAR_KEYWORD: "var_keyword",
            }
            params[pname] = {
                "default": (
                    param.default
                    if param.default is not inspect.Parameter.empty
                    else None
                ),
                "has_default": param.default is not inspect.Parameter.empty,
                "kind": kind_map.get(param.kind, "unknown"),
            }
    except (ValueError, TypeError):
        sig_str = "()"
        params = {}

    return {
        "name": name,
        "module": module,
        "signature": sig_str,
        "doc": doc,
        "params": params,
    }
