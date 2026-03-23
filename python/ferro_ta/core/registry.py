"""
Plugin / Extension Registry
============================

A lightweight registry that allows users to register custom indicators and
call any indicator (built-in or custom) by name.

Usage
-----
>>> import numpy as np
>>> import ferro_ta
>>> from ferro_ta.core.registry import register, run, get, list_indicators
>>>
>>> # Call a built-in indicator by name
>>> close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
>>> result = run("SMA", close, timeperiod=3)
>>>
>>> # Register a custom indicator
>>> def MY_IND(close, timeperiod=10):
...     \"\"\"Custom indicator: simple sum / timeperiod.\"\"\"
...     import numpy as np
...     out = np.full_like(close, np.nan)
...     for i in range(timeperiod - 1, len(close)):
...         out[i] = close[i - timeperiod + 1 : i + 1].sum() / timeperiod
...     return out
>>> register("MY_IND", MY_IND)
>>> result = run("MY_IND", close, timeperiod=3)

Writing a plugin
----------------
A plugin function must:

1. Accept at least one positional array argument (``close``, ``high``, etc.).
2. Accept keyword arguments for parameters (e.g. ``timeperiod=14``).
3. Return a single ``numpy.ndarray`` *or* a tuple of ``numpy.ndarray`` for
   multi-output indicators.

Example::

    def DOUBLE_RSI(close, timeperiod=14, smooth=3):
        import ferro_ta
        rsi = ferro_ta.RSI(close, timeperiod=timeperiod)
        return ferro_ta.SMA(rsi, timeperiod=smooth)

    from ferro_ta.core.registry import register
    register("DOUBLE_RSI", DOUBLE_RSI)

API
---
register(name, func)    — Register *func* under *name*.
unregister(name)        — Remove a registered indicator.
get(name)               — Return the callable for *name*.
run(name, *args, **kw)  — Look up *name* and call it with *args* / **kw*.
list_indicators()       — Return a sorted list of all registered names.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ferro_ta.core.exceptions import FerroTAError


class FerroTARegistryError(FerroTAError):
    """Raised when a registry lookup fails (unknown indicator name)."""


# ---------------------------------------------------------------------------
# Internal registry (module-level singleton dict)
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable[..., Any]] = {}


def register(name: str, func: Callable[..., Any]) -> None:
    """Register a callable under *name*.

    Parameters
    ----------
    name:
        Indicator name (case-sensitive; convention is ALL_CAPS for
        compatibility with TA-Lib naming).
    func:
        A callable that accepts at least one array-like positional argument
        and optional keyword arguments, and returns a ``numpy.ndarray`` or a
        tuple of ``numpy.ndarray``.

    Raises
    ------
    TypeError
        If *func* is not callable.
    """
    if not callable(func):
        raise TypeError(f"Expected a callable for '{name}', got {type(func).__name__}")
    _REGISTRY[name] = func


def unregister(name: str) -> None:
    """Remove the indicator registered under *name*.

    Parameters
    ----------
    name:
        Indicator name to remove.

    Raises
    ------
    FerroTARegistryError
        If *name* is not in the registry.
    """
    if name not in _REGISTRY:
        raise FerroTARegistryError(
            f"Indicator '{name}' is not registered. "
            f"Available indicators: {sorted(_REGISTRY)[:10]}…"
        )
    del _REGISTRY[name]


def get(name: str) -> Callable[..., Any]:
    """Return the callable registered under *name*.

    Parameters
    ----------
    name:
        Indicator name (case-sensitive).

    Returns
    -------
    Callable
        The registered function.

    Raises
    ------
    FerroTARegistryError
        If *name* is not in the registry.
    """
    if name not in _REGISTRY:
        raise FerroTARegistryError(
            f"Unknown indicator '{name}'. "
            f"Use list_indicators() to see all registered names."
        )
    return _REGISTRY[name]


def run(name: str, *args: Any, **kwargs: Any) -> Any:
    """Look up *name* in the registry and call it with *args* / *kwargs*.

    Parameters
    ----------
    name:
        Indicator name (case-sensitive).
    *args:
        Positional arguments forwarded to the indicator function.
    **kwargs:
        Keyword arguments forwarded to the indicator function.

    Returns
    -------
    numpy.ndarray or tuple of numpy.ndarray
        Whatever the indicator function returns.

    Raises
    ------
    FerroTARegistryError
        If *name* is not in the registry.
    """
    func = get(name)
    return func(*args, **kwargs)


def list_indicators() -> list[str]:
    """Return a sorted list of all registered indicator names.

    Returns
    -------
    list of str
        Sorted list of indicator names.
    """
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Auto-register all built-in indicators from ferro_ta.__all__
# ---------------------------------------------------------------------------


def _register_builtins() -> None:
    """Register every built-in indicator from ``ferro_ta.__all__``."""
    # Lazy import to avoid circular imports at module load time
    import ferro_ta  # noqa: PLC0415

    for _name in ferro_ta.__all__:  # type: ignore[attr-defined]
        _fn = getattr(ferro_ta, _name, None)
        if callable(_fn):
            _REGISTRY[_name] = _fn


_register_builtins()
