"""
ferro_ta.config — Global configuration and indicator defaults.

This module provides a simple configuration system that allows you to set
global default values for indicator parameters (e.g. default RSI period)
without having to pass them on every call.  Defaults are overridden by
explicit keyword arguments to any indicator function.

Usage
-----
>>> import ferro_ta.core.config as config
>>> config.set_default("timeperiod", 20)   # global fallback for all indicators
>>> config.set_default("RSI.timeperiod", 14)  # RSI-specific override

>>> from ferro_ta import RSI
>>> import numpy as np
>>> close = np.arange(1.0, 25.0)
>>> RSI(close)          # uses RSI.timeperiod=14 from config
>>> RSI(close, timeperiod=5)   # explicit argument wins

Context manager
---------------
Use :class:`Config` as a context manager for temporary overrides:

>>> with config.Config(timeperiod=5):
...     result = RSI(close)   # timeperiod=5 inside the block

Resetting
---------
>>> config.reset()   # remove all custom defaults

API
---
set_default(key, value)     — Set a global default.  *key* can be a plain
                              parameter name (``"timeperiod"``) or an
                              indicator-qualified name (``"RSI.timeperiod"``).
get_default(key, fallback)  — Get the current default for *key*.
reset(key=None)             — Reset one or all defaults to their built-in values.
Config(**overrides)         — Context manager: temporarily set defaults.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Thread-local storage — each thread can have independent config snapshots
# (rare in practice but safe for testing).
# ---------------------------------------------------------------------------
_local = threading.local()


def _store() -> dict[str, Any]:
    """Return the thread-local defaults store, creating it if necessary."""
    if not hasattr(_local, "defaults"):
        _local.defaults = {}
    return _local.defaults


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def set_default(key: str, value: Any) -> None:
    """Set a global default parameter value.

    Parameters
    ----------
    key : str
        Parameter name (e.g. ``"timeperiod"``) or indicator-qualified name
        (e.g. ``"RSI.timeperiod"``).  Indicator-qualified defaults take
        precedence over plain defaults when both are set.
    value : any
        Default value to store.

    Examples
    --------
    >>> import ferro_ta.core.config as config
    >>> config.set_default("timeperiod", 20)
    >>> config.set_default("RSI.timeperiod", 14)
    """
    _store()[key] = value


def get_default(key: str, fallback: Any = None) -> Any:
    """Return the current default for *key*, or *fallback* if not set.

    Parameters
    ----------
    key : str
        Parameter name (e.g. ``"timeperiod"``).
    fallback : any, optional
        Value returned when no default is set.

    Returns
    -------
    any
        The stored default value, or *fallback*.

    Examples
    --------
    >>> import ferro_ta.core.config as config
    >>> config.set_default("timeperiod", 20)
    >>> config.get_default("timeperiod")
    20
    >>> config.get_default("nonexistent", -1)
    -1
    """
    return _store().get(key, fallback)


def get_defaults_for(indicator_name: str) -> dict[str, Any]:
    """Return all applicable defaults for the given indicator.

    Indicator-qualified keys (``"RSI.timeperiod"``) override plain keys
    (``"timeperiod"``) in the returned dict.

    Parameters
    ----------
    indicator_name : str
        Name of the indicator (e.g. ``"RSI"``).

    Returns
    -------
    dict
        Merged defaults where indicator-specific values override global ones.

    Examples
    --------
    >>> import ferro_ta.core.config as config
    >>> config.set_default("timeperiod", 20)
    >>> config.set_default("RSI.timeperiod", 14)
    >>> config.get_defaults_for("RSI")
    {'timeperiod': 14}
    >>> config.get_defaults_for("SMA")
    {'timeperiod': 20}
    """
    store = _store()
    prefix = f"{indicator_name}."

    # Start with plain defaults
    result: dict[str, Any] = {}
    for k, v in store.items():
        if "." not in k:
            result[k] = v

    # Override with indicator-qualified defaults
    for k, v in store.items():
        if k.startswith(prefix):
            result[k[len(prefix) :]] = v

    return result


def reset(key: Optional[str] = None) -> None:
    """Reset defaults.

    Parameters
    ----------
    key : str, optional
        If given, remove only this key.  If ``None``, remove all defaults.

    Examples
    --------
    >>> import ferro_ta.core.config as config
    >>> config.set_default("timeperiod", 20)
    >>> config.reset("timeperiod")
    >>> config.get_default("timeperiod") is None
    True
    >>> config.reset()   # clear everything
    """
    store = _store()
    if key is None:
        store.clear()
    else:
        store.pop(key, None)


def list_defaults() -> dict[str, Any]:
    """Return a copy of all currently set defaults.

    Returns
    -------
    dict
        Copy of the current defaults store.

    Examples
    --------
    >>> import ferro_ta.core.config as config
    >>> config.set_default("timeperiod", 10)
    >>> config.list_defaults()
    {'timeperiod': 10}
    """
    return dict(_store())


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class Config:
    """Context manager for temporary configuration overrides.

    On entry, applies the specified overrides on top of the current defaults.
    On exit, restores the previous state exactly.

    Parameters
    ----------
    **overrides
        Key-value pairs to set temporarily.

    Examples
    --------
    >>> import numpy as np
    >>> import ferro_ta.core.config as config
    >>> from ferro_ta import RSI
    >>> close = np.arange(1.0, 25.0)
    >>> with config.Config(timeperiod=5):
    ...     config.get_default("timeperiod")
    5
    >>> config.get_default("timeperiod") is None   # restored after exit
    True
    """

    def __init__(self, **overrides: Any) -> None:
        self._overrides = overrides
        self._saved: dict[str, Any] = {}

    def __enter__(self) -> Config:
        store = _store()
        # Save current values for all keys we're about to change
        self._saved = {k: store.get(k) for k in self._overrides}
        # Apply overrides
        for k, v in self._overrides.items():
            store[k] = v
        return self

    def __exit__(self, *_: Any) -> None:
        store = _store()
        for k, saved_v in self._saved.items():
            if saved_v is None:
                store.pop(k, None)
            else:
                store[k] = saved_v


__all__ = [
    "set_default",
    "get_default",
    "get_defaults_for",
    "reset",
    "list_defaults",
    "Config",
]
