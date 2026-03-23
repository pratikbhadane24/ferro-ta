"""
Data-driven binding layer — generic wrapper for Rust indicator calls.

This module provides a single helper that performs validation, array conversion
(_to_f64), Rust call, and error normalization. Indicator modules can use it to
reduce repetitive wrapper code; a manifest (see _indicator_manifest.yaml) describes
each indicator so that wrappers or code generation can be driven from data.

Usage (manual wrapper):
    from ferro_ta._binding import binding_call
    def SMA(close, timeperiod=30):
        return binding_call(
            _sma,
            array_params=["close"],
            timeperiod_param="timeperiod",
            close=close,
            timeperiod=timeperiod,
        )

Future: A code generator can read the manifest and emit either full wrapper
functions or binding_call(...) invocations so that ~6000 lines of repetitive
wrapper code are generated from the manifest.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from ferro_ta._utils import _to_f64
from ferro_ta.core.exceptions import (
    _normalize_rust_error,
    check_equal_length,
    check_timeperiod,
)


def binding_call(
    rust_fn: Callable[..., Any],
    *,
    array_params: list[str],
    timeperiod_param: Optional[str] = None,
    timeperiod_min: int = 1,
    equal_length_groups: Optional[list[list[str]]] = None,
    **kwargs: Any,
) -> Any:
    """Call a Rust indicator with validation and array conversion.

    Parameters
    ----------
    rust_fn : callable
        The Rust function from _ferro_ta (e.g. _sma).
    array_params : list of str
        Names of keyword arguments that are array-like; they are converted
        with _to_f64 and passed in order as positional args to rust_fn.
    timeperiod_param : str, optional
        If set, the value of kwargs[timeperiod_param] is validated with
        check_timeperiod(..., minimum=timeperiod_min).
    timeperiod_min : int
        Minimum allowed value for timeperiod_param (default 1).
    equal_length_groups : list of list of str, optional
        Each inner list is a group of param names that must have equal length;
        check_equal_length is called with that group.
    **kwargs
        Keyword arguments to pass. Array params are converted and passed
        positionally; non-array params are passed as keyword arguments to
        rust_fn (caller must ensure rust_fn signature matches).

    Returns
    -------
    Result of rust_fn(...). Typically numpy.ndarray or tuple of ndarray.

    Raises
    ------
    FerroTAValueError, FerroTAInputError
        Via check_timeperiod / check_equal_length or _normalize_rust_error.
    """
    if timeperiod_param is not None and timeperiod_param in kwargs:
        check_timeperiod(
            kwargs[timeperiod_param],
            name=timeperiod_param,
            minimum=timeperiod_min,
        )
    if equal_length_groups is not None:
        for group in equal_length_groups:
            check_equal_length(**{k: kwargs[k] for k in group if k in kwargs})
    # Build positional args for rust_fn in array_params order, then remaining kwargs
    pos_args = [_to_f64(kwargs[p]) for p in array_params if p in kwargs]
    rest_kw = {k: v for k, v in kwargs.items() if k not in array_params}
    try:
        return rust_fn(*pos_args, **rest_kw)
    except ValueError as e:
        _normalize_rust_error(e)
