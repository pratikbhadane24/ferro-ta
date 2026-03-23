"""
ferro_ta.logging_utils — Logging integration and debug utilities.

Provides a structured logging interface for ferro_ta with configurable
verbosity, debug mode, and optional performance timing.

Usage
-----
>>> import ferro_ta.logging_utils as ft_log
>>> ft_log.enable_debug()          # turn on DEBUG-level output
>>> ft_log.disable_debug()         # back to WARNING level

>>> # Use as a context manager for a single call:
>>> with ft_log.debug_mode():
...     result = ferro_ta.SMA(close, timeperiod=20)

>>> # Access the ferro_ta logger directly:
>>> import logging
>>> logger = logging.getLogger("ferro_ta")
>>> logger.setLevel(logging.DEBUG)

API
---
get_logger()          — Return the ``ferro_ta`` :class:`logging.Logger`.
enable_debug()        — Set the ferro_ta logger to DEBUG level.
disable_debug()       — Reset the ferro_ta logger to WARNING level.
debug_mode()          — Context manager: temporarily enable debug logging.
log_call(func, ...)   — Log a function call with input shapes and timing.
benchmark(func, ...)  — Run *func* n times and return timing statistics.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import time
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

__all__ = [
    "get_logger",
    "enable_debug",
    "disable_debug",
    "debug_mode",
    "log_call",
    "benchmark",
]

# ---------------------------------------------------------------------------
# Logger setup — single ``ferro_ta`` logger, handlers added lazily.
# ---------------------------------------------------------------------------

_LOGGER_NAME = "ferro_ta"
_DEFAULT_FORMAT = "%(levelname)s [%(name)s] %(message)s"

F = TypeVar("F", bound=Callable[..., Any])


def get_logger() -> logging.Logger:
    """Return the ``ferro_ta`` package logger.

    The logger is created on first call.  A :class:`logging.NullHandler` is
    installed so that no output appears by default (following the best-practice
    for library loggers).  Call :func:`enable_debug` or configure the logger
    explicitly to see output.

    Returns
    -------
    logging.Logger
        The ``ferro_ta`` package logger.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def enable_debug(fmt: str = _DEFAULT_FORMAT) -> None:
    """Enable DEBUG-level logging for ferro_ta.

    Adds a :class:`logging.StreamHandler` that writes to *stderr* using *fmt*
    and sets the logger level to ``DEBUG``.  Calling this multiple times is
    safe — duplicate handlers are not added.

    Parameters
    ----------
    fmt:
        Log message format string passed to :class:`logging.Formatter`.
    """
    logger = get_logger()
    logger.setLevel(logging.DEBUG)
    # Avoid duplicate stream handlers
    has_stream = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    if not has_stream:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)


def disable_debug() -> None:
    """Reset the ferro_ta logger to WARNING level and remove stream handlers."""
    logger = get_logger()
    logger.setLevel(logging.WARNING)
    logger.handlers = [h for h in logger.handlers if isinstance(h, logging.NullHandler)]


@contextlib.contextmanager
def debug_mode(fmt: str = _DEFAULT_FORMAT) -> Iterator[logging.Logger]:
    """Context manager: enable debug logging for the duration of the block.

    Parameters
    ----------
    fmt:
        Log message format string.

    Yields
    ------
    logging.Logger
        The ``ferro_ta`` logger with DEBUG level active.

    Examples
    --------
    >>> import numpy as np
    >>> import ferro_ta.logging_utils as ft_log
    >>> close = np.arange(1.0, 30.0)
    >>> with ft_log.debug_mode():
    ...     pass  # ferro_ta calls inside here will log debug info
    """
    prev_level = get_logger().level
    enable_debug(fmt)
    try:
        yield get_logger()
    finally:
        disable_debug()
        get_logger().setLevel(prev_level)


# ---------------------------------------------------------------------------
# Helper: shape summary for numpy / pandas / polars arrays
# ---------------------------------------------------------------------------


def _shape_str(obj: Any) -> str:
    """Return a compact shape/type description for logging."""
    try:
        import numpy as np  # noqa: PLC0415

        if isinstance(obj, np.ndarray):
            return f"ndarray{obj.shape} dtype={obj.dtype}"
    except ImportError:
        pass

    if hasattr(obj, "shape"):
        return f"{type(obj).__name__}{obj.shape}"
    if hasattr(obj, "__len__"):
        return f"{type(obj).__name__}[{len(obj)}]"  # type: ignore[arg-type]
    return repr(obj)


# ---------------------------------------------------------------------------
# log_call: decorator / manual call logger
# ---------------------------------------------------------------------------


def log_call(
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call *func* with *args*/*kwargs*, logging input shapes and elapsed time.

    Parameters
    ----------
    func:
        The ferro_ta indicator function to call.
    *args:
        Positional arguments forwarded to *func*.
    **kwargs:
        Keyword arguments forwarded to *func*.

    Returns
    -------
    Any
        The return value of ``func(*args, **kwargs)``.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta import SMA
    >>> import ferro_ta.logging_utils as ft_log
    >>> ft_log.enable_debug()
    >>> close = np.arange(1.0, 30.0)
    >>> result = ft_log.log_call(SMA, close, timeperiod=5)
    """
    logger = get_logger()
    name = getattr(func, "__name__", repr(func))

    if logger.isEnabledFor(logging.DEBUG):
        arg_shapes = ", ".join(_shape_str(a) for a in args)
        kwarg_shapes = ", ".join(f"{k}={_shape_str(v)}" for k, v in kwargs.items())
        all_args = ", ".join(filter(None, [arg_shapes, kwarg_shapes]))
        logger.debug("calling %s(%s)", name, all_args)

    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if logger.isEnabledFor(logging.DEBUG):
        out_shape = (
            _shape_str(result)
            if not isinstance(result, tuple)
            else str(tuple(_shape_str(r) for r in result))
        )
        logger.debug("%s → %s  [%.3f ms]", name, out_shape, elapsed_ms)

    return result


# ---------------------------------------------------------------------------
# benchmark: run a function N times and report timing statistics
# ---------------------------------------------------------------------------


def benchmark(
    func: Callable[..., Any],
    *args: Any,
    n: int = 100,
    warmup: int = 5,
    **kwargs: Any,
) -> dict[str, float]:
    """Benchmark *func* by calling it *n* times and returning timing stats.

    Parameters
    ----------
    func:
        The ferro_ta indicator function to benchmark.
    *args:
        Positional arguments forwarded to *func* on each call.
    n:
        Number of timed iterations (default 100).
    warmup:
        Number of warm-up calls before timing starts (default 5).
    **kwargs:
        Keyword arguments forwarded to *func* on each call.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``"mean_ms"``, ``"min_ms"``, ``"max_ms"``,
        ``"total_ms"``, ``"n"``.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta import SMA
    >>> import ferro_ta.logging_utils as ft_log
    >>> close = np.random.randn(10_000)
    >>> stats = ft_log.benchmark(SMA, close, timeperiod=20, n=50)
    >>> print(f"mean={stats['mean_ms']:.3f} ms")
    mean=... ms
    """
    name = getattr(func, "__name__", repr(func))

    for _ in range(warmup):
        func(*args, **kwargs)

    times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000.0)

    total = sum(times)
    mean = total / n
    stats: dict[str, float] = {
        "mean_ms": mean,
        "min_ms": min(times),
        "max_ms": max(times),
        "total_ms": total,
        "n": float(n),
    }

    logger = get_logger()
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "benchmark %s  n=%d  mean=%.3f ms  min=%.3f ms  max=%.3f ms",
            name,
            n,
            stats["mean_ms"],
            stats["min_ms"],
            stats["max_ms"],
        )

    return stats


# ---------------------------------------------------------------------------
# traced: decorator that wraps a function with log_call behaviour
# ---------------------------------------------------------------------------


def traced(func: F) -> F:
    """Decorator: wrap *func* so every call is logged at DEBUG level.

    Parameters
    ----------
    func:
        Function to wrap.

    Returns
    -------
    Callable
        Wrapped function with identical signature.

    Examples
    --------
    >>> import ferro_ta.logging_utils as ft_log
    >>> @ft_log.traced
    ... def my_indicator(close, timeperiod=14):
    ...     return close  # placeholder
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return log_call(func, *args, **kwargs)

    return wrapper  # type: ignore[return-value]
