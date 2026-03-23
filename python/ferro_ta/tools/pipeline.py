"""
ferro_ta.pipeline — Indicator Pipeline and Composition API.

Build reusable pipelines that apply one or more indicators to price arrays
in a single call.  A :class:`Pipeline` collects named steps, runs them in
order, and returns the results as a dictionary.

This module is designed for:

- Backtesting workflows that need multiple indicators computed on the same data.
- Feature engineering for machine-learning pipelines.
- Batch scenarios where you want all indicator values in one dictionary.

Usage
-----
>>> import numpy as np
>>> from ferro_ta.tools.pipeline import Pipeline
>>> from ferro_ta import SMA, EMA, RSI
>>>
>>> close = np.array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10,
...                   45.15, 43.61, 44.33, 44.83, 45.10, 45.15, 43.61, 44.33])
>>>
>>> pipe = (
...     Pipeline()
...     .add("sma_10", SMA, timeperiod=10)
...     .add("ema_10", EMA, timeperiod=10)
...     .add("rsi_14", RSI, timeperiod=14)
... )
>>> results = pipe.run(close)
>>> print(list(results.keys()))
['sma_10', 'ema_10', 'rsi_14']
>>> results["sma_10"].shape
(15,)

Chaining convenience
--------------------
:meth:`Pipeline.add` returns ``self`` so calls can be chained.

The :func:`make_pipeline` function is a convenience wrapper:

>>> from ferro_ta.tools.pipeline import make_pipeline
>>> pipe = make_pipeline(sma_5=(SMA, {"timeperiod": 5}),
...                      rsi_14=(RSI, {"timeperiod": 14}))
>>> results = pipe.run(close)

Multi-output indicators
-----------------------
For indicators that return tuples (e.g. BBANDS, MACD) you can pass an
optional ``output_keys`` argument to unpack the tuple into named keys:

>>> from ferro_ta import BBANDS, MACD
>>> pipe = (
...     Pipeline()
...     .add("bb", BBANDS, output_keys=["bb_upper", "bb_mid", "bb_lower"],
...          timeperiod=5, nbdevup=2.0, nbdevdn=2.0)
...     .add("macd", MACD, output_keys=["macd", "signal", "hist"],
...          fastperiod=3, slowperiod=5, signalperiod=2)
... )
>>> results = pipe.run(close)
>>> list(results.keys())
['bb_upper', 'bb_mid', 'bb_lower', 'macd', 'signal', 'hist']
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike

from ferro_ta._utils import _to_f64

# ---------------------------------------------------------------------------
# Internal step type
# ---------------------------------------------------------------------------


class _Step:
    """A single pipeline step (one indicator call)."""

    __slots__ = ("name", "func", "kwargs", "output_keys")

    def __init__(
        self,
        name: str,
        func: Callable[..., Any],
        kwargs: dict[str, Any],
        output_keys: Optional[list[str]],
    ) -> None:
        self.name = name
        self.func = func
        self.kwargs = kwargs
        self.output_keys = output_keys


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """A reusable indicator pipeline.

    A Pipeline stores a sequence of named indicator steps and can be applied
    to one or more data arrays. Calling :meth:`run` returns a dictionary
    mapping step names to result arrays.

    Parameters
    ----------
    steps : list of (name, func, kwargs, output_keys), optional
        Pre-built steps (rarely needed; prefer :meth:`add`).

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta import SMA, RSI
    >>> from ferro_ta.tools.pipeline import Pipeline
    >>> close = np.arange(1.0, 20.0)
    >>> results = Pipeline().add("sma5", SMA, timeperiod=5).run(close)
    >>> results["sma5"].shape
    (19,)
    """

    def __init__(self, steps: Optional[list[_Step]] = None) -> None:
        self._steps: list[_Step] = list(steps) if steps else []

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def add(
        self,
        name: str,
        func: Callable[..., Any],
        output_keys: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Pipeline:
        """Add an indicator step to the pipeline.

        Parameters
        ----------
        name : str
            Key under which the result is stored in the output dict.
            For multi-output indicators with *output_keys*, this argument
            is ignored (the output_keys are used instead).
        func : callable
            Indicator function (e.g. ``SMA``, ``RSI``, ``BBANDS``).
        output_keys : list of str, optional
            For multi-output indicators that return a tuple (e.g. BBANDS,
            MACD), supply the names for each output.  If not provided and
            the indicator returns a tuple, the results are stored as
            ``name_0``, ``name_1``, … .
        **kwargs
            Keyword arguments forwarded to *func* (e.g. ``timeperiod=14``).

        Returns
        -------
        Pipeline
            Returns ``self`` for chaining.

        Raises
        ------
        ValueError
            If *name* is already used by an existing step (and no
            *output_keys* are supplied).
        TypeError
            If *func* is not callable.
        """
        if not callable(func):
            raise TypeError(f"func must be callable, got {type(func).__name__}")

        # Check for duplicate names (only when output_keys is not given)
        existing = self._output_names()
        if output_keys:
            for key in output_keys:
                if key in existing:
                    raise ValueError(f"Duplicate output key '{key}' in pipeline")
        else:
            if name in existing:
                raise ValueError(
                    f"A step named '{name}' already exists. "
                    "Use a different name or remove the existing step first."
                )

        self._steps.append(_Step(name, func, kwargs, output_keys))
        return self

    def remove(self, name: str) -> Pipeline:
        """Remove the step identified by *name* (or *output_keys* containing *name*).

        Parameters
        ----------
        name : str
            Step name or one of the output keys.

        Returns
        -------
        Pipeline
            Returns ``self`` for chaining.

        Raises
        ------
        KeyError
            If no step with the given name is found.
        """
        for i, step in enumerate(self._steps):
            if step.name == name or (step.output_keys and name in step.output_keys):
                del self._steps[i]
                return self
        raise KeyError(f"No step named '{name}' in pipeline")

    def steps(self) -> list[str]:
        """Return a list of step names (or output keys for multi-output steps)."""
        return self._output_names()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, close: ArrayLike, **extra: Any) -> dict[str, np.ndarray]:
        """Apply all pipeline steps to *close* and return results.

        Parameters
        ----------
        close : array-like
            Primary input array (close prices).  For indicators that need
            additional arrays (e.g. high/low/volume), pass them as keyword
            arguments (see *extra*).
        **extra
            Additional arrays (e.g. ``high=…``, ``low=…``, ``volume=…``).
            Each step's kwargs are merged with *extra* on a per-call basis;
            step-level kwargs take precedence.

        Returns
        -------
        dict of str → numpy.ndarray
            Mapping from output name to result array.

        Examples
        --------
        >>> import numpy as np
        >>> from ferro_ta import SMA, ATR
        >>> from ferro_ta.tools.pipeline import Pipeline
        >>> n = 20
        >>> close = np.random.rand(n) + 10
        >>> high  = close + 0.5
        >>> low   = close - 0.5
        >>> pipe = (
        ...     Pipeline()
        ...     .add("sma", SMA, timeperiod=5)
        ... )
        >>> out = pipe.run(close)
        >>> out["sma"].shape
        (20,)
        """
        close_arr = _to_f64(close)
        output: dict[str, np.ndarray] = {}

        for step in self._steps:
            # Build merged kwargs: extra is the base; step-level kwargs override
            merged = dict(extra)
            merged.update(step.kwargs)

            result = step.func(close_arr, **merged)

            if isinstance(result, tuple):
                if step.output_keys:
                    if len(step.output_keys) != len(result):
                        raise ValueError(
                            f"Step '{step.name}': output_keys has {len(step.output_keys)} "
                            f"entries but the function returned {len(result)} values."
                        )
                    for key, arr in zip(step.output_keys, result):
                        output[key] = np.asarray(arr, dtype=np.float64)
                else:
                    for i, arr in enumerate(result):
                        output[f"{step.name}_{i}"] = np.asarray(arr, dtype=np.float64)
            else:
                output[step.name] = np.asarray(result, dtype=np.float64)

        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _output_names(self) -> list[str]:
        names: list[str] = []
        for step in self._steps:
            if step.output_keys:
                names.extend(step.output_keys)
            else:
                names.append(step.name)
        return names

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        step_str = ", ".join(self._output_names())
        return f"Pipeline([{step_str}])"


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def make_pipeline(**named_steps: tuple[Callable[..., Any], dict[str, Any]]) -> Pipeline:
    """Build a :class:`Pipeline` from keyword arguments.

    Parameters
    ----------
    **named_steps
        Each keyword argument is a step: ``name=(func, kwargs_dict)``.

    Returns
    -------
    Pipeline

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta import SMA, RSI
    >>> from ferro_ta.tools.pipeline import make_pipeline
    >>> pipe = make_pipeline(sma_5=(SMA, {"timeperiod": 5}),
    ...                      rsi_14=(RSI, {"timeperiod": 14}))
    >>> results = pipe.run(np.arange(1.0, 25.0))
    >>> sorted(results.keys())
    ['rsi_14', 'sma_5']
    """
    pipe = Pipeline()
    for name, step in named_steps.items():
        func, kwargs = step
        pipe.add(name, func, **kwargs)
    return pipe


__all__ = [
    "Pipeline",
    "make_pipeline",
]
