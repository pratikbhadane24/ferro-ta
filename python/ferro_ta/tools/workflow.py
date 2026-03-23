"""
ferro_ta.workflow — End-to-End Workflow Orchestration
=====================================================

Provides a lightweight DAG/linear workflow that chains data acquisition,
resampling, indicator computation, strategy signal generation, and alerting
in a single call.  All heavy computation is delegated to existing ferro_ta
modules; this module is **pure orchestration** with no new algorithmic logic.

See ``docs/agentic.md`` for a full end-to-end example including LangChain
integration and scheduling.

Quick start
-----------
>>> import numpy as np
>>> from ferro_ta.tools.workflow import Workflow
>>>
>>> # Build a workflow
>>> wf = (
...     Workflow()
...     .add_indicator("sma_20", "SMA", timeperiod=20)
...     .add_indicator("rsi_14", "RSI", timeperiod=14)
...     .add_strategy("rsi_30_70")
... )
>>>
>>> close = np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, 100)) * 100
>>> result = wf.run(close)
>>> print(result.keys())

API
---
Workflow
    Fluent builder that chains: indicators → strategy → backtest → alerts.

run_pipeline(close, indicators, strategy, alert_level)
    Functional interface: single call that returns all outputs.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "Workflow",
    "run_pipeline",
]


class Workflow:
    """Fluent builder for an end-to-end ferro_ta workflow.

    A :class:`Workflow` chains these optional steps in order:

    1. **Indicators** — compute one or more named indicators on close prices.
    2. **Strategy** — optionally run a backtest strategy and capture the result.
    3. **Alerts** — optionally define threshold or cross alerts on any indicator
       output and collect firing bars.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.tools.workflow import Workflow
    >>> rng = np.random.default_rng(42)
    >>> close = np.cumprod(1 + rng.normal(0, 0.01, 200)) * 100
    >>> result = (
    ...     Workflow()
    ...     .add_indicator("sma_20", "SMA", timeperiod=20)
    ...     .add_indicator("rsi_14", "RSI", timeperiod=14)
    ...     .run(close)
    ... )
    >>> "sma_20" in result
    True
    >>> "rsi_14" in result
    True
    """

    def __init__(self) -> None:
        self._indicator_steps: list[tuple[str, str, dict[str, Any]]] = []
        self._strategy: Optional[str] = None
        self._strategy_kwargs: dict[str, Any] = {}
        self._alert_steps: list[tuple[str, str, float, int]] = []

    # ------------------------------------------------------------------
    # Fluent builders
    # ------------------------------------------------------------------

    def add_indicator(
        self,
        output_key: str,
        indicator_name: str,
        **kwargs: Any,
    ) -> Workflow:
        """Add an indicator step.

        Parameters
        ----------
        output_key : str
            Key under which the result will be stored in the output dict.
        indicator_name : str
            Name of the indicator (e.g. ``"SMA"``, ``"RSI"``).
        **kwargs
            Parameters forwarded to the indicator (e.g. ``timeperiod=14``).

        Returns
        -------
        Workflow
            Self, for chaining.
        """
        self._indicator_steps.append((output_key, indicator_name, kwargs))
        return self

    def add_strategy(
        self,
        strategy: str,
        **strategy_kwargs: Any,
    ) -> Workflow:
        """Set the backtest strategy to run.

        Only one strategy can be active at a time; calling this method again
        replaces the previous strategy.

        Parameters
        ----------
        strategy : str
            Strategy name (``"rsi_30_70"``, ``"sma_crossover"``, or
            ``"macd_crossover"``).
        **strategy_kwargs
            Extra parameters forwarded to the strategy function.

        Returns
        -------
        Workflow
            Self, for chaining.
        """
        self._strategy = strategy
        self._strategy_kwargs = dict(strategy_kwargs)
        return self

    def add_alert(
        self,
        indicator_key: str,
        level: float,
        direction: int = 1,
    ) -> Workflow:
        """Add a threshold crossing alert on an indicator output.

        The alert fires on bars where the specified indicator crosses *level*
        in *direction*.

        Parameters
        ----------
        indicator_key : str
            Key of an indicator already added via :meth:`add_indicator`.
        level : float
            Alert level (e.g. 30 for RSI oversold).
        direction : int
            ``+1`` → alert when series crosses *above* level.
            ``-1`` → alert when series crosses *below* level.

        Returns
        -------
        Workflow
            Self, for chaining.
        """
        alert_key = f"alert_{indicator_key}_{level:.4g}_{direction:+d}"
        self._alert_steps.append((alert_key, indicator_key, level, direction))
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        close: ArrayLike,
        commission_per_trade: float = 0.0,
        slippage_bps: float = 0.0,
    ) -> dict[str, Any]:
        """Execute the workflow and return all outputs.

        Parameters
        ----------
        close : array-like
            Close price series (1-D).
        commission_per_trade : float
            Commission forwarded to backtest (if strategy is set).
        slippage_bps : float
            Slippage in bps forwarded to backtest (if strategy is set).

        Returns
        -------
        dict
            Dictionary containing:

            * Each indicator key → ``numpy.ndarray`` result (or dict for
              multi-output indicators such as BBANDS/MACD).
            * ``"backtest"`` → summary dict (only if a strategy was added).
            * Each alert key → list of bar indices where alert fired
              (only if alerts were added).
        """
        from ferro_ta.tools import compute_indicator, run_backtest

        close_arr = np.asarray(close, dtype=np.float64)
        output: dict[str, Any] = {}

        # Step 1: compute indicators
        for output_key, indicator_name, kwargs in self._indicator_steps:
            output[output_key] = compute_indicator(indicator_name, close_arr, **kwargs)

        # Step 2: run backtest strategy (if set)
        if self._strategy is not None:
            output["backtest"] = run_backtest(
                self._strategy,
                close_arr,
                commission_per_trade=commission_per_trade,
                slippage_bps=slippage_bps,
                **self._strategy_kwargs,
            )

        # Step 3: compute alerts
        if self._alert_steps:
            from ferro_ta.tools.alerts import check_threshold, collect_alert_bars

            for alert_key, ind_key, level, direction in self._alert_steps:
                series = output.get(ind_key)
                if series is None:
                    continue
                # For multi-output indicators, skip alert silently
                if isinstance(series, dict):
                    continue
                arr = np.asarray(series, dtype=np.float64)
                mask = check_threshold(arr, level=level, direction=direction)
                output[alert_key] = collect_alert_bars(mask).tolist()

        return output


# ---------------------------------------------------------------------------
# Functional interface
# ---------------------------------------------------------------------------


def run_pipeline(
    close: ArrayLike,
    indicators: Optional[dict[str, dict[str, Any]]] = None,
    strategy: Optional[str] = None,
    strategy_kwargs: Optional[dict[str, Any]] = None,
    alert_level: Optional[float] = None,
    alert_indicator: Optional[str] = None,
    alert_direction: int = -1,
    commission_per_trade: float = 0.0,
    slippage_bps: float = 0.0,
) -> dict[str, Any]:
    """Run a full ferro_ta pipeline in one call.

    Functional wrapper around :class:`Workflow` for scripting and agent use.

    Parameters
    ----------
    close : array-like
        Close price series.
    indicators : dict of {str: dict}, optional
        Mapping of ``output_key → kwargs_dict`` for indicators to compute.
        The indicator name must be embedded as ``"name"`` in the kwargs dict.

        Example::

            indicators = {
                "sma_20": {"name": "SMA", "timeperiod": 20},
                "rsi_14": {"name": "RSI", "timeperiod": 14},
            }

    strategy : str, optional
        Built-in strategy name (``"rsi_30_70"`` etc.).
    strategy_kwargs : dict, optional
        Extra kwargs for the strategy.
    alert_level : float, optional
        If set, add a threshold alert on *alert_indicator* at this level.
    alert_indicator : str, optional
        Key of the indicator to alert on (must be in *indicators*).
    alert_direction : int
        Direction of the alert: ``+1`` cross-above, ``-1`` cross-below.
    commission_per_trade : float
        Backtest commission.
    slippage_bps : float
        Backtest slippage in bps.

    Returns
    -------
    dict
        Same structure as :meth:`Workflow.run`.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.tools.workflow import run_pipeline
    >>> rng = np.random.default_rng(0)
    >>> close = np.cumprod(1 + rng.normal(0, 0.01, 200)) * 100
    >>> result = run_pipeline(
    ...     close,
    ...     indicators={
    ...         "sma_20": {"name": "SMA", "timeperiod": 20},
    ...         "rsi_14": {"name": "RSI", "timeperiod": 14},
    ...     },
    ...     strategy="rsi_30_70",
    ... )
    >>> "sma_20" in result
    True
    >>> "backtest" in result
    True
    """
    wf = Workflow()

    if indicators:
        for key, params in indicators.items():
            params = dict(params)
            ind_name = params.pop("name")
            wf.add_indicator(key, ind_name, **params)

    if strategy:
        wf.add_strategy(strategy, **(strategy_kwargs or {}))

    if alert_level is not None and alert_indicator is not None:
        wf.add_alert(alert_indicator, level=alert_level, direction=alert_direction)

    return wf.run(
        close,
        commission_per_trade=commission_per_trade,
        slippage_bps=slippage_bps,
    )
