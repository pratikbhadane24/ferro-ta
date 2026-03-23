"""
ferro_ta.alerts — Alerts and notification hooks.
================================================

Provides an ``AlertManager`` for registering conditions (threshold crossings,
series cross-overs) and dispatching events to callbacks and/or webhooks.
Supports both **backtest** mode (collect alerts in a list for analysis) and
**live** mode (invoke callbacks or POST to webhook URLs on each condition fire).

Quick start
-----------
>>> import numpy as np
>>> from ferro_ta.tools.alerts import AlertManager
>>> np.random.seed(0)
>>> close = 100 + np.cumsum(np.random.randn(200) * 0.5)
>>> from ferro_ta import RSI
>>> rsi = RSI(close, timeperiod=14)
>>> am = AlertManager()
>>> am.add_threshold_condition("rsi_oversold", rsi, level=30, direction=-1)
>>> am.add_threshold_condition("rsi_overbought", rsi, level=70, direction=1)
>>> fired = am.run_backtest()
>>> print(fired)

API
---
AlertManager
    Registry for conditions and callbacks.  Use ``add_threshold_condition``
    or ``add_cross_condition`` to register conditions, then call
    ``run_backtest()`` to evaluate all conditions at once.

check_threshold(series, level, direction)
    Low-level: return int8 mask — 1 where *series* crosses *level*.

check_cross(fast, slow)
    Low-level: return int8 mask — 1 (cross up), -1 (cross down), 0 (no cross).

collect_alert_bars(mask)
    Low-level: return indices where *mask* is non-zero.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ferro_ta._ferro_ta import check_cross as _rust_check_cross
from ferro_ta._ferro_ta import check_threshold as _rust_check_threshold
from ferro_ta._ferro_ta import collect_alert_bars as _rust_collect_alert_bars
from ferro_ta._utils import _to_f64

_log = logging.getLogger(__name__)

__all__ = [
    "AlertEvent",
    "AlertManager",
    "check_threshold",
    "check_cross",
    "collect_alert_bars",
]


# ---------------------------------------------------------------------------
# Low-level wrappers
# ---------------------------------------------------------------------------


def check_threshold(
    series: ArrayLike,
    level: float,
    direction: int,
) -> NDArray[np.int8]:
    """Fire an alert when *series* crosses a threshold *level*.

    Parameters
    ----------
    series    : array-like — indicator values (e.g. RSI close prices)
    level     : float — threshold value
    direction : int
        ``1``  → fire when *series* crosses **above** *level*.
        ``-1`` → fire when *series* crosses **below** *level*.

    Returns
    -------
    numpy.ndarray of int8 — 1 at the bar where the crossing occurs, 0 elsewhere.
    """
    return np.asarray(
        _rust_check_threshold(_to_f64(series), float(level), int(direction)),
        dtype=np.int8,
    )


def check_cross(
    fast: ArrayLike,
    slow: ArrayLike,
) -> NDArray[np.int8]:
    """Detect cross-over / cross-under events between two series.

    Parameters
    ----------
    fast : array-like — the "fast" series (e.g. short SMA)
    slow : array-like — the "slow" series (e.g. long SMA)

    Returns
    -------
    numpy.ndarray of int8:
      ``1``  at bars where *fast* crosses **above** *slow* (bullish).
      ``-1`` at bars where *fast* crosses **below** *slow* (bearish).
      ``0``  elsewhere.
    """
    return np.asarray(
        _rust_check_cross(_to_f64(fast), _to_f64(slow)),
        dtype=np.int8,
    )


def collect_alert_bars(mask: ArrayLike) -> NDArray[np.int64]:
    """Return bar indices where *mask* is non-zero (condition fired).

    Parameters
    ----------
    mask : array-like of int8 — output of ``check_threshold`` or ``check_cross``

    Returns
    -------
    numpy.ndarray of int64 — indices of fired bars (ascending order)
    """
    m = np.asarray(mask, dtype=np.int8)
    return np.asarray(_rust_collect_alert_bars(m), dtype=np.int64)


# ---------------------------------------------------------------------------
# AlertEvent
# ---------------------------------------------------------------------------


class AlertEvent:
    """A single alert event.

    Attributes
    ----------
    condition_id : str  — user-supplied condition name
    bar_index    : int  — bar index where the condition fired
    value        : float or None — optional series value at the fired bar
    payload      : dict — extra metadata (e.g. symbol, direction)
    """

    __slots__ = ("condition_id", "bar_index", "value", "payload")

    def __init__(
        self,
        condition_id: str,
        bar_index: int,
        value: Optional[float] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        self.condition_id = condition_id
        self.bar_index = bar_index
        self.value = value
        self.payload = payload or {}

    def __repr__(self) -> str:
        return (
            f"AlertEvent(condition_id={self.condition_id!r}, "
            f"bar_index={self.bar_index}, value={self.value})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return event as a plain dict (suitable for JSON serialisation)."""
        return {
            "condition_id": self.condition_id,
            "bar_index": self.bar_index,
            "value": self.value,
            **self.payload,
        }


# ---------------------------------------------------------------------------
# Internal dataclass for condition storage
# ---------------------------------------------------------------------------


@dataclass
class _AlertCondition:
    """Internal representation of a registered alert condition."""

    kind: str  # "threshold" or "cross"
    condition_id: str
    series_a: np.ndarray  # primary series (or fast series for cross)
    series_b: Optional[np.ndarray]  # slow series for cross, else None
    level: Optional[float]  # threshold level (threshold only)
    direction: Optional[int]  # +1 / -1 (threshold) or None (cross)
    callback: Optional[Callable[..., Any]]
    webhook_url: Optional[str]
    extra_payload: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------


class AlertManager:
    """Registry for alert conditions.

    Supports both **backtest** mode (collect events in a list) and
    **live** mode (dispatch via callback and/or webhook).

    Parameters
    ----------
    symbol : str, optional
        Symbol name included in every event payload.
    live : bool
        If ``True``, ``run_live()`` is used and callbacks/webhooks are invoked
        immediately.  In backtest mode (``live=False``, default) no external
        calls are made unless ``force_live=True`` in ``run_backtest()``.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.tools.alerts import AlertManager
    >>> from ferro_ta import RSI, SMA
    >>> close = np.cumprod(1 + np.random.randn(100) * 0.01) * 100
    >>> rsi = RSI(close)
    >>> sma20 = SMA(close, 20)
    >>> sma50 = SMA(close, 50)
    >>> am = AlertManager(symbol="BTC")
    >>> am.add_threshold_condition("rsi_os", rsi, level=30, direction=-1)
    >>> am.add_cross_condition("sma_x", sma20, sma50)
    >>> events = am.run_backtest()
    >>> for ev in events:
    ...     print(ev)
    """

    def __init__(
        self,
        symbol: str = "",
        live: bool = False,
    ) -> None:
        self._symbol = symbol
        self._live = live
        self._conditions: list[_AlertCondition] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_threshold_condition(
        self,
        condition_id: str,
        series: ArrayLike,
        level: float,
        direction: int,
        callback: Optional[Callable[[AlertEvent], None]] = None,
        webhook_url: Optional[str] = None,
        **extra_payload: Any,
    ) -> None:
        """Register a threshold crossing condition.

        Parameters
        ----------
        condition_id : str — unique name for this condition
        series       : array-like — the indicator / price series to watch
        level        : float — threshold level
        direction    : int — ``1`` (cross above) or ``-1`` (cross below)
        callback     : callable, optional — ``callback(event)`` invoked on fire
        webhook_url  : str, optional — HTTP POST target (live mode only)
        **extra_payload : extra keys merged into ``AlertEvent.payload``
        """
        self._conditions.append(
            _AlertCondition(
                kind="threshold",
                condition_id=condition_id,
                series_a=np.asarray(series, dtype=np.float64),
                series_b=None,
                level=float(level),
                direction=int(direction),
                callback=callback,
                webhook_url=webhook_url,
                extra_payload=dict(extra_payload),
            )
        )

    def add_cross_condition(
        self,
        condition_id: str,
        fast: ArrayLike,
        slow: ArrayLike,
        callback: Optional[Callable[[AlertEvent], None]] = None,
        webhook_url: Optional[str] = None,
        **extra_payload: Any,
    ) -> None:
        """Register a series cross-over / cross-under condition.

        Parameters
        ----------
        condition_id : str — unique name for this condition
        fast         : array-like — the "fast" series
        slow         : array-like — the "slow" series
        callback     : callable, optional — ``callback(event)`` invoked on fire
        webhook_url  : str, optional — HTTP POST target (live mode only)
        **extra_payload : extra keys merged into ``AlertEvent.payload``
        """
        self._conditions.append(
            _AlertCondition(
                kind="cross",
                condition_id=condition_id,
                series_a=np.asarray(fast, dtype=np.float64),
                series_b=np.asarray(slow, dtype=np.float64),
                level=None,
                direction=None,
                callback=callback,
                webhook_url=webhook_url,
                extra_payload=dict(extra_payload),
            )
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        force_live: bool = False,
    ) -> list[AlertEvent]:
        """Evaluate all registered conditions in batch (backtest mode).

        No callbacks or webhooks are invoked unless ``force_live=True``.

        Parameters
        ----------
        force_live : bool
            If ``True``, invoke callbacks and webhooks even in backtest mode.

        Returns
        -------
        list of :class:`AlertEvent` — all events that fired, sorted by bar
        index (then condition_id for ties).
        """
        events: list[AlertEvent] = []
        do_live = self._live or force_live

        for cond in self._conditions:
            if cond.kind == "threshold":
                mask = _rust_check_threshold(
                    np.ascontiguousarray(cond.series_a, dtype=np.float64),
                    float(cond.level),  # type: ignore[arg-type]
                    int(cond.direction),  # type: ignore[arg-type]
                )
                bars = _rust_collect_alert_bars(mask)
                for bar_idx in bars:
                    ev = AlertEvent(
                        condition_id=cond.condition_id,
                        bar_index=int(bar_idx),
                        value=float(cond.series_a[int(bar_idx)]),
                        payload={
                            "symbol": self._symbol,
                            "direction": int(cond.direction),  # type: ignore[arg-type]
                            **cond.extra_payload,
                        },
                    )
                    events.append(ev)
                    if do_live:
                        self._dispatch(ev, cond.callback, cond.webhook_url)
            elif cond.kind == "cross":
                mask = _rust_check_cross(
                    np.ascontiguousarray(cond.series_a, dtype=np.float64),
                    np.ascontiguousarray(cond.series_b, dtype=np.float64),  # type: ignore[arg-type]
                )
                bars = _rust_collect_alert_bars(mask)
                for bar_idx in bars:
                    cross_dir = int(mask[int(bar_idx)])
                    ev = AlertEvent(
                        condition_id=cond.condition_id,
                        bar_index=int(bar_idx),
                        value=float(cond.series_a[int(bar_idx)]),
                        payload={
                            "symbol": self._symbol,
                            "direction": cross_dir,
                            **cond.extra_payload,
                        },
                    )
                    events.append(ev)
                    if do_live:
                        self._dispatch(ev, cond.callback, cond.webhook_url)

        events.sort(key=lambda e: (e.bar_index, e.condition_id))
        return events

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dispatch(
        event: AlertEvent,
        callback: Optional[Callable[[AlertEvent], None]],
        webhook_url: Optional[str],
    ) -> None:
        """Invoke callback and/or HTTP POST to webhook."""
        if callback is not None:
            try:
                callback(event)
            except Exception as exc:  # noqa: BLE001
                _log.warning("Alert callback raised an exception: %s", exc)

        if webhook_url:
            AlertManager._post_webhook(webhook_url, event.to_dict())

    @staticmethod
    def _post_webhook(url: str, payload: dict[str, Any]) -> None:
        """HTTP POST *payload* as JSON to *url* (best-effort, no retry)."""
        import urllib.error
        import urllib.request

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5):
                pass
        except (urllib.error.URLError, OSError, ValueError) as exc:
            _log.warning("Webhook POST to %s failed: %s", url, exc)
