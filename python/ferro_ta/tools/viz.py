"""
ferro_ta.viz — Charting and visualisation API.

Generates charts (matplotlib and/or Plotly) with indicators overlaid on price.

API
---
plot(ohlcv, indicators=None, *, backend='matplotlib', title=None,
     figsize=None, savefig=None, show=False)
    Generate a chart from OHLCV data and optional indicator series.
    Returns a figure object for further customisation.

Backends
--------
- ``'matplotlib'`` — requires ``matplotlib`` (recommended for static charts)
- ``'plotly'``     — requires ``plotly`` (recommended for interactive charts)

Install optional backends::

    pip install ferro-ta[plot]   # adds matplotlib + plotly
    pip install matplotlib      # matplotlib only
    pip install plotly          # plotly only

Examples
--------
>>> import numpy as np
>>> from ferro_ta import RSI, SMA
>>> from ferro_ta.tools.viz import plot
>>> rng = np.random.default_rng(0)
>>> n = 60
>>> close = np.cumprod(1 + rng.normal(0, 0.01, n)) * 100
>>> ohlcv = {"close": close, "open": close, "high": close * 1.01,
...          "low": close * 0.99, "volume": np.ones(n) * 1000}
>>> fig = plot(ohlcv, indicators={"RSI(14)": RSI(close, timeperiod=14),
...                                "SMA(20)": SMA(close, timeperiod=20)},
...            backend='matplotlib', show=False)
>>> fig is not None
True
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "plot",
]

# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


def plot(
    ohlcv: Any,
    indicators: Optional[dict[str, ArrayLike]] = None,
    *,
    backend: str = "matplotlib",
    title: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    savefig: Optional[str] = None,
    show: bool = True,
    volume: bool = True,
    close_col: str = "close",
    volume_col: str = "volume",
) -> Any:
    """Generate a chart from OHLCV data and optional indicator series.

    Parameters
    ----------
    ohlcv : dict, pandas.DataFrame, or array-like
        OHLCV data.  At minimum a ``close`` key/column is required.
    indicators : dict {label: array}, optional
        Additional indicator series to plot below the price panel.
        Each entry is plotted in its own subplot.
    backend : str
        ``'matplotlib'`` (default) or ``'plotly'``.
    title : str, optional
        Chart title.
    figsize : (width, height), optional
        Figure size in inches (matplotlib) or pixels (plotly).
    savefig : str, optional
        Save figure to this file path (e.g. ``'chart.png'``, ``'chart.html'``).
    show : bool
        If ``True``, call ``plt.show()`` or ``fig.show()`` interactively.
    volume : bool
        If ``True`` and a volume series is present, add a volume subplot.
    close_col, volume_col : str
        Column names when *ohlcv* is a DataFrame.

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure

    Raises
    ------
    ImportError
        If the requested backend is not installed.
    """
    close_arr, volume_arr = _extract_close_volume(ohlcv, close_col, volume_col)

    if backend == "matplotlib":
        return _plot_matplotlib(
            close_arr,
            volume_arr if volume else None,
            indicators,
            title=title,
            figsize=figsize,
            savefig=savefig,
            show=show,
        )
    elif backend == "plotly":
        return _plot_plotly(
            close_arr,
            volume_arr if volume else None,
            indicators,
            title=title,
            figsize=figsize,
            savefig=savefig,
            show=show,
        )
    else:
        raise ValueError(
            f"Unknown backend {backend!r}. Supported: 'matplotlib', 'plotly'."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_close_volume(
    ohlcv: Any,
    close_col: str,
    volume_col: str,
) -> tuple[NDArray[np.float64], Optional[NDArray[np.float64]]]:
    """Extract close and (optional) volume from various input formats."""
    try:
        import pandas as pd

        if isinstance(ohlcv, pd.DataFrame):
            close = ohlcv[close_col].values.astype(np.float64)
            volume = (
                ohlcv[volume_col].values.astype(np.float64)
                if volume_col in ohlcv.columns
                else None
            )
            return close, volume
    except ImportError:
        pass

    if isinstance(ohlcv, dict):
        close = np.asarray(
            ohlcv.get(close_col, ohlcv.get("close", [])), dtype=np.float64
        )
        vol_key = volume_col if volume_col in ohlcv else "volume"
        volume = (
            np.asarray(ohlcv[vol_key], dtype=np.float64) if vol_key in ohlcv else None
        )
        return close, volume

    # Plain array
    return np.asarray(ohlcv, dtype=np.float64), None


def _n_subplots(indicators: Optional[dict], volume_arr: Optional[NDArray]) -> int:
    n = 1  # price
    if volume_arr is not None:
        n += 1
    if indicators:
        n += len(indicators)
    return n


# ---------------------------------------------------------------------------
# Matplotlib backend
# ---------------------------------------------------------------------------


def _plot_matplotlib(
    close: NDArray,
    volume: Optional[NDArray],
    indicators: Optional[dict[str, ArrayLike]],
    *,
    title: Optional[str],
    figsize: Optional[tuple],
    savefig: Optional[str],
    show: bool,
) -> Any:
    try:
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for the 'matplotlib' backend.  "
            "Install with: pip install matplotlib"
        ) from exc

    n_subplots = _n_subplots(indicators, volume)
    height_ratios = [3] + [1] * (n_subplots - 1)
    fig_h = figsize[1] if figsize else 2.5 * n_subplots + 1
    fig_w = figsize[0] if figsize else 12.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(n_subplots, 1, height_ratios=height_ratios, hspace=0.35)

    ax_price = fig.add_subplot(gs[0])
    ax_price.plot(close, color="#1f77b4", linewidth=1.2, label="close")
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left", fontsize=8)
    ax_price.grid(alpha=0.3)
    if title:
        ax_price.set_title(title)

    row = 1
    if volume is not None:
        ax_vol = fig.add_subplot(gs[row], sharex=ax_price)
        ax_vol.bar(range(len(volume)), volume, color="#aec7e8", alpha=0.7, width=0.8)
        ax_vol.set_ylabel("Volume")
        ax_vol.grid(alpha=0.3)
        row += 1

    if indicators:
        colors = ["#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
        for idx, (label, arr) in enumerate(indicators.items()):
            ax_ind = fig.add_subplot(gs[row], sharex=ax_price)
            color = colors[idx % len(colors)]
            arr_np = np.asarray(arr, dtype=np.float64)
            ax_ind.plot(arr_np, color=color, linewidth=1.0, label=label)
            ax_ind.set_ylabel(label, fontsize=8)
            ax_ind.legend(loc="upper left", fontsize=8)
            ax_ind.grid(alpha=0.3)
            row += 1

    # Use tight_layout when possible but suppress known benign UserWarning
    # about incompatible Axes configurations.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout.*",
            category=UserWarning,
        )
        plt.tight_layout()

    if savefig:
        fig.savefig(savefig, dpi=100, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Plotly backend
# ---------------------------------------------------------------------------


def _plot_plotly(
    close: NDArray,
    volume: Optional[NDArray],
    indicators: Optional[dict[str, ArrayLike]],
    *,
    title: Optional[str],
    figsize: Optional[tuple],
    savefig: Optional[str],
    show: bool,
) -> Any:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "plotly is required for the 'plotly' backend.  "
            "Install with: pip install plotly"
        ) from exc

    n_subplots = _n_subplots(indicators, volume)
    row_heights = [0.5] + [0.1] * (n_subplots - 1)
    total = sum(row_heights)
    row_heights = [r / total for r in row_heights]
    shared_xaxes = True
    subplot_titles = ["Price"]
    if volume is not None:
        subplot_titles.append("Volume")
    if indicators:
        subplot_titles.extend(list(indicators.keys()))

    fig = make_subplots(
        rows=n_subplots,
        cols=1,
        shared_xaxes=shared_xaxes,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
        vertical_spacing=0.05,
    )
    x = list(range(len(close)))
    fig.add_trace(
        go.Scatter(
            x=x, y=close.tolist(), mode="lines", name="close", line={"color": "#1f77b4"}
        ),
        row=1,
        col=1,
    )

    row = 2
    if volume is not None:
        fig.add_trace(
            go.Bar(x=x, y=volume.tolist(), name="volume", marker_color="#aec7e8"),
            row=row,
            col=1,
        )
        row += 1

    if indicators:
        colors = ["#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
        for idx, (label, arr) in enumerate(indicators.items()):
            arr_np = np.asarray(arr, dtype=np.float64)
            color = colors[idx % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=arr_np.tolist(),
                    mode="lines",
                    name=label,
                    line={"color": color},
                ),
                row=row,
                col=1,
            )
            row += 1

    fig_w = figsize[0] if figsize else 900
    fig_h = figsize[1] if figsize else 500
    fig.update_layout(
        title=title or "ferro_ta Chart",
        width=fig_w,
        height=fig_h,
        showlegend=True,
    )

    if savefig:
        if savefig.endswith(".html"):
            fig.write_html(savefig)
        else:
            fig.write_image(savefig)
    if show:
        fig.show()
    return fig
