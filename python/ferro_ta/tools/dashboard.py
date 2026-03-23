"""
ferro_ta.dashboard — Interactive dashboards and exploration helpers.
===================================================================

Optional helpers for interactive exploration in Jupyter notebooks (via
ipywidgets) and a Streamlit template.  All widgets are optional: if ipywidgets
or streamlit are not installed, a clear ``ImportError`` is raised with install
instructions.

Functions
---------
indicator_widget(close, indicator_fn, param_name, param_range)
    Create an ipywidgets slider that updates an indicator plot in real time.

backtest_widget(close, strategy_fn, param_name, param_range)
    Create an ipywidgets slider that re-runs a backtest and shows equity curve.

streamlit_app()
    Launch a minimal Streamlit dashboard (call from a ``streamlit run`` script).

Notes
-----
To install optional dependencies::

    pip install ferro-ta[dashboard]          # installs ipywidgets
    pip install streamlit                   # for Streamlit app

Only the Python layer is in this module — all heavy computation delegated to
existing ferro-ta indicator and backtest functions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "indicator_widget",
    "backtest_widget",
    "streamlit_app",
]


# ---------------------------------------------------------------------------
# Jupyter / ipywidgets helpers
# ---------------------------------------------------------------------------


def indicator_widget(
    close: ArrayLike,
    indicator_fn: Callable[..., Any],
    param_name: str,
    param_range: Sequence[int],
    title: str = "Indicator",
) -> Any:
    """Create an interactive Jupyter widget with a parameter slider.

    Renders a ``matplotlib`` chart with the close price overlaid by the
    indicator output.  Dragging the slider updates the chart in real time.

    Parameters
    ----------
    close        : array-like — close price series
    indicator_fn : callable — indicator function, e.g. ``ferro_ta.SMA``.
        Signature: ``fn(close, **{param_name: value}) -> ndarray``.
    param_name   : str — name of the integer parameter to vary (e.g. ``'timeperiod'``).
    param_range  : sequence of int — values to iterate over (e.g. ``range(5, 51)``).
    title        : str — chart title.

    Returns
    -------
    ipywidgets ``Output`` widget — display it in a Jupyter cell.

    Requires
    --------
    ``ipywidgets``, ``matplotlib``

    Examples
    --------
    >>> from ferro_ta import SMA
    >>> from ferro_ta.tools.dashboard import indicator_widget
    >>> w = indicator_widget(close, SMA, 'timeperiod', range(5, 51))
    >>> display(w)  # in a Jupyter cell
    """
    try:
        import ipywidgets as widgets
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "indicator_widget requires ipywidgets and matplotlib.\n"
            "Install with: pip install ipywidgets matplotlib"
        ) from exc

    c = np.asarray(close, dtype=np.float64)
    param_values = list(param_range)

    out = widgets.Output()

    def update(change: Any) -> None:
        value = change["new"]
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(c, label="Close", alpha=0.5)
            ind_out = indicator_fn(c, **{param_name: value})
            if isinstance(ind_out, tuple):
                for arr in ind_out:
                    ax.plot(np.asarray(arr, dtype=np.float64), alpha=0.8)
            else:
                ax.plot(
                    np.asarray(ind_out, dtype=np.float64),
                    label=f"{indicator_fn.__name__}({param_name}={value})",
                )
            ax.set_title(f"{title} — {param_name}={value}")
            ax.legend()
            plt.tight_layout()
            plt.show()

    slider = widgets.IntSlider(
        value=param_values[len(param_values) // 2],
        min=min(param_values),
        max=max(param_values),
        step=1,
        description=param_name,
        continuous_update=False,
    )
    slider.observe(update, names="value")
    update({"new": slider.value})

    return widgets.VBox([slider, out])


def backtest_widget(
    close: ArrayLike,
    strategy: Union[str, Callable[..., Any]] = "rsi_30_70",
    param_name: str = "timeperiod",
    param_range: Sequence[int] = range(5, 30),
    title: str = "Backtest",
) -> Any:
    """Create an interactive Jupyter widget that re-runs a backtest on slider change.

    Parameters
    ----------
    close      : array-like — close prices
    strategy   : str or callable — backtest strategy (see ``ferro_ta.backtest.backtest``).
    param_name : str — strategy parameter name to vary.
    param_range: sequence of int — parameter values to iterate.
    title      : str — chart title.

    Returns
    -------
    ipywidgets ``VBox`` widget.

    Requires
    --------
    ``ipywidgets``, ``matplotlib``
    """
    try:
        import ipywidgets as widgets
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "backtest_widget requires ipywidgets and matplotlib.\n"
            "Install with: pip install ipywidgets matplotlib"
        ) from exc

    from ferro_ta.analysis.backtest import backtest

    c = np.asarray(close, dtype=np.float64)
    param_values = list(param_range)
    out = widgets.Output()

    def update(change: Any) -> None:
        value = change["new"]
        with out:
            out.clear_output(wait=True)
            result = backtest(c, strategy=strategy, **{param_name: value})
            fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            axes[0].plot(c, label="Close", alpha=0.7)
            axes[0].set_title(f"{title} — {param_name}={value}")
            axes[0].legend()
            axes[1].plot(result.equity, label="Equity", color="green")
            axes[1].axhline(1.0, color="gray", linestyle="--", alpha=0.5)
            axes[1].set_title(
                f"Equity (trades={result.n_trades}, final={result.final_equity:.3f})"
            )
            axes[1].legend()
            plt.tight_layout()
            plt.show()

    slider = widgets.IntSlider(
        value=param_values[len(param_values) // 2],
        min=min(param_values),
        max=max(param_values),
        step=1,
        description=param_name,
        continuous_update=False,
    )
    slider.observe(update, names="value")
    update({"new": slider.value})
    return widgets.VBox([slider, out])


# ---------------------------------------------------------------------------
# Streamlit app template
# ---------------------------------------------------------------------------


def streamlit_app() -> None:
    """Run a minimal Streamlit TA dashboard.

    Call this function from a Python script and run with::

        streamlit run your_script.py

    The dashboard provides:
    - A file uploader for OHLCV CSV data (or uses synthetic data as fallback).
    - An indicator selector (SMA, EMA, RSI, MACD, Bollinger Bands).
    - A parameter slider.
    - A price + indicator chart.
    - A backtest panel (RSI strategy) with equity curve.

    Requires
    --------
    ``streamlit``, ``matplotlib`` or ``plotly`` (optional)

    Examples
    --------
    Create a file ``ta_dashboard.py``::

        from ferro_ta.tools.dashboard import streamlit_app
        streamlit_app()

    Then run::

        streamlit run ta_dashboard.py
    """
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError(
            "streamlit_app requires streamlit.\nInstall with: pip install streamlit"
        ) from exc

    import ferro_ta as ft
    from ferro_ta.analysis.backtest import backtest

    st.title("ferro-ta Interactive Dashboard")

    # ---- Data ----
    st.sidebar.header("Data")
    uploaded = st.sidebar.file_uploader("Upload OHLCV CSV", type=["csv"])

    if uploaded is not None:
        try:
            import pandas as pd

            df = pd.read_csv(uploaded)
            cols = {c.lower(): c for c in df.columns}
            close = df[cols["close"]].values.astype(np.float64)
        except (ImportError, KeyError, ValueError) as e:
            st.error(f"Could not read CSV: {e}")
            close = _synthetic_close()
    else:
        st.info(
            "Using synthetic data.  Upload a CSV with a 'close' column to use real data."
        )
        close = _synthetic_close()

    n = len(close)
    st.sidebar.write(f"Bars loaded: {n}")

    # ---- Indicator ----
    st.sidebar.header("Indicator")
    indicator_name = st.sidebar.selectbox(
        "Indicator", ["SMA", "EMA", "RSI", "MACD", "BBANDS"]
    )
    timeperiod = st.sidebar.slider("Period", min_value=2, max_value=200, value=20)

    st.subheader(f"Price + {indicator_name}({timeperiod})")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(close, label="Close", alpha=0.5)

        if indicator_name == "SMA":
            ax.plot(np.asarray(ft.SMA(close, timeperiod=timeperiod)), label="SMA")
        elif indicator_name == "EMA":
            ax.plot(np.asarray(ft.EMA(close, timeperiod=timeperiod)), label="EMA")
        elif indicator_name == "RSI":
            fig2, ax2 = plt.subplots(figsize=(12, 2))
            ax2.plot(
                np.asarray(ft.RSI(close, timeperiod=timeperiod)),
                label="RSI",
                color="orange",
            )
            ax2.axhline(30, color="green", linestyle="--", alpha=0.5)
            ax2.axhline(70, color="red", linestyle="--", alpha=0.5)
            ax2.set_title("RSI")
            st.pyplot(fig2)
        elif indicator_name == "MACD":
            macd, signal, hist = ft.MACD(close)
            ax.plot(np.asarray(macd), label="MACD")
            ax.plot(np.asarray(signal), label="Signal")
        elif indicator_name == "BBANDS":
            upper, middle, lower = ft.BBANDS(close, timeperiod=timeperiod)
            ax.plot(np.asarray(upper), label="Upper", linestyle="--")
            ax.plot(np.asarray(middle), label="Middle")
            ax.plot(np.asarray(lower), label="Lower", linestyle="--")

        ax.legend()
        st.pyplot(fig)
    except (ImportError, ValueError, RuntimeError) as e:
        st.error(f"Error computing indicator: {e}")

    # ---- Backtest panel ----
    st.subheader("Backtest (RSI 30/70 strategy)")
    if st.button("Run Backtest"):
        result = backtest(close, strategy="rsi_30_70", timeperiod=timeperiod)
        try:
            import matplotlib.pyplot as plt

            fig3, ax3 = plt.subplots(figsize=(12, 3))
            ax3.plot(result.equity, color="green", label="Equity")
            ax3.axhline(1.0, color="gray", linestyle="--")
            ax3.set_title(
                f"Equity  trades={result.n_trades}  final={result.final_equity:.4f}"
            )
            ax3.legend()
            st.pyplot(fig3)
        except ImportError:
            st.write(
                f"Final equity: {result.final_equity:.4f}  trades: {result.n_trades}"
            )


def _synthetic_close(n: int = 500) -> NDArray:
    """Generate a synthetic close price series for the dashboard demo."""
    rng = np.random.default_rng(42)
    return np.cumprod(1 + rng.normal(0, 0.01, n)) * 100.0
