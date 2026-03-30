"""
Visualization utilities for backtest results.

plot_backtest(result, *, title="Backtest", show=True, return_fig=False)
    Generate an interactive Plotly chart with:
    - Top panel: equity curve (normalized to 1.0)
    - Middle panel: drawdown series (negative values, shaded red)
    - Bottom panel: position/signal over time
    Optional trade markers: entry (green triangle up) and exit (red triangle down) on equity curve.

Requires plotly -- raises ImportError with install hint if not available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__ = ["plot_backtest"]


def plot_backtest(
    result,  # AdvancedBacktestResult
    *,
    title: str = "Backtest",
    show: bool = True,
    return_fig: bool = False,
    benchmark: bool = True,
):
    """Plot equity curve, drawdown, and positions.

    Parameters
    ----------
    result : AdvancedBacktestResult
        Backtest result object with equity, drawdown_series, positions, and trades.
    title : str
        Chart title.
    show : bool
        Call fig.show() if True.
    return_fig : bool
        Return the plotly Figure object.
    benchmark : bool
        Overlay benchmark equity curve if result has benchmark returns.

    Returns
    -------
    plotly.graph_objects.Figure if return_fig=True, else None.

    Raises
    ------
    ImportError
        If plotly is not installed.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError(
            "plotly is required for visualization. Install with: pip install plotly"
        )

    import numpy as np

    # ------------------------------------------------------------------
    # Extract result fields
    # ------------------------------------------------------------------
    equity = np.asarray(result.equity, dtype=np.float64)
    n = len(equity)
    bars = np.arange(n)

    # Drawdown: prefer pre-computed drawdown_series, else compute from equity
    if hasattr(result, "drawdown_series") and result.drawdown_series is not None:
        drawdown = np.asarray(result.drawdown_series, dtype=np.float64)
    else:
        cum_max = np.maximum.accumulate(equity)
        drawdown = np.where(cum_max > 0, equity / cum_max - 1.0, 0.0)

    positions = (
        np.asarray(result.positions, dtype=np.float64)
        if hasattr(result, "positions")
        else np.zeros(n)
    )

    # Trades (may be empty or None)
    trades = getattr(result, "trades", None)

    # Benchmark equity (optional)
    benchmark_equity = None
    if (
        benchmark
        and hasattr(result, "benchmark_equity")
        and result.benchmark_equity is not None
    ):
        benchmark_equity = np.asarray(result.benchmark_equity, dtype=np.float64)

    # ------------------------------------------------------------------
    # Build 3-panel subplot
    # ------------------------------------------------------------------
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.04,
        subplot_titles=("Equity Curve", "Drawdown", "Positions"),
    )

    # ---- Panel 1: Equity curve ----------------------------------------
    fig.add_trace(
        go.Scatter(
            x=bars,
            y=equity,
            name="Strategy",
            line=dict(color="#00d4ff", width=1.5),
            hovertemplate="Bar %{x}<br>Equity: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Benchmark overlay
    if benchmark_equity is not None:
        fig.add_trace(
            go.Scatter(
                x=bars[: len(benchmark_equity)],
                y=benchmark_equity,
                name="Benchmark",
                line=dict(color="#f0a500", width=1.2, dash="dot"),
                hovertemplate="Bar %{x}<br>Benchmark: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Trade markers
    if trades is not None and hasattr(trades, "__len__") and len(trades) > 0:
        # trades may be a pd.DataFrame or a list of dicts
        try:
            # pandas DataFrame path
            entry_bars = trades["entry_bar"].values
            exit_bars = trades["exit_bar"].values
        except (TypeError, KeyError, AttributeError):
            # list-of-dicts path
            try:
                entry_bars = np.array([t["entry_bar"] for t in trades])
                exit_bars = np.array([t["exit_bar"] for t in trades])
            except (KeyError, TypeError):
                entry_bars = np.array([])
                exit_bars = np.array([])

        if len(entry_bars) > 0:
            # Clip indices to equity length
            entry_bars = np.clip(entry_bars.astype(int), 0, n - 1)
            exit_bars = np.clip(exit_bars.astype(int), 0, n - 1)

            fig.add_trace(
                go.Scatter(
                    x=entry_bars,
                    y=equity[entry_bars],
                    mode="markers",
                    name="Entry",
                    marker=dict(
                        symbol="triangle-up",
                        size=10,
                        color="lime",
                        line=dict(color="darkgreen", width=1),
                    ),
                    hovertemplate="Entry Bar %{x}<br>Equity: %{y:.4f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=exit_bars,
                    y=equity[exit_bars],
                    mode="markers",
                    name="Exit",
                    marker=dict(
                        symbol="triangle-down",
                        size=10,
                        color="red",
                        line=dict(color="darkred", width=1),
                    ),
                    hovertemplate="Exit Bar %{x}<br>Equity: %{y:.4f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    # ---- Panel 2: Drawdown -------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=bars,
            y=drawdown,
            name="Drawdown",
            fill="tozeroy",
            fillcolor="rgba(220, 50, 50, 0.25)",
            line=dict(color="rgba(220, 50, 50, 0.8)", width=1.0),
            hovertemplate="Bar %{x}<br>Drawdown: %{y:.2%}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # ---- Panel 3: Positions ------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=bars,
            y=positions,
            name="Position",
            fill="tozeroy",
            fillcolor="rgba(0, 150, 255, 0.2)",
            line=dict(color="rgba(0, 150, 255, 0.7)", width=1.0),
            hovertemplate="Bar %{x}<br>Position: %{y:.2f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # ------------------------------------------------------------------
    # Styling: dark theme + ferro-ta branding
    # ------------------------------------------------------------------
    metrics = getattr(result, "metrics", {})
    sharpe_str = f"Sharpe: {metrics.get('sharpe', float('nan')):.2f}" if metrics else ""
    dd_str = (
        f"Max DD: {metrics.get('max_drawdown', float('nan')):.1%}" if metrics else ""
    )
    subtitle = "  |  ".join(filter(None, [sharpe_str, dd_str]))

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" + (f"<br><sub>{subtitle}</sub>" if subtitle else ""),
            font=dict(size=18, color="#e0e0e0"),
        ),
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#b0b8c1", size=11),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
        height=700,
        margin=dict(l=60, r=40, t=80, b=40),
    )

    # Axis styling
    axis_style = dict(
        gridcolor="rgba(255,255,255,0.07)",
        zerolinecolor="rgba(255,255,255,0.15)",
        tickfont=dict(size=10),
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)

    # Y-axis labels
    fig.update_yaxes(title_text="Equity (norm.)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1)
    fig.update_yaxes(title_text="Position", row=3, col=1)
    fig.update_xaxes(title_text="Bar", row=3, col=1)

    # ------------------------------------------------------------------
    if show:
        fig.show()

    if return_fig:
        return fig

    return None
