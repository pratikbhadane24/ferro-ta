"""
ferro_ta.analysis — Portfolio analytics, strategy analysis, and financial modelling.


Sub-modules
-----------
* :mod:`ferro_ta.analysis.portfolio`    — Portfolio and multi-asset analytics
* :mod:`ferro_ta.analysis.backtest`     — Vectorised back-testing helpers
* :mod:`ferro_ta.analysis.regime`       — Market regime detection
* :mod:`ferro_ta.analysis.cross_asset`  — Cross-asset and relative-strength analysis
* :mod:`ferro_ta.analysis.attribution`  — Return attribution
* :mod:`ferro_ta.analysis.signals`      — Signal composition and screening
* :mod:`ferro_ta.analysis.features`     — Feature matrix and ML readiness helpers
* :mod:`ferro_ta.analysis.crypto`       — Crypto-specific indicators and helpers
* :mod:`ferro_ta.analysis.options`      — Options pricing, Greeks, IV, and smile analytics
* :mod:`ferro_ta.analysis.futures`      — Futures basis, curve, roll, and synthetic analytics
* :mod:`ferro_ta.analysis.options_strategy` — Typed derivatives strategy schemas
* :mod:`ferro_ta.analysis.derivatives_payoff` — Multi-leg payoff and Greeks aggregation
* :mod:`ferro_ta.analysis.resample`     — OHLCV bar aggregation utilities
* :mod:`ferro_ta.analysis.multitf`      — Multi-timeframe signal utilities
* :mod:`ferro_ta.analysis.adjust`       — Corporate action price adjustment utilities
* :mod:`ferro_ta.analysis.plot`         — Plotly-based backtest visualization

Example usage::

    from ferro_ta.analysis.portfolio import portfolio_returns
    from ferro_ta.analysis.backtest import backtest
    from ferro_ta.analysis.resample import resample_ohlcv, align_to_coarse, resample_ohlcv_labels
    from ferro_ta.analysis.multitf import MultiTimeframeEngine
    from ferro_ta.analysis.adjust import adjust_ohlcv, adjust_for_splits, adjust_for_dividends
    from ferro_ta.analysis.plot import plot_backtest
"""

import importlib as _importlib

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "detect_volatility_regime": (
        "ferro_ta.analysis.regime",
        "detect_volatility_regime",
    ),
    "detect_trend_regime": ("ferro_ta.analysis.regime", "detect_trend_regime"),
    "detect_combined_regime": ("ferro_ta.analysis.regime", "detect_combined_regime"),
    "RegimeFilter": ("ferro_ta.analysis.regime", "RegimeFilter"),
    "PortfolioOptimizer": ("ferro_ta.analysis.optimize", "PortfolioOptimizer"),
    "mean_variance_optimize": ("ferro_ta.analysis.optimize", "mean_variance_optimize"),
    "risk_parity_optimize": ("ferro_ta.analysis.optimize", "risk_parity_optimize"),
    "max_sharpe_optimize": ("ferro_ta.analysis.optimize", "max_sharpe_optimize"),
    "PaperTrader": ("ferro_ta.analysis.live", "PaperTrader"),
    "BarResult": ("ferro_ta.analysis.live", "BarResult"),
    "TradeRecord": ("ferro_ta.analysis.live", "TradeRecord"),
}


def __getattr__(name: str):
    """Lazy imports for heavy sub-modules to avoid startup cost."""
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path)
        obj = getattr(mod, attr)
        globals()[name] = obj  # cache so subsequent access skips __getattr__
        return obj
    raise AttributeError(f"module 'ferro_ta.analysis' has no attribute {name!r}")
