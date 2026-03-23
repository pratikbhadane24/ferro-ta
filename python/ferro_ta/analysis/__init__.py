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

Example usage::

    from ferro_ta.analysis.portfolio import portfolio_returns
    from ferro_ta.analysis.backtest import backtest
"""
