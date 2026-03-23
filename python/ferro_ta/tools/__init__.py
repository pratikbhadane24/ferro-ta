"""
ferro_ta.tools — Developer tools, visualisation, alerting, and workflow utilities.

Sub-modules
-----------
* :mod:`ferro_ta.tools.tools`    — General-purpose utility helpers (compute_indicator, run_backtest, …)
* :mod:`ferro_ta.tools.viz`      — Charting and visualisation API (matplotlib)
* :mod:`ferro_ta.tools.dashboard`— Interactive Streamlit/Dash dashboard helpers
* :mod:`ferro_ta.tools.alerts`   — Alert manager and threshold checks
* :mod:`ferro_ta.tools.dsl`      — Strategy expression DSL
* :mod:`ferro_ta.tools.pipeline` — Indicator pipeline builder
* :mod:`ferro_ta.tools.workflow` — Workflow automation helpers
* :mod:`ferro_ta.tools.api_info` — API discovery helpers (:func:`indicators`, :func:`info`)
* :mod:`ferro_ta.tools.gpu`      — GPU-accelerated indicator support (requires PyTorch)

Example usage::

    from ferro_ta.tools import compute_indicator, run_backtest, list_indicators
    from ferro_ta.tools.alerts import check_cross
"""

# Re-export the stable public API from tools.tools.
# tools/tools.py has no ferro_ta module-level imports, so this is safe.
from ferro_ta.tools.tools import (  # noqa: F401
    compute_indicator,
    describe_indicator,
    list_indicators,
    run_backtest,
)

