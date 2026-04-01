"""
Portfolio optimization utilities.

mean_variance_optimize(returns, target_return=None, allow_short=False)
    Minimum-variance portfolio (or target-return portfolio on efficient frontier).
    Uses scipy.optimize.minimize with SLSQP.
    Returns weight array summing to 1.

risk_parity_optimize(returns, risk_budget=None)
    Equal risk contribution portfolio (or custom risk budget).
    Each asset contributes equally to total portfolio volatility.
    Returns weight array summing to 1.

max_sharpe_optimize(returns, risk_free_rate=0.0)
    Maximize Sharpe ratio portfolio.
    Returns weight array.

PortfolioOptimizer
    Fluent builder that wraps the above functions and integrates with
    BacktestEngine for portfolio-level signal generation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray


def mean_variance_optimize(
    returns: ArrayLike,
    target_return: Optional[float] = None,
    allow_short: bool = False,
    risk_free_rate: float = 0.0,
) -> NDArray:
    """Compute minimum variance (or target return) portfolio weights.

    Parameters
    ----------
    returns : (T, N) array of asset returns
    target_return : float or None
        If None, return minimum-variance portfolio.
        If float, return minimum-variance portfolio with this expected return.
    allow_short : bool
        If False, weights are constrained to [0, 1].
    risk_free_rate : float
        Not used directly here (kept for API symmetry with max_sharpe).

    Returns
    -------
    weights : (N,) array summing to 1.0
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        raise ImportError(
            "scipy is required for portfolio optimization: pip install scipy"
        )

    r = np.asarray(returns, dtype=np.float64)
    if r.ndim == 1:
        r = r[:, np.newaxis]
    n_assets = r.shape[1]

    if n_assets == 1:
        return np.array([1.0])

    mu = r.mean(axis=0)
    cov = np.cov(r, rowvar=False)
    # Regularize to handle near-singular covariance matrices
    cov += 1e-8 * np.eye(n_assets)

    # Objective: minimize portfolio variance w^T @ cov @ w
    def portfolio_variance(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    def portfolio_variance_grad(w: np.ndarray) -> np.ndarray:
        return 2.0 * cov @ w

    # Constraints: weights sum to 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Optional target return constraint
    if target_return is not None:
        constraints.append(
            {"type": "eq", "fun": lambda w, mu=mu, tr=target_return: float(w @ mu) - tr}
        )

    # Bounds
    bounds = None if allow_short else [(0.0, 1.0)] * n_assets

    # Initial guess: equal weights
    w0 = np.ones(n_assets) / n_assets

    result = minimize(
        portfolio_variance,
        w0,
        jac=portfolio_variance_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    weights = result.x
    # Normalize to ensure exact sum=1 (numerical noise)
    weights = weights / weights.sum()
    if not allow_short:
        weights = np.maximum(weights, 0.0)
        s = weights.sum()
        if s > 0:
            weights /= s
    return weights


def risk_parity_optimize(
    returns: ArrayLike,
    risk_budget: Optional[ArrayLike] = None,
) -> NDArray:
    """Compute risk parity weights (equal risk contribution).

    Parameters
    ----------
    returns : (T, N) array of asset returns
    risk_budget : (N,) array or None
        Target risk contribution per asset (normalized internally). None = equal.

    Returns
    -------
    weights : (N,) array summing to 1.0
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        raise ImportError(
            "scipy is required for portfolio optimization: pip install scipy"
        )

    r = np.asarray(returns, dtype=np.float64)
    if r.ndim == 1:
        r = r[:, np.newaxis]
    n_assets = r.shape[1]

    if n_assets == 1:
        return np.array([1.0])

    cov = np.cov(r, rowvar=False)
    cov += 1e-8 * np.eye(n_assets)

    if risk_budget is None:
        budget = np.ones(n_assets) / n_assets
    else:
        budget = np.asarray(risk_budget, dtype=np.float64)
        budget = budget / budget.sum()

    def risk_contribution(w: np.ndarray) -> np.ndarray:
        """Return marginal risk contribution of each asset."""
        sigma = np.sqrt(w @ cov @ w)
        if sigma < 1e-12:
            return np.zeros(n_assets)
        mrc = cov @ w / sigma
        return w * mrc

    def objective(w: np.ndarray) -> float:
        """Minimize squared deviation from target risk budget."""
        rc = risk_contribution(w)
        total_rc = rc.sum()
        if total_rc < 1e-12:
            return float(np.sum((rc - budget) ** 2))
        rc_normalized = rc / total_rc
        return float(np.sum((rc_normalized - budget) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-6, 1.0)] * n_assets  # risk parity requires positive weights
    w0 = np.ones(n_assets) / n_assets

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )

    weights = result.x
    weights = np.maximum(weights, 0.0)
    s = weights.sum()
    if s > 0:
        weights /= s
    return weights


def max_sharpe_optimize(
    returns: ArrayLike,
    risk_free_rate: float = 0.0,
    allow_short: bool = False,
) -> NDArray:
    """Compute maximum Sharpe ratio portfolio weights.

    Returns
    -------
    weights : (N,) array summing to 1.0
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        raise ImportError(
            "scipy is required for portfolio optimization: pip install scipy"
        )

    r = np.asarray(returns, dtype=np.float64)
    if r.ndim == 1:
        r = r[:, np.newaxis]
    n_assets = r.shape[1]

    if n_assets == 1:
        return np.array([1.0])

    mu = r.mean(axis=0)
    cov = np.cov(r, rowvar=False)
    cov += 1e-8 * np.eye(n_assets)

    # Maximize Sharpe = minimize negative Sharpe
    def neg_sharpe(w: np.ndarray) -> float:
        port_return = float(w @ mu)
        port_vol = float(np.sqrt(w @ cov @ w))
        if port_vol < 1e-12:
            return 0.0
        return -(port_return - risk_free_rate) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = None if allow_short else [(0.0, 1.0)] * n_assets
    w0 = np.ones(n_assets) / n_assets

    result = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    weights = result.x
    weights = weights / weights.sum()
    if not allow_short:
        weights = np.maximum(weights, 0.0)
        s = weights.sum()
        if s > 0:
            weights /= s
    return weights


class PortfolioOptimizer:
    """Fluent interface for portfolio weight optimization.

    Example
    -------
    weights = (
        PortfolioOptimizer()
        .with_method("risk_parity")
        .with_lookback(252)
        .optimize(returns_matrix)
    )
    """

    def __init__(self) -> None:
        self._method: str = "min_variance"
        self._lookback: Optional[int] = None
        self._allow_short: bool = False
        self._risk_free_rate: float = 0.0
        self._target_return: Optional[float] = None
        self._risk_budget: Optional[NDArray] = None

    def with_method(self, method: str) -> PortfolioOptimizer:
        """Method: 'min_variance', 'risk_parity', 'max_sharpe'."""
        valid = ("min_variance", "risk_parity", "max_sharpe")
        if method not in valid:
            raise ValueError(f"method must be one of {valid}")
        self._method = method
        return self

    def with_lookback(self, n_bars: int) -> PortfolioOptimizer:
        """Use only the last n_bars for covariance estimation."""
        self._lookback = int(n_bars)
        return self

    def with_short_selling(self, allow: bool = True) -> PortfolioOptimizer:
        self._allow_short = allow
        return self

    def with_risk_free_rate(self, rate: float) -> PortfolioOptimizer:
        self._risk_free_rate = float(rate)
        return self

    def with_target_return(self, target: float) -> PortfolioOptimizer:
        self._target_return = float(target)
        return self

    def with_risk_budget(self, budget: ArrayLike) -> PortfolioOptimizer:
        self._risk_budget = np.asarray(budget, dtype=np.float64)
        return self

    def optimize(self, returns: ArrayLike) -> NDArray:
        """Run optimization and return weight array."""
        r = np.asarray(returns, dtype=np.float64)
        if self._lookback is not None:
            r = r[-self._lookback :]
        if self._method == "min_variance":
            return mean_variance_optimize(
                r, self._target_return, self._allow_short, self._risk_free_rate
            )
        elif self._method == "risk_parity":
            return risk_parity_optimize(r, self._risk_budget)
        else:
            return max_sharpe_optimize(r, self._risk_free_rate, self._allow_short)
