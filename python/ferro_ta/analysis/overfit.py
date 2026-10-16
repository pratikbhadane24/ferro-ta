"""
Backtest-overfitting statistics
===============================

Run enough parameter combinations and one of them will look brilliant by
chance. This module quantifies how much of a backtest's edge survives that
selection bias. Every heavy computation runs in Rust.

* :func:`probabilistic_sharpe_ratio` — P(true Sharpe > benchmark), corrected
  for sample length, skewness and fat tails (Bailey & López de Prado, 2012).
* :func:`deflated_sharpe_ratio` — PSR measured against the best Sharpe you
  would expect from *N* zero-skill trials (Bailey & López de Prado, 2014).
* :func:`pbo` — Probability of Backtest Overfitting via combinatorially
  symmetric cross-validation (Bailey, Borwein, López de Prado & Zhu, 2017).
* :func:`purged_kfold` — k-fold splits with purge/embargo gaps for labels that
  overlap in time.

All Sharpe ratios in this module are **per-period** (mean / std of the raw
returns, *not* annualised).

Example
-------
>>> import numpy as np
>>> from ferro_ta.analysis.overfit import deflated_sharpe_ratio, pbo, per_period_sharpe
>>> rng = np.random.default_rng(0)
>>> trials = rng.normal(0.0, 0.01, size=(1000, 50))   # 50 noise strategies
>>> best = trials[:, trials.mean(0).argmax()]
>>> dsr = deflated_sharpe_ratio(best, trial_sharpes=per_period_sharpe(trials))
>>> result = pbo(trials, n_splits=10)                 # ~0.5 for pure noise
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ferro_ta._ferro_ta import deflated_sharpe_ratio as _rust_dsr
from ferro_ta._ferro_ta import expected_max_sharpe as _rust_expected_max_sharpe
from ferro_ta._ferro_ta import pbo_cscv as _rust_pbo_cscv
from ferro_ta._ferro_ta import probabilistic_sharpe_ratio as _rust_psr
from ferro_ta._ferro_ta import purged_kfold_indices as _rust_purged_kfold_indices
from ferro_ta.core.exceptions import FerroTAInputError, _normalize_rust_error

__all__ = [
    "PBOResult",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "pbo",
    "per_period_sharpe",
    "probabilistic_sharpe_ratio",
    "purged_kfold",
    "purged_kfold_indices",
]


def _call_rust(fn: Callable[..., Any], *args: Any) -> Any:
    """Call a Rust binding, re-raising its ValueError as a FerroTA* error."""
    try:
        return fn(*args)
    except (ValueError, OverflowError) as err:
        _normalize_rust_error(ValueError(str(err)))


def _frozen(arr: Any) -> NDArray[Any]:
    """Read-only view, so frozen result objects really are immutable."""
    out = np.asarray(arr)
    out.setflags(write=False)
    return out


def _as_1d(returns: ArrayLike, name: str = "returns") -> NDArray[np.float64]:
    arr = np.ascontiguousarray(returns, dtype=np.float64)
    if arr.ndim != 1:
        raise FerroTAInputError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr


def _as_2d(returns_2d: ArrayLike) -> NDArray[np.float64]:
    arr = np.ascontiguousarray(returns_2d, dtype=np.float64)
    if arr.ndim != 2:
        raise FerroTAInputError(
            f"returns_2d must be 2-D (n_bars, n_strategies), got shape {arr.shape}"
        )
    return arr


def per_period_sharpe(returns_2d: ArrayLike) -> NDArray[np.float64]:
    """Per-period Sharpe (mean / sample std) of each column; 0 where std is 0."""
    arr = _as_2d(returns_2d)
    std = arr.std(axis=0, ddof=1)
    mean = arr.mean(axis=0)
    safe = np.where(std > 0, std, 1.0)
    return np.where(std > 0, mean / safe, 0.0)


def probabilistic_sharpe_ratio(
    returns: ArrayLike, benchmark_sharpe: float = 0.0
) -> float:
    """Probability that the true per-period Sharpe exceeds ``benchmark_sharpe``.

    Parameters
    ----------
    returns : array-like
        Per-period strategy returns (at least 3, finite, non-constant).
    benchmark_sharpe : float
        Per-period Sharpe to beat (default 0).

    Returns
    -------
    float
        PSR in [0, 1]; NaN when the skew/kurtosis correction is undefined.
    """
    return float(_call_rust(_rust_psr, _as_1d(returns), float(benchmark_sharpe)))


def expected_max_sharpe(n_trials: int, trial_sharpe_variance: float) -> float:
    """Expected best per-period Sharpe among ``n_trials`` zero-skill trials."""
    return float(
        _call_rust(
            _rust_expected_max_sharpe, int(n_trials), float(trial_sharpe_variance)
        )
    )


def deflated_sharpe_ratio(
    returns: ArrayLike,
    *,
    trial_sharpes: Optional[ArrayLike] = None,
    n_trials: Optional[int] = None,
    trial_sharpe_variance: Optional[float] = None,
) -> float:
    """Deflated Sharpe Ratio of the selected strategy.

    Pass either ``trial_sharpes`` (per-period Sharpe of *every* configuration
    tried, including the selected one) or both ``n_trials`` and
    ``trial_sharpe_variance``.

    A DSR above ~0.95 suggests the edge survives the multiple-testing
    correction; values near or below 0.5 mean the best result is consistent
    with luck.
    """
    if trial_sharpes is not None:
        if n_trials is not None or trial_sharpe_variance is not None:
            raise FerroTAInputError(
                "pass either trial_sharpes or (n_trials, trial_sharpe_variance), not both"
            )
        sharpes = _as_1d(trial_sharpes, "trial_sharpes")
        if len(sharpes) < 1 or not np.all(np.isfinite(sharpes)):
            raise FerroTAInputError("trial_sharpes must be non-empty and finite")
        n_trials = len(sharpes)
        trial_sharpe_variance = float(sharpes.var(ddof=1)) if n_trials > 1 else 0.0
    elif n_trials is None or trial_sharpe_variance is None:
        raise FerroTAInputError(
            "deflated_sharpe_ratio needs trial_sharpes or both n_trials and "
            "trial_sharpe_variance"
        )
    return float(
        _call_rust(
            _rust_dsr, _as_1d(returns), int(n_trials), float(trial_sharpe_variance)
        )
    )


@dataclasses.dataclass(frozen=True)
class PBOResult:
    """Result of :func:`pbo`; arrays have one entry per IS/OOS combination.

    Immutable: arrays are read-only.

    Attributes
    ----------
    pbo : float
        Probability of Backtest Overfitting — fraction of combinations where
        the in-sample winner landed at or below the out-of-sample median.
    logits : NDArray[np.float64]
        Logit of the IS winner's OOS relative rank (> 0 is good).
    is_sharpe, oos_sharpe : NDArray[np.float64]
        Per-period Sharpe of the IS winner, in and out of sample. Regressing
        ``oos_sharpe`` on ``is_sharpe`` gives the performance degradation.
    best_index : NDArray[np.int64]
        Column chosen as IS winner in each combination.
    """

    pbo: float
    logits: NDArray[np.float64]
    is_sharpe: NDArray[np.float64]
    oos_sharpe: NDArray[np.float64]
    best_index: NDArray[np.int64]

    @property
    def n_combinations(self) -> int:
        return int(len(self.logits))

    @property
    def prob_oos_loss(self) -> float:
        """Fraction of combinations where the IS winner lost money OOS."""
        return float(np.mean(self.oos_sharpe < 0.0))


def pbo(returns_2d: ArrayLike, n_splits: int = 16) -> PBOResult:
    """Probability of Backtest Overfitting via CSCV.

    Parameters
    ----------
    returns_2d : array-like, shape (n_bars, n_strategies)
        Per-period returns, one column per parameter configuration — e.g.
        :attr:`ferro_ta.analysis.sweep.SweepResult.returns`.
    n_splits : int
        Even number of contiguous time blocks (2–20). Evaluates
        C(n_splits, n_splits / 2) combinations: 12 870 for the default 16.
    """
    raw = _call_rust(_rust_pbo_cscv, _as_2d(returns_2d), int(n_splits))
    return PBOResult(
        pbo=float(raw["pbo"]),
        logits=_frozen(raw["logits"]),
        is_sharpe=_frozen(raw["is_sharpe"]),
        oos_sharpe=_frozen(raw["oos_sharpe"]),
        best_index=_frozen(raw["best_index"]),
    )


def purged_kfold_indices(
    n_bars: int, n_folds: int = 5, *, purge_bars: int = 0, embargo_bars: int = 0
) -> NDArray[np.int64]:
    """Fold boundaries, shape (n_folds, 4).

    Columns are ``[train_left_end, test_start, test_end, train_right_start]``:
    train = ``[0, train_left_end) ∪ [train_right_start, n_bars)``.
    """
    raw = _call_rust(
        _rust_purged_kfold_indices,
        int(n_bars),
        int(n_folds),
        int(purge_bars),
        int(embargo_bars),
    )
    return np.asarray(raw, dtype=np.int64)


def purged_kfold(
    n_bars: int, n_folds: int = 5, *, purge_bars: int = 0, embargo_bars: int = 0
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Purged/embargoed k-fold as ``(train_idx, test_idx)`` index arrays.

    ``purge_bars`` should be at least your label horizon; ``embargo_bars``
    guards against serial correlation leaking from test back into training.
    """
    bounds = purged_kfold_indices(
        n_bars, n_folds, purge_bars=purge_bars, embargo_bars=embargo_bars
    )
    return [
        (
            np.concatenate([np.arange(0, a), np.arange(d, n_bars)]).astype(np.int64),
            np.arange(b, c, dtype=np.int64),
        )
        for a, b, c, d in bounds.tolist()
    ]
