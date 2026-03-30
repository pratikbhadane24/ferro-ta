"""
Multi-timeframe signal utilities.

MultiTimeframeEngine wraps BacktestEngine with a higher-timeframe signal computation step.

Usage:
    from ferro_ta.analysis.multitf import MultiTimeframeEngine

    result = (
        MultiTimeframeEngine(factor=4)          # 4 fine bars per coarse bar
        .with_htf_strategy("rsi_30_70")         # strategy runs on coarse bars
        .with_ohlcv(high=h, low=l, open_=o)
        .with_stop_loss(0.02)
        .run(close_fine)
    )
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ferro_ta.analysis.backtest import AdvancedBacktestResult, BacktestEngine
from ferro_ta.analysis.resample import align_to_coarse, resample_ohlcv

__all__ = ["MultiTimeframeEngine"]


class MultiTimeframeEngine:
    """Backtests using signals computed on a higher timeframe (coarser bars).

    Parameters
    ----------
    factor : int
        Number of fine-resolution bars per coarse bar.
    """

    def __init__(self, factor: int) -> None:
        if factor < 1:
            raise ValueError(f"factor must be >= 1, got {factor}")
        self._factor = factor
        self._htf_strategy = "rsi_30_70"
        self._inner = BacktestEngine()

        # Store OHLCV separately so we can resample them
        self._high: np.ndarray | None = None
        self._low: np.ndarray | None = None
        self._open: np.ndarray | None = None

    def with_htf_strategy(self, strategy) -> MultiTimeframeEngine:
        """Set the strategy function or name used on coarse bars."""
        self._htf_strategy = strategy
        return self

    def with_ohlcv(self, *, high, low, open_) -> MultiTimeframeEngine:
        """Store OHLCV data for resampling and pass to inner engine after resampling."""
        self._high = np.asarray(high, dtype=np.float64)
        self._low = np.asarray(low, dtype=np.float64)
        self._open = np.asarray(open_, dtype=np.float64)
        return self

    def with_stop_loss(self, pct: float) -> MultiTimeframeEngine:
        self._inner.with_stop_loss(pct)
        return self

    def with_take_profit(self, pct: float) -> MultiTimeframeEngine:
        self._inner.with_take_profit(pct)
        return self

    def with_trailing_stop(self, pct: float) -> MultiTimeframeEngine:
        self._inner.with_trailing_stop(pct)
        return self

    def with_commission(self, rate: float) -> MultiTimeframeEngine:
        self._inner.with_commission(rate)
        return self

    def with_commission_model(self, model) -> MultiTimeframeEngine:
        self._inner.with_commission_model(model)
        return self

    def with_slippage(self, bps: float) -> MultiTimeframeEngine:
        self._inner.with_slippage(bps)
        return self

    def with_initial_capital(self, capital: float) -> MultiTimeframeEngine:
        self._inner.with_initial_capital(capital)
        return self

    def with_fill_mode(self, mode: str) -> MultiTimeframeEngine:
        self._inner.with_fill_mode(mode)
        return self

    def with_leverage(
        self, margin_ratio: float, margin_call_pct: float = 0.5
    ) -> MultiTimeframeEngine:
        self._inner.with_leverage(margin_ratio, margin_call_pct)
        return self

    def with_loss_limits(
        self, daily: float = 0.0, total: float = 0.0
    ) -> MultiTimeframeEngine:
        self._inner.with_loss_limits(daily, total)
        return self

    def run(
        self, close_fine: ArrayLike, **htf_strategy_kwargs
    ) -> AdvancedBacktestResult:
        """Run multi-timeframe backtest.

        1. Resample close_fine (and stored OHLCV) to coarse bars
        2. Run htf_strategy on coarse close to get coarse signals
        3. Align coarse signals back to fine resolution (repeat each coarse signal `factor` times)
        4. Run BacktestEngine on fine bars with aligned signals

        Parameters
        ----------
        close_fine : array-like
            Fine-resolution close prices.
        **htf_strategy_kwargs
            Extra keyword arguments passed to the HTF strategy.

        Returns
        -------
        AdvancedBacktestResult
        """
        c_fine = np.asarray(close_fine, dtype=np.float64)
        n_fine = len(c_fine)
        factor = self._factor

        # ------------------------------------------------------------------
        # 1. Resample close to coarse resolution
        # ------------------------------------------------------------------
        # Build dummy OHLCV if OHLCV not provided
        if self._high is not None and self._low is not None and self._open is not None:
            coarse_o, coarse_h, coarse_l, coarse_c, _ = resample_ohlcv(
                self._open,
                self._high,
                self._low,
                c_fine,
                np.ones(n_fine),  # volume placeholder
                factor,
            )
        else:
            coarse_o, coarse_h, coarse_l, coarse_c, _ = resample_ohlcv(
                c_fine,
                c_fine,
                c_fine,
                c_fine,
                np.ones(n_fine),
                factor,
            )

        # ------------------------------------------------------------------
        # 2. Compute coarse-bar signals via htf_strategy
        # ------------------------------------------------------------------
        from ferro_ta.analysis.backtest import _resolve_strategy

        strategy_fn = _resolve_strategy(self._htf_strategy)
        # Ensure the coarse close array is C-contiguous (required by Rust kernels)
        coarse_c = np.ascontiguousarray(coarse_c, dtype=np.float64)
        coarse_signals = np.asarray(
            strategy_fn(coarse_c, **htf_strategy_kwargs), dtype=np.float64
        )

        # ------------------------------------------------------------------
        # 3. Align coarse signals back to fine resolution
        # ------------------------------------------------------------------
        aligned_signals = align_to_coarse(coarse_signals, factor, n_fine)

        # ------------------------------------------------------------------
        # 4. Set up OHLCV on inner engine if provided and run
        # ------------------------------------------------------------------
        if self._high is not None and self._low is not None and self._open is not None:
            self._inner.with_ohlcv(
                high=self._high,
                low=self._low,
                open_=self._open,
            )

        # Use a passthrough lambda so the already-computed aligned_signals are used
        return self._inner.run(
            c_fine,
            strategy=lambda c, **kw: aligned_signals,
        )
