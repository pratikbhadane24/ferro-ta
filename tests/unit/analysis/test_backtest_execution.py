"""Execution realism in the OHLCV engine: volume caps and ATR slippage."""

from __future__ import annotations

import numpy as np
import pytest

from ferro_ta.analysis.backtest import BacktestEngine
from ferro_ta.core.exceptions import FerroTAValueError

N = 12


@pytest.fixture
def flat() -> dict[str, np.ndarray]:
    close = np.full(N, 100.0)
    return {"close": close, "high": close + 1, "low": close - 1, "open_": close}


def _always_long(c):
    return np.ones(len(c))


def _engine(flat) -> BacktestEngine:
    return BacktestEngine().with_ohlcv(
        high=flat["high"], low=flat["low"], open_=flat["open_"]
    )


class TestVolumeParticipation:
    def test_position_ramps_in_partial_fills(self, flat):
        # cap = 0.1 * 2500 * 100 / 100_000 = 0.25 weight per bar
        res = (
            _engine(flat)
            .with_volume_participation(np.full(N, 2_500.0), 0.1)
            .run(flat["close"], _always_long)
        )
        np.testing.assert_allclose(res.positions[:6], [0, 0.25, 0.5, 0.75, 1, 1])

    def test_ample_volume_is_unconstrained(self, flat):
        base = _engine(flat).run(flat["close"], _always_long)
        capped = (
            _engine(flat)
            .with_volume_participation(np.full(N, 1e12), 0.01)
            .run(flat["close"], _always_long)
        )
        np.testing.assert_array_equal(base.positions, capped.positions)
        np.testing.assert_array_equal(base.strategy_returns, capped.strategy_returns)

    def test_rejects_bad_participation(self, flat):
        with pytest.raises(FerroTAValueError):
            BacktestEngine().with_volume_participation(np.ones(N), 0.0)
        with pytest.raises(FerroTAValueError):
            BacktestEngine().with_volume_participation(np.ones(N), float("nan"))

    def test_volume_length_mismatch(self, flat):
        engine = _engine(flat).with_volume_participation(np.ones(3), 0.1)
        with pytest.raises(ValueError, match="volume length"):
            engine.run(flat["close"], _always_long)


class TestAtrSlippage:
    def test_entry_fill_pays_atr_slippage(self, flat):
        signals = np.zeros(N)
        signals[8] = 1.0
        res = (
            _engine(flat)
            .with_atr_slippage(0.5, period=3)
            .run(flat["close"], lambda c: signals)
        )
        # ATR of a constant 2.0 range is 2.0 -> 0.5 * 2 / 100 = 1% slippage
        assert res.fill_prices[9] == pytest.approx(101.0)

    def test_validation(self):
        with pytest.raises(FerroTAValueError):
            BacktestEngine().with_atr_slippage(-1.0)
        with pytest.raises(FerroTAValueError):
            BacktestEngine().with_atr_slippage(1.0, period=0)


def test_execution_options_require_ohlcv(flat):
    with pytest.raises(FerroTAValueError, match="require OHLCV"):
        BacktestEngine().with_atr_slippage(1.0).run(flat["close"], _always_long)
    with pytest.raises(FerroTAValueError, match="require OHLCV"):
        BacktestEngine().with_volume_participation(np.ones(N), 0.1).run(
            flat["close"], _always_long
        )
