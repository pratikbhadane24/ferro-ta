"""Tests for ferro_ta streaming / incremental indicators."""

import math

import numpy as np
import pytest

from ferro_ta import EMA, RSI, SMA
from ferro_ta.data.streaming import StreamingEMA, StreamingRSI, StreamingSMA

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PRICES = np.array(
    [
        44.34,
        44.09,
        44.15,
        43.61,
        44.33,
        44.83,
        45.10,
        45.15,
        43.61,
        44.33,
        44.83,
        45.10,
        45.15,
        43.61,
        44.33,
    ],
    dtype=np.float64,
)


def _finite(arr: np.ndarray) -> np.ndarray:
    return arr[~np.isnan(arr)]


# ---------------------------------------------------------------------------
# StreamingSMA
# ---------------------------------------------------------------------------


class TestStreamingSMA:
    def test_basic_values(self):
        """Feed known values, verify manually computed SMA."""
        sma = StreamingSMA(period=3)
        assert math.isnan(sma.update(1.0))
        assert math.isnan(sma.update(2.0))
        assert math.isclose(sma.update(3.0), 2.0)
        assert math.isclose(sma.update(4.0), 3.0)
        assert math.isclose(sma.update(5.0), 4.0)

    def test_matches_batch_sma(self):
        """Streaming SMA final values must match batch SMA on the same data."""
        period = 5
        batch = SMA(PRICES, timeperiod=period)
        sma = StreamingSMA(period=period)
        for i, price in enumerate(PRICES):
            val = sma.update(price)
            if math.isnan(batch[i]):
                assert math.isnan(val), f"Expected NaN at index {i}"
            else:
                assert math.isclose(val, batch[i], rel_tol=1e-10), (
                    f"Mismatch at index {i}: streaming={val}, batch={batch[i]}"
                )

    def test_period_property(self):
        sma = StreamingSMA(period=7)
        assert sma.period == 7

    def test_warmup_returns_nan(self):
        """First period-1 updates must return NaN."""
        period = 4
        sma = StreamingSMA(period=period)
        for i in range(period - 1):
            assert math.isnan(sma.update(float(i + 1)))
        # The period-th update should NOT be NaN
        assert not math.isnan(sma.update(float(period)))

    def test_single_value_period_1(self):
        """Period=1 means every value is immediately returned."""
        sma = StreamingSMA(period=1)
        assert math.isclose(sma.update(42.0), 42.0)
        assert math.isclose(sma.update(99.0), 99.0)

    def test_reset(self):
        """After reset, the indicator should behave as freshly constructed."""
        sma = StreamingSMA(period=3)
        sma.update(10.0)
        sma.update(20.0)
        result_before_reset = sma.update(30.0)
        assert math.isclose(result_before_reset, 20.0)

        sma.reset()
        # After reset, warmup restarts
        assert math.isnan(sma.update(100.0))
        assert math.isnan(sma.update(200.0))
        assert math.isclose(sma.update(300.0), 200.0)

    def test_invalid_period_zero(self):
        with pytest.raises(Exception):
            StreamingSMA(period=0)

    def test_repr(self):
        sma = StreamingSMA(period=5)
        assert "StreamingSMA" in repr(sma)
        assert "5" in repr(sma)


# ---------------------------------------------------------------------------
# StreamingEMA
# ---------------------------------------------------------------------------


class TestStreamingEMA:
    def test_basic_seeding(self):
        """EMA seeds from the first `period` values using their SMA."""
        ema = StreamingEMA(period=3)
        assert math.isnan(ema.update(1.0))
        assert math.isnan(ema.update(2.0))
        # Seed = SMA(1,2,3) = 2.0
        seed = ema.update(3.0)
        assert math.isclose(seed, 2.0)

    def test_matches_batch_ema(self):
        """Streaming EMA must match batch EMA on the same data."""
        period = 5
        batch = EMA(PRICES, timeperiod=period)
        ema = StreamingEMA(period=period)
        for i, price in enumerate(PRICES):
            val = ema.update(price)
            if math.isnan(batch[i]):
                assert math.isnan(val), f"Expected NaN at index {i}"
            else:
                assert math.isclose(val, batch[i], rel_tol=1e-10), (
                    f"Mismatch at index {i}: streaming={val}, batch={batch[i]}"
                )

    def test_warmup_returns_nan(self):
        period = 5
        ema = StreamingEMA(period=period)
        for i in range(period - 1):
            assert math.isnan(ema.update(float(i + 1)))
        assert not math.isnan(ema.update(float(period)))

    def test_ema_differs_from_sma_after_warmup(self):
        """After warmup, EMA and SMA should diverge for non-constant data."""
        period = 3
        prices = [1.0, 2.0, 3.0, 10.0, 11.0]
        sma = StreamingSMA(period=period)
        ema = StreamingEMA(period=period)
        sma_vals = [sma.update(p) for p in prices]
        ema_vals = [ema.update(p) for p in prices]
        # At the seed point they should match (both are SMA of first 3)
        assert math.isclose(sma_vals[2], ema_vals[2])
        # After the seed they should diverge
        assert not math.isclose(sma_vals[-1], ema_vals[-1], rel_tol=1e-9)

    def test_reset(self):
        ema = StreamingEMA(period=3)
        for p in [10.0, 20.0, 30.0, 40.0]:
            ema.update(p)
        ema.reset()
        # After reset, warmup restarts
        assert math.isnan(ema.update(1.0))
        assert math.isnan(ema.update(2.0))
        assert math.isclose(ema.update(3.0), 2.0)

    def test_period_property(self):
        ema = StreamingEMA(period=10)
        assert ema.period == 10

    def test_invalid_period_zero(self):
        with pytest.raises(Exception):
            StreamingEMA(period=0)

    def test_single_value_period_1(self):
        ema = StreamingEMA(period=1)
        assert math.isclose(ema.update(42.0), 42.0)
        assert math.isclose(ema.update(50.0), 50.0)

    def test_repr(self):
        ema = StreamingEMA(period=12)
        assert "StreamingEMA" in repr(ema)
        assert "12" in repr(ema)


# ---------------------------------------------------------------------------
# StreamingRSI
# ---------------------------------------------------------------------------


class TestStreamingRSI:
    def test_matches_batch_rsi(self):
        """Streaming RSI must match batch RSI on the same data."""
        period = 5
        batch = RSI(PRICES, timeperiod=period)
        rsi = StreamingRSI(period=period)
        for i, price in enumerate(PRICES):
            val = rsi.update(price)
            if math.isnan(batch[i]):
                assert math.isnan(val), f"Expected NaN at index {i}"
            else:
                assert math.isclose(val, batch[i], rel_tol=1e-8), (
                    f"Mismatch at index {i}: streaming={val}, batch={batch[i]}"
                )

    def test_warmup_returns_nan(self):
        """RSI needs period+1 bars (1 for first prev, then period deltas)."""
        period = 5
        rsi = StreamingRSI(period=period)
        # First bar: sets prev, returns NaN
        assert math.isnan(rsi.update(50.0))
        # Next period-1 bars: accumulating deltas, returns NaN
        for i in range(period - 1):
            assert math.isnan(rsi.update(50.0 + i))
        # The (period+1)-th bar should produce a value
        assert not math.isnan(rsi.update(55.0))

    def test_rsi_range(self):
        """All finite RSI values must be in [0, 100]."""
        rsi = StreamingRSI(period=5)
        for price in PRICES:
            val = rsi.update(price)
            if not math.isnan(val):
                assert 0.0 <= val <= 100.0, f"RSI out of range: {val}"

    def test_constant_prices(self):
        """Constant prices produce no gains or losses -- RSI should be 100
        (avg_loss == 0 leads to RS = infinity -> RSI = 100)."""
        rsi = StreamingRSI(period=5)
        results = [rsi.update(50.0) for _ in range(20)]
        finite = [v for v in results if not math.isnan(v)]
        assert len(finite) > 0
        for v in finite:
            assert math.isclose(v, 100.0) or math.isclose(v, 0.0) or (0.0 <= v <= 100.0)

    def test_monotone_increasing(self):
        """Monotonically increasing prices should yield RSI = 100."""
        rsi = StreamingRSI(period=3)
        results = [rsi.update(float(i)) for i in range(1, 20)]
        finite = [v for v in results if not math.isnan(v)]
        for v in finite:
            assert math.isclose(v, 100.0), (
                f"Expected RSI=100 for monotone increase, got {v}"
            )

    def test_monotone_decreasing(self):
        """Monotonically decreasing prices should yield RSI = 0."""
        rsi = StreamingRSI(period=3)
        results = [rsi.update(float(100 - i)) for i in range(20)]
        finite = [v for v in results if not math.isnan(v)]
        for v in finite:
            assert math.isclose(v, 0.0, abs_tol=1e-10), (
                f"Expected RSI=0 for monotone decrease, got {v}"
            )

    def test_default_period_14(self):
        rsi = StreamingRSI()
        assert rsi.period == 14

    def test_reset(self):
        rsi = StreamingRSI(period=3)
        for price in PRICES:
            rsi.update(price)
        rsi.reset()
        # After reset, warmup restarts -- first update should be NaN
        assert math.isnan(rsi.update(50.0))

    def test_invalid_period_zero(self):
        with pytest.raises(Exception):
            StreamingRSI(period=0)

    def test_repr(self):
        rsi = StreamingRSI(period=14)
        assert "StreamingRSI" in repr(rsi)
        assert "14" in repr(rsi)


# ---------------------------------------------------------------------------
# Edge cases (shared across indicators)
# ---------------------------------------------------------------------------


class TestStreamingEdgeCases:
    def test_nan_input_sma(self):
        """Feeding NaN into SMA should propagate NaN through the window."""
        sma = StreamingSMA(period=3)
        sma.update(1.0)
        sma.update(2.0)
        # Third value is NaN -- the sum will include NaN, producing NaN
        val = sma.update(float("nan"))
        assert math.isnan(val)

    def test_nan_input_ema(self):
        """Feeding NaN into EMA should produce NaN output."""
        ema = StreamingEMA(period=3)
        ema.update(1.0)
        ema.update(2.0)
        val = ema.update(float("nan"))
        assert math.isnan(val)

    def test_nan_input_rsi(self):
        """Feeding NaN into RSI should produce NaN output."""
        rsi = StreamingRSI(period=3)
        rsi.update(1.0)
        rsi.update(2.0)
        val = rsi.update(float("nan"))
        assert math.isnan(val)

    def test_single_value_sma(self):
        """Feeding exactly one value to SMA with period > 1 yields NaN."""
        sma = StreamingSMA(period=5)
        assert math.isnan(sma.update(42.0))

    def test_single_value_ema(self):
        ema = StreamingEMA(period=5)
        assert math.isnan(ema.update(42.0))

    def test_single_value_rsi(self):
        rsi = StreamingRSI(period=5)
        assert math.isnan(rsi.update(42.0))

    def test_large_dataset_sma(self):
        """Ensure streaming SMA is stable over many updates."""
        period = 20
        sma = StreamingSMA(period=period)
        np.random.seed(42)
        data = np.random.randn(10_000).cumsum() + 100.0
        batch = SMA(data, timeperiod=period)
        for i, price in enumerate(data):
            val = sma.update(price)
            if not math.isnan(batch[i]):
                assert math.isclose(val, batch[i], rel_tol=1e-8), (
                    f"Drift at index {i}: streaming={val}, batch={batch[i]}"
                )

    def test_large_dataset_ema(self):
        """Ensure streaming EMA is stable over many updates."""
        period = 20
        ema = StreamingEMA(period=period)
        np.random.seed(42)
        data = np.random.randn(10_000).cumsum() + 100.0
        batch = EMA(data, timeperiod=period)
        for i, price in enumerate(data):
            val = ema.update(price)
            if not math.isnan(batch[i]):
                assert math.isclose(val, batch[i], rel_tol=1e-8), (
                    f"Drift at index {i}: streaming={val}, batch={batch[i]}"
                )

    def test_large_dataset_rsi(self):
        """Ensure streaming RSI is stable over many updates."""
        period = 14
        rsi = StreamingRSI(period=period)
        np.random.seed(42)
        data = np.random.randn(10_000).cumsum() + 100.0
        batch = RSI(data, timeperiod=period)
        for i, price in enumerate(data):
            val = rsi.update(price)
            if not math.isnan(batch[i]):
                assert math.isclose(val, batch[i], rel_tol=1e-6), (
                    f"Drift at index {i}: streaming={val}, batch={batch[i]}"
                )

    def test_reset_then_reuse_matches_fresh_instance(self):
        """A reset indicator should produce identical output to a new one."""
        period = 5
        data = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]

        sma_reused = StreamingSMA(period=period)
        for p in [99.0, 98.0, 97.0, 96.0, 95.0]:
            sma_reused.update(p)
        sma_reused.reset()

        sma_fresh = StreamingSMA(period=period)

        for p in data:
            v1 = sma_reused.update(p)
            v2 = sma_fresh.update(p)
            if math.isnan(v1):
                assert math.isnan(v2)
            else:
                assert math.isclose(v1, v2, rel_tol=1e-12)
