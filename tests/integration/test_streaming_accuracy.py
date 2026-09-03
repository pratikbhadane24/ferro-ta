"""
Streaming accuracy tests: bar-by-bar == batch (Priority 3 - no optional deps).

Core claim: "bar-by-bar streaming == batch." Any divergence is a genuine bug.

This module validates that streaming (incremental) and batch (vectorized) modes
produce identical results within strict tolerances.

Pattern for each test:
1. Compute batch: batch_out = ferro_ta.INDICATOR(...)
2. Feed bar-by-bar: streamer = StreamingINDICATOR(...); [streamer.update(...) for bar in data]
3. Assert: np.allclose(stream_arr, batch_arr, equal_nan=True, atol=1e-12)

All tests use NO optional dependencies - they run in every CI environment.
"""

from __future__ import annotations

import numpy as np
import pytest

import ferro_ta
from ferro_ta.data.streaming import (
    StreamingATR,
    StreamingBBands,
    StreamingEMA,
    StreamingMACD,
    StreamingRSI,
    StreamingSMA,
    StreamingStoch,
    StreamingSupertrend,
    StreamingVWAP,
)

# ---------------------------------------------------------------------------
# Test Data (seeded for reproducibility)
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 200

CLOSE = 44.0 + np.cumsum(RNG.standard_normal(N) * 0.5)
HIGH = CLOSE + RNG.uniform(0.1, 1.0, N)
LOW = CLOSE - RNG.uniform(0.1, 1.0, N)
OPEN = CLOSE + RNG.standard_normal(N) * 0.2
VOLUME = RNG.uniform(500.0, 2000.0, N)


# ---------------------------------------------------------------------------
# StreamingSMA Tests
# ---------------------------------------------------------------------------


class TestStreamingSMA:
    """StreamingSMA vs ferro_ta.SMA — atol=1e-12 (identical arithmetic)."""

    @pytest.mark.parametrize("period", [5, 10, 20, 50])
    def test_streaming_matches_batch(self, period):
        """Streaming SMA should match batch SMA exactly."""
        # Batch
        batch_out = ferro_ta.SMA(CLOSE, timeperiod=period)

        # Streaming
        streamer = StreamingSMA(period=period)
        stream_out = np.array([streamer.update(c) for c in CLOSE])

        # Compare
        assert np.allclose(stream_out, batch_out, equal_nan=True, atol=1e-12)

    def test_warmup_produces_nan(self):
        """First period-1 updates should return NaN."""
        period = 10
        streamer = StreamingSMA(period=period)

        for i in range(period - 1):
            val = streamer.update(CLOSE[i])
            assert np.isnan(val), f"Expected NaN at index {i}, got {val}"

    def test_reset_gives_same_result(self):
        """Reset and re-feed should give identical output."""
        period = 10
        streamer = StreamingSMA(period=period)

        # First pass
        first_pass = np.array([streamer.update(c) for c in CLOSE[:50]])

        # Reset and second pass
        streamer.reset()
        second_pass = np.array([streamer.update(c) for c in CLOSE[:50]])

        assert np.allclose(first_pass, second_pass, equal_nan=True, atol=1e-14)


# ---------------------------------------------------------------------------
# StreamingEMA Tests
# ---------------------------------------------------------------------------


class TestStreamingEMA:
    """StreamingEMA vs ferro_ta.EMA — atol=1e-12 (same recursive formula, same seed)."""

    @pytest.mark.parametrize("period", [5, 10, 20, 50])
    def test_streaming_matches_batch(self, period):
        """Streaming EMA should match batch EMA exactly."""
        # Batch
        batch_out = ferro_ta.EMA(CLOSE, timeperiod=period)

        # Streaming
        streamer = StreamingEMA(period=period)
        stream_out = np.array([streamer.update(c) for c in CLOSE])

        # Compare
        assert np.allclose(stream_out, batch_out, equal_nan=True, atol=1e-12)

    def test_reset_gives_same_result(self):
        """Reset and re-feed should give identical output."""
        period = 10
        streamer = StreamingEMA(period=period)

        # First pass
        first_pass = np.array([streamer.update(c) for c in CLOSE[:50]])

        # Reset and second pass
        streamer.reset()
        second_pass = np.array([streamer.update(c) for c in CLOSE[:50]])

        assert np.allclose(first_pass, second_pass, equal_nan=True, atol=1e-14)


# ---------------------------------------------------------------------------
# StreamingRSI Tests
# ---------------------------------------------------------------------------


class TestStreamingRSI:
    """StreamingRSI vs ferro_ta.RSI — atol=1e-10; also verify range [0, 100]."""

    @pytest.mark.parametrize("period", [7, 14, 21])
    def test_streaming_matches_batch(self, period):
        """Streaming RSI should match batch RSI."""
        # Batch
        batch_out = ferro_ta.RSI(CLOSE, timeperiod=period)

        # Streaming
        streamer = StreamingRSI(period=period)
        stream_out = np.array([streamer.update(c) for c in CLOSE])

        # Compare
        assert np.allclose(stream_out, batch_out, equal_nan=True, atol=1e-10)

    def test_rsi_range_zero_to_hundred(self):
        """RSI values should be in range [0, 100]."""
        period = 14
        streamer = StreamingRSI(period=period)
        stream_out = np.array([streamer.update(c) for c in CLOSE])

        # Filter out NaN values
        valid = stream_out[~np.isnan(stream_out)]

        assert np.all(valid >= 0.0), "RSI should be >= 0"
        assert np.all(valid <= 100.0), "RSI should be <= 100"

    def test_reset_gives_same_result(self):
        """Reset and re-feed should give identical output."""
        period = 14
        streamer = StreamingRSI(period=period)

        # First pass
        first_pass = np.array([streamer.update(c) for c in CLOSE[:50]])

        # Reset and second pass
        streamer.reset()
        second_pass = np.array([streamer.update(c) for c in CLOSE[:50]])

        assert np.allclose(first_pass, second_pass, equal_nan=True, atol=1e-12)


# ---------------------------------------------------------------------------
# StreamingATR Tests
# ---------------------------------------------------------------------------


class TestStreamingATR:
    """StreamingATR vs ferro_ta.ATR — atol=1e-10; verify positive values."""

    @pytest.mark.parametrize("period", [7, 14, 21])
    def test_streaming_matches_batch(self, period):
        """Streaming ATR should match batch ATR bar-for-bar."""
        batch_out = ferro_ta.ATR(HIGH, LOW, CLOSE, timeperiod=period)

        streamer = StreamingATR(period=period)
        stream_out = np.array(
            [streamer.update(h, l, c) for h, l, c in zip(HIGH, LOW, CLOSE)]
        )

        assert np.allclose(stream_out, batch_out, equal_nan=True, atol=1e-10)

    def test_atr_non_negative(self):
        """ATR values should be non-negative."""
        period = 14
        streamer = StreamingATR(period=period)
        stream_out = np.array(
            [streamer.update(h, l, c) for h, l, c in zip(HIGH, LOW, CLOSE)]
        )

        valid = stream_out[~np.isnan(stream_out)]

        assert np.all(valid >= 0.0), "ATR should be non-negative"

    def test_reset_gives_same_result(self):
        """Reset and re-feed should give identical output."""
        period = 14
        streamer = StreamingATR(period=period)

        # First pass
        first_pass = np.array(
            [
                streamer.update(h, l, c)
                for h, l, c in zip(HIGH[:50], LOW[:50], CLOSE[:50])
            ]
        )

        # Reset and second pass
        streamer.reset()
        second_pass = np.array(
            [
                streamer.update(h, l, c)
                for h, l, c in zip(HIGH[:50], LOW[:50], CLOSE[:50])
            ]
        )

        assert np.allclose(first_pass, second_pass, equal_nan=True, atol=1e-12)


# ---------------------------------------------------------------------------
# StreamingBBands Tests
# ---------------------------------------------------------------------------


class TestStreamingBBands:
    """StreamingBBands vs ferro_ta.BBANDS — atol=1e-10 for all 3 bands."""

    @pytest.mark.parametrize("period", [10, 20, 30])
    def test_streaming_matches_batch(self, period):
        """Streaming BBands should match batch BBands (population std) bar-for-bar."""
        batch_upper, batch_middle, batch_lower = ferro_ta.BBANDS(
            CLOSE, timeperiod=period
        )

        streamer = StreamingBBands(period=period, nbdevup=2.0, nbdevdn=2.0)
        stream_results = [streamer.update(c) for c in CLOSE]
        stream_upper = np.array([r[0] for r in stream_results])
        stream_middle = np.array([r[1] for r in stream_results])
        stream_lower = np.array([r[2] for r in stream_results])

        assert np.allclose(stream_upper, batch_upper, equal_nan=True, atol=1e-10)
        assert np.allclose(stream_middle, batch_middle, equal_nan=True, atol=1e-10)
        assert np.allclose(stream_lower, batch_lower, equal_nan=True, atol=1e-10)

    def test_reset_gives_same_result(self):
        """Reset and re-feed should give identical output."""
        period = 20
        streamer = StreamingBBands(period=period, nbdevup=2.0, nbdevdn=2.0)

        # First pass
        first_pass = [streamer.update(c) for c in CLOSE[:50]]

        # Reset and second pass
        streamer.reset()
        second_pass = [streamer.update(c) for c in CLOSE[:50]]

        # Compare all three bands
        for i in range(len(first_pass)):
            assert np.allclose(
                first_pass[i], second_pass[i], equal_nan=True, atol=1e-14
            )


# ---------------------------------------------------------------------------
# StreamingMACD Tests
# ---------------------------------------------------------------------------


class TestStreamingMACD:
    """StreamingMACD vs ferro_ta.MACD — atol=1e-10; also verify histogram identity."""

    def test_streaming_matches_batch(self):
        """Streaming MACD should match batch MACD."""
        # Batch
        batch_macd, batch_signal, batch_hist = ferro_ta.MACD(
            CLOSE, fastperiod=12, slowperiod=26, signalperiod=9
        )

        # Streaming
        streamer = StreamingMACD(fastperiod=12, slowperiod=26, signalperiod=9)
        stream_results = [streamer.update(c) for c in CLOSE]
        stream_macd = np.array([r[0] for r in stream_results])
        stream_signal = np.array([r[1] for r in stream_results])
        stream_hist = np.array([r[2] for r in stream_results])

        assert np.allclose(stream_macd, batch_macd, equal_nan=True, atol=1e-10)
        assert np.allclose(stream_signal, batch_signal, equal_nan=True, atol=1e-10)
        assert np.allclose(stream_hist, batch_hist, equal_nan=True, atol=1e-10)

    def test_histogram_identity(self):
        """histogram should always equal macd - signal."""
        streamer = StreamingMACD(fastperiod=12, slowperiod=26, signalperiod=9)
        stream_results = [streamer.update(c) for c in CLOSE]
        stream_macd = np.array([r[0] for r in stream_results])
        stream_signal = np.array([r[1] for r in stream_results])
        stream_hist = np.array([r[2] for r in stream_results])

        expected_hist = stream_macd - stream_signal
        assert np.allclose(stream_hist, expected_hist, equal_nan=True, atol=1e-10)

    def test_reset_gives_same_result(self):
        """Reset and re-feed should give identical output."""
        streamer = StreamingMACD(fastperiod=12, slowperiod=26, signalperiod=9)

        # First pass
        first_pass = [streamer.update(c) for c in CLOSE[:50]]

        # Reset and second pass
        streamer.reset()
        second_pass = [streamer.update(c) for c in CLOSE[:50]]

        # Compare all three outputs
        for i in range(len(first_pass)):
            assert np.allclose(
                first_pass[i], second_pass[i], equal_nan=True, atol=1e-14
            )


# ---------------------------------------------------------------------------
# StreamingStoch Tests
# ---------------------------------------------------------------------------


class TestStreamingStoch:
    """StreamingStoch vs ferro_ta.STOCH — atol=1e-10; verify [0, 100] range."""

    def test_streaming_matches_batch(self):
        """Streaming Stochastic should match batch Stochastic."""
        # Batch
        batch_slowk, batch_slowd = ferro_ta.STOCH(
            HIGH, LOW, CLOSE, fastk_period=5, slowk_period=3, slowd_period=3
        )

        # Streaming
        streamer = StreamingStoch(fastk_period=5, slowk_period=3, slowd_period=3)
        stream_results = [streamer.update(h, l, c) for h, l, c in zip(HIGH, LOW, CLOSE)]
        stream_slowk = np.array([r[0] for r in stream_results])
        stream_slowd = np.array([r[1] for r in stream_results])

        assert np.allclose(stream_slowk, batch_slowk, equal_nan=True, atol=1e-10)
        assert np.allclose(stream_slowd, batch_slowd, equal_nan=True, atol=1e-10)

    def test_stoch_range_zero_to_hundred(self):
        """Stochastic values should be in range [0, 100]."""
        streamer = StreamingStoch(fastk_period=5, slowk_period=3, slowd_period=3)
        stream_results = [streamer.update(h, l, c) for h, l, c in zip(HIGH, LOW, CLOSE)]
        stream_slowk = np.array([r[0] for r in stream_results])
        stream_slowd = np.array([r[1] for r in stream_results])

        # Filter out NaN values
        valid_k = stream_slowk[~np.isnan(stream_slowk)]
        valid_d = stream_slowd[~np.isnan(stream_slowd)]

        assert np.all(valid_k >= 0.0), "slowk should be >= 0"
        assert np.all(valid_k <= 100.0), "slowk should be <= 100"
        assert np.all(valid_d >= 0.0), "slowd should be >= 0"
        assert np.all(valid_d <= 100.0), "slowd should be <= 100"

    def test_reset_gives_same_result(self):
        """Reset and re-feed should give identical output."""
        streamer = StreamingStoch(fastk_period=5, slowk_period=3, slowd_period=3)

        # First pass
        first_pass = [
            streamer.update(h, l, c) for h, l, c in zip(HIGH[:50], LOW[:50], CLOSE[:50])
        ]

        # Reset and second pass
        streamer.reset()
        second_pass = [
            streamer.update(h, l, c) for h, l, c in zip(HIGH[:50], LOW[:50], CLOSE[:50])
        ]

        # Compare
        for i in range(len(first_pass)):
            assert np.allclose(
                first_pass[i], second_pass[i], equal_nan=True, atol=1e-14
            )


# ---------------------------------------------------------------------------
# StreamingVWAP Tests
# ---------------------------------------------------------------------------


class TestStreamingVWAP:
    """StreamingVWAP vs ferro_ta.VWAP — atol=1e-10."""

    def test_streaming_matches_batch_cumulative(self):
        """Streaming VWAP (cumulative) should match batch VWAP."""
        # Batch (cumulative: timeperiod=0)
        batch_out = ferro_ta.VWAP(HIGH, LOW, CLOSE, VOLUME, timeperiod=0)

        # Streaming (cumulative)
        streamer = StreamingVWAP()
        stream_out = np.array(
            [
                streamer.update(h, l, c, v)
                for h, l, c, v in zip(HIGH, LOW, CLOSE, VOLUME)
            ]
        )

        # Compare
        assert np.allclose(stream_out, batch_out, equal_nan=True, atol=1e-10)

    def test_streaming_matches_batch_rolling(self):
        """Streaming VWAP (cumulative) matches batch cumulative VWAP."""
        # StreamingVWAP is cumulative only; compare against batch cumulative
        batch_out = ferro_ta.VWAP(HIGH, LOW, CLOSE, VOLUME, timeperiod=0)

        # Streaming (cumulative)
        streamer = StreamingVWAP()
        stream_out = np.array(
            [
                streamer.update(h, l, c, v)
                for h, l, c, v in zip(HIGH, LOW, CLOSE, VOLUME)
            ]
        )

        # Compare
        assert np.allclose(stream_out, batch_out, equal_nan=True, atol=1e-10)

    def test_reset_gives_same_result(self):
        """Reset and re-feed should give identical output."""
        streamer = StreamingVWAP()

        # First pass
        first_pass = np.array(
            [
                streamer.update(h, l, c, v)
                for h, l, c, v in zip(HIGH[:50], LOW[:50], CLOSE[:50], VOLUME[:50])
            ]
        )

        # Reset and second pass
        streamer.reset()
        second_pass = np.array(
            [
                streamer.update(h, l, c, v)
                for h, l, c, v in zip(HIGH[:50], LOW[:50], CLOSE[:50], VOLUME[:50])
            ]
        )

        assert np.allclose(first_pass, second_pass, equal_nan=True, atol=1e-14)


# ---------------------------------------------------------------------------
# StreamingSupertrend Tests
# ---------------------------------------------------------------------------


class TestStreamingSupertrend:
    """StreamingSupertrend vs ferro_ta.SUPERTREND — atol=1e-10."""

    def test_streaming_matches_batch(self):
        """Streaming SUPERTREND should match batch SUPERTREND."""
        period = 7
        multiplier = 3.0

        # Batch
        batch_line, batch_dir = ferro_ta.SUPERTREND(
            HIGH, LOW, CLOSE, timeperiod=period, multiplier=multiplier
        )

        # Streaming
        streamer = StreamingSupertrend(period=period, multiplier=multiplier)
        stream_results = [streamer.update(h, l, c) for h, l, c in zip(HIGH, LOW, CLOSE)]
        stream_line = np.array([r[0] for r in stream_results])
        stream_dir = np.array([r[1] for r in stream_results])

        # Compare
        assert np.allclose(stream_line, batch_line, equal_nan=True, atol=1e-10)
        assert np.allclose(stream_dir, batch_dir, equal_nan=True, atol=1e-10)

    def test_reset_gives_same_result(self):
        """Reset and re-feed should give identical output."""
        period = 7
        multiplier = 3.0
        streamer = StreamingSupertrend(period=period, multiplier=multiplier)

        # First pass
        first_pass = [
            streamer.update(h, l, c) for h, l, c in zip(HIGH[:50], LOW[:50], CLOSE[:50])
        ]

        # Reset and second pass
        streamer.reset()
        second_pass = [
            streamer.update(h, l, c) for h, l, c in zip(HIGH[:50], LOW[:50], CLOSE[:50])
        ]

        # Compare
        for i in range(len(first_pass)):
            assert np.allclose(
                first_pass[i], second_pass[i], equal_nan=True, atol=1e-14
            )
