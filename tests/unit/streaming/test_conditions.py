"""Tests for streaming conditions: dead-band cross, streak, cooldown."""

import math

import numpy as np
import pytest

from ferro_ta.data.streaming import StreamingCooldown, StreamingCross, StreamingStreak


def _cross_events(band: float, diffs: list[float]) -> list[int]:
    cross = StreamingCross(dead_band=band)
    return [cross.update(d, 0.0) for d in diffs]


class TestStreamingCross:
    def test_fires_on_clean_cross(self):
        cross = StreamingCross()
        assert [cross.update(a, 2.0) for a in (1.0, 1.0, 3.0)] == [0, 0, 1]

    def test_oscillation_inside_band_fires_once(self):
        diffs = [-1.0, 0.2, -0.2, 0.3, 0.6, 0.1, 0.7]
        assert _cross_events(0.5, diffs) == [0, 0, 0, 0, 1, 0, 0]

    def test_cross_down_mirrors_up(self):
        diffs = [1.0, -0.2, 0.2, -0.3, -0.6, -0.1, -0.7]
        assert _cross_events(0.5, diffs) == [0, 0, 0, 0, -1, 0, 0]

    def test_preview_does_not_arm_or_fire(self):
        cross = StreamingCross()
        cross.update(-1.0, 0.0)
        assert cross.update(1.0, 0.0, closed=False) == 1
        assert cross.update(1.0, 0.0, closed=False) == 1, "preview never commits"
        assert cross.update(1.0, 0.0) == 1
        assert cross.update(2.0, 0.0) == 0

    def test_nan_tick_skipped(self):
        cross = StreamingCross()
        cross.update(-1.0, 0.0)
        assert cross.update(math.nan, 0.0) == 0
        assert cross.update(1.0, 0.0) == 1

    @pytest.mark.parametrize("band", [-1.0, math.nan, math.inf])
    def test_invalid_band_rejected(self, band):
        with pytest.raises(ValueError):
            StreamingCross(dead_band=band)

    def test_repr(self):
        assert repr(StreamingCross(0.5)) == "StreamingCross(dead_band=0.5)"


class TestStreamingStreak:
    def test_fires_from_nth_true(self):
        streak = StreamingStreak(3)
        inputs = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        assert [streak.update(v) for v in inputs] == [0, 0, 0, 0, 0, 1, 1]
        assert streak.n == 3

    def test_zero_n_rejected(self):
        with pytest.raises(ValueError):
            StreamingStreak(0)


class TestStreamingCooldown:
    def test_blocks_next_bars(self):
        cooldown = StreamingCooldown(2)
        inputs = [1.0, 1.0, 1.0, 0.0, -1.0]
        assert [cooldown.update(v) for v in inputs] == [1, 0, 0, 0, -1]
        assert cooldown.bars == 2

    def test_compose_cross_then_cooldown(self):
        rng = np.random.default_rng(7)
        i = np.arange(500)
        price = 100 + 3 * np.sin(i / 4) + 0.3 * rng.standard_normal(500)

        raw_cross = StreamingCross(dead_band=0.2)
        raw = [k for k, p in enumerate(price) if raw_cross.update(p, 100.0)]

        cross, cooldown = StreamingCross(dead_band=0.2), StreamingCooldown(15)
        fired = [
            k for k, p in enumerate(price) if cooldown.update(cross.update(p, 100.0))
        ]
        assert 2 <= len(fired) < len(raw), "cooldown must suppress some crosses"
        assert all(b - a > 15 for a, b in zip(fired, fired[1:], strict=False))
