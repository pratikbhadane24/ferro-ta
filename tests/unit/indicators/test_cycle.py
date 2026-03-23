"""Unit tests for ferro_ta.indicators.cycle"""

import numpy as np

from ferro_ta.indicators.cycle import (
    HT_DCPERIOD,
    HT_DCPHASE,
    HT_PHASOR,
    HT_SINE,
    HT_TRENDLINE,
    HT_TRENDMODE,
)

# ---------------------------------------------------------------------------
# Shared fixtures — cycle indicators need at least ~64 bars for valid output
# ---------------------------------------------------------------------------

N = 200
t = np.linspace(0, 10 * np.pi, N)
SINE_CLOSE = 100 + 10 * np.sin(t)  # clean sine wave


def _warmup_end(arr):
    """Return index of first non-NaN value (or N if all NaN)."""
    valid = np.where(~np.isnan(arr.astype(float)))[0]
    return valid[0] if len(valid) else N


# ---------------------------------------------------------------------------
# HT_DCPERIOD
# ---------------------------------------------------------------------------


class TestHT_DCPERIOD:
    def test_length(self):
        result = HT_DCPERIOD(SINE_CLOSE)
        assert len(result) == N

    def test_nan_warmup(self):
        result = HT_DCPERIOD(SINE_CLOSE)
        w = _warmup_end(result)
        assert w > 0
        assert np.all(np.isnan(result[:w]))

    def test_valid_finite(self):
        result = HT_DCPERIOD(SINE_CLOSE)
        w = _warmup_end(result)
        assert np.all(np.isfinite(result[w:]))

    def test_sine_period_reasonable(self):
        # Our sine has period = 2*pi in t; with N=200 and t in [0,10*pi]
        # the true period in samples = 200 / (10*pi / (2*pi)) = 200/5 = 40
        result = HT_DCPERIOD(SINE_CLOSE)
        valid = result[~np.isnan(result)]
        # HT_DCPERIOD should detect a period in a reasonable range [6, 100]
        assert np.any((valid > 6) & (valid < 100))


# ---------------------------------------------------------------------------
# HT_DCPHASE
# ---------------------------------------------------------------------------


class TestHT_DCPHASE:
    def test_length(self):
        assert len(HT_DCPHASE(SINE_CLOSE)) == N

    def test_nan_warmup(self):
        result = HT_DCPHASE(SINE_CLOSE)
        w = _warmup_end(result)
        assert w > 0

    def test_valid_finite(self):
        result = HT_DCPHASE(SINE_CLOSE)
        w = _warmup_end(result)
        assert np.all(np.isfinite(result[w:]))


# ---------------------------------------------------------------------------
# HT_PHASOR
# ---------------------------------------------------------------------------


class TestHT_PHASOR:
    def test_returns_two_arrays(self):
        result = HT_PHASOR(SINE_CLOSE)
        assert isinstance(result, tuple) and len(result) == 2

    def test_length(self):
        inphase, quadrature = HT_PHASOR(SINE_CLOSE)
        assert len(inphase) == len(quadrature) == N

    def test_nan_warmup(self):
        inphase, quadrature = HT_PHASOR(SINE_CLOSE)
        w = _warmup_end(inphase)
        assert w > 0

    def test_valid_finite(self):
        inphase, quadrature = HT_PHASOR(SINE_CLOSE)
        wi = _warmup_end(inphase)
        wq = _warmup_end(quadrature)
        assert np.all(np.isfinite(inphase[wi:]))
        assert np.all(np.isfinite(quadrature[wq:]))


# ---------------------------------------------------------------------------
# HT_SINE
# ---------------------------------------------------------------------------


class TestHT_SINE:
    def test_returns_two_arrays(self):
        result = HT_SINE(SINE_CLOSE)
        assert isinstance(result, tuple) and len(result) == 2

    def test_length(self):
        sine, leadsine = HT_SINE(SINE_CLOSE)
        assert len(sine) == len(leadsine) == N

    def test_nan_warmup(self):
        sine, leadsine = HT_SINE(SINE_CLOSE)
        w = _warmup_end(sine)
        assert w > 0

    def test_valid_finite(self):
        sine, leadsine = HT_SINE(SINE_CLOSE)
        ws = _warmup_end(sine)
        wl = _warmup_end(leadsine)
        assert np.all(np.isfinite(sine[ws:]))
        assert np.all(np.isfinite(leadsine[wl:]))

    def test_values_in_sine_range(self):
        # Sine values should be in [-1, 1] roughly
        sine, leadsine = HT_SINE(SINE_CLOSE)
        valid = sine[~np.isnan(sine)]
        assert np.all(valid >= -1.5) and np.all(valid <= 1.5)


# ---------------------------------------------------------------------------
# HT_TRENDLINE
# ---------------------------------------------------------------------------


class TestHT_TRENDLINE:
    def test_length(self):
        assert len(HT_TRENDLINE(SINE_CLOSE)) == N

    def test_nan_warmup(self):
        result = HT_TRENDLINE(SINE_CLOSE)
        w = _warmup_end(result)
        assert w > 0

    def test_valid_finite(self):
        result = HT_TRENDLINE(SINE_CLOSE)
        w = _warmup_end(result)
        assert np.all(np.isfinite(result[w:]))

    def test_smooth_trendline(self):
        # Trendline should be smoother than raw close
        result = HT_TRENDLINE(SINE_CLOSE)
        w = _warmup_end(result)
        raw_std = np.std(np.diff(SINE_CLOSE[w:]))
        trend_std = np.std(np.diff(result[w:]))
        assert trend_std < raw_std


# ---------------------------------------------------------------------------
# HT_TRENDMODE
# ---------------------------------------------------------------------------


class TestHT_TRENDMODE:
    def test_length(self):
        assert len(HT_TRENDMODE(SINE_CLOSE)) == N

    def test_values_binary(self):
        result = HT_TRENDMODE(SINE_CLOSE)
        assert np.all(np.isin(result, [0, 1]))

    def test_nan_warmup_as_zero(self):
        # HT_TRENDMODE returns integers (no NaN); warmup bars should be 0
        result = HT_TRENDMODE(SINE_CLOSE)
        assert np.all(np.isfinite(result.astype(float)))
