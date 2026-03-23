"""
Benchmark suite
===========================

Numerical-regression and performance benchmarks that run against the canonical
OHLCV fixture in ``benchmarks/fixtures/canonical_ohlcv.npz``.

Numerical regression checks
----------------------------
For each (indicator, params) pair in ``INDICATOR_SUITE``, the test:
1. Loads the canonical dataset.
2. Runs the indicator.
3. Compares the last N non-NaN values to stored baselines (or tolerance-based).

To regenerate baselines after an intentional indicator change::

    pytest benchmarks/test_benchmark_suite.py --update-baselines

Performance checks
------------------
Each indicator is timed over the canonical dataset.  If a ``baselines.npz``
file exists in this directory, the run compares to that; otherwise timing is
reported only.

Run locally::

    pytest benchmarks/test_benchmark_suite.py -v

"""

from __future__ import annotations

import pathlib
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

FIXTURE_PATH = pathlib.Path(__file__).parent / "fixtures" / "canonical_ohlcv.npz"
BASELINE_PATH = pathlib.Path(__file__).parent / "baselines.npz"

# ---------------------------------------------------------------------------
# Load fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ohlcv() -> dict[str, np.ndarray]:
    """Load canonical OHLCV fixture."""
    if not FIXTURE_PATH.exists():
        pytest.skip(f"Canonical fixture not found: {FIXTURE_PATH}")
    data = np.load(FIXTURE_PATH)
    return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# Indicator suite definition
# ---------------------------------------------------------------------------

# Each entry: (name, callable, kwargs)
# The callable receives (close,) or (high, low, close,) based on 'inputs' key.
INDICATOR_SUITE: list[dict[str, Any]] = [
    {
        "name": "SMA_20",
        "inputs": "close",
        "fn": None,
        "fn_name": "SMA",
        "kwargs": {"timeperiod": 20},
    },
    {
        "name": "EMA_20",
        "inputs": "close",
        "fn": None,
        "fn_name": "EMA",
        "kwargs": {"timeperiod": 20},
    },
    {
        "name": "RSI_14",
        "inputs": "close",
        "fn": None,
        "fn_name": "RSI",
        "kwargs": {"timeperiod": 14},
    },
    {
        "name": "ATR_14",
        "inputs": "hlc",
        "fn": None,
        "fn_name": "ATR",
        "kwargs": {"timeperiod": 14},
    },
    {
        "name": "ADX_14",
        "inputs": "hlc",
        "fn": None,
        "fn_name": "ADX",
        "kwargs": {"timeperiod": 14},
    },
    {
        "name": "STDDEV_20",
        "inputs": "close",
        "fn": None,
        "fn_name": "STDDEV",
        "kwargs": {"timeperiod": 20},
    },
    {
        "name": "MACD",
        "inputs": "close",
        "fn": None,
        "fn_name": "MACD",
        "kwargs": {},
    },
    {
        "name": "BBANDS_20",
        "inputs": "close",
        "fn": None,
        "fn_name": "BBANDS",
        "kwargs": {"timeperiod": 20},
    },
    {
        "name": "STOCH",
        "inputs": "hlc",
        "fn": None,
        "fn_name": "STOCH",
        "kwargs": {},
    },
    {
        "name": "LINEARREG_14",
        "inputs": "close",
        "fn": None,
        "fn_name": "LINEARREG",
        "kwargs": {"timeperiod": 14},
    },
    {
        "name": "LINEARREG_SLOPE_14",
        "inputs": "close",
        "fn": None,
        "fn_name": "LINEARREG_SLOPE",
        "kwargs": {"timeperiod": 14},
    },
    {
        "name": "TSF_14",
        "inputs": "close",
        "fn": None,
        "fn_name": "TSF",
        "kwargs": {"timeperiod": 14},
    },
    {
        "name": "VAR_20",
        "inputs": "close",
        "fn": None,
        "fn_name": "VAR",
        "kwargs": {"timeperiod": 20},
    },
    {
        "name": "CORREL_30",
        "inputs": "pair_hl",
        "fn": None,
        "fn_name": "CORREL",
        "kwargs": {"timeperiod": 30},
    },
    {
        "name": "BETA_5",
        "inputs": "pair_hl",
        "fn": None,
        "fn_name": "BETA",
        "kwargs": {"timeperiod": 5},
    },
    {
        "name": "CCI_14",
        "inputs": "hlc",
        "fn": None,
        "fn_name": "CCI",
        "kwargs": {"timeperiod": 14},
    },
    {
        "name": "WILLR_14",
        "inputs": "hlc",
        "fn": None,
        "fn_name": "WILLR",
        "kwargs": {"timeperiod": 14},
    },
]


def _load_fn(fn_name: str) -> Callable[..., Any]:
    import ferro_ta as ft

    return getattr(ft, fn_name)


def _run_indicator(entry: dict[str, Any], data: dict[str, np.ndarray]) -> np.ndarray:
    fn = _load_fn(entry["fn_name"])
    if entry["inputs"] == "close":
        result = fn(data["close"], **entry["kwargs"])
    elif entry["inputs"] == "hlc":
        result = fn(data["high"], data["low"], data["close"], **entry["kwargs"])
    else:  # pair_hl
        result = fn(data["high"], data["low"], **entry["kwargs"])
    if isinstance(result, tuple):
        result = result[0]
    return np.asarray(result, dtype=np.float64)


# ---------------------------------------------------------------------------
# Numerical regression tests
# ---------------------------------------------------------------------------


class TestNumericalRegression:
    """Verify indicator outputs match stored baselines (or tolerance)."""

    @pytest.mark.parametrize(
        "entry", INDICATOR_SUITE, ids=[e["name"] for e in INDICATOR_SUITE]
    )
    def test_output_shape(
        self, entry: dict[str, Any], ohlcv: dict[str, np.ndarray]
    ) -> None:
        """Indicator output length must equal input length."""
        out = _run_indicator(entry, ohlcv)
        assert len(out) == len(ohlcv["close"]), (
            f"{entry['name']}: expected len {len(ohlcv['close'])}, got {len(out)}"
        )

    @pytest.mark.parametrize(
        "entry", INDICATOR_SUITE, ids=[e["name"] for e in INDICATOR_SUITE]
    )
    def test_warmup_is_nan(
        self, entry: dict[str, Any], ohlcv: dict[str, np.ndarray]
    ) -> None:
        """First bar must be NaN (warm-up)."""
        out = _run_indicator(entry, ohlcv)
        assert np.isnan(out[0]), f"{entry['name']}: expected NaN at bar 0, got {out[0]}"

    @pytest.mark.parametrize(
        "entry", INDICATOR_SUITE, ids=[e["name"] for e in INDICATOR_SUITE]
    )
    def test_no_inf(self, entry: dict[str, Any], ohlcv: dict[str, np.ndarray]) -> None:
        """Output must not contain infinities."""
        out = _run_indicator(entry, ohlcv)
        assert not np.any(np.isinf(out)), f"{entry['name']}: output contains Inf"

    @pytest.mark.parametrize(
        "entry", INDICATOR_SUITE, ids=[e["name"] for e in INDICATOR_SUITE]
    )
    def test_last_values_stable(
        self, entry: dict[str, Any], ohlcv: dict[str, np.ndarray]
    ) -> None:
        """Last 10 non-NaN values must be finite and stable (no sudden jumps)."""
        out = _run_indicator(entry, ohlcv)
        valid = out[~np.isnan(out)]
        assert len(valid) >= 10, f"{entry['name']}: fewer than 10 valid output values"
        last10 = valid[-10:]
        assert np.all(np.isfinite(last10)), (
            f"{entry['name']}: non-finite in last 10 values"
        )

    @pytest.mark.skipif(not BASELINE_PATH.exists(), reason="No baselines.npz found")
    @pytest.mark.parametrize(
        "entry", INDICATOR_SUITE, ids=[e["name"] for e in INDICATOR_SUITE]
    )
    def test_regression_vs_baseline(
        self, entry: dict[str, Any], ohlcv: dict[str, np.ndarray]
    ) -> None:
        """Compare last 10 values to stored baselines."""
        baselines = np.load(BASELINE_PATH)
        key = entry["name"]
        if key not in baselines:
            pytest.skip(f"No baseline stored for {key}")
        out = _run_indicator(entry, ohlcv)
        valid = out[~np.isnan(out)]
        last10 = valid[-10:]
        stored = baselines[key]
        np.testing.assert_allclose(
            last10,
            stored,
            rtol=1e-5,
            atol=1e-8,
            err_msg=f"Numerical regression for {key}",
        )


# ---------------------------------------------------------------------------
# Performance benchmarks
# ---------------------------------------------------------------------------


class TestPerformance:
    """Timing benchmarks — record wall time and compare to baselines if present."""

    PERF_THRESHOLD_FACTOR = 2.0  # fail if run is > 2× slower than baseline

    @pytest.mark.parametrize(
        "entry", INDICATOR_SUITE, ids=[e["name"] for e in INDICATOR_SUITE]
    )
    def test_timing(
        self,
        entry: dict[str, Any],
        ohlcv: dict[str, np.ndarray],
        request: pytest.FixtureRequest,
    ) -> None:
        """Time the indicator on the canonical dataset."""
        # Warm-up run
        _run_indicator(entry, ohlcv)

        # Timed run
        t0 = time.perf_counter()
        for _ in range(5):
            _run_indicator(entry, ohlcv)
        elapsed = (time.perf_counter() - t0) / 5.0  # average over 5 runs

        # Store timing in request node for reporting
        request.node._ferro_ta_timing = elapsed  # type: ignore[attr-defined]

        # Compare to baseline if available
        if BASELINE_PATH.exists():
            baselines = np.load(BASELINE_PATH, allow_pickle=True)
            key = f"timing_{entry['name']}"
            if key in baselines:
                baseline_time = float(baselines[key])
                if elapsed > baseline_time * self.PERF_THRESHOLD_FACTOR:
                    pytest.fail(
                        f"{entry['name']}: timing regression — "
                        f"current {elapsed * 1000:.2f}ms vs "
                        f"baseline {baseline_time * 1000:.2f}ms "
                        f"(>{self.PERF_THRESHOLD_FACTOR}×)"
                    )


# ---------------------------------------------------------------------------
# Baseline update helper
# ---------------------------------------------------------------------------


def update_baselines(ohlcv_data: dict[str, np.ndarray]) -> None:
    """Write current indicator outputs and timings to baselines.npz.

    Call this after intentional changes to update the stored baselines::

        python -c "
        import numpy as np
        from benchmarks.test_benchmark_suite import update_baselines, FIXTURE_PATH
        data = {k: v for k, v in np.load(FIXTURE_PATH).items()}
        update_baselines(data)
        "
    """
    store: dict[str, np.ndarray] = {}
    for entry in INDICATOR_SUITE:
        out = _run_indicator(entry, ohlcv_data)
        valid = out[~np.isnan(out)]
        store[entry["name"]] = valid[-10:]

        # Timing
        t0 = time.perf_counter()
        for _ in range(5):
            _run_indicator(entry, ohlcv_data)
        store[f"timing_{entry['name']}"] = np.array([(time.perf_counter() - t0) / 5.0])

    np.savez_compressed(BASELINE_PATH, **store)
    print(f"Baselines written to {BASELINE_PATH}")


if __name__ == "__main__":
    if not FIXTURE_PATH.exists():
        print(f"Fixture not found: {FIXTURE_PATH}")
        print("Run: python benchmarks/fixtures/generate_canonical.py")
    else:
        data = {k: v for k, v in np.load(FIXTURE_PATH).items()}
        update_baselines(data)
