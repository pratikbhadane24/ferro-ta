"""
Integration tests using the synthetic OHLCV fixture in tests/fixtures/.

These tests verify that:
- All major indicator categories produce finite output on realistic data.
- Output lengths match the input length.
- Error codes and suggestion hints are included in exception messages.
- ferro_ta.indicators() and ferro_ta.info() work correctly.
- Logging utilities (enable_debug, log_call, benchmark) work correctly.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load the OHLCV fixture
# ---------------------------------------------------------------------------

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "ohlcv_daily.csv"


def _load_fixture() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Return (open, high, low, close, volume) as float64 arrays."""
    rows = []
    with open(FIXTURE_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    open_ = np.array([float(r["open"]) for r in rows])
    high = np.array([float(r["high"]) for r in rows])
    low = np.array([float(r["low"]) for r in rows])
    close = np.array([float(r["close"]) for r in rows])
    volume = np.array([float(r["volume"]) for r in rows])
    return open_, high, low, close, volume


@pytest.fixture(scope="module")
def ohlcv():
    return _load_fixture()


# ---------------------------------------------------------------------------
# Fixture sanity
# ---------------------------------------------------------------------------


def test_fixture_loads(ohlcv):
    o, h, l, c, v = ohlcv
    assert len(c) == 252
    assert np.all(h >= l)
    assert np.all(v > 0)


# ---------------------------------------------------------------------------
# Overlap indicators on real OHLCV data
# ---------------------------------------------------------------------------


def test_sma_on_fixture(ohlcv):
    from ferro_ta import SMA

    _, _, _, close, _ = ohlcv
    result = SMA(close, timeperiod=20)
    assert len(result) == len(close)
    # First 19 values should be NaN, rest finite
    assert np.all(np.isnan(result[:19]))
    assert np.all(np.isfinite(result[19:]))


def test_ema_on_fixture(ohlcv):
    from ferro_ta import EMA

    _, _, _, close, _ = ohlcv
    result = EMA(close, timeperiod=14)
    assert len(result) == len(close)
    assert np.all(np.isfinite(result[13:]))


def test_bbands_on_fixture(ohlcv):
    from ferro_ta import BBANDS

    _, _, _, close, _ = ohlcv
    upper, mid, lower = BBANDS(close, timeperiod=20)
    assert len(upper) == len(close)
    assert np.all(upper[19:] >= mid[19:])
    assert np.all(mid[19:] >= lower[19:])


# ---------------------------------------------------------------------------
# Momentum indicators
# ---------------------------------------------------------------------------


def test_rsi_on_fixture(ohlcv):
    from ferro_ta import RSI

    _, _, _, close, _ = ohlcv
    result = RSI(close, timeperiod=14)
    assert len(result) == len(close)
    valid = result[~np.isnan(result)]
    assert np.all(valid >= 0) and np.all(valid <= 100)


def test_macd_on_fixture(ohlcv):
    from ferro_ta import MACD

    _, _, _, close, _ = ohlcv
    macd, signal, hist = MACD(close)
    assert len(macd) == len(close)


def test_adx_on_fixture(ohlcv):
    from ferro_ta import ADX

    _, high, low, close, _ = ohlcv
    result = ADX(high, low, close, timeperiod=14)
    assert len(result) == len(close)


def test_stoch_on_fixture(ohlcv):
    from ferro_ta import STOCH

    _, high, low, close, _ = ohlcv
    slowk, slowd = STOCH(high, low, close)
    assert len(slowk) == len(close)


# ---------------------------------------------------------------------------
# Volatility indicators
# ---------------------------------------------------------------------------


def test_atr_on_fixture(ohlcv):
    from ferro_ta import ATR

    _, high, low, close, _ = ohlcv
    result = ATR(high, low, close, timeperiod=14)
    assert len(result) == len(close)
    valid = result[~np.isnan(result)]
    assert np.all(valid >= 0)


# ---------------------------------------------------------------------------
# Volume indicators
# ---------------------------------------------------------------------------


def test_obv_on_fixture(ohlcv):
    from ferro_ta import OBV

    _, _, _, close, volume = ohlcv
    result = OBV(close, volume)
    assert len(result) == len(close)
    assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Error handling — error codes and suggestion hints
# ---------------------------------------------------------------------------


def test_value_error_has_code():
    from ferro_ta.core.exceptions import FerroTAValueError, check_timeperiod

    with pytest.raises(FerroTAValueError) as exc_info:
        check_timeperiod(0, "timeperiod", minimum=1)
    exc = exc_info.value
    assert exc.code == "FTERR001"
    assert "FTERR001" in str(exc)
    assert exc.suggestion is not None
    assert "Suggestion" in str(exc)


def test_input_error_length_mismatch_has_code():
    from ferro_ta.core.exceptions import FerroTAInputError, check_equal_length

    with pytest.raises(FerroTAInputError) as exc_info:
        check_equal_length(open=np.array([1.0, 2.0]), close=np.array([1.0]))
    exc = exc_info.value
    assert exc.code == "FTERR004"
    assert "Suggestion" in str(exc)


def test_input_error_too_short_has_code():
    from ferro_ta.core.exceptions import FerroTAInputError, check_min_length

    with pytest.raises(FerroTAInputError) as exc_info:
        check_min_length(np.array([1.0]), 10, "close")
    exc = exc_info.value
    assert exc.code == "FTERR003"
    assert "Suggestion" in str(exc)


def test_finite_check_error_has_code():
    from ferro_ta.core.exceptions import FerroTAInputError, check_finite

    arr = np.array([1.0, float("nan"), 3.0])
    with pytest.raises(FerroTAInputError) as exc_info:
        check_finite(arr, "close")
    exc = exc_info.value
    assert exc.code == "FTERR005"
    assert "Suggestion" in str(exc)


# ---------------------------------------------------------------------------
# API discovery
# ---------------------------------------------------------------------------


def test_indicators_returns_list():
    import ferro_ta

    result = ferro_ta.indicators()
    assert isinstance(result, list)
    assert len(result) > 20
    names = [d["name"] for d in result]
    assert "SMA" in names
    assert "RSI" in names
    assert "ATR" in names


def test_indicators_filter_by_category():
    import ferro_ta

    overlap = ferro_ta.indicators(category="overlap")
    assert all(d["category"] == "overlap" for d in overlap)
    assert any(d["name"] == "SMA" for d in overlap)


def test_info_by_function():
    import ferro_ta

    d = ferro_ta.info(ferro_ta.SMA)
    assert d["name"] == "SMA"
    assert "close" in d["params"]
    assert "timeperiod" in d["params"]
    assert isinstance(d["doc"], str)


def test_info_by_string():
    import ferro_ta

    d = ferro_ta.info("EMA")
    assert d["name"] == "EMA"


def test_info_unknown_raises():
    import ferro_ta

    with pytest.raises(ValueError, match="No indicator named"):
        ferro_ta.info("DOES_NOT_EXIST")


# ---------------------------------------------------------------------------
# Logging utilities
# ---------------------------------------------------------------------------


def test_get_logger_returns_logger():
    import ferro_ta

    logger = ferro_ta.get_logger()
    assert isinstance(logger, logging.Logger)
    assert logger.name == "ferro_ta"


def test_enable_disable_debug():
    import ferro_ta

    ferro_ta.enable_debug()
    assert ferro_ta.get_logger().level == logging.DEBUG
    ferro_ta.disable_debug()
    assert ferro_ta.get_logger().level == logging.WARNING


def test_debug_mode_context_manager():
    import ferro_ta

    with ferro_ta.debug_mode() as logger:
        assert logger.level == logging.DEBUG
    # After context, should be restored
    assert ferro_ta.get_logger().level == logging.WARNING


def test_log_call_returns_result(ohlcv):
    import ferro_ta
    from ferro_ta import SMA

    _, _, _, close, _ = ohlcv
    result = ferro_ta.log_call(SMA, close, timeperiod=10)
    assert len(result) == len(close)


def test_benchmark_returns_stats(ohlcv):
    import ferro_ta
    from ferro_ta import SMA

    _, _, _, close, _ = ohlcv
    stats = ferro_ta.benchmark(SMA, close, timeperiod=10, n=5, warmup=1)
    assert "mean_ms" in stats
    assert stats["mean_ms"] > 0
    assert stats["n"] == 5


def test_traced_decorator():
    import ferro_ta

    @ferro_ta.traced
    def dummy(x):
        return x * 2

    assert dummy(21) == 42
