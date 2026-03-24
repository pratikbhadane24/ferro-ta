"""
Cross-library speed benchmarks using pytest-benchmark.

Run:  pytest benchmarks/test_speed.py --benchmark-only -v
      pytest benchmarks/test_speed.py --benchmark-only --benchmark-json=benchmarks/results.json

Streaming benchmarks are in test_streaming_speed.py
"""

from __future__ import annotations

import pytest

from benchmarks.data_generator import LARGE
from benchmarks.wrapper_registry import (
    INDICATOR_CATEGORIES,
    available_libraries,
    execute_indicator,
    is_supported,
)

BENCH_DATA = LARGE  # 100k bars for main benchmarks
BENCH_LIBS = available_libraries()


def _make_bench(indicator: str, library: str):
    """Return a benchmark function that runs indicator on library (uses BENCH_DATA)."""

    def _fn():
        execute_indicator(library, indicator, BENCH_DATA)

    _fn.__name__ = f"{library}_{indicator}"
    return _fn


# ── Parametrize over all (indicator, library) combinations ───────────────────


def pytest_generate_tests(metafunc):
    if "indicator" in metafunc.fixturenames and "library" in metafunc.fixturenames:
        params = []
        for cat, inds in INDICATOR_CATEGORIES.items():
            for ind in inds:
                for lib in BENCH_LIBS:
                    params.append(pytest.param(ind, lib, id=f"{cat}/{ind}/{lib}"))
        metafunc.parametrize("indicator,library", params)


class TestSpeed:
    """One benchmark per (indicator, library) pair — all at 100k bars (LARGE dataset)."""

    def test_speed(self, benchmark, indicator, library):
        if not is_supported(library, indicator):
            pytest.skip(f"{library} does not implement {indicator}")
        fn = _make_bench(indicator, library)
        benchmark.pedantic(fn, iterations=5, rounds=20, warmup_rounds=2)


# ── Standalone head-to-head for the most important indicators ─────────────────


@pytest.mark.parametrize(
    "indicator,libs",
    [
        ("SMA", ["ferro_ta", "talib", "tulipy", "pandas_ta", "ta", "finta"]),
        ("EMA", ["ferro_ta", "talib", "tulipy", "pandas_ta", "ta", "finta"]),
        ("RSI", ["ferro_ta", "talib", "tulipy", "pandas_ta", "ta", "finta"]),
        ("MACD", ["ferro_ta", "talib", "tulipy", "pandas_ta", "ta", "finta"]),
        ("BBANDS", ["ferro_ta", "talib", "tulipy", "pandas_ta", "ta", "finta"]),
        ("ATR", ["ferro_ta", "talib", "tulipy", "pandas_ta", "ta", "finta"]),
        ("CCI", ["ferro_ta", "talib", "tulipy", "pandas_ta", "ta", "finta"]),
        ("WILLR", ["ferro_ta", "talib", "tulipy", "pandas_ta", "ta", "finta"]),
        ("OBV", ["ferro_ta", "talib", "tulipy", "pandas_ta", "ta", "finta"]),
        ("ADX", ["ferro_ta", "talib", "tulipy", "pandas_ta", "ta", "finta"]),
        ("MFI", ["ferro_ta", "talib", "tulipy", "pandas_ta", "ta", "finta"]),
        ("STOCH", ["ferro_ta", "talib", "tulipy", "pandas_ta", "ta", "finta"]),
    ],
)
def test_head_to_head(benchmark, indicator, libs):
    """Benchmark ferro_ta vs all peers — for README table generation."""
    if not is_supported("ferro_ta", indicator):
        pytest.skip(f"ferro_ta does not implement {indicator}")
    fn = _make_bench(indicator, "ferro_ta")
    benchmark.pedantic(fn, iterations=5, rounds=20, warmup_rounds=2)


# ── Large dataset benchmarks (100k bars) ─────────────────────────────────────


@pytest.mark.parametrize(
    "indicator",
    ["SMA", "EMA", "RSI", "MACD", "ATR", "BBANDS", "OBV", "CCI", "ADX", "MFI"],
)
def test_large_dataset(benchmark, indicator):
    """Scaling benchmark at 100k bars for ferro_ta."""
    if not is_supported("ferro_ta", indicator):
        pytest.skip(f"ferro_ta does not implement {indicator}")

    def _fn():
        execute_indicator("ferro_ta", indicator, LARGE)

    benchmark.pedantic(_fn, iterations=3, rounds=10, warmup_rounds=1)
