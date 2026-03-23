"""
ferro_ta vs TA-Lib speed comparison.

Measures throughput (M bars/s) for both libraries on the same data and parameters,
and reports speedup (talib_time / ferro_ta_time; > 1 means ferro_ta is faster).

Requirements:
    pip install ta-lib   # or conda install ta-lib

Run:
    python benchmarks/bench_vs_talib.py
    python benchmarks/bench_vs_talib.py --json results.json
    python benchmarks/bench_vs_talib.py --sizes 10000 100000  # default: 10k, 100k, 1M

If ta-lib is not installed, the script still runs and reports ferro_ta timings only (no speedup).
Methodology: same synthetic data, same parameters, median of 7 runs after warmup.
Environment: document Python version and OS when publishing results.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np

try:
    import talib  # noqa: F401
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    talib = None  # type: ignore[assignment]

import ferro_ta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_WARMUP = 1
N_RUNS = 7
DEFAULT_SIZES = [10_000, 100_000, 1_000_000]

_rng = np.random.default_rng(42)


def _git_info() -> dict[str, Any]:
    """Best-effort git metadata for benchmark reproducibility."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = None

    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except Exception:
        dirty = None

    return {"commit": commit, "dirty": dirty}


def _runtime_info() -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _summary_for_size(results: list[dict[str, Any]], size: int) -> dict[str, Any]:
    rows = [r for r in results if r.get("size") == size and "speedup" in r]
    if not rows:
        return {"size": size, "rows": 0}

    speedups = [float(r["speedup"]) for r in rows]
    wins = sum(1 for s in speedups if s > 1.0)
    speedups_sorted = sorted(speedups)
    mid = len(speedups_sorted) // 2
    if len(speedups_sorted) % 2:
        median = speedups_sorted[mid]
    else:
        median = (speedups_sorted[mid - 1] + speedups_sorted[mid]) / 2.0

    return {
        "size": size,
        "rows": len(rows),
        "wins": wins,
        "win_rate": wins / len(rows),
        "median_speedup": round(median, 4),
        "min_speedup": round(min(speedups), 4),
        "max_speedup": round(max(speedups), 4),
    }


def _synthetic_ohlcv(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Generate OHLCV so that ta crate DataItem constraints hold: low >= 0, volume >= 0,
    # and low <= open, close <= high, high >= open (see ta DataItemBuilder::build).
    close = 100.0 + np.cumsum(_rng.standard_normal(n) * 0.5)
    open_ = close + _rng.standard_normal(n) * 0.2
    high = np.maximum(open_, close) + np.abs(_rng.standard_normal(n) * 0.3)
    low = np.minimum(open_, close) - np.abs(_rng.standard_normal(n) * 0.3)
    # Enforce high >= low and low >= 0 (ta requires non-negative prices)
    high = np.maximum(high, low)
    low = np.maximum(low, 0.0)
    high = np.maximum(high, low)  # again after clamping low
    open_ = np.clip(open_, low, high)
    close = np.clip(close, low, high)
    volume = np.abs(_rng.standard_normal(n) * 1_000_000) + 500_000
    return open_, high, low, close, volume


def _median_time_ms(fn, *args, **kwargs) -> float:
    for _ in range(N_WARMUP):
        fn(*args, **kwargs)
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


# Each entry: (label, ferro_ta_callable, talib_callable, needs_ohlcv)
# ferro_ta_callable / talib_callable receive (open_, high, low, close, volume) and size;
# they return (args, ft_kwargs, ta_kwargs) or we use a simpler convention:
# we pass (o, h, l, c, v) and size; each runner knows how to slice and call.
def _run_ft_sma(o, h, l, c, v, n):
    return ferro_ta.SMA(c[:n], timeperiod=14)


def _run_ta_sma(o, h, l, c, v, n):
    return talib.SMA(c[:n], timeperiod=14)


def _run_ft_ema(o, h, l, c, v, n):
    return ferro_ta.EMA(c[:n], timeperiod=14)


def _run_ta_ema(o, h, l, c, v, n):
    return talib.EMA(c[:n], timeperiod=14)


def _run_ft_rsi(o, h, l, c, v, n):
    return ferro_ta.RSI(c[:n], timeperiod=14)


def _run_ta_rsi(o, h, l, c, v, n):
    return talib.RSI(c[:n], timeperiod=14)


def _run_ft_bbands(o, h, l, c, v, n):
    return ferro_ta.BBANDS(c[:n], timeperiod=20, nbdevup=2.0, nbdevdn=2.0)


def _run_ta_bbands(o, h, l, c, v, n):
    return talib.BBANDS(c[:n], timeperiod=20, nbdevup=2.0, nbdevdn=2.0)


def _run_ft_macd(o, h, l, c, v, n):
    return ferro_ta.MACD(c[:n], fastperiod=12, slowperiod=26, signalperiod=9)


def _run_ta_macd(o, h, l, c, v, n):
    return talib.MACD(c[:n], fastperiod=12, slowperiod=26, signalperiod=9)


def _run_ft_atr(o, h, l, c, v, n):
    return ferro_ta.ATR(h[:n], l[:n], c[:n], timeperiod=14)


def _run_ta_atr(o, h, l, c, v, n):
    return talib.ATR(h[:n], l[:n], c[:n], timeperiod=14)


def _run_ft_stoch(o, h, l, c, v, n):
    return ferro_ta.STOCH(h[:n], l[:n], c[:n])


def _run_ta_stoch(o, h, l, c, v, n):
    return talib.STOCH(h[:n], l[:n], c[:n])


def _run_ft_adx(o, h, l, c, v, n):
    return ferro_ta.ADX(h[:n], l[:n], c[:n], timeperiod=14)


def _run_ta_adx(o, h, l, c, v, n):
    return talib.ADX(h[:n], l[:n], c[:n], timeperiod=14)


def _run_ft_cci(o, h, l, c, v, n):
    return ferro_ta.CCI(h[:n], l[:n], c[:n], timeperiod=14)


def _run_ta_cci(o, h, l, c, v, n):
    return talib.CCI(h[:n], l[:n], c[:n], timeperiod=14)


def _run_ft_obv(o, h, l, c, v, n):
    return ferro_ta.OBV(c[:n], v[:n])


def _run_ta_obv(o, h, l, c, v, n):
    return talib.OBV(c[:n], v[:n])


def _run_ft_mfi(o, h, l, c, v, n):
    return ferro_ta.MFI(h[:n], l[:n], c[:n], v[:n], timeperiod=14)


def _run_ta_mfi(o, h, l, c, v, n):
    return talib.MFI(h[:n], l[:n], c[:n], v[:n], timeperiod=14)


def _run_ft_wma(o, h, l, c, v, n):
    return ferro_ta.WMA(c[:n], timeperiod=14)


def _run_ta_wma(o, h, l, c, v, n):
    return talib.WMA(c[:n], timeperiod=14)


# List of (indicator_name, ft_runner, ta_runner); skip 1M for very slow indicators if needed
COMPARISON_CASES = [
    ("SMA", _run_ft_sma, _run_ta_sma),
    ("EMA", _run_ft_ema, _run_ta_ema),
    ("RSI", _run_ft_rsi, _run_ta_rsi),
    ("BBANDS", _run_ft_bbands, _run_ta_bbands),
    ("MACD", _run_ft_macd, _run_ta_macd),
    ("ATR", _run_ft_atr, _run_ta_atr),
    ("STOCH", _run_ft_stoch, _run_ta_stoch),
    ("ADX", _run_ft_adx, _run_ta_adx),
    ("CCI", _run_ft_cci, _run_ta_cci),
    ("OBV", _run_ft_obv, _run_ta_obv),
    ("MFI", _run_ft_mfi, _run_ta_mfi),
    ("WMA", _run_ft_wma, _run_ta_wma),
]

# For STOCH/ADX and other heavier indicators, optionally skip 1M to keep runtime reasonable
SKIP_1M_FOR = {"STOCH", "ADX"}


def run_comparison(sizes: list[int], json_path: str | None) -> list[dict[str, Any]]:
    max_size = max(sizes)
    open_, high, low, close, volume = _synthetic_ohlcv(max_size)
    results = []
    col_label = 10
    col_size = 10
    col_ft_ms = 12
    col_ta_ms = 12
    col_speedup = 10
    col_ft_m = 12
    col_ta_m = 12

    if not TALIB_AVAILABLE:
        print("Note: ta-lib not installed — reporting ferro_ta timings only (no speedup).")
        print("Install with: pip install ta-lib (or conda install ta-lib) for comparison.\n")

    print(f"\nferro_ta vs TA-Lib — median of {N_RUNS} runs (after {N_WARMUP} warmup)")
    print(f"Sizes: {sizes}")
    print()
    header = (
        f"{'Indicator':<{col_label}} {'Size':<{col_size}} "
        f"{'ferro_ta(ms)':<{col_ft_ms}} {'TA-Lib(ms)':<{col_ta_ms}} "
        f"{'Speedup':<{col_speedup}} {'ferro_ta(M/s)':<{col_ft_m}} {'TA-Lib(M/s)':<{col_ta_m}}"
    )
    print(header)
    print("-" * len(header))

    for name, ft_run, ta_run in COMPARISON_CASES:
        for size in sizes:
            if size == 1_000_000 and name in SKIP_1M_FOR:
                continue
            ms_ft = _median_time_ms(ft_run, open_, high, low, close, volume, size)
            if TALIB_AVAILABLE:
                ms_ta = _median_time_ms(ta_run, open_, high, low, close, volume, size)
                speedup = ms_ta / ms_ft if ms_ft > 0 else float("inf")
                m_bars_ft = (size / 1e6) / (ms_ft / 1000) if ms_ft > 0 else 0
                m_bars_ta = (size / 1e6) / (ms_ta / 1000) if ms_ta > 0 else 0
                print(
                    f"{name:<{col_label}} {size:<{col_size}} "
                    f"{ms_ft:<{col_ft_ms}.3f} {ms_ta:<{col_ta_ms}.3f} "
                    f"{speedup:<{col_speedup}.2f}x {m_bars_ft:<{col_ft_m}.1f} {m_bars_ta:<{col_ta_m}.1f}"
                )
                row = {
                    "indicator": name,
                    "size": size,
                    "ferro_ta_ms": round(ms_ft, 4),
                    "talib_ms": round(ms_ta, 4),
                    "speedup": round(speedup, 4),
                    "ferro_ta_m_bars_s": round(m_bars_ft, 2),
                    "talib_m_bars_s": round(m_bars_ta, 2),
                }
            else:
                m_bars_ft = (size / 1e6) / (ms_ft / 1000) if ms_ft > 0 else 0
                print(
                    f"{name:<{col_label}} {size:<{col_size}} "
                    f"{ms_ft:<{col_ft_ms}.3f} {'N/A':<{col_ta_ms}} "
                    f"{'N/A':<{col_speedup}} {m_bars_ft:<{col_ft_m}.1f} {'N/A':<{col_ta_m}}"
                )
                row = {
                    "indicator": name,
                    "size": size,
                    "ferro_ta_ms": round(ms_ft, 4),
                    "ferro_ta_m_bars_s": round(m_bars_ft, 2),
                }
            results.append(row)

    print()
    if TALIB_AVAILABLE and results:
        wins = sum(1 for r in results if r.get("speedup", 0) > 1)
        total = len(results)
        print(f"Summary: ferro_ta faster on {wins}/{total} rows (speedup > 1).")
    print()
    if json_path:
        out = {
            "schema_version": 1,
            "command": "python benchmarks/bench_vs_talib.py",
            "n_warmup": N_WARMUP,
            "n_runs": N_RUNS,
            "sizes": sizes,
            "talib_available": TALIB_AVAILABLE,
            "runtime": _runtime_info(),
            "git": _git_info(),
            "summary": {
                "total_rows": len(results),
                "by_size": [_summary_for_size(results, s) for s in sizes],
            },
            "results": results,
        }
        if not TALIB_AVAILABLE:
            out["note"] = "ferro_ta only — ta-lib not installed"
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results written to {json_path}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="ferro_ta vs TA-Lib speed comparison")
    ap.add_argument("--json", default=None, help="Write results to JSON file")
    ap.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="Bar counts to benchmark (default: 10000 100000 1000000)",
    )
    args = ap.parse_args()
    run_comparison(args.sizes, args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
