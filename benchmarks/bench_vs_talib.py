"""
ferro_ta vs TA-Lib speed comparison.

Measures throughput (M bars/s) for both libraries on the same synthetic data
and parameters. The output is intentionally evidence-heavy:

- median timings
- per-run timing samples
- variability stats
- Python-tracked peak allocation snapshots
- machine, runtime, and build metadata

This is meant to support a narrow claim: ferro-ta is often faster on selected
indicators, not universally faster.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import tracemalloc
from typing import Any

import numpy as np

try:
    import talib  # noqa: F401

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    talib = None  # type: ignore[assignment]

import ferro_ta

try:
    from benchmarks.metadata import benchmark_metadata, package_versions
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from metadata import benchmark_metadata, package_versions

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_WARMUP = 1
N_RUNS = 7
DEFAULT_SIZES = [10_000, 100_000, 1_000_000]
TIE_EPSILON = 0.05

_rng = np.random.default_rng(42)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _summary_stats(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        return {
            "median_ms": 0.0,
            "mean_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "stddev_ms": 0.0,
            "cv_pct": 0.0,
        }

    mean_ms = sum(samples_ms) / len(samples_ms)
    variance = (
        sum((sample - mean_ms) ** 2 for sample in samples_ms) / (len(samples_ms) - 1)
        if len(samples_ms) > 1
        else 0.0
    )
    stddev_ms = math.sqrt(variance)
    cv_pct = (stddev_ms / mean_ms * 100.0) if mean_ms else 0.0
    return {
        "median_ms": round(_median(samples_ms), 4),
        "mean_ms": round(mean_ms, 4),
        "min_ms": round(min(samples_ms), 4),
        "max_ms": round(max(samples_ms), 4),
        "stddev_ms": round(stddev_ms, 4),
        "cv_pct": round(cv_pct, 3),
    }


def _outcome(speedup: float) -> str:
    if speedup > 1.0 + TIE_EPSILON:
        return "ferro_ta_win"
    if speedup < 1.0 - TIE_EPSILON:
        return "talib_win"
    return "tie"


def _summary_for_size(results: list[dict[str, Any]], size: int) -> dict[str, Any]:
    rows = [row for row in results if row.get("size") == size and "speedup" in row]
    if not rows:
        return {"size": size, "rows": 0}

    speedups = [float(row["speedup"]) for row in rows]
    wins = sum(1 for row in rows if row.get("outcome") == "ferro_ta_win")
    ties = sum(1 for row in rows if row.get("outcome") == "tie")
    losses = sum(1 for row in rows if row.get("outcome") == "talib_win")
    return {
        "size": size,
        "rows": len(rows),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": round(wins / len(rows), 4),
        "non_loss_rate": round((wins + ties) / len(rows), 4),
        "median_speedup": round(_median(speedups), 4),
        "min_speedup": round(min(speedups), 4),
        "max_speedup": round(max(speedups), 4),
        "talib_wins_or_ties": [
            row["indicator"]
            for row in rows
            if row.get("outcome") in {"talib_win", "tie"}
        ],
    }


def _synthetic_ohlcv(
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Generate OHLCV so that ta crate DataItem constraints hold: low >= 0,
    # volume >= 0, and low <= open, close <= high, high >= open.
    close = 100.0 + np.cumsum(_rng.standard_normal(n) * 0.5)
    open_ = close + _rng.standard_normal(n) * 0.2
    high = np.maximum(open_, close) + np.abs(_rng.standard_normal(n) * 0.3)
    low = np.minimum(open_, close) - np.abs(_rng.standard_normal(n) * 0.3)
    high = np.maximum(high, low)
    low = np.maximum(low, 0.0)
    high = np.maximum(high, low)
    open_ = np.clip(open_, low, high)
    close = np.clip(close, low, high)
    volume = np.abs(_rng.standard_normal(n) * 1_000_000) + 500_000
    return open_, high, low, close, volume


def _timed_runs_ms(fn, *args, **kwargs) -> list[float]:
    for _ in range(N_WARMUP):
        fn(*args, **kwargs)

    samples_ms: list[float] = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    return samples_ms


def _python_peak_bytes(fn, *args, **kwargs) -> int | None:
    try:
        tracemalloc.start()
        tracemalloc.reset_peak()
        fn(*args, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
        return int(peak)
    except Exception:
        return None
    finally:
        tracemalloc.stop()


def _throughput_m_bars_s(size: int, median_ms: float) -> float:
    if median_ms <= 0:
        return 0.0
    return (size / 1e6) / (median_ms / 1000.0)


# ---------------------------------------------------------------------------
# Benchmarked callables
# ---------------------------------------------------------------------------


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

SKIP_1M_FOR = {"STOCH", "ADX"}


def run_comparison(sizes: list[int], json_path: str | None) -> list[dict[str, Any]]:
    max_size = max(sizes)
    open_, high, low, close, volume = _synthetic_ohlcv(max_size)
    results: list[dict[str, Any]] = []

    col_label = 10
    col_size = 10
    col_ft_ms = 12
    col_ta_ms = 12
    col_speedup = 10
    col_ft_m = 12
    col_ta_m = 12

    if not TALIB_AVAILABLE:
        print("Note: ta-lib not installed. Reporting ferro_ta timings only.")
        print(
            "Install with: pip install ta-lib (or conda install ta-lib) for comparison.\n"
        )

    print(
        f"\nferro_ta vs TA-Lib — median of {N_RUNS} measured runs after {N_WARMUP} warmup"
    )
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

            ft_samples_ms = _timed_runs_ms(
                ft_run, open_, high, low, close, volume, size
            )
            ft_stats = _summary_stats(ft_samples_ms)
            ft_median_ms = float(ft_stats["median_ms"])
            ft_m_bars_s = _throughput_m_bars_s(size, ft_median_ms)
            ft_peak_bytes = _python_peak_bytes(
                ft_run, open_, high, low, close, volume, size
            )

            row: dict[str, Any] = {
                "indicator": name,
                "size": size,
                "input_layout": {
                    "dtype": "float64",
                    "contiguous": True,
                },
                "ferro_ta_ms": round(ft_median_ms, 4),
                "ferro_ta_m_bars_s": round(ft_m_bars_s, 2),
                "ferro_ta_runs_ms": [round(sample, 4) for sample in ft_samples_ms],
                "ferro_ta_stats": ft_stats,
                "python_peak_allocation_bytes": {
                    "ferro_ta": ft_peak_bytes,
                },
            }

            if TALIB_AVAILABLE:
                ta_samples_ms = _timed_runs_ms(
                    ta_run, open_, high, low, close, volume, size
                )
                ta_stats = _summary_stats(ta_samples_ms)
                ta_median_ms = float(ta_stats["median_ms"])
                ta_m_bars_s = _throughput_m_bars_s(size, ta_median_ms)
                speedup = (
                    ta_median_ms / ft_median_ms if ft_median_ms > 0 else float("inf")
                )
                outcome = _outcome(speedup)
                ta_peak_bytes = _python_peak_bytes(
                    ta_run, open_, high, low, close, volume, size
                )

                print(
                    f"{name:<{col_label}} {size:<{col_size}} "
                    f"{ft_median_ms:<{col_ft_ms}.3f} {ta_median_ms:<{col_ta_ms}.3f} "
                    f"{speedup:<{col_speedup}.2f}x {ft_m_bars_s:<{col_ft_m}.1f} {ta_m_bars_s:<{col_ta_m}.1f}"
                )

                row.update(
                    {
                        "talib_ms": round(ta_median_ms, 4),
                        "talib_m_bars_s": round(ta_m_bars_s, 2),
                        "talib_runs_ms": [round(sample, 4) for sample in ta_samples_ms],
                        "talib_stats": ta_stats,
                        "speedup": round(speedup, 4),
                        "outcome": outcome,
                    }
                )
                row["python_peak_allocation_bytes"]["talib"] = ta_peak_bytes
            else:
                print(
                    f"{name:<{col_label}} {size:<{col_size}} "
                    f"{ft_median_ms:<{col_ft_ms}.3f} {'N/A':<{col_ta_ms}} "
                    f"{'N/A':<{col_speedup}} {ft_m_bars_s:<{col_ft_m}.1f} {'N/A':<{col_ta_m}}"
                )

            results.append(row)

    print()
    if TALIB_AVAILABLE and results:
        wins = sum(1 for row in results if row.get("outcome") == "ferro_ta_win")
        total = len([row for row in results if "speedup" in row])
        print(f"Summary: ferro_ta ahead outside the tie band on {wins}/{total} rows.")
    print()

    if json_path:
        metadata = benchmark_metadata(
            "benchmark_vs_talib",
            extra={
                "dataset": {
                    "generator": "synthetic_ohlcv",
                    "sizes": sizes,
                    "dtype": "float64",
                    "array_layout": "C-contiguous",
                    "seed": 42,
                },
                "methodology": {
                    "warmup_runs": N_WARMUP,
                    "measured_runs": N_RUNS,
                    "reported_metric": "median_ms",
                    "speedup_definition": "talib_median_ms / ferro_ta_median_ms",
                    "tie_band": f"{1.0 - TIE_EPSILON:.2f} to {1.0 + TIE_EPSILON:.2f}",
                    "input_layout_notes": (
                        "Benchmarks use contiguous float64 arrays. If your workload "
                        "passes non-contiguous arrays or other dtypes, benchmark that "
                        "separately because wrapper overhead can dominate."
                    ),
                    "allocation_notes": (
                        "python_peak_allocation_bytes is a tracemalloc snapshot of "
                        "Python-tracked allocations only; it is not a full native RSS "
                        "or allocator profile."
                    ),
                },
                "packages": package_versions("numpy", "ferro-ta", "TA-Lib"),
            },
        )
        out = {
            "schema_version": 2,
            "command": " ".join(["python", *sys.argv]),
            "n_warmup": N_WARMUP,
            "n_runs": N_RUNS,
            "sizes": sizes,
            "talib_available": TALIB_AVAILABLE,
            "runtime": metadata["runtime"],
            "git": metadata["git"],
            "metadata": metadata,
            "summary": {
                "total_rows": len(results),
                "by_size": [_summary_for_size(results, size) for size in sizes],
            },
            "results": results,
        }
        if not TALIB_AVAILABLE:
            out["note"] = "ferro_ta only; ta-lib not installed"
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=2)
        print(f"Results written to {json_path}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="ferro_ta vs TA-Lib speed comparison")
    parser.add_argument("--json", default=None, help="Write results to JSON file")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="Bar counts to benchmark (default: 10000 100000 1000000)",
    )
    args = parser.parse_args()
    run_comparison(args.sizes, args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
