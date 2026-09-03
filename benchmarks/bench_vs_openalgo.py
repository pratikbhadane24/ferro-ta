"""
ferro_ta vs openalgo speed comparison (internal bench extra only).

Measures median runtime on C-contiguous float64 OHLCV at 10k and 100k bars.
Skips the whole run when the optional comparison extra is not installed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

try:
    from benchmarks.metadata import benchmark_metadata, package_versions
    from benchmarks.wrapper_registry import (
        INDICATOR_DEFAULTS,
        REGISTRY,
        available_libraries,
        openalgo_overlap_names,
    )
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from metadata import benchmark_metadata, package_versions
    from wrapper_registry import (
        INDICATOR_DEFAULTS,
        REGISTRY,
        available_libraries,
        openalgo_overlap_names,
    )

_rng = np.random.default_rng(42)

N_WARMUP = 1
N_RUNS = 7
DEFAULT_SIZES = [10_000, 100_000]
TIE_EPSILON = 0.05
DEFAULT_JSON = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "latest"
    / "benchmark_vs_openalgo.json"
)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _summary_stats(samples_us: list[float]) -> dict[str, float]:
    if not samples_us:
        return {
            "median_us": 0.0,
            "mean_us": 0.0,
            "min_us": 0.0,
            "max_us": 0.0,
            "stddev_us": 0.0,
            "cv_pct": 0.0,
        }

    mean_us = sum(samples_us) / len(samples_us)
    variance = (
        sum((sample - mean_us) ** 2 for sample in samples_us) / (len(samples_us) - 1)
        if len(samples_us) > 1
        else 0.0
    )
    stddev_us = math.sqrt(variance)
    cv_pct = (stddev_us / mean_us * 100.0) if mean_us else 0.0
    return {
        "median_us": round(_median(samples_us), 4),
        "mean_us": round(mean_us, 4),
        "min_us": round(min(samples_us), 4),
        "max_us": round(max(samples_us), 4),
        "stddev_us": round(stddev_us, 4),
        "cv_pct": round(cv_pct, 3),
    }


def _outcome(speedup: float) -> str:
    if speedup > 1.0 + TIE_EPSILON:
        return "ferro_ta_win"
    if speedup < 1.0 - TIE_EPSILON:
        return "openalgo_win"
    return "tie"


def _summary_for_size(results: list[dict[str, Any]], size: int) -> dict[str, Any]:
    rows = [row for row in results if row.get("size") == size and "speedup" in row]
    if not rows:
        return {"size": size, "rows": 0}

    speedups = [float(row["speedup"]) for row in rows]
    wins = sum(1 for row in rows if row.get("outcome") == "ferro_ta_win")
    ties = sum(1 for row in rows if row.get("outcome") == "tie")
    losses = sum(1 for row in rows if row.get("outcome") == "openalgo_win")
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
        "openalgo_wins_or_ties": [
            row["indicator"]
            for row in rows
            if row.get("outcome") in {"openalgo_win", "tie"}
        ],
    }


def _timed_runs_us(fn, *args, **kwargs) -> list[float]:
    for _ in range(N_WARMUP):
        fn(*args, **kwargs)

    samples_us: list[float] = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        samples_us.append((time.perf_counter() - t0) * 1_000_000.0)
    return samples_us


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


def _throughput_m_bars_s(size: int, median_us: float) -> float:
    if median_us <= 0:
        return 0.0
    return (size / 1e6) / (median_us / 1_000_000.0)


def _openalgo_available() -> bool:
    return "openalgo" in available_libraries()


def _synthetic_ohlcv(n: int) -> dict[str, np.ndarray]:
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
    return {
        "open": np.ascontiguousarray(open_, dtype=np.float64),
        "high": np.ascontiguousarray(high, dtype=np.float64),
        "low": np.ascontiguousarray(low, dtype=np.float64),
        "close": np.ascontiguousarray(close, dtype=np.float64),
        "volume": np.ascontiguousarray(volume, dtype=np.float64),
    }


def run_comparison(sizes: list[int], json_path: str | None) -> list[dict[str, Any]]:
    if not _openalgo_available():
        print("Note: openalgo extra is not installed. Skipping competitor comparison.")
        print("Install with: uv sync --extra comparison\n")
        return []

    names = openalgo_overlap_names()
    if not names:
        print("No overlapping indicator wrappers registered. Nothing to benchmark.")
        return []

    max_size = max(sizes)
    full = _synthetic_ohlcv(max_size)
    results: list[dict[str, Any]] = []

    col_label = 18
    col_size = 10
    col_ft = 14
    col_oa = 14
    col_speedup = 10

    print(
        f"\nferro_ta vs openalgo — median of {N_RUNS} measured runs after {N_WARMUP} warmup"
    )
    print(f"Sizes: {sizes}")
    print(f"Indicators: {len(names)}")
    print()

    header = (
        f"{'Indicator':<{col_label}} {'Size':<{col_size}} "
        f"{'ferro_ta(us)':<{col_ft}} {'openalgo(us)':<{col_oa}} "
        f"{'Speedup':<{col_speedup}}"
    )
    print(header)
    print("-" * len(header))

    for name in names:
        params = dict(INDICATOR_DEFAULTS.get(name, {}))
        ft_fn = REGISTRY[("ferro_ta", name)]
        oa_fn = REGISTRY[("openalgo", name)]

        for size in sizes:
            slice_data = {
                key: np.ascontiguousarray(value[:size], dtype=np.float64)
                for key, value in full.items()
            }

            def _run_ft(fn=ft_fn, data=slice_data, kw=params):
                return fn(data, None, **kw)

            def _run_oa(fn=oa_fn, data=slice_data, kw=params):
                return fn(data, None, **kw)

            ft_samples_us = _timed_runs_us(_run_ft)
            ft_stats = _summary_stats(ft_samples_us)
            ft_median_us = float(ft_stats["median_us"])
            ft_m_bars_s = _throughput_m_bars_s(size, ft_median_us)
            ft_peak_bytes = _python_peak_bytes(_run_ft)

            oa_samples_us = _timed_runs_us(_run_oa)
            oa_stats = _summary_stats(oa_samples_us)
            oa_median_us = float(oa_stats["median_us"])
            oa_m_bars_s = _throughput_m_bars_s(size, oa_median_us)
            speedup = oa_median_us / ft_median_us if ft_median_us > 0 else float("inf")
            outcome = _outcome(speedup)
            oa_peak_bytes = _python_peak_bytes(_run_oa)

            print(
                f"{name:<{col_label}} {size:<{col_size}} "
                f"{ft_median_us:<{col_ft}.1f} {oa_median_us:<{col_oa}.1f} "
                f"{speedup:<{col_speedup}.2f}x"
            )

            results.append(
                {
                    "indicator": name,
                    "size": size,
                    "input_layout": {
                        "dtype": "float64",
                        "contiguous": True,
                    },
                    "ferro_ta_us": round(ft_median_us, 4),
                    "ferro_ta_ms": round(ft_median_us / 1000.0, 4),
                    "ferro_ta_m_bars_s": round(ft_m_bars_s, 2),
                    "ferro_ta_runs_us": [round(sample, 4) for sample in ft_samples_us],
                    "ferro_ta_stats": ft_stats,
                    "openalgo_us": round(oa_median_us, 4),
                    "openalgo_ms": round(oa_median_us / 1000.0, 4),
                    "openalgo_m_bars_s": round(oa_m_bars_s, 2),
                    "openalgo_runs_us": [round(sample, 4) for sample in oa_samples_us],
                    "openalgo_stats": oa_stats,
                    "speedup": round(speedup, 4),
                    "outcome": outcome,
                    "python_peak_allocation_bytes": {
                        "ferro_ta": ft_peak_bytes,
                        "openalgo": oa_peak_bytes,
                    },
                }
            )

    print()
    if results:
        wins = sum(1 for row in results if row.get("outcome") == "ferro_ta_win")
        total = len(results)
        print(f"Summary: ferro_ta ahead outside the tie band on {wins}/{total} rows.")
        losses_100k = [
            row["indicator"]
            for row in results
            if row.get("size") == 100_000 and row.get("outcome") == "openalgo_win"
        ]
        if losses_100k:
            print(f"100k losses (optimize later): {', '.join(losses_100k)}")
    print()

    if json_path:
        out_path = Path(json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = benchmark_metadata(
            "benchmark_vs_openalgo",
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
                    "reported_metric": "median_us",
                    "speedup_definition": "openalgo_median_us / ferro_ta_median_us",
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
                "packages": package_versions("numpy", "ferro-ta", "openalgo"),
            },
        )
        payload = {
            "schema_version": 2,
            "command": " ".join(["python", *sys.argv]),
            "n_warmup": N_WARMUP,
            "n_runs": N_RUNS,
            "sizes": sizes,
            "openalgo_available": True,
            "indicators": names,
            "runtime": metadata["runtime"],
            "git": metadata["git"],
            "metadata": metadata,
            "summary": {
                "total_rows": len(results),
                "by_size": [_summary_for_size(results, size) for size in sizes],
            },
            "results": results,
        }
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Results written to {out_path}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ferro_ta vs openalgo speed comparison (internal extra)"
    )
    parser.add_argument(
        "--json",
        default=str(DEFAULT_JSON),
        help="Write results to JSON file (default: artifacts/latest)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not write a JSON artifact",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="Bar counts to benchmark (default: 10000 100000)",
    )
    args = parser.parse_args()
    json_path = None if args.no_json else args.json
    run_comparison(args.sizes, json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
