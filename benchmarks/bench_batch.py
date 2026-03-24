from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import ferro_ta

try:
    from benchmarks.metadata import benchmark_metadata
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from metadata import benchmark_metadata


def _time_fn(fn, *args, rounds: int = 5, **kwargs) -> float:
    fn(*args, **kwargs)
    times: list[float] = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return min(times)


def run_batch_benchmark(
    *,
    n_samples: int = 100_000,
    n_series: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    close2d = rng.uniform(100.0, 200.0, (n_samples, n_series))
    high2d = close2d + rng.uniform(0.1, 2.0, (n_samples, n_series))
    low2d = close2d - rng.uniform(0.1, 2.0, (n_samples, n_series))
    close1d = close2d[:, 0]
    high1d = high2d[:, 0]
    low1d = low2d[:, 0]

    batch_rows: list[dict[str, Any]] = []
    grouped_rows: list[dict[str, Any]] = []

    indicators = [
        (
            "SMA",
            lambda: ferro_ta.batch.batch_sma(close2d, timeperiod=14, parallel=True),
            lambda: ferro_ta.batch.batch_sma(close2d, timeperiod=14, parallel=False),
            lambda: [
                ferro_ta.SMA(close2d[:, j], timeperiod=14) for j in range(n_series)
            ],
        ),
        (
            "RSI",
            lambda: ferro_ta.batch.batch_rsi(close2d, timeperiod=14, parallel=True),
            lambda: ferro_ta.batch.batch_rsi(close2d, timeperiod=14, parallel=False),
            lambda: [
                ferro_ta.RSI(close2d[:, j], timeperiod=14) for j in range(n_series)
            ],
        ),
        (
            "ATR",
            lambda: ferro_ta.batch.batch_atr(
                high2d, low2d, close2d, timeperiod=14, parallel=True
            ),
            lambda: ferro_ta.batch.batch_atr(
                high2d, low2d, close2d, timeperiod=14, parallel=False
            ),
            lambda: [
                ferro_ta.ATR(high2d[:, j], low2d[:, j], close2d[:, j], timeperiod=14)
                for j in range(n_series)
            ],
        ),
        (
            "ADX",
            lambda: ferro_ta.batch.batch_adx(
                high2d, low2d, close2d, timeperiod=14, parallel=True
            ),
            lambda: ferro_ta.batch.batch_adx(
                high2d, low2d, close2d, timeperiod=14, parallel=False
            ),
            lambda: [
                ferro_ta.ADX(high2d[:, j], low2d[:, j], close2d[:, j], timeperiod=14)
                for j in range(n_series)
            ],
        ),
    ]

    for name, parallel_fn, sequential_fn, loop_fn in indicators:
        batch_parallel_s = _time_fn(parallel_fn)
        batch_sequential_s = _time_fn(sequential_fn)
        loop_s = _time_fn(loop_fn)
        batch_rows.append(
            {
                "indicator": name,
                "parallel_ms": round(batch_parallel_s * 1000, 4),
                "sequential_ms": round(batch_sequential_s * 1000, 4),
                "loop_ms": round(loop_s * 1000, 4),
                "parallel_speedup_vs_loop": round(loop_s / batch_parallel_s, 4),
                "sequential_speedup_vs_loop": round(loop_s / batch_sequential_s, 4),
            }
        )

    grouped_cases = [
        (
            "close_bundle_3",
            lambda: ferro_ta.batch.compute_many(
                [
                    ("SMA", {"timeperiod": 10}),
                    ("EMA", {"timeperiod": 12}),
                    ("RSI", {"timeperiod": 14}),
                ],
                close=close1d,
            ),
            lambda: (
                ferro_ta.SMA(close1d, timeperiod=10),
                ferro_ta.EMA(close1d, timeperiod=12),
                ferro_ta.RSI(close1d, timeperiod=14),
            ),
        ),
        (
            "hlc_bundle_3",
            lambda: ferro_ta.batch.compute_many(
                [
                    ("ATR", {"timeperiod": 14}),
                    ("ADX", {"timeperiod": 14}),
                    ("CCI", {"timeperiod": 14}),
                ],
                close=close1d,
                high=high1d,
                low=low1d,
            ),
            lambda: (
                ferro_ta.ATR(high1d, low1d, close1d, timeperiod=14),
                ferro_ta.ADX(high1d, low1d, close1d, timeperiod=14),
                ferro_ta.CCI(high1d, low1d, close1d, timeperiod=14),
            ),
        ),
    ]

    for name, grouped_fn, separate_fn in grouped_cases:
        grouped_s = _time_fn(grouped_fn)
        separate_s = _time_fn(separate_fn)
        grouped_rows.append(
            {
                "case": name,
                "grouped_ms": round(grouped_s * 1000, 4),
                "separate_ms": round(separate_s * 1000, 4),
                "speedup_vs_separate": round(separate_s / grouped_s, 4),
            }
        )

    return {
        "metadata": benchmark_metadata(
            "batch",
            extra={
                "dataset": {
                    "n_samples": n_samples,
                    "n_series": n_series,
                    "total_bars": n_samples * n_series,
                    "seed": seed,
                }
            },
        ),
        "results": batch_rows,
        "grouped_results": grouped_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark batch indicator execution.")
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--series", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()

    payload = run_batch_benchmark(
        n_samples=args.samples,
        n_series=args.series,
        seed=args.seed,
    )

    dataset = payload["metadata"]["dataset"]
    print(
        "Batch Benchmark: "
        f"{dataset['n_samples']} bars, {dataset['n_series']} series "
        f"(Total: {dataset['total_bars'] / 1e6:.1f} M bars)"
    )
    print("-" * 74)
    print(
        f"{'Indicator':<12} {'Parallel (ms)':>14} {'Sequential (ms)':>16} "
        f"{'Loop (ms)':>12} {'P speedup':>10}"
    )
    print("-" * 74)
    for row in payload["results"]:
        print(
            f"{row['indicator']:<12} {row['parallel_ms']:14.1f} "
            f"{row['sequential_ms']:16.1f} {row['loop_ms']:12.1f} "
            f"{row['parallel_speedup_vs_loop']:10.2f}x"
        )

    if payload["grouped_results"]:
        print("\nGrouped Multi-Indicator Calls")
        print("-" * 64)
        print(
            f"{'Case':<18} {'Grouped (ms)':>14} {'Separate (ms)':>16} {'Speedup':>12}"
        )
        print("-" * 64)
        for row in payload["grouped_results"]:
            print(
                f"{row['case']:<18} {row['grouped_ms']:14.1f} "
                f"{row['separate_ms']:16.1f} {row['speedup_vs_separate']:12.2f}x"
            )

    if args.json_path:
        json_path = Path(args.json_path)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON results to {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
