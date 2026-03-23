from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

import ferro_ta as ft

try:
    from benchmarks.metadata import benchmark_metadata
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from metadata import benchmark_metadata


def _time_min(fn: Callable[[], object], rounds: int = 5) -> float:
    fn()
    samples: list[float] = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return min(samples)


def _stream_close(close: np.ndarray, factory: Callable[[], Any]) -> float:
    streamer = factory()
    last = np.nan
    for value in close:
        last = streamer.update(float(value))
    return float(last) if not np.isnan(last) else np.nan


def _stream_hlc(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    factory: Callable[[], Any],
) -> float:
    streamer = factory()
    last = np.nan
    for high_value, low_value, close_value in zip(high, low, close):
        last = streamer.update(float(high_value), float(low_value), float(close_value))
    return float(last) if not np.isnan(last) else np.nan


def _stream_hlcv(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    factory: Callable[[], Any],
) -> float:
    streamer = factory()
    last = np.nan
    for high_value, low_value, close_value, volume_value in zip(high, low, close, volume):
        last = streamer.update(
            float(high_value),
            float(low_value),
            float(close_value),
            float(volume_value),
        )
    return float(last) if not np.isnan(last) else np.nan


def run_streaming_benchmark(
    *,
    n_bars: int = 100_000,
    seed: int = 2026,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n_bars)).astype(np.float64)
    high = close + rng.uniform(0.1, 2.0, n_bars)
    low = close - rng.uniform(0.1, 2.0, n_bars)
    volume = rng.uniform(1_000.0, 100_000.0, n_bars)

    cases = [
        (
            "StreamingSMA",
            "close",
            lambda: _stream_close(close, lambda: ft.StreamingSMA(period=20)),
            lambda: ft.SMA(close, timeperiod=20),
        ),
        (
            "StreamingEMA",
            "close",
            lambda: _stream_close(close, lambda: ft.StreamingEMA(period=20)),
            lambda: ft.EMA(close, timeperiod=20),
        ),
        (
            "StreamingRSI",
            "close",
            lambda: _stream_close(close, lambda: ft.StreamingRSI(period=14)),
            lambda: ft.RSI(close, timeperiod=14),
        ),
        (
            "StreamingATR",
            "hlc",
            lambda: _stream_hlc(
                high,
                low,
                close,
                lambda: ft.StreamingATR(period=14),
            ),
            lambda: ft.ATR(high, low, close, timeperiod=14),
        ),
        (
            "StreamingVWAP",
            "hlcv",
            lambda: _stream_hlcv(
                high,
                low,
                close,
                volume,
                lambda: ft.StreamingVWAP(),
            ),
            lambda: ft.VWAP(high, low, close, volume),
        ),
    ]

    rows: list[dict[str, Any]] = []
    for name, input_kind, stream_fn, batch_fn in cases:
        stream_s = _time_min(stream_fn)
        batch_s = _time_min(batch_fn)
        rows.append(
            {
                "indicator": name,
                "inputs": input_kind,
                "stream_total_ms": round(stream_s * 1000.0, 4),
                "batch_total_ms": round(batch_s * 1000.0, 4),
                "stream_ns_per_update": round(stream_s * 1e9 / n_bars, 2),
                "batch_ns_per_bar": round(batch_s * 1e9 / n_bars, 2),
                "updates_per_second": round(n_bars / stream_s, 2),
                "stream_over_batch_ratio": round(stream_s / batch_s, 4),
            }
        )

    return {
        "metadata": benchmark_metadata(
            "streaming",
            extra={
                "dataset": {
                    "n_bars": n_bars,
                    "seed": seed,
                }
            },
        ),
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark streaming indicator execution.")
    parser.add_argument("--bars", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()

    payload = run_streaming_benchmark(n_bars=args.bars, seed=args.seed)

    dataset = payload["metadata"]["dataset"]
    print(f"Streaming Benchmark: {dataset['n_bars']} bars")
    print("-" * 86)
    print(
        f"{'Indicator':<16} {'Stream (ms)':>12} {'Batch (ms)':>12} "
        f"{'ns/update':>12} {'upd/s':>12} {'ratio':>10}"
    )
    print("-" * 86)
    for row in payload["results"]:
        print(
            f"{row['indicator']:<16} {row['stream_total_ms']:12.2f} "
            f"{row['batch_total_ms']:12.2f} {row['stream_ns_per_update']:12.2f} "
            f"{row['updates_per_second']:12.1f} {row['stream_over_batch_ratio']:10.2f}"
        )

    if args.json_path:
        path = Path(args.json_path)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON results to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
