from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from benchmarks.bench_batch import run_batch_benchmark
    from benchmarks.bench_simd import run_simd_benchmark
    from benchmarks.bench_streaming import run_streaming_benchmark
    from benchmarks.bench_vs_talib import run_comparison
    from benchmarks.metadata import benchmark_metadata, file_info
    from benchmarks.profile_runtime_hotspots import build_hotspot_report
    from benchmarks.test_benchmark_suite import (
        FIXTURE_PATH,
        INDICATOR_SUITE,
        _run_indicator,
    )
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from bench_batch import run_batch_benchmark
    from bench_simd import run_simd_benchmark
    from bench_streaming import run_streaming_benchmark
    from bench_vs_talib import run_comparison
    from metadata import benchmark_metadata, file_info
    from profile_runtime_hotspots import build_hotspot_report
    from test_benchmark_suite import FIXTURE_PATH, INDICATOR_SUITE, _run_indicator


def _time_min(fn, rounds: int = 5) -> float:
    fn()
    samples: list[float] = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return min(samples) * 1000.0


def build_indicator_latency_report(*, rounds: int = 5) -> dict[str, Any]:
    if not FIXTURE_PATH.exists():
        raise FileNotFoundError(
            f"Canonical fixture not found: {FIXTURE_PATH}. "
            "Run benchmarks/fixtures/generate_canonical.py first."
        )

    fixture = np.load(FIXTURE_PATH)
    ohlcv = {key: fixture[key] for key in fixture.files}

    rows: list[dict[str, Any]] = []
    for entry in INDICATOR_SUITE:
        elapsed_ms = _time_min(lambda entry=entry: _run_indicator(entry, ohlcv), rounds=rounds)
        rows.append(
            {
                "name": entry["name"],
                "inputs": entry["inputs"],
                "kwargs": entry["kwargs"],
                "elapsed_ms": round(elapsed_ms, 4),
            }
        )

    rows.sort(key=lambda row: float(row["elapsed_ms"]), reverse=True)
    return {
        "metadata": benchmark_metadata(
            "indicator_latency",
            fixtures=[FIXTURE_PATH],
            extra={
                "dataset": {
                    "fixture": str(FIXTURE_PATH),
                    "bars": len(ohlcv["close"]),
                    "rounds": rounds,
                }
            },
        ),
        "results": rows,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate reproducible performance baseline artifacts."
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/artifacts/latest",
        help="Directory where benchmark JSON artifacts are written",
    )
    parser.add_argument("--indicator-rounds", type=int, default=5)
    parser.add_argument("--batch-samples", type=int, default=100_000)
    parser.add_argument("--batch-series", type=int, default=100)
    parser.add_argument("--batch-seed", type=int, default=42)
    parser.add_argument("--streaming-bars", type=int, default=100_000)
    parser.add_argument("--streaming-seed", type=int, default=2026)
    parser.add_argument("--price-bars", type=int, default=20_000)
    parser.add_argument("--iv-bars", type=int, default=50_000)
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument(
        "--skip-simd",
        action="store_true",
        help="Skip portable-vs-SIMD comparison",
    )
    parser.add_argument(
        "--talib-sizes",
        type=int,
        nargs="+",
        default=[10_000, 100_000],
        help="Bar counts used for the TA-Lib comparison suite",
    )
    parser.add_argument(
        "--skip-talib",
        action="store_true",
        help="Skip the TA-Lib comparison artifact",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}

    indicator_path = output_dir / "indicator_latency.json"
    _write_json(
        indicator_path,
        build_indicator_latency_report(rounds=args.indicator_rounds),
    )
    artifacts["indicator_latency"] = str(indicator_path)

    batch_path = output_dir / "batch.json"
    _write_json(
        batch_path,
        run_batch_benchmark(
            n_samples=args.batch_samples,
            n_series=args.batch_series,
            seed=args.batch_seed,
        ),
    )
    artifacts["batch"] = str(batch_path)

    streaming_path = output_dir / "streaming.json"
    _write_json(
        streaming_path,
        run_streaming_benchmark(
            n_bars=args.streaming_bars,
            seed=args.streaming_seed,
        ),
    )
    artifacts["streaming"] = str(streaming_path)

    hotspot_path = output_dir / "runtime_hotspots.json"
    _write_json(
        hotspot_path,
        build_hotspot_report(
            price_bars=args.price_bars,
            iv_bars=args.iv_bars,
            window=args.window,
        ),
    )
    artifacts["runtime_hotspots"] = str(hotspot_path)

    if not args.skip_simd:
        simd_path = output_dir / "simd.json"
        _write_json(
            simd_path,
            run_simd_benchmark(
                price_bars=args.price_bars,
                iv_bars=args.iv_bars,
                window=args.window,
            ),
        )
        artifacts["simd"] = str(simd_path)

    if not args.skip_talib:
        talib_path = output_dir / "benchmark_vs_talib.json"
        run_comparison(args.talib_sizes, str(talib_path))
        artifacts["benchmark_vs_talib"] = str(talib_path)

    wasm_path = output_dir / "wasm.json"
    if wasm_path.exists():
        artifacts["wasm"] = str(wasm_path)

    manifest = {
        "metadata": benchmark_metadata(
            "perf_contract",
            fixtures=[FIXTURE_PATH],
            extra={"output_dir": str(output_dir)},
        ),
        "artifacts": {
            name: file_info(path)
            for name, path in artifacts.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    print(f"Generated performance contract artifacts in {output_dir}")
    for name, path in artifacts.items():
        print(f" - {name}: {path}")
    print(f" - manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
