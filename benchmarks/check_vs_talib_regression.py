#!/usr/bin/env python3
"""
Validate benchmark-vs-TA-Lib results against guardrail thresholds.

This is intentionally conservative: it catches severe regressions and incomplete
benchmark outputs, without overfitting to one machine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_threshold_items(items: list[str]) -> dict[int, float]:
    thresholds: dict[int, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid threshold '{item}', expected SIZE=VALUE")
        size_s, value_s = item.split("=", 1)
        thresholds[int(size_s)] = float(value_s)
    return thresholds


def _percentile(values: list[float], q: float) -> float:
    """Return the q percentile using linear interpolation."""
    if not values:
        raise ValueError("Cannot compute percentile of empty sequence")
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)

    values = sorted(values)
    rank = (len(values) - 1) * (q / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check TA-Lib benchmark JSON against regression thresholds."
    )
    parser.add_argument(
        "--input",
        default="benchmark_vs_talib.json",
        help="Path to benchmark JSON produced by benchmarks/bench_vs_talib.py",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=10,
        help="Minimum benchmark rows required per size",
    )
    parser.add_argument(
        "--median-floor",
        action="append",
        default=["10000=0.35", "100000=0.35"],
        help="Required minimum median speedup per size, e.g. 100000=0.5 (repeatable)",
    )
    parser.add_argument(
        "--min-speedup-floor",
        action="append",
        default=["10000=0.10", "100000=0.10"],
        help="Hard minimum per-row speedup floor per size, e.g. 100000=0.1 (repeatable)",
    )
    parser.add_argument(
        "--tail-percentile",
        type=float,
        default=10.0,
        help="Tail percentile used for distribution-based slowdown checks (default: 10)",
    )
    parser.add_argument(
        "--tail-speedup-floor",
        action="append",
        default=["10000=0.20", "100000=0.20"],
        help="Required minimum tail percentile speedup per size, e.g. 100000=0.2 (repeatable)",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: benchmark file not found: {path}")
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("talib_available", False):
        print(
            "ERROR: TA-Lib was not available; cannot enforce TA-Lib regression policy."
        )
        return 1

    summary_by_size = {
        int(entry.get("size")): entry
        for entry in data.get("summary", {}).get("by_size", [])
        if entry.get("size") is not None
    }
    results_by_size: dict[int, list[dict[str, object]]] = {}
    for row in data.get("results", []):
        if "speedup" not in row or row.get("size") is None:
            continue
        size = int(row["size"])
        results_by_size.setdefault(size, []).append(row)

    median_floor = _parse_threshold_items(args.median_floor)
    min_speedup_floor = _parse_threshold_items(args.min_speedup_floor)
    tail_speedup_floor = _parse_threshold_items(args.tail_speedup_floor)
    required_sizes = sorted(
        set(median_floor) | set(min_speedup_floor) | set(tail_speedup_floor)
    )

    failures: list[str] = []
    for size in required_sizes:
        entry = summary_by_size.get(size)
        if entry is None:
            failures.append(f"missing summary for size={size}")
            continue
        rows_for_size = results_by_size.get(size, [])
        if not rows_for_size:
            failures.append(f"missing detailed rows for size={size}")
            continue

        rows = int(entry.get("rows", 0))
        med = float(entry.get("median_speedup", 0.0))
        min_s = float(entry.get("min_speedup", 0.0))
        speedups = [float(row["speedup"]) for row in rows_for_size]
        tail_s = _percentile(speedups, args.tail_percentile)
        print(
            "size="
            f"{size}: rows={rows}, median_speedup={med:.4f}, "
            f"p{args.tail_percentile:g}_speedup={tail_s:.4f}, min_speedup={min_s:.4f}"
        )

        if rows < args.min_rows:
            failures.append(f"size={size} rows {rows} < min_rows {args.min_rows}")
        if med < median_floor.get(size, float("-inf")):
            failures.append(
                f"size={size} median_speedup {med:.4f} < floor {median_floor[size]:.4f}"
            )
        if tail_s < tail_speedup_floor.get(size, float("-inf")):
            failures.append(
                "size="
                f"{size} p{args.tail_percentile:g}_speedup {tail_s:.4f} "
                f"< floor {tail_speedup_floor[size]:.4f}"
            )
        if min_s < min_speedup_floor.get(size, float("-inf")):
            failures.append(
                f"size={size} min_speedup {min_s:.4f} < floor {min_speedup_floor[size]:.4f}"
            )

    if failures:
        print("FAILED benchmark regression policy:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("PASS benchmark regression policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
