#!/usr/bin/env python3
"""
Validate hotspot benchmark JSON against conservative speedup floors.

This gate is intentionally lightweight: it checks that the optimized paths
remain faster than their bundled reference implementations and that all
expected cases were present in the report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_threshold_items(items: list[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid threshold '{item}', expected NAME=VALUE")
        name, value_s = item.split("=", 1)
        thresholds[name] = float(value_s)
    return thresholds


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check hotspot benchmark JSON against regression thresholds."
    )
    parser.add_argument(
        "--input",
        default="runtime_hotspots.json",
        help="Path to JSON produced by benchmarks/profile_runtime_hotspots.py",
    )
    parser.add_argument(
        "--min-speedup",
        action="append",
        default=[
            "CORREL=2.0",
            "BETA=2.0",
            "LINEARREG=2.0",
            "TSF=2.0",
            "iv_rank=1.1",
            "iv_percentile=1.1",
            "iv_zscore=1.05",
            "compute_many_close=0.85",
            "feature_matrix=0.80",
        ],
        help="Required minimum speedup per named case, e.g. CORREL=5.0 (repeatable)",
    )
    parser.add_argument(
        "--min-cases",
        type=int,
        default=9,
        help="Minimum number of benchmark rows expected in the report",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: hotspot benchmark file not found: {path}")
        return 1

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    if len(rows) < args.min_cases:
        print(
            f"ERROR: hotspot report contains {len(rows)} rows, expected at least {args.min_cases}"
        )
        return 1

    thresholds = _parse_threshold_items(args.min_speedup)
    rows_by_name = {str(row.get("name")): row for row in rows}
    failures: list[str] = []

    for name, floor in thresholds.items():
        row = rows_by_name.get(name)
        if row is None:
            failures.append(f"missing row for {name}")
            continue

        speedup = float(row.get("speedup_vs_reference", 0.0))
        fast_ms = float(row.get("fast_ms", 0.0))
        reference_ms = float(row.get("reference_ms", 0.0))
        print(
            f"{name}: fast_ms={fast_ms:.4f}, reference_ms={reference_ms:.4f}, "
            f"speedup={speedup:.4f}"
        )

        if fast_ms <= 0.0 or reference_ms <= 0.0:
            failures.append(f"{name} has non-positive timing values")
        if speedup < floor:
            failures.append(f"{name} speedup {speedup:.4f} < floor {floor:.4f}")

    if failures:
        print("FAILED hotspot regression policy:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("PASS hotspot regression policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
