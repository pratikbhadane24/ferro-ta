#!/usr/bin/env python3
"""
Generate the Speed Comparison markdown table from benchmarks/results.json.

Requires results from the full suite:
  pytest benchmarks/test_speed.py --benchmark-only --benchmark-json=benchmarks/results.json -v

Reads results.json and prints a markdown table: all indicators × all libraries.
Unsupported (indicator, library) pairs show N/A. Supported pairs missing benchmark
data show ERR (indicating the benchmark run was incomplete or failed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on path when run as script
_root = Path(__file__).resolve().parent.parent
if _root not in (Path(p).resolve() for p in sys.path):
    sys.path.insert(0, str(_root))

from benchmarks.wrapper_registry import (
    INDICATOR_CATEGORIES,
    is_supported,
)
from benchmarks.wrapper_registry import (
    LIBRARY_NAMES as LIBS,
)


def _all_indicators() -> list[str]:
    """All indicators in category order (matches test_speed parametrization)."""
    return [ind for cat in INDICATOR_CATEGORIES for ind in INDICATOR_CATEGORIES[cat]]


def main():
    p = Path(__file__).parent / "results.json"
    if not p.exists():
        print(
            "Run: pytest benchmarks/test_speed.py --benchmark-only --benchmark-json=benchmarks/results.json -v",
            file=sys.stderr,
        )
        sys.exit(1)
    raw = p.read_text().strip()
    if not raw:
        print(
            "results.json is empty. Run the full benchmark suite first.",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in results.json: {e}", file=sys.stderr)
        sys.exit(1)
    benchmarks = data.get("benchmarks", [])

    # Collect test_speed[Category/Indicator/library] -> median µs
    table: dict[str, dict[str, float]] = {}
    for b in benchmarks:
        name = b.get("name") or ""
        if "test_speed[" not in name:
            continue
        params = b.get("params") or {}
        ind = params.get("indicator")
        lib = params.get("library")
        if not ind or not lib or lib not in LIBS:
            continue
        median_sec = (b.get("stats") or {}).get("median")
        if median_sec is None:
            continue
        if ind not in table:
            table[ind] = {}
        table[ind][lib] = median_sec * 1e6  # to µs

    all_indicators = _all_indicators()
    if not all_indicators:
        print("No indicators from INDICATOR_CATEGORIES.", file=sys.stderr)
        sys.exit(1)

    # Header: Indicator | ferro_ta | talib | ...
    lib_header = " | ".join(LIBS)
    print(f"| Indicator | {lib_header} |")
    print("|-----------|" + "|".join(["--------:" for _ in LIBS]) + "|")

    for ind in all_indicators:
        row = table.get(ind, {})
        cells = []
        for lib in LIBS:
            if lib in row:
                cells.append(str(round(row[lib])))
            elif not is_supported(lib, ind):
                cells.append("N/A")
            else:
                cells.append("ERR")
        print(f"| {ind} | {' | '.join(cells)} |")

    print()
    print(
        "(Median time in µs, lower is better. N/A = unsupported pair. "
        "ERR = supported pair missing benchmark data. Source: results.json from full test_speed run.)"
    )


if __name__ == "__main__":
    main()
