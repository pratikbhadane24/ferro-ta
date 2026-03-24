from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from benchmarks.metadata import benchmark_metadata
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from metadata import benchmark_metadata

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _profile_variant(
    *,
    label: str,
    maturin_args: list[str],
    price_bars: int,
    iv_bars: int,
    window: int,
) -> dict[str, Any]:
    _run([sys.executable, "-m", "maturin", "develop", "--release", *maturin_args])
    with tempfile.TemporaryDirectory(prefix=f"ferro_ta_{label}_") as tmp_dir:
        json_path = Path(tmp_dir) / "runtime_hotspots.json"
        _run(
            [
                sys.executable,
                "benchmarks/profile_runtime_hotspots.py",
                "--price-bars",
                str(price_bars),
                "--iv-bars",
                str(iv_bars),
                "--window",
                str(window),
                "--json",
                str(json_path),
            ]
        )
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    return payload


def run_simd_benchmark(
    *,
    price_bars: int = 20_000,
    iv_bars: int = 50_000,
    window: int = 252,
) -> dict[str, Any]:
    variants = [
        ("portable_release", []),
        ("simd_release", ["--features", "simd"]),
    ]
    reports = {
        label: _profile_variant(
            label=label,
            maturin_args=args,
            price_bars=price_bars,
            iv_bars=iv_bars,
            window=window,
        )
        for label, args in variants
    }

    portable_rows = {row["name"]: row for row in reports["portable_release"]["results"]}
    simd_rows = {row["name"]: row for row in reports["simd_release"]["results"]}

    comparison: list[dict[str, Any]] = []
    for name in sorted(portable_rows):
        portable = portable_rows[name]
        simd = simd_rows.get(name)
        if simd is None:
            continue
        portable_ms = float(portable["fast_ms"])
        simd_ms = float(simd["fast_ms"])
        comparison.append(
            {
                "name": name,
                "category": portable["category"],
                "portable_ms": round(portable_ms, 4),
                "simd_ms": round(simd_ms, 4),
                "speedup_simd_vs_portable": round(
                    portable_ms / simd_ms if simd_ms > 0.0 else float("inf"), 4
                ),
            }
        )

    comparison.sort(
        key=lambda row: float(row["speedup_simd_vs_portable"]), reverse=True
    )

    # Restore the default portable editable build so the workspace ends in the
    # distributable configuration.
    _run([sys.executable, "-m", "maturin", "develop", "--release"])

    return {
        "metadata": benchmark_metadata(
            "simd",
            extra={
                "dataset": {
                    "price_bars": price_bars,
                    "iv_bars": iv_bars,
                    "window": window,
                },
                "variants": [label for label, _ in variants],
            },
        ),
        "results": comparison,
        "reports": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark portable vs SIMD-enabled ferro-ta builds."
    )
    parser.add_argument("--price-bars", type=int, default=20_000)
    parser.add_argument("--iv-bars", type=int, default=50_000)
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()

    payload = run_simd_benchmark(
        price_bars=args.price_bars,
        iv_bars=args.iv_bars,
        window=args.window,
    )

    print(f"{'Case':<20} {'Portable (ms)':>14} {'SIMD (ms)':>12} {'SIMD speedup':>14}")
    print("-" * 64)
    for row in payload["results"]:
        print(
            f"{row['name']:<20} {row['portable_ms']:14.4f} "
            f"{row['simd_ms']:12.4f} {row['speedup_simd_vs_portable']:14.2f}x"
        )

    if args.json_path:
        path = Path(args.json_path)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON results to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
