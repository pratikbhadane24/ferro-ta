"""
ferro_ta vs openalgo speed comparison (internal bench extra only).

Measures per-call runtime on C-contiguous float64 OHLCV at 10k and 100k bars.
Skips the whole run when the optional comparison extra is not installed.

Measurement design (see benchmarks/README.md for the noise evidence):

* The **minimum** of the samples is reported. Every sample is the true cost
  plus non-negative interference, so the min is the closest estimate of that
  cost; the median averages interference in, which answers a different
  question (observed latency) than this benchmark asks.
* ferro_ta and openalgo are timed **interleaved** (A/B per sample, order
  alternating) so thermal or scheduling drift lands on both sides.
* Sample counts are per size, since wall clock scales with the sample count.
* Samples are collected in **blocks** with the inputs reallocated between
  them, and the whole measurement is repeated in several **processes**. The
  dominant residual noise here is per-process: a row's minimum is reproducible
  to well under 1% inside one process, yet the same row can sit in a mode
  1.3-2.3x away in the next one (VWMA at 100k is either ~200us or ~460us).
  Each side's cross-process minimum is reported.
* A row is only called a win or a loss when the gap exceeds both the tie band
  and the row's own measured dispersion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

try:
    from benchmarks.metadata import (
        benchmark_metadata,
        package_versions,
        write_json_artifact,
    )
    from benchmarks.wrapper_registry import (
        INDICATOR_DEFAULTS,
        REGISTRY,
        available_libraries,
        openalgo_overlap_names,
    )
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from metadata import (  # type: ignore[no-redef]
        benchmark_metadata,
        package_versions,
        write_json_artifact,
    )
    from wrapper_registry import (
        INDICATOR_DEFAULTS,
        REGISTRY,
        available_libraries,
        openalgo_overlap_names,
    )

_rng = np.random.default_rng(42)

N_WARMUP = 2
N_BLOCKS = 3
N_PROCESSES = 3
N_RUNS_SMALL = 51
N_RUNS_LARGE = 21
SMALL_SIZE_THRESHOLD = 20_000
DEFAULT_SIZES = [10_000, 100_000]
TIE_EPSILON = 0.05
REPORTED_METRIC = "min_us"
DEFAULT_JSON = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "latest"
    / "benchmark_vs_openalgo.json"
)


def _measured_runs_for(size: int) -> int:
    """Sample count for one size; small inputs are cheap enough to sample hard."""
    return N_RUNS_SMALL if size <= SMALL_SIZE_THRESHOLD else N_RUNS_LARGE


def pin_to_single_cpu() -> dict[str, Any]:
    """Pin this process to one CPU where the OS supports it.

    ``os.sched_setaffinity`` exists on Linux only. macOS (and Windows) expose no
    equivalent, so there we record why pinning was skipped instead of failing.
    """
    set_affinity = getattr(os, "sched_setaffinity", None)
    get_affinity = getattr(os, "sched_getaffinity", None)
    if set_affinity is None or get_affinity is None:
        return {
            "pinned": False,
            "reason": (
                f"os.sched_setaffinity is unavailable on {sys.platform}; "
                "run-to-run migration noise cannot be removed on this platform"
            ),
        }
    try:
        available = sorted(get_affinity(0))
        if not available:
            return {"pinned": False, "reason": "no CPUs in the affinity mask"}
        chosen = available[-1]
        set_affinity(0, {chosen})
    except OSError as exc:
        return {"pinned": False, "reason": f"sched_setaffinity failed: {exc}"}
    return {"pinned": True, "cpu": chosen, "candidates": len(available)}


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _min_gap_pct(minima_us: list[float]) -> float:
    """Uncertainty of a minimum estimate, in percent.

    ``cv_pct`` describes the raw samples, which one contaminated sample
    inflates without moving the min, and the full spread of independent minima
    is just as easily dominated by one bad replicate. For a min estimator what
    matters is whether the *lowest* value is corroborated, so this reports the
    gap between the two lowest minima: near zero when replicates agree, large
    when the best measurement stands alone.
    """
    if len(minima_us) < 2:
        return 0.0
    lowest, second = sorted(minima_us)[:2]
    if lowest <= 0.0:
        return 0.0
    return (second - lowest) / lowest * 100.0


def _summary_stats(
    samples_us: list[float], block_mins_us: list[float] | None = None
) -> dict[str, float]:
    if not samples_us:
        return {
            "median_us": 0.0,
            "mean_us": 0.0,
            "min_us": 0.0,
            "max_us": 0.0,
            "stddev_us": 0.0,
            "cv_pct": 0.0,
            "min_gap_pct": 0.0,
            "samples": 0.0,
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
        "min_gap_pct": round(_min_gap_pct(block_mins_us or []), 3),
        "samples": float(len(samples_us)),
    }


def _required_margin(
    ft_stats: dict[str, float], oa_stats: dict[str, float]
) -> tuple[float, float]:
    """Margin a speedup must clear before the row is called, and the dispersion.

    A row cannot honestly be called a win or a loss when its own measurement
    noise is larger than the gap it claims to show, so the tie band widens to
    the noisier of the two sides.
    """
    dispersion_pct = max(
        float(ft_stats.get("min_gap_pct", 0.0)),
        float(oa_stats.get("min_gap_pct", 0.0)),
    )
    return max(TIE_EPSILON, dispersion_pct / 100.0), dispersion_pct


def _outcome(speedup: float, margin: float = TIE_EPSILON) -> str:
    if speedup > 1.0 + margin:
        return "ferro_ta_win"
    if speedup < 1.0 - margin:
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
    inconclusive = [
        row["indicator"] for row in rows if row.get("outcome_inconclusive") is True
    ]
    dispersions = [float(row.get("dispersion_pct", 0.0)) for row in rows]
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
        "median_dispersion_pct": round(_median(dispersions), 3),
        "max_dispersion_pct": round(max(dispersions), 3),
        "inconclusive_rows": len(inconclusive),
        "inconclusive": inconclusive,
        "openalgo_wins_or_ties": [
            row["indicator"]
            for row in rows
            if row.get("outcome") in {"openalgo_win", "tie"}
        ],
    }


def _time_once_us(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1_000_000.0


def _interleaved_samples_us(fn_a, fn_b, n_runs: int) -> tuple[list[float], list[float]]:
    """Time two callables alternately so slow drift cancels between them.

    Timing all of A then all of B lets a thermal or scheduling excursion land
    entirely on one side; alternating splits it across both, and the
    within-sample order flips so neither side owns the cache-cold position.
    """
    for _ in range(N_WARMUP):
        fn_a()
        fn_b()

    a_us: list[float] = []
    b_us: list[float] = []
    for index in range(n_runs):
        if index % 2:
            b_us.append(_time_once_us(fn_b))
            a_us.append(_time_once_us(fn_a))
        else:
            a_us.append(_time_once_us(fn_a))
            b_us.append(_time_once_us(fn_b))
    return a_us, b_us


def _blocked_measurement(
    make_pair, n_runs: int, n_blocks: int
) -> tuple[list[float], list[float], dict[str, float], dict[str, float]]:
    """Collect interleaved samples in ``n_blocks`` blocks of fresh inputs.

    ``make_pair`` returns a fresh ``(fn_a, fn_b)`` bound to newly allocated
    inputs, so each block re-warms and re-allocates instead of every sample
    inheriting one allocation's page layout. Returns the two sample lists
    followed by their stats.
    """
    per_block = max(1, math.ceil(n_runs / n_blocks))
    a_us: list[float] = []
    b_us: list[float] = []
    a_block_mins: list[float] = []
    b_block_mins: list[float] = []
    for _ in range(n_blocks):
        fn_a, fn_b = make_pair()
        a_block, b_block = _interleaved_samples_us(fn_a, fn_b, per_block)
        a_us.extend(a_block)
        b_us.extend(b_block)
        a_block_mins.append(min(a_block))
        b_block_mins.append(min(b_block))
    return (
        a_us,
        b_us,
        _summary_stats(a_us, a_block_mins),
        _summary_stats(b_us, b_block_mins),
    )


def _python_peak_bytes(fn) -> int | None:
    try:
        tracemalloc.start()
        tracemalloc.reset_peak()
        fn()
        _, peak = tracemalloc.get_traced_memory()
        return int(peak)
    except Exception:
        return None
    finally:
        tracemalloc.stop()


def _throughput_m_bars_s(size: int, elapsed_us: float) -> float:
    if elapsed_us <= 0:
        return 0.0
    return (size / 1e6) / (elapsed_us / 1_000_000.0)


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


def _measure_row(
    name: str,
    size: int,
    make_data,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Measure one ``(indicator, size)`` pair and build its artifact row."""
    ft_fn = REGISTRY[("ferro_ta", name)]
    oa_fn = REGISTRY[("openalgo", name)]

    def _make_pair():
        data = make_data()

        def _run_ft(fn=ft_fn, data=data, kw=params):
            return fn(data, None, **kw)

        def _run_oa(fn=oa_fn, data=data, kw=params):
            return fn(data, None, **kw)

        return _run_ft, _run_oa

    n_runs = _measured_runs_for(size)
    ft_us_samples, oa_us_samples, ft_stats, oa_stats = _blocked_measurement(
        _make_pair, n_runs, N_BLOCKS
    )

    ft_us = float(ft_stats[REPORTED_METRIC])
    oa_us = float(oa_stats[REPORTED_METRIC])
    speedup = oa_us / ft_us if ft_us > 0 else float("inf")

    margin, dispersion_pct = _required_margin(ft_stats, oa_stats)
    outcome = _outcome(speedup, margin)
    naive_outcome = _outcome(speedup, TIE_EPSILON)
    inconclusive = outcome != naive_outcome

    alloc_ft, alloc_oa = _make_pair()
    return {
        "indicator": name,
        "size": size,
        "input_layout": {
            "dtype": "float64",
            "contiguous": True,
        },
        "measured_runs": len(ft_us_samples),
        "measurement_blocks": N_BLOCKS,
        "ferro_ta_us": round(ft_us, 4),
        "ferro_ta_ms": round(ft_us / 1000.0, 4),
        "ferro_ta_m_bars_s": round(_throughput_m_bars_s(size, ft_us), 2),
        "ferro_ta_runs_us": [round(sample, 4) for sample in ft_us_samples],
        "ferro_ta_stats": ft_stats,
        "openalgo_us": round(oa_us, 4),
        "openalgo_ms": round(oa_us / 1000.0, 4),
        "openalgo_m_bars_s": round(_throughput_m_bars_s(size, oa_us), 2),
        "openalgo_runs_us": [round(sample, 4) for sample in oa_us_samples],
        "openalgo_stats": oa_stats,
        "speedup": round(speedup, 4),
        "outcome": outcome,
        "dispersion_pct": round(dispersion_pct, 3),
        "required_margin_pct": round(margin * 100.0, 3),
        "outcome_at_tie_band": naive_outcome,
        "outcome_inconclusive": inconclusive,
        "python_peak_allocation_bytes": {
            "ferro_ta": _python_peak_bytes(alloc_ft),
            "openalgo": _python_peak_bytes(alloc_oa),
        },
    }


def _methodology(
    sizes: list[int], pinning: dict[str, Any], processes: int
) -> dict[str, Any]:
    return {
        "warmup_runs": N_WARMUP,
        "measured_runs_by_size": {
            str(size): _measured_runs_for(size) for size in sizes
        },
        "measurement_blocks": N_BLOCKS,
        "processes": processes,
        "process_notes": (
            "The dominant residual noise is per-process, not per-sample: a "
            "row's minimum is reproducible to well under 1% inside one process "
            "yet the same row can sit in a mode 1.3-2.3x away in the next "
            "(VWMA at 100k is either ~200us or ~460us, MACD either ~490us or "
            "~657us, each mode stable across every sample of the process that "
            "sees it). No within-process statistic can see that, so the "
            "harness re-executes itself in `processes` processes and reports "
            "each side's cross-process minimum; the gap between the two lowest "
            "per-process minima becomes dispersion_pct. Per-run samples and "
            "*_stats come from the process with the fastest ferro_ta minimum."
        ),
        "block_notes": (
            "Within a process, samples are split across measurement_blocks "
            "blocks with freshly allocated inputs and a fresh warmup, so the "
            "per-block minima give a within-process error bar."
        ),
        "reported_metric": REPORTED_METRIC,
        "estimator_notes": (
            "The reported statistic is the minimum, not the median: every "
            "sample is the true cost plus non-negative interference, so the "
            "min is the least-biased estimate while the median averages "
            "interference in. median_us, mean_us, stddev_us and cv_pct are "
            "still recorded per row for the observed-latency view."
        ),
        "interleaved": True,
        "interleave_notes": (
            "ferro_ta and openalgo are timed alternately within each sample, "
            "with the within-sample order flipping every iteration, so thermal "
            "and scheduling drift is shared instead of landing on one side."
        ),
        "cpu_pinning": pinning,
        "speedup_definition": f"openalgo_{REPORTED_METRIC} / ferro_ta_{REPORTED_METRIC}",
        "tie_band": f"{1.0 - TIE_EPSILON:.2f} to {1.0 + TIE_EPSILON:.2f}",
        "outcome_notes": (
            "outcome widens the tie band to the row's own dispersion_pct (the "
            "gap between the two lowest independent minima, noisier side). A "
            "row whose noise exceeds its claimed gap is reported as a tie and "
            "flagged with outcome_inconclusive; outcome_at_tie_band records "
            "what the fixed 5% band alone would have said."
        ),
        "noise_floor_notes": (
            "Under the older median-of-7 sequential design, two runs against a "
            "byte-identical binary moved ferro_ta_us by up to ~55% on one row "
            "and flipped six verdicts. Do not quote one row from one run; "
            "re-run and treat differences inside roughly +-15% as unresolved."
        ),
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
    }


def _merge_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold one row's measurements from several processes into one row.

    Each side's cross-process minimum is the best available estimate of its true
    cost, and the spread of the per-process minima is the error bar that a
    single process cannot see.
    """
    ft_mins = [float(row["ferro_ta_us"]) for row in rows]
    oa_mins = [float(row["openalgo_us"]) for row in rows]
    merged = dict(min(rows, key=lambda row: float(row["ferro_ta_us"])))

    ft_us = min(ft_mins)
    oa_us = min(oa_mins)
    speedup = oa_us / ft_us if ft_us > 0 else float("inf")
    dispersion_pct = max(_min_gap_pct(ft_mins), _min_gap_pct(oa_mins))
    margin = max(TIE_EPSILON, dispersion_pct / 100.0)
    outcome = _outcome(speedup, margin)
    naive_outcome = _outcome(speedup, TIE_EPSILON)

    size = int(merged["size"])
    merged.update(
        {
            "processes": len(rows),
            "process_ferro_ta_min_us": [round(value, 4) for value in ft_mins],
            "process_openalgo_min_us": [round(value, 4) for value in oa_mins],
            "measured_runs": sum(int(row.get("measured_runs", 0)) for row in rows),
            "ferro_ta_us": round(ft_us, 4),
            "ferro_ta_ms": round(ft_us / 1000.0, 4),
            "ferro_ta_m_bars_s": round(_throughput_m_bars_s(size, ft_us), 2),
            "openalgo_us": round(oa_us, 4),
            "openalgo_ms": round(oa_us / 1000.0, 4),
            "openalgo_m_bars_s": round(_throughput_m_bars_s(size, oa_us), 2),
            "speedup": round(speedup, 4),
            "outcome": outcome,
            "dispersion_pct": round(dispersion_pct, 3),
            "required_margin_pct": round(margin * 100.0, 3),
            "outcome_at_tie_band": naive_outcome,
            "outcome_inconclusive": outcome != naive_outcome,
        }
    )
    return merged


def _merge_artifacts(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge per-process artifacts, keeping only rows every process measured."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    order: list[tuple[str, int]] = []
    for payload in payloads:
        for row in payload.get("results", []):
            key = (str(row["indicator"]), int(row["size"]))
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(row)
    return [
        _merge_row(grouped[key]) for key in order if len(grouped[key]) == len(payloads)
    ]


def _child_command(sizes: list[int], cpu_pin: bool, out_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--processes",
        "1",
        "--json",
        str(out_path),
        "--sizes",
        *[str(size) for size in sizes],
    ]
    if not cpu_pin:
        command.append("--no-cpu-pin")
    return command


def run_repeated(
    sizes: list[int], json_path: str | None, cpu_pin: bool, processes: int
) -> list[dict[str, Any]]:
    """Measure in ``processes`` separate processes and merge by cross-process min."""
    payloads: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="bench_vs_openalgo_") as tmp_dir:
        for index in range(processes):
            out_path = Path(tmp_dir) / f"process_{index}.json"
            print(f"measurement process {index + 1}/{processes} ...", flush=True)
            completed = subprocess.run(
                _child_command(sizes, cpu_pin, out_path),
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0 or not out_path.exists():
                print(
                    f"ERROR: measurement process {index + 1} failed "
                    f"(exit {completed.returncode}):"
                )
                print((completed.stderr or completed.stdout or "").strip()[-2000:])
                return []
            payloads.append(json.loads(out_path.read_text(encoding="utf-8")))

    if not payloads or not payloads[0].get("openalgo_available", False):
        print("Note: openalgo extra is not installed. Skipping competitor comparison.")
        return []

    results = _merge_artifacts(payloads)
    names = [str(name) for name in payloads[0].get("indicators", [])]
    pinning = payloads[0].get("cpu_pinning", {})
    _print_results(results, sizes, names, pinning, processes)
    if json_path and results:
        _write_artifact(Path(json_path), sizes, names, results, pinning, processes)
    return results


def _write_artifact(
    out_path: Path,
    sizes: list[int],
    names: list[str],
    results: list[dict[str, Any]],
    pinning: dict[str, Any],
    processes: int,
) -> None:
    """Write the schema_version 3 artifact for one merged or single-process run."""
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
            "methodology": _methodology(sizes, pinning, processes),
            "packages": package_versions("numpy", "ferro-ta", "openalgo"),
        },
    )
    payload = {
        "schema_version": 3,
        "command": " ".join(["python", *sys.argv]),
        "n_warmup": N_WARMUP,
        "n_runs": max(_measured_runs_for(size) for size in sizes) * processes,
        "n_runs_by_size": {
            str(size): _measured_runs_for(size) * processes for size in sizes
        },
        "n_processes": processes,
        "reported_metric": REPORTED_METRIC,
        "interleaved": True,
        "cpu_pinning": pinning,
        "sizes": sizes,
        "openalgo_available": True,
        "indicators": names,
        "runtime": metadata["runtime"],
        "git": metadata["git"],
        "metadata": metadata,
        "summary": {
            "total_rows": len(results),
            "inconclusive_rows": sum(
                1 for row in results if row.get("outcome_inconclusive")
            ),
            "by_size": [_summary_for_size(results, size) for size in sizes],
        },
        "results": results,
    }
    write_json_artifact(out_path, payload)
    print(f"Results written to {out_path}")


def _print_results(
    results: list[dict[str, Any]],
    sizes: list[int],
    names: list[str],
    pinning: dict[str, Any],
    processes: int,
) -> None:
    """Print the comparison table and the caveats a reader needs with it."""
    col_label, col_size, col_ft, col_oa, col_speedup = 18, 10, 14, 14, 10
    runs_by_size = ", ".join(
        f"{size}:{_measured_runs_for(size) * processes}" for size in sizes
    )
    print(
        f"\nferro_ta vs openalgo — {REPORTED_METRIC} of interleaved A/B samples "
        f"after {N_WARMUP} warmups"
    )
    print(
        f"Samples per row: {runs_by_size} ({processes} process(es) x {N_BLOCKS} blocks)"
    )
    print(f"Sizes: {sizes}")
    print(f"Indicators: {len(names)}")
    if pinning.get("pinned"):
        print(f"CPU pinning: pinned to cpu {pinning.get('cpu')}")
    else:
        print(f"CPU pinning: off — {pinning.get('reason')}")
    print(
        "Note: one run cannot resolve differences inside roughly +-15%; rows "
        "marked '?' are ties because their own noise exceeds the gap."
    )
    print()

    header = (
        f"{'Indicator':<{col_label}} {'Size':<{col_size}} "
        f"{'ferro_ta(us)':<{col_ft}} {'openalgo(us)':<{col_oa}} "
        f"{'Speedup':<{col_speedup}}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        flag = " ?" if row.get("outcome_inconclusive") else ""
        print(
            f"{row['indicator']:<{col_label}} {row['size']:<{col_size}} "
            f"{row['ferro_ta_us']:<{col_ft}.1f} {row['openalgo_us']:<{col_oa}.1f} "
            f"{row['speedup']:<{col_speedup}.2f}x{flag}"
        )

    print()
    if not results:
        return
    wins = sum(1 for row in results if row.get("outcome") == "ferro_ta_win")
    total = len(results)
    unresolved = sum(1 for row in results if row.get("outcome_inconclusive"))
    print(f"Summary: ferro_ta ahead outside the tie band on {wins}/{total} rows.")
    if unresolved:
        print(
            f"{unresolved}/{total} row(s) marked '?': their own measurement noise "
            "exceeds the gap, so they are reported as ties."
        )
    losses_100k = [
        row["indicator"]
        for row in results
        if row.get("size") == 100_000 and row.get("outcome") == "openalgo_win"
    ]
    if losses_100k:
        print(f"100k losses (optimize later): {', '.join(losses_100k)}")
    print()


def run_comparison(
    sizes: list[int], json_path: str | None, cpu_pin: bool = True
) -> list[dict[str, Any]]:
    """Measure every overlapping indicator once, in this process."""
    if not _openalgo_available():
        print("Note: openalgo extra is not installed. Skipping competitor comparison.")
        print("Install with: uv sync --extra comparison\n")
        return []

    names = openalgo_overlap_names()
    if not names:
        print("No overlapping indicator wrappers registered. Nothing to benchmark.")
        return []

    pinning = (
        pin_to_single_cpu() if cpu_pin else {"pinned": False, "reason": "--no-cpu-pin"}
    )
    full = _synthetic_ohlcv(max(sizes))
    results: list[dict[str, Any]] = []

    for name in names:
        params = dict(INDICATOR_DEFAULTS.get(name, {}))
        for size in sizes:

            def _make_data(source=full, length=size):
                return {
                    key: np.ascontiguousarray(value[:length], dtype=np.float64)
                    for key, value in source.items()
                }

            results.append(_measure_row(name, size, _make_data, params))

    _print_results(results, sizes, names, pinning, processes=1)
    if json_path:
        _write_artifact(Path(json_path), sizes, names, results, pinning, processes=1)
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
    parser.add_argument(
        "--processes",
        type=int,
        default=N_PROCESSES,
        help=(
            "Measure in this many separate processes and report the "
            "cross-process minimum per side. 1 measures in this process only. "
            f"(default: {N_PROCESSES}; the residual noise on this harness is "
            "per-process, so 1 is not reproducible)"
        ),
    )
    parser.add_argument(
        "--no-cpu-pin",
        action="store_true",
        help=(
            "Do not pin the process to a single CPU (Linux only; pinning is "
            "always skipped where os.sched_setaffinity does not exist)"
        ),
    )
    args = parser.parse_args()
    json_path = None if args.no_json else args.json
    cpu_pin = not args.no_cpu_pin
    if args.processes < 1:
        parser.error("--processes must be at least 1")
    if args.processes == 1:
        run_comparison(args.sizes, json_path, cpu_pin=cpu_pin)
    else:
        run_repeated(args.sizes, json_path, cpu_pin, args.processes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
