#!/usr/bin/env python3
"""
Diff two benchmark-vs-openalgo artifacts and fail on per-kernel regressions.

``benchmarks/bench_vs_openalgo.py`` emits a schema_version 2 artifact with one
row per ``(indicator, size)`` pair. The aggregate ``summary.win_rate`` can rise
while an individual kernel quietly gets slower or drops out of the win column,
so this script joins a baseline artifact against a candidate artifact on
``(indicator, size)`` and reports the per-row movement.

Usage::

    python benchmarks/check_vs_openalgo_regression.py \\
        --baseline before.json --candidate after.json

    # Enforce the project acceptance rule (every row a strict win at every size)
    python benchmarks/check_vs_openalgo_regression.py \\
        --baseline before.json --candidate after.json --require-all-wins

Checks performed:

1. Per-kernel ``ferro_ta_us`` slowdown beyond ``--max-slowdown-pct``.
2. Outcome transitions (``ferro_ta_win`` -> ``tie``/``openalgo_win`` fails;
   the reverse is reported as an improvement).
3. Rows present in only one artifact (a kernel leaving the bench is a
   coverage regression).
4. With ``--require-all-wins``: every size's ``openalgo_wins_or_ties`` must be
   empty and every row's ``speedup`` must exceed ``--speedup-floor``.

Advisory-only signals (never fail the run): ``openalgo_us`` drift between the
two artifacts, recorded ``cv_pct`` noise, ``runtime``/``git``/``metadata``
mismatches, and super-linear scaling between the smallest and largest size.

Exit codes: ``0`` pass, ``1`` regression policy failed, ``2`` unusable input.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_INPUT_ERROR = 2

TIE_EPSILON = 0.05
DEFAULT_SPEEDUP_FLOOR = 1.0 + TIE_EPSILON
DEFAULT_MAX_SLOWDOWN_PCT = 25.0
DEFAULT_NOISE_DRIFT_PCT = 20.0
DEFAULT_NONLINEARITY_FACTOR = 2.0
DEFAULT_TOP = 20
MIN_ROWS = 1
WIN = "ferro_ta_win"
LOSS = "openalgo_win"
TIE = "tie"
RUNTIME_KEYS = ("python_version", "platform", "cpu_model", "machine")


class ArtifactError(Exception):
    """Raised when an artifact is missing, malformed, or unusable."""


@dataclass(frozen=True)
class Row:
    """One ``(indicator, size)`` measurement from an artifact."""

    indicator: str
    size: int
    ferro_ta_us: float
    openalgo_us: float
    speedup: float
    outcome: str
    cv_pct: float


@dataclass(frozen=True)
class Artifact:
    """A parsed benchmark artifact plus its provenance fields."""

    path: Path
    rows: dict[tuple[str, int], Row]
    sizes: tuple[int, ...]
    runtime: dict[str, Any]
    git: dict[str, Any]
    n_runs: int
    openalgo_wins_or_ties: dict[int, tuple[str, ...]]


@dataclass(frozen=True)
class Delta:
    """The baseline-to-candidate movement for one joined row."""

    key: tuple[str, int]
    baseline: Row
    candidate: Row

    @property
    def label(self) -> str:
        return f"{self.key[0]}@{self.key[1]}"

    @property
    def ferro_ta_change_pct(self) -> float:
        return _pct_change(self.baseline.ferro_ta_us, self.candidate.ferro_ta_us)

    @property
    def openalgo_change_pct(self) -> float:
        return _pct_change(self.baseline.openalgo_us, self.candidate.openalgo_us)

    @property
    def speedup_delta(self) -> float:
        return self.candidate.speedup - self.baseline.speedup

    @property
    def noise_band_pct(self) -> float:
        """Combined recorded run-to-run noise for the two measurements."""
        return self.baseline.cv_pct + self.candidate.cv_pct


def _pct_change(before: float, after: float) -> float:
    if before <= 0.0:
        return float("inf") if after > 0.0 else 0.0
    return (after - before) / before * 100.0


def _as_float(row: dict[str, Any], key: str, where: str) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ArtifactError(f"{where}: field '{key}' is missing or not numeric")
    return float(value)


def _parse_row(raw: dict[str, Any], where: str) -> Row:
    indicator = raw.get("indicator")
    size = raw.get("size")
    if not isinstance(indicator, str) or not indicator:
        raise ArtifactError(f"{where}: row has no 'indicator'")
    if not isinstance(size, int) or isinstance(size, bool):
        raise ArtifactError(f"{where}: row '{indicator}' has no integer 'size'")

    stats = raw.get("ferro_ta_stats")
    cv_pct = 0.0
    if isinstance(stats, dict):
        raw_cv = stats.get("cv_pct")
        if isinstance(raw_cv, (int, float)) and not isinstance(raw_cv, bool):
            cv_pct = float(raw_cv)

    outcome = raw.get("outcome")
    if outcome not in {WIN, LOSS, TIE}:
        raise ArtifactError(
            f"{where}: row '{indicator}@{size}' has unknown outcome {outcome!r}"
        )

    return Row(
        indicator=indicator,
        size=size,
        ferro_ta_us=_as_float(raw, "ferro_ta_us", where),
        openalgo_us=_as_float(raw, "openalgo_us", where),
        speedup=_as_float(raw, "speedup", where),
        outcome=outcome,
        cv_pct=cv_pct,
    )


def load_artifact(path: Path) -> Artifact:
    """Read and validate one benchmark artifact."""
    if not path.exists():
        raise ArtifactError(f"benchmark file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"{path}: could not be read as JSON ({exc})") from exc
    if not isinstance(payload, dict):
        raise ArtifactError(f"{path}: expected a JSON object at the top level")
    if not payload.get("openalgo_available", False):
        raise ArtifactError(
            f"{path}: openalgo_available is false, so the bench skipped every "
            "comparison; install the optional openalgo extra and re-run"
        )

    raw_results = payload.get("results")
    if not isinstance(raw_results, list) or len(raw_results) < MIN_ROWS:
        raise ArtifactError(
            f"{path}: 'results' is empty or missing; the artifact is truncated"
        )

    rows: dict[tuple[str, int], Row] = {}
    for raw in raw_results:
        if not isinstance(raw, dict):
            raise ArtifactError(f"{path}: 'results' contains a non-object entry")
        row = _parse_row(raw, str(path))
        key = (row.indicator, row.size)
        if key in rows:
            raise ArtifactError(f"{path}: duplicate row for {row.indicator}@{row.size}")
        rows[key] = row

    summary = payload.get("summary")
    by_size = summary.get("by_size", []) if isinstance(summary, dict) else []
    wins_or_ties: dict[int, tuple[str, ...]] = {}
    if isinstance(by_size, list):
        for entry in by_size:
            if not isinstance(entry, dict) or not isinstance(entry.get("size"), int):
                continue
            names = entry.get("openalgo_wins_or_ties")
            wins_or_ties[int(entry["size"])] = (
                tuple(str(name) for name in names) if isinstance(names, list) else ()
            )

    runtime = payload.get("runtime")
    git = payload.get("git")
    n_runs = payload.get("n_runs")
    return Artifact(
        path=path,
        rows=rows,
        sizes=tuple(sorted({size for _, size in rows})),
        runtime=runtime if isinstance(runtime, dict) else {},
        git=git if isinstance(git, dict) else {},
        n_runs=int(n_runs)
        if isinstance(n_runs, int) and not isinstance(n_runs, bool)
        else 0,
        openalgo_wins_or_ties=wins_or_ties,
    )


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute median of empty sequence")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def provenance_warnings(baseline: Artifact, candidate: Artifact) -> list[str]:
    """Report artifact-level mismatches that make the comparison meaningless."""
    warnings: list[str] = []
    for key in RUNTIME_KEYS:
        before = baseline.runtime.get(key)
        after = candidate.runtime.get(key)
        if before != after:
            warnings.append(f"runtime.{key} differs: {before!r} -> {after!r}")
    if baseline.n_runs != candidate.n_runs:
        warnings.append(f"n_runs differs: {baseline.n_runs} -> {candidate.n_runs}")
    if baseline.git.get("commit") == candidate.git.get("commit"):
        warnings.append(
            f"both artifacts were produced at git commit "
            f"{str(baseline.git.get('commit'))[:12]}; nothing was rebuilt between runs"
        )
    for artifact in (baseline, candidate):
        if artifact.git.get("dirty"):
            warnings.append(f"{artifact.path.name} was produced from a dirty worktree")
    if baseline.sizes != candidate.sizes:
        warnings.append(
            f"benchmarked sizes differ: {list(baseline.sizes)} -> {list(candidate.sizes)}"
        )
    return warnings


def noise_warnings(deltas: list[Delta], drift_limit_pct: float) -> list[str]:
    """Qualify the comparison using competitor drift and recorded per-row noise."""
    warnings: list[str] = []
    if not deltas:
        return warnings

    drifts = [abs(delta.openalgo_change_pct) for delta in deltas]
    median_drift = _median(drifts)
    worst = max(deltas, key=lambda delta: abs(delta.openalgo_change_pct))
    if median_drift > drift_limit_pct:
        warnings.append(
            f"openalgo_us moved by a median of {median_drift:.1f}% between the two "
            f"artifacts (limit {drift_limit_pct:.1f}%). The machine was not quiet; "
            "treat every per-kernel delta below as unreliable and re-run."
        )
    warnings.append(
        f"openalgo_us drift: median {median_drift:.1f}%, worst "
        f"{worst.openalgo_change_pct:+.1f}% on {worst.label}"
    )

    noisy = [delta for delta in deltas if delta.noise_band_pct > drift_limit_pct]
    if noisy:
        loudest = max(noisy, key=lambda delta: delta.noise_band_pct)
        warnings.append(
            f"{len(noisy)} row(s) record a combined ferro_ta cv_pct above "
            f"{drift_limit_pct:.1f}% (loudest {loudest.label} at "
            f"{loudest.noise_band_pct:.1f}%); their deltas carry little signal"
        )
    return warnings


def scaling_warnings(artifact: Artifact, factor: float) -> list[str]:
    """Flag rows whose cost grows far faster than the data does."""
    if len(artifact.sizes) < 2:
        return []
    small, large = artifact.sizes[0], artifact.sizes[-1]
    size_ratio = large / small
    if size_ratio <= 0:
        return []

    warnings: list[str] = []
    for indicator in sorted({name for name, _ in artifact.rows}):
        low = artifact.rows.get((indicator, small))
        high = artifact.rows.get((indicator, large))
        if low is None or high is None or low.ferro_ta_us <= 0.0:
            continue
        time_ratio = high.ferro_ta_us / low.ferro_ta_us
        if time_ratio > size_ratio * factor:
            warnings.append(
                f"{indicator}: {time_ratio:.1f}x slower for {size_ratio:.0f}x the data "
                f"({low.ferro_ta_us:.1f}us -> {high.ferro_ta_us:.1f}us) - "
                "super-linear for an O(n) kernel, likely a bad measurement"
            )
    return warnings


def _format_delta(delta: Delta) -> str:
    return (
        f"{delta.label:<28} "
        f"{delta.baseline.ferro_ta_us:>11.2f}us -> {delta.candidate.ferro_ta_us:>11.2f}us "
        f"({delta.ferro_ta_change_pct:+7.1f}%)  "
        f"speedup {delta.baseline.speedup:>7.3f} -> {delta.candidate.speedup:>7.3f} "
        f"({delta.speedup_delta:+.3f})  "
        f"cv+-{delta.noise_band_pct:.1f}%"
    )


def _print_section(title: str, lines: list[str], top: int) -> None:
    if not lines:
        return
    print(f"\n{title}")
    for line in lines[:top]:
        print(f" - {line}")
    hidden = len(lines) - top
    if hidden > 0:
        print(f"   ... and {hidden} more (raise --top to see them)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diff two benchmark-vs-openalgo JSON artifacts and fail on per-kernel "
            "timing or outcome regressions."
        )
    )
    parser.add_argument(
        "--baseline",
        default="benchmarks/artifacts/latest/benchmark_vs_openalgo.json",
        help="Path to the reference artifact produced by benchmarks/bench_vs_openalgo.py",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Path to the new artifact to compare against the baseline",
    )
    parser.add_argument(
        "--max-slowdown-pct",
        type=float,
        default=DEFAULT_MAX_SLOWDOWN_PCT,
        help=(
            "Fail when a row's ferro_ta_us grows by more than this percentage "
            f"(default: {DEFAULT_MAX_SLOWDOWN_PCT:g}; small rows record cv_pct up to ~17%%)"
        ),
    )
    parser.add_argument(
        "--noise-drift-pct",
        type=float,
        default=DEFAULT_NOISE_DRIFT_PCT,
        help=(
            "Warn loudly when median openalgo_us drift or a row's recorded cv_pct "
            f"exceeds this percentage (default: {DEFAULT_NOISE_DRIFT_PCT:g})"
        ),
    )
    parser.add_argument(
        "--require-all-wins",
        action="store_true",
        help=(
            "Acceptance gate: fail unless every size's openalgo_wins_or_ties is empty "
            "and every candidate row's speedup exceeds --speedup-floor"
        ),
    )
    parser.add_argument(
        "--speedup-floor",
        type=float,
        default=DEFAULT_SPEEDUP_FLOOR,
        help=(
            "Per-row speedup floor used by --require-all-wins "
            f"(default: {DEFAULT_SPEEDUP_FLOOR:g}, the top of the tie band)"
        ),
    )
    parser.add_argument(
        "--nonlinearity-factor",
        type=float,
        default=DEFAULT_NONLINEARITY_FACTOR,
        help=(
            "Warn when a kernel's time ratio between the smallest and largest size "
            f"exceeds the size ratio by this factor (default: {DEFAULT_NONLINEARITY_FACTOR:g})"
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP,
        help=f"Maximum rows to print per section (default: {DEFAULT_TOP})",
    )
    return parser


def _collect_coverage_failures(
    baseline: Artifact, candidate: Artifact
) -> tuple[list[str], list[str]]:
    dropped = sorted(set(baseline.rows) - set(candidate.rows))
    added = sorted(set(candidate.rows) - set(baseline.rows))
    dropped_lines = [
        f"{indicator}@{size} is in the baseline but missing from the candidate"
        for indicator, size in dropped
    ]
    added_lines = [
        f"{indicator}@{size} is new in the candidate (no baseline to compare)"
        for indicator, size in added
    ]
    return dropped_lines, added_lines


def _acceptance_failures(candidate: Artifact, floor: float) -> list[str]:
    failures: list[str] = []
    for size in candidate.sizes:
        names = candidate.openalgo_wins_or_ties.get(size)
        if names is None:
            failures.append(f"size={size} has no summary.by_size entry to gate on")
        elif names:
            failures.append(
                f"size={size} still has {len(names)} openalgo win(s)/tie(s): "
                f"{', '.join(names)}"
            )
    below = [row for row in candidate.rows.values() if row.speedup <= floor]
    for row in sorted(below, key=lambda item: item.speedup):
        failures.append(
            f"{row.indicator}@{row.size} speedup {row.speedup:.3f} <= floor {floor:.3f} "
            f"({row.outcome})"
        )
    return failures


def main() -> int:
    args = build_parser().parse_args()

    try:
        baseline = load_artifact(Path(args.baseline))
        candidate = load_artifact(Path(args.candidate))
    except ArtifactError as exc:
        print(f"ERROR: {exc}")
        return EXIT_INPUT_ERROR

    print(f"baseline : {baseline.path} ({len(baseline.rows)} rows)")
    print(f"candidate: {candidate.path} ({len(candidate.rows)} rows)")

    shared = sorted(set(baseline.rows) & set(candidate.rows))
    deltas = [Delta(key, baseline.rows[key], candidate.rows[key]) for key in shared]

    for warning in provenance_warnings(baseline, candidate):
        print(f"WARNING: {warning}")
    for warning in noise_warnings(deltas, args.noise_drift_pct):
        print(f"WARNING: {warning}")
    for warning in scaling_warnings(candidate, args.nonlinearity_factor):
        print(f"WARNING: suspicious scaling: {warning}")

    slower = sorted(
        (d for d in deltas if d.ferro_ta_change_pct > args.max_slowdown_pct),
        key=lambda d: d.ferro_ta_change_pct,
        reverse=True,
    )
    faster = sorted(
        (d for d in deltas if d.ferro_ta_change_pct < -args.max_slowdown_pct),
        key=lambda d: d.ferro_ta_change_pct,
    )
    lost_wins = sorted(
        (d for d in deltas if d.baseline.outcome == WIN and d.candidate.outcome != WIN),
        key=lambda d: d.speedup_delta,
    )
    gained_wins = sorted(
        (d for d in deltas if d.baseline.outcome != WIN and d.candidate.outcome == WIN),
        key=lambda d: d.speedup_delta,
        reverse=True,
    )
    dropped_lines, added_lines = _collect_coverage_failures(baseline, candidate)

    print(
        f"\njoined {len(deltas)} row(s); "
        f"{len(slower)} slower than -{args.max_slowdown_pct:g}% tolerance, "
        f"{len(faster)} faster, {len(lost_wins)} lost win(s), "
        f"{len(gained_wins)} gained win(s)"
    )

    _print_section(
        f"REGRESSED ferro_ta_us (> {args.max_slowdown_pct:g}% slower), worst first:",
        [_format_delta(delta) for delta in slower],
        args.top,
    )
    _print_section(
        "LOST outcome (ferro_ta_win -> tie/openalgo_win), worst first:",
        [
            f"{delta.label:<28} {delta.baseline.outcome} -> {delta.candidate.outcome}  "
            f"speedup {delta.baseline.speedup:.3f} -> {delta.candidate.speedup:.3f}"
            for delta in lost_wins
        ],
        args.top,
    )
    _print_section("DROPPED rows (coverage regression):", dropped_lines, args.top)
    _print_section(
        f"IMPROVED ferro_ta_us (> {args.max_slowdown_pct:g}% faster), best first:",
        [_format_delta(delta) for delta in faster],
        args.top,
    )
    _print_section(
        "GAINED outcome (tie/openalgo_win -> ferro_ta_win), best first:",
        [
            f"{delta.label:<28} {delta.baseline.outcome} -> {delta.candidate.outcome}  "
            f"speedup {delta.baseline.speedup:.3f} -> {delta.candidate.speedup:.3f}"
            for delta in gained_wins
        ],
        args.top,
    )
    _print_section("NEW rows (informational):", added_lines, args.top)

    failures: list[str] = []
    failures += [
        f"{delta.label} ferro_ta_us {delta.baseline.ferro_ta_us:.2f}us -> "
        f"{delta.candidate.ferro_ta_us:.2f}us ({delta.ferro_ta_change_pct:+.1f}%) "
        f"exceeds tolerance {args.max_slowdown_pct:g}%"
        for delta in slower
    ]
    failures += [
        f"{delta.label} outcome {delta.baseline.outcome} -> {delta.candidate.outcome}"
        for delta in lost_wins
    ]
    failures += dropped_lines
    if args.require_all_wins:
        failures += _acceptance_failures(candidate, args.speedup_floor)

    if failures:
        print("\nFAILED openalgo regression policy:")
        for failure in failures[: args.top]:
            print(f" - {failure}")
        hidden = len(failures) - args.top
        if hidden > 0:
            print(f"   ... and {hidden} more (raise --top to see them)")
        return EXIT_FAIL

    print("\nPASS openalgo regression policy.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
