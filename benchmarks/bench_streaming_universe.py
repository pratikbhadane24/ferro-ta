"""ns/update of every registry streaming class through the Python bindings.

Every class generated from the streaming registry
(``crates/ferro_ta_core/src/streaming/registry.rs``) runs at its default
parameters over ``--bars`` bars of a seeded OHLCV random walk, twice:

* **commit** — ``update(*bar)`` (``closed=True``) for every bar;
* **preview** — ``update(*bar, closed=False)`` for every bar, on the final
  committed state (state is untouched, so every call sees full windows).

Each figure is the best of ``--runs`` fresh streams, in ns per call. Python
call overhead (argument parsing, float boxing, the result tuple) floors every
stream at a few hundred ns, so the 2 µs budget is judged on the Rust numbers
instead::

    cargo test -p ferro_ta_core --release --all-features --lib \\
        stream_ns_per_update -- --ignored --nocapture

Build the extension in release mode first (``maturin develop --release``).

Usage
-----
python benchmarks/bench_streaming_universe.py
python benchmarks/bench_streaming_universe.py --json out.json --filter vol
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import ferro_ta.data.streaming as streaming

DEFAULT_BARS = 10_000
DEFAULT_RUNS = 3
SEED = 20_240_601
START_PRICE = 100.0
DAILY_VOL = 0.012
MEAN_VOLUME = 1_000_000.0
# Smoothing span of the ``slow`` input (mirrors streaming/fixtures.rs).
SLOW_MEAN_BARS = 8

# Hand-written classes (and helpers) that predate the registry.
LEGACY = {
    "StreamingSMA",
    "StreamingEMA",
    "StreamingRSI",
    "StreamingATR",
    "StreamingBBands",
    "StreamingMACD",
    "StreamingStoch",
    "StreamingVWAP",
    "StreamingSupertrend",
    "StreamingCross",
    "StreamingStreak",
    "StreamingCooldown",
    "StreamingOptionGreeks",
    "Bar",
    "BarBuilder",
}

CLOSE_DERIVED = {"real", "real0", "value", "x", "asset", "a", "front", "position_size"}
OPEN_DERIVED = {"real1", "b", "benchmark", "next"}


def ohlcv_walk(n: int, seed: int = SEED) -> dict[str, list[float]]:
    """Seeded geometric random walk with consistent OHLC and log-normal volume."""
    rng = random.Random(seed)
    cols: dict[str, list[float]] = {
        k: [] for k in ("open", "high", "low", "close", "volume")
    }
    close = START_PRICE
    for _ in range(n):
        open_ = close * math.exp(rng.gauss(0.0, DAILY_VOL / 4))
        close = open_ * math.exp(rng.gauss(0.0, DAILY_VOL))
        high = max(open_, close) * math.exp(abs(rng.gauss(0.0, DAILY_VOL / 2)))
        low = min(open_, close) * math.exp(-abs(rng.gauss(0.0, DAILY_VOL / 2)))
        cols["open"].append(open_)
        cols["high"].append(high)
        cols["low"].append(low)
        cols["close"].append(close)
        cols["volume"].append(MEAN_VOLUME * math.exp(rng.gauss(0.0, 0.5)))
    return cols


def _trailing_mean(values: Sequence[float], k: int) -> list[float]:
    out = []
    for i in range(len(values)):
        window = values[max(0, i + 1 - k) : i + 1]
        out.append(sum(window) / len(window))
    return out


def input_columns(n: int) -> dict[str, list[float]]:
    """Every input name a registry stream takes, as an ``n``-bar column."""
    base = ohlcv_walk(n)
    close = base["close"]
    cols: dict[str, list[float]] = dict(base)
    cols.update({name: close for name in CLOSE_DERIVED})
    cols.update({name: base["open"] for name in OPEN_DERIVED})
    cols.update(
        {
            "fast": close,
            "slow": _trailing_mean(close, SLOW_MEAN_BARS),
            "series": [
                c + 3.0 * math.sin(2.1 * i) if (i // 40) % 3 == 2 else c
                for i, c in enumerate(close)
            ],
            "iv_series": [
                0.2 + 0.05 * math.sin(0.05 * i) + 0.03 * math.sin(0.31 * i)
                for i in range(n)
            ],
            "adx": [32.5 + 27.5 * math.sin(0.21 * i) for i in range(n)],
            "atr": [0.5 + 0.3 * math.sin(0.37 * i) for i in range(n)],
            "asset_returns": [
                0.01 * math.sin(0.7 * i) + 0.002 * math.cos(1.9 * i) for i in range(n)
            ],
            "benchmark_returns": [0.008 * math.cos(0.45 * i) for i in range(n)],
            "periods": [5.0 + i % 7 for i in range(n)],
            "next_weights": [min(1.0, i / 300) for i in range(n)],
            "funding_rate": [1e-4 * math.sin(i) for i in range(n)],
        }
    )
    return cols


def registry_classes() -> list[str]:
    return sorted(set(streaming.__all__) - LEGACY)


def update_inputs(cls: type) -> list[str]:
    params = inspect.signature(cls.update).parameters
    return [p for p in params if p not in ("self", "closed")]


def _time_loop(
    call: Callable[..., object], rows: list[tuple[float, ...]], closed: bool
) -> int:
    perf = time.perf_counter_ns
    if closed:
        start = perf()
        for row in rows:
            call(*row)
        return perf() - start
    start = perf()
    for row in rows:
        call(*row, closed=False)
    return perf() - start


def bench_class(
    name: str, cols: dict[str, list[float]], runs: int
) -> dict[str, object]:
    cls = getattr(streaming, name)
    inputs = update_inputs(cls)
    missing = [i for i in inputs if i not in cols]
    if missing:
        raise SystemExit(f"{name}: no benchmark column for inputs {missing}")
    rows = list(zip(*(cols[i] for i in inputs)))
    commit = preview = math.inf
    for _ in range(runs):
        stream = cls()
        commit = min(commit, _time_loop(stream.update, rows, closed=True) / len(rows))
        preview = min(
            preview, _time_loop(stream.update, rows, closed=False) / len(rows)
        )
    return {
        "stream": name,
        "inputs": len(inputs),
        "commit_ns": commit,
        "preview_ns": preview,
    }


def call_floor_ns(rows: int, runs: int) -> float:
    """ns per call of a trivial builtin with one float argument (loop + call floor)."""
    values = [(float(i),) for i in range(rows)]
    best = math.inf
    for _ in range(runs):
        best = min(best, _time_loop(abs, values, closed=True) / rows)
    return best


def markdown(results: list[dict[str, object]]) -> str:
    lines = ["| stream | inputs | commit ns | preview ns |", "|---|---:|---:|---:|"]
    for r in results:
        lines.append(
            f"| {r['stream']} | {r['inputs']} | {r['commit_ns']:.0f} | {r['preview_ns']:.0f} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bars", type=int, default=DEFAULT_BARS)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--filter", default="", help="keep class names containing this")
    parser.add_argument(
        "--json", type=Path, help="also write results to this JSON file"
    )
    args = parser.parse_args(argv)
    if args.bars < 1 or args.runs < 1:
        parser.error("--bars and --runs must be >= 1")

    cols = input_columns(args.bars)
    names = [n for n in registry_classes() if args.filter.lower() in n.lower()]
    if not names:
        parser.error(f"no registry class matches {args.filter!r}")
    results = [bench_class(n, cols, args.runs) for n in names]
    results.sort(key=lambda r: r["commit_ns"], reverse=True)
    floor = call_floor_ns(args.bars, args.runs)

    print(markdown(results))
    print(
        f"\n{len(results)} streams, {args.bars} bars, best of {args.runs}; "
        f"Python loop + call floor ~{floor:.0f} ns."
    )
    if args.json:
        payload = {
            "bars": args.bars,
            "runs": args.runs,
            "call_floor_ns": floor,
            "python": sys.version.split()[0],
            "results": results,
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
