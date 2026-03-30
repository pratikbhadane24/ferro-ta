"""
ferro_ta backtesting engine speed benchmark.

Measures throughput for single-asset, multi-asset, and analytics functions
across multiple bar sizes. Optional competitor comparison (vectorbt, backtrader)
is guarded behind try/except.

Usage:
    python benchmarks/bench_backtest.py
    python benchmarks/bench_backtest.py --sizes 10000 100000
    python benchmarks/bench_backtest.py --skip-competitors --json benchmarks/artifacts/bench_backtest_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from ferro_ta._ferro_ta import (
    backtest_core,
    backtest_multi_asset_core,
    backtest_ohlcv_core,
    compute_performance_metrics,
    kelly_fraction,
    monte_carlo_bootstrap,
    walk_forward_indices,
)

from ferro_ta.analysis.backtest import BacktestEngine

try:
    from benchmarks.metadata import benchmark_metadata
except ModuleNotFoundError:  # pragma: no cover
    from metadata import benchmark_metadata  # type: ignore[no-redef]

# Optional competitors -------------------------------------------------------
try:
    import vectorbt as vbt  # type: ignore[import]

    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None  # type: ignore[assignment]

try:
    import backtrader as bt  # type: ignore[import]

    BACKTRADER_AVAILABLE = True
except ImportError:
    BACKTRADER_AVAILABLE = False
    bt = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
N_WARMUP = 1
N_RUNS = 5
DEFAULT_SIZES = [10_000, 100_000, 1_000_000]
N_ASSETS = 50
N_SIMS = 500


# ---------------------------------------------------------------------------
# Timer helper
# ---------------------------------------------------------------------------


def _time_fn(
    fn, *args, n_warmup: int = N_WARMUP, n_runs: int = N_RUNS, **kwargs
) -> float:
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    times: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int, seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    close = np.cumprod(1 + rng.standard_normal(n) * 0.01) * 100.0
    high = close + rng.uniform(0.1, 1.5, n)
    low = close - rng.uniform(0.1, 1.5, n)
    open_ = close + rng.standard_normal(n) * 0.3
    return open_, high, low, close


def _make_signals(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = np.sign(rng.standard_normal(n))
    raw[raw == 0] = 1.0
    return raw.astype(np.float64)


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------


def bench_backtest_core_single(n: int) -> dict[str, Any]:
    _, _, _, close = _make_ohlcv(n)
    signals = _make_signals(n)

    t_ferro = _time_fn(backtest_core, close, signals)

    row: dict[str, Any] = {
        "n_bars": n,
        "ferro_ta_ms": round(t_ferro * 1000, 4),
        "ferro_ta_mbars_s": round(n / t_ferro / 1e6, 4),
    }

    if VECTORBT_AVAILABLE:
        import pandas as pd  # noqa: PLC0415

        close_s = pd.Series(close)
        sig_s = pd.Series(signals.astype(bool))

        def _vbt():
            pf = vbt.Portfolio.from_signals(close_s, sig_s, ~sig_s, freq="1D")
            return pf.total_return()

        t_vbt = _time_fn(_vbt)
        row["vectorbt_ms"] = round(t_vbt * 1000, 4)
        row["speedup_vs_vectorbt"] = round(t_vbt / t_ferro, 4)

    return row


def bench_backtest_ohlcv_core(n: int) -> dict[str, Any]:
    open_, high, low, close = _make_ohlcv(n)
    signals = _make_signals(n)

    t_ferro = _time_fn(
        backtest_ohlcv_core,
        open_,
        high,
        low,
        close,
        signals,
        fill_mode="market_open",
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
    )

    return {
        "n_bars": n,
        "ferro_ta_ms": round(t_ferro * 1000, 4),
        "ferro_ta_mbars_s": round(n / t_ferro / 1e6, 4),
    }


def bench_performance_metrics(n: int) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    returns = rng.standard_normal(n) * 0.01
    equity = np.cumprod(1 + returns)

    t_ferro = _time_fn(compute_performance_metrics, returns, equity)

    def _numpy_sharpe():
        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=1)
        _ = mean_r / std_r * np.sqrt(252)
        rolling_max = np.maximum.accumulate(equity)
        drawdown = (equity - rolling_max) / rolling_max
        _ = float(drawdown.min())

    t_numpy = _time_fn(_numpy_sharpe)

    return {
        "n_bars": n,
        "ferro_ta_ms": round(t_ferro * 1000, 4),
        "numpy_partial_ms": round(t_numpy * 1000, 4),
        "speedup_vs_numpy": round(t_numpy / t_ferro, 4),
        "note": "numpy_partial only computes sharpe+max_dd (2/23 metrics)",
    }


def bench_multi_asset(n: int, n_assets: int = N_ASSETS) -> dict[str, Any]:
    rng = np.random.default_rng(7)
    close_2d = np.ascontiguousarray(
        np.cumprod(1 + rng.standard_normal((n, n_assets)) * 0.01, axis=0) * 100.0
    )
    weights_2d = np.full((n, n_assets), 1.0 / n_assets)

    t_parallel = _time_fn(
        backtest_multi_asset_core, close_2d, weights_2d, parallel=True
    )
    t_serial = _time_fn(backtest_multi_asset_core, close_2d, weights_2d, parallel=False)

    def _numpy_loop():
        results = []
        for j in range(n_assets):
            col = np.ascontiguousarray(close_2d[:, j])
            sig = np.ones(n)
            _, _, sr, _ = backtest_core(col, sig)
            results.append(sr)
        return np.stack(results, axis=1)

    t_loop = _time_fn(_numpy_loop)

    return {
        "n_bars": n,
        "n_assets": n_assets,
        "parallel_ms": round(t_parallel * 1000, 4),
        "serial_ms": round(t_serial * 1000, 4),
        "loop_ms": round(t_loop * 1000, 4),
        "parallel_speedup_vs_loop": round(t_loop / t_parallel, 4),
        "parallel_speedup_vs_serial": round(t_serial / t_parallel, 4),
    }


def bench_monte_carlo(n: int, n_sims: int = N_SIMS) -> dict[str, Any]:
    rng = np.random.default_rng(3)
    returns = rng.standard_normal(n) * 0.01

    t_ferro = _time_fn(monte_carlo_bootstrap, returns, n_sims=n_sims, seed=42)

    def _numpy_mc():
        out = np.empty((n_sims, n))
        for i in range(n_sims):
            idx = np.random.choice(len(returns), size=len(returns), replace=True)
            out[i] = np.cumprod(1 + returns[idx])
        return out

    t_numpy = _time_fn(_numpy_mc)

    return {
        "n_bars": n,
        "n_sims": n_sims,
        "ferro_ta_ms": round(t_ferro * 1000, 4),
        "numpy_loop_ms": round(t_numpy * 1000, 4),
        "speedup_vs_numpy": round(t_numpy / t_ferro, 4),
    }


def bench_engine_pipeline(n: int) -> dict[str, Any]:
    _, high, low, open_ = _make_ohlcv(n)
    _, _, _, close = _make_ohlcv(n, seed=10)

    engine = (
        BacktestEngine()
        .with_commission(0.001)
        .with_slippage(5.0)
        .with_ohlcv(high=high, low=low, open_=open_)
        .with_stop_loss(0.02)
        .with_take_profit(0.04)
    )

    t_ferro = _time_fn(engine.run, close, "sma_crossover")

    return {
        "n_bars": n,
        "ferro_ta_ms": round(t_ferro * 1000, 4),
        "description": "Full pipeline: signals + OHLCV fill + 23 metrics + trades + drawdown",
    }


def bench_walk_forward_indices(n: int) -> dict[str, Any]:
    train = max(n // 5, 100)
    test = max(n // 20, 20)
    t = _time_fn(walk_forward_indices, n, train, test)
    return {
        "n_bars": n,
        "train_bars": train,
        "test_bars": test,
        "ferro_ta_us": round(t * 1_000_000, 4),
    }


def bench_kelly_fraction() -> dict[str, Any]:
    win_rates = np.linspace(0.3, 0.7, 1000)
    avg_wins = np.linspace(0.01, 0.05, 1000)
    avg_losses = np.linspace(0.005, 0.03, 1000)

    def _loop():
        for w, a, b in zip(win_rates, avg_wins, avg_losses):
            kelly_fraction(w, a, b)

    t = _time_fn(_loop)
    return {"n_calls": 1000, "ferro_ta_us": round(t * 1_000_000, 4)}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all(
    sizes: list[int],
    skip_competitors: bool,
    n_assets: int,
    n_sims: int,
) -> dict[str, Any]:
    results: dict[str, list[dict[str, Any]]] = {
        "backtest_core_single": [],
        "backtest_ohlcv_core": [],
        "performance_metrics": [],
        "multi_asset": [],
        "monte_carlo": [],
        "engine_full_pipeline": [],
        "walk_forward_indices": [],
    }

    for n in sizes:
        print(f"\n--- {n:,} bars ---")

        r = bench_backtest_core_single(n)
        results["backtest_core_single"].append(r)
        print(
            f"  backtest_core_single:  {r['ferro_ta_ms']:.2f} ms  ({r['ferro_ta_mbars_s']:.2f} M bars/s)"
        )

        r = bench_backtest_ohlcv_core(n)
        results["backtest_ohlcv_core"].append(r)
        print(
            f"  backtest_ohlcv_core:   {r['ferro_ta_ms']:.2f} ms  ({r['ferro_ta_mbars_s']:.2f} M bars/s)"
        )

        r = bench_performance_metrics(n)
        results["performance_metrics"].append(r)
        print(
            f"  performance_metrics:   {r['ferro_ta_ms']:.2f} ms  (numpy partial: {r['numpy_partial_ms']:.2f} ms, {r['speedup_vs_numpy']:.2f}x)"
        )

        r = bench_multi_asset(n, n_assets)
        results["multi_asset"].append(r)
        print(
            f"  multi_asset ({n_assets}):   parallel={r['parallel_ms']:.1f} ms  serial={r['serial_ms']:.1f} ms  loop={r['loop_ms']:.1f} ms  ({r['parallel_speedup_vs_loop']:.2f}x vs loop)"
        )

        r = bench_monte_carlo(n, n_sims)
        results["monte_carlo"].append(r)
        print(
            f"  monte_carlo ({n_sims} sims): {r['ferro_ta_ms']:.2f} ms  (numpy: {r['numpy_loop_ms']:.2f} ms, {r['speedup_vs_numpy']:.2f}x)"
        )

        r = bench_engine_pipeline(n)
        results["engine_full_pipeline"].append(r)
        print(f"  engine_full_pipeline:  {r['ferro_ta_ms']:.2f} ms")

        r = bench_walk_forward_indices(n)
        results["walk_forward_indices"].append(r)
        print(f"  walk_forward_indices:  {r['ferro_ta_us']:.1f} µs")

    kelly_row = bench_kelly_fraction()
    results["kelly_fraction"] = [kelly_row]
    print(f"\n  kelly_fraction (1k calls): {kelly_row['ferro_ta_us']:.1f} µs")

    return {
        "metadata": benchmark_metadata("backtest"),
        "results": results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark ferro-ta backtesting engine."
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        metavar="N",
        help="Bar counts to benchmark (default: 10000 100000 1000000)",
    )
    parser.add_argument(
        "--skip-competitors",
        action="store_true",
        help="Skip optional competitor benchmarks",
    )
    parser.add_argument(
        "--assets",
        type=int,
        default=N_ASSETS,
        help="Number of assets for multi-asset benchmark",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=N_SIMS,
        help="Number of simulations for Monte Carlo benchmark",
    )
    parser.add_argument(
        "--json", dest="json_path", help="Write JSON results to this path"
    )
    args = parser.parse_args()

    print(
        f"ferro-ta backtest benchmark | sizes={args.sizes} | assets={args.assets} | sims={args.sims}"
    )
    print("=" * 72)

    payload = run_all(
        sizes=args.sizes,
        skip_competitors=args.skip_competitors,
        n_assets=args.assets,
        n_sims=args.sims,
    )

    if args.json_path:
        json_path = Path(args.json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON results to {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
