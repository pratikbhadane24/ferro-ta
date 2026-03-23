from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

import ferro_ta as ft
from ferro_ta.analysis.features import feature_matrix
from ferro_ta.analysis.options import iv_percentile, iv_rank, iv_zscore
from ferro_ta.data.batch import compute_many

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
    return min(samples) * 1000.0


def _naive_correl(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=np.float64)
    for end in range(window - 1, len(x)):
        x_window = x[end + 1 - window : end + 1]
        y_window = y[end + 1 - window : end + 1]
        mean_x = float(np.sum(x_window)) / window
        mean_y = float(np.sum(y_window)) / window
        cov = float(np.sum((x_window - mean_x) * (y_window - mean_y)))
        std_x = float(np.sqrt(np.sum((x_window - mean_x) ** 2)))
        std_y = float(np.sqrt(np.sum((y_window - mean_y) ** 2)))
        denom = std_x * std_y
        out[end] = cov / denom if denom != 0.0 else np.nan
    return out


def _naive_beta(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=np.float64)
    for end in range(window, len(x)):
        start = end - window
        rx = np.array(
            [x[idx + 1] / x[idx] - 1.0 if x[idx] != 0.0 else np.nan for idx in range(start, end)],
            dtype=np.float64,
        )
        ry = np.array(
            [y[idx + 1] / y[idx] - 1.0 if y[idx] != 0.0 else np.nan for idx in range(start, end)],
            dtype=np.float64,
        )
        mean_x = float(np.sum(rx)) / window
        mean_y = float(np.sum(ry)) / window
        cov = float(np.sum((rx - mean_x) * (ry - mean_y))) / window
        var_x = float(np.sum((rx - mean_x) ** 2)) / window
        out[end] = cov / var_x if var_x != 0.0 else np.nan
    return out


def _naive_linearreg(series: np.ndarray, timeperiod: int, x_value: float) -> np.ndarray:
    out = np.full(len(series), np.nan, dtype=np.float64)
    xs = np.arange(timeperiod, dtype=np.float64)
    sum_x = float(np.sum(xs))
    sum_x2 = float(np.sum(xs * xs))
    for end in range(timeperiod - 1, len(series)):
        window = series[end + 1 - timeperiod : end + 1]
        sum_y = float(np.sum(window))
        sum_xy = float(np.sum(xs * window))
        denom = timeperiod * sum_x2 - sum_x * sum_x
        slope = (timeperiod * sum_xy - sum_x * sum_y) / denom if denom != 0.0 else 0.0
        intercept = (sum_y - slope * sum_x) / timeperiod
        out[end] = intercept + slope * x_value
    return out


def _old_iv_rank(iv: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(iv), np.nan, dtype=np.float64)
    for idx in range(window - 1, len(iv)):
        win = iv[idx - window + 1 : idx + 1]
        lower = float(np.nanmin(win))
        upper = float(np.nanmax(win))
        out[idx] = 0.0 if upper == lower else (iv[idx] - lower) / (upper - lower)
    return out


def _old_iv_percentile(iv: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(iv), np.nan, dtype=np.float64)
    for idx in range(window - 1, len(iv)):
        win = iv[idx - window + 1 : idx + 1]
        out[idx] = float(np.sum(win <= iv[idx])) / window
    return out


def _old_iv_zscore(iv: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(iv), np.nan, dtype=np.float64)
    for idx in range(window - 1, len(iv)):
        win = iv[idx - window + 1 : idx + 1]
        mean = float(np.nanmean(win))
        std = float(np.nanstd(win, ddof=0))
        out[idx] = np.nan if std == 0.0 else (iv[idx] - mean) / std
    return out


def build_hotspot_report(
    *,
    price_bars: int = 20_000,
    iv_bars: int = 50_000,
    window: int = 252,
) -> dict[str, Any]:
    rng = np.random.default_rng(2026)
    close = 100 + np.cumsum(rng.normal(0, 1, price_bars)).astype(np.float64)
    high = close + rng.uniform(0.1, 2.0, price_bars)
    low = close - rng.uniform(0.1, 2.0, price_bars)
    iv = rng.uniform(10.0, 40.0, iv_bars).astype(np.float64)
    ohlcv = {"close": close, "high": high, "low": low, "volume": np.full(price_bars, 1000.0)}

    rows = [
        (
            "rust_kernel",
            "CORREL",
            lambda: ft.CORREL(high, low, timeperiod=30),
            lambda: _naive_correl(high, low, 30),
        ),
        (
            "rust_kernel",
            "BETA",
            lambda: ft.BETA(high, low, timeperiod=5),
            lambda: _naive_beta(high, low, 5),
        ),
        (
            "rust_kernel",
            "LINEARREG",
            lambda: ft.LINEARREG(close, timeperiod=14),
            lambda: _naive_linearreg(close, 14, 13.0),
        ),
        (
            "rust_kernel",
            "TSF",
            lambda: ft.TSF(close, timeperiod=14),
            lambda: _naive_linearreg(close, 14, 14.0),
        ),
        (
            "python_analysis",
            "iv_rank",
            lambda: iv_rank(iv, window),
            lambda: _old_iv_rank(iv, window),
        ),
        (
            "python_analysis",
            "iv_percentile",
            lambda: iv_percentile(iv, window),
            lambda: _old_iv_percentile(iv, window),
        ),
        (
            "python_analysis",
            "iv_zscore",
            lambda: iv_zscore(iv, window),
            lambda: _old_iv_zscore(iv, window),
        ),
        (
            "ffi_grouping",
            "compute_many_close",
            lambda: compute_many(
                [
                    ("SMA", {"timeperiod": 10}),
                    ("EMA", {"timeperiod": 12}),
                    ("RSI", {"timeperiod": 14}),
                ],
                close=close,
            ),
            lambda: (
                ft.SMA(close, timeperiod=10),
                ft.EMA(close, timeperiod=12),
                ft.RSI(close, timeperiod=14),
            ),
        ),
        (
            "ffi_grouping",
            "feature_matrix",
            lambda: feature_matrix(
                ohlcv,
                [
                    ("SMA", {"timeperiod": 10}),
                    ("ATR", {"timeperiod": 14}),
                    ("ADX", {"timeperiod": 14}),
                ],
            ),
            lambda: {
                "SMA": ft.SMA(close, timeperiod=10),
                "ATR": ft.ATR(high, low, close, timeperiod=14),
                "ADX": ft.ADX(high, low, close, timeperiod=14),
            },
        ),
    ]

    results: list[dict[str, Any]] = []
    for category, name, fast_fn, reference_fn in rows:
        fast_ms = _time_min(fast_fn)
        reference_ms = _time_min(reference_fn, rounds=1)
        results.append(
            {
                "category": category,
                "name": name,
                "fast_ms": round(fast_ms, 4),
                "reference_ms": round(reference_ms, 4),
                "speedup_vs_reference": round(reference_ms / fast_ms, 4),
            }
        )

    results.sort(key=lambda row: row["fast_ms"], reverse=True)
    total_fast_ms = sum(float(row["fast_ms"]) for row in results) or 1.0
    for row in results:
        row["share_of_suite_pct"] = round(float(row["fast_ms"]) / total_fast_ms * 100.0, 2)

    return {
        "metadata": benchmark_metadata(
            "runtime_hotspots",
            extra={
                "dataset": {
                    "price_bars": price_bars,
                    "iv_bars": iv_bars,
                    "window": window,
                }
            },
        ),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile ferro-ta runtime hotspots.")
    parser.add_argument("--price-bars", type=int, default=20_000)
    parser.add_argument("--iv-bars", type=int, default=50_000)
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()

    payload = build_hotspot_report(
        price_bars=args.price_bars,
        iv_bars=args.iv_bars,
        window=args.window,
    )

    print(f"{'Category':<16} {'Case':<18} {'Fast (ms)':>10} {'Ref (ms)':>10} {'Speedup':>10}")
    print("-" * 70)
    for row in payload["results"]:
        print(
            f"{row['category']:<16} {row['name']:<18} {row['fast_ms']:10.2f} "
            f"{row['reference_ms']:10.2f} {row['speedup_vs_reference']:10.2f}x"
        )

    if args.json_path:
        path = Path(args.json_path)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON results to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
