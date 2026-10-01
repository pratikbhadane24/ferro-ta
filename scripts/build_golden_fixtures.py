#!/usr/bin/env python3
"""Generate language-neutral golden fixtures by calling the C ABI via ctypes.

Every export described in ``crates/ferro_ta_ffi/ffi_spec.json`` is called on a
slice of ``tests/fixtures/ohlcv_daily.csv`` with representative parameters,
and the results are written to ``tests/fixtures/golden/ffi_golden.json``.
Language bindings (Go today; C#, JVM, ... later) replay these cases through
their own wrappers, which verifies argument order, types and output order.

The chain of trust: ``ferro_ta_core`` is checked against TA-Lib by the Python
suite, the C ABI is checked bit-for-bit against the core by
``crates/ferro_ta_ffi/tests``, and each binding is checked against this file.

Regenerate after a core change that alters numbers (fixtures are not checked
for freshness in CI: last-ulp results may differ across CPU architectures)::

    cargo build -p ferro_ta_ffi --release
    python3 scripts/build_golden_fixtures.py
"""

from __future__ import annotations

import csv
import ctypes
import json
import math
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "crates" / "ferro_ta_ffi" / "ffi_spec.json"
CSV_PATH = ROOT / "tests" / "fixtures" / "ohlcv_daily.csv"
OUT_PATH = ROOT / "tests" / "fixtures" / "golden" / "ffi_golden.json"
LIB_DIR = ROOT / "target" / "release"

BARS = 120
SIGNIFICANT_DIGITS = 15
PATTERN_BARS = 1000
PATTERN_SEED = 7

# Multi-candle patterns that need exact bar sequences and never fire on the
# synthetic data. Their wrappers come from the same generated code path as
# the 43 patterns that do fire, which the golden replay does exercise.
KNOWN_SILENT_PATTERNS = frozenset(
    {
        "ft_cdl2crows",
        "ft_cdl3blackcrows",
        "ft_cdl3linestrike",
        "ft_cdl3starsinsouth",
        "ft_cdl3whitesoldiers",
        "ft_cdlabandonedbaby",
        "ft_cdlconcealbabyswall",
        "ft_cdlcounterattack",
        "ft_cdldarkcloudcover",
        "ft_cdlgapsidesidewhite",
        "ft_cdlinneck",
        "ft_cdlkicking",
        "ft_cdlkickingbylength",
        "ft_cdlmathold",
        "ft_cdlonneck",
        "ft_cdlpiercing",
        "ft_cdlrisefall3methods",
        "ft_cdlupsidegap2crows",
    }
)

# Per-function parameter values chosen so the output actually varies on the
# fixture data (a constant output cannot catch a wrapper bug).
PARAM_OVERRIDES: dict[str, dict[str, float]] = {
    # atr_pct_threshold is compared with the ATR/close *fraction* (~0.01-0.03).
    "ft_regime_combined": {"adx_threshold": 50.0, "atr_pct_threshold": 0.01},
    "ft_detect_breaks_cusum": {"threshold": 1.0, "slack": 0.001},
    "ft_rolling_variance_break": {"threshold": 1.5},
}

# Input argument name -> fixture column.
INPUT_COLUMNS = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "real": "close",
    "real0": "close",
    "real1": "open",
    "a": "close",
    "b": "open",
    "s1": "close",
    "s2": "open",
    "prices": "close",
    "value": "close",
    "periods": "periods",
    "iv_series": "iv",
    "front": "close",
    "next": "next",
    "next_weights": "weights",
    "adx": "oscillator",
    "atr": "range",
    "series": "returns",
    "asset": "returns",
    "benchmark": "benchmark_returns",
    "asset_returns": "returns",
    "benchmark_returns": "benchmark_returns",
    "x": "close",
    "fast": "close",
    "slow": "slow",
    "position_size": "position",
    "funding_rate": "funding",
    "values": "gappy",
}

# Period-like params whose relationships matter (fast < slow, etc.).
PERIOD_VALUES = {
    "fastperiod": 3,
    "slowperiod": 8,
    "signalperiod": 4,
    "fastk_period": 5,
    "slowk_period": 3,
    "slowd_period": 3,
    "fastd_period": 3,
    "minperiod": 2,
    "maxperiod": 10,
    "timeperiod1": 3,
    "timeperiod2": 6,
    "timeperiod3": 12,
    "tenkan_period": 3,
    "kijun_period": 8,
    "senkou_b_period": 15,
    "longperiod": 10,
    "shortperiod": 4,
    "jaw_period": 8,
    "teeth_period": 5,
    "lips_period": 3,
    "roc1": 3,
    "roc2": 4,
    "roc3": 5,
    "roc4": 6,
    "sma1": 3,
    "sma2": 3,
    "sma3": 3,
    "sma4": 4,
    "streakperiod": 2,
    "rankperiod": 20,
    "cycleperiod": 5,
    "wma_period": 5,
    "bins": 4,
    "window": 10,
    "short_window": 5,
    "long_window": 20,
}
DEFAULT_PERIOD = 5
DEFAULT_COUNT = 2

FLOAT_VALUES = {
    "nbdevup": 2.0,
    "nbdevdn": 2.0,
    "nbdev": 1.0,
    "multiplier": 3.0,
    "acceleration": 0.02,
    "maximum": 0.2,
    "vfactor": 0.7,
    "fastlimit": 0.5,
    "slowlimit": 0.05,
    "percent": 2.5,
    "annual": 252.0,
    "offset": 0.85,
    "sigma": 6.0,
    "scale": 1.0,
    "startvalue": 0.0,
    "offsetonreverse": 0.0,
    "accelerationinitlong": 0.02,
    "accelerationlong": 0.02,
    "accelerationmaxlong": 0.2,
    "accelerationinitshort": 0.02,
    "accelerationshort": 0.02,
    "accelerationmaxshort": 0.2,
    "trading_days": 252.0,
    "threshold": 25.0,
    "slack": 0.5,
    "adx_threshold": 25.0,
    "atr_pct_threshold": 1.0,
    "hedge": 0.8,
    "oversold": 30.0,
    "overbought": 70.0,
    # Scalar (option / futures / sizing) parameters.
    "spot": 100.0,
    "strike": 105.0,
    "forward": 101.0,
    "underlying": 100.0,
    "rate": 0.03,
    "dividend_yield": 0.01,
    "carry": 0.02,
    "time_to_expiry": 0.5,
    "volatility": 0.25,
    "iv": 0.25,
    "call_price": 6.0,
    "put_price": 5.5,
    "target_price": 7.5,
    "future": 101.5,
    "front_price": 100.0,
    "next_price": 101.0,
    "days_to_expiry": 30.0,
    "trading_days_per_year": 252.0,
    "initial_guess": 0.2,
    "tolerance": 1e-8,
    "win_rate": 0.55,
    "avg_win": 2.0,
    "avg_loss": 1.0,
}

COUNT_VALUES = {"max_iterations": 100}

C_SCALAR = {
    "int64_t": ctypes.c_int64,
    "int32_t": ctypes.c_int32,
    "double": ctypes.c_double,
}
C_ELEM = {
    "f64": ctypes.c_double,
    "i32": ctypes.c_int32,
    "i64": ctypes.c_int64,
    "i8": ctypes.c_int8,
}


def load_library(spec: dict) -> ctypes.CDLL:
    """Build the C ABI library and load it, refusing a stale build.

    Calling a library built from older sources with the current spec's argument
    lists would be undefined behaviour recorded as truth, so the build is
    always refreshed and its version checked against the spec.
    """
    build = subprocess.run(
        ["cargo", "build", "-q", "-p", "ferro_ta_ffi", "--release"], cwd=ROOT
    )
    if build.returncode != 0:
        sys.exit("cargo build -p ferro_ta_ffi --release failed")
    for name in ("libferro_ta_ffi.dylib", "libferro_ta_ffi.so", "ferro_ta_ffi.dll"):
        path = LIB_DIR / name
        if path.exists():
            lib = ctypes.CDLL(str(path))
            lib.ft_version.restype = ctypes.c_char_p
            version = lib.ft_version().decode()
            if version != spec["version"]:
                sys.exit(f"library version {version} != spec version {spec['version']}")
            return lib
    sys.exit(f"native library not found in {LIB_DIR}")


def load_inputs() -> dict[str, list[float]]:
    with CSV_PATH.open() as fh:
        rows = list(csv.DictReader(fh))[:BARS]
    cols = {
        key: [float(r[key]) for r in rows]
        for key in ("open", "high", "low", "close", "volume")
    }
    n = len(rows)
    close = cols["close"]
    # Deterministic derived columns for inputs that are not raw OHLCV.
    cols["periods"] = [float(2 + i % 9) for i in range(n)]  # MAVP periods in 2..10
    cols["iv"] = [0.2 + 0.05 * math.sin(i * 0.3) for i in range(n)]
    cols["oscillator"] = [50.0 + 40.0 * math.sin(i * 0.2) for i in range(n)]
    cols["range"] = [h - lo for h, lo in zip(cols["high"], cols["low"])]
    cols["returns"] = [0.0] + [close[i] / close[i - 1] - 1.0 for i in range(1, n)]
    cols["benchmark_returns"] = [
        0.6 * r + 0.001 * math.cos(i) for i, r in enumerate(cols["returns"])
    ]
    cols["next"] = [c * 1.01 for c in close]
    cols["weights"] = [min(1.0, max(0.0, (i - n / 3) / (n / 3))) for i in range(n)]
    cols["slow"] = [
        sum(close[max(0, i - 4) : i + 1]) / (i - max(0, i - 4) + 1) for i in range(n)
    ]
    cols["position"] = [1.0 if (i // 10) % 2 == 0 else -1.0 for i in range(n)]
    cols["funding"] = [0.0001 * math.sin(i * 0.5) for i in range(n)]
    cols["gappy"] = [math.nan if i % 7 in (0, 3) else c for i, c in enumerate(close)]
    return cols


def synthetic_ohlc(n: int, seed: int = PATTERN_SEED) -> dict[str, list[float]]:
    """Seeded OHLC with varied candle shapes (dojis, marubozus, long shadows,
    gaps, trend runs) so most candlestick patterns fire at least once.
    `random.Random` is deterministic across platforms and Python versions."""
    rng = random.Random(seed)
    cols: dict[str, list[float]] = {k: [] for k in ("open", "high", "low", "close")}
    price, trend = 100.0, 0.0
    for i in range(n):
        if i % 15 == 0:
            trend = rng.choice([-1.0, 0.0, 1.0]) * rng.uniform(0.3, 1.2)
        gap = rng.choice([0.0, 0.0, 0.0, rng.uniform(-2.0, 2.0)])
        open_ = price + gap
        body = rng.choice([0.0, 0.02, rng.uniform(0.1, 0.6), rng.uniform(0.8, 3.0)])
        bias = 0.15 * ((trend > 0) - (trend < 0))
        direction = 1.0 if rng.random() < 0.5 + bias else -1.0
        close = open_ + direction * body + trend * 0.3
        upper = rng.choice([0.0, rng.uniform(0.0, 0.3), rng.uniform(0.3, 2.5)])
        lower = rng.choice([0.0, rng.uniform(0.0, 0.3), rng.uniform(0.3, 2.5)])
        cols["open"].append(round(open_, 4))
        cols["close"].append(round(close, 4))
        cols["high"].append(round(max(open_, close) + upper, 4))
        cols["low"].append(round(min(open_, close) - lower, 4))
        price = close
    cols["volume"] = [1000.0 + 37.0 * (i % 11) for i in range(n)]
    return cols


def param_value(param: dict) -> float | int:
    kind, name = param["kind"], param["name"]
    if kind == "period":
        return max(PERIOD_VALUES.get(name, DEFAULT_PERIOD), param["min"])
    if kind == "count":
        return COUNT_VALUES.get(name, DEFAULT_COUNT)
    if kind in ("matype", "enum"):
        return 0
    if name not in FLOAT_VALUES:
        sys.exit(f"no golden value for float param {name!r}; add it to FLOAT_VALUES")
    return FLOAT_VALUES[name]


def encode(value: float) -> float | str | None:
    """JSON has no NaN/Inf: NaN -> null, +-Inf -> "inf"/"-inf"."""
    if math.isnan(value):
        return None
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def distinct_params(name: str, params_spec: list[dict]) -> dict[str, float | int]:
    """Representative params, made pairwise distinct within each type group.

    Equal values for two same-typed params would let a wrapper that swaps
    them pass the golden replay. Integers step up by one, floats by 1/8 of
    their magnitude, and MA types take distinct non-zero values.
    """
    overrides = PARAM_OVERRIDES.get(name, {})
    used: dict[str, set] = {"int": set(), "float": set(), "matype": set()}
    next_matype = 1
    values: dict[str, float | int] = {}
    for p in params_spec:
        kind = p["kind"]
        if kind == "matype":
            value: float | int = next_matype
            next_matype += 1
            group = "matype"
        else:
            value = overrides.get(p["name"], param_value(p))
            group = "float" if kind == "float" else "int"
            while value in used[group]:
                if group == "int":
                    value += 1
                else:
                    value = round(value + (abs(value) / 8 or 0.125), 10)
        used[group].add(value)
        values[p["name"]] = value
    return values


def call_function(lib: ctypes.CDLL, fn: dict, cols: dict[str, list[float]]) -> dict:
    n = len(cols["close"])
    input_names = [INPUT_COLUMNS[name] for name in fn["inputs"]]
    if len(set(input_names)) != len(input_names):
        sys.exit(f"{fn['name']}: two inputs share a column ({input_names})")
    params = distinct_params(fn["name"], fn["params"])
    in_arrays = [(ctypes.c_double * n)(*cols[col]) for col in input_names]
    out_arrays = [(C_ELEM[o["elem"]] * n)() for o in fn["outputs"]]
    args = [*in_arrays, ctypes.c_size_t(n)]
    args += [C_SCALAR[p["c_type"]](params[p["name"]]) for p in fn["params"]]
    args += out_arrays
    func = getattr(lib, fn["name"])
    func.restype = ctypes.c_int32
    status = func(*args)
    if status != 0:
        sys.exit(f"{fn['name']} returned status {status} for params {params}")
    outputs = {}
    for out, arr in zip(fn["outputs"], out_arrays):
        values = list(arr)
        outputs[out["name"]] = (
            [encode(v) for v in values] if out["elem"] == "f64" else values
        )
    return {
        "fn": fn["name"],
        "inputs": dict(zip(fn["inputs"], input_names)),
        "params": params,
        "outputs": outputs,
    }


def call_scalar(lib: ctypes.CDLL, fn: dict, variant: int, enums: dict) -> dict:
    params = distinct_params(fn["name"], fn["params"])
    # Enum params rotate through their values with an offset per position, so
    # two enum params never share an index and both cases differ.
    enum_params = [p for p in fn["params"] if p["kind"] == "enum"]
    for position, p in enumerate(enum_params):
        values = enums[p["enum"]]
        params[p["name"]] = values[(variant + position) % len(values)]["value"]
    outs = [C_ELEM[o["elem"]]() for o in fn["outputs"]]
    args = [C_SCALAR[p["c_type"]](params[p["name"]]) for p in fn["params"]]
    func = getattr(lib, fn["name"])
    func.restype = ctypes.c_int32
    status = func(*args, *[ctypes.byref(o) for o in outs])
    if status != 0:
        sys.exit(f"{fn['name']} returned status {status} for params {params}")
    return {
        "fn": fn["name"],
        "params": params,
        "outputs": {o["name"]: encode(v.value) for o, v in zip(fn["outputs"], outs)},
    }


def scalar_cases(lib: ctypes.CDLL, spec: dict) -> list[dict]:
    """One case per scalar; a second, with rotated enum values, if it has enums."""
    enums = {e["name"]: e["values"] for e in spec["enums"]}
    cases = []
    for fn in spec["scalars"]:
        cases.append(call_scalar(lib, fn, 0, enums))
        if any(p["kind"] == "enum" for p in fn["params"]):
            cases.append(call_scalar(lib, fn, 1, enums))
    return cases


def run_stream(lib: ctypes.CDLL, stream: dict, cols: dict[str, list[float]]) -> dict:
    params = distinct_params(stream["name"], stream["params"])
    input_names = [INPUT_COLUMNS[name] for name in stream["inputs"]]
    handle = ctypes.c_void_p()
    new = getattr(lib, stream["new_fn"])
    new.restype = ctypes.c_int32
    args = [C_SCALAR[p["c_type"]](params[p["name"]]) for p in stream["params"]]
    if new(*args, ctypes.byref(handle)) != 0:
        sys.exit(f"{stream['new_fn']} failed for {params}")
    update = getattr(lib, stream["update_fn"])
    update.restype = ctypes.c_int32
    free = getattr(lib, stream["free_fn"])
    series: dict[str, list] = {o["name"]: [] for o in stream["outputs"]}
    try:
        for i in range(len(cols["close"])):
            outs = [C_ELEM[o["elem"]]() for o in stream["outputs"]]
            bar = [ctypes.c_double(cols[col][i]) for col in input_names]
            if update(handle, *bar, *[ctypes.byref(o) for o in outs]) != 0:
                sys.exit(f"{stream['update_fn']} failed at bar {i}")
            for spec_out, value in zip(stream["outputs"], outs):
                v = value.value
                series[spec_out["name"]].append(
                    encode(v) if spec_out["elem"] == "f64" else v
                )
    finally:
        free(handle)
    return {
        "stream": stream["name"],
        "inputs": dict(zip(stream["inputs"], input_names)),
        "params": params,
        "outputs": series,
    }


def dataset_for(fn: dict) -> str:
    """Candlestick patterns replay on synthetic bars where most of them fire."""
    return "patterns" if fn["name"].startswith("ft_cdl") else "daily"


def weak_outputs(case: dict) -> list[str]:
    """Outputs that cannot detect a broken wrapper: all-NaN or constant."""
    return [
        name
        for name, values in case["outputs"].items()
        if all(v is None for v in values) or len(set(values)) <= 1
    ]


def validate_strength(cases: list[dict]) -> None:
    """Fail on any weak output, except documented never-firing patterns."""
    errors, now_firing = [], []
    for case in cases:
        label = case.get("fn") or case.get("stream")
        weak = weak_outputs(case)
        if label in KNOWN_SILENT_PATTERNS:
            if not weak:
                now_firing.append(label)
            continue
        errors += [f"{label}.{out}" for out in weak]
    if now_firing:
        print(
            f"note: remove from KNOWN_SILENT_PATTERNS (now firing): {', '.join(now_firing)}"
        )
    if errors:
        sys.exit(
            "weak golden outputs (all-NaN or constant) cannot catch wrapper bugs; "
            f"adjust inputs or PARAM_OVERRIDES: {', '.join(errors)}"
        )


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text())
    lib = load_library(spec)
    datasets = {"daily": load_inputs(), "patterns": synthetic_ohlc(PATTERN_BARS)}
    functions = [
        {
            **call_function(lib, fn, datasets[dataset_for(fn)]),
            "dataset": dataset_for(fn),
        }
        for fn in spec["functions"]
    ]
    streams = [
        {**run_stream(lib, s, datasets["daily"]), "dataset": "daily"}
        for s in spec["streams"]
    ]
    validate_strength(functions + streams)
    golden = {
        "note": 'Generated by scripts/build_golden_fixtures.py. NaN is null; Inf is "inf"/"-inf".',
        "version": spec["version"],
        "datasets": {
            name: {k: [encode(v) for v in vals] for k, vals in cols.items()}
            for name, cols in datasets.items()
        },
        "functions": functions,
        "scalars": scalar_cases(lib, spec),
        "streams": streams,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(golden, separators=(",", ":")) + "\n")
    size_kb = OUT_PATH.stat().st_size // 1024
    print(
        f"wrote {OUT_PATH.relative_to(ROOT)} ({len(functions)} functions, "
        f"{len(golden['scalars'])} scalar cases, {len(streams)} streams, {size_kb} KiB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
