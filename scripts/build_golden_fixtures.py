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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "crates" / "ferro_ta_ffi" / "ffi_spec.json"
CSV_PATH = ROOT / "tests" / "fixtures" / "ohlcv_daily.csv"
OUT_PATH = ROOT / "tests" / "fixtures" / "golden" / "ffi_golden.json"
LIB_DIR = ROOT / "target" / "release"

BARS = 120
SIGNIFICANT_DIGITS = 15

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
}

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


def load_library() -> ctypes.CDLL:
    for name in ("libferro_ta_ffi.dylib", "libferro_ta_ffi.so", "ferro_ta_ffi.dll"):
        path = LIB_DIR / name
        if path.exists():
            return ctypes.CDLL(str(path))
    sys.exit("native library not found; run: cargo build -p ferro_ta_ffi --release")


def load_inputs() -> dict[str, list[float]]:
    with CSV_PATH.open() as fh:
        rows = list(csv.DictReader(fh))[:BARS]
    cols = {
        key: [float(r[key]) for r in rows]
        for key in ("open", "high", "low", "close", "volume")
    }
    # Variable-period input for MAVP: deterministic cycle through 2..10.
    cols["periods"] = [float(2 + i % 9) for i in range(len(rows))]
    return cols


def param_value(param: dict) -> float | int:
    kind, name = param["kind"], param["name"]
    if kind == "period":
        return max(PERIOD_VALUES.get(name, DEFAULT_PERIOD), param["min"])
    if kind == "count":
        return DEFAULT_COUNT
    if kind == "matype":
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


def call_function(lib: ctypes.CDLL, fn: dict, cols: dict[str, list[float]]) -> dict:
    n = len(cols["close"])
    input_names = [INPUT_COLUMNS[name] for name in fn["inputs"]]
    params = {p["name"]: param_value(p) for p in fn["params"]}
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


def run_stream(lib: ctypes.CDLL, stream: dict, cols: dict[str, list[float]]) -> dict:
    params = {p["name"]: param_value(p) for p in stream["params"]}
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


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text())
    lib = load_library()
    cols = load_inputs()
    golden = {
        "note": 'Generated by scripts/build_golden_fixtures.py. NaN is null; Inf is "inf"/"-inf".',
        "version": spec["version"],
        "bars": len(cols["close"]),
        "columns": {k: [encode(v) for v in vals] for k, vals in cols.items()},
        "functions": [call_function(lib, fn, cols) for fn in spec["functions"]],
        "streams": [run_stream(lib, s, cols) for s in spec["streams"]],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(golden, separators=(",", ":")) + "\n")
    all_nan = [
        case["fn"]
        for case in golden["functions"]
        if all(v is None for vals in case["outputs"].values() for v in vals)
    ]
    size_kb = OUT_PATH.stat().st_size // 1024
    print(
        f"wrote {OUT_PATH.relative_to(ROOT)} ({len(golden['functions'])} functions, "
        f"{len(golden['streams'])} streams, {size_kb} KiB)"
    )
    if all_nan:
        print(f"warning: all-NaN output (weak coverage) for: {', '.join(all_nan)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
