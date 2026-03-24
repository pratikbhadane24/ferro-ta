from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import ferro_ta

ROOT = Path(__file__).resolve().parents[2]
WASM_DIR = ROOT / "wasm"
PKG_JS = WASM_DIR / "pkg" / "ferro_ta_wasm.js"
SCRIPT = WASM_DIR / "conformance_node.js"


def _write_node_conformance_script(path: Path) -> None:
    path.write_text(
        """
const wasm = require("./pkg/ferro_ta_wasm.js");

function toArray(x) {
  return Array.from(x, (v) => (Number.isNaN(v) ? null : Number(v)));
}

const close = new Float64Array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.1, 45.42, 45.84, 46.08, 45.89, 46.03, 46.21, 46.02, 45.78]);
const high = new Float64Array([44.71, 44.5, 44.6, 44.09, 44.79, 45.2, 45.44, 45.73, 46.01, 46.44, 46.21, 46.39, 46.53, 46.3, 46.12]);
const low = new Float64Array([43.9, 43.8, 43.9, 43.2, 43.9, 44.2, 44.6, 44.8, 45.2, 45.5, 45.4, 45.5, 45.7, 45.6, 45.4]);
const volume = new Float64Array([1200, 1320, 1250, 1460, 1500, 1670, 1720, 1810, 1900, 2020, 1980, 2100, 2170, 2140, 2080]);

const payload = {
  sma: toArray(wasm.sma(close, 5)),
  ema: toArray(wasm.ema(close, 5)),
  wma: toArray(wasm.wma(close, 5)),
  rsi: toArray(wasm.rsi(close, 5)),
  adx: toArray(wasm.adx(high, low, close, 5)),
  mfi: toArray(wasm.mfi(high, low, close, volume, 5)),
};

process.stdout.write(JSON.stringify(payload));
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _run_node_conformance() -> dict[str, list[float | None]]:
    if shutil.which("node") is None:
        pytest.skip("node is required for wasm/node conformance test")
    if not PKG_JS.exists():
        pytest.skip(
            "wasm/pkg not found; run `wasm-pack build --target nodejs --out-dir pkg`"
        )

    _write_node_conformance_script(SCRIPT)
    try:
        out = subprocess.check_output(
            ["node", str(SCRIPT)],
            cwd=WASM_DIR,
            text=True,
        )
    finally:
        if SCRIPT.exists():
            SCRIPT.unlink()
    return json.loads(out)


def _to_jsonable(arr: np.ndarray) -> list[float | None]:
    vals = np.asarray(arr, dtype=np.float64)
    return [None if np.isnan(x) else float(x) for x in vals]


def _assert_close_with_null_nan(
    actual: list[float | None],
    expected: list[float | None],
    *,
    atol: float,
) -> None:
    assert len(actual) == len(expected)
    a = np.array([np.nan if v is None else float(v) for v in actual], dtype=np.float64)
    e = np.array(
        [np.nan if v is None else float(v) for v in expected], dtype=np.float64
    )
    np.testing.assert_allclose(a, e, atol=atol, rtol=0.0, equal_nan=True)


def test_wasm_node_matches_python_core_indicators() -> None:
    close = np.array(
        [
            44.34,
            44.09,
            44.15,
            43.61,
            44.33,
            44.83,
            45.10,
            45.42,
            45.84,
            46.08,
            45.89,
            46.03,
            46.21,
            46.02,
            45.78,
        ],
        dtype=np.float64,
    )
    high = np.array(
        [
            44.71,
            44.50,
            44.60,
            44.09,
            44.79,
            45.20,
            45.44,
            45.73,
            46.01,
            46.44,
            46.21,
            46.39,
            46.53,
            46.30,
            46.12,
        ],
        dtype=np.float64,
    )
    low = np.array(
        [
            43.90,
            43.80,
            43.90,
            43.20,
            43.90,
            44.20,
            44.60,
            44.80,
            45.20,
            45.50,
            45.40,
            45.50,
            45.70,
            45.60,
            45.40,
        ],
        dtype=np.float64,
    )
    volume = np.array(
        [
            1200.0,
            1320.0,
            1250.0,
            1460.0,
            1500.0,
            1670.0,
            1720.0,
            1810.0,
            1900.0,
            2020.0,
            1980.0,
            2100.0,
            2170.0,
            2140.0,
            2080.0,
        ],
        dtype=np.float64,
    )

    node_payload = _run_node_conformance()

    py_expected = {
        "sma": _to_jsonable(ferro_ta.SMA(close, 5)),
        "ema": _to_jsonable(ferro_ta.EMA(close, 5)),
        "wma": _to_jsonable(ferro_ta.WMA(close, 5)),
        "rsi": _to_jsonable(ferro_ta.RSI(close, 5)),
        "adx": _to_jsonable(ferro_ta.ADX(high, low, close, 5)),
        "mfi": _to_jsonable(ferro_ta.MFI(high, low, close, volume, 5)),
    }

    for name, expected in py_expected.items():
        assert name in node_payload
        _assert_close_with_null_nan(node_payload[name], expected, atol=1e-9)
