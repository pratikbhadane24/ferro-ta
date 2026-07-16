#!/usr/bin/env python3
"""Generate the Flutter (flutter_rust_bridge) API module from the WASM bindings.

The WASM crate (`wasm/src/lib.rs`) is the signature ground-truth for every
value-returning indicator: each free ``#[wasm_bindgen] pub fn`` already wraps a
``ferro_ta_core`` call. This script ports those free functions into plain-Rust
``pub fn`` wrappers that flutter_rust_bridge can scan, keeping the three
bindings (Python / WASM / Flutter) in lockstep from a single source.

Mapping applied
---------------
- ``&Float64Array`` / ``Float64Array``  -> ``Vec<f64>``     (core arg gets ``&``)
- ``&str``                              -> ``String``       (core arg gets ``&``)
- ``f64`` / ``usize`` / ``i32`` / ``bool`` -> unchanged (passed by value)
- return ``Float64Array`` / ``Vec<f64>``   -> ``Vec<f64>``
- return ``js_sys::Int8Array``             -> ``Vec<i8>``
- return ``js_sys::Int32Array``            -> ``Vec<i32>``
- return ``Array`` (N ``.push`` calls)     -> ``(Vec<f64>, .. * N)`` tuple
- return ``f64`` / ``usize`` / ``i32`` / ``bool`` -> unchanged

Struct / enum / stateful surfaces (``CommissionModel``, ``WasmStreamingVWAP``,
options enums, ``String``, ``Vec<Vec<f64>>``/``Vec<usize>``/``Vec<i64>``) are
intentionally skipped here — they need bespoke flutter_rust_bridge opaque-type
design and are tracked as follow-up. Every skip is reported.

Usage
-----
python3 scripts/build_flutter_bridge.py            # write the api module
python3 scripts/build_flutter_bridge.py --check     # fail if out of date
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WASM_SRC = ROOT / "wasm" / "src" / "lib.rs"
OUT_PATH = ROOT / "flutter" / "rust" / "src" / "api" / "indicators.rs"

# A top-level free function starts at column 0 with ``pub fn``; struct-impl
# methods in the WASM crate are always indented, so column-0 filtering cleanly
# separates the two.
FREE_FN_RE = re.compile(r"(?m)^pub fn\s+([A-Za-z0-9_]+)\s*\(")
CORE_CALL_RE = re.compile(r"ferro_ta_core::[A-Za-z0-9_:]+")

ARRAY_PARAM_TYPES = {"&Float64Array", "Float64Array"}

# Wrappers whose WASM body performs non-trivial marshalling (type conversion,
# struct flattening, Result handling) that a positional passthrough cannot
# reproduce — e.g. options greeks/pricing (struct results), backtest engines,
# crossover-signal indices, and array-of-array batch ops. These need a
# hand-written bridge wrapper and are tracked as follow-up; excluding them keeps
# the generated module clean and compiling. Verified empirically: every name
# here is one the Rust compiler rejected as a naive passthrough.
MANUAL_EXCLUDE: set[str] = {
    "aggregate_greeks_dense",
    "aggregate_time_bars",
    "american_price",
    "atm_index",
    "backtest_core",
    "backtest_ohlcv",
    "black_76_greeks",
    "black_76_price",
    "black_scholes_greeks",
    "black_scholes_price",
    "bottom_n_indices",
    "compute_performance_metrics",
    "continuous_bar_labels",
    "curve_summary",
    "digital_greeks",
    "digital_price",
    "drawdown_series",
    "dtw_distance",
    "early_exercise_premium",
    "expected_move",
    "extended_greeks",
    "half_kelly_fraction",
    "ht_trendmode",
    "implied_volatility",
    "kelly_fraction",
    "label_moneyness",
    "macd_crossover_signals",
    "make_chunk_ranges",
    "mark_session_boundaries",
    "model_greeks",
    "model_price",
    "model_theta",
    "monthly_contribution",
    "ohlcv_agg",
    "price_lower_bound",
    "price_upper_bound",
    "rolling_maxindex",
    "rolling_minindex",
    "select_strike_by_delta",
    "select_strike_by_offset",
    "signal_attribution",
    "sma_crossover_signals",
    "smile_metrics",
    "stochf",
    "strategy_payoff_dense",
    "strategy_value_grid",
    "supertrend",
    "top_n_indices",
    "vol_cone",
    "walk_forward_indices",
}

# Optional override while re-discovering the exclusion set after a core/wasm
# change (see scripts/discover_flutter_excludes note in the docstring).
_env_exclude = __import__("os").environ.get("FERRO_FLUTTER_EXCLUDE", "")
if _env_exclude:
    MANUAL_EXCLUDE |= {n.strip() for n in _env_exclude.split(",") if n.strip()}


@dataclass(frozen=True)
class Param:
    name: str
    dart_type: str  # transformed Rust type exposed to the bridge
    is_ref_at_core: bool  # whether the core call needs a leading ``&``


@dataclass(frozen=True)
class Wrapper:
    name: str
    params: tuple[Param, ...]
    ret: str  # transformed Rust return type
    core_path: str
    doc: str


def _match_braces(text: str, open_idx: int) -> int:
    """Return the index just past the ``}`` matching the ``{`` at ``open_idx``."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError("unbalanced braces")


def _parse_param(raw: str) -> Param | None:
    raw = raw.strip()
    if (
        not raw
        or raw.startswith("&self")
        or raw == "self"
        or raw.startswith("&mut self")
    ):
        return None
    if ":" not in raw:
        return None
    name, ty = (part.strip() for part in raw.split(":", 1))
    if ty in ARRAY_PARAM_TYPES:
        return Param(name, "Vec<f64>", is_ref_at_core=True)
    if ty == "&str":
        return Param(name, "String", is_ref_at_core=True)
    if ty in {"f64", "usize", "i32", "i64", "u8", "bool"}:
        return Param(name, ty, is_ref_at_core=False)
    # Unknown parameter type — signal the caller to skip this wrapper.
    return Param(name, f"__UNSUPPORTED__{ty}", is_ref_at_core=False)


def _transform_return(ret: str, body: str) -> str | None:
    ret = ret.strip()
    if ret in {"Float64Array", "Vec<f64>"}:
        return "Vec<f64>"
    if ret == "js_sys::Int8Array":
        return "Vec<i8>"
    if ret == "js_sys::Int32Array":
        return "Vec<i32>"
    if ret in {"f64", "usize", "i32", "i64", "u8", "bool"}:
        return ret
    if ret == "Array":
        arity = body.count(".push(")
        if arity < 2:
            return None
        return "(" + ", ".join(["Vec<f64>"] * arity) + ")"
    # CommissionModel, WasmStreamingVWAP, Self, enums, String, Vec<Vec<f64>>, ...
    return None


def parse_wrappers(text: str) -> tuple[list[Wrapper], list[tuple[str, str]]]:
    wrappers: list[Wrapper] = []
    skipped: list[tuple[str, str]] = []

    for m in FREE_FN_RE.finditer(text):
        name = m.group(1)
        if name in MANUAL_EXCLUDE:
            skipped.append(
                (name, "needs hand-written bridge (non-trivial marshalling)")
            )
            continue
        sig_start = m.start()
        paren_open = text.index("(", m.end() - 1)
        # Find the end of the parameter list (matching paren).
        depth = 0
        i = paren_open
        while i < len(text):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        paren_close = i
        params_raw = text[paren_open + 1 : paren_close]

        # Return type sits between ``)`` and the opening ``{`` of the body.
        brace_open = text.index("{", paren_close)
        between = text[paren_close + 1 : brace_open].strip()
        ret = "()"
        if between.startswith("->"):
            ret = between[2:].strip()

        body_end = _match_braces(text, brace_open)
        body = text[brace_open:body_end]

        # Leading doc comment (/// lines) immediately above the fn.
        doc = _leading_doc(text, sig_start)

        # Parse params.
        parts = _split_params(params_raw)
        params: list[Param] = []
        unsupported = None
        for part in parts:
            p = _parse_param(part)
            if p is None:
                continue
            if p.dart_type.startswith("__UNSUPPORTED__"):
                unsupported = p.dart_type.removeprefix("__UNSUPPORTED__")
                break
            params.append(p)
        if unsupported is not None:
            skipped.append((name, f"unsupported param type `{unsupported}`"))
            continue

        new_ret = _transform_return(ret, body)
        if new_ret is None:
            skipped.append((name, f"unsupported return type `{ret}`"))
            continue

        core_match = CORE_CALL_RE.search(body)
        if core_match is None:
            skipped.append((name, "no ferro_ta_core call found in body"))
            continue

        wrappers.append(
            Wrapper(
                name=name,
                params=tuple(params),
                ret=new_ret,
                core_path=core_match.group(0),
                doc=doc,
            )
        )

    wrappers.sort(key=lambda w: w.name)
    skipped.sort()
    return wrappers, skipped


def _split_params(params_raw: str) -> list[str]:
    """Split a parameter list on top-level commas (ignoring generics)."""
    parts: list[str] = []
    depth = 0
    current = ""
    for ch in params_raw:
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current)
    return parts


def _leading_doc(text: str, sig_start: int) -> str:
    """Collect contiguous ``///`` lines directly above ``sig_start``."""
    prefix = text[:sig_start].rstrip("\n")
    lines = prefix.split("\n")
    doc_lines: list[str] = []
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith("///"):
            doc_lines.append(stripped)
        elif stripped.startswith("#["):
            continue
        else:
            break
    return "\n".join(reversed(doc_lines))


def render(wrappers: list[Wrapper]) -> str:
    lines: list[str] = [
        "// @generated by scripts/build_flutter_bridge.py — DO NOT EDIT.",
        "// Regenerate with: python3 scripts/build_flutter_bridge.py",
        "//",
        "// Thin flutter_rust_bridge wrappers over `ferro_ta_core`, ported from the",
        "// WASM bindings so all language surfaces stay in lockstep.",
        "#![allow(clippy::too_many_arguments)]",
        "",
    ]
    for w in wrappers:
        if w.doc:
            lines.append(w.doc)
        param_sig = ", ".join(f"{p.name}: {p.dart_type}" for p in w.params)
        core_args = ", ".join(
            (f"&{p.name}" if p.is_ref_at_core else p.name) for p in w.params
        )
        lines.append(f"pub fn {w.name}({param_sig}) -> {w.ret} {{")
        lines.append(f"    {w.core_path}({core_args})")
        lines.append("}")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the generated file is stale (for CI).",
    )
    args = parser.parse_args()

    text = WASM_SRC.read_text(encoding="utf-8")
    wrappers, skipped = parse_wrappers(text)
    rendered = render(wrappers)

    print(f"Ported {len(wrappers)} wrappers; skipped {len(skipped)}.")
    if skipped:
        print("Skipped (need bespoke bridge design):")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")

    if args.check:
        current = OUT_PATH.read_text(encoding="utf-8") if OUT_PATH.exists() else ""
        if current != rendered:
            print(
                f"\nERROR: {OUT_PATH.relative_to(ROOT)} is stale. Run "
                "`python3 scripts/build_flutter_bridge.py`.",
                file=sys.stderr,
            )
            return 1
        print(f"OK: {OUT_PATH.relative_to(ROOT)} is up to date.")
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"Wrote {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
