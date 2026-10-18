"""Golden stream outputs match the golden batch outputs of the same function.

``tests/fixtures/golden/ffi_golden.json`` records every C ABI stream replayed
bar by bar and every batch function called once, on the same data and with
the same params. A stream named like a function (``add`` <-> ``ft_add``) with
identical params and inputs must satisfy the streaming parity contract against
it; a binding or generator that swaps params or inputs breaks this.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

GOLDEN_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "ffi_golden.json"
)
REL_TOL = 1e-9

GOLDEN = json.loads(GOLDEN_PATH.read_text())
FUNCTIONS = {case["fn"]: case for case in GOLDEN["functions"]}


def _paired_streams() -> list[tuple[dict, dict]]:
    pairs = []
    for stream in GOLDEN["streams"]:
        fn = FUNCTIONS.get("ft_" + stream["stream"])
        if (
            fn is not None
            and fn["params"] == stream["params"]
            and fn["inputs"] == stream["inputs"]
            and fn.get("dataset") == stream.get("dataset")
        ):
            pairs.append((stream, fn))
    return pairs


PAIRS = _paired_streams()


def _decode(value: object) -> float | None:
    if value is None:
        return None
    if value in ("inf", "-inf"):
        return math.inf if value == "inf" else -math.inf
    return float(value)  # type: ignore[arg-type]


def test_registry_streams_are_paired():
    names = {stream["stream"] for stream, _ in PAIRS}
    assert {"add", "sum", "rolling_maxindex", "avgprice"} <= names


@pytest.mark.parametrize("stream,fn", PAIRS, ids=[s["stream"] for s, _ in PAIRS])
def test_stream_golden_matches_function_golden(stream: dict, fn: dict):
    assert stream["outputs"].keys() == fn["outputs"].keys()
    for out, got_series in stream["outputs"].items():
        want_series = fn["outputs"][out]
        assert len(got_series) == len(want_series)
        for i, (got, want) in enumerate(zip(got_series, want_series)):
            if isinstance(want, int) and not isinstance(want, bool):
                assert got == want, f"{out}[{i}]: {got} != {want}"
                continue
            g, w = _decode(got), _decode(want)
            assert (g is None) == (w is None), f"{out}[{i}]: NaN mask {got} vs {want}"
            if g is not None and w is not None:
                assert abs(g - w) <= REL_TOL * max(1.0, abs(w)), (
                    f"{out}[{i}]: {g} != {w}"
                )
