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
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from stream_names import is_registry_stream  # noqa: E402

GOLDEN_PATH = ROOT / "tests" / "fixtures" / "golden" / "ffi_golden.json"
SPEC_PATH = ROOT / "crates" / "ferro_ta_ffi" / "ffi_spec.json"
REL_TOL = 1e-9

GOLDEN = json.loads(GOLDEN_PATH.read_text())
FUNCTIONS = {case["fn"]: case for case in GOLDEN["functions"]}
SPEC = json.loads(SPEC_PATH.read_text())
SPEC_FUNCTIONS = {fn["name"]: fn for fn in SPEC["functions"]}
REGISTRY_STREAMS = [s for s in SPEC["streams"] if is_registry_stream(s["name"])]

# Registry streams deliberately without an `ft_<name>` batch peer.
NO_BATCH_PEER: frozenset[str] = frozenset()
# Registry streams that stream only some of their function's outputs:
# stream name -> the function output names kept, in order.
OUTPUT_SUBSET: dict[str, tuple[str, ...]] = {
    # chikou is close[i + displacement]: it needs future bars.
    "ichimoku": ("out_tenkan", "out_kijun", "out_senkou_a", "out_senkou_b"),
}


def _param_sig(params: list[dict]) -> list[tuple]:
    return [(p["name"], p["kind"], p.get("min")) for p in params]


@pytest.mark.parametrize("stream", REGISTRY_STREAMS, ids=lambda s: s["name"])
def test_registry_stream_signature_matches_function(stream: dict):
    """Params (name, kind, min), inputs and outputs equal ``ft_<name>``'s."""
    name = stream["name"]
    fn = SPEC_FUNCTIONS.get("ft_" + name)
    if name in NO_BATCH_PEER:
        assert fn is None, f"{name} has ft_{name}; drop it from NO_BATCH_PEER"
        return
    assert fn is not None, f"registry stream {name} has no ft_{name} batch function"
    assert _param_sig(stream["params"]) == _param_sig(fn["params"])
    assert stream["inputs"] == fn["inputs"]
    fn_outputs = [(o["name"], o["elem"]) for o in fn["outputs"]]
    if name in OUTPUT_SUBSET:
        kept = OUTPUT_SUBSET[name]
        fn_outputs = [o for o in fn_outputs if o[0] in kept]
        assert [o[0] for o in fn_outputs] == list(kept)
    assert [(o["name"], o["elem"]) for o in stream["outputs"]] == fn_outputs


def _paired_streams() -> list[tuple[dict, dict]]:
    """Golden stream / function cases replayed with the same params and data.

    A registry stream that fails to pair is an error (see
    ``test_every_registry_golden_stream_is_paired``), never a silent skip.
    """
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


def test_every_registry_golden_stream_is_paired():
    paired = {stream["stream"] for stream, _ in PAIRS}
    expected = {
        s["stream"]
        for s in GOLDEN["streams"]
        if is_registry_stream(s["stream"]) and s["stream"] not in NO_BATCH_PEER
    }
    assert expected, "no registry streams in the golden fixtures"
    assert expected - paired == set(), "registry streams without a golden peer"


def _decode(value: object) -> float | None:
    if value is None:
        return None
    if value in ("inf", "-inf"):
        return math.inf if value == "inf" else -math.inf
    return float(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("stream,fn", PAIRS, ids=[s["stream"] for s, _ in PAIRS])
def test_stream_golden_matches_function_golden(stream: dict, fn: dict):
    kept = OUTPUT_SUBSET.get(stream["stream"], tuple(fn["outputs"]))
    assert list(stream["outputs"]) == list(kept)
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
