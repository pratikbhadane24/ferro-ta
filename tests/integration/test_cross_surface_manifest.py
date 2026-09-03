from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_api_manifest import (
    COVERAGE_RST_REL,
    build_manifest,
    canonical_key,
    render_coverage_rst,
    support_matrix_count_snippets,
)


def test_api_manifest_is_deterministic_and_current() -> None:
    manifest_path = ROOT / "docs" / "api_manifest.json"
    coverage_rst_path = ROOT / COVERAGE_RST_REL
    assert manifest_path.exists(), "docs/api_manifest.json is missing"
    assert coverage_rst_path.exists(), f"{COVERAGE_RST_REL} is missing"

    expected = build_manifest(ROOT, include_runtime_metadata=False)
    actual = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert actual == expected
    assert coverage_rst_path.read_text(encoding="utf-8") == render_coverage_rst(
        expected
    )


def test_canonical_key_normalizes_surface_names() -> None:
    assert canonical_key("SMA") == canonical_key("sma")
    assert canonical_key("PLUS_DI") == canonical_key("plus_di")
    assert canonical_key("CDLDOJI") == canonical_key("cdldoji")
    assert canonical_key("TRIX") == canonical_key("trix_indicator")
    assert (
        canonical_key("SIN")
        == canonical_key("transform_sin")
        == canonical_key("math_sin")
    )
    assert canonical_key("StreamingSMA") == canonical_key("WasmStreamingSMA")
    assert canonical_key("ADD") == canonical_key("math_add")


def test_normalized_coverage_includes_shared_indicators() -> None:
    manifest = build_manifest(ROOT, include_runtime_metadata=False)
    rows = {row["key"]: row for row in manifest["coverage"]["rows"]}

    for name in ("SMA", "RSI", "MACD", "BBANDS", "CDLDOJI", "PLUS_DI"):
        row = rows[canonical_key(name)]
        assert row["core"], name
        assert row["python"], name
        assert row["wasm"], name

    sma = rows[canonical_key("SMA")]
    assert sma["flutter"], "SMA should be generated for Flutter"

    streaming = rows[canonical_key("StreamingSMA")]
    assert streaming["core"] and streaming["python"] and streaming["wasm"]

    counts = manifest["coverage"]["counts"]
    assert (
        counts["common_python_wasm_count"]
        > manifest["parity_summary"]["common_python_wasm_count"]
    )
    assert "flutter" in manifest["surfaces"]
    assert manifest["surfaces"]["flutter"]["manual_exclude"]
    assert any(row["flutter_excluded"] for row in manifest["coverage"]["rows"])

    names = {row["name"] for row in manifest["coverage"]["rows"]}
    for method in ("new", "period", "reset", "update"):
        assert method not in names, f"{method} is an impl method, not a core export"
    assert canonical_key("StreamingSMA") in {
        row["key"] for row in manifest["coverage"]["rows"]
    }

    support_matrix = (ROOT / "docs" / "support_matrix.rst").read_text(encoding="utf-8")
    for snippet in support_matrix_count_snippets(counts):
        assert snippet in support_matrix, snippet
