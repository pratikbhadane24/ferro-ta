from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_api_manifest import build_manifest


def test_api_manifest_is_deterministic_and_current() -> None:
    manifest_path = ROOT / "docs" / "api_manifest.json"
    assert manifest_path.exists(), "docs/api_manifest.json is missing"

    expected = build_manifest(ROOT, include_runtime_metadata=False)
    actual = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert actual == expected
