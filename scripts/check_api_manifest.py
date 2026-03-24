#!/usr/bin/env python3
"""
Check that docs/api_manifest.json is up-to-date.

This script regenerates the deterministic manifest in-memory and compares it to
the committed file. It exits non-zero if drift is detected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    python_root = str(root / "python")
    if python_root not in sys.path:
        sys.path.insert(0, python_root)
    scripts_root = str(root / "scripts")
    if scripts_root not in sys.path:
        sys.path.insert(0, scripts_root)

    from build_api_manifest import build_manifest

    manifest_path = root / "docs" / "api_manifest.json"

    if not manifest_path.exists():
        print(
            "docs/api_manifest.json is missing. Run:\n"
            "  python scripts/build_api_manifest.py --output docs/api_manifest.json"
        )
        return 1

    expected = build_manifest(root, include_runtime_metadata=False)
    actual = json.loads(manifest_path.read_text(encoding="utf-8"))

    if actual != expected:
        print(
            "docs/api_manifest.json is out of date.\n"
            "Run:\n"
            "  python scripts/build_api_manifest.py --output docs/api_manifest.json\n"
            "and commit the updated file."
        )
        return 1

    print("docs/api_manifest.json is up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
