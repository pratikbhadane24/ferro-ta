#!/usr/bin/env python3
"""
Check that docs/api_manifest.json, the coverage RST include, and
support-matrix count phrases are up-to-date.

This script regenerates the deterministic manifest in-memory and compares it to
the committed files. It exits non-zero if drift is detected.
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

    from build_api_manifest import (
        COVERAGE_RST_REL,
        build_manifest,
        render_coverage_rst,
        support_matrix_count_snippets,
    )

    manifest_path = root / "docs" / "api_manifest.json"
    coverage_rst_path = root / COVERAGE_RST_REL
    support_matrix_path = root / "docs" / "support_matrix.rst"

    missing = []
    if not manifest_path.exists():
        missing.append("docs/api_manifest.json")
    if not coverage_rst_path.exists():
        missing.append(str(COVERAGE_RST_REL))
    if not support_matrix_path.exists():
        missing.append("docs/support_matrix.rst")
    if missing:
        print(
            "Missing generated API coverage files:\n  "
            + "\n  ".join(missing)
            + "\nRun:\n"
            "  python scripts/build_api_manifest.py"
        )
        return 1

    expected = build_manifest(root, include_runtime_metadata=False)
    actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_rst = render_coverage_rst(expected)
    actual_rst = coverage_rst_path.read_text(encoding="utf-8")
    support_matrix_text = support_matrix_path.read_text(encoding="utf-8")
    missing_counts = [
        snippet
        for snippet in support_matrix_count_snippets(expected["coverage"]["counts"])
        if snippet not in support_matrix_text
    ]

    drifted = []
    if actual != expected:
        drifted.append("docs/api_manifest.json")
    if actual_rst != expected_rst:
        drifted.append(str(COVERAGE_RST_REL))
    if missing_counts:
        drifted.append("docs/support_matrix.rst")
    if drifted:
        extra = ""
        if missing_counts:
            extra = (
                "\nSupport-matrix count phrases missing or stale:\n  "
                + "\n  ".join(missing_counts)
            )
        print(
            "Generated API coverage files are out of date:\n  "
            + "\n  ".join(drifted)
            + extra
            + "\nRun:\n"
            "  python scripts/build_api_manifest.py\n"
            "and commit the updated files (including support-matrix counts)."
        )
        return 1

    print(
        "docs/api_manifest.json, coverage RST include, and support-matrix "
        "counts are up to date."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
