#!/usr/bin/env python3
"""Validate that CHANGELOG.md keeps a single top-level [Unreleased] section."""

from __future__ import annotations

import re
from pathlib import Path


def main() -> int:
    changelog = Path("CHANGELOG.md")
    if not changelog.exists():
        print("ERROR: CHANGELOG.md not found.")
        return 1

    text = changelog.read_text(encoding="utf-8")
    headings = list(re.finditer(r"^## \[(.+?)\]\s*$", text, flags=re.MULTILINE))
    unreleased = [m for m in headings if m.group(1) == "Unreleased"]

    if not unreleased:
        print("ERROR: CHANGELOG.md is missing a '## [Unreleased]' heading.")
        return 1
    if len(unreleased) > 1:
        print("ERROR: CHANGELOG.md contains multiple '## [Unreleased]' headings.")
        return 1

    if headings and headings[0].group(1) != "Unreleased":
        print("ERROR: '## [Unreleased]' must be the first top-level changelog section.")
        return 1

    print("OK: CHANGELOG.md contains a single top-level [Unreleased] section.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
