#!/usr/bin/env python3
"""Update or verify ferro-ta version strings across release files.

Usage
-----
python3 scripts/bump_version.py 1.0.3
python3 scripts/bump_version.py --check
python3 scripts/bump_version.py --show
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


@dataclass(frozen=True)
class VersionCarrier:
    label: str
    path: Path
    pattern: str
    replacement: str

    def read(self) -> str:
        text = self.path.read_text(encoding="utf-8")
        match = re.search(self.pattern, text, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"Could not find version for {self.label} in {self.path}")
        return match.group(2)

    def write(self, version: str) -> bool:
        text = self.path.read_text(encoding="utf-8")
        updated, count = re.subn(
            self.pattern,
            rf"\g<1>{version}\g<3>",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise ValueError(f"Could not update {self.label} in {self.path}")
        changed = updated != text
        if changed:
            self.path.write_text(updated, encoding="utf-8")
        return changed


CARRIERS = [
    VersionCarrier(
        "cargo_root",
        ROOT / "Cargo.toml",
        r'(?m)^(version = ")([^"]+)(")$',
        r"\g<1>{version}\g<3>",
    ),
    VersionCarrier(
        "cargo_core_dep",
        ROOT / "Cargo.toml",
        r'(ferro_ta_core = \{ path = "crates/ferro_ta_core", version = ")([^"]+)(" \})',
        r"\g<1>{version}\g<3>",
    ),
    VersionCarrier(
        "cargo_core_crate",
        ROOT / "crates" / "ferro_ta_core" / "Cargo.toml",
        r'(?m)^(version = ")([^"]+)(")$',
        r"\g<1>{version}\g<3>",
    ),
    VersionCarrier(
        "cargo_core_readme",
        ROOT / "crates" / "ferro_ta_core" / "README.md",
        r'(ferro_ta_core = ")([^"]+)(")',
        r"\g<1>{version}\g<3>",
    ),
    VersionCarrier(
        "pyproject",
        ROOT / "pyproject.toml",
        r'(?m)^(version = ")([^"]+)(")$',
        r"\g<1>{version}\g<3>",
    ),
    VersionCarrier(
        "wasm_cargo",
        ROOT / "wasm" / "Cargo.toml",
        r'(?m)^(version = ")([^"]+)(")$',
        r"\g<1>{version}\g<3>",
    ),
    VersionCarrier(
        "wasm_package",
        ROOT / "wasm" / "package.json",
        r'("version": ")([^"]+)(")',
        r"\g<1>{version}\g<3>",
    ),
    VersionCarrier(
        "conda",
        ROOT / "conda" / "meta.yaml",
        r'({% set version = ")([^"]+)(" %})',
        r"\g<1>{version}\g<3>",
    ),
    VersionCarrier(
        "docs_changelog",
        ROOT / "docs" / "changelog.rst",
        r"(These docs track package version ``)([^`]+)(``\.)",
        r"\g<1>{version}\g<3>",
    ),
    VersionCarrier(
        "docs_support_matrix",
        ROOT / "docs" / "support_matrix.rst",
        r"(These docs track package version ``)([^`]+)(``\.)",
        r"\g<1>{version}\g<3>",
    ),
]


def _read_versions() -> dict[str, str]:
    return {carrier.label: carrier.read() for carrier in CARRIERS}


def _print_versions(versions: dict[str, str]) -> None:
    for label, version in versions.items():
        print(f"{label:20} {version}")


def _check_versions() -> int:
    versions = _read_versions()
    unique = sorted(set(versions.values()))
    _print_versions(versions)
    if len(unique) != 1:
        print()
        print(f"ERROR: version mismatch detected: {', '.join(unique)}")
        return 1
    print()
    print(f"OK: all tracked versions match {unique[0]}")
    return 0


def _set_version(version: str) -> int:
    if not SEMVER_RE.match(version):
        print(f"ERROR: expected MAJOR.MINOR.PATCH, got {version!r}")
        return 1

    changed_paths: list[Path] = []
    for carrier in CARRIERS:
        if carrier.write(version):
            changed_paths.append(carrier.path)

    if changed_paths:
        print(f"Updated version to {version}:")
        for path in sorted(set(changed_paths)):
            print(f" - {path.relative_to(ROOT)}")
    else:
        print(f"No changes needed. All tracked files already use {version}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", nargs="?", help="New version to write")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if tracked version strings do not match",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print tracked version strings without modifying files",
    )
    args = parser.parse_args()

    if args.check:
        return _check_versions()
    if args.show:
        _print_versions(_read_versions())
        return 0
    if args.version:
        return _set_version(args.version)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
