#!/usr/bin/env python3
"""
Build a cross-surface API manifest for ferro-ta.

The generated manifest summarizes:
- Python indicator/method exposure (from ferro_ta.tools.api_info)
- Core Rust crate public functions (ferro_ta_core)
- WASM/Node exported functions (from wasm pkg d.ts)

Output is written to `docs/api_manifest.json`.
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_api_info_module(root: Path, module_path: Path):
    python_root = str(root / "python")
    if python_root not in sys.path:
        sys.path.insert(0, python_root)
    spec = importlib.util.spec_from_file_location(
        "ferro_ta_tools_api_info", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def _module_file(root: Path, module_name: str) -> Path | None:
    module_rel = module_name.replace(".", "/")
    file_path = root / "python" / f"{module_rel}.py"
    if file_path.exists():
        return file_path
    init_path = root / "python" / module_rel / "__init__.py"
    if init_path.exists():
        return init_path
    return None


def _extract_dunder_all(file_path: Path) -> list[str]:
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except Exception:
        return []

    exports: list[str] = []
    for node in tree.body:
        value_node = None
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    value_node = node.value
                    break
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == "__all__":
                value_node = node.value
        if value_node is None:
            continue
        try:
            value = ast.literal_eval(value_node)
        except Exception:
            continue
        if isinstance(value, str):
            exports = [value]
        elif isinstance(value, (list, tuple)):
            exports = [item for item in value if isinstance(item, str)]
    return exports


def _module_exports(root: Path, module_name: str) -> list[str]:
    file_path = _module_file(root, module_name)
    if file_path is None:
        return []
    return _extract_dunder_all(file_path)


def _extract_python_api(root: Path) -> dict[str, Any]:
    module_path = root / "python" / "ferro_ta" / "tools" / "api_info.py"
    api_info_module = _load_api_info_module(root, module_path)

    category_modules = dict(getattr(api_info_module, "_CATEGORY_MODULES", {}))
    method_modules = dict(getattr(api_info_module, "_METHOD_MODULES", {}))

    indicators: list[dict[str, Any]] = []
    seen_indicators: set[str] = set()
    for category, module_name in category_modules.items():
        for name in _module_exports(root, module_name):
            if name in seen_indicators:
                continue
            seen_indicators.add(name)
            indicators.append(
                {
                    "name": name,
                    "category": category,
                    "module": module_name,
                    "doc": "",
                    "params": [],
                }
            )

    methods: list[dict[str, Any]] = []
    seen_methods: set[tuple[str, str]] = set()
    for category, module_name in method_modules.items():
        for name in _module_exports(root, module_name):
            key = (module_name, name)
            if key in seen_methods:
                continue
            seen_methods.add(key)
            methods.append(
                {
                    "name": name,
                    "category": category,
                    "module": module_name,
                    "doc": "",
                    "params": [],
                }
            )

    indicators.sort(key=lambda entry: entry["name"])
    methods.sort(key=lambda entry: (entry["category"], entry["name"]))

    categories = sorted({entry["category"] for entry in indicators})

    if not indicators:
        raise RuntimeError(
            "No Python indicators discovered from source exports. "
            "Check `python/ferro_ta/tools/api_info.py` mappings and module __all__ declarations."
        )

    return {
        "indicator_count": len(indicators),
        "method_count": len(methods),
        "categories": categories,
        "indicators": indicators,
        "methods": methods,
    }


def _extract_core_exports(root: Path) -> list[dict[str, str]]:
    core_src = root / "crates" / "ferro_ta_core" / "src"
    entries: list[dict[str, str]] = []

    for rs_file in sorted(core_src.rglob("*.rs")):
        rel = rs_file.relative_to(core_src).as_posix()
        module = rel[:-3].replace("/", ".")
        text = rs_file.read_text(encoding="utf-8")
        for match in re.finditer(r"(?m)^\s*pub\s+fn\s+([A-Za-z0-9_]+)\s*\(", text):
            entries.append(
                {
                    "module": module,
                    "function": match.group(1),
                    "file": rel,
                }
            )

    entries.sort(key=lambda item: (item["module"], item["function"]))
    return entries


def _extract_wasm_exports(root: Path) -> list[str]:
    exports: set[str] = set()

    # Source exports are the canonical declaration of the WASM/Node API and
    # avoid drift when a stale wasm/pkg folder is present locally.
    wasm_lib = root / "wasm" / "src" / "lib.rs"
    if wasm_lib.exists():
        text = wasm_lib.read_text(encoding="utf-8")
        for match in re.finditer(
            r"(?ms)#\s*\[wasm_bindgen(?:\([^\)]*\))?\]\s*pub\s+fn\s+([A-Za-z0-9_]+)\s*\(",
            text,
        ):
            exports.add(match.group(1))
        if exports:
            return sorted(exports)

    # Fallback to generated declarations if source parsing did not find exports.
    dts_path = root / "wasm" / "pkg" / "ferro_ta_wasm.d.ts"
    if dts_path.exists():
        for line in dts_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("export function "):
                name = line[len("export function ") :].split("(")[0].strip()
                if name:
                    exports.add(name)

    return sorted(exports)


def _safe_git_head(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    value = completed.stdout.strip()
    return value or None


def build_manifest(
    root: Path, include_runtime_metadata: bool = False
) -> dict[str, Any]:
    python_api = _extract_python_api(root)
    rust_core = _extract_core_exports(root)
    wasm_exports = _extract_wasm_exports(root)

    python_indicator_names = {entry["name"] for entry in python_api["indicators"]}
    python_indicator_names_lc = {name.lower() for name in python_indicator_names}
    wasm_set = set(wasm_exports)
    wasm_set_lc = {name.lower() for name in wasm_set}
    common_with_wasm = sorted(python_indicator_names_lc.intersection(wasm_set_lc))

    manifest: dict[str, Any] = {
        "surfaces": {
            "python": python_api,
            "rust_core": {
                "public_function_count": len(rust_core),
                "functions": rust_core,
            },
            "wasm_node": {
                "export_count": len(wasm_exports),
                "exports": wasm_exports,
            },
        },
        "parity_summary": {
            "python_indicator_count": len(python_indicator_names_lc),
            "wasm_export_count": len(wasm_set),
            "common_python_wasm_count": len(common_with_wasm),
            "common_python_wasm": common_with_wasm,
            "python_only_vs_wasm": sorted(python_indicator_names_lc - wasm_set_lc),
            "wasm_only_vs_python": sorted(wasm_set_lc - python_indicator_names_lc),
        },
    }

    if include_runtime_metadata:
        manifest["generated_at_utc"] = _dt.datetime.now(tz=_dt.UTC).isoformat()
        manifest["git_head"] = _safe_git_head(root)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cross-surface API manifest")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/api_manifest.json"),
        help="Output JSON path relative to repo root (default: docs/api_manifest.json)",
    )
    parser.add_argument(
        "--include-runtime-metadata",
        action="store_true",
        help=(
            "Include non-deterministic metadata fields (timestamp, git head). "
            "Disabled by default to keep manifest reproducible for CI checks."
        ),
    )
    args = parser.parse_args()

    root = _repo_root()
    output_path = (root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        root, include_runtime_metadata=args.include_runtime_metadata
    )
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote API manifest to {output_path}")


if __name__ == "__main__":
    main()
