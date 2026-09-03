#!/usr/bin/env python3
"""
Build a cross-surface API manifest for ferro-ta.

The generated manifest summarizes:
- Python indicator/method exposure (from ferro_ta.tools.api_info)
- Core Rust crate public functions (ferro_ta_core)
- WASM/Node exported functions (from wasm/src/lib.rs)
- Flutter/Dart wrappers (flutter/rust/src/api/indicators.rs) plus MANUAL_EXCLUDE

Output:
- `docs/api_manifest.json`
- `docs/languages/_coverage.inc.rst` (checked-in coverage table)
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

COVERAGE_RST_REL = Path("docs/languages/_coverage.inc.rst")

# Element-wise / rolling math names used to strip WASM `transform_` / `math_`
# prefixes and core `math_*` wrappers onto the TA-Lib short name (SIN, ADD, …).
_MATH_SHORT = {
    "add",
    "sub",
    "mult",
    "div",
    "sum",
    "max",
    "min",
    "maxindex",
    "minindex",
    "acos",
    "asin",
    "atan",
    "ceil",
    "cos",
    "cosh",
    "exp",
    "floor",
    "ln",
    "log10",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
}

# Remaining known aliases after underscore-fold + prefix stripping.
_EXPLICIT_ALIASES = {
    "trixindicator": "trix",
    "betarolling": "beta",
    "drawdownseries": "drawdown",
    "fundingcumulativepnl": "fundingpnl",
    "aggregatetickbars": "aggregateticks",
    "marksessionboundaries": "sessionboundaries",
    "rollingsum": "sum",
    "rollingmax": "max",
    "rollingmin": "min",
    "rollingmaxindex": "maxindex",
    "rollingminindex": "minindex",
}

_CORE_HELPER_SUFFIXES = ("_into",)
_CORE_HELPER_NAMES = {
    "body_size",
    "upper_shadow",
    "lower_shadow",
    "candle_range",
    "is_bullish",
    "is_bearish",
    "sliding_max",
    "sliding_min",
}

_UNARY_TRANSFORM_RE = re.compile(r"(?m)^unary_transform!\(\s*([A-Za-z0-9_]+)\s*,")
_CDL_WRAPPER_RE = re.compile(r"(?ms)cdl_wrapper!\((.*?)\)\s*;")
_MATH_TRANSFORM_WRAPPER_RE = re.compile(
    r"(?m)^math_transform_wrapper!\(\s*([A-Za-z0-9_]+)\s*,"
)
_TOPLEVEL_PUB_FN_RE = re.compile(r"(?m)^pub\s+fn\s+([A-Za-z0-9_]+)\s*\(")
_TOPLEVEL_PUB_STRUCT_RE = re.compile(r"(?m)^pub\s+struct\s+([A-Za-z0-9_]+)\b")
_WASM_BINDGEN_TOPLEVEL_ITEM_RE = re.compile(
    r"(?m)^(?:#\[[^\]]+\]\s*)+pub\s+(?:fn|struct)\s+([A-Za-z0-9_]+)\b"
)
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def fold_key(name: str) -> str:
    """Lowercase and strip non-alphanumerics so SMA / sma / Sma match."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def canonical_key(name: str) -> str:
    """Map a surface-local name onto a shared coverage key."""
    key = fold_key(name)
    if key.startswith("wasm") and len(key) > 4:
        key = key[4:]
    if key.startswith("transform") and key[9:] in _MATH_SHORT:
        key = key[9:]
    if key.startswith("math") and key[4:] in _MATH_SHORT:
        key = key[4:]
    if key.endswith("indicator"):
        stripped = key[: -len("indicator")]
        if stripped == "trix":
            key = stripped
    return _EXPLICIT_ALIASES.get(key, key)


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


def _is_core_helper(function: str) -> bool:
    if function in _CORE_HELPER_NAMES:
        return True
    return any(function.endswith(suffix) for suffix in _CORE_HELPER_SUFFIXES)


def _extract_core_exports(root: Path) -> list[dict[str, str]]:
    core_src = root / "crates" / "ferro_ta_core" / "src"
    entries: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for rs_file in sorted(core_src.rglob("*.rs")):
        rel = rs_file.relative_to(core_src).as_posix()
        module = rel[:-3].replace("/", ".")
        text = rs_file.read_text(encoding="utf-8")
        names = [match.group(1) for match in _TOPLEVEL_PUB_FN_RE.finditer(text)]
        names.extend(_UNARY_TRANSFORM_RE.findall(text))
        for name in names:
            key = (module, name)
            if key in seen:
                continue
            seen.add(key)
            entries.append(
                {
                    "module": module,
                    "function": name,
                    "file": rel,
                }
            )

    entries.sort(key=lambda item: (item["module"], item["function"]))
    return entries


def _extract_core_types(root: Path) -> list[str]:
    """Public structs used as coverage peers (streaming classes, etc.)."""
    core_src = root / "crates" / "ferro_ta_core" / "src"
    names: set[str] = set()
    for rs_file in core_src.rglob("*.rs"):
        text = rs_file.read_text(encoding="utf-8")
        for name in _TOPLEVEL_PUB_STRUCT_RE.findall(text):
            if name.startswith("Streaming") and name not in {
                "StreamingError",
                "StreamingBacktest",
                "StreamingBarResult",
                "StreamingSummary",
            }:
                names.add(name)
    return sorted(names)


def _extract_wasm_exports(root: Path) -> list[str]:
    exports: set[str] = set()

    # Source exports are the canonical declaration of the WASM/Node API and
    # avoid drift when a stale wasm/pkg folder is present locally.
    wasm_lib = root / "wasm" / "src" / "lib.rs"
    if wasm_lib.exists():
        text = wasm_lib.read_text(encoding="utf-8")
        for match in _WASM_BINDGEN_TOPLEVEL_ITEM_RE.finditer(text):
            if "wasm_bindgen" in match.group(0):
                exports.add(match.group(1))
        for block in _CDL_WRAPPER_RE.findall(text):
            exports.update(_IDENT_RE.findall(block))
        exports.update(_MATH_TRANSFORM_WRAPPER_RE.findall(text))
        if exports:
            return sorted(exports)

    # Fallback to generated declarations if source parsing did not find exports.
    dts_path = root / "wasm" / "node" / "ferro_ta_wasm.d.ts"
    if dts_path.exists():
        for line in dts_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("export function "):
                name = line[len("export function ") :].split("(")[0].strip()
                if name:
                    exports.add(name)

    return sorted(exports)


def _load_flutter_manual_exclude(root: Path) -> list[str]:
    """Parse MANUAL_EXCLUDE from the Flutter generator without importing it."""
    module_path = root / "scripts" / "build_flutter_bridge.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    for node in tree.body:
        target_names: list[str] = []
        value_node = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_names = [node.target.id]
            value_node = node.value
        elif isinstance(node, ast.Assign):
            target_names = [
                target.id for target in node.targets if isinstance(target, ast.Name)
            ]
            value_node = node.value
        if "MANUAL_EXCLUDE" not in target_names or value_node is None:
            continue
        try:
            value = ast.literal_eval(value_node)
        except Exception as exc:
            raise RuntimeError(
                f"Could not parse MANUAL_EXCLUDE in {module_path}"
            ) from exc
        if isinstance(value, (set, list, tuple)):
            return sorted(str(item) for item in value)
        raise RuntimeError(f"MANUAL_EXCLUDE in {module_path} is not a set/list")
    raise RuntimeError(f"MANUAL_EXCLUDE not found in {module_path}")


def _extract_flutter_exports(root: Path) -> list[str]:
    indicators_rs = root / "flutter" / "rust" / "src" / "api" / "indicators.rs"
    if not indicators_rs.exists():
        return []
    text = indicators_rs.read_text(encoding="utf-8")
    return sorted(_TOPLEVEL_PUB_FN_RE.findall(text))


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


def _index_by_canonical(names: list[str]) -> dict[str, str]:
    """Map canonical key → first original name (stable: sorted input preferred)."""
    index: dict[str, str] = {}
    for name in names:
        key = canonical_key(name)
        index.setdefault(key, name)
    return index


def _build_coverage_rows(
    python_indicators: list[dict[str, Any]],
    rust_core: list[dict[str, str]],
    rust_core_types: list[str],
    wasm_exports: list[str],
    flutter_exports: list[str],
    flutter_excluded: list[str],
) -> list[dict[str, Any]]:
    python_by_key = _index_by_canonical([entry["name"] for entry in python_indicators])
    python_category = {
        canonical_key(entry["name"]): entry["category"] for entry in python_indicators
    }
    core_by_key: dict[str, str] = {}
    core_category: dict[str, str] = {}
    for entry in rust_core:
        function = entry["function"]
        if _is_core_helper(function):
            continue
        key = canonical_key(function)
        core_by_key.setdefault(key, function)
        module = entry["module"].split(".")[0]
        core_category.setdefault(key, module)
    for type_name in rust_core_types:
        key = canonical_key(type_name)
        core_by_key.setdefault(key, type_name)
        core_category.setdefault(key, "streaming")
    wasm_by_key = _index_by_canonical(wasm_exports)
    flutter_by_key = _index_by_canonical(flutter_exports)
    excluded_by_key = _index_by_canonical(flutter_excluded)

    keys = sorted(
        set(python_by_key)
        | set(core_by_key)
        | set(wasm_by_key)
        | set(flutter_by_key)
        | set(excluded_by_key)
    )

    rows: list[dict[str, Any]] = []
    for key in keys:
        python_name = python_by_key.get(key)
        core_name = core_by_key.get(key)
        wasm_name = wasm_by_key.get(key)
        flutter_name = flutter_by_key.get(key)
        excluded_name = excluded_by_key.get(key)
        display = (
            python_name
            or core_name
            or wasm_name
            or flutter_name
            or excluded_name
            or key
        )
        category = python_category.get(key) or core_category.get(key) or "other"
        flutter_excluded_flag = flutter_name is None and excluded_name is not None
        rows.append(
            {
                "key": key,
                "name": display,
                "category": category,
                "core": core_name is not None,
                "python": python_name is not None,
                "wasm": wasm_name is not None,
                "flutter": flutter_name is not None,
                "flutter_excluded": flutter_excluded_flag,
                "python_name": python_name,
                "core_name": core_name,
                "wasm_name": wasm_name,
                "flutter_name": flutter_name,
            }
        )
    return rows


def _coverage_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "row_count": len(rows),
        "core_count": sum(1 for row in rows if row["core"]),
        "python_count": sum(1 for row in rows if row["python"]),
        "wasm_count": sum(1 for row in rows if row["wasm"]),
        "flutter_count": sum(1 for row in rows if row["flutter"]),
        "flutter_excluded_count": sum(1 for row in rows if row["flutter_excluded"]),
        "common_python_wasm_count": sum(
            1 for row in rows if row["python"] and row["wasm"]
        ),
        "common_all_four_count": sum(
            1
            for row in rows
            if row["core"] and row["python"] and row["wasm"] and row["flutter"]
        ),
    }


def support_matrix_count_snippets(counts: dict[str, int]) -> list[str]:
    """Phrases that docs/support_matrix.rst must keep in sync with coverage.counts."""
    return [
        f"{counts['python_count']} names on the coverage spine",
        f"{counts['core_count']} core symbols",
        f"{counts['wasm_count']} exports",
        f"{counts['flutter_count']} generated wrappers",
        f"{counts['flutter_excluded_count']} ``MANUAL_EXCLUDE``",
        f"{counts['common_python_wasm_count']} names shared with Python",
        f"{counts['common_all_four_count']} names are present on all four",
    ]


def _cell(present: bool, excluded: bool = False) -> str:
    if present:
        return "yes"
    if excluded:
        return "excluded"
    return "—"


def render_coverage_rst(manifest: dict[str, Any]) -> str:
    """Render the checked-in Sphinx include for language coverage."""
    coverage = manifest["coverage"]
    counts = coverage["counts"]
    rows = coverage["rows"]

    lines = [
        ".. Generated by scripts/build_api_manifest.py — do not edit by hand.",
        "",
        ".. list-table:: Coverage counts",
        "   :header-rows: 1",
        "   :widths: 20 12 12 12 12 16",
        "",
        "   * - Rows",
        "     - Core",
        "     - Python",
        "     - WASM",
        "     - Flutter",
        "     - Flutter excluded",
        f"   * - {counts['row_count']}",
        f"     - {counts['core_count']}",
        f"     - {counts['python_count']}",
        f"     - {counts['wasm_count']}",
        f"     - {counts['flutter_count']}",
        f"     - {counts['flutter_excluded_count']}",
        "",
        ".. list-table:: Cross-language indicator coverage",
        "   :header-rows: 1",
        "   :widths: 22 16 10 10 10 12",
        "",
        "   * - Name",
        "     - Category",
        "     - Core",
        "     - Python",
        "     - WASM",
        "     - Flutter",
    ]
    for row in rows:
        flutter_cell = _cell(row["flutter"], row["flutter_excluded"])
        lines.extend(
            [
                f"   * - ``{row['name']}``",
                f"     - {row['category']}",
                f"     - {_cell(row['core'])}",
                f"     - {_cell(row['python'])}",
                f"     - {_cell(row['wasm'])}",
                f"     - {flutter_cell}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def build_manifest(
    root: Path, include_runtime_metadata: bool = False
) -> dict[str, Any]:
    python_api = _extract_python_api(root)
    rust_core = _extract_core_exports(root)
    rust_core_types = _extract_core_types(root)
    wasm_exports = _extract_wasm_exports(root)
    flutter_exports = _extract_flutter_exports(root)
    flutter_excluded = _load_flutter_manual_exclude(root)

    python_indicator_names = {entry["name"] for entry in python_api["indicators"]}
    python_indicator_names_lc = {name.lower() for name in python_indicator_names}
    wasm_set = set(wasm_exports)
    wasm_set_lc = {name.lower() for name in wasm_set}
    common_with_wasm = sorted(python_indicator_names_lc.intersection(wasm_set_lc))

    coverage_rows = _build_coverage_rows(
        python_api["indicators"],
        rust_core,
        rust_core_types,
        wasm_exports,
        flutter_exports,
        flutter_excluded,
    )
    python_keys = {canonical_key(name) for name in python_indicator_names}
    wasm_keys = {canonical_key(name) for name in wasm_exports}

    manifest: dict[str, Any] = {
        "surfaces": {
            "python": python_api,
            "rust_core": {
                "public_function_count": len(rust_core),
                "functions": rust_core,
                "streaming_types": rust_core_types,
            },
            "wasm_node": {
                "export_count": len(wasm_exports),
                "exports": wasm_exports,
            },
            "flutter": {
                "export_count": len(flutter_exports),
                "exports": flutter_exports,
                "manual_exclude_count": len(flutter_excluded),
                "manual_exclude": flutter_excluded,
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
        "normalized_parity": {
            "python_indicator_count": len(python_keys),
            "wasm_export_count": len(wasm_keys),
            "common_python_wasm_count": len(python_keys & wasm_keys),
            "python_only_vs_wasm": sorted(python_keys - wasm_keys),
            "wasm_only_vs_python": sorted(wasm_keys - python_keys),
        },
        "coverage": {
            "counts": _coverage_counts(coverage_rows),
            "rows": coverage_rows,
        },
    }

    if include_runtime_metadata:
        manifest["generated_at_utc"] = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
        manifest["git_head"] = _safe_git_head(root)

    return manifest


def write_manifest_outputs(
    root: Path,
    manifest: dict[str, Any],
    output_path: Path,
    coverage_rst_path: Path | None = None,
) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    rst_path = coverage_rst_path or (root / COVERAGE_RST_REL)
    rst_path.parent.mkdir(parents=True, exist_ok=True)
    rst_path.write_text(render_coverage_rst(manifest), encoding="utf-8")
    return output_path, rst_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cross-surface API manifest")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/api_manifest.json"),
        help="Output JSON path relative to repo root (default: docs/api_manifest.json)",
    )
    parser.add_argument(
        "--coverage-rst",
        type=Path,
        default=COVERAGE_RST_REL,
        help="Coverage RST include path relative to repo root",
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
    coverage_rst_path = (root / args.coverage_rst).resolve()

    manifest = build_manifest(
        root, include_runtime_metadata=args.include_runtime_metadata
    )
    json_path, rst_path = write_manifest_outputs(
        root, manifest, output_path, coverage_rst_path
    )
    print(f"Wrote API manifest to {json_path}")
    print(f"Wrote coverage table to {rst_path}")


if __name__ == "__main__":
    main()
