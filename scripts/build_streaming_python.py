#!/usr/bin/env python3
"""Generate the Python stubs and exports of the registry streaming classes.

The registry (``crates/ferro_ta_core/src/streaming/registry.rs``) generates one
PyO3 class per stream. This script rewrites the generated block (between
``# BEGIN GENERATED STREAMS`` and ``# END GENERATED STREAMS``) in

- ``python/ferro_ta/__init__.pyi``         (class stubs)
- ``python/ferro_ta/data/streaming.py``    (imports + ``__all__``)

Streams, params (kinds) and inputs come from ``ffi_spec.json`` (every registry
stream is also a C ABI stream with the same names); defaults come from the
built extension: the class ``__text_signature__`` must list the spec params in
order, and the values are read back from a default-constructed instance.

Usage
-----
python3 scripts/build_streaming_python.py           # rewrite the blocks
python3 scripts/build_streaming_python.py --check   # fail if a block is stale

Needs the extension built first (``maturin develop --release``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stream_names import is_registry_stream, stream_class  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "crates" / "ferro_ta_ffi" / "ffi_spec.json"
STUB_PATH = ROOT / "python" / "ferro_ta" / "__init__.pyi"
EXPORTS_PATH = ROOT / "python" / "ferro_ta" / "data" / "streaming.py"

BEGIN = "# BEGIN GENERATED STREAMS (scripts/build_streaming_python.py)"
END = "# END GENERATED STREAMS"
# Where a missing block is inserted.
STUB_ANCHOR = "class Bar:\n"
LINE_LENGTH = 88

PARAM_TYPES = {"period": "int", "count": "int", "matype": "int", "float": "float"}
ELEM_TYPES = {"f64": "float", "i32": "int", "i64": "int", "i8": "int"}


def render_def(name: str, args: list[str], ret: str, indent: str = "    ") -> list[str]:
    """A stub ``def`` laid out the way ``ruff format`` lays it out."""
    one_line = f"{indent}def {name}({', '.join(args)}) -> {ret}: ..."
    if len(one_line) <= LINE_LENGTH:
        return [one_line]
    hugged = f"{indent}    {', '.join(args)}"
    if len(hugged) <= LINE_LENGTH:
        return [f"{indent}def {name}(", hugged, f"{indent}) -> {ret}: ..."]
    return [
        f"{indent}def {name}(",
        *[f"{indent}    {arg}," for arg in args],
        f"{indent}) -> {ret}: ...",
    ]


def return_type(stream: dict) -> str:
    elems = [ELEM_TYPES[o["elem"]] for o in stream["outputs"]]
    return elems[0] if len(elems) == 1 else f"tuple[{', '.join(elems)}]"


def signature_names(text_signature: str | None) -> list[str]:
    """Parameter names of a ``__text_signature__`` (without self / ``*``)."""
    if not text_signature:
        return []
    inner = text_signature.strip()[1:-1]
    names = [part.split("=")[0].strip() for part in inner.split(",") if part.strip()]
    return [n for n in names if n not in ("$self", "*", "/")]


def stream_defaults(stream: dict, module: object) -> dict[str, object]:
    """Param defaults of the built class, checked against the spec."""
    name = stream_class(stream["name"])
    cls = getattr(module, name, None)
    if cls is None:
        sys.exit(f"{name} missing from the extension; run maturin develop --release")
    params = [p["name"] for p in stream["params"]]
    if signature_names(cls.__text_signature__) != params:
        sys.exit(f"{name}{cls.__text_signature__} does not match spec params {params}")
    inputs = [*stream["inputs"], "closed"]
    if signature_names(cls.update.__text_signature__) != inputs:
        sys.exit(
            f"{name}.update{cls.update.__text_signature__} != spec inputs {inputs}"
        )
    instance = cls()
    return {p: getattr(instance, p) for p in params}


def render_stub(stream: dict, defaults: dict[str, object]) -> list[str]:
    name = stream_class(stream["name"])
    params = [(p["name"], PARAM_TYPES[p["kind"]]) for p in stream["params"]]
    init_args = ["self", *[f"{p}: {t} = {defaults[p]!r}" for p, t in params]]
    update_args = [
        "self",
        *[f"{i}: float" for i in stream["inputs"]],
        "*",
        "closed: bool = True",
    ]
    return [
        f"class {name}:",
        *[f"    {p}: {t}" for p, t in params],
        *render_def("__init__", init_args, "None"),
        *render_def("update", update_args, return_type(stream)),
        "    def reset(self) -> None: ...",
        "    @property",
        "    def count(self) -> int: ...",
        "    @property",
        "    def is_ready(self) -> bool: ...",
        "    def to_json(self) -> str: ...",
        "    @staticmethod",
        f"    def from_json(json: str) -> {name}: ...",
        "    def __reduce__(self) -> tuple[object, tuple[str]]: ...",
        f"    def __copy__(self) -> {name}: ...",
        f"    def __deepcopy__(self, memo: object) -> {name}: ...",
    ]


def render_stub_block(streams: list[dict], module: object) -> str:
    lines = [BEGIN]
    for stream in streams:
        lines += ["", *render_stub(stream, stream_defaults(stream, module))]
    return "\n".join([*lines, "", END]) + "\n"


def render_exports_block(streams: list[dict]) -> str:
    # Case-insensitive, as ruff's isort orders imported names (`MaEnvelopes`
    # after `Macdfix`), so the pre-commit hook leaves the block unchanged.
    # Ruff uses *natural* order (digit runs compare as numbers); this key
    # agrees only while no two names differ in a multi-digit number
    # (`Foo2` vs `Foo10`) — switch to a natural-order key if one appears.
    names = sorted(
        (stream_class(s["name"]) for s in streams), key=lambda n: (n.lower(), n)
    )
    return (
        "\n".join(
            [
                BEGIN,
                "from ferro_ta._ferro_ta import (  # noqa: E402",
                *[f"    {n}," for n in names],
                ")",
                "",
                "__all__ += [",
                *[f'    "{n}",' for n in names],
                "]",
                END,
            ]
        )
        + "\n"
    )


def splice(text: str, block: str, anchor: str | None) -> str:
    """Replace the generated block, or insert it before ``anchor`` / at the end."""
    if BEGIN in text:
        start = text.index(BEGIN)
        end = text.index(END, start) + len(END) + 1
        return text[:start] + block + text[end:]
    if anchor is not None:
        if anchor not in text:
            sys.exit(f"anchor {anchor!r} not found")
        at = text.index(anchor)
        return text[:at] + block + "\n" + text[at:]
    return text.rstrip("\n") + "\n\n" + block


def load_extension() -> object:
    """The compiled ``ferro_ta._ferro_ta``: installed, or the in-tree build.

    The in-tree ``.so`` is loaded directly so this runs under a bare
    ``python3`` (the package ``__init__`` needs numpy; the extension does not).
    """
    try:
        import ferro_ta._ferro_ta as module

        return module
    except ImportError:
        pass
    for path in sorted((ROOT / "python" / "ferro_ta").glob("_ferro_ta*.*")):
        if path.suffix not in (".so", ".pyd"):
            continue
        spec = importlib.util.spec_from_file_location("ferro_ta._ferro_ta", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    sys.exit("ferro_ta extension not built; run: maturin develop --release")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if stale")
    args = parser.parse_args()

    module = load_extension()

    spec = json.loads(SPEC_PATH.read_text())
    streams = [s for s in spec["streams"] if is_registry_stream(s["name"])]
    outputs = {
        STUB_PATH: splice(
            STUB_PATH.read_text(), render_stub_block(streams, module), STUB_ANCHOR
        ),
        EXPORTS_PATH: splice(
            EXPORTS_PATH.read_text(), render_exports_block(streams), None
        ),
    }
    stale = [p for p, text in outputs.items() if p.read_text() != text]
    if args.check:
        for path in stale:
            print(f"out of date: {path.relative_to(ROOT)}", file=sys.stderr)
        if stale:
            print("run: python3 scripts/build_streaming_python.py", file=sys.stderr)
            return 1
        print(f"streaming stubs up to date ({len(streams)} registry streams)")
        return 0
    for path in stale:
        path.write_text(outputs[path])
    print(f"wrote {len(stale)} files ({len(streams)} registry streams)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
