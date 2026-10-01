#!/usr/bin/env python3
"""Generate the C header and C-ABI language wrappers from the FFI spec.

``crates/ferro_ta_ffi`` records the signature of every export it defines (the
``ffi_exports!`` / ``ffi_streams!`` macros emit metadata next to each
``extern "C"`` function). Its ``dump_spec`` example serialises that metadata to
``crates/ferro_ta_ffi/ffi_spec.json``; this script turns the spec into:

- ``crates/ferro_ta_ffi/include/ferro_ta.h``  (the public C header)
- ``bindings/go/include/ferro_ta.h``           (copy; Go modules are self-contained)
- ``bindings/go/<group>_gen.go``               (idiomatic Go wrappers)

cbindgen is not used because it cannot see macro-generated items on stable Rust.

Usage
-----
python3 scripts/build_ffi_bindings.py           # regenerate spec, header and wrappers
python3 scripts/build_ffi_bindings.py --check   # fail if anything is out of date
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ffi_codegen.c_header import render_header  # noqa: E402
from ffi_codegen.go import render_go  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FFI_DIR = ROOT / "crates" / "ferro_ta_ffi"
SPEC_PATH = FFI_DIR / "ffi_spec.json"
HEADER_PATH = FFI_DIR / "include" / "ferro_ta.h"
GO_DIR = ROOT / "bindings" / "go"
GO_HEADER_PATH = GO_DIR / "include" / "ferro_ta.h"


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


def dump_spec() -> str:
    """Run the Rust dump_spec example and return its JSON output."""
    result = subprocess.run(
        ["cargo", "run", "-q", "-p", "ferro_ta_ffi", "--example", "dump_spec"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"dump_spec failed:\n{result.stderr}")
    return result.stdout


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="fail if outputs are stale"
    )
    args = parser.parse_args()

    spec_text = dump_spec()
    spec = json.loads(spec_text)
    header = render_header(spec)
    outputs: dict[Path, str] = {
        SPEC_PATH: spec_text,
        HEADER_PATH: header,
        GO_HEADER_PATH: header,
        **render_go(spec, GO_DIR),
    }

    stale_generated = {
        p
        for p in [*GO_DIR.glob("*_gen.go"), *GO_DIR.glob("*_gen_test.go")]
        if p not in outputs
    }
    if args.check:
        stale = [
            p for p, text in outputs.items() if not p.exists() or p.read_text() != text
        ]
        stale += sorted(stale_generated)
        if stale:
            for path in stale:
                print(f"out of date: {path.relative_to(ROOT)}", file=sys.stderr)
            print("run: python3 scripts/build_ffi_bindings.py", file=sys.stderr)
            return 1
        print(
            f"ffi bindings up to date ({len(spec['functions'])} functions, {len(spec['streams'])} streams)"
        )
        return 0

    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for path in stale_generated:
        path.unlink()
    print(
        f"wrote {len(outputs)} files ({len(spec['functions'])} functions, {len(spec['streams'])} streams)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
