"""Render the Go module's generated files from the FFI spec."""

from __future__ import annotations

from pathlib import Path

from .common import with_requires

GO_ELEM = {"f64": "float64", "i32": "int32", "i64": "int64", "i8": "int8"}
C_ELEM = {"f64": "C.double", "i32": "C.int32_t", "i64": "C.int64_t", "i8": "C.int8_t"}


def go_name(snake: str) -> str:
    """`linearreg_slope` -> `LinearregSlope`, `cdl3blackcrows` -> `Cdl3blackcrows`."""
    return "".join(part[:1].upper() + part[1:] for part in snake.split("_") if part)


def go_ident(snake: str) -> str:
    """Lower-camel local identifier: `fastk_period` -> `fastkPeriod`."""
    name = go_name(snake)
    return name[:1].lower() + name[1:]


def go_out_name(out_name: str) -> str:
    """`out` -> `out`, `out_upper` -> `upper`."""
    return "out" if out_name == "out" else go_ident(out_name.removeprefix("out_"))


def go_param_type(param: dict) -> str:
    if param["kind"] == "enum":
        return param["enum"]
    return {"period": "int", "count": "int", "matype": "MAType", "float": "float64"}[
        param["kind"]
    ]


def go_param_arg(param: dict) -> str:
    ident = go_ident(param["name"])
    return {
        "period": f"C.int64_t({ident})",
        "count": f"C.int64_t({ident})",
        "matype": f"C.int32_t({ident})",
        "float": f"C.double({ident})",
        "enum": f"C.int32_t({ident})",
    }[param["kind"]]


def go_doc(name: str, summary: str, doc: str) -> list[str]:
    lines = [f"// {name} {summary}"]
    if doc:
        lines.append("//")
        lines += [f"// {line}".rstrip() for line in doc.splitlines()]
    return lines


def go_signature_params(inputs: list[str], params: list[dict], input_type: str) -> str:
    parts = [f"{go_ident(name)} {input_type}" for name in inputs]
    parts += [f"{go_ident(p['name'])} {go_param_type(p)}" for p in params]
    return ", ".join(parts)


def go_function(fn: dict) -> list[str]:
    core = fn["name"].removeprefix("ft_")
    name = go_name(core)
    inputs = [go_ident(i) for i in fn["inputs"]]
    outs = [(go_out_name(o["name"]), o) for o in fn["outputs"]]
    single = len(outs) == 1
    if single:
        results = f"([]{GO_ELEM[outs[0][1]['elem']]}, error)"
    else:
        named = ", ".join(f"{n} []{GO_ELEM[o['elem']]}" for n, o in outs)
        results = f"({named}, err error)"
    zero_return = "nil, err" if single else ", ".join(["nil"] * len(outs)) + ", err"

    body = [
        *go_doc(
            name,
            f"wraps ferro_ta_core {fn['group']}::{core}.",
            with_requires(fn, "ErrInvalidParam"),
        ),
        f"func {name}({go_signature_params(fn['inputs'], fn['params'], '[]float64')}) {results} {{",
        f"\tn := len({inputs[0]})",
    ]
    if len(inputs) > 1:
        names = ", ".join(f'"{i}"' for i in fn["inputs"])
        lens = ", ".join(f"len({i})" for i in inputs)
        body += [
            f'\tif err := checkLengths("{name}", []string{{{names}}}, []int{{{lens}}}); err != nil {{',
            f"\t\treturn {zero_return}",
            "\t}",
        ]
    for out_name, out in outs:
        body.append(f"\t{out_name}Buf := make([]{GO_ELEM[out['elem']]}, n)")
    args = [f"inPtr({i})" for i in inputs]
    args.append("C.size_t(n)")
    args += [go_param_arg(p) for p in fn["params"]]
    args += [f"(*{C_ELEM[o['elem']]})(outPtr({n}Buf))" for n, o in outs]
    body += [
        f'\tif err := statusError("{name}", C.{fn["name"]}({", ".join(args)})); err != nil {{',
        f"\t\treturn {zero_return}",
        "\t}",
        f"\treturn {', '.join(f'{n}Buf' for n, _ in outs)}, nil",
        "}",
    ]
    return body


def go_enum_const(c_prefix: str, value_name: str) -> str:
    """`FT_OPTION` + `CALL` -> `OptionCall`; `FT_MODEL` + `BLACK_76` -> `ModelBlack76`."""
    words = c_prefix.removeprefix("FT_").split("_") + value_name.split("_")
    return "".join(w[:1] + w[1:].lower() for w in words)


def go_enum(enum: dict) -> list[str]:
    lines = [
        f"// {enum['name']} mirrors the C {enum['c_prefix']}_* constants.",
        f"type {enum['name']} int32",
        "",
        f"// {enum['name']} values.",
        "const (",
    ]
    names = [go_enum_const(enum["c_prefix"], v["name"]) for v in enum["values"]]
    width = max(len(n) for n in names)  # gofmt aligns the const block
    lines += [
        f"\t{n:<{width}} {enum['name']} = {v['value']}"
        for n, v in zip(names, enum["values"])
    ]
    lines.append(")")
    return lines


def go_scalar(fn: dict) -> list[str]:
    core = fn["name"].removeprefix("ft_")
    name = go_name(core)
    outs = [(go_out_name(o["name"]), o) for o in fn["outputs"]]
    single = len(outs) == 1
    params = ", ".join(
        f"{go_ident(p['name'])} {go_param_type(p)}" for p in fn["params"]
    )
    if single:
        results = f"({GO_ELEM[outs[0][1]['elem']]}, error)"
    else:
        named = ", ".join(f"{n} {GO_ELEM[o['elem']]}" for n, o in outs)
        results = f"({named}, err error)"
    zero_return = "0, err" if single else ", ".join(["0"] * len(outs)) + ", err"
    locals_ = [f"\tvar {n}Out {GO_ELEM[o['elem']]}" for n, o in outs]
    args = [go_param_arg(p) for p in fn["params"]]
    args += [f"(*{C_ELEM[o['elem']]})(unsafe.Pointer(&{n}Out))" for n, o in outs]
    return [
        *go_doc(name, f"wraps ferro_ta_core {fn['group']}::{core}.", fn["doc"]),
        f"func {name}({params}) {results} {{",
        *locals_,
        f'\tif err := statusError("{name}", C.{fn["name"]}({", ".join(args)})); err != nil {{',
        f"\t\treturn {zero_return}",
        "\t}",
        f"\treturn {', '.join(f'{n}Out' for n, _ in outs)}, nil",
        "}",
    ]


def go_stream(stream: dict) -> list[str]:
    name = "Stream" + go_name(stream["name"])
    handle = f"*C.{stream['handle']}"
    outs = [(go_out_name(o["name"]), o) for o in stream["outputs"]]
    single = len(outs) == 1
    if single:
        update_results = f"({GO_ELEM[outs[0][1]['elem']]}, error)"
    else:
        named = ", ".join(f"{n} {GO_ELEM[o['elem']]}" for n, o in outs)
        update_results = f"({named}, err error)"
    update_zero = "0, err" if single else ", ".join(["0"] * len(outs)) + ", err"
    closed_return = update_zero.replace("err", f'closedError("{name}.Update")')
    new_params = ", ".join(
        f"{go_ident(p['name'])} {go_param_type(p)}" for p in stream["params"]
    )
    new_args = [go_param_arg(p) for p in stream["params"]] + ["&h"]
    update_params = ", ".join(f"{go_ident(i)} float64" for i in stream["inputs"])
    update_args = ["s.h"] + [f"C.double({go_ident(i)})" for i in stream["inputs"]]
    update_args += [f"(*{C_ELEM[o['elem']]})(unsafe.Pointer(&{n}))" for n, o in outs]
    return [
        *go_doc(
            name,
            "is a stateful, bar-by-bar indicator backed by a Rust handle.",
            stream["doc"],
        ),
        "//",
        "// Methods are safe for concurrent use. Call Close when done; a finalizer",
        "// releases the handle if Close is forgotten.",
        f"type {name} struct {{",
        "\tmu sync.Mutex",
        f"\th  {handle}",
        "}",
        "",
        f"// New{name} creates a {name}.",
        f"func New{name}({new_params}) (*{name}, error) {{",
        f"\tvar h {handle}",
        f'\tif err := statusError("New{name}", C.{stream["new_fn"]}({", ".join(new_args)})); err != nil {{',
        "\t\treturn nil, err",
        "\t}",
        f"\ts := &{name}{{h: h}}",
        f"\truntime.SetFinalizer(s, (*{name}).Close)",
        "\treturn s, nil",
        "}",
        "",
        "// Update feeds one bar and returns this bar's value(s) (NaN during warm-up).",
        f"func (s *{name}) Update({update_params}) {update_results} {{",
        # Multi-output Update uses named results; a single output needs a local.
        *([f"\tvar {outs[0][0]} {GO_ELEM[outs[0][1]['elem']]}"] if single else []),
        "\ts.mu.Lock()",
        "\tdefer s.mu.Unlock()",
        "\tif s.h == nil {",
        f"\t\treturn {closed_return}",
        "\t}",
        f'\tif err := statusError("{name}.Update", C.{stream["update_fn"]}({", ".join(update_args)})); err != nil {{',
        f"\t\treturn {update_zero}",
        "\t}",
        f"\treturn {', '.join(n for n, _ in outs)}, nil",
        "}",
        "",
        "// Reset clears all state; subsequent bars warm up again.",
        f"func (s *{name}) Reset() error {{",
        "\ts.mu.Lock()",
        "\tdefer s.mu.Unlock()",
        "\tif s.h == nil {",
        f'\t\treturn closedError("{name}.Reset")',
        "\t}",
        f'\treturn statusError("{name}.Reset", C.{stream["reset_fn"]}(s.h))',
        "}",
        "",
        "// Close releases the underlying handle. It is safe to call more than once.",
        f"func (s *{name}) Close() error {{",
        "\ts.mu.Lock()",
        "\tdefer s.mu.Unlock()",
        "\tif s.h != nil {",
        f"\t\tC.{stream['free_fn']}(s.h)",
        "\t\ts.h = nil",
        "\t\truntime.SetFinalizer(s, nil)",
        "\t}",
        "\treturn nil",
        "}",
    ]


def go_file(imports: list[str], blocks: list[list[str]], cgo: bool = True) -> str:
    out = [
        "// Code generated by scripts/build_ffi_bindings.py from ffi_spec.json. DO NOT EDIT.",
        "",
        "package ferrota",
    ]
    if cgo:
        out += ["", "/*", '#include "ferro_ta.h"', "*/", 'import "C"']
    if imports:
        out += ["", "import ("] + [f'\t"{i}"' for i in imports] + [")"]
    for block in blocks:
        out += [""] + block
    return "\n".join(out) + "\n"


def go_param_from_map(param: dict) -> str:
    """Golden-test expression converting a JSON number to the wrapper's param type."""
    key = f'p["{param["name"]}"]'
    if param["kind"] == "enum":
        return f"{param['enum']}({key})"
    return {
        "period": f"int({key})",
        "count": f"int({key})",
        "matype": f"MAType({key})",
        "float": key,
    }[param["kind"]]


def go_golden_function(fn: dict) -> list[str]:
    name = go_name(fn["name"].removeprefix("ft_"))
    args = [f'in["{i}"]' for i in fn["inputs"]] + [
        go_param_from_map(p) for p in fn["params"]
    ]
    outs = [go_out_name(o["name"]) for o in fn["outputs"]]
    results = ", ".join(outs)
    pairs = ", ".join(f'"{o["name"]}": {n}' for o, n in zip(fn["outputs"], outs))
    return [
        f'	"{fn["name"]}": func(in map[string][]float64, p map[string]float64) (map[string]any, error) {{',
        f"		{results}, err := {name}({', '.join(args)})",
        f"		return map[string]any{{{pairs}}}, err",
        "	},",
    ]


def go_golden_scalar(fn: dict) -> list[str]:
    name = go_name(fn["name"].removeprefix("ft_"))
    args = ", ".join(go_param_from_map(p) for p in fn["params"])
    outs = [go_out_name(o["name"]) for o in fn["outputs"]]
    pairs = ", ".join(f'"{o["name"]}": {n}' for o, n in zip(fn["outputs"], outs))
    return [
        f'\t"{fn["name"]}": func(p map[string]float64) (map[string]any, error) {{',
        f"\t\t{', '.join(outs)}, err := {name}({args})",
        f"\t\treturn map[string]any{{{pairs}}}, err",
        "\t},",
    ]


def go_golden_stream(stream: dict) -> list[str]:
    name = "Stream" + go_name(stream["name"])
    ctor_args = ", ".join(go_param_from_map(p) for p in stream["params"])
    bar_args = ", ".join(f'bar["{i}"]' for i in stream["inputs"])
    outs = [go_out_name(o["name"]) for o in stream["outputs"]]
    pairs = ", ".join(f'"{o["name"]}": {n}' for o, n in zip(stream["outputs"], outs))
    return [
        f'	"{stream["name"]}": func(p map[string]float64) (goldenStepper, error) {{',
        f"		s, err := New{name}({ctor_args})",
        "		if err != nil {",
        "			return goldenStepper{}, err",
        "		}",
        "		step := func(bar map[string]float64) (map[string]any, error) {",
        f"			{', '.join(outs)}, err := s.Update({bar_args})",
        f"			return map[string]any{{{pairs}}}, err",
        "		}",
        "		return goldenStepper{step: step, close: s.Close}, nil",
        "	},",
    ]


def render_go_golden_dispatch(spec: dict) -> str:
    out = [
        "// Code generated by scripts/build_ffi_bindings.py from ffi_spec.json. DO NOT EDIT.",
        "",
        "package ferrota",
        "",
        "// goldenFunctions maps each C export to its Go wrapper so golden_test.go",
        "// can replay tests/fixtures/golden/ffi_golden.json by name.",
        "var goldenFunctions = map[string]func(map[string][]float64, map[string]float64) (map[string]any, error){",
    ]
    for fn in spec["functions"]:
        out += go_golden_function(fn)
    out += [
        "}",
        "",
        "// goldenScalars maps each scalar C export to its Go wrapper.",
        "var goldenScalars = map[string]func(map[string]float64) (map[string]any, error){",
    ]
    for fn in spec["scalars"]:
        out += go_golden_scalar(fn)
    out += [
        "}",
        "",
        "// goldenStreams builds each streaming wrapper for golden replay.",
        "var goldenStreams = map[string]func(map[string]float64) (goldenStepper, error){",
    ]
    for stream in spec["streams"]:
        out += go_golden_stream(stream)
    out += ["}", ""]
    return "\n".join(out)


def render_go(spec: dict, go_dir: Path) -> dict[Path, str]:
    groups: dict[str, list[dict]] = {}
    for fn in spec["functions"]:
        groups.setdefault(fn["group"], []).append(fn)
    files = {
        go_dir / f"{group}_gen.go": go_file([], [go_function(fn) for fn in fns])
        for group, fns in groups.items()
    }
    scalar_groups: dict[str, list[dict]] = {}
    for fn in spec["scalars"]:
        scalar_groups.setdefault(fn["group"], []).append(fn)
    for group, fns in scalar_groups.items():
        files[go_dir / f"{group}_scalar_gen.go"] = go_file(
            ["unsafe"], [go_scalar(fn) for fn in fns]
        )
    files[go_dir / "enums_gen.go"] = go_file(
        [], [go_enum(e) for e in spec["enums"]], cgo=False
    )
    files[go_dir / "streaming_gen.go"] = go_file(
        ["runtime", "sync", "unsafe"], [go_stream(s) for s in spec["streams"]]
    )
    files[go_dir / "golden_dispatch_gen_test.go"] = render_go_golden_dispatch(spec)
    files[go_dir / "version_gen.go"] = "\n".join(
        [
            "// Code generated by scripts/build_ffi_bindings.py from ffi_spec.json. DO NOT EDIT.",
            "",
            "package ferrota",
            "",
            "// Version is the ferro_ta release this module wraps.",
            f'const Version = "{spec["version"]}"',
            "",
        ]
    )
    return files
