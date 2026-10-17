"""Render Go wrappers for kernel (multi-shape array) exports."""

from __future__ import annotations

from .go import (
    C_ELEM,
    GO_ELEM,
    check_unique_idents,
    go_doc,
    go_ident,
    go_name,
    go_out_name,
    go_param_arg,
    go_param_type,
)


def _dim_expr(dim: dict) -> str:
    return str(dim["const"]) if "const" in dim else go_ident(dim["name"])


def _product(dims: list[dict]) -> str:
    return " * ".join(_dim_expr(d) for d in dims)


def _dim_list(dims: list[dict]) -> str:
    return ", ".join(_dim_expr(d) for d in dims)


def _dim_params(fn: dict) -> list[dict]:
    """Integer params that size an array (guarded before any allocation)."""
    named = {
        d["name"]
        for a in fn["args"]
        if a["role"] in ("in", "out")
        for d in a["dims"]
        if "name" in d
    }
    return [
        a
        for a in fn["args"]
        if a["role"] == "param"
        and a["name"] in named
        and a["kind"] in ("period", "count")
    ]


def _inferred_sizes(fn: dict) -> dict[str, dict]:
    """size name -> first input whose leading dim it is (non-derived only)."""
    derived = {a["name"] for a in fn["args"] if a["role"] == "size" and a["derived"]}
    out: dict[str, dict] = {}
    for arg in fn["args"]:
        if arg["role"] == "in":
            lead = arg["dims"][0].get("name")
            if lead and lead not in derived and lead not in out:
                out[lead] = arg
    return out


def _outputs(fn: dict) -> list[tuple[str, dict]]:
    return [
        (go_out_name(a["name"]), a)
        for a in fn["args"]
        if a["role"] in ("out", "out_scalar")
    ]


def _go_type(arg: dict) -> str:
    elem = GO_ELEM[arg["elem"]]
    return f"[]{elem}" if arg["role"] in ("in", "out") else elem


def kernel_uses_unsafe(fn: dict) -> bool:
    return any(a["role"] == "out_scalar" for a in fn["args"])


def go_kernel(fn: dict) -> list[str]:
    core = fn["name"].removeprefix("ft_")
    name = go_name(core)
    params_by_name = {a["name"]: a for a in fn["args"] if a["role"] == "param"}
    inferred = _inferred_sizes(fn)
    sizes = [a for a in fn["args"] if a["role"] == "size"]
    explicit = [a for a in sizes if not a["derived"] and a["name"] not in inferred]
    outs = _outputs(fn)

    sig_args = []
    for arg in fn["args"]:
        if arg["role"] == "in":
            sig_args.append(f"{go_ident(arg['name'])} {_go_type(arg)}")
        elif arg["role"] == "size" and arg in explicit:
            sig_args.append(f"{go_ident(arg['name'])} int")
        elif arg["role"] == "param":
            sig_args.append(f"{go_ident(arg['name'])} {go_param_type(arg)}")
    check_unique_idents(
        fn["name"],
        [s.split(" ")[0] for s in sig_args]
        + [n for n, _ in outs]
        + [go_ident(a["name"]) for a in sizes if a not in explicit],
    )

    results = ", ".join(f"{n} {_go_type(a)}" for n, a in outs)
    zero = ", ".join("nil" if a["role"] == "out" else "0" for _, a in outs)
    fail = f"return {zero}, err"

    shapes = [
        f"\t{go_ident(a['name']) if a['role'] == 'in' else go_out_name(a['name'])}: "
        f"{_product(a['dims'])} elements (row-major)"
        for a in fn["args"]
        if a["role"] in ("in", "out")
    ]
    doc = fn["doc"] + ("\nShapes:\n\n" + "\n".join(shapes) if shapes else "")
    body = [
        *go_doc(name, f"wraps ferro_ta_core {fn['group']}::{core}.", doc),
        f"func {name}({', '.join(sig_args)}) ({results}, err error) {{",
    ]
    guards = [(go_ident(a["name"]), 0) for a in explicit]
    guards += [(go_ident(a["name"]), a["min"]) for a in _dim_params(fn)]
    for ident, lo in guards:
        body += [
            f"\tif {ident} < {lo} || {ident} > maxDim {{",
            f'\t\terr = statusError("{name}", C.FT_ERR_INVALID_PARAM)',
            f"\t\t{fail}",
            "\t}",
        ]
    for arg in sizes:
        if not arg["derived"]:
            continue
        ident = go_ident(arg["name"])
        via_args = [go_param_arg(params_by_name[p]) for p in arg["derived"]["params"]]
        body += [
            f"\tvar {ident}C C.int64_t",
            f'\tif err = statusError("{name}", C.{arg["derived"]["via"]}({", ".join(via_args + [f"&{ident}C"])})); err != nil {{',
            f"\t\t{fail}",
            "\t}",
            f"\t{ident} := int({ident}C)",
        ]
    for size_name, source in inferred.items():
        ident = go_ident(size_name)
        src = go_ident(source["name"])
        rest = source["dims"][1:]
        if rest:
            body += [
                f'\t{ident}Rest, err := shapeLen("{name}", {_dim_list(rest)})',
                "\tif err != nil {",
                f"\t\t{fail}",
                "\t}",
                f"\t{ident} := 0",
                f"\tif {ident}Rest > 0 {{",
                f"\t\t{ident} = len({src}) / {ident}Rest",
                "\t}",
            ]
        else:
            body.append(f"\t{ident} := len({src})")
    trivially_sized = {id(src) for src in inferred.values() if len(src["dims"]) == 1}
    for arg in fn["args"]:
        if arg["role"] != "in" or id(arg) in trivially_sized:
            continue
        ident = go_ident(arg["name"])
        body += [
            f'\t{ident}Len, err := shapeLen("{name}", {_dim_list(arg["dims"])})',
            "\tif err != nil {",
            f"\t\t{fail}",
            "\t}",
            f"\tif len({ident}) != {ident}Len {{",
            f'\t\terr = shapeError("{name}", "{ident}", len({ident}), {ident}Len)',
            f"\t\t{fail}",
            "\t}",
        ]
    for out_name, arg in outs:
        if arg["role"] == "out":
            body += [
                f'\t{out_name}Len, err := shapeLen("{name}", {_dim_list(arg["dims"])})',
                "\tif err != nil {",
                f"\t\t{fail}",
                "\t}",
                f"\t{out_name}Buf := make([]{GO_ELEM[arg['elem']]}, {out_name}Len)",
            ]
        else:
            body.append(f"\tvar {out_name}Val {GO_ELEM[arg['elem']]}")

    call_args = []
    for arg in fn["args"]:
        ident = go_ident(arg["name"])
        role = arg["role"]
        if role == "in":
            helper = "inPtr" if arg["elem"] == "f64" else "inPtrI64"
            call_args.append(f"{helper}({ident})")
        elif role == "size":
            call_args.append(f"C.size_t({ident})")
        elif role == "param":
            call_args.append(go_param_arg(arg))
        elif role == "out":
            call_args.append(
                f"(*{C_ELEM[arg['elem']]})(outPtr({go_out_name(arg['name'])}Buf))"
            )
        else:
            call_args.append(
                f"(*{C_ELEM[arg['elem']]})(unsafe.Pointer(&{go_out_name(arg['name'])}Val))"
            )
    returns = ", ".join(f"{n}Buf" if a["role"] == "out" else f"{n}Val" for n, a in outs)
    body += [
        f'\tif err = statusError("{name}", C.{fn["name"]}({", ".join(call_args)})); err != nil {{',
        f"\t\t{fail}",
        "\t}",
        f"\treturn {returns}, nil",
        "}",
    ]
    return body
