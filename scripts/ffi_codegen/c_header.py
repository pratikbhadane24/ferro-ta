"""Render ``ferro_ta.h`` from the FFI spec."""

from __future__ import annotations

from .common import GENERATED_NOTE, with_requires


def c_doc(doc: str, indent: str = "") -> list[str]:
    if not doc:
        return []
    lines = [f"{indent}/**"]
    lines += [f"{indent} * {line}".rstrip() for line in doc.splitlines()]
    lines.append(f"{indent} */")
    return lines


# Hand-written in crates/ferro_ta_ffi/src/strategy.rs: legs cross the ABI as
# a struct array, which the generated-export spec cannot describe.
STRATEGY_BLOCK = """/* ---- option strategy ------------------------------------------------- */

/* FtLeg.kind values. */
#define FT_LEG_OPTION 0
#define FT_LEG_FUTURE 1
#define FT_LEG_STOCK 2

/**
 * One strategy leg. quantity is signed (+ long, - short). Option fields are
 * ignored for futures/stock, entry_price for options; right uses FT_OPTION_*.
 * Requires: finite values, quantity != 0, multiplier > 0, strike > 0,
 * premium >= 0, expiry_years >= 0, volatility >= 0, entry_price > 0.
 */
typedef struct FtLeg {
    int32_t kind;
    int32_t right;
    double quantity;
    double multiplier;
    double strike;
    double premium;
    double expiry_years;
    double volatility;
    double entry_price;
} FtLeg;

/*
 * Expiry analytics (payoff, breakevens, max profit/loss, probability of
 * profit) need every option leg to share one expiry_years, else
 * FT_ERR_INVALID_PARAM. Spot is confined to [0, inf).
 */
/** Expiry P&L at n_spots spots into out[n_spots]. */
int32_t ft_strategy_payoff(const FtLeg *legs, size_t n_legs, const double *spots, size_t n_spots, double *out);
/** Breakevens ascending; *out_len gets the count (<= 2*n_legs+2); cap too small -> FT_ERR_INVALID_PARAM. */
int32_t ft_strategy_breakevens(const FtLeg *legs, size_t n_legs, double *out, size_t cap, size_t *out_len);
/** Best expiry P&L and lowest spot attaining it; *out_unlimited = 1 (value/spot NaN) if unbounded. */
int32_t ft_strategy_max_profit(const FtLeg *legs, size_t n_legs, double *out_value, double *out_spot, int32_t *out_unlimited);
/** Worst expiry P&L (usually negative); *out_unlimited = 1 (value/spot NaN) if unbounded. */
int32_t ft_strategy_max_loss(const FtLeg *legs, size_t n_legs, double *out_value, double *out_spot, int32_t *out_unlimited);
/** Risk-neutral probability that expiry P&L > 0 (lognormal underlying). */
int32_t ft_strategy_probability_of_profit(const FtLeg *legs, size_t n_legs, double spot, double volatility, double time_years, double rate, double carry, double *out);
/** Black-Scholes greeks summed over legs (each option at its own expiry/vol). */
int32_t ft_strategy_greeks(const FtLeg *legs, size_t n_legs, double spot, double rate, double carry, double *out_delta, double *out_gamma, double *out_vega, double *out_theta, double *out_rho);
/** Mark-to-model P&L, row-major [vol_shift][spot] into out[n_shifts*n_spots]. */
int32_t ft_strategy_scenario_grid(const FtLeg *legs, size_t n_legs, const double *spots, size_t n_spots, const double *vol_shifts, size_t n_shifts, double days_forward, double rate, double carry, double *out);"""


def c_function(fn: dict) -> str:
    args = [f"const double *{name}" for name in fn["inputs"]]
    args.append("size_t len")
    args += [f"{p['c_type']} {p['name']}" for p in fn["params"]]
    args += [f"{o['c_type']} *{o['name']}" for o in fn["outputs"]]
    return f"int32_t {fn['name']}({', '.join(args)});"


def c_stream(stream: dict) -> list[str]:
    handle = stream["handle"]
    new_args = [f"{p['c_type']} {p['name']}" for p in stream["params"]]
    new_args.append(f"{handle} **out_handle")
    update_args = [f"{handle} *handle"]
    update_args += [f"double {name}" for name in stream["inputs"]]
    update_args += [f"{o['c_type']} *{o['name']}" for o in stream["outputs"]]
    return [
        *c_doc(stream["doc"]),
        f"typedef struct {handle} {handle};",
        f"int32_t {stream['new_fn']}({', '.join(new_args)});",
        f"int32_t {stream['update_fn']}({', '.join(update_args)});",
        f"int32_t {stream['reset_fn']}({handle} *handle);",
        f"void {stream['free_fn']}({handle} *handle);",
    ]


def c_scalar(fn: dict) -> str:
    args = [f"{p['c_type']} {p['name']}" for p in fn["params"]]
    args += [f"{o['c_type']} *{o['name']}" for o in fn["outputs"]]
    return f"int32_t {fn['name']}({', '.join(args)});"


def dims_text(dims: list[dict]) -> str:
    """`[{"name": "n"}, {"const": 4}]` -> `n * 4`."""
    return " * ".join(str(d["const"]) if "const" in d else d["name"] for d in dims)


def kernel_doc(fn: dict) -> str:
    """Kernel doc plus the element count of every array argument."""
    shapes = [
        f"  {a['name']}[{dims_text(a['dims'])}]"
        + (" (optional: may be NULL)" if a.get("optional") else "")
        for a in fn["args"]
        if a["role"] in ("in", "out")
    ]
    derived = [
        f"  {a['name']} = {a['derived']['via']}({', '.join(a['derived']['params'])})"
        for a in fn["args"]
        if a["role"] == "size" and a["derived"]
    ]
    out = fn["doc"]
    if shapes:
        out += "\nShapes (row-major element counts):\n" + "\n".join(shapes)
    if derived:
        out += "\nSizes computed by a helper:\n" + "\n".join(derived)
    return out


def c_kernel_arg(arg: dict) -> str:
    role = arg["role"]
    if role == "in":
        return f"const {arg['c_type']} *{arg['name']}"
    if role in ("out", "out_scalar"):
        return f"{arg['c_type']} *{arg['name']}"
    return f"{arg['c_type']} {arg['name']}"


def c_kernel(fn: dict) -> str:
    return f"int32_t {fn['name']}({', '.join(c_kernel_arg(a) for a in fn['args'])});"


def c_enums(spec: dict) -> list[str]:
    out: list[str] = []
    for enum in spec["enums"]:
        out += ["", f"/* {enum['name']} values (int32_t parameters of that kind). */"]
        out += [
            f"#define {enum['c_prefix']}_{v['name']} {v['value']}"
            for v in enum["values"]
        ]
    return out


def render_header(spec: dict) -> str:
    out = [
        f"/* {GENERATED_NOTE} */",
        "",
        "/*",
        f" * ferro_ta C API v{spec['version']}",
        " *",
        " * Conventions:",
        " *  - Every function returns an int32_t status (FT_OK == 0); see ft_status_message().",
        " *  - Array inputs are `const double *` sharing one `size_t len`.",
        " *  - Outputs are caller-allocated arrays of length `len`, written only on success.",
        " *    An output may alias an input (in-place). Any array may be NULL when len == 0.",
        " *  - Periods are int64_t in [minimum, 2^24], MA types int32_t in 0..8.",
        " *  - Cross-parameter rules (e.g. fastperiod < slowperiod) are listed as",
        " *    'Requires:' on each function; violations return FT_ERR_INVALID_PARAM.",
        " *  - Float parameters must be finite; enum parameters must be a listed value.",
        " *  - Scalar functions write each result through an output pointer. A NaN",
        " *    result with FT_OK means the inputs are outside the model's domain",
        " *    (e.g. spot <= 0) or a solver found no solution.",
        " *  - Leading warm-up values are NaN; output length always equals input length.",
        " *  - Kernel functions take arrays of different shapes: each array's row-major",
        " *    element count is listed under 'Shapes:' as a product of size_t / integer",
        " *    arguments. 'Derived sizes' must equal the named helper's result.",
        " *  - Streaming handles are not thread-safe; free each exactly once. After",
        " *    FT_ERR_PANIC from an update the handle's state is unspecified: reset it.",
        " */",
        "",
        "#ifndef FERRO_TA_H",
        "#define FERRO_TA_H",
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        f'#define FT_VERSION "{spec["version"]}"',
        "",
    ]
    out += [f"#define {s['name']} {s['value']}" for s in spec["status_codes"]]
    out += [
        "",
        "/** Static description of a status code. Never NULL; do not free. */",
        "const char *ft_status_message(int32_t code);",
        "/** Library version string. Static; do not free. */",
        "const char *ft_version(void);",
        *c_enums(spec),
    ]
    group = None
    for fn in spec["functions"]:
        if fn["group"] != group:
            group = fn["group"]
            out += ["", f"/* ---- {group} " + "-" * (70 - len(group)) + " */"]
        out += ["", *c_doc(with_requires(fn)), c_function(fn)]
    group = None
    for fn in spec["scalars"]:
        if fn["group"] != group:
            group = fn["group"]
            out += ["", f"/* ---- {group} (scalar) " + "-" * (61 - len(group)) + " */"]
        out += ["", *c_doc(fn["doc"]), c_scalar(fn)]
    group = None
    for fn in spec.get("kernels", []):
        if fn["group"] != group:
            group = fn["group"]
            out += ["", f"/* ---- {group} (kernel) " + "-" * (61 - len(group)) + " */"]
        out += ["", *c_doc(kernel_doc(fn)), c_kernel(fn)]
    out += ["", "/* ---- streaming " + "-" * 61 + " */"]
    for stream in spec["streams"]:
        out += ["", *c_stream(stream)]
    out += ["", STRATEGY_BLOCK]
    out += [
        "",
        "#ifdef __cplusplus",
        '}  /* extern "C" */',
        "#endif",
        "",
        "#endif  /* FERRO_TA_H */",
        "",
    ]
    return "\n".join(out)
