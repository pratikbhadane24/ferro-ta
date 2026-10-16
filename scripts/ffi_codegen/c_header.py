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


# Hand-written in crates/ferro_ta_ffi/src/bars.rs: timestamped input and an
# optional struct output don't fit the generated stream shape.
BAR_BUILDER_BLOCK = """/* ---- bar builder ----------------------------------------------------- */

/* ft_bar_builder_on_tick status codes (int32_t). */
#define FT_TICK_ACCEPTED 0
#define FT_TICK_LATE 1
#define FT_TICK_DUPLICATE 2
#define FT_TICK_OUT_OF_SESSION 3
#define FT_TICK_INVALID 4

/** One OHLCV bar covering [start_ms, end_ms) in UTC epoch milliseconds. */
typedef struct FtBar {
    int64_t start_ms;
    int64_t end_ms;
    double open;
    double high;
    double low;
    double close;
    double volume;
    uint32_t ticks;
} FtBar;

/**
 * Session-aligned tick -> bar builder: bars start at session_open + k*interval
 * (last bar cut at session_close); gaps produce no bars. Session times are
 * seconds after local midnight; local = UTC + utc_offset_secs.
 * Requires: 0 < interval_secs <= 86400; 0 <= session_open_secs <
 * session_close_secs <= 86400; |utc_offset_secs| <= 50400.
 */
typedef struct FtBarBuilder FtBarBuilder;
int32_t ft_bar_builder_new(int64_t interval_secs, int64_t session_open_secs, int64_t session_close_secs, int64_t utc_offset_secs, FtBarBuilder **out_handle);
/** Feed a tick; *out_status is FT_TICK_*; *out_closed is 1 with *out_bar set when it closed a bar. */
int32_t ft_bar_builder_on_tick(FtBarBuilder *handle, int64_t ts_ms, double price, double qty, int32_t *out_status, int32_t *out_closed, FtBar *out_bar);
/** Close the forming bar once now_ms reaches its end; *out_closed is 1 with *out_bar set if so. */
int32_t ft_bar_builder_on_time(FtBarBuilder *handle, int64_t now_ms, int32_t *out_closed, FtBar *out_bar);
/** The bar being built: *out_has is 1 with *out_bar set, else 0. */
int32_t ft_bar_builder_forming(FtBarBuilder *handle, int32_t *out_has, FtBar *out_bar);
int32_t ft_bar_builder_reset(FtBarBuilder *handle);
void ft_bar_builder_free(FtBarBuilder *handle);"""


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
        f"int32_t {stream['peek_fn']}({', '.join(update_args)});",
        f"int32_t {stream['reset_fn']}({handle} *handle);",
        f"void {stream['free_fn']}({handle} *handle);",
    ]


def c_scalar(fn: dict) -> str:
    args = [f"{p['c_type']} {p['name']}" for p in fn["params"]]
    args += [f"{o['c_type']} *{o['name']}" for o in fn["outputs"]]
    return f"int32_t {fn['name']}({', '.join(args)});"


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
    out += ["", "/* ---- streaming " + "-" * 61 + " */"]
    for stream in spec["streams"]:
        out += ["", *c_stream(stream)]
    out += ["", BAR_BUILDER_BLOCK]
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
