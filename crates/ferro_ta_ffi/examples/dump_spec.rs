//! Print the machine-readable ABI description as JSON.
//!
//! ```sh
//! cargo run -q -p ferro_ta_ffi --example dump_spec > crates/ferro_ta_ffi/ffi_spec.json
//! ```
//!
//! `scripts/build_ffi_bindings.py` turns this file into `include/ferro_ta.h`
//! and every language's generated wrappers.

use ferro_ta_ffi::spec::{ElemType, OutputSpec, ParamKind, ParamSpec};
use ferro_ta_ffi::{all_specs, all_stream_specs};
use serde_json::{json, Value};

fn param(p: &ParamSpec) -> Value {
    match p.kind {
        ParamKind::Period { min } => {
            json!({"name": p.name, "kind": "period", "c_type": "int64_t", "min": min})
        }
        ParamKind::Count => json!({"name": p.name, "kind": "count", "c_type": "int64_t", "min": 0}),
        ParamKind::MaType => {
            json!({"name": p.name, "kind": "matype", "c_type": "int32_t", "min": 0, "max": 8})
        }
        ParamKind::Float => json!({"name": p.name, "kind": "float", "c_type": "double"}),
    }
}

fn output(o: &OutputSpec) -> Value {
    let elem = match o.elem {
        ElemType::F64 => "f64",
        ElemType::I32 => "i32",
        ElemType::I64 => "i64",
        ElemType::I8 => "i8",
    };
    json!({"name": o.name, "elem": elem, "c_type": o.elem.c_name()})
}

fn doc(raw: &str) -> String {
    raw.lines()
        .map(str::trim)
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn main() {
    let functions: Vec<Value> = all_specs()
        .map(|f| {
            json!({
                "name": f.name,
                "group": f.group,
                "doc": doc(f.doc),
                "requires": f.requires,
                "inputs": f.inputs,
                "params": f.params.iter().map(param).collect::<Vec<_>>(),
                "outputs": f.outputs.iter().map(output).collect::<Vec<_>>(),
            })
        })
        .collect();
    let streams: Vec<Value> = all_stream_specs()
        .map(|s| {
            json!({
                "name": s.name,
                "handle": s.handle,
                "doc": doc(s.doc),
                "new_fn": s.new_fn,
                "update_fn": s.update_fn,
                "reset_fn": s.reset_fn,
                "free_fn": s.free_fn,
                "params": s.params.iter().map(param).collect::<Vec<_>>(),
                "inputs": s.inputs,
                "outputs": s.outputs.iter().map(output).collect::<Vec<_>>(),
            })
        })
        .collect();
    let spec = json!({
        "version": env!("CARGO_PKG_VERSION"),
        "status_codes": [
            {"name": "FT_OK", "value": ferro_ta_ffi::FT_OK},
            {"name": "FT_ERR_NULL_PTR", "value": ferro_ta_ffi::FT_ERR_NULL_PTR},
            {"name": "FT_ERR_INVALID_PARAM", "value": ferro_ta_ffi::FT_ERR_INVALID_PARAM},
            {"name": "FT_ERR_LENGTH_MISMATCH", "value": ferro_ta_ffi::FT_ERR_LENGTH_MISMATCH},
            {"name": "FT_ERR_PANIC", "value": ferro_ta_ffi::FT_ERR_PANIC},
        ],
        "functions": functions,
        "streams": streams,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&spec).expect("spec serialises")
    );
}
