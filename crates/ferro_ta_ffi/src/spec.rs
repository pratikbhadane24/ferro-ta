//! Machine-readable description of every exported function.
//!
//! cbindgen cannot see functions produced by `macro_rules!` on stable Rust, so
//! the export macros record their own signature here instead. The
//! `dump_spec` example serialises [`all_specs`](crate::all_specs) to
//! `ffi_spec.json`, from which `scripts/build_ffi_bindings.py` generates both
//! `ferro_ta.h` and the per-language wrappers.

/// How a scalar parameter crosses the ABI and how it is validated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    /// `int64_t` window length, must be `>= min`.
    Period { min: i64 },
    /// `int64_t` non-negative count (shifts, displacement).
    Count,
    /// `int32_t` moving-average type, `0..=8`.
    MaType,
    /// `double`, passed through unchanged.
    Float,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemType {
    F64,
    I32,
    I64,
    I8,
}

impl ElemType {
    pub const fn c_name(self) -> &'static str {
        match self {
            ElemType::F64 => "double",
            ElemType::I32 => "int32_t",
            ElemType::I64 => "int64_t",
            ElemType::I8 => "int8_t",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ParamSpec {
    pub name: &'static str,
    pub kind: ParamKind,
}

#[derive(Debug, Clone, Copy)]
pub struct OutputSpec {
    pub name: &'static str,
    pub elem: ElemType,
}

/// One array-in / array-out export: `ft_x(inputs..., len, params..., outputs...)`.
#[derive(Debug, Clone, Copy)]
pub struct FnSpec {
    pub name: &'static str,
    /// Module the export belongs to (`overlap`, `momentum`, ...).
    pub group: &'static str,
    pub doc: &'static str,
    pub inputs: &'static [&'static str],
    pub params: &'static [ParamSpec],
    pub outputs: &'static [OutputSpec],
}
