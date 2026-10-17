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
    /// `int64_t` RNG seed: any value, reinterpreted as `u64` (never sizes
    /// anything, so no cap).
    Seed,
    /// `int32_t` moving-average type, `0..=8`.
    MaType,
    /// `double`, must be finite.
    Float,
    /// `int32_t` value of a C enum (see [`EnumSpec`]).
    Enum(&'static EnumSpec),
}

/// A C enum: `#define <c_prefix>_<NAME> <value>` in the header and a typed
/// constant set in each language binding.
#[derive(Debug, PartialEq, Eq)]
pub struct EnumSpec {
    /// Type name in bindings (`OptionKind`).
    pub name: &'static str,
    /// Prefix of the C constants (`FT_OPTION` -> `FT_OPTION_CALL`).
    pub c_prefix: &'static str,
    pub values: &'static [(&'static str, i32)],
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
    /// Cross-parameter rule (Rust expression text), or `""` if none.
    pub requires: &'static str,
    pub inputs: &'static [&'static str],
    pub params: &'static [ParamSpec],
    pub outputs: &'static [OutputSpec],
}

/// One streaming (stateful, bar-by-bar) indicator exposed as an opaque handle:
///
/// ```c
/// int32_t ft_stream_x_new(params..., FtStreamX **out_handle);
/// int32_t ft_stream_x_update(FtStreamX *h, inputs..., outputs...);
/// int32_t ft_stream_x_reset(FtStreamX *h);
/// void    ft_stream_x_free(FtStreamX *h);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct StreamSpec {
    /// Short name (`sma`, `bbands`, ...).
    pub name: &'static str,
    /// Opaque C handle type name (`FtStreamSma`).
    pub handle: &'static str,
    pub doc: &'static str,
    pub new_fn: &'static str,
    pub update_fn: &'static str,
    pub reset_fn: &'static str,
    pub free_fn: &'static str,
    pub params: &'static [ParamSpec],
    /// Scalar `double` inputs to `update`.
    pub inputs: &'static [&'static str],
    /// Scalar outputs written by `update` through pointers.
    pub outputs: &'static [OutputSpec],
}

/// One scalar export: `ft_x(params..., outputs...)` with every output a single
/// value written through a pointer (struct results are flattened).
#[derive(Debug, Clone, Copy)]
pub struct ScalarSpec {
    pub name: &'static str,
    pub group: &'static str,
    pub doc: &'static str,
    pub params: &'static [ParamSpec],
    pub outputs: &'static [OutputSpec],
}

/// One dimension of a kernel array: a named size/integer param, or a constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dim {
    /// Name of a `size` argument or integer (`period` / `count`) param.
    Named(&'static str),
    Const(usize),
}

/// A `size` argument that bindings compute by calling a scalar export
/// (`via(params...) -> [out_value: i64]`) instead of asking the caller.
#[derive(Debug, Clone, Copy)]
pub struct DerivedSize {
    pub via: &'static str,
    pub params: &'static [&'static str],
}

/// One argument of a kernel export, in C declaration order.
#[derive(Debug, Clone, Copy)]
pub enum KernelArg {
    /// `const T *name`; element count is the product of `dims`. An
    /// `optional` input may be NULL, meaning "not supplied".
    In {
        name: &'static str,
        elem: ElemType,
        dims: &'static [Dim],
        optional: bool,
    },
    /// `size_t name`. Bindings infer it from the length of the first input
    /// whose leading dim it is, derive it via `derived`, or take it from the
    /// caller.
    Size {
        name: &'static str,
        derived: Option<DerivedSize>,
    },
    /// Validated scalar parameter (same kinds as array exports).
    Param(ParamSpec),
    /// Caller-allocated `T *name` of `product(dims)` elements, written only on
    /// success.
    Out {
        name: &'static str,
        elem: ElemType,
        dims: &'static [Dim],
    },
    /// `T *name` written once on success.
    OutScalar { name: &'static str, elem: ElemType },
}

/// A kernel export: arrays of different (row-major, multi-dimensional)
/// shapes, explicit sizes, params and scalar results in one call:
/// `int32_t ft_x(args...)`.
#[derive(Debug, Clone, Copy)]
pub struct KernelSpec {
    pub name: &'static str,
    pub group: &'static str,
    pub doc: &'static str,
    pub args: &'static [KernelArg],
}
