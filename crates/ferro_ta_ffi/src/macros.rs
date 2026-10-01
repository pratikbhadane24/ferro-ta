//! `ffi_exports!` — declares array-in / array-out C exports over `ferro_ta_core`.
//!
//! ```ignore
//! ffi_exports! {
//!     group: "overlap",
//!     /// Simple Moving Average.
//!     ft_sma = overlap::sma(close)[timeperiod: period 1] -> [out: f64];
//! }
//! ```
//!
//! expands to
//!
//! ```c
//! int32_t ft_sma(const double *close, size_t len, int64_t timeperiod, double *out);
//! ```
//!
//! plus a `SPECS` entry describing it. Every array shares one `len`; scalar
//! params follow in core order, then caller-allocated outputs of length `len`.
//!
//! Param kinds:
//! * `period N`: int64, must be `>= N` (N defaults to 1)
//! * `count`: int64, must be `>= 0`
//! * `matype`: int32, `0..=8`
//! * `float`: double
//!
//! Output element types: `f64`, `i32`, `i64`, `i8`.
//!
//! An optional trailing `if <condition>` adds a cross-parameter rule checked
//! after conversion (e.g. `if long_window > short_window`); a violation returns
//! `FT_ERR_INVALID_PARAM`.

/// C-side type of a param kind.
#[macro_export]
#[doc(hidden)]
macro_rules! ffi_cty {
    (period) => {
        i64
    };
    (count) => {
        i64
    };
    (matype) => {
        i32
    };
    (float) => {
        f64
    };
}

/// Core-side (Rust) type of a param kind.
#[macro_export]
#[doc(hidden)]
macro_rules! ffi_rty {
    (period) => {
        usize
    };
    (count) => {
        usize
    };
    (matype) => {
        u8
    };
    (float) => {
        f64
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! ffi_kind {
    (period) => {
        $crate::spec::ParamKind::Period { min: 1 }
    };
    (period $min:literal) => {
        $crate::spec::ParamKind::Period { min: $min }
    };
    (count) => {
        $crate::spec::ParamKind::Count
    };
    (matype) => {
        $crate::spec::ParamKind::MaType
    };
    (float) => {
        $crate::spec::ParamKind::Float
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! ffi_elem {
    (f64) => {
        $crate::spec::ElemType::F64
    };
    (i32) => {
        $crate::spec::ElemType::I32
    };
    (i64) => {
        $crate::spec::ElemType::I64
    };
    (i8) => {
        $crate::spec::ElemType::I8
    };
}

#[macro_export]
macro_rules! ffi_exports {
    (
        group: $group:literal,
        $(
            $(#[doc = $doc:literal])*
            $name:ident = $($path:ident)::+ ( $($input:ident),+ )
                [ $($param:ident : $kind:ident $($min:literal)?),* ]
                -> [ $($out:ident : $elem:ident),+ ] $(if $requires:expr)?;
        )+
    ) => {
        $(
            $(#[doc = $doc])*
            ///
            /// # Safety
            /// Each input pointer must be valid for `len` reads and each output
            /// pointer valid for `len` writes (pointers may be NULL when
            /// `len == 0`). An output may alias an input for in-place use.
            #[no_mangle]
            pub unsafe extern "C" fn $name(
                $($input: *const f64,)+
                len: usize,
                $($param: $crate::ffi_cty!($kind),)*
                $($out: *mut $elem,)+
            ) -> i32 {
                $crate::status::guard(|| {
                    $(
                        let $param: $crate::ffi_rty!($kind) = $crate::marshal::Param::from_raw(
                            $param,
                            $crate::ffi_kind!($kind $($min)?),
                        )?;
                    )*
                    $(
                        if !($requires) {
                            return Err($crate::status::Status::InvalidParam);
                        }
                    )?
                    $( $crate::marshal::check_output($out, len)?; )+
                    if len == 0 {
                        return Ok(());
                    }
                    // SAFETY: pointer validity for `len` reads is the caller's contract.
                    $( let $input = unsafe { $crate::marshal::input($input, len)? }; )+
                    let result = ferro_ta_core::$($path)::+($($input,)+ $($param,)*);
                    // SAFETY: outputs checked non-null above; valid for `len` writes per contract.
                    unsafe { $crate::marshal::WriteOutputs::write_to(result, ($($out,)+), len) }
                })
            }
        )+

        /// Signature metadata for every export in this module.
        pub(crate) const SPECS: &[$crate::spec::FnSpec] = &[
            $(
                $crate::spec::FnSpec {
                    name: stringify!($name),
                    group: $group,
                    doc: concat!($($doc, "\n",)*),
                    requires: concat!("", $(stringify!($requires))?),
                    inputs: &[$(stringify!($input)),+],
                    params: &[$(
                        $crate::spec::ParamSpec {
                            name: stringify!($param),
                            kind: $crate::ffi_kind!($kind $($min)?),
                        }
                    ),*],
                    outputs: &[$(
                        $crate::spec::OutputSpec {
                            name: stringify!($out),
                            elem: $crate::ffi_elem!($elem),
                        }
                    ),+],
                }
            ),+
        ];
    };
}
