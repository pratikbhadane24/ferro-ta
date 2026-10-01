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
    (option_kind) => {
        i32
    };
    (digital_kind) => {
        i32
    };
    (pricing_model) => {
        i32
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
    (option_kind) => {
        ferro_ta_core::options::OptionKind
    };
    (digital_kind) => {
        ferro_ta_core::options::digital::DigitalKind
    };
    (pricing_model) => {
        ferro_ta_core::options::PricingModel
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
    (option_kind) => {
        $crate::spec::ParamKind::Enum(&$crate::enums::OPTION_KIND)
    };
    (digital_kind) => {
        $crate::spec::ParamKind::Enum(&$crate::enums::DIGITAL_KIND)
    };
    (pricing_model) => {
        $crate::spec::ParamKind::Enum(&$crate::enums::PRICING_MODEL)
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
            // `!(rule)` is deliberate: for float rules such as `sigma > 0.0` it
            // also rejects NaN, which `sigma <= 0.0` would let through.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
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

/// `ffi_scalar_exports!` — scalar-in / scalar-out C exports.
///
/// ```ignore
/// ffi_scalar_exports! {
///     group: "futures",
///     /// Futures basis: futures - spot.
///     ft_basis(spot: float, future: float) -> [out_value: f64]
///         = futures::basis::basis(spot, future);
/// }
/// ```
///
/// expands to `int32_t ft_basis(double spot, double future, double *out_value);`.
/// The body is any expression over the (validated, converted) params; it may
/// flatten a struct into a tuple or use `?` on a `Result<_, Status>`.
/// Enum param kinds: `option_kind`, `digital_kind`, `pricing_model`.
#[macro_export]
macro_rules! ffi_scalar_exports {
    (
        group: $group:literal,
        $(
            $(#[doc = $doc:literal])*
            $name:ident ( $($param:ident : $kind:ident $($min:literal)?),* )
                -> [ $($out:ident : $elem:ident),+ ] = $body:expr;
        )+
    ) => {
        $(
            $(#[doc = $doc])*
            ///
            /// # Safety
            /// Each output pointer must be valid for one write.
            #[no_mangle]
            pub unsafe extern "C" fn $name(
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
                    if $( $out.is_null() )||+ {
                        return Err($crate::status::Status::NullPtr);
                    }
                    #[allow(clippy::redundant_closure_call)]
                    let result = $body;
                    // SAFETY: outputs checked non-null; one write each per contract.
                    unsafe { $crate::marshal::WriteScalars::write_to(result, ($($out,)+)) };
                    Ok(())
                })
            }
        )+

        /// Signature metadata for every scalar export in this module.
        pub(crate) const SCALAR_SPECS: &[$crate::spec::ScalarSpec] = &[
            $(
                $crate::spec::ScalarSpec {
                    name: stringify!($name),
                    group: $group,
                    doc: concat!($($doc, "\n",)*),
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
