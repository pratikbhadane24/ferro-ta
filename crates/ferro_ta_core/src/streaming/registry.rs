//! Declarative registry of streaming indicators generated from batch kernels.
//!
//! Every new stream is **one entry** in a family macro under
//! `streaming/families/`. The same entry drives the core type, the Python,
//! WASM and C ABI (and through it Go) classes, the generated `.pyi` stubs, and
//! the registry tests (`registry/tests.rs`) — adding an entry is enough to test and ship it.
//! The 14 hand-written legacy streams (`StreamingSMA` …) are not in the
//! registry.
//!
//! # Entry grammar
//!
//! ```text
//! <name> => <Type>, doc "<one-line doc>",
//!     params(<p>: <kind>[(<min>)] = <default>, ...),
//!     inputs(<input>, ...),
//!     outputs(<out>: <elem>, ...),
//!     windowed(lookback = <expr>, batch = <expr> [, requires = <expr>]);
//! | ...
//!     native(lookback = <expr>, batch = <expr>);
//! ```
//!
//! * `<name>` — the batch function name without `ft_` (`rolling_maxindex`).
//!   It is the snapshot kind and the C stem (`ft_stream_<name>_new`, handle
//!   `FtStream<UpperCamel>`).
//! * `<Type>` — exactly `Streaming<UpperCamel(name)>` (`StreamingRollingMaxindex`);
//!   the WASM class is `Wasm<Type>`. `registry_names_follow_convention` checks it.
//! * `params` / `inputs` / `outputs` — the names, order and kinds of the batch
//!   function in `crates/ferro_ta_ffi/ffi_spec.json` (`ft_<name>`). Param
//!   kinds: `period` (`usize`, `min..=MAX_STREAM_PERIOD`, default min 1),
//!   `count` (`usize`, `<= MAX_STREAM_PERIOD`; both caps also apply on
//!   snapshot restore),
//!   `float` (`f64`, finite), `matype` (`u8`, `0..=8`) — the `ffi_kind!`
//!   tokens. `<default>` is the Python default as one token tree: a bare
//!   literal (`30`, `2.0`; PyO3 renders it in the signature) or, if negative,
//!   parenthesised (`(-1.0)`). Output elems: `f64`, `i32`,
//!   `i64`, `i8`; inputs are always `f64`.
//! * `lookback` / `batch` / `requires` are expressions in which every param
//!   is a local of its Rust type and (in `batch`) every input is a `&[f64]`,
//!   all by name. Use absolute `$crate::` paths. `batch` is the batch
//!   reference: it returns `Vec<elem>` for one output, a tuple of `Vec<f64>`
//!   for several.
//! * `windowed` — the stream is `Windowed<<name>::Kernel>`: `batch` is replayed
//!   on the last `lookback` bars and its last element emitted. `lookback` must
//!   cover every bar the batch value depends on; the parity test runs with
//!   doubled periods to catch one that is short. Optional `requires` is a
//!   cross-parameter rule (e.g. `fastperiod < slowperiod`).
//! * `native` — `<Type>` is a hand-written struct in the family file with
//!   `new(params...) -> Result<Self, StreamingError>`,
//!   `update(&mut self, inputs...) -> Output`, `reset`, `validate`, and a
//!   getter per param; its serde derive lives next to it. `batch` and
//!   `lookback` are used only by the tests.
//! * Entries end with `;`. Every stream also needs an input column mapping in
//!   `streaming::fixtures::input_column` for its input names. Param names
//!   must not collide with generated members and stream names with
//!   `streaming` submodules (`registry_names_do_not_collide`).
//!
//! # Wiring
//!
//! Each family macro (`stream_family_math!` …) holds its entries and forwards
//! them through [`__stream_chain!`]; [`for_each_stream!`]`(gen)` chains every
//! family and calls `gen! { <all entries> }` once, so a generator can emit
//! whole-registry items (a spec table, a `register` function). Invoke it in
//! item position (the generator should emit items, e.g. a `const`). A new family
//! file adds `$crate::stream_family_<x>;` to the `rest` list in
//! `for_each_stream!` and invokes itself with `__stream_define` (see
//! `families/math.rs`).

use super::StreamingError;

/// Call `$gen! { <every registry entry> }` once.
#[macro_export]
macro_rules! for_each_stream {
    ($gen:path) => {
        $crate::__stream_chain! {
            []
            gen = $gen;
            acc = [];
            rest = [$crate::stream_family_math;];
        }
    };
}

/// Accumulates family entries, then hands all of them to the generator.
#[macro_export]
#[doc(hidden)]
macro_rules! __stream_chain {
    (
        [$($new:tt)*]
        gen = $gen:path;
        acc = [$($acc:tt)*];
        rest = [$next:path; $($more:path;)*];
    ) => {
        $next! {
            gen = $gen;
            acc = [$($acc)* $($new)*];
            rest = [$($more;)*];
        }
    };
    (
        [$($new:tt)*]
        gen = $gen:path;
        acc = [$($acc:tt)*];
        rest = [];
    ) => {
        $gen! { $($acc)* $($new)* }
    };
}

/// Rust type of a param kind (same mapping as the C ABI's `ffi_rty!`).
#[macro_export]
#[doc(hidden)]
macro_rules! __stream_param_ty {
    (period) => {
        usize
    };
    (count) => {
        usize
    };
    (float) => {
        f64
    };
    (matype) => {
        u8
    };
}

/// `f64` per input, as one value or a tuple.
#[macro_export]
#[doc(hidden)]
macro_rules! __stream_input_ty {
    ($one:ident) => {
        f64
    };
    ($($i:ident),+) => {
        ($($crate::__stream_input_ty!($i)),+)
    };
}

/// Output elems as one value or a tuple.
#[macro_export]
#[doc(hidden)]
macro_rules! __stream_output_ty {
    ($one:ident) => {
        $one
    };
    ($($e:ident),+) => {
        ($($e),+)
    };
}

/// Validate one param by kind.
#[macro_export]
#[doc(hidden)]
macro_rules! __stream_check_param {
    (period; $v:expr, $name:expr) => {
        $crate::streaming::registry::check_period($v, $name, 1)
    };
    (period($min:literal); $v:expr, $name:expr) => {
        $crate::streaming::registry::check_period($v, $name, $min)
    };
    (count; $v:expr, $name:expr) => {
        $crate::streaming::registry::check_count($v, $name)
    };
    (float; $v:expr, $name:expr) => {
        $crate::streaming::registry::check_float($v, $name)
    };
    (matype; $v:expr, $name:expr) => {
        $crate::streaming::registry::check_matype($v, $name)
    };
}

/// Core generator: defines the types for each entry (invoked per family).
#[macro_export]
#[doc(hidden)]
macro_rules! __stream_define {
    ($(
        $name:ident => $Ty:ident, doc $doc:literal,
            params($($p:ident: $kind:ident $(($min:literal))? = $default:tt),* $(,)?),
            inputs($($i:ident),+ $(,)?),
            outputs($($o:ident: $elem:ident),+ $(,)?),
            $imp:ident($($body:tt)*);
    )*) => {
        $(
            $crate::__stream_define_one! {
                $imp($($body)*)
                $name => $Ty, doc $doc,
                params($($p: $kind $(($min))?),*),
                inputs($($i),+),
                outputs($($elem),+)
            }
        )*
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! __stream_define_one {
    (
        windowed(lookback = $lb:expr, batch = $batch:expr $(, requires = $req:expr)? $(,)?)
        $name:ident => $Ty:ident, doc $doc:literal,
        params($($p:ident: $kind:ident $(($min:literal))?),*),
        inputs($($i:ident),+),
        outputs($($elem:ident),+)
    ) => {
        #[doc(hidden)]
        pub mod $name {
            #[doc = concat!("Parameters of [`", stringify!($Ty), "`](super::", stringify!($Ty), ").")]
            #[derive(Clone, Debug, PartialEq)]
            #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
            pub struct Params {
                $(pub $p: $crate::__stream_param_ty!($kind),)*
            }

            /// Batch kernel replayed by the windowed stream.
            #[derive(Clone, Debug)]
            pub struct Kernel;

            impl $crate::streaming::WindowKernel for Kernel {
                const KIND: &'static str = stringify!($name);
                type Params = Params;
                type Input = $crate::__stream_input_ty!($($i),+);
                type Output = $crate::__stream_output_ty!($($elem),+);

                #[allow(unused_variables)]
                fn lookback(params: &Params) -> usize {
                    $(let $p = params.$p;)*
                    $lb
                }

                #[allow(unused_variables)]
                fn validate(params: &Params) -> Result<(), $crate::streaming::StreamingError> {
                    $($crate::__stream_check_param!($kind $(($min))?; params.$p, stringify!($p))?;)*
                    $(let $p = params.$p;)*
                    $(
                        if !($req) {
                            return Err($crate::streaming::StreamingError(format!(
                                "{}: parameters violate {}",
                                stringify!($name),
                                stringify!($req)
                            )));
                        }
                    )?
                    Ok(())
                }

                #[allow(unused_variables)]
                fn last(params: &Params, window: &[&[f64]]) -> Self::Output {
                    $(let $p = params.$p;)*
                    let &[$($i),+] = window else {
                        unreachable!("{}: window has {} inputs", stringify!($name), window.len())
                    };
                    let out = $batch;
                    $crate::streaming::BatchOutput::<Self::Output>::last_bar(&out)
                }
            }
        }

        #[doc = $doc]
        pub type $Ty = $crate::streaming::Windowed<$name::Kernel>;

        impl $Ty {
            pub fn new(
                $($p: $crate::__stream_param_ty!($kind)),*
            ) -> Result<Self, $crate::streaming::StreamingError> {
                Self::from_params($name::Params { $($p),* })
            }

            $(
                pub fn $p(&self) -> $crate::__stream_param_ty!($kind) {
                    self.params().$p
                }
            )*
        }
    };
    (
        native(lookback = $lb:expr, batch = $batch:expr $(,)?)
        $name:ident => $Ty:ident, doc $doc:literal,
        params($($p:ident: $kind:ident $(($min:literal))?),*),
        inputs($($i:ident),+),
        outputs($($elem:ident),+)
    ) => {
        impl $crate::streaming::StreamingIndicator for $Ty {
            const KIND: &'static str = stringify!($name);
            type Input = $crate::__stream_input_ty!($($i),+);
            type Output = $crate::__stream_output_ty!($($elem),+);

            fn update(&mut self, input: Self::Input) -> Self::Output {
                #[allow(unused_parens)]
                let ($($i),+) = input;
                <$Ty>::update(self, $($i),+)
            }

            fn reset(&mut self) {
                <$Ty>::reset(self)
            }

            fn validate(&self) -> Result<(), $crate::streaming::StreamingError> {
                <$Ty>::validate(self)
            }
        }
    };
}

/// Largest `period` / `count` a registry stream accepts — the C ABI's
/// `MAX_PERIOD`. Checked at construction *and* on snapshot restore, so an
/// edited snapshot cannot smuggle in a window no constructor would allow.
pub const MAX_STREAM_PERIOD: usize = 1 << 24;

fn check_cap(value: usize, name: &str) -> Result<(), StreamingError> {
    if value > MAX_STREAM_PERIOD {
        return Err(StreamingError(format!(
            "{name} must be <= {MAX_STREAM_PERIOD}, got {value}"
        )));
    }
    Ok(())
}

/// `min <= value <= MAX_STREAM_PERIOD` for a `period` param.
pub(crate) fn check_period(value: usize, name: &str, min: usize) -> Result<(), StreamingError> {
    super::validate_timeperiod(value, name, min)?;
    check_cap(value, name)
}

/// `value <= MAX_STREAM_PERIOD` for a `count` param.
#[allow(dead_code)] // first `count` param arrives with a later family
pub(crate) fn check_count(value: usize, name: &str) -> Result<(), StreamingError> {
    check_cap(value, name)
}

/// Finite `float` param.
#[allow(dead_code)] // first `float` param arrives with a later family
pub(crate) fn check_float(value: f64, name: &str) -> Result<(), StreamingError> {
    super::validate_finite(value, name)
}

/// `matype` param in `0..=8`.
#[allow(dead_code)] // first `matype` param arrives with a later family
pub(crate) fn check_matype(value: u8, name: &str) -> Result<(), StreamingError> {
    const MAX_MATYPE: u8 = 8;
    if value > MAX_MATYPE {
        return Err(StreamingError(format!(
            "{name} must be in 0..={MAX_MATYPE}, got {value}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
