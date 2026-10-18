//! Declarative registry of streaming indicators generated from batch kernels.
//!
//! Every new stream is **one entry** in a family macro under
//! `streaming/families/`. The same entry drives the core type, the Python,
//! WASM and C ABI (and through it Go) classes, the generated `.pyi` stubs, and
//! the registry tests below — adding an entry is enough to test and ship it.
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
//!   kinds: `period` (`usize`, `>= min`, default min 1), `count` (`usize`),
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
//!   `streaming::fixtures::input_column` for its input names.
//!
//! # Wiring
//!
//! Each family macro (`stream_family_math!` …) holds its entries and forwards
//! them through [`__stream_chain!`]; [`for_each_stream!`]`(gen)` chains every
//! family and calls `gen! { <all entries> }` once, so a generator can emit
//! whole-registry items (a spec table, a `register` function). A new family
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
        Ok::<(), $crate::streaming::StreamingError>(())
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

/// `value >= min` for a `period` param.
pub(crate) fn check_period(value: usize, name: &str, min: usize) -> Result<(), StreamingError> {
    super::validate_timeperiod(value, name, min)
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
mod tests {
    use crate::streaming::fixtures::input_column;
    use crate::streaming::{Live, StreamValue, StreamingIndicator};

    const N_BARS: usize = 600;
    const TICKS_PER_BAR: usize = 4;
    /// Parity tolerance from the streaming contract: `1e-9 · max(1, |b|)`.
    const REL_TOL: f64 = 1e-9;

    /// Snapshot bound, so the same harness compiles with and without `serde`.
    #[cfg(feature = "serde")]
    trait Snap: serde::Serialize + serde::de::DeserializeOwned {}
    #[cfg(feature = "serde")]
    impl<T: serde::Serialize + serde::de::DeserializeOwned> Snap for T {}
    #[cfg(not(feature = "serde"))]
    trait Snap {}
    #[cfg(not(feature = "serde"))]
    impl<T> Snap for T {}

    /// A registry stream behind `Live`, fed rows of `f64` in input order.
    trait DynStream {
        fn update(&mut self, row: &[f64], closed: bool) -> Vec<Option<f64>>;
        #[cfg(feature = "serde")]
        fn save(&self) -> String;
        #[cfg(feature = "serde")]
        fn restore(&self, json: &str) -> Box<dyn DynStream>;
    }

    struct Erased<T: StreamingIndicator> {
        live: Live<T>,
        input_of: fn(&[f64]) -> T::Input,
    }

    impl<T: StreamingIndicator + Snap + 'static> DynStream for Erased<T> {
        fn update(&mut self, row: &[f64], closed: bool) -> Vec<Option<f64>> {
            self.live.update((self.input_of)(row), closed).to_parts()
        }
        #[cfg(feature = "serde")]
        fn save(&self) -> String {
            self.live.save().unwrap()
        }
        #[cfg(feature = "serde")]
        fn restore(&self, json: &str) -> Box<dyn DynStream> {
            Box::new(Erased::<T> {
                live: Live::load(json).unwrap(),
                input_of: self.input_of,
            })
        }
    }

    /// One bar's output as `StreamValue::to_parts`.
    type Parts = Vec<Option<f64>>;

    /// Batch results as per-bar `to_parts` rows.
    fn batch_rows<O: StreamValue, B: crate::streaming::BatchOutput<O>>(b: &B) -> Vec<Parts> {
        (0..b.bars()).map(|i| b.at(i).to_parts()).collect()
    }

    /// One registry entry, type-erased. `scale` multiplies every
    /// `period`-kind parameter (1 = defaults).
    struct Case {
        name: &'static str,
        ty: &'static str,
        kind: &'static str,
        inputs: &'static [&'static str],
        exact: &'static [bool],
        /// Snapshot save points; only the `serde` tests read it.
        #[cfg_attr(not(feature = "serde"), allow(dead_code))]
        lookback: fn(usize) -> usize,
        batch: fn(&[Vec<f64>], usize) -> Vec<Parts>,
        stream: fn(usize) -> Box<dyn DynStream>,
    }

    macro_rules! scaled {
        (period, $default:expr, $scale:expr) => {
            $default * $scale
        };
        ($other:ident, $default:expr, $scale:expr) => {
            $default
        };
    }

    /// Integer outputs must match exactly; `f64` within `REL_TOL`.
    macro_rules! exact {
        (f64) => {
            false
        };
        ($int:ident) => {
            true
        };
    }

    macro_rules! registry_cases {
        ($(
            $name:ident => $Ty:ident, doc $doc:literal,
                params($($p:ident: $kind:ident $(($min:literal))? = $default:tt),* $(,)?),
                inputs($($i:ident),+ $(,)?),
                outputs($($o:ident: $elem:ident),+ $(,)?),
                $imp:ident(lookback = $lb:expr, batch = $batch:expr $(, requires = $req:expr)? $(,)?);
        )*) => {
            #[allow(unused_variables, clippy::redundant_closure_call)]
            fn cases() -> Vec<Case> {
                vec![$(
                    Case {
                        name: stringify!($name),
                        ty: stringify!($Ty),
                        kind: <crate::streaming::$Ty as StreamingIndicator>::KIND,
                        inputs: &[$(stringify!($i)),+],
                        exact: &[$(exact!($elem)),+],
                        lookback: |scale| {
                            $(let $p: $crate::__stream_param_ty!($kind) = scaled!($kind, $default, scale);)*
                            $lb
                        },
                        batch: |cols, scale| {
                            $(let $p: $crate::__stream_param_ty!($kind) = scaled!($kind, $default, scale);)*
                            let slices: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
                            let &[$($i),+] = slices.as_slice() else {
                                panic!("{}: wrong input count", stringify!($name))
                            };
                            batch_rows::<<crate::streaming::$Ty as StreamingIndicator>::Output, _>(&$batch)
                        },
                        stream: |scale| {
                            $(let $p: $crate::__stream_param_ty!($kind) = scaled!($kind, $default, scale);)*
                            let inner = crate::streaming::$Ty::new($($p),*).unwrap();
                            Box::new(Erased {
                                live: Live::new(inner),
                                input_of: |row| {
                                    let mut it = row.iter().copied();
                                    #[allow(unused_parens)]
                                    let input = ($({
                                        let _ = stringify!($i);
                                        it.next().unwrap()
                                    }),+);
                                    input
                                },
                            })
                        },
                    }
                ),*]
            }
        };
    }

    crate::for_each_stream!(registry_cases);

    fn columns(case: &Case) -> Vec<Vec<f64>> {
        case.inputs
            .iter()
            .map(|n| input_column(n, N_BARS))
            .collect()
    }

    fn row(cols: &[Vec<f64>], i: usize) -> Vec<f64> {
        cols.iter().map(|c| c[i]).collect()
    }

    fn assert_parity(case: &Case, bar: usize, got: &[Option<f64>], want: &[Option<f64>]) {
        assert_eq!(
            got.len(),
            want.len(),
            "{} bar {bar}: output arity",
            case.name
        );
        for (k, (g, w)) in got.iter().zip(want).enumerate() {
            let ok = match (g, w) {
                (None, None) => true,
                (Some(g), Some(w)) if case.exact[k] => g == w,
                (Some(g), Some(w)) => (g - w).abs() <= REL_TOL * w.abs().max(1.0),
                _ => false,
            };
            assert!(
                ok,
                "{} bar {bar} output {k}: stream {g:?} != batch {w:?}",
                case.name
            );
        }
    }

    /// Every registry stream matches its batch function at every bar (NaN
    /// mask included), with default params and with every period doubled —
    /// the second run catches a lookback that is one bar short.
    #[test]
    fn registry_parity_all_streams() {
        let cases = cases();
        assert!(!cases.is_empty());
        for case in &cases {
            let cols = columns(case);
            for scale in [1, 2] {
                let want = (case.batch)(&cols, scale);
                assert_eq!(want.len(), N_BARS, "{}: batch length", case.name);
                let mut stream = (case.stream)(scale);
                for (i, w) in want.iter().enumerate() {
                    let got = stream.update(&row(&cols, i), true);
                    assert_parity(case, i, &got, w);
                }
            }
        }
    }

    fn same(a: &[Option<f64>], b: &[Option<f64>]) -> bool {
        format!("{a:?}") == format!("{b:?}")
    }

    /// Previews before every close must not perturb committed state.
    #[test]
    fn previews_are_pure_all_streams() {
        for case in &cases() {
            let cols = columns(case);
            let mut plain = (case.stream)(1);
            let mut live = (case.stream)(1);
            for i in 0..N_BARS {
                let bar = row(&cols, i);
                for t in 0..TICKS_PER_BAR {
                    let wobble = 0.01 * (t as f64 + 1.0);
                    let tick: Vec<f64> = bar
                        .iter()
                        .zip(case.inputs)
                        .map(|(&v, &name)| if name == "periods" { v } else { v + wobble })
                        .collect();
                    let _ = live.update(&tick, false);
                }
                let expected = plain.update(&bar, true);
                let preview = live.update(&bar, false);
                let got = live.update(&bar, true);
                assert!(same(&preview, &expected), "{} bar {i}: preview", case.name);
                assert!(same(&got, &expected), "{} bar {i}: commit", case.name);
            }
        }
    }

    /// Restoring at warm-up edges (empty, one bar short, just full, end)
    /// continues bit-identically.
    #[cfg(feature = "serde")]
    #[test]
    fn snapshot_round_trip_all_streams() {
        for case in &cases() {
            let cols = columns(case);
            let lookback = (case.lookback)(1);
            for save_at in [0, lookback.saturating_sub(1), lookback, N_BARS - 1] {
                let mut original = (case.stream)(1);
                for i in 0..save_at {
                    original.update(&row(&cols, i), true);
                }
                let mut restored = original.restore(&original.save());
                for i in save_at..N_BARS {
                    let a = original.update(&row(&cols, i), true);
                    let b = restored.update(&row(&cols, i), true);
                    assert!(same(&a, &b), "{} saved at {save_at}, bar {i}", case.name);
                }
            }
        }
    }

    /// A windowed snapshot holding more than `2 · lookback` bars is corrupt.
    #[cfg(feature = "serde")]
    #[test]
    fn windowed_rejects_oversized_buffer() {
        use crate::streaming::StreamingSum;
        let mut live = Live::new(StreamingSum::new(3).unwrap());
        for v in [1.0, 2.0, 3.0, 4.0] {
            live.update(v, true);
        }
        let json = live.save().unwrap();
        assert!(Live::<StreamingSum>::load(&json).is_ok());
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let cols = &mut value["state"]["inner"]["cols"];
        assert!(cols.is_array(), "snapshot shape: {json}");
        cols[0] = serde_json::json!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let oversized = value.to_string();
        assert!(Live::<StreamingSum>::load(&oversized).is_err());
    }

    fn upper_camel(snake: &str) -> String {
        snake
            .split('_')
            .filter(|p| !p.is_empty())
            .map(|p| p[..1].to_uppercase() + &p[1..])
            .collect()
    }

    /// Type names follow `Streaming<UpperCamel(name)>` (the WASM, C and Go
    /// generators derive their names from `name` alone), snapshot kinds equal
    /// the name, and names are unique.
    #[test]
    fn registry_names_follow_convention() {
        let cases = cases();
        let mut seen = std::collections::BTreeSet::new();
        for case in &cases {
            assert_eq!(case.ty, format!("Streaming{}", upper_camel(case.name)));
            assert_eq!(case.kind, case.name);
            assert!(seen.insert(case.name), "duplicate stream {}", case.name);
        }
    }

    #[test]
    fn param_kind_checks() {
        use super::{check_float, check_matype, check_period};
        assert!(check_period(2, "p", 2).is_ok() && check_period(1, "p", 2).is_err());
        assert!(check_float(0.5, "f").is_ok() && check_float(f64::NAN, "f").is_err());
        assert!(check_matype(8, "m").is_ok() && check_matype(9, "m").is_err());
    }

    #[test]
    fn invalid_params_are_rejected() {
        use crate::streaming::{StreamingRollingMaxindex, StreamingSum};
        assert!(StreamingSum::new(0).is_err());
        assert!(StreamingRollingMaxindex::new(0).is_err());
        assert_eq!(StreamingSum::new(7).unwrap().timeperiod(), 7);
    }
}
