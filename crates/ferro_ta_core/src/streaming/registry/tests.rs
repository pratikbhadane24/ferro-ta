//! Registry-driven tests: every entry is checked against its batch reference.
use crate::streaming::fixtures::{input_column, pattern_fixture, PATTERN_BARS};
use crate::streaming::{Live, StreamValue, StreamingIndicator};

const N_BARS: usize = 600;
/// Fires every candlestick pattern must show on `pattern_fixture`.
const MIN_PATTERN_FIRES: usize = 3;
/// Candlestick streams in the registry.
const PATTERN_STREAMS: usize = 61;
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
    /// `Live::load` of a possibly corrupt snapshot (`None` on `Err`).
    #[cfg(feature = "serde")]
    fn try_restore(&self, json: &str) -> Option<Box<dyn DynStream>>;
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
    #[cfg(feature = "serde")]
    fn try_restore(&self, json: &str) -> Option<Box<dyn DynStream>> {
        let live = Live::<T>::load(json).ok()?;
        Some(Box::new(Erased::<T> {
            live,
            input_of: self.input_of,
        }))
    }
}

/// One bar's output as `StreamValue::to_parts`.
type Parts = Vec<Option<f64>>;

/// Batch results as per-bar `to_parts` rows.
fn batch_rows<O: StreamValue, B: crate::streaming::BatchOutput<O>>(b: &B) -> Vec<Parts> {
    (0..b.bars()).map(|i| b.at(i).to_parts()).collect()
}

/// Parameter set a harness runs: `scale` multiplies every `period`-kind
/// parameter, `matype` (if set) replaces every `matype`-kind parameter;
/// everything else stays at its default.
#[derive(Clone, Copy, Debug)]
struct Variant {
    scale: usize,
    matype: Option<u8>,
}

const DEFAULTS: Variant = Variant {
    scale: 1,
    matype: None,
};

/// One registry entry, type-erased.
struct Case {
    name: &'static str,
    ty: &'static str,
    kind: &'static str,
    params: &'static [&'static str],
    inputs: &'static [&'static str],
    exact: &'static [bool],
    /// Output element types (`f64`, `i8`, ...), as written in the entry.
    elems: &'static [&'static str],
    /// Has a `matype`-kind parameter (swept over `0..=8`).
    has_matype: bool,
    /// Snapshot save points; only the `serde` tests read it.
    #[cfg_attr(not(feature = "serde"), allow(dead_code))]
    lookback: fn(Variant) -> usize,
    batch: fn(&[Vec<f64>], Variant) -> Vec<Parts>,
    stream: fn(Variant) -> Box<dyn DynStream>,
}

impl Case {
    /// Defaults, then (for `matype` streams) every `matype` at scale 1.
    fn variants(&self) -> Vec<Variant> {
        let sweep = (0..=8u8).map(|m| Variant {
            scale: 1,
            matype: Some(m),
        });
        std::iter::once(DEFAULTS)
            .chain(sweep.filter(|_| self.has_matype))
            .collect()
    }
}

/// A parameter's value under a [`Variant`].
macro_rules! param {
    (period, $default:expr, $v:expr) => {
        $default * $v.scale
    };
    (matype, $default:expr, $v:expr) => {
        $v.matype.unwrap_or($default)
    };
    ($other:ident, $default:expr, $v:expr) => {
        $default
    };
}

macro_rules! is_matype {
    (matype) => {
        true
    };
    ($other:ident) => {
        false
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
                    params: &[$(stringify!($p)),*],
                    inputs: &[$(stringify!($i)),+],
                    exact: &[$(exact!($elem)),+],
                    elems: &[$(stringify!($elem)),+],
                    has_matype: false $(|| is_matype!($kind))*,
                    lookback: |variant| {
                        $(let $p: $crate::__stream_param_ty!($kind) = param!($kind, $default, variant);)*
                        $lb
                    },
                    batch: |cols, variant| {
                        $(let $p: $crate::__stream_param_ty!($kind) = param!($kind, $default, variant);)*
                        let slices: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
                        let &[$($i),+] = slices.as_slice() else {
                            panic!("{}: wrong input count", stringify!($name))
                        };
                        batch_rows::<<crate::streaming::$Ty as StreamingIndicator>::Output, _>(&$batch)
                    },
                    stream: |variant| {
                        $(let $p: $crate::__stream_param_ty!($kind) = param!($kind, $default, variant);)*
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

/// First mismatch between a stream bar and its batch bar, if any.
fn parity_error(
    case: &Case,
    bar: usize,
    got: &[Option<f64>],
    want: &[Option<f64>],
) -> Option<String> {
    if got.len() != want.len() {
        return Some(format!("{} bar {bar}: output arity", case.name));
    }
    got.iter().zip(want).enumerate().find_map(|(k, (g, w))| {
        let ok = match (g, w) {
            (None, None) => true,
            (Some(g), Some(w)) if case.exact[k] => g == w,
            (Some(g), Some(w)) => (g - w).abs() <= REL_TOL * w.abs().max(1.0),
            _ => false,
        };
        (!ok).then(|| {
            format!(
                "{} bar {bar} output {k}: stream {g:?} != batch {w:?}",
                case.name
            )
        })
    })
}

/// Input columns of a case on `fixture` bars: the pattern fixture for
/// open / high / low / close, the usual per-name columns otherwise.
fn pattern_columns(case: &Case) -> Vec<Vec<f64>> {
    let (open, high, low, close) = pattern_fixture(PATTERN_BARS);
    case.inputs
        .iter()
        .map(|n| match *n {
            "open" => open.clone(),
            "high" => high.clone(),
            "low" => low.clone(),
            "close" | "real" => close.clone(),
            other => input_column(other, PATTERN_BARS),
        })
        .collect()
}

/// Every registry stream matches its batch function at every bar (NaN
/// mask included), with default params, with every period doubled — which
/// catches a lookback that is one bar short — and, for streams with a
/// `matype` param, with every `matype` in `0..=8`. Every `i32` stream (the
/// candlestick codes, which a random walk almost never fires) also runs on
/// `pattern_fixture`. Reports the first mismatch of every failing stream.
#[test]
fn registry_parity_all_streams() {
    let cases = cases();
    assert!(!cases.is_empty());
    let mut failures = Vec::new();
    for case in &cases {
        let doubled = Variant {
            scale: 2,
            matype: None,
        };
        let mut variants = case.variants();
        variants.insert(1, doubled);
        let mut datasets = vec![(columns(case), N_BARS)];
        if case.elems.contains(&"i32") {
            datasets.push((pattern_columns(case), PATTERN_BARS));
        }
        let first = datasets.iter().find_map(|(cols, bars)| {
            variants.iter().find_map(|&variant| {
                let want = (case.batch)(cols, variant);
                assert_eq!(want.len(), *bars, "{}: batch length", case.name);
                let mut stream = (case.stream)(variant);
                want.iter().enumerate().find_map(|(i, w)| {
                    let got = stream.update(&row(cols, i), true);
                    parity_error(case, i, &got, w).map(|e| format!("{e} ({variant:?})"))
                })
            })
        });
        failures.extend(first);
    }
    assert!(
        failures.is_empty(),
        "parity failures:\n{}",
        failures.join("\n")
    );
}

/// Candlestick streams (`i32` outputs over open / high / low / close) must
/// fire at least [`MIN_PATTERN_FIRES`] times on `pattern_fixture`, otherwise
/// the parity run on it proves nothing for that pattern.
#[test]
fn every_pattern_fires_on_fixture() {
    let mut checked = 0;
    let mut quiet = Vec::new();
    for case in &cases() {
        if !(case.elems == ["i32"] && case.inputs == ["open", "high", "low", "close"]) {
            continue;
        }
        checked += 1;
        let fires = (case.batch)(&pattern_columns(case), DEFAULTS)
            .iter()
            .filter(|p| p[0] != Some(0.0))
            .count();
        if fires < MIN_PATTERN_FIRES {
            quiet.push(format!("{} ({fires})", case.name));
        }
    }
    assert_eq!(checked, PATTERN_STREAMS, "pattern stream count");
    assert!(
        quiet.is_empty(),
        "patterns firing < {MIN_PATTERN_FIRES}x: {quiet:?}"
    );
}

/// Streams that emit event / regime codes (every `i8` output, plus the
/// float-coded threshold signal) must show at least two distinct values on
/// the fixture, otherwise parity against batch would be vacuous.
#[test]
fn signal_streams_are_not_degenerate_on_the_fixture() {
    const FLOAT_CODED: &[&str] = &["rsi_threshold_signals"];
    /// Streams whose fixture must produce all of -1, 0 and 1.
    const FULL_TRINARY: &[&str] = &["check_cross", "rsi_threshold_signals"];
    let mut checked = 0;
    for case in &cases() {
        if !(case.elems.contains(&"i8") || FLOAT_CODED.contains(&case.name)) {
            continue;
        }
        checked += 1;
        let cols = columns(case);
        let mut stream = (case.stream)(DEFAULTS);
        let mut distinct = std::collections::BTreeSet::new();
        for i in 0..N_BARS {
            // `None` parts are NaN (warm-up) and are not signal values.
            for part in stream.update(&row(&cols, i), true).into_iter().flatten() {
                distinct.insert(part as i64);
            }
        }
        if FULL_TRINARY.contains(&case.name) {
            let want: std::collections::BTreeSet<i64> = [-1, 0, 1].into();
            assert_eq!(
                distinct, want,
                "{}: fixture must show -1, 0 and 1",
                case.name
            );
        }
        assert!(
            distinct.len() >= 2,
            "{}: fixture only produces {distinct:?}",
            case.name
        );
    }
    assert!(checked >= 6, "expected the i8 streams, saw {checked}");
}

fn same(a: &[Option<f64>], b: &[Option<f64>]) -> bool {
    format!("{a:?}") == format!("{b:?}")
}

/// Previews before every close must not perturb committed state.
#[test]
fn previews_are_pure_all_streams() {
    for case in &cases() {
        let cols = columns(case);
        for variant in case.variants() {
            let mut plain = (case.stream)(variant);
            let mut live = (case.stream)(variant);
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
                let label = format!("{} {variant:?} bar {i}", case.name);
                assert!(same(&preview, &expected), "{label}: preview");
                assert!(same(&got, &expected), "{label}: commit");
            }
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
        for variant in case.variants() {
            let lookback = (case.lookback)(variant);
            // Warm-up edges, then just after the first buffer drain (windowed
            // buffers hold at most 2 · lookback bars), then the end.
            let points = [
                0,
                lookback.saturating_sub(1),
                lookback,
                2 * lookback,
                2 * lookback + 1,
                N_BARS - 1,
            ];
            for save_at in points.into_iter().filter(|&k| k < N_BARS) {
                let mut original = (case.stream)(variant);
                for i in 0..save_at {
                    original.update(&row(&cols, i), true);
                }
                let mut restored = original.restore(&original.save());
                for i in save_at..N_BARS {
                    let a = original.update(&row(&cols, i), true);
                    let b = restored.update(&row(&cols, i), true);
                    let label = format!("{} {variant:?} saved at {save_at}, bar {i}", case.name);
                    assert!(same(&a, &b), "{label}");
                }
            }
        }
    }
}

#[cfg(feature = "serde")]
mod snapshot_fuzz;

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

/// Load `json` after `edit` rewrites its JSON state.
#[cfg(feature = "serde")]
fn edited<T: StreamingIndicator + Snap>(
    json: &str,
    edit: impl FnOnce(&mut serde_json::Value),
) -> Result<Live<T>, crate::streaming::StreamingError> {
    let mut value: serde_json::Value = serde_json::from_str(json).unwrap();
    edit(&mut value["state"]["inner"]);
    Live::<T>::load(&value.to_string())
}

/// A snapshot cannot carry a period no constructor accepts.
#[cfg(feature = "serde")]
#[test]
fn restore_rejects_period_above_cap() {
    use crate::streaming::{StreamingRollingMaxindex, StreamingSum, MAX_STREAM_PERIOD};
    assert!(StreamingSum::new(MAX_STREAM_PERIOD + 1).is_err());
    assert!(StreamingRollingMaxindex::new(MAX_STREAM_PERIOD + 1).is_err());
    let too_big = serde_json::json!(MAX_STREAM_PERIOD + 1);

    let json = Live::new(StreamingSum::new(3).unwrap()).save().unwrap();
    let ok = edited::<StreamingSum>(&json, |s| s["params"]["timeperiod"] = serde_json::json!(4));
    assert!(ok.is_ok(), "fixture shape: {json}");
    let bad = edited::<StreamingSum>(&json, |s| s["params"]["timeperiod"] = too_big.clone());
    assert!(bad.is_err());

    let json = Live::new(StreamingRollingMaxindex::new(3).unwrap())
        .save()
        .unwrap();
    let bad = edited::<StreamingRollingMaxindex>(&json, |s| s["inner"]["timeperiod"] = too_big);
    assert!(bad.is_err());
}

/// Bar counters and stored indices from a snapshot cannot overflow.
#[cfg(feature = "serde")]
#[test]
fn index_restore_rejects_overflowing_counters() {
    use crate::streaming::StreamingRollingMinindex;
    let mut live = Live::new(StreamingRollingMinindex::new(3).unwrap());
    for v in [3.0, 1.0, 2.0, 4.0] {
        live.update(v, true);
    }
    let json = live.save().unwrap();
    type Idx = StreamingRollingMinindex;
    assert!(
        edited::<Idx>(&json, |_| {}).is_ok(),
        "fixture shape: {json}"
    );
    for bars in [u64::MAX, i64::MAX as u64 + 1] {
        let bad = edited::<Idx>(&json, |s| s["inner"]["bars"] = serde_json::json!(bars));
        assert!(bad.is_err(), "bars = {bars}");
        // An empty deque passes the window checks, so the counter bound
        // itself must reject it.
        let empty = edited::<Idx>(&json, |s| {
            s["inner"]["bars"] = serde_json::json!(bars);
            s["inner"]["entries"] = serde_json::json!([]);
        });
        assert!(empty.is_err(), "empty deque, bars = {bars}");
    }
    let future = edited::<Idx>(&json, |s| {
        s["inner"]["entries"][0][0] = serde_json::json!(4)
    });
    assert!(future.is_err(), "entry index must be < bars");
    let stale = edited::<Idx>(&json, |s| {
        s["inner"]["entries"][0][0] = serde_json::json!(0)
    });
    assert!(stale.is_err(), "entry index must be inside the window");
}

/// Generated members (Python / WASM methods, getters, macro locals) that a
/// param name would shadow or collide with.
const RESERVED_PARAMS: &[&str] = &[
    "count",
    "reset",
    "update",
    "peek",
    "is_ready",
    "to_json",
    "from_json",
    "save_state",
    "load_state",
    "params",
    "from_params",
    "validate",
    "last",
    "inner",
    "new",
    "closed",
    "out",
    "self",
];
/// Modules glob re-exported into `streaming`, which a stream's
/// `#[doc(hidden)] pub mod <name>` would clash with.
const STREAMING_MODULES: &[&str] = &[
    "bars",
    "live",
    "windowed",
    "registry",
    "fixtures",
    "families",
    "snapshot",
    "conditions",
    "option_greeks",
    "primitives",
    "math",
    "overlap",
    "ma_osc",
];

#[test]
fn registry_names_do_not_collide() {
    for case in &cases() {
        assert!(
            !STREAMING_MODULES.contains(&case.name),
            "stream name {}",
            case.name
        );
        let mut seen = std::collections::BTreeSet::new();
        for p in case.params {
            assert!(
                !RESERVED_PARAMS.contains(p),
                "{}: param {p} is reserved",
                case.name
            );
            assert!(
                !case.inputs.contains(p),
                "{}: param {p} shadows an input",
                case.name
            );
            assert!(seen.insert(*p), "{}: duplicate param {p}", case.name);
        }
    }
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
    use crate::streaming::registry::{
        check_count, check_float, check_matype, check_period, MAX_STREAM_PERIOD,
    };
    assert!(check_period(2, "p", 2).is_ok() && check_period(1, "p", 2).is_err());
    assert!(check_period(MAX_STREAM_PERIOD, "p", 1).is_ok());
    assert!(check_period(MAX_STREAM_PERIOD + 1, "p", 1).is_err());
    assert!(check_count(0, "c").is_ok() && check_count(MAX_STREAM_PERIOD + 1, "c").is_err());
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
