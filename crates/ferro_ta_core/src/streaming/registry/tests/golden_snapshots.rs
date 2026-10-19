//! Snapshot-format guard for every registry kind: the saved state of each
//! stream (default params, after [`WARM`] bars) is pinned in a committed
//! fixture, so any change to a stream's serialized layout fails here until
//! `STATE_FORMAT_VERSION` is bumped and the fixtures are re-recorded.
//!
//! Record: `UPDATE_SNAPSHOT_FIXTURES=1 cargo test -p ferro_ta_core --features serde golden`.
use super::{cases, columns_n, row, same, DEFAULTS};
use crate::streaming::STATE_FORMAT_VERSION;
use std::collections::BTreeMap;

/// Bars committed before the snapshot is taken.
const WARM: usize = 50;
/// Bars replayed after it, on the committed and on a fresh stream.
const TAIL: usize = 50;

fn fixture_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "tests/fixtures/streaming_registry_snapshots_v{STATE_FORMAT_VERSION}.json"
    ))
}

fn load_fixtures(path: &std::path::Path) -> BTreeMap<String, String> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|_| {
        panic!(
            "missing {}; run with UPDATE_SNAPSHOT_FIXTURES=1",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap()
}

/// Relative tolerance for float leaves: libm results differ by an ulp or so
/// across platforms, which must not read as a layout change.
const FLOAT_REL_TOL: f64 = 1e-12;

/// Structural comparison of two snapshot JSON texts: identical shape (keys,
/// array lengths, value types), exact integers/bools/strings, floats within
/// [`FLOAT_REL_TOL`]. Returns the first difference as `Err`.
fn layout_diff(actual: &str, golden: &str) -> Result<(), String> {
    let a: serde_json::Value = serde_json::from_str(actual).map_err(|e| e.to_string())?;
    let g: serde_json::Value = serde_json::from_str(golden).map_err(|e| e.to_string())?;
    value_diff(&a, &g, "$")
}

fn value_diff(a: &serde_json::Value, g: &serde_json::Value, at: &str) -> Result<(), String> {
    use serde_json::Value::{Array, Bool, Null, Number, Object, String as Str};
    match (a, g) {
        (Null, Null) => Ok(()),
        (Bool(x), Bool(y)) if x == y => Ok(()),
        (Str(x), Str(y)) if x == y => Ok(()),
        (Number(x), Number(y)) => number_diff(x, y, at),
        (Array(x), Array(y)) => {
            if x.len() != y.len() {
                return Err(format!("{at}: array length {} vs {}", x.len(), y.len()));
            }
            x.iter()
                .zip(y)
                .enumerate()
                .try_for_each(|(i, (p, q))| value_diff(p, q, &format!("{at}[{i}]")))
        }
        (Object(x), Object(y)) => {
            if x.keys().ne(y.keys()) {
                return Err(format!("{at}: object keys differ"));
            }
            x.iter()
                .try_for_each(|(k, p)| value_diff(p, &y[k], &format!("{at}.{k}")))
        }
        _ => Err(format!("{at}: value or type differs")),
    }
}

fn number_diff(x: &serde_json::Number, y: &serde_json::Number, at: &str) -> Result<(), String> {
    if x.is_f64() != y.is_f64() {
        return Err(format!("{at}: integer vs float"));
    }
    if !x.is_f64() {
        return if x == y {
            Ok(())
        } else {
            Err(format!("{at}: {x} vs {y}"))
        };
    }
    let (p, q) = (x.as_f64().unwrap(), y.as_f64().unwrap());
    if (p - q).abs() <= FLOAT_REL_TOL * p.abs().max(q.abs()).max(1.0) {
        Ok(())
    } else {
        Err(format!("{at}: {p} vs {q}"))
    }
}

#[test]
fn golden_registry_snapshots_match_current_format() {
    let record = std::env::var_os("UPDATE_SNAPSHOT_FIXTURES").is_some();
    let path = fixture_path();
    let committed = if record {
        BTreeMap::new()
    } else {
        load_fixtures(&path)
    };
    let mut recorded = BTreeMap::new();
    let mut failures = Vec::new();
    let cases = cases();
    for case in &cases {
        let cols = columns_n(case, WARM + TAIL);
        let mut fresh = (case.stream)(DEFAULTS);
        for i in 0..WARM {
            fresh.update(&row(&cols, i), true);
        }
        let json = fresh.save();
        if record {
            recorded.insert(case.name.to_string(), json);
            continue;
        }
        let Some(golden) = committed.get(case.name) else {
            failures.push(format!("{}: no golden snapshot", case.name));
            continue;
        };
        if let Err(why) = layout_diff(&json, golden) {
            failures.push(format!("{}: state layout changed ({why})", case.name));
            continue;
        }
        let mut restored = fresh.restore(golden);
        let diverged = (WARM..WARM + TAIL).find(|&i| {
            let bar = row(&cols, i);
            !same(&fresh.update(&bar, true), &restored.update(&bar, true))
        });
        if let Some(i) = diverged {
            failures.push(format!(
                "{}: committed snapshot diverges at bar {i}",
                case.name
            ));
        }
    }
    if record {
        let text = serde_json::to_string_pretty(&recorded).unwrap() + "\n";
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, text).unwrap();
        return;
    }
    let stale: Vec<&String> = committed
        .keys()
        .filter(|k| !cases.iter().any(|c| c.name == k.as_str()))
        .collect();
    assert!(
        stale.is_empty(),
        "fixture kinds not in the registry: {stale:?}"
    );
    assert!(
        failures.is_empty(),
        "bump STATE_FORMAT_VERSION and regenerate (UPDATE_SNAPSHOT_FIXTURES=1 \
         cargo test -p ferro_ta_core --features serde golden):\n{}",
        failures.join("\n")
    );
}

#[cfg(test)]
mod layout_diff_tests {
    use super::layout_diff;

    const BASE: &str = r#"{"a":1,"b":[0.5,2.5],"c":null,"d":"x","e":true}"#;

    #[test]
    fn one_ulp_float_difference_passes() {
        let nudged = r#"{"a":1,"b":[0.5000000000000001,2.5],"c":null,"d":"x","e":true}"#;
        assert!(layout_diff(nudged, BASE).is_ok());
    }

    #[test]
    fn real_float_change_fails() {
        let moved = r#"{"a":1,"b":[0.51,2.5],"c":null,"d":"x","e":true}"#;
        assert!(layout_diff(moved, BASE).is_err());
    }

    #[test]
    fn shape_changes_fail() {
        for changed in [
            r#"{"a":1,"b":[0.5,2.5],"c":null,"d":"x"}"#,
            r#"{"a":1,"b":[0.5,2.5],"c":null,"d":"x","e":true,"f":0}"#,
            r#"{"a":1,"b":[0.5],"c":null,"d":"x","e":true}"#,
            r#"{"a":1,"b":[0.5,2.5],"c":0.0,"d":"x","e":true}"#,
            r#"{"a":1.0,"b":[0.5,2.5],"c":null,"d":"x","e":true}"#,
            r#"{"a":2,"b":[0.5,2.5],"c":null,"d":"x","e":true}"#,
            r#"{"a":1,"b":[0.5,2.5],"c":null,"d":"y","e":true}"#,
            r#"{"a":1,"b":[0.5,2.5],"c":null,"d":"x","e":false}"#,
        ] {
            assert!(layout_diff(changed, BASE).is_err(), "{changed}");
        }
    }
}
