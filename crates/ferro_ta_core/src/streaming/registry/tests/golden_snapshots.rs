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
        if &json != golden {
            failures.push(format!("{}: state layout changed", case.name));
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
