//! Corrupt snapshots must be rejected or accepted, never panic the loader.
use super::*;
use serde_json::Value;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Bars fed after a corrupt snapshot loads.
const BARS_AFTER_LOAD: usize = 5;

/// Mutants per stream variant above which each node gets one mutation
/// (rotating through its kinds) instead of all of them.
const FULL_MUTATION_BUDGET: usize = 1_000;

/// Replacements for one node (applied one at a time); for an object, one
/// copy per removed field.
fn local_mutations(v: &Value) -> Vec<Value> {
    match v {
        Value::Array(a) => {
            let mut out = vec![Value::Array(vec![])];
            if !a.is_empty() {
                out.push(Value::Array(a[..a.len() - 1].to_vec()));
            }
            out
        }
        Value::Number(_) | Value::Null => [
            serde_json::json!(0),
            serde_json::json!(-1),
            serde_json::json!(1e300),
            serde_json::json!(1u64 << 53),
        ]
        .to_vec(),
        Value::Bool(b) => vec![Value::Bool(!b)],
        Value::String(_) => vec![Value::String(String::new())],
        Value::Object(o) => o
            .keys()
            .map(|key| {
                let mut removed = o.clone();
                removed.remove(key);
                Value::Object(removed)
            })
            .collect(),
    }
}

/// JSON pointer of every node of `v` (the root is `""`).
fn pointers(v: &Value, at: String, out: &mut Vec<String>) {
    match v {
        Value::Array(a) => a
            .iter()
            .enumerate()
            .for_each(|(i, c)| pointers(c, format!("{at}/{i}"), out)),
        Value::Object(o) => o.iter().for_each(|(k, c)| {
            let key = k.replace('~', "~0").replace('/', "~1");
            pointers(c, format!("{at}/{key}"), out)
        }),
        _ => {}
    }
    out.push(at);
}

/// Single-node mutations of `doc` as whole documents: every node mutated,
/// with all of its mutation kinds while the total stays within
/// [`FULL_MUTATION_BUDGET`], else one kind per node.
fn mutants(doc: &Value) -> Vec<Value> {
    let mut paths = Vec::new();
    pointers(doc, String::new(), &mut paths);
    let per_node: Vec<Vec<Value>> = paths
        .iter()
        .map(|p| local_mutations(doc.pointer(p).unwrap()))
        .collect();
    let full = per_node.iter().map(Vec::len).sum::<usize>() <= FULL_MUTATION_BUDGET;
    let mut out = Vec::new();
    for (k, (path, options)) in paths.iter().zip(per_node).enumerate() {
        if options.is_empty() {
            continue;
        }
        let picked: Vec<Value> = if full {
            options
        } else {
            let pick = k % options.len();
            options.into_iter().skip(pick).take(1).collect()
        };
        for replacement in picked {
            let mut copy = doc.clone();
            *copy.pointer_mut(path).unwrap() = replacement;
            out.push(copy);
        }
    }
    out
}

/// Every registry stream (each `matype` variant too): load single-node
/// mutations (every node at least once) of a warm snapshot under `catch_unwind`; a
/// snapshot that loads must also survive a few more bars.
#[test]
fn snapshot_load_never_panics_all_streams() {
    let mut panics = Vec::new();
    let mut vacuous = Vec::new();
    for case in &cases() {
        let mut accepted = 0usize;
        let cols = columns(case);
        for variant in case.variants() {
            let mut stream = (case.stream)(variant);
            let warm = ((case.lookback)(variant) + 3).min(N_BARS - BARS_AFTER_LOAD);
            for i in 0..warm {
                stream.update(&row(&cols, i), true);
            }
            let doc: Value = serde_json::from_str(&stream.save()).unwrap();
            // Sanity: the unmutated snapshot loads and continues exactly
            // like the stream it came from (else every mutant would be
            // rejected for the wrong reason and the fuzz would pass vacuously).
            let mut twin = stream.try_restore(&doc.to_string()).unwrap_or_else(|| {
                panic!("{} {variant:?}: own snapshot failed to load", case.name)
            });
            for i in warm..warm + BARS_AFTER_LOAD {
                let want = stream.update(&row(&cols, i), true);
                let got = twin.update(&row(&cols, i), true);
                assert_eq!(
                    format!("{want:?}"),
                    format!("{got:?}"),
                    "{} {variant:?}: restored stream diverged at bar {i}",
                    case.name
                );
            }
            for mutant in mutants(&doc) {
                let json = mutant.to_string();
                let outcome = catch_unwind(AssertUnwindSafe(|| {
                    if let Some(mut loaded) = stream.try_restore(&json) {
                        accepted += 1;
                        for i in warm..warm + BARS_AFTER_LOAD {
                            loaded.update(&row(&cols, i), false);
                            loaded.update(&row(&cols, i), true);
                        }
                    }
                }));
                if outcome.is_err() {
                    panics.push(format!("{} {variant:?}: {json}", case.name));
                }
            }
        }
        if accepted == 0 {
            vacuous.push(case.name);
        }
    }
    assert!(
        vacuous.is_empty(),
        "no mutant of these streams was ever accepted, so the post-load \
         update path is untested: {vacuous:?}"
    );
    assert!(
        panics.is_empty(),
        "{} snapshot(s) panicked:\n{}",
        panics.len(),
        panics.join("\n")
    );
}
