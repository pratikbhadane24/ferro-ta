//! ns/update of every registry stream at default params. Run in release:
//!
//! ```text
//! cargo test -p ferro_ta_core --release --all-features --lib \
//!     stream_ns_per_update -- --ignored --nocapture
//! ```
//!
//! Commit: `closed = true` over [`BENCH_BARS`] fixture bars. Preview:
//! `closed = false` over the same bars on the final committed state. Best of
//! [`BENCH_RUNS`] fresh streams. Prints a markdown table, slowest commit
//! first. `FERRO_STREAM_BENCH_JSON=<path>` also writes the numbers as JSON;
//! `FERRO_STREAM_BENCH_FILTER=<substr>` keeps only matching stream names.
use super::*;

const BENCH_BARS: usize = 10_000;
const BENCH_RUNS: usize = 3;
/// Per-update budget of the commit path.
const BUDGET_NS: f64 = 2_000.0;

struct Timing {
    name: &'static str,
    imp: &'static str,
    commit_ns: f64,
    preview_ns: f64,
}

fn per_update(d: Duration) -> f64 {
    d.as_nanos() as f64 / BENCH_BARS as f64
}

fn time_case(case: &Case) -> Timing {
    let cols = columns_n(case, BENCH_BARS);
    let rows: Vec<Vec<f64>> = (0..BENCH_BARS).map(|i| row(&cols, i)).collect();
    let (mut commit_ns, mut preview_ns) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..BENCH_RUNS {
        let (commit, preview) = (case.stream)(DEFAULTS).time_updates(&rows);
        commit_ns = commit_ns.min(per_update(commit));
        preview_ns = preview_ns.min(per_update(preview));
    }
    Timing {
        name: case.name,
        imp: case.imp,
        commit_ns,
        preview_ns,
    }
}

fn to_json(rows: &[Timing]) -> String {
    let body: Vec<String> = rows
        .iter()
        .map(|t| {
            format!(
                "  {{\"stream\": \"{}\", \"impl\": \"{}\", \"commit_ns\": {:.1}, \"preview_ns\": {:.1}}}",
                t.name, t.imp, t.commit_ns, t.preview_ns
            )
        })
        .collect();
    format!("[\n{}\n]\n", body.join(",\n"))
}

#[test]
#[ignore = "benchmark: run in release with --ignored --nocapture"]
fn stream_ns_per_update() {
    let filter = std::env::var("FERRO_STREAM_BENCH_FILTER").unwrap_or_default();
    let mut rows: Vec<Timing> = cases()
        .iter()
        .filter(|c| c.name.contains(filter.as_str()))
        .map(time_case)
        .collect();
    rows.sort_by(|a, b| b.commit_ns.total_cmp(&a.commit_ns));
    println!("\n| stream | impl | commit ns | preview ns |\n|---|---|---:|---:|");
    for t in &rows {
        println!(
            "| {} | {} | {:.0} | {:.0} |",
            t.name, t.imp, t.commit_ns, t.preview_ns
        );
    }
    let over: Vec<&str> = rows
        .iter()
        .filter(|t| t.commit_ns > BUDGET_NS)
        .map(|t| t.name)
        .collect();
    println!(
        "\n{} streams; {} over {BUDGET_NS} ns on commit: {over:?}",
        rows.len(),
        over.len()
    );
    if let Ok(path) = std::env::var("FERRO_STREAM_BENCH_JSON") {
        std::fs::write(&path, to_json(&rows)).unwrap();
    }
}
