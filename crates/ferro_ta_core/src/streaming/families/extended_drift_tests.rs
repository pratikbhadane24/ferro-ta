//! Drift check for the windowed `extended_*` streams whose batch kernel
//! carries a running or prefix sum from bar 0 (see `drift_tests.rs`): the
//! stream must equal the batch kernel on the trailing window exactly, and
//! `measure_extended_batch_drift` prints how far the full-series batch
//! drifts from that (`cargo test ... -- --ignored --nocapture`).
use super::drift_tests::{assert_exact_window, run, worst_gap, Batch, Data};
use super::*;
use crate::streaming::registry::BATCH_DRIFT_KERNELS;

/// `(name, lookback, batch over bars lo..hi)` at the Python defaults; one
/// entry per output (`name.k` for output `k` of a multi-output kernel).
fn candidates() -> Vec<(&'static str, usize, Batch)> {
    macro_rules! c {
        ($name:literal, $lb:expr, |$d:ident, $r:ident| $e:expr) => {
            ($name, $lb, Box::new(|$d: &Data, lo: usize, hi: usize| {
                let $r = lo..hi;
                $e
            }) as Batch)
        };
    }
    use crate::extended as x;
    vec![
        c!("choppiness_index", 15, |d, r| x::choppiness_index(
            &d.high[r.clone()],
            &d.low[r.clone()],
            &d.close[r],
            14
        )),
        c!("hull_ma", 19, |d, r| x::hull_ma(&d.close[r], 16)),
        c!("rwi.0", 15, |d, r| x::rwi(
            &d.high[r.clone()],
            &d.low[r.clone()],
            &d.close[r],
            14
        )
        .0),
        c!("rwi.1", 15, |d, r| x::rwi(
            &d.high[r.clone()],
            &d.low[r.clone()],
            &d.close[r],
            14
        )
        .1),
        c!("ao", 34, |d, r| x::ao(&d.high[r.clone()], &d.low[r], 5, 34)),
        c!("ac", 38, |d, r| x::ac(
            &d.high[r.clone()],
            &d.low[r],
            5,
            34,
            5
        )),
        c!("po", 21, |d, r| x::po(&d.close[r], 10, 21)),
        c!("dpo", 20, |d, r| x::dpo(&d.close[r], 20)),
        c!("kst.0", 53, |d, r| x::kst(
            &d.close[r],
            10,
            15,
            20,
            30,
            10,
            10,
            10,
            15,
            9
        )
        .0),
        c!("kst.1", 53, |d, r| x::kst(
            &d.close[r],
            10,
            15,
            20,
            30,
            10,
            10,
            10,
            15,
            9
        )
        .1),
        c!("vortex.0", 15, |d, r| x::vortex(
            &d.high[r.clone()],
            &d.low[r.clone()],
            &d.close[r],
            14
        )
        .0),
        c!("vortex.1", 15, |d, r| x::vortex(
            &d.high[r.clone()],
            &d.low[r.clone()],
            &d.close[r],
            14
        )
        .1),
        c!("coppock", 24, |d, r| x::coppock(&d.close[r], 10, 14, 11)),
        c!("bbpercent", 5, |d, r| x::bbpercent(
            &d.close[r],
            5,
            2.0,
            2.0
        )),
        c!("bbwidth", 5, |d, r| x::bbwidth(&d.close[r], 5, 2.0, 2.0)),
        c!("cmf", 20, |d, r| x::cmf(
            &d.high[r.clone()],
            &d.low[r.clone()],
            &d.close[r.clone()],
            &d.volume[r],
            20
        )),
        c!("emv", 15, |d, r| x::emv(
            &d.high[r.clone()],
            &d.low[r.clone()],
            &d.volume[r],
            14,
            10000.0
        )),
        c!("volosc", 10, |d, r| x::volosc(&d.volume[r], 5, 10)),
        c!("rvol", 20, |d, r| x::rvol(&d.volume[r], 20)),
        c!("vwma", 20, |d, r| x::vwma(
            &d.close[r.clone()],
            &d.volume[r],
            20
        )),
    ]
}

/// Prints the measured drift of every candidate kernel.
#[test]
#[ignore = "measurement; run with --nocapture"]
fn measure_extended_batch_drift() {
    let d = Data::new();
    for (name, lookback, batch) in candidates() {
        println!(
            "{name} lookback={lookback} max rel diff {:e}",
            worst_gap(&d, lookback, &batch)
        );
    }
}

#[test]
fn extended_windowed_streams_equal_batch_on_trailing_window_long_series() {
    let d = Data::new();
    let cands = candidates();
    let check = |name: &str, got: Vec<f64>| {
        let (_, lb, batch) = cands.iter().find(|c| c.0 == name).unwrap();
        assert_exact_window(name, &got, &d, *lb, batch);
    };
    let hlc = |i: usize| (d.high[i], d.low[i], d.close[i]);
    let hl = |i: usize| (d.high[i], d.low[i]);
    let close = |i: usize| d.close[i];
    let vol = |i: usize| d.volume[i];

    check(
        "choppiness_index",
        run(StreamingChoppinessIndex::new(14).unwrap(), hlc),
    );
    check("hull_ma", run(StreamingHullMa::new(16).unwrap(), close));
    let rwi = run(StreamingRwi::new(14).unwrap(), hlc);
    check("rwi.0", rwi.iter().map(|o| o.0).collect());
    check("rwi.1", rwi.iter().map(|o| o.1).collect());
    check("ao", run(StreamingAo::new(5, 34).unwrap(), hl));
    check("ac", run(StreamingAc::new(5, 34, 5).unwrap(), hl));
    check("po", run(StreamingPo::new(10, 21).unwrap(), close));
    check("dpo", run(StreamingDpo::new(20).unwrap(), close));
    let kst = run(
        StreamingKst::new(10, 15, 20, 30, 10, 10, 10, 15, 9).unwrap(),
        close,
    );
    check("kst.0", kst.iter().map(|o| o.0).collect());
    check("kst.1", kst.iter().map(|o| o.1).collect());
    let vortex = run(StreamingVortex::new(14).unwrap(), hlc);
    check("vortex.0", vortex.iter().map(|o| o.0).collect());
    check("vortex.1", vortex.iter().map(|o| o.1).collect());
    check(
        "coppock",
        run(StreamingCoppock::new(10, 14, 11).unwrap(), close),
    );
    check(
        "bbpercent",
        run(StreamingBbpercent::new(5, 2.0, 2.0).unwrap(), close),
    );
    check(
        "bbwidth",
        run(StreamingBbwidth::new(5, 2.0, 2.0).unwrap(), close),
    );
    check(
        "cmf",
        run(StreamingCmf::new(20).unwrap(), |i| {
            (d.high[i], d.low[i], d.close[i], d.volume[i])
        }),
    );
    check(
        "emv",
        run(StreamingEmv::new(14, 10000.0).unwrap(), |i| {
            (d.high[i], d.low[i], d.volume[i])
        }),
    );
    check("volosc", run(StreamingVolosc::new(5, 10).unwrap(), vol));
    check("rvol", run(StreamingRvol::new(20).unwrap(), vol));
    check(
        "vwma",
        run(StreamingVwma::new(20).unwrap(), |i| {
            (d.close[i], d.volume[i])
        }),
    );
    for name in EXTENDED_DRIFT_KERNELS {
        assert!(BATCH_DRIFT_KERNELS.contains(name), "{name}");
    }
}

/// Extended kernels whose measured gap (20k bars near 50,000) is within 10x
/// of the 1e-9 contract or beyond it: bbpercent 6.8e-8 (the `bbands`
/// running sum of squares), ao 5.2e-10, po 3.6e-10, dpo 2.3e-10 (`sma`
/// running sums under a near-zero difference; the gap scales with price).
/// Every other kernel above is <= 1.1e-11 (hull 6.1e-12, ac 1.0e-11,
/// volosc 1.7e-12, bbwidth 1.4e-12, emv 7.2e-13, vwma 8.2e-13, the rest
/// <= 5e-14). RVI and Ulcer (gap 0 / 3.7e-17 here) are native instead: their
/// running sums decide the NaN mask on flat windows (see their docs).
const EXTENDED_DRIFT_KERNELS: &[&str] = &["bbpercent", "ao", "po", "dpo"];
