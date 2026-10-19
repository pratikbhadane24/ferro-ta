//! Windowed streams recompute each window exactly, so they equal the batch
//! kernel applied to just the trailing window. A few batch kernels keep
//! rolling sums from bar 0 and drift on long, high-priced series; this pins
//! the stream side and measures the gap (`cargo test ... -- --ignored --nocapture`).
use super::*;
use crate::streaming::fixtures::bars_seeded;
use crate::streaming::registry::BATCH_DRIFT_KERNELS;
use crate::streaming::{Live, StreamingIndicator};

pub(super) const LONG_BARS: usize = 20_000;
pub(super) const SAMPLE_STEP: usize = 97;
const START_PRICE: f64 = 50_000.0;
const REL_TOL: f64 = 1e-9;

pub(super) struct Data {
    pub(super) open: Vec<f64>,
    pub(super) high: Vec<f64>,
    pub(super) low: Vec<f64>,
    pub(super) close: Vec<f64>,
    pub(super) volume: Vec<f64>,
    pub(super) other: Vec<f64>,
}

impl Data {
    pub(super) fn new() -> Self {
        let rows = bars_seeded(12345, LONG_BARS);
        let shift = START_PRICE - 100.0;
        let col = |f: fn(&(f64, f64, f64, f64)) -> f64| -> Vec<f64> {
            rows.iter().map(|r| f(r) + shift).collect()
        };
        let close = col(|r| r.2);
        Data {
            open: close
                .iter()
                .scan(START_PRICE, |p, &c| Some(std::mem::replace(p, c)))
                .collect(),
            high: col(|r| r.0),
            low: col(|r| r.1),
            other: close
                .iter()
                .enumerate()
                .map(|(i, c)| c * 0.5 + 1000.0 + (i % 7) as f64)
                .collect(),
            volume: rows.iter().map(|r| r.3).collect(),
            close,
        }
    }
}

pub(super) type Batch = Box<dyn Fn(&Data, usize, usize) -> Vec<f64>>;

/// `(name, lookback, batch over bars lo..hi)` for the drift-suspect kernels.
fn candidates() -> Vec<(&'static str, usize, Batch)> {
    macro_rules! c {
        ($name:literal, $lb:expr, |$d:ident, $r:ident| $e:expr) => {
            ($name, $lb, Box::new(|$d: &Data, lo: usize, hi: usize| {
                let $r = lo..hi;
                $e
            }) as Batch)
        };
    }
    vec![
        c!("correl", 30, |d, r| crate::statistic::correl(
            &d.close[r.clone()],
            &d.other[r],
            30
        )),
        c!("correl", 5, |d, r| crate::statistic::correl(
            &d.close[r.clone()],
            &d.other[r],
            5
        )),
        c!("beta", 6, |d, r| crate::statistic::beta(
            &d.close[r.clone()],
            &d.other[r],
            5
        )),
        c!("linearreg_slope", 14, |d, r| {
            crate::statistic::linearreg_slope(&d.close[r], 14)
        }),
        c!("linearreg_angle", 14, |d, r| {
            crate::statistic::linearreg_angle(&d.close[r], 14)
        }),
        c!("linearreg", 14, |d, r| crate::statistic::linearreg(
            &d.close[r],
            14
        )),
        c!("linearreg_intercept", 14, |d, r| {
            crate::statistic::linearreg_intercept(&d.close[r], 14)
        }),
        c!("tsf", 14, |d, r| crate::statistic::tsf(&d.close[r], 14)),
        c!("stddev", 5, |d, r| crate::statistic::stddev(
            &d.close[r],
            5,
            1.0
        )),
        c!("var", 5, |d, r| crate::statistic::var(&d.close[r], 5, 1.0)),
        c!("sum", 30, |d, r| crate::math::sum(&d.close[r], 30)),
        c!("rolling_sum", 30, |d, r| crate::math_ops::rolling_sum(
            &d.close[r],
            30
        )),
        c!("wma", 30, |d, r| crate::overlap::wma(&d.close[r], 30)),
        c!("trima", 30, |d, r| crate::overlap::trima(&d.close[r], 30)),
        c!("cci", 14, |d, r| crate::momentum::cci(
            &d.high[r.clone()],
            &d.low[r.clone()],
            &d.close[r],
            14
        )),
        c!("mfi", 15, |d, r| crate::volume::mfi(
            &d.high[r.clone()],
            &d.low[r.clone()],
            &d.close[r.clone()],
            &d.volume[r],
            14
        )),
        c!("rolling_beta", 20, |d, r| crate::portfolio::rolling_beta(
            &d.close[r.clone()],
            &d.other[r],
            20
        )),
        c!("zscore_series", 20, |d, r| crate::portfolio::zscore_series(
            &d.close[r],
            20
        )),
        c!("close_to_close_vol", 21, |d, r| {
            crate::options::realized_vol::close_to_close_vol(&d.close[r], 20, 252.0)
        }),
        c!("garman_klass_vol", 20, |d, r| {
            crate::options::realized_vol::garman_klass_vol(
                &d.open[r.clone()],
                &d.high[r.clone()],
                &d.low[r.clone()],
                &d.close[r],
                20,
                252.0,
            )
        }),
        c!("yang_zhang_vol", 21, |d, r| {
            crate::options::realized_vol::yang_zhang_vol(
                &d.open[r.clone()],
                &d.high[r.clone()],
                &d.low[r.clone()],
                &d.close[r],
                20,
                252.0,
            )
        }),
    ]
}

/// Worst relative gap between batch-on-full-series and batch-on-trailing-window.
pub(super) fn worst_gap(d: &Data, lookback: usize, batch: &Batch) -> f64 {
    let full = batch(d, 0, LONG_BARS);
    (lookback..LONG_BARS)
        .step_by(SAMPLE_STEP)
        .filter_map(|i| {
            let w = *batch(d, i + 1 - lookback, i + 1).last().unwrap();
            let b = full[i];
            (w.is_finite() && b.is_finite()).then(|| (w - b).abs() / b.abs().max(1.0))
        })
        .fold(0.0, f64::max)
}

/// Prints the measured drift of every candidate kernel.
#[test]
#[ignore = "measurement; run with --nocapture"]
fn measure_batch_drift() {
    let d = Data::new();
    for (name, lookback, batch) in candidates() {
        println!(
            "{name} lookback={lookback} max rel diff {:e}",
            worst_gap(&d, lookback, &batch)
        );
    }
}

pub(super) fn run<S: StreamingIndicator>(s: S, feed: impl Fn(usize) -> S::Input) -> Vec<S::Output> {
    let mut live = Live::new(s);
    (0..LONG_BARS).map(|i| live.update(feed(i), true)).collect()
}

pub(super) fn assert_exact_window(
    name: &str,
    got: &[f64],
    d: &Data,
    lookback: usize,
    batch: &Batch,
) {
    for i in (lookback..LONG_BARS).step_by(SAMPLE_STEP) {
        let want = *batch(d, i + 1 - lookback, i + 1).last().unwrap();
        let ok = (got[i].is_nan() && want.is_nan())
            || (got[i] - want).abs() <= REL_TOL * want.abs().max(1.0);
        assert!(
            ok,
            "{name} bar {i}: stream {} != trailing-window batch {want}",
            got[i]
        );
    }
}

#[test]
fn windowed_streams_equal_batch_on_trailing_window_long_series() {
    let d = Data::new();
    let cands = candidates();
    let find = |name: &str, lb: usize| cands.iter().find(|c| c.0 == name && c.1 == lb).unwrap();
    macro_rules! check {
        ($name:literal, $lb:expr, $stream:expr, |$i:ident| $feed:expr) => {{
            let (_, lb, batch) = find($name, $lb);
            let got = run($stream.unwrap(), |$i| $feed);
            assert_exact_window($name, &got, &d, *lb, batch);
        }};
    }
    check!("correl", 30, StreamingCorrel::new(30), |i| (
        d.close[i], d.other[i]
    ));
    check!("correl", 5, StreamingCorrel::new(5), |i| (
        d.close[i], d.other[i]
    ));
    check!("beta", 6, StreamingBeta::new(5), |i| (
        d.close[i], d.other[i]
    ));
    check!(
        "linearreg_slope",
        14,
        StreamingLinearregSlope::new(14),
        |i| d.close[i]
    );
    check!(
        "linearreg_angle",
        14,
        StreamingLinearregAngle::new(14),
        |i| d.close[i]
    );
    check!("linearreg", 14, StreamingLinearreg::new(14), |i| d.close[i]);
    check!(
        "linearreg_intercept",
        14,
        StreamingLinearregIntercept::new(14),
        |i| d.close[i]
    );
    check!("tsf", 14, StreamingTsf::new(14), |i| d.close[i]);
    check!("stddev", 5, StreamingStddev::new(5, 1.0), |i| d.close[i]);
    check!("var", 5, StreamingVar::new(5, 1.0), |i| d.close[i]);
    // mfi is native (it mirrors the batch running flow sums), so it is
    // measured above but not pinned to the trailing window here.
    check!("cci", 14, StreamingCci::new(14), |i| (
        d.high[i], d.low[i], d.close[i]
    ));
    // Measured (max rel diff vs batch, 20k bars near 50,000): correl 1.1e-3 /
    // 3.3e-5 (p=5 / 30), linearreg_slope 3.9e-8, linearreg_angle 2.2e-6,
    // stddev 1.7e-8, var 5.0e-9; every other windowed kernel here is <= 2e-12.
    for name in [
        "correl",
        "linearreg_slope",
        "linearreg_angle",
        "stddev",
        "var",
    ] {
        assert!(BATCH_DRIFT_KERNELS.contains(&name), "{name}");
    }
}
