//! SAR, SAREXT, MAMA, the Hilbert-transform streams and Ichimoku on inputs and
//! parameters the registry run (defaults, doubled periods) never reaches:
//! signed start values, offset on reverse, asymmetric accelerations, other
//! MAMA limits, a falling open, flat and sine prices, and Ichimoku
//! displacements other than the default. Each matches its batch kernel at
//! every bar, NaN mask included.
use super::*;
use crate::streaming::fixtures::input_column;
use crate::streaming::primitives::tests::assert_series;
use crate::streaming::StreamingIndicator;

const N: usize = 2000;

/// `(high, low)` of the fixture walk and of its mirror image, which opens
/// falling (`high[1] < high[0]` exactly when the walk opens rising).
fn hl_layouts() -> Vec<(String, Vec<f64>, Vec<f64>)> {
    let high = input_column("high", N);
    let low = input_column("low", N);
    let mirror_high: Vec<f64> = low.iter().map(|l| 200.0 - l).collect();
    let mirror_low: Vec<f64> = high.iter().map(|h| 200.0 - h).collect();
    assert_ne!(
        high[1] >= high[0],
        mirror_high[1] >= mirror_high[0],
        "layouts must open in opposite directions"
    );
    vec![
        ("walk".into(), high, low),
        ("mirror".into(), mirror_high, mirror_low),
    ]
}

/// Closes: the fixture walk, a flat line (zero phasor) and a 20-bar sine.
fn close_layouts() -> Vec<(&'static str, Vec<f64>)> {
    vec![
        ("walk", input_column("close", N)),
        ("flat", vec![100.0; N]),
        (
            "sine",
            (0..N)
                .map(|i| 100.0 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 20.0).sin())
                .collect(),
        ),
    ]
}

#[test]
fn sar_matches_batch_off_defaults() {
    for (tag, h, l) in hl_layouts() {
        for (acc, max) in [(0.02, 0.2), (0.05, 0.3), (0.01, 0.1), (0.2, 0.2)] {
            let mut s = StreamingSar::new(acc, max).unwrap();
            let got: Vec<f64> = (0..N).map(|i| s.update(h[i], l[i])).collect();
            let want = crate::overlap::sar(&h, &l, acc, max);
            assert_series(&format!("sar {tag} {acc}/{max}"), &got, &want);
        }
    }
}

#[test]
fn sarext_matches_batch_with_start_offset_and_asymmetric_acceleration() {
    let starts = [0.0, 95.0, -105.0];
    let offsets = [0.0, 0.01, 0.05];
    let accels = [
        [0.02, 0.02, 0.2, 0.02, 0.02, 0.2],
        [0.01, 0.03, 0.25, 0.04, 0.01, 0.15],
    ];
    for (tag, h, l) in hl_layouts() {
        for start in starts {
            for offset in offsets {
                for a in accels {
                    let mut s =
                        StreamingSarext::new(start, offset, a[0], a[1], a[2], a[3], a[4], a[5])
                            .unwrap();
                    let got: Vec<f64> = (0..N).map(|i| s.update(h[i], l[i])).collect();
                    let want = crate::overlap::sarext(
                        &h, &l, start, offset, a[0], a[1], a[2], a[3], a[4], a[5],
                    );
                    assert_series(&format!("sarext {tag} {start}/{offset}/{a:?}"), &got, &want);
                }
            }
        }
    }
}

#[test]
fn mama_matches_batch_off_defaults() {
    for (tag, c) in close_layouts() {
        for (fast, slow) in [(0.5, 0.05), (0.3, 0.1), (0.5, 0.5), (0.9, 0.01)] {
            let mut s = StreamingMama::new(fast, slow).unwrap();
            let (gm, gf): (Vec<f64>, Vec<f64>) = c.iter().map(|&x| s.update(x)).unzip();
            let (wm, wf) = crate::overlap::mama(&c, fast, slow);
            assert_series(&format!("mama {tag} {fast}/{slow}"), &gm, &wm);
            assert_series(&format!("fama {tag} {fast}/{slow}"), &gf, &wf);
        }
    }
}

#[test]
fn mama_rejects_inverted_limits() {
    assert!(StreamingMama::new(0.05, 0.5).is_err());
    assert!(StreamingMama::new(f64::NAN, 0.05).is_err());
}

#[test]
fn ht_streams_match_batch_on_flat_and_sine_prices() {
    use crate::cycle as ht;
    for (tag, c) in close_layouts() {
        for n in [ht::HT_LOOKBACK, ht::HT_LOOKBACK + 1, N] {
            let c = &c[..n];
            let run = |f: &mut dyn FnMut(f64) -> f64| c.iter().map(|&x| f(x)).collect::<Vec<_>>();
            let label = |name: &str| format!("{name} {tag} n={n}");
            let mut s = StreamingHtTrendline::new().unwrap();
            assert_series(
                &label("trendline"),
                &run(&mut |x| s.update(x)),
                &ht::ht_trendline(c),
            );
            let mut s = StreamingHtDcperiod::new().unwrap();
            assert_series(
                &label("dcperiod"),
                &run(&mut |x| s.update(x)),
                &ht::ht_dcperiod(c),
            );
            let mut s = StreamingHtDcphase::new().unwrap();
            assert_series(
                &label("dcphase"),
                &run(&mut |x| s.update(x)),
                &ht::ht_dcphase(c),
            );
            let (wi, wq) = ht::ht_phasor(c);
            let mut s = StreamingHtPhasor::new().unwrap();
            let (gi, gq): (Vec<f64>, Vec<f64>) = c.iter().map(|&x| s.update(x)).unzip();
            assert_series(&label("inphase"), &gi, &wi);
            assert_series(&label("quadrature"), &gq, &wq);
            let (ws, wl) = ht::ht_sine(c);
            let mut s = StreamingHtSine::new().unwrap();
            let (gs, gl): (Vec<f64>, Vec<f64>) = c.iter().map(|&x| s.update(x)).unzip();
            assert_series(&label("sine"), &gs, &ws);
            assert_series(&label("leadsine"), &gl, &wl);
            let mut s = StreamingHtTrendmode::new().unwrap();
            let got: Vec<i32> = c.iter().map(|&x| s.update(x)).collect();
            assert_eq!(got, ht::ht_trendmode(c), "{}", label("trendmode"));
        }
    }
}

#[test]
fn ichimoku_matches_batch_across_displacements() {
    let (h, l, c) = (
        input_column("high", N),
        input_column("low", N),
        input_column("close", N),
    );
    let params = [
        (9usize, 26usize, 52usize, 26usize),
        (9, 26, 52, 0),
        (1, 1, 1, 0),
        (2, 3, 4, 2),
        (3, 3, 3, 100),
        (52, 26, 9, 1),
    ];
    for (t, k, b, d) in params {
        let mut s = StreamingIchimoku::new(t, k, b, d).unwrap();
        let rows: Vec<(f64, f64, f64, f64)> =
            (0..N).map(|i| s.update((h[i], l[i], c[i]))).collect();
        let want = crate::extended::ichimoku(&h, &l, &c, t, k, b, d);
        let tag = format!("ichimoku {t}/{k}/{b}/{d}");
        let col = |f: fn(&(f64, f64, f64, f64)) -> f64| rows.iter().map(f).collect::<Vec<_>>();
        assert_series(&format!("{tag} tenkan"), &col(|r| r.0), &want.0);
        assert_series(&format!("{tag} kijun"), &col(|r| r.1), &want.1);
        assert_series(&format!("{tag} senkou_a"), &col(|r| r.2), &want.2);
        assert_series(&format!("{tag} senkou_b"), &col(|r| r.3), &want.3);
    }
}
