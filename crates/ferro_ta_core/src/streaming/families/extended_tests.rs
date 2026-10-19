//! Native `extended_*` streams on inputs the registry fixture never
//! produces: flat stretches (zero range, zero momentum, equal volume), period
//! 1, zero shifts and off-default periods. Each must match its batch kernel
//! at every bar, NaN mask included.
use super::*;
use crate::extended as x;
use crate::streaming::fixtures::input_column;
use crate::streaming::primitives::tests::assert_series;

const N: usize = 400;
const PERIODS: [usize; 5] = [1, 2, 3, 5, 14];
const FLATS: [usize; 3] = [0, 3, 40];

/// `[open, high, low, close, volume]`; the first `flat` bars and every 50th
/// bar after them are frozen at one price and one volume.
fn bars_with_flats(flat: usize) -> [Vec<f64>; 5] {
    let mut cols = ["open", "high", "low", "close", "volume"].map(|n| input_column(n, N));
    for i in 0..N {
        if i < flat || i.is_multiple_of(50) {
            for col in cols.iter_mut().take(4) {
                col[i] = 100.0;
            }
            cols[4][i] = 1000.0;
        }
    }
    cols
}

/// Run a stream over every bar; `step` returns that bar's outputs.
fn run_cols<S>(mut s: S, mut step: impl FnMut(&mut S, usize) -> Vec<f64>) -> Vec<Vec<f64>> {
    let rows: Vec<Vec<f64>> = (0..N).map(|i| step(&mut s, i)).collect();
    (0..rows[0].len())
        .map(|k| rows.iter().map(|r| r[k]).collect())
        .collect()
}

fn check(tag: &str, got: &[Vec<f64>], want: &[&[f64]]) {
    assert_eq!(got.len(), want.len(), "{tag}: output count");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert_series(&format!("{tag} out{k}"), g, w);
    }
}

/// Every flat layout, with `[open, high, low, close, volume]` bound.
fn for_each_layout(mut f: impl FnMut(usize, &[Vec<f64>; 5])) {
    for flat in FLATS {
        f(flat, &bars_with_flats(flat));
    }
}

#[test]
fn band_streams_match_batch_on_flats_and_short_periods() {
    for_each_layout(|flat, [_, h, l, c, _]| {
        for p in PERIODS {
            for q in [1, 3, 14] {
                let tag = format!("p={p} q={q} flat={flat}");
                let got = run_cols(StreamingKeltnerChannels::new(p, q, 2.0).unwrap(), |s, i| {
                    let o = s.update(h[i], l[i], c[i]);
                    vec![o.0, o.1, o.2]
                });
                let w = x::keltner_channels(h, l, c, p, q, 2.0);
                check(&format!("keltner {tag}"), &got, &[&w.0, &w.1, &w.2]);
                let got = run_cols(StreamingStarc::new(p, q, 2.0).unwrap(), |s, i| {
                    let o = s.update(h[i], l[i], c[i]);
                    vec![o.0, o.1, o.2]
                });
                let w = x::starc(h, l, c, p, q, 2.0);
                check(&format!("starc {tag}"), &got, &[&w.0, &w.1, &w.2]);
                let got = run_cols(StreamingMedianBands::new(p, q, 2.0).unwrap(), |s, i| {
                    let o = s.update(h[i], l[i], c[i]);
                    vec![o.0, o.1, o.2, o.3]
                });
                let w = x::median_bands(h, l, c, p, q, 2.0);
                check(
                    &format!("median_bands {tag}"),
                    &got,
                    &[&w.0, &w.1, &w.2, &w.3],
                );
                let got = run_cols(StreamingChandeKrollStop::new(p, 1.0, q).unwrap(), |s, i| {
                    let o = s.update(h[i], l[i], c[i]);
                    vec![o.0, o.1]
                });
                let w = x::chande_kroll_stop(h, l, c, p, 1.0, q);
                check(&format!("chande_kroll {tag}"), &got, &[&w.0, &w.1]);
                let got = run_cols(StreamingChaikinVol::new(p, q).unwrap(), |s, i| {
                    vec![s.update(h[i], l[i])]
                });
                check(
                    &format!("chaikin_vol {tag}"),
                    &got,
                    &[&x::chaikin_vol(h, l, p, q)],
                );
                // A flat stretch makes the double EMA 0, so the ratio is NaN
                // and the Mass sum restarts.
                let got = run_cols(StreamingMass::new(p, q).unwrap(), |s, i| {
                    vec![s.update(h[i], l[i])]
                });
                check(&format!("mass {tag}"), &got, &[&x::mass(h, l, p, q)]);
            }
            let got = run_cols(StreamingChandelierExit::new(p, 3.0).unwrap(), |s, i| {
                let o = s.update(h[i], l[i], c[i]);
                vec![o.0, o.1]
            });
            let w = x::chandelier_exit(h, l, c, p, 3.0);
            check(
                &format!("chandelier p={p} flat={flat}"),
                &got,
                &[&w.0, &w.1],
            );
            for matype in 0..=8u8 {
                let got = run_cols(
                    StreamingMaEnvelopes::new(p, 2.5, matype).unwrap(),
                    |s, i| {
                        let o = s.update(c[i]);
                        vec![o.0, o.1, o.2]
                    },
                );
                let w = x::ma_envelopes(c, p, 2.5, matype);
                let tag = format!("ma_envelopes p={p} m={matype} flat={flat}");
                check(&tag, &got, &[&w.0, &w.1, &w.2]);
            }
        }
    });
}

#[test]
fn oscillator_streams_match_batch_on_flats_and_short_periods() {
    for_each_layout(|flat, [_, h, l, c, _]| {
        for p in PERIODS {
            let tag = format!("p={p} flat={flat}");
            let got = run_cols(StreamingDmi::new(p).unwrap(), |s, i| {
                let o = s.update(h[i], l[i], c[i]);
                vec![o.0, o.1, o.2]
            });
            let w = x::dmi(h, l, c, p);
            check(&format!("dmi {tag}"), &got, &[&w.0, &w.1, &w.2]);
            let got = run_cols(StreamingElderRay::new(p).unwrap(), |s, i| {
                let o = s.update(h[i], l[i], c[i]);
                vec![o.0, o.1]
            });
            let w = x::elder_ray(h, l, c, p);
            check(&format!("elder_ray {tag}"), &got, &[&w.0, &w.1]);
            let got = run_cols(StreamingFisher::new(p).unwrap(), |s, i| {
                let o = s.update(h[i], l[i]);
                vec![o.0, o.1]
            });
            let w = x::fisher(h, l, p);
            check(&format!("fisher {tag}"), &got, &[&w.0, &w.1]);
            for (sp, rank) in [(1, 1), (2, 5), (2, 100)] {
                let got = run_cols(StreamingCrsi::new(p, sp, rank).unwrap(), |s, i| {
                    vec![s.update(c[i])]
                });
                let tag = format!("crsi {tag} sp={sp} rank={rank}");
                check(&tag, &got, &[&x::crsi(c, p, sp, rank)]);
            }
            for (long, short, signal) in [(p, p, p), (p, 1, 3), (25, 13, p)] {
                let got = run_cols(StreamingTsi::new(long, short, signal).unwrap(), |s, i| {
                    let o = s.update(c[i]);
                    vec![o.0, o.1]
                });
                let w = x::tsi(c, long, short, signal);
                check(
                    &format!("tsi {long}/{short}/{signal} flat={flat}"),
                    &got,
                    &[&w.0, &w.1],
                );
            }
            for (cycle, d1, d2) in [(p, 1, 1), (10, 3, p)] {
                let got = run_cols(
                    StreamingStc::new(p, p + 3, cycle, d1, d2).unwrap(),
                    |s, i| vec![s.update(c[i])],
                );
                let tag = format!("stc {tag} cycle={cycle} d={d1}/{d2}");
                check(&tag, &got, &[&x::stc(c, p, p + 3, cycle, d1, d2)]);
            }
        }
    });
}

#[test]
fn trend_streams_match_batch_on_flats_and_short_periods() {
    for_each_layout(|flat, [_, h, l, c, _]| {
        for p in PERIODS {
            let tag = format!("p={p} flat={flat}");
            for shift in [0, 1, 8] {
                let args = (p, shift, p + 1, shift / 2, 1, 0);
                let got = run_cols(
                    StreamingAlligator::new(args.0, args.1, args.2, args.3, args.4, args.5)
                        .unwrap(),
                    |s, i| {
                        let o = s.update(h[i], l[i]);
                        vec![o.0, o.1, o.2]
                    },
                );
                let w = x::alligator(h, l, args.0, args.1, args.2, args.3, args.4, args.5);
                check(
                    &format!("alligator {tag} shift={shift}"),
                    &got,
                    &[&w.0, &w.1, &w.2],
                );
                let got = run_cols(
                    StreamingGator::new(args.0, args.1, args.2, args.3, args.4, args.5).unwrap(),
                    |s, i| {
                        let o = s.update(h[i], l[i]);
                        vec![o.0, o.1]
                    },
                );
                let w = x::gator(h, l, args.0, args.1, args.2, args.3, args.4, args.5);
                check(&format!("gator {tag} shift={shift}"), &got, &[&w.0, &w.1]);
            }
            let got = run_cols(StreamingZlema::new(p).unwrap(), |s, i| vec![s.update(c[i])]);
            check(&format!("zlema {tag}"), &got, &[&x::zlema(c, p)]);
            let got = run_cols(StreamingMcginley::new(p).unwrap(), |s, i| {
                vec![s.update(c[i])]
            });
            check(&format!("mcginley {tag}"), &got, &[&x::mcginley(c, p)]);
            if p >= 2 {
                let got = run_cols(StreamingFrama::new(p).unwrap(), |s, i| vec![s.update(c[i])]);
                check(&format!("frama {tag}"), &got, &[&x::frama(c, p)]);
            }
            for cmo in [1, 9] {
                let got = run_cols(StreamingVidya::new(p, cmo).unwrap(), |s, i| {
                    vec![s.update(c[i])]
                });
                check(
                    &format!("vidya {tag} cmo={cmo}"),
                    &got,
                    &[&x::vidya(c, p, cmo)],
                );
            }
        }
    });
}

#[test]
fn volume_streams_match_batch_on_flats_and_short_periods() {
    for_each_layout(|flat, [_, h, l, c, v]| {
        let tag = format!("flat={flat}");
        let got = run_cols(StreamingNvi::new().unwrap(), |s, i| {
            vec![s.update(c[i], v[i])]
        });
        check(&format!("nvi {tag}"), &got, &[&x::nvi(c, v)]);
        let got = run_cols(StreamingPvi::new().unwrap(), |s, i| {
            vec![s.update(c[i], v[i])]
        });
        check(&format!("pvi {tag}"), &got, &[&x::pvi(c, v)]);
        let got = run_cols(StreamingPvt::new().unwrap(), |s, i| {
            vec![s.update(c[i], v[i])]
        });
        check(&format!("pvt {tag}"), &got, &[&x::pvt(c, v)]);
        for p in PERIODS {
            let tag = format!("p={p} {tag}");
            let got = run_cols(StreamingCho::new(p, p + 2).unwrap(), |s, i| {
                vec![s.update(h[i], l[i], c[i], v[i])]
            });
            check(
                &format!("cho {tag}"),
                &got,
                &[&x::cho(h, l, c, v, p, p + 2)],
            );
            let got = run_cols(StreamingForceIndex::new(p).unwrap(), |s, i| {
                vec![s.update(c[i], v[i])]
            });
            check(
                &format!("force_index {tag}"),
                &got,
                &[&x::force_index(c, v, p)],
            );
            let got = run_cols(StreamingNviWithEma::new(p).unwrap(), |s, i| {
                let o = s.update(c[i], v[i]);
                vec![o.0, o.1]
            });
            let w = x::nvi_with_ema(c, v, p);
            check(&format!("nvi_with_ema {tag}"), &got, &[&w.0, &w.1]);
            for signal in [1, 3] {
                let got = run_cols(StreamingKvo::new(p, p + 1, signal).unwrap(), |s, i| {
                    let o = s.update(h[i], l[i], c[i], v[i]);
                    vec![o.0, o.1]
                });
                let w = x::kvo(h, l, c, v, p, p + 1, signal);
                check(&format!("kvo {tag} signal={signal}"), &got, &[&w.0, &w.1]);
            }
            for matype in 0..=8u8 {
                let got = run_cols(StreamingObvSmoothed::new(p, matype).unwrap(), |s, i| {
                    vec![s.update(c[i], v[i])]
                });
                let want = x::obv_smoothed(c, v, p, matype);
                check(&format!("obv_smoothed {tag} m={matype}"), &got, &[&want]);
                let got = run_cols(StreamingPviWithSignal::new(p, matype).unwrap(), |s, i| {
                    let o = s.update(c[i], v[i]);
                    vec![o.0, o.1]
                });
                let w = x::pvi_with_signal(c, v, p, matype);
                check(
                    &format!("pvi_with_signal {tag} m={matype}"),
                    &got,
                    &[&w.0, &w.1],
                );
            }
        }
    });
}

#[test]
fn fast_slow_and_shift_rules_are_enforced() {
    assert!(StreamingCho::new(10, 10).is_err());
    assert!(StreamingKvo::new(55, 34, 13).is_err());
    assert!(StreamingStc::new(50, 23, 10, 3, 3).is_err());
    assert!(StreamingAo::new(34, 5).is_err());
    assert!(StreamingAlma::new(21, 0.85, 0.0).is_err());
    assert!(StreamingFrama::new(1).is_err());
    assert!(StreamingRwi::new(1).is_err());
    assert!(StreamingAlligator::new(13, 0, 8, 0, 5, 0).is_ok());
    assert!(
        StreamingAlligator::new(13, crate::streaming::MAX_STREAM_PERIOD + 1, 8, 5, 5, 3).is_err()
    );
}
