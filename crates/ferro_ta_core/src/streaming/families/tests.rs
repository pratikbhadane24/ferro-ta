//! Overlap / MA-oscillator streams across every `matype` (the registry
//! parity test only runs the defaults).
use super::*;
use crate::streaming::fixtures::input_column;
use crate::streaming::primitives::tests::assert_series;

const N: usize = 600;

fn close() -> Vec<f64> {
    input_column("close", N)
}

fn run<O>(input: &[f64], step: impl FnMut(f64) -> O) -> Vec<O> {
    input.iter().copied().map(step).collect()
}

fn unzip3(rows: Vec<(f64, f64, f64)>) -> [Vec<f64>; 3] {
    let mut out: [Vec<f64>; 3] = Default::default();
    for (a, b, c) in rows {
        out[0].push(a);
        out[1].push(b);
        out[2].push(c);
    }
    out
}

fn assert_triple(label: &str, got: Vec<(f64, f64, f64)>, want: (Vec<f64>, Vec<f64>, Vec<f64>)) {
    let got = unzip3(got);
    for (k, w) in [want.0, want.1, want.2].iter().enumerate() {
        assert_series(&format!("{label} output {k}"), &got[k], w);
    }
}

#[test]
fn apo_and_ppo_match_batch_for_every_matype() {
    let close = close();
    for matype in 0..=8u8 {
        let mut apo = StreamingApo::new(5, 12, matype).unwrap();
        let want = crate::momentum::apo(&close, 5, 12, matype);
        assert_series(
            &format!("apo matype={matype}"),
            &run(&close, |x| apo.update(x)),
            &want,
        );

        let mut ppo = StreamingPpo::new(5, 12, 4, matype).unwrap();
        let want = crate::momentum::ppo(&close, 5, 12, 4, matype);
        assert_triple(
            &format!("ppo matype={matype}"),
            run(&close, |x| ppo.update(x)),
            want,
        );
    }
}

#[test]
fn macdext_matches_batch_for_every_matype_and_mixed_legs() {
    let close = close();
    let mut combos: Vec<(u8, u8, u8)> = (0..=8).map(|t| (t, t, t)).collect();
    combos.extend([(6, 0, 3), (7, 1, 6), (3, 4, 2), (0, 8, 5), (2, 6, 7)]);
    for (fast_t, slow_t, sig_t) in combos {
        let mut s = StreamingMacdext::new(5, fast_t, 12, slow_t, 4, sig_t).unwrap();
        let want = crate::overlap::macdext(&close, 5, fast_t, 12, slow_t, 4, sig_t);
        let label = format!("macdext {fast_t}/{slow_t}/{sig_t}");
        assert_triple(&label, run(&close, |x| s.update(x)), want);
    }
}

#[test]
fn macdfix_matches_batch_for_several_signal_periods() {
    let close = close();
    for signal in [1, 2, 9, 30] {
        let mut s = StreamingMacdfix::new(signal).unwrap();
        let want = crate::overlap::macdfix(&close, signal);
        assert_triple(
            &format!("macdfix {signal}"),
            run(&close, |x| s.update(x)),
            want,
        );
    }
}

#[test]
fn mavp_clamps_rounded_periods_for_every_matype() {
    let close = close();
    // 0.4 .. 36.4: rounds to 0..=36, so both clamp bounds are exercised.
    let periods: Vec<f64> = (0..N).map(|i| (i % 37) as f64 + 0.4).collect();
    for matype in 0..=8u8 {
        let mut s = StreamingMavp::new(3, 20, matype).unwrap();
        let want = crate::overlap::mavp(&close, &periods, 3, 20, matype);
        let got: Vec<f64> = close
            .iter()
            .zip(&periods)
            .map(|(&c, &p)| s.update(c, p))
            .collect();
        assert_series(&format!("mavp matype={matype}"), &got, &want);
    }
}

#[test]
fn t3_trix_and_kama_match_batch_off_default() {
    let close = close();
    for (p, v) in [(1, 0.7), (3, 0.0), (4, 0.5), (8, 1.2)] {
        let mut s = StreamingT3::new(p, v).unwrap();
        let want = crate::overlap::t3(&close, p, v);
        assert_series(&format!("t3 {p} {v}"), &run(&close, |x| s.update(x)), &want);
    }
    for p in [1, 2, 5, 40] {
        let mut s = StreamingTrix::new(p).unwrap();
        let want = crate::momentum::trix(&close, p);
        assert_series(&format!("trix {p}"), &run(&close, |x| s.update(x)), &want);
        let mut s = StreamingKama::new(p).unwrap();
        let want = crate::overlap::kama(&close, p);
        assert_series(&format!("kama {p}"), &run(&close, |x| s.update(x)), &want);
    }
}

#[test]
fn invalid_params_are_rejected() {
    assert!(StreamingApo::new(12, 12, 1).is_err());
    assert!(StreamingPpo::new(5, 12, 0, 1).is_err());
    assert!(StreamingPpo::new(5, 12, 9, 9).is_err());
    assert!(StreamingMacdext::new(26, 1, 12, 1, 9, 1).is_err());
    assert!(StreamingMacdext::new(12, 1, 26, 1, 9, 9).is_err());
    assert!(StreamingMacdfix::new(0).is_err());
    assert!(StreamingMavp::new(10, 5, 0).is_err());
    assert!(StreamingMa::new(30, 9).is_err());
    assert!(StreamingT3::new(5, f64::INFINITY).is_err());
    assert!(StreamingDema::new(0).is_err());
    assert_eq!(StreamingMavp::new(2, 30, 3).unwrap().maxperiod(), 30);
}

/// A snapshot cannot swap the average type or count under the params.
#[cfg(feature = "serde")]
#[test]
fn restore_rejects_state_that_contradicts_params() {
    use crate::streaming::Live;
    fn load_edited<T>(live: &Live<T>, edit: impl FnOnce(&mut serde_json::Value)) -> bool
    where
        T: crate::streaming::StreamingIndicator + serde::Serialize + serde::de::DeserializeOwned,
    {
        let mut value: serde_json::Value = serde_json::from_str(&live.save().unwrap()).unwrap();
        edit(&mut value["state"]["inner"]);
        Live::<T>::load(&value.to_string()).is_ok()
    }
    let ma = Live::new(StreamingMa::new(5, 0).unwrap());
    assert!(load_edited(&ma, |_| {}));
    assert!(!load_edited(&ma, |s| s["matype"] = serde_json::json!(2)));
    let mavp = Live::new(StreamingMavp::new(2, 4, 0).unwrap());
    assert!(load_edited(&mavp, |_| {}));
    assert!(!load_edited(&mavp, |s| s["maxperiod"] = serde_json::json!(5)));
    let macdext = Live::new(StreamingMacdext::new(3, 1, 6, 1, 2, 1).unwrap());
    assert!(!load_edited(&macdext, |s| s["slowmatype"] = serde_json::json!(4)));
}
