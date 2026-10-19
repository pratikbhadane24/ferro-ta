//! Each primitive against the batch kernel it mirrors.
use super::*;
use crate::streaming::fixtures::input_column;

const N: usize = 600;
const REL_TOL: f64 = 1e-9;
const PERIODS: [usize; 6] = [1, 2, 3, 7, 14, 30];

/// Same NaN mask, values within `REL_TOL · max(1, |want|)`.
pub(crate) fn assert_series(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let ok = if w.is_nan() {
            g.is_nan()
        } else {
            (g - w).abs() <= REL_TOL * w.abs().max(1.0)
        };
        assert!(ok, "{label} bar {i}: stream {g} != batch {w}");
    }
}

fn run(input: &[f64], mut step: impl FnMut(f64) -> f64) -> Vec<f64> {
    input.iter().map(|&x| step(x)).collect()
}

fn close() -> Vec<f64> {
    input_column("close", N)
}

#[test]
fn ma_stream_matches_compute_ma_by_type_for_every_matype() {
    let close = close();
    for matype in 0..=8u8 {
        for p in PERIODS {
            let mut s = MaStream::new(p, matype).unwrap();
            let want = crate::overlap::ma(&close, p, matype);
            assert_series(
                &format!("ma p={p} matype={matype}"),
                &run(&close, |x| s.update(x)),
                &want,
            );
            s.validate_as(p, matype).unwrap();
        }
    }
}

#[test]
fn ema_ta_skips_a_leading_nan_prefix() {
    let mut input = vec![f64::NAN; 5];
    input.extend(close());
    for p in PERIODS {
        let mut s = EmaTa::new(p).unwrap();
        let want = crate::overlap::ema(&input, p);
        assert_series(&format!("ema p={p}"), &run(&input, |x| s.update(x)), &want);
    }
}

#[test]
fn ema_first_finite_restarts_its_seed_run_on_non_finite() {
    let mut input = close();
    // Gaps inside the first seed windows, then a poisoning NaN late on.
    input[3] = f64::NAN;
    input[9] = f64::INFINITY;
    input[500] = f64::NAN;
    for p in PERIODS {
        let mut s = EmaFirstFinite::new(p).unwrap();
        let want = crate::overlap::ema_from_first_finite(&input, p);
        assert_series(
            &format!("ema_ff p={p}"),
            &run(&input, |x| s.update(x)),
            &want,
        );
    }
}

#[test]
fn ema_macd_matches_macd_legs() {
    let close = close();
    let mut fast = EmaMacd::new(3).unwrap();
    let mut slow = EmaMacd::new(5).unwrap();
    let mut signal = EmaMacd::new(2).unwrap();
    let (want, _, _) = crate::overlap::macd(&close, 3, 5, 2);
    let got = run(&close, |x| {
        let (f, s) = (fast.update(x), slow.update(x));
        if s.is_nan() {
            return f64::NAN;
        }
        let m = f - s;
        if signal.update(m).is_nan() {
            f64::NAN
        } else {
            m
        }
    });
    assert_series("macd line", &got, &want);
}

#[test]
fn wilder_matches_atr() {
    let (high, low, close) = (input_column("high", N), input_column("low", N), close());
    for p in PERIODS {
        let mut s = Wilder::new(p).unwrap();
        let want = crate::volatility::atr(&high, &low, &close, p);
        let mut got = vec![f64::NAN];
        got.extend(
            (1..N).map(|i| s.update(crate::volatility::true_range(high[i], low[i], close[i - 1]))),
        );
        assert_series(&format!("wilder p={p}"), &got, &want);
    }
}

#[test]
fn smma_matches_alligator() {
    let (high, low) = (input_column("high", N), input_column("low", N));
    for p in PERIODS {
        let mut s = Smma::new(p).unwrap();
        let (want, _, _) = crate::extended::alligator(&high, &low, p, 0, 1, 0, 1, 0);
        let got: Vec<f64> = high
            .iter()
            .zip(&low)
            .map(|(h, l)| s.update((h + l) / 2.0))
            .collect();
        assert_series(&format!("smma p={p}"), &got, &want);
    }
}

#[test]
fn reset_restarts_from_scratch() {
    let close = close();
    for matype in 0..=8u8 {
        let mut s = MaStream::new(7, matype).unwrap();
        let first = run(&close, |x| s.update(x));
        s.reset();
        let again = run(&close, |x| s.update(x));
        assert_eq!(
            format!("{first:?}"),
            format!("{again:?}"),
            "matype {matype}"
        );
    }
}

#[test]
fn ma_stream_rejects_bad_params_and_mismatched_state() {
    assert!(MaStream::new(0, 1).is_err());
    assert!(MaStream::new(5, 9).is_err());
    assert!(MaStream::new(crate::streaming::MAX_STREAM_PERIOD + 1, 0).is_err());
    let t3 = MaStream::new(5, 7).unwrap();
    assert!(t3.validate_as(5, 8).is_ok(), "7 and 8 are both T3");
    assert!(t3.validate_as(5, 3).is_err());
    assert!(t3.validate_as(6, 7).is_err());
    assert!(EmaCascade::new(3, 2).unwrap().validate(3).is_err());
    assert!(T3::new(3, f64::NAN).is_err());
}
