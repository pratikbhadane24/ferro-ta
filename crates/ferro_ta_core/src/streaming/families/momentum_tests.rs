//! Momentum / volatility / volume streams on inputs the registry fixture
//! never produces: flat ranges, zero divisors, tied extremes, period 1 and
//! off-default periods.
use super::*;
use crate::streaming::fixtures::input_column;
use crate::streaming::primitives::tests::assert_series;

const N: usize = 400;

/// `(open, high, low, close, volume)` with the first `flat` bars frozen at
/// one price (zero true range) and every 50th bar after that flat too.
fn bars_with_flats(flat: usize) -> [Vec<f64>; 5] {
    let mut cols = ["open", "high", "low", "close", "volume"].map(|n| input_column(n, N));
    for i in 0..N {
        if i < flat || i % 50 == 0 {
            let price = 100.0;
            for col in 0..4 {
                cols[col][i] = price;
            }
        }
    }
    cols
}

fn collect(cols: &[Vec<f64>; 5], mut step: impl FnMut(&[f64; 5]) -> f64) -> Vec<f64> {
    (0..N)
        .map(|i| step(&[cols[0][i], cols[1][i], cols[2][i], cols[3][i], cols[4][i]]))
        .collect()
}

#[test]
fn directional_streams_match_batch_with_flat_prefixes_and_period_one() {
    for flat in [0, 3, 40] {
        let [_, h, l, c, _] = bars_with_flats(flat);
        for p in [1, 2, 3, 5, 14] {
            let cols = bars_with_flats(flat);
            let tag = |name: &str| format!("{name} p={p} flat={flat}");
            let mut s = StreamingPlusDm::new(p).unwrap();
            let got = collect(&cols, |r| s.update(r[1], r[2]));
            assert_series(&tag("plus_dm"), &got, &crate::momentum::plus_dm(&h, &l, p));
            let mut s = StreamingMinusDm::new(p).unwrap();
            let got = collect(&cols, |r| s.update(r[1], r[2]));
            assert_series(
                &tag("minus_dm"),
                &got,
                &crate::momentum::minus_dm(&h, &l, p),
            );
            let mut s = StreamingPlusDi::new(p).unwrap();
            let got = collect(&cols, |r| s.update(r[1], r[2], r[3]));
            assert_series(
                &tag("plus_di"),
                &got,
                &crate::momentum::plus_di(&h, &l, &c, p),
            );
            let mut s = StreamingMinusDi::new(p).unwrap();
            let got = collect(&cols, |r| s.update(r[1], r[2], r[3]));
            assert_series(
                &tag("minus_di"),
                &got,
                &crate::momentum::minus_di(&h, &l, &c, p),
            );
            let mut s = StreamingDx::new(p).unwrap();
            let got = collect(&cols, |r| s.update(r[1], r[2], r[3]));
            assert_series(&tag("dx"), &got, &crate::momentum::dx(&h, &l, &c, p));
            let mut s = StreamingAdx::new(p).unwrap();
            let got = collect(&cols, |r| s.update(r[1], r[2], r[3]));
            let want = crate::momentum::adx(&h, &l, &c, p);
            assert_series(&tag("adx"), &got, &want);
            let mut s = StreamingAdxr::new(p).unwrap();
            let got = collect(&cols, |r| s.update(r[1], r[2], r[3]));
            assert_series(&tag("adxr"), &got, &crate::momentum::adxr(&h, &l, &c, p));
        }
    }
}

/// A frozen start leaves a seed DX at NaN, so the batch never emits an ADX;
/// the stream must stay dead too (and the case must really occur).
#[test]
fn adx_stays_nan_when_a_seed_dx_is_undefined() {
    let [_, h, l, c, _] = bars_with_flats(40);
    let batch = crate::momentum::adx(&h, &l, &c, 5);
    assert!(
        batch.iter().all(|v| v.is_nan()),
        "fixture no longer kills ADX"
    );
    let mut s = StreamingAdx::new(5).unwrap();
    assert!((0..N).all(|i| s.update(h[i], l[i], c[i]).is_nan()));
}

#[test]
fn range_oscillators_match_batch_on_flat_ranges_and_ties() {
    let cols = bars_with_flats(10);
    let [o, h, l, c, v] = &cols;
    for p in [1, 2, 5, 14] {
        let mut s = StreamingStochf::new(p, 3, 0).unwrap();
        let (fk, fd) = crate::momentum::stochf(h, l, c, p, 3, 0);
        let got: Vec<(f64, f64)> = (0..N).map(|i| s.update(h[i], l[i], c[i])).collect();
        assert_series(
            &format!("stochf fastk p={p}"),
            &got.iter().map(|g| g.0).collect::<Vec<_>>(),
            &fk,
        );
        assert_series(
            &format!("stochf fastd p={p}"),
            &got.iter().map(|g| g.1).collect::<Vec<_>>(),
            &fd,
        );
        let mut w = StreamingWillr::new(p).unwrap();
        let got = collect(&cols, |r| w.update((r[1], r[2], r[3])));
        assert_series(
            &format!("willr p={p}"),
            &got,
            &crate::momentum::willr(h, l, c, p),
        );
        let mut a = StreamingAroon::new(p).unwrap();
        let (down, up) = crate::momentum::aroon(h, l, p);
        let got: Vec<(f64, f64)> = (0..N).map(|i| a.update((h[i], l[i]))).collect();
        assert_series(
            &format!("aroon down p={p}"),
            &got.iter().map(|g| g.0).collect::<Vec<_>>(),
            &down,
        );
        assert_series(
            &format!("aroon up p={p}"),
            &got.iter().map(|g| g.1).collect::<Vec<_>>(),
            &up,
        );
    }
    let mut b = StreamingBop::new().unwrap();
    assert_series(
        "bop",
        &collect(&cols, |r| b.update((r[0], r[1], r[2], r[3]))),
        &crate::momentum::bop(o, h, l, c),
    );
    let mut m = StreamingMfi::new(5).unwrap();
    let got = collect(&cols, |r| m.update(r[1], r[2], r[3], r[4]));
    assert_series("mfi", &got, &crate::volume::mfi(h, l, c, v, 5));
}

#[test]
fn zero_divisors_give_nan_like_batch() {
    // Closes touch 0 so roc*/natr see a zero divisor.
    let mut close = input_column("close", N);
    for i in (0..N).step_by(7) {
        close[i] = 0.0;
    }
    let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
    let low: Vec<f64> = close.iter().map(|c| (c - 1.0).max(0.0)).collect();
    for p in [1, 3, 7] {
        let mut r = StreamingRoc::new(p).unwrap();
        let got: Vec<f64> = close.iter().map(|&x| r.update(x)).collect();
        assert_series(
            &format!("roc p={p}"),
            &got,
            &crate::momentum::roc(&close, p),
        );
        let mut r = StreamingRocr100::new(p).unwrap();
        let got: Vec<f64> = close.iter().map(|&x| r.update(x)).collect();
        assert_series(
            &format!("rocr100 p={p}"),
            &got,
            &crate::momentum::rocr100(&close, p),
        );
        let mut n = StreamingNatr::new(p).unwrap();
        let got: Vec<f64> = (0..N)
            .map(|i| n.update(high[i], low[i], close[i]))
            .collect();
        assert_series(
            &format!("natr p={p}"),
            &got,
            &crate::volatility::natr(&high, &low, &close, p),
        );
    }
}

#[test]
fn volume_and_oscillator_streams_match_batch_off_default() {
    let cols = bars_with_flats(10);
    let [_, h, l, c, v] = &cols;
    let mut obv = StreamingObv::new().unwrap();
    assert_series(
        "obv",
        &collect(&cols, |r| obv.update(r[3], r[4])),
        &crate::volume::obv(c, v),
    );
    let mut ad = StreamingAd::new().unwrap();
    assert_series(
        "ad",
        &collect(&cols, |r| ad.update(r[1], r[2], r[3], r[4])),
        &crate::volume::ad(h, l, c, v),
    );
    for (fast, slow) in [(1, 1), (1, 2), (3, 10), (5, 3), (12, 40)] {
        let mut s = StreamingAdosc::new(fast, slow).unwrap();
        let got = collect(&cols, |r| s.update(r[1], r[2], r[3], r[4]));
        let want = crate::volume::adosc(h, l, c, v, fast, slow);
        assert_series(&format!("adosc {fast}/{slow}"), &got, &want);
    }
    for p in [1, 2, 14, 30] {
        let mut s = StreamingCmo::new(p).unwrap();
        let got = collect(&cols, |r| s.update(r[3]));
        assert_series(&format!("cmo p={p}"), &got, &crate::momentum::cmo(c, p));
    }
}

#[test]
fn stochrsi_matches_batch_for_every_matype_and_flat_stretches() {
    let cols = bars_with_flats(30);
    let c = &cols[3];
    for matype in 0..=8u8 {
        for (p, k, d) in [(1, 1, 1), (3, 2, 2), (5, 4, 3), (14, 5, 3)] {
            let mut s = StreamingStochrsi::new(p, k, d, matype).unwrap();
            let (fk, fd) = crate::momentum::stochrsi(c, p, k, d, matype);
            let got: Vec<(f64, f64)> = c.iter().map(|&x| s.update(x)).collect();
            let label = format!("stochrsi {p}/{k}/{d} matype={matype}");
            assert_series(
                &format!("{label} fastk"),
                &got.iter().map(|g| g.0).collect::<Vec<_>>(),
                &fk,
            );
            assert_series(
                &format!("{label} fastd"),
                &got.iter().map(|g| g.1).collect::<Vec<_>>(),
                &fd,
            );
        }
    }
}

#[test]
fn stochf_and_ultosc_match_batch_for_every_matype_and_off_default_periods() {
    let cols = bars_with_flats(5);
    let [_, h, l, c, _] = &cols;
    for matype in 0..=8u8 {
        let mut s = StreamingStochf::new(4, 3, matype).unwrap();
        let (fk, fd) = crate::momentum::stochf(h, l, c, 4, 3, matype);
        let got: Vec<(f64, f64)> = (0..N).map(|i| s.update(h[i], l[i], c[i])).collect();
        let label = format!("stochf matype={matype}");
        assert_series(
            &format!("{label} fastk"),
            &got.iter().map(|g| g.0).collect::<Vec<_>>(),
            &fk,
        );
        assert_series(
            &format!("{label} fastd"),
            &got.iter().map(|g| g.1).collect::<Vec<_>>(),
            &fd,
        );
    }
    for (a, b, d) in [(7, 14, 28), (14, 7, 3), (1, 1, 1), (28, 14, 7), (2, 9, 4)] {
        let mut s = StreamingUltosc::new(a, b, d).unwrap();
        let got = collect(&cols, |r| s.update((r[1], r[2], r[3])));
        let want = crate::momentum::ultosc(h, l, c, a, b, d);
        assert_series(&format!("ultosc {a}/{b}/{d}"), &got, &want);
    }
}

#[test]
fn invalid_params_are_rejected() {
    assert!(StreamingAdx::new(0).is_err());
    assert!(StreamingPlusDm::new(0).is_err());
    assert!(StreamingCmo::new(0).is_err());
    assert!(StreamingStochf::new(0, 3, 0).is_err());
    assert!(StreamingStochf::new(5, 3, 9).is_err());
    assert!(StreamingStochrsi::new(14, 5, 0, 0).is_err());
    assert!(StreamingStochrsi::new(14, 5, 3, 9).is_err());
    assert!(StreamingAdosc::new(0, 10).is_err());
    assert!(StreamingNatr::new(0).is_err());
    assert!(StreamingAdx::new(crate::streaming::MAX_STREAM_PERIOD + 1).is_err());
}

/// Edited snapshots cannot desynchronise a ring from its counters or swap
/// the %D flavour under `fastd_matype`.
#[cfg(feature = "serde")]
#[test]
fn restore_rejects_state_that_contradicts_params() {
    use crate::streaming::{Live, StreamingIndicator};
    use serde_json::{json, Value};

    fn load_edited<T>(
        stream: T,
        input: impl Fn(usize) -> T::Input,
        edit: impl FnOnce(&mut Value),
    ) -> bool
    where
        T: StreamingIndicator + serde::Serialize + serde::de::DeserializeOwned,
    {
        let mut live = Live::new(stream);
        for i in 0..40 {
            live.update(input(i), true);
        }
        let mut value: Value = serde_json::from_str(&live.save().unwrap()).unwrap();
        edit(&mut value["state"]["inner"]);
        Live::<T>::load(&value.to_string()).is_ok()
    }
    let [_, h, l, c, v] = bars_with_flats(0);
    let hlc = |i: usize| (h[i], l[i], c[i]);
    let hlcv = |i: usize| (h[i], l[i], c[i], v[i]);

    let adxr = || StreamingAdxr::new(5).unwrap();
    assert!(load_edited(adxr(), hlc, |_| {}));
    assert!(!load_edited(adxr(), hlc, |s| s["adx_ring"] = json!([1.0])));
    assert!(!load_edited(adxr(), hlc, |s| s["state"]["period"] = json!(0)));
    let adx = || StreamingAdx::new(5).unwrap();
    assert!(load_edited(adx(), hlc, |_| {}));
    assert!(!load_edited(adx(), hlc, |s| s["state"]["period"] = json!(1 << 25)));

    let sma_d = || StreamingStochrsi::new(5, 4, 3, 0).unwrap();
    let typed_d = || StreamingStochrsi::new(5, 4, 3, 1).unwrap();
    assert!(load_edited(sma_d(), |i| c[i], |_| {}));
    assert!(load_edited(typed_d(), |i| c[i], |_| {}));
    assert!(!load_edited(
        sma_d(),
        |i| c[i],
        |s| s["fastd_matype"] = json!(1)
    ));
    assert!(!load_edited(
        typed_d(),
        |i| c[i],
        |s| s["fastd_matype"] = json!(0)
    ));
    assert!(!load_edited(
        sma_d(),
        |i| c[i],
        |s| s["rsi_window"] = json!([1, 2, 3, 4, 5, 6])
    ));
    let stochf = || StreamingStochf::new(5, 3, 1).unwrap();
    assert!(load_edited(stochf(), hlc, |_| {}));
    assert!(!load_edited(stochf(), hlc, |s| s["lows"] = json!([1.0])));
    assert!(!load_edited(stochf(), hlc, |s| s["fastd_matype"] = json!(0)));
    assert!(!load_edited(stochf(), hlc, |s| s["fastk_period"] = json!(0)));
    let adosc = || StreamingAdosc::new(3, 10).unwrap();
    assert!(load_edited(adosc(), hlcv, |_| {}));
    assert!(!load_edited(adosc(), hlcv, |s| s["slowperiod"] = json!(0)));
    let natr = || StreamingNatr::new(5).unwrap();
    assert!(load_edited(natr(), hlc, |_| {}));
    assert!(!load_edited(natr(), hlc, |s| s["atr"]["period"] = json!(0)));
}
