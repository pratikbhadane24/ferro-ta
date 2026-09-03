//! Parity tests for the generated Flutter bridge wrappers.
//!
//! The wrappers in `api::indicators` are generated from the WASM signatures and
//! must be pure passthroughs to `ferro_ta_core`. These tests assert that the
//! bridge surface returns exactly what the core crate returns, so a bad
//! generator transform (wrong argument order, dropped parameter) fails here
//! rather than silently shipping wrong numbers to Flutter apps.

use ferro_ta_flutter::api::indicators;

fn closes() -> Vec<f64> {
    vec![
        10.0, 11.0, 12.0, 11.5, 13.0, 14.0, 13.5, 15.0, 16.0, 15.5, 17.0, 18.0, 17.5, 19.0, 20.0,
    ]
}

fn highs() -> Vec<f64> {
    closes().iter().map(|c| c + 1.0).collect()
}

fn lows() -> Vec<f64> {
    closes().iter().map(|c| c - 1.0).collect()
}

fn volumes() -> Vec<f64> {
    (1..=15).map(|i| (i as f64) * 100.0).collect()
}

/// Compare two f64 slices treating NaN as equal (warm-up values are NaN).
fn assert_same(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        if a.is_nan() && e.is_nan() {
            continue;
        }
        assert!((a - e).abs() < 1e-12, "index {i}: {a} != {e}");
    }
}

#[test]
fn sma_matches_core() {
    let c = closes();
    assert_same(
        &indicators::sma(c.clone(), 3),
        &ferro_ta_core::overlap::sma(&c, 3),
    );
}

#[test]
fn ema_matches_core() {
    let c = closes();
    assert_same(
        &indicators::ema(c.clone(), 5),
        &ferro_ta_core::overlap::ema(&c, 5),
    );
}

#[test]
fn wma_matches_core() {
    let c = closes();
    assert_same(
        &indicators::wma(c.clone(), 4),
        &ferro_ta_core::overlap::wma(&c, 4),
    );
}

#[test]
fn rsi_matches_core() {
    let c = closes();
    assert_same(
        &indicators::rsi(c.clone(), 5),
        &ferro_ta_core::momentum::rsi(&c, 5),
    );
}

#[test]
fn bbands_tuple_order_matches_core() {
    let c = closes();
    let (u, m, l) = indicators::bbands(c.clone(), 5, 2.0, 2.0, 0);
    let (cu, cm, cl) = ferro_ta_core::overlap::bbands(&c, 5, 2.0, 2.0, 0);
    assert_same(&u, &cu);
    assert_same(&m, &cm);
    assert_same(&l, &cl);

    // `matype` must reach the core: the T3-centred bands differ from the SMA
    // default, and the generated wrapper has to forward the argument.
    let (u8_, m8, l8) = indicators::bbands(c.clone(), 5, 2.0, 2.0, 8);
    let (cu8, cm8, cl8) = ferro_ta_core::overlap::bbands(&c, 5, 2.0, 2.0, 8);
    assert_same(&u8_, &cu8);
    assert_same(&m8, &cm8);
    assert_same(&l8, &cl8);
}

#[test]
fn macd_tuple_order_matches_core() {
    let c = closes();
    let (m, s, h) = indicators::macd(c.clone(), 3, 6, 3);
    let (cm, cs, ch) = ferro_ta_core::overlap::macd(&c, 3, 6, 3);
    assert_same(&m, &cm);
    assert_same(&s, &cs);
    assert_same(&h, &ch);
}

/// Multi-array input: argument order (high, low, close) must be preserved.
#[test]
fn atr_multi_array_arg_order_matches_core() {
    let (h, l, c) = (highs(), lows(), closes());
    assert_same(
        &indicators::atr(h.clone(), l.clone(), c.clone(), 5),
        &ferro_ta_core::volatility::atr(&h, &l, &c, 5),
    );
}

#[test]
fn adx_multi_array_arg_order_matches_core() {
    let (h, l, c) = (highs(), lows(), closes());
    assert_same(
        &indicators::adx(h.clone(), l.clone(), c.clone(), 5),
        &ferro_ta_core::momentum::adx(&h, &l, &c, 5),
    );
}

#[test]
fn obv_matches_core() {
    let (c, v) = (closes(), volumes());
    assert_same(
        &indicators::obv(c.clone(), v.clone()),
        &ferro_ta_core::volume::obv(&c, &v),
    );
}

/// Four-array input — the widest arg-order surface in the generated module.
#[test]
fn ad_four_array_arg_order_matches_core() {
    let (h, l, c, v) = (highs(), lows(), closes(), volumes());
    assert_same(
        &indicators::ad(h.clone(), l.clone(), c.clone(), v.clone()),
        &ferro_ta_core::volume::ad(&h, &l, &c, &v),
    );
}

/// Scalar-returning wrapper.
#[test]
fn scalar_return_matches_core() {
    let c = closes();
    let a = indicators::rolling_max(c.clone(), 5);
    let b = ferro_ta_core::math_ops::rolling_max(&c, 5);
    assert_same(&a, &b);
}

/// `Vec<i8>` return path (alerts/pattern surfaces).
#[test]
fn int8_return_matches_core() {
    let c = closes();
    let a = indicators::check_threshold(c.clone(), 14.0, 1);
    let b = ferro_ta_core::alerts::check_threshold(&c, 14.0, 1);
    assert_eq!(a, b);
}

// ---------------------------------------------------------------------------
// Extended catalog parity
//
// These wrappers carry the widest positional surfaces in the generated module
// (eight `usize` shift/period arguments, ten ROC/SMA windows, `u8` matype
// selectors, four output arrays). A generator transform that dropped or
// reordered one of those would still compile, so each is pinned against the
// core kernel here.
// ---------------------------------------------------------------------------

/// 40 bars — long enough for the slower warmups (KST, KVO, STC).
fn long_closes() -> Vec<f64> {
    (0..40)
        .map(|i| {
            let x = i as f64;
            100.0 + x * 0.5 + (x * 0.7).sin() * 3.0
        })
        .collect()
}

fn long_opens() -> Vec<f64> {
    long_closes().iter().map(|c| c - 0.25).collect()
}

fn long_highs() -> Vec<f64> {
    long_closes().iter().map(|c| c + 1.5).collect()
}

fn long_lows() -> Vec<f64> {
    long_closes().iter().map(|c| c - 1.5).collect()
}

fn long_volumes() -> Vec<f64> {
    (0..40).map(|i| 1000.0 + ((i % 7) as f64) * 250.0).collect()
}

/// Eight positional period/shift arguments — the widest `usize` surface.
#[test]
fn alligator_arg_order_matches_core() {
    let (h, l) = (long_highs(), long_lows());
    let (jaw, teeth, lips) = indicators::alligator(h.clone(), l.clone(), 13, 8, 8, 5, 5, 3);
    let (cj, ct, cl) = ferro_ta_core::extended::alligator(&h, &l, 13, 8, 8, 5, 5, 3);
    assert_same(&jaw, &cj);
    assert_same(&teeth, &ct);
    assert_same(&lips, &cl);
}

#[test]
fn gator_arg_order_matches_core() {
    let (h, l) = (long_highs(), long_lows());
    let (upper, lower) = indicators::gator(h.clone(), l.clone(), 13, 8, 8, 5, 5, 3);
    let (cu, cl) = ferro_ta_core::extended::gator(&h, &l, 13, 8, 8, 5, 5, 3);
    assert_same(&upper, &cu);
    assert_same(&lower, &cl);
}

/// Ten positional windows — the widest scalar surface in the module.
#[test]
fn kst_arg_order_matches_core() {
    let c = long_closes();
    let (kst, signal) = indicators::kst(c.clone(), 10, 15, 20, 30, 10, 10, 10, 15, 9);
    let (ck, cs) = ferro_ta_core::extended::kst(&c, 10, 15, 20, 30, 10, 10, 10, 15, 9);
    assert_same(&kst, &ck);
    assert_same(&signal, &cs);
}

#[test]
fn kvo_arg_order_matches_core() {
    let (h, l, c, v) = (long_highs(), long_lows(), long_closes(), long_volumes());
    let (kvo, signal) = indicators::kvo(h.clone(), l.clone(), c.clone(), v.clone(), 34, 55, 13);
    let (ck, cs) = ferro_ta_core::extended::kvo(&h, &l, &c, &v, 34, 55, 13);
    assert_same(&kvo, &ck);
    assert_same(&signal, &cs);
}

/// Four output arrays.
#[test]
fn median_bands_tuple_order_matches_core() {
    let (h, l, c) = (long_highs(), long_lows(), long_closes());
    let (m, u, lo, ema) = indicators::median_bands(h.clone(), l.clone(), c.clone(), 3, 14, 2.0);
    let (cm, cu, clo, cema) = ferro_ta_core::extended::median_bands(&h, &l, &c, 3, 14, 2.0);
    assert_same(&m, &cm);
    assert_same(&u, &cu);
    assert_same(&lo, &clo);
    assert_same(&ema, &cema);
}

/// Four input arrays in OHLC order (not the HLCV order used elsewhere).
#[test]
fn rvi_arg_order_matches_core() {
    let (o, h, l, c) = (long_opens(), long_highs(), long_lows(), long_closes());
    let (rvi, signal) = indicators::rvi(o.clone(), h.clone(), l.clone(), c.clone(), 10);
    let (cr, cs) = ferro_ta_core::extended::rvi(&o, &h, &l, &c, 10);
    assert_same(&rvi, &cr);
    assert_same(&signal, &cs);
}

#[test]
fn dmi_tuple_order_matches_core() {
    let (h, l, c) = (long_highs(), long_lows(), long_closes());
    let (p, m, adx) = indicators::dmi(h.clone(), l.clone(), c.clone(), 14);
    let (cp, cm, cadx) = ferro_ta_core::extended::dmi(&h, &l, &c, 14);
    assert_same(&p, &cp);
    assert_same(&m, &cm);
    assert_same(&adx, &cadx);
}

#[test]
fn starc_tuple_order_matches_core() {
    let (h, l, c) = (long_highs(), long_lows(), long_closes());
    let (u, m, lo) = indicators::starc(h.clone(), l.clone(), c.clone(), 15, 15, 2.0);
    let (cu, cm, clo) = ferro_ta_core::extended::starc(&h, &l, &c, 15, 15, 2.0);
    assert_same(&u, &cu);
    assert_same(&m, &cm);
    assert_same(&lo, &clo);
}

#[test]
fn chande_kroll_stop_arg_order_matches_core() {
    let (h, l, c) = (long_highs(), long_lows(), long_closes());
    let (long_stop, short_stop) =
        indicators::chande_kroll_stop(h.clone(), l.clone(), c.clone(), 10, 1.0, 9);
    let (cls, css) = ferro_ta_core::extended::chande_kroll_stop(&h, &l, &c, 10, 1.0, 9);
    assert_same(&long_stop, &cls);
    assert_same(&short_stop, &css);
}

#[test]
fn williams_fractals_tuple_order_matches_core() {
    let (h, l) = (long_highs(), long_lows());
    let (up, down) = indicators::williams_fractals(h.clone(), l.clone(), 2);
    let (cu, cd) = ferro_ta_core::extended::williams_fractals(&h, &l, 2);
    assert_same(&up, &cu);
    assert_same(&down, &cd);
}

#[test]
fn rwi_tuple_order_matches_core() {
    let (h, l, c) = (long_highs(), long_lows(), long_closes());
    let (hi, lo) = indicators::rwi(h.clone(), l.clone(), c.clone(), 10);
    let (chi, clo) = ferro_ta_core::extended::rwi(&h, &l, &c, 10);
    assert_same(&hi, &chi);
    assert_same(&lo, &clo);
}

/// `u8` matype selector must survive the transform.
#[test]
fn matype_selectors_match_core() {
    let (c, v) = (long_closes(), long_volumes());
    assert_same(
        &indicators::obv_smoothed(c.clone(), v.clone(), 10, 1),
        &ferro_ta_core::extended::obv_smoothed(&c, &v, 10, 1),
    );
    let (u, m, lo) = indicators::ma_envelopes(c.clone(), 10, 2.5, 0);
    let (cu, cm, clo) = ferro_ta_core::extended::ma_envelopes(&c, 10, 2.5, 0);
    assert_same(&u, &cu);
    assert_same(&m, &cm);
    assert_same(&lo, &clo);
    let (pvi, signal) = indicators::pvi_with_signal(c.clone(), v.clone(), 10, 0);
    let (cp, cs) = ferro_ta_core::extended::pvi_with_signal(&c, &v, 10, 0);
    assert_same(&pvi, &cp);
    assert_same(&signal, &cs);
}

/// Mixed `usize` + `f64` tail arguments.
#[test]
fn scalar_tail_arguments_match_core() {
    let c = long_closes();
    assert_same(
        &indicators::historical_volatility(c.clone(), 20, 252.0),
        &ferro_ta_core::extended::historical_volatility(&c, 20, 252.0),
    );
    assert_same(
        &indicators::crsi(c.clone(), 3, 2, 20),
        &ferro_ta_core::extended::crsi(&c, 3, 2, 20),
    );
    assert_same(
        &indicators::mode(c.clone(), 5, 4),
        &ferro_ta_core::extended::mode(&c, 5, 4),
    );
    assert_same(
        &indicators::stc(c.clone(), 23, 50, 10, 3, 3),
        &ferro_ta_core::extended::stc(&c, 23, 50, 10, 3, 3),
    );
}

/// Signal utilities: two-series and condition/value argument order.
#[test]
fn signal_utilities_match_core() {
    let c = long_closes();
    let fast = ferro_ta_core::overlap::sma(&c, 3);
    let slow = ferro_ta_core::overlap::sma(&c, 8);
    assert_same(
        &indicators::crossunder(fast.clone(), slow.clone()),
        &ferro_ta_core::utils::crossunder(&fast, &slow),
    );
    assert_same(
        &indicators::cross(fast.clone(), slow.clone()),
        &ferro_ta_core::utils::cross(&fast, &slow),
    );
    let up = ferro_ta_core::utils::crossover(&fast, &slow);
    let down = ferro_ta_core::utils::crossunder(&fast, &slow);
    assert_same(
        &indicators::exrem(up.clone(), down.clone()),
        &ferro_ta_core::utils::exrem(&up, &down),
    );
    assert_same(
        &indicators::flip(up.clone(), down.clone()),
        &ferro_ta_core::utils::flip(&up, &down),
    );
    // condition first, then the value series.
    assert_same(
        &indicators::valuewhen(up.clone(), c.clone(), 1),
        &ferro_ta_core::utils::valuewhen(&up, &c, 1),
    );
    assert_same(
        &indicators::lowest(c.clone(), 5),
        &ferro_ta_core::utils::lowest(&c, 5),
    );
    assert_same(
        &indicators::change(c.clone(), 3),
        &ferro_ta_core::utils::change(&c, 3),
    );
    assert_same(
        &indicators::rising(c.clone(), 3),
        &ferro_ta_core::utils::rising(&c, 3),
    );
    assert_same(
        &indicators::falling(c.clone(), 3),
        &ferro_ta_core::utils::falling(&c, 3),
    );
}
