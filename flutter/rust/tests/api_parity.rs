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
    let (u, m, l) = indicators::bbands(c.clone(), 5, 2.0, 2.0);
    let (cu, cm, cl) = ferro_ta_core::overlap::bbands(&c, 5, 2.0, 2.0);
    assert_same(&u, &cu);
    assert_same(&m, &cm);
    assert_same(&l, &cl);
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
