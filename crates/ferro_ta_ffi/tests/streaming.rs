//! Streaming handle lifecycle over the C ABI: new / update / reset / free.

use ferro_ta_core::streaming::{StreamingBBands, StreamingSupertrend, StreamingVWAP};
use ferro_ta_core::{momentum, overlap};
use ferro_ta_ffi::*;
use std::ptr;

fn series(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 100.0 + (i as f64 * 0.7).sin() * 5.0 + i as f64 * 0.1)
        .collect()
}

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// Streaming keeps running sums while batch kernels re-reduce windows, so the
/// two legitimately differ in the last ulp; compare them with a tolerance.
fn close_enough(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-9 * b.abs().max(1.0)
}

#[test]
fn sma_stream_matches_batch() {
    let close = series(50);
    let batch = overlap::sma(&close, 10);
    let mut h = ptr::null_mut();
    assert_eq!(unsafe { ft_stream_sma_new(10, &mut h) }, FT_OK);
    assert!(!h.is_null());
    for (i, x) in close.iter().enumerate() {
        let mut out = 0.0;
        assert_eq!(unsafe { ft_stream_sma_update(h, *x, &mut out) }, FT_OK);
        assert!(
            close_enough(out, batch[i]),
            "bar {i}: {out} vs {}",
            batch[i]
        );
    }
    unsafe { ft_stream_sma_free(h) };
}

#[test]
fn rsi_stream_matches_batch_tail() {
    let close = series(80);
    let batch = momentum::rsi(&close, 14);
    let mut h = ptr::null_mut();
    assert_eq!(unsafe { ft_stream_rsi_new(14, &mut h) }, FT_OK);
    let mut last = f64::NAN;
    for x in &close {
        assert_eq!(unsafe { ft_stream_rsi_update(h, *x, &mut last) }, FT_OK);
    }
    assert!((last - batch[close.len() - 1]).abs() < 1e-9);
    unsafe { ft_stream_rsi_free(h) };
}

#[test]
fn bbands_stream_three_outputs_match_core_stream() {
    let close = series(40);
    let mut core = StreamingBBands::new(20, 2.0, 2.0).unwrap();
    let mut h = ptr::null_mut();
    assert_eq!(unsafe { ft_stream_bbands_new(20, 2.0, 2.0, &mut h) }, FT_OK);
    for x in &close {
        let (mut u, mut m, mut l) = (0.0, 0.0, 0.0);
        assert_eq!(
            unsafe { ft_stream_bbands_update(h, *x, &mut u, &mut m, &mut l) },
            FT_OK
        );
        let (eu, em, el) = core.update(*x);
        assert!(same(u, eu) && same(m, em) && same(l, el));
    }
    unsafe { ft_stream_bbands_free(h) };
}

#[test]
fn supertrend_stream_mixed_outputs() {
    let close = series(40);
    let mut core = StreamingSupertrend::new(7, 3.0).unwrap();
    let mut h = ptr::null_mut();
    assert_eq!(unsafe { ft_stream_supertrend_new(7, 3.0, &mut h) }, FT_OK);
    for c in &close {
        let (mut line, mut dir) = (0.0, 0i8);
        let status =
            unsafe { ft_stream_supertrend_update(h, c + 1.0, c - 1.0, *c, &mut line, &mut dir) };
        assert_eq!(status, FT_OK);
        let (el, ed) = core.update(c + 1.0, c - 1.0, *c);
        assert!(same(line, el));
        assert_eq!(dir, ed);
    }
    unsafe { ft_stream_supertrend_free(h) };
}

#[test]
fn vwap_infallible_constructor() {
    let mut core = StreamingVWAP::new();
    let mut h = ptr::null_mut();
    assert_eq!(unsafe { ft_stream_vwap_new(&mut h) }, FT_OK);
    let mut out = 0.0;
    assert_eq!(
        unsafe { ft_stream_vwap_update(h, 11.0, 9.0, 10.0, 500.0, &mut out) },
        FT_OK
    );
    assert!(same(out, core.update(11.0, 9.0, 10.0, 500.0)));
    unsafe { ft_stream_vwap_free(h) };
}

#[test]
fn reset_restarts_warmup() {
    let mut h = ptr::null_mut();
    assert_eq!(unsafe { ft_stream_sma_new(3, &mut h) }, FT_OK);
    let mut out = 0.0;
    for x in [1.0, 2.0, 3.0] {
        unsafe { ft_stream_sma_update(h, x, &mut out) };
    }
    assert_eq!(out, 2.0);
    assert_eq!(unsafe { ft_stream_sma_reset(h) }, FT_OK);
    unsafe { ft_stream_sma_update(h, 5.0, &mut out) };
    assert!(out.is_nan(), "warm-up must restart after reset");
    unsafe { ft_stream_sma_free(h) };
}

#[test]
fn invalid_period_returns_error_and_null_handle() {
    let mut h = ptr::null_mut();
    assert_eq!(
        unsafe { ft_stream_sma_new(0, &mut h) },
        FT_ERR_INVALID_PARAM
    );
    assert!(h.is_null());
    assert_eq!(
        unsafe { ft_stream_sma_new(-3, &mut h) },
        FT_ERR_INVALID_PARAM
    );
    assert!(h.is_null());
}

#[test]
fn core_constructor_rejection_maps_to_invalid_param() {
    // StreamingMACD itself rejects fastperiod >= slowperiod.
    let mut h = ptr::null_mut();
    assert_eq!(
        unsafe { ft_stream_macd_new(26, 12, 9, &mut h) },
        FT_ERR_INVALID_PARAM
    );
    assert!(h.is_null());
}

#[test]
fn non_finite_constructor_float_is_rejected() {
    let mut h = ptr::null_mut();
    assert_eq!(
        unsafe { ft_stream_supertrend_new(7, f64::NAN, &mut h) },
        FT_ERR_INVALID_PARAM
    );
    assert!(h.is_null());
}

#[test]
fn null_handles_and_outputs_are_rejected() {
    let mut out = 0.0;
    assert_eq!(
        unsafe { ft_stream_sma_update(ptr::null_mut(), 1.0, &mut out) },
        FT_ERR_NULL_PTR
    );
    assert_eq!(
        unsafe { ft_stream_sma_reset(ptr::null_mut()) },
        FT_ERR_NULL_PTR
    );
    assert_eq!(
        unsafe { ft_stream_sma_new(3, ptr::null_mut()) },
        FT_ERR_NULL_PTR
    );
    let mut h = ptr::null_mut();
    assert_eq!(unsafe { ft_stream_sma_new(3, &mut h) }, FT_OK);
    assert_eq!(
        unsafe { ft_stream_sma_update(h, 1.0, ptr::null_mut()) },
        FT_ERR_NULL_PTR
    );
    unsafe { ft_stream_sma_free(h) };
}

#[test]
fn free_null_is_noop() {
    unsafe { ft_stream_sma_free(ptr::null_mut()) };
}

#[test]
fn every_streaming_type_has_a_spec() {
    let names: Vec<_> = all_stream_specs().map(|s| s.name).collect();
    for want in [
        "sma",
        "ema",
        "rsi",
        "atr",
        "bbands",
        "macd",
        "stoch",
        "vwap",
        "supertrend",
    ] {
        assert!(names.contains(&want), "missing stream spec {want}");
    }
}

#[test]
fn oversized_stream_period_is_rejected() {
    let mut h = ptr::null_mut();
    assert_eq!(
        unsafe { ft_stream_sma_new(1 << 40, &mut h) },
        FT_ERR_INVALID_PARAM
    );
    assert!(h.is_null());
}

#[test]
fn peek_previews_without_committing() {
    let mut h = ptr::null_mut();
    assert_eq!(unsafe { ft_stream_sma_new(3, &mut h) }, FT_OK);
    let mut out = 0.0;
    for x in [1.0, 2.0, 3.0] {
        unsafe { ft_stream_sma_update(h, x, &mut out) };
    }
    let mut preview = 0.0;
    for tick in [50.0, 9.0] {
        assert_eq!(unsafe { ft_stream_sma_peek(h, tick, &mut preview) }, FT_OK);
    }
    assert_eq!(preview, (2.0 + 3.0 + 9.0) / 3.0);
    unsafe { ft_stream_sma_update(h, 9.0, &mut out) };
    assert_eq!(out, preview, "peek must equal the committed value");
    unsafe { ft_stream_sma_free(h) };
}

#[test]
fn peek_rejects_null_pointers() {
    let mut h = ptr::null_mut();
    assert_eq!(unsafe { ft_stream_sma_new(3, &mut h) }, FT_OK);
    let mut out = 0.0;
    assert_eq!(
        unsafe { ft_stream_sma_peek(ptr::null_mut(), 1.0, &mut out) },
        FT_ERR_NULL_PTR
    );
    assert_eq!(
        unsafe { ft_stream_sma_peek(h, 1.0, ptr::null_mut()) },
        FT_ERR_NULL_PTR
    );
    unsafe { ft_stream_sma_free(h) };
}

#[test]
fn nan_input_is_skipped_not_poisoning() {
    let mut h = ptr::null_mut();
    assert_eq!(unsafe { ft_stream_sma_new(2, &mut h) }, FT_OK);
    let mut out = 0.0;
    unsafe { ft_stream_sma_update(h, 1.0, &mut out) };
    unsafe { ft_stream_sma_update(h, f64::NAN, &mut out) };
    assert!(out.is_nan(), "NaN tick reports NaN");
    unsafe { ft_stream_sma_update(h, 3.0, &mut out) };
    assert_eq!(out, 2.0, "state continues as if the NaN never arrived");
    unsafe { ft_stream_sma_free(h) };
}

#[test]
fn cross_stream_fires_once_inside_band() {
    let mut h = ptr::null_mut();
    assert_eq!(unsafe { ft_stream_cross_new(0.5, &mut h) }, FT_OK);
    let mut events = Vec::new();
    for d in [-1.0, 0.2, -0.2, 0.3, 0.6, 0.1, 0.7] {
        let mut out = 9_i8;
        assert_eq!(
            unsafe { ft_stream_cross_update(h, d, 0.0, &mut out) },
            FT_OK
        );
        events.push(out);
    }
    assert_eq!(events, vec![0, 0, 0, 0, 1, 0, 0]);
    unsafe { ft_stream_cross_free(h) };
}

#[test]
fn condition_streams_reject_bad_params() {
    let mut h = ptr::null_mut();
    assert_eq!(
        unsafe { ft_stream_cross_new(-1.0, &mut h) },
        FT_ERR_INVALID_PARAM
    );
    assert!(h.is_null());
    let mut s = ptr::null_mut();
    assert_eq!(
        unsafe { ft_stream_streak_new(0, &mut s) },
        FT_ERR_INVALID_PARAM
    );
    assert!(s.is_null());
}

#[test]
fn option_greeks_stream_solves_iv_and_keeps_seed_on_failure() {
    use ferro_ta_core::options::{
        pricing::model_price, OptionContract, OptionEvaluation, OptionKind, PricingModel,
    };
    let contract = OptionContract {
        model: PricingModel::BlackScholes,
        underlying: 102.0,
        strike: 100.0,
        rate: 0.05,
        carry: 0.0,
        time_to_expiry: 0.5,
        kind: OptionKind::Call,
    };
    let price = model_price(OptionEvaluation {
        contract,
        volatility: 0.25,
    });
    let mut h = ptr::null_mut();
    assert_eq!(
        unsafe { ft_stream_option_greeks_new(0, 0, 100.0, 0.05, 0.0, &mut h) },
        FT_OK
    );
    let mut out = [0.0_f64; 6];
    let [iv, delta, gamma, vega, theta, rho] = &mut out;
    let feed = |p: f64, s: f64, o: (&mut f64, &mut f64, &mut f64, &mut f64, &mut f64, &mut f64)| unsafe {
        ft_stream_option_greeks_update(h, p, s, 0.5, o.0, o.1, o.2, o.3, o.4, o.5)
    };
    assert_eq!(
        feed(price, 102.0, (iv, delta, gamma, vega, theta, rho)),
        FT_OK
    );
    assert!((*iv - 0.25).abs() < 1e-8);
    assert!(*delta > 0.0 && *delta < 1.0);
    assert_eq!(
        feed(1.0, 125.0, (iv, delta, gamma, vega, theta, rho)),
        FT_OK
    );
    assert!(iv.is_nan() && rho.is_nan(), "unsolvable price writes NaN");
    unsafe { ft_stream_option_greeks_free(h) };
}

#[test]
fn option_greeks_stream_rejects_bad_enum_and_strike() {
    let mut h = ptr::null_mut();
    assert_eq!(
        unsafe { ft_stream_option_greeks_new(0, 7, 100.0, 0.0, 0.0, &mut h) },
        FT_ERR_INVALID_PARAM
    );
    assert_eq!(
        unsafe { ft_stream_option_greeks_new(0, 0, -1.0, 0.0, 0.0, &mut h) },
        FT_ERR_INVALID_PARAM
    );
    assert!(h.is_null());
}

#[test]
fn save_load_round_trip_continues_bit_identically() {
    let close = series(80);
    let mut h = ptr::null_mut();
    assert_eq!(unsafe { ft_stream_rsi_new(14, &mut h) }, FT_OK);
    let mut out = 0.0;
    for x in &close[..40] {
        unsafe { ft_stream_rsi_update(h, *x, &mut out) };
    }
    let mut need = 0usize;
    assert_eq!(
        unsafe { ft_stream_rsi_save(h, ptr::null_mut(), 0, &mut need) },
        FT_ERR_LENGTH_MISMATCH
    );
    assert!(need > 0);
    let mut buf = vec![0u8; need];
    let mut len = 0usize;
    assert_eq!(
        unsafe { ft_stream_rsi_save(h, buf.as_mut_ptr().cast(), buf.len(), &mut len) },
        FT_OK
    );
    assert_eq!(len, need);
    let mut restored = ptr::null_mut();
    assert_eq!(
        unsafe { ft_stream_rsi_load(buf.as_ptr().cast(), len, &mut restored) },
        FT_OK
    );
    for x in &close[40..] {
        let (mut a, mut b) = (0.0, 0.0);
        unsafe { ft_stream_rsi_update(h, *x, &mut a) };
        unsafe { ft_stream_rsi_update(restored, *x, &mut b) };
        assert!(same(a, b));
    }
    unsafe { ft_stream_rsi_free(h) };
    unsafe { ft_stream_rsi_free(restored) };
}

#[test]
fn load_rejects_wrong_stream_and_garbage() {
    let mut h = ptr::null_mut();
    assert_eq!(unsafe { ft_stream_sma_new(3, &mut h) }, FT_OK);
    let mut buf = vec![0u8; 4096];
    let mut len = 0usize;
    assert_eq!(
        unsafe { ft_stream_sma_save(h, buf.as_mut_ptr().cast(), buf.len(), &mut len) },
        FT_OK
    );
    let mut ema = ptr::null_mut();
    assert_eq!(
        unsafe { ft_stream_ema_load(buf.as_ptr().cast(), len, &mut ema) },
        FT_ERR_INVALID_PARAM
    );
    assert!(ema.is_null());
    let junk = [0xFFu8, 0xFE];
    let mut sma = ptr::null_mut();
    assert_eq!(
        unsafe { ft_stream_sma_load(junk.as_ptr().cast(), junk.len(), &mut sma) },
        FT_ERR_INVALID_PARAM,
        "non-UTF-8"
    );
    assert_eq!(
        unsafe { ft_stream_sma_load(ptr::null(), 5, &mut sma) },
        FT_ERR_NULL_PTR
    );
    unsafe { ft_stream_sma_free(h) };
}
