//! ABI-level parity tests: every case calls the `extern "C"` symbol exactly as a
//! foreign caller would (raw pointers + caller-allocated outputs) and asserts the
//! result is bit-for-bit the `ferro_ta_core` output.

use ferro_ta_core::{extended, momentum, overlap, pattern, volatility};
use ferro_ta_ffi::*;
use std::ffi::CStr;
use std::ptr;

fn series(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 100.0 + (i as f64 * 0.7).sin() * 5.0 + i as f64 * 0.1)
        .collect()
}

fn ohlc(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let close = series(n);
    let open: Vec<f64> = close.iter().map(|c| c - 0.4).collect();
    let high: Vec<f64> = close.iter().map(|c| c + 1.3).collect();
    let low: Vec<f64> = close.iter().map(|c| c - 1.1).collect();
    (open, high, low, close)
}

/// NaN-aware bitwise equality: the ABI must not perturb a single ulp.
fn assert_same(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.to_bits() == e.to_bits() || (a.is_nan() && e.is_nan()),
            "index {i}: {a} != {e}"
        );
    }
}

#[test]
fn sma_matches_core() {
    let close = series(64);
    let mut out = vec![0.0; close.len()];
    let status = unsafe { ft_sma(close.as_ptr(), close.len(), 10, out.as_mut_ptr()) };
    assert_eq!(status, FT_OK);
    assert_same(&out, &overlap::sma(&close, 10));
}

#[test]
fn bbands_three_outputs_match_core() {
    let close = series(64);
    let n = close.len();
    let (mut up, mut mid, mut lo) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let status = unsafe {
        ft_bbands(
            close.as_ptr(),
            n,
            20,
            2.0,
            2.0,
            0,
            up.as_mut_ptr(),
            mid.as_mut_ptr(),
            lo.as_mut_ptr(),
        )
    };
    assert_eq!(status, FT_OK);
    let (eu, em, el) = overlap::bbands(&close, 20, 2.0, 2.0, 0);
    assert_same(&up, &eu);
    assert_same(&mid, &em);
    assert_same(&lo, &el);
}

#[test]
fn atr_hlc_inputs_match_core() {
    let (_, high, low, close) = ohlc(64);
    let mut out = vec![0.0; close.len()];
    let status = unsafe {
        ft_atr(
            high.as_ptr(),
            low.as_ptr(),
            close.as_ptr(),
            close.len(),
            14,
            out.as_mut_ptr(),
        )
    };
    assert_eq!(status, FT_OK);
    assert_same(&out, &volatility::atr(&high, &low, &close, 14));
}

#[test]
fn cdl_pattern_returns_i32() {
    let (open, high, low, close) = ohlc(64);
    let mut out = vec![0i32; close.len()];
    let status = unsafe {
        ft_cdldoji(
            open.as_ptr(),
            high.as_ptr(),
            low.as_ptr(),
            close.as_ptr(),
            close.len(),
            out.as_mut_ptr(),
        )
    };
    assert_eq!(status, FT_OK);
    assert_eq!(out, pattern::cdldoji(&open, &high, &low, &close));
}

#[test]
fn supertrend_mixed_output_types() {
    let (_, high, low, close) = ohlc(64);
    let n = close.len();
    let (mut line, mut dir) = (vec![0.0; n], vec![0i8; n]);
    let status = unsafe {
        ft_supertrend(
            high.as_ptr(),
            low.as_ptr(),
            close.as_ptr(),
            n,
            7,
            3.0,
            line.as_mut_ptr(),
            dir.as_mut_ptr(),
        )
    };
    assert_eq!(status, FT_OK);
    let (el, ed) = extended::supertrend(&high, &low, &close, 7, 3.0);
    assert_same(&line, &el);
    assert_eq!(dir, ed);
}

#[test]
fn period_below_minimum_is_invalid_param() {
    let close = series(16);
    let mut out = vec![0.0; close.len()];
    assert_eq!(
        unsafe { ft_sma(close.as_ptr(), close.len(), 0, out.as_mut_ptr()) },
        FT_ERR_INVALID_PARAM
    );
    assert_eq!(
        unsafe { ft_sma(close.as_ptr(), close.len(), -5, out.as_mut_ptr()) },
        FT_ERR_INVALID_PARAM
    );
}

#[test]
fn rejection_leaves_output_untouched() {
    let close = series(16);
    let mut out = vec![7.0; close.len()];
    unsafe { ft_sma(close.as_ptr(), close.len(), 0, out.as_mut_ptr()) };
    assert!(out.iter().all(|v| *v == 7.0));
}

#[test]
fn matype_out_of_range_is_invalid_param() {
    let close = series(32);
    let mut out = vec![0.0; close.len()];
    let status = unsafe { ft_ma(close.as_ptr(), close.len(), 5, 9, out.as_mut_ptr()) };
    assert_eq!(status, FT_ERR_INVALID_PARAM);
}

#[test]
fn null_pointer_with_data_is_rejected() {
    let mut out = vec![0.0; 8];
    let status = unsafe { ft_sma(ptr::null(), 8, 3, out.as_mut_ptr()) };
    assert_eq!(status, FT_ERR_NULL_PTR);
    let close = series(8);
    let status = unsafe { ft_sma(close.as_ptr(), 8, 3, ptr::null_mut()) };
    assert_eq!(status, FT_ERR_NULL_PTR);
}

#[test]
fn empty_input_accepts_null_pointers() {
    let status = unsafe { ft_sma(ptr::null(), 0, 3, ptr::null_mut()) };
    assert_eq!(status, FT_OK);
}

#[test]
fn empty_input_still_validates_params() {
    let status = unsafe { ft_sma(ptr::null(), 0, 0, ptr::null_mut()) };
    assert_eq!(status, FT_ERR_INVALID_PARAM);
}

#[test]
fn in_place_output_is_allowed() {
    let close = series(32);
    let expected = momentum::rsi(&close, 14);
    let mut buf = close.clone();
    let status = unsafe { ft_rsi(buf.as_ptr(), buf.len(), 14, buf.as_mut_ptr()) };
    assert_eq!(status, FT_OK);
    assert_same(&buf, &expected);
}

#[test]
fn shift_params_accept_zero() {
    let (_, high, low, _) = ohlc(64);
    let n = high.len();
    let (mut a, mut b, mut c) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let status = unsafe {
        ft_alligator(
            high.as_ptr(),
            low.as_ptr(),
            n,
            13,
            0,
            8,
            0,
            5,
            0,
            a.as_mut_ptr(),
            b.as_mut_ptr(),
            c.as_mut_ptr(),
        )
    };
    assert_eq!(status, FT_OK);
}

#[test]
fn status_messages_are_static_c_strings() {
    for code in [
        FT_OK,
        FT_ERR_NULL_PTR,
        FT_ERR_INVALID_PARAM,
        FT_ERR_LENGTH_MISMATCH,
        FT_ERR_PANIC,
        -999,
    ] {
        let msg = unsafe { CStr::from_ptr(ft_status_message(code)) };
        assert!(!msg.to_str().unwrap().is_empty());
    }
}

#[test]
fn version_matches_crate() {
    let v = unsafe { CStr::from_ptr(ft_version()) }.to_str().unwrap();
    assert_eq!(v, env!("CARGO_PKG_VERSION"));
}

#[test]
fn every_spec_name_is_unique_and_prefixed() {
    let mut seen = std::collections::HashSet::new();
    for spec in all_specs() {
        assert!(spec.name.starts_with("ft_"), "{}", spec.name);
        assert!(seen.insert(spec.name), "duplicate export {}", spec.name);
        assert!(
            !spec.inputs.is_empty() && !spec.outputs.is_empty(),
            "{}",
            spec.name
        );
    }
    assert!(
        seen.len() > 200,
        "expected full indicator coverage, got {}",
        seen.len()
    );
}

#[test]
fn non_finite_float_params_are_invalid() {
    let close = series(32);
    let n = close.len();
    let (mut up, mut mid, mut lo) = (vec![7.0; n], vec![7.0; n], vec![7.0; n]);
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let status = unsafe {
            ft_bbands(
                close.as_ptr(),
                n,
                20,
                bad,
                2.0,
                0,
                up.as_mut_ptr(),
                mid.as_mut_ptr(),
                lo.as_mut_ptr(),
            )
        };
        assert_eq!(status, FT_ERR_INVALID_PARAM, "nbdevup={bad}");
    }
    assert!(up.iter().chain(&mid).chain(&lo).all(|v| *v == 7.0));
}

#[test]
fn finite_edge_float_params_are_accepted() {
    let close = series(32);
    let n = close.len();
    let (mut up, mut mid, mut lo) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for ok in [0.0, -1.5, f64::MAX] {
        let status = unsafe {
            ft_bbands(
                close.as_ptr(),
                n,
                20,
                ok,
                ok,
                0,
                up.as_mut_ptr(),
                mid.as_mut_ptr(),
                lo.as_mut_ptr(),
            )
        };
        assert_eq!(status, FT_OK, "nbdev={ok}");
    }
}

#[test]
fn cross_param_rule_is_enforced() {
    let series = series(64);
    let mut out = vec![9i8; series.len()];
    // rolling_variance_break requires long_window > short_window.
    let status = unsafe {
        ft_rolling_variance_break(series.as_ptr(), series.len(), 10, 10, 2.0, out.as_mut_ptr())
    };
    assert_eq!(status, FT_ERR_INVALID_PARAM);
    assert!(out.iter().all(|v| *v == 9));
    let status = unsafe {
        ft_rolling_variance_break(series.as_ptr(), series.len(), 5, 20, 2.0, out.as_mut_ptr())
    };
    assert_eq!(status, FT_OK);
}

#[test]
fn cross_param_rule_is_recorded_in_spec() {
    let spec = all_specs()
        .find(|s| s.name == "ft_rolling_variance_break")
        .unwrap();
    assert_eq!(spec.requires, "long_window > short_window");
    let sma = all_specs().find(|s| s.name == "ft_sma").unwrap();
    assert_eq!(sma.requires, "");
}

#[test]
fn every_fast_slow_export_requires_fast_below_slow() {
    for spec in all_specs() {
        let names: Vec<_> = spec.params.iter().map(|p| p.name).collect();
        if names.contains(&"fastperiod") && names.contains(&"slowperiod") {
            assert_eq!(spec.requires, "fastperiod < slowperiod", "{}", spec.name);
        }
    }
}

#[test]
fn fast_not_below_slow_is_invalid_param() {
    let close = series(64);
    let n = close.len();
    let (mut a, mut b, mut c) = (vec![7.0; n], vec![7.0; n], vec![7.0; n]);
    for (fast, slow) in [(26, 12), (12, 12)] {
        let status = unsafe {
            ft_macd(
                close.as_ptr(),
                n,
                fast,
                slow,
                9,
                a.as_mut_ptr(),
                b.as_mut_ptr(),
                c.as_mut_ptr(),
            )
        };
        assert_eq!(status, FT_ERR_INVALID_PARAM, "fast={fast} slow={slow}");
    }
    assert!(a.iter().all(|v| *v == 7.0));
}

#[test]
fn mavp_max_below_min_is_invalid_param() {
    let close = series(32);
    let periods = vec![5.0; 32];
    let mut out = vec![0.0; 32];
    let status = unsafe {
        ft_mavp(
            close.as_ptr(),
            periods.as_ptr(),
            32,
            10,
            5,
            0,
            out.as_mut_ptr(),
        )
    };
    assert_eq!(status, FT_ERR_INVALID_PARAM);
    let status = unsafe {
        ft_mavp(
            close.as_ptr(),
            periods.as_ptr(),
            32,
            5,
            5,
            0,
            out.as_mut_ptr(),
        )
    };
    assert_eq!(
        status, FT_OK,
        "maxperiod == minperiod is allowed, as in Python"
    );
}

#[test]
fn alma_non_positive_sigma_is_invalid_param() {
    let close = series(32);
    let mut out = vec![0.0; 32];
    for sigma in [0.0, -1.0] {
        let status = unsafe { ft_alma(close.as_ptr(), 32, 9, 0.85, sigma, out.as_mut_ptr()) };
        assert_eq!(status, FT_ERR_INVALID_PARAM, "sigma={sigma}");
    }
}

#[test]
fn oversized_period_is_rejected_instead_of_aborting() {
    // Without a cap, core allocations sized by the period abort the process.
    let close = series(8);
    let mut out = vec![0.0; 8];
    for period in [1i64 << 40, i64::MAX] {
        let status = unsafe {
            ft_rvi(
                close.as_ptr(),
                close.as_ptr(),
                close.as_ptr(),
                close.as_ptr(),
                8,
                period,
                out.as_mut_ptr(),
                out.as_mut_ptr(),
            )
        };
        assert_eq!(status, FT_ERR_INVALID_PARAM, "period={period}");
    }
    let at_cap = unsafe {
        ft_sma(
            close.as_ptr(),
            8,
            ferro_ta_ffi::marshal::MAX_PERIOD,
            out.as_mut_ptr(),
        )
    };
    assert_eq!(at_cap, FT_OK, "the cap itself is a valid period");
}
