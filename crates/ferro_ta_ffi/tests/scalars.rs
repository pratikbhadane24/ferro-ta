//! Scalar exports: struct flattening, enum validation and Result mapping.

use ferro_ta_core::backtest;
use ferro_ta_core::options::{self, OptionKind};
use ferro_ta_ffi::*;
use std::ptr;

const CALL: i32 = 0;
const PUT: i32 = 1;

#[test]
fn black_scholes_price_matches_core_for_both_kinds() {
    for (raw, kind) in [(CALL, OptionKind::Call), (PUT, OptionKind::Put)] {
        let mut v = 0.0;
        let status =
            unsafe { ft_black_scholes_price(100.0, 105.0, 0.05, 0.01, 0.5, 0.2, raw, &mut v) };
        assert_eq!(status, FT_OK);
        let want = options::pricing::black_scholes_price(100.0, 105.0, 0.05, 0.01, 0.5, 0.2, kind);
        assert_eq!(v.to_bits(), want.to_bits());
    }
}

#[test]
fn greeks_are_flattened_in_declared_order() {
    let (mut d, mut g, mut v, mut t, mut r) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let status = unsafe {
        ft_black_scholes_greeks(
            100.0, 100.0, 0.03, 0.0, 1.0, 0.25, CALL, &mut d, &mut g, &mut v, &mut t, &mut r,
        )
    };
    assert_eq!(status, FT_OK);
    let want =
        options::greeks::black_scholes_greeks(100.0, 100.0, 0.03, 0.0, 1.0, 0.25, OptionKind::Call);
    assert_eq!(
        [d, g, v, t, r],
        [want.delta, want.gamma, want.vega, want.theta, want.rho]
    );
}

#[test]
fn unknown_enum_value_is_invalid_param() {
    let mut v = 7.0;
    for bad in [-1, 2, i32::MAX] {
        let status =
            unsafe { ft_black_scholes_price(100.0, 100.0, 0.0, 0.0, 1.0, 0.2, bad, &mut v) };
        assert_eq!(status, FT_ERR_INVALID_PARAM, "kind={bad}");
    }
    assert_eq!(v, 7.0, "output untouched on rejection");
    let status = unsafe {
        ft_implied_volatility(
            10.0, 2, 100.0, 100.0, 0.0, 0.0, 1.0, CALL, 0.2, 1e-8, 100, &mut v,
        )
    };
    assert_eq!(
        status, FT_ERR_INVALID_PARAM,
        "pricing model 2 does not exist"
    );
}

#[test]
fn implied_volatility_round_trips_a_price() {
    let mut price = 0.0;
    unsafe { ft_black_scholes_price(100.0, 110.0, 0.02, 0.02, 0.75, 0.3, PUT, &mut price) };
    let mut iv = 0.0;
    let status = unsafe {
        ft_implied_volatility(
            price, 0, 100.0, 110.0, 0.02, 0.02, 0.75, PUT, 0.2, 1e-10, 200, &mut iv,
        )
    };
    assert_eq!(status, FT_OK);
    assert!((iv - 0.3).abs() < 1e-6, "iv = {iv}");
}

#[test]
fn core_result_error_maps_to_invalid_param() {
    let mut v = 0.0;
    let (rate, win, loss) = (1.5, 2.0, 1.0); // win_rate outside [0, 1]
    assert!(backtest::kelly_fraction(rate, win, loss).is_err());
    assert_eq!(
        unsafe { ft_kelly_fraction(rate, win, loss, &mut v) },
        FT_ERR_INVALID_PARAM
    );
    assert_eq!(unsafe { ft_kelly_fraction(0.6, 2.0, 1.0, &mut v) }, FT_OK);
    assert_eq!(v, backtest::kelly_fraction(0.6, 2.0, 1.0).unwrap());
}

#[test]
fn null_scalar_output_is_rejected() {
    let mut d = 0.0;
    let status = unsafe {
        ft_digital_greeks(
            100.0,
            100.0,
            0.0,
            0.0,
            1.0,
            0.2,
            CALL,
            0,
            &mut d,
            ptr::null_mut(),
            &mut d,
        )
    };
    assert_eq!(status, FT_ERR_NULL_PTR);
}

#[test]
fn non_finite_scalar_param_is_rejected() {
    let mut v = 0.0;
    assert_eq!(
        unsafe { ft_basis(f64::NAN, 100.0, &mut v) },
        FT_ERR_INVALID_PARAM
    );
}

#[test]
fn every_scalar_spec_is_unique() {
    let mut seen = std::collections::HashSet::new();
    for spec in all_scalar_specs() {
        assert!(seen.insert(spec.name), "duplicate {}", spec.name);
        assert!(
            all_specs().all(|f| f.name != spec.name),
            "{} collides with an array export",
            spec.name
        );
    }
    assert_eq!(seen.len(), 24);
}
