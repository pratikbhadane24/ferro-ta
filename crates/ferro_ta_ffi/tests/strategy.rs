//! Option strategy analytics over the C ABI.

use ferro_ta_ffi::*;
use std::ptr;

fn option(qty: f64, right: i32, strike: f64, premium: f64) -> FtLeg {
    FtLeg {
        kind: FT_LEG_OPTION,
        right,
        quantity: qty,
        multiplier: 1.0,
        strike,
        premium,
        expiry_years: 0.25,
        volatility: 0.2,
        entry_price: 0.0,
    }
}

fn iron_condor() -> [FtLeg; 4] {
    [
        option(1.0, 1, 80.0, 1.0),
        option(-1.0, 1, 90.0, 3.0),
        option(-1.0, 0, 110.0, 3.0),
        option(1.0, 0, 120.0, 1.0),
    ]
}

#[test]
fn iron_condor_breakevens_extremes_and_payoff() {
    let legs = iron_condor();
    let mut out = [0.0; 10];
    let mut n = 0usize;
    assert_eq!(
        unsafe { ft_strategy_breakevens(legs.as_ptr(), 4, out.as_mut_ptr(), out.len(), &mut n) },
        FT_OK
    );
    assert_eq!(&out[..n], &[86.0, 114.0]);

    let (mut v, mut s, mut u) = (0.0, 0.0, -1);
    assert_eq!(
        unsafe { ft_strategy_max_profit(legs.as_ptr(), 4, &mut v, &mut s, &mut u) },
        FT_OK
    );
    assert_eq!((v, u), (4.0, 0));
    assert_eq!(
        unsafe { ft_strategy_max_loss(legs.as_ptr(), 4, &mut v, &mut s, &mut u) },
        FT_OK
    );
    assert_eq!((v, u), (-6.0, 0));

    let spots = [70.0, 100.0, 130.0];
    let mut pnl = [0.0; 3];
    assert_eq!(
        unsafe { ft_strategy_payoff(legs.as_ptr(), 4, spots.as_ptr(), 3, pnl.as_mut_ptr()) },
        FT_OK
    );
    assert_eq!(pnl, [-6.0, 4.0, -6.0]);
}

#[test]
fn unlimited_loss_and_small_buffer() {
    let naked = [option(-1.0, 0, 100.0, 5.0)];
    let (mut v, mut s, mut u) = (0.0, 0.0, 0);
    assert_eq!(
        unsafe { ft_strategy_max_loss(naked.as_ptr(), 1, &mut v, &mut s, &mut u) },
        FT_OK
    );
    assert_eq!(u, 1);
    assert!(v.is_nan() && s.is_nan());

    let legs = iron_condor();
    let mut n = 0usize;
    assert_eq!(
        unsafe { ft_strategy_breakevens(legs.as_ptr(), 4, ptr::null_mut(), 0, &mut n) },
        FT_ERR_INVALID_PARAM
    );
    assert_eq!(n, 2, "count reported even when the buffer is too small");
}

#[test]
fn pop_greeks_and_scenario_grid() {
    let legs = iron_condor();
    let mut p = 0.0;
    assert_eq!(
        unsafe {
            ft_strategy_probability_of_profit(legs.as_ptr(), 4, 100.0, 0.2, 0.25, 0.0, 0.0, &mut p)
        },
        FT_OK
    );
    assert!(p > 0.5 && p < 1.0, "{p}");
    let mut g = [0.0; 5];
    let [d, ga, ve, th, rh] = &mut g;
    assert_eq!(
        unsafe { ft_strategy_greeks(legs.as_ptr(), 4, 100.0, 0.0, 0.0, d, ga, ve, th, rh) },
        FT_OK
    );
    assert!(g[2] < 0.0, "short premium is short vega");
    let (spots, shifts) = ([95.0, 105.0], [0.0, 0.05, 0.1]);
    let mut grid = [0.0; 6];
    assert_eq!(
        unsafe {
            ft_strategy_scenario_grid(
                legs.as_ptr(),
                4,
                spots.as_ptr(),
                2,
                shifts.as_ptr(),
                3,
                0.0,
                0.0,
                0.0,
                grid.as_mut_ptr(),
            )
        },
        FT_OK
    );
    assert!(grid[4] < grid[0], "more vol, less value for the condor");
}

#[test]
fn rejects_bad_legs_mixed_expiries_and_nulls() {
    let mut bad = iron_condor();
    bad[0].right = 7;
    let mut p = 0.0;
    let call = |legs: &[FtLeg], p: &mut f64| unsafe {
        ft_strategy_probability_of_profit(legs.as_ptr(), legs.len(), 100.0, 0.2, 0.25, 0.0, 0.0, p)
    };
    assert_eq!(call(&bad, &mut p), FT_ERR_INVALID_PARAM);
    let mut zero_qty = iron_condor();
    zero_qty[1].quantity = 0.0;
    assert_eq!(call(&zero_qty, &mut p), FT_ERR_INVALID_PARAM);
    let mut calendar = iron_condor();
    calendar[3].expiry_years = 0.5;
    assert_eq!(
        call(&calendar, &mut p),
        FT_ERR_INVALID_PARAM,
        "mixed expiries"
    );
    assert_eq!(call(&[], &mut p), FT_ERR_INVALID_PARAM, "empty strategy");
    assert_eq!(
        unsafe {
            ft_strategy_probability_of_profit(ptr::null(), 2, 100.0, 0.2, 0.25, 0.0, 0.0, &mut p)
        },
        FT_ERR_NULL_PTR
    );
    let legs = iron_condor();
    assert_eq!(
        unsafe {
            ft_strategy_probability_of_profit(
                legs.as_ptr(),
                4,
                100.0,
                0.2,
                0.25,
                0.0,
                0.0,
                ptr::null_mut(),
            )
        },
        FT_ERR_NULL_PTR
    );
}
