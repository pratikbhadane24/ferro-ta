//! Hand-written C ABI for multi-leg option strategies.
//!
//! Legs cross the boundary as an array of `FtLeg` structs, which the
//! generated-export spec cannot describe, so these exports (and their header
//! block in `scripts/ffi_codegen/c_header.py`) are written by hand. Every
//! call is stateless: pass the leg array each time.

use crate::marshal::kernel_in;
use crate::status::{guard, Status};
use ferro_ta_core::options::strategy::{Extreme, Leg, Strategy};
use ferro_ta_core::options::OptionKind;

/// Leg kind codes (`FtLeg::kind`).
pub const FT_LEG_OPTION: i32 = 0;
pub const FT_LEG_FUTURE: i32 = 1;
pub const FT_LEG_STOCK: i32 = 2;

/// One strategy leg. `quantity` is signed (+ long, − short). Option fields
/// are ignored for futures/stock and `entry_price` for options; `right` uses
/// the `FT_OPTION_*` codes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FtLeg {
    pub kind: i32,
    pub right: i32,
    pub quantity: f64,
    pub multiplier: f64,
    pub strike: f64,
    pub premium: f64,
    pub expiry_years: f64,
    pub volatility: f64,
    pub entry_price: f64,
}

impl TryFrom<&FtLeg> for Leg {
    type Error = Status;

    fn try_from(l: &FtLeg) -> Result<Self, Status> {
        Ok(match l.kind {
            FT_LEG_OPTION => {
                let right = match l.right {
                    0 => OptionKind::Call,
                    1 => OptionKind::Put,
                    _ => return Err(Status::InvalidParam),
                };
                Leg::option(
                    l.quantity,
                    right,
                    l.strike,
                    l.premium,
                    l.expiry_years,
                    l.volatility,
                    l.multiplier,
                )
            }
            FT_LEG_FUTURE => Leg::future(l.quantity, l.entry_price, l.multiplier),
            FT_LEG_STOCK => Leg::stock(l.quantity, l.entry_price, l.multiplier),
            _ => return Err(Status::InvalidParam),
        })
    }
}

/// Validate and build a strategy from a caller leg array.
///
/// # Safety
/// `legs` valid for `n_legs` reads.
unsafe fn strategy(legs: *const FtLeg, n_legs: usize) -> Result<Strategy, Status> {
    // SAFETY: caller contract.
    let raw = unsafe { kernel_in(legs, n_legs) }?;
    let legs = raw
        .iter()
        .map(Leg::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    Strategy::new(legs).map_err(|_| Status::InvalidParam)
}

fn non_null<T>(ptrs: &[*mut T]) -> Result<(), Status> {
    if ptrs.iter().any(|p| p.is_null()) {
        return Err(Status::NullPtr);
    }
    Ok(())
}

/// Expiry P&L at each of `n_spots` spots into `out` (`n_spots` values).
/// Mixed option expiries return `FT_ERR_INVALID_PARAM`.
///
/// # Safety
/// `legs` valid for `n_legs` reads; `spots` and `out` for `n_spots` elements.
#[no_mangle]
pub unsafe extern "C" fn ft_strategy_payoff(
    legs: *const FtLeg,
    n_legs: usize,
    spots: *const f64,
    n_spots: usize,
    out: *mut f64,
) -> i32 {
    guard(|| {
        if n_spots > 0 {
            non_null(&[out])?;
        }
        // SAFETY: caller contract.
        let s = unsafe { strategy(legs, n_legs) }?;
        // SAFETY: caller contract.
        let spots = unsafe { kernel_in(spots, n_spots) }?;
        let values = s.payoff(spots).map_err(|_| Status::InvalidParam)?;
        // SAFETY: `out` valid for `n_spots` writes; lengths match.
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), out, values.len()) };
        Ok(())
    })
}

/// Breakeven spots (ascending) into `out`. `*out_len` always receives the
/// count; at most `2 * n_legs + 2`, so a buffer that size always suffices.
/// When `cap` is smaller than the count nothing is copied and
/// `FT_ERR_INVALID_PARAM` is returned.
///
/// # Safety
/// `legs` valid for `n_legs` reads; `out` valid for `cap` writes when
/// `cap > 0`; `out_len` valid for one write.
#[no_mangle]
pub unsafe extern "C" fn ft_strategy_breakevens(
    legs: *const FtLeg,
    n_legs: usize,
    out: *mut f64,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        non_null(&[out_len])?;
        // SAFETY: caller contract.
        let s = unsafe { strategy(legs, n_legs) }?;
        let roots = s.breakevens().map_err(|_| Status::InvalidParam)?;
        // SAFETY: checked non-null.
        unsafe { out_len.write(roots.len()) };
        if roots.len() > cap {
            return Err(Status::InvalidParam);
        }
        if !roots.is_empty() {
            non_null(&[out])?;
            // SAFETY: `out` holds at least `roots.len()` values.
            unsafe { std::ptr::copy_nonoverlapping(roots.as_ptr(), out, roots.len()) };
        }
        Ok(())
    })
}

/// Write an extreme: `*unlimited` = 1 with value/spot NaN, else 0.
///
/// # Safety
/// All pointers non-null and valid for one write.
unsafe fn write_extreme(e: Extreme, value: *mut f64, spot: *mut f64, unlimited: *mut i32) {
    let (v, s, u) = match e {
        Extreme::Finite { value, spot } => (value, spot, 0),
        Extreme::Unlimited => (f64::NAN, f64::NAN, 1),
    };
    // SAFETY: caller contract.
    unsafe {
        value.write(v);
        spot.write(s);
        unlimited.write(u);
    }
}

/// Best expiry P&L over spot ≥ 0 and the lowest spot attaining it;
/// `*out_unlimited` = 1 (value/spot NaN) when profit is unbounded.
///
/// # Safety
/// `legs` valid for `n_legs` reads; outputs valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ft_strategy_max_profit(
    legs: *const FtLeg,
    n_legs: usize,
    out_value: *mut f64,
    out_spot: *mut f64,
    out_unlimited: *mut i32,
) -> i32 {
    guard(|| {
        non_null(&[out_value, out_spot])?;
        non_null(&[out_unlimited])?;
        // SAFETY: caller contract.
        let e = unsafe { strategy(legs, n_legs) }?
            .max_profit()
            .map_err(|_| Status::InvalidParam)?;
        // SAFETY: checked non-null.
        unsafe { write_extreme(e, out_value, out_spot, out_unlimited) };
        Ok(())
    })
}

/// Worst expiry P&L over spot ≥ 0 (a P&L value, usually negative);
/// `*out_unlimited` = 1 (value/spot NaN) when loss is unbounded.
///
/// # Safety
/// As `ft_strategy_max_profit`.
#[no_mangle]
pub unsafe extern "C" fn ft_strategy_max_loss(
    legs: *const FtLeg,
    n_legs: usize,
    out_value: *mut f64,
    out_spot: *mut f64,
    out_unlimited: *mut i32,
) -> i32 {
    guard(|| {
        non_null(&[out_value, out_spot])?;
        non_null(&[out_unlimited])?;
        // SAFETY: caller contract.
        let e = unsafe { strategy(legs, n_legs) }?
            .max_loss()
            .map_err(|_| Status::InvalidParam)?;
        // SAFETY: checked non-null.
        unsafe { write_extreme(e, out_value, out_spot, out_unlimited) };
        Ok(())
    })
}

/// Risk-neutral probability that expiry P&L > 0, lognormal underlying.
///
/// # Safety
/// `legs` valid for `n_legs` reads; `out` valid for one write.
#[no_mangle]
pub unsafe extern "C" fn ft_strategy_probability_of_profit(
    legs: *const FtLeg,
    n_legs: usize,
    spot: f64,
    volatility: f64,
    time_years: f64,
    rate: f64,
    carry: f64,
    out: *mut f64,
) -> i32 {
    guard(|| {
        non_null(&[out])?;
        // SAFETY: caller contract.
        let p = unsafe { strategy(legs, n_legs) }?
            .probability_of_profit(spot, volatility, time_years, rate, carry)
            .map_err(|_| Status::InvalidParam)?;
        // SAFETY: checked non-null.
        unsafe { out.write(p) };
        Ok(())
    })
}

/// Black-Scholes greeks summed over legs.
///
/// # Safety
/// `legs` valid for `n_legs` reads; each output valid for one write.
#[no_mangle]
pub unsafe extern "C" fn ft_strategy_greeks(
    legs: *const FtLeg,
    n_legs: usize,
    spot: f64,
    rate: f64,
    carry: f64,
    out_delta: *mut f64,
    out_gamma: *mut f64,
    out_vega: *mut f64,
    out_theta: *mut f64,
    out_rho: *mut f64,
) -> i32 {
    guard(|| {
        non_null(&[out_delta, out_gamma, out_vega, out_theta, out_rho])?;
        // SAFETY: caller contract.
        let g = unsafe { strategy(legs, n_legs) }?.greeks(spot, rate, carry);
        // SAFETY: checked non-null.
        unsafe {
            out_delta.write(g.delta);
            out_gamma.write(g.gamma);
            out_vega.write(g.vega);
            out_theta.write(g.theta);
            out_rho.write(g.rho);
        }
        Ok(())
    })
}

/// Mark-to-model P&L, row-major `[vol_shift][spot]` into `out`
/// (`n_shifts * n_spots` values), `days_forward` days ahead.
///
/// # Safety
/// `legs` valid for `n_legs` reads; `spots`/`vol_shifts` for their counts;
/// `out` valid for `n_shifts * n_spots` writes.
#[no_mangle]
pub unsafe extern "C" fn ft_strategy_scenario_grid(
    legs: *const FtLeg,
    n_legs: usize,
    spots: *const f64,
    n_spots: usize,
    vol_shifts: *const f64,
    n_shifts: usize,
    days_forward: f64,
    rate: f64,
    carry: f64,
    out: *mut f64,
) -> i32 {
    guard(|| {
        let total = n_spots.checked_mul(n_shifts).ok_or(Status::InvalidParam)?;
        if total > 0 {
            non_null(&[out])?;
        }
        // SAFETY: caller contract.
        let s = unsafe { strategy(legs, n_legs) }?;
        // SAFETY: caller contract.
        let (spots, shifts) =
            unsafe { (kernel_in(spots, n_spots)?, kernel_in(vol_shifts, n_shifts)?) };
        let grid = s.scenario_grid(spots, shifts, days_forward, rate, carry);
        if grid.len() != total {
            return Err(Status::Internal);
        }
        // SAFETY: `out` valid for `total` writes.
        unsafe { std::ptr::copy_nonoverlapping(grid.as_ptr(), out, total) };
        Ok(())
    })
}
