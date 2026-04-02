//! Pure-Rust (no PyO3, no numpy) strategy payoff and value functions.
//!
//! NOTE: `crates/ferro_ta_core/src/options/mod.rs` must declare `pub mod payoff;`
//! for this module to be reachable from the rest of the crate and from the PyO3 bridge.

use super::pricing::black_scholes_price;
use super::OptionKind;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Instrument codes: 0=option, 1=future, 2=stock.
const INSTRUMENT_OPTION: i64 = 0;
const INSTRUMENT_FUTURE: i64 = 1;
const INSTRUMENT_STOCK: i64 = 2;

/// Side sign from encoded value: 1=long (+1.0), -1=short (-1.0).
#[inline]
fn side_sign(v: i64) -> f64 {
    if v == 1 {
        1.0
    } else if v == -1 {
        -1.0
    } else {
        f64::NAN
    }
}

/// Option kind from encoded value: 1=call, -1=put.
#[inline]
fn option_kind(v: i64) -> Option<OptionKind> {
    match v {
        1 => Some(OptionKind::Call),
        -1 => Some(OptionKind::Put),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// strategy_payoff_dense
// ---------------------------------------------------------------------------

/// Aggregate strategy payoff over a spot grid.
///
/// Parameters (all slices of length n_legs):
/// - `instruments`: 0=option, 1=future, 2=stock
/// - `sides`: 1=long, -1=short
/// - `option_types`: 1=call, -1=put (ignored for futures/stocks)
/// - `strikes`: strike for options
/// - `premiums`: premium for options
/// - `entry_prices`: entry price for futures/stocks
/// - `quantities`, `multipliers`: applied to all instruments
///
/// Returns a Vec<f64> of length spot_grid.len() with aggregate P&L per spot point.
#[allow(clippy::too_many_arguments)]
pub fn strategy_payoff_dense(
    spot_grid: &[f64],
    instruments: &[i64],
    sides: &[i64],
    option_types: &[i64],
    strikes: &[f64],
    premiums: &[f64],
    entry_prices: &[f64],
    quantities: &[f64],
    multipliers: &[f64],
) -> Vec<f64> {
    let n_legs = instruments.len();
    // Validate that all leg slices are the same length; return zeros if not.
    if sides.len() != n_legs
        || option_types.len() != n_legs
        || strikes.len() != n_legs
        || premiums.len() != n_legs
        || entry_prices.len() != n_legs
        || quantities.len() != n_legs
        || multipliers.len() != n_legs
    {
        return vec![0.0; spot_grid.len()];
    }

    let mut total = vec![0.0_f64; spot_grid.len()];

    for leg_idx in 0..n_legs {
        let inst = instruments[leg_idx];
        let sign = side_sign(sides[leg_idx]);
        if sign.is_nan() {
            // Invalid side — skip leg (treat as zero contribution).
            continue;
        }
        let leg_scale = sign * quantities[leg_idx] * multipliers[leg_idx];

        match inst {
            INSTRUMENT_OPTION => {
                let kind = match option_kind(option_types[leg_idx]) {
                    Some(k) => k,
                    None => continue, // Invalid option type — skip.
                };
                let k = strikes[leg_idx];
                let p = premiums[leg_idx];
                for (i, &s) in spot_grid.iter().enumerate() {
                    let intrinsic = match kind {
                        OptionKind::Call => (s - k).max(0.0),
                        OptionKind::Put => (k - s).max(0.0),
                    };
                    total[i] += leg_scale * (intrinsic - p);
                }
            }
            INSTRUMENT_FUTURE | INSTRUMENT_STOCK => {
                let e = entry_prices[leg_idx];
                for (i, &s) in spot_grid.iter().enumerate() {
                    total[i] += leg_scale * (s - e);
                }
            }
            _ => {
                // Unknown instrument code — skip leg (NaN would propagate; zeros are safer).
            }
        }
    }

    total
}

// ---------------------------------------------------------------------------
// strategy_value_dense / strategy_value_grid
// ---------------------------------------------------------------------------

/// Current BSM value of a strategy at a single spot (pre-expiry).
///
/// Unlike `strategy_payoff_dense`, this uses BSM pricing for option legs rather
/// than intrinsic value.
///
/// Parameters: same as `strategy_payoff_dense` plus per-leg BSM inputs:
/// - `time_to_expiries`: TTE for each option leg (ignored for futures/stocks)
/// - `volatilities`: vol for each option leg (ignored for futures/stocks)
/// - `rates`: risk-free rate for each leg
/// - `carries`: carry/dividend yield for each option leg
///
/// Returns a scalar f64 (strategy P&L at the given spot).
#[allow(clippy::too_many_arguments)]
pub fn strategy_value_dense(
    spot: f64,
    instruments: &[i64],
    sides: &[i64],
    option_types: &[i64],
    strikes: &[f64],
    premiums: &[f64],
    entry_prices: &[f64],
    quantities: &[f64],
    multipliers: &[f64],
    time_to_expiries: &[f64],
    volatilities: &[f64],
    rates: &[f64],
    carries: &[f64],
) -> f64 {
    let n_legs = instruments.len();
    // Validate that all leg slices are the same length; return NaN if not.
    if sides.len() != n_legs
        || option_types.len() != n_legs
        || strikes.len() != n_legs
        || premiums.len() != n_legs
        || entry_prices.len() != n_legs
        || quantities.len() != n_legs
        || multipliers.len() != n_legs
        || time_to_expiries.len() != n_legs
        || volatilities.len() != n_legs
        || rates.len() != n_legs
        || carries.len() != n_legs
    {
        return f64::NAN;
    }

    let mut total = 0.0_f64;

    for leg_idx in 0..n_legs {
        let inst = instruments[leg_idx];
        let sign = side_sign(sides[leg_idx]);
        if sign.is_nan() {
            continue;
        }
        let leg_scale = sign * quantities[leg_idx] * multipliers[leg_idx];

        match inst {
            INSTRUMENT_OPTION => {
                let kind = match option_kind(option_types[leg_idx]) {
                    Some(k) => k,
                    None => continue,
                };
                let bsm = black_scholes_price(
                    spot,
                    strikes[leg_idx],
                    rates[leg_idx],
                    carries[leg_idx],
                    time_to_expiries[leg_idx],
                    volatilities[leg_idx],
                    kind,
                );
                total += leg_scale * (bsm - premiums[leg_idx]);
            }
            INSTRUMENT_FUTURE | INSTRUMENT_STOCK => {
                total += leg_scale * (spot - entry_prices[leg_idx]);
            }
            _ => {}
        }
    }

    total
}

// ---------------------------------------------------------------------------
// aggregate_greeks_dense
// ---------------------------------------------------------------------------

/// Aggregate BSM Greeks for a multi-leg strategy at a single spot.
///
/// Parameters (all slices of length n_legs):
/// - `instruments`: 0=option, 1=future, 2=stock
/// - `sides`: 1=long, -1=short
/// - `option_types`: 1=call, -1=put (ignored for futures/stocks)
/// - `strikes`: strike price for option legs
/// - `volatilities`: implied vol for option legs
/// - `time_to_expiries`: TTE in years for option legs
/// - `rates`: risk-free rate for each leg
/// - `carries`: carry/dividend yield for option legs
/// - `quantities`, `multipliers`: applied to all instruments
///
/// Returns `(delta, gamma, vega, theta, rho)` aggregate across all legs.
/// Future/stock legs contribute `leg_scale` to delta only (all other Greeks = 0).
#[allow(clippy::too_many_arguments)]
pub fn aggregate_greeks_dense(
    spot: f64,
    instruments: &[i64],
    sides: &[i64],
    option_types: &[i64],
    strikes: &[f64],
    volatilities: &[f64],
    time_to_expiries: &[f64],
    rates: &[f64],
    carries: &[f64],
    quantities: &[f64],
    multipliers: &[f64],
) -> (f64, f64, f64, f64, f64) {
    use super::greeks::model_greeks;
    use super::{OptionContract, OptionEvaluation, PricingModel};

    let n_legs = instruments.len();
    if sides.len() != n_legs
        || option_types.len() != n_legs
        || strikes.len() != n_legs
        || volatilities.len() != n_legs
        || time_to_expiries.len() != n_legs
        || rates.len() != n_legs
        || carries.len() != n_legs
        || quantities.len() != n_legs
        || multipliers.len() != n_legs
    {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }

    let mut delta = 0.0_f64;
    let mut gamma = 0.0_f64;
    let mut vega = 0.0_f64;
    let mut theta = 0.0_f64;
    let mut rho = 0.0_f64;

    for i in 0..n_legs {
        let sign = side_sign(sides[i]);
        if sign.is_nan() {
            continue;
        }
        let leg_scale = sign * quantities[i] * multipliers[i];

        match instruments[i] {
            INSTRUMENT_FUTURE | INSTRUMENT_STOCK => {
                delta += leg_scale;
            }
            INSTRUMENT_OPTION => {
                let kind = match option_kind(option_types[i]) {
                    Some(k) => k,
                    None => continue,
                };
                let greeks = model_greeks(OptionEvaluation {
                    contract: OptionContract {
                        model: PricingModel::BlackScholes,
                        underlying: spot,
                        strike: strikes[i],
                        rate: rates[i],
                        carry: carries[i],
                        time_to_expiry: time_to_expiries[i],
                        kind,
                    },
                    volatility: volatilities[i],
                });
                delta += leg_scale * greeks.delta;
                gamma += leg_scale * greeks.gamma;
                vega += leg_scale * greeks.vega;
                theta += leg_scale * greeks.theta;
                rho += leg_scale * greeks.rho;
            }
            _ => {}
        }
    }

    (delta, gamma, vega, theta, rho)
}

/// Evaluate `strategy_value_dense` for each point in `spot_grid`.
///
/// Returns a `Vec<f64>` of length `spot_grid.len()`.
#[allow(clippy::too_many_arguments)]
pub fn strategy_value_grid(
    spot_grid: &[f64],
    instruments: &[i64],
    sides: &[i64],
    option_types: &[i64],
    strikes: &[f64],
    premiums: &[f64],
    entry_prices: &[f64],
    quantities: &[f64],
    multipliers: &[f64],
    time_to_expiries: &[f64],
    volatilities: &[f64],
    rates: &[f64],
    carries: &[f64],
) -> Vec<f64> {
    spot_grid
        .iter()
        .map(|&s| {
            strategy_value_dense(
                s,
                instruments,
                sides,
                option_types,
                strikes,
                premiums,
                entry_prices,
                quantities,
                multipliers,
                time_to_expiries,
                volatilities,
                rates,
                carries,
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payoff_single_call() {
        let grid = vec![90.0, 100.0, 110.0, 120.0];
        let out = strategy_payoff_dense(
            &grid,
            &[0],
            &[1],
            &[1],
            &[100.0],
            &[5.0],
            &[0.0],
            &[1.0],
            &[1.0],
        );
        assert!(out[0] < 0.0); // below strike, loss = premium
        assert!((out[0] - (-5.0)).abs() < 1e-10);
        assert!((out[2] - 5.0).abs() < 1e-10); // at 110, intrinsic=10, net=10-5=5
    }

    #[test]
    fn stock_leg_linear() {
        let grid = vec![90.0, 100.0, 110.0];
        let out = strategy_payoff_dense(
            &grid,
            &[2],
            &[1],
            &[0],
            &[0.0],
            &[0.0],
            &[100.0],
            &[1.0],
            &[1.0],
        );
        assert!((out[0] - (-10.0)).abs() < 1e-10);
        assert!((out[1] - 0.0).abs() < 1e-10);
        assert!((out[2] - 10.0).abs() < 1e-10);
    }
}
