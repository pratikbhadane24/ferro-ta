//! Smile and surface analytics helpers.

use super::chain::atm_index;
use super::greeks::model_greeks;
use super::{ChainGreeksContext, OptionContract, OptionEvaluation, OptionKind, PricingModel};

/// Smile summary metrics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmileMetrics {
    pub atm_iv: f64,
    pub risk_reversal_25d: f64,
    pub butterfly_25d: f64,
    pub skew_slope: f64,
    pub convexity: f64,
}

/// Linear interpolation helper.
pub fn linear_interpolate(xs: &[f64], ys: &[f64], target: f64) -> f64 {
    if xs.len() != ys.len() || xs.is_empty() {
        return f64::NAN;
    }
    if target <= xs[0] {
        return ys[0];
    }
    for i in 1..xs.len() {
        if target <= xs[i] {
            let x0 = xs[i - 1];
            let x1 = xs[i];
            let y0 = ys[i - 1];
            let y1 = ys[i];
            let w = if x1 == x0 {
                0.0
            } else {
                (target - x0) / (x1 - x0)
            };
            return y0 + w * (y1 - y0);
        }
    }
    ys[ys.len() - 1]
}

/// ATM implied volatility by nearest strike.
pub fn atm_iv(strikes: &[f64], vols: &[f64], reference_price: f64) -> f64 {
    if strikes.len() != vols.len() || strikes.is_empty() || !reference_price.is_finite() {
        return f64::NAN;
    }
    atm_index(strikes, reference_price)
        .and_then(|idx| vols.get(idx).copied())
        .unwrap_or(f64::NAN)
}

fn regression_slope(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.len() < 2 {
        return f64::NAN;
    }
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        cov += (x - mean_x) * (y - mean_y);
        var += (x - mean_x) * (x - mean_x);
    }
    if var == 0.0 {
        f64::NAN
    } else {
        cov / var
    }
}

fn closest_delta_iv(
    strikes: &[f64],
    vols: &[f64],
    context: ChainGreeksContext,
    target_delta: f64,
) -> f64 {
    let mut best_iv = f64::NAN;
    let mut best_distance = f64::INFINITY;
    for (&strike, &vol) in strikes.iter().zip(vols.iter()) {
        if !strike.is_finite() || !vol.is_finite() {
            continue;
        }
        let delta = model_greeks(OptionEvaluation {
            contract: OptionContract {
                model: context.model,
                underlying: context.reference_price,
                strike,
                rate: context.rate,
                carry: context.carry,
                time_to_expiry: context.time_to_expiry,
                kind: context.kind,
            },
            volatility: vol,
        })
        .delta;
        if !delta.is_finite() {
            continue;
        }
        let distance = (delta - target_delta).abs();
        if distance < best_distance {
            best_distance = distance;
            best_iv = vol;
        }
    }
    best_iv
}

/// Smile metrics from a single expiry slice.
pub fn smile_metrics(
    strikes: &[f64],
    vols: &[f64],
    reference_price: f64,
    rate: f64,
    carry: f64,
    time_to_expiry: f64,
    model: PricingModel,
) -> SmileMetrics {
    if strikes.len() != vols.len() || strikes.len() < 3 || reference_price <= 0.0 {
        return SmileMetrics {
            atm_iv: f64::NAN,
            risk_reversal_25d: f64::NAN,
            butterfly_25d: f64::NAN,
            skew_slope: f64::NAN,
            convexity: f64::NAN,
        };
    }

    let atm_idx = match atm_index(strikes, reference_price) {
        Some(idx) => idx,
        None => {
            return SmileMetrics {
                atm_iv: f64::NAN,
                risk_reversal_25d: f64::NAN,
                butterfly_25d: f64::NAN,
                skew_slope: f64::NAN,
                convexity: f64::NAN,
            }
        }
    };
    let atm_iv = vols[atm_idx];

    let call_25 = closest_delta_iv(
        strikes,
        vols,
        ChainGreeksContext {
            model,
            reference_price,
            rate,
            carry,
            time_to_expiry,
            kind: OptionKind::Call,
        },
        0.25,
    );
    let put_25 = closest_delta_iv(
        strikes,
        vols,
        ChainGreeksContext {
            model,
            reference_price,
            rate,
            carry,
            time_to_expiry,
            kind: OptionKind::Put,
        },
        -0.25,
    );
    let risk_reversal_25d = call_25 - put_25;
    let butterfly_25d = 0.5 * (call_25 + put_25) - atm_iv;

    let log_moneyness: Vec<f64> = strikes
        .iter()
        .map(|&k| (k / reference_price).ln())
        .collect();
    let skew_slope = regression_slope(&log_moneyness, vols);
    let convexity = if atm_idx > 0 && atm_idx + 1 < strikes.len() {
        let x0 = log_moneyness[atm_idx - 1];
        let x1 = log_moneyness[atm_idx];
        let x2 = log_moneyness[atm_idx + 1];
        let y0 = vols[atm_idx - 1];
        let y1 = vols[atm_idx];
        let y2 = vols[atm_idx + 1];
        let left = if x1 == x0 { 0.0 } else { (y1 - y0) / (x1 - x0) };
        let right = if x2 == x1 { 0.0 } else { (y2 - y1) / (x2 - x1) };
        right - left
    } else {
        f64::NAN
    };

    SmileMetrics {
        atm_iv,
        risk_reversal_25d,
        butterfly_25d,
        skew_slope,
        convexity,
    }
}

/// Term-structure slope from (tenor, atm_iv) points.
pub fn term_structure_slope(tenors: &[f64], atm_ivs: &[f64]) -> f64 {
    regression_slope(tenors, atm_ivs)
}

/// Expected ±1σ move over `days_to_expiry` calendar days.
///
/// Returns `(lower_move, upper_move)` as absolute changes from `spot`.
/// Example: if spot=100 and upper_move=5.0 then the 1σ upper bound is 105.
///
/// Uses the log-normal approximation: `spot × e^{±σ√(days/trading_days)} − spot`.
pub fn expected_move(
    spot: f64,
    iv: f64,
    days_to_expiry: f64,
    trading_days_per_year: f64,
) -> (f64, f64) {
    if !spot.is_finite()
        || !iv.is_finite()
        || !days_to_expiry.is_finite()
        || !trading_days_per_year.is_finite()
        || spot <= 0.0
        || iv < 0.0
        || days_to_expiry < 0.0
        || trading_days_per_year <= 0.0
    {
        return (f64::NAN, f64::NAN);
    }
    let sigma_sqrt_t = iv * (days_to_expiry / trading_days_per_year).sqrt();
    let upper = spot * sigma_sqrt_t.exp() - spot;
    let lower = spot * (-sigma_sqrt_t).exp() - spot;
    (lower, upper)
}

#[cfg(test)]
mod tests {
    use super::{atm_iv, smile_metrics, term_structure_slope};
    use crate::options::PricingModel;

    #[test]
    fn atm_selection_works() {
        let strikes = [90.0, 100.0, 110.0];
        let vols = [0.24, 0.20, 0.22];
        assert!((atm_iv(&strikes, &vols, 102.0) - 0.20).abs() < 1e-12);
    }

    #[test]
    fn smile_metrics_are_finite() {
        let strikes = [80.0, 90.0, 100.0, 110.0, 120.0];
        let vols = [0.30, 0.25, 0.20, 0.22, 0.27];
        let metrics = smile_metrics(
            &strikes,
            &vols,
            100.0,
            0.02,
            0.0,
            0.5,
            PricingModel::BlackScholes,
        );
        assert!(metrics.atm_iv.is_finite());
        assert!(metrics.skew_slope.is_finite());
    }

    #[test]
    fn term_slope_is_reasonable() {
        let tenors = [0.1, 0.5, 1.0];
        let vols = [0.18, 0.20, 0.22];
        assert!(term_structure_slope(&tenors, &vols) > 0.0);
    }
}
