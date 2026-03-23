//! Implied volatility inversion and IV-series helpers.

use super::greeks::model_greeks;
use super::pricing::{model_price, price_lower_bound, price_upper_bound};
use super::{IvSolverConfig, OptionContract, OptionEvaluation};

/// Solve implied volatility with guarded Newton iterations and bisection fallback.
pub fn implied_volatility(
    contract: OptionContract,
    target_price: f64,
    config: IvSolverConfig,
) -> f64 {
    if !target_price.is_finite()
        || !contract.underlying.is_finite()
        || !contract.strike.is_finite()
        || !contract.rate.is_finite()
        || !contract.carry.is_finite()
        || !contract.time_to_expiry.is_finite()
        || target_price < 0.0
        || contract.underlying <= 0.0
        || contract.strike <= 0.0
        || contract.time_to_expiry < 0.0
    {
        return f64::NAN;
    }
    if contract.time_to_expiry == 0.0 {
        return 0.0;
    }

    let lower = price_lower_bound(contract);
    let upper = price_upper_bound(contract);
    if target_price < lower - config.tolerance || target_price > upper + config.tolerance {
        return f64::NAN;
    }
    if (target_price - lower).abs() <= config.tolerance {
        return 0.0;
    }

    let mut low_vol = 1e-9;
    let mut high_vol = config.initial_guess.max(0.25).max(low_vol * 10.0);
    let mut high_price = model_price(OptionEvaluation {
        contract,
        volatility: high_vol,
    });
    while high_price < target_price && high_vol < 10.0 {
        high_vol *= 2.0;
        high_price = model_price(OptionEvaluation {
            contract,
            volatility: high_vol,
        });
    }
    if high_price < target_price {
        return f64::NAN;
    }

    let mut vol = config.initial_guess.clamp(low_vol, high_vol).max(1e-4);
    for _ in 0..config.max_iterations.max(1) {
        let price = model_price(OptionEvaluation {
            contract,
            volatility: vol,
        });
        let diff = price - target_price;
        if diff.abs() <= config.tolerance {
            return vol;
        }

        if diff > 0.0 {
            high_vol = high_vol.min(vol);
        } else {
            low_vol = low_vol.max(vol);
        }

        let vega = model_greeks(OptionEvaluation {
            contract,
            volatility: vol,
        })
        .vega;

        let next = if vega.is_finite() && vega.abs() > 1e-10 {
            let candidate = vol - diff / vega;
            if candidate > low_vol && candidate < high_vol {
                candidate
            } else {
                0.5 * (low_vol + high_vol)
            }
        } else {
            0.5 * (low_vol + high_vol)
        };
        vol = next;
    }

    let final_price = model_price(OptionEvaluation {
        contract,
        volatility: vol,
    });
    if (final_price - target_price).abs() <= config.tolerance * 10.0 {
        vol
    } else {
        f64::NAN
    }
}

fn validate_window(window: usize) -> bool {
    window >= 1
}

/// Rolling IV rank.
pub fn iv_rank(iv_series: &[f64], window: usize) -> Vec<f64> {
    let n = iv_series.len();
    let mut out = vec![f64::NAN; n];
    if !validate_window(window) || n < window {
        return out;
    }

    for end in (window - 1)..n {
        let start = end + 1 - window;
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for &v in &iv_series[start..=end] {
            if v.is_finite() {
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }
        }
        let current = iv_series[end];
        if !current.is_finite() || !min_v.is_finite() || !max_v.is_finite() {
            out[end] = f64::NAN;
            continue;
        }
        let spread = max_v - min_v;
        out[end] = if spread == 0.0 {
            0.0
        } else {
            (current - min_v) / spread
        };
    }
    out
}

/// Rolling IV percentile.
pub fn iv_percentile(iv_series: &[f64], window: usize) -> Vec<f64> {
    let n = iv_series.len();
    let mut out = vec![f64::NAN; n];
    if !validate_window(window) || n < window {
        return out;
    }

    for end in (window - 1)..n {
        let start = end + 1 - window;
        let current = iv_series[end];
        let count = iv_series[start..=end]
            .iter()
            .filter(|&&v| v <= current)
            .count();
        out[end] = count as f64 / window as f64;
    }
    out
}

/// Rolling IV z-score.
pub fn iv_zscore(iv_series: &[f64], window: usize) -> Vec<f64> {
    let n = iv_series.len();
    let mut out = vec![f64::NAN; n];
    if !validate_window(window) || n < window {
        return out;
    }

    for end in (window - 1)..n {
        let start = end + 1 - window;
        let mut count = 0usize;
        let mut sum = 0.0;
        for &v in &iv_series[start..=end] {
            if v.is_finite() {
                count += 1;
                sum += v;
            }
        }
        if count == 0 {
            out[end] = f64::NAN;
            continue;
        }
        let mean = sum / count as f64;
        let mut var = 0.0;
        for &v in &iv_series[start..=end] {
            if v.is_finite() {
                let d = v - mean;
                var += d * d;
            }
        }
        let std = (var / count as f64).sqrt();
        let current = iv_series[end];
        out[end] = if !current.is_finite() || std == 0.0 {
            f64::NAN
        } else {
            (current - mean) / std
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{implied_volatility, iv_percentile, iv_rank, iv_zscore};
    use crate::options::pricing::black_scholes_price;
    use crate::options::{IvSolverConfig, OptionContract, OptionKind, PricingModel};

    #[test]
    fn solver_recovers_input_vol() {
        let price = black_scholes_price(100.0, 100.0, 0.05, 0.0, 1.0, 0.2, OptionKind::Call);
        let iv = implied_volatility(
            OptionContract {
                model: PricingModel::BlackScholes,
                underlying: 100.0,
                strike: 100.0,
                rate: 0.05,
                carry: 0.0,
                time_to_expiry: 1.0,
                kind: OptionKind::Call,
            },
            price,
            IvSolverConfig {
                initial_guess: 0.3,
                tolerance: 1e-8,
                max_iterations: 100,
            },
        );
        assert!((iv - 0.2).abs() < 1e-6);
    }

    #[test]
    fn iv_helpers_match_expected_values() {
        let iv = [10.0, 20.0, 30.0, 15.0, 22.0];
        let rank = iv_rank(&iv, 3);
        let pct = iv_percentile(&iv, 3);
        let z = iv_zscore(&iv, 3);
        assert!(rank[0].is_nan() && rank[1].is_nan());
        assert!((rank[2] - 1.0).abs() < 1e-12);
        assert!((pct[3] - (1.0 / 3.0)).abs() < 1e-12);
        assert!((z[2] - 1.224_744_871).abs() < 1e-6);
    }
}
