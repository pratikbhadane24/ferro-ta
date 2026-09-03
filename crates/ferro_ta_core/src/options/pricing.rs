//! Option pricing models.

use super::normal::cdf;
use super::{OptionContract, OptionEvaluation, OptionKind, PricingModel};

fn invalid_inputs(underlying: f64, strike: f64, time_to_expiry: f64, volatility: f64) -> bool {
    !underlying.is_finite()
        || !strike.is_finite()
        || !time_to_expiry.is_finite()
        || !volatility.is_finite()
        || underlying <= 0.0
        || strike <= 0.0
        || time_to_expiry < 0.0
        || volatility < 0.0
}

/// Black-Scholes-Merton price with continuous carry/dividend yield.
pub fn black_scholes_price(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    time_to_expiry: f64,
    volatility: f64,
    kind: OptionKind,
) -> f64 {
    if invalid_inputs(spot, strike, time_to_expiry, volatility) || !rate.is_finite() {
        return f64::NAN;
    }
    if time_to_expiry == 0.0 {
        return match kind {
            OptionKind::Call => (spot - strike).max(0.0),
            OptionKind::Put => (strike - spot).max(0.0),
        };
    }

    let discount = (-rate * time_to_expiry).exp();
    let carry_discount = (-dividend_yield * time_to_expiry).exp();
    if volatility == 0.0 {
        return match kind {
            OptionKind::Call => (spot * carry_discount - strike * discount).max(0.0),
            OptionKind::Put => (strike * discount - spot * carry_discount).max(0.0),
        };
    }

    let sqrt_t = time_to_expiry.sqrt();
    let sigma_sqrt_t = volatility * sqrt_t;
    let d1 = ((spot / strike).ln()
        + (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry)
        / sigma_sqrt_t;
    let d2 = d1 - sigma_sqrt_t;

    match kind {
        OptionKind::Call => spot * carry_discount * cdf(d1) - strike * discount * cdf(d2),
        OptionKind::Put => strike * discount * cdf(-d2) - spot * carry_discount * cdf(-d1),
    }
}

/// Black-76 price using the forward price as the underlying input.
pub fn black_76_price(
    forward: f64,
    strike: f64,
    rate: f64,
    time_to_expiry: f64,
    volatility: f64,
    kind: OptionKind,
) -> f64 {
    if invalid_inputs(forward, strike, time_to_expiry, volatility) || !rate.is_finite() {
        return f64::NAN;
    }
    let discount = (-rate * time_to_expiry).exp();
    if time_to_expiry == 0.0 {
        return discount
            * match kind {
                OptionKind::Call => (forward - strike).max(0.0),
                OptionKind::Put => (strike - forward).max(0.0),
            };
    }
    if volatility == 0.0 {
        return discount
            * match kind {
                OptionKind::Call => (forward - strike).max(0.0),
                OptionKind::Put => (strike - forward).max(0.0),
            };
    }

    let sqrt_t = time_to_expiry.sqrt();
    let sigma_sqrt_t = volatility * sqrt_t;
    let d1 =
        ((forward / strike).ln() + 0.5 * volatility * volatility * time_to_expiry) / sigma_sqrt_t;
    let d2 = d1 - sigma_sqrt_t;

    let signed = kind.sign();
    discount * signed * (forward * cdf(signed * d1) - strike * cdf(signed * d2))
}

/// Model-dispatched option price.
pub fn model_price(input: OptionEvaluation) -> f64 {
    let contract = input.contract;
    match contract.model {
        PricingModel::BlackScholes => black_scholes_price(
            contract.underlying,
            contract.strike,
            contract.rate,
            contract.carry,
            contract.time_to_expiry,
            input.volatility,
            contract.kind,
        ),
        PricingModel::Black76 => black_76_price(
            contract.underlying,
            contract.strike,
            contract.rate,
            contract.time_to_expiry,
            input.volatility,
            contract.kind,
        ),
    }
}

/// Put-call parity deviation: `C - P - (S·e^{-q·T} - K·e^{-r·T})`.
///
/// Returns 0.0 when no arbitrage exists. A non-zero value indicates the
/// magnitude of mispricing or data error.
pub fn put_call_parity_deviation(
    call_price: f64,
    put_price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    carry: f64,
    time_to_expiry: f64,
) -> f64 {
    if !call_price.is_finite()
        || !put_price.is_finite()
        || !spot.is_finite()
        || !strike.is_finite()
        || !rate.is_finite()
        || !carry.is_finite()
        || !time_to_expiry.is_finite()
        || spot <= 0.0
        || strike <= 0.0
        || time_to_expiry < 0.0
    {
        return f64::NAN;
    }
    let pv_forward = spot * (-carry * time_to_expiry).exp();
    let pv_strike = strike * (-rate * time_to_expiry).exp();
    call_price - put_price - (pv_forward - pv_strike)
}

/// Lower no-arbitrage bound for the option price.
pub fn price_lower_bound(contract: OptionContract) -> f64 {
    match contract.model {
        PricingModel::BlackScholes => {
            let discount = (-contract.rate * contract.time_to_expiry).exp();
            let carry_discount = (-contract.carry * contract.time_to_expiry).exp();
            match contract.kind {
                OptionKind::Call => {
                    (contract.underlying * carry_discount - contract.strike * discount).max(0.0)
                }
                OptionKind::Put => {
                    (contract.strike * discount - contract.underlying * carry_discount).max(0.0)
                }
            }
        }
        PricingModel::Black76 => {
            let discount = (-contract.rate * contract.time_to_expiry).exp();
            discount
                * match contract.kind {
                    OptionKind::Call => (contract.underlying - contract.strike).max(0.0),
                    OptionKind::Put => (contract.strike - contract.underlying).max(0.0),
                }
        }
    }
}

/// Upper no-arbitrage bound for the option price.
pub fn price_upper_bound(contract: OptionContract) -> f64 {
    match contract.model {
        PricingModel::BlackScholes => match contract.kind {
            OptionKind::Call => {
                contract.underlying * (-contract.carry * contract.time_to_expiry).exp()
            }
            OptionKind::Put => contract.strike * (-contract.rate * contract.time_to_expiry).exp(),
        },
        PricingModel::Black76 => {
            let discount = (-contract.rate * contract.time_to_expiry).exp();
            discount
                * match contract.kind {
                    OptionKind::Call => contract.underlying,
                    OptionKind::Put => contract.strike,
                }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{black_76_price, black_scholes_price};
    use crate::options::OptionKind;

    #[test]
    fn black_scholes_prices_are_reasonable() {
        let call = black_scholes_price(100.0, 100.0, 0.05, 0.0, 1.0, 0.2, OptionKind::Call);
        let put = black_scholes_price(100.0, 100.0, 0.05, 0.0, 1.0, 0.2, OptionKind::Put);
        assert!((call - 10.4506).abs() < 1e-3);
        assert!((put - 5.5735).abs() < 1e-3);
    }

    #[test]
    fn black_76_prices_are_reasonable() {
        let call = black_76_price(100.0, 100.0, 0.03, 1.0, 0.2, OptionKind::Call);
        let put = black_76_price(100.0, 100.0, 0.03, 1.0, 0.2, OptionKind::Put);
        assert!((call - 7.730_148).abs() < 1e-3);
        assert!((put - 7.730_148).abs() < 1e-3);
    }

    #[test]
    fn bsm_d1_d2_lock_in_price() {
        use super::put_call_parity_deviation;
        use crate::options::normal::cdf;
        let spot = 100.0_f64;
        let strike = 105.0_f64;
        let rate = 0.05_f64;
        let q = 0.02_f64;
        let t = 1.25_f64;
        let vol = 0.22_f64;
        let sigma_sqrt_t = vol * t.sqrt();
        let d1 = ((spot / strike).ln() + (rate - q + 0.5 * vol * vol) * t) / sigma_sqrt_t;
        let d2 = d1 - sigma_sqrt_t;
        assert!((d2 - (d1 - sigma_sqrt_t)).abs() < 1e-15);
        let discount = (-rate * t).exp();
        let carry_discount = (-q * t).exp();
        let expected_call = spot * carry_discount * cdf(d1) - strike * discount * cdf(d2);
        let expected_put = strike * discount * cdf(-d2) - spot * carry_discount * cdf(-d1);
        let call = black_scholes_price(spot, strike, rate, q, t, vol, OptionKind::Call);
        let put = black_scholes_price(spot, strike, rate, q, t, vol, OptionKind::Put);
        assert!((call - expected_call).abs() < 1e-12);
        assert!((put - expected_put).abs() < 1e-12);
        let parity = put_call_parity_deviation(call, put, spot, strike, rate, q, t);
        assert!(parity.abs() < 1e-12, "put-call parity residual {parity}");
        let fwd = spot * carry_discount - strike * discount;
        assert!(((call - put) - fwd).abs() < 1e-12);
    }

    #[test]
    fn black_76_d1_d2_lock_in_price() {
        use crate::options::normal::cdf;
        let forward = 100.0_f64;
        let strike = 95.0_f64;
        let rate = 0.03_f64;
        let t = 0.75_f64;
        let vol = 0.18_f64;
        let sigma_sqrt_t = vol * t.sqrt();
        let d1 = ((forward / strike).ln() + 0.5 * vol * vol * t) / sigma_sqrt_t;
        let d2 = d1 - sigma_sqrt_t;
        let discount = (-rate * t).exp();
        let expected = discount * (forward * cdf(d1) - strike * cdf(d2));
        let call = black_76_price(forward, strike, rate, t, vol, OptionKind::Call);
        assert!((call - expected).abs() < 1e-12);
    }
}
