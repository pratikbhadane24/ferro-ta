//! Option Greeks.

use super::normal::{cdf, pdf};
use super::pricing::{black_76_price, black_scholes_price};
use super::{ExtendedGreeks, Greeks, OptionEvaluation, OptionKind, PricingModel};

fn bs_inputs_valid(
    underlying: f64,
    strike: f64,
    rate: f64,
    carry: f64,
    time_to_expiry: f64,
    volatility: f64,
) -> bool {
    underlying.is_finite()
        && strike.is_finite()
        && rate.is_finite()
        && carry.is_finite()
        && time_to_expiry.is_finite()
        && volatility.is_finite()
        && underlying > 0.0
        && strike > 0.0
        && time_to_expiry > 0.0
        && volatility > 0.0
}

fn numerical_theta<F>(time_to_expiry: f64, price_fn: F) -> f64
where
    F: Fn(f64) -> f64,
{
    if time_to_expiry <= 0.0 {
        return 0.0;
    }
    let h = time_to_expiry.clamp(1e-6, 1.0 / 365.0);
    let t_minus = (time_to_expiry - h).max(1e-8);
    let t_plus = time_to_expiry + h;
    let price_minus = price_fn(t_minus);
    let price_plus = price_fn(t_plus);
    (price_minus - price_plus) / (t_plus - t_minus)
}

/// Black-Scholes-Merton Greeks.
pub fn black_scholes_greeks(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    time_to_expiry: f64,
    volatility: f64,
    kind: OptionKind,
) -> Greeks {
    if !bs_inputs_valid(
        spot,
        strike,
        rate,
        dividend_yield,
        time_to_expiry,
        volatility,
    ) {
        return Greeks {
            delta: f64::NAN,
            gamma: f64::NAN,
            vega: f64::NAN,
            theta: f64::NAN,
            rho: f64::NAN,
        };
    }

    let sqrt_t = time_to_expiry.sqrt();
    let sigma_sqrt_t = volatility * sqrt_t;
    let discount = (-rate * time_to_expiry).exp();
    let carry_discount = (-dividend_yield * time_to_expiry).exp();
    let d1 = ((spot / strike).ln()
        + (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry)
        / sigma_sqrt_t;
    let d2 = d1 - sigma_sqrt_t;
    let pdf_d1 = pdf(d1);

    let delta = match kind {
        OptionKind::Call => carry_discount * cdf(d1),
        OptionKind::Put => carry_discount * (cdf(d1) - 1.0),
    };
    let gamma = carry_discount * pdf_d1 / (spot * sigma_sqrt_t);
    let vega = spot * carry_discount * pdf_d1 * sqrt_t;
    let theta = match kind {
        OptionKind::Call => {
            -(spot * carry_discount * pdf_d1 * volatility) / (2.0 * sqrt_t)
                - rate * strike * discount * cdf(d2)
                + dividend_yield * spot * carry_discount * cdf(d1)
        }
        OptionKind::Put => {
            -(spot * carry_discount * pdf_d1 * volatility) / (2.0 * sqrt_t)
                + rate * strike * discount * cdf(-d2)
                - dividend_yield * spot * carry_discount * cdf(-d1)
        }
    };
    let rho = match kind {
        OptionKind::Call => strike * time_to_expiry * discount * cdf(d2),
        OptionKind::Put => -strike * time_to_expiry * discount * cdf(-d2),
    };

    Greeks {
        delta,
        gamma,
        vega,
        theta,
        rho,
    }
}

/// Black-76 Greeks with respect to the forward.
pub fn black_76_greeks(
    forward: f64,
    strike: f64,
    rate: f64,
    time_to_expiry: f64,
    volatility: f64,
    kind: OptionKind,
) -> Greeks {
    if !bs_inputs_valid(forward, strike, rate, 0.0, time_to_expiry, volatility) {
        return Greeks {
            delta: f64::NAN,
            gamma: f64::NAN,
            vega: f64::NAN,
            theta: f64::NAN,
            rho: f64::NAN,
        };
    }

    let sqrt_t = time_to_expiry.sqrt();
    let sigma_sqrt_t = volatility * sqrt_t;
    let discount = (-rate * time_to_expiry).exp();
    let d1 =
        ((forward / strike).ln() + 0.5 * volatility * volatility * time_to_expiry) / sigma_sqrt_t;
    let d2 = d1 - sigma_sqrt_t;
    let pdf_d1 = pdf(d1);

    let delta = match kind {
        OptionKind::Call => discount * cdf(d1),
        OptionKind::Put => -discount * cdf(-d1),
    };
    let gamma = discount * pdf_d1 / (forward * sigma_sqrt_t);
    let vega = discount * forward * pdf_d1 * sqrt_t;
    let theta = numerical_theta(time_to_expiry, |t| {
        black_76_price(forward, strike, rate, t, volatility, kind)
    });
    // Analytical ρ holding the forward fixed: −T · K · e^{−rT} · N(±d2).
    // (−T · price) is the total derivative through the discount factor on the
    // whole premium, not the strike-discount ρ used here.
    let rho = match kind {
        OptionKind::Call => -time_to_expiry * strike * discount * cdf(d2),
        OptionKind::Put => -time_to_expiry * strike * discount * cdf(-d2),
    };

    Greeks {
        delta,
        gamma,
        vega,
        theta,
        rho,
    }
}

/// Model-dispatched Greeks.
pub fn model_greeks(input: OptionEvaluation) -> Greeks {
    let contract = input.contract;
    match contract.model {
        PricingModel::BlackScholes => black_scholes_greeks(
            contract.underlying,
            contract.strike,
            contract.rate,
            contract.carry,
            contract.time_to_expiry,
            input.volatility,
            contract.kind,
        ),
        PricingModel::Black76 => black_76_greeks(
            contract.underlying,
            contract.strike,
            contract.rate,
            contract.time_to_expiry,
            input.volatility,
            contract.kind,
        ),
    }
}

/// Price derivative with respect to calendar time using the selected model.
pub fn model_theta(input: OptionEvaluation) -> f64 {
    let contract = input.contract;
    numerical_theta(contract.time_to_expiry, |t| match contract.model {
        PricingModel::BlackScholes => black_scholes_price(
            contract.underlying,
            contract.strike,
            contract.rate,
            contract.carry,
            t,
            input.volatility,
            contract.kind,
        ),
        PricingModel::Black76 => black_76_price(
            contract.underlying,
            contract.strike,
            contract.rate,
            t,
            input.volatility,
            contract.kind,
        ),
    })
}

/// Extended Greeks under Black-Scholes-Merton (closed-form).
///
/// All inputs must be positive finite; returns NaN fields for invalid inputs.
pub fn black_scholes_extended_greeks(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    time_to_expiry: f64,
    volatility: f64,
    kind: OptionKind,
) -> ExtendedGreeks {
    if !bs_inputs_valid(
        spot,
        strike,
        rate,
        dividend_yield,
        time_to_expiry,
        volatility,
    ) {
        return ExtendedGreeks {
            vanna: f64::NAN,
            volga: f64::NAN,
            charm: f64::NAN,
            speed: f64::NAN,
            color: f64::NAN,
        };
    }

    let sqrt_t = time_to_expiry.sqrt();
    let sigma_sqrt_t = volatility * sqrt_t;
    let carry_discount = (-dividend_yield * time_to_expiry).exp();
    let d1 = ((spot / strike).ln()
        + (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry)
        / sigma_sqrt_t;
    let d2 = d1 - sigma_sqrt_t;
    let pdf_d1 = pdf(d1);

    let gamma = carry_discount * pdf_d1 / (spot * sigma_sqrt_t);

    let vanna = -carry_discount * pdf_d1 * d2 / volatility;
    let volga = spot * carry_discount * pdf_d1 * sqrt_t * d1 * d2 / volatility;
    // Charm = ∂Δ/∂τ. The n(d1) term is shared; the q N(±d1) term differs
    // for puts (Haug / Wikipedia).
    let charm_common = pdf_d1
        * (2.0 * (rate - dividend_yield) * time_to_expiry - d2 * sigma_sqrt_t)
        / (2.0 * time_to_expiry * sigma_sqrt_t);
    let charm_extra = match kind {
        OptionKind::Call => dividend_yield * cdf(d1),
        OptionKind::Put => -dividend_yield * cdf(-d1),
    };
    let charm = -carry_discount * (charm_common + charm_extra);
    let speed = -gamma / spot * (d1 / sigma_sqrt_t + 1.0);
    // Color = ∂Γ/∂τ is identical for calls and puts (Γ itself is).
    let color = -carry_discount * pdf_d1 / (2.0 * spot * time_to_expiry * sigma_sqrt_t)
        * (2.0 * (rate - dividend_yield) * time_to_expiry + 1.0
            - d1 * (2.0 * (rate - dividend_yield) * time_to_expiry - d2 * sigma_sqrt_t)
                / sigma_sqrt_t);

    ExtendedGreeks {
        vanna,
        volga,
        charm,
        speed,
        color,
    }
}

/// Model-dispatched extended Greeks.
/// Only BSM is supported with closed-form; Black-76 is not yet supported (returns NaN).
pub fn model_extended_greeks(input: OptionEvaluation) -> ExtendedGreeks {
    let contract = input.contract;
    match contract.model {
        PricingModel::BlackScholes => black_scholes_extended_greeks(
            contract.underlying,
            contract.strike,
            contract.rate,
            contract.carry,
            contract.time_to_expiry,
            input.volatility,
            contract.kind,
        ),
        PricingModel::Black76 => ExtendedGreeks {
            vanna: f64::NAN,
            volga: f64::NAN,
            charm: f64::NAN,
            speed: f64::NAN,
            color: f64::NAN,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{black_76_greeks, black_scholes_extended_greeks, black_scholes_greeks};
    use crate::options::OptionKind;

    #[test]
    fn bsm_greeks_are_finite() {
        let g = black_scholes_greeks(100.0, 100.0, 0.05, 0.0, 1.0, 0.2, OptionKind::Call);
        assert!(g.delta.is_finite());
        assert!(g.gamma.is_finite());
        assert!(g.vega.is_finite());
        assert!(g.theta.is_finite());
        assert!(g.rho.is_finite());
    }

    #[test]
    fn black_76_greeks_are_finite() {
        let g = black_76_greeks(100.0, 100.0, 0.03, 1.0, 0.2, OptionKind::Put);
        assert!(g.delta.is_finite());
        assert!(g.gamma.is_finite());
        assert!(g.vega.is_finite());
        assert!(g.theta.is_finite());
        assert!(g.rho.is_finite());
    }

    #[test]
    fn extended_greeks_finite_for_valid_inputs() {
        let eg = black_scholes_extended_greeks(100.0, 100.0, 0.05, 0.0, 1.0, 0.2, OptionKind::Call);
        assert!(eg.vanna.is_finite());
        assert!(eg.volga.is_finite());
        assert!(eg.charm.is_finite());
        assert!(eg.speed.is_finite());
        assert!(eg.color.is_finite());
        // Volga must be positive (convex in vol)
        assert!(eg.volga >= 0.0);
    }

    fn expected_charm(
        spot: f64,
        strike: f64,
        rate: f64,
        dividend_yield: f64,
        time_to_expiry: f64,
        volatility: f64,
        kind: OptionKind,
    ) -> f64 {
        use crate::options::normal::{cdf, pdf};
        let sqrt_t = time_to_expiry.sqrt();
        let sigma_sqrt_t = volatility * sqrt_t;
        let carry_discount = (-dividend_yield * time_to_expiry).exp();
        let d1 = ((spot / strike).ln()
            + (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry)
            / sigma_sqrt_t;
        let d2 = d1 - sigma_sqrt_t;
        let common = pdf(d1) * (2.0 * (rate - dividend_yield) * time_to_expiry - d2 * sigma_sqrt_t)
            / (2.0 * time_to_expiry * sigma_sqrt_t);
        let extra = match kind {
            OptionKind::Call => dividend_yield * cdf(d1),
            OptionKind::Put => -dividend_yield * cdf(-d1),
        };
        -carry_discount * (common + extra)
    }

    #[test]
    fn extended_greeks_charm_depends_on_kind_when_carry_nonzero() {
        let call =
            black_scholes_extended_greeks(100.0, 100.0, 0.05, 0.03, 1.0, 0.2, OptionKind::Call);
        let put =
            black_scholes_extended_greeks(100.0, 100.0, 0.05, 0.03, 1.0, 0.2, OptionKind::Put);
        let exp_call = expected_charm(100.0, 100.0, 0.05, 0.03, 1.0, 0.2, OptionKind::Call);
        let exp_put = expected_charm(100.0, 100.0, 0.05, 0.03, 1.0, 0.2, OptionKind::Put);
        assert!(
            (call.charm - exp_call).abs() < 1e-10,
            "call charm {} != {}",
            call.charm,
            exp_call
        );
        assert!(
            (put.charm - exp_put).abs() < 1e-10,
            "put charm {} != {}",
            put.charm,
            exp_put
        );
        assert!(
            (call.charm - put.charm).abs() > 1e-6,
            "charm must differ for call vs put when q≠0"
        );
    }

    #[test]
    fn black_76_rho_is_analytical_strike_discount() {
        use crate::options::normal::cdf;
        let forward = 100.0_f64;
        let strike = 100.0_f64;
        let rate = 0.03_f64;
        let t = 1.0_f64;
        let vol = 0.2_f64;
        let sqrt_t = t.sqrt();
        let d1 = ((forward / strike).ln() + 0.5 * vol * vol * t) / (vol * sqrt_t);
        let d2 = d1 - vol * sqrt_t;
        let discount = (-rate * t).exp();
        let exp_call = -t * strike * discount * cdf(d2);
        let exp_put = -t * strike * discount * cdf(-d2);
        let call = black_76_greeks(forward, strike, rate, t, vol, OptionKind::Call);
        let put = black_76_greeks(forward, strike, rate, t, vol, OptionKind::Put);
        assert!(
            (call.rho - exp_call).abs() < 1e-10,
            "Black-76 call ρ {} != {}",
            call.rho,
            exp_call
        );
        assert!(
            (put.rho - exp_put).abs() < 1e-10,
            "Black-76 put ρ {} != {}",
            put.rho,
            exp_put
        );
        // The old −T·price formula is a different number for this ATM case.
        let price = crate::options::pricing::black_76_price(
            forward,
            strike,
            rate,
            t,
            vol,
            OptionKind::Call,
        );
        assert!((call.rho - (-t * price)).abs() > 1.0);
    }
}
