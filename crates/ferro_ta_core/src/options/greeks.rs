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
    let rho =
        -time_to_expiry * black_76_price(forward, strike, rate, time_to_expiry, volatility, kind);

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

    // dd1/dT — shared by charm and color.
    let dd1_dt = (2.0 * (rate - dividend_yield) * time_to_expiry - d2 * sigma_sqrt_t)
        / (2.0 * time_to_expiry * sigma_sqrt_t);

    // charm = dDelta/dt = -dDelta/dT. The dividend term is kind-dependent and
    // vanishes only when dividend_yield == 0:
    //   call: delta = e^{-qT} N(d1)   → charm = +q e^{-qT} N(d1)  - e^{-qT} phi(d1) dd1_dT
    //   put:  delta = -e^{-qT} N(-d1) → charm = -q e^{-qT} N(-d1) - e^{-qT} phi(d1) dd1_dT
    let charm_dividend_term = match kind {
        OptionKind::Call => dividend_yield * carry_discount * cdf(d1),
        OptionKind::Put => -dividend_yield * carry_discount * cdf(-d1),
    };
    let charm = charm_dividend_term - carry_discount * pdf_d1 * dd1_dt;

    let speed = -gamma / spot * (d1 / sigma_sqrt_t + 1.0);

    // color = dGamma/dt = -dGamma/dT. From gamma = e^{-qT} phi(d1)/(S sigma sqrt(T)):
    //   dGamma/dT = gamma * (-q - d1 * dd1_dT - 1/(2T))
    // so color = gamma * (q + d1 * dd1_dT + 1/(2T)).
    let color = gamma * (dividend_yield + d1 * dd1_dt + 1.0 / (2.0 * time_to_expiry));

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

    /// Every extended Greek is a derivative of a first-order Greek, so each
    /// closed form must agree with a central difference of the first-order
    /// Greeks this module already computes. Finiteness assertions alone let
    /// sign errors and missing dividend terms through.
    #[test]
    fn extended_greeks_match_finite_differences() {
        // (spot, strike, rate, dividend_yield, T, vol, kind, label)
        let cases = [
            (
                100.0,
                100.0,
                0.05,
                0.03,
                1.0,
                0.2,
                OptionKind::Call,
                "call q=3%",
            ),
            (
                100.0,
                100.0,
                0.05,
                0.0,
                1.0,
                0.2,
                OptionKind::Call,
                "call q=0",
            ),
            (
                100.0,
                100.0,
                0.05,
                0.03,
                1.0,
                0.2,
                OptionKind::Put,
                "put q=3%",
            ),
            (
                110.0,
                100.0,
                0.04,
                0.06,
                0.5,
                0.3,
                OptionKind::Call,
                "call q>r",
            ),
            (
                95.0,
                100.0,
                0.02,
                0.01,
                2.0,
                0.25,
                OptionKind::Put,
                "put long T",
            ),
        ];
        let h = 1e-5;
        for (s, k, r, q, t, v, kind, label) in cases {
            let eg = black_scholes_extended_greeks(s, k, r, q, t, v, kind);
            let g = |s: f64, t: f64, v: f64| black_scholes_greeks(s, k, r, q, t, v, kind);

            // charm = dDelta/dt and color = dGamma/dt, where t is calendar
            // time: d/dt == -d/dT.
            let fd_charm = -(g(s, t + h, v).delta - g(s, t - h, v).delta) / (2.0 * h);
            let fd_color = -(g(s, t + h, v).gamma - g(s, t - h, v).gamma) / (2.0 * h);
            // vanna = dDelta/dsigma, speed = dGamma/dS.
            let fd_vanna = (g(s, t, v + h).delta - g(s, t, v - h).delta) / (2.0 * h);
            let fd_speed = (g(s + h, t, v).gamma - g(s - h, t, v).gamma) / (2.0 * h);
            // volga = dVega/dsigma (vega is per unit vol, not per 1%).
            let fd_volga = (g(s, t, v + h).vega - g(s, t, v - h).vega) / (2.0 * h);

            let close = |a: f64, b: f64| (a - b).abs() <= 1e-4 * b.abs().max(1.0);
            assert!(
                close(eg.charm, fd_charm),
                "{label} charm: {} vs {}",
                eg.charm,
                fd_charm
            );
            assert!(
                close(eg.color, fd_color),
                "{label} color: {} vs {}",
                eg.color,
                fd_color
            );
            assert!(
                close(eg.vanna, fd_vanna),
                "{label} vanna: {} vs {}",
                eg.vanna,
                fd_vanna
            );
            assert!(
                close(eg.speed, fd_speed),
                "{label} speed: {} vs {}",
                eg.speed,
                fd_speed
            );
            assert!(
                close(eg.volga, fd_volga),
                "{label} volga: {} vs {}",
                eg.volga,
                fd_volga
            );
        }
    }
}
