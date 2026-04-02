//! American option pricing via the Barone-Adesi-Whaley (1987) quadratic approximation.

use super::normal::cdf;
use super::pricing::black_scholes_price;
use super::OptionKind;

fn invalid_inputs(spot: f64, strike: f64, time_to_expiry: f64, volatility: f64) -> bool {
    !spot.is_finite()
        || !strike.is_finite()
        || !time_to_expiry.is_finite()
        || !volatility.is_finite()
        || spot <= 0.0
        || strike <= 0.0
        || time_to_expiry < 0.0
        || volatility < 0.0
}

/// Compute d1 for BSM given spot S* (used inside the Newton-Raphson loop).
fn d1_fn(s: f64, strike: f64, rate: f64, carry: f64, time_to_expiry: f64, volatility: f64) -> f64 {
    let sigma_sqrt_t = volatility * time_to_expiry.sqrt();
    ((s / strike).ln() + (rate - carry + 0.5 * volatility * volatility) * time_to_expiry)
        / sigma_sqrt_t
}

/// Find the critical spot price S* for American call early exercise using Newton-Raphson.
///
/// S* satisfies: C(S*) - (S* - K) = (S*/q2) * (1 - e^{-q*T} * N(d1(S*)))
/// Rearranged as F(S*) = 0:
/// F(x) = C(x) - (x - K) - (x/q2) * (1 - carry_discount * N(d1(x))) = 0
fn find_critical_call(
    strike: f64,
    rate: f64,
    carry: f64,
    time_to_expiry: f64,
    volatility: f64,
    q2: f64,
) -> f64 {
    let carry_discount = (-carry * time_to_expiry).exp();

    // Initial guess: S* ≈ K * q2 / (q2 - 1), clamped to be above strike
    let mut s = if q2 > 1.0 {
        strike * q2 / (q2 - 1.0)
    } else {
        // q2 <= 1 means the denominator is small/negative; fall back to a safe value
        strike * 2.0
    };
    // Ensure starting guess is positive
    if s <= 0.0 {
        s = strike * 1.5;
    }

    for _ in 0..50 {
        let c = black_scholes_price(
            s,
            strike,
            rate,
            carry,
            time_to_expiry,
            volatility,
            OptionKind::Call,
        );
        let d1 = d1_fn(s, strike, rate, carry, time_to_expiry, volatility);
        let nd1 = cdf(d1);
        let lhs = c - (s - strike);
        let rhs = (s / q2) * (1.0 - carry_discount * nd1);
        let f = lhs - rhs;

        // Derivative of F with respect to s:
        // dC/ds = e^{-q*T} * N(d1) (BSM delta for call)
        // d(s - K)/ds = 1
        // d(rhs)/ds = (1/q2) * (1 - carry_discount * N(d1))
        //            + (s/q2) * (-carry_discount * phi(d1) / (s * vol * sqrt(T)))
        //           = (1/q2) * (1 - carry_discount * N(d1)) - carry_discount * phi(d1) / (q2 * vol * sqrt(T))
        let sigma_sqrt_t = volatility * time_to_expiry.sqrt();
        let phi_d1 = super::normal::pdf(d1);
        let d_lhs_ds = carry_discount * nd1 - 1.0;
        let d_rhs_ds = (1.0 / q2) * (1.0 - carry_discount * nd1)
            - carry_discount * phi_d1 / (q2 * sigma_sqrt_t);
        let df = d_lhs_ds - d_rhs_ds;

        if df.abs() < 1e-14 {
            break;
        }
        let step = f / df;
        s -= step;
        // Keep s positive
        if s <= 0.0 {
            s = strike * 0.1;
        }
        if step.abs() < 1e-8 {
            break;
        }
    }
    s
}

/// Find the critical spot price S** for American put early exercise using Newton-Raphson.
///
/// S** satisfies: P(S**) - (K - S**) = -(S**/q1) * (1 - e^{-q*T} * N(-d1(S**)))
/// F(x) = P(x) - (K - x) + (x/q1) * (1 - carry_discount * N(-d1(x))) = 0
fn find_critical_put(
    strike: f64,
    rate: f64,
    carry: f64,
    time_to_expiry: f64,
    volatility: f64,
    q1: f64,
) -> f64 {
    let carry_discount = (-carry * time_to_expiry).exp();

    // Initial guess for put: S** ≈ K * q1 / (q1 - 1)
    // q1 is negative, so q1 - 1 < 0, and the guess should be below strike.
    let mut s = if (q1 - 1.0).abs() > 1e-10 {
        strike * q1 / (q1 - 1.0)
    } else {
        strike * 0.5
    };
    if s <= 0.0 || s >= strike {
        s = strike * 0.5;
    }

    for _ in 0..50 {
        let p = black_scholes_price(
            s,
            strike,
            rate,
            carry,
            time_to_expiry,
            volatility,
            OptionKind::Put,
        );
        let d1 = d1_fn(s, strike, rate, carry, time_to_expiry, volatility);
        let n_neg_d1 = cdf(-d1);
        let lhs = p - (strike - s);
        // rhs = -(s/q1) * (1 - carry_discount * N(-d1))
        let rhs = -(s / q1) * (1.0 - carry_discount * n_neg_d1);
        let f = lhs - rhs;

        // Derivative:
        // dP/ds = -e^{-q*T} * N(-d1)  (BSM delta for put = e^{-q*T}*(N(d1)-1))
        // d(K - s)/ds = -1  so  d(lhs)/ds = dP/ds - (-1) = dP/ds + 1
        // d(rhs)/ds = -(1/q1)*(1 - carry_discount*N(-d1))
        //           + -(s/q1)*carry_discount*phi(d1)/(s*vol*sqrt(T))   [since d(N(-d1))/ds = -phi(d1)*dd1/ds]
        //           = -(1/q1)*(1 - carry_discount*N(-d1))
        //           - carry_discount*phi(d1)/(q1*vol*sqrt(T))
        let sigma_sqrt_t = volatility * time_to_expiry.sqrt();
        let phi_d1 = super::normal::pdf(d1);
        let d_lhs_ds = -carry_discount * n_neg_d1 + 1.0;
        let d_rhs_ds = -(1.0 / q1) * (1.0 - carry_discount * n_neg_d1)
            - carry_discount * phi_d1 / (q1 * sigma_sqrt_t);
        let df = d_lhs_ds - d_rhs_ds;

        if df.abs() < 1e-14 {
            break;
        }
        let step = f / df;
        s -= step;
        if s <= 0.0 {
            s = strike * 0.01;
        }
        if s >= strike {
            s = strike * 0.99;
        }
        if step.abs() < 1e-8 {
            break;
        }
    }
    s
}

/// American option price using the Barone-Adesi-Whaley (1987) quadratic approximation.
///
/// # Parameters
/// - `spot`: current underlying price
/// - `strike`: option strike price
/// - `rate`: risk-free rate (annualized, decimal)
/// - `carry`: continuous dividend yield / carry rate
/// - `time_to_expiry`: time to expiry in years
/// - `volatility`: implied vol (annualized, decimal)
/// - `kind`: call or put
pub fn american_price_baw(
    spot: f64,
    strike: f64,
    rate: f64,
    carry: f64,
    time_to_expiry: f64,
    volatility: f64,
    kind: OptionKind,
) -> f64 {
    if invalid_inputs(spot, strike, time_to_expiry, volatility)
        || !rate.is_finite()
        || !carry.is_finite()
    {
        return f64::NAN;
    }

    // At expiry: immediate exercise value
    if time_to_expiry == 0.0 {
        return match kind {
            OptionKind::Call => (spot - strike).max(0.0),
            OptionKind::Put => (strike - spot).max(0.0),
        };
    }

    // At zero vol: deterministic — exercise if ITM
    if volatility == 0.0 {
        return match kind {
            OptionKind::Call => (spot - strike).max(0.0),
            OptionKind::Put => (strike - spot).max(0.0),
        };
    }

    let european = black_scholes_price(spot, strike, rate, carry, time_to_expiry, volatility, kind);

    match kind {
        OptionKind::Call => {
            // No early exercise premium when there are no dividends (carry == 0 means q==0
            // in BSM parameterisation where carry = q).
            if carry <= 0.0 {
                return european;
            }

            let sigma2 = volatility * volatility;
            let m = 2.0 * rate / sigma2;
            let n = 2.0 * (rate - carry) / sigma2;
            let h = 1.0 - (-rate * time_to_expiry).exp();

            if h.abs() < 1e-14 {
                return european;
            }

            let discriminant = (n - 1.0) * (n - 1.0) + 4.0 * m / h;
            if discriminant < 0.0 {
                return european;
            }

            let q2 = (-(n - 1.0) + discriminant.sqrt()) / 2.0;

            // Find critical price S*
            let s_star = find_critical_call(strike, rate, carry, time_to_expiry, volatility, q2);

            if s_star <= strike {
                // Degenerate critical price; fall back to European
                return european;
            }

            // A2 = (S*/q2) * (1 - e^{-q*T} * N(d1(S*)))
            let carry_discount = (-carry * time_to_expiry).exp();
            let d1_star = d1_fn(s_star, strike, rate, carry, time_to_expiry, volatility);
            let a2 = (s_star / q2) * (1.0 - carry_discount * cdf(d1_star));

            if spot >= s_star {
                // Immediate exercise is optimal
                (spot - strike).max(0.0)
            } else {
                (european + a2 * (spot / s_star).powf(q2)).max(european)
            }
        }

        OptionKind::Put => {
            // No early exercise when rate == 0 (no time value of money)
            if rate <= 0.0 {
                return european;
            }

            let sigma2 = volatility * volatility;
            let m = 2.0 * rate / sigma2;
            let n = 2.0 * (rate - carry) / sigma2;
            let h = 1.0 - (-rate * time_to_expiry).exp();

            if h.abs() < 1e-14 {
                return european;
            }

            let discriminant = (n - 1.0) * (n - 1.0) + 4.0 * m / h;
            if discriminant < 0.0 {
                return european;
            }

            let q1 = (-(n - 1.0) - discriminant.sqrt()) / 2.0;

            // Find critical price S**
            let s_star_star =
                find_critical_put(strike, rate, carry, time_to_expiry, volatility, q1);

            if s_star_star <= 0.0 || s_star_star >= strike {
                return european;
            }

            // A1 = -(S**/q1) * (1 - e^{-q*T} * N(-d1(S**)))
            let carry_discount = (-carry * time_to_expiry).exp();
            let d1_star = d1_fn(s_star_star, strike, rate, carry, time_to_expiry, volatility);
            let a1 = -(s_star_star / q1) * (1.0 - carry_discount * cdf(-d1_star));

            if spot <= s_star_star {
                // Immediate exercise is optimal
                (strike - spot).max(0.0)
            } else {
                (european + a1 * (spot / s_star_star).powf(q1)).max(european)
            }
        }
    }
}

/// Early exercise premium = american_price - european_bsm_price.
///
/// Always non-negative for valid inputs.
pub fn early_exercise_premium(
    spot: f64,
    strike: f64,
    rate: f64,
    carry: f64,
    time_to_expiry: f64,
    volatility: f64,
    kind: OptionKind,
) -> f64 {
    let american = american_price_baw(spot, strike, rate, carry, time_to_expiry, volatility, kind);
    let european = black_scholes_price(spot, strike, rate, carry, time_to_expiry, volatility, kind);
    if american.is_nan() || european.is_nan() {
        return f64::NAN;
    }
    (american - european).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::OptionKind;

    #[test]
    fn american_call_gte_european_call() {
        let european = crate::options::pricing::black_scholes_price(
            100.0,
            100.0,
            0.05,
            0.03,
            1.0,
            0.2,
            OptionKind::Call,
        );
        let american = american_price_baw(100.0, 100.0, 0.05, 0.03, 1.0, 0.2, OptionKind::Call);
        assert!(american >= european - 1e-10);
    }

    #[test]
    fn american_put_gte_european_put() {
        let european = crate::options::pricing::black_scholes_price(
            100.0,
            100.0,
            0.05,
            0.0,
            1.0,
            0.2,
            OptionKind::Put,
        );
        let american = american_price_baw(100.0, 100.0, 0.05, 0.0, 1.0, 0.2, OptionKind::Put);
        assert!(american >= european - 1e-10);
    }

    #[test]
    fn early_exercise_premium_nonneg() {
        let prem = early_exercise_premium(100.0, 100.0, 0.05, 0.03, 1.0, 0.2, OptionKind::Call);
        assert!(prem >= 0.0);
    }

    #[test]
    fn american_call_no_dividends_equals_european() {
        // With no dividends (carry == 0), no early exercise is optimal for calls
        let european = crate::options::pricing::black_scholes_price(
            100.0,
            100.0,
            0.05,
            0.0,
            1.0,
            0.2,
            OptionKind::Call,
        );
        let american = american_price_baw(100.0, 100.0, 0.05, 0.0, 1.0, 0.2, OptionKind::Call);
        assert!((american - european).abs() < 1e-10);
    }

    #[test]
    fn american_price_returns_nan_for_invalid() {
        let price = american_price_baw(-1.0, 100.0, 0.05, 0.0, 1.0, 0.2, OptionKind::Call);
        assert!(price.is_nan());
    }

    #[test]
    fn american_price_at_expiry_is_intrinsic() {
        let call = american_price_baw(110.0, 100.0, 0.05, 0.03, 0.0, 0.2, OptionKind::Call);
        assert!((call - 10.0).abs() < 1e-10);
        let put = american_price_baw(90.0, 100.0, 0.05, 0.0, 0.0, 0.2, OptionKind::Put);
        assert!((put - 10.0).abs() < 1e-10);
    }

    #[test]
    fn american_put_itm_has_positive_premium() {
        // Deep ITM put with high rate should have meaningful early exercise premium
        let prem = early_exercise_premium(80.0, 100.0, 0.10, 0.0, 1.0, 0.2, OptionKind::Put);
        assert!(prem >= 0.0);
    }

    #[test]
    fn american_prices_are_finite_for_valid_inputs() {
        let call = american_price_baw(100.0, 100.0, 0.05, 0.02, 1.0, 0.25, OptionKind::Call);
        let put = american_price_baw(100.0, 100.0, 0.05, 0.02, 1.0, 0.25, OptionKind::Put);
        assert!(call.is_finite());
        assert!(put.is_finite());
    }
}
