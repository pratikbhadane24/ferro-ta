//! Digital (binary) option pricing.

use super::normal::cdf;
use super::OptionKind;

/// Type of digital option payoff.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DigitalKind {
    /// Pays 1 unit of cash if option expires in the money.
    CashOrNothing,
    /// Pays the underlying asset if option expires in the money.
    AssetOrNothing,
}

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

/// Price a digital (binary) option under BSM.
///
/// # Parameters
/// - `spot`: current underlying price
/// - `strike`: option strike price
/// - `rate`: risk-free rate (annualized, decimal)
/// - `carry`: continuous dividend yield / carry rate
/// - `time_to_expiry`: time to expiry in years
/// - `volatility`: implied vol (annualized, decimal)
/// - `option_kind`: call or put
/// - `digital_kind`: cash-or-nothing or asset-or-nothing
#[allow(clippy::too_many_arguments)]
pub fn digital_price(
    spot: f64,
    strike: f64,
    rate: f64,
    carry: f64,
    time_to_expiry: f64,
    volatility: f64,
    option_kind: OptionKind,
    digital_kind: DigitalKind,
) -> f64 {
    if invalid_inputs(spot, strike, time_to_expiry, volatility)
        || !rate.is_finite()
        || !carry.is_finite()
    {
        return f64::NAN;
    }

    // At expiry: pay intrinsic based on ITM status
    if time_to_expiry == 0.0 {
        let itm = match option_kind {
            OptionKind::Call => spot > strike,
            OptionKind::Put => spot < strike,
        };
        return if itm {
            match digital_kind {
                DigitalKind::CashOrNothing => 1.0,
                DigitalKind::AssetOrNothing => spot,
            }
        } else {
            0.0
        };
    }

    let discount = (-rate * time_to_expiry).exp();
    let carry_discount = (-carry * time_to_expiry).exp();

    // At zero vol: deterministic payoff
    if volatility == 0.0 {
        let forward = spot * (carry_discount / discount); // S * e^{(r-q)*T} equivalent: S*e^{-q*T}/e^{-r*T}
                                                          // forward = S * e^{(r-q)*T}; ITM if forward > K for call
        let itm = match option_kind {
            OptionKind::Call => spot * carry_discount > strike * discount,
            OptionKind::Put => spot * carry_discount < strike * discount,
        };
        let _ = forward; // suppress unused warning
        return if itm {
            match digital_kind {
                DigitalKind::CashOrNothing => discount,
                DigitalKind::AssetOrNothing => spot * carry_discount,
            }
        } else {
            0.0
        };
    }

    let sqrt_t = time_to_expiry.sqrt();
    let sigma_sqrt_t = volatility * sqrt_t;
    let d1 = ((spot / strike).ln()
        + (rate - carry + 0.5 * volatility * volatility) * time_to_expiry)
        / sigma_sqrt_t;
    let d2 = d1 - sigma_sqrt_t;

    match digital_kind {
        DigitalKind::CashOrNothing => match option_kind {
            OptionKind::Call => discount * cdf(d2),
            OptionKind::Put => discount * cdf(-d2),
        },
        DigitalKind::AssetOrNothing => match option_kind {
            OptionKind::Call => spot * carry_discount * cdf(d1),
            OptionKind::Put => spot * carry_discount * cdf(-d1),
        },
    }
}

/// Compute numerical delta, gamma, and vega for a digital option.
///
/// Uses central finite differences:
/// - delta/gamma: bump spot by ε = spot * 1e-3
/// - vega: bump volatility by 1e-3
///
/// Returns `(delta, gamma, vega)`.
#[allow(clippy::too_many_arguments)]
pub fn digital_greeks(
    spot: f64,
    strike: f64,
    rate: f64,
    carry: f64,
    time_to_expiry: f64,
    volatility: f64,
    option_kind: OptionKind,
    digital_kind: DigitalKind,
) -> (f64, f64, f64) {
    let eps = spot * 1e-3;
    if eps <= 0.0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }

    let price_mid = digital_price(
        spot,
        strike,
        rate,
        carry,
        time_to_expiry,
        volatility,
        option_kind,
        digital_kind,
    );
    let price_up = digital_price(
        spot + eps,
        strike,
        rate,
        carry,
        time_to_expiry,
        volatility,
        option_kind,
        digital_kind,
    );
    let price_dn = digital_price(
        spot - eps,
        strike,
        rate,
        carry,
        time_to_expiry,
        volatility,
        option_kind,
        digital_kind,
    );

    let delta = (price_up - price_dn) / (2.0 * eps);
    let gamma = (price_up - 2.0 * price_mid + price_dn) / (eps * eps);

    let vol_bump = 1e-3;
    let vega = if volatility + vol_bump > 0.0 && volatility - vol_bump > 0.0 {
        let price_vup = digital_price(
            spot,
            strike,
            rate,
            carry,
            time_to_expiry,
            volatility + vol_bump,
            option_kind,
            digital_kind,
        );
        let price_vdn = digital_price(
            spot,
            strike,
            rate,
            carry,
            time_to_expiry,
            volatility - vol_bump,
            option_kind,
            digital_kind,
        );
        (price_vup - price_vdn) / (2.0 * vol_bump)
    } else {
        // vol too close to zero; one-sided bump
        let price_vup = digital_price(
            spot,
            strike,
            rate,
            carry,
            time_to_expiry,
            volatility + vol_bump,
            option_kind,
            digital_kind,
        );
        (price_vup - price_mid) / vol_bump
    };

    (delta, gamma, vega)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::OptionKind;

    #[test]
    fn cash_or_nothing_call_atm() {
        // ATM cash-or-nothing call: price = e^{-rT} * N(d2)
        // At S=K=100, r=0.05, q=0, T=1, σ=0.2:
        //   d1 = (0 + 0.07) / 0.2 = 0.35,  d2 = 0.15  →  N(0.15) ≈ 0.5596
        //   price ≈ e^{-0.05} * 0.5596 ≈ 0.532
        let price = digital_price(
            100.0,
            100.0,
            0.05,
            0.0,
            1.0,
            0.2,
            OptionKind::Call,
            DigitalKind::CashOrNothing,
        );
        assert!(
            price > 0.0 && price < 1.0,
            "price should be between 0 and 1"
        );
        assert!((price - 0.532).abs() < 0.01, "price ≈ 0.532, got {price}");
    }

    #[test]
    fn asset_or_nothing_call_at_zero_vol() {
        // At zero vol, ITM asset-or-nothing call should equal S * e^{-q*T}
        let price = digital_price(
            110.0,
            100.0,
            0.05,
            0.0,
            1.0,
            0.0,
            OptionKind::Call,
            DigitalKind::AssetOrNothing,
        );
        assert!((price - 110.0).abs() < 1e-6);
    }

    #[test]
    fn digital_price_returns_nan_for_invalid() {
        let price = digital_price(
            -1.0,
            100.0,
            0.05,
            0.0,
            1.0,
            0.2,
            OptionKind::Call,
            DigitalKind::CashOrNothing,
        );
        assert!(price.is_nan());
    }

    #[test]
    fn cash_or_nothing_put_call_parity() {
        // Cash-or-nothing call + cash-or-nothing put = e^{-rT}
        let call = digital_price(
            100.0,
            100.0,
            0.05,
            0.02,
            1.0,
            0.25,
            OptionKind::Call,
            DigitalKind::CashOrNothing,
        );
        let put = digital_price(
            100.0,
            100.0,
            0.05,
            0.02,
            1.0,
            0.25,
            OptionKind::Put,
            DigitalKind::CashOrNothing,
        );
        let discount = (-0.05_f64).exp();
        assert!((call + put - discount).abs() < 1e-10);
    }

    #[test]
    fn asset_or_nothing_put_call_parity() {
        // Asset-or-nothing call + asset-or-nothing put = S * e^{-q*T}
        let s = 100.0_f64;
        let q = 0.02_f64;
        let call = digital_price(
            s,
            100.0,
            0.05,
            q,
            1.0,
            0.25,
            OptionKind::Call,
            DigitalKind::AssetOrNothing,
        );
        let put = digital_price(
            s,
            100.0,
            0.05,
            q,
            1.0,
            0.25,
            OptionKind::Put,
            DigitalKind::AssetOrNothing,
        );
        let expected = s * (-q).exp();
        assert!((call + put - expected).abs() < 1e-10);
    }

    #[test]
    fn digital_greeks_are_finite_for_valid_inputs() {
        let (delta, gamma, vega) = digital_greeks(
            100.0,
            100.0,
            0.05,
            0.0,
            1.0,
            0.2,
            OptionKind::Call,
            DigitalKind::CashOrNothing,
        );
        assert!(delta.is_finite());
        assert!(gamma.is_finite());
        assert!(vega.is_finite());
    }

    #[test]
    fn digital_at_expiry_itm_returns_intrinsic() {
        let price = digital_price(
            110.0,
            100.0,
            0.05,
            0.0,
            0.0,
            0.2,
            OptionKind::Call,
            DigitalKind::CashOrNothing,
        );
        assert!((price - 1.0).abs() < 1e-10);
        let price2 = digital_price(
            110.0,
            100.0,
            0.05,
            0.0,
            0.0,
            0.2,
            OptionKind::Call,
            DigitalKind::AssetOrNothing,
        );
        assert!((price2 - 110.0).abs() < 1e-10);
    }

    #[test]
    fn digital_at_expiry_otm_returns_zero() {
        let price = digital_price(
            90.0,
            100.0,
            0.05,
            0.0,
            0.0,
            0.2,
            OptionKind::Call,
            DigitalKind::CashOrNothing,
        );
        assert!((price - 0.0).abs() < 1e-10);
    }
}
