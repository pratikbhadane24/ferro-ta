//! Synthetic futures helpers built from put-call parity.

/// Synthetic forward price from call/put parity.
pub fn synthetic_forward(
    call_price: f64,
    put_price: f64,
    strike: f64,
    rate: f64,
    time_to_expiry: f64,
) -> f64 {
    if !call_price.is_finite()
        || !put_price.is_finite()
        || !strike.is_finite()
        || !rate.is_finite()
        || !time_to_expiry.is_finite()
        || strike <= 0.0
        || time_to_expiry < 0.0
    {
        return f64::NAN;
    }
    (call_price - put_price) * (rate * time_to_expiry).exp() + strike
}

/// Synthetic spot price implied by call/put parity with continuous carry.
pub fn synthetic_spot(
    call_price: f64,
    put_price: f64,
    strike: f64,
    rate: f64,
    carry: f64,
    time_to_expiry: f64,
) -> f64 {
    if !call_price.is_finite()
        || !put_price.is_finite()
        || !strike.is_finite()
        || !rate.is_finite()
        || !carry.is_finite()
        || !time_to_expiry.is_finite()
        || strike <= 0.0
        || time_to_expiry < 0.0
    {
        return f64::NAN;
    }
    (call_price - put_price + strike * (-rate * time_to_expiry).exp())
        * (carry * time_to_expiry).exp()
}

/// Put-call parity residual. Zero means the inputs are parity-consistent.
pub fn parity_gap(
    call_price: f64,
    put_price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    carry: f64,
    time_to_expiry: f64,
) -> f64 {
    call_price
        - put_price
        - (spot * (-carry * time_to_expiry).exp() - strike * (-rate * time_to_expiry).exp())
}

#[cfg(test)]
mod tests {
    use super::{parity_gap, synthetic_forward};

    #[test]
    fn synthetic_forward_is_consistent() {
        let forward = synthetic_forward(8.0, 5.0, 100.0, 0.02, 0.5);
        assert!(forward > 100.0);
    }

    #[test]
    fn parity_gap_zero_when_consistent() {
        let gap = parity_gap(10.45, 5.57, 100.0, 100.0, 0.05, 0.0, 1.0);
        assert!(gap.abs() < 0.05);
    }
}
