//! Basis and carry analytics.

/// Futures basis: futures - spot.
pub fn basis(spot: f64, future: f64) -> f64 {
    if !spot.is_finite() || !future.is_finite() {
        f64::NAN
    } else {
        future - spot
    }
}

/// Annualized simple basis return.
pub fn annualized_basis(spot: f64, future: f64, time_to_expiry: f64) -> f64 {
    if !spot.is_finite()
        || !future.is_finite()
        || !time_to_expiry.is_finite()
        || spot <= 0.0
        || time_to_expiry <= 0.0
    {
        return f64::NAN;
    }
    (future / spot - 1.0) / time_to_expiry
}

/// Implied continuously compounded carry rate.
pub fn implied_carry_rate(spot: f64, future: f64, time_to_expiry: f64) -> f64 {
    if !spot.is_finite()
        || !future.is_finite()
        || !time_to_expiry.is_finite()
        || spot <= 0.0
        || future <= 0.0
        || time_to_expiry <= 0.0
    {
        return f64::NAN;
    }
    (future / spot).ln() / time_to_expiry
}

/// Carry spread relative to the risk-free rate.
pub fn carry_spread(spot: f64, future: f64, rate: f64, time_to_expiry: f64) -> f64 {
    implied_carry_rate(spot, future, time_to_expiry) - rate
}

#[cfg(test)]
mod tests {
    use super::{annualized_basis, basis, carry_spread, implied_carry_rate};

    #[test]
    fn basis_helpers_work() {
        assert_eq!(basis(100.0, 103.0), 3.0);
        assert!(annualized_basis(100.0, 103.0, 0.25) > 0.0);
        assert!(implied_carry_rate(100.0, 103.0, 0.25) > 0.0);
        assert!(carry_spread(100.0, 103.0, 0.02, 0.25).is_finite());
    }
}
