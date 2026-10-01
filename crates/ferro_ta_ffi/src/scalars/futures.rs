//! Scalar C exports for `ferro_ta_core::futures`.

use ferro_ta_core::futures;

crate::ffi_scalar_exports! {
    group: "futures",
    /// Futures basis: futures - spot.
    ft_basis(spot: float, future: float) -> [out_value: f64] = futures::basis::basis(spot, future);
    /// Annualized simple basis return.
    ft_annualized_basis(spot: float, future: float, time_to_expiry: float) -> [out_value: f64]
        = futures::basis::annualized_basis(spot, future, time_to_expiry);
    /// Implied continuously compounded carry rate.
    ft_implied_carry_rate(spot: float, future: float, time_to_expiry: float) -> [out_value: f64]
        = futures::basis::implied_carry_rate(spot, future, time_to_expiry);
    /// Carry spread relative to the risk-free rate.
    ft_carry_spread(spot: float, future: float, rate: float, time_to_expiry: float) -> [out_value: f64]
        = futures::basis::carry_spread(spot, future, rate, time_to_expiry);
    /// Annualized roll yield from front and next prices.
    ft_roll_yield(front_price: float, next_price: float, time_to_expiry: float) -> [out_value: f64]
        = futures::roll::roll_yield(front_price, next_price, time_to_expiry);
    /// Synthetic forward price from call/put parity.
    ft_synthetic_forward(call_price: float, put_price: float, strike: float, rate: float, time_to_expiry: float)
        -> [out_value: f64]
        = futures::synthetic::synthetic_forward(call_price, put_price, strike, rate, time_to_expiry);
    /// Synthetic spot price implied by call/put parity with continuous carry.
    ft_synthetic_spot(call_price: float, put_price: float, strike: float, rate: float, carry: float, time_to_expiry: float)
        -> [out_value: f64]
        = futures::synthetic::synthetic_spot(call_price, put_price, strike, rate, carry, time_to_expiry);
    /// Put-call parity residual. Zero means the inputs are parity-consistent.
    ft_parity_gap(call_price: float, put_price: float, spot: float, strike: float, rate: float, carry: float, time_to_expiry: float)
        -> [out_value: f64]
        = futures::synthetic::parity_gap(call_price, put_price, spot, strike, rate, carry, time_to_expiry);
}
