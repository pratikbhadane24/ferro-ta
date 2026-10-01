//! Scalar C exports for `ferro_ta_core::options`.

use ferro_ta_core::options::{self, IvSolverConfig, OptionContract};

crate::ffi_scalar_exports! {
    group: "options",
    /// Black-Scholes-Merton price with continuous carry/dividend yield.
    ft_black_scholes_price(spot: float, strike: float, rate: float, dividend_yield: float, time_to_expiry: float, volatility: float, kind: option_kind)
        -> [out_value: f64]
        = options::pricing::black_scholes_price(spot, strike, rate, dividend_yield, time_to_expiry, volatility, kind);
    /// Black-76 price using the forward price as the underlying input.
    ft_black_76_price(forward: float, strike: float, rate: float, time_to_expiry: float, volatility: float, kind: option_kind)
        -> [out_value: f64]
        = options::pricing::black_76_price(forward, strike, rate, time_to_expiry, volatility, kind);
    /// Black-Scholes-Merton Greeks.
    ft_black_scholes_greeks(spot: float, strike: float, rate: float, dividend_yield: float, time_to_expiry: float, volatility: float, kind: option_kind)
        -> [out_delta: f64, out_gamma: f64, out_vega: f64, out_theta: f64, out_rho: f64]
        = {
            let g = options::greeks::black_scholes_greeks(spot, strike, rate, dividend_yield, time_to_expiry, volatility, kind);
            (g.delta, g.gamma, g.vega, g.theta, g.rho)
        };
    /// Black-76 Greeks with respect to the forward.
    ft_black_76_greeks(forward: float, strike: float, rate: float, time_to_expiry: float, volatility: float, kind: option_kind)
        -> [out_delta: f64, out_gamma: f64, out_vega: f64, out_theta: f64, out_rho: f64]
        = {
            let g = options::greeks::black_76_greeks(forward, strike, rate, time_to_expiry, volatility, kind);
            (g.delta, g.gamma, g.vega, g.theta, g.rho)
        };
    /// Extended Greeks under Black-Scholes-Merton (closed-form).
    ft_black_scholes_extended_greeks(spot: float, strike: float, rate: float, dividend_yield: float, time_to_expiry: float, volatility: float, kind: option_kind)
        -> [out_vanna: f64, out_volga: f64, out_charm: f64, out_speed: f64, out_color: f64]
        = {
            let g = options::greeks::black_scholes_extended_greeks(spot, strike, rate, dividend_yield, time_to_expiry, volatility, kind);
            (g.vanna, g.volga, g.charm, g.speed, g.color)
        };
    /// American option price using the Barone-Adesi-Whaley (1987) quadratic approximation.
    ft_american_price_baw(spot: float, strike: float, rate: float, carry: float, time_to_expiry: float, volatility: float, kind: option_kind)
        -> [out_value: f64]
        = options::american::american_price_baw(spot, strike, rate, carry, time_to_expiry, volatility, kind);
    /// Early exercise premium = american_price - european_bsm_price.
    ft_early_exercise_premium(spot: float, strike: float, rate: float, carry: float, time_to_expiry: float, volatility: float, kind: option_kind)
        -> [out_value: f64]
        = options::american::early_exercise_premium(spot, strike, rate, carry, time_to_expiry, volatility, kind);
    /// Price a digital (binary) option under BSM.
    ft_digital_price(spot: float, strike: float, rate: float, carry: float, time_to_expiry: float, volatility: float, option_kind: option_kind, digital_kind: digital_kind)
        -> [out_value: f64]
        = options::digital::digital_price(spot, strike, rate, carry, time_to_expiry, volatility, option_kind, digital_kind);
    /// Compute numerical delta, gamma, and vega for a digital option.
    ft_digital_greeks(spot: float, strike: float, rate: float, carry: float, time_to_expiry: float, volatility: float, option_kind: option_kind, digital_kind: digital_kind)
        -> [out_delta: f64, out_gamma: f64, out_vega: f64]
        = options::digital::digital_greeks(spot, strike, rate, carry, time_to_expiry, volatility, option_kind, digital_kind);
    /// Put-call parity deviation: `C - P - (S·e^{-q·T} - K·e^{-r·T})`.
    ft_put_call_parity_deviation(call_price: float, put_price: float, spot: float, strike: float, rate: float, carry: float, time_to_expiry: float)
        -> [out_value: f64]
        = options::pricing::put_call_parity_deviation(call_price, put_price, spot, strike, rate, carry, time_to_expiry);
    /// Expected ±1σ move over `days_to_expiry` calendar days.
    ft_expected_move(spot: float, iv: float, days_to_expiry: float, trading_days_per_year: float)
        -> [out_lower: f64, out_upper: f64]
        = options::surface::expected_move(spot, iv, days_to_expiry, trading_days_per_year);
    /// Solve implied volatility with guarded Newton iterations and bisection fallback.
    /// The result is NaN (with FT_OK) when no volatility reproduces `target_price`
    /// (outside the no-arbitrage bounds) or the solver does not converge.
    ft_implied_volatility(target_price: float, model: pricing_model, underlying: float, strike: float, rate: float, carry: float, time_to_expiry: float, kind: option_kind, initial_guess: float, tolerance: float, max_iterations: count)
        -> [out_value: f64]
        = options::iv::implied_volatility(
            OptionContract { model, underlying, strike, rate, carry, time_to_expiry, kind },
            target_price,
            IvSolverConfig { initial_guess, tolerance, max_iterations },
        );
    /// Lower no-arbitrage bound for the option price.
    ft_price_lower_bound(model: pricing_model, underlying: float, strike: float, rate: float, carry: float, time_to_expiry: float, kind: option_kind)
        -> [out_value: f64]
        = options::pricing::price_lower_bound(OptionContract { model, underlying, strike, rate, carry, time_to_expiry, kind });
    /// Upper no-arbitrage bound for the option price.
    ft_price_upper_bound(model: pricing_model, underlying: float, strike: float, rate: float, carry: float, time_to_expiry: float, kind: option_kind)
        -> [out_value: f64]
        = options::pricing::price_upper_bound(OptionContract { model, underlying, strike, rate, carry, time_to_expiry, kind });
}
