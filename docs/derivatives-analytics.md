# Derivatives Analytics

`ferro-ta` now includes a Rust-backed derivatives analytics layer focused on
research, simulation, and risk analysis.

## Modules

- `ferro_ta.analysis.options`
  - Black-Scholes-Merton and Black-76 pricing
  - Delta, gamma, vega, theta, rho
  - Implied volatility inversion with guarded Newton + bisection fallback
  - IV rank / percentile / z-score
  - Smile metrics: ATM IV, 25-delta risk reversal, butterfly, skew slope, convexity
  - Chain helpers: moneyness labels and strike selection by offset or delta
- `ferro_ta.analysis.futures`
  - Synthetic forwards and parity diagnostics
  - Basis, annualized basis, implied carry, carry spread
  - Continuous contract stitching: weighted, back-adjusted, ratio-adjusted
  - Curve analytics: calendar spreads, slope, contango summary
- `ferro_ta.analysis.options_strategy`
  - Typed strategy schemas for expiry selectors, strike selectors, multi-leg presets,
    risk controls, cost assumptions, and simulation limits
- `ferro_ta.analysis.derivatives_payoff`
  - Multi-leg payoff aggregation
  - Portfolio-level Greeks aggregation across option and futures legs

## Model conventions

- `model="bsm"` expects the underlying input to be spot and `carry` to represent
  a continuous dividend yield or generic carry term.
- `model="black76"` expects the underlying input to be the forward price.
- Volatility and rates use decimal units:
  - `0.20` means 20% annualized volatility
  - `0.05` means 5% annualized rate
- `time_to_expiry` is expressed in years.

## Quick examples

```python
from ferro_ta.analysis.options import greeks, implied_volatility, option_price

price = option_price(100.0, 100.0, 0.05, 1.0, 0.20, option_type="call")
iv = implied_volatility(price, 100.0, 100.0, 0.05, 1.0, option_type="call")
g = greeks(100.0, 100.0, 0.05, 1.0, 0.20, option_type="call")
print(price, iv, g.delta)
```

```python
from ferro_ta.analysis.futures import basis, curve_summary

print(basis(100.0, 103.0))
print(curve_summary(100.0, [0.1, 0.5, 1.0], [101.0, 102.0, 104.0]))
```

```python
from ferro_ta.analysis.derivatives_payoff import PayoffLeg, strategy_payoff

legs = [
    PayoffLeg("option", "long", option_type="call", strike=100.0, premium=5.0),
    PayoffLeg("future", "long", entry_price=100.0),
]
grid = [90.0, 100.0, 110.0]
print(strategy_payoff(grid, legs=legs))
```

## Notes

- Existing `iv_rank`, `iv_percentile`, and `iv_zscore` names are preserved.
- The derivatives layer is analytics-only: there is no broker connectivity,
  order routing, or execution workflow in this API.
