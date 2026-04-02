# Derivatives Analytics

`ferro-ta` ships a Rust-backed derivatives analytics layer focused on
research, simulation, and risk analysis.  All functions are implemented in
Rust core and exposed to Python (via PyO3) and WebAssembly (via wasm-bindgen).

---

## Modules

### `ferro_ta.analysis.options`

| Category | Functions |
|---|---|
| **Pricing** | `black_scholes_price`, `black_76_price`, `option_price` |
| **Greeks** | `greeks`, `extended_greeks` |
| **Implied vol** | `implied_volatility`, `iv_rank`, `iv_percentile`, `iv_zscore` |
| **Digital options** | `digital_option_price`, `digital_option_greeks` |
| **American options** | `american_option_price`, `early_exercise_premium` |
| **Smile / surface** | `smile_metrics`, `term_structure_slope`, `expected_move` |
| **Chain helpers** | `label_moneyness`, `select_strike` |
| **Realised vol** | `close_to_close_vol`, `parkinson_vol`, `garman_klass_vol`, `rogers_satchell_vol`, `yang_zhang_vol` |
| **Vol cone** | `vol_cone` |
| **Diagnostics** | `put_call_parity_deviation` |

### `ferro_ta.analysis.futures`

- Synthetic forwards and parity diagnostics
- Basis, annualized basis, implied carry, carry spread
- Continuous contract stitching: weighted, back-adjusted, ratio-adjusted
- Curve analytics: calendar spreads, slope, contango summary

### `ferro_ta.analysis.options_strategy`

Typed strategy schemas: expiry selectors, strike selectors, multi-leg presets
(`STRADDLE`, `STRANGLE`, `IRON_CONDOR`, `BULL_CALL_SPREAD`, `BEAR_PUT_SPREAD`),
risk controls, cost assumptions, and simulation limits.

### `ferro_ta.analysis.derivatives_payoff`

Multi-leg payoff and Greeks aggregation supporting **option**, **future**, and
**stock** instrument types.

| Function | Description |
|---|---|
| `option_leg_payoff` | Expiry P/L for a single option leg |
| `futures_leg_payoff` | Linear P/L for a futures leg |
| `stock_leg_payoff` | Linear P/L for a stock/equity leg |
| `strategy_payoff` | Aggregate expiry payoff across all legs |
| `strategy_value` | Pre-expiry BSM mid-price value of a multi-leg strategy |
| `aggregate_greeks` | Portfolio-level Greeks across option, futures, and stock legs |

---

## Model conventions

| Parameter | Convention |
|---|---|
| `model="bsm"` | Underlying is spot; `carry` = continuous dividend yield |
| `model="black76"` | Underlying is the forward price |
| `volatility` / `rate` / `carry` | Decimal annual (e.g. `0.20` = 20 %, `0.05` = 5 %) |
| `time_to_expiry` | Years (e.g. `0.25` = 3 months) |

---

## Quick examples

### BSM pricing and Greeks

```python
from ferro_ta.analysis.options import greeks, implied_volatility, option_price

price = option_price(100.0, 100.0, 0.05, 1.0, 0.20, option_type="call")
iv    = implied_volatility(price, 100.0, 100.0, 0.05, 1.0, option_type="call")
g     = greeks(100.0, 100.0, 0.05, 1.0, 0.20, option_type="call")
print(price, iv, g.delta, g.gamma)
```

### Extended (second-order) Greeks

```python
from ferro_ta.analysis.options import extended_greeks

eg = extended_greeks(100.0, 100.0, 0.05, 1.0, 0.20, option_type="call")
print(eg.vanna, eg.volga, eg.charm, eg.speed, eg.color)
```

### Digital options

```python
from ferro_ta.analysis.options import digital_option_price, digital_option_greeks

# Cash-or-nothing call at ATM ≈ e^{-rT} * N(d2) ≈ 0.53
price = digital_option_price(100.0, 100.0, 0.05, 1.0, 0.20,
                              option_type="call", digital_type="cash_or_nothing")
g = digital_option_greeks(100.0, 100.0, 0.05, 1.0, 0.20,
                           option_type="call", digital_type="cash_or_nothing")
print(price, g.delta, g.gamma, g.vega)
```

### American options (BAW approximation)

```python
from ferro_ta.analysis.options import american_option_price, early_exercise_premium

# American put — may have meaningful early exercise premium
american = american_option_price(100.0, 100.0, 0.05, 1.0, 0.20, option_type="put")
premium  = early_exercise_premium(100.0, 100.0, 0.05, 1.0, 0.20, option_type="put")
print(american, premium)
```

### Historical volatility estimators

```python
import numpy as np
from ferro_ta.analysis.options import (
    close_to_close_vol, garman_klass_vol, parkinson_vol,
    rogers_satchell_vol, yang_zhang_vol,
)

# Assume daily OHLC arrays of length N
open_p, high_p, low_p, close_p = ...  # numpy arrays

ctc = close_to_close_vol(close_p, window=20)           # close-only
park = parkinson_vol(high_p, low_p, window=20)         # high-low
gk   = garman_klass_vol(open_p, high_p, low_p, close_p, window=20)
rs   = rogers_satchell_vol(open_p, high_p, low_p, close_p, window=20)
yz   = yang_zhang_vol(open_p, high_p, low_p, close_p, window=20)
```

### Volatility cone

```python
from ferro_ta.analysis.options import vol_cone

cone = vol_cone(close_p, windows=(21, 42, 63, 126, 252))
# Overlay current IV against the cone to gauge richness/cheapness
for w, med in zip(cone.windows, cone.median):
    print(f"window={int(w):3d}  median_rv={med:.1%}")
```

### Put-call parity check

```python
from ferro_ta.analysis.options import option_price, put_call_parity_deviation

call = option_price(100.0, 100.0, 0.05, 1.0, 0.20, option_type="call")
put  = option_price(100.0, 100.0, 0.05, 1.0, 0.20, option_type="put")
dev  = put_call_parity_deviation(call, put, 100.0, 100.0, 0.05, 1.0)
# dev ≈ 0.0 for BSM-consistent prices; non-zero signals stale/mismatched quotes
```

### Expected move

```python
from ferro_ta.analysis.options import expected_move

lower, upper = expected_move(100.0, 0.20, days_to_expiry=30)
print(f"Expected ±1σ range: [{100+lower:.2f}, {100+upper:.2f}]")
```

### Multi-leg strategies with stock

```python
import numpy as np
from ferro_ta.analysis.derivatives_payoff import PayoffLeg, strategy_payoff, strategy_value

# Covered Call: long 100 shares + short 1 OTM call
spot_grid = np.linspace(80, 130, 100)
legs = [
    PayoffLeg("stock",  "long",  entry_price=100.0),
    PayoffLeg("option", "short", option_type="call",
              strike=110.0, premium=3.0, volatility=0.20, time_to_expiry=0.25),
]

# Expiry P/L
payoff = strategy_payoff(spot_grid, legs=legs)

# Pre-expiry BSM value (T=3 months remaining)
value = strategy_value(spot_grid, legs=legs, time_to_expiry=0.25, volatility=0.20)
```

### Futures analytics

```python
from ferro_ta.analysis.futures import basis, curve_summary

print(basis(100.0, 103.0))
print(curve_summary(100.0, [0.1, 0.5, 1.0], [101.0, 102.0, 104.0]))
```

---

## Instrument types in `PayoffLeg` / `StrategyLeg`

| `instrument` | Required fields | Payoff |
|---|---|---|
| `"option"` | `option_type`, `strike`, `expiry_selector`, `strike_selector` | `max(φ(S−K), 0) − premium` |
| `"future"` | `entry_price` | `S − entry_price` |
| `"stock"` | `entry_price` | `S − entry_price` (identical to future, no margin) |

---

## Volatility estimator efficiency comparison

| Estimator | Relative efficiency vs close-to-close | Handles overnight gaps |
|---|---|---|
| Close-to-close | 1× (baseline) | N/A (uses close only) |
| Parkinson | ~5× | No |
| Garman-Klass | ~7.4× | No |
| Rogers-Satchell | ~8× | No |
| Yang-Zhang | ~14× | Yes |

*Use Yang-Zhang when you have overnight gaps (futures, crypto). Use Parkinson
or Garman-Klass for continuous trading sessions.*

---

## Notes

- All existing function names (`iv_rank`, `iv_percentile`, `iv_zscore`, `greeks`,
  `option_price`, etc.) are preserved — fully backward compatible.
- The derivatives layer is analytics-only: no broker connectivity, order routing,
  or execution workflow.
- WASM: all functions in this layer are also exported as WebAssembly bindings
  (see `wasm/src/lib.rs`).
