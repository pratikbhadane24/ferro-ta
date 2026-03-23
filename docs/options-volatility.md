# Options and Implied Volatility

`ferro-ta` exposes options analytics from `ferro_ta.analysis.options`.

## Scope

The module now covers both classic IV-series helpers and model-based option
analytics:

- `iv_rank`, `iv_percentile`, `iv_zscore`
- Black-Scholes-Merton pricing
- Black-76 pricing
- Delta, gamma, vega, theta, rho
- Implied volatility inversion
- Smile metrics and chain helpers

Heavy computation runs in Rust through the `_ferro_ta` extension.

## IV-series helpers

The original rolling helpers remain available and keep their public names:

```python
import numpy as np
from ferro_ta.analysis.options import iv_rank, iv_percentile, iv_zscore

iv = np.array([18.5, 22.3, 19.1, 25.0, 30.2, 27.8, 21.4, 19.0])
rank = iv_rank(iv, window=5)
pct = iv_percentile(iv, window=5)
z = iv_zscore(iv, window=5)
```

These helpers accept a 1-D IV series and return rolling statistics with
`NaN` during the warmup period.

## Pricing and Greeks

```python
from ferro_ta.analysis.options import greeks, implied_volatility, option_price

price = option_price(100.0, 100.0, 0.05, 1.0, 0.20, option_type="call")
iv = implied_volatility(price, 100.0, 100.0, 0.05, 1.0, option_type="call")
g = greeks(100.0, 100.0, 0.05, 1.0, 0.20, option_type="call")
```

Conventions:

- Volatility is decimal annualized volatility: `0.20` means 20%.
- Rates are decimal annualized rates: `0.05` means 5%.
- `time_to_expiry` is measured in years.
- `model="bsm"` uses spot as the underlying input.
- `model="black76"` uses forward as the underlying input.

## Smile and chain helpers

```python
from ferro_ta.analysis.options import label_moneyness, select_strike, smile_metrics

strikes = [80, 90, 100, 110, 120]
vols = [0.30, 0.25, 0.20, 0.22, 0.27]

metrics = smile_metrics(strikes, vols, 100.0, 0.5)
labels = label_moneyness(strikes, 100.0, option_type="call")
atm = select_strike(strikes, 100.0, selector="ATM")
delta_strike = select_strike(
    strikes,
    100.0,
    selector="DELTA0.25",
    option_type="call",
    volatilities=vols,
    time_to_expiry=0.5,
)
```

## Related futures analytics

See `ferro_ta.analysis.futures` and
[`docs/derivatives-analytics.md`](./derivatives-analytics.md) for synthetic
forwards, basis, carry, curve, and roll analytics.
