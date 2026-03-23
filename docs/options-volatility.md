# Options and Implied Volatility

ferro-ta provides optional helpers for implied volatility (IV) analysis
via the `ferro_ta.options` module.  This document describes the scope,
data format, dependency strategy, and limitations.

---

## Scope

The `ferro_ta.options` module focuses on **IV series analysis**:

- **IV rank** — where today's IV sits relative to the min/max over a look-back window.
- **IV percentile** — fraction of observations over a look-back window at or below today's IV.
- **IV z-score** — how many standard deviations today's IV is above the rolling mean.

These functions accept any 1-D IV series (e.g. VIX daily closes, single-name
30-day IV, etc.) and return rolling statistics.

**Out of scope (for now):** Black-Scholes pricing, Greeks, option chain
parsing, synthetic forward construction, dividend adjustment.  For full
option-pricing functionality consider `py_vollib`, `mibian`, or similar.

---

## Data format

All functions accept a 1-D NumPy array (or any array-like) of IV values.
IV values are typically in **percentage points** (e.g. VIX = 20 means 20%
annualised volatility), but the helpers are unit-agnostic — they only
compare values within the rolling window.

```python
import numpy as np
from ferro_ta.options import iv_rank, iv_percentile, iv_zscore

# VIX-like daily close series
iv = np.array([18.5, 22.3, 19.1, 25.0, 30.2, 27.8, 21.4, 19.0])

rank = iv_rank(iv, window=5)        # rolling IV rank in [0, 1]
pct  = iv_percentile(iv, window=5)  # rolling IV percentile in [0, 1]
z    = iv_zscore(iv, window=5)      # rolling z-score
```

---

## Dependency strategy

The `ferro_ta.options` module uses **only NumPy** (already a core dependency).
No additional packages are required for the helpers described here.

For advanced option analytics (Black-Scholes, volatility surface
interpolation), install the optional extra:

```bash
pip install "ferro-ta[options]"
```

This may install additional packages in the future (e.g. `py_vollib`).

---

## API reference

### `iv_rank(iv_series, window=252)`

Rolling IV rank.

```
rank_t = (IV_t - min(IV[t-window+1:t+1])) / (max(IV[t-window+1:t+1]) - min(IV[t-window+1:t+1]))
```

Returns values in [0, 1].  NaN for the first `window - 1` bars.

### `iv_percentile(iv_series, window=252)`

Rolling IV percentile: fraction of the *window* bars whose IV was at or
below the current value.

### `iv_zscore(iv_series, window=252)`

Rolling z-score: `(IV_t - rolling_mean) / rolling_std`.

---

## Limitations

- All functions use **O(n × window)** time complexity (pure Python loops).
  For large windows or series consider vectorised alternatives.
- No option chain support; the module assumes IV series as input.
- Streaming (bar-by-bar) versions of these functions are not yet
  implemented.  For live use, maintain a rolling buffer and call the
  functions on the buffer at each bar.

---

## See also

- `ferro_ta.options` — module source.
- `ferro_ta.statistic` — general statistical functions (STDDEV, VAR, CORREL, etc.).
- `ferro_ta.volatility` — price-based volatility indicators (ATR, NATR).
