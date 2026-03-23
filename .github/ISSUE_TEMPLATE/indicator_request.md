---
name: Indicator request
about: Request a new technical analysis indicator or a variant of an existing one
title: "[Indicator] "
labels: new-indicator
assignees: ''
---

## Indicator Name

<!-- The standard name, e.g. "Chande Momentum Oscillator (CMO)" or "Kaufman Adaptive MA (KAMA) variant" -->

## Category

<!-- Which category does this indicator belong to? -->
- [ ] Overlap Studies (moving averages, bands)
- [ ] Momentum Indicators
- [ ] Volatility Indicators
- [ ] Volume Indicators
- [ ] Price Transform
- [ ] Statistic Functions
- [ ] Cycle Indicators
- [ ] Extended / Multi-output Indicators
- [ ] Other: ___

## Reference / Formula

<!-- Link to the authoritative reference (book, paper, website) and/or the mathematical formula. -->

**Formula:**
```
# e.g.
# CMO = 100 × (SumUp − SumDown) / (SumUp + SumDown)
```

**Reference:** <!-- URL or book citation -->

## TA-Lib equivalent

<!-- If this indicator exists in TA-Lib, what is the function name? -->
- [ ] This indicator is in TA-Lib as: `TALIB_FUNCTION_NAME`
- [ ] This indicator is NOT in TA-Lib (extension indicator)

## Expected API

<!-- How should the function look in ferro_ta? -->
```python
from ferro_ta import MY_INDICATOR
import numpy as np

close = np.array([...])
result = MY_INDICATOR(close, timeperiod=14)
# result: np.ndarray of float64, same length as close
```

## Use Case

<!-- Why do you need this indicator? What trading strategy or analysis does it support? -->

## Priority / Urgency

- [ ] Nice to have
- [ ] Would significantly improve my workflow
- [ ] Blocking me from using ferro_ta

## Willingness to Contribute

- [ ] I'd like to implement this myself (see `CONTRIBUTING.md`)
- [ ] I can help test / validate the implementation
- [ ] I just want to request it — up to the maintainers
