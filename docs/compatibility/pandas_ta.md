# Compatibility: ferro-ta vs pandas-ta

ferro-ta provides indicators that match [pandas-ta](https://github.com/twopirllc/pandas-ta)
results to within numerical tolerance. This guide explains how to migrate from
pandas-ta and how to run the cross-library validation tests.

## Installation

```bash
pip install ferro-ta
# Optional: install pandas-ta to run comparison tests
pip install pandas-ta
```

## API Comparison

### pandas-ta style (accessor)

```python
import pandas as pd
import pandas_ta as ta

close = pd.Series([...])
sma = close.ta.sma(length=20)
ema = close.ta.ema(length=14)
rsi = close.ta.rsi(length=14)
```

### ferro-ta equivalent

```python
import numpy as np
import ferro_ta as ft

close = np.array([...])
sma = ft.SMA(close, timeperiod=20)
ema = ft.EMA(close, timeperiod=14)
rsi = ft.RSI(close, timeperiod=14)
```

> **Note**: ferro-ta operates on NumPy arrays. If you have a `pd.Series`, pass
> it directly — ferro-ta will convert it automatically.

## Indicator Mapping

| pandas-ta | ferro-ta | Notes |
|---|---|---|
| `ta.sma(length=N)` | `ft.SMA(close, timeperiod=N)` | Exact match |
| `ta.ema(length=N)` | `ft.EMA(close, timeperiod=N)` | Tail convergence within 1e-6 |
| `ta.wma(length=N)` | `ft.WMA(close, timeperiod=N)` | Exact match |
| `ta.rsi(length=N)` | `ft.RSI(close, timeperiod=N)` | Tail convergence |
| `ta.macd(fast, slow, signal)` | `ft.MACD(close, fastperiod, slowperiod, signalperiod)` | Tail convergence |
| `ta.bbands(length=N, std=2)` | `ft.BBANDS(close, timeperiod=N, nbdevup=2, nbdevdn=2)` | Exact match |
| `ta.stoch(high, low, close)` | `ft.STOCH(high, low, close, ...)` | Tail convergence |
| `ta.cci(high, low, close, length=N)` | `ft.CCI(high, low, close, timeperiod=N)` | Exact match |
| `ta.mom(length=N)` | `ft.MOM(close, timeperiod=N)` | Exact match |
| `ta.roc(length=N)` | `ft.ROC(close, timeperiod=N)` | Exact match |
| `ta.trima(length=N)` | `ft.TRIMA(close, timeperiod=N)` | Exact match |
| `ta.hma(length=N)` | `ft.HT_MA(close, timeperiod=N)` | Hull MA variant |
| `ta.ichimoku(...)` | `ft.ICHIMOKU(high, low, close)` | Tenkan/Kijun match |
| `ta.kc(high, low, close, ...)` | `ft.KELTNER(high, low, close, ...)` | Tail convergence |

## Batch Execution

ferro-ta supports running many indicators at once via the batch API:

```python
import numpy as np
import ferro_ta as ft

data = np.random.randn(1000, 50)  # 50 instruments × 1000 bars

# Run SMA(20) across all 50 instruments in one call
results = ft.batch_compute(data, "SMA", timeperiod=20)
```

## Running the Cross-Library Tests

Cross-library comparison tests live in `tests/integration/test_vs_pandas_ta.py`.
They are automatically **skipped** when pandas-ta is not installed.

```bash
# Install pandas-ta first
pip install pandas-ta

# Run comparison tests
pytest tests/integration/test_vs_pandas_ta.py -v
```

## Known Differences

- **Seeding period**: EMA results during the first `timeperiod` bars may differ
  due to different initialization strategies (SMA seed vs EMA seed). Results
  converge after the seeding window.
- **MACD signal line**: The signal EMA is seeded from the first valid MACD value.
  Exact match begins after 2× `slowperiod` bars.
- **STOCH smoothing**: ferro-ta defaults match TA-Lib (SMA slowk, SMA slowd).
  pandas-ta uses different defaults; pass matching parameters explicitly.

## Performance Comparison

ferro-ta is 10–100× faster than pandas-ta for large arrays because the core
computation is written in Rust:

```bash
# Run the benchmark
pytest benchmarks/test_speed.py --benchmark-only --benchmark-json=benchmarks/results.json
```
