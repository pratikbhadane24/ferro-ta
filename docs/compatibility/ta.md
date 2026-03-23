# Compatibility: ferro-ta vs ta (Bukosabino)

ferro-ta provides indicators that match [ta](https://github.com/bukosabino/ta)
(Bukosabino's library) results to within numerical tolerance. This guide
explains how to migrate from `ta` and how to run the cross-library validation
tests.

## Installation

```bash
pip install ferro-ta
# Optional: install ta to run comparison tests
pip install ta
```

## API Comparison

### ta style

```python
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import SMAIndicator, EMAIndicator, MACD, CCIIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.others import DailyReturnIndicator

close = pd.Series([...])
high = pd.Series([...])
low = pd.Series([...])
volume = pd.Series([...])

rsi = RSIIndicator(close, window=14).rsi()
sma = SMAIndicator(close, window=20).sma_indicator()
ema = EMAIndicator(close, window=14).ema_indicator()
```

### ferro-ta equivalent

```python
import numpy as np
import ferro_ta as ft

close = np.array([...])
high = np.array([...])
low = np.array([...])
volume = np.array([...])

rsi = ft.RSI(close, timeperiod=14)
sma = ft.SMA(close, timeperiod=20)
ema = ft.EMA(close, timeperiod=14)
```

> **Note**: ferro-ta operates on NumPy arrays. If you have a `pd.Series`, pass
> it directly — ferro-ta will convert it automatically.

## Indicator Mapping

| ta | ferro-ta | Notes |
|---|---|---|
| `SMAIndicator(close, window=N).sma_indicator()` | `ft.SMA(close, timeperiod=N)` | Exact match |
| `EMAIndicator(close, window=N).ema_indicator()` | `ft.EMA(close, timeperiod=N)` | Tail convergence |
| `BollingerBands(close, window=N, window_dev=2)` | `ft.BBANDS(close, timeperiod=N, nbdevup=2, nbdevdn=2)` | Exact match |
| `RSIIndicator(close, window=N).rsi()` | `ft.RSI(close, timeperiod=N)` | Tail convergence |
| `MACD(close, window_slow, window_fast, window_sign)` | `ft.MACD(close, fastperiod, slowperiod, signalperiod)` | Tail convergence |
| `StochasticOscillator(high, low, close, window, smooth_window)` | `ft.STOCH(high, low, close, ...)` | Tail convergence |
| `AverageTrueRange(high, low, close, window=N)` | `ft.ATR(high, low, close, timeperiod=N)` | Tail convergence |
| `WilliamsRIndicator(high, low, close, lbp=N)` | `ft.WILLR(high, low, close, timeperiod=N)` | Exact match |
| `OnBalanceVolumeIndicator(close, volume)` | `ft.OBV(close, volume)` | Exact match |
| `CCIIndicator(high, low, close, window=N)` | `ft.CCI(high, low, close, timeperiod=N)` | Exact match |

## Running the Cross-Library Tests

Cross-library comparison tests live in `tests/integration/test_vs_ta.py`.
They are automatically **skipped** when `ta` is not installed.

```bash
# Install ta first
pip install ta

# Run comparison tests
pytest tests/integration/test_vs_ta.py -v
```

## Known Differences

- **EMA seeding**: `ta` uses pandas `ewm` with `adjust=True` by default, which
  produces different warm-up values. Results converge after `2 × timeperiod` bars.
- **ATR**: `ta` uses a simple rolling mean for ATR by default; ferro-ta uses
  Wilder's smoothing (same as TA-Lib). Values converge after the warm-up window.
- **STOCH**: `ta` and ferro-ta use different default smoothing periods. Pass
  matching `window` / `smooth_window` values to get tail convergence.

## Performance Comparison

ferro-ta is significantly faster than `ta` for large arrays because the core
computation is written in Rust:

```bash
pytest benchmarks/test_speed.py --benchmark-only --benchmark-json=benchmarks/results.json
```

`ta` is a pure-Python/pandas library; ferro-ta processes 100k-bar arrays
in microseconds vs milliseconds for pandas-based implementations.
