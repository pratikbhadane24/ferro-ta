# ferro-ta ↔ Tulipy Compatibility

[Tulipy](https://github.com/cirla/tulipy) is the Python binding for
[Tulip Indicators](https://tulipindicators.org/) — 104 technical analysis
functions written in pure ANSI C99, designed for absolute speed with zero
external dependencies.

---

## Key architectural differences

| Aspect | ferro-ta | Tulipy |
|--------|---------|--------|
| **Backend** | Rust/C + SIMD | ANSI C99 |
| **Input type** | NumPy array or list | `np.float64` contiguous array |
| **Output length** | Same as input (NaN-padded) | Truncated (lookback bars shorter) |
| **NaN handling** | Pads warmup with NaN | Strips warmup entirely |
| **Multi-output** | Returns tuple | Returns tuple |
| **Pandas support** | Yes (via `ArrayLike`) | No |
| **Streaming** | Yes (StreamingXxx classes) | No |

---

## Output length difference

Tulipy **truncates** output instead of NaN-padding.  When comparing results
you must align by the **trailing** elements:

```python
import tulipy as ti
import ferro_ta
import numpy as np

close = np.ascontiguousarray(np.random.randn(100).cumsum() + 100, dtype=np.float64)

ti_sma = ti.sma(close, period=20)          # len = 81
ft_sma = ferro_ta.SMA(close, timeperiod=20)  # len = 100 (19 leading NaN)

# Align: compare last 81 values
n = len(ti_sma)
assert np.allclose(ti_sma, ft_sma[-n:][np.isfinite(ft_sma[-n:])], atol=1e-8)
```

---

## Function signature mapping

Tulipy uses lowercase function names.  The `period` argument is always a
positional-or-keyword integer.

| Indicator | ferro-ta | Tulipy |
|-----------|---------|--------|
| SMA | `SMA(close, timeperiod=20)` | `sma(close, period=20)` |
| EMA | `EMA(close, timeperiod=20)` | `ema(close, period=20)` |
| WMA | `WMA(close, timeperiod=14)` | `wma(close, period=14)` |
| RSI | `RSI(close, timeperiod=14)` | `rsi(close, period=14)` |
| MACD | `MACD(close, 12, 26, 9)` | `macd(close, short_period=12, long_period=26, signal_period=9)` |
| BBANDS | `BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)` → (upper, mid, lower) | `bbands(close, period=20, stddev=2.0)` → (lower, mid, upper) ⚠️ reversed! |
| ATR | `ATR(high, low, close, timeperiod=14)` | `atr(high, low, close, period=14)` |
| OBV | `OBV(close, volume)` | `obv(close, volume)` |
| CCI | `CCI(high, low, close, timeperiod=14)` | `cci(high, low, close, period=14)` |
| WILLR | `WILLR(high, low, close, timeperiod=14)` | `willr(high, low, close, period=14)` |
| STOCH | `STOCH(high, low, close, 5, 3, 3)` | `stoch(high, low, close, ...)` |
| HMA | Not supported | `hma(close, period=14)` |
| DEMA | `DEMA(close, timeperiod=30)` | `dema(close, period=30)` |
| TEMA | `TEMA(close, timeperiod=30)` | `tema(close, period=30)` |
| AROON | `AROONOSC(high, low, timeperiod=14)` | `aroonosc(high, low, period=14)` |
| MFI | `MFI(high, low, close, volume, timeperiod=14)` | `mfi(high, low, close, volume, period=14)` |
| TRANGE | `TRANGE(high, low, close)` | `tr(high, low, close)` |

⚠️ **BBANDS tuple order**: Tulipy returns `(lower, middle, upper)`;
ferro-ta and TA-Lib return `(upper, middle, lower)`.

---

## Memory requirements

Tulipy requires **strictly contiguous** `np.float64` arrays.  Passing a
Pandas Series slice or a non-contiguous array causes an error:

```python
# Wrong — may be a non-contiguous view
close = df["close"].values
ti.sma(close, period=20)  # may raise ValueError

# Correct — explicit contiguous cast
close = np.ascontiguousarray(df["close"].values, dtype=np.float64)
ti.sma(close, period=20)  # always works
```

ferro-ta accepts any `ArrayLike` and handles the conversion internally.

---

## Numerical accuracy

Tulipy and ferro-ta agree closely for SMA, WMA, and other non-recursive
indicators (differences < 1e-8).  For EMA-based indicators the first
`timeperiod` values differ due to initialisation seed choice:

- **Tulipy**: uses the first data value as the EMA seed.
- **ferro-ta**: follows TA-Lib convention (SMA of first `timeperiod` bars).

Values converge after approximately 2–3× the `timeperiod`.

---

## Speed comparison

On 10,000 bars (median µs, Apple M-series):

| Indicator | ferro-ta | Tulipy | Winner |
|-----------|--------:|-------:|--------|
| SMA       | 16.7    | 21.2   | ferro-ta |
| MACD      | 70.4    | 30.2   | Tulipy |
| ATR       | 51.4    | 27.6   | Tulipy |

Tulipy's C99 implementation excels for recursive indicators (ATR, MACD).
ferro-ta is faster for sliding-window indicators (SMA) thanks to SIMD
vectorisation.

---

## Migration guide

```python
# FROM Tulipy
import tulipy as ti
import numpy as np

close = np.ascontiguousarray(close_series.values, dtype=np.float64)
sma_values = ti.sma(close, period=20)  # length: n - 19

# TO ferro-ta (drop-in, same numeric result in the tail)
import ferro_ta

sma_values = ferro_ta.SMA(close, timeperiod=20)  # length: n (19 leading NaN)
# Strip warmup if needed:
sma_values = sma_values[~np.isnan(sma_values)]
```
