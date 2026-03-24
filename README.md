<div align="center">

# ⚡ ferro-ta

### Rust-powered Python technical analysis with a TA-Lib-compatible API

**Focused on one primary job: fast, reproducible technical analysis for Python users who want TA-Lib-style ergonomics without native build friction.**

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pratikbhadane24/ferro-ta/HEAD?labpath=examples%2Fquickstart.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pratikbhadane24/ferro-ta/blob/main/examples/quickstart.ipynb)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://pratikbhadane24.github.io/ferro-ta/)

</div>

---

> `ferro-ta` is a Rust-backed Python technical analysis library with a TA-Lib-compatible API for NumPy-first workloads.
>
> Performance varies by indicator, array layout, warmup, build flags, and machine. Public checked-in runs show `ferro-ta` is often faster on selected indicators, while TA-Lib still wins or ties on others. The benchmark workflow, artifacts, and caveats are published in [`benchmarks/README.md`](benchmarks/README.md).

---

## 🚀 What ferro-ta is

| | TA-Lib | ferro-ta |
|---|---|---|
| **Primary product** | C-backed Python TA library | Rust-backed Python TA library |
| **API shape** | `talib.SMA(close, 20)` | `ferro_ta.SMA(close, 20)` |
| **Installation** | Often requires native/system setup | Pre-built wheels on supported targets |
| **Performance claim** | Established baseline | Often faster on selected indicators; see reproducible benchmarks |
| **Scope** | Technical indicators | Technical indicators first; other tooling is optional and secondary |

---

## ⚡ Benchmark evidence

The latest checked-in TA-Lib comparison artifact uses contiguous `float64`
arrays at 10k and 100k bars on an `Apple M3 Max`, `CPython 3.13.5`, `Rust
1.91.1`, default release profile (`lto = true`, `codegen-units = 1`), with no
extra `RUSTFLAGS`:

- `ferro-ta` is ahead outside the tie band on 6 of 12 indicators at 10k bars and 6 of 12 at 100k bars.
- At 100k bars, the stronger public wins are `SMA` (`2.28x`), `BBANDS` (`2.34x`), `MACD` (`1.38x`), `MFI` (`3.04x`), and `WMA` (`2.39x`).
- TA-Lib still wins on `STOCH` and `ADX` in the current checked-in 10k and 100k runs, and still wins or ties on `EMA`, `RSI`, `ATR`, and `OBV`.
- The published JSON now includes per-run samples, variance stats, and Python-tracked allocation snapshots, not just a single median.

The point of the benchmark suite is not to claim universal wins. It is to let readers reproduce the results, inspect the raw artifact, and see where each library is stronger.

### 🏆 Reproduce the public comparison

- Methodology, artifact format, and result tables: [`benchmarks/README.md`](benchmarks/README.md)
- Latest checked-in artifact bundle: [`benchmarks/artifacts/latest/`](benchmarks/artifacts/latest/)
- TA-Lib head-to-head script: `benchmarks/bench_vs_talib.py`
- Cross-library suite: `benchmarks/test_speed.py`

```bash
# Reproduce the TA-Lib comparison yourself
pip install ferro-ta ta-lib
python benchmarks/bench_vs_talib.py --sizes 10000 100000 --json benchmark_vs_talib.json

# or with uv
uv run python benchmarks/bench_vs_talib.py --sizes 10000 100000 --json benchmark_vs_talib.json
uv run python benchmarks/check_vs_talib_regression.py --input benchmark_vs_talib.json

# full cross-library speed suite (100k bars)
uv run pytest benchmarks/test_speed.py --benchmark-only --benchmark-json=benchmarks/results.json -v

# generate the comparison table from results.json
uv run python benchmarks/benchmark_table.py
```

---

## 🎯 Core capabilities

- **TA-Lib-style API** for 160+ indicators, including the common `SMA`, `EMA`, `RSI`, `MACD`, and `BBANDS` entry points.
- **Pre-built wheels** for supported Python and OS targets, so the common install path stays `pip install ferro-ta`.
- **NumPy-first execution** with pandas and polars adapters, plus explicit guidance on contiguous-array fast paths.
- **Batch and streaming APIs** for multi-series and bar-by-bar workloads.
- **Compatibility and support docs** covering parity status, supported wheels, supported Python versions, and experimental modules.
- **Type stubs, error model, API discovery, and examples** for day-to-day library use.

## 🧪 Adjacent and experimental modules

These ship in the repo, but they are not the primary product story:

- **Adjacent analytics:** derivatives helpers, backtesting utilities, portfolio and cross-asset analysis, feature generation, and charting.
- **Experimental or optional tooling:** GPU backend, plugin system, WASM package, agent/tool wrappers, and the MCP server.
- **Docs posture:** these modules are now called out separately in the docs nav and support matrix so the core TA library remains the main narrative.

---

## 📦 Installation

```bash
pip install ferro-ta
```

Optional extras:

```bash
pip install "ferro-ta[pandas]"   # transparent pandas.Series support
pip install "ferro-ta[polars]"   # transparent polars.Series support
pip install "ferro-ta[gpu]"      # GPU-accelerated SMA/EMA/RSI via PyTorch (CUDA/MPS)
pip install "ferro-ta[options]"  # Derivatives analytics helpers
pip install "ferro-ta[mcp]"      # MCP server for Cursor/Claude agent integration
pip install "ferro-ta[all]"      # all optional extras (excluding gpu)
```

---

## ⚡ Quick Start

```python
import numpy as np
from ferro_ta import SMA, EMA, RSI, MACD, BBANDS

close = np.array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.15,
                  43.61, 44.33, 44.83, 45.10, 45.15, 43.61, 44.33])

# Simple Moving Average
sma = SMA(close, timeperiod=5)

# Exponential Moving Average
ema = EMA(close, timeperiod=5)

# Relative Strength Index
rsi = RSI(close, timeperiod=14)

# MACD (returns macd_line, signal_line, histogram)
macd_line, signal, histogram = MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

# Bollinger Bands (returns upper, middle, lower)
upper, middle, lower = BBANDS(close, timeperiod=5, nbdevup=2.0, nbdevdn=2.0)
```

## Δ Derivatives Analytics

```python
from ferro_ta.analysis.options import greeks, implied_volatility, option_price
from ferro_ta.analysis.futures import basis, curve_summary

price = option_price(100.0, 100.0, 0.05, 1.0, 0.20, option_type="call", model="bsm")
iv = implied_volatility(price, 100.0, 100.0, 0.05, 1.0, option_type="call", model="bsm")
g = greeks(100.0, 100.0, 0.05, 1.0, 0.20, option_type="call", model="bsm")

front_basis = basis(100.0, 103.0)
curve = curve_summary(100.0, [0.1, 0.5, 1.0], [101.0, 102.0, 104.0])
```

The derivatives layer is analytics-only. It includes:

- options pricing under Black-Scholes-Merton and Black-76
- delta, gamma, vega, theta, and rho
- implied volatility inversion and smile metrics
- futures basis, carry, curve, and continuous-roll helpers
- typed strategy schemas and multi-leg payoff/Greeks aggregation

**Migrating from TA-Lib?** Just swap the import — the API is identical:

```python
# Before (TA-Lib)
import talib
sma = talib.SMA(close, timeperiod=20)
rsi = talib.RSI(close, timeperiod=14)

# After (ferro-ta — same call signature)
import ferro_ta
sma = ferro_ta.SMA(close, timeperiod=20)
rsi = ferro_ta.RSI(close, timeperiod=14)
```

---

## 🛠️ Development Setup

Requires Rust and **Python 3.10–3.13** (PyO3 supports up to 3.13; for Python 3.14+ use a compatible interpreter or set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` to attempt a build).

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install build tool and dependencies
pip install maturin numpy pytest pandas

# Compile and install in editable mode
maturin develop --release

# Run tests
pytest tests/unit/ tests/integration/
# or: uv run pytest tests/unit/ tests/integration/

# Run TA-Lib comparison tests (requires ta-lib package)
pip install "ferro-ta[comparison]"   # or: pip install ta-lib
pytest tests/integration/test_vs_talib.py -v

# Build Sphinx documentation (requires sphinx + sphinx-rtd-theme)
pip install "ferro-ta[docs]"
cd docs && make html
# Output: docs/_build/html/index.html
```

---

## 📊 Full TA-Lib Compatibility

ferro-ta covers **100% of TA-Lib's function set** (162+ indicators). The table below shows implementation status and numerical accuracy vs TA-Lib.

**Legend**

| Symbol | Meaning |
|--------|---------|
| ✅ Exact | Values match TA-Lib to floating-point precision |
| ✅ Close | Values match after a short convergence window (EMA-seed difference) |
| ⚠️ Corr | Strong correlation (> 0.95) but not numerically identical (Wilder smoothing seed or algorithm variant) |
| ⚠️ Shape | Same output shape / NaN structure; values differ due to algorithm variant |
| ❌ | Not yet implemented |

### Overlap Studies

| TA-Lib Function | ferro-ta | Accuracy | Notes |
|-----------------|---------|----------|-------|
| `BBANDS` | ✅ | ✅ Exact | Bollinger Bands |
| `DEMA` | ✅ | ✅ Close | Double EMA; converges after ~20 bars |
| `EMA` | ✅ | ✅ Close | Exponential Moving Average; converges after ~20 bars |
| `KAMA` | ✅ | ✅ Exact | Kaufman Adaptive MA (values match after seed bar) |
| `MA` | ✅ | ✅ Exact | Moving average (generic, type-selectable) |
| `MAMA` | ✅ | ⚠️ Corr | MESA Adaptive MA |
| `MAVP` | ✅ | ✅ Exact | MA with variable period |
| `MIDPOINT` | ✅ | ✅ Exact | Midpoint over period |
| `MIDPRICE` | ✅ | ✅ Exact | Midpoint price over period |
| `SAR` | ✅ | ⚠️ Shape | Parabolic SAR (same shape; reversal history diverges) |
| `SAREXT` | ✅ | ⚠️ Shape | Parabolic SAR Extended |
| `SMA` | ✅ | ✅ Exact | Simple Moving Average |
| `T3` | ✅ | ✅ Close | Triple Exponential MA (T3); converges after ~50 bars |
| `TEMA` | ✅ | ✅ Close | Triple EMA; converges after ~20 bars |
| `TRIMA` | ✅ | ✅ Exact | Triangular Moving Average |
| `WMA` | ✅ | ✅ Exact | Weighted Moving Average |

### Momentum Indicators

| TA-Lib Function | ferro-ta | Accuracy | Notes |
|-----------------|---------|----------|-------|
| `ADX` | ✅ | ✅ Close | Avg Directional Movement Index (TA-Lib Wilder sum-seeding) |
| `ADXR` | ✅ | ✅ Close | ADX Rating (inherits ADX; TA-Lib seeding) |
| `APO` | ✅ | ✅ Close | Absolute Price Oscillator (EMA-based) |
| `AROON` | ✅ | ✅ Exact | Aroon Up/Down |
| `AROONOSC` | ✅ | ✅ Exact | Aroon Oscillator |
| `BOP` | ✅ | ✅ Exact | Balance Of Power |
| `CCI` | ✅ | ✅ Exact | Commodity Channel Index (TA-Lib–compatible MAD formula) |
| `CMO` | ✅ | ✅ Close | Chande Momentum Oscillator (rolling window, TA-Lib–compatible) |
| `DX` | ✅ | ✅ Close | Directional Movement Index (TA-Lib Wilder sum-seeding) |
| `MACD` | ✅ | ✅ Close | MACD (EMA-based; converges after ~30 bars) |
| `MACDEXT` | ✅ | ✅ Close | MACD with controllable MA type (EMA-based; converges) |
| `MACDFIX` | ✅ | ✅ Close | MACD Fixed 12/26 (EMA-based; converges) |
| `MFI` | ✅ | ✅ Exact | Money Flow Index |
| `MINUS_DI` | ✅ | ✅ Close | Minus Directional Indicator (TA-Lib Wilder sum-seeding) |
| `MINUS_DM` | ✅ | ✅ Close | Minus Directional Movement (TA-Lib Wilder sum-seeding) |
| `MOM` | ✅ | ✅ Exact | Momentum |
| `PLUS_DI` | ✅ | ✅ Close | Plus Directional Indicator (TA-Lib Wilder sum-seeding) |
| `PLUS_DM` | ✅ | ✅ Close | Plus Directional Movement (TA-Lib Wilder sum-seeding) |
| `PPO` | ✅ | ✅ Close | Percentage Price Oscillator (EMA-based) |
| `ROC` | ✅ | ✅ Exact | Rate of Change |
| `ROCP` | ✅ | ✅ Exact | Rate of Change Percentage |
| `ROCR` | ✅ | ✅ Exact | Rate of Change Ratio |
| `ROCR100` | ✅ | ✅ Exact | Rate of Change Ratio × 100 |
| `RSI` | ✅ | ✅ Close | Relative Strength Index (TA-Lib Wilder seeding; converges after ~1 seed bar) |
| `STOCH` | ✅ | ✅ Close | Stochastic (TA-Lib–compatible SMA smoothing for slowk and slowd) |
| `STOCHF` | ✅ | ✅ Exact | Stochastic Fast (%K exact; %D NaN offset ±2) |
| `STOCHRSI` | ✅ | ✅ Close | Stochastic RSI (TA-Lib–compatible; SMA fastd, Wilder-seeded RSI) |
| `TRIX` | ✅ | ✅ Close | 1-day ROC of Triple EMA (EMA-based; converges) |
| `ULTOSC` | ✅ | ✅ Exact | Ultimate Oscillator |
| `WILLR` | ✅ | ✅ Exact | Williams' %R |

### Volume Indicators

| TA-Lib Function | ferro-ta | Accuracy | Notes |
|-----------------|---------|----------|-------|
| `AD` | ✅ | ✅ Exact | Chaikin A/D Line |
| `ADOSC` | ✅ | ✅ Exact | Chaikin A/D Oscillator |
| `OBV` | ✅ | ✅ Exact | On Balance Volume (increments identical; constant offset at bar 0) |

### Volatility Indicators

| TA-Lib Function | ferro-ta | Accuracy | Notes |
|-----------------|---------|----------|-------|
| `ATR` | ✅ | ✅ Close | Average True Range (TA-Lib Wilder seeding; matches from bar timeperiod) |
| `NATR` | ✅ | ✅ Close | Normalized ATR (TA-Lib Wilder seeding) |
| `TRANGE` | ✅ | ✅ Exact | True Range (bar 0 differs; all others identical) |

### Cycle Indicators

| TA-Lib Function | ferro-ta | Accuracy | Notes |
|-----------------|---------|----------|-------|
| `HT_DCPERIOD` | ✅ | ⚠️ Shape | Hilbert Transform Dominant Cycle Period (Ehlers algorithm) |
| `HT_DCPHASE` | ✅ | ⚠️ Shape | Hilbert Transform Dominant Cycle Phase |
| `HT_PHASOR` | ✅ | ⚠️ Shape | Hilbert Transform Phasor Components (inphase, quadrature) |
| `HT_SINE` | ✅ | ⚠️ Shape | Hilbert Transform SineWave (sine, leadsine) |
| `HT_TRENDLINE` | ✅ | ⚠️ Shape | Hilbert Transform Instantaneous Trendline |
| `HT_TRENDMODE` | ✅ | ⚠️ Shape | Hilbert Transform Trend vs Cycle Mode (1=trend, 0=cycle) |

### Price Transformations

| TA-Lib Function | ferro-ta | Accuracy | Notes |
|-----------------|---------|----------|-------|
| `AVGPRICE` | ✅ | ✅ Exact | Average Price |
| `MEDPRICE` | ✅ | ✅ Exact | Median Price |
| `TYPPRICE` | ✅ | ✅ Exact | Typical Price |
| `WCLPRICE` | ✅ | ✅ Exact | Weighted Close Price |

### Statistic Functions

| TA-Lib Function | ferro-ta | Accuracy | Notes |
|-----------------|---------|----------|-------|
| `BETA` | ✅ | ✅ Close | Beta coefficient (returns-based regression matching TA-Lib) |
| `CORREL` | ✅ | ✅ Exact | Pearson Correlation Coefficient |
| `LINEARREG` | ✅ | ✅ Exact | Linear Regression |
| `LINEARREG_ANGLE` | ✅ | ✅ Exact | Linear Regression Angle |
| `LINEARREG_INTERCEPT` | ✅ | ✅ Exact | Linear Regression Intercept |
| `LINEARREG_SLOPE` | ✅ | ✅ Exact | Linear Regression Slope |
| `STDDEV` | ✅ | ✅ Exact | Standard Deviation |
| `TSF` | ✅ | ✅ Exact | Time Series Forecast |
| `VAR` | ✅ | ✅ Exact | Variance |

### Pattern Recognition

ferro-ta implements all 61 candlestick patterns. All return the same `{-100, 0, 100}`
convention as TA-Lib. Pattern thresholds may differ slightly from the full TA-Lib
implementation.

| TA-Lib Function | ferro-ta | Notes |
|-----------------|---------|-------|
| `CDL2CROWS` | ✅ | Two Crows |
| `CDL3BLACKCROWS` | ✅ | Three Black Crows |
| `CDL3INSIDE` | ✅ | Three Inside Up/Down |
| `CDL3LINESTRIKE` | ✅ | Three-Line Strike |
| `CDL3OUTSIDE` | ✅ | Three Outside Up/Down |
| `CDL3STARSINSOUTH` | ✅ | Three Stars In The South |
| `CDL3WHITESOLDIERS` | ✅ | Three Advancing White Soldiers |
| `CDLABANDONEDBABY` | ✅ | Abandoned Baby |
| `CDLADVANCEBLOCK` | ✅ | Advance Block |
| `CDLBELTHOLD` | ✅ | Belt-hold |
| `CDLBREAKAWAY` | ✅ | Breakaway |
| `CDLCLOSINGMARUBOZU` | ✅ | Closing Marubozu |
| `CDLCONCEALBABYSWALL` | ✅ | Concealing Baby Swallow |
| `CDLCOUNTERATTACK` | ✅ | Counterattack |
| `CDLDARKCLOUDCOVER` | ✅ | Dark Cloud Cover |
| `CDLDOJI` | ✅ | Doji |
| `CDLDOJISTAR` | ✅ | Doji Star |
| `CDLDRAGONFLYDOJI` | ✅ | Dragonfly Doji |
| `CDLENGULFING` | ✅ | Engulfing Pattern |
| `CDLEVENINGDOJISTAR` | ✅ | Evening Doji Star |
| `CDLEVENINGSTAR` | ✅ | Evening Star |
| `CDLGAPSIDESIDEWHITE` | ✅ | Up/Down-gap side-by-side white lines |
| `CDLGRAVESTONEDOJI` | ✅ | Gravestone Doji |
| `CDLHAMMER` | ✅ | Hammer |
| `CDLHANGINGMAN` | ✅ | Hanging Man |
| `CDLHARAMI` | ✅ | Harami Pattern |
| `CDLHARAMICROSS` | ✅ | Harami Cross Pattern |
| `CDLHIGHWAVE` | ✅ | High-Wave Candle |
| `CDLHIKKAKE` | ✅ | Hikkake Pattern |
| `CDLHIKKAKEMOD` | ✅ | Modified Hikkake Pattern |
| `CDLHOMINGPIGEON` | ✅ | Homing Pigeon |
| `CDLIDENTICAL3CROWS` | ✅ | Identical Three Crows |
| `CDLINNECK` | ✅ | In-Neck Pattern |
| `CDLINVERTEDHAMMER` | ✅ | Inverted Hammer |
| `CDLKICKING` | ✅ | Kicking |
| `CDLKICKINGBYLENGTH` | ✅ | Kicking by the longer Marubozu |
| `CDLLADDERBOTTOM` | ✅ | Ladder Bottom |
| `CDLLONGLEGGEDDOJI` | ✅ | Long Legged Doji |
| `CDLLONGLINE` | ✅ | Long Line Candle |
| `CDLMARUBOZU` | ✅ | Marubozu |
| `CDLMATCHINGLOW` | ✅ | Matching Low |
| `CDLMATHOLD` | ✅ | Mat Hold |
| `CDLMORNINGDOJISTAR` | ✅ | Morning Doji Star |
| `CDLMORNINGSTAR` | ✅ | Morning Star |
| `CDLONNECK` | ✅ | On-Neck Pattern |
| `CDLPIERCING` | ✅ | Piercing Pattern |
| `CDLRICKSHAWMAN` | ✅ | Rickshaw Man |
| `CDLRISEFALL3METHODS` | ✅ | Rising/Falling Three Methods |
| `CDLSEPARATINGLINES` | ✅ | Separating Lines |
| `CDLSHOOTINGSTAR` | ✅ | Shooting Star |
| `CDLSHORTLINE` | ✅ | Short Line Candle |
| `CDLSPINNINGTOP` | ✅ | Spinning Top |
| `CDLSTALLEDPATTERN` | ✅ | Stalled Pattern |
| `CDLSTICKSANDWICH` | ✅ | Stick Sandwich |
| `CDLTAKURI` | ✅ | Takuri (Dragonfly Doji with very long lower shadow) |
| `CDLTASUKIGAP` | ✅ | Tasuki Gap |
| `CDLTHRUSTING` | ✅ | Thrusting Pattern |
| `CDLTRISTAR` | ✅ | Tristar Pattern |
| `CDLUNIQUE3RIVER` | ✅ | Unique 3 River |
| `CDLUPSIDEGAP2CROWS` | ✅ | Upside Gap Two Crows |
| `CDLXSIDEGAP3METHODS` | ✅ | Upside/Downside Gap Three Methods |

### Math Operators / Math Transforms

ferro-ta provides TA-Lib–compatible wrappers for all arithmetic and math-transform functions.
Rolling functions (SUM, MAX, MIN) produce NaN for the first `timeperiod - 1` bars.

| TA-Lib Function | ferro-ta | Notes |
|-----------------|---------|-------|
| `ADD` | ✅ | Element-wise addition |
| `SUB` | ✅ | Element-wise subtraction |
| `MULT` | ✅ | Element-wise multiplication |
| `DIV` | ✅ | Element-wise division |
| `SUM` | ✅ | Rolling sum over *timeperiod* |
| `MAX` / `MAXINDEX` | ✅ | Rolling maximum / index |
| `MIN` / `MININDEX` | ✅ | Rolling minimum / index |
| `ACOS` / `ASIN` / `ATAN` | ✅ | Arc trig transforms |
| `CEIL` / `FLOOR` | ✅ | Round up / down |
| `COS` / `SIN` / `TAN` | ✅ | Trig transforms |
| `COSH` / `SINH` / `TANH` | ✅ | Hyperbolic transforms |
| `EXP` / `LN` / `LOG10` | ✅ | Exponential / log transforms |
| `SQRT` | ✅ | Square root |

### Pandas API

**Contract:** All indicators accept `pandas.Series` (or 1-D DataFrame columns) and return
`pandas.Series` — or a **tuple of Series** for multi-output functions like `MACD`, `BBANDS` —
with the **original index preserved**.

**Default OHLCV column names:** When using a DataFrame with OHLCV data, the conventional names
are `open`, `high`, `low`, `close`, `volume`. To use different column names, use the helper
:func:`ferro_ta.utils.get_ohlcv` (or pass Series/arrays extracted from your DataFrame).

**Single Series or tuple of Series:**

```python
import pandas as pd
from ferro_ta import SMA, BBANDS, MACD, CDLDOJI

close = pd.Series([44.34, 44.09, 44.15, 43.61, 44.33], index=pd.date_range("2024-01-01", 5))

# Single-output: returns Series
sma = SMA(close, timeperiod=3)  # pd.Series with same index

# Multi-output: returns tuple of Series
upper, mid, lower = BBANDS(close, timeperiod=3)  # all pd.Series
```

**DataFrame with OHLCV columns (configurable names):**

```python
import pandas as pd
from ferro_ta import ATR, RSI
from ferro_ta.utils import get_ohlcv  # or: from ferro_ta._utils import get_ohlcv

df = pd.DataFrame({
    "Open": [1, 2, 3], "High": [1.1, 2.1, 3.1],
    "Low": [0.9, 1.9, 2.9], "Close": [1.05, 2.05, 3.05],
}, index=pd.date_range("2024-01-01", periods=3, freq="D"))

# Extract with default names (open, high, low, close, volume)
o, h, l, c, v = get_ohlcv(df, open_col="Open", high_col="High", low_col="Low", close_col="Close")
atr = ATR(h, l, c, timeperiod=2)   # index preserved
rsi = RSI(c, timeperiod=2)         # index preserved
```

### Extended Indicators

ferro-ta includes popular indicators that go beyond the TA-Lib standard set.
These are available in `ferro_ta.extended` and importable directly from `ferro_ta`.

| Function | ferro-ta | Notes |
|----------|---------|-------|
| `VWAP` | ✅ | Volume Weighted Average Price — cumulative (session) or rolling window |
| `SUPERTREND` | ✅ | ATR-based trend signal; returns (supertrend_line, direction) |
| `ICHIMOKU` | ✅ | Ichimoku Cloud — Tenkan, Kijun, Senkou A/B, Chikou Span |
| `DONCHIAN` | ✅ | Donchian Channels — rolling highest high / lowest low |
| `PIVOT_POINTS` | ✅ | Pivot points — Classic, Fibonacci, Camarilla methods |
| `KELTNER_CHANNELS` | ✅ | EMA ± (ATR × multiplier) bands; returns (upper, middle, lower) |
| `HULL_MA` | ✅ | Hull Moving Average — fast, low-lag WMA-based MA |
| `CHANDELIER_EXIT` | ✅ | ATR-based trailing stop levels; returns (long_exit, short_exit) |
| `VWMA` | ✅ | Volume Weighted Moving Average — rolling sum(close*vol) / sum(vol) |
| `CHOPPINESS_INDEX` | ✅ | Market choppiness/trending strength index (0–100) |

```python
from ferro_ta import VWAP, SUPERTREND, ICHIMOKU, DONCHIAN, PIVOT_POINTS
from ferro_ta import KELTNER_CHANNELS, HULL_MA, CHANDELIER_EXIT, VWMA, CHOPPINESS_INDEX
import numpy as np

close = np.array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.15])
high  = close + 0.5
low   = close - 0.5
vol   = np.full(len(close), 1_000_000.0)

# Cumulative / rolling VWAP
vwap = VWAP(high, low, close, vol)
rolling_vwap = VWAP(high, low, close, vol, timeperiod=5)

# Supertrend (trend line and direction: 1=up, -1=down)
st_line, direction = SUPERTREND(high, low, close, timeperiod=7, multiplier=3.0)

# Ichimoku Cloud
tenkan, kijun, senkou_a, senkou_b, chikou = ICHIMOKU(high, low, close)

# Donchian Channels
dc_upper, dc_mid, dc_lower = DONCHIAN(high, low, timeperiod=5)

# Pivot Points
pivot, r1, s1, r2, s2 = PIVOT_POINTS(high, low, close, method="classic")
# method options: "classic", "fibonacci", "camarilla"

# Keltner Channels
kc_upper, kc_mid, kc_lower = KELTNER_CHANNELS(high, low, close, timeperiod=20, atr_period=10)

# Hull Moving Average
hull = HULL_MA(close, timeperiod=16)

# Chandelier Exit
long_exit, short_exit = CHANDELIER_EXIT(high, low, close, timeperiod=22, multiplier=3.0)

# Volume Weighted Moving Average
vwma = VWMA(close, vol, timeperiod=20)

# Choppiness Index (100 = choppy, 0 = strong trend)
ci = CHOPPINESS_INDEX(high, low, close, timeperiod=14)
```

### Streaming / Live-Trading API

For real-time / bar-by-bar processing, import classes from `ferro_ta.streaming`.
Each class maintains state internally and returns `NaN` during the warmup window:

```python
from ferro_ta.streaming import StreamingSMA, StreamingEMA, StreamingRSI, StreamingATR
from ferro_ta.streaming import StreamingBBands, StreamingMACD, StreamingStoch
from ferro_ta.streaming import StreamingVWAP, StreamingSupertrend

sma = StreamingSMA(period=20)
rsi = StreamingRSI(period=14)
atr = StreamingATR(period=14)
bb  = StreamingBBands(period=20, nbdevup=2.0, nbdevdn=2.0)
macd = StreamingMACD(fastperiod=12, slowperiod=26, signalperiod=9)
stoch = StreamingStoch(fastk_period=5, slowk_period=3, slowd_period=3)
vwap = StreamingVWAP()  # reset() at session open
st = StreamingSupertrend(period=7, multiplier=3.0)

for bar in live_data_feed:
    current_sma  = sma.update(bar.close)
    current_rsi  = rsi.update(bar.close)
    current_atr  = atr.update(bar.high, bar.low, bar.close)
    upper, mid, lower = bb.update(bar.close)
    macd_line, signal, histogram = macd.update(bar.close)
    slowk, slowd = stoch.update(bar.high, bar.low, bar.close)
    current_vwap = vwap.update(bar.high, bar.low, bar.close, bar.volume)
    st_line, trend_dir = st.update(bar.high, bar.low, bar.close)  # 1=up, -1=down
```

### 📈 Implementation Coverage Summary

| Category | Implemented | Not Implemented |
|----------|:-----------:|:---------------:|
| Overlap Studies | 19 | 0 |
| Momentum Indicators | 28 | 0 |
| Volume Indicators | 3 | 0 |
| Volatility Indicators | 3 | 0 |
| Cycle Indicators | 6 | 0 |
| Price Transforms | 4 | 0 |
| Statistic Functions | 9 | 0 |
| Pattern Recognition | 61 | 0 |
| Math Operators / Transforms | 24 | 0 |
| Extended Indicators | 10 | — |
| Streaming Classes | 9 | — |
| **Total** | **162+** | **0** |

> 🎉 **100% of TA-Lib's function set is implemented.** NaN values are placed at the beginning of each output array for the warmup period.

---

## 🔄 Batch Execution API

Run indicators on multiple price series (symbols) in a single call. Dedicated Rust-backed functions for SMA, EMA, RSI, ATR, STOCH, and ADX; use `batch_apply` for any other indicator.

```python
import numpy as np
from ferro_ta.batch import batch_sma, batch_ema, batch_rsi, batch_atr, batch_stoch, batch_adx, batch_apply

# 100 bars × 5 symbols
close = np.random.rand(100, 5) + 50.0
high = close + 0.1
low = close - 0.1

sma_out = batch_sma(close, timeperiod=14)       # (100, 5)
ema_out = batch_ema(close, timeperiod=14)       # (100, 5)
rsi_out = batch_rsi(close, timeperiod=14)       # (100, 5)
atr_out = batch_atr(high, low, close, timeperiod=14)
stoch_k, stoch_d = batch_stoch(high, low, close)
adx_out = batch_adx(high, low, close, timeperiod=14)

# Any single-series function via batch_apply
from ferro_ta import BBANDS
def bbands_upper(c, **kw):
    return BBANDS(c, **kw)[0]
upper = batch_apply(close, bbands_upper, timeperiod=20)
```

---

## 🦀 Pure Rust Core Library

ferro-ta is structured as a Cargo workspace with two crates:

| Crate | Purpose |
|-------|---------|
| `ferro_ta` (root) | PyO3 `#[pyfunction]` wrappers — converts numpy ↔ `&[f64]`; builds the Python wheel |
| `crates/ferro_ta_core` | Pure Rust indicators — no PyO3/numpy dependency; usable from any Rust project |

```bash
# Build and test the core crate directly
cargo build -p ferro_ta_core
cargo test -p ferro_ta_core
```

```rust
use ferro_ta_core::overlap;

let close = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let sma = overlap::sma(&close, 3);
```

### Rust Module Structure

The main `ferro_ta` crate (`src/`) uses a **consistent directory-based module layout** matching the TA-Lib category structure. Every module is a directory with `mod.rs` declaring sub-modules and a `register()` function; each indicator (or closely related group) lives in its own `.rs` file:

```
src/
├── lib.rs                   # PyModule entry point — calls each module's register()
├── overlap/                 # Overlap Studies (SMA, EMA, BBANDS, MACD, SAR, …)
│   ├── mod.rs
│   ├── sma.rs, ema.rs, wma.rs, dema.rs, tema.rs, trima.rs, kama.rs, t3.rs
│   ├── bbands.rs, macd.rs, macdfix.rs, macdext.rs
│   ├── sar.rs, sarext.rs, mama.rs, midpoint.rs, midprice.rs
│   └── ma_mavp.rs
├── momentum/                # Momentum Indicators (RSI, STOCH, ADX, CCI, …)
│   ├── mod.rs
│   └── rsi.rs, mom.rs, roc.rs, willr.rs, aroon.rs, cci.rs, mfi.rs,
│       bop.rs, stochf.rs, stoch.rs, stochrsi.rs, apo.rs, ppo.rs, cmo.rs,
│       adx.rs, trix.rs, ultosc.rs
├── volatility/              # Volatility Indicators (ATR, NATR, TRANGE)
│   ├── mod.rs
│   ├── common.rs            # shared TR computation
│   ├── trange.rs, atr.rs, natr.rs
├── volume/                  # Volume Indicators (AD, ADOSC, OBV)
│   ├── mod.rs
│   └── ad.rs, adosc.rs, obv.rs
├── statistic/               # Statistic Functions (STDDEV, VAR, LINEARREG*, BETA, CORREL)
│   ├── mod.rs
│   ├── common.rs            # shared linreg() helper
│   └── stddev.rs, var.rs, linearreg.rs, beta.rs, correl.rs
├── price_transform/         # Price Transformations (AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE)
│   ├── mod.rs
│   └── avgprice.rs, medprice.rs, typprice.rs, wclprice.rs
├── cycle/                   # Cycle Indicators (HT_TRENDLINE, HT_DCPERIOD, …)
│   ├── mod.rs
│   ├── common.rs            # shared HT core pipeline (compute_ht_core)
│   └── ht_trendline.rs, ht_dcperiod.rs, ht_dcphase.rs,
│       ht_phasor.rs, ht_sine.rs, ht_trendmode.rs
└── pattern/                 # Pattern Recognition (CDL2CROWS, CDLDOJI, …)
    ├── mod.rs
    ├── common.rs            # shared candle utilities
    └── cdl*.rs              # one file per pattern (61 patterns)
```

This layout makes it easy to add, review, or modify individual indicators in isolation — simply edit or add the relevant `.rs` file and update `mod.rs`.

### Python sub-package layout

The `python/ferro_ta/` package is organized into sub-packages by concern.
Backward-compat stubs at the old flat paths (e.g. `ferro_ta.momentum`) re-export
from the new locations, so existing code continues to work without changes.

```
python/ferro_ta/
├── __init__.py        # top-level re-exports and public API
├── core/              # Exceptions, configuration, registry, logging, raw FFI bindings
├── indicators/        # Technical indicators (momentum, overlap, volatility, volume,
│                      #   statistic, cycle, pattern, price_transform, math_ops, extended)
├── data/              # Streaming, batch, chunked, resampling, aggregation, adapters
├── analysis/          # Portfolio, backtest, regime, cross_asset, attribution,
│                      #   signals, features, crypto, options, futures,
│                      #   options_strategy, derivatives_payoff
├── tools/             # Visualisation, alerting, DSL, pipeline, workflow,
│                      #   api_info, GPU support
└── mcp/               # Model Context Protocol server
```



## 🌐 Other Languages (WebAssembly / Node.js)

A WebAssembly binding is available in the `wasm/` directory, exposing SMA, EMA, BBANDS,
RSI, ATR, OBV, and MACD for use in Node.js and browsers.

```javascript
// Node.js (after `wasm-pack build --target nodejs --out-dir pkg` in wasm/)
const { sma, rsi, macd } = require('./wasm/pkg/ferro_ta_wasm.js');

const close = new Float64Array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10]);
const smaOut = sma(close, 3);       // Float64Array — first 2 values are NaN
const rsiOut = rsi(close, 5);       // Float64Array — first 5 values are NaN

// MACD — returns [macd_line, signal_line, histogram] as a js_sys::Array
const [macdLine, signal, hist] = macd(close, 3, 5, 2);
```

See [`wasm/README.md`](wasm/README.md) for build instructions, the full list of exposed
functions, and browser usage examples.

---

## 🔥 GPU Acceleration (Optional)

For very large arrays (millions of bars), an optional GPU-accelerated path is available
via [PyTorch](https://pytorch.org/). Pass a `torch.Tensor` on CUDA or MPS and get a tensor back;
NumPy in → NumPy out (CPU fallback).

```bash
pip install "ferro-ta[gpu]"
# or install PyTorch yourself (e.g. with CUDA or MPS support):
# pip install torch
```

```python
import torch
from ferro_ta.gpu import sma, ema, rsi

# Use CUDA or MPS (Apple Silicon)
close_gpu = torch.tensor(
    [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.15, 43.61, 44.33],
    device="cuda",  # or device="mps" on Apple Silicon
    dtype=torch.float64,
)

result = sma(close_gpu, timeperiod=5)   # torch.Tensor on same device
result_cpu = result.cpu().numpy()       # back to NumPy if needed
```

PyTorch tensors in → PyTorch tensors out; NumPy arrays in → NumPy arrays out (CPU).
See [`docs/gpu-backend.md`](docs/gpu-backend.md) for supported indicators, limitations,
and benchmark data.

---

## 📉 Backtesting

A minimal vectorized backtester is available at `ferro_ta.backtest`:

```python
import numpy as np
from ferro_ta.backtest import backtest

np.random.seed(42)
close = np.cumprod(1 + np.random.randn(200) * 0.01) * 100

# Run an RSI 30/70 strategy
result = backtest(close, strategy="rsi_30_70", timeperiod=14)
print(f"Final equity: {result.final_equity:.4f}")
print(f"Number of trades: {result.n_trades}")

# Or use SMA crossover
result2 = backtest(close, strategy="sma_crossover", fast=10, slow=30)
result3 = backtest(close, strategy="macd_crossover", commission_per_trade=0.001, slippage_bps=5)
```

> **Note:** This is a *minimal harness* for testing strategies. Optional `commission_per_trade` and `slippage_bps` are supported; for margin or full order types consider `backtrader`, `zipline`, or `vectorbt`.
> For production use consider `backtrader`, `zipline`, or `vectorbt`.

---

## 🔗 Indicator Pipeline

Compose multiple indicators into a reusable pipeline:

```python
import numpy as np
from ferro_ta import SMA, EMA, RSI, BBANDS
from ferro_ta.pipeline import Pipeline

close = np.cumprod(1 + np.random.randn(200) * 0.01) * 100

pipe = (
    Pipeline()
    .add("sma_20", SMA, timeperiod=20)
    .add("ema_20", EMA, timeperiod=20)
    .add("rsi_14", RSI, timeperiod=14)
    .add("bb", BBANDS, output_keys=["bb_upper", "bb_mid", "bb_lower"],
         timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
)

results = pipe.run(close)
# {'sma_20': array([...]), 'ema_20': array([...]), ..., 'bb_lower': array([...])}
print(list(results.keys()))
```

---

## ⚙️ Configuration Defaults

Set global parameter defaults to avoid repeating them on every call:

```python
import ferro_ta.config as config

config.set_default("timeperiod", 20)       # applies to all indicators
config.set_default("RSI.timeperiod", 14)   # RSI-specific override

from ferro_ta import RSI, SMA
# RSI(close) uses timeperiod=14; SMA(close) uses timeperiod=20

# Context manager for temporary overrides
with config.Config(timeperiod=5):
    result = SMA(close)   # timeperiod=5 inside this block
# back to timeperiod=20 after the block

config.reset()   # clear all custom defaults
```

---

## 🔌 Plugin Registry

Register and call any indicator (built-in or custom) by name. See the
`Writing a plugin <docs/plugins.rst>`_ doc for the plugin contract and a full example
(``examples/custom_indicator.py``).

```python
import numpy as np
from ferro_ta.registry import register, run, list_indicators

# Call a built-in by name
close = np.array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10])
sma = run("SMA", close, timeperiod=3)

# Register a custom indicator
def DOUBLE_RSI(close, timeperiod=14, smooth=3):
    import ferro_ta
    rsi = ferro_ta.RSI(close, timeperiod=timeperiod)
    return ferro_ta.SMA(rsi, timeperiod=smooth)

register("DOUBLE_RSI", DOUBLE_RSI)
result = run("DOUBLE_RSI", close, timeperiod=5, smooth=2)

# List all registered indicators
print(list_indicators()[:5])  # ['AD', 'ADOSC', 'ADX', 'ADXR', 'APO']
```

---

## 🛡️ Error Handling

ferro-ta provides a typed exception hierarchy with **error codes** and **actionable suggestions**:

```python
from ferro_ta import FerroTAError, FerroTAValueError, FerroTAInputError
from ferro_ta.exceptions import check_timeperiod, check_equal_length

# Catch any ferro-ta error
try:
    result = SMA(close, timeperiod=0)
except FerroTAValueError as e:
    print(e.code)        # "FTERR001"
    print(e.suggestion)  # "Set timeperiod=1 or higher."
    print(e)             # "[FTERR001] timeperiod must be >= 1, got 0\n  Suggestion: ..."

# Validate inputs before calling
check_equal_length(open=open_, close=close)   # raises FerroTAInputError (FTERR004) on mismatch
check_timeperiod(timeperiod)                   # raises FerroTAValueError (FTERR001) if < 1
```

Error code reference:

| Code | Exception | Meaning |
|------|-----------|---------|
| `FTERR001` | `FerroTAValueError` | Invalid parameter value |
| `FTERR002` | `FerroTAInputError` | Invalid input array |
| `FTERR003` | `FerroTAInputError` | Input array too short |
| `FTERR004` | `FerroTAInputError` | Mismatched array lengths |
| `FTERR005` | `FerroTAInputError` | Array contains NaN/Inf (strict mode) |
| `FTERR006` | `FerroTAValueError/InputError` | Rust-bridge error |

## 🔍 Observability & Logging

ferro-ta ships a lightweight logging module that integrates with Python's standard `logging` library:

```python
import ferro_ta

# Enable DEBUG-level logging (writes to stderr)
ferro_ta.enable_debug()
result = ferro_ta.SMA(close, timeperiod=20)
# DEBUG [ferro_ta] calling SMA(ndarray(252,) dtype=float64, timeperiod=20)
# DEBUG [ferro_ta] SMA → ndarray(252,)  [0.042 ms]
ferro_ta.disable_debug()

# Context manager: temporary debug output
with ferro_ta.debug_mode():
    ferro_ta.RSI(close, timeperiod=14)

# Call with automatic shape + timing log
result = ferro_ta.log_call(ferro_ta.ATR, high, low, close, timeperiod=14)

# Benchmark: returns {mean_ms, min_ms, max_ms, total_ms, n}
stats = ferro_ta.benchmark(ferro_ta.SMA, close, timeperiod=20, n=500)
print(f"SMA mean: {stats['mean_ms']:.3f} ms")

# Decorator: wrap any function with automatic logging
@ferro_ta.traced
def my_strategy(close):
    sma = ferro_ta.SMA(close, timeperiod=20)
    rsi = ferro_ta.RSI(close, timeperiod=14)
    return sma, rsi
```

## 🔎 API Discovery

```python
import ferro_ta

# List all 160+ indicators with metadata
all_indicators = ferro_ta.indicators()
print(len(all_indicators))  # 160+

# Filter by category
overlap = ferro_ta.indicators(category="overlap")
momentum = ferro_ta.indicators(category="momentum")

# Get parameter info for any indicator
d = ferro_ta.info(ferro_ta.SMA)
print(d["signature"])   # (close: ArrayLike, timeperiod: int = 30) -> NDArray[float64]
print(d["params"])      # {"close": {"default": None, ...}, "timeperiod": {"default": 30, ...}}

# By name string
d = ferro_ta.info("MACD")
```

See [`PLATFORMS.md`](PLATFORMS.md) for supported OS and Python versions.
See [`CHANGELOG.md`](CHANGELOG.md) and [`VERSIONING.md`](VERSIONING.md) for release notes and versioning policy.
See [`RELEASE.md`](RELEASE.md) for the step-by-step release playbook.
See [`examples/`](examples/) for Jupyter notebook examples (quickstart, streaming, backtesting, and more).

## 🗺️ Multi-Timeframe, Portfolio, and ML Features

### OHLCV Resampling and Multi-Timeframe API (`ferro_ta.resampling`)

```python
from ferro_ta.resampling import resample, volume_bars, multi_timeframe
from ferro_ta import RSI
import pandas as pd

# Resample 1-minute data to 5-minute bars (requires pandas)
df5 = resample(ohlcv_df, '5min')

# Volume bars (every 10,000 units of volume) — Rust backend
vbars = volume_bars(ohlcv_df, volume_threshold=10_000)

# Multi-timeframe RSI in one call
mtf = multi_timeframe(ohlcv_df, ['5min', '15min'], indicator=RSI,
                      indicator_kwargs={'timeperiod': 14})
# mtf = {'5min': array(...), '15min': array(...)}
```

### Tick Aggregation Pipeline (`ferro_ta.aggregation`)

```python
from ferro_ta.aggregation import aggregate_ticks, TickAggregator

# Tick bars, volume bars, time bars — all Rust-backed
tick_bars   = aggregate_ticks(ticks, rule='tick:100')
volume_bars = aggregate_ticks(ticks, rule='volume:500')
time_bars   = aggregate_ticks(ticks, rule='time:60')

# Class-based API
agg = TickAggregator(rule='tick:100')
bars = agg.aggregate(ticks)  # → pandas DataFrame or dict
```

### Strategy Expression DSL (`ferro_ta.dsl`)

```python
from ferro_ta.dsl import Strategy, evaluate

# Parse and evaluate expression strings
strat = Strategy("RSI(14) < 30 and close > SMA(20)")
signal = strat.evaluate({"close": close_arr})  # 1/0 integer array
```

### Signal Composition and Screening (`ferro_ta.signals`)

```python
from ferro_ta.signals import compose, screen, rank_signals

# Weighted combination of signal columns (Rust-backed)
score = compose(signals_df, weights=[0.4, 0.35, 0.25])

# Screening
top2 = screen({'AAPL': 0.8, 'MSFT': 0.9, 'GOOG': 0.5}, top_n=2)
# {'MSFT': 0.9, 'AAPL': 0.8}
```

### Portfolio Analytics (`ferro_ta.portfolio`)

```python
from ferro_ta.portfolio import correlation_matrix, portfolio_volatility, beta, drawdown

corr   = correlation_matrix(returns_df)               # Pearson corr matrix
vol    = portfolio_volatility(returns_df, weights,     # sqrt(w'Σw)
                              annualise=252)
b      = beta(asset_returns, benchmark_returns)        # OLS beta
rb     = beta(asset_returns, benchmark_returns,        # rolling beta
              window=30)
dd, mx = drawdown(equity_curve)                        # drawdown series + max
```

### Cross-Asset Relative Strength (`ferro_ta.cross_asset`)

```python
from ferro_ta.cross_asset import relative_strength, spread, ratio, zscore, rolling_beta

rs = relative_strength(asset_rets, bench_rets)  # cumulative return ratio
sp = spread(price_a, price_b, hedge=1.0)        # A - hedge * B
z  = zscore(sp, window=20)                       # rolling Z-score
```

### Feature Matrix for ML (`ferro_ta.features`)

```python
from ferro_ta.features import feature_matrix

fm = feature_matrix(ohlcv, [
    ('RSI', {'timeperiod': 14}),
    ('SMA', {'timeperiod': 20}),
    ('ATR', {'timeperiod': 14}),
], nan_policy='drop')
# fm is a pandas DataFrame with one column per indicator
# Use with sklearn: clf.fit(fm.values, labels)
```

### Charting and Visualization (`ferro_ta.viz`)

```python
from ferro_ta.viz import plot
from ferro_ta import RSI, SMA

fig = plot(ohlcv_df, indicators={'RSI(14)': RSI(close), 'SMA(20)': SMA(close)},
           backend='matplotlib', savefig='chart.png')
# Also supports 'plotly' backend for interactive charts
```

### Market Data Adapters (`ferro_ta.adapters`)

```python
from ferro_ta.adapters import CsvAdapter, InMemoryAdapter, register_adapter, DataAdapter

# Load from CSV
adapter = CsvAdapter('data.csv', index_col='date')
ohlcv = adapter.fetch()

# Custom adapter
class MyAdapter(DataAdapter):
    def fetch(self, **kwargs): return ...

register_adapter('mybroker', MyAdapter)
```

---

## 🤝 Community

[![GitHub Discussions](https://img.shields.io/badge/discussions-GitHub-blue?logo=github)](https://github.com/pratikbhadane24/ferro-ta/discussions)

- **GitHub Discussions** — Ask questions, share strategies, and request features in our [Discussions](https://github.com/pratikbhadane24/ferro-ta/discussions) space.  Categories: **Q&A**, **Ideas**, **Show & Tell**, **Announcements**.
- **Contributing**: See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, code style, and PR guidelines.
- **Code of Conduct**: All participants are expected to follow the [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
- **Governance**: Decision-making process and maintainer info in [`GOVERNANCE.md`](GOVERNANCE.md).
- **Roadmap**: Development plan in [`ROADMAP.md`](ROADMAP.md).
- **Security**: Responsible disclosure policy in [`SECURITY.md`](SECURITY.md).
- **Migration from TA-Lib**: Step-by-step guide in the [documentation](docs/migration_talib.rst).
- **Library Compatibility Guides** — drop-in migration instructions and cross-library test results:
  - [TA-Lib compatibility](docs/compatibility/talib.md) — full indicator mapping, API differences, and migration guide
  - [pandas-ta compatibility](docs/compatibility/pandas_ta.md) — indicator mapping, known differences, and comparison tests
  - [ta (Bukosabino) compatibility](docs/compatibility/ta.md) — indicator mapping, known differences, and comparison tests
  - [Tulipy compatibility](docs/compatibility/tulipy.md) — C99 Tulip Indicators: output truncation, memory requirements, signature mapping
  - [finta compatibility](docs/compatibility/finta.md) — pure-Pandas library: DataFrame requirements, speed comparison, migration guide

- **Cross-Library Benchmarks** — accuracy and speed comparison across all 6 libraries:
  - [Benchmarks README](benchmarks/README.md) — real timing results (µs), accuracy methodology, and known limitations
  - [Performance Roadmap](PERFORMANCE_ROADMAP.md) — plan to achieve 100x speedup over Tulipy

---

<div align="center">

**ferro-ta** — Built with ❤️ and Rust. [Star ⭐ on GitHub](https://github.com/pratikbhadane24/ferro-ta) to support the project.

</div>
