# TA-Lib Compatibility

`ferro-ta` covers **100% of TA-Lib's function set** (`162+` indicators). This
file keeps the full GitHub-facing parity matrix in one place so the root
`README.md` can stay product-focused.

See also:

- [docs/migration_talib.rst](docs/migration_talib.rst)
- [docs/compatibility/talib.md](docs/compatibility/talib.md)
- [docs/support_matrix.rst](docs/support_matrix.rst)

## Legend


| Symbol   | Meaning                                                                                                |
| -------- | ------------------------------------------------------------------------------------------------------ |
| ✅ Exact  | Values match TA-Lib to floating-point precision                                                        |
| ✅ Close  | Values match after a short convergence window (EMA-seed difference)                                    |
| ⚠️ Corr  | Strong correlation (> 0.95) but not numerically identical (Wilder smoothing seed or algorithm variant) |
| ⚠️ Shape | Same output shape / NaN structure; values differ due to algorithm variant                              |
| ❌        | Not yet implemented                                                                                    |


## Overlap Studies


| TA-Lib Function | ferro-ta | Accuracy | Notes                                                 |
| --------------- | -------- | -------- | ----------------------------------------------------- |
| `BBANDS`        | ✅        | ✅ Exact  | Bollinger Bands                                       |
| `DEMA`          | ✅        | ✅ Close  | Double EMA; converges after ~20 bars                  |
| `EMA`           | ✅        | ✅ Close  | Exponential Moving Average; converges after ~20 bars  |
| `KAMA`          | ✅        | ✅ Exact  | Kaufman Adaptive MA (values match after seed bar)     |
| `MA`            | ✅        | ✅ Exact  | Moving average (generic, type-selectable)             |
| `MAMA`          | ✅        | ⚠️ Corr  | MESA Adaptive MA                                      |
| `MAVP`          | ✅        | ✅ Exact  | MA with variable period                               |
| `MIDPOINT`      | ✅        | ✅ Exact  | Midpoint over period                                  |
| `MIDPRICE`      | ✅        | ✅ Exact  | Midpoint price over period                            |
| `SAR`           | ✅        | ⚠️ Shape | Parabolic SAR (same shape; reversal history diverges) |
| `SAREXT`        | ✅        | ⚠️ Shape | Parabolic SAR Extended                                |
| `SMA`           | ✅        | ✅ Exact  | Simple Moving Average                                 |
| `T3`            | ✅        | ✅ Close  | Triple Exponential MA (T3); converges after ~50 bars  |
| `TEMA`          | ✅        | ✅ Close  | Triple EMA; converges after ~20 bars                  |
| `TRIMA`         | ✅        | ✅ Exact  | Triangular Moving Average                             |
| `WMA`           | ✅        | ✅ Exact  | Weighted Moving Average                               |


## Momentum Indicators


| TA-Lib Function | ferro-ta | Accuracy | Notes                                                                        |
| --------------- | -------- | -------- | ---------------------------------------------------------------------------- |
| `ADX`           | ✅        | ✅ Close  | Avg Directional Movement Index (TA-Lib Wilder sum-seeding)                   |
| `ADXR`          | ✅        | ✅ Close  | ADX Rating (inherits ADX; TA-Lib seeding)                                    |
| `APO`           | ✅        | ✅ Close  | Absolute Price Oscillator (EMA-based)                                        |
| `AROON`         | ✅        | ✅ Exact  | Aroon Up/Down                                                                |
| `AROONOSC`      | ✅        | ✅ Exact  | Aroon Oscillator                                                             |
| `BOP`           | ✅        | ✅ Exact  | Balance Of Power                                                             |
| `CCI`           | ✅        | ✅ Exact  | Commodity Channel Index (TA-Lib-compatible MAD formula)                      |
| `CMO`           | ✅        | ✅ Close  | Chande Momentum Oscillator (rolling window, TA-Lib-compatible)               |
| `DX`            | ✅        | ✅ Close  | Directional Movement Index (TA-Lib Wilder sum-seeding)                       |
| `MACD`          | ✅        | ✅ Close  | MACD (EMA-based; converges after ~30 bars)                                   |
| `MACDEXT`       | ✅        | ✅ Close  | MACD with controllable MA type (EMA-based; converges)                        |
| `MACDFIX`       | ✅        | ✅ Close  | MACD Fixed 12/26 (EMA-based; converges)                                      |
| `MFI`           | ✅        | ✅ Exact  | Money Flow Index                                                             |
| `MINUS_DI`      | ✅        | ✅ Close  | Minus Directional Indicator (TA-Lib Wilder sum-seeding)                      |
| `MINUS_DM`      | ✅        | ✅ Close  | Minus Directional Movement (TA-Lib Wilder sum-seeding)                       |
| `MOM`           | ✅        | ✅ Exact  | Momentum                                                                     |
| `PLUS_DI`       | ✅        | ✅ Close  | Plus Directional Indicator (TA-Lib Wilder sum-seeding)                       |
| `PLUS_DM`       | ✅        | ✅ Close  | Plus Directional Movement (TA-Lib Wilder sum-seeding)                        |
| `PPO`           | ✅        | ✅ Close  | Percentage Price Oscillator (EMA-based)                                      |
| `ROC`           | ✅        | ✅ Exact  | Rate of Change                                                               |
| `ROCP`          | ✅        | ✅ Exact  | Rate of Change Percentage                                                    |
| `ROCR`          | ✅        | ✅ Exact  | Rate of Change Ratio                                                         |
| `ROCR100`       | ✅        | ✅ Exact  | Rate of Change Ratio × 100                                                   |
| `RSI`           | ✅        | ✅ Close  | Relative Strength Index (TA-Lib Wilder seeding; converges after ~1 seed bar) |
| `STOCH`         | ✅        | ✅ Close  | Stochastic (TA-Lib-compatible SMA smoothing for slowk and slowd)             |
| `STOCHF`        | ✅        | ✅ Exact  | Stochastic Fast (%K exact; %D NaN offset ±2)                                 |
| `STOCHRSI`      | ✅        | ✅ Close  | Stochastic RSI (TA-Lib-compatible; SMA fastd, Wilder-seeded RSI)             |
| `TRIX`          | ✅        | ✅ Close  | 1-day ROC of Triple EMA (EMA-based; converges)                               |
| `ULTOSC`        | ✅        | ✅ Exact  | Ultimate Oscillator                                                          |
| `WILLR`         | ✅        | ✅ Exact  | Williams' %R                                                                 |


## Volume Indicators


| TA-Lib Function | ferro-ta | Accuracy | Notes                                                              |
| --------------- | -------- | -------- | ------------------------------------------------------------------ |
| `AD`            | ✅        | ✅ Exact  | Chaikin A/D Line                                                   |
| `ADOSC`         | ✅        | ✅ Exact  | Chaikin A/D Oscillator                                             |
| `OBV`           | ✅        | ✅ Exact  | On Balance Volume (increments identical; constant offset at bar 0) |


## Volatility Indicators


| TA-Lib Function | ferro-ta | Accuracy | Notes                                                                   |
| --------------- | -------- | -------- | ----------------------------------------------------------------------- |
| `ATR`           | ✅        | ✅ Close  | Average True Range (TA-Lib Wilder seeding; matches from bar timeperiod) |
| `NATR`          | ✅        | ✅ Close  | Normalized ATR (TA-Lib Wilder seeding)                                  |
| `TRANGE`        | ✅        | ✅ Exact  | True Range (bar 0 differs; all others identical)                        |


## Cycle Indicators


| TA-Lib Function | ferro-ta | Accuracy | Notes                                                      |
| --------------- | -------- | -------- | ---------------------------------------------------------- |
| `HT_DCPERIOD`   | ✅        | ⚠️ Shape | Hilbert Transform Dominant Cycle Period (Ehlers algorithm) |
| `HT_DCPHASE`    | ✅        | ⚠️ Shape | Hilbert Transform Dominant Cycle Phase                     |
| `HT_PHASOR`     | ✅        | ⚠️ Shape | Hilbert Transform Phasor Components (inphase, quadrature)  |
| `HT_SINE`       | ✅        | ⚠️ Shape | Hilbert Transform SineWave (sine, leadsine)                |
| `HT_TRENDLINE`  | ✅        | ⚠️ Shape | Hilbert Transform Instantaneous Trendline                  |
| `HT_TRENDMODE`  | ✅        | ⚠️ Shape | Hilbert Transform Trend vs Cycle Mode (1=trend, 0=cycle)   |


## Price Transformations


| TA-Lib Function | ferro-ta | Accuracy | Notes                |
| --------------- | -------- | -------- | -------------------- |
| `AVGPRICE`      | ✅        | ✅ Exact  | Average Price        |
| `MEDPRICE`      | ✅        | ✅ Exact  | Median Price         |
| `TYPPRICE`      | ✅        | ✅ Exact  | Typical Price        |
| `WCLPRICE`      | ✅        | ✅ Exact  | Weighted Close Price |


## Statistic Functions


| TA-Lib Function       | ferro-ta | Accuracy | Notes                                                       |
| --------------------- | -------- | -------- | ----------------------------------------------------------- |
| `BETA`                | ✅        | ✅ Close  | Beta coefficient (returns-based regression matching TA-Lib) |
| `CORREL`              | ✅        | ✅ Exact  | Pearson Correlation Coefficient                             |
| `LINEARREG`           | ✅        | ✅ Exact  | Linear Regression                                           |
| `LINEARREG_ANGLE`     | ✅        | ✅ Exact  | Linear Regression Angle                                     |
| `LINEARREG_INTERCEPT` | ✅        | ✅ Exact  | Linear Regression Intercept                                 |
| `LINEARREG_SLOPE`     | ✅        | ✅ Exact  | Linear Regression Slope                                     |
| `STDDEV`              | ✅        | ✅ Exact  | Standard Deviation                                          |
| `TSF`                 | ✅        | ✅ Exact  | Time Series Forecast                                        |
| `VAR`                 | ✅        | ✅ Exact  | Variance                                                    |


## Pattern Recognition

`ferro-ta` implements all 61 candlestick patterns. All return the same
`{-100, 0, 100}` convention as TA-Lib. Pattern thresholds may differ slightly
from the full TA-Lib implementation.


| TA-Lib Function       | ferro-ta | Notes                                               |
| --------------------- | -------- | --------------------------------------------------- |
| `CDL2CROWS`           | ✅        | Two Crows                                           |
| `CDL3BLACKCROWS`      | ✅        | Three Black Crows                                   |
| `CDL3INSIDE`          | ✅        | Three Inside Up/Down                                |
| `CDL3LINESTRIKE`      | ✅        | Three-Line Strike                                   |
| `CDL3OUTSIDE`         | ✅        | Three Outside Up/Down                               |
| `CDL3STARSINSOUTH`    | ✅        | Three Stars In The South                            |
| `CDL3WHITESOLDIERS`   | ✅        | Three Advancing White Soldiers                      |
| `CDLABANDONEDBABY`    | ✅        | Abandoned Baby                                      |
| `CDLADVANCEBLOCK`     | ✅        | Advance Block                                       |
| `CDLBELTHOLD`         | ✅        | Belt-hold                                           |
| `CDLBREAKAWAY`        | ✅        | Breakaway                                           |
| `CDLCLOSINGMARUBOZU`  | ✅        | Closing Marubozu                                    |
| `CDLCONCEALBABYSWALL` | ✅        | Concealing Baby Swallow                             |
| `CDLCOUNTERATTACK`    | ✅        | Counterattack                                       |
| `CDLDARKCLOUDCOVER`   | ✅        | Dark Cloud Cover                                    |
| `CDLDOJI`             | ✅        | Doji                                                |
| `CDLDOJISTAR`         | ✅        | Doji Star                                           |
| `CDLDRAGONFLYDOJI`    | ✅        | Dragonfly Doji                                      |
| `CDLENGULFING`        | ✅        | Engulfing Pattern                                   |
| `CDLEVENINGDOJISTAR`  | ✅        | Evening Doji Star                                   |
| `CDLEVENINGSTAR`      | ✅        | Evening Star                                        |
| `CDLGAPSIDESIDEWHITE` | ✅        | Up/Down-gap side-by-side white lines                |
| `CDLGRAVESTONEDOJI`   | ✅        | Gravestone Doji                                     |
| `CDLHAMMER`           | ✅        | Hammer                                              |
| `CDLHANGINGMAN`       | ✅        | Hanging Man                                         |
| `CDLHARAMI`           | ✅        | Harami Pattern                                      |
| `CDLHARAMICROSS`      | ✅        | Harami Cross Pattern                                |
| `CDLHIGHWAVE`         | ✅        | High-Wave Candle                                    |
| `CDLHIKKAKE`          | ✅        | Hikkake Pattern                                     |
| `CDLHIKKAKEMOD`       | ✅        | Modified Hikkake Pattern                            |
| `CDLHOMINGPIGEON`     | ✅        | Homing Pigeon                                       |
| `CDLIDENTICAL3CROWS`  | ✅        | Identical Three Crows                               |
| `CDLINNECK`           | ✅        | In-Neck Pattern                                     |
| `CDLINVERTEDHAMMER`   | ✅        | Inverted Hammer                                     |
| `CDLKICKING`          | ✅        | Kicking                                             |
| `CDLKICKINGBYLENGTH`  | ✅        | Kicking by the longer Marubozu                      |
| `CDLLADDERBOTTOM`     | ✅        | Ladder Bottom                                       |
| `CDLLONGLEGGEDDOJI`   | ✅        | Long Legged Doji                                    |
| `CDLLONGLINE`         | ✅        | Long Line Candle                                    |
| `CDLMARUBOZU`         | ✅        | Marubozu                                            |
| `CDLMATCHINGLOW`      | ✅        | Matching Low                                        |
| `CDLMATHOLD`          | ✅        | Mat Hold                                            |
| `CDLMORNINGDOJISTAR`  | ✅        | Morning Doji Star                                   |
| `CDLMORNINGSTAR`      | ✅        | Morning Star                                        |
| `CDLONNECK`           | ✅        | On-Neck Pattern                                     |
| `CDLPIERCING`         | ✅        | Piercing Pattern                                    |
| `CDLRICKSHAWMAN`      | ✅        | Rickshaw Man                                        |
| `CDLRISEFALL3METHODS` | ✅        | Rising/Falling Three Methods                        |
| `CDLSEPARATINGLINES`  | ✅        | Separating Lines                                    |
| `CDLSHOOTINGSTAR`     | ✅        | Shooting Star                                       |
| `CDLSHORTLINE`        | ✅        | Short Line Candle                                   |
| `CDLSPINNINGTOP`      | ✅        | Spinning Top                                        |
| `CDLSTALLEDPATTERN`   | ✅        | Stalled Pattern                                     |
| `CDLSTICKSANDWICH`    | ✅        | Stick Sandwich                                      |
| `CDLTAKURI`           | ✅        | Takuri (Dragonfly Doji with very long lower shadow) |
| `CDLTASUKIGAP`        | ✅        | Tasuki Gap                                          |
| `CDLTHRUSTING`        | ✅        | Thrusting Pattern                                   |
| `CDLTRISTAR`          | ✅        | Tristar Pattern                                     |
| `CDLUNIQUE3RIVER`     | ✅        | Unique 3 River                                      |
| `CDLUPSIDEGAP2CROWS`  | ✅        | Upside Gap Two Crows                                |
| `CDLXSIDEGAP3METHODS` | ✅        | Upside/Downside Gap Three Methods                   |


## Math Operators / Math Transforms

`ferro-ta` provides TA-Lib-compatible wrappers for all arithmetic and
math-transform functions. Rolling functions (`SUM`, `MAX`, `MIN`) produce `NaN`
for the first `timeperiod - 1` bars.


| TA-Lib Function          | ferro-ta | Notes                         |
| ------------------------ | -------- | ----------------------------- |
| `ADD`                    | ✅        | Element-wise addition         |
| `SUB`                    | ✅        | Element-wise subtraction      |
| `MULT`                   | ✅        | Element-wise multiplication   |
| `DIV`                    | ✅        | Element-wise division         |
| `SUM`                    | ✅        | Rolling sum over *timeperiod* |
| `MAX` / `MAXINDEX`       | ✅        | Rolling maximum / index       |
| `MIN` / `MININDEX`       | ✅        | Rolling minimum / index       |
| `ACOS` / `ASIN` / `ATAN` | ✅        | Arc trig transforms           |
| `CEIL` / `FLOOR`         | ✅        | Round up / down               |
| `COS` / `SIN` / `TAN`    | ✅        | Trig transforms               |
| `COSH` / `SINH` / `TANH` | ✅        | Hyperbolic transforms         |
| `EXP` / `LN` / `LOG10`   | ✅        | Exponential / log transforms  |
| `SQRT`                   | ✅        | Square root                   |


## Implementation Coverage Summary


| Category                    | Implemented | Not Implemented |
| --------------------------- | ----------- | --------------- |
| Overlap Studies             | 19          | 0               |
| Momentum Indicators         | 28          | 0               |
| Volume Indicators           | 3           | 0               |
| Volatility Indicators       | 3           | 0               |
| Cycle Indicators            | 6           | 0               |
| Price Transforms            | 4           | 0               |
| Statistic Functions         | 9           | 0               |
| Pattern Recognition         | 61          | 0               |
| Math Operators / Transforms | 24          | 0               |
| Extended Indicators         | 10          | -               |
| Streaming Classes           | 9           | -               |
| **Total**                   | **162+**    | **0**           |


> `ferro-ta` implements 100% of TA-Lib's function set. NaN values are placed
> at the beginning of each output array for the warmup period.

