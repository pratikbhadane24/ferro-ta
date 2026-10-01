//! C exports for `ferro_ta_core::overlap`.

crate::ffi_exports! {
    group: "overlap",
    /// Compute Bollinger Bands, returning `(upper, middle, lower)`.
    ft_bbands = overlap::bbands(close)[timeperiod: period, nbdevup: float, nbdevdn: float, matype: matype]
        -> [out_upper: f64, out_middle: f64, out_lower: f64];
    /// MACD with fixed 12/26 periods.
    ft_macdfix = overlap::macdfix(close)[signalperiod: period]
        -> [out_macd: f64, out_signal: f64, out_hist: f64];
    /// MACD with configurable MA types for fast/slow/signal.
    ft_macdext = overlap::macdext(close)[fastperiod: period, fastmatype: matype, slowperiod: period, slowmatype: matype, signalperiod: period, signalmatype: matype]
        -> [out_macd: f64, out_signal: f64, out_hist: f64];
    /// Generic Moving Average. matype: 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA,
    ft_ma = overlap::ma(close)[timeperiod: period, matype: matype]
        -> [out: f64];
    /// Moving Average with Variable Period per bar.
    ft_mavp = overlap::mavp(close, periods)[minperiod: period, maxperiod: period, matype: matype]
        -> [out: f64];
    /// Compute the Exponential Moving Average (EMA).
    ft_ema = overlap::ema(close)[timeperiod: period]
        -> [out: f64];
    /// Double Exponential Moving Average: `2*EMA - EMA(EMA)`.
    ft_dema = overlap::dema(close)[timeperiod: period]
        -> [out: f64];
    /// Triple Exponential Moving Average: `3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))`.
    ft_tema = overlap::tema(close)[timeperiod: period]
        -> [out: f64];
    /// Tillson T3: 6 cascaded SMA-seeded EMAs with volume factor.
    ft_t3 = overlap::t3(close)[timeperiod: period, vfactor: float]
        -> [out: f64];
    /// Kaufman Adaptive Moving Average (TA-Lib).
    ft_kama = overlap::kama(close)[timeperiod: period]
        -> [out: f64];
    /// Compute the Moving Average Convergence/Divergence (MACD).
    ft_macd = overlap::macd(close)[fastperiod: period, slowperiod: period, signalperiod: period]
        -> [out_macd: f64, out_signal: f64, out_hist: f64];
    /// MESA Adaptive Moving Average. Returns `(mama, fama)`.
    ft_mama = overlap::mama(close)[fastlimit: float, slowlimit: float]
        -> [out_mama: f64, out_fama: f64];
    /// Midpoint: `(max(close) + min(close)) / 2` over rolling window.
    ft_midpoint = overlap::midpoint(close)[timeperiod: period]
        -> [out: f64];
    /// MidPrice: `(highest_high + lowest_low) / 2` over rolling window.
    ft_midprice = overlap::midprice(high, low)[timeperiod: period]
        -> [out: f64];
    /// Parabolic SAR.
    ft_sar = overlap::sar(high, low)[acceleration: float, maximum: float]
        -> [out: f64];
    /// Parabolic SAR Extended with configurable acceleration factors.
    ft_sarext = overlap::sarext(high, low)[startvalue: float, offsetonreverse: float, accelerationinitlong: float, accelerationlong: float, accelerationmaxlong: float, accelerationinitshort: float, accelerationshort: float, accelerationmaxshort: float]
        -> [out: f64];
    /// Compute the Simple Moving Average (SMA) over a rolling window.
    ft_sma = overlap::sma(close)[timeperiod: period]
        -> [out: f64];
    /// Triangular Moving Average (triangle-weighted).
    ft_trima = overlap::trima(close)[timeperiod: period]
        -> [out: f64];
    /// Compute the Weighted Moving Average (WMA).
    ft_wma = overlap::wma(close)[timeperiod: period]
        -> [out: f64];
}
