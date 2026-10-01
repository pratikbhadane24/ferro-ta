//! C exports for `ferro_ta_core::momentum`.

crate::ffi_exports! {
    group: "momentum",
    /// Compute the Plus Directional Movement (+DM), Wilder smoothed.
    ft_plus_dm = momentum::plus_dm(high, low)[timeperiod: period]
        -> [out: f64];
    /// Compute the Minus Directional Movement (-DM), Wilder smoothed.
    ft_minus_dm = momentum::minus_dm(high, low)[timeperiod: period]
        -> [out: f64];
    /// Compute the Plus Directional Indicator (+DI), Wilder smoothed.
    ft_plus_di = momentum::plus_di(high, low, close)[timeperiod: period]
        -> [out: f64];
    /// Compute the Minus Directional Indicator (-DI), Wilder smoothed.
    ft_minus_di = momentum::minus_di(high, low, close)[timeperiod: period]
        -> [out: f64];
    /// Compute the Directional Movement Index (DX).
    ft_dx = momentum::dx(high, low, close)[timeperiod: period]
        -> [out: f64];
    /// Compute the Average Directional Movement Index (ADX).
    ft_adx = momentum::adx(high, low, close)[timeperiod: period]
        -> [out: f64];
    /// Compute the ADX Rating (ADXR).
    ft_adxr = momentum::adxr(high, low, close)[timeperiod: period]
        -> [out: f64];
    /// Aroon indicator. Returns `(aroon_down, aroon_up)`.
    ft_aroon = momentum::aroon(high, low)[timeperiod: period]
        -> [out_aroondown: f64, out_aroonup: f64];
    /// Aroon Oscillator: `aroon_up - aroon_down`.
    ft_aroonosc = momentum::aroonosc(high, low)[timeperiod: period]
        -> [out: f64];
    /// Absolute Price Oscillator: `fast MA - slow MA`.
    ft_apo = momentum::apo(close)[fastperiod: period, slowperiod: period, matype: matype]
        -> [out: f64] if fastperiod < slowperiod;
    /// Percentage Price Oscillator: `(fast MA - slow MA) / slow MA * 100`. Returns `(ppo_line, signal_line, histogram)`.
    ft_ppo = momentum::ppo(close)[fastperiod: period, slowperiod: period, signalperiod: period, matype: matype]
        -> [out_ppo: f64, out_signal: f64, out_hist: f64] if fastperiod < slowperiod;
    /// TRIX: 1-period rate of change of triple-smoothed EMA.
    ft_trix = momentum::trix(close)[timeperiod: period]
        -> [out: f64];
    /// Williams %R: `-100 * (HH - close) / (HH - LL)` over the window. Returns values in `[-100, 0]`.
    ft_willr = momentum::willr(high, low, close)[timeperiod: period]
        -> [out: f64];
    /// Commodity Channel Index: `(tp - SMA(tp)) / (0.015 * MAD)`.
    ft_cci = momentum::cci(high, low, close)[timeperiod: period]
        -> [out: f64];
    /// Balance of Power: `(close - open) / (high - low)`.
    ft_bop = momentum::bop(open, high, low, close)[]
        -> [out: f64];
    /// Compute the Momentum indicator: `close[i] - close[i - timeperiod]`.
    ft_mom = momentum::mom(close)[timeperiod: period]
        -> [out: f64];
    /// Rate of Change: `(close[i] - close[i-p]) / close[i-p] * 100`.
    ft_roc = momentum::roc(close)[timeperiod: period]
        -> [out: f64];
    /// Rate of Change Percentage: `(close[i] - close[i-p]) / close[i-p]`.
    ft_rocp = momentum::rocp(close)[timeperiod: period]
        -> [out: f64];
    /// Rate of Change Ratio: `close[i] / close[i-p]`.
    ft_rocr = momentum::rocr(close)[timeperiod: period]
        -> [out: f64];
    /// Rate of Change Ratio x 100: `close[i] / close[i-p] * 100`.
    ft_rocr100 = momentum::rocr100(close)[timeperiod: period]
        -> [out: f64];
    /// Compute the Relative Strength Index (RSI).
    ft_rsi = momentum::rsi(close)[timeperiod: period]
        -> [out: f64];
    /// Chande Momentum Oscillator: `100 * (gains - losses) / (gains + losses)`.
    ft_cmo = momentum::cmo(close)[timeperiod: period]
        -> [out: f64];
    /// Compute the Stochastic Oscillator (TA-Lib compatible).
    ft_stoch = momentum::stoch(high, low, close)[fastk_period: period, slowk_period: period, slowk_matype: matype, slowd_period: period, slowd_matype: matype]
        -> [out_slowk: f64, out_slowd: f64];
    /// Fast Stochastic. Returns `(fastk, fastd)`.
    ft_stochf = momentum::stochf(high, low, close)[fastk_period: period, fastd_period: period, fastd_matype: matype]
        -> [out_fastk: f64, out_fastd: f64];
    /// Stochastic RSI. Returns `(fastk, fastd)`.
    ft_stochrsi = momentum::stochrsi(close)[timeperiod: period, fastk_period: period, fastd_period: period, fastd_matype: matype]
        -> [out_fastk: f64, out_fastd: f64];
    /// Ultimate Oscillator: weighted average of buying pressure over three periods.
    ft_ultosc = momentum::ultosc(high, low, close)[timeperiod1: period, timeperiod2: period, timeperiod3: period]
        -> [out: f64];
}
