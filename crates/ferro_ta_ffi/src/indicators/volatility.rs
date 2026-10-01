//! C exports for `ferro_ta_core::volatility`.

crate::ffi_exports! {
    group: "volatility",
    /// Compute the Average True Range (ATR), Wilder smoothed (TA-Lib compatible).
    ft_atr = volatility::atr(high, low, close)[timeperiod: period]
        -> [out: f64];
    /// Compute the True Range for each bar.
    ft_trange = volatility::trange(high, low, close)[]
        -> [out: f64];
    /// Normalized Average True Range: `ATR / close * 100`.
    ft_natr = volatility::natr(high, low, close)[timeperiod: period]
        -> [out: f64];
}
