//! C exports for `ferro_ta_core::regime`.

crate::ffi_exports! {
    group: "regime",
    /// Label each bar as trend (1) or range (0) based on ADX level.
    ft_regime_adx = regime::regime_adx(adx)[threshold: float] -> [out: i8];
    /// Label each bar as trend (1) or range (0) using ADX + ATR-ratio rule.
    ft_regime_combined = regime::regime_combined(adx, atr, close)[adx_threshold: float, atr_pct_threshold: float]
        -> [out: i8];
    /// Detect structural breaks using a CUSUM (cumulative sum) approach.
    ft_detect_breaks_cusum = regime::detect_breaks_cusum(series)[window: period 2, threshold: float, slack: float]
        -> [out: i8];
    /// Detect volatility regime breaks using rolling variance ratio.
    ft_rolling_variance_break = regime::rolling_variance_break(series)[short_window: period 2, long_window: period, threshold: float]
        -> [out: i8] if long_window > short_window;
}
