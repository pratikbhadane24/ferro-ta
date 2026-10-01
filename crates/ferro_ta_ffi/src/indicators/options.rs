//! C exports for the array-shaped parts of `ferro_ta_core::options`.

crate::ffi_exports! {
    group: "options",
    /// Rolling IV rank.
    ft_iv_rank = options::iv::iv_rank(iv_series)[window: period] -> [out: f64];
    /// Rolling IV percentile.
    ft_iv_percentile = options::iv::iv_percentile(iv_series)[window: period] -> [out: f64];
    /// Rolling IV z-score.
    ft_iv_zscore = options::iv::iv_zscore(iv_series)[window: period] -> [out: f64];
    /// Rolling close-to-close realized volatility.
    ft_close_to_close_vol = options::realized_vol::close_to_close_vol(close)[window: period, trading_days: float]
        -> [out: f64];
    /// Rolling Parkinson high-low realized volatility estimator.
    ft_parkinson_vol = options::realized_vol::parkinson_vol(high, low)[window: period, trading_days: float]
        -> [out: f64];
    /// Rolling Garman-Klass OHLC realized volatility estimator.
    ft_garman_klass_vol = options::realized_vol::garman_klass_vol(open, high, low, close)[window: period, trading_days: float]
        -> [out: f64];
    /// Rolling Rogers-Satchell OHLC realized volatility estimator.
    ft_rogers_satchell_vol = options::realized_vol::rogers_satchell_vol(open, high, low, close)[window: period, trading_days: float]
        -> [out: f64];
    /// Rolling Yang-Zhang OHLC realized volatility estimator.
    ft_yang_zhang_vol = options::realized_vol::yang_zhang_vol(open, high, low, close)[window: period, trading_days: float]
        -> [out: f64];
}
