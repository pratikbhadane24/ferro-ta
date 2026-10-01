//! C exports for small array utilities from `signals`, `alerts`, `crypto`,
//! `chunked` and `backtest`.

crate::ffi_exports! {
    group: "utilities",
    /// Compute fractional rank of each element (1-based, ascending). Ties receive the average of their rank positions.
    ft_rank_values = signals::rank_values(x)[] -> [out: f64];
    /// Detect cross-over / cross-under events between two series.
    ft_check_cross = alerts::check_cross(fast, slow)[] -> [out: i8];
    /// Compute the cumulative PnL from funding rate payments.
    ft_funding_cumulative_pnl = crypto::funding_cumulative_pnl(position_size, funding_rate)[] -> [out: f64];
    /// Forward-fill NaN values in a 1-D array. Leading NaN values are preserved until the first non-NaN value appears.
    ft_forward_fill_nan = chunked::forward_fill_nan(values)[] -> [out: f64];
    /// RSI threshold strategy: +1 when RSI <= oversold, -1 when RSI >= overbought, 0 otherwise.
    ft_rsi_threshold_signals = backtest::rsi_threshold_signals(close)[timeperiod: period, oversold: float, overbought: float]
        -> [out: f64];
}
