//! C exports for the array-shaped parts of `ferro_ta_core::portfolio`.

crate::ffi_exports! {
    group: "portfolio",
    /// Compute rolling beta of `asset` vs `benchmark` over a sliding `window`.
    ft_rolling_beta = portfolio::rolling_beta(asset, benchmark)[window: period 2] -> [out: f64];
    /// Compute relative strength of an asset vs a benchmark.
    ft_relative_strength = portfolio::relative_strength(asset_returns, benchmark_returns)[] -> [out: f64];
    /// Compute the spread between two series: a - hedge * b.
    ft_spread = portfolio::spread(a, b)[hedge: float] -> [out: f64];
    /// Compute the ratio between two series: a / b.
    ft_ratio = portfolio::ratio(a, b)[] -> [out: f64];
    /// Compute the rolling Z-score of a 1-D series.
    ft_zscore_series = portfolio::zscore_series(x)[window: period 2] -> [out: f64];
}
