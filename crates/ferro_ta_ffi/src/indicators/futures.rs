//! C exports for the array-shaped parts of `ferro_ta_core::futures`.

crate::ffi_exports! {
    group: "futures",
    /// Weighted stitching using next-contract weights in [0, 1].
    ft_weighted_continuous = futures::roll::weighted_continuous(front, next, next_weights)[] -> [out: f64];
    /// Back-adjusted continuous series using the roll date implied by the weights.
    ft_back_adjusted_continuous = futures::roll::back_adjusted_continuous(front, next, next_weights)[] -> [out: f64];
    /// Ratio-adjusted continuous series using the roll date implied by the weights.
    ft_ratio_adjusted_continuous = futures::roll::ratio_adjusted_continuous(front, next, next_weights)[] -> [out: f64];
}
