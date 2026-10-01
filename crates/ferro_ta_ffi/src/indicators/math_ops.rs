//! C exports for `ferro_ta_core::math_ops`.

crate::ffi_exports! {
    group: "math_ops",
    /// Rolling sum over `timeperiod` bars using an O(n) sliding window. Leading `timeperiod - 1` values are NaN.
    ft_rolling_sum = math_ops::rolling_sum(real)[timeperiod: period]
        -> [out: f64];
    /// Rolling maximum over `timeperiod` bars (O(n) monotonic deque). Delegates to `math::sliding_max`.
    ft_rolling_max = math_ops::rolling_max(real)[timeperiod: period]
        -> [out: f64];
    /// Rolling minimum over `timeperiod` bars (O(n) monotonic deque). Delegates to `math::sliding_min`.
    ft_rolling_min = math_ops::rolling_min(real)[timeperiod: period]
        -> [out: f64];
    /// Index of rolling maximum over `timeperiod` bars. Returns 0-based index. During warmup the value is `-1`.
    ft_rolling_maxindex = math_ops::rolling_maxindex(real)[timeperiod: period]
        -> [out: i64];
    /// Index of rolling minimum over `timeperiod` bars. Returns 0-based index. During warmup the value is `-1`.
    ft_rolling_minindex = math_ops::rolling_minindex(real)[timeperiod: period]
        -> [out: i64];
}
