//! C exports for `ferro_ta_core::math`.

crate::ffi_exports! {
    group: "math",
    /// Compute the rolling sum over `timeperiod` bars.
    ft_sum = math::sum(real)[timeperiod: period]
        -> [out: f64];
    /// Compute the rolling maximum over `timeperiod` bars.
    ft_max = math::max(real)[timeperiod: period]
        -> [out: f64];
    /// Compute the rolling minimum over `timeperiod` bars.
    ft_min = math::min(real)[timeperiod: period]
        -> [out: f64];
    /// Compute the sliding maximum over `timeperiod` bars in O(n) time.
    ft_sliding_max = math::sliding_max(real)[timeperiod: period]
        -> [out: f64];
    /// Compute the sliding minimum over `timeperiod` bars in O(n) time.
    ft_sliding_min = math::sliding_min(real)[timeperiod: period]
        -> [out: f64];
    /// Element-wise addition of two arrays.
    ft_add = math::add(a, b)[]
        -> [out: f64];
    /// Element-wise subtraction of two arrays.
    ft_sub = math::sub(a, b)[]
        -> [out: f64];
    /// Element-wise multiplication of two arrays.
    ft_mult = math::mult(a, b)[]
        -> [out: f64];
    /// Element-wise division of two arrays (NaN where b=0).
    ft_div = math::div(a, b)[]
        -> [out: f64];
}
