//! Scalar C exports sizing the backtest kernels.

use crate::status::Status;
use ferro_ta_core::backtest;

crate::ffi_scalar_exports! {
    group: "backtest",
    /// Number of folds ft_walk_forward_indices writes for these arguments (anchored must be 0 or 1).
    ft_walk_forward_n_folds(n_bars: count, train_bars: period, test_bars: period, anchored: count, step_bars: count) -> [out_value: i64]
        = {
            let anchored = match anchored {
                0 => false,
                1 => true,
                _ => return Err(Status::InvalidParam),
            };
            backtest::walk_forward_indices(n_bars, train_bars, test_bars, anchored, step_bars)
                .map_err(|_| Status::InvalidParam)?
                .len() as i64
        };
    /// Number of benchmark-relative metrics ft_performance_metrics writes (FT_BENCH_* indices are below it).
    ft_benchmark_n_metrics() -> [out_value: i64]
        = crate::kernels::backtest::N_BENCHMARK_METRICS as i64;
    /// Length of the settings vector this library understands (FT_BT_* indices are below it).
    ft_backtest_n_settings() -> [out_value: i64]
        = crate::kernels::backtest::N_SETTINGS as i64;
}
