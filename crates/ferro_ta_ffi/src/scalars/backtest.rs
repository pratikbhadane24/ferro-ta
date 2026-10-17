//! Scalar C exports for the backtest settings vector.

crate::ffi_scalar_exports! {
    group: "backtest",
    /// Length of the settings vector this library understands (FT_BT_* indices are below it).
    ft_backtest_n_settings() -> [out_value: i64]
        = crate::kernels::backtest::N_SETTINGS as i64;
}
