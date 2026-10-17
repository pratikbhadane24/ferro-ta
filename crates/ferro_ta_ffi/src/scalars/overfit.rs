//! Scalar C exports for `ferro_ta_core::overfit` and `ferro_ta_core::sweep`.

use crate::status::Status;
use ferro_ta_core::overfit;

crate::ffi_scalar_exports! {
    group: "overfit",
    /// Expected maximum per-period Sharpe among `n_trials` zero-skill trials whose Sharpe estimates have variance `trial_sharpe_variance`.
    ft_expected_max_sharpe(n_trials: period, trial_sharpe_variance: float) -> [out_value: f64]
        = overfit::expected_max_sharpe(n_trials, trial_sharpe_variance)
            .map_err(|_| Status::InvalidParam)?;
    /// Number of IS/OOS combinations `ft_pbo_cscv` evaluates: C(n_splits, n_splits/2). n_splits must be even and at most 20.
    ft_pbo_n_combinations(n_splits: period 2) -> [out_value: i64]
        = overfit::cscv_combinations(n_splits).map_err(|_| Status::InvalidParam)? as i64;
    /// Number of parameters (columns of `combos`) a built-in sweep strategy takes, in canonical order: RSI_THRESHOLD (timeperiod, oversold, overbought), SMA_CROSSOVER (fast, slow), MACD_CROSSOVER (fastperiod, slowperiod, signalperiod).
    ft_sweep_strategy_n_params(strategy: sweep_strategy) -> [out_value: i64]
        = strategy.param_names().len() as i64;
    /// Number of columns in the sweep metric matrices (FT_METRIC_* indices are below it). Pass it as `n_metric_cols` so a caller built against an older header can never under-allocate.
    ft_sweep_n_metrics() -> [out_value: i64]
        = ferro_ta_core::backtest::BacktestMetrics::N_SCALAR_FIELDS as i64;
}
