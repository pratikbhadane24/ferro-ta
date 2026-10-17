//! Kernel C exports for `ferro_ta_core::sweep` (parameter sweeps).
//!
//! Metric rows have `n_metric_cols` columns (must equal
//! `ft_sweep_n_metrics()`), indexed by the `FT_METRIC_*` constants. Sweeps run
//! on all cores (Rayon).

use super::invalid;
use crate::marshal::{check_output, elems, kernel_in, kernel_out};
use crate::status::Status;
use ferro_ta_core::backtest::BacktestMetrics;
use ferro_ta_core::sweep::{self as core_sweep, grid, SweepConfig, SweepOutput};

const N_METRICS: usize = BacktestMetrics::N_SCALAR_FIELDS;

fn config(
    commission_per_trade: f64,
    slippage_bps: f64,
    initial_capital: f64,
    periods_per_year: f64,
    risk_free_rate: f64,
) -> SweepConfig {
    SweepConfig {
        commission_per_trade,
        slippage_bps,
        initial_capital,
        periods_per_year,
        risk_free_rate,
        parallel: true,
    }
}

/// Spread `out` (columns = kept combos) into full `(n_bars, n_combos)` /
/// `(n_combos, N_METRICS)` buffers, NaN for skipped combos.
fn scatter(out: &SweepOutput, kept: &[usize], n_combos: usize) -> (Vec<f64>, Vec<f64>) {
    let n_kept = kept.len();
    let mut returns = vec![f64::NAN; out.n_bars * n_combos];
    for i in 0..out.n_bars {
        for (j, &c) in kept.iter().enumerate() {
            returns[i * n_combos + c] = out.returns[i * n_kept + j];
        }
    }
    let matrix = out.metrics_matrix();
    let mut metrics = vec![f64::NAN; n_combos * N_METRICS];
    for (j, &c) in kept.iter().enumerate() {
        metrics[c * N_METRICS..(c + 1) * N_METRICS]
            .copy_from_slice(&matrix[j * N_METRICS..(j + 1) * N_METRICS]);
    }
    (returns, metrics)
}

crate::ffi_kernel_exports! {
    group: "sweep",
    /// Backtest every column of a row-major (n_bars, n_combos) signal matrix against `close` (same semantics as backtest_core; NaN signals hold the position). Writes per-bar strategy returns (n_bars, n_combos) and a (n_combos, n_metric_cols) metric matrix indexed by FT_METRIC_*; n_metric_cols must equal ft_sweep_n_metrics().
    ft_sweep_signals(
        close: in f64 [n_bars],
        n_bars: size usize,
        signals: in f64 [n_bars, n_combos],
        n_combos: size usize,
        commission_per_trade: param float,
        slippage_bps: param float,
        initial_capital: param float,
        periods_per_year: param float,
        risk_free_rate: param float,
        n_metric_cols: size usize <- ft_sweep_n_metrics(),
        out_returns: out f64 [n_bars, n_combos],
        out_metrics: out f64 [n_combos, n_metric_cols],
    ) => {
        if n_metric_cols != N_METRICS {
            return Err(Status::InvalidParam);
        }
        let n_returns = elems(&[n_bars, n_combos])?;
        let n_metrics = elems(&[n_combos, N_METRICS])?;
        check_output(out_returns, n_returns)?;
        check_output(out_metrics, n_metrics)?;
        // SAFETY: caller contract — inputs valid for their shapes.
        let (c, s) = unsafe { (kernel_in(close, n_bars)?, kernel_in(signals, n_returns)?) };
        let cfg = config(
            commission_per_trade,
            slippage_bps,
            initial_capital,
            periods_per_year,
            risk_free_rate,
        );
        let out = core_sweep::sweep_signals(c, s, n_combos, &cfg).map_err(invalid)?;
        // SAFETY: outputs checked non-null and valid for their shapes.
        unsafe {
            kernel_out(out_metrics, &out.metrics_matrix(), n_metrics)?;
            kernel_out(out_returns, &out.returns, n_returns)
        }
    };
    /// Whole sweep of a built-in strategy in Rust, signal generation included. `combos` is row-major (n_combos, n_params) in the strategy's canonical parameter order (see ft_sweep_strategy_n_params). Combos violating the strategy's constraints (e.g. fast >= slow, non-integer periods) get out_valid = 0 and NaN returns/metrics; FT_ERR_INVALID_PARAM if none is valid.
    ft_sweep_builtin(
        close: in f64 [n_bars],
        n_bars: size usize,
        strategy: param sweep_strategy,
        combos: in f64 [n_combos, n_params],
        n_combos: size usize,
        n_params: size usize <- ft_sweep_strategy_n_params(strategy),
        commission_per_trade: param float,
        slippage_bps: param float,
        initial_capital: param float,
        periods_per_year: param float,
        risk_free_rate: param float,
        n_metric_cols: size usize <- ft_sweep_n_metrics(),
        out_valid: out i8 [n_combos],
        out_returns: out f64 [n_bars, n_combos],
        out_metrics: out f64 [n_combos, n_metric_cols],
    ) => {
        let names = strategy.param_names();
        if n_params != names.len() {
            return Err(Status::InvalidParam);
        }
        if n_metric_cols != N_METRICS {
            return Err(Status::InvalidParam);
        }
        let n_returns = elems(&[n_bars, n_combos])?;
        let n_metrics = elems(&[n_combos, N_METRICS])?;
        check_output(out_valid, n_combos)?;
        check_output(out_returns, n_returns)?;
        check_output(out_metrics, n_metrics)?;
        // SAFETY: caller contract — inputs valid for their shapes.
        let (c, grid_rows) = unsafe {
            (kernel_in(close, n_bars)?, kernel_in(combos, elems(&[n_combos, n_params])?)?)
        };
        let cfg = config(
            commission_per_trade,
            slippage_bps,
            initial_capital,
            periods_per_year,
            risk_free_rate,
        );
        let res = core_sweep::sweep_builtin(c, strategy, &names, grid_rows, &cfg)
            .map_err(invalid)?;
        let (returns, metrics) = scatter(&res.output, &res.kept, n_combos);
        let mut valid = vec![0i8; n_combos];
        for &k in &res.kept {
            valid[k] = 1;
        }
        // SAFETY: outputs checked non-null and valid for their shapes.
        unsafe {
            kernel_out(out_valid, &valid, n_combos)?;
            kernel_out(out_metrics, &metrics, n_metrics)?;
            kernel_out(out_returns, &returns, n_returns)
        }
    };
    /// Plateau-robust scores: mean of `values` over each combination's grid neighbourhood (combinations within `radius` steps on every axis; NaN ignored). `coords` is row-major (n_combos, n_axes) per-axis grid indices (>= 0).
    ft_sweep_plateau_scores(
        values: in f64 [n_combos],
        n_combos: size usize,
        coords: in i64 [n_combos, n_axes],
        n_axes: size usize,
        radius: param count,
        out: out f64 [n_combos],
    ) => {
        let n = n_combos;
        check_output(out, n)?;
        // SAFETY: caller contract — inputs valid for their shapes.
        let (v, raw) = unsafe { (kernel_in(values, n)?, kernel_in(coords, elems(&[n, n_axes])?)?) };
        let idx: Vec<usize> = raw
            .iter()
            .map(|&c| usize::try_from(c).map_err(invalid))
            .collect::<Result<_, _>>()?;
        let scores = grid::plateau_scores(v, &idx, n_axes, radius).map_err(invalid)?;
        // SAFETY: checked non-null; valid for n writes.
        unsafe { kernel_out(out, &scores, n) }
    };
}
