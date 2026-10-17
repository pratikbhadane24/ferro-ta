//! Parameter sweeps: one close series against many parameter configurations.
//!
//! - [`sweep_signals`] backtests caller-supplied signal columns.
//! - [`sweep_builtin`] also generates the signals, from a
//!   [`BuiltinStrategy`] and a grid of parameter values, so an entire sweep
//!   runs in Rust from any binding.
//! - [`grid`] holds Cartesian expansion and plateau-robust scoring.
//!
//! With the `parallel` feature, combinations run across a Rayon pool when
//! [`SweepConfig::parallel`] is set; without it (e.g. WASM) they run serially
//! with identical results.

pub mod builtin;
pub mod grid;

pub use builtin::BuiltinStrategy;

use crate::backtest::{backtest_core, compute_performance_metrics_with_positions, BacktestMetrics};

/// Cost and annualisation settings shared by every combination in a sweep.
#[derive(Clone, Copy, Debug)]
pub struct SweepConfig {
    pub commission_per_trade: f64,
    pub slippage_bps: f64,
    pub initial_capital: f64,
    pub periods_per_year: f64,
    pub risk_free_rate: f64,
    /// Use the Rayon pool when the `parallel` feature is compiled in.
    pub parallel: bool,
}

impl Default for SweepConfig {
    fn default() -> Self {
        Self {
            commission_per_trade: 0.0,
            slippage_bps: 0.0,
            initial_capital: 100_000.0,
            periods_per_year: 252.0,
            risk_free_rate: 0.0,
            parallel: true,
        }
    }
}

/// Per-combination output: strategy returns plus its metrics.
#[derive(Clone, Debug)]
pub struct SweepComboResult {
    pub strategy_returns: Vec<f64>,
    pub metrics: BacktestMetrics,
}

/// Whole-sweep output.
#[derive(Clone, Debug)]
pub struct SweepOutput {
    pub n_bars: usize,
    /// Row-major `(n_bars, n_combos)` strategy returns.
    pub returns: Vec<f64>,
    /// One entry per combination, in column order.
    pub metrics: Vec<BacktestMetrics>,
}

impl SweepOutput {
    pub fn n_combos(&self) -> usize {
        self.metrics.len()
    }

    /// Row-major `(n_combos, N_SCALAR_FIELDS)` metric matrix; column names
    /// are [`metric_names`].
    pub fn metrics_matrix(&self) -> Vec<f64> {
        self.metrics
            .iter()
            .flat_map(|m| m.scalar_fields().map(|(_, v)| v))
            .collect()
    }

    fn from_columns(n_bars: usize, cols: Vec<SweepComboResult>) -> Self {
        let n_combos = cols.len();
        let mut returns = vec![0.0; n_bars * n_combos];
        for (j, col) in cols.iter().enumerate() {
            for (i, &r) in col.strategy_returns.iter().enumerate() {
                returns[i * n_combos + j] = r;
            }
        }
        Self {
            n_bars,
            returns,
            metrics: cols.into_iter().map(|c| c.metrics).collect(),
        }
    }
}

/// Names of the per-combination metrics, in [`SweepOutput::metrics_matrix`]
/// column order.
pub fn metric_names() -> [&'static str; BacktestMetrics::N_SCALAR_FIELDS] {
    BacktestMetrics::scalar_field_names()
}

/// Backtest one signal vector with the same semantics as [`backtest_core`]
/// (NaN signals hold the prior position) and compute its metrics.
pub fn run_sweep_combo(
    close: &[f64],
    signals: &[f64],
    cfg: &SweepConfig,
) -> Result<SweepComboResult, String> {
    let bt = backtest_core(
        close,
        signals,
        None,
        cfg.slippage_bps,
        cfg.initial_capital,
        cfg.commission_per_trade,
    )?;
    let metrics = compute_performance_metrics_with_positions(
        &bt.strategy_returns,
        &bt.equity,
        cfg.periods_per_year,
        cfg.risk_free_rate,
        None,
        Some(&bt.positions),
    )?;
    Ok(SweepComboResult {
        strategy_returns: bt.strategy_returns,
        metrics,
    })
}

/// Map `f` over `0..n`, on the Rayon pool when available and requested.
fn map_combos<T, F>(n: usize, parallel: bool, f: F) -> Result<Vec<T>, String>
where
    T: Send,
    F: Fn(usize) -> Result<T, String> + Sync + Send,
{
    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        return (0..n).into_par_iter().map(f).collect();
    }
    let _ = parallel;
    (0..n).map(f).collect()
}

fn validate_close(close: &[f64]) -> Result<usize, String> {
    if close.len() < 2 {
        return Err("close must have at least 2 bars".to_string());
    }
    Ok(close.len())
}

/// Backtest every column of a row-major `(n_bars, n_combos)` signal matrix.
pub fn sweep_signals(
    close: &[f64],
    signals: &[f64],
    n_combos: usize,
    cfg: &SweepConfig,
) -> Result<SweepOutput, String> {
    let n_bars = validate_close(close)?;
    if n_combos == 0 {
        return Err("need at least 1 signal column".to_string());
    }
    if signals.len() != n_bars * n_combos {
        return Err(format!(
            "signals length ({}) must equal n_bars ({n_bars}) * n_combos ({n_combos})",
            signals.len()
        ));
    }
    let cols = map_combos(n_combos, cfg.parallel, |j| {
        let column: Vec<f64> = (0..n_bars).map(|i| signals[i * n_combos + j]).collect();
        run_sweep_combo(close, &column, cfg)
    })?;
    Ok(SweepOutput::from_columns(n_bars, cols))
}

/// Result of [`sweep_builtin`]: which grid rows ran, and their output.
#[derive(Clone, Debug)]
pub struct BuiltinSweep {
    /// Indices (into the input combos) that satisfied the strategy's
    /// constraints and were run; the output's columns follow this order.
    pub kept: Vec<usize>,
    pub output: SweepOutput,
}

/// Generate signals for each parameter combination and backtest them.
///
/// `combos` is row-major `(n_combos, names.len())`; parameters not named use
/// the strategy's defaults. Combinations violating the strategy's
/// constraints (e.g. `fast >= slow`) are skipped and omitted from
/// [`BuiltinSweep::kept`]; it is an error if none remain.
pub fn sweep_builtin(
    close: &[f64],
    strategy: BuiltinStrategy,
    names: &[&str],
    combos: &[f64],
    cfg: &SweepConfig,
) -> Result<BuiltinSweep, String> {
    let n_bars = validate_close(close)?;
    if names.is_empty() {
        return Err("need at least one parameter name".to_string());
    }
    strategy.validate_names(names)?;
    if combos.is_empty() || !combos.len().is_multiple_of(names.len()) {
        return Err(format!(
            "combos length ({}) must be a non-zero multiple of the number of names ({})",
            combos.len(),
            names.len()
        ));
    }
    let (kept, resolved): (Vec<usize>, Vec<Vec<f64>>) = combos
        .chunks_exact(names.len())
        .enumerate()
        .filter_map(|(c, row)| strategy.resolve(names, row).map(|p| (c, p)))
        .unzip();
    if kept.is_empty() {
        return Err("no parameter combination satisfies the strategy's constraints".to_string());
    }
    let cols = map_combos(resolved.len(), cfg.parallel, |j| {
        let signals = strategy.signals(close, &resolved[j])?;
        run_sweep_combo(close, &signals, cfg)
    })?;
    Ok(BuiltinSweep {
        kept,
        output: SweepOutput::from_columns(n_bars, cols),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SweepConfig {
        SweepConfig {
            parallel: false,
            ..SweepConfig::default()
        }
    }

    fn wave(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| 100.0 + (i as f64 * 0.11).sin() * 4.0 + i as f64 * 0.01)
            .collect()
    }

    #[test]
    fn combo_matches_backtest_core() {
        let close = [100.0, 101.0, 99.0, 102.0, 103.0, 101.0];
        let signals = [1.0, 1.0, -1.0, f64::NAN, 1.0, 0.0];
        let res = run_sweep_combo(&close, &signals, &cfg()).unwrap();
        let bt = backtest_core(&close, &signals, None, 0.0, 100_000.0, 0.0).unwrap();
        assert_eq!(res.strategy_returns, bt.strategy_returns);
        let direct = compute_performance_metrics_with_positions(
            &bt.strategy_returns,
            &bt.equity,
            252.0,
            0.0,
            None,
            Some(&bt.positions),
        )
        .unwrap();
        assert_eq!(res.metrics.sharpe, direct.sharpe);
        assert_eq!(res.metrics.n_trades, direct.n_trades);
    }

    #[test]
    fn combo_rejects_length_mismatch() {
        assert!(run_sweep_combo(&[1.0, 2.0, 3.0], &[1.0, 0.0], &cfg()).is_err());
    }

    #[test]
    fn scalar_fields_names_unique() {
        let mut names = metric_names().to_vec();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), BacktestMetrics::N_SCALAR_FIELDS);
    }

    #[test]
    fn sweep_signals_layout_matches_single_runs() {
        let close = wave(200);
        let a: Vec<f64> = close
            .iter()
            .map(|c| if *c > 100.0 { 1.0 } else { -1.0 })
            .collect();
        let b = vec![1.0; 200];
        let mut flat = vec![0.0; 400];
        for i in 0..200 {
            flat[i * 2] = a[i];
            flat[i * 2 + 1] = b[i];
        }
        let out = sweep_signals(&close, &flat, 2, &cfg()).unwrap();
        let ra = run_sweep_combo(&close, &a, &cfg()).unwrap();
        let rb = run_sweep_combo(&close, &b, &cfg()).unwrap();
        for i in 0..200 {
            assert_eq!(out.returns[i * 2], ra.strategy_returns[i]);
            assert_eq!(out.returns[i * 2 + 1], rb.strategy_returns[i]);
        }
        assert_eq!(
            out.metrics_matrix().len(),
            2 * BacktestMetrics::N_SCALAR_FIELDS
        );
        assert!(sweep_signals(&close, &flat, 3, &cfg()).is_err());
        assert!(sweep_signals(&close, &[], 0, &cfg()).is_err());
    }

    #[test]
    fn sweep_builtin_skips_invalid_combos() {
        let close = wave(300);
        let combos = grid::expand_grid(&[vec![5.0, 20.0, 40.0], vec![10.0, 30.0]]).unwrap();
        let res = sweep_builtin(
            &close,
            BuiltinStrategy::SmaCrossover,
            &["fast", "slow"],
            &combos,
            &cfg(),
        )
        .unwrap();
        // (5,10) (5,30) (20,30) valid; (20,10) (40,10) (40,30) skipped.
        assert_eq!(res.kept, vec![0, 1, 3]);
        assert_eq!(res.output.n_combos(), 3);
        let sig = sma_signals(&close, 20, 30);
        let single = run_sweep_combo(&close, &sig, &cfg()).unwrap();
        for i in 0..300 {
            assert_eq!(res.output.returns[i * 3 + 2], single.strategy_returns[i]);
        }
    }

    fn sma_signals(close: &[f64], fast: usize, slow: usize) -> Vec<f64> {
        crate::backtest::sma_crossover_signals(close, fast, slow).unwrap()
    }

    #[test]
    fn sweep_builtin_errors() {
        let close = wave(100);
        let s = BuiltinStrategy::SmaCrossover;
        assert!(sweep_builtin(&close, s, &["fast"], &[], &cfg()).is_err());
        assert!(sweep_builtin(&close, s, &["nope"], &[5.0], &cfg()).is_err());
        assert!(sweep_builtin(&close, s, &["fast", "slow"], &[5.0], &cfg()).is_err());
        assert!(sweep_builtin(&close, s, &["fast", "slow"], &[50.0, 10.0], &cfg()).is_err());
        assert!(sweep_builtin(&[1.0], s, &["fast"], &[5.0], &cfg()).is_err());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_equals_serial() {
        let close = wave(500);
        let combos = grid::expand_grid(&[vec![3.0, 5.0, 8.0, 13.0], vec![21.0, 34.0]]).unwrap();
        let names = ["fast", "slow"];
        let s = BuiltinStrategy::SmaCrossover;
        let serial = sweep_builtin(&close, s, &names, &combos, &cfg()).unwrap();
        let par_cfg = SweepConfig {
            parallel: true,
            ..cfg()
        };
        let par = sweep_builtin(&close, s, &names, &combos, &par_cfg).unwrap();
        assert_eq!(serial.output.returns, par.output.returns);
        assert_eq!(serial.output.metrics_matrix(), par.output.metrics_matrix());
    }
}
