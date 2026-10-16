//! Parameter-sweep building block.
//!
//! A sweep runs the same close series against many signal vectors (one per
//! parameter configuration). The core stays single-threaded and allocation-
//! light; bindings fan [`run_sweep_combo`] out across threads.

use crate::backtest::{backtest_core, compute_performance_metrics_with_positions, BacktestMetrics};

/// Cost and annualisation settings shared by every combination in a sweep.
#[derive(Clone, Copy, Debug)]
pub struct SweepConfig {
    pub commission_per_trade: f64,
    pub slippage_bps: f64,
    pub initial_capital: f64,
    pub periods_per_year: f64,
    pub risk_free_rate: f64,
}

/// Per-combination output: strategy returns plus its metrics.
#[derive(Clone, Debug)]
pub struct SweepComboResult {
    pub strategy_returns: Vec<f64>,
    pub metrics: BacktestMetrics,
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

#[cfg(test)]
mod tests {
    use super::*;

    const CFG: SweepConfig = SweepConfig {
        commission_per_trade: 0.0,
        slippage_bps: 0.0,
        initial_capital: 100_000.0,
        periods_per_year: 252.0,
        risk_free_rate: 0.0,
    };

    #[test]
    fn combo_matches_backtest_core() {
        let close = [100.0, 101.0, 99.0, 102.0, 103.0, 101.0];
        let signals = [1.0, 1.0, -1.0, f64::NAN, 1.0, 0.0];
        let res = run_sweep_combo(&close, &signals, &CFG).unwrap();
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
        assert!(run_sweep_combo(&[1.0, 2.0, 3.0], &[1.0, 0.0], &CFG).is_err());
    }

    #[test]
    fn scalar_fields_names_unique() {
        let res = run_sweep_combo(&[1.0, 2.0, 3.0, 2.0], &[1.0; 4], &CFG).unwrap();
        let fields = res.metrics.scalar_fields();
        let mut names: Vec<&str> = fields.iter().map(|(n, _)| *n).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), BacktestMetrics::N_SCALAR_FIELDS);
    }
}
