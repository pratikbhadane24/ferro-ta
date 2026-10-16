//! Parallel parameter sweep: one close series, many signal columns.

use ferro_ta_core::backtest::BacktestMetrics;
use ferro_ta_core::sweep::{run_sweep_combo, SweepComboResult, SweepConfig};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

/// Backtest every column of `signals_2d` (shape `(n_bars, n_combos)`) against
/// `close` and return `(strategy_returns_2d, metrics)`.
///
/// `strategy_returns_2d` has the same shape as `signals_2d`; `metrics` maps
/// each metric name to a float64 array of length `n_combos`. Runs with the
/// GIL released, across threads when `parallel` is true.
#[pyfunction]
#[pyo3(signature = (
    close, signals_2d,
    commission_per_trade = 0.0,
    slippage_bps = 0.0,
    initial_capital = 100_000.0,
    periods_per_year = 252.0,
    risk_free_rate = 0.0,
    parallel = true,
))]
#[allow(clippy::too_many_arguments)]
pub fn backtest_sweep_core<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    signals_2d: PyReadonlyArray2<'py, f64>,
    commission_per_trade: f64,
    slippage_bps: f64,
    initial_capital: f64,
    periods_per_year: f64,
    risk_free_rate: f64,
    parallel: bool,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>)> {
    let c = close.as_slice()?.to_vec();
    let s_arr = signals_2d.as_array();
    let (n_bars, n_combos) = s_arr.dim();
    if n_bars != c.len() {
        return Err(PyValueError::new_err(format!(
            "signals_2d has {n_bars} rows but close has {} bars",
            c.len()
        )));
    }
    if n_combos == 0 {
        return Err(PyValueError::new_err(
            "signals_2d must have at least 1 column",
        ));
    }
    let columns: Vec<Vec<f64>> = (0..n_combos).map(|j| s_arr.column(j).to_vec()).collect();
    let cfg = SweepConfig {
        commission_per_trade,
        slippage_bps,
        initial_capital,
        periods_per_year,
        risk_free_rate,
    };

    let results: Vec<SweepComboResult> = py
        .allow_threads(|| {
            let run = |sig: &Vec<f64>| run_sweep_combo(&c, sig, &cfg);
            if parallel {
                columns.par_iter().map(run).collect::<Result<Vec<_>, _>>()
            } else {
                columns.iter().map(run).collect::<Result<Vec<_>, _>>()
            }
        })
        .map_err(PyValueError::new_err)?;

    let mut returns_2d = Array2::<f64>::zeros((n_bars, n_combos));
    for (j, res) in results.iter().enumerate() {
        for (i, &r) in res.strategy_returns.iter().enumerate() {
            returns_2d[[i, j]] = r;
        }
    }

    let field_values: Vec<_> = results.iter().map(|r| r.metrics.scalar_fields()).collect();
    let metrics = PyDict::new(py);
    for k in 0..BacktestMetrics::N_SCALAR_FIELDS {
        let name = field_values[0][k].0;
        let column: Vec<f64> = field_values.iter().map(|f| f[k].1).collect();
        metrics.set_item(name, column.into_pyarray(py))?;
    }

    Ok((returns_2d.into_pyarray(py), metrics))
}
