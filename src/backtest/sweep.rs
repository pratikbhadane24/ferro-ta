//! Thin PyO3 wrappers over `ferro_ta_core::sweep`.

use ferro_ta_core::backtest::BacktestMetrics;
use ferro_ta_core::sweep::{self as core_sweep, grid, BuiltinStrategy, SweepConfig, SweepOutput};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

type SweepArrays<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>);

fn contiguous<'a>(arr: &'a PyReadonlyArray2<'_, f64>, name: &str) -> PyResult<&'a [f64]> {
    arr.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a C-contiguous float64 array")))
}

/// `(returns_2d, {metric_name: array})` from a core sweep output.
fn to_python<'py>(py: Python<'py>, out: SweepOutput) -> PyResult<SweepArrays<'py>> {
    let n_combos = out.n_combos();
    let matrix = out.metrics_matrix();
    let metrics = PyDict::new(py);
    for (k, name) in BacktestMetrics::SCALAR_FIELD_NAMES.iter().enumerate() {
        let column: Vec<f64> = (0..n_combos)
            .map(|j| matrix[j * BacktestMetrics::N_SCALAR_FIELDS + k])
            .collect();
        metrics.set_item(*name, column.into_pyarray(py))?;
    }
    let returns = Array2::from_shape_vec((out.n_bars, n_combos), out.returns)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((returns.into_pyarray(py), metrics))
}

fn sweep_config(
    commission_per_trade: f64,
    slippage_bps: f64,
    initial_capital: f64,
    periods_per_year: f64,
    risk_free_rate: f64,
    parallel: bool,
) -> SweepConfig {
    SweepConfig {
        commission_per_trade,
        slippage_bps,
        initial_capital,
        periods_per_year,
        risk_free_rate,
        parallel,
    }
}

/// Backtest every column of `signals_2d` (shape `(n_bars, n_combos)`) against
/// `close` and return `(strategy_returns_2d, metrics)`. GIL released.
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
) -> PyResult<SweepArrays<'py>> {
    let c = close.as_slice()?;
    let n_combos = signals_2d.as_array().ncols();
    let s = contiguous(&signals_2d, "signals_2d")?;
    let cfg = sweep_config(
        commission_per_trade,
        slippage_bps,
        initial_capital,
        periods_per_year,
        risk_free_rate,
        parallel,
    );
    let out = py
        .allow_threads(|| core_sweep::sweep_signals(c, s, n_combos, &cfg))
        .map_err(PyValueError::new_err)?;
    to_python(py, out)
}

/// Run a whole sweep in Rust: built-in `strategy` over `combos_2d`
/// (shape `(n_combos, len(param_names))`). Returns `(kept, returns_2d,
/// metrics)` where `kept` indexes the combos that satisfied the strategy's
/// constraints (the others are skipped).
#[pyfunction]
#[pyo3(signature = (
    close, strategy, param_names, combos_2d,
    commission_per_trade = 0.0,
    slippage_bps = 0.0,
    initial_capital = 100_000.0,
    periods_per_year = 252.0,
    risk_free_rate = 0.0,
    parallel = true,
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn backtest_sweep_builtin_core<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    strategy: &str,
    param_names: Vec<String>,
    combos_2d: PyReadonlyArray2<'py, f64>,
    commission_per_trade: f64,
    slippage_bps: f64,
    initial_capital: f64,
    periods_per_year: f64,
    risk_free_rate: f64,
    parallel: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyDict>,
)> {
    let strat = BuiltinStrategy::from_name(strategy).map_err(PyValueError::new_err)?;
    if combos_2d.as_array().ncols() != param_names.len() {
        return Err(PyValueError::new_err(format!(
            "combos_2d has {} columns but {} param_names were given",
            combos_2d.as_array().ncols(),
            param_names.len()
        )));
    }
    let c = close.as_slice()?;
    let combos = contiguous(&combos_2d, "combos_2d")?;
    let names: Vec<&str> = param_names.iter().map(String::as_str).collect();
    let cfg = sweep_config(
        commission_per_trade,
        slippage_bps,
        initial_capital,
        periods_per_year,
        risk_free_rate,
        parallel,
    );
    let res = py
        .allow_threads(|| core_sweep::sweep_builtin(c, strat, &names, combos, &cfg))
        .map_err(PyValueError::new_err)?;
    let kept: Vec<i64> = res.kept.iter().map(|&k| k as i64).collect();
    let (returns, metrics) = to_python(py, res.output)?;
    Ok((kept.into_pyarray(py), returns, metrics))
}

/// Neighbourhood-mean scores over a `(n_combos, n_axes)` grid-index matrix.
#[pyfunction]
pub fn sweep_plateau_scores<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    coords_2d: PyReadonlyArray2<'py, i64>,
    radius: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let coords = coords_2d.as_array();
    let n_axes = coords.ncols();
    let flat: Vec<usize> = coords
        .iter()
        .map(|&v| usize::try_from(v))
        .collect::<Result<_, _>>()
        .map_err(|_| PyValueError::new_err("coords must be non-negative"))?;
    let scores = grid::plateau_scores(values.as_slice()?, &flat, n_axes, radius)
        .map_err(PyValueError::new_err)?;
    Ok(scores.into_pyarray(py))
}

/// Canonical names of the strategies [`backtest_sweep_builtin_core`] runs.
#[pyfunction]
pub fn sweep_builtin_strategies() -> Vec<&'static str> {
    BuiltinStrategy::NAMES.to_vec()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sweep_builtin_strategies, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_sweep_core, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_sweep_builtin_core, m)?)?;
    m.add_function(wrap_pyfunction!(sweep_plateau_scores, m)?)?;
    Ok(())
}
