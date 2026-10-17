//! Thin PyO3 wrappers over `ferro_ta_core::overfit`.

use ferro_ta_core::overfit as core_of;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
#[pyo3(signature = (returns, benchmark_sharpe = 0.0))]
pub fn probabilistic_sharpe_ratio(
    returns: PyReadonlyArray1<'_, f64>,
    benchmark_sharpe: f64,
) -> PyResult<f64> {
    core_of::probabilistic_sharpe_ratio(returns.as_slice()?, benchmark_sharpe)
        .map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn expected_max_sharpe(n_trials: usize, trial_sharpe_variance: f64) -> PyResult<f64> {
    core_of::expected_max_sharpe(n_trials, trial_sharpe_variance).map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn deflated_sharpe_ratio(
    returns: PyReadonlyArray1<'_, f64>,
    n_trials: usize,
    trial_sharpe_variance: f64,
) -> PyResult<f64> {
    core_of::deflated_sharpe_ratio(returns.as_slice()?, n_trials, trial_sharpe_variance)
        .map_err(PyValueError::new_err)
}

/// PBO via CSCV over a C-contiguous `(n_bars, n_strategies)` returns matrix.
#[pyfunction]
#[pyo3(signature = (returns_2d, n_splits = 16))]
pub fn pbo_cscv<'py>(
    py: Python<'py>,
    returns_2d: PyReadonlyArray2<'py, f64>,
    n_splits: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let n_strategies = returns_2d.as_array().ncols();
    let flat = returns_2d
        .as_slice()
        .map_err(|_| PyValueError::new_err("returns_2d must be a C-contiguous float64 array"))?;
    let res = py
        .allow_threads(|| core_of::pbo_cscv(flat, n_strategies, n_splits))
        .map_err(PyValueError::new_err)?;

    let best_index: Vec<i64> = res.best_index.iter().map(|&i| i as i64).collect();
    let dict = PyDict::new(py);
    dict.set_item("pbo", res.pbo)?;
    dict.set_item("logits", res.logits.into_pyarray(py))?;
    dict.set_item("is_sharpe", res.is_sharpe.into_pyarray(py))?;
    dict.set_item("oos_sharpe", res.oos_sharpe.into_pyarray(py))?;
    dict.set_item("best_index", best_index.into_pyarray(py))?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (n_bars, n_folds, purge_bars = 0, embargo_bars = 0))]
pub fn purged_kfold_indices<'py>(
    py: Python<'py>,
    n_bars: usize,
    n_folds: usize,
    purge_bars: usize,
    embargo_bars: usize,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let folds = core_of::purged_kfold_indices(n_bars, n_folds, purge_bars, embargo_bars)
        .map_err(PyValueError::new_err)?;
    let flat: Vec<i64> = folds.iter().flatten().copied().collect();
    let arr = Array2::from_shape_vec((folds.len(), 4), flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(probabilistic_sharpe_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(expected_max_sharpe, m)?)?;
    m.add_function(wrap_pyfunction!(deflated_sharpe_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(pbo_cscv, m)?)?;
    m.add_function(wrap_pyfunction!(purged_kfold_indices, m)?)?;
    Ok(())
}
