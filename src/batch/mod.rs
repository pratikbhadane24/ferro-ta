//! Rust-side batch execution — run SMA/EMA/RSI over all columns of a 2-D
//! array in a **single GIL release**, avoiding per-column Python round-trips.
//!
//! Python shapes: `(n_samples, n_series)` — C-contiguous row-major.
//! Rust iterates over columns (series) and rows (time) inside native code.
//!
//! When `parallel = true` (default), columns are processed in parallel via
//! [Rayon](https://docs.rs/rayon) after releasing the GIL.  For small inputs
//! the sequential path (`parallel = false`) may be faster due to thread-pool
//! overhead.
//!
//! All indicator logic lives in `ferro_ta_core::batch`.  This module is a thin
//! PyO3 wrapper that converts numpy ↔ Rust types and optionally adds Rayon
//! parallelism.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// numpy ↔ Vec<Vec<f64>> helpers
// ---------------------------------------------------------------------------

/// Convert a numpy (n_samples, n_series) array into `Vec<Vec<f64>>` where
/// `result[j]` is column j (one time-series of length n_samples).
fn numpy2d_to_columns(arr: &ndarray::ArrayView2<'_, f64>) -> Vec<Vec<f64>> {
    let (_n_samples, n_series) = arr.dim();
    (0..n_series)
        .map(|j| arr.column(j).to_vec())
        .collect()
}

/// Convert `Vec<Vec<f64>>` (columns) back into a numpy (n_samples, n_series) array.
fn columns_to_numpy2d<'py>(
    py: Python<'py>,
    n_samples: usize,
    columns: Vec<Vec<f64>>,
) -> Bound<'py, PyArray2<f64>> {
    let n_series = columns.len();
    let mut result = Array2::<f64>::from_elem((n_samples, n_series), f64::NAN);
    for (j, col) in columns.into_iter().enumerate() {
        for (i, val) in col.into_iter().enumerate() {
            result[[i, j]] = val;
        }
    }
    result.into_pyarray(py)
}

/// Convert a pair of column-vectors into a pair of numpy 2-D arrays.
fn column_pair_to_numpy2d<'py>(
    py: Python<'py>,
    n_samples: usize,
    cols_a: Vec<Vec<f64>>,
    cols_b: Vec<Vec<f64>>,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>) {
    (
        columns_to_numpy2d(py, n_samples, cols_a),
        columns_to_numpy2d(py, n_samples, cols_b),
    )
}

fn validate_same_shape(
    expected: (usize, usize),
    actual: (usize, usize),
    name: &str,
) -> PyResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must have shape {:?}, got {:?}",
            expected, actual
        )))
    }
}

fn map_core_err(err: String) -> PyErr {
    PyValueError::new_err(err)
}

// ---------------------------------------------------------------------------
// Parallel-aware unary batch helper
// ---------------------------------------------------------------------------

/// Run a unary batch function.  When `parallel` is true, split column extraction
/// across Rayon threads and process in parallel; otherwise delegate sequentially
/// to `ferro_ta_core::batch`.
fn run_unary_batch_par<'py, F>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    parallel: bool,
    per_col: F,
) -> PyResult<Bound<'py, PyArray2<f64>>>
where
    F: Fn(&[f64]) -> Vec<f64> + Sync,
{
    let arr = data.as_array();
    let (n_samples, _n_series) = arr.dim();
    let columns = numpy2d_to_columns(&arr);

    let col_results: Vec<Vec<f64>> = py.allow_threads(|| {
        if parallel {
            columns.par_iter().map(|col| per_col(col)).collect()
        } else {
            columns.iter().map(|col| per_col(col)).collect()
        }
    });

    Ok(columns_to_numpy2d(py, n_samples, col_results))
}

// ---------------------------------------------------------------------------
// batch_sma
// ---------------------------------------------------------------------------

/// Batch Simple Moving Average — applies SMA to every column of a 2-D array.
///
/// Parameters
/// ----------
/// data : numpy array, shape (n_samples, n_series), dtype float64
/// timeperiod : int
/// parallel : bool, default True
///   When True, columns are processed in parallel via Rayon (GIL released).
///
/// Returns
/// -------
/// numpy array, shape (n_samples, n_series), dtype float64
///   Same shape as input; first ``timeperiod-1`` rows are NaN.
#[pyfunction]
#[pyo3(signature = (data, timeperiod = 30, parallel = true))]
pub fn batch_sma<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    timeperiod: usize,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if timeperiod == 0 {
        return Err(PyValueError::new_err("timeperiod must be >= 1"));
    }
    let (n_samples, n_series) = data.as_array().dim();
    log::debug!(
        "batch_sma: timeperiod={timeperiod}, shape=({n_samples}, {n_series}), parallel={parallel}"
    );
    run_unary_batch_par(py, data, parallel, |col| {
        ferro_ta_core::overlap::sma(col, timeperiod)
    })
}

// ---------------------------------------------------------------------------
// batch_ema
// ---------------------------------------------------------------------------

/// Batch Exponential Moving Average — applies EMA to every column.
#[pyfunction]
#[pyo3(signature = (data, timeperiod = 30, parallel = true))]
pub fn batch_ema<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    timeperiod: usize,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if timeperiod == 0 {
        return Err(PyValueError::new_err("timeperiod must be >= 1"));
    }
    let (n_samples, n_series) = data.as_array().dim();
    log::debug!(
        "batch_ema: timeperiod={timeperiod}, shape=({n_samples}, {n_series}), parallel={parallel}"
    );
    run_unary_batch_par(py, data, parallel, |col| {
        ferro_ta_core::overlap::ema(col, timeperiod)
    })
}

// ---------------------------------------------------------------------------
// batch_rsi
// ---------------------------------------------------------------------------

/// Batch RSI — applies RSI (Wilder seeding) to every column.
#[pyfunction]
#[pyo3(signature = (data, timeperiod = 14, parallel = true))]
pub fn batch_rsi<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    timeperiod: usize,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if timeperiod == 0 {
        return Err(PyValueError::new_err("timeperiod must be >= 1"));
    }
    let (n_samples, n_series) = data.as_array().dim();
    log::debug!(
        "batch_rsi: timeperiod={timeperiod}, shape=({n_samples}, {n_series}), parallel={parallel}"
    );
    run_unary_batch_par(py, data, parallel, |col| {
        ferro_ta_core::momentum::rsi(col, timeperiod)
    })
}

// ---------------------------------------------------------------------------
// batch_atr
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 14, parallel = true))]
pub fn batch_atr<'py>(
    py: Python<'py>,
    high: PyReadonlyArray2<'py, f64>,
    low: PyReadonlyArray2<'py, f64>,
    close: PyReadonlyArray2<'py, f64>,
    timeperiod: usize,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if timeperiod == 0 {
        return Err(PyValueError::new_err("timeperiod must be >= 1"));
    }
    let arr_h = high.as_array();
    let arr_l = low.as_array();
    let arr_c = close.as_array();
    let (n_samples, n_series) = arr_h.dim();
    validate_same_shape((n_samples, n_series), arr_l.dim(), "low")?;
    validate_same_shape((n_samples, n_series), arr_c.dim(), "close")?;

    let h_cols = numpy2d_to_columns(&arr_h);
    let l_cols = numpy2d_to_columns(&arr_l);
    let c_cols = numpy2d_to_columns(&arr_c);

    let col_results: Vec<Vec<f64>> = py.allow_threads(|| {
        let process = |i: usize| {
            ferro_ta_core::volatility::atr(&h_cols[i], &l_cols[i], &c_cols[i], timeperiod)
        };
        if parallel {
            (0..n_series).into_par_iter().map(process).collect()
        } else {
            (0..n_series).map(process).collect()
        }
    });
    Ok(columns_to_numpy2d(py, n_samples, col_results))
}

// ---------------------------------------------------------------------------
// batch_stoch
// ---------------------------------------------------------------------------

type StochBatchResult<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>);

#[pyfunction]
#[pyo3(signature = (high, low, close, fastk_period = 5, slowk_period = 3, slowd_period = 3, parallel = true))]
#[allow(clippy::too_many_arguments)]
pub fn batch_stoch<'py>(
    py: Python<'py>,
    high: PyReadonlyArray2<'py, f64>,
    low: PyReadonlyArray2<'py, f64>,
    close: PyReadonlyArray2<'py, f64>,
    fastk_period: usize,
    slowk_period: usize,
    slowd_period: usize,
    parallel: bool,
) -> PyResult<StochBatchResult<'py>> {
    let arr_h = high.as_array();
    let arr_l = low.as_array();
    let arr_c = close.as_array();
    let (n_samples, n_series) = arr_h.dim();
    validate_same_shape((n_samples, n_series), arr_l.dim(), "low")?;
    validate_same_shape((n_samples, n_series), arr_c.dim(), "close")?;

    let h_cols = numpy2d_to_columns(&arr_h);
    let l_cols = numpy2d_to_columns(&arr_l);
    let c_cols = numpy2d_to_columns(&arr_c);

    let col_results: Vec<(Vec<f64>, Vec<f64>)> = py.allow_threads(|| {
        let process = |i: usize| {
            ferro_ta_core::momentum::stoch(
                &h_cols[i],
                &l_cols[i],
                &c_cols[i],
                fastk_period,
                slowk_period,
                slowd_period,
            )
        };
        if parallel {
            (0..n_series).into_par_iter().map(process).collect()
        } else {
            (0..n_series).map(process).collect()
        }
    });

    let (all_k, all_d): (Vec<Vec<f64>>, Vec<Vec<f64>>) = col_results.into_iter().unzip();
    Ok(column_pair_to_numpy2d(py, n_samples, all_k, all_d))
}

// ---------------------------------------------------------------------------
// batch_adx
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 14, parallel = true))]
pub fn batch_adx<'py>(
    py: Python<'py>,
    high: PyReadonlyArray2<'py, f64>,
    low: PyReadonlyArray2<'py, f64>,
    close: PyReadonlyArray2<'py, f64>,
    timeperiod: usize,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if timeperiod == 0 {
        return Err(PyValueError::new_err("timeperiod must be >= 1"));
    }
    let arr_h = high.as_array();
    let arr_l = low.as_array();
    let arr_c = close.as_array();
    let (n_samples, n_series) = arr_h.dim();
    validate_same_shape((n_samples, n_series), arr_l.dim(), "low")?;
    validate_same_shape((n_samples, n_series), arr_c.dim(), "close")?;

    let h_cols = numpy2d_to_columns(&arr_h);
    let l_cols = numpy2d_to_columns(&arr_l);
    let c_cols = numpy2d_to_columns(&arr_c);

    let col_results: Vec<Vec<f64>> = py.allow_threads(|| {
        let process = |i: usize| {
            ferro_ta_core::momentum::adx(&h_cols[i], &l_cols[i], &c_cols[i], timeperiod)
        };
        if parallel {
            (0..n_series).into_par_iter().map(process).collect()
        } else {
            (0..n_series).map(process).collect()
        }
    });
    Ok(columns_to_numpy2d(py, n_samples, col_results))
}

// ---------------------------------------------------------------------------
// grouped 1-D execution
// ---------------------------------------------------------------------------

type IndicatorArrayList = Vec<Py<PyArray1<f64>>>;

#[pyfunction]
#[pyo3(signature = (close, names, timeperiods, parallel = true))]
pub fn run_close_indicators<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    names: Vec<String>,
    timeperiods: Vec<usize>,
    parallel: bool,
) -> PyResult<IndicatorArrayList> {
    let close_values = close.as_slice()?;

    if parallel {
        // Parallel path: call core per-indicator in parallel via Rayon
        let results: Vec<Result<Vec<f64>, String>> = py.allow_threads(|| {
            (0..names.len())
                .into_par_iter()
                .map(|idx| {
                    ferro_ta_core::batch::run_close_indicators(
                        close_values,
                        &[names[idx].clone()],
                        &[timeperiods[idx]],
                    )
                    .map(|mut v| v.remove(0))
                })
                .collect()
        });
        results
            .into_iter()
            .map(|r| r.map(|v| v.into_pyarray(py).unbind()).map_err(map_core_err))
            .collect()
    } else {
        let results = ferro_ta_core::batch::run_close_indicators(close_values, &names, &timeperiods)
            .map_err(map_core_err)?;
        Ok(results
            .into_iter()
            .map(|v| v.into_pyarray(py).unbind())
            .collect())
    }
}

#[pyfunction]
#[pyo3(signature = (high, low, close, names, timeperiods, parallel = true))]
pub fn run_hlc_indicators<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    names: Vec<String>,
    timeperiods: Vec<usize>,
    parallel: bool,
) -> PyResult<IndicatorArrayList> {
    let high_values = high.as_slice()?;
    let low_values = low.as_slice()?;
    let close_values = close.as_slice()?;

    if high_values.len() != low_values.len() || high_values.len() != close_values.len() {
        return Err(PyValueError::new_err(
            "high, low, and close must have equal length",
        ));
    }

    if parallel {
        let results: Vec<Result<Vec<f64>, String>> = py.allow_threads(|| {
            (0..names.len())
                .into_par_iter()
                .map(|idx| {
                    ferro_ta_core::batch::run_hlc_indicators(
                        high_values,
                        low_values,
                        close_values,
                        &[names[idx].clone()],
                        &[timeperiods[idx]],
                    )
                    .map(|mut v| v.remove(0))
                })
                .collect()
        });
        results
            .into_iter()
            .map(|r| r.map(|v| v.into_pyarray(py).unbind()).map_err(map_core_err))
            .collect()
    } else {
        let results = ferro_ta_core::batch::run_hlc_indicators(
            high_values,
            low_values,
            close_values,
            &names,
            &timeperiods,
        )
        .map_err(map_core_err)?;
        Ok(results
            .into_iter()
            .map(|v| v.into_pyarray(py).unbind())
            .collect())
    }
}

// ---------------------------------------------------------------------------
// register
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(batch_sma, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(batch_ema, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(batch_rsi, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(batch_atr, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(batch_stoch, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(batch_adx, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(run_close_indicators, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(run_hlc_indicators, m)?)?;
    Ok(())
}
