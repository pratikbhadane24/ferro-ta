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

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

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
    let arr = data.as_array();
    let (n_samples, n_series) = arr.dim();
    log::debug!(
        "batch_sma: timeperiod={timeperiod}, shape=({n_samples}, {n_series}), parallel={parallel}"
    );

    // Extract columns to owned Vecs so we can release the GIL for parallel work.
    let columns: Vec<Vec<f64>> = (0..n_series)
        .map(|j| (0..n_samples).map(|i| arr[[i, j]]).collect())
        .collect();

    let process_col = |col: &Vec<f64>| -> Vec<f64> { ferro_ta_core::overlap::sma(col, timeperiod) };

    let col_results: Vec<Vec<f64>> = py.allow_threads(|| {
        if parallel {
            columns.par_iter().map(process_col).collect()
        } else {
            columns.iter().map(process_col).collect()
        }
    });

    let mut result = Array2::<f64>::from_elem((n_samples, n_series), f64::NAN);
    for (j, col_result) in col_results.iter().enumerate() {
        for (i, &val) in col_result.iter().enumerate() {
            result[[i, j]] = val;
        }
    }
    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// batch_ema
// ---------------------------------------------------------------------------

/// Batch Exponential Moving Average — applies EMA to every column.
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
    let arr = data.as_array();
    let (n_samples, n_series) = arr.dim();
    log::debug!(
        "batch_ema: timeperiod={timeperiod}, shape=({n_samples}, {n_series}), parallel={parallel}"
    );

    let columns: Vec<Vec<f64>> = (0..n_series)
        .map(|j| (0..n_samples).map(|i| arr[[i, j]]).collect())
        .collect();

    let process_col = |col: &Vec<f64>| -> Vec<f64> { ferro_ta_core::overlap::ema(col, timeperiod) };

    let col_results: Vec<Vec<f64>> = py.allow_threads(|| {
        if parallel {
            columns.par_iter().map(process_col).collect()
        } else {
            columns.iter().map(process_col).collect()
        }
    });

    let mut result = Array2::<f64>::from_elem((n_samples, n_series), f64::NAN);
    for (j, col_result) in col_results.iter().enumerate() {
        for (i, &val) in col_result.iter().enumerate() {
            result[[i, j]] = val;
        }
    }
    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// batch_rsi
// ---------------------------------------------------------------------------

/// Batch RSI — applies RSI (Wilder seeding) to every column.
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
///   Values in [0, 100]; NaN during warmup.
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
    let arr = data.as_array();
    let (n_samples, n_series) = arr.dim();
    log::debug!(
        "batch_rsi: timeperiod={timeperiod}, shape=({n_samples}, {n_series}), parallel={parallel}"
    );

    let columns: Vec<Vec<f64>> = (0..n_series)
        .map(|j| (0..n_samples).map(|i| arr[[i, j]]).collect())
        .collect();

    let period_f = timeperiod as f64;
    let process_col = |col: &Vec<f64>| -> Vec<f64> {
        let mut col_result = vec![f64::NAN; n_samples];
        if n_samples <= timeperiod {
            return col_result;
        }
        let mut avg_gain = 0.0_f64;
        let mut avg_loss = 0.0_f64;
        for i in 1..=timeperiod {
            let delta = col[i] - col[i - 1];
            if delta > 0.0 {
                avg_gain += delta;
            } else {
                avg_loss += -delta;
            }
        }
        avg_gain /= period_f;
        avg_loss /= period_f;
        let rs = if avg_loss == 0.0 {
            f64::MAX
        } else {
            avg_gain / avg_loss
        };
        col_result[timeperiod] = 100.0 - 100.0 / (1.0 + rs);
        for i in (timeperiod + 1)..n_samples {
            let delta = col[i] - col[i - 1];
            let (gain, loss) = if delta > 0.0 {
                (delta, 0.0)
            } else {
                (0.0, -delta)
            };
            avg_gain = (avg_gain * (period_f - 1.0) + gain) / period_f;
            avg_loss = (avg_loss * (period_f - 1.0) + loss) / period_f;
            let rs = if avg_loss == 0.0 {
                f64::MAX
            } else {
                avg_gain / avg_loss
            };
            col_result[i] = 100.0 - 100.0 / (1.0 + rs);
        }
        col_result
    };

    let col_results: Vec<Vec<f64>> = py.allow_threads(|| {
        if parallel {
            columns.par_iter().map(process_col).collect()
        } else {
            columns.iter().map(process_col).collect()
        }
    });

    let mut result = Array2::<f64>::from_elem((n_samples, n_series), f64::NAN);
    for (j, col_result) in col_results.iter().enumerate() {
        for (i, &val) in col_result.iter().enumerate() {
            result[[i, j]] = val;
        }
    }
    Ok(result.into_pyarray(py))
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

    let cols_h: Vec<Vec<f64>> = (0..n_series)
        .map(|j| (0..n_samples).map(|i| arr_h[[i, j]]).collect())
        .collect();
    let cols_l: Vec<Vec<f64>> = (0..n_series)
        .map(|j| (0..n_samples).map(|i| arr_l[[i, j]]).collect())
        .collect();
    let cols_c: Vec<Vec<f64>> = (0..n_series)
        .map(|j| (0..n_samples).map(|i| arr_c[[i, j]]).collect())
        .collect();

    let col_results: Vec<Vec<f64>> = py.allow_threads(|| {
        let process_col = |j: usize| -> Vec<f64> {
            ferro_ta_core::volatility::atr(&cols_h[j], &cols_l[j], &cols_c[j], timeperiod)
        };
        if parallel {
            (0..n_series).into_par_iter().map(process_col).collect()
        } else {
            (0..n_series).map(process_col).collect()
        }
    });

    let mut result = Array2::<f64>::from_elem((n_samples, n_series), f64::NAN);
    for (j, col_result) in col_results.iter().enumerate() {
        for (i, &val) in col_result.iter().enumerate() {
            result[[i, j]] = val;
        }
    }
    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// batch_stoch
// ---------------------------------------------------------------------------

/// Stoch batch result type (slowk, slowd arrays).
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

    let cols_h: Vec<Vec<f64>> = (0..n_series)
        .map(|j| (0..n_samples).map(|i| arr_h[[i, j]]).collect())
        .collect();
    let cols_l: Vec<Vec<f64>> = (0..n_series)
        .map(|j| (0..n_samples).map(|i| arr_l[[i, j]]).collect())
        .collect();
    let cols_c: Vec<Vec<f64>> = (0..n_series)
        .map(|j| (0..n_samples).map(|i| arr_c[[i, j]]).collect())
        .collect();

    let col_results: Vec<(Vec<f64>, Vec<f64>)> = py.allow_threads(|| {
        let process_col = |j: usize| -> (Vec<f64>, Vec<f64>) {
            ferro_ta_core::momentum::stoch(
                &cols_h[j],
                &cols_l[j],
                &cols_c[j],
                fastk_period,
                slowk_period,
                slowd_period,
            )
        };
        if parallel {
            (0..n_series).into_par_iter().map(process_col).collect()
        } else {
            (0..n_series).map(process_col).collect()
        }
    });

    let mut result_k = Array2::<f64>::from_elem((n_samples, n_series), f64::NAN);
    let mut result_d = Array2::<f64>::from_elem((n_samples, n_series), f64::NAN);
    for (j, (k_col, d_col)) in col_results.iter().enumerate() {
        for i in 0..n_samples {
            result_k[[i, j]] = k_col[i];
            result_d[[i, j]] = d_col[i];
        }
    }
    Ok((result_k.into_pyarray(py), result_d.into_pyarray(py)))
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

    let cols_h: Vec<Vec<f64>> = (0..n_series)
        .map(|j| (0..n_samples).map(|i| arr_h[[i, j]]).collect())
        .collect();
    let cols_l: Vec<Vec<f64>> = (0..n_series)
        .map(|j| (0..n_samples).map(|i| arr_l[[i, j]]).collect())
        .collect();
    let cols_c: Vec<Vec<f64>> = (0..n_series)
        .map(|j| (0..n_samples).map(|i| arr_c[[i, j]]).collect())
        .collect();

    let col_results: Vec<Vec<f64>> = py.allow_threads(|| {
        let process_col = |j: usize| -> Vec<f64> {
            ferro_ta_core::momentum::adx(&cols_h[j], &cols_l[j], &cols_c[j], timeperiod)
        };
        if parallel {
            (0..n_series).into_par_iter().map(process_col).collect()
        } else {
            (0..n_series).map(process_col).collect()
        }
    });

    let mut result = Array2::<f64>::from_elem((n_samples, n_series), f64::NAN);
    for (j, col_result) in col_results.iter().enumerate() {
        for (i, &val) in col_result.iter().enumerate() {
            result[[i, j]] = val;
        }
    }
    Ok(result.into_pyarray(py))
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
    Ok(())
}
