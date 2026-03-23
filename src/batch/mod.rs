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

use ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use ta::indicators::{Maximum, Minimum};
use ta::Next;

fn transpose_to_series_major(data: ArrayView2<'_, f64>) -> Array2<f64> {
    let (n_samples, n_series) = data.dim();
    Array2::from_shape_vec((n_series, n_samples), data.t().iter().copied().collect())
        .expect("shape matches transposed data")
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

fn finish_single_output<'py>(
    py: Python<'py>,
    n_samples: usize,
    n_series: usize,
    col_results: Vec<Vec<f64>>,
) -> Bound<'py, PyArray2<f64>> {
    let mut result = Array2::<f64>::from_elem((n_samples, n_series), f64::NAN);
    for (series_idx, values) in col_results.into_iter().enumerate() {
        debug_assert_eq!(values.len(), n_samples);
        for (sample_idx, value) in values.into_iter().enumerate() {
            result[[sample_idx, series_idx]] = value;
        }
    }
    result.into_pyarray(py)
}

fn finish_pair_output<'py>(
    py: Python<'py>,
    n_samples: usize,
    n_series: usize,
    col_results: Vec<(Vec<f64>, Vec<f64>)>,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>) {
    let mut result_k = Array2::<f64>::from_elem((n_samples, n_series), f64::NAN);
    let mut result_d = Array2::<f64>::from_elem((n_samples, n_series), f64::NAN);

    for (series_idx, (k_values, d_values)) in col_results.into_iter().enumerate() {
        debug_assert_eq!(k_values.len(), n_samples);
        debug_assert_eq!(d_values.len(), n_samples);
        for (sample_idx, value) in k_values.into_iter().enumerate() {
            result_k[[sample_idx, series_idx]] = value;
        }
        for (sample_idx, value) in d_values.into_iter().enumerate() {
            result_d[[sample_idx, series_idx]] = value;
        }
    }

    (result_k.into_pyarray(py), result_d.into_pyarray(py))
}

fn run_unary_batch<'py, F>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    parallel: bool,
    process_col: F,
) -> Bound<'py, PyArray2<f64>>
where
    F: Fn(&[f64]) -> Vec<f64> + Sync,
{
    let arr = data.as_array();
    let (n_samples, n_series) = arr.dim();
    let series_major = transpose_to_series_major(arr);

    let col_results: Vec<Vec<f64>> = py.allow_threads(|| {
        let run = |series_idx: usize| {
            let column_row = series_major.row(series_idx);
            let column = column_row
                .as_slice()
                .expect("series-major rows are contiguous");
            process_col(column)
        };
        if parallel {
            (0..n_series).into_par_iter().map(run).collect()
        } else {
            (0..n_series).map(run).collect()
        }
    });

    finish_single_output(py, n_samples, n_series, col_results)
}

fn validate_indicator_requests(names: &[String], timeperiods: &[usize]) -> PyResult<()> {
    if names.len() != timeperiods.len() {
        return Err(PyValueError::new_err(format!(
            "names length ({}) must equal timeperiods length ({})",
            names.len(),
            timeperiods.len()
        )));
    }
    for (name, &timeperiod) in names.iter().zip(timeperiods.iter()) {
        if timeperiod == 0 {
            return Err(PyValueError::new_err(format!(
                "{name}: timeperiod must be >= 1"
            )));
        }
    }
    Ok(())
}

fn compute_cci(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = high.len();
    let typical_price: Vec<f64> = high
        .iter()
        .zip(low.iter())
        .zip(close.iter())
        .map(|((&h, &l), &c)| (h + l + c) / 3.0)
        .collect();

    let mut result = vec![f64::NAN; n];
    for end in (timeperiod - 1)..n {
        let window = &typical_price[(end + 1 - timeperiod)..=end];
        let mean = window.iter().sum::<f64>() / timeperiod as f64;
        let mad = window
            .iter()
            .map(|&value| (value - mean).abs())
            .sum::<f64>()
            / timeperiod as f64;
        result[end] = if mad != 0.0 {
            (typical_price[end] - mean) / (0.015 * mad)
        } else {
            0.0
        };
    }
    result
}

fn compute_willr(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    timeperiod: usize,
) -> PyResult<Vec<f64>> {
    let n = high.len();
    let mut result = vec![f64::NAN; n];
    let mut max_ind =
        Maximum::new(timeperiod).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let mut min_ind =
        Minimum::new(timeperiod).map_err(|err| PyValueError::new_err(err.to_string()))?;

    for (idx, ((&high_value, &low_value), &close_value)) in
        high.iter().zip(low.iter()).zip(close.iter()).enumerate()
    {
        let highest = max_ind.next(high_value);
        let lowest = min_ind.next(low_value);
        if idx + 1 >= timeperiod {
            let range = highest - lowest;
            result[idx] = if range != 0.0 {
                -100.0 * (highest - close_value) / range
            } else {
                -50.0
            };
        }
    }

    Ok(result)
}

fn compute_close_indicator(name: &str, close: &[f64], timeperiod: usize) -> PyResult<Vec<f64>> {
    match name {
        "SMA" => Ok(ferro_ta_core::overlap::sma(close, timeperiod)),
        "EMA" => Ok(ferro_ta_core::overlap::ema(close, timeperiod)),
        "RSI" => Ok(ferro_ta_core::momentum::rsi(close, timeperiod)),
        "STDDEV" => Ok(ferro_ta_core::statistic::stddev(close, timeperiod, 1.0)),
        "VAR" => Ok(ferro_ta_core::statistic::stddev(close, timeperiod, 1.0)
            .into_iter()
            .map(|value| if value.is_nan() { value } else { value * value })
            .collect()),
        "LINEARREG" => {
            use crate::statistic::common::rolling_linreg_apply;
            let last_x = (timeperiod - 1) as f64;
            Ok(rolling_linreg_apply(
                close,
                timeperiod,
                |slope: f64, intercept: f64| intercept + slope * last_x,
            ))
        }
        "LINEARREG_SLOPE" => {
            use crate::statistic::common::rolling_linreg_apply;
            Ok(rolling_linreg_apply(
                close,
                timeperiod,
                |slope: f64, _: f64| slope,
            ))
        }
        "LINEARREG_INTERCEPT" => {
            use crate::statistic::common::rolling_linreg_apply;
            Ok(rolling_linreg_apply(
                close,
                timeperiod,
                |_: f64, intercept: f64| intercept,
            ))
        }
        "LINEARREG_ANGLE" => {
            use crate::statistic::common::rolling_linreg_apply;
            Ok(rolling_linreg_apply(
                close,
                timeperiod,
                |slope: f64, _: f64| slope.atan() * 180.0 / std::f64::consts::PI,
            ))
        }
        "TSF" => {
            use crate::statistic::common::rolling_linreg_apply;
            let forecast_x = timeperiod as f64;
            Ok(rolling_linreg_apply(
                close,
                timeperiod,
                |slope: f64, intercept: f64| intercept + slope * forecast_x,
            ))
        }
        _ => Err(PyValueError::new_err(format!(
            "unsupported close indicator for grouped execution: {name}"
        ))),
    }
}

fn compute_hlc_indicator(
    name: &str,
    high: &[f64],
    low: &[f64],
    close: &[f64],
    timeperiod: usize,
) -> PyResult<Vec<f64>> {
    match name {
        "ATR" => Ok(ferro_ta_core::volatility::atr(high, low, close, timeperiod)),
        "NATR" => {
            let atr = ferro_ta_core::volatility::atr(high, low, close, timeperiod);
            Ok(atr
                .into_iter()
                .zip(close.iter())
                .map(|(atr_value, &close_value)| {
                    if atr_value.is_nan() || close_value == 0.0 {
                        f64::NAN
                    } else {
                        (atr_value / close_value) * 100.0
                    }
                })
                .collect())
        }
        "ADX" => Ok(ferro_ta_core::momentum::adx(high, low, close, timeperiod)),
        "ADXR" => Ok(ferro_ta_core::momentum::adxr(high, low, close, timeperiod)),
        "CCI" => Ok(compute_cci(high, low, close, timeperiod)),
        "WILLR" => compute_willr(high, low, close, timeperiod),
        _ => Err(PyValueError::new_err(format!(
            "unsupported HLC indicator for grouped execution: {name}"
        ))),
    }
}

type IndicatorArrayList = Vec<Py<PyArray1<f64>>>;

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
    Ok(run_unary_batch(py, data, parallel, |col| {
        ferro_ta_core::overlap::sma(col, timeperiod)
    }))
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
    let (n_samples, n_series) = data.as_array().dim();
    log::debug!(
        "batch_ema: timeperiod={timeperiod}, shape=({n_samples}, {n_series}), parallel={parallel}"
    );
    Ok(run_unary_batch(py, data, parallel, |col| {
        ferro_ta_core::overlap::ema(col, timeperiod)
    }))
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
    let (n_samples, n_series) = data.as_array().dim();
    log::debug!(
        "batch_rsi: timeperiod={timeperiod}, shape=({n_samples}, {n_series}), parallel={parallel}"
    );

    let period_f = timeperiod as f64;
    Ok(run_unary_batch(py, data, parallel, |col| {
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
    }))
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

    let high_by_series = transpose_to_series_major(arr_h);
    let low_by_series = transpose_to_series_major(arr_l);
    let close_by_series = transpose_to_series_major(arr_c);

    let col_results: Vec<Vec<f64>> = py.allow_threads(|| {
        let process_col = |series_idx: usize| -> Vec<f64> {
            let high_row = high_by_series.row(series_idx);
            let low_row = low_by_series.row(series_idx);
            let close_row = close_by_series.row(series_idx);
            let high_col = high_row
                .as_slice()
                .expect("series-major rows are contiguous");
            let low_col = low_row
                .as_slice()
                .expect("series-major rows are contiguous");
            let close_col = close_row
                .as_slice()
                .expect("series-major rows are contiguous");
            ferro_ta_core::volatility::atr(high_col, low_col, close_col, timeperiod)
        };
        if parallel {
            (0..n_series).into_par_iter().map(process_col).collect()
        } else {
            (0..n_series).map(process_col).collect()
        }
    });
    Ok(finish_single_output(py, n_samples, n_series, col_results))
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
    validate_same_shape((n_samples, n_series), arr_l.dim(), "low")?;
    validate_same_shape((n_samples, n_series), arr_c.dim(), "close")?;

    let high_by_series = transpose_to_series_major(arr_h);
    let low_by_series = transpose_to_series_major(arr_l);
    let close_by_series = transpose_to_series_major(arr_c);

    let col_results: Vec<(Vec<f64>, Vec<f64>)> = py.allow_threads(|| {
        let process_col = |series_idx: usize| -> (Vec<f64>, Vec<f64>) {
            let high_row = high_by_series.row(series_idx);
            let low_row = low_by_series.row(series_idx);
            let close_row = close_by_series.row(series_idx);
            let high_col = high_row
                .as_slice()
                .expect("series-major rows are contiguous");
            let low_col = low_row
                .as_slice()
                .expect("series-major rows are contiguous");
            let close_col = close_row
                .as_slice()
                .expect("series-major rows are contiguous");
            ferro_ta_core::momentum::stoch(
                high_col,
                low_col,
                close_col,
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
    Ok(finish_pair_output(py, n_samples, n_series, col_results))
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

    let high_by_series = transpose_to_series_major(arr_h);
    let low_by_series = transpose_to_series_major(arr_l);
    let close_by_series = transpose_to_series_major(arr_c);

    let col_results: Vec<Vec<f64>> = py.allow_threads(|| {
        let process_col = |series_idx: usize| -> Vec<f64> {
            let high_row = high_by_series.row(series_idx);
            let low_row = low_by_series.row(series_idx);
            let close_row = close_by_series.row(series_idx);
            let high_col = high_row
                .as_slice()
                .expect("series-major rows are contiguous");
            let low_col = low_row
                .as_slice()
                .expect("series-major rows are contiguous");
            let close_col = close_row
                .as_slice()
                .expect("series-major rows are contiguous");
            ferro_ta_core::momentum::adx(high_col, low_col, close_col, timeperiod)
        };
        if parallel {
            (0..n_series).into_par_iter().map(process_col).collect()
        } else {
            (0..n_series).map(process_col).collect()
        }
    });
    Ok(finish_single_output(py, n_samples, n_series, col_results))
}

// ---------------------------------------------------------------------------
// grouped 1-D execution
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (close, names, timeperiods, parallel = true))]
pub fn run_close_indicators<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    names: Vec<String>,
    timeperiods: Vec<usize>,
    parallel: bool,
) -> PyResult<IndicatorArrayList> {
    validate_indicator_requests(&names, &timeperiods)?;
    let close_values = close.as_slice()?;

    let results: Vec<PyResult<Vec<f64>>> = py.allow_threads(|| {
        let run = |idx: usize| compute_close_indicator(&names[idx], close_values, timeperiods[idx]);
        if parallel {
            (0..names.len()).into_par_iter().map(run).collect()
        } else {
            (0..names.len()).map(run).collect()
        }
    });

    results
        .into_iter()
        .map(|result| result.map(|values| values.into_pyarray(py).unbind()))
        .collect()
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
    validate_indicator_requests(&names, &timeperiods)?;
    let high_values = high.as_slice()?;
    let low_values = low.as_slice()?;
    let close_values = close.as_slice()?;

    if high_values.len() != low_values.len() || high_values.len() != close_values.len() {
        return Err(PyValueError::new_err(
            "high, low, and close must have equal length",
        ));
    }

    let results: Vec<PyResult<Vec<f64>>> = py.allow_threads(|| {
        let run = |idx: usize| {
            compute_hlc_indicator(
                &names[idx],
                high_values,
                low_values,
                close_values,
                timeperiods[idx],
            )
        };
        if parallel {
            (0..names.len()).into_par_iter().map(run).collect()
        } else {
            (0..names.len()).map(run).collect()
        }
    });

    results
        .into_iter()
        .map(|result| result.map(|values| values.into_pyarray(py).unbind()))
        .collect()
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
