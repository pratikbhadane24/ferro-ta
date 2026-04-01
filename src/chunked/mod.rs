//! Chunked / out-of-core execution helpers (thin PyO3 wrapper over ferro_ta_core::chunked).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Remove the first *overlap* elements from an array.
#[pyfunction]
pub fn trim_overlap<'py>(
    py: Python<'py>,
    chunk_out: PyReadonlyArray1<'py, f64>,
    overlap: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let s = chunk_out.as_slice()?;
    if overlap > s.len() {
        return Err(PyValueError::new_err(format!(
            "overlap ({overlap}) must be <= chunk length ({})",
            s.len()
        )));
    }
    let result = ferro_ta_core::chunked::trim_overlap(s, overlap);
    Ok(result.into_pyarray(py))
}

/// Concatenate a list of trimmed chunk results into a single output array.
#[pyfunction]
pub fn stitch_chunks<'py>(
    py: Python<'py>,
    chunks: Vec<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let vecs: Vec<Vec<f64>> = chunks
        .iter()
        .map(|c| c.as_slice().map(|s| s.to_vec()))
        .collect::<Result<_, _>>()?;
    let refs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();
    let result = ferro_ta_core::chunked::stitch_chunks(&refs);
    Ok(result.into_pyarray(py))
}

/// Compute (start, end) index pairs for chunked processing.
#[pyfunction]
pub fn make_chunk_ranges<'py>(
    py: Python<'py>,
    n: usize,
    chunk_size: usize,
    overlap: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    if chunk_size == 0 {
        return Err(PyValueError::new_err("chunk_size must be >= 1"));
    }
    let result = ferro_ta_core::chunked::make_chunk_ranges(n, chunk_size, overlap);
    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// chunk_apply_close_indicator — stays in PyO3 (dispatches to ferro_ta_core indicators)
// ---------------------------------------------------------------------------

fn compute_close_indicator(
    indicator: &str,
    series: &[f64],
    timeperiod: usize,
) -> PyResult<Vec<f64>> {
    match indicator {
        "SMA" => Ok(ferro_ta_core::overlap::sma(series, timeperiod)),
        "EMA" => Ok(ferro_ta_core::overlap::ema(series, timeperiod)),
        "RSI" => Ok(ferro_ta_core::momentum::rsi(series, timeperiod)),
        _ => Err(PyValueError::new_err(format!(
            "chunk_apply_close_indicator does not support indicator '{indicator}'"
        ))),
    }
}

/// Run chunked execution for close-only indicators in Rust.
#[pyfunction]
#[pyo3(signature = (series, indicator, timeperiod, chunk_size = 10_000, overlap = 100))]
pub fn chunk_apply_close_indicator<'py>(
    py: Python<'py>,
    series: PyReadonlyArray1<'py, f64>,
    indicator: &str,
    timeperiod: usize,
    chunk_size: usize,
    overlap: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if timeperiod == 0 {
        return Err(PyValueError::new_err("timeperiod must be >= 1"));
    }
    if chunk_size == 0 {
        return Err(PyValueError::new_err("chunk_size must be >= 1"));
    }

    let values = series.as_slice()?;
    if values.is_empty() {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    let name = indicator.to_ascii_uppercase();
    let n = values.len();
    let mut stitched: Vec<f64> = Vec::with_capacity(n);
    let mut start = 0usize;
    let mut chunk_index = 0usize;

    loop {
        let end = (start + chunk_size + overlap).min(n);
        let chunk = &values[start..end];
        let out = compute_close_indicator(name.as_str(), chunk, timeperiod)?;

        let discard = if chunk_index == 0 { 0 } else { overlap };
        if discard > out.len() {
            return Err(PyValueError::new_err(format!(
                "overlap ({discard}) must be <= chunk output length ({})",
                out.len()
            )));
        }
        stitched.extend_from_slice(&out[discard..]);

        if end >= n {
            break;
        }
        start = end.saturating_sub(overlap);
        chunk_index += 1;
    }

    if stitched.len() != n {
        return Err(PyValueError::new_err(format!(
            "internal chunk stitching error: expected output length {n}, got {}",
            stitched.len()
        )));
    }

    Ok(stitched.into_pyarray(py))
}

/// Forward-fill NaN values in a 1-D array.
#[pyfunction]
pub fn forward_fill_nan<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let input = values.as_slice()?;
    let result = ferro_ta_core::chunked::forward_fill_nan(input);
    Ok(result.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trim_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(stitch_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(make_chunk_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_apply_close_indicator, m)?)?;
    m.add_function(wrap_pyfunction!(forward_fill_nan, m)?)?;
    Ok(())
}
