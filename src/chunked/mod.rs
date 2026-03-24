//! Chunked / out-of-core execution helpers.
//!
//! These functions support running indicators on data that is too large for
//! memory by processing it in chunks.  The caller splits a large series into
//! overlapping chunks (overlap = indicator warm-up period), runs an indicator
//! on each chunk, and then stitches the results by trimming the overlap from
//! the front of each chunk's output.
//!
//! Functions
//! ---------
//! - `trim_overlap`               — remove the first *overlap* elements from
//!   an array (to strip the warm-up from a chunk's indicator output).
//! - `stitch_chunks`              — concatenate trimmed chunk results into one
//!   array.
//! - `make_chunk_ranges`          — compute start/end indices for a series
//!   given chunk size and overlap, for use by the Python caller.
//! - `chunk_apply_close_indicator`— run chunked close-only indicators fully in
//!   Rust (SMA/EMA/RSI).
//! - `forward_fill_nan`           — forward-fill NaN values in a 1-D array.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// trim_overlap
// ---------------------------------------------------------------------------

/// Remove the first *overlap* elements from an array.
///
/// After running an indicator on a chunk that includes a warm-up prefix, the
/// first *overlap* output values are unreliable (NaN or influenced by
/// artificial padding).  This function discards them.
///
/// Parameters
/// ----------
/// chunk_out : 1-D float64 array — indicator output for a chunk
/// overlap   : int — number of leading elements to discard
///
/// Returns
/// -------
/// 1-D float64 array — the trailing ``len(chunk_out) - overlap`` elements
#[pyfunction]
pub fn trim_overlap<'py>(
    py: Python<'py>,
    chunk_out: PyReadonlyArray1<'py, f64>,
    overlap: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let s = chunk_out.as_slice()?;
    let n = s.len();
    if overlap > n {
        return Err(PyValueError::new_err(format!(
            "overlap ({overlap}) must be <= chunk length ({n})"
        )));
    }
    let out = s[overlap..].to_vec();
    Ok(out.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// stitch_chunks
// ---------------------------------------------------------------------------

/// Concatenate a list of trimmed chunk results into a single output array.
///
/// Parameters
/// ----------
/// chunks : list of 1-D float64 arrays — trimmed outputs from each chunk
///
/// Returns
/// -------
/// 1-D float64 array — concatenated result
#[pyfunction]
pub fn stitch_chunks<'py>(
    py: Python<'py>,
    chunks: Vec<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mut out: Vec<f64> = Vec::new();
    for chunk in &chunks {
        out.extend_from_slice(chunk.as_slice()?);
    }
    Ok(out.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// make_chunk_ranges
// ---------------------------------------------------------------------------

/// Compute the (start, end) index pairs for chunked processing.
///
/// Each range ``[start, end)`` specifies a slice of the input series that the
/// caller should pass to the indicator function.  The first *overlap* elements
/// of each range (except the very first range) are the warm-up prefix from the
/// previous chunk.
///
/// Parameters
/// ----------
/// n          : int — total length of the series
/// chunk_size : int — desired number of *output* bars per chunk (>= 1)
/// overlap    : int — number of warm-up bars prepended to each chunk (>= 0)
///
/// Returns
/// -------
/// list of (start: int, end: int) pairs as a flattened 1-D int64 array of
/// length 2 × n_chunks.  Caller unpacks with ``ranges.reshape(-1, 2)``.
///
/// Example
/// -------
/// For n=10, chunk_size=4, overlap=2 the ranges would cover:
///   [0, 4), [2, 8), [6, 10)  (start of chunk 2 = end of prev chunk - overlap)
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
    let mut ranges: Vec<i64> = Vec::new();
    if n == 0 {
        return Ok(ranges.into_pyarray(py));
    }
    let mut start: usize = 0;
    loop {
        let end = (start + chunk_size + overlap).min(n);
        ranges.push(start as i64);
        ranges.push(end as i64);
        if end >= n {
            break;
        }
        // Next chunk starts at end - overlap (so the next chunk has its overlap prefix)
        start = end.saturating_sub(overlap);
    }
    Ok(ranges.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// chunk_apply_close_indicator
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
///
/// Parameters
/// ----------
/// series     : 1-D float64 array
/// indicator  : one of {"SMA", "EMA", "RSI"}
/// timeperiod : indicator period (>= 1)
/// chunk_size : output bars per chunk (>= 1)
/// overlap    : warm-up bars prepended to each chunk
///
/// Returns
/// -------
/// 1-D float64 array with the same length as `series`.
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

// ---------------------------------------------------------------------------
// forward_fill_nan
// ---------------------------------------------------------------------------

/// Forward-fill NaN values in a 1-D array.
///
/// Leading NaN values are preserved until the first non-NaN value appears.
#[pyfunction]
pub fn forward_fill_nan<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let input = values.as_slice()?;
    let mut out = Vec::with_capacity(input.len());
    let mut last = f64::NAN;

    for &value in input {
        if value.is_nan() {
            out.push(last);
        } else {
            last = value;
            out.push(value);
        }
    }

    Ok(out.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// Register
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trim_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(stitch_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(make_chunk_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_apply_close_indicator, m)?)?;
    m.add_function(wrap_pyfunction!(forward_fill_nan, m)?)?;
    Ok(())
}
