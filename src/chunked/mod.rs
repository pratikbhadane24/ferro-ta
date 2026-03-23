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
//! - `trim_overlap`      — remove the first *overlap* elements from an array
//!   (to strip the warm-up from a chunk's indicator output).
//! - `stitch_chunks`     — concatenate trimmed chunk results into one array.
//! - `make_chunk_ranges` — compute start/end indices for a series given chunk
//!   size and overlap, for use by the Python caller.

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
// Register
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trim_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(stitch_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(make_chunk_ranges, m)?)?;
    Ok(())
}
