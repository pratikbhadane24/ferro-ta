//! Signal processing helpers — Rust implementations.
//!
//! - `rank_series`      — cross-sectional rank of a 1-D array (fractional rank)
//! - `top_n_indices`    — indices of the N largest values in a 1-D array
//! - `bottom_n_indices` — indices of the N smallest values in a 1-D array

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn rank_values(xv: &[f64]) -> Vec<f64> {
    let n = xv.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        xv[a]
            .partial_cmp(&xv[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut ranks = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let val = xv[order[i]];
        let mut j = i + 1;
        while j < n && xv[order[j]] == val {
            j += 1;
        }
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[order[k]] = avg_rank;
        }
        i = j;
    }
    ranks
}

// ---------------------------------------------------------------------------
// rank_series
// ---------------------------------------------------------------------------

/// Compute the fractional rank of each element (1-based, ascending).
///
/// Ties receive the average of their rank positions (same as pandas default).
///
/// Parameters
/// ----------
/// x : 1-D float64 array
///
/// Returns
/// -------
/// 1-D float64 array — ranks in [1, n]
#[pyfunction]
pub fn rank_series<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xv = x.as_slice()?;
    let n = xv.len();
    if n == 0 {
        return Err(PyValueError::new_err("x must be non-empty"));
    }
    Ok(rank_values(xv).into_pyarray(py))
}

// ---------------------------------------------------------------------------
// compose_rank
// ---------------------------------------------------------------------------

/// Compute rank-based composite scores for a 2-D signal matrix.
///
/// Each column is ranked independently (ascending, fractional ranks for ties),
/// and the per-row ranks are summed across columns.
#[pyfunction]
pub fn compose_rank<'py>(
    py: Python<'py>,
    signals: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let arr = signals.as_array();
    let (n_bars, n_sigs) = arr.dim();
    if n_bars == 0 || n_sigs == 0 {
        return Err(PyValueError::new_err(
            "signals must be a non-empty 2-D array",
        ));
    }

    let scores = py.allow_threads(|| {
        let mut scores = vec![0.0_f64; n_bars];
        for sig_idx in 0..n_sigs {
            let column: Vec<f64> = arr.column(sig_idx).iter().copied().collect();
            let ranks = rank_values(&column);
            for (bar_idx, rank) in ranks.into_iter().enumerate() {
                scores[bar_idx] += rank;
            }
        }
        scores
    });

    Ok(scores.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// top_n_indices
// ---------------------------------------------------------------------------

/// Return the indices of the N largest values in `x` (unsorted).
///
/// Parameters
/// ----------
/// x : 1-D float64 array
/// n : int — number of top elements to return
///
/// Returns
/// -------
/// 1-D int64 array of length min(n, len(x))
#[pyfunction]
pub fn top_n_indices<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let xv = x.as_slice()?;
    let len = xv.len();
    let k = n.min(len);
    let mut order: Vec<usize> = (0..len).collect();
    order.sort_by(|&a, &b| {
        xv[b]
            .partial_cmp(&xv[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let result: Vec<i64> = order[..k].iter().map(|&i| i as i64).collect();
    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// bottom_n_indices
// ---------------------------------------------------------------------------

/// Return the indices of the N smallest values in `x` (unsorted).
#[pyfunction]
pub fn bottom_n_indices<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let xv = x.as_slice()?;
    let len = xv.len();
    let k = n.min(len);
    let mut order: Vec<usize> = (0..len).collect();
    order.sort_by(|&a, &b| {
        xv[a]
            .partial_cmp(&xv[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let result: Vec<i64> = order[..k].iter().map(|&i| i as i64).collect();
    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// Register
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rank_series, m)?)?;
    m.add_function(wrap_pyfunction!(compose_rank, m)?)?;
    m.add_function(wrap_pyfunction!(top_n_indices, m)?)?;
    m.add_function(wrap_pyfunction!(bottom_n_indices, m)?)?;
    Ok(())
}
