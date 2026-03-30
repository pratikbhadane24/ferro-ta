//! Signal processing helpers (thin PyO3 wrapper over ferro_ta_core::signals).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Compute the fractional rank of each element (1-based, ascending).
/// Ties receive the average of their rank positions.
#[pyfunction]
pub fn rank_series<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xv = x.as_slice()?;
    if xv.is_empty() {
        return Err(PyValueError::new_err("x must be non-empty"));
    }
    let result = ferro_ta_core::signals::rank_values(xv);
    Ok(result.into_pyarray(py))
}

/// Compute rank-based composite scores for a 2-D signal matrix.
/// Each column is ranked independently, per-row ranks are summed.
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
        let columns: Vec<Vec<f64>> = (0..n_sigs)
            .map(|sig_idx| arr.column(sig_idx).iter().copied().collect())
            .collect();
        let col_refs: Vec<&[f64]> = columns.iter().map(|c| c.as_slice()).collect();
        ferro_ta_core::signals::compose_rank(&col_refs)
    });

    Ok(scores.into_pyarray(py))
}

/// Return the indices of the N largest values in `x`.
#[pyfunction]
pub fn top_n_indices<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let xv = x.as_slice()?;
    let result = ferro_ta_core::signals::top_n_indices(xv, n);
    Ok(result.into_pyarray(py))
}

/// Return the indices of the N smallest values in `x`.
#[pyfunction]
pub fn bottom_n_indices<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let xv = x.as_slice()?;
    let result = ferro_ta_core::signals::bottom_n_indices(xv, n);
    Ok(result.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rank_series, m)?)?;
    m.add_function(wrap_pyfunction!(compose_rank, m)?)?;
    m.add_function(wrap_pyfunction!(top_n_indices, m)?)?;
    m.add_function(wrap_pyfunction!(bottom_n_indices, m)?)?;
    Ok(())
}
