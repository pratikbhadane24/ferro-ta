use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Dynamic Time Warping — distance and optimal warping path between two 1-D series.
///
/// Returns a tuple `(distance, path)` where `path` is a NumPy array of shape
/// `(N, 2)` containing `(i, j)` index pairs from `(0, 0)` to `(n-1, m-1)`.
///
/// Local cost: `|series1[i] - series2[j]|` (Euclidean, matches `dtaidistance`).
#[pyfunction]
#[pyo3(signature = (series1, series2, window = None))]
pub fn dtw<'py>(
    py: Python<'py>,
    series1: PyReadonlyArray1<'py, f64>,
    series2: PyReadonlyArray1<'py, f64>,
    window: Option<usize>,
) -> PyResult<(f64, Bound<'py, PyArray2<usize>>)> {
    let s1 = series1.as_slice()?;
    let s2 = series2.as_slice()?;
    if s1.is_empty() || s2.is_empty() {
        return Err(PyValueError::new_err(
            "series1 and series2 must not be empty",
        ));
    }
    let (dist, path) = ferro_ta_core::statistic::dtw_path(s1, s2, window);
    let n = path.len();
    let flat: Vec<usize> = path.iter().flat_map(|&(i, j)| [i, j]).collect();
    let arr =
        Array2::from_shape_vec((n, 2), flat).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((dist, arr.into_pyarray(py)))
}

/// Dynamic Time Warping — distance only (faster, no path reconstruction).
///
/// Returns the accumulated Euclidean cost along the optimal warping path.
/// Use this when you only need the distance, not the alignment path.
#[pyfunction]
#[pyo3(signature = (series1, series2, window = None))]
pub fn dtw_distance<'py>(
    _py: Python<'py>,
    series1: PyReadonlyArray1<'py, f64>,
    series2: PyReadonlyArray1<'py, f64>,
    window: Option<usize>,
) -> PyResult<f64> {
    let s1 = series1.as_slice()?;
    let s2 = series2.as_slice()?;
    if s1.is_empty() || s2.is_empty() {
        return Err(PyValueError::new_err(
            "series1 and series2 must not be empty",
        ));
    }
    Ok(ferro_ta_core::statistic::dtw_distance(s1, s2, window))
}

/// Batch Dynamic Time Warping — compute DTW distance from each row of a 2-D matrix
/// to a single reference series, in parallel.
///
/// Parameters
/// ----------
/// matrix : np.ndarray, shape (N, L)
///     N time series of length L. Each row is compared against `reference`.
/// reference : np.ndarray, shape (L,)
///     The reference series.
/// window : int, optional
///     Sakoe-Chiba band width. `None` = unconstrained.
///
/// Returns
/// -------
/// np.ndarray, shape (N,)
///     DTW distances, one per row.
#[pyfunction]
#[pyo3(signature = (matrix, reference, window = None))]
pub fn batch_dtw<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f64>,
    reference: PyReadonlyArray1<'py, f64>,
    window: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mat = matrix.as_array();
    let ref_slice = reference.as_slice()?;

    if ref_slice.is_empty() {
        return Err(PyValueError::new_err("reference must not be empty"));
    }

    let (n_rows, _) = mat.dim();
    let rows: Vec<Vec<f64>> = (0..n_rows).map(|i| mat.row(i).to_vec()).collect();

    let result: Vec<f64> = rows
        .par_iter()
        .map(|series| ferro_ta_core::statistic::dtw_distance(series, ref_slice, window))
        .collect();

    Ok(result.into_pyarray(py))
}
