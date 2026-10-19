use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// MESA Adaptive Moving Average. Returns (mama, fama). Uses Hilbert Transform–based period.
///
/// Raises `ValueError` unless both limits are finite and
/// `slowlimit <= fastlimit` (the same rule as `StreamingMama`).
#[pyfunction]
#[pyo3(signature = (close, fastlimit = 0.5, slowlimit = 0.05))]
#[allow(clippy::type_complexity)]
pub fn mama<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    fastlimit: f64,
    slowlimit: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    if !(fastlimit.is_finite() && slowlimit.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "mama: fastlimit ({fastlimit}) and slowlimit ({slowlimit}) must be finite"
        )));
    }
    if slowlimit > fastlimit {
        return Err(PyValueError::new_err(format!(
            "mama: slowlimit ({slowlimit}) must be <= fastlimit ({fastlimit})"
        )));
    }
    let prices = close.as_slice()?;
    let (mama_arr, fama_arr) = ferro_ta_core::overlap::mama(prices, fastlimit, slowlimit);
    Ok((mama_arr.into_pyarray(py), fama_arr.into_pyarray(py)))
}
