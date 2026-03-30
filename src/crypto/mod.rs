//! Crypto and 24/7 market helpers (thin PyO3 wrapper over ferro_ta_core::crypto).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Compute the cumulative PnL from funding rate payments.
#[pyfunction]
pub fn funding_cumulative_pnl<'py>(
    py: Python<'py>,
    position_size: PyReadonlyArray1<'py, f64>,
    funding_rate: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let pos = position_size.as_slice()?;
    let rate = funding_rate.as_slice()?;
    if pos.len() != rate.len() {
        return Err(PyValueError::new_err(
            "position_size and funding_rate must have the same length",
        ));
    }
    let result = ferro_ta_core::crypto::funding_cumulative_pnl(pos, rate);
    Ok(result.into_pyarray(py))
}

/// Assign a sequential integer label per bar based on a fixed-size period.
#[pyfunction]
pub fn continuous_bar_labels<'py>(
    py: Python<'py>,
    n_bars: usize,
    period_bars: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    if period_bars == 0 {
        return Err(PyValueError::new_err("period_bars must be >= 1"));
    }
    let result = ferro_ta_core::crypto::continuous_bar_labels(n_bars, period_bars);
    Ok(result.into_pyarray(py))
}

/// Return bar indices where a new UTC day begins.
#[pyfunction]
pub fn mark_session_boundaries<'py>(
    py: Python<'py>,
    timestamps_ns: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let ts = timestamps_ns.as_slice()?;
    let result = ferro_ta_core::crypto::mark_session_boundaries(ts);
    Ok(result.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(funding_cumulative_pnl, m)?)?;
    m.add_function(wrap_pyfunction!(continuous_bar_labels, m)?)?;
    m.add_function(wrap_pyfunction!(mark_session_boundaries, m)?)?;
    Ok(())
}
