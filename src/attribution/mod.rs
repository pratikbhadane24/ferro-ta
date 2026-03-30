//! Performance attribution (thin PyO3 wrapper over ferro_ta_core::attribution).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::validation;

/// Compute trade-level statistics from trade PnL and hold durations.
#[pyfunction]
pub fn trade_stats(
    pnl: PyReadonlyArray1<'_, f64>,
    hold_bars: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    let p = pnl.as_slice()?;
    let h = hold_bars.as_slice()?;
    let n = p.len();
    if n == 0 {
        return Err(PyValueError::new_err("pnl must be non-empty"));
    }
    validation::validate_equal_length(&[(n, "pnl"), (h.len(), "hold_bars")])?;
    Ok(ferro_ta_core::attribution::trade_stats(p, h))
}

/// Group per-bar returns by month index and sum each month's contribution.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn monthly_contribution<'py>(
    py: Python<'py>,
    bar_returns: PyReadonlyArray1<'py, f64>,
    month_index: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)> {
    let ret = bar_returns.as_slice()?;
    let mi = month_index.as_slice()?;
    let n = ret.len();
    validation::validate_equal_length(&[(n, "bar_returns"), (mi.len(), "month_index")])?;
    let (months, contributions) = ferro_ta_core::attribution::monthly_contribution(ret, mi);
    Ok((months.into_pyarray(py), contributions.into_pyarray(py)))
}

/// Attribute per-bar returns to each signal label.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn signal_attribution<'py>(
    py: Python<'py>,
    bar_returns: PyReadonlyArray1<'py, f64>,
    signal_labels: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)> {
    let ret = bar_returns.as_slice()?;
    let lbl = signal_labels.as_slice()?;
    let n = ret.len();
    validation::validate_equal_length(&[(n, "bar_returns"), (lbl.len(), "signal_labels")])?;
    let (labels, contributions) = ferro_ta_core::attribution::signal_attribution(ret, lbl);
    Ok((labels.into_pyarray(py), contributions.into_pyarray(py)))
}

/// Extract trade-level pnl and hold durations from positions and strategy returns.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn extract_trades<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray1<'py, f64>,
    strategy_returns: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let pos = positions.as_slice()?;
    let ret = strategy_returns.as_slice()?;
    let n = pos.len();
    validation::validate_equal_length(&[(n, "positions"), (ret.len(), "strategy_returns")])?;
    let (pnl, hold) = ferro_ta_core::attribution::extract_trades(pos, ret);
    Ok((pnl.into_pyarray(py), hold.into_pyarray(py)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trade_stats, m)?)?;
    m.add_function(wrap_pyfunction!(monthly_contribution, m)?)?;
    m.add_function(wrap_pyfunction!(signal_attribution, m)?)?;
    m.add_function(wrap_pyfunction!(extract_trades, m)?)?;
    Ok(())
}
