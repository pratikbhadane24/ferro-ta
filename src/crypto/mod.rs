//! Crypto and 24/7 market helpers.
//!
//! Functions designed for continuous (24/7) markets such as crypto:
//!
//! - `funding_cumulative_pnl` — cumulative PnL from periodic funding rate
//!   payments, given a constant position size.
//! - `continuous_bar_labels`  — assign a sequential integer label to each bar
//!   based on a fixed period size (e.g. daily UTC buckets).
//! - `mark_session_boundaries` — return indices where the session rolls over
//!   (useful for calendar-free resampling on continuous data).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// funding_cumulative_pnl
// ---------------------------------------------------------------------------

/// Compute the cumulative PnL from funding rate payments.
///
/// Crypto perpetual contracts charge a periodic funding rate to the holder.
/// If you hold ``position_size`` contracts and the funding rate is ``rate[i]``,
/// the PnL at period *i* is ``-position_size * rate[i]`` (longs pay when rate
/// is positive).  This function returns the **cumulative** funding PnL.
///
/// Parameters
/// ----------
/// position_size : 1-D float64 array — signed position size per funding period
///     (positive = long, negative = short).  Must have the same length as
///     ``funding_rate``.
/// funding_rate  : 1-D float64 array — periodic funding rate (decimal, e.g.
///     0.0001 = 0.01%).
///
/// Returns
/// -------
/// 1-D float64 array — cumulative funding PnL (same length as inputs).
#[pyfunction]
pub fn funding_cumulative_pnl<'py>(
    py: Python<'py>,
    position_size: PyReadonlyArray1<'py, f64>,
    funding_rate: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let pos = position_size.as_slice()?;
    let rate = funding_rate.as_slice()?;
    let n = pos.len();
    if n != rate.len() {
        return Err(PyValueError::new_err(
            "position_size and funding_rate must have the same length",
        ));
    }
    let mut out = vec![0.0_f64; n];
    let mut cumulative = 0.0_f64;
    for i in 0..n {
        // Longs pay when rate > 0; shorts receive
        cumulative += -pos[i] * rate[i];
        out[i] = cumulative;
    }
    Ok(out.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// continuous_bar_labels
// ---------------------------------------------------------------------------

/// Assign a sequential integer label per bar based on a fixed-size period.
///
/// For 24/7 data (no session gaps), this groups bars into equal-sized buckets:
/// bars 0…(period_bars-1) get label 0, bars period_bars…(2*period_bars-1) get
/// label 1, etc.  Useful for resampling continuous data (e.g. group every 24
/// one-hour bars into a "day").
///
/// Parameters
/// ----------
/// n_bars     : int — total number of bars
/// period_bars: int — number of bars per period (must be >= 1)
///
/// Returns
/// -------
/// 1-D int64 array of length *n_bars* — period labels (0-based).
#[pyfunction]
pub fn continuous_bar_labels<'py>(
    py: Python<'py>,
    n_bars: usize,
    period_bars: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    if period_bars == 0 {
        return Err(PyValueError::new_err("period_bars must be >= 1"));
    }
    let out: Vec<i64> = (0..n_bars).map(|i| (i / period_bars) as i64).collect();
    Ok(out.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// mark_session_boundaries
// ---------------------------------------------------------------------------

/// Return bar indices where a new session begins.
///
/// Given UTC timestamps in nanoseconds (int64), marks boundaries at the start
/// of each new UTC day (midnight).  Returns the bar indices where the UTC day
/// changes.  Bar 0 is always included as the first boundary.
///
/// Parameters
/// ----------
/// timestamps_ns : 1-D int64 array — UTC timestamps in nanoseconds
///     (e.g. from ``pandas.DatetimeIndex.astype('int64')``).
///
/// Returns
/// -------
/// 1-D int64 array — indices of bars at the start of each new UTC day.
#[pyfunction]
pub fn mark_session_boundaries<'py>(
    py: Python<'py>,
    timestamps_ns: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let ts = timestamps_ns.as_slice()?;
    let n = ts.len();
    if n == 0 {
        return Ok(Vec::<i64>::new().into_pyarray(py));
    }
    // Nanoseconds per day
    const NS_PER_DAY: i64 = 86_400_000_000_000;
    let mut out = vec![0i64]; // bar 0 is always a boundary
    let mut prev_day = ts[0].div_euclid(NS_PER_DAY);
    for (i, &t) in ts.iter().enumerate().skip(1) {
        let day = t.div_euclid(NS_PER_DAY);
        if day != prev_day {
            out.push(i as i64);
            prev_day = day;
        }
    }
    Ok(out.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// Register
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(funding_cumulative_pnl, m)?)?;
    m.add_function(wrap_pyfunction!(continuous_bar_labels, m)?)?;
    m.add_function(wrap_pyfunction!(mark_session_boundaries, m)?)?;
    Ok(())
}
