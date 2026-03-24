//! Performance attribution and trade analysis.
//!
//! Functions
//! ---------
//! - `trade_stats`           — compute win rate, avg win/loss, hold time,
//!   profit factor from a list of trade PnLs and hold durations.
//! - `monthly_contribution`  — group bar returns by month index and sum, for
//!   time-based performance attribution.
//! - `signal_attribution`    — given signal labels per bar and bar returns,
//!   compute the PnL contribution of each signal.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// trade_stats
// ---------------------------------------------------------------------------

/// Compute trade-level statistics from trade PnL and hold durations.
///
/// Parameters
/// ----------
/// pnl        : 1-D float64 array — per-trade profit/loss (positive = win)
/// hold_bars  : 1-D float64 array — hold duration in bars for each trade
///   (same length as *pnl*)
///
/// Returns
/// -------
/// tuple of 5 floats:
///   ``(win_rate, avg_win, avg_loss, profit_factor, avg_hold_bars)``
///
/// - **win_rate**      : fraction of trades with PnL > 0
/// - **avg_win**       : mean PnL of winning trades (or 0 if none)
/// - **avg_loss**      : mean PnL of losing trades (negative; or 0 if none)
/// - **profit_factor** : gross profit / |gross loss|  (inf if no losses)
/// - **avg_hold_bars** : mean hold duration across all trades
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
    if n != h.len() {
        return Err(PyValueError::new_err(
            "pnl and hold_bars must have the same length",
        ));
    }

    let mut wins: Vec<f64> = Vec::new();
    let mut losses: Vec<f64> = Vec::new();
    for &v in p.iter() {
        if v > 0.0 {
            wins.push(v);
        } else if v < 0.0 {
            losses.push(v);
        }
    }

    let win_rate = wins.len() as f64 / n as f64;
    let avg_win = if wins.is_empty() {
        0.0
    } else {
        wins.iter().sum::<f64>() / wins.len() as f64
    };
    let avg_loss = if losses.is_empty() {
        0.0
    } else {
        losses.iter().sum::<f64>() / losses.len() as f64
    };

    let gross_profit: f64 = wins.iter().sum();
    let gross_loss: f64 = losses.iter().map(|v| v.abs()).sum();
    let profit_factor = if gross_loss == 0.0 {
        f64::INFINITY
    } else {
        gross_profit / gross_loss
    };

    let avg_hold = h.iter().sum::<f64>() / n as f64;

    Ok((win_rate, avg_win, avg_loss, profit_factor, avg_hold))
}

// ---------------------------------------------------------------------------
// monthly_contribution
// ---------------------------------------------------------------------------

/// Group per-bar returns by month index and sum each month's contribution.
///
/// The ``month_index`` array assigns each bar to a month bucket (0-based
/// integer, e.g. 0 = January year 1, 1 = February year 1, …).  The function
/// returns the **unique sorted month indices** and the corresponding
/// **total return** for each month.
///
/// Parameters
/// ----------
/// bar_returns  : 1-D float64 array — per-bar strategy returns
/// month_index  : 1-D int64 array — month bucket for each bar (same length)
///
/// Returns
/// -------
/// tuple ``(months, contributions)``:
///   - ``months``        : 1-D int64 array — sorted unique month indices
///   - ``contributions`` : 1-D float64 array — summed return per month
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
    if n != mi.len() {
        return Err(PyValueError::new_err(
            "bar_returns and month_index must have the same length",
        ));
    }

    // Accumulate contributions by month
    let mut map: HashMap<i64, f64> = HashMap::new();
    for i in 0..n {
        if !ret[i].is_nan() {
            *map.entry(mi[i]).or_insert(0.0) += ret[i];
        }
    }

    // Sort by month index
    let mut months: Vec<i64> = map.keys().copied().collect();
    months.sort_unstable();
    let contributions: Vec<f64> = months.iter().map(|m| map[m]).collect();

    Ok((months.into_pyarray(py), contributions.into_pyarray(py)))
}

// ---------------------------------------------------------------------------
// signal_attribution
// ---------------------------------------------------------------------------

/// Attribute per-bar returns to each signal label.
///
/// Each bar has a *signal_label* (integer) indicating which signal or rule
/// triggered the trade.  ``-1`` means "no signal / flat".  The function sums
/// bar returns per signal label.
///
/// Parameters
/// ----------
/// bar_returns   : 1-D float64 array — per-bar strategy returns
/// signal_labels : 1-D int64 array — signal label per bar (same length)
///
/// Returns
/// -------
/// tuple ``(labels, contributions)``:
///   - ``labels``        : 1-D int64 array — sorted unique signal labels
///   - ``contributions`` : 1-D float64 array — summed return per label
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
    if n != lbl.len() {
        return Err(PyValueError::new_err(
            "bar_returns and signal_labels must have the same length",
        ));
    }

    let mut map: HashMap<i64, f64> = HashMap::new();
    for i in 0..n {
        if !ret[i].is_nan() {
            *map.entry(lbl[i]).or_insert(0.0) += ret[i];
        }
    }

    let mut labels: Vec<i64> = map.keys().copied().collect();
    labels.sort_unstable();
    let contributions: Vec<f64> = labels.iter().map(|l| map[l]).collect();

    Ok((labels.into_pyarray(py), contributions.into_pyarray(py)))
}

// ---------------------------------------------------------------------------
// extract_trades
// ---------------------------------------------------------------------------

/// Extract trade-level pnl and hold durations from positions and strategy returns.
///
/// A trade is a maximal contiguous run of non-zero position values.
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
    if n != ret.len() {
        return Err(PyValueError::new_err(
            "positions and strategy_returns must have the same length",
        ));
    }

    let mut pnl = Vec::<f64>::new();
    let mut hold = Vec::<f64>::new();

    let mut i = 0usize;
    while i < n {
        if pos[i] == 0.0 {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while j < n && pos[j] == pos[i] {
            j += 1;
        }
        let mut trade_pnl = 0.0_f64;
        for v in ret.iter().take(j).skip(i) {
            trade_pnl += *v;
        }
        pnl.push(trade_pnl);
        hold.push((j - i) as f64);
        i = j;
    }

    Ok((pnl.into_pyarray(py), hold.into_pyarray(py)))
}

// ---------------------------------------------------------------------------
// Register
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trade_stats, m)?)?;
    m.add_function(wrap_pyfunction!(monthly_contribution, m)?)?;
    m.add_function(wrap_pyfunction!(signal_attribution, m)?)?;
    m.add_function(wrap_pyfunction!(extract_trades, m)?)?;
    Ok(())
}
