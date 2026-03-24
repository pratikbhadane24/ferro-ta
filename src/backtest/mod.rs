//! Rust-backed strategy signal generation and backtest core.
//!
//! These functions move the hot loops from Python into Rust while preserving
//! the public Python behavior.

use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn nan_to_num_with_numpy_defaults(v: f64) -> f64 {
    if v.is_nan() {
        0.0
    } else if v.is_infinite() {
        if v.is_sign_positive() {
            f64::MAX
        } else {
            -f64::MAX
        }
    } else {
        v
    }
}

// ---------------------------------------------------------------------------
// Strategy signal helpers
// ---------------------------------------------------------------------------

/// RSI threshold strategy:
/// +1 when RSI <= oversold, -1 when RSI >= overbought, 0 otherwise.
/// Warm-up bars are NaN.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 14, oversold = 30.0, overbought = 70.0))]
pub fn rsi_threshold_signals<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    oversold: f64,
    overbought: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let prices = close.as_slice()?;
    let rsi = ferro_ta_core::momentum::rsi(prices, timeperiod);
    let out: Vec<f64> = rsi
        .iter()
        .map(|&v| {
            if v.is_nan() {
                f64::NAN
            } else if v <= oversold {
                1.0
            } else if v >= overbought {
                -1.0
            } else {
                0.0
            }
        })
        .collect();
    Ok(out.into_pyarray(py))
}

/// SMA crossover strategy:
/// +1 when fast SMA > slow SMA, -1 otherwise. Warm-up bars are NaN.
#[pyfunction]
#[pyo3(signature = (close, fast = 10, slow = 30))]
pub fn sma_crossover_signals<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    fast: usize,
    slow: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(fast, "fast", 1)?;
    validation::validate_timeperiod(slow, "slow", 1)?;
    if fast >= slow {
        return Err(PyValueError::new_err(format!(
            "fast ({fast}) must be less than slow ({slow})"
        )));
    }
    let prices = close.as_slice()?;
    let sma_fast = ferro_ta_core::overlap::sma(prices, fast);
    let sma_slow = ferro_ta_core::overlap::sma(prices, slow);
    let out: Vec<f64> = sma_fast
        .iter()
        .zip(sma_slow.iter())
        .map(|(&f, &s)| {
            if f.is_nan() || s.is_nan() {
                f64::NAN
            } else if f > s {
                1.0
            } else {
                -1.0
            }
        })
        .collect();
    Ok(out.into_pyarray(py))
}

/// MACD crossover strategy:
/// +1 when MACD line > signal line, -1 otherwise. Warm-up bars are NaN.
#[pyfunction]
#[pyo3(signature = (close, fastperiod = 12, slowperiod = 26, signalperiod = 9))]
pub fn macd_crossover_signals<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    fastperiod: usize,
    slowperiod: usize,
    signalperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(fastperiod, "fastperiod", 1)?;
    validation::validate_timeperiod(slowperiod, "slowperiod", 1)?;
    validation::validate_timeperiod(signalperiod, "signalperiod", 1)?;
    if fastperiod >= slowperiod {
        return Err(PyValueError::new_err(format!(
            "fastperiod ({fastperiod}) must be less than slowperiod ({slowperiod})"
        )));
    }

    let prices = close.as_slice()?;
    let (macd_line, signal_line, _) =
        ferro_ta_core::overlap::macd(prices, fastperiod, slowperiod, signalperiod);
    let out: Vec<f64> = macd_line
        .iter()
        .zip(signal_line.iter())
        .map(|(&m, &s)| {
            if m.is_nan() || s.is_nan() {
                f64::NAN
            } else if m > s {
                1.0
            } else {
                -1.0
            }
        })
        .collect();
    Ok(out.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// Backtest core
// ---------------------------------------------------------------------------

/// Backtest core loop over close prices and strategy signals.
///
/// Returns `(positions, bar_returns, strategy_returns, equity)`.
#[pyfunction]
#[pyo3(signature = (close, signals, commission_per_trade = 0.0, slippage_bps = 0.0))]
#[allow(clippy::type_complexity)]
pub fn backtest_core<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    signals: PyReadonlyArray1<'py, f64>,
    commission_per_trade: f64,
    slippage_bps: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let c = close.as_slice()?;
    let s = signals.as_slice()?;
    let n = c.len();
    validation::validate_equal_length(&[(n, "close"), (s.len(), "signals")])?;

    let mut positions = vec![0.0_f64; n];
    if n > 1 {
        for i in 1..n {
            positions[i] = nan_to_num_with_numpy_defaults(s[i - 1]);
        }
    }

    let mut bar_returns = vec![0.0_f64; n];
    for i in 1..n {
        bar_returns[i] = (c[i] - c[i - 1]) / c[i - 1];
    }

    let mut strategy_returns = vec![0.0_f64; n];
    for i in 0..n {
        strategy_returns[i] = positions[i] * bar_returns[i];
    }

    let mut position_changed = vec![false; n];
    for i in 1..n {
        position_changed[i] = positions[i] != positions[i - 1];
    }

    if slippage_bps > 0.0 {
        let slip = slippage_bps / 10_000.0;
        for i in 0..n {
            if position_changed[i] {
                strategy_returns[i] -= slip;
            }
        }
    }

    let mut equity = vec![1.0_f64; n];
    if n > 0 {
        if commission_per_trade <= 0.0 {
            let mut gross = 1.0_f64;
            for i in 0..n {
                gross *= 1.0 + strategy_returns[i];
                equity[i] = gross;
            }
        } else {
            let mut gross_equity = vec![1.0_f64; n];
            let mut gross = 1.0_f64;
            for i in 0..n {
                gross *= 1.0 + strategy_returns[i];
                gross_equity[i] = gross;
            }

            if gross_equity.contains(&0.0) {
                equity[0] = 1.0;
                for i in 1..n {
                    equity[i] = equity[i - 1] * (1.0 + strategy_returns[i]);
                    if position_changed[i] {
                        equity[i] -= commission_per_trade;
                    }
                }
            } else {
                let mut discounted_commissions = 0.0_f64;
                for i in 0..n {
                    if position_changed[i] {
                        discounted_commissions += commission_per_trade / gross_equity[i];
                    }
                    equity[i] = gross_equity[i] * (1.0 - discounted_commissions);
                }
            }
        }
    }

    Ok((
        positions.into_pyarray(py),
        bar_returns.into_pyarray(py),
        strategy_returns.into_pyarray(py),
        equity.into_pyarray(py),
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rsi_threshold_signals, m)?)?;
    m.add_function(wrap_pyfunction!(sma_crossover_signals, m)?)?;
    m.add_function(wrap_pyfunction!(macd_crossover_signals, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_core, m)?)?;
    Ok(())
}
