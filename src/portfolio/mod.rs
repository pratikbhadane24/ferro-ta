//! Portfolio Analytics — thin PyO3 wrappers delegating to `ferro_ta_core::portfolio`.
//!
//! Compute-intensive portfolio metrics implemented in Rust:
//! - `portfolio_volatility`  — sqrt(w' Σ w) given weights and a covariance matrix
//! - `beta_series`           — rolling or full beta of asset vs benchmark
//! - `drawdown_series`       — per-bar drawdown and underwater series
//! - `correlation_matrix`    — pairwise Pearson correlation (n_assets × n_assets)
//! - `relative_strength`     — cumulative return ratio (asset / benchmark)
//! - `spread`                — A - hedge * B
//! - `zscore_series`         — rolling Z-score of a 1-D series
//! - `rolling_beta`          — rolling beta (hedge ratio) of two series

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// portfolio_volatility
// ---------------------------------------------------------------------------

/// Compute portfolio volatility: sqrt(w' Σ w).
///
/// Parameters
/// ----------
/// cov_matrix : 2-D float64 array, shape (n, n) — covariance matrix
/// weights    : 1-D float64 array, length n — portfolio weights (need not sum to 1)
///
/// Returns
/// -------
/// float — portfolio volatility (annualisation is the caller's responsibility)
#[pyfunction]
pub fn portfolio_volatility<'py>(
    cov_matrix: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    let cov = cov_matrix.as_array();
    let w = weights.as_slice()?;
    let n = w.len();
    let (rows, cols) = cov.dim();
    if rows != n || cols != n {
        return Err(PyValueError::new_err(format!(
            "cov_matrix must be ({n}, {n}), got ({rows}, {cols})"
        )));
    }
    // Convert ndarray rows to Vec<Vec<f64>> for core
    let cov_rows: Vec<Vec<f64>> = (0..rows)
        .map(|i| (0..cols).map(|j| cov[[i, j]]).collect())
        .collect();
    Ok(ferro_ta_core::portfolio::portfolio_volatility(&cov_rows, w))
}

// ---------------------------------------------------------------------------
// beta_full
// ---------------------------------------------------------------------------

/// Compute the full-sample beta of `asset_returns` to `benchmark_returns`.
///
/// Beta = Cov(asset, bench) / Var(bench)  (OLS regression slope).
///
/// Parameters
/// ----------
/// asset_returns, benchmark_returns : 1-D float64 arrays (equal length, >= 2 elements)
///
/// Returns
/// -------
/// float — beta
#[pyfunction]
pub fn beta_full<'py>(
    asset_returns: PyReadonlyArray1<'py, f64>,
    benchmark_returns: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    let a = asset_returns.as_slice()?;
    let b = benchmark_returns.as_slice()?;
    let n = a.len();
    if n < 2 || b.len() != n {
        return Err(PyValueError::new_err(
            "asset_returns and benchmark_returns must have equal length >= 2",
        ));
    }
    Ok(ferro_ta_core::portfolio::beta_full(a, b))
}

// ---------------------------------------------------------------------------
// rolling_beta
// ---------------------------------------------------------------------------

/// Compute rolling beta of `asset` vs `benchmark` over a sliding window.
///
/// Parameters
/// ----------
/// asset, benchmark : 1-D float64 arrays (equal length)
/// window : int — rolling window size (must be >= 2)
///
/// Returns
/// -------
/// 1-D float64 array — NaN for first `window-1` positions.
#[pyfunction]
#[pyo3(signature = (asset, benchmark, window))]
pub fn rolling_beta<'py>(
    py: Python<'py>,
    asset: PyReadonlyArray1<'py, f64>,
    benchmark: PyReadonlyArray1<'py, f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if window < 2 {
        return Err(PyValueError::new_err("window must be >= 2"));
    }
    let a = asset.as_slice()?;
    let b = benchmark.as_slice()?;
    let n = a.len();
    if n == 0 || b.len() != n {
        return Err(PyValueError::new_err(
            "asset and benchmark must be non-empty and equal length",
        ));
    }
    let result = ferro_ta_core::portfolio::rolling_beta(a, b, window);
    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// drawdown_series
// ---------------------------------------------------------------------------

/// Compute the drawdown series and maximum drawdown for an equity/price series.
///
/// Drawdown at bar i = (equity[i] - running_max) / running_max  (always <= 0).
///
/// Parameters
/// ----------
/// equity : 1-D float64 array — equity or price series
///
/// Returns
/// -------
/// (drawdown, max_drawdown)
///   drawdown     : 1-D float64 array (same length as equity)
///   max_drawdown : float — worst (most negative) drawdown observed
#[pyfunction]
pub fn drawdown_series<'py>(
    py: Python<'py>,
    equity: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let eq = equity.as_slice()?;
    if eq.is_empty() {
        return Err(PyValueError::new_err("equity must be non-empty"));
    }
    let (dd, max_dd) = ferro_ta_core::portfolio::drawdown_series(eq);
    Ok((dd.into_pyarray(py), max_dd))
}

// ---------------------------------------------------------------------------
// correlation_matrix
// ---------------------------------------------------------------------------

/// Compute the pairwise Pearson correlation matrix for a returns DataFrame.
///
/// Parameters
/// ----------
/// data : 2-D float64 array, shape (n_bars, n_assets) — returns per bar/asset
///
/// Returns
/// -------
/// 2-D float64 array, shape (n_assets, n_assets) — correlation matrix
#[pyfunction]
pub fn correlation_matrix<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = data.as_array();
    let (n_bars, n_assets) = arr.dim();
    if n_bars < 2 {
        return Err(PyValueError::new_err("data must have at least 2 rows"));
    }
    // Core expects column vectors: data[j][i] = asset j at bar i
    let columns: Vec<Vec<f64>> = (0..n_assets)
        .map(|j| (0..n_bars).map(|i| arr[[i, j]]).collect())
        .collect();
    let corr = ferro_ta_core::portfolio::correlation_matrix(&columns);
    // Convert Vec<Vec<f64>> back to ndarray::Array2
    let mut result = Array2::<f64>::zeros((n_assets, n_assets));
    for j1 in 0..n_assets {
        for j2 in 0..n_assets {
            result[[j1, j2]] = corr[j1][j2];
        }
    }
    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// relative_strength
// ---------------------------------------------------------------------------

/// Compute relative strength of an asset vs a benchmark.
///
/// result[i] = (1 + asset_returns[i]).cumprod() / (1 + bench_returns[i]).cumprod()
/// starting from 1.0.
///
/// Parameters
/// ----------
/// asset_returns, benchmark_returns : 1-D float64 arrays (equal length)
///   Fractional returns per bar (e.g. 0.01 for +1%).
///
/// Returns
/// -------
/// 1-D float64 array — relative strength (ratio of cumulative returns).
#[pyfunction]
pub fn relative_strength<'py>(
    py: Python<'py>,
    asset_returns: PyReadonlyArray1<'py, f64>,
    benchmark_returns: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = asset_returns.as_slice()?;
    let b = benchmark_returns.as_slice()?;
    let n = a.len();
    if n == 0 || b.len() != n {
        return Err(PyValueError::new_err(
            "asset_returns and benchmark_returns must be non-empty and equal length",
        ));
    }
    let result = ferro_ta_core::portfolio::relative_strength(a, b);
    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// spread
// ---------------------------------------------------------------------------

/// Compute the spread between two series: A - hedge * B.
///
/// Parameters
/// ----------
/// a, b     : 1-D float64 arrays (equal length)
/// hedge    : float — hedge ratio (default 1.0)
///
/// Returns
/// -------
/// 1-D float64 array
#[pyfunction]
#[pyo3(signature = (a, b, hedge = 1.0))]
pub fn spread<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    hedge: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let av = a.as_slice()?;
    let bv = b.as_slice()?;
    let n = av.len();
    if n == 0 || bv.len() != n {
        return Err(PyValueError::new_err(
            "a and b must be non-empty and equal length",
        ));
    }
    let result = ferro_ta_core::portfolio::spread(av, bv, hedge);
    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// ratio
// ---------------------------------------------------------------------------

/// Compute the ratio between two series: A / B.
///
/// Where B is 0, returns NaN.
#[pyfunction]
pub fn ratio<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let av = a.as_slice()?;
    let bv = b.as_slice()?;
    let n = av.len();
    if n == 0 || bv.len() != n {
        return Err(PyValueError::new_err(
            "a and b must be non-empty and equal length",
        ));
    }
    let result = ferro_ta_core::portfolio::ratio(av, bv);
    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// zscore_series
// ---------------------------------------------------------------------------

/// Compute the rolling Z-score of a 1-D series.
///
/// Z[i] = (x[i] - mean(x[i-window+1..=i])) / std(x[i-window+1..=i])
///
/// Parameters
/// ----------
/// x      : 1-D float64 array
/// window : int — rolling window (must be >= 2)
///
/// Returns
/// -------
/// 1-D float64 array — NaN for first `window-1` positions.
#[pyfunction]
#[pyo3(signature = (x, window))]
pub fn zscore_series<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if window < 2 {
        return Err(PyValueError::new_err("window must be >= 2"));
    }
    let xv = x.as_slice()?;
    if xv.is_empty() {
        return Err(PyValueError::new_err("x must be non-empty"));
    }
    let result = ferro_ta_core::portfolio::zscore_series(xv, window);
    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// compose_weighted
// ---------------------------------------------------------------------------

/// Weighted combination of multiple signal columns.
///
/// Parameters
/// ----------
/// data    : 2-D float64 array, shape (n_bars, n_signals)
/// weights : 1-D float64 array, length n_signals
///
/// Returns
/// -------
/// 1-D float64 array — weighted sum per bar
#[pyfunction]
pub fn compose_weighted<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let arr = data.as_array();
    let w = weights.as_slice()?;
    let (n_bars, n_sigs) = arr.dim();
    if w.len() != n_sigs {
        return Err(PyValueError::new_err(format!(
            "weights length ({}) must equal number of signal columns ({})",
            w.len(),
            n_sigs
        )));
    }
    // Core expects column vectors: data[j][i] = signal j at bar i
    let columns: Vec<Vec<f64>> = (0..n_sigs)
        .map(|j| (0..n_bars).map(|i| arr[[i, j]]).collect())
        .collect();
    let result = ferro_ta_core::portfolio::compose_weighted(&columns, w);
    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// Register
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(portfolio_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(beta_full, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_beta, m)?)?;
    m.add_function(wrap_pyfunction!(drawdown_series, m)?)?;
    m.add_function(wrap_pyfunction!(correlation_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(relative_strength, m)?)?;
    m.add_function(wrap_pyfunction!(spread, m)?)?;
    m.add_function(wrap_pyfunction!(ratio, m)?)?;
    m.add_function(wrap_pyfunction!(zscore_series, m)?)?;
    m.add_function(wrap_pyfunction!(compose_weighted, m)?)?;
    Ok(())
}
