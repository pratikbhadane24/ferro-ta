//! Regime detection and structural breaks.
//!
//! Functions
//! ---------
//! - `regime_adx`             — label each bar as trend (1) or range (0)
//!   using an ADX threshold.
//! - `regime_combined`        — combine ADX + ATR-ratio rule for more robust
//!   regime labelling.
//! - `detect_breaks_cusum`    — detect structural breaks using a CUSUM-style
//!   cumulative sum approach; returns a binary mask.
//! - `rolling_variance_break` — find indices where rolling variance changes
//!   significantly (volatility regime break).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// regime_adx
// ---------------------------------------------------------------------------

/// Label each bar as **trend** (1) or **range** (0) based on ADX level.
///
/// A bar is labelled "trend" when ``adx[i] > threshold`` (default 25).
///
/// Parameters
/// ----------
/// adx       : 1-D float64 array — ADX values (NaN during warm-up)
/// threshold : float — ADX level above which a bar is "trending" (default 25.0)
///
/// Returns
/// -------
/// 1-D int8 array — ``1`` = trend, ``0`` = range, ``-1`` = NaN (warm-up)
#[pyfunction]
pub fn regime_adx<'py>(
    py: Python<'py>,
    adx: PyReadonlyArray1<'py, f64>,
    threshold: f64,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let a = adx.as_slice()?;
    let out: Vec<i8> = a
        .iter()
        .map(|&v| {
            if v.is_nan() {
                -1i8
            } else if v > threshold {
                1i8
            } else {
                0i8
            }
        })
        .collect();
    Ok(out.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// regime_combined
// ---------------------------------------------------------------------------

/// Label each bar as trend (1) or range (0) using ADX + ATR-ratio rule.
///
/// A bar is "trending" when:
///   ``adx[i] > adx_threshold``  **AND**  ``atr[i] / close[i] > atr_pct_threshold``
///
/// The second condition (ATR as % of price) ensures that the trend has
/// meaningful volatility (avoids labelling flat micro-trends as trending).
///
/// Parameters
/// ----------
/// adx               : 1-D float64 — ADX values
/// atr               : 1-D float64 — ATR values
/// close             : 1-D float64 — close prices (for ATR normalisation)
/// adx_threshold     : float — ADX threshold (default 25.0)
/// atr_pct_threshold : float — minimum ATR/close ratio (default 0.005 = 0.5%)
///
/// Returns
/// -------
/// 1-D int8 — ``1`` = trend, ``0`` = range, ``-1`` = NaN
#[pyfunction]
pub fn regime_combined<'py>(
    py: Python<'py>,
    adx: PyReadonlyArray1<'py, f64>,
    atr: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    adx_threshold: f64,
    atr_pct_threshold: f64,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let a = adx.as_slice()?;
    let r = atr.as_slice()?;
    let c = close.as_slice()?;
    let n = a.len();
    if n != r.len() || n != c.len() {
        return Err(PyValueError::new_err(
            "adx, atr, and close must have the same length",
        ));
    }
    let out: Vec<i8> = (0..n)
        .map(|i| {
            let av = a[i];
            let rv = r[i];
            let cv = c[i];
            if av.is_nan() || rv.is_nan() || cv.is_nan() || cv == 0.0 {
                -1i8
            } else if av > adx_threshold && (rv / cv) > atr_pct_threshold {
                1i8
            } else {
                0i8
            }
        })
        .collect();
    Ok(out.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// detect_breaks_cusum
// ---------------------------------------------------------------------------

/// Detect structural breaks using a CUSUM (cumulative sum) approach.
///
/// CUSUM accumulates deviations from a rolling mean.  When the cumulative
/// sum exceeds ``threshold * std(series)``, a break is flagged.
///
/// Algorithm (simplified one-sided CUSUM on demeaned series):
/// 1. Compute a rolling mean and std over *window* bars.
/// 2. Accumulate the standardised deviation: ``S_i = max(0, S_{i-1} + z_i - slack)``.
/// 3. When ``S_i > threshold``, mark a break and reset the accumulator.
///
/// Parameters
/// ----------
/// series    : 1-D float64 array — the series to monitor (e.g. close prices)
/// window    : int — lookback for mean/std estimation (>= 2)
/// threshold : float — CUSUM threshold in units of std (default 3.0)
/// slack     : float — allowance term (default 0.5)
///
/// Returns
/// -------
/// 1-D int8 array — ``1`` at break bars, ``0`` elsewhere
#[pyfunction]
pub fn detect_breaks_cusum<'py>(
    py: Python<'py>,
    series: PyReadonlyArray1<'py, f64>,
    window: usize,
    threshold: f64,
    slack: f64,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    if window < 2 {
        return Err(PyValueError::new_err("window must be >= 2"));
    }
    let s = series.as_slice()?;
    let n = s.len();
    let mut out = vec![0i8; n];
    if n < window {
        return Ok(out.into_pyarray(py));
    }
    let mut cusum_pos = 0.0_f64;
    let mut cusum_neg = 0.0_f64;
    for i in window..n {
        // Rolling mean and std over [i-window, i)
        let slice = &s[(i - window)..i];
        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        let var: f64 =
            slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (window - 1) as f64;
        let std = var.sqrt();
        if std == 0.0 || std.is_nan() || s[i].is_nan() {
            continue;
        }
        let z = (s[i] - mean) / std;
        cusum_pos = (cusum_pos + z - slack).max(0.0);
        cusum_neg = (cusum_neg - z - slack).max(0.0);
        if cusum_pos > threshold || cusum_neg > threshold {
            out[i] = 1;
            cusum_pos = 0.0;
            cusum_neg = 0.0;
        }
    }
    Ok(out.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// rolling_variance_break
// ---------------------------------------------------------------------------

/// Detect volatility regime breaks using a rolling variance change test.
///
/// Compares the variance in a short lookback window (*short_window*) to a
/// longer reference window (*long_window*).  When their ratio exceeds
/// *threshold*, a volatility break is flagged.
///
/// Parameters
/// ----------
/// series       : 1-D float64 array — returns or price series
/// short_window : int — short lookback for recent variance (>= 2)
/// long_window  : int — long lookback for baseline variance (> short_window)
/// threshold    : float — ratio short_var / long_var above which a break fires
///   (default 2.0)
///
/// Returns
/// -------
/// 1-D int8 array — ``1`` at break bars, ``0`` elsewhere
#[pyfunction]
pub fn rolling_variance_break<'py>(
    py: Python<'py>,
    series: PyReadonlyArray1<'py, f64>,
    short_window: usize,
    long_window: usize,
    threshold: f64,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    if short_window < 2 {
        return Err(PyValueError::new_err("short_window must be >= 2"));
    }
    if long_window <= short_window {
        return Err(PyValueError::new_err("long_window must be > short_window"));
    }
    let s = series.as_slice()?;
    let n = s.len();
    let mut out = vec![0i8; n];
    if n < long_window {
        return Ok(out.into_pyarray(py));
    }

    let variance = |slice: &[f64]| -> f64 {
        let k = slice.len();
        let mean: f64 = slice.iter().sum::<f64>() / k as f64;
        slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (k - 1) as f64
    };

    for i in long_window..n {
        let long_slice = &s[(i - long_window)..i];
        let short_slice = &s[(i - short_window)..i];
        let long_var = variance(long_slice);
        let short_var = variance(short_slice);
        if long_var == 0.0 || long_var.is_nan() || short_var.is_nan() {
            continue;
        }
        if short_var / long_var > threshold {
            out[i] = 1;
        }
    }
    Ok(out.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// Register
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(regime_adx, m)?)?;
    m.add_function(wrap_pyfunction!(regime_combined, m)?)?;
    m.add_function(wrap_pyfunction!(detect_breaks_cusum, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_variance_break, m)?)?;
    Ok(())
}
