//! Extended Indicators — Rust implementations of indicators not in TA-Lib.
//!
//! All compute-heavy work (sequential loops, rolling windows) is done in Rust.
//! Python wrappers in `python/ferro_ta/extended.py` are thin call-throughs that
//! handle input conversion and pandas/polars wrapping.

#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

use std::collections::VecDeque;

use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute ATR array using Wilder smoothing (same as src/volatility/atr.rs).
fn compute_atr(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = high.len();
    let mut result = vec![f64::NAN; n];
    if n <= timeperiod {
        return result;
    }
    // Seed: SMA of first `timeperiod` true range values
    let mut seed_sum = high[0] - low[0]; // first TR has no prev_close
    for i in 1..timeperiod {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        seed_sum += hl.max(hc).max(lc);
    }
    let mut atr = seed_sum / timeperiod as f64;
    result[timeperiod - 1] = atr;
    let pf = (timeperiod - 1) as f64;
    for i in timeperiod..n {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        let tr = hl.max(hc).max(lc);
        atr = (atr * pf + tr) / timeperiod as f64;
        result[i] = atr;
    }
    result
}

/// Compute EMA using ferro_ta_core (SMA-seeded, matches Python EMA).
fn compute_ema_ta(prices: &[f64], timeperiod: usize) -> Vec<f64> {
    ferro_ta_core::overlap::ema(prices, timeperiod)
}

/// Compute WMA array using ferro_ta_core O(n) implementation.
fn compute_wma(prices: &[f64], timeperiod: usize) -> Vec<f64> {
    ferro_ta_core::overlap::wma(prices, timeperiod)
}

// ---------------------------------------------------------------------------
// VWAP
// ---------------------------------------------------------------------------

/// Volume Weighted Average Price (cumulative or rolling).
///
/// Parameters
/// ----------
/// high, low, close, volume : 1-D float64 arrays (equal length)
/// timeperiod : 0 = cumulative from bar 0; >= 1 = rolling window
///
/// Returns
/// -------
/// 1-D float64 array of VWAP values.
#[pyfunction]
#[pyo3(signature = (high, low, close, volume, timeperiod = 0))]
pub fn vwap<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let c = close.as_slice()?;
    let v = volume.as_slice()?;
    let n = h.len();
    validation::validate_equal_length(&[
        (n, "high"),
        (lo.len(), "low"),
        (c.len(), "close"),
        (v.len(), "volume"),
    ])?;

    let mut result = vec![f64::NAN; n];
    let mut cum_tpv = 0.0f64;
    let mut cum_vol = 0.0f64;

    if timeperiod == 0 {
        for i in 0..n {
            let tp = (h[i] + lo[i] + c[i]) / 3.0;
            cum_tpv += tp * v[i];
            cum_vol += v[i];
            result[i] = if cum_vol != 0.0 {
                cum_tpv / cum_vol
            } else {
                f64::NAN
            };
        }
    } else {
        // Pre-compute cumulative sums for O(n) rolling window
        let mut cum_tpv_arr = vec![0.0f64; n];
        let mut cum_vol_arr = vec![0.0f64; n];
        for i in 0..n {
            let tp = (h[i] + lo[i] + c[i]) / 3.0;
            let tpv = tp * v[i];
            cum_tpv_arr[i] = tpv + if i > 0 { cum_tpv_arr[i - 1] } else { 0.0 };
            cum_vol_arr[i] = v[i] + if i > 0 { cum_vol_arr[i - 1] } else { 0.0 };
        }
        for i in (timeperiod - 1)..n {
            let prev_tpv = if i >= timeperiod {
                cum_tpv_arr[i - timeperiod]
            } else {
                0.0
            };
            let prev_vol = if i >= timeperiod {
                cum_vol_arr[i - timeperiod]
            } else {
                0.0
            };
            let w_tpv = cum_tpv_arr[i] - prev_tpv;
            let w_vol = cum_vol_arr[i] - prev_vol;
            result[i] = if w_vol != 0.0 {
                w_tpv / w_vol
            } else {
                f64::NAN
            };
        }
    }
    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// VWMA
// ---------------------------------------------------------------------------

/// Volume Weighted Moving Average.
///
/// VWMA = sum(close * volume, n) / sum(volume, n)
#[pyfunction]
#[pyo3(signature = (close, volume, timeperiod = 20))]
pub fn vwma<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let c = close.as_slice()?;
    let v = volume.as_slice()?;
    let n = c.len();
    validation::validate_equal_length(&[(n, "close"), (v.len(), "volume")])?;

    let mut cum_cv = vec![0.0f64; n];
    let mut cum_v = vec![0.0f64; n];
    for i in 0..n {
        cum_cv[i] = c[i] * v[i] + if i > 0 { cum_cv[i - 1] } else { 0.0 };
        cum_v[i] = v[i] + if i > 0 { cum_v[i - 1] } else { 0.0 };
    }

    let mut result = vec![f64::NAN; n];
    for i in (timeperiod - 1)..n {
        let prev_cv = if i >= timeperiod {
            cum_cv[i - timeperiod]
        } else {
            0.0
        };
        let prev_v = if i >= timeperiod {
            cum_v[i - timeperiod]
        } else {
            0.0
        };
        let w_cv = cum_cv[i] - prev_cv;
        let w_v = cum_v[i] - prev_v;
        result[i] = if w_v != 0.0 { w_cv / w_v } else { f64::NAN };
    }
    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// SUPERTREND
// ---------------------------------------------------------------------------

/// ATR-based Supertrend indicator.
///
/// Returns (supertrend_line, direction) where direction is an int8 array:
/// 1 = uptrend, -1 = downtrend, 0 = warmup.
#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 7, multiplier = 3.0))]
pub fn supertrend<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    multiplier: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i8>>)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let c = close.as_slice()?;
    let n = h.len();
    validation::validate_equal_length(&[(n, "high"), (lo.len(), "low"), (c.len(), "close")])?;

    let atr = compute_atr(h, lo, c, timeperiod);

    let mut supertrend_out = vec![f64::NAN; n];
    let mut direction = vec![0i8; n];
    let mut upper_band = vec![f64::NAN; n];
    let mut lower_band = vec![f64::NAN; n];

    let first_valid = timeperiod - 1;
    if first_valid >= n || atr[first_valid].is_nan() {
        return Ok((supertrend_out.into_pyarray(py), direction.into_pyarray(py)));
    }

    // Compute basic bands
    let mut upper_basic = vec![f64::NAN; n];
    let mut lower_basic = vec![f64::NAN; n];
    for i in 0..n {
        if !atr[i].is_nan() {
            let hl2 = (h[i] + lo[i]) / 2.0;
            upper_basic[i] = hl2 + multiplier * atr[i];
            lower_basic[i] = hl2 - multiplier * atr[i];
        }
    }

    // Initialize band state at first valid ATR bar
    upper_band[first_valid] = upper_basic[first_valid];
    lower_band[first_valid] = lower_basic[first_valid];
    // Keep supertrend_out NaN and direction 0 for indices 0..timeperiod (warmup)

    for i in (first_valid + 1)..n {
        if atr[i].is_nan() {
            continue;
        }

        // Adjust lower band
        lower_band[i] = if lower_basic[i] > lower_band[i - 1] || c[i - 1] < lower_band[i - 1] {
            lower_basic[i]
        } else {
            lower_band[i - 1]
        };

        // Adjust upper band
        upper_band[i] = if upper_basic[i] < upper_band[i - 1] || c[i - 1] > upper_band[i - 1] {
            upper_basic[i]
        } else {
            upper_band[i - 1]
        };

        // Direction and output only from index timeperiod (warmup = 0, NaN)
        if i >= timeperiod {
            let prev_dir = direction[i - 1];
            direction[i] = if prev_dir == 0 {
                // First output bar: bootstrap with -1 (downtrend) or 1 from price vs band
                if c[i] > upper_band[i] {
                    1
                } else {
                    -1
                }
            } else if prev_dir == -1 {
                if c[i] > upper_band[i] {
                    1
                } else {
                    -1
                }
            } else if c[i] < lower_band[i] {
                -1
            } else {
                1
            };
            supertrend_out[i] = if direction[i] == 1 {
                lower_band[i]
            } else {
                upper_band[i]
            };
        }
    }

    Ok((supertrend_out.into_pyarray(py), direction.into_pyarray(py)))
}

// ---------------------------------------------------------------------------
// DONCHIAN
// ---------------------------------------------------------------------------

/// Donchian Channels — rolling highest high / lowest low.
///
/// Returns (upper, middle, lower) arrays.
#[pyfunction]
#[pyo3(signature = (high, low, timeperiod = 20))]
pub fn donchian<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let n = h.len();
    validation::validate_equal_length(&[(n, "high"), (lo.len(), "low")])?;

    let mut upper = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];
    let mut middle = vec![f64::NAN; n];

    // Use monotonic deque for O(n) sliding max / min
    let mut max_dq: VecDeque<usize> = VecDeque::new();
    let mut min_dq: VecDeque<usize> = VecDeque::new();

    for i in 0..n {
        // Remove out-of-window indices
        while max_dq
            .front()
            .map(|&j| j + timeperiod <= i)
            .unwrap_or(false)
        {
            max_dq.pop_front();
        }
        while min_dq
            .front()
            .map(|&j| j + timeperiod <= i)
            .unwrap_or(false)
        {
            min_dq.pop_front();
        }
        // Maintain decreasing deque for max
        while max_dq.back().map(|&j| h[j] <= h[i]).unwrap_or(false) {
            max_dq.pop_back();
        }
        max_dq.push_back(i);
        // Maintain increasing deque for min
        while min_dq.back().map(|&j| lo[j] >= lo[i]).unwrap_or(false) {
            min_dq.pop_back();
        }
        min_dq.push_back(i);

        if i + 1 >= timeperiod {
            upper[i] = h[*max_dq.front().unwrap()];
            lower[i] = lo[*min_dq.front().unwrap()];
            middle[i] = (upper[i] + lower[i]) / 2.0;
        }
    }

    Ok((
        upper.into_pyarray(py),
        middle.into_pyarray(py),
        lower.into_pyarray(py),
    ))
}

// ---------------------------------------------------------------------------
// CHOPPINESS_INDEX
// ---------------------------------------------------------------------------

/// Choppiness Index — measures market choppiness vs trending.
///
/// Values near 100 → choppy; near 0 → trending.
/// Leading `timeperiod` values are NaN.
#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 14))]
pub fn choppiness_index<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let c = close.as_slice()?;
    let n = h.len();
    validation::validate_equal_length(&[(n, "high"), (lo.len(), "low"), (c.len(), "close")])?;

    // ATR(1) = True Range per bar
    let mut tr = vec![0.0f64; n];
    tr[0] = h[0] - lo[0];
    for i in 1..n {
        let hl = h[i] - lo[i];
        let hc = (h[i] - c[i - 1]).abs();
        let lc = (lo[i] - c[i - 1]).abs();
        tr[i] = hl.max(hc).max(lc);
    }

    // Cumulative TR for rolling sum
    let mut cum_tr = vec![0.0f64; n];
    cum_tr[0] = tr[0];
    for i in 1..n {
        cum_tr[i] = cum_tr[i - 1] + tr[i];
    }

    let log_n = (timeperiod as f64).log10();
    let mut result = vec![f64::NAN; n];

    // Rolling max and min using monotonic deques
    let mut max_dq: VecDeque<usize> = VecDeque::new();
    let mut min_dq: VecDeque<usize> = VecDeque::new();

    for i in 0..n {
        while max_dq
            .front()
            .map(|&j| j + timeperiod <= i)
            .unwrap_or(false)
        {
            max_dq.pop_front();
        }
        while min_dq
            .front()
            .map(|&j| j + timeperiod <= i)
            .unwrap_or(false)
        {
            min_dq.pop_front();
        }
        while max_dq.back().map(|&j| h[j] <= h[i]).unwrap_or(false) {
            max_dq.pop_back();
        }
        max_dq.push_back(i);
        while min_dq.back().map(|&j| lo[j] >= lo[i]).unwrap_or(false) {
            min_dq.pop_back();
        }
        min_dq.push_back(i);

        if i + 1 > timeperiod {
            let prev_cum = if i >= timeperiod {
                cum_tr[i - timeperiod]
            } else {
                0.0
            };
            let sum_tr = cum_tr[i] - prev_cum;
            let hh = h[*max_dq.front().unwrap()];
            let ll = lo[*min_dq.front().unwrap()];
            let hl_range = hh - ll;
            if hl_range > 0.0 && log_n > 0.0 {
                result[i] = 100.0 * (sum_tr / hl_range).log10() / log_n;
            }
        }
    }

    Ok(result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// KELTNER_CHANNELS
// ---------------------------------------------------------------------------

/// Keltner Channels — EMA ± (multiplier × ATR).
///
/// Returns (upper, middle, lower) arrays.
#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 20, atr_period = 10, multiplier = 2.0))]
pub fn keltner_channels<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    atr_period: usize,
    multiplier: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    validation::validate_timeperiod(atr_period, "atr_period", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let c = close.as_slice()?;
    let n = h.len();
    validation::validate_equal_length(&[(n, "high"), (lo.len(), "low"), (c.len(), "close")])?;

    let middle = compute_ema_ta(c, timeperiod);
    let atr = compute_atr(h, lo, c, atr_period);

    let mut upper = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];
    for i in 0..n {
        if !middle[i].is_nan() && !atr[i].is_nan() {
            let band = multiplier * atr[i];
            upper[i] = middle[i] + band;
            lower[i] = middle[i] - band;
        }
    }

    Ok((
        upper.into_pyarray(py),
        middle.into_pyarray(py),
        lower.into_pyarray(py),
    ))
}

// ---------------------------------------------------------------------------
// HULL_MA
// ---------------------------------------------------------------------------

/// Hull Moving Average (HMA).
///
/// Formula: HMA(n) = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 16))]
pub fn hull_ma<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let c = close.as_slice()?;
    let n = c.len();

    let half = (timeperiod / 2).max(1);
    let sqrt_p = ((timeperiod as f64).sqrt().round() as usize).max(1);

    let wma_full = compute_wma(c, timeperiod);
    let wma_half = compute_wma(c, half);

    // raw = 2 * wma_half - wma_full
    let mut raw = vec![f64::NAN; n];
    for i in 0..n {
        if !wma_full[i].is_nan() && !wma_half[i].is_nan() {
            raw[i] = 2.0 * wma_half[i] - wma_full[i];
        }
    }

    // Find first valid index in raw
    let first_valid = raw.iter().position(|x| !x.is_nan()).unwrap_or(n);
    let mut hull = vec![f64::NAN; n];
    if first_valid < n {
        let raw_valid = &raw[first_valid..];
        let hma_slice = compute_wma(raw_valid, sqrt_p);
        for (k, &v) in hma_slice.iter().enumerate() {
            hull[first_valid + k] = v;
        }
    }

    Ok(hull.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// CHANDELIER_EXIT
// ---------------------------------------------------------------------------

/// Chandelier Exit — ATR-based trailing stop levels.
///
/// Returns (long_exit, short_exit) arrays.
#[pyfunction]
#[pyo3(signature = (high, low, close, timeperiod = 22, multiplier = 3.0))]
pub fn chandelier_exit<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    multiplier: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let c = close.as_slice()?;
    let n = h.len();
    validation::validate_equal_length(&[(n, "high"), (lo.len(), "low"), (c.len(), "close")])?;

    let atr = compute_atr(h, lo, c, timeperiod);

    // Rolling max/min using monotonic deques
    let mut max_dq: VecDeque<usize> = VecDeque::new();
    let mut min_dq: VecDeque<usize> = VecDeque::new();
    let mut highest_high = vec![f64::NAN; n];
    let mut lowest_low = vec![f64::NAN; n];

    for i in 0..n {
        while max_dq
            .front()
            .map(|&j| j + timeperiod <= i)
            .unwrap_or(false)
        {
            max_dq.pop_front();
        }
        while min_dq
            .front()
            .map(|&j| j + timeperiod <= i)
            .unwrap_or(false)
        {
            min_dq.pop_front();
        }
        while max_dq.back().map(|&j| h[j] <= h[i]).unwrap_or(false) {
            max_dq.pop_back();
        }
        max_dq.push_back(i);
        while min_dq.back().map(|&j| lo[j] >= lo[i]).unwrap_or(false) {
            min_dq.pop_back();
        }
        min_dq.push_back(i);

        if i + 1 >= timeperiod {
            highest_high[i] = h[*max_dq.front().unwrap()];
            lowest_low[i] = lo[*min_dq.front().unwrap()];
        }
    }

    let mut long_exit = vec![f64::NAN; n];
    let mut short_exit = vec![f64::NAN; n];
    for i in 0..n {
        if !highest_high[i].is_nan() && !atr[i].is_nan() {
            long_exit[i] = highest_high[i] - multiplier * atr[i];
            short_exit[i] = lowest_low[i] + multiplier * atr[i];
        }
    }

    Ok((long_exit.into_pyarray(py), short_exit.into_pyarray(py)))
}

// ---------------------------------------------------------------------------
// ICHIMOKU
// ---------------------------------------------------------------------------

/// Ichimoku Cloud (Ichimoku Kinko Hyo).
///
/// Returns (tenkan, kijun, senkou_a, senkou_b, chikou) arrays.
#[pyfunction]
#[pyo3(signature = (high, low, close, tenkan_period = 9, kijun_period = 26, senkou_b_period = 52, displacement = 26))]
pub fn ichimoku<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    tenkan_period: usize,
    kijun_period: usize,
    senkou_b_period: usize,
    displacement: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    validation::validate_timeperiod(tenkan_period, "tenkan_period", 1)?;
    validation::validate_timeperiod(kijun_period, "kijun_period", 1)?;
    validation::validate_timeperiod(senkou_b_period, "senkou_b_period", 1)?;
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let c = close.as_slice()?;
    let n = h.len();
    validation::validate_equal_length(&[(n, "high"), (lo.len(), "low"), (c.len(), "close")])?;

    // Helper: rolling (H+L)/2 using monotonic deques
    let midpoint_rolling = |period: usize| -> Vec<f64> {
        let mut result = vec![f64::NAN; n];
        let mut max_dq: VecDeque<usize> = VecDeque::new();
        let mut min_dq: VecDeque<usize> = VecDeque::new();
        for i in 0..n {
            while max_dq.front().map(|&j| j + period <= i).unwrap_or(false) {
                max_dq.pop_front();
            }
            while min_dq.front().map(|&j| j + period <= i).unwrap_or(false) {
                min_dq.pop_front();
            }
            while max_dq.back().map(|&j| h[j] <= h[i]).unwrap_or(false) {
                max_dq.pop_back();
            }
            max_dq.push_back(i);
            while min_dq.back().map(|&j| lo[j] >= lo[i]).unwrap_or(false) {
                min_dq.pop_back();
            }
            min_dq.push_back(i);
            if i + 1 >= period {
                result[i] = (h[*max_dq.front().unwrap()] + lo[*min_dq.front().unwrap()]) / 2.0;
            }
        }
        result
    };

    let tenkan = midpoint_rolling(tenkan_period);
    let kijun = midpoint_rolling(kijun_period);
    let raw_b = midpoint_rolling(senkou_b_period);

    // Senkou A: (tenkan + kijun) / 2 shifted back `displacement` bars
    let mut senkou_a = vec![f64::NAN; n];
    if n > displacement {
        for i in displacement..n {
            if !tenkan[i].is_nan() && !kijun[i].is_nan() {
                senkou_a[i - displacement] = (tenkan[i] + kijun[i]) / 2.0;
            }
        }
    }

    // Senkou B: raw_b shifted back `displacement` bars
    let mut senkou_b = vec![f64::NAN; n];
    if n > displacement {
        senkou_b[..n - displacement].copy_from_slice(&raw_b[displacement..]);
    }

    // Chikou: close shifted forward `displacement` bars
    let mut chikou = vec![f64::NAN; n];
    if n > displacement {
        chikou[displacement..].copy_from_slice(&c[..n - displacement]);
    }

    Ok((
        tenkan.into_pyarray(py),
        kijun.into_pyarray(py),
        senkou_a.into_pyarray(py),
        senkou_b.into_pyarray(py),
        chikou.into_pyarray(py),
    ))
}

// ---------------------------------------------------------------------------
// PIVOT_POINTS
// ---------------------------------------------------------------------------

/// Pivot Points — support / resistance levels computed from previous bar.
///
/// method: "classic" | "fibonacci" | "camarilla"
/// Returns (pivot, r1, s1, r2, s2) arrays.
#[pyfunction]
#[pyo3(signature = (high, low, close, method = "classic"))]
pub fn pivot_points<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    method: &str,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let h = high.as_slice()?;
    let lo = low.as_slice()?;
    let c = close.as_slice()?;
    let n = h.len();
    validation::validate_equal_length(&[(n, "high"), (lo.len(), "low"), (c.len(), "close")])?;

    let mut pivot = vec![f64::NAN; n];
    let mut r1 = vec![f64::NAN; n];
    let mut s1 = vec![f64::NAN; n];
    let mut r2 = vec![f64::NAN; n];
    let mut s2 = vec![f64::NAN; n];

    let method_lower = method.to_lowercase();
    for i in 1..n {
        let ph = h[i - 1];
        let pl = lo[i - 1];
        let pc = c[i - 1];
        let hl = ph - pl;
        let p = (ph + pl + pc) / 3.0;
        pivot[i] = p;
        match method_lower.as_str() {
            "classic" => {
                r1[i] = 2.0 * p - pl;
                s1[i] = 2.0 * p - ph;
                r2[i] = p + hl;
                s2[i] = p - hl;
            }
            "fibonacci" => {
                r1[i] = p + 0.382 * hl;
                s1[i] = p - 0.382 * hl;
                r2[i] = p + 0.618 * hl;
                s2[i] = p - 0.618 * hl;
            }
            "camarilla" => {
                r1[i] = pc + 1.1 * hl / 12.0;
                s1[i] = pc - 1.1 * hl / 12.0;
                r2[i] = pc + 1.1 * hl / 6.0;
                s2[i] = pc - 1.1 * hl / 6.0;
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown pivot method '{}'. Use 'classic', 'fibonacci', or 'camarilla'.",
                    method
                )));
            }
        }
    }

    Ok((
        pivot.into_pyarray(py),
        r1.into_pyarray(py),
        s1.into_pyarray(py),
        r2.into_pyarray(py),
        s2.into_pyarray(py),
    ))
}

// ---------------------------------------------------------------------------
// register
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(vwap, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(vwma, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(supertrend, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(donchian, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(choppiness_index, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(keltner_channels, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(hull_ma, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(chandelier_exit, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ichimoku, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(pivot_points, m)?)?;
    Ok(())
}
