//! Streaming / Incremental Indicators — bar-by-bar stateful classes.
//!
//! All classes are exposed as PyO3 `#[pyclass]` types.  Each class:
//! - Accepts one value per call to `update()`.
//! - Returns `NaN` (or a NaN tuple) during the warm-up window.
//! - Exposes a `reset()` method to restart from scratch.
//! - Has a `period` property (where applicable).
//!
//! Internal EMA state is shared via the non-pyclass `EmaState` helper so
//! composite classes (`StreamingMACD`, `StreamingSupertrend`) can hold
//! multiple EMA states without additional allocations.

use std::collections::VecDeque;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Internal helper: EMA state (not a pyclass — used inside composite classes)
// ---------------------------------------------------------------------------

struct EmaState {
    period: usize,
    alpha: f64,
    ema: f64,
    seed_buf: Vec<f64>,
    seeded: bool,
}

impl EmaState {
    fn new(period: usize) -> Self {
        Self {
            period,
            alpha: 2.0 / (period as f64 + 1.0),
            ema: 0.0,
            seed_buf: Vec::with_capacity(period),
            seeded: false,
        }
    }

    fn update(&mut self, value: f64) -> f64 {
        if !self.seeded {
            self.seed_buf.push(value);
            if self.seed_buf.len() < self.period {
                return f64::NAN;
            }
            let seed = self.seed_buf.iter().sum::<f64>() / self.period as f64;
            self.ema = seed;
            self.seeded = true;
            log::debug!(
                "EmaState warm-up complete: period={}, seed={seed:.6}",
                self.period
            );
            return seed;
        }
        self.ema += self.alpha * (value - self.ema);
        self.ema
    }

    fn reset(&mut self) {
        self.ema = 0.0;
        self.seed_buf.clear();
        self.seeded = false;
    }
}

// ---------------------------------------------------------------------------
// Internal helper: ATR state (Wilder smoothing)
// ---------------------------------------------------------------------------

struct AtrState {
    period: usize,
    prev_close: f64,
    tr_buf: Vec<f64>,
    atr: f64,
    seeded: bool,
    has_prev: bool,
}

impl AtrState {
    fn new(period: usize) -> Self {
        Self {
            period,
            prev_close: 0.0,
            tr_buf: Vec::with_capacity(period),
            atr: 0.0,
            seeded: false,
            has_prev: false,
        }
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> f64 {
        let tr = if self.has_prev {
            let hl = high - low;
            let hc = (high - self.prev_close).abs();
            let lc = (low - self.prev_close).abs();
            hl.max(hc).max(lc)
        } else {
            high - low
        };
        self.prev_close = close;
        self.has_prev = true;

        if !self.seeded {
            self.tr_buf.push(tr);
            if self.tr_buf.len() < self.period {
                return f64::NAN;
            }
            let seed = self.tr_buf.iter().sum::<f64>() / self.period as f64;
            self.atr = seed;
            self.seeded = true;
            return f64::NAN; // first `period` bars (including this one) return NaN
        }
        let pf = (self.period - 1) as f64;
        self.atr = (self.atr * pf + tr) / self.period as f64;
        self.atr
    }

    fn reset(&mut self) {
        self.prev_close = 0.0;
        self.has_prev = false;
        self.tr_buf.clear();
        self.atr = 0.0;
        self.seeded = false;
    }
}

// ---------------------------------------------------------------------------
// StreamingSMA
// ---------------------------------------------------------------------------

/// Simple Moving Average — O(1) per update via running sum.
///
/// Returns NaN during the first `period - 1` bars.
#[pyclass(module = "ferro_ta._ferro_ta")]
pub struct StreamingSMA {
    period: usize,
    buf: VecDeque<f64>,
    running_sum: f64,
    count: usize,
}

#[pymethods]
impl StreamingSMA {
    #[new]
    #[pyo3(signature = (period))]
    pub fn new(period: usize) -> PyResult<Self> {
        if period < 1 {
            return Err(PyValueError::new_err("period must be >= 1"));
        }
        Ok(Self {
            period,
            buf: VecDeque::with_capacity(period + 1),
            running_sum: 0.0,
            count: 0,
        })
    }

    /// Add a new bar and return the current SMA (NaN during warmup).
    pub fn update(&mut self, value: f64) -> f64 {
        if self.buf.len() == self.period {
            if let Some(old) = self.buf.pop_front() {
                self.running_sum -= old;
            }
        }
        self.buf.push_back(value);
        self.running_sum += value;
        self.count += 1;
        if self.count < self.period {
            f64::NAN
        } else {
            self.running_sum / self.period as f64
        }
    }

    /// Reset state to initial condition.
    pub fn reset(&mut self) {
        self.buf.clear();
        self.running_sum = 0.0;
        self.count = 0;
    }

    #[getter]
    pub fn period(&self) -> usize {
        self.period
    }

    fn __repr__(&self) -> String {
        format!("StreamingSMA(period={})", self.period)
    }
}

// ---------------------------------------------------------------------------
// StreamingEMA
// ---------------------------------------------------------------------------

/// Exponential Moving Average with SMA seeding.
///
/// Uses a simple SMA for the first `period` bars to seed the EMA, then
/// switches to the standard EMA formula (alpha = 2 / (period + 1)).
/// Returns NaN during the warmup window.
#[pyclass(module = "ferro_ta._ferro_ta")]
pub struct StreamingEMA {
    inner: EmaState,
}

#[pymethods]
impl StreamingEMA {
    #[new]
    #[pyo3(signature = (period))]
    pub fn new(period: usize) -> PyResult<Self> {
        if period < 1 {
            return Err(PyValueError::new_err("period must be >= 1"));
        }
        Ok(Self {
            inner: EmaState::new(period),
        })
    }

    /// Add a new bar and return the current EMA (NaN during warmup).
    pub fn update(&mut self, value: f64) -> f64 {
        self.inner.update(value)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    pub fn period(&self) -> usize {
        self.inner.period
    }

    fn __repr__(&self) -> String {
        format!("StreamingEMA(period={})", self.inner.period)
    }
}

// ---------------------------------------------------------------------------
// StreamingRSI
// ---------------------------------------------------------------------------

/// Relative Strength Index with TA-Lib–compatible Wilder seeding.
///
/// Returns NaN during the first `period` bars.
#[pyclass(module = "ferro_ta._ferro_ta")]
pub struct StreamingRSI {
    period: usize,
    prev: f64,
    has_prev: bool,
    gains: Vec<f64>,
    losses: Vec<f64>,
    avg_gain: f64,
    avg_loss: f64,
    seeded: bool,
}

#[pymethods]
impl StreamingRSI {
    #[new]
    #[pyo3(signature = (period = 14))]
    pub fn new(period: usize) -> PyResult<Self> {
        if period < 1 {
            return Err(PyValueError::new_err("period must be >= 1"));
        }
        Ok(Self {
            period,
            prev: 0.0,
            has_prev: false,
            gains: Vec::with_capacity(period),
            losses: Vec::with_capacity(period),
            avg_gain: 0.0,
            avg_loss: 0.0,
            seeded: false,
        })
    }

    /// Add a new close and return RSI in [0, 100] (NaN during warmup).
    pub fn update(&mut self, value: f64) -> f64 {
        if !self.has_prev {
            self.prev = value;
            self.has_prev = true;
            return f64::NAN;
        }
        let delta = value - self.prev;
        self.prev = value;
        let gain = if delta > 0.0 { delta } else { 0.0 };
        let loss = if delta < 0.0 { -delta } else { 0.0 };

        if !self.seeded {
            self.gains.push(gain);
            self.losses.push(loss);
            if self.gains.len() < self.period {
                return f64::NAN;
            }
            self.avg_gain = self.gains.iter().sum::<f64>() / self.period as f64;
            self.avg_loss = self.losses.iter().sum::<f64>() / self.period as f64;
            self.seeded = true;
            log::debug!("StreamingRSI warm-up complete: period={}", self.period);
        } else {
            let pf = (self.period - 1) as f64;
            self.avg_gain = (self.avg_gain * pf + gain) / self.period as f64;
            self.avg_loss = (self.avg_loss * pf + loss) / self.period as f64;
        }

        if self.avg_loss == 0.0 {
            return 100.0;
        }
        let rs = self.avg_gain / self.avg_loss;
        100.0 - 100.0 / (1.0 + rs)
    }

    pub fn reset(&mut self) {
        self.prev = 0.0;
        self.has_prev = false;
        self.gains.clear();
        self.losses.clear();
        self.avg_gain = 0.0;
        self.avg_loss = 0.0;
        self.seeded = false;
    }

    #[getter]
    pub fn period(&self) -> usize {
        self.period
    }

    fn __repr__(&self) -> String {
        format!("StreamingRSI(period={})", self.period)
    }
}

// ---------------------------------------------------------------------------
// StreamingATR
// ---------------------------------------------------------------------------

/// Average True Range with TA-Lib–compatible Wilder seeding.
///
/// Accepts (high, low, close) per bar.
/// Returns NaN during the first `period` bars.
#[pyclass(module = "ferro_ta._ferro_ta")]
pub struct StreamingATR {
    inner: AtrState,
}

#[pymethods]
impl StreamingATR {
    #[new]
    #[pyo3(signature = (period = 14))]
    pub fn new(period: usize) -> PyResult<Self> {
        if period < 1 {
            return Err(PyValueError::new_err("period must be >= 1"));
        }
        Ok(Self {
            inner: AtrState::new(period),
        })
    }

    /// Add a new bar (high, low, close) and return ATR (NaN during warmup).
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> f64 {
        self.inner.update(high, low, close)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    pub fn period(&self) -> usize {
        self.inner.period
    }

    fn __repr__(&self) -> String {
        format!("StreamingATR(period={})", self.inner.period)
    }
}

// ---------------------------------------------------------------------------
// StreamingBBands
// ---------------------------------------------------------------------------

/// Bollinger Bands — streaming variant.
///
/// Returns (upper, middle, lower) as a Python tuple.
/// NaN tuple during the warmup window.
#[pyclass(module = "ferro_ta._ferro_ta")]
pub struct StreamingBBands {
    period: usize,
    nbdevup: f64,
    nbdevdn: f64,
    buf: VecDeque<f64>,
}

#[pymethods]
impl StreamingBBands {
    #[new]
    #[pyo3(signature = (period = 20, nbdevup = 2.0, nbdevdn = 2.0))]
    pub fn new(period: usize, nbdevup: f64, nbdevdn: f64) -> PyResult<Self> {
        if period < 2 {
            return Err(PyValueError::new_err("period must be >= 2"));
        }
        Ok(Self {
            period,
            nbdevup,
            nbdevdn,
            buf: VecDeque::with_capacity(period + 1),
        })
    }

    /// Add a new bar; return (upper, middle, lower). NaN tuple during warmup.
    pub fn update(&mut self, value: f64) -> (f64, f64, f64) {
        if self.buf.len() == self.period {
            self.buf.pop_front();
        }
        self.buf.push_back(value);
        if self.buf.len() < self.period {
            return (f64::NAN, f64::NAN, f64::NAN);
        }
        let n = self.period as f64;
        // Single-pass: compute sum and sum-of-squares simultaneously
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for &x in &self.buf {
            sum += x;
            sum_sq += x * x;
        }
        let mean = sum / n;
        // Sample variance: (Σx² - n·mean²) / (n-1)
        let variance = (sum_sq - n * mean * mean).max(0.0) / (n - 1.0);
        let std = variance.sqrt();
        (mean + self.nbdevup * std, mean, mean - self.nbdevdn * std)
    }

    pub fn reset(&mut self) {
        self.buf.clear();
    }

    #[getter]
    pub fn period(&self) -> usize {
        self.period
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamingBBands(period={}, nbdevup={}, nbdevdn={})",
            self.period, self.nbdevup, self.nbdevdn
        )
    }
}

// ---------------------------------------------------------------------------
// StreamingMACD
// ---------------------------------------------------------------------------

/// MACD — fast EMA, slow EMA, signal EMA.
///
/// Returns (macd_line, signal_line, histogram) as a Python tuple.
/// NaN values during warmup.
#[pyclass(module = "ferro_ta._ferro_ta")]
pub struct StreamingMACD {
    fast: EmaState,
    slow: EmaState,
    signal: EmaState,
}

#[pymethods]
impl StreamingMACD {
    #[new]
    #[pyo3(signature = (fastperiod = 12, slowperiod = 26, signalperiod = 9))]
    pub fn new(fastperiod: usize, slowperiod: usize, signalperiod: usize) -> PyResult<Self> {
        if fastperiod >= slowperiod {
            return Err(PyValueError::new_err("fastperiod must be < slowperiod"));
        }
        if fastperiod < 1 || signalperiod < 1 {
            return Err(PyValueError::new_err("periods must be >= 1"));
        }
        Ok(Self {
            fast: EmaState::new(fastperiod),
            slow: EmaState::new(slowperiod),
            signal: EmaState::new(signalperiod),
        })
    }

    /// Add a new close; return (macd_line, signal_line, histogram).
    pub fn update(&mut self, value: f64) -> (f64, f64, f64) {
        let fast_val = self.fast.update(value);
        let slow_val = self.slow.update(value);

        if slow_val.is_nan() {
            return (f64::NAN, f64::NAN, f64::NAN);
        }

        let macd = fast_val - slow_val;
        let signal = self.signal.update(macd);
        if signal.is_nan() {
            return (macd, f64::NAN, f64::NAN);
        }
        (macd, signal, macd - signal)
    }

    pub fn reset(&mut self) {
        self.fast.reset();
        self.slow.reset();
        self.signal.reset();
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamingMACD(fastperiod={}, slowperiod={}, signalperiod={})",
            self.fast.period, self.slow.period, self.signal.period
        )
    }
}

// ---------------------------------------------------------------------------
// StreamingStoch
// ---------------------------------------------------------------------------

/// Slow Stochastic (SMA-smoothed).
///
/// Returns (slowk, slowd) as a Python tuple.
/// NaN tuple during warmup.
#[pyclass(module = "ferro_ta._ferro_ta")]
pub struct StreamingStoch {
    fastk_period: usize,
    slowk_period: usize,
    slowd_period: usize,
    high_buf: VecDeque<f64>,
    low_buf: VecDeque<f64>,
    close_buf: VecDeque<f64>,
    fastk_buf: VecDeque<f64>,
    slowk_buf: VecDeque<f64>,
}

#[pymethods]
impl StreamingStoch {
    #[new]
    #[pyo3(signature = (fastk_period = 5, slowk_period = 3, slowd_period = 3))]
    pub fn new(fastk_period: usize, slowk_period: usize, slowd_period: usize) -> PyResult<Self> {
        if fastk_period < 1 || slowk_period < 1 || slowd_period < 1 {
            return Err(PyValueError::new_err("all periods must be >= 1"));
        }
        Ok(Self {
            fastk_period,
            slowk_period,
            slowd_period,
            high_buf: VecDeque::with_capacity(fastk_period + 1),
            low_buf: VecDeque::with_capacity(fastk_period + 1),
            close_buf: VecDeque::with_capacity(fastk_period + 1),
            fastk_buf: VecDeque::with_capacity(slowk_period + 1),
            slowk_buf: VecDeque::with_capacity(slowd_period + 1),
        })
    }

    /// Add a new bar (high, low, close); return (slowk, slowd).
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> (f64, f64) {
        if self.high_buf.len() == self.fastk_period {
            self.high_buf.pop_front();
            self.low_buf.pop_front();
            self.close_buf.pop_front();
        }
        self.high_buf.push_back(high);
        self.low_buf.push_back(low);
        self.close_buf.push_back(close);

        if self.high_buf.len() < self.fastk_period {
            return (f64::NAN, f64::NAN);
        }

        let max_h = self
            .high_buf
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_l = self.low_buf.iter().cloned().fold(f64::INFINITY, f64::min);

        let fastk = if max_h != min_l {
            100.0 * (close - min_l) / (max_h - min_l)
        } else {
            0.0
        };

        if self.fastk_buf.len() == self.slowk_period {
            self.fastk_buf.pop_front();
        }
        self.fastk_buf.push_back(fastk);
        if self.fastk_buf.len() < self.slowk_period {
            return (f64::NAN, f64::NAN);
        }

        let slowk = self.fastk_buf.iter().sum::<f64>() / self.slowk_period as f64;

        if self.slowk_buf.len() == self.slowd_period {
            self.slowk_buf.pop_front();
        }
        self.slowk_buf.push_back(slowk);
        if self.slowk_buf.len() < self.slowd_period {
            return (slowk, f64::NAN);
        }

        let slowd = self.slowk_buf.iter().sum::<f64>() / self.slowd_period as f64;
        (slowk, slowd)
    }

    pub fn reset(&mut self) {
        self.high_buf.clear();
        self.low_buf.clear();
        self.close_buf.clear();
        self.fastk_buf.clear();
        self.slowk_buf.clear();
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamingStoch(fastk_period={}, slowk_period={}, slowd_period={})",
            self.fastk_period, self.slowk_period, self.slowd_period
        )
    }
}

// ---------------------------------------------------------------------------
// StreamingVWAP
// ---------------------------------------------------------------------------

/// Cumulative Volume Weighted Average Price.
///
/// Resets automatically when `reset()` is called (e.g. at session open).
/// Accepts (high, low, close, volume) per bar.
#[pyclass(module = "ferro_ta._ferro_ta")]
#[derive(Default)]
pub struct StreamingVWAP {
    cum_tpv: f64,
    cum_vol: f64,
}

#[pymethods]
impl StreamingVWAP {
    #[new]
    pub fn new() -> Self {
        Self {
            cum_tpv: 0.0,
            cum_vol: 0.0,
        }
    }

    /// Add a new bar (high, low, close, volume) and return cumulative VWAP.
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        let tp = (high + low + close) / 3.0;
        self.cum_tpv += tp * volume;
        self.cum_vol += volume;
        if self.cum_vol == 0.0 {
            f64::NAN
        } else {
            self.cum_tpv / self.cum_vol
        }
    }

    /// Reset for a new session.
    pub fn reset(&mut self) {
        self.cum_tpv = 0.0;
        self.cum_vol = 0.0;
    }

    fn __repr__(&self) -> String {
        "StreamingVWAP()".to_string()
    }
}

// ---------------------------------------------------------------------------
// StreamingSupertrend
// ---------------------------------------------------------------------------

/// ATR-based Supertrend — streaming variant.
///
/// Accepts (high, low, close) per bar.
/// Returns (supertrend_line, direction) as a Python tuple.
/// direction: 1 = uptrend, -1 = downtrend, 0 = warmup.
#[pyclass(module = "ferro_ta._ferro_ta")]
pub struct StreamingSupertrend {
    period: usize,
    multiplier: f64,
    atr: AtrState,
    upper_band: f64,
    lower_band: f64,
    has_bands: bool,
    direction: i8,
    prev_close: f64,
    has_prev: bool,
}

#[pymethods]
impl StreamingSupertrend {
    #[new]
    #[pyo3(signature = (period = 7, multiplier = 3.0))]
    pub fn new(period: usize, multiplier: f64) -> PyResult<Self> {
        if period < 1 {
            return Err(PyValueError::new_err("period must be >= 1"));
        }
        Ok(Self {
            period,
            multiplier,
            atr: AtrState::new(period),
            upper_band: 0.0,
            lower_band: 0.0,
            has_bands: false,
            direction: 0,
            prev_close: 0.0,
            has_prev: false,
        })
    }

    /// Add a new bar (high, low, close); return (supertrend_line, direction).
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> (f64, i8) {
        let atr = self.atr.update(high, low, close);
        if atr.is_nan() {
            self.prev_close = close;
            self.has_prev = true;
            return (f64::NAN, 0);
        }

        let hl2 = (high + low) / 2.0;
        let upper_basic = hl2 + self.multiplier * atr;
        let lower_basic = hl2 - self.multiplier * atr;

        if !self.has_bands {
            self.upper_band = upper_basic;
            self.lower_band = lower_basic;
            self.has_bands = true;
            self.direction = -1;
            self.prev_close = close;
            self.has_prev = true;
            return (self.upper_band, self.direction);
        }

        let prev_close = self.prev_close;

        let new_lower = if lower_basic > self.lower_band || prev_close < self.lower_band {
            lower_basic
        } else {
            self.lower_band
        };
        let new_upper = if upper_basic < self.upper_band || prev_close > self.upper_band {
            upper_basic
        } else {
            self.upper_band
        };

        self.lower_band = new_lower;
        self.upper_band = new_upper;

        self.direction = if self.direction == -1 {
            if close > new_upper {
                1
            } else {
                -1
            }
        } else if close < new_lower {
            -1
        } else {
            1
        };

        self.prev_close = close;
        let line = if self.direction == 1 {
            new_lower
        } else {
            new_upper
        };
        (line, self.direction)
    }

    pub fn reset(&mut self) {
        self.atr.reset();
        self.upper_band = 0.0;
        self.lower_band = 0.0;
        self.has_bands = false;
        self.direction = 0;
        self.prev_close = 0.0;
        self.has_prev = false;
    }

    #[getter]
    pub fn period(&self) -> usize {
        self.period
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamingSupertrend(period={}, multiplier={})",
            self.period, self.multiplier
        )
    }
}

// ---------------------------------------------------------------------------
// register
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StreamingSMA>()?;
    m.add_class::<StreamingEMA>()?;
    m.add_class::<StreamingRSI>()?;
    m.add_class::<StreamingATR>()?;
    m.add_class::<StreamingBBands>()?;
    m.add_class::<StreamingMACD>()?;
    m.add_class::<StreamingStoch>()?;
    m.add_class::<StreamingVWAP>()?;
    m.add_class::<StreamingSupertrend>()?;
    Ok(())
}
