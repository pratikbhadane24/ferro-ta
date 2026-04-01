//! Streaming / Incremental Indicators — bar-by-bar stateful classes.
//!
//! Thin PyO3 wrappers that delegate to `ferro_ta_core::streaming`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ferro_ta_core::streaming as core;

// ---------------------------------------------------------------------------
// Helper: convert core StreamingError to PyValueError
// ---------------------------------------------------------------------------

fn to_py_err(e: core::StreamingError) -> PyErr {
    PyValueError::new_err(e.0)
}

// ---------------------------------------------------------------------------
// StreamingSMA
// ---------------------------------------------------------------------------

/// Simple Moving Average — O(1) per update via running sum.
///
/// Returns NaN during the first `period - 1` bars.
#[pyclass(module = "ferro_ta._ferro_ta")]
pub struct StreamingSMA {
    inner: core::StreamingSMA,
}

#[pymethods]
impl StreamingSMA {
    #[new]
    #[pyo3(signature = (period))]
    pub fn new(period: usize) -> PyResult<Self> {
        Ok(Self {
            inner: core::StreamingSMA::new(period).map_err(to_py_err)?,
        })
    }

    /// Add a new bar and return the current SMA (NaN during warmup).
    pub fn update(&mut self, value: f64) -> f64 {
        self.inner.update(value)
    }

    /// Reset state to initial condition.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    pub fn period(&self) -> usize {
        self.inner.period()
    }

    fn __repr__(&self) -> String {
        format!("StreamingSMA(period={})", self.inner.period())
    }
}

// ---------------------------------------------------------------------------
// StreamingEMA
// ---------------------------------------------------------------------------

/// Exponential Moving Average with SMA seeding.
#[pyclass(module = "ferro_ta._ferro_ta")]
pub struct StreamingEMA {
    inner: core::StreamingEMA,
}

#[pymethods]
impl StreamingEMA {
    #[new]
    #[pyo3(signature = (period))]
    pub fn new(period: usize) -> PyResult<Self> {
        Ok(Self {
            inner: core::StreamingEMA::new(period).map_err(to_py_err)?,
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
        self.inner.period()
    }

    fn __repr__(&self) -> String {
        format!("StreamingEMA(period={})", self.inner.period())
    }
}

// ---------------------------------------------------------------------------
// StreamingRSI
// ---------------------------------------------------------------------------

/// Relative Strength Index with TA-Lib–compatible Wilder seeding.
#[pyclass(module = "ferro_ta._ferro_ta")]
pub struct StreamingRSI {
    inner: core::StreamingRSI,
}

#[pymethods]
impl StreamingRSI {
    #[new]
    #[pyo3(signature = (period = 14))]
    pub fn new(period: usize) -> PyResult<Self> {
        Ok(Self {
            inner: core::StreamingRSI::new(period).map_err(to_py_err)?,
        })
    }

    /// Add a new close and return RSI in [0, 100] (NaN during warmup).
    pub fn update(&mut self, value: f64) -> f64 {
        self.inner.update(value)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    pub fn period(&self) -> usize {
        self.inner.period()
    }

    fn __repr__(&self) -> String {
        format!("StreamingRSI(period={})", self.inner.period())
    }
}

// ---------------------------------------------------------------------------
// StreamingATR
// ---------------------------------------------------------------------------

/// Average True Range with TA-Lib–compatible Wilder seeding.
#[pyclass(module = "ferro_ta._ferro_ta")]
pub struct StreamingATR {
    inner: core::StreamingATR,
}

#[pymethods]
impl StreamingATR {
    #[new]
    #[pyo3(signature = (period = 14))]
    pub fn new(period: usize) -> PyResult<Self> {
        Ok(Self {
            inner: core::StreamingATR::new(period).map_err(to_py_err)?,
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
        self.inner.period()
    }

    fn __repr__(&self) -> String {
        format!("StreamingATR(period={})", self.inner.period())
    }
}

// ---------------------------------------------------------------------------
// StreamingBBands
// ---------------------------------------------------------------------------

/// Bollinger Bands — streaming variant using Welford's online algorithm.
#[pyclass(module = "ferro_ta._ferro_ta")]
pub struct StreamingBBands {
    inner: core::StreamingBBands,
}

#[pymethods]
impl StreamingBBands {
    #[new]
    #[pyo3(signature = (period = 20, nbdevup = 2.0, nbdevdn = 2.0))]
    pub fn new(period: usize, nbdevup: f64, nbdevdn: f64) -> PyResult<Self> {
        Ok(Self {
            inner: core::StreamingBBands::new(period, nbdevup, nbdevdn).map_err(to_py_err)?,
        })
    }

    /// Add a new bar; return (upper, middle, lower). NaN tuple during warmup.
    pub fn update(&mut self, value: f64) -> (f64, f64, f64) {
        self.inner.update(value)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    pub fn period(&self) -> usize {
        self.inner.period()
    }

    fn __repr__(&self) -> String {
        format!("StreamingBBands(period={})", self.inner.period())
    }
}

// ---------------------------------------------------------------------------
// StreamingMACD
// ---------------------------------------------------------------------------

/// MACD — fast EMA, slow EMA, signal EMA.
#[pyclass(module = "ferro_ta._ferro_ta")]
pub struct StreamingMACD {
    inner: core::StreamingMACD,
}

#[pymethods]
impl StreamingMACD {
    #[new]
    #[pyo3(signature = (fastperiod = 12, slowperiod = 26, signalperiod = 9))]
    pub fn new(fastperiod: usize, slowperiod: usize, signalperiod: usize) -> PyResult<Self> {
        Ok(Self {
            inner: core::StreamingMACD::new(fastperiod, slowperiod, signalperiod)
                .map_err(to_py_err)?,
        })
    }

    /// Add a new close; return (macd_line, signal_line, histogram).
    pub fn update(&mut self, value: f64) -> (f64, f64, f64) {
        self.inner.update(value)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamingMACD(fastperiod={}, slowperiod={}, signalperiod={})",
            self.inner.fast_period(),
            self.inner.slow_period(),
            self.inner.signal_period()
        )
    }
}

// ---------------------------------------------------------------------------
// StreamingStoch
// ---------------------------------------------------------------------------

/// Slow Stochastic (SMA-smoothed).
#[pyclass(module = "ferro_ta._ferro_ta")]
pub struct StreamingStoch {
    inner: core::StreamingStoch,
}

#[pymethods]
impl StreamingStoch {
    #[new]
    #[pyo3(signature = (fastk_period = 5, slowk_period = 3, slowd_period = 3))]
    pub fn new(fastk_period: usize, slowk_period: usize, slowd_period: usize) -> PyResult<Self> {
        Ok(Self {
            inner: core::StreamingStoch::new(fastk_period, slowk_period, slowd_period)
                .map_err(to_py_err)?,
        })
    }

    /// Add a new bar (high, low, close); return (slowk, slowd).
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> (f64, f64) {
        self.inner.update(high, low, close)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        format!("StreamingStoch(fastk_period={})", self.inner.period())
    }
}

// ---------------------------------------------------------------------------
// StreamingVWAP
// ---------------------------------------------------------------------------

/// Cumulative Volume Weighted Average Price.
#[pyclass(module = "ferro_ta._ferro_ta")]
pub struct StreamingVWAP {
    inner: core::StreamingVWAP,
}

#[pymethods]
impl StreamingVWAP {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: core::StreamingVWAP::new(),
        }
    }

    /// Add a new bar (high, low, close, volume) and return cumulative VWAP.
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        self.inner.update(high, low, close, volume)
    }

    /// Reset for a new session.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        "StreamingVWAP()".to_string()
    }
}

// ---------------------------------------------------------------------------
// StreamingSupertrend
// ---------------------------------------------------------------------------

/// ATR-based Supertrend — streaming variant.
#[pyclass(module = "ferro_ta._ferro_ta")]
pub struct StreamingSupertrend {
    inner: core::StreamingSupertrend,
}

#[pymethods]
impl StreamingSupertrend {
    #[new]
    #[pyo3(signature = (period = 7, multiplier = 3.0))]
    pub fn new(period: usize, multiplier: f64) -> PyResult<Self> {
        Ok(Self {
            inner: core::StreamingSupertrend::new(period, multiplier).map_err(to_py_err)?,
        })
    }

    /// Add a new bar (high, low, close); return (supertrend_line, direction).
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> (f64, i8) {
        self.inner.update(high, low, close)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    pub fn period(&self) -> usize {
        self.inner.period()
    }

    fn __repr__(&self) -> String {
        format!("StreamingSupertrend(period={})", self.inner.period())
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
