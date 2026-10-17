//! Streaming / Incremental Indicators — bar-by-bar stateful classes.
//!
//! Thin PyO3 wrappers over `ferro_ta_core::streaming::Live<T>`, which adds
//! live-bar previews (`closed=False`), NaN-tick handling and introspection on
//! top of each core indicator.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ferro_ta_core::streaming as core;

mod bars;

// ---------------------------------------------------------------------------
// Helper: convert core StreamingError to PyValueError
// ---------------------------------------------------------------------------

pub(crate) fn to_py_err(e: core::StreamingError) -> PyErr {
    PyValueError::new_err(e.0)
}

// ---------------------------------------------------------------------------
// Class generator
// ---------------------------------------------------------------------------

/// Generates a `#[pyclass]` wrapping `core::Live<$core>`.
///
/// `#[pymethods]` cannot be split across blocks without the
/// `multiple-pymethods` feature, so the shared methods and the per-class
/// constructor / extras are spliced into one block here.
macro_rules! streaming_pyclass {
    (
        $(#[$doc:meta])*
        $name:ident wraps $core:ty,
        new($($arg:ident: $arg_ty:ty $(= $default:expr)?),*) => $ctor:expr,
        update($($input:ident),+) -> $out:ty,
        repr(|$ind:ident| $repr:expr),
        getters($($getter:ident),*) $(,)?
    ) => {
        $(#[$doc])*
        #[pyclass(module = "ferro_ta._ferro_ta")]
        #[derive(Clone)]
        pub struct $name {
            inner: core::Live<$core>,
        }

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = ($($arg $(= $default)?),*))]
            pub fn new($($arg: $arg_ty),*) -> PyResult<Self> {
                let indicator: $core = $ctor.map_err(to_py_err)?;
                Ok(Self {
                    inner: core::Live::new(indicator),
                })
            }

            /// Feed one tick.
            ///
            /// ``closed=True`` (default) commits the bar. ``closed=False``
            /// previews the still-forming bar from committed state without
            /// changing it, so repeated ticks of one bar replace each other.
            #[pyo3(signature = ($($input),+, *, closed = true))]
            pub fn update(&mut self, $($input: f64),+, closed: bool) -> $out {
                self.inner.update(($($input),+), closed)
            }

            /// Reset state to initial condition.
            pub fn reset(&mut self) {
                self.inner.reset();
            }

            /// Number of committed (closed) bars.
            #[getter]
            pub fn count(&self) -> usize {
                self.inner.count()
            }

            /// True once the last committed bar produced a non-NaN value.
            #[getter]
            pub fn is_ready(&self) -> bool {
                self.inner.is_ready()
            }

            /// Versioned JSON snapshot of the full state (see ``from_json``).
            pub fn to_json(&self) -> PyResult<String> {
                self.inner.save().map_err(to_py_err)
            }

            /// Restore a snapshot made by ``to_json`` of this same class.
            #[staticmethod]
            pub fn from_json(json: &str) -> PyResult<Self> {
                Ok(Self {
                    inner: core::Live::load(json).map_err(to_py_err)?,
                })
            }

            fn __reduce__<'py>(
                slf: &Bound<'py, Self>,
            ) -> PyResult<(Bound<'py, PyAny>, (String,))> {
                let snapshot = slf.borrow().inner.save().map_err(to_py_err)?;
                Ok((slf.get_type().getattr("from_json")?, (snapshot,)))
            }

            fn __copy__(&self) -> Self {
                self.clone()
            }

            fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
                self.clone()
            }

            fn __repr__(&self) -> String {
                let $ind = self.inner.inner();
                $repr
            }

            $(
                #[getter]
                pub fn $getter(&self) -> usize {
                    self.inner.inner().$getter()
                }
            )*
        }
    };
}

// ---------------------------------------------------------------------------
// Classes
// ---------------------------------------------------------------------------

streaming_pyclass! {
    /// Simple Moving Average — O(1) per update via running sum.
    ///
    /// Returns NaN during the first `period - 1` bars.
    StreamingSMA wraps core::StreamingSMA,
    new(period: usize) => core::StreamingSMA::new(period),
    update(value) -> f64,
    repr(|ind| format!("StreamingSMA(period={})", ind.period())),
    getters(period),
}

streaming_pyclass! {
    /// Exponential Moving Average with SMA seeding.
    StreamingEMA wraps core::StreamingEMA,
    new(period: usize) => core::StreamingEMA::new(period),
    update(value) -> f64,
    repr(|ind| format!("StreamingEMA(period={})", ind.period())),
    getters(period),
}

streaming_pyclass! {
    /// Relative Strength Index with TA-Lib–compatible Wilder seeding.
    StreamingRSI wraps core::StreamingRSI,
    new(period: usize = 14) => core::StreamingRSI::new(period),
    update(value) -> f64,
    repr(|ind| format!("StreamingRSI(period={})", ind.period())),
    getters(period),
}

streaming_pyclass! {
    /// Average True Range with TA-Lib–compatible Wilder seeding.
    StreamingATR wraps core::StreamingATR,
    new(period: usize = 14) => core::StreamingATR::new(period),
    update(high, low, close) -> f64,
    repr(|ind| format!("StreamingATR(period={})", ind.period())),
    getters(period),
}

streaming_pyclass! {
    /// Bollinger Bands — returns (upper, middle, lower).
    StreamingBBands wraps core::StreamingBBands,
    new(period: usize = 20, nbdevup: f64 = 2.0, nbdevdn: f64 = 2.0) =>
        core::StreamingBBands::new(period, nbdevup, nbdevdn),
    update(value) -> (f64, f64, f64),
    repr(|ind| format!("StreamingBBands(period={})", ind.period())),
    getters(period),
}

streaming_pyclass! {
    /// MACD — returns (macd_line, signal_line, histogram).
    StreamingMACD wraps core::StreamingMACD,
    new(fastperiod: usize = 12, slowperiod: usize = 26, signalperiod: usize = 9) =>
        core::StreamingMACD::new(fastperiod, slowperiod, signalperiod),
    update(value) -> (f64, f64, f64),
    repr(|ind| format!(
        "StreamingMACD(fastperiod={}, slowperiod={}, signalperiod={})",
        ind.fast_period(),
        ind.slow_period(),
        ind.signal_period()
    )),
    getters(),
}

streaming_pyclass! {
    /// Slow Stochastic (SMA-smoothed) — returns (slowk, slowd).
    StreamingStoch wraps core::StreamingStoch,
    new(fastk_period: usize = 5, slowk_period: usize = 3, slowd_period: usize = 3) =>
        core::StreamingStoch::new(fastk_period, slowk_period, slowd_period),
    update(high, low, close) -> (f64, f64),
    repr(|ind| format!("StreamingStoch(fastk_period={})", ind.period())),
    getters(),
}

streaming_pyclass! {
    /// Cumulative Volume Weighted Average Price. Call `reset()` at session open.
    StreamingVWAP wraps core::StreamingVWAP,
    new() => Ok::<_, core::StreamingError>(core::StreamingVWAP::new()),
    update(high, low, close, volume) -> f64,
    repr(|_ind| "StreamingVWAP()".to_string()),
    getters(),
}

streaming_pyclass! {
    /// ATR-based Supertrend — returns (supertrend_line, direction).
    StreamingSupertrend wraps core::StreamingSupertrend,
    new(period: usize = 7, multiplier: f64 = 3.0) =>
        core::StreamingSupertrend::new(period, multiplier),
    update(high, low, close) -> (f64, i8),
    repr(|ind| format!("StreamingSupertrend(period={})", ind.period())),
    getters(period),
}

streaming_pyclass! {
    /// Cross of `a` over `b` with a dead band — emits 1 (up), -1 (down), 0.
    ///
    /// Inside ``±dead_band`` the side is unchanged, so noise around the level
    /// fires at most once per real crossing.
    StreamingCross wraps core::StreamingCross,
    new(dead_band: f64 = 0.0) => core::StreamingCross::new(dead_band),
    update(a, b) -> i8,
    repr(|ind| format!("StreamingCross(dead_band={})", ind.dead_band())),
    getters(),
}

streaming_pyclass! {
    /// "N bars in a row" — emits 1 while the run of non-zero inputs is >= n.
    StreamingStreak wraps core::StreamingStreak,
    new(n: usize) => core::StreamingStreak::new(n),
    update(condition) -> i8,
    repr(|ind| format!("StreamingStreak(n={})", ind.n())),
    getters(n),
}

streaming_pyclass! {
    /// Passes the first non-zero signal's sign, then mutes the next `bars` bars.
    StreamingCooldown wraps core::StreamingCooldown,
    new(bars: usize) => core::StreamingCooldown::new(bars),
    update(signal) -> i8,
    repr(|ind| format!("StreamingCooldown(bars={})", ind.bars())),
    getters(bars),
}

streaming_pyclass! {
    /// Live implied volatility and greeks for one option contract.
    ///
    /// ``update(option_price, underlying, time_to_expiry)`` returns
    /// ``(iv, delta, gamma, vega, theta, rho)``. The IV solver is seeded with
    /// the previous tick's IV; a price no volatility can reproduce yields
    /// all-NaN and keeps the seed.
    StreamingOptionGreeks wraps core::StreamingOptionGreeks,
    new(
        strike: f64,
        option_type: &str = "call",
        model: &str = "bsm",
        rate: f64 = 0.0,
        carry: f64 = 0.0
    ) => {
        let kind = crate::options::parse_option_kind(option_type)?;
        let model = crate::options::parse_pricing_model(model)?;
        core::StreamingOptionGreeks::new(model, kind, strike, rate, carry)
    },
    update(option_price, underlying, time_to_expiry) -> (f64, f64, f64, f64, f64, f64),
    repr(|ind| format!("StreamingOptionGreeks(strike={})", ind.strike())),
    getters(),
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
    m.add_class::<StreamingCross>()?;
    m.add_class::<StreamingStreak>()?;
    m.add_class::<StreamingCooldown>()?;
    m.add_class::<StreamingOptionGreeks>()?;
    m.add_class::<bars::BarBuilder>()?;
    m.add_class::<bars::Bar>()?;
    Ok(())
}
