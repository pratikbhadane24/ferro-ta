//! PyO3 wrapper for `ferro_ta_core::streaming::BarBuilder`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ferro_ta_core::streaming as core;

use super::to_py_err;

/// One OHLCV bar covering ``[start_ms, end_ms)`` (UTC epoch milliseconds).
#[pyclass(module = "ferro_ta._ferro_ta", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub struct Bar {
    start_ms: i64,
    end_ms: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    ticks: u32,
}

impl From<core::Bar> for Bar {
    fn from(b: core::Bar) -> Self {
        Self {
            start_ms: b.start_ms,
            end_ms: b.end_ms,
            open: b.open,
            high: b.high,
            low: b.low,
            close: b.close,
            volume: b.volume,
            ticks: b.ticks,
        }
    }
}

#[pymethods]
impl Bar {
    fn __repr__(&self) -> String {
        format!(
            "Bar(start_ms={}, end_ms={}, open={}, high={}, low={}, close={}, volume={}, ticks={})",
            self.start_ms,
            self.end_ms,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            self.ticks
        )
    }
}

/// Parse ``"HH:MM"`` or ``"HH:MM:SS"`` (``"24:00"`` allowed) to seconds.
fn parse_clock(text: &str, name: &str) -> PyResult<i64> {
    let bad = || {
        PyValueError::new_err(format!(
            "{name} must be 'HH:MM' or 'HH:MM:SS', got {text:?}"
        ))
    };
    let parts: Vec<&str> = text.split(':').collect();
    if !(2..=3).contains(&parts.len()) {
        return Err(bad());
    }
    let nums: Vec<i64> = parts
        .iter()
        .map(|p| p.parse::<i64>().map_err(|_| bad()))
        .collect::<PyResult<_>>()?;
    let (h, m, s) = (nums[0], nums[1], nums.get(2).copied().unwrap_or(0));
    let in_range = (0..=24).contains(&h) && (0..60).contains(&m) && (0..60).contains(&s);
    if !in_range || (h == 24 && (m, s) != (0, 0)) {
        return Err(bad());
    }
    Ok(h * 3_600 + m * 60 + s)
}

/// Session-aligned tick → bar builder.
///
/// Bars sit on a session-local grid (``session_open + k * interval``; the
/// last bar is cut at ``session_close``), so 5-minute NSE bars start at
/// 09:15, 09:20, … Gaps without ticks produce no bars. Late, duplicate,
/// out-of-session and invalid ticks are rejected with a status string.
#[pyclass(module = "ferro_ta._ferro_ta")]
#[derive(Clone)]
pub struct BarBuilder {
    inner: core::BarBuilder,
}

#[pymethods]
impl BarBuilder {
    #[new]
    #[pyo3(signature = (interval_secs, *, session_open = "09:15", session_close = "15:30", utc_offset_minutes = 330))]
    fn new(
        interval_secs: i64,
        session_open: &str,
        session_close: &str,
        utc_offset_minutes: i64,
    ) -> PyResult<Self> {
        let spec = core::BarSpec {
            interval_secs,
            session_open_secs: parse_clock(session_open, "session_open")?,
            session_close_secs: parse_clock(session_close, "session_close")?,
            utc_offset_secs: utc_offset_minutes
                .checked_mul(60)
                .ok_or_else(|| PyValueError::new_err("utc_offset_minutes out of range"))?,
        };
        Ok(Self {
            inner: core::BarBuilder::new(spec).map_err(to_py_err)?,
        })
    }

    /// Feed one tick. Returns ``(status, closed_bar_or_None)`` where status
    /// is ``"accepted"``, ``"late"``, ``"duplicate"``, ``"out_of_session"``
    /// or ``"invalid"``.
    #[pyo3(signature = (ts_ms, price, qty = 0.0))]
    fn on_tick(&mut self, ts_ms: i64, price: f64, qty: f64) -> (&'static str, Option<Bar>) {
        let (status, closed) = self.inner.on_tick(ts_ms, price, qty);
        (status.as_str(), closed.map(Bar::from))
    }

    /// Close the forming bar once ``now_ms`` reaches its end (no tick needed).
    fn on_time(&mut self, now_ms: i64) -> Option<Bar> {
        self.inner.on_time(now_ms).map(Bar::from)
    }

    /// The bar currently being built, if any.
    #[getter]
    fn forming(&self) -> Option<Bar> {
        self.inner.forming().map(Bar::from)
    }

    #[getter]
    fn interval_secs(&self) -> i64 {
        self.inner.spec().interval_secs
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Versioned JSON snapshot, including the forming bar (see ``from_json``).
    fn to_json(&self) -> PyResult<String> {
        self.inner.save().map_err(to_py_err)
    }

    /// Restore a snapshot made by ``to_json``.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        Ok(Self {
            inner: core::BarBuilder::load(json).map_err(to_py_err)?,
        })
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
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
        format!(
            "BarBuilder(interval_secs={})",
            self.inner.spec().interval_secs
        )
    }
}
