use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Bollinger Bands. Returns (upper, middle, lower).
///
/// The middle band is a moving average of type `matype` (default `0` = SMA,
/// TA-Lib's default and this wrapper's historical behaviour); the outer bands
/// are offset by ± `nbdev` × the window's population standard deviation.
///
/// `0`–`6` and `8` match TA-Lib's numbering; `7` is T3 here where TA-Lib's `7`
/// is MAMA, and MAMA is not reachable through any `matype` (use `ferro_ta.MAMA`).
///
/// Like TA-Lib, the deviation is always measured about the window **SMA**, not
/// about the selected MA (`ta_BBANDS.c` passes `optInMAType` to `TA_MA` for the
/// centre but never to its `TA_STDDEV` call). For `matype != 0` the centre and
/// the deviation reference are therefore different series: the bands are not an
/// `nbdev`-sigma envelope of the series they are centred on, and if the selected
/// MA drifts further from the window SMA than `nbdev * sigma`, the envelope stops
/// bracketing the SMA altogether. Warm-up follows the selected MA's lookback and
/// is identical in all three vectors.
#[pyfunction]
#[pyo3(signature = (close, timeperiod = 5, nbdevup = 2.0, nbdevdn = 2.0, matype = 0))]
#[allow(clippy::type_complexity)]
pub fn bbands<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    timeperiod: usize,
    nbdevup: f64,
    nbdevdn: f64,
    matype: u8,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    validation::validate_timeperiod(timeperiod, "timeperiod", 1)?;
    if matype > 8 {
        return Err(PyValueError::new_err(
            "matype must be 0–8 (SMA/EMA/WMA/DEMA/TEMA/TRIMA/KAMA/T3; 8 aliases T3)",
        ));
    }
    let prices = close.as_slice()?;
    log::debug!(
        "BBANDS: timeperiod={timeperiod}, matype={matype}, n={}",
        prices.len()
    );
    let (upper, middle, lower) = py.allow_threads(|| {
        ferro_ta_core::overlap::bbands(prices, timeperiod, nbdevup, nbdevdn, matype)
    });
    Ok((
        upper.into_pyarray(py),
        middle.into_pyarray(py),
        lower.into_pyarray(py),
    ))
}
