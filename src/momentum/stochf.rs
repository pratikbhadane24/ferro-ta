use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ta::indicators::{ExponentialMovingAverage, FastStochastic};
use ta::{DataItem, Next};

/// Fast Stochastic. Returns (fastk, fastd). %K from high-low range; %D is EMA of %K.
#[pyfunction]
#[pyo3(signature = (high, low, close, fastk_period = 5, fastd_period = 3))]
#[allow(clippy::type_complexity)]
pub fn stochf<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    fastk_period: usize,
    fastd_period: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validation::validate_timeperiod(fastk_period, "fastk_period", 1)?;
    validation::validate_timeperiod(fastd_period, "fastd_period", 1)?;
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let closes = close.as_slice()?;
    let n = highs.len();
    validation::validate_equal_length(&[
        (n, "high"),
        (lows.len(), "low"),
        (closes.len(), "close"),
    ])?;

    let mut fast_stoch =
        FastStochastic::new(fastk_period).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut d_ema = ExponentialMovingAverage::new(fastd_period)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let warmup_k = fastk_period - 1;
    let warmup_d = warmup_k + fastd_period - 1;

    let mut fastk = vec![f64::NAN; n];
    let mut fastd = vec![f64::NAN; n];

    for (i, ((&h, &l), &c)) in highs.iter().zip(lows.iter()).zip(closes.iter()).enumerate() {
        let bar = DataItem::builder()
            .high(h)
            .low(l)
            .close(c)
            .open(c)
            .volume(0.0)
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let k = fast_stoch.next(&bar);
        if i >= warmup_k {
            fastk[i] = k;
            let d = d_ema.next(k);
            if i >= warmup_d {
                fastd[i] = d;
            }
        }
    }
    Ok((fastk.into_pyarray(py), fastd.into_pyarray(py)))
}
