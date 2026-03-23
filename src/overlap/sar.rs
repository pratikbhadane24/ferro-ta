use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Parabolic SAR. Same shape as TA-Lib; reversal history may differ slightly.
#[pyfunction]
#[pyo3(signature = (high, low, acceleration = 0.02, maximum = 0.2))]
pub fn sar<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    acceleration: f64,
    maximum: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let highs = high.as_slice()?;
    let lows = low.as_slice()?;
    let n = highs.len();
    validation::validate_equal_length(&[(n, "high"), (lows.len(), "low")])?;
    if n < 2 {
        return Ok(vec![f64::NAN; n].into_pyarray(py));
    }

    let mut result = vec![f64::NAN; n];

    let mut is_rising = highs[1] >= highs[0];
    let mut af = acceleration;
    let mut ep: f64;
    let mut sar_val: f64;

    if is_rising {
        sar_val = lows[0];
        ep = highs[1];
    } else {
        sar_val = highs[0];
        ep = lows[1];
    }
    result[1] = sar_val;

    for i in 2..n {
        let prev_sar = sar_val;
        sar_val = prev_sar + af * (ep - prev_sar);

        if is_rising {
            sar_val = sar_val.min(lows[i - 1]).min(lows[i - 2]);
            if lows[i] < sar_val {
                is_rising = false;
                sar_val = ep;
                ep = lows[i];
                af = acceleration;
            } else if highs[i] > ep {
                ep = highs[i];
                af = (af + acceleration).min(maximum);
            }
        } else {
            sar_val = sar_val.max(highs[i - 1]).max(highs[i - 2]);
            if highs[i] > sar_val {
                is_rising = true;
                sar_val = ep;
                ep = highs[i];
                af = acceleration;
            } else if lows[i] < ep {
                ep = lows[i];
                af = (af + acceleration).min(maximum);
            }
        }
        result[i] = sar_val;
    }

    Ok(result.into_pyarray(py))
}
