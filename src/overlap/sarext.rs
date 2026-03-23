use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Parabolic SAR Extended: SAR with configurable start value and long/short acceleration.
#[pyfunction]
#[pyo3(signature = (high, low, startvalue = 0.0, offsetonreverse = 0.0, accelerationinitlong = 0.02, accelerationlong = 0.02, accelerationmaxlong = 0.2, accelerationinitshort = 0.02, accelerationshort = 0.02, accelerationmaxshort = 0.2))]
#[allow(clippy::too_many_arguments)]
pub fn sarext<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    startvalue: f64,
    offsetonreverse: f64,
    accelerationinitlong: f64,
    accelerationlong: f64,
    accelerationmaxlong: f64,
    accelerationinitshort: f64,
    accelerationshort: f64,
    accelerationmaxshort: f64,
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

    let (mut af, af_step, af_max) = if is_rising {
        (accelerationinitlong, accelerationlong, accelerationmaxlong)
    } else {
        (
            accelerationinitshort,
            accelerationshort,
            accelerationmaxshort,
        )
    };

    let mut ep: f64;
    let mut sar_val: f64;

    if is_rising {
        sar_val = if startvalue != 0.0 {
            startvalue
        } else {
            lows[0]
        };
        ep = highs[1];
    } else {
        sar_val = if startvalue != 0.0 {
            -startvalue
        } else {
            highs[0]
        };
        ep = lows[1];
    }

    result[1] = sar_val;

    let mut af_step_cur = af_step;
    let mut af_max_cur = af_max;

    for i in 2..n {
        let prev_sar = sar_val;
        sar_val = prev_sar + af * (ep - prev_sar);

        if is_rising {
            sar_val = sar_val.min(lows[i - 1]).min(lows[i - 2]);
            if lows[i] < sar_val {
                is_rising = false;
                sar_val = ep + sar_val.abs() * offsetonreverse;
                ep = lows[i];
                af = accelerationinitshort;
                af_step_cur = accelerationshort;
                af_max_cur = accelerationmaxshort;
            } else if highs[i] > ep {
                ep = highs[i];
                af = (af + af_step_cur).min(af_max_cur);
            }
        } else {
            sar_val = sar_val.max(highs[i - 1]).max(highs[i - 2]);
            if highs[i] > sar_val {
                is_rising = true;
                sar_val = ep - sar_val.abs() * offsetonreverse;
                ep = highs[i];
                af = accelerationinitlong;
                af_step_cur = accelerationlong;
                af_max_cur = accelerationmaxlong;
            } else if lows[i] < ep {
                ep = lows[i];
                af = (af + af_step_cur).min(af_max_cur);
            }
        }
        result[i] = sar_val;
    }

    Ok(result.into_pyarray(py))
}
