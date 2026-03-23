use crate::validation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
pub fn weighted_continuous_contract<'py>(
    py: Python<'py>,
    front: PyReadonlyArray1<'py, f64>,
    next: PyReadonlyArray1<'py, f64>,
    next_weights: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let front = front.as_slice()?;
    let next = next.as_slice()?;
    let next_weights = next_weights.as_slice()?;
    validation::validate_equal_length(&[
        (front.len(), "front"),
        (next.len(), "next"),
        (next_weights.len(), "next_weights"),
    ])?;
    Ok(
        ferro_ta_core::futures::roll::weighted_continuous(front, next, next_weights)
            .into_pyarray(py),
    )
}

#[pyfunction]
pub fn back_adjusted_continuous_contract<'py>(
    py: Python<'py>,
    front: PyReadonlyArray1<'py, f64>,
    next: PyReadonlyArray1<'py, f64>,
    next_weights: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let front = front.as_slice()?;
    let next = next.as_slice()?;
    let next_weights = next_weights.as_slice()?;
    validation::validate_equal_length(&[
        (front.len(), "front"),
        (next.len(), "next"),
        (next_weights.len(), "next_weights"),
    ])?;
    Ok(
        ferro_ta_core::futures::roll::back_adjusted_continuous(front, next, next_weights)
            .into_pyarray(py),
    )
}

#[pyfunction]
pub fn ratio_adjusted_continuous_contract<'py>(
    py: Python<'py>,
    front: PyReadonlyArray1<'py, f64>,
    next: PyReadonlyArray1<'py, f64>,
    next_weights: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let front = front.as_slice()?;
    let next = next.as_slice()?;
    let next_weights = next_weights.as_slice()?;
    validation::validate_equal_length(&[
        (front.len(), "front"),
        (next.len(), "next"),
        (next_weights.len(), "next_weights"),
    ])?;
    Ok(
        ferro_ta_core::futures::roll::ratio_adjusted_continuous(front, next, next_weights)
            .into_pyarray(py),
    )
}

#[pyfunction]
pub fn roll_yield(front_price: f64, next_price: f64, time_to_expiry: f64) -> PyResult<f64> {
    Ok(ferro_ta_core::futures::roll::roll_yield(
        front_price,
        next_price,
        time_to_expiry,
    ))
}
