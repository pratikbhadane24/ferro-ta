//! Validation helpers used by PyO3 functions. They raise PyValueError with
//! messages that match the Python check_* helpers; the Python wrapper layer
//! converts these to FerroTAValueError / FerroTAInputError via _normalize_rust_error.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Parse a period parameter from Python (signed int). Validates >= minimum,
/// returns Ok(usize). Use this for PyO3 signatures that take `i64` so negative
/// or out-of-range values are caught in Rust with a clear error.
pub fn parse_timeperiod(value: i64, name: &str, minimum: i64) -> PyResult<usize> {
    if value < minimum {
        return Err(PyValueError::new_err(format!(
            "{} must be >= {}, got {}",
            name, minimum, value
        )));
    }
    let u = usize::try_from(value).map_err(|_| {
        PyValueError::new_err(format!("{} must be >= {}, got {}", name, minimum, value))
    })?;
    Ok(u)
}

/// Validate that a period parameter is >= minimum. On failure raises PyValueError
/// (message format matches Python check_timeperiod; Python normalizes to FerroTAValueError).
pub fn validate_timeperiod(value: usize, name: &str, minimum: usize) -> PyResult<()> {
    if value < minimum {
        return Err(PyValueError::new_err(format!(
            "{} must be >= {}, got {}",
            name, minimum, value
        )));
    }
    Ok(())
}

/// Validate that all named lengths are equal. On failure raises PyValueError
/// (message includes "same length" so Python normalizes to FerroTAInputError).
pub fn validate_equal_length(lengths_and_names: &[(usize, &str)]) -> PyResult<()> {
    if lengths_and_names.len() < 2 {
        return Ok(());
    }
    let first = lengths_and_names[0].0;
    for (len, _name) in lengths_and_names.iter().skip(1) {
        if *len != first {
            let detail = lengths_and_names
                .iter()
                .map(|(l, n)| format!("{}={}", n, l))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(PyValueError::new_err(format!(
                "All input arrays must have the same length. Got: {}",
                detail
            )));
        }
    }
    Ok(())
}
