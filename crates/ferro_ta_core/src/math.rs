//! Math utilities.

use std::collections::VecDeque;

/// Compute the rolling sum over `timeperiod` bars.
///
/// Returns a `Vec<f64>` of length `n`. The first `timeperiod - 1` values
/// are `NaN`. Uses an incremental algorithm (add new, subtract old) for O(n).
///
/// # Arguments
/// * `real` - Input series.
/// * `timeperiod` - Rolling window size (must be >= 1).
pub fn sum(real: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = real.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 || n < timeperiod {
        return result;
    }
    let mut win: f64 = real[..timeperiod].iter().sum();
    result[timeperiod - 1] = win;
    for i in timeperiod..n {
        win += real[i] - real[i - timeperiod];
        result[i] = win;
    }
    result
}

/// Compute the rolling maximum over `timeperiod` bars.
///
/// Delegates to [`sliding_max`] for O(n) performance via a monotonic deque.
/// The first `timeperiod - 1` values are `NaN`.
///
/// # Arguments
/// * `real` - Input series.
/// * `timeperiod` - Rolling window size (must be >= 1).
pub fn max(real: &[f64], timeperiod: usize) -> Vec<f64> {
    sliding_max(real, timeperiod)
}

/// Compute the rolling minimum over `timeperiod` bars.
///
/// Delegates to [`sliding_min`] for O(n) performance via a monotonic deque.
/// The first `timeperiod - 1` values are `NaN`.
///
/// # Arguments
/// * `real` - Input series.
/// * `timeperiod` - Rolling window size (must be >= 1).
pub fn min(real: &[f64], timeperiod: usize) -> Vec<f64> {
    sliding_min(real, timeperiod)
}

/// Compute the sliding maximum over `timeperiod` bars in O(n) time.
///
/// Uses a monotonic decreasing deque so each element is pushed/popped at
/// most once. The first `timeperiod - 1` values are `NaN`.
///
/// # Arguments
/// * `real` - Input series.
/// * `timeperiod` - Rolling window size (must be >= 1).
pub fn sliding_max(real: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = real.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 || n < timeperiod {
        return result;
    }
    let mut dq: VecDeque<usize> = VecDeque::new();
    for i in 0..n {
        // Remove indices outside the window
        while dq.front().map(|&j| j + timeperiod <= i).unwrap_or(false) {
            dq.pop_front();
        }
        // Maintain decreasing deque
        while dq.back().map(|&j| real[j] <= real[i]).unwrap_or(false) {
            dq.pop_back();
        }
        dq.push_back(i);
        if i + 1 >= timeperiod {
            result[i] = real[*dq.front().unwrap()];
        }
    }
    result
}

/// Compute the sliding minimum over `timeperiod` bars in O(n) time.
///
/// Uses a monotonic increasing deque so each element is pushed/popped at
/// most once. The first `timeperiod - 1` values are `NaN`.
///
/// # Arguments
/// * `real` - Input series.
/// * `timeperiod` - Rolling window size (must be >= 1).
pub fn sliding_min(real: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = real.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 || n < timeperiod {
        return result;
    }
    let mut dq: VecDeque<usize> = VecDeque::new();
    for i in 0..n {
        // Remove indices outside the window
        while dq.front().map(|&j| j + timeperiod <= i).unwrap_or(false) {
            dq.pop_front();
        }
        // Maintain increasing deque
        while dq.back().map(|&j| real[j] >= real[i]).unwrap_or(false) {
            dq.pop_back();
        }
        dq.push_back(i);
        if i + 1 >= timeperiod {
            result[i] = real[*dq.front().unwrap()];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Element-wise arithmetic operators
// ---------------------------------------------------------------------------

/// Element-wise addition of two arrays.
pub fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// Element-wise subtraction of two arrays.
pub fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
}

/// Element-wise multiplication of two arrays.
pub fn mult(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
}

/// Element-wise division of two arrays (NaN where b=0).
pub fn div(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| if y != 0.0 { x / y } else { f64::NAN })
        .collect()
}

// ---------------------------------------------------------------------------
// Element-wise math transforms
// ---------------------------------------------------------------------------

macro_rules! unary_transform {
    ($name:ident, $method:ident) => {
        pub fn $name(real: &[f64]) -> Vec<f64> {
            real.iter().map(|&x| x.$method()).collect()
        }
    };
}

unary_transform!(math_acos, acos);
unary_transform!(math_asin, asin);
unary_transform!(math_atan, atan);
unary_transform!(math_ceil, ceil);
unary_transform!(math_cos, cos);
unary_transform!(math_cosh, cosh);
unary_transform!(math_exp, exp);
unary_transform!(math_floor, floor);
unary_transform!(math_ln, ln);
unary_transform!(math_log10, log10);
unary_transform!(math_sin, sin);
unary_transform!(math_sinh, sinh);
unary_transform!(math_sqrt, sqrt);
unary_transform!(math_tan, tan);
unary_transform!(math_tanh, tanh);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_basic() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let r = sum(&v, 3);
        assert!(r[0].is_nan());
        assert!((r[2] - 6.0).abs() < 1e-10);
        assert!((r[4] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn max_basic() {
        let v = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let r = max(&v, 3);
        assert!((r[2] - 4.0).abs() < 1e-10);
        assert!((r[4] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn sliding_max_matches_naive() {
        let v = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let naive = max(&v, 3);
        let fast = sliding_max(&v, 3);
        for i in 0..v.len() {
            assert_eq!(naive[i].is_nan(), fast[i].is_nan());
            if !naive[i].is_nan() {
                assert!((naive[i] - fast[i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn sliding_min_matches_naive() {
        let v = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let naive = min(&v, 3);
        let fast = sliding_min(&v, 3);
        for i in 0..v.len() {
            assert_eq!(naive[i].is_nan(), fast[i].is_nan());
            if !naive[i].is_nan() {
                assert!((naive[i] - fast[i]).abs() < 1e-10);
            }
        }
    }
}
