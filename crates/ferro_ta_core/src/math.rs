//! Math utilities.

use std::collections::VecDeque;

/// Rolling sum over `timeperiod` bars.
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

/// Rolling maximum over `timeperiod` bars — O(n) via monotonic deque.
pub fn max(real: &[f64], timeperiod: usize) -> Vec<f64> {
    sliding_max(real, timeperiod)
}

/// Rolling minimum over `timeperiod` bars — O(n) via monotonic deque.
pub fn min(real: &[f64], timeperiod: usize) -> Vec<f64> {
    sliding_min(real, timeperiod)
}

/// Sliding maximum over `timeperiod` bars — O(n) via monotonic deque.
///
/// Equivalent to `max` but uses a monotonic deque for O(n) total time.
/// Leading `timeperiod - 1` values are NaN.
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

/// Sliding minimum over `timeperiod` bars — O(n) via monotonic deque.
///
/// Equivalent to `min` but uses a monotonic deque for O(n) total time.
/// Leading `timeperiod - 1` values are NaN.
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
