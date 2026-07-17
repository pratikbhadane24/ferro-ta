//! Rolling math operators — O(n) sliding window implementations.
//!
//! - `rolling_sum`      — rolling sum over `timeperiod` bars (prefix-sum based)
//! - `rolling_max`      — rolling maximum (O(n) monotonic deque)
//! - `rolling_min`      — rolling minimum (O(n) monotonic deque)
//! - `rolling_maxindex` — index of rolling maximum
//! - `rolling_minindex` — index of rolling minimum

use std::collections::VecDeque;

/// Rolling sum over `timeperiod` bars using an O(n) sliding window.
/// Leading `timeperiod - 1` values are NaN.
///
/// A NaN input only invalidates the windows that actually contain it; once it
/// slides out, sums resume (matching `rolling_max` / `rolling_min`). A
/// prefix-sum formulation would instead poison every later window.
pub fn rolling_sum(real: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = real.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod == 0 || n < timeperiod {
        return result;
    }
    let mut sum = 0.0_f64;
    let mut nan_count: usize = 0;
    for i in 0..n {
        if real[i].is_nan() {
            nan_count += 1;
        } else {
            sum += real[i];
        }
        if i >= timeperiod {
            let old = real[i - timeperiod];
            if old.is_nan() {
                nan_count -= 1;
            } else {
                sum -= old;
            }
        }
        if i + 1 >= timeperiod {
            result[i] = if nan_count > 0 { f64::NAN } else { sum };
        }
    }
    result
}

/// Rolling maximum over `timeperiod` bars (O(n) monotonic deque).
/// Delegates to `math::sliding_max`.
pub fn rolling_max(real: &[f64], timeperiod: usize) -> Vec<f64> {
    crate::math::sliding_max(real, timeperiod)
}

/// Rolling minimum over `timeperiod` bars (O(n) monotonic deque).
/// Delegates to `math::sliding_min`.
pub fn rolling_min(real: &[f64], timeperiod: usize) -> Vec<f64> {
    crate::math::sliding_min(real, timeperiod)
}

/// Index of rolling maximum over `timeperiod` bars.
/// Returns 0-based index. During warmup the value is `-1`.
pub fn rolling_maxindex(real: &[f64], timeperiod: usize) -> Vec<i64> {
    let n = real.len();
    let mut result = vec![-1i64; n];
    if timeperiod == 0 || n < timeperiod {
        return result;
    }
    let mut dq: VecDeque<usize> = VecDeque::new();
    for i in 0..n {
        while dq.front().map(|&j| j + timeperiod <= i).unwrap_or(false) {
            dq.pop_front();
        }
        while dq.back().map(|&j| real[j] <= real[i]).unwrap_or(false) {
            dq.pop_back();
        }
        dq.push_back(i);
        if i + 1 >= timeperiod {
            result[i] = *dq.front().unwrap() as i64;
        }
    }
    result
}

/// Index of rolling minimum over `timeperiod` bars.
/// Returns 0-based index. During warmup the value is `-1`.
pub fn rolling_minindex(real: &[f64], timeperiod: usize) -> Vec<i64> {
    let n = real.len();
    let mut result = vec![-1i64; n];
    if timeperiod == 0 || n < timeperiod {
        return result;
    }
    let mut dq: VecDeque<usize> = VecDeque::new();
    for i in 0..n {
        while dq.front().map(|&j| j + timeperiod <= i).unwrap_or(false) {
            dq.pop_front();
        }
        while dq.back().map(|&j| real[j] >= real[i]).unwrap_or(false) {
            dq.pop_back();
        }
        dq.push_back(i);
        if i + 1 >= timeperiod {
            result[i] = *dq.front().unwrap() as i64;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_sum(&data, 3);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 6.0).abs() < 1e-10); // 1+2+3
        assert!((result[3] - 9.0).abs() < 1e-10); // 2+3+4
        assert!((result[4] - 12.0).abs() < 1e-10); // 3+4+5
    }

    #[test]
    fn test_rolling_sum_recovers_after_nan_leaves_window() {
        let data = vec![1.0, f64::NAN, 2.0, 3.0, 4.0];
        let result = rolling_sum(&data, 2);
        assert!(result[0].is_nan()); // warmup
        assert!(result[1].is_nan()); // window [1, NaN]
        assert!(result[2].is_nan()); // window [NaN, 2]
                                     // NaN has slid out — these windows are clean.
        assert!((result[3] - 5.0).abs() < 1e-10); // 2+3
        assert!((result[4] - 7.0).abs() < 1e-10); // 3+4
    }

    #[test]
    fn test_rolling_max() {
        let data = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let result = rolling_max(&data, 3);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 3.0).abs() < 1e-10);
        assert!((result[3] - 5.0).abs() < 1e-10);
        assert!((result[4] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_min() {
        let data = vec![5.0, 3.0, 4.0, 1.0, 2.0];
        let result = rolling_min(&data, 3);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 3.0).abs() < 1e-10);
        assert!((result[3] - 1.0).abs() < 1e-10);
        assert!((result[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_maxindex() {
        let data = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let result = rolling_maxindex(&data, 3);
        assert_eq!(result[0], -1);
        assert_eq!(result[1], -1);
        assert_eq!(result[2], 1); // max(1,3,2) at index 1
        assert_eq!(result[3], 3); // max(3,2,5) at index 3
        assert_eq!(result[4], 3); // max(2,5,4) at index 3
    }

    #[test]
    fn test_rolling_minindex() {
        let data = vec![5.0, 3.0, 4.0, 1.0, 2.0];
        let result = rolling_minindex(&data, 3);
        assert_eq!(result[0], -1);
        assert_eq!(result[1], -1);
        assert_eq!(result[2], 1); // min(5,3,4) at index 1
        assert_eq!(result[3], 3); // min(3,4,1) at index 3
        assert_eq!(result[4], 3); // min(4,1,2) at index 3
    }

    #[test]
    fn test_short_input() {
        let data = vec![1.0, 2.0];
        let result = rolling_sum(&data, 5);
        assert!(result.iter().all(|v| v.is_nan()));
    }
}
