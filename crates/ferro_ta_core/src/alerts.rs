//! Alerts — condition evaluation helpers.
//!
//! - `check_threshold` — fires when a series crosses above/below a level
//! - `check_cross`     — fires when *fast* crosses above or below *slow*
//! - `collect_alert_bars` — returns indices of bars where a mask is non-zero

/// Fire an alert when `series` crosses a threshold level.
///
/// `direction`: `1` = cross above, `-1` = cross below.
///
/// Returns a `Vec<i8>` with `1` at crossing bars, `0` elsewhere.
/// Element 0 is always 0.
pub fn check_threshold(series: &[f64], level: f64, direction: i32) -> Vec<i8> {
    let n = series.len();
    let mut out = vec![0i8; n];
    if n < 2 {
        return out;
    }
    for i in 1..n {
        let prev = series[i - 1];
        let curr = series[i];
        if prev.is_nan() || curr.is_nan() {
            continue;
        }
        if (direction == 1 && prev <= level && curr > level)
            || (direction == -1 && prev >= level && curr < level)
        {
            out[i] = 1;
        }
    }
    out
}

/// Detect cross-over / cross-under events between two series.
///
/// Returns `Vec<i8>`: `1` = bullish cross (fast above slow), `-1` = bearish, `0` = none.
/// Element 0 is always 0.
pub fn check_cross(fast: &[f64], slow: &[f64]) -> Vec<i8> {
    let n = fast.len();
    let mut out = vec![0i8; n];
    if n < 2 {
        return out;
    }
    for i in 1..n {
        let fp = fast[i - 1];
        let fc = fast[i];
        let sp = slow[i - 1];
        let sc = slow[i];
        if fp.is_nan() || fc.is_nan() || sp.is_nan() || sc.is_nan() {
            continue;
        }
        if fp <= sp && fc > sc {
            out[i] = 1;
        } else if fp >= sp && fc < sc {
            out[i] = -1;
        }
    }
    out
}

/// Collect bar indices where `mask` is non-zero.
pub fn collect_alert_bars(mask: &[i8]) -> Vec<i64> {
    mask.iter()
        .enumerate()
        .filter(|(_, &v)| v != 0)
        .map(|(i, _)| i as i64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_threshold_cross_above() {
        let series = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let result = check_threshold(&series, 25.0, 1);
        assert_eq!(result, vec![0, 0, 1, 0, 0]);
    }

    #[test]
    fn test_check_threshold_cross_below() {
        let series = vec![50.0, 40.0, 30.0, 20.0, 10.0];
        let result = check_threshold(&series, 25.0, -1);
        assert_eq!(result, vec![0, 0, 0, 1, 0]);
    }

    #[test]
    fn test_check_cross_bullish() {
        let fast = vec![1.0, 2.0, 5.0];
        let slow = vec![3.0, 3.0, 3.0];
        let result = check_cross(&fast, &slow);
        assert_eq!(result, vec![0, 0, 1]);
    }

    #[test]
    fn test_check_cross_bearish() {
        let fast = vec![5.0, 4.0, 1.0];
        let slow = vec![3.0, 3.0, 3.0];
        let result = check_cross(&fast, &slow);
        assert_eq!(result, vec![0, 0, -1]);
    }

    #[test]
    fn test_collect_alert_bars() {
        let mask = vec![0i8, 1, 0, -1, 0, 1];
        let result = collect_alert_bars(&mask);
        assert_eq!(result, vec![1, 3, 5]);
    }

    #[test]
    fn test_empty() {
        assert_eq!(check_threshold(&[], 0.0, 1), Vec::<i8>::new());
        assert_eq!(check_cross(&[], &[]), Vec::<i8>::new());
        assert_eq!(collect_alert_bars(&[]), Vec::<i64>::new());
    }

    #[test]
    fn test_nan_handling() {
        let series = vec![10.0, f64::NAN, 30.0, 40.0];
        let result = check_threshold(&series, 25.0, 1);
        // NaN bars are skipped
        assert_eq!(result[1], 0);
        assert_eq!(result[2], 0); // prev is NaN
    }
}
