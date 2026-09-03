//! Signal utilities — crossovers, rolling extremes, change, and state helpers.
//!
//! Boolean-style outputs use `1.0` / `0.0`. Warmup bars that lack a lookback
//! (`CHANGE`, `RISING`, `FALLING`, `HIGHEST`, `LOWEST`, `VALUEWHEN`) are `NaN`.

/// True when `v` is a finite, non-zero signal (treats `0` / `NaN` / `Inf` as off).
#[inline]
fn is_signal(v: f64) -> bool {
    v.is_finite() && v != 0.0
}

/// `1.0` on the bar where `real0` crosses strictly above `real1`.
///
/// A cross is `real0[i-1] <= real1[i-1]` and `real0[i] > real1[i]`.
/// The first bar is `0.0`. Bars with any non-finite input are `0.0`.
pub fn crossover(real0: &[f64], real1: &[f64]) -> Vec<f64> {
    let n = real0.len();
    let mut result = vec![0.0; n];
    if n < 2 || real1.len() != n {
        return result;
    }
    for i in 1..n {
        let a0 = real0[i - 1];
        let a1 = real0[i];
        let b0 = real1[i - 1];
        let b1 = real1[i];
        if a0.is_finite()
            && a1.is_finite()
            && b0.is_finite()
            && b1.is_finite()
            && a0 <= b0
            && a1 > b1
        {
            result[i] = 1.0;
        }
    }
    result
}

/// `1.0` on the bar where `real0` crosses strictly below `real1`.
///
/// A cross is `real0[i-1] >= real1[i-1]` and `real0[i] < real1[i]`.
/// The first bar is `0.0`. Bars with any non-finite input are `0.0`.
pub fn crossunder(real0: &[f64], real1: &[f64]) -> Vec<f64> {
    let n = real0.len();
    let mut result = vec![0.0; n];
    if n < 2 || real1.len() != n {
        return result;
    }
    for i in 1..n {
        let a0 = real0[i - 1];
        let a1 = real0[i];
        let b0 = real1[i - 1];
        let b1 = real1[i];
        if a0.is_finite()
            && a1.is_finite()
            && b0.is_finite()
            && b1.is_finite()
            && a0 >= b0
            && a1 < b1
        {
            result[i] = 1.0;
        }
    }
    result
}

/// `1.0` on any bar where `real0` crosses `real1` in either direction.
pub fn cross(real0: &[f64], real1: &[f64]) -> Vec<f64> {
    let up = crossover(real0, real1);
    let down = crossunder(real0, real1);
    up.into_iter()
        .zip(down)
        .map(|(a, b)| if a > 0.0 || b > 0.0 { 1.0 } else { 0.0 })
        .collect()
}

/// Rolling highest value over `timeperiod` bars. Wraps [`crate::math::max`].
///
/// Leading `timeperiod - 1` values are `NaN`.
pub fn highest(real: &[f64], timeperiod: usize) -> Vec<f64> {
    crate::math::max(real, timeperiod)
}

/// Rolling lowest value over `timeperiod` bars. Wraps [`crate::math::min`].
///
/// Leading `timeperiod - 1` values are `NaN`.
pub fn lowest(real: &[f64], timeperiod: usize) -> Vec<f64> {
    crate::math::min(real, timeperiod)
}

/// Lookback difference: `real[i] - real[i - timeperiod]`.
///
/// Leading `timeperiod` values are `NaN`.
pub fn change(real: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = real.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 {
        return result;
    }
    for i in timeperiod..n {
        let cur = real[i];
        let prev = real[i - timeperiod];
        if cur.is_finite() && prev.is_finite() {
            result[i] = cur - prev;
        }
    }
    result
}

/// `1.0` when `real[i]` is strictly greater than `real[i - timeperiod]`.
///
/// Leading `timeperiod` values are `NaN`. Equivalent to `CHANGE(...) > 0`
/// on finite bars.
pub fn rising(real: &[f64], timeperiod: usize) -> Vec<f64> {
    signed_compare(real, timeperiod, true)
}

/// `1.0` when `real[i]` is strictly less than `real[i - timeperiod]`.
///
/// Leading `timeperiod` values are `NaN`. Equivalent to `CHANGE(...) < 0`
/// on finite bars.
pub fn falling(real: &[f64], timeperiod: usize) -> Vec<f64> {
    signed_compare(real, timeperiod, false)
}

fn signed_compare(real: &[f64], timeperiod: usize, want_up: bool) -> Vec<f64> {
    let n = real.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 {
        return result;
    }
    for i in timeperiod..n {
        let cur = real[i];
        let prev = real[i - timeperiod];
        if cur.is_finite() && prev.is_finite() {
            let cmp = if want_up { cur > prev } else { cur < prev };
            result[i] = if cmp { 1.0 } else { 0.0 };
        }
    }
    result
}

/// Excess-removal: keep the first `primary` signal and suppress further
/// primaries until a `secondary` signal occurs.
///
/// Same-bar primary and secondary: the primary is emitted (if latched off)
/// and the latch then resets so the next primary can fire.
pub fn exrem(primary: &[f64], secondary: &[f64]) -> Vec<f64> {
    let n = primary.len();
    let mut result = vec![0.0; n];
    if secondary.len() != n {
        return result;
    }
    let mut waiting_for_secondary = false;
    for i in 0..n {
        if is_signal(primary[i]) && !waiting_for_secondary {
            result[i] = 1.0;
            waiting_for_secondary = true;
        }
        if is_signal(secondary[i]) {
            waiting_for_secondary = false;
        }
    }
    result
}

/// Hold `1.0` from a `primary` signal until a `secondary` signal clears it.
///
/// Same-bar primary and secondary leaves the state off (secondary wins).
pub fn flip(primary: &[f64], secondary: &[f64]) -> Vec<f64> {
    let n = primary.len();
    let mut result = vec![0.0; n];
    if secondary.len() != n {
        return result;
    }
    let mut on = false;
    for i in 0..n {
        if is_signal(primary[i]) {
            on = true;
        }
        if is_signal(secondary[i]) {
            on = false;
        }
        if on {
            result[i] = 1.0;
        }
    }
    result
}

/// Value of `real` at the `occurrence`-th most recent true `condition`.
///
/// `occurrence = 1` is the most recent hit (including the current bar).
/// Bars that have not yet seen that many hits are `NaN`.
pub fn valuewhen(condition: &[f64], real: &[f64], occurrence: usize) -> Vec<f64> {
    let n = condition.len();
    let mut result = vec![f64::NAN; n];
    if occurrence < 1 || real.len() != n {
        return result;
    }
    let mut hist: Vec<f64> = Vec::with_capacity(occurrence);
    for i in 0..n {
        if is_signal(condition[i]) && real[i].is_finite() {
            hist.insert(0, real[i]);
            if hist.len() > occurrence {
                hist.pop();
            }
        }
        if hist.len() >= occurrence {
            result[i] = hist[occurrence - 1];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_all_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert_eq!(
                a.is_nan(),
                e.is_nan(),
                "NaN mismatch: {actual:?} vs {expected:?}"
            );
            if !e.is_nan() {
                assert!((a - e).abs() < 1e-10, "{a} != {e}");
            }
        }
    }

    #[test]
    fn crossover_golden() {
        let fast = [1.0, 2.0, 5.0];
        let slow = [3.0, 3.0, 3.0];
        assert_eq!(crossover(&fast, &slow), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn crossunder_golden() {
        let fast = [5.0, 4.0, 1.0];
        let slow = [3.0, 3.0, 3.0];
        assert_eq!(crossunder(&fast, &slow), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn cross_either_direction() {
        let a = [1.0, 4.0, 1.0];
        let b = [2.0, 2.0, 2.0];
        assert_eq!(cross(&a, &b), vec![0.0, 1.0, 1.0]);
        assert_eq!(crossover(&a, &b), vec![0.0, 1.0, 0.0]);
        assert_eq!(crossunder(&a, &b), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn crossover_equal_then_above() {
        let a = [2.0, 3.0];
        let b = [2.0, 2.0];
        assert_eq!(crossover(&a, &b), vec![0.0, 1.0]);
    }

    #[test]
    fn crossover_nan_is_not_a_cross() {
        let a = [1.0, f64::NAN, 5.0];
        let b = [3.0, 3.0, 3.0];
        assert_eq!(crossover(&a, &b), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn highest_wraps_max() {
        let v = [3.0, 1.0, 4.0, 1.0, 5.0];
        let h = highest(&v, 3);
        let m = crate::math::max(&v, 3);
        assert_all_close(&h, &m);
        assert!(h[0].is_nan() && h[1].is_nan());
        assert!((h[2] - 4.0).abs() < 1e-10);
        assert!((h[4] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn lowest_wraps_min() {
        let v = [3.0, 1.0, 4.0, 1.0, 5.0];
        assert_all_close(&lowest(&v, 3), &crate::math::min(&v, 3));
    }

    #[test]
    fn change_golden_period2() {
        let prices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = change(&prices, 2);
        assert!(result[0].is_nan() && result[1].is_nan());
        for v in &result[2..] {
            assert!((v - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn change_matches_mom() {
        let prices = [1.0, 3.0, 2.0, 8.0, 5.0];
        assert_all_close(&change(&prices, 2), &crate::momentum::mom(&prices, 2));
    }

    #[test]
    fn rising_falling_vs_n_bars_ago() {
        let x = [1.0, 2.0, 1.5, 3.0];
        // period 2: compare to i-2
        assert_all_close(&rising(&x, 2), &[f64::NAN, f64::NAN, 1.0, 1.0]);
        assert_all_close(&falling(&x, 2), &[f64::NAN, f64::NAN, 0.0, 0.0]);
        let down = [5.0, 4.0, 3.0];
        assert_all_close(&falling(&down, 1), &[f64::NAN, 1.0, 1.0]);
        assert_all_close(&rising(&down, 1), &[f64::NAN, 0.0, 0.0]);
    }

    #[test]
    fn exrem_keeps_first_until_reset() {
        let primary = [1.0, 1.0, 0.0, 1.0, 0.0];
        let secondary = [0.0, 0.0, 1.0, 0.0, 0.0];
        assert_eq!(exrem(&primary, &secondary), vec![1.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn exrem_same_bar_fires_then_resets() {
        let primary = [1.0, 1.0];
        let secondary = [1.0, 0.0];
        assert_eq!(exrem(&primary, &secondary), vec![1.0, 1.0]);
    }

    #[test]
    fn flip_holds_until_off() {
        let on = [1.0, 0.0, 0.0, 0.0];
        let off = [0.0, 0.0, 1.0, 0.0];
        assert_eq!(flip(&on, &off), vec![1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn flip_same_bar_secondary_wins() {
        let on = [1.0];
        let off = [1.0];
        assert_eq!(flip(&on, &off), vec![0.0]);
    }

    #[test]
    fn valuewhen_occurrence_one_is_most_recent() {
        let cond = [0.0, 1.0, 0.0, 1.0, 0.0];
        let src = [10.0, 20.0, 30.0, 40.0, 50.0];
        assert_all_close(
            &valuewhen(&cond, &src, 1),
            &[f64::NAN, 20.0, 20.0, 40.0, 40.0],
        );
        assert_all_close(
            &valuewhen(&cond, &src, 2),
            &[f64::NAN, f64::NAN, f64::NAN, 20.0, 20.0],
        );
    }

    #[test]
    fn empty_and_mismatch_are_safe() {
        assert!(crossover(&[], &[]).is_empty());
        assert_eq!(crossover(&[1.0], &[1.0, 2.0]), vec![0.0]);
        assert!(change(&[], 1).is_empty());
        assert!(valuewhen(&[], &[], 1).is_empty());
        assert!(valuewhen(&[1.0], &[2.0], 0)[0].is_nan());
    }
}
