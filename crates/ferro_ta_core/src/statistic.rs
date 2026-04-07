//! Statistic functions.

/// Compute the rolling population standard deviation, scaled by `nbdev`.
///
/// Uses population variance (`ddof = 0`). Returns `nbdev * stddev` for
/// each window. The first `timeperiod - 1` values are `NaN`.
///
/// # Arguments
/// * `real` - Input series.
/// * `timeperiod` - Rolling window size (must be >= 1).
/// * `nbdev` - Multiplier applied to the standard deviation (use 1.0 for raw stddev).
pub fn stddev(real: &[f64], timeperiod: usize, nbdev: f64) -> Vec<f64> {
    let n = real.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 || n < timeperiod {
        return result;
    }
    for i in (timeperiod - 1)..n {
        let window = &real[i + 1 - timeperiod..=i];
        let mean: f64 = window.iter().sum::<f64>() / timeperiod as f64;
        let var: f64 = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / timeperiod as f64;
        result[i] = var.sqrt() * nbdev;
    }
    result
}

/// Rolling population variance, scaled by `nbdev²`.
pub fn var(real: &[f64], timeperiod: usize, nbdev: f64) -> Vec<f64> {
    let n = real.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 || n < timeperiod {
        return result;
    }
    for i in (timeperiod - 1)..n {
        let window = &real[i + 1 - timeperiod..=i];
        let mean: f64 = window.iter().sum::<f64>() / timeperiod as f64;
        let variance: f64 =
            window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / timeperiod as f64;
        result[i] = variance * nbdev * nbdev;
    }
    result
}

// ---------------------------------------------------------------------------
// Linear regression helpers
// ---------------------------------------------------------------------------

fn rolling_linreg_apply<F>(prices: &[f64], timeperiod: usize, mut map: F) -> Vec<f64>
where
    F: FnMut(f64, f64) -> f64,
{
    let n = prices.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod == 0 || n < timeperiod {
        return result;
    }
    let period = timeperiod as f64;
    let last_x = (timeperiod - 1) as f64;
    let sum_x = last_x * period / 2.0;
    let sum_x2 = last_x * period * (2.0 * period - 1.0) / 6.0;
    let denom = period * sum_x2 - sum_x * sum_x;

    let mut sum_y: f64 = prices[..timeperiod].iter().sum();
    let mut sum_xy: f64 = prices[..timeperiod]
        .iter()
        .enumerate()
        .map(|(idx, &v)| idx as f64 * v)
        .sum();

    for end in (timeperiod - 1)..n {
        let slope = if denom != 0.0 {
            (period * sum_xy - sum_x * sum_y) / denom
        } else {
            0.0
        };
        let intercept = (sum_y - slope * sum_x) / period;
        result[end] = map(slope, intercept);
        if end + 1 < n {
            let outgoing = prices[end + 1 - timeperiod];
            let incoming = prices[end + 1];
            let prev_sum_y = sum_y;
            sum_y = prev_sum_y - outgoing + incoming;
            sum_xy = sum_xy - (prev_sum_y - outgoing) + last_x * incoming;
        }
    }
    result
}

/// Linear regression fitted value at the last point of the window.
pub fn linearreg(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let last_x = if timeperiod > 0 {
        (timeperiod - 1) as f64
    } else {
        0.0
    };
    rolling_linreg_apply(close, timeperiod, |slope, intercept| {
        intercept + slope * last_x
    })
}

/// Slope of the rolling linear regression line.
pub fn linearreg_slope(close: &[f64], timeperiod: usize) -> Vec<f64> {
    rolling_linreg_apply(close, timeperiod, |slope, _| slope)
}

/// Intercept of the rolling linear regression line.
pub fn linearreg_intercept(close: &[f64], timeperiod: usize) -> Vec<f64> {
    rolling_linreg_apply(close, timeperiod, |_, intercept| intercept)
}

/// Angle of the regression line in degrees.
pub fn linearreg_angle(close: &[f64], timeperiod: usize) -> Vec<f64> {
    rolling_linreg_apply(close, timeperiod, |slope, _| {
        slope.atan() * 180.0 / std::f64::consts::PI
    })
}

/// Time Series Forecast: linear regression extrapolated one period ahead.
pub fn tsf(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let forecast_x = timeperiod as f64;
    rolling_linreg_apply(close, timeperiod, |slope, intercept| {
        intercept + slope * forecast_x
    })
}

// ---------------------------------------------------------------------------
// Beta (rolling, return-based)
// ---------------------------------------------------------------------------

/// Rolling beta: regression of real1 daily returns on real0 daily returns.
pub fn beta(real0: &[f64], real1: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = real0.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod == 0 || n <= timeperiod {
        return result;
    }

    let price_return = |curr: f64, prev: f64| -> f64 {
        if prev != 0.0 {
            curr / prev - 1.0
        } else {
            f64::NAN
        }
    };
    let rx: Vec<f64> = real0.windows(2).map(|w| price_return(w[1], w[0])).collect();
    let ry: Vec<f64> = real1.windows(2).map(|w| price_return(w[1], w[0])).collect();

    let period = timeperiod as f64;
    let mut sum_rx = 0.0_f64;
    let mut sum_ry = 0.0_f64;
    let mut sum_rx2 = 0.0_f64;
    let mut sum_rxry = 0.0_f64;
    let mut invalid = 0usize;

    for idx in 0..timeperiod {
        let (ret_x, ret_y) = (rx[idx], ry[idx]);
        if ret_x.is_finite() && ret_y.is_finite() {
            sum_rx += ret_x;
            sum_ry += ret_y;
            sum_rx2 += ret_x * ret_x;
            sum_rxry += ret_x * ret_y;
        } else {
            invalid += 1;
        }
    }

    for end in timeperiod..n {
        result[end] = if invalid == 0 {
            let denom = period * sum_rx2 - sum_rx * sum_rx;
            if denom != 0.0 {
                (period * sum_rxry - sum_rx * sum_ry) / denom
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        };

        if end + 1 < n {
            let out = end - timeperiod;
            let (ox, oy) = (rx[out], ry[out]);
            if ox.is_finite() && oy.is_finite() {
                sum_rx -= ox;
                sum_ry -= oy;
                sum_rx2 -= ox * ox;
                sum_rxry -= ox * oy;
            } else {
                invalid -= 1;
            }
            let (ix, iy) = (rx[end], ry[end]);
            if ix.is_finite() && iy.is_finite() {
                sum_rx += ix;
                sum_ry += iy;
                sum_rx2 += ix * ix;
                sum_rxry += ix * iy;
            } else {
                invalid += 1;
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Correlation (rolling Pearson)
// ---------------------------------------------------------------------------

/// Rolling Pearson correlation coefficient between two series.
pub fn correl(real0: &[f64], real1: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = real0.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod == 0 || n < timeperiod {
        return result;
    }

    let period = timeperiod as f64;
    let mut sum_x: f64 = real0[..timeperiod].iter().sum();
    let mut sum_y: f64 = real1[..timeperiod].iter().sum();
    let mut sum_x2: f64 = real0[..timeperiod].iter().map(|v| v * v).sum();
    let mut sum_y2: f64 = real1[..timeperiod].iter().map(|v| v * v).sum();
    let mut sum_xy: f64 = real0[..timeperiod]
        .iter()
        .zip(real1[..timeperiod].iter())
        .map(|(&a, &b)| a * b)
        .sum();

    #[allow(clippy::needless_range_loop)]
    for end in (timeperiod - 1)..n {
        let denom_x = period * sum_x2 - sum_x * sum_x;
        let denom_y = period * sum_y2 - sum_y * sum_y;
        result[end] = if denom_x > 0.0 && denom_y > 0.0 {
            (period * sum_xy - sum_x * sum_y) / (denom_x * denom_y).sqrt()
        } else {
            f64::NAN
        };

        if end + 1 < n {
            let out = end + 1 - timeperiod;
            let inc = end + 1;
            sum_x += real0[inc] - real0[out];
            sum_y += real1[inc] - real1[out];
            sum_x2 += real0[inc] * real0[inc] - real0[out] * real0[out];
            sum_y2 += real1[inc] * real1[inc] - real1[out] * real1[out];
            sum_xy += real0[inc] * real1[inc] - real0[out] * real1[out];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Dynamic Time Warping (DTW)
// ---------------------------------------------------------------------------

/// Internal helper: build the full DTW accumulated-cost matrix.
///
/// Local cost: `|s1[i] - s2[j]|` (Euclidean / L1 for 1-D series).
/// This matches the convention used by `dtaidistance.dtw.distance()`.
///
/// Out-of-band cells (Sakoe-Chiba constraint) are set to `f64::INFINITY`.
fn dtw_matrix(s1: &[f64], s2: &[f64], window: Option<usize>) -> Vec<Vec<f64>> {
    let n = s1.len();
    let m = s2.len();
    let mut dp = vec![vec![f64::INFINITY; m]; n];
    for i in 0..n {
        // Window convention matches dtaidistance: window=w means |i-j| < w.
        // None = unconstrained (full matrix).
        let (j_lo, j_hi) = match window {
            None => (0, m),
            Some(w) => {
                let lo = i.saturating_sub(w.saturating_sub(1));
                let hi = i.saturating_add(w).min(m);
                (lo, hi)
            }
        };
        for j in j_lo..j_hi {
            // Squared Euclidean local cost — matches dtaidistance convention.
            // The final sqrt is applied only once at the top level (not per-step).
            let cost = (s1[i] - s2[j]).powi(2);
            let prev = if i == 0 && j == 0 {
                0.0
            } else if i == 0 {
                dp[0][j - 1]
            } else if j == 0 {
                dp[i - 1][0]
            } else {
                dp[i - 1][j - 1].min(dp[i - 1][j]).min(dp[i][j - 1])
            };
            dp[i][j] = cost + prev;
        }
    }
    dp
}

/// Compute the Dynamic Time Warping distance between two 1-D series.
///
/// Returns the accumulated Euclidean cost along the optimal warping path.
/// Uses `|s1[i] - s2[j]|` as the local cost, matching `dtaidistance` convention.
///
/// # Arguments
/// * `s1` - First time series.
/// * `s2` - Second time series.
/// * `window` - Optional Sakoe-Chiba band width. `None` = unconstrained.
///
/// Returns `f64::NAN` if either input is empty.
pub fn dtw_distance(s1: &[f64], s2: &[f64], window: Option<usize>) -> f64 {
    if s1.is_empty() || s2.is_empty() {
        return f64::NAN;
    }
    let dp = dtw_matrix(s1, s2, window);
    // sqrt applied once at the end — matches dtaidistance.dtw.distance() convention.
    dp[s1.len() - 1][s2.len() - 1].sqrt()
}

/// Compute the DTW distance and the optimal warping path between two 1-D series.
///
/// The warping path is a `Vec<(usize, usize)>` of `(i, j)` index pairs,
/// starting at `(0, 0)` and ending at `(n-1, m-1)`, monotonically non-decreasing.
///
/// # Arguments
/// * `s1` - First time series.
/// * `s2` - Second time series.
/// * `window` - Optional Sakoe-Chiba band width. `None` = unconstrained.
///
/// Returns `(f64::NAN, vec![])` if either input is empty.
pub fn dtw_path(s1: &[f64], s2: &[f64], window: Option<usize>) -> (f64, Vec<(usize, usize)>) {
    if s1.is_empty() || s2.is_empty() {
        return (f64::NAN, vec![]);
    }
    let dp = dtw_matrix(s1, s2, window);
    let dist = dp[s1.len() - 1][s2.len() - 1].sqrt();

    // Backtrace from (n-1, m-1) to (0, 0)
    let mut path = Vec::new();
    let (mut i, mut j) = (s1.len() - 1, s2.len() - 1);
    path.push((i, j));
    while i > 0 || j > 0 {
        let (ni, nj) = match (i, j) {
            (0, _) => (0, j - 1),
            (_, 0) => (i - 1, 0),
            _ => {
                let diag = dp[i - 1][j - 1];
                let up = dp[i - 1][j];
                let left = dp[i][j - 1];
                let best = diag.min(up).min(left);
                if best == diag {
                    (i - 1, j - 1)
                } else if best == up {
                    (i - 1, j)
                } else {
                    (i, j - 1)
                }
            }
        };
        i = ni;
        j = nj;
        path.push((i, j));
    }
    path.reverse();
    (dist, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stddev_constant() {
        let prices = vec![5.0; 5];
        let result = stddev(&prices, 3, 1.0);
        for v in result.iter().filter(|v| !v.is_nan()) {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn dtw_identical_series_is_zero() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(dtw_distance(&a, &a, None), 0.0);
    }

    #[test]
    fn dtw_known_shifted_series() {
        // [0,1,2] vs [1,2,3]: DTW uses squared Euclidean local cost + final sqrt.
        // Optimal path (0,0)→(1,0)→(2,1)→(2,2), accumulated cost = 1+0+0+1 = 2, sqrt(2).
        // Matches dtaidistance.dtw.distance([0,1,2],[1,2,3]) = 1.4142...
        let a = vec![0.0, 1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let expected = 2.0_f64.sqrt();
        let result = dtw_distance(&a, &b, None);
        assert!(
            (result - expected).abs() < 1e-12,
            "got {result}, expected {expected}"
        );
    }

    #[test]
    fn dtw_known_even_shift() {
        // [0,2,4] vs [1,3,5]: diagonal path, squared costs 1+1+1=3, sqrt(3).
        // Matches dtaidistance.dtw.distance([0,2,4],[1,3,5]) = 1.7320...
        let a = vec![0.0, 2.0, 4.0];
        let b = vec![1.0, 3.0, 5.0];
        let expected = 3.0_f64.sqrt();
        let result = dtw_distance(&a, &b, None);
        assert!(
            (result - expected).abs() < 1e-12,
            "got {result}, expected {expected}"
        );
    }

    #[test]
    fn dtw_single_element() {
        let a = vec![3.0];
        let b = vec![7.0];
        assert_eq!(dtw_distance(&a, &b, None), 4.0);
    }

    #[test]
    fn dtw_empty_returns_nan() {
        assert!(dtw_distance(&[], &[1.0, 2.0], None).is_nan());
        assert!(dtw_distance(&[1.0, 2.0], &[], None).is_nan());
    }

    #[test]
    fn dtw_path_endpoints() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.5, 2.5, 3.5, 4.5];
        let (_, path) = dtw_path(&a, &b, None);
        assert_eq!(path.first(), Some(&(0, 0)));
        assert_eq!(path.last(), Some(&(3, 3)));
    }

    #[test]
    fn dtw_path_is_monotone() {
        let a = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let b = vec![2.0, 1.0, 4.0, 3.0, 6.0];
        let (_, path) = dtw_path(&a, &b, None);
        for k in 1..path.len() {
            assert!(path[k].0 >= path[k - 1].0);
            assert!(path[k].1 >= path[k - 1].1);
        }
    }

    #[test]
    fn dtw_path_distance_matches_distance_only() {
        let a = vec![1.0, 4.0, 2.0, 8.0, 3.0];
        let b = vec![2.0, 3.0, 7.0, 4.0, 5.0];
        let d1 = dtw_distance(&a, &b, None);
        let (d2, _) = dtw_path(&a, &b, None);
        assert!((d1 - d2).abs() < 1e-12);
    }

    #[test]
    fn dtw_window_constrained_ge_unconstrained() {
        // window convention matches dtaidistance: Some(w) means |i-j| < w.
        // A narrow window restricts warping, so constrained distance >= unconstrained.
        let a: Vec<f64> = (0..20).map(|x| x as f64).collect();
        let b: Vec<f64> = (0..20).map(|x| x as f64 + 3.0).collect();
        let d_full = dtw_distance(&a, &b, None);
        let d_narrow = dtw_distance(&a, &b, Some(3));
        assert!(d_narrow >= d_full - 1e-12);
    }
}
