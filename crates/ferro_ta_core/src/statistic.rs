//! Statistic functions.

use crate::rolling::RollingVariance;

/// Drive a single [`RollingVariance`] across `real` and map each window's
/// population variance to an output value.
///
/// This is the O(n) core shared by [`stddev`] and [`var`]. It replaces the
/// previous O(n * timeperiod) shape — a mean pass plus a squared-deviation
/// pass for *every* window — with one Welford accumulator advanced a single
/// step per bar.
///
/// # Numerical notes
///
/// [`RollingVariance`] is the rolling Welford (West's update) already vetted
/// in-tree by `overlap::bbands`, deliberately **not** the naive
/// `sum(x^2)/N - mean^2` form that TA-Lib's own `TA_VAR` uses. The naive
/// form's relative error scales as `(mean / sigma)^2 * eps`: harmless at
/// price 100 / sigma 0.05, but ~18% relative at index or crypto levels
/// (price 1e5, sigma 0.01), where it can also drive the second moment
/// negative and hand `sqrt` a negative argument.
///
/// Welford removes that cancellation, but no *rolling* second moment is
/// unconditionally accurate, and the second moment's `m2 < 0.0` clamp is not
/// what makes it so. Two things a bare rolling Welford gets wrong, both from
/// entirely **finite** input, and what [`RollingVariance`] does about them:
///
/// * **Non-finite accumulator.** `m2` can overflow to `+inf` on one bar and
///   take `-inf` on the next, leaving `NaN`. The clamp is inert there
///   (`NaN < 0.0` is false), so the `NaN` would persist for every subsequent
///   window. [`RollingVariance::needs_reseed`] detects a non-finite `mean` or
///   `m2` over an all-finite window and recomputes exactly.
/// * **Dynamic range.** A single large-but-finite bar destroys `m2`'s
///   low-order bits while it is in the window, and the damage surfaces once it
///   *leaves* — where the clamp turns the wreckage into a confident `0.0`.
///   [`RollingVariance`] tracks the peak `m2` since its last exact recompute
///   and forces one when `m2` has collapsed by more than
///   `rolling::MAX_ACCUMULATOR_RANGE` from that peak.
///
/// With both triggers plus the periodic reseed every
/// `rolling::RESEED_INTERVAL` advances, the variance these kernels report stays
/// within `2^-26` (~1.5e-8) relative of a two-pass recompute over the same
/// window, independent of series length.
///
/// # Non-finite input
///
/// A mid-series non-finite value corrupts exactly the `timeperiod` outputs
/// whose window contains it and then recovers, matching the previous two-pass
/// behaviour, because [`RollingVariance::advance`] recomputes exactly once the
/// last non-finite entrant has left the window. An output is `NaN` only where a
/// two-pass recompute over the same window would also be `NaN`; a window that
/// merely *overflows* (all-finite bars whose squared deviations exceed `f64`)
/// reports `+inf`, again as the two-pass form does.
fn rolling_population_var_apply<F>(real: &[f64], timeperiod: usize, mut map: F) -> Vec<f64>
where
    F: FnMut(f64) -> f64,
{
    let n = real.len();
    if timeperiod < 1 || n < timeperiod {
        return vec![f64::NAN; n];
    }
    // Pre-size and store by index. `Vec::with_capacity` + `push` does skip the
    // NaN fill's store pass, but it charges a capacity check and a vec-header
    // reload per bar, which measured slower than the one-time prologue. The
    // `timeperiod - 1` warm-up slots keep their initialized `NaN`. Note the
    // accumulator still reads its window straight out of `real`, so the slide
    // and any reseed see exactly the same slice as before.
    let mut out = vec![f64::NAN; n];

    let mut acc = RollingVariance::new(&real[..timeperiod]);
    out[timeperiod - 1] = map(acc.population_var());
    for i in timeperiod..n {
        let window = &real[i + 1 - timeperiod..=i];
        acc.advance(real[i], real[i - timeperiod], window);
        out[i] = map(acc.population_var());
    }
    out
}

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
    rolling_population_var_apply(real, timeperiod, |variance| variance.sqrt() * nbdev)
}

/// Rolling population variance, scaled by `nbdev**2`.
pub fn var(real: &[f64], timeperiod: usize, nbdev: f64) -> Vec<f64> {
    rolling_population_var_apply(real, timeperiod, |variance| variance * nbdev * nbdev)
}

// ---------------------------------------------------------------------------
// Linear regression helpers
// ---------------------------------------------------------------------------

fn rolling_linreg_apply<F>(prices: &[f64], timeperiod: usize, mut map: F) -> Vec<f64>
where
    F: FnMut(f64, f64) -> f64,
{
    let n = prices.len();
    if timeperiod == 0 || n < timeperiod {
        return vec![f64::NAN; n];
    }
    // See the note in `rolling_population_var_apply`: pre-size and store by
    // index; `push` costs more per bar than the NaN fill costs once.
    let mut result = vec![f64::NAN; n];
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
    if timeperiod == 0 || n <= timeperiod {
        return vec![f64::NAN; n];
    }
    // Pre-size and store by index (see `rolling_population_var_apply`). Unlike
    // its siblings this one stores **unconditionally**, selecting `NaN` in the
    // degenerate arms rather than skipping the write. `beta` does the least
    // per-bar arithmetic of any kernel here, so a nested
    // `if invalid == 0 { if denom != 0.0 { .. } }` around the store measured
    // ~13% slower than an unconditional store of a select --- the branches cost
    // more than the store they were avoiding. The `push` form had the same
    // unconditional shape, which is why it did not show the regression the
    // other kernels did; the fix is to keep the select, not to keep `push`.
    let mut result = vec![f64::NAN; n];

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
        let denom = period * sum_rx2 - sum_rx * sum_rx;
        result[end] = if invalid == 0 && denom != 0.0 {
            (period * sum_rxry - sum_rx * sum_ry) / denom
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
    if timeperiod == 0 || n < timeperiod {
        return vec![f64::NAN; n];
    }
    // Pre-size and store by index (see `rolling_population_var_apply`); a
    // degenerate denominator leaves the slot at its initialized `NaN`.
    let mut result = vec![f64::NAN; n];

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

    for end in (timeperiod - 1)..n {
        let denom_x = period * sum_x2 - sum_x * sum_x;
        let denom_y = period * sum_y2 - sum_y * sum_y;
        if denom_x > 0.0 && denom_y > 0.0 {
            result[end] = (period * sum_xy - sum_x * sum_y) / (denom_x * denom_y).sqrt();
        }

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

    // -- Two-pass reference: the pre-rewrite implementation, kept verbatim as
    // -- the equivalence oracle for the rolling Welford form.

    fn reference_stddev(real: &[f64], timeperiod: usize, nbdev: f64) -> Vec<f64> {
        let n = real.len();
        let mut result = vec![f64::NAN; n];
        if timeperiod < 1 || n < timeperiod {
            return result;
        }
        for i in (timeperiod - 1)..n {
            let window = &real[i + 1 - timeperiod..=i];
            let mean: f64 = window.iter().sum::<f64>() / timeperiod as f64;
            let var: f64 =
                window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / timeperiod as f64;
            result[i] = var.sqrt() * nbdev;
        }
        result
    }

    fn reference_var(real: &[f64], timeperiod: usize, nbdev: f64) -> Vec<f64> {
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

    /// The naive `sum(x^2)/N - mean^2` accumulator this rewrite deliberately
    /// rejects (the form TA-Lib's own `TA_VAR` uses). Present only so the
    /// large-mean/small-sigma test can show how badly it fails.
    fn naive_sumsq_var(real: &[f64], timeperiod: usize) -> Vec<f64> {
        let n = real.len();
        let mut result = vec![f64::NAN; n];
        if timeperiod < 1 || n < timeperiod {
            return result;
        }
        let p = timeperiod as f64;
        let mut sum: f64 = real[..timeperiod].iter().sum();
        let mut sum_sq: f64 = real[..timeperiod].iter().map(|&x| x * x).sum();
        result[timeperiod - 1] = sum_sq / p - (sum / p) * (sum / p);
        for i in timeperiod..n {
            let x_new = real[i];
            let x_old = real[i - timeperiod];
            sum += x_new - x_old;
            sum_sq += x_new * x_new - x_old * x_old;
            result[i] = sum_sq / p - (sum / p) * (sum / p);
        }
        result
    }

    /// Deterministic pseudo-random series (LCG), so the tests need no dev-dep.
    fn noisy_series(n: usize, mean: f64, amplitude: f64) -> Vec<f64> {
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = ((state >> 11) as f64) / ((1u64 << 53) as f64);
                mean + amplitude * (unit - 0.5)
            })
            .collect()
    }

    /// Mixed relative/absolute comparison.
    ///
    /// A pure relative bound is the wrong gate for a rolling variance: the
    /// error of the Welford recurrence scales with `eps * mean^2`, so a window
    /// whose dispersion is tiny next to its mean (`timeperiod = 2` on a noisy
    /// price series, say) has a large *relative* error and a negligible
    /// absolute one. The absolute floor is what the binding gate actually
    /// cares about; `1e-8` is two orders below that gate's `atol = 1e-6`.
    ///
    /// The worst observed case across the cases below is `timeperiod = 2` on a
    /// strongly trending series (dispersion 0.185 against a level of 235,
    /// i.e. `mean / sigma ~ 1.3e3`), where the recurrence accumulates ~4e-10
    /// absolute before the `RESEED_INTERVAL` recompute clears it.
    const ABS_FLOOR: f64 = 1e-8;

    fn assert_close(got: &[f64], want: &[f64], rtol: f64, label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: length");
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            if w.is_nan() {
                assert!(g.is_nan(), "{label}: i={i} expected NaN, got {g}");
                continue;
            }
            assert!(g.is_finite(), "{label}: i={i} non-finite {g}");
            let tolerance = rtol * w.abs() + ABS_FLOOR;
            assert!(
                (g - w).abs() <= tolerance,
                "{label}: i={i} got={g} want={w} diff={} tol={tolerance}",
                (g - w).abs()
            );
        }
    }

    #[test]
    fn stddev_constant() {
        let prices = vec![5.0; 5];
        let result = stddev(&prices, 3, 1.0);
        for v in result.iter().filter(|v| !v.is_nan()) {
            // Exactly zero, not merely tiny: on a constant series every Welford
            // `delta` is exactly `0.0`, so no rounding enters `m2` at all. The
            // `m2 < 0.0` clamp is not what produces this and must not be relied
            // on to; see the accuracy contract on `rolling::RollingVariance`.
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn stddev_and_var_match_the_two_pass_reference() {
        // Ordinary equity-like scales. `rtol = 1e-11` is ~2 orders above the
        // observed ~1e-13 relative deviation between the rolling Welford and a
        // fresh two-pass recompute, and 5 orders *below* the 1e-6 absolute gate
        // in `tests/integration/test_vs_talib.py`.
        const RTOL: f64 = 1e-11;
        let cases: Vec<(&str, Vec<f64>)> = vec![
            ("noisy_100", noisy_series(500, 100.0, 4.0)),
            ("noisy_1", noisy_series(500, 1.0, 0.4)),
            (
                "trending",
                (0..500).map(|i| 50.0 + i as f64 * 0.37).collect(),
            ),
            ("constant", vec![7.5; 200]),
            (
                "alternating",
                (0..300)
                    .map(|i| if i % 2 == 0 { 10.0 } else { 20.0 })
                    .collect(),
            ),
        ];
        for (label, data) in &cases {
            for &p in &[1usize, 2, 5, 14, 20, 63] {
                assert_close(
                    &stddev(data, p, 1.0),
                    &reference_stddev(data, p, 1.0),
                    RTOL,
                    &format!("stddev {label} p={p}"),
                );
                assert_close(
                    &var(data, p, 1.0),
                    &reference_var(data, p, 1.0),
                    RTOL,
                    &format!("var {label} p={p}"),
                );
                assert_close(
                    &stddev(data, p, 2.5),
                    &reference_stddev(data, p, 2.5),
                    RTOL,
                    &format!("stddev nbdev {label} p={p}"),
                );
                assert_close(
                    &var(data, p, 2.5),
                    &reference_var(data, p, 2.5),
                    RTOL,
                    &format!("var nbdev {label} p={p}"),
                );
            }
        }
    }

    #[test]
    fn welford_survives_large_mean_small_sigma_where_naive_sumsq_fails() {
        // Index / crypto scale: mean 1e5, sigma ~1e-2, so (mean/sigma)^2 ~ 1e14
        // and the naive `sum(x^2)/N - mean^2` cancellation is catastrophic.
        let data = noisy_series(600, 1e5, 0.035);
        let p = 20;
        let expected_var = reference_var(&data, p, 1.0);
        let expected_stddev = reference_stddev(&data, p, 1.0);

        // Welford: `rtol = 1e-6` on the *variance*. At sigma ~1e-2 that is
        // ~1e-10 absolute on the variance and ~5e-11 absolute on the stddev,
        // i.e. still four-plus orders below the 1e-6 TA-Lib gate. The observed
        // figure is ~1e-8 relative; the loose bound is deliberate headroom for
        // the scalar / SIMD reseed paths.
        assert_close(&var(&data, p, 1.0), &expected_var, 1e-6, "welford var");
        assert_close(
            &stddev(&data, p, 1.0),
            &expected_stddev,
            1e-6,
            "welford stddev",
        );

        // And the contrast that justifies the choice: the naive form is off by
        // percent-level *relative* error on the same data.
        let naive = naive_sumsq_var(&data, p);
        let worst = naive
            .iter()
            .zip(expected_var.iter())
            .filter(|(_, w)| w.is_finite() && **w > 0.0)
            .map(|(g, w)| (g - w).abs() / w)
            .fold(0.0_f64, f64::max);
        assert!(
            worst > 1e-3,
            "naive sum-of-squares was unexpectedly accurate here (worst rel err {worst}); \
             the premise of this test no longer holds"
        );
    }

    #[test]
    fn mid_series_nan_corrupts_exactly_timeperiod_outputs() {
        let p = 8;
        let mut data = noisy_series(120, 100.0, 3.0);
        let nan_at = 40;
        data[nan_at] = f64::NAN;

        for out in [stddev(&data, p, 1.0), var(&data, p, 1.0)] {
            // Warmup NaNs.
            for (i, v) in out.iter().enumerate().take(p - 1) {
                assert!(v.is_nan(), "warmup i={i}");
            }
            // Exactly the `p` windows that contain the NaN are corrupted.
            for i in nan_at..nan_at + p {
                assert!(out[i].is_nan(), "expected NaN at i={i}");
            }
            // Clean before and after — the accumulator recovers immediately.
            for (i, v) in out.iter().enumerate().take(nan_at).skip(p - 1) {
                assert!(v.is_finite(), "pre-NaN i={i} was {v}");
            }
            for (i, v) in out.iter().enumerate().skip(nan_at + p) {
                assert!(v.is_finite(), "post-NaN i={i} was {v}");
            }
        }
        // Recovery is exact, not merely finite.
        let clean = reference_stddev(&data, p, 1.0);
        let got = stddev(&data, p, 1.0);
        for i in (nan_at + p)..data.len() {
            assert!(
                (got[i] - clean[i]).abs() <= 1e-11 * clean[i].abs() + ABS_FLOOR,
                "post-NaN recovery i={i} got={} want={}",
                got[i],
                clean[i]
            );
        }
    }

    #[test]
    fn stddev_and_var_hand_computed_golden() {
        // real = [2, 4, 4, 4, 5, 5, 7, 9], timeperiod = 4, population (ddof=0).
        //
        //   i=3  [2,4,4,4]  mean 14/4 = 3.5   SSD 2.25+.25+.25+.25 = 3.00  var 0.7500
        //   i=4  [4,4,4,5]  mean 17/4 = 4.25  SSD .0625*3+.5625  = 0.75    var 0.1875
        //   i=5  [4,4,5,5]  mean 18/4 = 4.5   SSD .25*4          = 1.00    var 0.2500
        //   i=6  [4,5,5,7]  mean 21/4 = 5.25  SSD 1.5625+.0625*2+3.0625    var 1.1875
        //   i=7  [5,5,7,9]  mean 26/4 = 6.5   SSD 2.25+2.25+.25+6.25 = 11  var 2.7500
        //
        // Every mean and SSD above is exact in binary64, so the variances are
        // exact and the stddevs are the exact square roots of them.
        let real = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let expected_var = [0.75, 0.1875, 0.25, 1.1875, 2.75];

        let got_var = var(&real, 4, 1.0);
        let got_stddev = stddev(&real, 4, 1.0);
        for i in 0..3 {
            assert!(got_var[i].is_nan());
            assert!(got_stddev[i].is_nan());
        }
        for (k, &want) in expected_var.iter().enumerate() {
            let i = k + 3;
            assert!(
                (got_var[i] - want).abs() < 1e-12,
                "var i={i} got={} want={want}",
                got_var[i]
            );
            assert!(
                (got_stddev[i] - want.sqrt()).abs() < 1e-12,
                "stddev i={i} got={} want={}",
                got_stddev[i],
                want.sqrt()
            );
        }

        // nbdev scaling: stddev scales linearly, var quadratically.
        let scaled_stddev = stddev(&real, 4, 2.0);
        let scaled_var = var(&real, 4, 2.0);
        for k in 0..expected_var.len() {
            let i = k + 3;
            assert!((scaled_stddev[i] - 2.0 * got_stddev[i]).abs() < 1e-12);
            assert!((scaled_var[i] - 4.0 * got_var[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn timeperiod_one_has_zero_dispersion() {
        let data = noisy_series(50, 100.0, 5.0);
        for (i, (&s, &v)) in stddev(&data, 1, 1.0)
            .iter()
            .zip(var(&data, 1, 1.0).iter())
            .enumerate()
        {
            assert_eq!(s, 0.0, "stddev i={i}");
            assert_eq!(v, 0.0, "var i={i}");
        }
    }

    #[test]
    fn stddev_and_var_degenerate_inputs() {
        let data = noisy_series(10, 100.0, 5.0);

        // timeperiod == n: exactly one non-NaN output, at the last slot.
        let full = stddev(&data, data.len(), 1.0);
        assert_eq!(full.len(), data.len());
        assert!(full[..data.len() - 1].iter().all(|v| v.is_nan()));
        assert!(full[data.len() - 1].is_finite());

        // timeperiod > n: all NaN, length preserved.
        for out in [stddev(&data, 11, 1.0), var(&data, 11, 1.0)] {
            assert_eq!(out.len(), data.len());
            assert!(out.iter().all(|v| v.is_nan()));
        }

        // timeperiod == 0 is rejected by the `timeperiod < 1` guard.
        assert!(stddev(&data, 0, 1.0).iter().all(|v| v.is_nan()));
        assert!(var(&data, 0, 1.0).iter().all(|v| v.is_nan()));

        // Empty input.
        assert!(stddev(&[], 5, 1.0).is_empty());
        assert!(var(&[], 5, 1.0).is_empty());
        assert!(stddev(&[], 0, 1.0).is_empty());
    }

    #[test]
    fn stddev_crosses_the_reseed_interval_without_a_discontinuity() {
        // 20_000 bars with timeperiod 12 drives 19_988 advances, so the exact
        // reseed at every RESEED_INTERVAL = 8192 advances fires twice.
        let data = noisy_series(20_000, 250.0, 6.0);
        let p = 12;
        let got = stddev(&data, p, 1.0);
        let want = reference_stddev(&data, p, 1.0);
        assert_close(&got, &want, 1e-11, "reseed stddev");
        assert_close(
            &var(&data, p, 1.0),
            &reference_var(&data, p, 1.0),
            1e-11,
            "reseed var",
        );

        // No step change at either reseed boundary: the reseed replaces the
        // accumulator with an exact recompute, so neighbouring outputs must
        // stay as close to each other as the reference's neighbours are.
        for advance in [8192usize, 16_384] {
            let i = p - 1 + advance;
            let jump = (got[i] - got[i - 1]).abs();
            let reference_jump = (want[i] - want[i - 1]).abs();
            assert!(
                (jump - reference_jump).abs() < 1e-10,
                "discontinuity at reseed advance {advance}: jump={jump} reference={reference_jump}"
            );
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
    fn dtw_nan_in_input_propagates() {
        // NaN in either input must propagate to the distance (IEEE 754 semantics).
        let a = vec![1.0, 2.0, f64::NAN, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        assert!(dtw_distance(&a, &b, None).is_nan());
        assert!(dtw_distance(&b, &a, None).is_nan());
    }

    #[test]
    fn dtw_is_symmetric() {
        let a = vec![1.0, 4.0, 2.0, 8.0, 3.0, 6.0, 5.0];
        let b = vec![2.0, 3.0, 7.0, 4.0, 5.0, 1.0, 9.0];
        let d_ab = dtw_distance(&a, &b, None);
        let d_ba = dtw_distance(&b, &a, None);
        assert!((d_ab - d_ba).abs() < 1e-12);
    }

    #[test]
    fn dtw_path_length_bounded() {
        // A valid warp path has length between max(n, m) and n + m - 1.
        let a: Vec<f64> = (0..7).map(|x| x as f64).collect();
        let b: Vec<f64> = (0..10).map(|x| (x as f64).sin()).collect();
        let (_, path) = dtw_path(&a, &b, None);
        let n = a.len();
        let m = b.len();
        assert!(path.len() >= n.max(m));
        assert!(path.len() < n + m);
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

    // -----------------------------------------------------------------
    // Reseed-policy regression tests.
    //
    // Each case below returned a *wrong* value before the dynamic-range and
    // non-finite-accumulator reseed triggers were added to
    // `rolling::RollingVariance`, and every one of them is built entirely from
    // finite inputs. The recorded "was" figures are the pre-fix outputs.
    // -----------------------------------------------------------------

    /// Exact two-pass population variance, independent of the kernel.
    fn exact_population_var(window: &[f64]) -> f64 {
        let p = window.len() as f64;
        let mean = window.iter().sum::<f64>() / p;
        window.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / p
    }

    /// Assert `var`/`stddev` agree with the exact two-pass form on every
    /// full window of `data`, and that `stddev == sqrt(var)`.
    fn assert_matches_exact(data: &[f64], p: usize, rtol: f64, label: &str) {
        let got_var = var(data, p, 1.0);
        let got_sd = stddev(data, p, 1.0);
        for i in (p - 1)..data.len() {
            let want = exact_population_var(&data[i + 1 - p..=i]);
            if want.is_nan() {
                assert!(
                    got_var[i].is_nan(),
                    "{label} i={i}: want NaN, got {}",
                    got_var[i]
                );
                continue;
            }
            assert!(
                !got_var[i].is_nan(),
                "{label} i={i}: NaN from a window whose two-pass variance is {want}"
            );
            if want.is_infinite() {
                assert!(
                    got_var[i].is_infinite() && got_var[i].is_sign_positive(),
                    "{label} i={i}: want +inf, got {}",
                    got_var[i]
                );
                continue;
            }
            let tol = rtol * want.abs() + f64::MIN_POSITIVE;
            assert!(
                (got_var[i] - want).abs() <= tol,
                "{label} i={i}: var got={} want={want} rel={}",
                got_var[i],
                (got_var[i] - want).abs() / want.abs()
            );
            let want_sd = want.sqrt();
            assert!(
                (got_sd[i] - want_sd).abs() <= rtol * want_sd + f64::MIN_POSITIVE,
                "{label} i={i}: stddev got={} want={want_sd}",
                got_sd[i]
            );
        }
    }

    /// Manifestation 1: `m2` overflows to `+inf` on the seed window, the next
    /// slide adds `-inf`, and `m2` becomes `NaN`. The `m2 < 0.0` clamp does not
    /// catch it (`NaN < 0.0` is false), so before the non-finite-accumulator
    /// trigger every later window — over entirely ordinary prices — returned
    /// `NaN`. Was: `NaN` at i=4,5,6; want 0.16666666666666666.
    #[test]
    fn overflowing_m2_does_not_poison_later_windows() {
        let data = [1e300, -1e300, 100.0, 100.5, 101.0, 101.5, 102.0];
        assert_matches_exact(&data, 3, 1e-12, "m2 overflow");

        // Explicitly: the overflowing window reports `+inf` (as the two-pass
        // form does), and recovery on the first clean window is immediate.
        let got = var(&data, 3, 1.0);
        assert!(got[2].is_infinite() && got[3].is_infinite());
        for i in 4..data.len() {
            assert!(
                (got[i] - 1.0 / 6.0).abs() < 1e-15,
                "i={i} got={} want=1/6",
                got[i]
            );
        }

        // Extending the series must not reintroduce the poisoning: 26 further
        // ordinary bars, all of which returned `NaN` before the fix.
        let mut long = vec![1e300, -1e300];
        long.extend((0..30).map(|k| 100.0 + 0.5 * k as f64));
        assert_matches_exact(&long, 3, 1e-12, "m2 overflow extended");
        assert!(var(&long, 3, 1.0)[4..].iter().all(|v| v.is_finite()));
    }

    /// Manifestation 2: a lone `1e8` spike on a price-100 series. Once it left
    /// the window the accumulator held pure rounding residue, which the
    /// `m2 < 0.0` clamp rounded up to a confident hard zero. Was:
    /// `var = 0.0`, `stddev = 0.0` at i=6,7,8; want ~6.667e-3.
    #[test]
    fn a_finite_spike_leaving_the_window_does_not_pin_variance_to_zero() {
        let data = [100.0, 100.5, 101.0, 1e8, 100.2, 100.3, 100.4, 100.5, 100.6];
        assert_matches_exact(&data, 3, 1e-9, "1e8 spike");

        let got_var = var(&data, 3, 1.0);
        let got_sd = stddev(&data, 3, 1.0);
        for i in 6..data.len() {
            assert!(got_var[i] > 0.0, "var[{i}] pinned to {}", got_var[i]);
            assert!(got_sd[i] > 0.0, "stddev[{i}] pinned to {}", got_sd[i]);
        }

        // The pre-fix error was non-monotone in the spike magnitude, which is
        // the signature of precision loss rather than a logic error. Sweep it.
        for exp in 4..=20 {
            let spike = 10f64.powi(exp);
            let mut d = data.to_vec();
            d[3] = spike;
            assert_matches_exact(&d, 3, 1e-9, &format!("spike 1e{exp}"));
        }
    }

    /// Manifestation 3: a subnormal / huge-exponent mix. Was: `5.635e270` at
    /// i=4 against an exact `1.742e202` — off by 68 orders of magnitude.
    #[test]
    fn subnormal_and_huge_exponent_mix_stays_exact() {
        let data = [1.39e-309, 3.05e143, 2.80e101, -7.34e83, 2.20e-106];
        assert_matches_exact(&data, 3, 1e-12, "exponent mix");
        let got = var(&data, 3, 1.0);
        let want = exact_population_var(&data[2..5]);
        assert!(
            (got[4] - want).abs() <= 1e-12 * want,
            "i=4 got={} want={want}",
            got[4]
        );
    }

    /// A spike that transits a *long* window, followed by many ordinary bars —
    /// the manifestation-2 shape at a period where the pre-fix damage would
    /// have persisted for the rest of the series.
    #[test]
    fn spike_transiting_a_long_window_then_ordinary_bars() {
        let p = 30;
        let mut data = noisy_series(400, 100.0, 2.0);
        data[50] = 1e12;
        assert_matches_exact(&data, p, 1e-8, "long-window spike");
        // Everything from the first spike-free window onwards must agree with
        // the two-pass reference to the crate's own tight tolerance.
        let got = var(&data, p, 1.0);
        let want = reference_var(&data, p, 1.0);
        for i in (50 + p)..data.len() {
            assert!(
                (got[i] - want[i]).abs() <= 1e-11 * want[i] + ABS_FLOOR,
                "post-spike i={i} got={} want={}",
                got[i],
                want[i]
            );
        }
    }

    /// Values near `1e-161` have squared deviations in the **subnormal** range,
    /// where `m2` carries only a handful of significant bits and any rolling
    /// recurrence drifts by percent-level *relative* error per step — with a
    /// dynamic range of 1, so no magnitude-ratio trigger can see it. This was
    /// the second `fuzz_stddev` finding after the three reproducers above.
    #[test]
    fn subnormal_second_moment_stays_exact() {
        let base = 1.7e-161;
        let data: Vec<f64> = (0..64)
            .map(|i| base * (1.0 + 0.25 * ((i % 5) as f64)))
            .collect();
        let probe = exact_population_var(&data[..2]);
        assert!(
            probe > 0.0 && probe < f64::MIN_POSITIVE,
            "premise broken: m2 is {probe}, not subnormal"
        );
        for &p in &[2usize, 3, 7, 14] {
            let got = var(&data, p, 1.0);
            for i in (p - 1)..data.len() {
                let want = exact_population_var(&data[i + 1 - p..=i]);
                assert_eq!(
                    got[i].to_bits(),
                    want.to_bits(),
                    "p={p} i={i}: got={:e} want={want:e}",
                    got[i]
                );
            }
        }
    }
}
