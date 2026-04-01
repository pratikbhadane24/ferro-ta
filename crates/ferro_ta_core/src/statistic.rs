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
}
