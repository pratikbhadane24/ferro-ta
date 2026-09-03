//! Simple Moving Average.

/// Compute the Simple Moving Average (SMA) over a rolling window.
///
/// Returns a `Vec<f64>` of the same length as `close`. The first
/// `timeperiod - 1` values are `NaN` (warmup period).
///
/// # Arguments
/// * `close` - Price series.
/// * `timeperiod` - Rolling window size (must be >= 1).
///
/// # Edge Cases
/// Returns all-NaN when `timeperiod < 1` or `close.len() < timeperiod`.
pub fn sma(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    sma_into(close, timeperiod, &mut result, 0);
    result
}

/// Write a Simple Moving Average directly into a pre-allocated buffer.
///
/// Values before `dest_offset + timeperiod - 1` are left untouched.
/// This avoids an intermediate allocation when composing indicators
/// (e.g., Stochastic slow %K and slow %D).
///
/// # Arguments
/// * `src` - Input price series.
/// * `timeperiod` - Rolling window size (must be >= 1).
/// * `dest` - Output buffer (must be at least `dest_offset + src.len()` long).
/// * `dest_offset` - Starting index in `dest` to write results.
pub fn sma_into(src: &[f64], timeperiod: usize, dest: &mut [f64], dest_offset: usize) {
    let n = src.len();
    if timeperiod < 1 || n < timeperiod {
        return;
    }

    // Seed the rolling window with a runtime-dispatched reduction. The O(n)
    // streaming recurrence below is inherently sequential, so SIMD only ever
    // applies to this initial window sum.
    let mut window_sum = crate::simd::sum(&src[..timeperiod]);
    let tp_f64 = timeperiod as f64;
    dest[dest_offset + timeperiod - 1] = window_sum / tp_f64;

    let mut i = timeperiod;
    while i + 1 < n {
        let old0 = src[i - timeperiod];
        let new0 = src[i];
        window_sum += new0 - old0;
        dest[dest_offset + i] = window_sum / tp_f64;

        let old1 = src[i + 1 - timeperiod];
        let new1 = src[i + 1];
        window_sum += new1 - old1;
        dest[dest_offset + i + 1] = window_sum / tp_f64;

        i += 2;
    }
    if i < n {
        window_sum += src[i] - src[i - timeperiod];
        dest[dest_offset + i] = window_sum / tp_f64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sma_basic() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&prices, 3);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
        assert!((result[4] - 4.0).abs() < 1e-10);
    }
}
