//! Overlap studies — moving averages and trend indicators.
//!
//! All functions return a `Vec<f64>` of the same length as the input.
//! Leading values are `f64::NAN` for the warm-up period.

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

    #[cfg(feature = "simd")]
    let window_sum_init = {
        use wide::f64x4;
        let p_data = &src[..timeperiod];
        let mut sum = f64x4::splat(0.0);
        let mut chunks = p_data.chunks_exact(4);
        for chunk in &mut chunks {
            sum += f64x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        let arr = sum.to_array();
        let mut total = arr[0] + arr[1] + arr[2] + arr[3];
        for &v in chunks.remainder() {
            total += v;
        }
        total
    };

    #[cfg(not(feature = "simd"))]
    let window_sum_init: f64 = src[..timeperiod].iter().sum();

    let mut window_sum = window_sum_init;
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

/// Compute the Exponential Moving Average (EMA).
///
/// The EMA is seeded with the SMA of the first `timeperiod` bars and uses
/// a smoothing factor of `k = 2 / (timeperiod + 1)`. Returns a `Vec<f64>`
/// of the same length as `close`; the first `timeperiod - 1` values are `NaN`.
///
/// # Arguments
/// * `close` - Price series.
/// * `timeperiod` - Lookback period (must be >= 1).
pub fn ema(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 || n < timeperiod {
        return result;
    }
    let k = 2.0 / (timeperiod as f64 + 1.0);
    let seed: f64 = close[..timeperiod].iter().sum::<f64>() / timeperiod as f64;
    result[timeperiod - 1] = seed;
    for i in timeperiod..n {
        result[i] = (result[i - 1] * (1.0 - k)).mul_add(1.0, close[i] * k);
    }
    result
}

/// Compute the Weighted Moving Average (WMA).
///
/// Assigns linearly increasing weights (1, 2, ..., timeperiod) to the window.
/// Uses an O(n) incremental recurrence to avoid recomputing weights each bar.
/// Returns a `Vec<f64>` of length `n`; the first `timeperiod - 1` values are `NaN`.
///
/// # Arguments
/// * `close` - Price series.
/// * `timeperiod` - Rolling window size (must be >= 1).
pub fn wma(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 || n < timeperiod {
        return result;
    }
    let denom: f64 = (timeperiod * (timeperiod + 1) / 2) as f64;
    let p = timeperiod as f64;

    // Seed: compute T and S for the first window.
    #[cfg(feature = "simd")]
    let (mut t, mut s) = {
        use wide::f64x4;
        let p_data = &close[..timeperiod];
        let mut t_simd = f64x4::splat(0.0);
        let mut s_simd = f64x4::splat(0.0);
        let mut chunks = p_data.chunks_exact(4);
        let mut idx = 1.0;
        let step = f64x4::new([0.0, 1.0, 2.0, 3.0]);

        for chunk in &mut chunks {
            let vals = f64x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let mults = f64x4::splat(idx) + step;
            t_simd += vals * mults;
            s_simd += vals;
            idx += 4.0;
        }
        let t_arr = t_simd.to_array();
        let s_arr = s_simd.to_array();
        let mut t = t_arr[0] + t_arr[1] + t_arr[2] + t_arr[3];
        let mut s = s_arr[0] + s_arr[1] + s_arr[2] + s_arr[3];
        for &v in chunks.remainder() {
            t += v * idx;
            s += v;
            idx += 1.0;
        }
        (t, s)
    };

    #[cfg(not(feature = "simd"))]
    let (mut t, mut s) = {
        let t_val: f64 = close[..timeperiod]
            .iter()
            .enumerate()
            .map(|(k, &v)| v * (k + 1) as f64)
            .sum();
        let s_val: f64 = close[..timeperiod].iter().sum();
        (t_val, s_val)
    };

    result[timeperiod - 1] = t / denom;

    let mut i = timeperiod;
    while i + 1 < n {
        t += p * close[i] - s;
        s += close[i] - close[i - timeperiod];
        result[i] = t / denom;

        t += p * close[i + 1] - s;
        s += close[i + 1] - close[i + 1 - timeperiod];
        result[i + 1] = t / denom;

        i += 2;
    }
    if i < n {
        t += p * close[i] - s;
        result[i] = t / denom;
    }
    result
}

/// Compute Bollinger Bands, returning `(upper, middle, lower)`.
///
/// The middle band is the SMA; upper and lower bands are offset by
/// `nbdevup` and `nbdevdn` standard deviations respectively. Uses
/// Welford's rolling algorithm for numerically stable variance in O(n).
///
/// # Arguments
/// * `close` - Price series.
/// * `timeperiod` - SMA / standard deviation window (must be >= 1).
/// * `nbdevup` - Number of standard deviations above the mean for the upper band.
/// * `nbdevdn` - Number of standard deviations below the mean for the lower band.
///
/// # Returns
/// `(upper, middle, lower)` -- each `Vec<f64>` of length `n`. The first
/// `timeperiod - 1` values in each vector are `NaN`.
///
/// ## Welford's rolling algorithm
///
/// We maintain `mean` and `m2` (sum of squared deviations from the current
/// mean) across a sliding window of size `N`.  When a new value `x_new`
/// replaces an old value `x_old` (window size stays constant):
///
/// ```text
/// delta     = x_new - x_old
/// old_mean  = mean
/// mean     += delta / N
/// m2       += delta * ((x_new - mean) + (x_old - old_mean))
///
/// variance  = m2 / N               // population variance
/// stddev    = sqrt(variance)
/// ```
///
/// The initial window is seeded using the standard (non-rolling) Welford
/// incremental algorithm.
///
/// This avoids the catastrophic cancellation inherent in the naïve
/// `Σx²/N − mean²` formula when values are large but close together.
pub fn bbands(
    close: &[f64],
    timeperiod: usize,
    nbdevup: f64,
    nbdevdn: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = close.len();
    let nan = vec![f64::NAN; n];
    if timeperiod < 1 || n < timeperiod {
        return (nan.clone(), nan.clone(), nan);
    }
    let mut upper = vec![f64::NAN; n];
    let mut middle = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];
    let p = timeperiod as f64;

    // --- Seed: build initial mean and m2 for the first window using
    //     Welford's incremental (non-rolling) algorithm. ---
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;
    for (k, &x) in close[..timeperiod].iter().enumerate() {
        let count = (k + 1) as f64;
        let delta = x - mean;
        mean += delta / count;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }

    let var = (m2 / p).max(0.0);
    let std = var.sqrt();
    middle[timeperiod - 1] = mean;
    upper[timeperiod - 1] = mean + nbdevup * std;
    lower[timeperiod - 1] = mean - nbdevdn * std;

    // --- Rolling phase: slide the window one element at a time,
    //     removing the oldest value and adding the newest. ---

    /// Inline helper: replace `x_old` with `x_new` in the Welford accumulator
    /// (constant window size `p`), then write band values into the output slots.
    ///
    /// Combined rolling Welford update (window size stays constant at N):
    ///
    /// ```text
    /// delta     = x_new - x_old
    /// old_mean  = mean
    /// mean     += delta / N
    /// m2       += delta * ((x_new - mean) + (x_old - old_mean))
    /// ```
    ///
    /// This is algebraically equivalent to removing `x_old` and adding `x_new`
    /// in two separate Welford steps, but avoids the intermediate N-1 state.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn welford_step(
        x_old: f64,
        x_new: f64,
        mean: &mut f64,
        m2: &mut f64,
        p: f64,
        nbdevup: f64,
        nbdevdn: f64,
        upper: &mut f64,
        middle: &mut f64,
        lower: &mut f64,
    ) {
        let delta = x_new - x_old;
        let old_mean = *mean;
        *mean += delta / p;
        // Update m2 using both old and new deviations.
        *m2 += delta * ((x_new - *mean) + (x_old - old_mean));

        // Clamp m2 to zero to guard against floating-point drift.
        if *m2 < 0.0 {
            *m2 = 0.0;
        }

        let var = *m2 / p;
        let std = var.sqrt();
        *middle = *mean;
        *upper = *mean + nbdevup * std;
        *lower = *mean - nbdevdn * std;
    }

    // Process two iterations at a time (loop unrolling) for throughput.
    let mut i = timeperiod;
    while i + 1 < n {
        welford_step(
            close[i - timeperiod],
            close[i],
            &mut mean,
            &mut m2,
            p,
            nbdevup,
            nbdevdn,
            &mut upper[i],
            &mut middle[i],
            &mut lower[i],
        );
        welford_step(
            close[i + 1 - timeperiod],
            close[i + 1],
            &mut mean,
            &mut m2,
            p,
            nbdevup,
            nbdevdn,
            &mut upper[i + 1],
            &mut middle[i + 1],
            &mut lower[i + 1],
        );
        i += 2;
    }
    if i < n {
        welford_step(
            close[i - timeperiod],
            close[i],
            &mut mean,
            &mut m2,
            p,
            nbdevup,
            nbdevdn,
            &mut upper[i],
            &mut middle[i],
            &mut lower[i],
        );
    }

    (upper, middle, lower)
}

/// Compute the Moving Average Convergence/Divergence (MACD).
///
/// `MACD = EMA(close, fastperiod) - EMA(close, slowperiod)`.
/// The signal line is `EMA(macd, signalperiod)` and the histogram is
/// `macd - signal`. TA-Lib compatible: leading values are `NaN` up to
/// the point where all three outputs are valid.
///
/// # Arguments
/// * `close` - Price series.
/// * `fastperiod` - Fast EMA period (must be < `slowperiod`).
/// * `slowperiod` - Slow EMA period.
/// * `signalperiod` - Signal line EMA period.
///
/// # Returns
/// `(macd_line, signal_line, histogram)` -- each `Vec<f64>` of length `n`.
pub fn macd(
    close: &[f64],
    fastperiod: usize,
    slowperiod: usize,
    signalperiod: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = close.len();
    let nan_vec = || vec![f64::NAN; n];
    if fastperiod < 1 || slowperiod < 1 || signalperiod < 1 || fastperiod >= slowperiod {
        return (nan_vec(), nan_vec(), nan_vec());
    }
    if n < slowperiod {
        return (nan_vec(), nan_vec(), nan_vec());
    }

    let kf = 2.0 / (fastperiod as f64 + 1.0);
    let ks = 2.0 / (slowperiod as f64 + 1.0);

    // Seed fast EMA from SMA of first fastperiod bars.
    let mut fast_val: f64 = close[..fastperiod].iter().sum::<f64>() / fastperiod as f64;
    // Seed slow EMA from SMA of first slowperiod bars.
    let mut slow_val: f64 = close[..slowperiod].iter().sum::<f64>() / slowperiod as f64;

    let mut macd_line = nan_vec();

    // From fastperiod-1 to slowperiod-2: advance fast EMA only.
    for &price in close.iter().take(slowperiod - 1).skip(fastperiod) {
        fast_val = price * kf + fast_val * (1.0 - kf);
    }

    // From fastperiod to slowperiod-1: advance fastEMA and compute initial MACD at slowperiod-1
    // Actually, fast_val currently holds the value for `slowperiod - 2` after `take(slowperiod - 1)`
    // So we apply it for `slowperiod - 1`.
    fast_val = close[slowperiod - 1] * kf + fast_val * (1.0 - kf);
    macd_line[slowperiod - 1] = fast_val - slow_val;
    for i in slowperiod..n {
        fast_val = close[i] * kf + fast_val * (1.0 - kf);
        slow_val = close[i] * ks + slow_val * (1.0 - ks);
        macd_line[i] = fast_val - slow_val;
    }

    // Signal line: EMA of macd_line, seeded from the first valid macd value.
    // The signal line starts producing values after slowperiod - 1 + signalperiod - 1 bars.
    let sig_start = slowperiod - 1 + signalperiod - 1;
    let mut signal_line = nan_vec();
    let mut histogram = nan_vec();

    if sig_start >= n {
        // If we can't compute signal, TA-Lib clears MACD!
        for v in macd_line.iter_mut().take(n) {
            *v = f64::NAN;
        }
        return (macd_line, signal_line, histogram);
    }

    let ksig = 2.0 / (signalperiod as f64 + 1.0);
    // Seed signal EMA with SMA of the first signalperiod macd values.
    let sig_seed: f64 = macd_line[(slowperiod - 1)..(slowperiod - 1 + signalperiod)]
        .iter()
        .sum::<f64>()
        / signalperiod as f64;
    signal_line[sig_start] = sig_seed;
    histogram[sig_start] = macd_line[sig_start] - signal_line[sig_start];

    for i in (sig_start + 1)..n {
        signal_line[i] = macd_line[i] * ksig + signal_line[i - 1] * (1.0 - ksig);
    }
    for i in (sig_start + 1)..n {
        histogram[i] = macd_line[i] - signal_line[i];
    }

    // TA-Lib pads the MACD line itself with NaNs up to `sig_start`!
    for v in macd_line.iter_mut().take(sig_start) {
        *v = f64::NAN;
    }

    (macd_line, signal_line, histogram)
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

    #[test]
    fn ema_basic() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ema(&prices, 3);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-10); // seed = SMA(3)
    }

    #[test]
    fn wma_basic() {
        let prices = vec![1.0, 2.0, 3.0];
        let result = wma(&prices, 3);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // weights: 1, 2, 3; denom 6 => (1*1 + 2*2 + 3*3)/6 = 14/6
        assert!((result[2] - 14.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn bbands_basic() {
        let prices = vec![2.0, 2.0, 2.0, 2.0, 2.0];
        let (upper, middle, lower) = bbands(&prices, 3, 2.0, 2.0);
        assert!((middle[2] - 2.0).abs() < 1e-10);
        assert!((upper[2] - 2.0).abs() < 1e-10); // std = 0
        assert!((lower[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn bbands_varying_prices() {
        // Verify against hand-computed values for a small window.
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (upper, middle, lower) = bbands(&prices, 3, 2.0, 2.0);

        // First two values should be NaN (warmup).
        assert!(middle[0].is_nan());
        assert!(middle[1].is_nan());

        // Window [1,2,3]: mean = 2.0, pop_var = 2/3, std = sqrt(2/3)
        let expected_mean = 2.0;
        let expected_std = (2.0_f64 / 3.0).sqrt();
        assert!((middle[2] - expected_mean).abs() < 1e-10);
        assert!((upper[2] - (expected_mean + 2.0 * expected_std)).abs() < 1e-10);
        assert!((lower[2] - (expected_mean - 2.0 * expected_std)).abs() < 1e-10);

        // Window [2,3,4]: mean = 3.0, pop_var = 2/3, std = sqrt(2/3)
        assert!((middle[3] - 3.0).abs() < 1e-10);
        assert!((upper[3] - (3.0 + 2.0 * expected_std)).abs() < 1e-10);

        // Window [3,4,5]: mean = 4.0, pop_var = 2/3, std = sqrt(2/3)
        assert!((middle[4] - 4.0).abs() < 1e-10);
        assert!((upper[4] - (4.0 + 2.0 * expected_std)).abs() < 1e-10);
    }

    #[test]
    fn bbands_numerical_stability() {
        // Large offset with tiny variation — this is where the naïve sum_sq
        // formula suffers from catastrophic cancellation.
        let base = 1e12;
        let prices: Vec<f64> = (0..100).map(|i| base + (i as f64) * 0.01).collect();
        let (upper, middle, lower) = bbands(&prices, 20, 2.0, 2.0);

        // Check that middle band matches SMA.
        for i in 19..100 {
            let window = &prices[i - 19..=i];
            let expected_mean: f64 = window.iter().sum::<f64>() / 20.0;
            assert!(
                (middle[i] - expected_mean).abs() < 1e-4,
                "mean mismatch at {i}: got {} expected {}",
                middle[i],
                expected_mean,
            );
            // Bands should be above/below middle.
            assert!(upper[i] >= middle[i]);
            assert!(lower[i] <= middle[i]);
        }
    }

    #[test]
    fn bbands_edge_cases() {
        // timeperiod == 1: every bar should have std = 0, bands == price.
        let prices = vec![10.0, 20.0, 30.0];
        let (upper, middle, lower) = bbands(&prices, 1, 2.0, 2.0);
        for i in 0..3 {
            assert!((middle[i] - prices[i]).abs() < 1e-10);
            assert!((upper[i] - prices[i]).abs() < 1e-10);
            assert!((lower[i] - prices[i]).abs() < 1e-10);
        }

        // Input shorter than timeperiod: all NaN.
        let (u, m, l) = bbands(&[1.0, 2.0], 5, 2.0, 2.0);
        assert!(u.iter().all(|v| v.is_nan()));
        assert!(m.iter().all(|v| v.is_nan()));
        assert!(l.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn macd_basic() {
        // 40 bars of linearly increasing prices — MACD line should converge
        let prices: Vec<f64> = (1..=40).map(|i| i as f64).collect();
        let (macd_line, signal_line, histogram) = macd(&prices, 3, 5, 2);
        // TA-Lib pads MACD line with NaN up to sig_start = slowperiod-1 + signalperiod-1 = 5
        for i in 0..5 {
            assert!(macd_line[i].is_nan(), "expected NaN at {i}");
        }
        // First valid macd bar is at index 5 (sig_start)
        assert!(!macd_line[5].is_nan());
        // First valid signal bar is at index 5
        assert!(!signal_line[5].is_nan());
        // histogram = macd - signal
        assert!((histogram[5] - (macd_line[5] - signal_line[5])).abs() < 1e-10);
    }

    #[test]
    fn macd_invalid_params() {
        let prices = vec![1.0; 50];
        // fastperiod >= slowperiod should return all-NaN
        let (m, s, h) = macd(&prices, 5, 3, 9);
        assert!(m.iter().all(|v| v.is_nan()));
        assert!(s.iter().all(|v| v.is_nan()));
        assert!(h.iter().all(|v| v.is_nan()));
    }
}
