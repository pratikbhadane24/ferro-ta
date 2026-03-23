//! Overlap studies — moving averages and trend indicators.
//!
//! All functions return a `Vec<f64>` of the same length as the input.
//! Leading values are `f64::NAN` for the warm-up period.

/// Simple Moving Average over `timeperiod` bars.
///
/// # Edge Cases
/// Returns all-NaN when `timeperiod < 1` or `close.len() < timeperiod`.
pub fn sma(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    sma_into(close, timeperiod, &mut result, 0);
    result
}

/// Simple Moving Average written directly into `dest` starting at `dest_offset`.
/// Leaves values before `dest_offset + timeperiod - 1` untouched (e.g. they can be NaN).
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

/// Exponential Moving Average — seeded with SMA of first `timeperiod` bars.
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

/// Weighted Moving Average — O(n) incremental algorithm using running weighted sum.
///
/// Recurrence: `T[i] = T[i-1] + n*close[i] - S[i-1]`
/// where `S[i]` is the rolling sum over `timeperiod` bars.
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

/// Bollinger Bands — returns `(upper, middle, lower)`.
///
/// Middle is SMA; bands are `± nbdev * stddev`.
/// Uses O(n) sliding `sum` and `sum_sq` windows for mean and variance.
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

    // Seed sliding sums for the first window.
    #[cfg(feature = "simd")]
    let (mut sum, mut sum_sq) = {
        use wide::f64x4;
        let p_data = &close[..timeperiod];
        let mut sum_simd = f64x4::splat(0.0);
        let mut sq_simd = f64x4::splat(0.0);
        let mut chunks = p_data.chunks_exact(4);
        for chunk in &mut chunks {
            let vals = f64x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]);
            sum_simd += vals;
            sq_simd += vals * vals;
        }
        let s_arr = sum_simd.to_array();
        let sq_arr = sq_simd.to_array();
        let mut sum = s_arr[0] + s_arr[1] + s_arr[2] + s_arr[3];
        let mut sum_sq = sq_arr[0] + sq_arr[1] + sq_arr[2] + sq_arr[3];
        for &v in chunks.remainder() {
            sum += v;
            sum_sq += v * v;
        }
        (sum, sum_sq)
    };

    #[cfg(not(feature = "simd"))]
    let (mut sum, mut sum_sq) = {
        let s: f64 = close[..timeperiod].iter().sum();
        let sq: f64 = close[..timeperiod].iter().map(|&x| x * x).sum();
        (s, sq)
    };

    let mean = sum / p;
    let var = (sum_sq / p - mean * mean).max(0.0);
    let std = var.sqrt();
    middle[timeperiod - 1] = mean;
    upper[timeperiod - 1] = mean + nbdevup * std;
    lower[timeperiod - 1] = mean - nbdevdn * std;

    let mut i = timeperiod;
    while i + 1 < n {
        let old0 = close[i - timeperiod];
        sum += close[i] - old0;
        sum_sq += close[i] * close[i] - old0 * old0;
        let mean = sum / p;
        let var = (sum_sq / p - mean * mean).max(0.0);
        let std = var.sqrt();
        middle[i] = mean;
        upper[i] = mean + nbdevup * std;
        lower[i] = mean - nbdevdn * std;

        let old1 = close[i + 1 - timeperiod];
        sum += close[i + 1] - old1;
        sum_sq += close[i + 1] * close[i + 1] - old1 * old1;
        let mean1 = sum / p;
        let var1 = (sum_sq / p - mean1 * mean1).max(0.0);
        let std1 = var1.sqrt();
        middle[i + 1] = mean1;
        upper[i + 1] = mean1 + nbdevup * std1;
        lower[i + 1] = mean1 - nbdevdn * std1;

        i += 2;
    }
    if i < n {
        let old = close[i - timeperiod];
        sum += close[i] - old;
        sum_sq += close[i] * close[i] - old * old;
        let mean = sum / p;
        let var = (sum_sq / p - mean * mean).max(0.0);
        let std = var.sqrt();
        middle[i] = mean;
        upper[i] = mean + nbdevup * std;
        lower[i] = mean - nbdevdn * std;
    }
    (upper, middle, lower)
}

/// MACD — EMA(fastperiod) minus EMA(slowperiod), signal = EMA(macd, signalperiod).
///
/// Returns `(macd_line, signal_line, histogram)`, each of length `n`.
/// Leading values are `NaN` during warmup.
/// `fastperiod` must be less than `slowperiod`.
///
/// Fast and slow EMAs are computed in a **single combined loop** to minimise
/// memory round-trips, then the signal EMA is computed in a second pass.
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
