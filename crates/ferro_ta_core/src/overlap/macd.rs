//! Moving Average Convergence/Divergence.

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
