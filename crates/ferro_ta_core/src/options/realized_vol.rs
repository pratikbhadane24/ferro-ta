//! Historical (realized) volatility estimators and volatility cone.

/// Rolling close-to-close realized volatility.
///
/// Returns a `Vec<f64>` of the same length as `close`. The first `window` values
/// are NaN (we need `window` log-returns, which require `window+1` prices, so the
/// first valid output sits at index `window`).
///
/// Annualization: `sqrt(sum(r²) / window * trading_days)`.
pub fn close_to_close_vol(close: &[f64], window: usize, trading_days: f64) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if window == 0 || n <= window {
        return out;
    }

    // Precompute log-returns; returns[i] = ln(close[i+1] / close[i])
    let mut returns = vec![f64::NAN; n - 1];
    for i in 0..(n - 1) {
        if close[i] > 0.0 && close[i + 1] > 0.0 {
            returns[i] = (close[i + 1] / close[i]).ln();
        }
    }

    // Rolling sum of squared returns over `window` bars.
    // The output at position `end` (in the original close array) uses
    // returns[end-window .. end-1], i.e. `window` returns.
    for end in window..n {
        let slice = &returns[(end - window)..end];
        let sum_sq: f64 = slice.iter().map(|&r| r * r).sum();
        let var = sum_sq / window as f64 * trading_days;
        out[end] = if var >= 0.0 { var.sqrt() } else { f64::NAN };
    }
    out
}

/// Rolling Parkinson high-low realized volatility estimator.
///
/// Returns a `Vec<f64>` of the same length as `high`. The first `window-1` values
/// are NaN.
#[allow(clippy::needless_range_loop)]
pub fn parkinson_vol(high: &[f64], low: &[f64], window: usize, trading_days: f64) -> Vec<f64> {
    let n = high.len();
    let mut out = vec![f64::NAN; n];
    if window == 0 || n < window || low.len() != n {
        return out;
    }

    let factor = 1.0 / (4.0 * 2_f64.ln());

    for end in (window - 1)..n {
        let start = end + 1 - window;
        let mut sum_sq = 0.0;
        let mut valid = true;
        for i in start..=end {
            if high[i] <= 0.0 || low[i] <= 0.0 || !high[i].is_finite() || !low[i].is_finite() {
                valid = false;
                break;
            }
            let u = (high[i] / low[i]).ln();
            sum_sq += u * u;
        }
        if valid {
            let var = factor * sum_sq / window as f64 * trading_days;
            out[end] = if var >= 0.0 { var.sqrt() } else { f64::NAN };
        }
    }
    out
}

/// Rolling Garman-Klass OHLC realized volatility estimator.
///
/// Returns a `Vec<f64>` of the same length as the inputs. The first `window-1`
/// values are NaN. All four slices must have the same length.
pub fn garman_klass_vol(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    window: usize,
    trading_days: f64,
) -> Vec<f64> {
    let n = open.len();
    let mut out = vec![f64::NAN; n];
    if window == 0 || n < window || high.len() != n || low.len() != n || close.len() != n {
        return out;
    }

    let ln2 = 2_f64.ln();

    // Precompute per-bar GK contributions.
    let mut gk = vec![f64::NAN; n];
    for i in 0..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        if o > 0.0
            && h > 0.0
            && l > 0.0
            && c > 0.0
            && o.is_finite()
            && h.is_finite()
            && l.is_finite()
            && c.is_finite()
        {
            let u = (h / o).ln();
            let d = (l / o).ln();
            let ci = (c / o).ln();
            gk[i] = 0.5 * (u - d).powi(2) - (2.0 * ln2 - 1.0) * ci * ci;
        }
    }

    for end in (window - 1)..n {
        let start = end + 1 - window;
        let slice = &gk[start..=end];
        if slice.iter().all(|v| v.is_finite()) {
            let sum: f64 = slice.iter().sum();
            let var = sum / window as f64 * trading_days;
            out[end] = if var >= 0.0 { var.sqrt() } else { f64::NAN };
        }
    }
    out
}

/// Compute the Rogers-Satchell per-bar variance contribution.
fn rs_bar(open: f64, high: f64, low: f64, close: f64) -> f64 {
    let u = (high / close).ln();
    let d = (low / close).ln();
    let uo = (high / open).ln();
    let do_ = (low / open).ln();
    u * uo + d * do_
}

/// Rolling Rogers-Satchell OHLC realized volatility estimator.
///
/// Returns a `Vec<f64>` of the same length as the inputs. The first `window-1`
/// values are NaN. All four slices must have the same length.
pub fn rogers_satchell_vol(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    window: usize,
    trading_days: f64,
) -> Vec<f64> {
    let n = open.len();
    let mut out = vec![f64::NAN; n];
    if window == 0 || n < window || high.len() != n || low.len() != n || close.len() != n {
        return out;
    }

    // Precompute per-bar RS contributions.
    let mut rs = vec![f64::NAN; n];
    for i in 0..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        if o > 0.0
            && h > 0.0
            && l > 0.0
            && c > 0.0
            && o.is_finite()
            && h.is_finite()
            && l.is_finite()
            && c.is_finite()
        {
            rs[i] = rs_bar(o, h, l, c);
        }
    }

    for end in (window - 1)..n {
        let start = end + 1 - window;
        let slice = &rs[start..=end];
        if slice.iter().all(|v| v.is_finite()) {
            let sum: f64 = slice.iter().sum();
            let var = sum / window as f64 * trading_days;
            out[end] = if var >= 0.0 { var.sqrt() } else { f64::NAN };
        }
    }
    out
}

/// Rolling Yang-Zhang OHLC realized volatility estimator.
///
/// Handles overnight gaps. Returns a `Vec<f64>` of the same length as the inputs.
/// The first `window` values are NaN (we need `window` bars plus the prior close
/// for overnight returns, so valid output starts at index `window`).
/// All four slices must have the same length.
pub fn yang_zhang_vol(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    window: usize,
    trading_days: f64,
) -> Vec<f64> {
    let n = open.len();
    let mut out = vec![f64::NAN; n];
    if window == 0 || n <= window || high.len() != n || low.len() != n || close.len() != n {
        return out;
    }

    let k = 0.34 / (1.34 + (window as f64 + 1.0) / (window as f64 - 1.0).max(1e-10));

    // Precompute per-bar components; index 0 has no overnight return.
    // overnight[i] = ln(O_i / C_{i-1}), valid for i >= 1
    // openclose[i] = ln(C_i / O_i)
    // rs[i] = Rogers-Satchell for bar i
    let mut overnight = vec![f64::NAN; n];
    let mut openclose = vec![f64::NAN; n];
    let mut rs = vec![f64::NAN; n];

    for i in 0..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        if o > 0.0
            && h > 0.0
            && l > 0.0
            && c > 0.0
            && o.is_finite()
            && h.is_finite()
            && l.is_finite()
            && c.is_finite()
        {
            openclose[i] = (c / o).ln();
            rs[i] = rs_bar(o, h, l, c);

            if i > 0 {
                let prev_c = close[i - 1];
                if prev_c > 0.0 && prev_c.is_finite() {
                    overnight[i] = (o / prev_c).ln();
                }
            }
        }
    }

    // Valid windows start at index `window` (using bars [end-window+1 .. end],
    // all of which have valid overnight returns since they start at index >= 1).
    for end in window..n {
        let start = end + 1 - window; // start >= 1 because end >= window

        let o_slice = &overnight[start..=end];
        let c_slice = &openclose[start..=end];
        let r_slice = &rs[start..=end];

        if !o_slice.iter().all(|v| v.is_finite())
            || !c_slice.iter().all(|v| v.is_finite())
            || !r_slice.iter().all(|v| v.is_finite())
        {
            continue;
        }

        let w = window as f64;

        let o_sum: f64 = o_slice.iter().sum();
        let o_sum_sq: f64 = o_slice.iter().map(|&x| x * x).sum();
        let overnight_var = o_sum_sq / (w - 1.0) - (o_sum / w).powi(2) * w / (w - 1.0);

        let c_sum: f64 = c_slice.iter().sum();
        let c_sum_sq: f64 = c_slice.iter().map(|&x| x * x).sum();
        let openclose_var = c_sum_sq / (w - 1.0) - (c_sum / w).powi(2) * w / (w - 1.0);

        let rs_sum: f64 = r_slice.iter().sum();
        let rs_var = rs_sum / w;

        let yz_var = overnight_var + k * openclose_var + (1.0 - k) * rs_var;
        let annualized = yz_var * trading_days;
        out[end] = if annualized >= 0.0 {
            annualized.sqrt()
        } else {
            f64::NAN
        };
    }
    out
}

/// Summary statistics of realized vol distribution for one window length.
#[derive(Clone, Copy, Debug)]
pub struct VolConeSlice {
    pub window: usize,
    pub min: f64,
    pub p25: f64,
    pub median: f64,
    pub p75: f64,
    pub max: f64,
}

/// Compute a percentile via linear interpolation on a sorted slice.
///
/// `sorted` must be non-empty and already sorted ascending.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = (n - 1) as f64 * p;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Compute vol cone: distribution of realized vols across multiple window lengths.
///
/// For each window in `windows`, the close-to-close rolling vol is computed,
/// NaN values are filtered out, and the distribution statistics (min, p25,
/// median, p75, max) are derived via linear interpolation.
pub fn vol_cone(close: &[f64], windows: &[usize], trading_days: f64) -> Vec<VolConeSlice> {
    windows
        .iter()
        .map(|&w| {
            let vols = close_to_close_vol(close, w, trading_days);
            let mut valid: Vec<f64> = vols.into_iter().filter(|v| v.is_finite()).collect();
            valid.sort_by(|a, b| a.partial_cmp(b).unwrap());

            if valid.is_empty() {
                return VolConeSlice {
                    window: w,
                    min: f64::NAN,
                    p25: f64::NAN,
                    median: f64::NAN,
                    p75: f64::NAN,
                    max: f64::NAN,
                };
            }

            VolConeSlice {
                window: w,
                min: valid[0],
                p25: percentile_sorted(&valid, 0.25),
                median: percentile_sorted(&valid, 0.5),
                p75: percentile_sorted(&valid, 0.75),
                max: *valid.last().unwrap(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_prices(n: usize) -> Vec<f64> {
        // simple synthetic price series
        let mut prices = vec![100.0_f64; n];
        for i in 1..n {
            prices[i] = prices[i - 1] * (1.0 + 0.01 * (i as f64 % 7_f64 - 3.0) * 0.01);
        }
        prices
    }

    #[test]
    fn close_to_close_returns_nans_for_warmup() {
        let close = fake_prices(100);
        let result = close_to_close_vol(&close, 20, 252.0);
        assert_eq!(result.len(), 100);
        // first 20 values should be NaN (window-1 of returns warmup + 1 for diff)
        for i in 0..20 {
            assert!(result[i].is_nan(), "result[{i}] should be NaN");
        }
        assert!(result[20].is_finite());
    }

    #[test]
    fn parkinson_vol_is_positive() {
        let close = fake_prices(100);
        let high: Vec<f64> = close.iter().map(|&c| c * 1.01).collect();
        let low: Vec<f64> = close.iter().map(|&c| c * 0.99).collect();
        let result = parkinson_vol(&high, &low, 20, 252.0);
        for v in result.iter().skip(19) {
            assert!(v.is_finite() && *v >= 0.0);
        }
    }

    #[test]
    fn vol_cone_is_ordered() {
        let close = fake_prices(300);
        let cones = vol_cone(&close, &[20, 60], 252.0);
        assert_eq!(cones.len(), 2);
        for cone in &cones {
            assert!(cone.min <= cone.p25);
            assert!(cone.p25 <= cone.median);
            assert!(cone.median <= cone.p75);
            assert!(cone.p75 <= cone.max);
        }
    }

    #[test]
    fn garman_klass_returns_nans_for_warmup() {
        let close = fake_prices(50);
        let high: Vec<f64> = close.iter().map(|&c| c * 1.01).collect();
        let low: Vec<f64> = close.iter().map(|&c| c * 0.99).collect();
        let result = garman_klass_vol(&close, &high, &low, &close, 10, 252.0);
        assert_eq!(result.len(), 50);
        for i in 0..9 {
            assert!(result[i].is_nan(), "result[{i}] should be NaN");
        }
        assert!(result[9].is_finite());
    }

    #[test]
    fn rogers_satchell_returns_nans_for_warmup() {
        let close = fake_prices(50);
        let high: Vec<f64> = close.iter().map(|&c| c * 1.01).collect();
        let low: Vec<f64> = close.iter().map(|&c| c * 0.99).collect();
        let result = rogers_satchell_vol(&close, &high, &low, &close, 10, 252.0);
        assert_eq!(result.len(), 50);
        for i in 0..9 {
            assert!(result[i].is_nan(), "result[{i}] should be NaN");
        }
        assert!(result[9].is_finite());
    }

    #[test]
    fn yang_zhang_returns_nans_for_warmup() {
        let close = fake_prices(50);
        let high: Vec<f64> = close.iter().map(|&c| c * 1.01).collect();
        let low: Vec<f64> = close.iter().map(|&c| c * 0.99).collect();
        let result = yang_zhang_vol(&close, &high, &low, &close, 10, 252.0);
        assert_eq!(result.len(), 50);
        for i in 0..10 {
            assert!(result[i].is_nan(), "result[{i}] should be NaN");
        }
        assert!(result[10].is_finite());
    }

    #[test]
    fn mismatched_lengths_return_all_nan() {
        let a = vec![100.0_f64; 20];
        let b = vec![101.0_f64; 15]; // wrong length
        let result = parkinson_vol(&a, &b, 5, 252.0);
        assert!(result.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn window_larger_than_data_returns_all_nan() {
        let close = fake_prices(10);
        let result = close_to_close_vol(&close, 20, 252.0);
        assert!(result.iter().all(|v| v.is_nan()));
    }
}
