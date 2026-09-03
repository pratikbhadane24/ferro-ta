//! Volume-weighted price kernels: VWAP (cumulative and rolling) and VWMA.

// ---------------------------------------------------------------------------
// VWAP
// ---------------------------------------------------------------------------

/// Volume Weighted Average Price (cumulative or rolling).
///
/// # Arguments
/// * `high`, `low`, `close`, `volume` — equal-length price/volume slices.
/// * `timeperiod` — 0 for cumulative VWAP from bar 0; >= 1 for a rolling window.
///
/// # Returns
/// A `Vec<f64>` of VWAP values. For rolling mode the first `timeperiod - 1`
/// entries are `NaN`. Mismatched input lengths yield all `NaN`.
pub fn vwap(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    timeperiod: usize,
) -> Vec<f64> {
    let n = high.len();
    if low.len() != n || close.len() != n || volume.len() != n {
        return vec![f64::NAN; n];
    }

    if timeperiod == 0 {
        let mut result = vec![f64::NAN; n];
        let mut cum_tpv = 0.0_f64;
        let mut cum_vol = 0.0_f64;
        for i in 0..n {
            let tp = (high[i] + low[i] + close[i]) / 3.0;
            cum_tpv += tp * volume[i];
            cum_vol += volume[i];
            if cum_vol != 0.0 {
                result[i] = cum_tpv / cum_vol;
            }
        }
        return result;
    }

    // Pre-compute cumulative sums for O(n) rolling window. These stay
    // zero-initialized: a zero fill is a lazily-mapped page, unlike `NaN`.
    let mut cum_tpv_arr = vec![0.0_f64; n];
    let mut cum_vol_arr = vec![0.0_f64; n];
    for i in 0..n {
        let tp = (high[i] + low[i] + close[i]) / 3.0;
        let tpv = tp * volume[i];
        cum_tpv_arr[i] = tpv + if i > 0 { cum_tpv_arr[i - 1] } else { 0.0 };
        cum_vol_arr[i] = volume[i] + if i > 0 { cum_vol_arr[i - 1] } else { 0.0 };
    }

    let warmup = (timeperiod - 1).min(n);
    let mut result = vec![f64::NAN; n];
    for i in warmup..n {
        let prev_tpv = if i >= timeperiod {
            cum_tpv_arr[i - timeperiod]
        } else {
            0.0
        };
        let prev_vol = if i >= timeperiod {
            cum_vol_arr[i - timeperiod]
        } else {
            0.0
        };
        let w_tpv = cum_tpv_arr[i] - prev_tpv;
        let w_vol = cum_vol_arr[i] - prev_vol;
        if w_vol != 0.0 {
            result[i] = w_tpv / w_vol;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// VWMA
// ---------------------------------------------------------------------------

/// Volume Weighted Moving Average.
///
/// `VWMA = sum(close * volume, n) / sum(volume, n)`
///
/// # Arguments
/// * `close` — price series.
/// * `volume` — volume series (same length as `close`).
/// * `timeperiod` — rolling window size (>= 1).
///
/// # Returns
/// A `Vec<f64>` with `NaN` for the first `timeperiod - 1` entries. Mismatched
/// input lengths yield all `NaN`.
pub fn vwma(close: &[f64], volume: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    if timeperiod < 1 || n < timeperiod || volume.len() != n {
        return vec![f64::NAN; n];
    }

    let mut cum_cv = vec![0.0_f64; n];
    let mut cum_v = vec![0.0_f64; n];
    for i in 0..n {
        cum_cv[i] = close[i] * volume[i] + if i > 0 { cum_cv[i - 1] } else { 0.0 };
        cum_v[i] = volume[i] + if i > 0 { cum_v[i - 1] } else { 0.0 };
    }

    let mut result = vec![f64::NAN; n];
    for i in (timeperiod - 1)..n {
        let prev_cv = if i >= timeperiod {
            cum_cv[i - timeperiod]
        } else {
            0.0
        };
        let prev_v = if i >= timeperiod {
            cum_v[i - timeperiod]
        } else {
            0.0
        };
        let w_cv = cum_cv[i] - prev_cv;
        let w_v = cum_v[i] - prev_v;
        if w_v != 0.0 {
            result[i] = w_cv / w_v;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extended::test_support::sample_ohlcv;

    // -----------------------------------------------------------------------
    // VWAP tests
    // -----------------------------------------------------------------------

    #[test]
    fn vwap_cumulative_basic() {
        let (h, l, c, v) = sample_ohlcv();
        let result = vwap(&h, &l, &c, &v, 0);
        assert_eq!(result.len(), h.len());
        // First bar: tp = (11+9+10)/3 = 10.0, tpv = 1000.0, vol = 100.0 => 10.0
        assert!((result[0] - 10.0).abs() < 1e-10);
        // All values should be non-NaN for cumulative
        for val in &result {
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn vwap_empty_input() {
        let result = vwap(&[], &[], &[], &[], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn vwap_rolling_basic() {
        let (h, l, c, v) = sample_ohlcv();
        let result = vwap(&h, &l, &c, &v, 3);
        assert_eq!(result.len(), h.len());
        // First 2 values should be NaN
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // From index 2 onward should be valid
        assert!(!result[2].is_nan());
    }

    // -----------------------------------------------------------------------
    // VWMA tests
    // -----------------------------------------------------------------------

    #[test]
    fn vwma_basic() {
        let (_, _, c, v) = sample_ohlcv();
        let result = vwma(&c, &v, 3);
        assert_eq!(result.len(), c.len());
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // Index 2: sum(c*v, 0..3) / sum(v, 0..3) = (1000+1650+2400)/(100+150+200) = 5050/450
        let expected = (10.0 * 100.0 + 11.0 * 150.0 + 12.0 * 200.0) / (100.0 + 150.0 + 200.0);
        assert!((result[2] - expected).abs() < 1e-10);
    }

    #[test]
    fn vwma_empty_input() {
        let result = vwma(&[], &[], 3);
        assert!(result.is_empty());
    }

    #[test]
    fn vwma_period_larger_than_data() {
        let result = vwma(&[1.0, 2.0], &[100.0, 200.0], 5);
        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|v| v.is_nan()));
    }
}
