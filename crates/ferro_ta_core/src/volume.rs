//! Volume indicators.

/// Compute On-Balance Volume (OBV).
///
/// OBV is a cumulative indicator that adds volume on up-close bars and
/// subtracts volume on down-close bars. Unchanged closes contribute zero.
/// Returns a `Vec<f64>` of length `n` with no `NaN` values.
///
/// # Arguments
/// * `close` - Price series.
/// * `volume` - Volume series (same length as `close`).
pub fn obv(close: &[f64], volume: &[f64]) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![0.0_f64; n];
    if n == 0 {
        return result;
    }
    // result[0] stays 0; accumulation starts from bar 1
    for i in 1..n {
        result[i] = result[i - 1]
            + if close[i] > close[i - 1] {
                volume[i]
            } else if close[i] < close[i - 1] {
                -volume[i]
            } else {
                0.0
            };
    }
    result
}

/// Compute the Money Flow Index (MFI).
///
/// MFI is a volume-weighted RSI, returning values in `[0, 100]`.
/// `typical_price = (H + L + C) / 3`; money flow is positive when
/// typical price rises, negative when it falls. The first `timeperiod`
/// values are `NaN`.
///
/// # Arguments
/// * `high` / `low` / `close` - OHLC price series (same length).
/// * `volume` - Volume series (same length).
/// * `timeperiod` - Lookback window (typically 14).
pub fn mfi(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    timeperiod: usize,
) -> Vec<f64> {
    let n = high.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 || n <= timeperiod {
        return result;
    }

    let mut pos_flow = vec![0.0_f64; n];
    let mut neg_flow = vec![0.0_f64; n];
    let mut tp_prev = (high[0] + low[0] + close[0]) / 3.0;

    for i in 1..n {
        let tp_cur = (high[i] + low[i] + close[i]) / 3.0;
        let rmf = tp_cur * volume[i];
        if tp_cur > tp_prev {
            pos_flow[i] = rmf;
        } else if tp_cur < tp_prev {
            neg_flow[i] = rmf;
        }
        tp_prev = tp_cur;
    }

    // Sliding window sum over timeperiod bars (indices i+1-timeperiod ..= i).
    // First valid window: indices 1..=timeperiod.
    let mut pos_sum: f64 = pos_flow[1..=timeperiod].iter().sum();
    let mut neg_sum: f64 = neg_flow[1..=timeperiod].iter().sum();
    // TA-Lib convention: zero total money flow (flat window / zero volume)
    // yields 0, not 100. 100*p/(p+n) is algebraically 100 - 100/(1+p/n).
    let mfi_val = |pos: f64, neg: f64| {
        let denom = pos + neg;
        if denom == 0.0 {
            0.0
        } else {
            100.0 * pos / denom
        }
    };
    result[timeperiod] = mfi_val(pos_sum, neg_sum);

    for i in (timeperiod + 1)..n {
        pos_sum += pos_flow[i] - pos_flow[i - timeperiod];
        neg_sum += neg_flow[i] - neg_flow[i - timeperiod];
        result[i] = mfi_val(pos_sum, neg_sum);
    }
    result
}

/// Chaikin Accumulation/Distribution Line.
///
/// Cumulates `(close - low - (high - close)) / (high - low) * volume`.
pub fn ad(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
    let n = high.len();
    let mut result = vec![0.0_f64; n];
    let mut ad_val = 0.0_f64;
    for i in 0..n {
        let hl = high[i] - low[i];
        let clv = if hl != 0.0 {
            ((close[i] - low[i]) - (high[i] - close[i])) / hl
        } else {
            0.0
        };
        ad_val += clv * volume[i];
        result[i] = ad_val;
    }
    result
}

/// Chaikin A/D Oscillator: fast EMA of AD minus slow EMA of AD.
///
/// Both EMAs are seeded with the *first* A/D value and run from bar 0, with
/// the first output at `slowperiod - 1` (TA-Lib's `ta_ADOSC.c` convention).
/// Note this differs from `overlap::ema`, which uses an SMA seed.
pub fn adosc(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    fastperiod: usize,
    slowperiod: usize,
) -> Vec<f64> {
    let n = high.len();
    let mut result = vec![f64::NAN; n];
    if fastperiod < 1 || slowperiod < 1 || n == 0 {
        return result;
    }
    let ad_vals = ad(high, low, close, volume);
    let fast_k = 2.0 / (fastperiod as f64 + 1.0);
    let slow_k = 2.0 / (slowperiod as f64 + 1.0);

    let mut fast_ema = ad_vals[0];
    let mut slow_ema = ad_vals[0];
    let warmup = slowperiod - 1;
    if warmup == 0 {
        result[0] = fast_ema - slow_ema;
    }
    for i in 1..n {
        fast_ema = fast_k * ad_vals[i] + (1.0 - fast_k) * fast_ema;
        slow_ema = slow_k * ad_vals[i] + (1.0 - slow_k) * slow_ema;
        if i >= warmup {
            result[i] = fast_ema - slow_ema;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obv_up_trend() {
        let c = vec![1.0, 2.0, 3.0];
        let v = vec![100.0, 200.0, 300.0];
        let result = obv(&c, &v);
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 200.0).abs() < 1e-10);
        assert!((result[2] - 500.0).abs() < 1e-10);
    }

    #[test]
    fn ad_basic() {
        let h = vec![10.0, 12.0, 11.0];
        let l = vec![8.0, 9.0, 9.0];
        let c = vec![9.0, 11.0, 10.0];
        let v = vec![1000.0, 2000.0, 1500.0];
        let result = ad(&h, &l, &c, &v);
        assert_eq!(result.len(), 3);
        // CLV[0] = ((9-8) - (10-9)) / (10-8) = (1 - 1) / 2 = 0
        assert!((result[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn adosc_basic() {
        let n = 30;
        let h: Vec<f64> = (1..=n).map(|i| i as f64 + 1.0).collect();
        let l: Vec<f64> = (1..=n).map(|i| i as f64 - 1.0).collect();
        let c: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let v: Vec<f64> = vec![1000.0; n];
        let result = adosc(&h, &l, &c, &v, 3, 10);
        assert_eq!(result.len(), n);
        // Warmup period should be NaN
        for i in 0..9 {
            assert!(result[i].is_nan());
        }
    }

    #[test]
    fn mfi_range() {
        let n = 50;
        let high: Vec<f64> = (1..=n).map(|i| i as f64 + 0.5).collect();
        let low: Vec<f64> = (1..=n).map(|i| i as f64 - 0.5).collect();
        let close: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let volume: Vec<f64> = vec![1_000_000.0; n];
        let result = mfi(&high, &low, &close, &volume, 14);
        for v in result.iter().filter(|v| !v.is_nan()) {
            assert!(*v >= 0.0 && *v <= 100.0, "MFI out of range: {v}");
        }
    }
}
