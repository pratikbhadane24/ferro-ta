//! Volume indicators.

/// On-Balance Volume.
pub fn obv(close: &[f64], volume: &[f64]) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![0.0_f64; n];
    if n == 0 {
        return result;
    }
    result[0] = volume[0];
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

/// Money Flow Index — O(n) sliding-window implementation without per-bar allocation.
///
/// MFI = 100 - 100 / (1 + positive_flow / negative_flow) over `timeperiod` bars.
/// typical_price = (high + low + close) / 3; raw_money_flow = typical_price * volume.
/// Leading `timeperiod` values are NaN.
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
    let mfr = if neg_sum == 0.0 {
        f64::MAX
    } else {
        pos_sum / neg_sum
    };
    result[timeperiod] = 100.0 - 100.0 / (1.0 + mfr);

    for i in (timeperiod + 1)..n {
        pos_sum += pos_flow[i] - pos_flow[i - timeperiod];
        neg_sum += neg_flow[i] - neg_flow[i - timeperiod];
        let mfr = if neg_sum == 0.0 {
            f64::MAX
        } else {
            pos_sum / neg_sum
        };
        result[i] = 100.0 - 100.0 / (1.0 + mfr);
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
        assert!((result[0] - 100.0).abs() < 1e-10);
        assert!((result[1] - 300.0).abs() < 1e-10);
        assert!((result[2] - 600.0).abs() < 1e-10);
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
