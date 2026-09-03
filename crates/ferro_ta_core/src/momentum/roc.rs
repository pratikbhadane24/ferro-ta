//! Momentum and the Rate-of-Change family: `mom`, `roc`, `rocp`, `rocr`
//! and `rocr100`.

/// Compute the Momentum indicator: `close[i] - close[i - timeperiod]`.
///
/// Returns a `Vec<f64>` of length `n`. The first `timeperiod` values are `NaN`.
/// Positive values indicate upward price movement over the lookback window.
///
/// # Arguments
/// * `close` - Price series.
/// * `timeperiod` - Number of bars to look back (must be >= 1).
pub fn mom(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 {
        return result;
    }
    for i in timeperiod..n {
        result[i] = close[i] - close[i - timeperiod];
    }
    result
}

// ---------------------------------------------------------------------------
// Rate of Change variants
// ---------------------------------------------------------------------------

// The four variants share one shape: `NaN` for the first `timeperiod` bars,
// then a bar-to-bar ratio against `close[i - timeperiod]`, with `NaN` wherever
// that divisor is exactly zero.
//
// They allocate with `vec![f64::NAN; n]` and write by index. An earlier
// revision used `Vec::with_capacity` + `push` to dodge the `NaN` prologue --- a
// real 800 KB store pass at `n = 100_000`, since `NaN` is not zero and so
// cannot be a lazily-mapped zero page. That was **measured as a regression
// here**: `push` carries a per-bar capacity check and a reload of the vec
// header, and these are the lightest kernels in the crate (~1.2 ns/bar, close
// to the O(n) floor), so the per-bar cost outweighs the one-time prologue by a
// wide margin. The same reversal was measured independently on `median`
// (+20.3% at 100k) and on `obv`.
//
// The rule: reclaim a `NaN` prologue only when the kernel does enough per-bar
// work to absorb a capacity check. For a light kernel, keep the `vec!` and
// spend the effort on bounds-check elision instead.

/// Rate of Change: `(close[i] - close[i-p]) / close[i-p] * 100`.
pub fn roc(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    if timeperiod == 0 {
        return vec![f64::NAN; n];
    }
    let mut result = vec![f64::NAN; n];
    for i in timeperiod..n {
        let prev = close[i - timeperiod];
        if prev != 0.0 {
            result[i] = (close[i] - prev) / prev * 100.0;
        }
    }
    result
}

/// Rate of Change Percentage: `(close[i] - close[i-p]) / close[i-p]`.
pub fn rocp(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    if timeperiod == 0 {
        return vec![f64::NAN; n];
    }
    let mut result = vec![f64::NAN; n];
    for i in timeperiod..n {
        let prev = close[i - timeperiod];
        if prev != 0.0 {
            result[i] = (close[i] - prev) / prev;
        }
    }
    result
}

/// Rate of Change Ratio: `close[i] / close[i-p]`.
pub fn rocr(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    if timeperiod == 0 {
        return vec![f64::NAN; n];
    }
    let mut result = vec![f64::NAN; n];
    for i in timeperiod..n {
        let prev = close[i - timeperiod];
        if prev != 0.0 {
            result[i] = close[i] / prev;
        }
    }
    result
}

/// Rate of Change Ratio x 100: `close[i] / close[i-p] * 100`.
pub fn rocr100(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    if timeperiod == 0 {
        return vec![f64::NAN; n];
    }
    let mut result = vec![f64::NAN; n];
    for i in timeperiod..n {
        let prev = close[i - timeperiod];
        if prev != 0.0 {
            result[i] = close[i] / prev * 100.0;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::momentum::test_support::assert_bits;

    #[test]
    fn mom_basic() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = mom(&prices, 2);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-10);
    }

    // -- Group E: the dropped NaN prologue is bit-identical -------------------

    /// The pre-rewrite `roc` family, verbatim, parameterized by the one
    /// expression that differs.
    fn reference_roc_family(close: &[f64], timeperiod: usize, variant: usize) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];
        if timeperiod == 0 {
            return result;
        }
        for i in timeperiod..n {
            let prev = close[i - timeperiod];
            if prev != 0.0 {
                result[i] = match variant {
                    0 => (close[i] - prev) / prev * 100.0,
                    1 => (close[i] - prev) / prev,
                    2 => close[i] / prev,
                    _ => close[i] / prev * 100.0,
                };
            }
        }
        result
    }

    #[test]
    fn roc_family_is_bit_identical_after_dropping_the_nan_prologue() {
        // Includes exact zeros (the `prev != 0.0` skip branch), negatives and
        // a length shorter than the period.
        let mut series: Vec<f64> = (0..60).map(|i| 10.0 + (i % 7) as f64 - 3.0).collect();
        series[5] = 0.0;
        series[6] = -0.0;
        series[41] = 0.0;
        /// Signature shared by the whole ROC family.
        type RocFn = fn(&[f64], usize) -> Vec<f64>;
        let variants: [(&str, RocFn); 4] = [
            ("roc", roc),
            ("rocp", rocp),
            ("rocr", rocr),
            ("rocr100", rocr100),
        ];
        for (v, (name, f)) in variants.iter().enumerate() {
            for tp in [0, 1, 2, 7, 59, 60, 61, 200] {
                for len in [0, 1, 60] {
                    let data = &series[..len];
                    let want = reference_roc_family(data, tp, v);
                    let got = f(data, tp);
                    assert_bits(&got, &want, &format!("{name} tp={tp} len={len}"));
                }
            }
        }
    }
}
