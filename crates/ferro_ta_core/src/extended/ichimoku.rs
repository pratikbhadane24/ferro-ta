//! Ichimoku Kinko Hyo and the shifted rolling-midpoint helper it uses.

use crate::rolling::{MaxDeque, MinDeque};

// ---------------------------------------------------------------------------
// ICHIMOKU
// ---------------------------------------------------------------------------

/// Ichimoku Cloud (Ichimoku Kinko Hyo).
///
/// Write the rolling `(highest high + lowest low) / 2` of each full `period`
/// window into `out[i + shift]`, in **one** traversal — both extremum deques
/// driven together, so a midpoint line costs one pass and zero intermediate
/// `Vec`s.
///
/// `shift` lets a forward-projected line be written straight into its
/// destination slot: the value derived from bar `i` lands at `i + shift`, so
/// it can only ever depend on bar `i` and earlier, making the no-look-ahead
/// property structural. Slots whose source window is incomplete, or which
/// fall past the end, keep whatever `out` already held.
///
/// The `NaN` test is on the window high alone, matching the form this
/// replaced: a `NaN` low then flows through the addition on its own.
fn midpoint_into_shifted(high: &[f64], low: &[f64], period: usize, shift: usize, out: &mut [f64]) {
    let n = high.len();
    if period < 1 || n < period {
        return;
    }
    let mut hi = MaxDeque::with_window(period, n);
    let mut lo = MinDeque::with_window(period, n);
    for i in 0..n {
        hi.advance(i, high[i], period);
        lo.advance(i, low[i], period);
        if i + 1 < period {
            continue;
        }
        let dst = i + shift;
        if dst >= n {
            // `dst` only grows, so nothing further can land in range.
            break;
        }
        let hh = hi.front();
        if !hh.is_nan() {
            out[dst] = (hh + lo.front()) / 2.0;
        }
    }
}

/// # Returns
/// `(tenkan, kijun, senkou_a, senkou_b, chikou)` arrays. Mismatched input
/// lengths yield all `NaN`.
#[allow(clippy::type_complexity)]
pub fn ichimoku(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    tenkan_period: usize,
    kijun_period: usize,
    senkou_b_period: usize,
    displacement: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = high.len();
    let nan = || vec![f64::NAN; n];

    if tenkan_period < 1
        || kijun_period < 1
        || senkou_b_period < 1
        || low.len() != n
        || close.len() != n
    {
        return (nan(), nan(), nan(), nan(), nan());
    }

    let mut tenkan = nan();
    let mut kijun = nan();
    let mut senkou_a = nan();
    let mut senkou_b = nan();
    let mut chikou = nan();

    // Three fused passes writing midpoints straight into the destination
    // slots. Senkou B's pass carries the displacement, which is what
    // eliminates the `raw_b` temporary.
    midpoint_into_shifted(high, low, tenkan_period, 0, &mut tenkan);
    midpoint_into_shifted(high, low, kijun_period, 0, &mut kijun);
    midpoint_into_shifted(high, low, senkou_b_period, displacement, &mut senkou_b);

    // Senkou A: (tenkan + kijun) / 2 projected forward `displacement` bars,
    // so senkou_a[i] only uses data from bar i - displacement (no look-ahead).
    for i in displacement..n {
        let src = i - displacement;
        if tenkan[src].is_finite() && kijun[src].is_finite() {
            senkou_a[i] = (tenkan[src] + kijun[src]) / 2.0;
        }
    }

    // Chikou (lagging span): close plotted `displacement` bars back, i.e.
    // chikou[i] = close[i + displacement] (standard close.shift(-displacement)).
    if n > displacement {
        chikou[..n - displacement].copy_from_slice(&close[displacement..]);
    }

    (tenkan, kijun, senkou_a, senkou_b, chikou)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extended::test_support::{assert_bit_eq, same, stress_cases};

    // -----------------------------------------------------------------------
    // ICHIMOKU tests
    // -----------------------------------------------------------------------

    #[test]
    fn ichimoku_basic() {
        // Use a larger dataset for ichimoku
        let n = 60;
        let high: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 + 1.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 - 1.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();

        let (tenkan, kijun, senkou_a, senkou_b, chikou) =
            ichimoku(&high, &low, &close, 9, 26, 52, 26);

        assert_eq!(tenkan.len(), n);
        assert_eq!(kijun.len(), n);
        assert_eq!(senkou_a.len(), n);
        assert_eq!(senkou_b.len(), n);
        assert_eq!(chikou.len(), n);

        // Tenkan: period 9, first valid at index 8
        assert!(tenkan[7].is_nan());
        assert!(!tenkan[8].is_nan());

        // Kijun: period 26, first valid at index 25
        assert!(kijun[24].is_nan());
        assert!(!kijun[25].is_nan());

        // Chikou (lagging span): close plotted 26 bars back —
        // chikou[i] == close[i + 26], NaN for the last 26 bars.
        assert!(!chikou[0].is_nan());
        assert!((chikou[0] - close[26]).abs() < 1e-10);
        assert!((chikou[n - 27] - close[n - 1]).abs() < 1e-10);
        assert!(chikou[n - 26].is_nan());
        assert!(chikou[n - 1].is_nan());

        // Senkou spans are projected 26 bars forward: leading NaNs, and no
        // value may depend on a future bar.
        assert!(senkou_a[25].is_nan());
        assert!(senkou_b[25].is_nan());
        // senkou_a[i] == (tenkan[i-26] + kijun[i-26]) / 2
        assert!(!senkou_a[51].is_nan());
        assert!((senkou_a[51] - (tenkan[25] + kijun[25]) / 2.0).abs() < 1e-10);
    }

    #[test]
    fn ichimoku_empty_input() {
        let (t, k, sa, sb, ch) = ichimoku(&[], &[], &[], 9, 26, 52, 26);
        assert!(t.is_empty());
        assert!(k.is_empty());
        assert!(sa.is_empty());
        assert!(sb.is_empty());
        assert!(ch.is_empty());
    }

    fn rolling_hl_midpoint(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
        let n = high.len();
        let mut out = vec![f64::NAN; n];
        if period < 1 || n < period {
            return out;
        }
        for i in (period - 1)..n {
            let start = i + 1 - period;
            let hh = high[start..=i]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let ll = low[start..=i].iter().copied().fold(f64::INFINITY, f64::min);
            out[i] = (hh + ll) / 2.0;
        }
        out
    }

    /// Senkou at index `i` must use tenkan/kijun/raw_b at `i - d`, never `> i`.
    #[test]
    fn ichimoku_senkou_no_lookahead() {
        let n = 20;
        let high: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 + 2.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 - 2.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();

        let tenkan_p = 3usize;
        let kijun_p = 4usize;
        let senkou_b_p = 5usize;
        let d = 2usize;

        let (tenkan, kijun, senkou_a, senkou_b, chikou) =
            ichimoku(&high, &low, &close, tenkan_p, kijun_p, senkou_b_p, d);

        let expected_tenkan = rolling_hl_midpoint(&high, &low, tenkan_p);
        let expected_kijun = rolling_hl_midpoint(&high, &low, kijun_p);
        let expected_raw_b = rolling_hl_midpoint(&high, &low, senkou_b_p);

        for i in 0..n {
            assert!(
                (tenkan[i] - expected_tenkan[i]).abs() < 1e-10
                    || (tenkan[i].is_nan() && expected_tenkan[i].is_nan())
            );
            assert!(
                (kijun[i] - expected_kijun[i]).abs() < 1e-10
                    || (kijun[i].is_nan() && expected_kijun[i].is_nan())
            );

            if i < d {
                assert!(
                    senkou_a[i].is_nan(),
                    "senkou_a[{i}] must be NaN (no past bar)"
                );
                assert!(
                    senkou_b[i].is_nan(),
                    "senkou_b[{i}] must be NaN (no past bar)"
                );
                continue;
            }

            let src = i - d;
            assert!(
                src <= i,
                "Senkou source index {src} must not be ahead of {i}"
            );

            let exp_a = if expected_tenkan[src].is_finite() && expected_kijun[src].is_finite() {
                (expected_tenkan[src] + expected_kijun[src]) / 2.0
            } else {
                f64::NAN
            };
            if exp_a.is_nan() {
                assert!(
                    senkou_a[i].is_nan(),
                    "senkou_a[{i}] should be NaN (tenkan/kijun[{src}] not both finite)"
                );
            } else {
                assert!(
                    (senkou_a[i] - exp_a).abs() < 1e-10,
                    "senkou_a[{i}] = {} expected {} from tenkan/kijun[{src}]",
                    senkou_a[i],
                    exp_a
                );
            }

            let exp_b = expected_raw_b[src];
            if exp_b.is_nan() {
                assert!(senkou_b[i].is_nan(), "senkou_b[{i}] should be NaN");
            } else {
                assert!(
                    (senkou_b[i] - exp_b).abs() < 1e-10,
                    "senkou_b[{i}] = {} expected raw_b[{src}] = {}",
                    senkou_b[i],
                    exp_b
                );
            }

            let chikou_src = i + d;
            if chikou_src < n {
                assert!((chikou[i] - close[chikou_src]).abs() < 1e-10);
            } else {
                assert!(chikou[i].is_nan());
            }
        }

        // Old mapping wrote senkou_a[i] from tenkan/kijun[i + d] (lookahead).
        let future = d;
        let lookahead = (expected_tenkan[future] + expected_kijun[future]) / 2.0;
        assert!(
            senkou_a[0].is_nan() || (senkou_a[0] - lookahead).abs() > 1e-10,
            "senkou_a[0] must not equal the lookahead value from tenkan/kijun[{future}]"
        );
    }

    /// Hand-computed Senkou values: span at i equals the midpoint from i - d.
    #[test]
    fn ichimoku_senkou_golden_displaced_past() {
        let high = vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0];
        let low = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let close = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];
        let d = 2usize;

        let (_, _, senkou_a, senkou_b, chikou) = ichimoku(&high, &low, &close, 2, 3, 4, d);

        // tenkan p=2: [NaN, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5]
        // kijun  p=3: [NaN, NaN,  11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
        // raw_b  p=4: [NaN, NaN,  NaN,  11.5, 12.5, 13.5, 14.5, 15.5]
        // senkou_a[i] = (tenkan[i-2] + kijun[i-2]) / 2
        for i in 0..4 {
            assert!(senkou_a[i].is_nan(), "senkou_a[{i}] warmup");
        }
        assert!((senkou_a[4] - 11.25).abs() < 1e-10); // (11.5 + 11.0) / 2
        assert!((senkou_a[5] - 12.25).abs() < 1e-10);
        assert!((senkou_a[6] - 13.25).abs() < 1e-10);
        assert!((senkou_a[7] - 14.25).abs() < 1e-10);

        // senkou_b[i] = raw_b[i-2]
        for i in 0..5 {
            assert!(senkou_b[i].is_nan(), "senkou_b[{i}] warmup");
        }
        assert!((senkou_b[5] - 11.5).abs() < 1e-10);
        assert!((senkou_b[6] - 12.5).abs() < 1e-10);
        assert!((senkou_b[7] - 13.5).abs() < 1e-10);

        // Chikou is close plotted `d` bars back: chikou[i] = close[i + d].
        assert!((chikou[0] - 12.0).abs() < 1e-10);
        assert!((chikou[5] - 17.0).abs() < 1e-10);
        assert!(chikou[6].is_nan());
        assert!(chikou[7].is_nan());
    }

    /// A spike on the last bar must not leak into earlier Senkou values.
    #[test]
    fn ichimoku_senkou_ignores_future_spike() {
        let n = 12;
        let mut high: Vec<f64> = (0..n).map(|i| 10.0 + i as f64).collect();
        let mut low: Vec<f64> = (0..n).map(|i| 8.0 + i as f64).collect();
        let close: Vec<f64> = (0..n).map(|i| 9.0 + i as f64).collect();
        high[n - 1] = 1000.0;
        low[n - 1] = 999.0;

        let d = 3usize;
        let (_, _, senkou_a, senkou_b, _) = ichimoku(&high, &low, &close, 2, 2, 2, d);

        for i in 0..n {
            if senkou_a[i].is_finite() {
                assert!(
                    senkou_a[i] < 100.0,
                    "senkou_a[{i}] = {} leaked a future spike (lookahead)",
                    senkou_a[i]
                );
            }
            if senkou_b[i].is_finite() {
                assert!(
                    senkou_b[i] < 100.0,
                    "senkou_b[{i}] = {} leaked a future spike (lookahead)",
                    senkou_b[i]
                );
            }
        }
    }

    #[allow(clippy::type_complexity)]
    fn reference_ichimoku(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        tenkan_period: usize,
        kijun_period: usize,
        senkou_b_period: usize,
        displacement: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = high.len();
        let nan = || vec![f64::NAN; n];
        if tenkan_period < 1 || kijun_period < 1 || senkou_b_period < 1 {
            return (nan(), nan(), nan(), nan(), nan());
        }
        let midpoint_rolling = |period: usize| -> Vec<f64> {
            let hh = crate::math::sliding_max(high, period);
            let ll = crate::math::sliding_min(low, period);
            let mut result = vec![f64::NAN; n];
            for i in 0..n {
                if !hh[i].is_nan() {
                    result[i] = (hh[i] + ll[i]) / 2.0;
                }
            }
            result
        };
        let tenkan = midpoint_rolling(tenkan_period);
        let kijun = midpoint_rolling(kijun_period);
        let raw_b = midpoint_rolling(senkou_b_period);
        let mut senkou_a = nan();
        let mut senkou_b = nan();
        let mut chikou = nan();
        if n > displacement {
            for i in displacement..n {
                let src = i - displacement;
                if tenkan[src].is_finite() && kijun[src].is_finite() {
                    senkou_a[i] = (tenkan[src] + kijun[src]) / 2.0;
                }
            }
            senkou_b[displacement..].copy_from_slice(&raw_b[..n - displacement]);
            chikou[..n - displacement].copy_from_slice(&close[displacement..]);
        }
        (tenkan, kijun, senkou_a, senkou_b, chikou)
    }

    /// Bit-identical: the midpoints are the same deque output, and Senkou B's
    /// displaced write reproduces the `copy_from_slice` index mapping exactly.
    #[test]
    fn ichimoku_matches_reference_bitwise() {
        let param_sets = [
            (9usize, 26usize, 52usize, 26usize),
            (1, 1, 1, 0),
            (2, 3, 4, 2),
            (3, 3, 3, 100),
            (0, 26, 52, 26),
            (9, 0, 52, 26),
            (9, 26, 0, 26),
        ];
        for (name, h, l, c) in stress_cases() {
            for (t, k, b, d) in param_sets {
                let got = ichimoku(&h, &l, &c, t, k, b, d);
                let want = reference_ichimoku(&h, &l, &c, t, k, b, d);
                let label = format!("ichimoku({name}, {t}/{k}/{b}/{d})");
                assert_bit_eq(&got.0, &want.0, &format!("{label}.tenkan"));
                assert_bit_eq(&got.1, &want.1, &format!("{label}.kijun"));
                assert_bit_eq(&got.2, &want.2, &format!("{label}.senkou_a"));
                assert_bit_eq(&got.3, &want.3, &format!("{label}.senkou_b"));
                assert_bit_eq(&got.4, &want.4, &format!("{label}.chikou"));
            }
        }
    }

    /// The rewrite writes `senkou_b[i + displacement]` from bar `i`, which is
    /// exactly where look-ahead gets reintroduced by accident. Replacing the
    /// tail of the input with garbage must not move any earlier output.
    ///
    /// `chikou` is excluded by construction: it is a *lagging* span, defined
    /// as `close[i + displacement]`, so it reads forward on purpose.
    #[test]
    fn ichimoku_outputs_do_not_depend_on_the_future() {
        let n = 120usize;
        let cut = 70usize;
        let (t_p, k_p, b_p, d) = (9usize, 26usize, 52usize, 26usize);

        let high: Vec<f64> = (0..n).map(|i| 100.0 + (i % 11) as f64 + 1.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 100.0 + (i % 11) as f64 - 1.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i % 11) as f64).collect();

        let mut gh = high.clone();
        let mut gl = low.clone();
        let mut gc = close.clone();
        for i in cut..n {
            gh[i] = 1.0e9;
            gl[i] = -1.0e9;
            gc[i] = f64::NAN;
        }

        let base = ichimoku(&high, &low, &close, t_p, k_p, b_p, d);
        let garbled = ichimoku(&gh, &gl, &gc, t_p, k_p, b_p, d);

        // tenkan/kijun at bar i read only [i - p + 1, i].
        for i in 0..cut {
            assert!(same(base.0[i], garbled.0[i]), "tenkan[{i}] saw the future");
            assert!(same(base.1[i], garbled.1[i]), "kijun[{i}] saw the future");
        }
        // The Senkou spans at bar i read only bars up to i - d.
        for i in 0..(cut + d) {
            assert!(
                same(base.2[i], garbled.2[i]),
                "senkou_a[{i}] saw the future"
            );
            assert!(
                same(base.3[i], garbled.3[i]),
                "senkou_b[{i}] saw the future"
            );
        }
        // ... and the test has teeth: the first slot whose source window does
        // touch the garbage must differ.
        assert!(
            !same(base.3[cut + d], garbled.3[cut + d]),
            "senkou_b[{}] should have changed",
            cut + d
        );
    }
}
