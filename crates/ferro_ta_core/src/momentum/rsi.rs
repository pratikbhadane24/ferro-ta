//! RSI and CMO — the two Wilder-smoothed momentum oscillators.
//!
//! See the module note on Wilder smoothing and the hoisted reciprocal in
//! [`super`] for why both advance their averages with a reciprocal multiply.

/// Compute the Relative Strength Index (RSI).
///
/// Returns values in the range `[0, 100]`. Uses Wilder's smoothing method
/// (TA-Lib compatible), seeding avg_gain/avg_loss with the SMA of the first
/// `timeperiod` price changes. The first `timeperiod` values are `NaN`.
///
/// # Arguments
/// * `close` - Price series.
/// * `timeperiod` - Lookback period (typically 14).
pub fn rsi(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    if n <= timeperiod || timeperiod < 1 {
        return vec![f64::NAN; n];
    }
    let mut avg_gain = 0.0_f64;
    let mut avg_loss = 0.0_f64;
    for i in 1..=timeperiod {
        let diff = close[i] - close[i - 1];
        let abs_diff = diff.abs();
        avg_gain += (diff + abs_diff) * 0.5;
        avg_loss += (abs_diff - diff) * 0.5;
    }
    avg_gain /= timeperiod as f64;
    avg_loss /= timeperiod as f64;
    let p = timeperiod as f64;
    // `p` is loop-invariant, but LLVM may not turn `x / p` into `x * (1 / p)`
    // on its own: that is a value-changing transform outside fast-math. Doing
    // it by hand trades a ~14-cycle divide in the critical dependency chain
    // for a ~4-cycle multiply. See the module note on Wilder smoothing.
    let inv_p = 1.0 / p;
    // TA-Lib convention: a fully flat window (no gains AND no losses) yields
    // 0, not 100. 100*g/(g+l) is algebraically 100 - 100/(1+g/l).
    // gain/(gain+loss) is mathematically in [0, 1], but rounding can push it a
    // hair past 1.0 when loss is denormal, so clamp the documented range.
    let rsi_val = |gain: f64, loss: f64| {
        let denom = gain + loss;
        if denom == 0.0 {
            0.0
        } else {
            (100.0 * gain / denom).clamp(0.0, 100.0)
        }
    };
    let mut result = vec![f64::NAN; n];
    result[timeperiod] = rsi_val(avg_gain, avg_loss);
    for i in (timeperiod + 1)..n {
        let diff = close[i] - close[i - 1];
        let abs_diff = diff.abs();
        let gain = (diff + abs_diff) * 0.5;
        let loss = (abs_diff - diff) * 0.5;
        avg_gain = (avg_gain * (p - 1.0) + gain) * inv_p;
        avg_loss = (avg_loss * (p - 1.0) + loss) * inv_p;
        result[i] = rsi_val(avg_gain, avg_loss);
    }
    result
}

// ---------------------------------------------------------------------------
// CMO
// ---------------------------------------------------------------------------

/// Chande Momentum Oscillator: `100 * (gains - losses) / (gains + losses)`.
///
/// Uses Wilder's smoothing with an SMA seed, matching TA-Lib's `ta_CMO.c`
/// (which is "mostly identical to RSI"). A plain rolling-window sum would
/// diverge from TA-Lib permanently rather than converging.
pub fn cmo(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    if timeperiod < 1 || n <= timeperiod {
        return vec![f64::NAN; n];
    }

    let mut avg_gain = 0.0_f64;
    let mut avg_loss = 0.0_f64;
    for i in 1..=timeperiod {
        let diff = close[i] - close[i - 1];
        let abs_diff = diff.abs();
        avg_gain += (diff + abs_diff) * 0.5;
        avg_loss += (abs_diff - diff) * 0.5;
    }
    avg_gain /= timeperiod as f64;
    avg_loss /= timeperiod as f64;

    // Clamp for the same rounding reason as `rsi`: (g-l)/(g+l) is
    // mathematically in [-1, 1].
    let cmo_val = |gain: f64, loss: f64| {
        let denom = gain + loss;
        if denom == 0.0 {
            0.0
        } else {
            (100.0 * (gain - loss) / denom).clamp(-100.0, 100.0)
        }
    };
    let mut result = vec![f64::NAN; n];
    result[timeperiod] = cmo_val(avg_gain, avg_loss);

    let p = timeperiod as f64;
    // Reciprocal hoist, as in `rsi`; see the module note on Wilder smoothing.
    let inv_p = 1.0 / p;
    for i in (timeperiod + 1)..n {
        let diff = close[i] - close[i - 1];
        let abs_diff = diff.abs();
        avg_gain = (avg_gain * (p - 1.0) + (diff + abs_diff) * 0.5) * inv_p;
        avg_loss = (avg_loss * (p - 1.0) + (abs_diff - diff) * 0.5) * inv_p;
        result[i] = cmo_val(avg_gain, avg_loss);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::momentum::test_support::ultosc_ohlc;

    #[test]
    fn rsi_range() {
        let prices: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let result = rsi(&prices, 14);
        for v in result.iter().filter(|v| !v.is_nan()) {
            assert!(*v >= 0.0 && *v <= 100.0);
        }
    }

    /// `100 * gain / (gain + loss)` is mathematically bounded by 100, but the
    /// two roundings (the multiply, then the divide) can land one ulp above it:
    /// `(100.0 * g) / g == 100.00000000000001` for this g. The previous
    /// `100 - 100/(1 + rs)` form was bounded by construction, so the rewrite to
    /// TA-Lib's expression needed an explicit clamp. Found by fuzz_rsi.
    #[test]
    fn rsi_cannot_exceed_100_on_rounding_boundary() {
        let g = f64::from_bits(0x65aa_9c82_79f2_48b0);
        assert!(
            (100.0 * g) / g > 100.0,
            "test value no longer triggers rounding"
        );

        // timeperiod = 1: a single positive diff gives gain = g, loss = 0.
        let result = rsi(&[0.0, g], 1);
        assert!(
            result[1] <= 100.0,
            "RSI must stay within [0, 100], got {}",
            result[1]
        );
    }

    #[test]
    fn cmo_stays_within_bounds_on_rounding_boundary() {
        let g = f64::from_bits(0x65aa_9c82_79f2_48b0);
        let result = cmo(&[0.0, g], 1);
        assert!(
            result[1] >= -100.0 && result[1] <= 100.0,
            "CMO must stay within [-100, 100], got {}",
            result[1]
        );
    }

    #[test]
    fn cmo_wilder_golden_period3() {
        // Hand-computed CMO(3) with Wilder gain/loss (same seed as RSI).
        // close = [1, 2, 3, 2, 4, 3, 5]
        // changes: +1, +1, -1, +2, -1, +2
        //
        // Seed (first 3 changes): avg_gain=2/3, avg_loss=1/3
        // CMO[3] = 100*(2/3-1/3)/(2/3+1/3) = 100/3
        // i=4: g=10/9, l=2/9 → 200/3
        // i=5: g=20/27, l=13/27 → 700/33
        // i=6: g=94/81, l=26/81 → 170/3
        //
        // A plain rolling sum (the old bug) also yields 100/3 at index 3,
        // but 50.0 at index 4 — so 200/3 is the Wilder lock-in.
        let close = [1.0, 2.0, 3.0, 2.0, 4.0, 3.0, 5.0];
        let result = cmo(&close, 3);
        for (i, &v) in result.iter().enumerate().take(3) {
            assert!(v.is_nan(), "expected NaN warmup at {i}, got {v}");
        }
        let expected = [100.0 / 3.0, 200.0 / 3.0, 700.0 / 33.0, 170.0 / 3.0];
        for (offset, &exp) in expected.iter().enumerate() {
            let i = 3 + offset;
            assert!(
                result[i].is_finite(),
                "expected finite CMO at {i}, got {}",
                result[i]
            );
            assert!(
                (result[i] - exp).abs() < 1e-10,
                "CMO[{i}]: got {} expected {exp} (must be Wilder, not rolling sum)",
                result[i]
            );
        }
        // Same seed as RSI: CMO = 2*RSI - 100 on this non-flat series.
        let rsi_vals = rsi(&close, 3);
        for i in 3..close.len() {
            assert!((result[i] - (2.0 * rsi_vals[i] - 100.0)).abs() < 1e-10);
        }
    }

    // -- Wilder reciprocal hoist (rsi / cmo) ---------------------------------

    /// `rsi` with the original `/ p` division form.
    fn reference_rsi_div(close: &[f64], timeperiod: usize) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];
        if n <= timeperiod || timeperiod < 1 {
            return result;
        }
        let mut avg_gain = 0.0_f64;
        let mut avg_loss = 0.0_f64;
        for i in 1..=timeperiod {
            let diff = close[i] - close[i - 1];
            let abs_diff = diff.abs();
            avg_gain += (diff + abs_diff) * 0.5;
            avg_loss += (abs_diff - diff) * 0.5;
        }
        avg_gain /= timeperiod as f64;
        avg_loss /= timeperiod as f64;
        let p = timeperiod as f64;
        let rsi_val = |gain: f64, loss: f64| {
            let denom = gain + loss;
            if denom == 0.0 {
                0.0
            } else {
                (100.0 * gain / denom).clamp(0.0, 100.0)
            }
        };
        result[timeperiod] = rsi_val(avg_gain, avg_loss);
        for i in (timeperiod + 1)..n {
            let diff = close[i] - close[i - 1];
            let abs_diff = diff.abs();
            let gain = (diff + abs_diff) * 0.5;
            let loss = (abs_diff - diff) * 0.5;
            avg_gain = (avg_gain * (p - 1.0) + gain) / p;
            avg_loss = (avg_loss * (p - 1.0) + loss) / p;
            result[i] = rsi_val(avg_gain, avg_loss);
        }
        result
    }

    /// `cmo` with the original `/ p` division form.
    fn reference_cmo_div(close: &[f64], timeperiod: usize) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];
        if timeperiod < 1 || n <= timeperiod {
            return result;
        }
        let mut avg_gain = 0.0_f64;
        let mut avg_loss = 0.0_f64;
        for i in 1..=timeperiod {
            let diff = close[i] - close[i - 1];
            let abs_diff = diff.abs();
            avg_gain += (diff + abs_diff) * 0.5;
            avg_loss += (abs_diff - diff) * 0.5;
        }
        avg_gain /= timeperiod as f64;
        avg_loss /= timeperiod as f64;
        let cmo_val = |gain: f64, loss: f64| {
            let denom = gain + loss;
            if denom == 0.0 {
                0.0
            } else {
                (100.0 * (gain - loss) / denom).clamp(-100.0, 100.0)
            }
        };
        result[timeperiod] = cmo_val(avg_gain, avg_loss);
        let p = timeperiod as f64;
        for i in (timeperiod + 1)..n {
            let diff = close[i] - close[i - 1];
            let abs_diff = diff.abs();
            avg_gain = (avg_gain * (p - 1.0) + (diff + abs_diff) * 0.5) / p;
            avg_loss = (avg_loss * (p - 1.0) + (abs_diff - diff) * 0.5) / p;
            result[i] = cmo_val(avg_gain, avg_loss);
        }
        result
    }

    /// The recurrence is a contraction (`(p - 1) / p < 1`), so the at-most-one-
    /// ulp-per-step difference between `* inv_p` and `/ p` decays instead of
    /// accumulating. Steady-state relative error is `O(p * eps)` — under
    /// `1e-14` at these periods, six orders inside the `atol = 1e-8` gate.
    /// The bound is asserted over 20k bars so any *accumulating* error would
    /// show up.
    #[test]
    fn wilder_reciprocal_hoist_stays_within_one_ulp_per_step() {
        let n = 20_000;
        let (_, _, close) = ultosc_ohlc(n);
        for tp in [1, 2, 3, 14, 20, 64] {
            let want = reference_rsi_div(&close, tp);
            let got = rsi(&close, tp);
            assert_eq!(got.len(), want.len(), "rsi tp={tp}: length");
            let want_cmo = reference_cmo_div(&close, tp);
            let got_cmo = cmo(&close, tp);
            for i in 0..n {
                assert_eq!(got[i].is_nan(), want[i].is_nan(), "rsi tp={tp} [{i}]");
                if !want[i].is_nan() {
                    assert!(
                        (got[i] - want[i]).abs() <= 1e-11,
                        "rsi tp={tp} [{i}]: {} vs {}",
                        got[i],
                        want[i]
                    );
                    assert!((0.0..=100.0).contains(&got[i]), "rsi range tp={tp} [{i}]");
                }
                assert_eq!(
                    got_cmo[i].is_nan(),
                    want_cmo[i].is_nan(),
                    "cmo tp={tp} [{i}]"
                );
                if !want_cmo[i].is_nan() {
                    assert!(
                        (got_cmo[i] - want_cmo[i]).abs() <= 1e-11,
                        "cmo tp={tp} [{i}]: {} vs {}",
                        got_cmo[i],
                        want_cmo[i]
                    );
                    assert!(
                        (-100.0..=100.0).contains(&got_cmo[i]),
                        "cmo range tp={tp} [{i}]"
                    );
                }
            }
        }
    }
}
