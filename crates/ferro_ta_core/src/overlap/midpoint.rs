//! Rolling midpoint / midprice.

// ---------------------------------------------------------------------------
// MIDPOINT / MIDPRICE
// ---------------------------------------------------------------------------

/// NaN-propagating window maximum.
///
/// `f64::max` **ignores** `NaN` — `f64::max(f64::NAN, 3.0) == 3.0` — so
/// reducing a window with it lets a contaminated window produce a finite
/// answer computed from whatever values happened to be clean. This returns
/// `NaN` for such a window instead. For all-finite input the result is the
/// same value `f64::max` would have produced.
#[inline]
fn window_max(window: &[f64]) -> f64 {
    let mut mx = f64::NEG_INFINITY;
    for &v in window {
        if v.is_nan() {
            return f64::NAN;
        }
        if v > mx {
            mx = v;
        }
    }
    mx
}

/// NaN-propagating window minimum. See [`window_max`].
#[inline]
fn window_min(window: &[f64]) -> f64 {
    let mut mn = f64::INFINITY;
    for &v in window {
        if v.is_nan() {
            return f64::NAN;
        }
        if v < mn {
            mn = v;
        }
    }
    mn
}

/// Midpoint: `(max(close) + min(close)) / 2` over rolling window.
///
/// # NaN semantics
///
/// The first `timeperiod - 1` values are `NaN` (warm-up). Beyond that, a
/// window containing a `NaN` yields `NaN` — the contamination is propagated,
/// matching every other kernel in this crate (see the `rolling` module's
/// bit-identity contract, where a `NaN` can legitimately surface as the window
/// extreme). TA-Lib's `TA_MIDPOINT` is written in C over raw `double`
/// comparisons, which silently *skip* `NaN` inputs; that is an artifact of
/// `<`/`>` being false for `NaN` rather than a defined stance, and TA-Lib does
/// not specify behaviour for non-finite input. Propagation is therefore the
/// defensible choice: parity against TA-Lib is only asserted on finite series,
/// and for finite input the two agree exactly.
///
/// `±inf` is *not* special-cased: a window of all `+inf` gives `+inf`, and a
/// window spanning both infinities gives `NaN` from `inf + -inf`.
pub fn midpoint(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    if timeperiod == 0 || n < timeperiod {
        return vec![f64::NAN; n];
    }
    // Indexed stores over a pre-filled `NaN` buffer, not `push`. Note that a
    // `NaN` written by the loop is a *computed* `NaN` from a contaminated
    // window, distinct from the untouched warm-up `NaN`s --- the propagation
    // documented above is unchanged.
    let mut result = vec![f64::NAN; n];
    for i in (timeperiod - 1)..n {
        let window = &close[(i + 1 - timeperiod)..=i];
        let mx = window_max(window);
        let mn = window_min(window);
        result[i] = (mx + mn) / 2.0;
    }
    result
}

/// MidPrice: `(highest_high + lowest_low) / 2` over rolling window.
///
/// # NaN semantics
///
/// The first `timeperiod - 1` values are `NaN` (warm-up). Beyond that, a `NaN`
/// anywhere in the `high` window **or** the `low` window yields `NaN` — either
/// extreme becomes `NaN` and the sum carries it through. See [`midpoint`] for
/// why propagation is preferred over TA-Lib's incidental skip-`NaN` behaviour.
/// `±inf` is not special-cased.
pub fn midprice(high: &[f64], low: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = high.len();
    if timeperiod == 0 || n < timeperiod {
        return vec![f64::NAN; n];
    }
    let mut result = vec![f64::NAN; n];
    for i in (timeperiod - 1)..n {
        let start = i + 1 - timeperiod;
        let mx = window_max(&high[start..=i]);
        let mn = window_min(&low[start..=i]);
        result[i] = (mx + mn) / 2.0;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overlap::test_support::*;

    // -----------------------------------------------------------------------
    // Legacy oracles: the pre-fix `f64::max` / `f64::min` reductions, verbatim.
    // Used **only on all-finite input**, where the NaN fix is a no-op, to prove
    // that nothing but the NaN path moved.
    // -----------------------------------------------------------------------

    fn legacy_midpoint(close: &[f64], timeperiod: usize) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];
        if timeperiod == 0 || n < timeperiod {
            return result;
        }
        for i in (timeperiod - 1)..n {
            let window = &close[(i + 1 - timeperiod)..=i];
            let mx = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mn = window.iter().cloned().fold(f64::INFINITY, f64::min);
            result[i] = (mx + mn) / 2.0;
        }
        result
    }

    fn legacy_midprice(high: &[f64], low: &[f64], timeperiod: usize) -> Vec<f64> {
        let n = high.len();
        let mut result = vec![f64::NAN; n];
        if timeperiod == 0 || n < timeperiod {
            return result;
        }
        for i in (timeperiod - 1)..n {
            let start = i + 1 - timeperiod;
            let mx = high[start..=i]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let mn = low[start..=i].iter().cloned().fold(f64::INFINITY, f64::min);
            result[i] = (mx + mn) / 2.0;
        }
        result
    }

    #[test]
    fn finite_input_is_bit_identical_to_the_legacy_reduction() {
        let close = synthetic_series(600);
        let (high, low) = synthetic_hl(600);
        for &p in &[1usize, 2, 14, 30, 601] {
            assert_bits_eq(
                &midpoint(&close, p),
                &legacy_midpoint(&close, p),
                &format!("midpoint p={p}"),
            );
            assert_bits_eq(
                &midprice(&high, &low, p),
                &legacy_midprice(&high, &low, p),
                &format!("midprice p={p}"),
            );
        }
        // Degenerate period must still return exactly `n` NaNs.
        assert_eq!(midpoint(&close, 0).len(), 600);
        assert!(midpoint(&close, 0).iter().all(|v| v.is_nan()));
        assert_eq!(midprice(&high, &low, 0).len(), 600);
    }

    #[test]
    fn midpoint_propagates_nan_from_every_window_position() {
        let base = [10.0, 12.0, 14.0, 16.0, 18.0];
        let p = 3;
        // A NaN at index `j` contaminates every window that contains `j`,
        // i.e. outputs j ..= j + p - 1 (clipped to the series).
        for j in 0..base.len() {
            let mut series = base;
            series[j] = f64::NAN;
            let out = midpoint(&series, p);
            assert_eq!(out.len(), base.len());
            for (i, &v) in out.iter().enumerate() {
                if i < p - 1 {
                    assert!(v.is_nan(), "warmup at {i} (nan at {j})");
                    continue;
                }
                let window_covers_nan = j + p > i && j <= i;
                assert_eq!(
                    v.is_nan(),
                    window_covers_nan,
                    "nan at {j}: output {i} = {v} (window covers nan: {window_covers_nan})"
                );
            }
        }
    }

    #[test]
    fn midpoint_all_nan_window_is_nan_not_neg_inf_plus_inf() {
        // Regression guard for the old `fold` behaviour, which reduced an
        // all-NaN window to (NEG_INFINITY + INFINITY) / 2 -- also NaN, but for
        // the wrong reason -- while a *partially* NaN window silently produced
        // a finite number.
        let series = vec![f64::NAN; 5];
        assert!(midpoint(&series, 3).iter().all(|v| v.is_nan()));

        // The bug in one line: the middle value must not survive on its own.
        let contaminated = [f64::NAN, 7.0, f64::NAN];
        let out = midpoint(&contaminated, 3);
        assert!(out[2].is_nan(), "got {} -- NaN was swallowed", out[2]);
    }

    #[test]
    fn midprice_propagates_nan_from_high_or_low_independently() {
        let high = [11.0, 13.0, 15.0, 17.0];
        let low = [9.0, 11.0, 13.0, 15.0];
        let p = 2;

        // NaN in `high` only.
        let mut h = high;
        h[2] = f64::NAN;
        let out = midprice(&h, &low, p);
        assert!(out[0].is_nan(), "warmup");
        assert!(!out[1].is_nan());
        assert!(out[2].is_nan(), "high NaN not propagated: {}", out[2]);
        assert!(out[3].is_nan(), "high NaN not propagated: {}", out[3]);

        // NaN in `low` only.
        let mut l = low;
        l[2] = f64::NAN;
        let out = midprice(&high, &l, p);
        assert!(!out[1].is_nan());
        assert!(out[2].is_nan(), "low NaN not propagated: {}", out[2]);
        assert!(out[3].is_nan(), "low NaN not propagated: {}", out[3]);
    }

    #[test]
    fn infinities_are_not_special_cased() {
        // All +inf: the max is +inf, the min is +inf, the midpoint is +inf.
        let series = [f64::INFINITY; 3];
        assert_eq!(midpoint(&series, 3)[2], f64::INFINITY);

        // Both infinities in one window: inf + -inf == NaN.
        let series = [f64::INFINITY, 1.0, f64::NEG_INFINITY];
        assert!(midpoint(&series, 3)[2].is_nan());

        // A single -inf floors the minimum.
        let series = [1.0, f64::NEG_INFINITY, 3.0];
        assert_eq!(midpoint(&series, 3)[2], f64::NEG_INFINITY);

        // midprice: +inf high, finite low.
        let high = [f64::INFINITY, f64::INFINITY];
        let low = [1.0, 2.0];
        assert_eq!(midprice(&high, &low, 2)[1], f64::INFINITY);
    }

    #[test]
    fn midprice_finite_smoke() {
        let high = [1.0, 2.0, 3.0];
        let low = [0.0, 1.0, 2.0];
        let out = midprice(&high, &low, 2);
        assert_eq!(out.len(), 3);
        assert!(out[0].is_nan());
        assert_eq!(out[1], 1.0);
    }
}
