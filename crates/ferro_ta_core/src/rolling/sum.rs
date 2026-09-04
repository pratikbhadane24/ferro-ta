//! Running sum over a fixed-size window.

use super::reseed::{
    is_subnormal_magnitude, NonFiniteGuard, MAX_ACCUMULATOR_RANGE, RESEED_INTERVAL,
};

/// Running sum over a fixed-size window, usable *inside* another kernel's loop.
///
/// `math_ops::rolling_sum` computes the same thing but returns a `Vec`, so it
/// cannot be composed into a kernel that already owns the traversal.
///
/// Prefer [`Self::advance`], which folds the drift reseed, the non-finite
/// recovery and the dynamic-range recovery into the same call.
///
/// # Accuracy contract
///
/// A sum's error is *first order* in the magnitudes it absorbs, unlike a second
/// moment, so it degrades far more gently — but it still degrades absolutely.
/// `[1e300, 1.0, 1.0, 1.0]` at `timeperiod = 3` is the extreme: `1e300 + 1 + 1`
/// rounds back to `1e300`, so when `1e300` leaves the recurrence subtracts it
/// from itself and reports `0.0` for a window whose true sum is `3.0`.
/// [`MAX_ACCUMULATOR_RANGE`] bounds that: the reported sum stays within `2^-26`
/// of an exact recompute over the same window, relative to
/// `max(|sum|, |x_new|)` (see [`Self::cur_mag`] for why the incoming bar is in
/// the denominator, and why for a sum of non-negative terms — every in-tree
/// consumer — that is simply `2^-26` relative to the sum).
pub(crate) struct RollingSum {
    sum: f64,
    /// Advances since the last exact recompute.
    since_reseed: usize,
    /// Largest magnitude the sum has held, or absorbed as an incoming bar,
    /// since the last exact recompute. See [`MAX_ACCUMULATOR_RANGE`].
    ///
    /// The incoming bar counts as well as the running sum: `sum += x_new -
    /// x_old` rounds the difference to the ulp of `max(|x_new|, |x_old|)`, which
    /// a window holding two large values of opposite sign can push above any
    /// magnitude `sum` itself ever reaches.
    mag_max: f64,
    /// Proxy for the magnitude scale of the *current* window: the larger of
    /// `|sum|` and the magnitude of the bar that just entered.
    ///
    /// This, and not `|sum|` alone, is what `mag_max` is measured against.
    /// A window whose values cancel — `[1.0, -1.0]` — has `|sum| == 0` with no
    /// error whatsoever, and comparing against zero would force an exact
    /// recompute on every single bar of an alternating series. `|x_new|` keeps
    /// the denominator on the scale of the data instead of the scale of the
    /// cancellation. For a sum of *non-negative* terms — which is every
    /// in-tree consumer (`ultosc`'s buying pressure and true range,
    /// `choppiness`'s true range) — `|sum| >= |x_new|` always, so the two
    /// coincide and the guarantee is relative to `|sum|` as usual.
    cur_mag: f64,
    guard: NonFiniteGuard,
}

impl RollingSum {
    /// Seed from the first full window (length `timeperiod`).
    pub(crate) fn new(window: &[f64]) -> Self {
        let mut out = Self {
            sum: 0.0,
            since_reseed: 0,
            mag_max: 0.0,
            cur_mag: 0.0,
            guard: NonFiniteGuard::seed(window),
        };
        out.seed_from(window);
        out
    }

    /// Recompute the sum and the magnitude tracker exactly from `window`.
    fn seed_from(&mut self, window: &[f64]) {
        self.sum = crate::simd::sum(window);
        // The seed sum absorbed every magnitude in the window, so the tracker
        // has to start from the largest of them and not merely from `|sum|`.
        self.mag_max = window
            .iter()
            .fold(self.sum.abs(), |acc, x| acc.max(x.abs()));
        // At the seed the whole window is the "recent" data, so its own
        // magnitude is the scale to measure against. Both trackers therefore
        // start equal, and `range_is_lossy` is necessarily false: an exact
        // recompute always clears the range trigger, which is why this
        // accumulator needs no stall flag to keep from spinning.
        self.cur_mag = self.mag_max;
    }

    /// True when the sum has gone non-finite over a window that is entirely
    /// finite. That is pure damage — an overflow to `±inf` followed by the
    /// opposite infinity leaves a permanent `NaN` — and is never stalled: an
    /// exact recompute either clears it or reproduces a faithful `±inf`. In the
    /// latter case — an all-finite window whose sum genuinely overflows — it
    /// recomputes on every bar for as long as such a window persists, which
    /// costs O(p) per bar but never returns a wrong answer.
    #[inline]
    fn accumulator_is_damaged(&self) -> bool {
        self.guard.window_is_clean() && !self.sum.is_finite()
    }

    /// True when the sum has absorbed a magnitude that dominates its current
    /// value badly enough to have destroyed [`MAX_ACCUMULATOR_RANGE`]'s
    /// precision guarantee.
    ///
    /// Cannot spin: `mag_max` is reset by every reseed and `cur_mag` tracks the
    /// current window, so an exact recompute always clears the condition. An
    /// earlier revision carried a stall flag for this and it was removed as
    /// unsound — it latched on a structurally-cancelling window (`[1, -1, 1,
    /// -1, …]`, where `|sum|` is 0 with no error at all) and then suppressed a
    /// genuine fire two bars later. Using `max(|sum|, |x_new|)` as the
    /// denominator keeps the comparison on the scale of the *data* rather than
    /// the scale of the cancellation, which is what removed the need for it.
    #[inline]
    fn range_is_lossy(&self) -> bool {
        self.guard.window_is_clean()
            && self.sum.is_finite()
            && (self.mag_max > MAX_ACCUMULATOR_RANGE * self.cur_mag
                || is_subnormal_magnitude(self.sum))
    }

    /// Current window sum.
    #[inline]
    pub(crate) fn value(&self) -> f64 {
        self.sum
    }

    /// Slide by one bar: `x_new` enters, `x_old` leaves. Returns the new sum.
    ///
    /// Raw recurrence — no reseed, no non-finite recovery. Use
    /// [`Self::advance`] unless you are driving those yourself.
    #[inline]
    pub(crate) fn push_pop(&mut self, x_new: f64, x_old: f64) -> f64 {
        self.sum += x_new - x_old;
        self.since_reseed += 1;
        // `f64::max` returns the other operand for a NaN, so a non-finite bar
        // cannot poison either tracker; `NonFiniteGuard` covers that case.
        self.cur_mag = self.sum.abs().max(x_new.abs());
        self.mag_max = self.mag_max.max(self.cur_mag);
        self.guard.slide(x_new, x_old);
        self.sum
    }

    /// True when the accumulator should be recomputed exactly: drift has had
    /// [`RESEED_INTERVAL`] chances to accumulate, a non-finite value entered and
    /// has since left the window, or an absorbed magnitude now dominates the sum
    /// by more than [`MAX_ACCUMULATOR_RANGE`].
    #[inline]
    pub(crate) fn needs_reseed(&self) -> bool {
        self.since_reseed >= RESEED_INTERVAL
            || self.guard.needs_recompute()
            || self.accumulator_is_damaged()
            || self.range_is_lossy()
    }

    /// Recompute the sum exactly from `window` (the current window).
    pub(crate) fn reseed(&mut self, window: &[f64]) -> f64 {
        self.since_reseed = 0;
        self.guard.clear();
        self.seed_from(window);
        self.sum
    }

    /// Slide by one bar, reseeding exactly when required.
    ///
    /// `window` must be the window **after** the slide, i.e. the `timeperiod`
    /// bars ending at the bar `x_new` came from.
    #[inline]
    pub(crate) fn advance(&mut self, x_new: f64, x_old: f64, window: &[f64]) -> f64 {
        self.push_pop(x_new, x_old);
        if self.needs_reseed() {
            self.reseed(window);
        }
        self.sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rolling::test_support::drive_sum;

    #[test]
    fn rolling_sum_tracks_the_two_pass_sum() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64) * 0.25 + 3.0).collect();
        let p = 7;
        let mut rs = RollingSum::new(&data[..p]);
        for i in p..data.len() {
            let got = rs.advance(data[i], data[i - p], &data[i + 1 - p..=i]);
            let want: f64 = data[i + 1 - p..=i].iter().sum();
            assert!((got - want).abs() < 1e-9, "i={i} got={got} want={want}");
        }
    }

    #[test]
    fn rolling_sum_recovers_once_a_nan_leaves_the_window() {
        let mut data: Vec<f64> = (0..30).map(|i| i as f64).collect();
        data[10] = f64::NAN;
        let p = 4;
        let mut rs = RollingSum::new(&data[..p]);
        for i in p..data.len() {
            let got = rs.advance(data[i], data[i - p], &data[i + 1 - p..=i]);
            let window = &data[i + 1 - p..=i];
            if window.iter().any(|x| x.is_nan()) {
                assert!(got.is_nan(), "i={i} expected NaN, got {got}");
            } else {
                let want: f64 = window.iter().sum();
                assert!((got - want).abs() < 1e-9, "i={i} got={got} want={want}");
            }
        }
    }

    /// The `RollingSum` analogue of the spike case, and the reason the sum needs
    /// the same trigger despite its error being only first order: `1e300 + 1 + 1`
    /// rounds back to `1e300`, so when `1e300` leaves the recurrence subtracts it
    /// from itself and reports `0.0` for a window whose true sum is `3.0`.
    #[test]
    fn rolling_sum_recovers_after_a_dominating_value_leaves_the_window() {
        let mut data = vec![1e300];
        data.extend(std::iter::repeat_n(1.0, 10));
        let p = 3;
        let (got, reseeds) = drive_sum(&data, p);
        for (k, &g) in got.iter().enumerate() {
            let i = k + p - 1;
            let want: f64 = data[i + 1 - p..=i].iter().sum();
            assert!(
                (g - want).abs() <= 1e-12 * want.abs().max(1.0),
                "i={i} got={g} want={want}"
            );
        }
        assert_eq!(reseeds, 1, "expected one dynamic-range reseed");
    }

    /// A sum that overflows to `+inf` and then takes `-inf` becomes `NaN`
    /// permanently without the non-finite-accumulator trigger — the same
    /// class-1 defect as the variance's, with no `m2` involved.
    #[test]
    fn rolling_sum_recovers_from_an_overflow_to_nan() {
        let mut data = vec![1e308, 1e308, -1e308];
        data.extend((0..20).map(|k| 10.0 + k as f64));
        let p = 3;
        let (got, _) = drive_sum(&data, p);
        for (k, &g) in got.iter().enumerate() {
            let i = k + p - 1;
            let want: f64 = data[i + 1 - p..=i].iter().sum();
            if want.is_finite() {
                assert!(
                    (g - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "i={i} got={g} want={want}"
                );
            } else {
                assert!(!g.is_nan(), "i={i} got NaN where the exact sum is {want}");
            }
        }
    }

    /// The case the `cur_mag` denominator exists for: a window whose values
    /// cancel to a sum far smaller than themselves. `|sum|` alone as the
    /// denominator would force an exact recompute on every bar of an
    /// alternating series — an O(n * p) regression on data that is not remotely
    /// pathological, and one no correctness test would catch.
    #[test]
    fn rolling_sum_does_not_spin_on_a_structurally_cancelling_window() {
        let data: Vec<f64> = (0..500)
            .map(|i| if i % 2 == 0 { 1e6 } else { -1e6 })
            .collect();
        let (got, reseeds) = drive_sum(&data, 2);
        assert!(got.iter().all(|g| *g == 0.0), "exact sums are all zero");
        assert_eq!(reseeds, 0, "range trigger fired on a cancelling window");
    }

    /// The same for a subnormal *sum*.
    #[test]
    fn subnormal_sum_forces_an_exact_recompute() {
        let data: Vec<f64> = (0..40).map(|i| 1e-320 * (1.0 + (i % 3) as f64)).collect();
        let p = 3;
        let (got, _) = drive_sum(&data, p);
        for (k, &g) in got.iter().enumerate() {
            let i = k + p - 1;
            let want: f64 = data[i + 1 - p..=i].iter().sum();
            assert!(
                want > 0.0 && want < f64::MIN_POSITIVE,
                "premise broken: sum {want} is not subnormal"
            );
            assert_eq!(g.to_bits(), want.to_bits(), "i={i} got={g:e} want={want:e}");
        }
    }
}
