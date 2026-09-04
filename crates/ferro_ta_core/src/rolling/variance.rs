//! Running mean and population variance over a fixed-size window.

use super::reseed::{
    is_subnormal_magnitude, NonFiniteGuard, MAX_ACCUMULATOR_RANGE, RESEED_INTERVAL,
};

/// Running mean and population variance over a fixed-size window, via
/// Welford's rolling update.
///
/// # Rolling Welford
///
/// We maintain `mean` and `m2` (the sum of squared deviations from the current
/// mean). When `x_new` replaces `x_old` and the window size `N` is constant:
///
/// ```text
/// delta     = x_new - x_old
/// old_mean  = mean
/// mean     += delta / N
/// m2       += delta * ((x_new - mean) + (x_old - old_mean))
///
/// variance  = m2 / N               // population variance
/// ```
///
/// That is algebraically the two separate Welford steps (remove `x_old`, add
/// `x_new`) without the intermediate `N - 1` state. The initial window is
/// seeded with the standard incremental Welford recurrence.
///
/// # Accuracy contract
///
/// `m2` is clamped at zero, which guarantees `sqrt` a non-negative argument for
/// any *finite* `m2`. It is not an accuracy defence: a negative `m2` means the
/// recurrence already lost every significant bit, and the clamp reports that as
/// `0.0`. It is also inert for a `NaN` `m2` (`NaN < 0.0` is false), which the
/// rolling update can produce from entirely finite input by overflowing `m2` to
/// `+inf` and then adding `-inf`.
///
/// Accuracy is instead maintained by [`Self::needs_reseed`], which forces an
/// exact O(p) recompute whenever
///
/// * `mean` or `m2` has gone non-finite over an all-finite window,
/// * `m2` has collapsed by more than [`MAX_ACCUMULATOR_RANGE`] from its peak
///   since the last recompute (a dominating bar left the window), or
/// * [`RESEED_INTERVAL`] advances have passed.
///
/// Driven through [`Self::advance`], the reported population variance therefore
/// stays within `2^-26` relative of an exact two-pass recompute, and is `NaN`
/// only where a two-pass recompute over the same window would also be.
///
/// # Why not `Σx² / N − mean²`
///
/// That naive form (which TA-Lib itself uses) has relative error scaling as
/// `(mean / sigma)^2 * eps`. At price 100 / sigma 0.05 that is ~1e-9 relative
/// and invisible; at index or crypto levels (price 1e5, sigma 0.01) it is ~2e-2
/// **relative** — a 2% error — and `m2` can go negative outright. Welford
/// avoids the cancellation entirely.
pub(crate) struct RollingVariance {
    mean: f64,
    m2: f64,
    /// Window size as `f64`, to keep the conversion out of the hot loop.
    p: f64,
    since_reseed: usize,
    /// Largest value `m2` has held since the last exact recompute. See
    /// [`MAX_ACCUMULATOR_RANGE`].
    ///
    /// Tracking the peak of `m2` itself — rather than the peak input magnitude —
    /// is what makes the trigger scale-free: the residue left behind when a
    /// dominating bar leaves the window is bounded by the ulp of the largest
    /// `m2` the accumulator ever held, so the ratio `m2_max / m2` *is* the
    /// precision-loss factor, in the same units, needing no notion of the
    /// window's price level. It also means a constant window (`m2 == m2_max ==
    /// 0`) never triggers, so the flat-series case costs nothing.
    m2_max: f64,
    guard: NonFiniteGuard,
}

impl RollingVariance {
    /// Seed from the first full window (length `timeperiod`, must be non-empty).
    pub(crate) fn new(window: &[f64]) -> Self {
        let mut out = Self {
            mean: 0.0,
            m2: 0.0,
            p: window.len() as f64,
            since_reseed: 0,
            m2_max: 0.0,
            guard: NonFiniteGuard::seed(window),
        };
        out.seed_from(window);
        out
    }

    /// Exact two-pass recompute over the whole window: `mean = sum / p`, then
    /// `m2 = sum (x - mean)^2`.
    ///
    /// # Why two-pass and not incremental Welford
    ///
    /// Incremental Welford is the right *streaming* seed when the window length
    /// is not known up front. Here it is: the window is a slice, so the mean is
    /// available in one pass and the deviations in a second. Two passes are
    /// strictly more accurate, cost the same O(p), and — being the identical
    /// expression the two-pass reference and the fuzz target evaluate — make the
    /// seed agree with them to the last bit rather than to a tolerance.
    ///
    /// That matters in the **subnormal** range, where incremental Welford loses
    /// several relative digits: below `f64::MIN_POSITIVE` the mantissa is
    /// truncated, so `delta / count` throws away bits the two-pass form never
    /// forms in the first place (subnormal addition and subtraction are exact,
    /// the exponent being pinned at its minimum). At `x ~ 1e-323` the
    /// incremental seed was ~4e-6 relative off a two-pass recompute — around
    /// three of the two or three significant bits such a value even has.
    ///
    /// `m2` is a sum of squares, so it is non-negative or `NaN` by construction
    /// and needs no clamp; the final value is also its peak, so `m2_max` needs
    /// no per-step tracking.
    fn seed_from(&mut self, window: &[f64]) {
        debug_assert!(!window.is_empty());
        // A plain left-to-right sum, deliberately not `simd::sum`. Lane-wise
        // reassociation would buy a few percent on a path that runs O(p) work
        // once per reseed, and would cost bit-identity with the canonical
        // `window.iter().sum::<f64>() / p` two-pass form. Where `m2` is
        // subnormal that identity is the whole ballgame: `m2` then has only a
        // handful of significant bits, so a one-ulp difference in `mean` can
        // move the result by percent-level *relative* error. See the subnormal
        // branch of `range_is_lossy`.
        let mean = window.iter().sum::<f64>() / self.p;
        let mut m2 = 0.0_f64;
        for &x in window {
            let d = x - mean;
            m2 += d * d;
        }
        debug_assert!(m2 >= 0.0 || m2.is_nan(), "seed m2 went negative: {m2}");
        self.mean = mean;
        self.m2 = m2;
        self.m2_max = m2;
    }

    /// True when the accumulator has lost the [`MAX_ACCUMULATOR_RANGE`]
    /// precision guarantee: `m2` has collapsed relative to the peak `m2` the
    /// accumulator absorbed, or `m2` / `mean` has gone non-finite over a window
    /// that is entirely finite (pure damage, never a faithful result).
    ///
    /// Cannot spin, for the same reason as
    /// [`super::RollingSum::range_is_lossy`]: `m2_max` is reset to `m2` by
    /// every reseed and is measured in the same units, so the condition is
    /// always cleared by an exact recompute. Neither accumulator carries a
    /// stall flag; one was written for the sum and removed as unsound.
    #[inline]
    fn range_is_lossy(&self) -> bool {
        self.guard.window_is_clean()
            && (!self.m2.is_finite()
                || !self.mean.is_finite()
                || self.m2_max > MAX_ACCUMULATOR_RANGE * self.m2
                || is_subnormal_magnitude(self.m2))
    }

    /// Current window mean.
    #[inline]
    pub(crate) fn mean(&self) -> f64 {
        self.mean
    }

    /// Current population variance.
    ///
    /// Non-negative whenever it is a number at all; `NaN` is reachable only
    /// while the window itself holds a non-finite value (see the accuracy
    /// contract on the type), which [`Self::advance`] clears as soon as that
    /// value leaves.
    #[inline]
    pub(crate) fn population_var(&self) -> f64 {
        self.m2 / self.p
    }

    /// Slide by one bar: `x_new` enters, `x_old` leaves. Raw recurrence only.
    #[inline]
    pub(crate) fn push_pop(&mut self, x_new: f64, x_old: f64) {
        let delta = x_new - x_old;
        let old_mean = self.mean;
        self.mean += delta / self.p;
        self.m2 += delta * ((x_new - self.mean) + (x_old - old_mean));
        // Guarantees `sqrt` a non-negative argument, and nothing else. A
        // negative `m2` here means the recurrence has *already* lost every
        // significant bit, and this clamp reports that total loss as a
        // confident `0.0` — so it must never be the only thing standing between
        // a damaged accumulator and the caller. `range_is_lossy` is what
        // actually detects the loss; see `MAX_ACCUMULATOR_RANGE`. Note it also
        // does nothing for a `NaN` `m2`, since `NaN < 0.0` is false.
        if self.m2 < 0.0 {
            self.m2 = 0.0;
        }
        self.since_reseed += 1;
        // `f64::max` returns the other operand for a NaN, so this cannot poison
        // the tracker; the non-finite branch of `range_is_lossy` covers NaN.
        self.m2_max = self.m2_max.max(self.m2);
        self.guard.slide(x_new, x_old);
    }

    /// True when the accumulator should be recomputed exactly: drift has had
    /// [`RESEED_INTERVAL`] chances to accumulate, a non-finite value entered and
    /// has since left the window, or a dominating bar has left the window and
    /// taken more than [`MAX_ACCUMULATOR_RANGE`] of `m2`'s precision with it.
    #[inline]
    pub(crate) fn needs_reseed(&self) -> bool {
        self.since_reseed >= RESEED_INTERVAL
            || self.guard.needs_recompute()
            || self.range_is_lossy()
    }

    /// Recompute mean and `m2` exactly from `window` (the current window).
    pub(crate) fn reseed(&mut self, window: &[f64]) {
        debug_assert_eq!(window.len(), self.p as usize);
        self.seed_from(window);
        self.since_reseed = 0;
        self.guard.clear();
    }

    /// Slide by one bar, reseeding exactly when required.
    ///
    /// `window` must be the window **after** the slide.
    #[inline]
    pub(crate) fn advance(&mut self, x_new: f64, x_old: f64, window: &[f64]) {
        self.push_pop(x_new, x_old);
        if self.needs_reseed() {
            self.reseed(window);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rolling::test_support::drive_variance;

    #[test]
    fn rolling_variance_matches_a_recomputed_welford() {
        // Large mean, tiny sigma — the case that breaks `Σx²/N − mean²`.
        let data: Vec<f64> = (0..400)
            .map(|i| 100_000.0 + ((i % 11) as f64) * 0.01)
            .collect();
        let p = 20;
        let mut rv = RollingVariance::new(&data[..p]);
        for i in p..data.len() {
            let window = &data[i + 1 - p..=i];
            rv.advance(data[i], data[i - p], window);
            let mean = window.iter().sum::<f64>() / p as f64;
            let var = window.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / p as f64;
            assert!((rv.mean() - mean).abs() < 1e-8, "mean i={i}");
            assert!(rv.population_var() >= 0.0, "negative variance i={i}");
            // Rolling Welford drifts a few ULPs of `mean^2` relative to a
            // fresh recompute, which at this mean/sigma ratio (1e7) shows up
            // as ~1e-9 *relative* error in the variance. The point of the case
            // is the contrast: the naive `Σx²/N − mean²` form is ~2e-2
            // relative here, seven orders of magnitude worse, and can even
            // produce a negative `m2`.
            assert!(
                (rv.population_var() - var).abs() < 1e-6 * var.max(1e-12),
                "var i={i} got={} want={var}",
                rv.population_var()
            );
        }
    }

    #[test]
    fn rolling_variance_recovers_once_a_nan_leaves_the_window() {
        let mut data: Vec<f64> = (0..40).map(|i| 10.0 + (i % 3) as f64).collect();
        data[15] = f64::NAN;
        let p = 5;
        let mut rv = RollingVariance::new(&data[..p]);
        for i in p..data.len() {
            let window = &data[i + 1 - p..=i];
            rv.advance(data[i], data[i - p], window);
            if !window.iter().any(|x| x.is_nan()) {
                let mean = window.iter().sum::<f64>() / p as f64;
                assert!((rv.mean() - mean).abs() < 1e-9, "i={i} mean={}", rv.mean());
            }
        }
    }

    fn exact_population_var(window: &[f64]) -> f64 {
        let p = window.len() as f64;
        let mean = window.iter().sum::<f64>() / p;
        window.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / p
    }

    fn assert_variance_exact(data: &[f64], p: usize, rtol: f64, label: &str) {
        let (got, _) = drive_variance(data, p);
        for (k, &g) in got.iter().enumerate() {
            let i = k + p - 1;
            let want = exact_population_var(&data[i + 1 - p..=i]);
            if want.is_nan() {
                assert!(g.is_nan(), "{label} i={i}: want NaN got {g}");
            } else if want.is_infinite() {
                assert!(g.is_infinite(), "{label} i={i}: want inf got {g}");
            } else {
                assert!(
                    (g - want).abs() <= rtol * want.abs() + f64::MIN_POSITIVE,
                    "{label} i={i}: got={g} want={want}"
                );
            }
        }
    }

    /// A `1e8` spike on a price-100 series: the accumulator's `m2` is pure
    /// rounding residue once the spike leaves, which the `m2 < 0.0` clamp used
    /// to round up to a hard `0.0`.
    #[test]
    fn variance_recovers_after_a_finite_spike_leaves_the_window() {
        let data = [100.0, 100.5, 101.0, 1e8, 100.2, 100.3, 100.4, 100.5, 100.6];
        assert_variance_exact(&data, 3, 1e-9, "1e8 spike");
        let (got, reseeds) = drive_variance(&data, 3);
        // The spike-free windows must have real dispersion, not a clamped zero.
        for (k, &g) in got.iter().enumerate().skip(4) {
            assert!(g > 0.0, "window {k} pinned to {g}");
        }
        // Exactly one recompute: the bar on which the spike left.
        assert_eq!(reseeds, 1, "expected one dynamic-range reseed");
    }

    /// Sweeping the spike magnitude: the pre-fix error was non-monotone in it
    /// (1e8 gave a hard zero while 1e9 and 1e11 gave nonzero-but-wrong values),
    /// which is the signature of precision loss rather than a logic error.
    #[test]
    fn variance_is_exact_across_the_whole_spike_magnitude_sweep() {
        for exp in 2..=150 {
            let spike = 10f64.powi(exp);
            let data = [
                100.0, 100.5, 101.0, spike, 100.2, 100.3, 100.4, 100.5, 100.6,
            ];
            assert_variance_exact(&data, 3, 1e-9, &format!("spike 1e{exp}"));
        }
    }

    /// `m2` overflows to `+inf`, then the next slide adds `-inf` and leaves
    /// `NaN`. The clamp is inert (`NaN < 0.0` is false), so without the
    /// non-finite-accumulator trigger the `NaN` is permanent.
    #[test]
    fn variance_recovers_from_an_m2_overflow_to_nan() {
        let mut data = vec![1e300, -1e300];
        data.extend((0..30).map(|k| 100.0 + 0.5 * k as f64));
        assert_variance_exact(&data, 3, 1e-12, "m2 overflow");
        let (got, _) = drive_variance(&data, 3);
        // The two overflowing windows report `+inf` — exactly what a two-pass
        // recompute over the same bars yields — and every later window is
        // finite and correct.
        assert!(got[0].is_infinite() && got[1].is_infinite());
        for (k, &g) in got.iter().enumerate().skip(2) {
            assert!(g.is_finite(), "window {k} is {g}");
            assert!((g - 1.0 / 6.0).abs() < 1e-15, "window {k} got {g}");
        }
    }

    /// Subnormal / huge-exponent mix: bar 4 was off by 68 orders of magnitude
    /// (`5.635e270` against an exact `1.742e202`).
    #[test]
    fn variance_survives_a_subnormal_and_huge_exponent_mix() {
        let data = [1.39e-309, 3.05e143, 2.80e101, -7.34e83, 2.20e-106];
        assert_variance_exact(&data, 3, 1e-12, "exponent mix");
    }

    /// A regime change *does* have to fire — once per collapse, not once per
    /// bar. `m2_max` is reset by the recompute, so the trigger is self-limiting.
    #[test]
    fn a_variance_collapse_costs_one_recompute_not_one_per_bar() {
        let p = 5;
        // A 30 -> 0.01 amplitude collapse: a 9e6-fold drop in `m2`, far past
        // `MAX_ACCUMULATOR_RANGE`, while staying inside the conditioning where
        // a *relative* comparison to a two-pass recompute means anything
        // (`mean / sigma ~ 1e4`, so `eps * (mean / sigma)^2 ~ 2e-8`).
        let mut data: Vec<f64> = (0..200).map(|i| 100.0 + 30.0 * ((i % 3) as f64)).collect();
        for x in data.iter_mut().skip(100) {
            *x = 100.0 + 0.01 * ((*x - 100.0) / 30.0);
        }
        assert_variance_exact(&data, p, 1e-6, "regime collapse");
        let (_, reseeds) = drive_variance(&data, p);
        assert!(
            reseeds <= p + 1,
            "a single variance collapse cost {reseeds} recomputes"
        );
    }

    /// Values around `1e-161` are perfectly normal `f64`s, but their squared
    /// deviations are **subnormal**, so `m2` itself has only a handful of
    /// significant bits and the rolling recurrence loses whole percent of it
    /// per step. `MAX_ACCUMULATOR_RANGE` cannot see this — the dynamic range is
    /// 1 — which is why `is_subnormal_magnitude` is a trigger of its own.
    #[test]
    fn subnormal_m2_forces_an_exact_recompute() {
        let base = 1.7e-161;
        let data: Vec<f64> = (0..40)
            .map(|i| base * (1.0 + 0.25 * ((i % 5) as f64)))
            .collect();
        // `m2` really is subnormal here, or the test proves nothing.
        let probe = exact_population_var(&data[..2]);
        assert!(
            probe > 0.0 && probe < f64::MIN_POSITIVE,
            "premise broken: m2 is {probe}, not subnormal"
        );
        for &p in &[2usize, 3, 7] {
            // Bit-exact, not merely close: the recompute evaluates the same
            // canonical two-pass expression as `exact_population_var`.
            let (got, _) = drive_variance(&data, p);
            for (k, &g) in got.iter().enumerate() {
                let i = k + p - 1;
                let want = exact_population_var(&data[i + 1 - p..=i]);
                assert_eq!(
                    g.to_bits(),
                    want.to_bits(),
                    "p={p} i={i}: got={g:e} want={want:e}"
                );
            }
        }
    }
}
