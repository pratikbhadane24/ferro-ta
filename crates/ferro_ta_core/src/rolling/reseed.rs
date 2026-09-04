//! Reseed policy shared by the running accumulators.

/// Number of window advances after which a running accumulator is recomputed
/// exactly from the current window.
///
/// The amortized cost is `timeperiod / 8192` operations per bar — effectively
/// free — and it bounds accumulated rounding drift at `O(sqrt(8192) * eps)`
/// regardless of series length. That independence from `n` is what makes the
/// chunked and streaming code paths agree with the one-shot path.
pub(crate) const RESEED_INTERVAL: usize = 8192;

/// Largest tolerated ratio between the biggest magnitude a running accumulator
/// has absorbed since its last exact recompute and the accumulator's current
/// value, before an exact recompute is *forced*.
///
/// # Why a second trigger at all
///
/// [`RESEED_INTERVAL`] bounds *drift* — the slow random walk of per-step
/// rounding on well-conditioned data. It does nothing about a **single**
/// perfectly finite bar whose magnitude dwarfs the rest of the window: that bar
/// destroys the accumulator's low-order bits the moment it is absorbed, and the
/// damage only becomes visible once it *leaves*, at which point the recurrence
/// has nothing left to subtract it from. Without this trigger the next up to
/// `RESEED_INTERVAL` outputs are wrong, and — because the `m2 < 0.0` clamp
/// rounds the wreckage up to `0.0` — wrong in a way that looks like a confident
/// "this window is constant".
///
/// # Derivation of the bound
///
/// Every update of a running accumulator rounds to a multiple of the ulp of the
/// magnitudes it is handling. If the largest magnitude absorbed since the last
/// exact recompute is `D` and there have been `N` updates since, the absolute
/// residue is `O(N * 2^-53 * D)` — `f64` carries a 53-bit mantissa. Measured
/// against the accumulator's current value `s`, the retained precision is
///
/// ```text
/// bits = 53 - log2(N) - log2(D / s)
/// ```
///
/// `N <= RESEED_INTERVAL = 2^13`, so forcing a recompute once `D / s > 2^14`
/// guarantees at least `53 - 13 - 14 = 26` bits, i.e. a **relative error below
/// `2^-26 ~= 1.5e-8`** on every value either accumulator reports — five orders
/// below the loosest tolerance any consumer or fuzz target applies, and two
/// orders below the crate's own `1e-6` TA-Lib gate.
///
/// For [`super::RollingVariance`] the ratio is taken between second moments, which are
/// already squared, so `2^14` on `m2` is a `2^7 = 128`-fold dynamic range in the
/// *values* — the point at which `sigma` is still resolved to 26 bits.
pub(super) const MAX_ACCUMULATOR_RANGE: f64 = 16_384.0;

/// True for a non-zero value below `f64::MIN_POSITIVE`, i.e. one in the
/// subnormal range.
///
/// [`MAX_ACCUMULATOR_RANGE`]'s derivation assumes the ulp is *relative* — a
/// fixed `2^-53` of the magnitude at hand. Below `f64::MIN_POSITIVE` the
/// exponent is pinned at its minimum and the ulp becomes **absolute**
/// (`2^-1074`), so a subnormal accumulator has however many significant bits
/// its exponent happens to leave it, sometimes fewer than ten. No running
/// recurrence has useful relative precision there whatever the dynamic range,
/// so both accumulators force an exact recompute instead, which reproduces the
/// canonical two-pass expression bit for bit.
///
/// It costs one comparison per bar and can only fire on a variance below
/// `1e-308` (deviations under ~`1.5e-154`) or a sum below `1e-308`.
#[inline]
pub(super) fn is_subnormal_magnitude(x: f64) -> bool {
    x != 0.0 && x.abs() < f64::MIN_POSITIVE
}

/// Tracks how many non-finite values are inside the window and whether the
/// accumulator has been contaminated by one.
///
/// Two-pass kernels have *localized* NaN semantics: a single `NaN` input
/// corrupts exactly the `timeperiod` outputs whose window contains it, then
/// results resume. A naive running accumulator regresses to permanent
/// poisoning, because `NaN - NaN` is `NaN`. Recording contamination and
/// recomputing exactly once the window is clean again preserves the two-pass
/// behaviour for one always-false `is_finite` compare per bar.
///
/// This guard covers non-finite *input* only. A finite input can still drive the
/// accumulator itself non-finite (overflow to `±inf`, then `inf + -inf = NaN`)
/// or merely destroy its precision; [`MAX_ACCUMULATOR_RANGE`] is what catches
/// those. [`Self::window_is_clean`] is the shared prerequisite: a non-finite
/// accumulator is damage worth recomputing only when the window holds nothing
/// non-finite of its own.
#[derive(Default)]
pub(super) struct NonFiniteGuard {
    /// Non-finite values currently inside the window.
    inside: usize,
    /// A non-finite value has entered since the last exact recompute.
    contaminated: bool,
}

impl NonFiniteGuard {
    pub(super) fn seed(window: &[f64]) -> Self {
        let inside = window.iter().filter(|x| !x.is_finite()).count();
        Self {
            inside,
            contaminated: inside > 0,
        }
    }

    #[inline]
    pub(super) fn slide(&mut self, x_new: f64, x_old: f64) {
        if !x_new.is_finite() {
            self.inside += 1;
            self.contaminated = true;
        }
        if !x_old.is_finite() {
            self.inside -= 1;
        }
    }

    /// True once every non-finite value has left the window but the
    /// accumulator still carries their damage.
    #[inline]
    pub(super) fn needs_recompute(&self) -> bool {
        self.contaminated && self.inside == 0
    }

    /// True when the window itself holds no non-finite value, so a non-finite
    /// *accumulator* can only be damage rather than a faithful result.
    #[inline]
    pub(super) fn window_is_clean(&self) -> bool {
        self.inside == 0
    }

    #[inline]
    pub(super) fn clear(&mut self) {
        self.contaminated = self.inside > 0;
    }
}

#[cfg(test)]
mod tests {
    use crate::rolling::test_support::{drive_sum, drive_variance};

    // -----------------------------------------------------------------
    // Reseed-policy tests.
    //
    // Every case here is built from **finite** inputs, so `NonFiniteGuard`
    // never fires; they exercise the `MAX_ACCUMULATOR_RANGE` and non-finite-
    // accumulator triggers alone.
    // -----------------------------------------------------------------

    /// The cost side of the policy. A trigger that fires on ordinary data would
    /// restore the O(n * p) shape these accumulators exist to remove, and no
    /// correctness test would catch it.
    #[test]
    fn the_reseed_triggers_do_not_fire_on_ordinary_data() {
        // Noisy prices, trending prices, a constant series (m2 == m2_max == 0,
        // the case a value-magnitude trigger would fire on every bar), and an
        // alternating series whose windows sum to zero (the case the sum's
        // magnitude trigger would spin on).
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let noisy: Vec<f64> = (0..4000)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = ((state >> 11) as f64) / ((1u64 << 53) as f64);
                100.0 + 4.0 * (unit - 0.5)
            })
            .collect();
        let trending: Vec<f64> = (0..4000).map(|i| 50.0 + i as f64 * 0.37).collect();
        let constant = vec![7.5; 4000];
        let zero_sum: Vec<f64> = (0..4000)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        // A volatility regime change: a calm stretch, a volatile stretch, then
        // calm again. Real market data does this, and it must cost O(1)
        // recomputes, not one per bar.
        let regimes: Vec<f64> = (0..4000)
            .map(|i| {
                let amp = if (1000..2000).contains(&i) { 20.0 } else { 0.2 };
                100.0 + amp * ((i % 7) as f64 - 3.0)
            })
            .collect();

        for (label, data) in [
            ("noisy", &noisy),
            ("trending", &trending),
            ("constant", &constant),
            ("zero_sum", &zero_sum),
            ("regimes", &regimes),
        ] {
            for &p in &[2usize, 5, 14, 30] {
                let (_, var_reseeds) = drive_variance(data, p);
                let (_, sum_reseeds) = drive_sum(data, p);
                // A handful of regime-change recomputes is fine; a rate
                // anywhere near one per bar is the regression this guards.
                // The budget is on *work*, not on the trigger count: each
                // recompute is O(p), so `reseeds * p` extra operations against
                // `data.len()` bars is the overhead the policy actually costs.
                // Capped at 25%, versus the 100x it would be at one reseed per
                // bar with p = 30.
                let budget = data.len() / (4 * p);
                assert!(
                    var_reseeds <= budget,
                    "variance reseeded {var_reseeds} times on {label} p={p} (budget {budget})"
                );
                assert!(
                    sum_reseeds <= budget,
                    "sum reseeded {sum_reseeds} times on {label} p={p} (budget {budget})"
                );
            }
        }
    }
}
