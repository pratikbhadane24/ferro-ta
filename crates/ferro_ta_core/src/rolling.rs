//! Shared rolling-window machinery (crate-internal).
//!
//! Indicator kernels repeatedly need the same four primitives over a sliding
//! window: the extreme value, the *index* of the extreme, a running sum, and a
//! running variance. This module holds one vetted copy of each, so kernels can
//! compose them in a **single traversal** instead of materializing an
//! intermediate `Vec` per pass.
//!
//! # Why not `VecDeque`
//!
//! The monotonic-deque sliding extremum is O(n), but a `VecDeque<usize>`
//! implementation pays for it twice: every front/back access does ring-buffer
//! mask arithmetic, and the pop predicate has to *gather* `real[j]` to recover
//! the value behind a stored index. [`MaxDeque`] / [`MinDeque`] instead keep
//! **parallel `Vec<f64>` values and `Vec<u32>` indices** with a `head` cursor,
//! so the hot pop predicate reads `vals[len - 1]` — sequential, in L1, no
//! gather. `pop_front` is `head += 1`; the buffers are compacted only when the
//! deque drains completely (the common case, free) or when `head` has walked a
//! full window (amortized one element move per bar), which bounds both buffers
//! at `2 * timeperiod + 2` entries with no reallocation.
//!
//! # Bit-identity contract
//!
//! The comparison predicates here are byte-for-byte the ones the previous
//! `math::sliding_max` / `math::sliding_min` used, and that is **load-bearing**:
//!
//! * **NaN behaviour.** A `NaN` satisfies neither `<=` nor `>=`, so it is never
//!   popped and can legitimately surface as the window extreme. Switching to
//!   `f64::max` (which ignores NaN) would silently change `DONCHIAN`, `AROON`
//!   and `WILLR` output.
//! * **Tie-breaking.** `<=` (not `<`) means an incoming equal value pops the
//!   entire monotonic run *including the front*, so among equal extremes the
//!   **most recent** index wins. `AROON` depends on exactly this; inverting it
//!   shifts `AROON` output by `100 / period`.
//!
//! The `tests` module keeps a private verbatim copy of each pre-existing
//! implementation and asserts `to_bits()` equality against it over adversarial
//! inputs (plateaus, monotone runs, NaN/±inf, degenerate periods).

// This module is the shared machinery; several primitives land here one commit
// ahead of the kernels that consume them. The alternative is a scattering of
// per-item `#[allow]`s that would all have to be removed again.
#![allow(dead_code)]

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
/// For [`RollingVariance`] the ratio is taken between second moments, which are
/// already squared, so `2^14` on `m2` is a `2^7 = 128`-fold dynamic range in the
/// *values* — the point at which `sigma` is still resolved to 26 bits.
const MAX_ACCUMULATOR_RANGE: f64 = 16_384.0;

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
fn is_subnormal_magnitude(x: f64) -> bool {
    x != 0.0 && x.abs() < f64::MIN_POSITIVE
}

/// Assert the series is addressable by the deques' `u32` index arrays.
///
/// The bound is 4 billion bars — 32 GB of `f64`, unreachable through any of
/// the crate's bindings — so this documents the limit rather than defends it.
#[inline]
fn debug_assert_indexable(n: usize) {
    debug_assert!(
        n <= u32::MAX as usize,
        "series longer than u32::MAX bars is not supported by the rolling deques"
    );
}

// ---------------------------------------------------------------------------
// Monotonic deques and their single-series drivers
// ---------------------------------------------------------------------------

/// Emit a monotonic deque plus its two whole-series drivers.
///
/// `$cmp` is the pop-back predicate (`<=` for a max deque, `>=` for a min
/// deque). Two monomorphized types beat one type with a runtime `bool` flag:
/// the comparison in the hot loop stays a single branch-free instruction.
macro_rules! mono_deque {
    ($name:ident, $cmp:tt, $kind:expr, $val_into:ident, $idx_into:ident) => {
        #[doc = concat!("Monotonic deque whose front is the window ", $kind, ".")]
        ///
        #[doc = concat!("Pops back while `vals[back] ", stringify!($cmp), " x`, so equal extremes")]
        /// resolve to the most recent bar and `NaN` entries are never popped.
        pub(crate) struct $name {
            /// Values of the live entries, monotonic from `head` to the end.
            vals: Vec<f64>,
            /// Bar index of each entry in `vals`, strictly increasing.
            idxs: Vec<u32>,
            /// Index of the deque front inside `vals` / `idxs`.
            head: usize,
        }

        impl $name {
            /// Allocate for a `timeperiod` window over an `n`-bar series.
            ///
            /// The capacity is chosen so the compaction in [`Self::advance`]
            /// can never trigger a reallocation.
            pub(crate) fn with_window(timeperiod: usize, n: usize) -> Self {
                let cap = 2 * (timeperiod.min(n) + 1);
                Self {
                    vals: Vec::with_capacity(cap),
                    idxs: Vec::with_capacity(cap),
                    head: 0,
                }
            }

            /// Slide the window to bar `i`, whose value is `x`.
            ///
            /// Evicts entries that left the window, pops the monotonic run `x`
            /// dominates, then pushes `x`. Afterwards [`Self::front`] is the
            /// window extreme, valid once `i + 1 >= timeperiod`.
            #[inline]
            pub(crate) fn advance(&mut self, i: usize, x: f64, timeperiod: usize) {
                // Drop indices that have left the window.
                while self.head < self.idxs.len()
                    && self.idxs[self.head] as usize + timeperiod <= i
                {
                    self.head += 1;
                }
                if self.head == self.vals.len() {
                    // Fully drained — the common case. Rewind for free.
                    self.vals.clear();
                    self.idxs.clear();
                    self.head = 0;
                } else if self.head > timeperiod {
                    // `head` walked a whole window without the deque ever
                    // draining (a strictly monotone series does this). Compact
                    // so the buffers stay bounded: at most `timeperiod` live
                    // entries move, and only once per `timeperiod` bars.
                    self.vals.drain(..self.head);
                    self.idxs.drain(..self.head);
                    self.head = 0;
                }

                // Pop the dominated run. `vals[len - 1]` is hot and sequential;
                // a `VecDeque<usize>` would gather `real[j]` here instead.
                while self.vals.len() > self.head && self.vals[self.vals.len() - 1] $cmp x {
                    self.vals.pop();
                    self.idxs.pop();
                }
                self.vals.push(x);
                self.idxs.push(i as u32);
            }

            /// Extreme value currently in the window.
            ///
            /// # Panics
            /// Panics if called before the first [`Self::advance`].
            #[inline]
            pub(crate) fn front(&self) -> f64 {
                self.vals[self.head]
            }

            /// Bar index of the extreme currently in the window; among equal
            /// extremes, the **most recent** index.
            ///
            /// # Panics
            /// Panics if called before the first [`Self::advance`].
            #[inline]
            pub(crate) fn front_index(&self) -> usize {
                self.idxs[self.head] as usize
            }
        }

        #[doc = concat!("Write the sliding ", $kind, " over `timeperiod` bars into `out[..real.len()]`.")]
        ///
        /// The first `timeperiod - 1` slots are `NaN`, and every slot is `NaN`
        /// when `timeperiod < 1` or `real.len() < timeperiod` — bit-identical
        /// to the `VecDeque` implementation this replaced.
        ///
        /// # Arguments
        /// * `real` - Input series.
        /// * `timeperiod` - Rolling window size (must be >= 1).
        /// * `out` - Output buffer, at least `real.len()` long.
        pub(crate) fn $val_into(real: &[f64], timeperiod: usize, out: &mut [f64]) {
            let n = real.len();
            debug_assert!(out.len() >= n);
            debug_assert_indexable(n);
            if timeperiod < 1 || n < timeperiod {
                out[..n].fill(f64::NAN);
                return;
            }
            out[..timeperiod - 1].fill(f64::NAN);
            let mut dq = $name::with_window(timeperiod, n);
            for i in 0..n {
                dq.advance(i, real[i], timeperiod);
                if i + 1 >= timeperiod {
                    out[i] = dq.front();
                }
            }
        }

        #[doc = concat!("Write the 0-based index of the sliding ", $kind, " into `out[..real.len()]`.")]
        ///
        /// Warmup slots and degenerate inputs are `-1`. Among equal extremes
        /// the most recent index wins.
        pub(crate) fn $idx_into(real: &[f64], timeperiod: usize, out: &mut [i64]) {
            let n = real.len();
            debug_assert!(out.len() >= n);
            debug_assert_indexable(n);
            if timeperiod < 1 || n < timeperiod {
                out[..n].fill(-1);
                return;
            }
            out[..timeperiod - 1].fill(-1);
            let mut dq = $name::with_window(timeperiod, n);
            for i in 0..n {
                dq.advance(i, real[i], timeperiod);
                if i + 1 >= timeperiod {
                    out[i] = dq.front_index() as i64;
                }
            }
        }
    };
}

mono_deque!(MaxDeque, <=, "maximum", sliding_max_into, sliding_maxindex_into);
mono_deque!(MinDeque, >=, "minimum", sliding_min_into, sliding_minindex_into);

/// Write the sliding maximum of `real_high` and the sliding minimum of
/// `real_low` in **one** traversal.
///
/// Each output is bit-identical to the corresponding single-series call. This
/// is what lets channel indicators (Donchian, Ichimoku, Williams %R) collapse
/// a `sliding_max` + `sliding_min` + fix-up triple into one pass.
///
/// # Arguments
/// * `real_high` - Series to take the rolling maximum of.
/// * `real_low` - Series to take the rolling minimum of (same length).
/// * `timeperiod` - Rolling window size (must be >= 1).
/// * `out_max` - Buffer for the rolling maximum.
/// * `out_min` - Buffer for the rolling minimum.
pub(crate) fn sliding_min_max_into(
    real_high: &[f64],
    real_low: &[f64],
    timeperiod: usize,
    out_max: &mut [f64],
    out_min: &mut [f64],
) {
    let n = real_high.len().min(real_low.len());
    debug_assert!(out_max.len() >= n && out_min.len() >= n);
    debug_assert_indexable(n);
    if timeperiod < 1 || n < timeperiod {
        out_max[..n].fill(f64::NAN);
        out_min[..n].fill(f64::NAN);
        return;
    }
    out_max[..timeperiod - 1].fill(f64::NAN);
    out_min[..timeperiod - 1].fill(f64::NAN);
    let mut hi = MaxDeque::with_window(timeperiod, n);
    let mut lo = MinDeque::with_window(timeperiod, n);
    for i in 0..n {
        hi.advance(i, real_high[i], timeperiod);
        lo.advance(i, real_low[i], timeperiod);
        if i + 1 >= timeperiod {
            out_max[i] = hi.front();
            out_min[i] = lo.front();
        }
    }
}

/// Drive both extremum deques and hand the caller the *indices* only.
///
/// Calls `f(i, max_index, min_index)` for every bar `i` whose window is full
/// (`i + 1 >= window_size`), where `max_index` is the argmax of `high` and
/// `min_index` the argmin of `low` over `[i + 1 - window_size, i]`. Among equal
/// extremes the most recent bar wins, matching the naive `>=` / `<=` scans the
/// Aroon family uses today.
///
/// Nothing is materialized, so an Aroon-style "bars since the extreme" kernel
/// costs one traversal and zero index arrays.
pub(crate) fn sliding_arg_extrema(
    high: &[f64],
    low: &[f64],
    window_size: usize,
    mut f: impl FnMut(usize, usize, usize),
) {
    let n = high.len().min(low.len());
    debug_assert_indexable(n);
    if window_size < 1 || n < window_size {
        return;
    }
    let mut hi = MaxDeque::with_window(window_size, n);
    let mut lo = MinDeque::with_window(window_size, n);
    for i in 0..n {
        hi.advance(i, high[i], window_size);
        lo.advance(i, low[i], window_size);
        if i + 1 >= window_size {
            f(i, hi.front_index(), lo.front_index());
        }
    }
}

// ---------------------------------------------------------------------------
// Running accumulators
// ---------------------------------------------------------------------------

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
struct NonFiniteGuard {
    /// Non-finite values currently inside the window.
    inside: usize,
    /// A non-finite value has entered since the last exact recompute.
    contaminated: bool,
}

impl NonFiniteGuard {
    fn seed(window: &[f64]) -> Self {
        let inside = window.iter().filter(|x| !x.is_finite()).count();
        Self {
            inside,
            contaminated: inside > 0,
        }
    }

    #[inline]
    fn slide(&mut self, x_new: f64, x_old: f64) {
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
    fn needs_recompute(&self) -> bool {
        self.contaminated && self.inside == 0
    }

    /// True when the window itself holds no non-finite value, so a non-finite
    /// *accumulator* can only be damage rather than a faithful result.
    #[inline]
    fn window_is_clean(&self) -> bool {
        self.inside == 0
    }

    #[inline]
    fn clear(&mut self) {
        self.contaminated = self.inside > 0;
    }
}

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
    /// precision guarantee. Stallable — see [`Self::range_stalled`].
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
    /// Unlike [`RollingSum::range_is_lossy`] this needs no stall flag: `m2_max`
    /// is reset to `m2` by every reseed and is measured in the same units, so
    /// the condition is always cleared by an exact recompute and can never spin.
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
    use std::collections::VecDeque;

    // -- Pre-existing implementations, kept verbatim as the equivalence oracle.

    fn reference_sliding_max(real: &[f64], timeperiod: usize) -> Vec<f64> {
        let n = real.len();
        let mut result = vec![f64::NAN; n];
        if timeperiod < 1 || n < timeperiod {
            return result;
        }
        let mut dq: VecDeque<usize> = VecDeque::new();
        for i in 0..n {
            while dq.front().map(|&j| j + timeperiod <= i).unwrap_or(false) {
                dq.pop_front();
            }
            while dq.back().map(|&j| real[j] <= real[i]).unwrap_or(false) {
                dq.pop_back();
            }
            dq.push_back(i);
            if i + 1 >= timeperiod {
                result[i] = real[*dq.front().unwrap()];
            }
        }
        result
    }

    fn reference_sliding_min(real: &[f64], timeperiod: usize) -> Vec<f64> {
        let n = real.len();
        let mut result = vec![f64::NAN; n];
        if timeperiod < 1 || n < timeperiod {
            return result;
        }
        let mut dq: VecDeque<usize> = VecDeque::new();
        for i in 0..n {
            while dq.front().map(|&j| j + timeperiod <= i).unwrap_or(false) {
                dq.pop_front();
            }
            while dq.back().map(|&j| real[j] >= real[i]).unwrap_or(false) {
                dq.pop_back();
            }
            dq.push_back(i);
            if i + 1 >= timeperiod {
                result[i] = real[*dq.front().unwrap()];
            }
        }
        result
    }

    fn reference_maxindex(real: &[f64], timeperiod: usize) -> Vec<i64> {
        let n = real.len();
        let mut result = vec![-1i64; n];
        if timeperiod == 0 || n < timeperiod {
            return result;
        }
        let mut dq: VecDeque<usize> = VecDeque::new();
        for i in 0..n {
            while dq.front().map(|&j| j + timeperiod <= i).unwrap_or(false) {
                dq.pop_front();
            }
            while dq.back().map(|&j| real[j] <= real[i]).unwrap_or(false) {
                dq.pop_back();
            }
            dq.push_back(i);
            if i + 1 >= timeperiod {
                result[i] = *dq.front().unwrap() as i64;
            }
        }
        result
    }

    fn reference_minindex(real: &[f64], timeperiod: usize) -> Vec<i64> {
        let n = real.len();
        let mut result = vec![-1i64; n];
        if timeperiod == 0 || n < timeperiod {
            return result;
        }
        let mut dq: VecDeque<usize> = VecDeque::new();
        for i in 0..n {
            while dq.front().map(|&j| j + timeperiod <= i).unwrap_or(false) {
                dq.pop_front();
            }
            while dq.back().map(|&j| real[j] >= real[i]).unwrap_or(false) {
                dq.pop_back();
            }
            dq.push_back(i);
            if i + 1 >= timeperiod {
                result[i] = *dq.front().unwrap() as i64;
            }
        }
        result
    }

    /// Adversarial series: plateaus (locks tie-breaking), monotone runs (locks
    /// the compaction path, which only triggers when the deque never drains),
    /// all-equal, single mid-series NaN / +inf / -inf, and a low-cardinality
    /// pseudo-random walk.
    fn series() -> Vec<(&'static str, Vec<f64>)> {
        let plateau: Vec<f64> = (0..200).map(|i| (i % 5) as f64).collect();
        let up: Vec<f64> = (0..200).map(|i| i as f64).collect();
        let down: Vec<f64> = (0..200).map(|i| -(i as f64)).collect();
        let flat = vec![7.5; 200];
        let mut with_nan = plateau.clone();
        with_nan[97] = f64::NAN;
        let mut with_pinf = plateau.clone();
        with_pinf[97] = f64::INFINITY;
        let mut with_ninf = plateau.clone();
        with_ninf[97] = f64::NEG_INFINITY;
        // Low-cardinality LCG so equal values recur often.
        let mut state = 12_345u64;
        let random: Vec<f64> = (0..500)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                ((state >> 33) % 7) as f64
            })
            .collect();
        vec![
            ("empty", vec![]),
            ("single", vec![42.0]),
            ("plateau", plateau),
            ("monotone_up", up),
            ("monotone_down", down),
            ("all_equal", flat),
            ("mid_nan", with_nan),
            ("mid_pos_inf", with_pinf),
            ("mid_neg_inf", with_ninf),
            ("low_cardinality_random", random),
        ]
    }

    fn periods(n: usize) -> Vec<usize> {
        let mut p = vec![0, 1, 2, 3, 14, 20];
        if n > 0 {
            p.push(n);
            p.push(n + 1);
        }
        p.push(1000);
        p
    }

    #[test]
    fn sliding_max_is_bit_identical_to_reference() {
        for (name, data) in series() {
            for tp in periods(data.len()) {
                let want = reference_sliding_max(&data, tp);
                let mut got = vec![0.0; data.len()];
                sliding_max_into(&data, tp, &mut got);
                for i in 0..data.len() {
                    assert_eq!(
                        got[i].to_bits(),
                        want[i].to_bits(),
                        "max mismatch: {name} tp={tp} i={i}"
                    );
                }
            }
        }
    }

    #[test]
    fn sliding_min_is_bit_identical_to_reference() {
        for (name, data) in series() {
            for tp in periods(data.len()) {
                let want = reference_sliding_min(&data, tp);
                let mut got = vec![0.0; data.len()];
                sliding_min_into(&data, tp, &mut got);
                for i in 0..data.len() {
                    assert_eq!(
                        got[i].to_bits(),
                        want[i].to_bits(),
                        "min mismatch: {name} tp={tp} i={i}"
                    );
                }
            }
        }
    }

    #[test]
    fn extrema_indices_are_bit_identical_to_reference() {
        for (name, data) in series() {
            for tp in periods(data.len()) {
                let mut got_max = vec![0i64; data.len()];
                let mut got_min = vec![0i64; data.len()];
                sliding_maxindex_into(&data, tp, &mut got_max);
                sliding_minindex_into(&data, tp, &mut got_min);
                assert_eq!(got_max, reference_maxindex(&data, tp), "{name} tp={tp}");
                assert_eq!(got_min, reference_minindex(&data, tp), "{name} tp={tp}");
            }
        }
    }

    #[test]
    fn fused_min_max_matches_the_separate_passes() {
        for (name, data) in series() {
            let flipped: Vec<f64> = data.iter().map(|x| -x).collect();
            for tp in periods(data.len()) {
                let want_max = reference_sliding_max(&data, tp);
                let want_min = reference_sliding_min(&flipped, tp);
                let mut got_max = vec![0.0; data.len()];
                let mut got_min = vec![0.0; data.len()];
                sliding_min_max_into(&data, &flipped, tp, &mut got_max, &mut got_min);
                for i in 0..data.len() {
                    assert_eq!(
                        got_max[i].to_bits(),
                        want_max[i].to_bits(),
                        "fused max mismatch: {name} tp={tp} i={i}"
                    );
                    assert_eq!(
                        got_min[i].to_bits(),
                        want_min[i].to_bits(),
                        "fused min mismatch: {name} tp={tp} i={i}"
                    );
                }
            }
        }
    }

    #[test]
    fn arg_extrema_matches_the_index_passes() {
        for (name, data) in series() {
            for tp in periods(data.len()) {
                let want_max = reference_maxindex(&data, tp);
                let want_min = reference_minindex(&data, tp);
                let mut visited = 0usize;
                sliding_arg_extrema(&data, &data, tp, |i, mx, mn| {
                    assert_eq!(mx as i64, want_max[i], "{name} tp={tp} i={i}");
                    assert_eq!(mn as i64, want_min[i], "{name} tp={tp} i={i}");
                    visited += 1;
                });
                let expected = if tp == 0 || data.len() < tp {
                    0
                } else {
                    data.len() - tp + 1
                };
                assert_eq!(visited, expected, "{name} tp={tp}");
            }
        }
    }

    /// A strictly monotone series never lets the deque drain, so it is the only
    /// input that exercises the `drain(..head)` compaction. Assert the buffers
    /// stay bounded instead of growing to `O(n)`.
    #[test]
    fn monotone_input_keeps_the_deque_bounded() {
        let tp = 8;
        let n = 10_000;
        let mut dq = MaxDeque::with_window(tp, n);
        let cap = dq.vals.capacity();
        for i in 0..n {
            dq.advance(i, -(i as f64), tp);
            assert!(
                dq.vals.len() <= 2 * tp + 2,
                "buffer grew to {}",
                dq.vals.len()
            );
        }
        assert_eq!(dq.vals.capacity(), cap, "compaction reallocated");
    }

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

    // -----------------------------------------------------------------
    // Reseed-policy tests.
    //
    // Every case here is built from **finite** inputs, so `NonFiniteGuard`
    // never fires; they exercise the `MAX_ACCUMULATOR_RANGE` and non-finite-
    // accumulator triggers alone.
    // -----------------------------------------------------------------

    fn exact_population_var(window: &[f64]) -> f64 {
        let p = window.len() as f64;
        let mean = window.iter().sum::<f64>() / p;
        window.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / p
    }

    /// Drive a `RollingVariance` over `data`, returning every full window's
    /// reported variance and the number of exact recomputes it took.
    ///
    /// Deliberately open-codes `advance` so the reseed count is observable:
    /// a policy that is correct but fires constantly turns an O(n) kernel back
    /// into O(n * p), and no correctness assertion would notice.
    fn drive_variance(data: &[f64], p: usize) -> (Vec<f64>, usize) {
        let mut acc = RollingVariance::new(&data[..p]);
        let mut out = vec![acc.population_var()];
        let mut reseeds = 0usize;
        for i in p..data.len() {
            acc.push_pop(data[i], data[i - p]);
            if acc.needs_reseed() {
                acc.reseed(&data[i + 1 - p..=i]);
                reseeds += 1;
            }
            out.push(acc.population_var());
        }
        (out, reseeds)
    }

    /// Same for `RollingSum`.
    fn drive_sum(data: &[f64], p: usize) -> (Vec<f64>, usize) {
        let mut acc = RollingSum::new(&data[..p]);
        let mut out = vec![acc.value()];
        let mut reseeds = 0usize;
        for i in p..data.len() {
            acc.push_pop(data[i], data[i - p]);
            if acc.needs_reseed() {
                acc.reseed(&data[i + 1 - p..=i]);
                reseeds += 1;
            }
            out.push(acc.value());
        }
        (out, reseeds)
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
