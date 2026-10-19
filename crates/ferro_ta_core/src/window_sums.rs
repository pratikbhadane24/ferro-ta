//! Drift-safe running window sums for batch kernels (crate-internal).
//!
//! Two failure modes of the plain `sum += new - old` recurrence, and the
//! accumulator here that removes each:
//!
//! * **Price-level cancellation.** CORREL, the LINEARREG family and the
//!   SMA-difference oscillators (AO, PO, DPO) report a small difference of
//!   large running sums — `n * Σxy - Σx * Σy`, or `SMA(fast) - SMA(slow)`.
//!   Near price 50,000 every update rounds to the ulp of the *price level*,
//!   and the difference inherits that error amplified by `(price / spread)`;
//!   carried from bar 0 it drifted up to 1e-1 (CORREL, `timeperiod = 5`)
//!   from an exact two-pass recompute of the same window. The kernels here
//!   instead sum *deviations from an anchor* — a bar of the window itself,
//!   re-chosen with an exact recompute every
//!   [`anchor_interval`]`(timeperiod)` bars — so every rounding is on the
//!   scale of the window's spread plus its recent excursion, independent of
//!   the price level, and the error stays at ~1e-12 of the result.
//! * **Zero residue.** [`ZeroExactSum`] keeps a running volume sum that reads
//!   *exactly* `0.0` whenever its window holds only zeros, where a plain
//!   running sum keeps the rounding residue of the non-zero values that left
//!   (see the type).
//!
//! Both anchored kernels also recompute exactly once a non-finite bar has
//! left the window ([`NonFiniteGuard`]), so a `NaN` affects exactly the
//! windows that contain it instead of every later bar.

use crate::rolling::{anchor_interval, NonFiniteGuard, MAX_ACCUMULATOR_RANGE};

/// `0.0` for a pair of finite values, `NaN` otherwise (`inf * 0` and
/// `NaN * 0` are `NaN`), so a two-input bar can drive a single-series
/// [`NonFiniteGuard`]. Branch-free.
#[inline]
fn finite_flag(a: f64, b: f64) -> f64 {
    a * 0.0 + b * 0.0
}

/// True when a dispersion `v` has collapsed by more than
/// [`MAX_ACCUMULATOR_RANGE`] from its peak since the last recompute — a
/// wide stretch has left the window and taken `v`'s low bits with it. A
/// negative (rounded-through-zero) `v` counts as collapsed.
#[inline]
fn collapsed(v: f64, peak: f64) -> bool {
    peak > MAX_ACCUMULATOR_RANGE * v.max(0.0)
}

// ---------------------------------------------------------------------------
// CORREL
// ---------------------------------------------------------------------------

/// Deviation sums of a window of `(x, y)` pairs about the anchor `(ax, ay)`.
#[derive(Default)]
struct CoMoments {
    ax: f64,
    ay: f64,
    sx: f64,
    sy: f64,
    sxx: f64,
    syy: f64,
    sxy: f64,
}

impl CoMoments {
    /// Exact recompute, anchored at the window's last pair. A window bar as
    /// anchor (rather than the mean) makes a constant window's deviations
    /// exactly zero, so its dispersion is an exact `0.0`, as two-pass gives.
    fn seed(x: &[f64], y: &[f64]) -> Self {
        let last = x.len() - 1;
        let mut m = Self {
            ax: x[last],
            ay: y[last],
            ..Self::default()
        };
        for (&a, &b) in x.iter().zip(y) {
            let (dx, dy) = (a - m.ax, b - m.ay);
            m.sx += dx;
            m.sy += dy;
            m.sxx += dx * dx;
            m.syy += dy * dy;
            m.sxy += dx * dy;
        }
        m
    }

    #[inline]
    fn slide(&mut self, x_new: f64, y_new: f64, x_old: f64, y_old: f64) {
        let (dxi, dyi) = (x_new - self.ax, y_new - self.ay);
        let (dxo, dyo) = (x_old - self.ax, y_old - self.ay);
        self.sx += dxi - dxo;
        self.sy += dyi - dyo;
        self.sxx += dxi * dxi - dxo * dxo;
        self.syy += dyi * dyi - dyo * dyo;
        self.sxy += dxi * dyi - dxo * dyo;
    }

    /// `(p Σdx² - (Σdx)², p Σdy² - (Σdy)², p Σdxdy - Σdx Σdy)`: `p²` times
    /// the window's variances and covariance (anchor-free).
    #[inline]
    fn spreads(&self, p: f64) -> (f64, f64, f64) {
        (
            p * self.sxx - self.sx * self.sx,
            p * self.syy - self.sy * self.sy,
            p * self.sxy - self.sx * self.sy,
        )
    }
}

/// Rolling Pearson correlation over anchored co-moment sums; `NaN` where
/// either side has zero (or non-finite) dispersion. See the module docs.
///
/// Recomputed exactly every [`anchor_interval`] bars, once a non-finite pair
/// has left the window, and when either dispersion collapses by more than
/// [`MAX_ACCUMULATOR_RANGE`] from its peak since the last recompute — the
/// last is what makes a window that turns constant report an exact zero
/// dispersion (`NaN`) rather than a rounding residue.
pub(crate) fn correl(real0: &[f64], real1: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = real0.len();
    let mut out = vec![f64::NAN; n];
    if timeperiod == 0 || n < timeperiod || real1.len() != n {
        return out;
    }
    let p = timeperiod as f64;
    let interval = anchor_interval(timeperiod);
    let emit = |(vx, vy, cxy): (f64, f64, f64)| {
        if vx > 0.0 && vy > 0.0 {
            cxy / (vx * vy).sqrt()
        } else {
            f64::NAN
        }
    };
    let bad = (0..timeperiod)
        .filter(|&j| finite_flag(real0[j], real1[j]).is_nan())
        .count();
    let mut guard = NonFiniteGuard::from_count(bad);
    let mut m = CoMoments::seed(&real0[..timeperiod], &real1[..timeperiod]);
    let mut spreads = m.spreads(p);
    let (mut peak_x, mut peak_y) = (spreads.0.max(0.0), spreads.1.max(0.0));
    let mut since = 0usize;
    out[timeperiod - 1] = emit(spreads);
    for end in timeperiod..n {
        let old = end - timeperiod;
        m.slide(real0[end], real1[end], real0[old], real1[old]);
        guard.slide(
            finite_flag(real0[end], real1[end]),
            finite_flag(real0[old], real1[old]),
        );
        since += 1;
        spreads = m.spreads(p);
        let (vx, vy, _) = spreads;
        peak_x = peak_x.max(vx);
        peak_y = peak_y.max(vy);
        let lossy = guard.window_is_clean() && (collapsed(vx, peak_x) || collapsed(vy, peak_y));
        if since >= interval || guard.needs_recompute() || lossy {
            let start = end + 1 - timeperiod;
            m = CoMoments::seed(&real0[start..=end], &real1[start..=end]);
            spreads = m.spreads(p);
            (peak_x, peak_y) = (spreads.0.max(0.0), spreads.1.max(0.0));
            guard.clear();
            since = 0;
        }
        out[end] = emit(spreads);
    }
    out
}

// ---------------------------------------------------------------------------
// LINEARREG family
// ---------------------------------------------------------------------------

/// `Σ (y - a)` and `Σ j (y_j - a)` (`j = 0..p`) of a window about the anchor
/// `a`, the window's last bar.
struct LinregSums {
    anchor: f64,
    sy: f64,
    sxy: f64,
}

impl LinregSums {
    fn seed(window: &[f64]) -> Self {
        let anchor = window[window.len() - 1];
        let (mut sy, mut sxy) = (0.0, 0.0);
        for (j, &v) in window.iter().enumerate() {
            let d = v - anchor;
            sy += d;
            sxy += j as f64 * d;
        }
        Self { anchor, sy, sxy }
    }

    /// Shift the window by one bar: every x index drops by one, so
    /// `Σ j d_j` loses `Σ d` (minus the leaving bar), then gains the new bar
    /// at `x = p - 1`.
    #[inline]
    fn slide(&mut self, incoming: f64, outgoing: f64, last_x: f64) {
        let d_in = incoming - self.anchor;
        let d_out = outgoing - self.anchor;
        let prev_sy = self.sy;
        self.sy = prev_sy - d_out + d_in;
        self.sxy = self.sxy - (prev_sy - d_out) + last_x * d_in;
    }
}

/// Least-squares fit of every `timeperiod` window against `x = 0..p`,
/// mapped through `map(slope, intercept)`.
///
/// The slope is anchor-free; the intercept is `(Σd - slope Σx) / p + a`.
/// Recomputed exactly every [`anchor_interval`] bars and once a non-finite
/// bar has left the window. See the module docs.
pub(crate) fn linreg_apply<F>(prices: &[f64], timeperiod: usize, mut map: F) -> Vec<f64>
where
    F: FnMut(f64, f64) -> f64,
{
    let n = prices.len();
    if timeperiod == 0 || n < timeperiod {
        return vec![f64::NAN; n];
    }
    // Pre-size and store by index; `push` costs more per bar than the NaN
    // fill costs once (see `statistic::rolling_population_var_apply`).
    let mut result = vec![f64::NAN; n];
    let period = timeperiod as f64;
    let last_x = (timeperiod - 1) as f64;
    let sum_x = last_x * period / 2.0;
    let sum_x2 = last_x * period * (2.0 * period - 1.0) / 6.0;
    let denom = period * sum_x2 - sum_x * sum_x;
    let interval = anchor_interval(timeperiod);
    let mut emit = |s: &LinregSums| {
        let slope = if denom != 0.0 {
            (period * s.sxy - sum_x * s.sy) / denom
        } else {
            0.0
        };
        let intercept = (s.sy - slope * sum_x) / period + s.anchor;
        map(slope, intercept)
    };

    let mut guard = NonFiniteGuard::seed(&prices[..timeperiod]);
    let mut sums = LinregSums::seed(&prices[..timeperiod]);
    let mut since = 0usize;
    result[timeperiod - 1] = emit(&sums);
    for end in timeperiod..n {
        let outgoing = prices[end - timeperiod];
        sums.slide(prices[end], outgoing, last_x);
        guard.slide(prices[end], outgoing);
        since += 1;
        if since >= interval || guard.needs_recompute() {
            sums = LinregSums::seed(&prices[end + 1 - timeperiod..=end]);
            guard.clear();
            since = 0;
        }
        result[end] = emit(&sums);
    }
    result
}

// ---------------------------------------------------------------------------
// SMA differences (AO, PO, DPO)
// ---------------------------------------------------------------------------

/// Means of the `periods[k]`-bar windows ending at every bar, as
/// `mean_k - anchor` with one shared anchor, passed to
/// `emit(bar, anchor, [mean_k - anchor])`.
///
/// For a kernel that reports a *difference* of means (or a bar minus a
/// mean), subtracting the shifted means cancels the anchor exactly, so the
/// result never rounds at the price level. Output starts at the bar where
/// the longest window is full and is `NaN` while that window holds a
/// non-finite bar. Recomputed exactly every [`anchor_interval`] bars (of the
/// longest period) and once a non-finite bar has left the window; the
/// anchor is the current bar at each recompute.
pub(crate) fn anchored_window_means<const K: usize>(
    src: &[f64],
    periods: [usize; K],
    mut emit: impl FnMut(usize, f64, [f64; K]) -> f64,
) -> Vec<f64> {
    let n = src.len();
    let mut out = vec![f64::NAN; n];
    let longest = periods.iter().copied().max().unwrap_or(0);
    if K == 0 || periods.contains(&0) || n < longest {
        return out;
    }
    let reseed = |end: usize| -> (f64, [f64; K]) {
        let anchor = src[end];
        let sums = periods.map(|p| src[end + 1 - p..=end].iter().map(|v| v - anchor).sum());
        (anchor, sums)
    };
    let interval = anchor_interval(longest);
    let first = longest - 1;
    let mut guard = NonFiniteGuard::seed(&src[..longest]);
    let (mut anchor, mut sums) = reseed(first);
    let mut since = 0usize;
    for i in first..n {
        if i > first {
            let d_new = src[i] - anchor;
            for (sum, &p) in sums.iter_mut().zip(&periods) {
                *sum += d_new - (src[i - p] - anchor);
            }
            guard.slide(src[i], src[i - longest]);
            since += 1;
            if since >= interval || guard.needs_recompute() {
                (anchor, sums) = reseed(i);
                guard.clear();
                since = 0;
            }
        }
        if guard.window_is_clean() {
            let mut k = 0;
            let means = sums.map(|s| {
                let m = s / periods[k] as f64;
                k += 1;
                m
            });
            out[i] = emit(i, anchor, means);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Zero-exact running sums (volume kernels)
// ---------------------------------------------------------------------------

/// A running window sum that is exactly `0.0` whenever every value in the
/// window is zero.
///
/// `sum += new - old` over fractional values leaves a rounding residue once
/// they have all left: a window of all-zero volume then reads as, e.g.,
/// `-2e-16` instead of `0`, and a kernel's `!= 0` / `== 0` denominator rule
/// emits finite garbage (CMF `-0.14`, VOLOSC `-21.7`) where its own rule says
/// `NaN`. Tracking the run of trailing zeros (`NaN` breaks a run) and
/// resetting the sum to `0.0` on the bar the run first covers the window is
/// an exact recompute of the all-zero window at O(1) cost; while the run
/// lasts, `0.0 + (0 - 0)` stays exactly `0.0`. Otherwise the recurrence —
/// operation for operation — is the plain one, so every other output is
/// unchanged bit for bit, and the reset sits behind a cold branch so the
/// loop-carried `sum` chain is not lengthened.
/// A running window sum that reads exactly `0.0` whenever every value in
/// the window is zero.
///
/// `sum += new - old` over fractional values leaves a rounding residue once
/// they have all left: a window of all-zero volume then reads as, e.g.,
/// `-2e-16` instead of `0`, and a kernel's `!= 0` / `== 0` denominator rule
/// emits finite garbage (CMF `-0.14`, VOLOSC `-21.7`) where its own rule says
/// `NaN`. This tracks the run of trailing zeros (`NaN` breaks a run) and
/// reports `0.0` while the run covers the window — the exact sum of that
/// window.
///
/// The running sum itself is left alone: it is the plain recurrence,
/// operation for operation, so every window holding a non-zero value reads
/// exactly what it did before (bit for bit). Resetting the carried sum
/// instead would put a select on its loop-carried dependency chain, which
/// measured ~2x slower on these otherwise latency-bound loops; the residue it
/// would clear is of the order `eps` times the volumes that left, as before.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct ZeroExactSum {
    /// The plain running sum (may hold a residue over an all-zero window).
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::streaming::snapshot::nan_as_null")
    )]
    sum: f64,
    /// Consecutive zeros ending at the newest bar (may exceed `period`).
    zero_run: usize,
    /// Window length.
    period: usize,
}

impl ZeroExactSum {
    /// Seed from `window` with the caller's own seed reduction `sum`, so
    /// each kernel keeps its historical seed bits.
    pub(crate) fn seeded(sum: f64, window: &[f64]) -> Self {
        Self {
            sum,
            zero_run: window.iter().rev().take_while(|v| **v == 0.0).count(),
            period: window.len(),
        }
    }

    /// Slide by one bar (`x_new` enters, `x_old` leaves); returns
    /// [`Self::value`].
    #[inline]
    pub(crate) fn slide(&mut self, x_new: f64, x_old: f64) -> f64 {
        self.sum += x_new - x_old;
        // Branch-free: `(run + 1)` while zeros continue, `0` on any other
        // value (a run cannot reach `usize::MAX` bars).
        self.zero_run = usize::from(x_new == 0.0) * (self.zero_run + 1);
        self.value()
    }

    /// The window sum: exactly `0.0` for an all-zero window, else the
    /// running sum.
    #[inline]
    pub(crate) fn value(&self) -> f64 {
        if self.zero_run >= self.period {
            0.0
        } else {
            self.sum
        }
    }

    /// Snapshot-restore check for the streams that mirror this accumulator:
    /// the window length and trailing-zero run match `window` (oldest first,
    /// full).
    pub(crate) fn matches<'a>(&self, window: impl DoubleEndedIterator<Item = &'a f64>) -> bool {
        let mut len = 0;
        let mut trailing = 0;
        let mut in_run = true;
        for v in window.rev() {
            len += 1;
            in_run &= *v == 0.0;
            trailing += usize::from(in_run);
        }
        len == self.period && self.zero_run.min(len) == trailing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_exact_sum_snaps_an_all_zero_window_to_zero() {
        let v = [0.37, 0.74, 1.11, 0.0, 0.0, 0.0, 0.5];
        let p = 3;
        let mut s = ZeroExactSum::seeded(v[..p].iter().sum(), &v[..p]);
        let mut got = vec![s.value()];
        for i in p..v.len() {
            got.push(s.slide(v[i], v[i - p]));
        }
        // Bar 5 holds only zeros: exactly 0, not the 0.37+0.74+1.11 residue.
        assert_eq!(got[3], 0.0);
        assert!(got[2] > 0.0 && (got[4] - 0.5).abs() < 1e-15);
        assert!(s.matches(v[v.len() - p..].iter()));
    }

    #[test]
    fn anchored_kernels_recover_once_a_nan_leaves_the_window() {
        let mut x: Vec<f64> = (0..60).map(|i| 50_000.0 + (i % 7) as f64).collect();
        let y: Vec<f64> = (0..60).map(|i| 100.0 + (i % 5) as f64).collect();
        x[20] = f64::NAN;
        let p = 5;
        let c = correl(&x, &y, p);
        let s = linreg_apply(&x, p, |slope, _| slope);
        let m = anchored_window_means(&x, [p], |_, _, m| m[0]);
        for i in p - 1..x.len() {
            let dirty = (i + 1 - p..=i).contains(&20);
            for (name, v) in [("correl", c[i]), ("slope", s[i]), ("mean", m[i])] {
                assert_eq!(v.is_nan(), dirty, "{name} bar {i}: {v}");
            }
        }
    }

    #[test]
    fn a_window_turning_constant_reports_exact_zero_dispersion() {
        let mut x: Vec<f64> = (0..40)
            .map(|i| 50_000.0 + (i as f64 * 0.37).sin())
            .collect();
        x[20..].fill(50_000.125);
        let y: Vec<f64> = (0..40).map(|i| 3.0 + (i as f64 * 0.11).cos()).collect();
        let c = correl(&x, &y, 5);
        assert!(c[19].is_finite());
        assert!(c[24..].iter().all(|v| v.is_nan()), "{:?}", &c[24..]);
    }
}
