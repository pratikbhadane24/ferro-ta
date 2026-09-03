//! Statistic extended indicators: MEDIAN, MEDIAN_BANDS, MODE.

use crate::rolling::{MaxDeque, MinDeque};

/// Ascending order of the window, as `sort_by(partial_cmp)` produces it.
///
/// `partial_cmp` is a total order on the finite values this kernel keeps, so
/// `unwrap_or(Equal)` never fires. `sort_by` is *stable*, and that is
/// load-bearing: `-0.0` and `0.0` compare equal yet stay distinguishable in
/// the output, so the tie order decides the sign of a returned zero.
#[inline]
fn sort_window_ascending(window: &mut [f64]) {
    window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

/// Central value of an already-ascending window.
///
/// Even windows average the two central values (same as NumPy).
#[inline]
fn median_of_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

/// Shortest `timeperiod` for which sliding the sorted window beats re-sorting
/// it from scratch.
///
/// Sliding is amortized O(p) *moves* against re-sorting's O(p log p)
/// *comparisons*, so the crossover depends on how sorted the arriving windows
/// already are. Measured at n = 100k on Apple Silicon, `slide / rebuild`
/// crosses 1.0 at p = 7 on a random walk but only at p = 12 on a smooth
/// (locally monotone) series, where insertion-sorting a window costs barely
/// one comparison per element. 12 is the conservative choice — past it sliding
/// wins on *both* distributions, so neither regresses — and from there the win
/// only grows: 4.9x at p = 20, 24x at p = 257.
const MEDIAN_SLIDE_MIN_PERIOD: usize = 12;

/// Replace `old` with `x` in the ascending `buf`, keeping it ascending.
///
/// `buf` holds the previous window sorted; `old` just left the window and `x`
/// just entered it, so the length never changes and one shift suffices.
/// **Bit-identical to re-sorting**: within a class of equal-comparing values
/// (only `+0.0`/`-0.0` can differ observably) `buf` stays in bar order,
/// because `partition_point(|y| y < old)` drops the left-most — oldest —
/// member of `old`'s class, and both shift loops place `x` after every value
/// `<= x`, i.e. as the newest member of its own. That is exactly the order a
/// stable sort of the window produces.
///
/// The shift is a linear scan, not a `memmove` over the whole buffer: adjacent
/// bars of a price series are close, so `x` usually lands within a few slots
/// of `old`.
#[inline]
fn slide_sorted_window(buf: &mut [f64], old: f64, x: f64) {
    let len = buf.len();
    let drop_at = buf.partition_point(|&y| y < old);
    // Walk right while the neighbour still belongs left of `x`.
    let mut at = drop_at;
    while at + 1 < len && buf[at + 1] <= x {
        buf[at] = buf[at + 1];
        at += 1;
    }
    if at == drop_at {
        // `x` sorts at or before the vacated slot: walk left instead.
        while at > 0 && buf[at - 1] > x {
            buf[at] = buf[at - 1];
            at -= 1;
        }
    }
    buf[at] = x;
}

/// Middle value of a 3-bar window, plus whether the window is finite.
///
/// `timeperiod = 3` is MEDIAN's declared default, so it gets a path that keeps
/// the window in registers: no scratch buffer, no `memcpy`, no separate
/// finiteness scan. The comparisons are a stable insertion sort unrolled
/// (`swap` only on a *strict* `<`), so equal-comparing values keep bar order
/// and the result is bit-identical to the general path — which matters because
/// the middle of `[+0.0, -0.0, x]` is a zero whose sign the tie order picks.
#[inline]
fn median3(window: &[f64; 3]) -> (f64, bool) {
    let [mut a, mut b, c0] = *window;
    let all_finite = a.is_finite() && b.is_finite() && c0.is_finite();
    let mut c = c0;
    if b < a {
        std::mem::swap(&mut a, &mut b);
    }
    if c < b {
        std::mem::swap(&mut b, &mut c);
        if b < a {
            std::mem::swap(&mut a, &mut b);
        }
    }
    (b, all_finite)
}

/// Rolling median of `real` over `timeperiod`.
///
/// Even windows use the average of the two central values (same as NumPy).
/// The first `timeperiod - 1` entries are `NaN`. A window that contains any
/// non-finite value also yields `NaN`.
pub fn median(real: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = real.len();
    if timeperiod < 1 || n < timeperiod {
        return vec![f64::NAN; n];
    }

    // `vec![0.0; n]` rather than `vec![NAN; n]`: `0.0` is all-zero bits, so
    // this is an `alloc_zeroed` the OS can hand over as lazily-mapped pages,
    // whereas a `NaN` fill is a real 800 KB store pass at n = 100k that the
    // kernel then overwrites. Every slot below is assigned exactly once, so
    // the zeros are never observable. (`Vec::with_capacity` + `push` measured
    // *slower* than either: at ~1.7 ns/bar the per-push capacity check costs
    // more than the fill it removes.)
    let mut out = vec![0.0_f64; n];
    out[..timeperiod - 1].fill(f64::NAN);

    // The median of a single value needs no ordering at all.
    if timeperiod == 1 {
        for (slot, &v) in out.iter_mut().zip(real) {
            *slot = if v.is_finite() { v } else { f64::NAN };
        }
        return out;
    }

    if timeperiod == 3 {
        for (start, slot) in out[2..].iter_mut().enumerate() {
            let window: &[f64; 3] = real[start..start + 3].try_into().expect("3-bar window");
            let (mid, all_finite) = median3(window);
            *slot = if all_finite { mid } else { f64::NAN };
        }
        return out;
    }

    // One scratch window, reused for every bar.
    let mut buf = vec![0.0_f64; timeperiod];

    if timeperiod < MEDIAN_SLIDE_MIN_PERIOD {
        // Short windows: one pass that copies, orders, and checks finiteness
        // together — down from a finiteness scan, a `memcpy`, and a sort.
        // Insertion sort is what `slice::sort_by` runs at these lengths
        // anyway, so this sheds only the comparator closure and the extra
        // passes; the shifting is stable, hence bit-identical.
        for (start, slot) in out[timeperiod - 1..].iter_mut().enumerate() {
            let window = &real[start..start + timeperiod];
            let mut all_finite = true;
            for (i, &key) in window.iter().enumerate() {
                all_finite &= key.is_finite();
                let mut j = i;
                while j > 0 && key < buf[j - 1] {
                    buf[j] = buf[j - 1];
                    j -= 1;
                }
                buf[j] = key;
            }
            *slot = if all_finite {
                median_of_sorted(&buf)
            } else {
                f64::NAN
            };
        }
        return out;
    }

    // Long windows: keep `buf` sorted across bars and slide it one value at a
    // time, and track the window's non-finite population with a rolling count
    // so the skip rule costs O(1) per bar instead of a rescan. An integer
    // count over exact membership cannot drift.
    let mut nonfinite = real[..timeperiod - 1]
        .iter()
        .filter(|v| !v.is_finite())
        .count();
    let mut buf_holds_previous_window = false;
    for (start, slot) in out[timeperiod - 1..].iter_mut().enumerate() {
        let end = start + timeperiod - 1;
        if !real[end].is_finite() {
            nonfinite += 1;
        }
        if nonfinite != 0 {
            // Nothing sorted to carry forward across the gap.
            buf_holds_previous_window = false;
            *slot = f64::NAN;
        } else {
            if buf_holds_previous_window {
                slide_sorted_window(&mut buf, real[start - 1], real[end]);
            } else {
                buf.copy_from_slice(&real[start..=end]);
                sort_window_ascending(&mut buf);
                buf_holds_previous_window = true;
            }
            *slot = median_of_sorted(&buf);
        }
        if !real[start].is_finite() {
            nonfinite -= 1;
        }
    }
    out
}

/// Median bands: rolling median of `(high + low) / 2`, ATR envelopes, and an
/// EMA of the median.
///
/// Returns `(median, upper, lower, median_ema)`.
///
/// * `timeperiod` — median / EMA window (typically 3).
/// * `atr_period` — ATR smoothing period (typically 14).
/// * `multiplier` — ATR width (`upper = median + multiplier * ATR`).
pub fn median_bands(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    timeperiod: usize,
    atr_period: usize,
    multiplier: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = high.len();
    if low.len() != n || close.len() != n {
        return (
            vec![f64::NAN; n],
            vec![f64::NAN; n],
            vec![f64::NAN; n],
            vec![f64::NAN; n],
        );
    }
    let mut source = Vec::with_capacity(n);
    source.extend(high.iter().zip(low).map(|(&h, &l)| 0.5 * (h + l)));
    let mid = median(&source, timeperiod);
    let atr = crate::volatility::atr(high, low, close, atr_period);
    let mid_ema = crate::overlap::ema(&mid, timeperiod);
    debug_assert_eq!(mid.len(), n);
    debug_assert_eq!(atr.len(), n);
    // Zeroed allocations (see `median`); every slot is assigned below.
    let mut upper = vec![0.0_f64; n];
    let mut lower = vec![0.0_f64; n];
    for (((u, l), &m), &a) in upper.iter_mut().zip(lower.iter_mut()).zip(&mid).zip(&atr) {
        if m.is_finite() && a.is_finite() {
            let width = multiplier * a;
            *u = m + width;
            *l = m - width;
        } else {
            *u = f64::NAN;
            *l = f64::NAN;
        }
    }
    (mid, upper, lower, mid_ema)
}

/// Window minimum, maximum, and whether every value is finite, in one pass.
///
/// The comparison predicates are byte-for-byte the ones the two separate
/// `min` / `max` scans used, so the extremes are bit-identical down to the
/// sign of a zero. The extremes are meaningless when `false` is returned, and
/// the caller must not use them.
#[inline]
fn window_extremes(window: &[f64]) -> (f64, f64, bool) {
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    let mut all_finite = true;
    for &v in window {
        all_finite &= v.is_finite();
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
    }
    (min_v, max_v, all_finite)
}

/// Shortest `timeperiod` for which the monotonic deques beat the fused
/// per-bar rescan in [`window_extremes`].
///
/// The deques are amortized O(1) per bar against the rescan's O(p), but carry
/// per-bar bookkeeping the rescan does not. Measured at n = 100k, bins = 10 on
/// Apple Silicon: the rescan wins at p = 20 (3.45 ms vs 4.06 ms), the deques
/// win from p = 25 (4.10 ms vs 4.25 ms), reaching 1.7x by p = 100.
const MODE_DEQUE_MIN_PERIOD: usize = 25;

/// Discretize one finite window and return the centre of its most populous bin.
///
/// `min_v` / `max_v` must be the window extremes. `counts` is a reusable
/// `bins`-long scratch histogram; its previous contents are irrelevant.
#[inline]
fn mode_of_window(window: &[f64], min_v: f64, max_v: f64, counts: &mut [usize]) -> f64 {
    // Constant window: return the value itself. `window[0]` rather than
    // `min_v`, because a `+0.0`/`-0.0` mix compares constant while still
    // carrying an observable sign, and the leading element is what an
    // `INFINITY`-seeded first-minimum scan latches.
    if max_v == min_v {
        return window[0];
    }
    let bins = counts.len();
    counts.fill(0);
    let width = (max_v - min_v) / bins as f64;
    for &v in window {
        // `v >= min_v` so the quotient is non-negative; the clamp only fires
        // for `v == max_v` (and, if `width` underflows to zero, for all of
        // them, since the quotient is then infinite and the cast saturates).
        let mut idx = ((v - min_v) / width).floor() as usize;
        if idx >= bins {
            idx = bins - 1;
        }
        counts[idx] += 1;
    }
    let mut best = 0_usize;
    let mut best_c = counts[0];
    for (i, &c) in counts.iter().enumerate().skip(1) {
        if c > best_c {
            best = i;
            best_c = c;
        }
    }
    min_v + (best as f64 + 0.5) * width
}

/// Rolling mode via equal-width discretization of each window.
///
/// Values are placed into `bins` buckets between the window min and max.
/// The output is the centre of the most populous bin. Ties take the
/// left-most bin. A constant window returns that constant. The first
/// `timeperiod - 1` entries are `NaN`.
pub fn mode(real: &[f64], timeperiod: usize, bins: usize) -> Vec<f64> {
    let n = real.len();
    if timeperiod < 1 || bins < 1 || n < timeperiod {
        return vec![f64::NAN; n];
    }

    // See `median` for why this is a zeroed allocation and not a `NaN` fill.
    let mut out = vec![0.0_f64; n];
    out[..timeperiod - 1].fill(f64::NAN);
    let mut counts = vec![0_usize; bins];

    if timeperiod < MODE_DEQUE_MIN_PERIOD {
        // Short windows: one fused pass for the NaN test and both extremes,
        // then the histogram. Two passes per bar, down from four.
        for (start, slot) in out[timeperiod - 1..].iter_mut().enumerate() {
            let window = &real[start..start + timeperiod];
            let (min_v, max_v, all_finite) = window_extremes(window);
            *slot = if all_finite {
                mode_of_window(window, min_v, max_v, &mut counts)
            } else {
                f64::NAN
            };
        }
        return out;
    }

    // Long windows: extremes from the shared monotonic deques (amortized O(1)
    // per bar) plus a rolling non-finite count, leaving only the histogram
    // O(timeperiod). Just the extreme *values* are read, and those are a
    // property of the window multiset, so the deques' most-recent-wins
    // tie-break is not observable here.
    let mut hi = MaxDeque::with_window(timeperiod, n);
    let mut lo = MinDeque::with_window(timeperiod, n);
    let mut nonfinite = real[..timeperiod - 1]
        .iter()
        .filter(|v| !v.is_finite())
        .count();
    for (i, &v) in real[..timeperiod - 1].iter().enumerate() {
        hi.advance(i, v, timeperiod);
        lo.advance(i, v, timeperiod);
    }

    for (start, slot) in out[timeperiod - 1..].iter_mut().enumerate() {
        let end = start + timeperiod - 1;
        let x = real[end];
        hi.advance(end, x, timeperiod);
        lo.advance(end, x, timeperiod);
        if !x.is_finite() {
            nonfinite += 1;
        }
        *slot = if nonfinite == 0 {
            mode_of_window(&real[start..=end], lo.front(), hi.front(), &mut counts)
        } else {
            f64::NAN
        };
        if !real[start].is_finite() {
            nonfinite -= 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pre-optimization references, copied verbatim from the implementations
    // these kernels replaced; every fast path is asserted `to_bits()`-equal
    // against them. Exact equality is the right bar because these kernels
    // *select* or *centre on* actual input values, so any epsilon tolerance
    // would hide a behaviour change rather than absorb rounding.

    fn reference_window_median(window: &mut [f64]) -> f64 {
        window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = window.len();
        if n % 2 == 1 {
            window[n / 2]
        } else {
            0.5 * (window[n / 2 - 1] + window[n / 2])
        }
    }

    fn reference_median(real: &[f64], timeperiod: usize) -> Vec<f64> {
        let n = real.len();
        let mut result = vec![f64::NAN; n];
        if timeperiod < 1 || n < timeperiod {
            return result;
        }
        let mut buf = vec![0.0_f64; timeperiod];
        for end in (timeperiod - 1)..n {
            let start = end + 1 - timeperiod;
            let window = &real[start..=end];
            if window.iter().any(|v| !v.is_finite()) {
                continue;
            }
            buf.copy_from_slice(window);
            result[end] = reference_window_median(&mut buf);
        }
        result
    }

    fn reference_mode(real: &[f64], timeperiod: usize, bins: usize) -> Vec<f64> {
        let n = real.len();
        let mut result = vec![f64::NAN; n];
        if timeperiod < 1 || bins < 1 || n < timeperiod {
            return result;
        }
        let mut counts = vec![0_usize; bins];
        for end in (timeperiod - 1)..n {
            let window = &real[end + 1 - timeperiod..=end];
            if window.iter().any(|v| !v.is_finite()) {
                continue;
            }
            let mut min_v = f64::INFINITY;
            let mut max_v = f64::NEG_INFINITY;
            for &v in window {
                if v < min_v {
                    min_v = v;
                }
                if v > max_v {
                    max_v = v;
                }
            }
            if max_v == min_v {
                result[end] = min_v;
                continue;
            }
            counts.fill(0);
            let width = (max_v - min_v) / bins as f64;
            for &v in window {
                let mut idx = ((v - min_v) / width).floor() as usize;
                if idx >= bins {
                    idx = bins - 1;
                }
                counts[idx] += 1;
            }
            let mut best = 0_usize;
            let mut best_c = counts[0];
            for (i, &c) in counts.iter().enumerate().skip(1) {
                if c > best_c {
                    best = i;
                    best_c = c;
                }
            }
            result[end] = min_v + (best as f64 + 0.5) * width;
        }
        result
    }

    #[allow(clippy::type_complexity)]
    fn reference_median_bands(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        timeperiod: usize,
        atr_period: usize,
        multiplier: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = high.len();
        if low.len() != n || close.len() != n {
            return (
                vec![f64::NAN; n],
                vec![f64::NAN; n],
                vec![f64::NAN; n],
                vec![f64::NAN; n],
            );
        }
        let mut source = vec![f64::NAN; n];
        for i in 0..n {
            source[i] = 0.5 * (high[i] + low[i]);
        }
        let mid = reference_median(&source, timeperiod);
        let atr = crate::volatility::atr(high, low, close, atr_period);
        let mid_ema = crate::overlap::ema(&mid, timeperiod);
        let mut upper = vec![f64::NAN; n];
        let mut lower = vec![f64::NAN; n];
        for i in 0..n {
            if mid[i].is_finite() && atr[i].is_finite() {
                let width = multiplier * atr[i];
                upper[i] = mid[i] + width;
                lower[i] = mid[i] - width;
            }
        }
        (mid, upper, lower, mid_ema)
    }

    /// Bit-for-bit equality: `NaN` matches `NaN`, `-0.0` does not match `0.0`.
    fn assert_bits_eq(actual: &[f64], expected: &[f64], case: &str) {
        assert_eq!(actual.len(), expected.len(), "length mismatch for {case}");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "bit mismatch at {i} for {case}: {a} vs {e}"
            );
        }
    }

    /// Deterministic pseudo-random walk (no RNG dependency).
    fn walk(n: usize) -> Vec<f64> {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut price = 100.0_f64;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let step = ((state >> 32) as f64 / (1_u64 << 31) as f64) - 1.0;
                price += step;
                price
            })
            .collect()
    }

    /// Adversarial series: plateaus, constant runs, mixed zero signs, and
    /// non-finite values.
    fn edge_series() -> Vec<Vec<f64>> {
        vec![
            vec![],
            vec![1.0],
            vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 1.0],
            vec![0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0],
            vec![-0.0, 0.0, 1.0, -0.0, 0.0, -1.0, 0.0],
            vec![1.0, 2.0, f64::NAN, 3.0, 4.0, 5.0, 6.0, 7.0],
            vec![1.0, 2.0, 3.0, f64::INFINITY, 4.0, 5.0, 6.0],
            vec![1.0, 2.0, 3.0, f64::NEG_INFINITY, 4.0, 5.0, 6.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            vec![-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0],
            walk(64),
            walk(257),
        ]
    }

    #[test]
    fn median_matches_reference_bitwise() {
        for series in edge_series() {
            let n = series.len();
            for timeperiod in [0, 1, 2, 3, 4, 5, 7, 20, 21, 33, n, n + 1, n + 5] {
                let case = format!("median n={n} p={timeperiod}");
                assert_bits_eq(
                    &median(&series, timeperiod),
                    &reference_median(&series, timeperiod),
                    &case,
                );
            }
        }
    }

    #[test]
    fn mode_matches_reference_bitwise() {
        for series in edge_series() {
            let n = series.len();
            for timeperiod in [0, 1, 2, 3, 5, 20, n, n + 1] {
                for bins in [0, 1, 2, 3, 10, 64] {
                    let case = format!("mode n={n} p={timeperiod} bins={bins}");
                    assert_bits_eq(
                        &mode(&series, timeperiod, bins),
                        &reference_mode(&series, timeperiod, bins),
                        &case,
                    );
                }
            }
        }
    }

    #[test]
    fn mode_bin_boundary_tie_matches_reference() {
        // width lands exactly on representable boundaries, and the top value
        // hits the clamp: [0, 1, 2, 3] with bins=2 gives width 1.5, counts
        // [2, 2], and the left-most bin wins the tie -> 0.75.
        let real = [0.0, 1.0, 2.0, 3.0];
        let out = mode(&real, 4, 2);
        assert_eq!(out[3].to_bits(), 0.75_f64.to_bits());
        assert_bits_eq(&out, &reference_mode(&real, 4, 2), "mode boundary tie");

        // Exact bin edge with an odd count: width 2, indices 0/1/clamped 1.
        let exact = [0.0, 2.0, 4.0];
        assert_bits_eq(
            &mode(&exact, 3, 2),
            &reference_mode(&exact, 3, 2),
            "mode edge",
        );
    }

    #[test]
    fn median_bands_matches_reference_bitwise() {
        let close = walk(200);
        let high: Vec<f64> = close.iter().map(|c| c + 1.5).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.5).collect();
        for (timeperiod, atr_period) in [(1, 1), (3, 14), (4, 3), (25, 14), (200, 14), (201, 14)] {
            let (mid, upper, lower, ema) =
                median_bands(&high, &low, &close, timeperiod, atr_period, 2.0);
            let (rmid, rupper, rlower, rema) =
                reference_median_bands(&high, &low, &close, timeperiod, atr_period, 2.0);
            let case = format!("median_bands p={timeperiod} atr={atr_period}");
            assert_bits_eq(&mid, &rmid, &case);
            assert_bits_eq(&upper, &rupper, &case);
            assert_bits_eq(&lower, &rlower, &case);
            assert_bits_eq(&ema, &rema, &case);
        }
        let empty: [f64; 0] = [];
        let (mid, upper, lower, ema) = median_bands(&empty, &empty, &empty, 3, 14, 2.0);
        assert!(mid.is_empty() && upper.is_empty() && lower.is_empty() && ema.is_empty());
    }

    #[test]
    fn median_odd_window_golden() {
        let close = [1.0, 5.0, 3.0, 4.0, 2.0];
        let result = median(&close, 3);
        assert!(result[0].is_nan() && result[1].is_nan());
        assert!((result[2] - 3.0).abs() < 1e-12);
        assert!((result[3] - 4.0).abs() < 1e-12);
        assert!((result[4] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn median_even_window_averages_centres() {
        let close = [1.0, 2.0, 3.0, 4.0];
        let result = median(&close, 4);
        assert!(result[0].is_nan() && result[1].is_nan() && result[2].is_nan());
        assert!((result[3] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn median_period_one_is_identity() {
        let close = [7.0, 8.0, 9.0];
        let result = median(&close, 1);
        assert_eq!(result, close);
    }

    #[test]
    fn median_nan_in_window() {
        let close = [1.0, f64::NAN, 3.0];
        let result = median(&close, 3);
        assert!(result[2].is_nan());
    }

    #[test]
    fn median_empty_and_short() {
        assert!(median(&[], 3).is_empty());
        let short = median(&[1.0, 2.0], 3);
        assert_eq!(short.len(), 2);
        assert!(short.iter().all(|v| v.is_nan()));
    }

    /// Long series with periodic non-finites and plateaus of equal values:
    /// forces the sliding buffer to be torn down and rebuilt at every gap and
    /// exercises the equal-value tie order at scale.
    #[test]
    fn median_and_mode_survive_plateaus_and_gaps() {
        let mut series = walk(2_000);
        for (i, v) in series.iter_mut().enumerate() {
            if i % 97 == 0 {
                *v = f64::NAN;
            } else if i % 401 == 0 {
                *v = f64::INFINITY;
            } else if i % 37 < 9 {
                *v = 100.0;
            } else if i % 37 < 12 {
                *v = -0.0;
            } else if i % 37 < 15 {
                *v = 0.0;
            }
        }
        for timeperiod in [1, 2, 6, 7, 8, 13, 24, 25, 26, 40, 129] {
            assert_bits_eq(
                &median(&series, timeperiod),
                &reference_median(&series, timeperiod),
                &format!("median gaps p={timeperiod}"),
            );
            assert_bits_eq(
                &mode(&series, timeperiod, 10),
                &reference_mode(&series, timeperiod, 10),
                &format!("mode gaps p={timeperiod}"),
            );
        }
    }

    #[test]
    fn median_and_mode_path_thresholds_agree_with_reference() {
        // Straddle both hybrid thresholds so every path runs on the same data.
        let series = walk(300);
        for timeperiod in [
            MEDIAN_SLIDE_MIN_PERIOD - 1,
            MEDIAN_SLIDE_MIN_PERIOD,
            MEDIAN_SLIDE_MIN_PERIOD + 1,
            64,
        ] {
            assert_bits_eq(
                &median(&series, timeperiod),
                &reference_median(&series, timeperiod),
                &format!("median straddle p={timeperiod}"),
            );
        }
        for timeperiod in [
            MODE_DEQUE_MIN_PERIOD - 1,
            MODE_DEQUE_MIN_PERIOD,
            MODE_DEQUE_MIN_PERIOD + 1,
            64,
        ] {
            assert_bits_eq(
                &mode(&series, timeperiod, 10),
                &reference_mode(&series, timeperiod, 10),
                &format!("mode straddle p={timeperiod}"),
            );
        }
    }

    #[test]
    fn median_bands_compose_from_hl2_and_atr() {
        let high = [11.0, 12.0, 13.0, 14.0, 16.0, 17.0, 18.0, 19.0];
        let low = [9.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0, 17.0];
        let close = [10.0, 11.0, 12.0, 13.0, 15.0, 16.0, 17.0, 18.0];
        let (mid, upper, lower, mid_ema) = median_bands(&high, &low, &close, 3, 3, 2.0);
        let mut hl2 = [0.0; 8];
        for i in 0..8 {
            hl2[i] = 0.5 * (high[i] + low[i]);
        }
        let expected_mid = median(&hl2, 3);
        let atr = crate::volatility::atr(&high, &low, &close, 3);
        let expected_ema = crate::overlap::ema(&expected_mid, 3);
        for i in 0..8 {
            assert!(
                (mid[i].is_nan() && expected_mid[i].is_nan())
                    || (mid[i] - expected_mid[i]).abs() < 1e-12
            );
            if mid[i].is_finite() && atr[i].is_finite() {
                assert!((upper[i] - (mid[i] + 2.0 * atr[i])).abs() < 1e-12);
                assert!((lower[i] - (mid[i] - 2.0 * atr[i])).abs() < 1e-12);
            } else {
                assert!(upper[i].is_nan() && lower[i].is_nan());
            }
            assert!(
                (mid_ema[i].is_nan() && expected_ema[i].is_nan())
                    || (mid_ema[i] - expected_ema[i]).abs() < 1e-12
            );
        }
    }

    #[test]
    fn mode_constant_window() {
        let real = [4.0, 4.0, 4.0, 4.0];
        let result = mode(&real, 3, 10);
        assert!(result[0].is_nan() && result[1].is_nan());
        assert!((result[2] - 4.0).abs() < 1e-12);
        assert!((result[3] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn mode_binned_golden() {
        // Window [1, 1, 1, 2, 3], bins=2 → min=1, max=3, width=1.
        // bin0 [1, 2) gets three 1s; bin1 [2, 3] gets 2 and 3.
        // Mode is the centre of bin0: 1.5.
        let real = [1.0, 1.0, 1.0, 2.0, 3.0];
        let result = mode(&real, 5, 2);
        assert!((result[4] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn mode_empty() {
        assert!(mode(&[], 5, 10).is_empty());
    }

    #[test]
    fn median_bands_length_mismatch_returns_nan() {
        let high = [11.0, 12.0, 13.0, 14.0];
        let low = [9.0, 10.0, 11.0, 12.0];
        let close = [10.0, 11.0, 12.0, 13.0];
        let short = [1.0, 2.0];
        let long = [1.0, 2.0, 3.0, 4.0, 5.0];
        for other in [short.as_slice(), long.as_slice()] {
            for (mid, upper, lower, mid_ema) in [
                median_bands(&high, other, &close, 3, 3, 2.0),
                median_bands(&high, &low, other, 3, 3, 2.0),
            ] {
                for out in [&mid, &upper, &lower, &mid_ema] {
                    assert_eq!(out.len(), high.len());
                    assert!(out.iter().all(|v| v.is_nan()));
                }
            }
        }
    }
}
