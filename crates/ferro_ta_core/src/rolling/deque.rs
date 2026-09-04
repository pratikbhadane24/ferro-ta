//! Monotonic deques and their single-series drivers.

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
}
