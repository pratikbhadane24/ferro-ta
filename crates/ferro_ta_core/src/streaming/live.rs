//! Live-bar support for streaming indicators.
//!
//! Exchanges deliver many ticks per bar. [`Live::update`] with
//! `closed = false` computes the indicator for the still-forming bar from the
//! last *committed* state without mutating it, so every tick of the forming
//! bar replaces the previous preview. `closed = true` commits the bar.

use super::{
    StreamingATR, StreamingBBands, StreamingCooldown, StreamingCross, StreamingEMA, StreamingMACD,
    StreamingRSI, StreamingSMA, StreamingStoch, StreamingStreak, StreamingSupertrend,
    StreamingVWAP,
};

// ---------------------------------------------------------------------------
// StreamValue — inputs and outputs of streaming indicators
// ---------------------------------------------------------------------------

/// A value flowing into or out of a streaming indicator.
pub trait StreamValue: Copy + std::fmt::Debug {
    /// True if any floating-point component is NaN.
    fn has_nan(&self) -> bool;
    /// True if every floating-point component is finite (no NaN / ±inf).
    fn all_finite(&self) -> bool;
    /// The "no value" sentinel emitted during warmup.
    fn nan() -> Self;
}

impl StreamValue for f64 {
    fn has_nan(&self) -> bool {
        self.is_nan()
    }
    fn all_finite(&self) -> bool {
        self.is_finite()
    }
    fn nan() -> Self {
        f64::NAN
    }
}

impl StreamValue for (f64, f64) {
    fn has_nan(&self) -> bool {
        self.0.is_nan() || self.1.is_nan()
    }
    fn all_finite(&self) -> bool {
        self.0.is_finite() && self.1.is_finite()
    }
    fn nan() -> Self {
        (f64::NAN, f64::NAN)
    }
}

impl StreamValue for (f64, f64, f64) {
    fn has_nan(&self) -> bool {
        self.0.is_nan() || self.1.is_nan() || self.2.is_nan()
    }
    fn all_finite(&self) -> bool {
        self.0.is_finite() && self.1.is_finite() && self.2.is_finite()
    }
    fn nan() -> Self {
        (f64::NAN, f64::NAN, f64::NAN)
    }
}

impl StreamValue for (f64, f64, f64, f64) {
    fn has_nan(&self) -> bool {
        self.0.is_nan() || self.1.is_nan() || self.2.is_nan() || self.3.is_nan()
    }
    fn all_finite(&self) -> bool {
        self.0.is_finite() && self.1.is_finite() && self.2.is_finite() && self.3.is_finite()
    }
    fn nan() -> Self {
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    }
}

/// Condition events (`StreamingCross` etc.): no NaN state; `0` = no event.
impl StreamValue for i8 {
    fn has_nan(&self) -> bool {
        false
    }
    fn all_finite(&self) -> bool {
        true
    }
    fn nan() -> Self {
        0
    }
}

/// Supertrend output: (line, direction). Direction 0 means warmup.
impl StreamValue for (f64, i8) {
    fn has_nan(&self) -> bool {
        self.0.is_nan()
    }
    fn all_finite(&self) -> bool {
        self.0.is_finite()
    }
    fn nan() -> Self {
        (f64::NAN, 0)
    }
}

// ---------------------------------------------------------------------------
// StreamingIndicator — common interface over the concrete structs
// ---------------------------------------------------------------------------

/// Common interface implemented by every streaming indicator.
pub trait StreamingIndicator: Clone {
    type Input: StreamValue;
    type Output: StreamValue;

    /// Commit one bar and return the indicator value for it.
    fn update(&mut self, input: Self::Input) -> Self::Output;

    /// Restart from scratch.
    fn reset(&mut self);

    /// Value `update(input)` would return, without changing state.
    fn peek(&self, input: Self::Input) -> Self::Output {
        self.clone().update(input)
    }
}

macro_rules! impl_streaming_indicator {
    (
        $ty:ty, $input:ty, $output:ty, |$s:ident, $pat:pat_param| $call:expr
        $(, peek = |$ps:ident, $ppat:pat_param| $peek:expr)? $(,)?
    ) => {
        impl StreamingIndicator for $ty {
            type Input = $input;
            type Output = $output;

            fn update(&mut self, input: $input) -> $output {
                let $s = self;
                let $pat = input;
                $call
            }

            fn reset(&mut self) {
                <$ty>::reset(self)
            }

            $(
                /// Allocation-free override of the clone-based default.
                fn peek(&self, input: $input) -> $output {
                    let $ps = self;
                    let $ppat = input;
                    $peek
                }
            )?
        }
    };
}

impl_streaming_indicator!(
    StreamingSMA,
    f64,
    f64,
    |s, v| s.update(v),
    peek = |s, v| s.peek_value(v)
);
impl_streaming_indicator!(StreamingEMA, f64, f64, |s, v| s.update(v));
impl_streaming_indicator!(StreamingRSI, f64, f64, |s, v| s.update(v));
impl_streaming_indicator!(
    StreamingBBands,
    f64,
    (f64, f64, f64),
    |s, v| s.update(v),
    peek = |s, v| s.peek_value(v)
);
impl_streaming_indicator!(StreamingMACD, f64, (f64, f64, f64), |s, v| s.update(v));
impl_streaming_indicator!(StreamingATR, (f64, f64, f64), f64, |s, (h, l, c)| s
    .update(h, l, c));
impl_streaming_indicator!(
    StreamingStoch,
    (f64, f64, f64),
    (f64, f64),
    |s, (h, l, c)| s.update(h, l, c),
    peek = |s, (h, l, c)| s.peek_value(h, l, c)
);
impl_streaming_indicator!(StreamingSupertrend, (f64, f64, f64), (f64, i8), |s,
                                                                            (
    h,
    l,
    c,
)| s
    .update(h, l, c));
impl_streaming_indicator!(
    StreamingVWAP,
    (f64, f64, f64, f64),
    f64,
    |s, (h, l, c, v)| s.update(h, l, c, v)
);

impl_streaming_indicator!(StreamingCross, (f64, f64), i8, |s, (a, b)| s.update(a, b));
impl_streaming_indicator!(StreamingStreak, f64, i8, |s, v| s.update(v));
impl_streaming_indicator!(StreamingCooldown, f64, i8, |s, v| s.update(v));

// ---------------------------------------------------------------------------
// Live — closed-bar commits, forming-bar previews, NaN guard
// ---------------------------------------------------------------------------

/// Wraps a streaming indicator with live-bar semantics and introspection.
#[derive(Clone, Debug)]
pub struct Live<T: StreamingIndicator> {
    inner: T,
    last: T::Output,
    count: usize,
}

impl<T: StreamingIndicator> Live<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            last: T::Output::nan(),
            count: 0,
        }
    }

    /// Feed one tick.
    ///
    /// `closed = true` commits the bar. `closed = false` previews the forming
    /// bar from committed state; state is left untouched.
    pub fn update(&mut self, input: T::Input, closed: bool) -> T::Output {
        if !input.all_finite() {
            return self.on_bad_input();
        }
        if !closed {
            return self.inner.peek(input);
        }
        self.last = self.inner.update(input);
        self.count += 1;
        self.last
    }

    /// Policy for a tick whose input contains NaN or ±inf (bad print, feed
    /// gap).
    ///
    /// Feeding a non-finite value into the indicator poisons running sums
    /// (inf - inf = NaN once it leaves the window) and recursive averages
    /// forever, so it never reaches `self.inner.update`.
    ///
    /// Policy: *skip*. The tick yields the NaN sentinel so the gap stays
    /// visible to the caller (holding the last value would let a strategy act
    /// on stale data), and state, `last` and `count` are left untouched — the
    /// next clean bar continues as if the bad tick never arrived.
    fn on_bad_input(&self) -> T::Output {
        T::Output::nan()
    }

    /// Restart from scratch.
    pub fn reset(&mut self) {
        self.inner.reset();
        self.last = T::Output::nan();
        self.count = 0;
    }

    /// Number of committed (closed) bars.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Value emitted by the last committed bar (NaN before the first).
    pub fn last(&self) -> T::Output {
        self.last
    }

    /// True once the last committed bar produced a non-NaN value.
    pub fn is_ready(&self) -> bool {
        !self.last.has_nan()
    }

    pub fn inner(&self) -> &T {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const N_BARS: usize = 300;
    const TICKS_PER_BAR: usize = 4;

    /// Deterministic OHLCV path (LCG) so tests need no RNG dependency.
    fn bars(n: usize) -> Vec<(f64, f64, f64, f64)> {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut close = 100.0;
        (0..n)
            .map(|_| {
                close += next() - 0.5;
                let high = close + next();
                let low = close - next();
                (high, low, close, 1.0 + 10.0 * next())
            })
            .collect()
    }

    /// Same NaN-aware bitwise equality the batch parity tests rely on.
    fn same<O: std::fmt::Debug>(a: O, b: O) -> bool {
        format!("{a:?}") == format!("{b:?}")
    }

    /// Previews interleaved before every close must not perturb the committed
    /// stream: output must be bit-identical to plain `update` calls.
    fn assert_previews_are_pure<T, F>(make: impl Fn() -> T, input_of: F)
    where
        T: StreamingIndicator,
        F: Fn((f64, f64, f64, f64)) -> T::Input,
    {
        let data = bars(N_BARS);
        let mut plain = make();
        let mut live = Live::new(make());
        for (i, &bar) in data.iter().enumerate() {
            for t in 0..TICKS_PER_BAR {
                let wobble = 0.01 * (t as f64 + 1.0);
                let tick = (bar.0 + wobble, bar.1 - wobble, bar.2 + wobble, bar.3);
                let _ = live.update(input_of(tick), false);
            }
            let expected = plain.update(input_of(bar));
            let preview = live.update(input_of(bar), false);
            let got = live.update(input_of(bar), true);
            assert!(
                same(preview, expected),
                "bar {i}: preview {preview:?} != {expected:?}"
            );
            assert!(
                same(got, expected),
                "bar {i}: commit {got:?} != {expected:?}"
            );
        }
        assert_eq!(live.count(), N_BARS);
    }

    #[test]
    fn previews_are_pure_for_every_indicator() {
        let close = |b: (f64, f64, f64, f64)| b.2;
        let hlc = |b: (f64, f64, f64, f64)| (b.0, b.1, b.2);
        assert_previews_are_pure(|| StreamingSMA::new(10).unwrap(), close);
        assert_previews_are_pure(|| StreamingEMA::new(10).unwrap(), close);
        assert_previews_are_pure(|| StreamingRSI::new(14).unwrap(), close);
        assert_previews_are_pure(|| StreamingBBands::new(20, 2.0, 2.0).unwrap(), close);
        assert_previews_are_pure(|| StreamingMACD::new(12, 26, 9).unwrap(), close);
        assert_previews_are_pure(|| StreamingATR::new(14).unwrap(), hlc);
        assert_previews_are_pure(|| StreamingStoch::new(5, 3, 3).unwrap(), hlc);
        assert_previews_are_pure(|| StreamingSupertrend::new(7, 3.0).unwrap(), hlc);
        assert_previews_are_pure(StreamingVWAP::new, |b| b);
        assert_previews_are_pure(|| StreamingCross::new(0.1).unwrap(), |b| (b.2, b.0 - 0.5));
        assert_previews_are_pure(|| StreamingStreak::new(3).unwrap(), |b| b.2 - b.1 - 0.5);
        assert_previews_are_pure(|| StreamingCooldown::new(2).unwrap(), |b| b.2 - 100.0);
    }

    #[test]
    fn introspection_tracks_committed_bars_only() {
        let mut live = Live::new(StreamingSMA::new(3).unwrap());
        assert!(!live.is_ready());
        live.update(1.0, true);
        live.update(2.0, true);
        live.update(99.0, false);
        assert_eq!(live.count(), 2);
        assert!(!live.is_ready());
        assert_eq!(live.update(3.0, true), 2.0);
        assert!(live.is_ready());
        assert_eq!(live.last(), 2.0);

        live.reset();
        assert_eq!(live.count(), 0);
        assert!(!live.is_ready());
        assert!(live.last().is_nan());
    }

    #[test]
    fn clone_snapshot_diverges_independently() {
        let mut a = Live::new(StreamingEMA::new(3).unwrap());
        for v in [1.0, 2.0, 3.0] {
            a.update(v, true);
        }
        let mut b = a.clone();
        assert_eq!(a.update(10.0, true), b.update(10.0, true));
        assert_ne!(a.update(0.0, true), b.update(20.0, true));
    }

    /// A NaN tick must not poison state: once it is behind us, outputs must
    /// match a stream that never saw it.
    #[test]
    fn nan_input_does_not_poison_state() {
        let clean = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut reference = StreamingSMA::new(3).unwrap();
        let expected: Vec<f64> = clean.iter().map(|&v| reference.update(v)).collect();

        let mut live = Live::new(StreamingSMA::new(3).unwrap());
        let mut got = Vec::new();
        for (i, &v) in clean.iter().enumerate() {
            if i == 3 {
                assert!(live.update(f64::NAN, false).is_nan());
                assert!(live.update(f64::NAN, true).is_nan());
                assert_eq!(live.count(), 3, "NaN tick must not count as a bar");
                assert_eq!(live.last(), 2.0, "NaN tick must not overwrite last");
            }
            got.push(live.update(v, true));
        }
        assert!(
            same(got.as_slice(), expected.as_slice()),
            "{got:?} != {expected:?}"
        );
    }

    /// Infinity is as poisonous as NaN for running sums (inf - inf = NaN once
    /// it leaves the window), so it must be skipped the same way.
    #[test]
    fn infinite_input_does_not_poison_state() {
        let mut reference = StreamingSMA::new(2).unwrap();
        let mut live = Live::new(StreamingSMA::new(2).unwrap());
        for v in [1.0, 2.0] {
            reference.update(v);
            live.update(v, true);
        }
        for bad in [f64::INFINITY, f64::NEG_INFINITY] {
            assert!(live.update(bad, false).is_nan());
            assert!(live.update(bad, true).is_nan());
        }
        assert_eq!(live.count(), 2);
        for v in [3.0, 4.0, 5.0] {
            assert_eq!(live.update(v, true), reference.update(v));
        }
    }

    /// The allocation-free `peek_value` overrides must be bit-identical to the
    /// clone-and-update reference at every bar, warm-up included.
    #[test]
    fn peek_matches_clone_update_without_allocating() {
        let data = bars(N_BARS);
        let mut sma = StreamingSMA::new(10).unwrap();
        let mut bb = StreamingBBands::new(20, 2.0, 1.5).unwrap();
        let mut stoch = StreamingStoch::new(5, 3, 3).unwrap();
        for (i, &(h, l, c, _)) in data.iter().enumerate() {
            let probe = c + 0.37;
            assert!(
                same(sma.peek_value(probe), sma.clone().update(probe)),
                "sma bar {i}"
            );
            assert!(
                same(bb.peek_value(probe), bb.clone().update(probe)),
                "bbands bar {i}"
            );
            assert!(
                same(
                    stoch.peek_value(h + 0.2, l - 0.2, probe),
                    stoch.clone().update(h + 0.2, l - 0.2, probe)
                ),
                "stoch bar {i}"
            );
            sma.update(c);
            bb.update(c);
            stoch.update(h, l, c);
        }
    }
}
