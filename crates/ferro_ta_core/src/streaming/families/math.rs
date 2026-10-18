//! Price transforms, element-wise math and rolling math streams.

use std::collections::VecDeque;

use crate::streaming::{validate_timeperiod, StreamingError};

/// Registry entries for `price_transform`, `math` and `math_ops`.
#[macro_export]
#[doc(hidden)]
macro_rules! stream_family_math {
    ($($ctx:tt)*) => {
        $crate::__stream_chain! {
            [
                avgprice => StreamingAvgprice, doc "Streaming average price: (open + high + low + close) / 4.",
                    params(), inputs(open, high, low, close), outputs(out: f64),
                    windowed(lookback = 1, batch = $crate::price_transform::avgprice(open, high, low, close));
                medprice => StreamingMedprice, doc "Streaming median price: (high + low) / 2.",
                    params(), inputs(high, low), outputs(out: f64),
                    windowed(lookback = 1, batch = $crate::price_transform::medprice(high, low));
                typprice => StreamingTypprice, doc "Streaming typical price: (high + low + close) / 3.",
                    params(), inputs(high, low, close), outputs(out: f64),
                    windowed(lookback = 1, batch = $crate::price_transform::typprice(high, low, close));
                wclprice => StreamingWclprice, doc "Streaming weighted close price: (high + low + 2 * close) / 4.",
                    params(), inputs(high, low, close), outputs(out: f64),
                    windowed(lookback = 1, batch = $crate::price_transform::wclprice(high, low, close));
                add => StreamingAdd, doc "Streaming element-wise a + b.",
                    params(), inputs(a, b), outputs(out: f64),
                    windowed(lookback = 1, batch = $crate::math::add(a, b));
                sub => StreamingSub, doc "Streaming element-wise a - b.",
                    params(), inputs(a, b), outputs(out: f64),
                    windowed(lookback = 1, batch = $crate::math::sub(a, b));
                mult => StreamingMult, doc "Streaming element-wise a * b.",
                    params(), inputs(a, b), outputs(out: f64),
                    windowed(lookback = 1, batch = $crate::math::mult(a, b));
                div => StreamingDiv, doc "Streaming element-wise a / b (NaN where b = 0).",
                    params(), inputs(a, b), outputs(out: f64),
                    windowed(lookback = 1, batch = $crate::math::div(a, b));
                sum => StreamingSum, doc "Streaming rolling sum over `timeperiod` bars.",
                    params(timeperiod: period = 30), inputs(real), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::math::sum(real, timeperiod));
                max => StreamingMax, doc "Streaming rolling maximum over `timeperiod` bars.",
                    params(timeperiod: period = 30), inputs(real), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::math::max(real, timeperiod));
                min => StreamingMin, doc "Streaming rolling minimum over `timeperiod` bars.",
                    params(timeperiod: period = 30), inputs(real), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::math::min(real, timeperiod));
                sliding_max => StreamingSlidingMax, doc "Streaming sliding maximum over `timeperiod` bars.",
                    params(timeperiod: period = 30), inputs(real), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::math::sliding_max(real, timeperiod));
                sliding_min => StreamingSlidingMin, doc "Streaming sliding minimum over `timeperiod` bars.",
                    params(timeperiod: period = 30), inputs(real), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::math::sliding_min(real, timeperiod));
                rolling_sum => StreamingRollingSum, doc "Streaming rolling sum over `timeperiod` bars.",
                    params(timeperiod: period = 30), inputs(real), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::math_ops::rolling_sum(real, timeperiod));
                rolling_max => StreamingRollingMax, doc "Streaming rolling maximum over `timeperiod` bars.",
                    params(timeperiod: period = 30), inputs(real), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::math_ops::rolling_max(real, timeperiod));
                rolling_min => StreamingRollingMin, doc "Streaming rolling minimum over `timeperiod` bars.",
                    params(timeperiod: period = 30), inputs(real), outputs(out: f64),
                    windowed(lookback = timeperiod, batch = $crate::math_ops::rolling_min(real, timeperiod));
                rolling_maxindex => StreamingRollingMaxindex, doc "Streaming index of the rolling maximum: the absolute 0-based bar index since the first bar (most recent on ties), -1 during warm-up.",
                    params(timeperiod: period = 30), inputs(real), outputs(out: i64),
                    native(lookback = timeperiod, batch = $crate::math_ops::rolling_maxindex(real, timeperiod));
                rolling_minindex => StreamingRollingMinindex, doc "Streaming index of the rolling minimum: the absolute 0-based bar index since the first bar (most recent on ties), -1 during warm-up.",
                    params(timeperiod: period = 30), inputs(real), outputs(out: i64),
                    native(lookback = timeperiod, batch = $crate::math_ops::rolling_minindex(real, timeperiod));
            ]
            $($ctx)*
        }
    };
}

crate::stream_family_math! {
    gen = crate::__stream_define;
    acc = [];
    rest = [];
}

// ---------------------------------------------------------------------------
// Native: rolling extreme index
// ---------------------------------------------------------------------------

/// Monotonic deque of `(absolute bar index, value)` whose front is the window
/// extreme. A windowed replay would report window-relative indices, so the
/// index streams count committed bars themselves.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct ExtremeIndex {
    timeperiod: usize,
    entries: VecDeque<(u64, f64)>,
    /// Committed bars so far (the next bar's absolute index).
    bars: u64,
}

impl ExtremeIndex {
    fn new(timeperiod: usize) -> Result<Self, StreamingError> {
        validate_timeperiod(timeperiod, "timeperiod", 1)?;
        Ok(Self {
            timeperiod,
            entries: VecDeque::new(),
            bars: 0,
        })
    }

    /// Commit `x`. `dominated(old, x)` pops `old` from the back; popping on
    /// equality makes the most recent of equal extremes win, as in batch.
    fn update(&mut self, x: f64, dominated: fn(f64, f64) -> bool) -> i64 {
        let i = self.bars;
        self.bars += 1;
        let period = self.timeperiod as u64;
        while self.entries.front().is_some_and(|&(j, _)| j + period <= i) {
            self.entries.pop_front();
        }
        while self.entries.back().is_some_and(|&(_, v)| dominated(v, x)) {
            self.entries.pop_back();
        }
        self.entries.push_back((i, x));
        if i + 1 >= period {
            self.entries.front().map_or(-1, |&(j, _)| j as i64)
        } else {
            -1
        }
    }

    fn reset(&mut self) {
        self.entries.clear();
        self.bars = 0;
    }

    fn validate(&self, dominated: fn(f64, f64) -> bool) -> Result<(), StreamingError> {
        validate_timeperiod(self.timeperiod, "timeperiod", 1)?;
        let period = self.timeperiod as u64;
        let in_window = |j: u64| j < self.bars && j + period >= self.bars;
        let ordered = self
            .entries
            .iter()
            .zip(self.entries.iter().skip(1))
            .all(|(a, b)| a.0 < b.0 && !dominated(a.1, b.1));
        if !self.entries.iter().all(|&(j, _)| in_window(j)) || !ordered {
            return Err(StreamingError(
                "extreme-index deque is inconsistent with its window".to_string(),
            ));
        }
        Ok(())
    }
}

macro_rules! extreme_index_stream {
    ($(#[$doc:meta])* $Ty:ident, $dominated:expr) => {
        $(#[$doc])*
        #[derive(Clone, Debug)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        pub struct $Ty {
            inner: ExtremeIndex,
        }

        impl $Ty {
            const DOMINATED: fn(f64, f64) -> bool = $dominated;

            pub fn new(timeperiod: usize) -> Result<Self, StreamingError> {
                Ok(Self {
                    inner: ExtremeIndex::new(timeperiod)?,
                })
            }

            /// Commit one bar; returns the absolute bar index of the extreme
            /// (`-1` during the first `timeperiod - 1` bars).
            pub fn update(&mut self, real: f64) -> i64 {
                self.inner.update(real, Self::DOMINATED)
            }

            pub fn reset(&mut self) {
                self.inner.reset();
            }

            pub fn validate(&self) -> Result<(), StreamingError> {
                self.inner.validate(Self::DOMINATED)
            }

            pub fn timeperiod(&self) -> usize {
                self.inner.timeperiod
            }
        }
    };
}

extreme_index_stream!(
    /// Streaming index of the rolling maximum (absolute bar index).
    StreamingRollingMaxindex,
    |old, x| old <= x
);
extreme_index_stream!(
    /// Streaming index of the rolling minimum (absolute bar index).
    StreamingRollingMinindex,
    |old, x| old >= x
);
