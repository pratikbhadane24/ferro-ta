//! Session-aligned tick → bar builder.
//!
//! Buckets ticks into bars on a session-local grid (`open + k·interval`,
//! the last bar cut at the session close), so 5-minute NSE bars start at
//! 09:15, 09:20, … regardless of when the first tick arrives. Gaps without
//! ticks produce no bars, so holidays need no calendar. Feed the forming bar
//! to an indicator with `closed = false` and each returned closed bar with
//! `closed = true`.

use super::StreamingError;

const MS_PER_SEC: i64 = 1_000;
const SECS_PER_DAY: i64 = 86_400;
const MS_PER_DAY: i64 = SECS_PER_DAY * MS_PER_SEC;
/// Real-world UTC offsets span -12:00 .. +14:00.
const MAX_UTC_OFFSET_SECS: i64 = 14 * 3_600;

/// Bar grid: interval plus a daily session window in local time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BarSpec {
    pub interval_secs: i64,
    /// Session open, seconds after local midnight.
    pub session_open_secs: i64,
    /// Session close (exclusive), seconds after local midnight.
    pub session_close_secs: i64,
    /// Local time = UTC + this offset. Fixed: no time-zone or DST rules, so
    /// for DST markets build a new builder when the offset changes.
    pub utc_offset_secs: i64,
}

impl BarSpec {
    /// NSE/BSE cash and F&O: 09:15–15:30 IST (UTC+05:30).
    pub fn nse(interval_secs: i64) -> Self {
        Self {
            interval_secs,
            session_open_secs: 9 * 3_600 + 15 * 60,
            session_close_secs: 15 * 3_600 + 30 * 60,
            utc_offset_secs: 5 * 3_600 + 30 * 60,
        }
    }

    fn validate(&self) -> Result<(), StreamingError> {
        let err = |msg: String| Err(StreamingError(msg));
        if self.interval_secs <= 0 || self.interval_secs > SECS_PER_DAY {
            return err(format!(
                "interval_secs must be in 1..={SECS_PER_DAY}, got {}",
                self.interval_secs
            ));
        }
        if self.session_open_secs < 0 || self.session_close_secs > SECS_PER_DAY {
            return err(format!(
                "session must lie within one day (0..={SECS_PER_DAY} s), got {}..{}",
                self.session_open_secs, self.session_close_secs
            ));
        }
        if self.session_open_secs >= self.session_close_secs {
            return err(format!(
                "session_open_secs must be < session_close_secs, got {} >= {}",
                self.session_open_secs, self.session_close_secs
            ));
        }
        if self.utc_offset_secs.abs() > MAX_UTC_OFFSET_SECS {
            return err(format!(
                "utc_offset_secs must be within ±{MAX_UTC_OFFSET_SECS}, got {}",
                self.utc_offset_secs
            ));
        }
        Ok(())
    }
}

/// One OHLCV bar covering `[start_ms, end_ms)` (UTC epoch milliseconds).
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Bar {
    pub start_ms: i64,
    pub end_ms: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub ticks: u32,
}

/// What happened to a tick. Only `Accepted` ticks change state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum TickStatus {
    Accepted = 0,
    /// Older than the last accepted tick, or inside an already-closed bar.
    Late = 1,
    /// Identical (timestamp, price, qty) to the last accepted tick.
    Duplicate = 2,
    /// Outside the daily session window.
    OutOfSession = 3,
    /// Non-finite price/qty or negative qty.
    Invalid = 4,
}

impl TickStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Late => "late",
            Self::Duplicate => "duplicate",
            Self::OutOfSession => "out_of_session",
            Self::Invalid => "invalid",
        }
    }
}

/// Builds session-aligned OHLCV bars from ticks.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BarBuilder {
    spec: BarSpec,
    forming: Option<Bar>,
    last_tick: Option<(i64, f64, f64)>,
    /// End of the last closed bar; earlier ticks are late.
    closed_until_ms: i64,
}

impl BarBuilder {
    pub fn new(spec: BarSpec) -> Result<Self, StreamingError> {
        spec.validate()?;
        Ok(Self {
            spec,
            forming: None,
            last_tick: None,
            closed_until_ms: i64::MIN,
        })
    }

    /// Feed one tick. Returns its status and the bar it closed, if any.
    pub fn on_tick(&mut self, ts_ms: i64, price: f64, qty: f64) -> (TickStatus, Option<Bar>) {
        if !price.is_finite() || !qty.is_finite() || qty < 0.0 {
            return (TickStatus::Invalid, None);
        }
        let (start_ms, end_ms) = match self.bucket(ts_ms) {
            Ok(Some(bounds)) => bounds,
            Ok(None) => return (TickStatus::OutOfSession, None),
            Err(status) => return (status, None),
        };
        if ts_ms < self.closed_until_ms || self.last_tick.is_some_and(|(t, ..)| ts_ms < t) {
            return (TickStatus::Late, None);
        }
        if self.last_tick == Some((ts_ms, price, qty)) {
            return (TickStatus::Duplicate, None);
        }
        self.last_tick = Some((ts_ms, price, qty));

        let closed = match &mut self.forming {
            Some(bar) if bar.start_ms == start_ms => {
                bar.high = bar.high.max(price);
                bar.low = bar.low.min(price);
                bar.close = price;
                bar.volume += qty;
                bar.ticks += 1;
                None
            }
            _ => self.forming.take(),
        };
        if let Some(bar) = closed {
            self.closed_until_ms = bar.end_ms;
        }
        if self.forming.is_none() {
            self.forming = Some(Bar {
                start_ms,
                end_ms,
                open: price,
                high: price,
                low: price,
                close: price,
                volume: qty,
                ticks: 1,
            });
        }
        (TickStatus::Accepted, closed)
    }

    /// Close the forming bar once the clock reaches its end, even with no
    /// further ticks. Never creates bars.
    pub fn on_time(&mut self, now_ms: i64) -> Option<Bar> {
        let bar = self.forming.filter(|bar| now_ms >= bar.end_ms)?;
        self.forming = None;
        self.closed_until_ms = bar.end_ms;
        Some(bar)
    }

    /// The bar currently being built, if any.
    pub fn forming(&self) -> Option<Bar> {
        self.forming
    }

    pub fn spec(&self) -> BarSpec {
        self.spec
    }

    pub fn reset(&mut self) {
        self.forming = None;
        self.last_tick = None;
        self.closed_until_ms = i64::MIN;
    }

    /// `[start, end)` of the bar containing `ts_ms`, `None` off-session, or
    /// `Err(Invalid)` when the timestamp is too extreme to place.
    fn bucket(&self, ts_ms: i64) -> Result<Option<(i64, i64)>, TickStatus> {
        // Spec fields are validated to be within a day, so only arithmetic
        // involving `ts_ms` can overflow; such timestamps are invalid.
        let overflow = TickStatus::Invalid;
        let offset_ms = self.spec.utc_offset_secs * MS_PER_SEC;
        let local = ts_ms.checked_add(offset_ms).ok_or(overflow)?;
        let day_start = local
            .div_euclid(MS_PER_DAY)
            .checked_mul(MS_PER_DAY)
            .ok_or(overflow)?;
        let time_of_day = local - day_start;
        let open = self.spec.session_open_secs * MS_PER_SEC;
        let close = self.spec.session_close_secs * MS_PER_SEC;
        if time_of_day < open || time_of_day >= close {
            return Ok(None);
        }
        let interval = self.spec.interval_secs * MS_PER_SEC;
        let start = open + (time_of_day - open) / interval * interval;
        let end = (start + interval).min(close);
        let to_utc = |t: i64| day_start.checked_add(t)?.checked_sub(offset_ms);
        match (to_utc(start), to_utc(end)) {
            (Some(s), Some(e)) => Ok(Some((s, e))),
            _ => Err(overflow),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Arbitrary trading day (days since the Unix epoch).
    const DAY: i64 = 20_377;
    const IST_OFFSET_MS: i64 = 19_800_000;

    /// UTC epoch ms for an IST wall-clock time on `DAY` (+ `day_shift`).
    fn ist(day_shift: i64, h: i64, m: i64, s: i64, ms: i64) -> i64 {
        let local = ((h * 60 + m) * 60 + s) * 1000 + ms;
        (DAY + day_shift) * 86_400_000 + local - IST_OFFSET_MS
    }

    fn five_min() -> BarBuilder {
        BarBuilder::new(BarSpec::nse(300)).unwrap()
    }

    #[test]
    fn ticks_bucket_on_session_grid() {
        let mut b = five_min();
        assert_eq!(
            b.on_tick(ist(0, 9, 15, 0, 500), 100.0, 2.0),
            (TickStatus::Accepted, None)
        );
        assert_eq!(
            b.on_tick(ist(0, 9, 17, 0, 0), 103.0, 1.0).0,
            TickStatus::Accepted
        );
        assert_eq!(
            b.on_tick(ist(0, 9, 18, 0, 0), 99.0, 4.0).0,
            TickStatus::Accepted
        );
        assert_eq!(
            b.on_tick(ist(0, 9, 19, 59, 0), 101.0, 3.0).0,
            TickStatus::Accepted
        );
        let bar = b.forming().unwrap();
        assert_eq!(
            bar,
            Bar {
                start_ms: ist(0, 9, 15, 0, 0),
                end_ms: ist(0, 9, 20, 0, 0),
                open: 100.0,
                high: 103.0,
                low: 99.0,
                close: 101.0,
                volume: 10.0,
                ticks: 4,
            }
        );
    }

    #[test]
    fn tick_on_boundary_opens_next_bar() {
        let mut b = five_min();
        b.on_tick(ist(0, 9, 19, 0, 0), 100.0, 1.0);
        let (status, closed) = b.on_tick(ist(0, 9, 20, 0, 0), 105.0, 1.0);
        assert_eq!(status, TickStatus::Accepted);
        let closed = closed.expect("boundary tick closes the 09:15 bar");
        assert_eq!(closed.end_ms, ist(0, 9, 20, 0, 0));
        assert_eq!(closed.close, 100.0);
        assert_eq!(b.forming().unwrap().start_ms, ist(0, 9, 20, 0, 0));
        assert_eq!(b.forming().unwrap().open, 105.0);
    }

    #[test]
    fn on_time_closes_bar_without_ticks() {
        let mut b = five_min();
        b.on_tick(ist(0, 9, 16, 0, 0), 100.0, 1.0);
        assert_eq!(b.on_time(ist(0, 9, 19, 59, 999)), None);
        let closed = b.on_time(ist(0, 9, 20, 0, 0)).expect("closes on time");
        assert_eq!(closed.start_ms, ist(0, 9, 15, 0, 0));
        assert_eq!(b.forming(), None);
        assert_eq!(b.on_time(ist(0, 9, 30, 0, 0)), None, "no synthetic bars");
    }

    #[test]
    fn late_and_duplicate_ticks_dropped() {
        let mut b = five_min();
        b.on_tick(ist(0, 9, 16, 0, 0), 100.0, 1.0);
        assert_eq!(
            b.on_tick(ist(0, 9, 16, 0, 0), 100.0, 1.0),
            (TickStatus::Duplicate, None)
        );
        assert_eq!(
            b.on_tick(ist(0, 9, 16, 0, 0), 100.5, 1.0).0,
            TickStatus::Accepted,
            "same ms, different trade"
        );
        assert_eq!(
            b.on_tick(ist(0, 9, 15, 59, 0), 99.0, 1.0),
            (TickStatus::Late, None)
        );
        assert_eq!(b.forming().unwrap().ticks, 2);

        b.on_time(ist(0, 9, 20, 0, 0));
        assert_eq!(
            b.on_tick(ist(0, 9, 16, 30, 0), 101.0, 1.0).0,
            TickStatus::Late,
            "tick for a bar already closed by on_time"
        );
    }

    #[test]
    fn out_of_session_tick_is_rejected() {
        let mut b = five_min();
        assert_eq!(
            b.on_tick(ist(0, 9, 14, 59, 0), 100.0, 1.0),
            (TickStatus::OutOfSession, None)
        );
        assert_eq!(
            b.on_tick(ist(0, 15, 30, 0, 0), 100.0, 1.0),
            (TickStatus::OutOfSession, None)
        );
        assert_eq!(b.forming(), None);
    }

    #[test]
    fn invalid_ticks_rejected() {
        let mut b = five_min();
        for (price, qty) in [
            (f64::NAN, 1.0),
            (f64::INFINITY, 1.0),
            (100.0, -1.0),
            (100.0, f64::NAN),
        ] {
            assert_eq!(
                b.on_tick(ist(0, 9, 16, 0, 0), price, qty),
                (TickStatus::Invalid, None)
            );
        }
        assert_eq!(b.forming(), None);
    }

    #[test]
    fn last_bar_truncated_at_close() {
        let mut b = BarBuilder::new(BarSpec::nse(3600)).unwrap();
        b.on_tick(ist(0, 15, 20, 0, 0), 100.0, 1.0);
        let bar = b.forming().unwrap();
        assert_eq!(bar.start_ms, ist(0, 15, 15, 0, 0));
        assert_eq!(bar.end_ms, ist(0, 15, 30, 0, 0));
    }

    #[test]
    fn no_bars_for_gaps() {
        let mut b = five_min();
        b.on_tick(ist(0, 9, 16, 0, 0), 100.0, 1.0);
        let (_, closed) = b.on_tick(ist(0, 9, 41, 0, 0), 102.0, 1.0);
        assert_eq!(closed.unwrap().start_ms, ist(0, 9, 15, 0, 0));
        assert_eq!(b.forming().unwrap().start_ms, ist(0, 9, 40, 0, 0));
    }

    #[test]
    fn next_session_tick_closes_previous_day_bar() {
        let mut b = five_min();
        b.on_tick(ist(0, 15, 29, 0, 0), 100.0, 1.0);
        let (_, closed) = b.on_tick(ist(1, 9, 15, 1, 0), 101.0, 1.0);
        assert_eq!(closed.unwrap().end_ms, ist(0, 15, 30, 0, 0));
        assert_eq!(b.forming().unwrap().start_ms, ist(1, 9, 15, 0, 0));
    }

    #[test]
    fn invalid_spec_rejected() {
        let spec = |interval, open, close, offset| BarSpec {
            interval_secs: interval,
            session_open_secs: open,
            session_close_secs: close,
            utc_offset_secs: offset,
        };
        assert!(BarBuilder::new(spec(0, 33_300, 55_800, 19_800)).is_err());
        assert!(BarBuilder::new(spec(i64::MAX, 33_300, 55_800, 19_800)).is_err());
        assert!(BarBuilder::new(spec(86_401, 33_300, 55_800, 19_800)).is_err());
        assert!(BarBuilder::new(spec(300, 55_800, 33_300, 19_800)).is_err());
        assert!(BarBuilder::new(spec(300, -1, 55_800, 19_800)).is_err());
        assert!(BarBuilder::new(spec(300, 33_300, 86_401, 19_800)).is_err());
        assert!(BarBuilder::new(spec(300, 33_300, 55_800, 15 * 3600)).is_err());
    }

    #[test]
    fn extreme_timestamps_rejected_not_wrapped() {
        let full_day = BarSpec {
            interval_secs: 3600,
            session_open_secs: 0,
            session_close_secs: 86_400,
            utc_offset_secs: 19_800,
        };
        let mut b = BarBuilder::new(full_day).unwrap();
        for ts in [i64::MAX, i64::MAX - 1_000, i64::MIN, i64::MIN + 1_000] {
            assert_eq!(
                b.on_tick(ts, 100.0, 1.0),
                (TickStatus::Invalid, None),
                "ts {ts}"
            );
        }
        assert_eq!(b.forming(), None);
    }
}
