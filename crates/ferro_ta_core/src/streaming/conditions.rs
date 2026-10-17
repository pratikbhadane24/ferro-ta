//! Streaming condition helpers: dead-band cross, N-bars-in-a-row, cooldown.
//!
//! Each is a [`StreamingIndicator`](super::StreamingIndicator) emitting `i8`
//! events, so it composes with [`Live`](super::Live) previews and the
//! non-finite input guard like any indicator.

use super::StreamingError;

/// Which side of the dead band a series pair was last seen on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum Side {
    #[default]
    Unknown,
    Below,
    Above,
}

/// Cross of `a` over `b` with a dead band (hysteresis).
///
/// With `d = a - b`, the pair is *above* when `d > dead_band` and *below*
/// when `d < -dead_band`; values inside the band leave the side unchanged.
/// Emits `1` when the side flips below → above, `-1` above → below, else
/// `0`. The first bar outside the band only arms the side. Noise that stays
/// inside the band can therefore fire at most once per real crossing.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingCross {
    dead_band: f64,
    side: Side,
}

impl StreamingCross {
    pub fn new(dead_band: f64) -> Result<Self, StreamingError> {
        if !dead_band.is_finite() || dead_band < 0.0 {
            return Err(StreamingError(format!(
                "dead_band must be finite and >= 0, got {dead_band}"
            )));
        }
        Ok(Self {
            dead_band,
            side: Side::Unknown,
        })
    }

    /// Feed one bar of the two series; return the cross event.
    pub fn update(&mut self, a: f64, b: f64) -> i8 {
        let d = a - b;
        let now = if d > self.dead_band {
            Side::Above
        } else if d < -self.dead_band {
            Side::Below
        } else {
            return 0;
        };
        let before = std::mem::replace(&mut self.side, now);
        match (before, now) {
            (Side::Below, Side::Above) => 1,
            (Side::Above, Side::Below) => -1,
            _ => 0,
        }
    }

    pub fn reset(&mut self) {
        self.side = Side::Unknown;
    }

    pub fn dead_band(&self) -> f64 {
        self.dead_band
    }

    /// Re-check invariants (used when restoring a snapshot).
    pub fn validate(&self) -> Result<(), StreamingError> {
        Self::new(self.dead_band).map(|_| ())
    }
}

/// "N bars in a row": emits `1` while the current run of true (non-zero)
/// inputs is at least `n` bars long, else `0`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingStreak {
    n: usize,
    run: usize,
}

impl StreamingStreak {
    pub fn new(n: usize) -> Result<Self, StreamingError> {
        if n == 0 {
            return Err(StreamingError("n must be >= 1, got 0".to_string()));
        }
        Ok(Self { n, run: 0 })
    }

    /// Feed one bar's condition (non-zero = true).
    pub fn update(&mut self, condition: f64) -> i8 {
        self.run = if condition != 0.0 { self.run + 1 } else { 0 };
        i8::from(self.run >= self.n)
    }

    pub fn reset(&mut self) {
        self.run = 0;
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Re-check invariants (used when restoring a snapshot).
    pub fn validate(&self) -> Result<(), StreamingError> {
        Self::new(self.n).map(|_| ())
    }
}

/// Passes the sign of the first non-zero signal, then mutes the next
/// `bars` bars, so one event cannot re-fire on every following bar.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamingCooldown {
    bars: usize,
    remaining: usize,
}

impl StreamingCooldown {
    /// Infallible today; `Result` keeps the constructor uniform with the
    /// other streaming types for the binding generators.
    pub fn new(bars: usize) -> Result<Self, StreamingError> {
        Ok(Self { bars, remaining: 0 })
    }

    /// Feed one bar's signal; return its sign unless cooling down.
    pub fn update(&mut self, signal: f64) -> i8 {
        if self.remaining > 0 {
            self.remaining -= 1;
            return 0;
        }
        if signal == 0.0 {
            return 0;
        }
        self.remaining = self.bars;
        if signal > 0.0 {
            1
        } else {
            -1
        }
    }

    pub fn reset(&mut self) {
        self.remaining = 0;
    }

    pub fn bars(&self) -> usize {
        self.bars
    }

    /// Re-check invariants (used when restoring a snapshot).
    pub fn validate(&self) -> Result<(), StreamingError> {
        if self.remaining > self.bars {
            return Err(StreamingError(format!(
                "cooldown remaining {} exceeds bars {}",
                self.remaining, self.bars
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_cross(band: f64, diffs: &[f64]) -> Vec<i8> {
        let mut cross = StreamingCross::new(band).unwrap();
        diffs.iter().map(|&d| cross.update(d, 0.0)).collect()
    }

    #[test]
    fn cross_fires_on_clean_cross() {
        let mut cross = StreamingCross::new(0.0).unwrap();
        let got: Vec<i8> = [(1.0, 2.0), (1.0, 2.0), (3.0, 2.0)]
            .iter()
            .map(|&(a, b)| cross.update(a, b))
            .collect();
        assert_eq!(got, vec![0, 0, 1]);
    }

    #[test]
    fn oscillation_inside_band_fires_once() {
        let got = run_cross(0.5, &[-1.0, 0.2, -0.2, 0.3, 0.6, 0.1, 0.7]);
        assert_eq!(got, vec![0, 0, 0, 0, 1, 0, 0]);
    }

    #[test]
    fn cross_down_mirrors_up() {
        let got = run_cross(0.5, &[1.0, -0.2, 0.2, -0.3, -0.6, -0.1, -0.7]);
        assert_eq!(got, vec![0, 0, 0, 0, -1, 0, 0]);
    }

    #[test]
    fn cross_touching_zero_band_matches_batch_check_cross() {
        let diffs = [-1.0, 0.0, 1.0, 0.0, -1.0];
        let batch = crate::alerts::check_cross(&diffs, &[0.0; 5]);
        assert_eq!(run_cross(0.0, &diffs), batch);
    }

    #[test]
    fn streak_fires_from_nth_true() {
        let mut streak = StreamingStreak::new(3).unwrap();
        let got: Vec<i8> = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
            .iter()
            .map(|&v| streak.update(v))
            .collect();
        assert_eq!(got, vec![0, 0, 0, 0, 0, 1, 1]);
    }

    #[test]
    fn cooldown_blocks_next_bars() {
        let mut cooldown = StreamingCooldown::new(2).unwrap();
        let got: Vec<i8> = [1.0, 1.0, 1.0, 0.0, -1.0]
            .iter()
            .map(|&v| cooldown.update(v))
            .collect();
        assert_eq!(got, vec![1, 0, 0, 0, -1]);
    }

    #[test]
    fn reset_clears_condition_state() {
        let mut cross = StreamingCross::new(0.0).unwrap();
        cross.update(-1.0, 0.0);
        cross.reset();
        assert_eq!(
            cross.update(1.0, 0.0),
            0,
            "first side after reset only arms"
        );

        let mut cooldown = StreamingCooldown::new(5).unwrap();
        cooldown.update(1.0);
        cooldown.reset();
        assert_eq!(cooldown.update(-1.0), -1);
    }

    #[test]
    fn invalid_params_rejected() {
        assert!(StreamingCross::new(-1.0).is_err());
        assert!(StreamingCross::new(f64::NAN).is_err());
        assert!(StreamingCross::new(f64::INFINITY).is_err());
        assert!(StreamingStreak::new(0).is_err());
    }
}
