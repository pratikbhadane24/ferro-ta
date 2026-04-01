//! Cycle indicators — Hilbert Transform-based cycle analysis (Ehlers).
//!
//! Based on John Ehlers' Discrete Hilbert Transform as implemented in TA-Lib.
//! Reference: "Cybernetic Analysis for Stocks and Futures" by J.F. Ehlers
//!
//! All HT functions share a 63-bar lookback period.

use std::f64::consts::PI;

/// Number of leading bars that are set to NaN / zero.
pub const HT_LOOKBACK: usize = 63;

/// Shared output from the core Hilbert Transform computation.
pub struct HtCore {
    pub trendline: Vec<f64>,
    pub dc_period: Vec<f64>,
    pub dc_phase: Vec<f64>,
    pub inphase: Vec<f64>,
    pub quadrature: Vec<f64>,
    pub trend_mode: Vec<i32>,
}

/// Run the full Hilbert Transform pipeline on a slice of close prices.
pub fn compute_ht_core(prices: &[f64]) -> HtCore {
    let n = prices.len();

    let mut trendline = vec![f64::NAN; n];
    let mut dc_period = vec![f64::NAN; n];
    let mut dc_phase = vec![f64::NAN; n];
    let mut inphase = vec![f64::NAN; n];
    let mut quadrature = vec![f64::NAN; n];
    let mut trend_mode = vec![0i32; n];

    if n <= HT_LOOKBACK {
        return HtCore {
            trendline,
            dc_period,
            dc_phase,
            inphase,
            quadrature,
            trend_mode,
        };
    }

    // Step 1: Smooth the price series (4-bar weighted average)
    let mut smooth = vec![0.0f64; n];
    for i in 0..n {
        smooth[i] = if i >= 3 {
            (4.0 * prices[i] + 3.0 * prices[i - 1] + 2.0 * prices[i - 2] + prices[i - 3]) / 10.0
        } else {
            prices[i]
        };
    }

    // Step 2: Full Hilbert Transform pipeline
    let mut detrender = vec![0.0f64; n];
    let mut q1 = vec![0.0f64; n];
    let mut i1 = vec![0.0f64; n];
    let mut ji = vec![0.0f64; n];
    let mut jq = vec![0.0f64; n];
    let mut i2 = vec![0.0f64; n];
    let mut q2 = vec![0.0f64; n];
    let mut re = vec![0.0f64; n];
    let mut im = vec![0.0f64; n];
    let mut period = vec![0.0f64; n];
    let mut smooth_period = vec![0.0f64; n];
    let mut phase = vec![0.0f64; n];

    for i in 6..n {
        let prev_period = period[i - 1];
        // Alpha coefficient for HT filters depends on the current period estimate
        let alpha = 0.075 * prev_period + 0.54;

        // Discrete Hilbert Transform of smooth price (detrender)
        detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i - 2]
            - 0.5769 * smooth[i - 4]
            - 0.0962 * smooth[i - 6])
            * alpha;

        // Q1: HT of detrender
        if i >= 12 {
            q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i - 2]
                - 0.5769 * detrender[i - 4]
                - 0.0962 * detrender[i - 6])
                * alpha;
        }

        // I1: delayed detrender
        if i >= 9 {
            i1[i] = detrender[i - 3];
        }

        // jI: HT of I1
        if i >= 15 {
            ji[i] = (0.0962 * i1[i] + 0.5769 * i1[i - 2] - 0.5769 * i1[i - 4] - 0.0962 * i1[i - 6])
                * alpha;
        }

        // jQ: HT of Q1
        if i >= 18 {
            jq[i] = (0.0962 * q1[i] + 0.5769 * q1[i - 2] - 0.5769 * q1[i - 4] - 0.0962 * q1[i - 6])
                * alpha;
        }

        // Phase components
        let i2_raw = i1[i] - jq[i];
        let q2_raw = q1[i] + ji[i];

        // EMA smoothing of I2 and Q2
        let i2_prev = i2[i - 1];
        let q2_prev = q2[i - 1];
        i2[i] = 0.2 * i2_raw + 0.8 * i2_prev;
        q2[i] = 0.2 * q2_raw + 0.8 * q2_prev;

        // Cross-product for period estimation
        let re_raw = i2[i] * i2_prev + q2[i] * q2_prev;
        let im_raw = i2[i] * q2_prev - q2[i] * i2_prev;

        // EMA smoothing of Re and Im
        re[i] = 0.2 * re_raw + 0.8 * re[i - 1];
        im[i] = 0.2 * im_raw + 0.8 * im[i - 1];

        // Compute period from cross-product of consecutive phasors.
        let mut p = if re[i] != 0.0 && im[i] != 0.0 && re[i] > 0.0 {
            2.0 * PI / (im[i] / re[i]).atan()
        } else {
            prev_period
        };

        // Clamp period relative to previous
        if prev_period > 0.0 {
            if p > 1.5 * prev_period {
                p = 1.5 * prev_period;
            }
            if p < 0.67 * prev_period {
                p = 0.67 * prev_period;
            }
        }
        // Hard clamp to [6, 50] bars
        p = p.clamp(6.0, 50.0);

        // EMA smooth the period
        period[i] = 0.2 * p + 0.8 * prev_period;

        // Smooth the smoothed period once more
        smooth_period[i] = 0.33 * period[i] + 0.67 * smooth_period[i - 1];

        // Phase from I1 and Q1
        phase[i] = if i1[i] != 0.0 {
            q1[i].atan2(i1[i]) * 180.0 / PI
        } else if q1[i] > 0.0 {
            90.0
        } else if q1[i] < 0.0 {
            -90.0
        } else {
            0.0
        };

        // Write outputs once past lookback
        if i >= HT_LOOKBACK {
            dc_period[i] = smooth_period[i];
            dc_phase[i] = phase[i];
            inphase[i] = i1[i];
            quadrature[i] = q1[i];

            // Trend mode: cycle when SmoothPeriod >= 20, trend when < 20
            trend_mode[i] = if smooth_period[i] < 20.0 { 1 } else { 0 };
        }
    }

    // Trendline: average over the current dominant cycle period
    for i in HT_LOOKBACK..n {
        let sp = smooth_period[i];
        let dc = (sp.round() as usize).max(1).min(i + 1);
        let sum: f64 = (0..dc).map(|j| smooth[i - j]).sum();
        trendline[i] = sum / dc as f64;
    }

    HtCore {
        trendline,
        dc_period,
        dc_phase,
        inphase,
        quadrature,
        trend_mode,
    }
}

// ---------------------------------------------------------------------------
// Public indicator functions
// ---------------------------------------------------------------------------

/// Hilbert Transform Instantaneous Trendline (Ehlers).
/// Smooths price over the dominant cycle period.
pub fn ht_trendline(close: &[f64]) -> Vec<f64> {
    compute_ht_core(close).trendline
}

/// Hilbert Transform Dominant Cycle Period in bars.
pub fn ht_dcperiod(close: &[f64]) -> Vec<f64> {
    compute_ht_core(close).dc_period
}

/// Hilbert Transform Dominant Cycle Phase in degrees.
pub fn ht_dcphase(close: &[f64]) -> Vec<f64> {
    compute_ht_core(close).dc_phase
}

/// Hilbert Transform Phasor components. Returns `(inphase, quadrature)`.
pub fn ht_phasor(close: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let core = compute_ht_core(close);
    (core.inphase, core.quadrature)
}

/// Hilbert Transform SineWave. Returns `(sine, leadsine)` where leadsine
/// leads sine by 45 degrees.
pub fn ht_sine(close: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    let core = compute_ht_core(close);

    let mut sine = vec![f64::NAN; n];
    let mut lead_sine = vec![f64::NAN; n];

    for i in HT_LOOKBACK..n {
        if !core.dc_phase[i].is_nan() {
            let phase_rad = core.dc_phase[i] * PI / 180.0;
            sine[i] = phase_rad.sin();
            lead_sine[i] = (phase_rad + PI / 4.0).sin(); // 45-degree lead
        }
    }

    (sine, lead_sine)
}

/// Hilbert Transform Trend vs Cycle Mode: 1 = trending, 0 = cycling.
pub fn ht_trendmode(close: &[f64]) -> Vec<i32> {
    compute_ht_core(close).trend_mode
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a simple sine wave for testing cycle detection.
    fn sine_wave(n: usize, period: f64) -> Vec<f64> {
        (0..n)
            .map(|i| 100.0 + 10.0 * (2.0 * PI * i as f64 / period).sin())
            .collect()
    }

    /// Flat price series for baseline testing.
    fn flat_prices(n: usize) -> Vec<f64> {
        vec![100.0; n]
    }

    #[test]
    fn test_ht_trendline_length_and_lookback() {
        let close = sine_wave(200, 20.0);
        let result = ht_trendline(&close);
        assert_eq!(result.len(), close.len());
        // First HT_LOOKBACK values must be NaN
        for v in &result[..HT_LOOKBACK] {
            assert!(v.is_nan(), "expected NaN in lookback region");
        }
        // Values after lookback must be finite
        for v in &result[HT_LOOKBACK..] {
            assert!(v.is_finite(), "expected finite value after lookback");
        }
    }

    #[test]
    fn test_ht_dcperiod_length_and_lookback() {
        let close = sine_wave(200, 20.0);
        let result = ht_dcperiod(&close);
        assert_eq!(result.len(), close.len());
        for v in &result[..HT_LOOKBACK] {
            assert!(v.is_nan());
        }
        // After lookback, period should be positive and finite
        for v in &result[HT_LOOKBACK..] {
            assert!(v.is_finite());
            assert!(*v >= 6.0 && *v <= 50.0, "period {} out of [6,50]", v);
        }
    }

    #[test]
    fn test_ht_dcphase_length_and_lookback() {
        let close = sine_wave(200, 20.0);
        let result = ht_dcphase(&close);
        assert_eq!(result.len(), close.len());
        for v in &result[..HT_LOOKBACK] {
            assert!(v.is_nan());
        }
        for v in &result[HT_LOOKBACK..] {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_ht_phasor_dual_output() {
        let close = sine_wave(200, 20.0);
        let (inp, quad) = ht_phasor(&close);
        assert_eq!(inp.len(), close.len());
        assert_eq!(quad.len(), close.len());
        for v in &inp[..HT_LOOKBACK] {
            assert!(v.is_nan());
        }
        for v in &quad[..HT_LOOKBACK] {
            assert!(v.is_nan());
        }
    }

    #[test]
    fn test_ht_sine_dual_output() {
        let close = sine_wave(200, 20.0);
        let (s, ls) = ht_sine(&close);
        assert_eq!(s.len(), close.len());
        assert_eq!(ls.len(), close.len());
        for v in &s[..HT_LOOKBACK] {
            assert!(v.is_nan());
        }
        // Sine values should be in [-1, 1]
        for v in &s[HT_LOOKBACK..] {
            assert!(v.is_finite());
            assert!(*v >= -1.0 && *v <= 1.0, "sine {} out of [-1,1]", v);
        }
        for v in &ls[HT_LOOKBACK..] {
            assert!(v.is_finite());
            assert!(*v >= -1.0 && *v <= 1.0, "leadsine {} out of [-1,1]", v);
        }
    }

    #[test]
    fn test_ht_trendmode_values() {
        let close = sine_wave(200, 20.0);
        let result = ht_trendmode(&close);
        assert_eq!(result.len(), close.len());
        // All values must be 0 or 1
        for v in &result {
            assert!(*v == 0 || *v == 1, "trend_mode {} not 0 or 1", v);
        }
    }

    #[test]
    fn test_short_input_all_nan() {
        let close = vec![100.0; HT_LOOKBACK]; // exactly HT_LOOKBACK, not enough
        let tl = ht_trendline(&close);
        assert!(tl.iter().all(|v| v.is_nan()));
        let dp = ht_dcperiod(&close);
        assert!(dp.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_flat_prices_trendline_equals_price() {
        let close = flat_prices(200);
        let tl = ht_trendline(&close);
        // For a flat price, trendline after lookback should be very close to the price
        for v in &tl[HT_LOOKBACK..] {
            assert!(
                (v - 100.0).abs() < 1e-6,
                "trendline {} diverged from flat price",
                v
            );
        }
    }
}
