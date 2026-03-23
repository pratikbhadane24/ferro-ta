// Shared Hilbert Transform Core
// Based on John Ehlers' Discrete Hilbert Transform as implemented in TA-Lib.
// Reference: "Cybernetic Analysis for Stocks and Futures" by J.F. Ehlers
//
// All HT functions share a 63-bar lookback period.

use std::f64::consts::PI;

pub(super) const HT_LOOKBACK: usize = 63;

/// Shared output from the core Hilbert Transform computation.
pub(super) struct HtCore {
    pub(super) trendline: Vec<f64>,
    pub(super) dc_period: Vec<f64>,
    pub(super) dc_phase: Vec<f64>,
    pub(super) inphase: Vec<f64>,
    pub(super) quadrature: Vec<f64>,
    pub(super) trend_mode: Vec<i32>,
}

/// Run the full Hilbert Transform pipeline on a slice of close prices.
pub(super) fn compute_ht_core(prices: &[f64]) -> HtCore {
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
        // Uses atan(Im/Re) per Ehlers' convention; guard against negative Re
        // which would flip the sign of the period estimate.
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
