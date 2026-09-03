//! MESA Adaptive Moving Average.

// ---------------------------------------------------------------------------
// MAMA — MESA Adaptive Moving Average
// ---------------------------------------------------------------------------

/// MESA Adaptive Moving Average. Returns `(mama, fama)`.
pub fn mama(close: &[f64], fastlimit: f64, slowlimit: f64) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    let lookback = 32;
    let mut mama_arr = vec![f64::NAN; n];
    let mut fama_arr = vec![f64::NAN; n];
    if n <= lookback {
        return (mama_arr, fama_arr);
    }

    let mut smooth = vec![0.0f64; n];
    for i in 0..n {
        smooth[i] = if i >= 3 {
            (4.0 * close[i] + 3.0 * close[i - 1] + 2.0 * close[i - 2] + close[i - 3]) / 10.0
        } else {
            close[i]
        };
    }

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
    let mut phase = vec![0.0f64; n];
    let mut mama_val = close[0];
    let mut fama_val = close[0];

    for i in 6..n {
        let prev_period = period[i - 1].max(1.0);
        let alpha = 0.075 * prev_period + 0.54;
        detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i - 2]
            - 0.5769 * smooth[i - 4]
            - 0.0962 * smooth[i - 6])
            * alpha;
        if i >= 12 {
            q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i - 2]
                - 0.5769 * detrender[i - 4]
                - 0.0962 * detrender[i - 6])
                * alpha;
        }
        if i >= 9 {
            i1[i] = detrender[i - 3];
        }
        if i >= 15 {
            ji[i] = (0.0962 * i1[i] + 0.5769 * i1[i - 2] - 0.5769 * i1[i - 4] - 0.0962 * i1[i - 6])
                * alpha;
        }
        if i >= 18 {
            jq[i] = (0.0962 * q1[i] + 0.5769 * q1[i - 2] - 0.5769 * q1[i - 4] - 0.0962 * q1[i - 6])
                * alpha;
        }
        let i2_raw = i1[i] - jq[i];
        let q2_raw = q1[i] + ji[i];
        i2[i] = 0.2 * i2_raw + 0.8 * i2[i - 1];
        q2[i] = 0.2 * q2_raw + 0.8 * q2[i - 1];
        re[i] = 0.2 * (i2[i] * i2[i - 1] + q2[i] * q2[i - 1]) + 0.8 * re[i - 1];
        im[i] = 0.2 * (i2[i] * q2[i - 1] - q2[i] * i2[i - 1]) + 0.8 * im[i - 1];
        let mut p = if re[i] != 0.0 && im[i] != 0.0 && re[i] > 0.0 {
            std::f64::consts::PI * 2.0 / (im[i] / re[i]).atan()
        } else {
            prev_period
        };
        p = p
            .clamp(0.67 * prev_period, 1.5 * prev_period)
            .clamp(6.0, 50.0);
        period[i] = 0.2 * p + 0.8 * prev_period;
        phase[i] = if i1[i] != 0.0 {
            q1[i].atan2(i1[i]) * 180.0 / std::f64::consts::PI
        } else if q1[i] > 0.0 {
            90.0
        } else if q1[i] < 0.0 {
            -90.0
        } else {
            0.0
        };
        let mut delta_phase = phase[i - 1] - phase[i];
        if delta_phase < 1.0 {
            delta_phase = 1.0;
        }
        let adaptive_alpha = (fastlimit / delta_phase).clamp(slowlimit, fastlimit);
        if i >= lookback {
            mama_val = adaptive_alpha * close[i] + (1.0 - adaptive_alpha) * mama_val;
            fama_val = 0.5 * adaptive_alpha * mama_val + (1.0 - 0.5 * adaptive_alpha) * fama_val;
            mama_arr[i] = mama_val;
            fama_arr[i] = fama_val;
        } else {
            mama_val = close[i];
            fama_val = close[i];
        }
    }
    (mama_arr, fama_arr)
}
