//! Hybrid extended indicators: DMI, WILLIAMS_FRACTALS, RWI.

/// Directional Movement Index as `(PLUS_DI, MINUS_DI, ADX)`.
///
/// One fused ADX-family pass so the three series share the same Wilder seed
/// as [`crate::momentum::plus_di`], [`crate::momentum::minus_di`], and
/// [`crate::momentum::adx`].
pub fn dmi(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    timeperiod: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (_, _, plus_di, minus_di, _, adx) = crate::momentum::adx_all(high, low, close, timeperiod);
    (plus_di, minus_di, adx)
}

/// Williams Fractals: local swing high / swing low with `timeperiod` bars
/// on each side of the pivot.
///
/// Returns `(up, down)`. An up-fractal writes `high[i]` when `high[i]` is
/// strictly greater than the `timeperiod` highs on both sides; a down-fractal
/// writes `low[i]` when `low[i]` is strictly lower than the neighbouring
/// lows. Unconfirmed edges (first and last `timeperiod` bars) and non-pivot
/// bars are `NaN`.
pub fn williams_fractals(high: &[f64], low: &[f64], timeperiod: usize) -> (Vec<f64>, Vec<f64>) {
    let n = high.len();
    let mut up = vec![f64::NAN; n];
    let mut down = vec![f64::NAN; n];
    if timeperiod < 1 || n < 2 * timeperiod + 1 {
        return (up, down);
    }
    for i in timeperiod..(n - timeperiod) {
        let mut is_up = true;
        let mut is_down = true;
        for k in 1..=timeperiod {
            if high[i] <= high[i - k] || high[i] <= high[i + k] {
                is_up = false;
            }
            if low[i] >= low[i - k] || low[i] >= low[i + k] {
                is_down = false;
            }
            if !is_up && !is_down {
                break;
            }
        }
        if is_up {
            up[i] = high[i];
        }
        if is_down {
            down[i] = low[i];
        }
    }
    (up, down)
}

/// Random Walk Index (Poulos): max over lookbacks `2..=timeperiod` of
///
/// * `RWI High = (high[i] - low[i-n]) / (SMA(TR, n) * sqrt(n))`
/// * `RWI Low  = (high[i-n] - low[i]) / (SMA(TR, n) * sqrt(n))`
///
/// The first `timeperiod` values are `NaN`. Values above 1 indicate a move
/// stronger than a random walk of that length.
pub fn rwi(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> (Vec<f64>, Vec<f64>) {
    let n = high.len();
    let mut rwi_high = vec![f64::NAN; n];
    let mut rwi_low = vec![f64::NAN; n];
    if timeperiod < 2 || n <= timeperiod {
        return (rwi_high, rwi_low);
    }

    let mut tr = vec![0.0_f64; n];
    tr[0] = high[0] - low[0];
    for i in 1..n {
        tr[i] = crate::volatility::true_range(high[i], low[i], close[i - 1]);
    }
    let mut ps = vec![0.0_f64; n + 1];
    for i in 0..n {
        ps[i + 1] = ps[i] + tr[i];
    }

    for i in timeperiod..n {
        let mut max_h = f64::NEG_INFINITY;
        let mut max_l = f64::NEG_INFINITY;
        for lookback in 2..=timeperiod {
            let atr = (ps[i + 1] - ps[i + 1 - lookback]) / lookback as f64;
            if !atr.is_finite() || atr <= 0.0 {
                continue;
            }
            let denom = atr * (lookback as f64).sqrt();
            if denom == 0.0 {
                continue;
            }
            let rh = (high[i] - low[i - lookback]) / denom;
            let rl = (high[i - lookback] - low[i]) / denom;
            if rh > max_h {
                max_h = rh;
            }
            if rl > max_l {
                max_l = rl;
            }
        }
        if max_h.is_finite() {
            rwi_high[i] = max_h;
        }
        if max_l.is_finite() {
            rwi_low[i] = max_l;
        }
    }
    (rwi_high, rwi_low)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_hlc() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let h: Vec<f64> = (1..=40).map(|i| i as f64 + 1.0).collect();
        let l: Vec<f64> = (1..=40).map(|i| i as f64).collect();
        let c: Vec<f64> = (1..=40).map(|i| i as f64 + 0.5).collect();
        (h, l, c)
    }

    #[test]
    fn dmi_matches_plus_di_minus_di_adx() {
        let (h, l, c) = sample_hlc();
        let (pdi, mdi, adx) = dmi(&h, &l, &c, 5);
        let exp_p = crate::momentum::plus_di(&h, &l, &c, 5);
        let exp_m = crate::momentum::minus_di(&h, &l, &c, 5);
        let exp_a = crate::momentum::adx(&h, &l, &c, 5);
        assert_eq!(pdi.len(), h.len());
        for i in 0..h.len() {
            assert!((pdi[i].is_nan() && exp_p[i].is_nan()) || (pdi[i] - exp_p[i]).abs() < 1e-14);
            assert!((mdi[i].is_nan() && exp_m[i].is_nan()) || (mdi[i] - exp_m[i]).abs() < 1e-14);
            assert!((adx[i].is_nan() && exp_a[i].is_nan()) || (adx[i] - exp_a[i]).abs() < 1e-14);
        }
    }

    #[test]
    fn dmi_empty() {
        let (p, m, a) = dmi(&[], &[], &[], 14);
        assert!(p.is_empty() && m.is_empty() && a.is_empty());
    }

    #[test]
    fn williams_fractals_peak_and_trough() {
        let high = [1.0, 2.0, 5.0, 2.0, 1.0, 3.0, 4.0];
        let low = [0.0, 1.0, 3.0, 1.0, 0.0, 2.0, 3.0];
        let (up, down) = williams_fractals(&high, &low, 2);
        assert_eq!(up.len(), 7);
        assert!(up[2] - 5.0 < 1e-12 && (up[2] - 5.0).abs() < 1e-12);
        assert!(down[4] - 0.0 < 1e-12 && down[4].abs() < 1e-12);
        // Edges unconfirmed.
        assert!(up[0].is_nan() && up[1].is_nan() && up[5].is_nan() && up[6].is_nan());
        assert!(down[0].is_nan() && down[1].is_nan() && down[5].is_nan() && down[6].is_nan());
        assert!(up[3].is_nan() && up[4].is_nan());
        assert!(down[2].is_nan() && down[3].is_nan());
    }

    #[test]
    fn williams_fractals_insufficient() {
        let high = [1.0, 2.0, 3.0];
        let low = [0.0, 1.0, 2.0];
        let (up, down) = williams_fractals(&high, &low, 2);
        assert!(up.iter().all(|v| v.is_nan()));
        assert!(down.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn rwi_golden_period2() {
        let high = [10.0, 11.0, 12.0, 13.0, 14.0];
        let low = [9.0, 10.0, 11.0, 12.0, 13.0];
        let close = [9.5, 10.5, 11.5, 12.5, 13.5];
        let (rh, rl) = rwi(&high, &low, &close, 2);
        assert!(rh[0].is_nan() && rh[1].is_nan());
        assert!(rl[0].is_nan() && rl[1].is_nan());
        // i=2, n=2: TR[1]=1.5, TR[2]=1.5, ATR=1.5, denom=1.5*sqrt(2)
        // RH=(high[2]-low[0])/denom = 3/denom
        // RL=(high[0]-low[2])/denom = -1/denom
        let denom = 1.5 * 2.0_f64.sqrt();
        assert!((rh[2] - 3.0 / denom).abs() < 1e-12);
        assert!((rl[2] + 1.0 / denom).abs() < 1e-12);
        assert!(rh[2].is_finite() && rh[3].is_finite() && rh[4].is_finite());
    }

    #[test]
    fn rwi_warmup_and_empty() {
        let (h, l, c) = sample_hlc();
        let (rh, rl) = rwi(&h, &l, &c, 8);
        for v in rh.iter().take(8) {
            assert!(v.is_nan());
        }
        for v in rl.iter().take(8) {
            assert!(v.is_nan());
        }
        assert!(rh[8].is_finite());
        let (eh, el) = rwi(&[], &[], &[], 14);
        assert!(eh.is_empty() && el.is_empty());
    }
}
