//! Exponential moving averages and their compositions (EMA, DEMA, TEMA, T3).

/// Compute the Exponential Moving Average (EMA).
///
/// The EMA is seeded with the SMA of the first `timeperiod` **finite** bars and
/// uses a smoothing factor of `k = 2 / (timeperiod + 1)`. Returns a `Vec<f64>`
/// of the same length as `close`.
///
/// # Warm-up
///
/// A leading `NaN` prefix in the input is skipped before seeding, so the
/// warm-up is `start + timeperiod - 1` values of `NaN`, where `start` is the
/// index of the first non-`NaN` input. For an input with no `NaN` prefix
/// (`start == 0`) that reduces to the usual `timeperiod - 1`. If fewer than
/// `timeperiod` bars follow `start`, or the input is entirely `NaN`, the whole
/// output is `NaN`. A `NaN` *after* the seed window poisons the recurrence from
/// that bar onward.
///
/// # Arguments
/// * `close` - Price series.
/// * `timeperiod` - Lookback period (must be >= 1).
pub fn ema(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    if timeperiod < 1 {
        return vec![f64::NAN; n];
    }
    // Skip any leading NaN prefix so EMA-of-EMA compositions (DEMA, TEMA,
    // TRIX, PPO signal, MA types 3/4) seed from the first valid window
    // instead of poisoning the entire output with a NaN seed.
    let start = match close.iter().position(|v| !v.is_nan()) {
        Some(s) => s,
        None => return vec![f64::NAN; n],
    };
    if n - start < timeperiod {
        return vec![f64::NAN; n];
    }
    let k = 2.0 / (timeperiod as f64 + 1.0);
    let seed: f64 = close[start..start + timeperiod].iter().sum::<f64>() / timeperiod as f64;
    // `vec![NaN; n]` + indexed stores, not `Vec::with_capacity` + `push`.
    // `push` costs a capacity check and a vec-header reload on every bar; the
    // `NaN` prologue costs one store pass. The two are within noise for a
    // standalone `ema`, but `ema_from_first_finite` below is inlined into
    // `dema`/`tema`/`t3` 2/3/6 times over, where the indexed form measured
    // 15-30% faster end to end, so both use it. The early returns above
    // guarantee `start + timeperiod - 1 < n`, and the warm-up prefix simply
    // keeps its initialized `NaN`s.
    let mut result = vec![f64::NAN; n];
    result[start + timeperiod - 1] = seed;
    let mut prev = seed;
    for i in start + timeperiod..n {
        prev = prev.mul_add(1.0 - k, close[i] * k);
        result[i] = prev;
    }
    result
}

/// EMA seeded from the first window of `timeperiod` consecutive finite inputs.
///
/// Output is aligned to the original index: the seed is written at the last
/// bar of that window, and the recurrence continues from there. Used by
/// composed-EMA indicators (DEMA, TEMA, TRIX, T3, PPO signal) so leading NaNs
/// from an inner EMA do not poison the outer seed.
///
/// # Warm-up
///
/// The warm-up is `seed_end - 1` values of `NaN`, where `seed_end` is one past
/// the last bar of the first run of `timeperiod` consecutive finite inputs —
/// **not** a fixed `timeperiod - 1`. Any non-finite value (`NaN` or `±inf`)
/// restarts the run, so a gap anywhere before the first complete window pushes
/// the warm-up out by that much. If no such run exists the whole output is
/// `NaN`.
pub(crate) fn ema_from_first_finite(input: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = input.len();
    if timeperiod < 1 || n < timeperiod {
        return vec![f64::NAN; n];
    }

    let mut run = 0usize;
    let mut seed_end = None;
    for (i, &v) in input.iter().enumerate() {
        if v.is_finite() {
            run += 1;
            if run == timeperiod {
                seed_end = Some(i + 1);
                break;
            }
        } else {
            run = 0;
        }
    }
    let Some(seed_end) = seed_end else {
        return vec![f64::NAN; n];
    };
    let start = seed_end - timeperiod;

    let k = 2.0 / (timeperiod as f64 + 1.0);
    let seed: f64 = input[start..seed_end].iter().sum::<f64>() / timeperiod as f64;
    // Indexed stores over a pre-filled `NaN` buffer (see `ema`). The warm-up
    // here is *data-dependent* --- `seed_end` comes from the scan above, not
    // from `timeperiod` --- but writing the seed at `seed_end - 1` and nothing
    // before it reproduces exactly the leading-`NaN` count that
    // `resize(seed_end - 1, NaN)` produced. The two early returns cover the
    // cases with no seed at all (all-`NaN` input, or fewer than `timeperiod`
    // bars), and `seed_end <= n` keeps the index in bounds.
    let mut result = vec![f64::NAN; n];
    result[seed_end - 1] = seed;
    let mut prev = seed;
    for i in seed_end..n {
        prev = prev.mul_add(1.0 - k, input[i] * k);
        result[i] = prev;
    }
    result
}

// ---------------------------------------------------------------------------
// DEMA — Double Exponential Moving Average
// ---------------------------------------------------------------------------

/// Double Exponential Moving Average: `2*EMA - EMA(EMA)`.
///
/// # Warm-up
///
/// `2 * (timeperiod - 1)` is a *lower bound*, not the exact count: the two
/// stages are [`ema_from_first_finite`], whose warm-ups stack and each of which
/// slips forward past any non-finite input (see its "Warm-up" section). With a
/// leading `NaN` prefix of length `start`, the first finite output is at
/// `start + 2 * (timeperiod - 1)`; a gap inside either stage's seed window
/// pushes it later still. Bars where either stage is `NaN` are `NaN`.
pub fn dema(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    if timeperiod == 0 {
        return vec![f64::NAN; n];
    }
    let warmup = 2 * (timeperiod - 1);
    let ema1 = ema_from_first_finite(close, timeperiod);
    let ema2 = ema_from_first_finite(&ema1, timeperiod);
    let mut result = vec![f64::NAN; n];
    for i in warmup..n {
        if !ema1[i].is_nan() && !ema2[i].is_nan() {
            result[i] = 2.0 * ema1[i] - ema2[i];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// TEMA — Triple Exponential Moving Average
// ---------------------------------------------------------------------------

/// Triple Exponential Moving Average: `3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))`.
///
/// # Warm-up
///
/// `3 * (timeperiod - 1)` is a *lower bound*, not the exact count — the three
/// [`ema_from_first_finite`] stages stack and each slips past non-finite input.
/// With a leading `NaN` prefix of length `start` the first finite output is at
/// `start + 3 * (timeperiod - 1)`, or later if a gap falls inside a stage's
/// seed window. Bars where any stage is `NaN` are `NaN`.
pub fn tema(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    if timeperiod == 0 {
        return vec![f64::NAN; n];
    }
    let warmup = 3 * (timeperiod - 1);
    let ema1 = ema_from_first_finite(close, timeperiod);
    let ema2 = ema_from_first_finite(&ema1, timeperiod);
    let ema3 = ema_from_first_finite(&ema2, timeperiod);
    let mut result = vec![f64::NAN; n];
    for i in warmup..n {
        if !ema1[i].is_nan() && !ema2[i].is_nan() && !ema3[i].is_nan() {
            result[i] = 3.0 * ema1[i] - 3.0 * ema2[i] + ema3[i];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// T3 — Tillson T3
// ---------------------------------------------------------------------------

/// Tillson T3: 6 cascaded SMA-seeded EMAs with volume factor.
///
/// Each stage seeds from the first window of finite inputs of the prior
/// stage ([`ema_from_first_finite`]), matching TA-Lib.
///
/// # Warm-up
///
/// `6 * (timeperiod - 1)` is a *lower bound*, not the exact count — the six
/// stages' warm-ups stack and each slips past non-finite input. With a leading
/// `NaN` prefix of length `start` the first finite output is at
/// `start + 6 * (timeperiod - 1)`, or later if a gap falls inside a stage's
/// seed window. Bars where any contributing stage is non-finite are `NaN`.
pub fn t3(close: &[f64], timeperiod: usize, vfactor: f64) -> Vec<f64> {
    let n = close.len();
    if timeperiod == 0 {
        return vec![f64::NAN; n];
    }
    let v = vfactor;
    let c1 = -(v * v * v);
    let c2 = 3.0 * v * v + 3.0 * v * v * v;
    let c3 = -6.0 * v * v - 3.0 * v - 3.0 * v * v * v;
    let c4 = 1.0 + 3.0 * v + v * v * v + 3.0 * v * v;
    let warmup = 6 * (timeperiod - 1);
    let e1 = ema_from_first_finite(close, timeperiod);
    let e2 = ema_from_first_finite(&e1, timeperiod);
    let e3 = ema_from_first_finite(&e2, timeperiod);
    let e4 = ema_from_first_finite(&e3, timeperiod);
    let e5 = ema_from_first_finite(&e4, timeperiod);
    let e6 = ema_from_first_finite(&e5, timeperiod);
    let mut result = vec![f64::NAN; n];
    for i in warmup..n {
        if e3[i].is_finite() && e4[i].is_finite() && e5[i].is_finite() && e6[i].is_finite() {
            result[i] = c1 * e6[i] + c2 * e5[i] + c3 * e4[i] + c4 * e3[i];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overlap::test_support::*;

    fn reference_ema(close: &[f64], timeperiod: usize) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];
        if timeperiod < 1 {
            return result;
        }
        let start = match close.iter().position(|v| !v.is_nan()) {
            Some(s) => s,
            None => return result,
        };
        if n - start < timeperiod {
            return result;
        }
        let k = 2.0 / (timeperiod as f64 + 1.0);
        let seed: f64 = close[start..start + timeperiod].iter().sum::<f64>() / timeperiod as f64;
        result[start + timeperiod - 1] = seed;
        for i in start + timeperiod..n {
            result[i] = result[i - 1].mul_add(1.0 - k, close[i] * k);
        }
        result
    }

    fn reference_ema_from_first_finite(input: &[f64], timeperiod: usize) -> Vec<f64> {
        let n = input.len();
        let mut result = vec![f64::NAN; n];
        if timeperiod < 1 || n < timeperiod {
            return result;
        }
        let mut run = 0usize;
        let mut seed_end = None;
        for (i, &v) in input.iter().enumerate() {
            if v.is_finite() {
                run += 1;
                if run == timeperiod {
                    seed_end = Some(i + 1);
                    break;
                }
            } else {
                run = 0;
            }
        }
        let Some(seed_end) = seed_end else {
            return result;
        };
        let start = seed_end - timeperiod;
        let k = 2.0 / (timeperiod as f64 + 1.0);
        let seed: f64 = input[start..seed_end].iter().sum::<f64>() / timeperiod as f64;
        result[seed_end - 1] = seed;
        for i in seed_end..n {
            result[i] = result[i - 1].mul_add(1.0 - k, input[i] * k);
        }
        result
    }

    // -- Group E: output-construction changes must be bit-identical --------

    #[test]
    fn group_e_output_construction_is_bit_identical() {
        let close = synthetic_series(600);
        // A leading NaN run plus a mid-series NaN exercises every early-return
        // and skip branch in the converted kernels.
        let mut gappy = close.clone();
        gappy[0] = f64::NAN;
        gappy[1] = f64::NAN;
        gappy[300] = f64::NAN;

        for &p in &[1usize, 2, 14, 30] {
            for series in [&close, &gappy] {
                assert_bits_eq(
                    &ema(series, p),
                    &reference_ema(series, p),
                    &format!("ema p={p}"),
                );
                assert_bits_eq(
                    &ema_from_first_finite(series, p),
                    &reference_ema_from_first_finite(series, p),
                    &format!("ema_from_first_finite p={p}"),
                );
                // DEMA / TEMA / T3 are compositions of `ema_from_first_finite`,
                // so equality here also covers their own converted tails.
                let r1 = reference_ema_from_first_finite(series, p);
                let r2 = reference_ema_from_first_finite(&r1, p);
                let r3 = reference_ema_from_first_finite(&r2, p);
                let mut want_dema = vec![f64::NAN; series.len()];
                for i in 2 * (p - 1)..series.len() {
                    if !r1[i].is_nan() && !r2[i].is_nan() {
                        want_dema[i] = 2.0 * r1[i] - r2[i];
                    }
                }
                assert_bits_eq(&dema(series, p), &want_dema, &format!("dema p={p}"));
                let mut want_tema = vec![f64::NAN; series.len()];
                for i in 3 * (p - 1)..series.len() {
                    if !r1[i].is_nan() && !r2[i].is_nan() && !r3[i].is_nan() {
                        want_tema[i] = 3.0 * r1[i] - 3.0 * r2[i] + r3[i];
                    }
                }
                assert_bits_eq(&tema(series, p), &want_tema, &format!("tema p={p}"));
            }
        }
        // Warmup longer than the series: every converted kernel must still
        // return exactly `n` NaNs rather than panicking on the resize.
        let short = &close[..4];
        for k in [dema(short, 5), tema(short, 5), t3(short, 5, 0.7)] {
            assert_eq!(k.len(), 4);
            assert!(k.iter().all(|v| v.is_nan()));
        }
    }

    #[test]
    fn ema_basic() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ema(&prices, 3);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-10); // seed = SMA(3)
    }

    #[test]
    fn dema_golden_period3() {
        // Hand-computed DEMA(3) on 1..=10.
        // k = 2/(3+1) = 0.5
        // EMA1 seed SMA(1,2,3)=2; then 3,4,5,6,7,8,9
        // EMA2 seeds from the first 3 finite EMA1 values: SMA(2,3,4)=3 at index 4
        // DEMA = 2*EMA1 - EMA2 => 5,6,7,8,9,10 after warmup 2*(3-1)=4
        let prices: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let period = 3;
        let result = dema(&prices, period);
        let warmup = 2 * (period - 1);
        for (i, &v) in result.iter().enumerate().take(warmup) {
            assert!(v.is_nan(), "expected NaN warmup at {i}, got {v}");
        }
        let expected = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        for (offset, &exp) in expected.iter().enumerate() {
            let i = warmup + offset;
            assert!(
                result[i].is_finite(),
                "expected finite DEMA at {i}, got {}",
                result[i]
            );
            assert!(
                (result[i] - exp).abs() < 1e-10,
                "DEMA[{i}]: got {} expected {exp}",
                result[i]
            );
        }
    }

    #[test]
    fn tema_golden_period3() {
        // Hand-computed TEMA(3) on 1..=10.
        // EMA1/EMA2 as in dema_golden_period3; EMA3 seeds SMA(3,4,5)=4 at index 6
        // TEMA = 3*EMA1 - 3*EMA2 + EMA3 => 7,8,9,10 after warmup 3*(3-1)=6
        let prices: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let period = 3;
        let result = tema(&prices, period);
        let warmup = 3 * (period - 1);
        for (i, &v) in result.iter().enumerate().take(warmup) {
            assert!(v.is_nan(), "expected NaN warmup at {i}, got {v}");
        }
        let expected = [7.0, 8.0, 9.0, 10.0];
        for (offset, &exp) in expected.iter().enumerate() {
            let i = warmup + offset;
            assert!(
                result[i].is_finite(),
                "expected finite TEMA at {i}, got {}",
                result[i]
            );
            assert!(
                (result[i] - exp).abs() < 1e-10,
                "TEMA[{i}]: got {} expected {exp}",
                result[i]
            );
        }
    }

    #[test]
    fn t3_golden_sma_seeded_period3() {
        // Hand-computed T3(3, v=0.7) on 1..=16.
        // Six cascaded SMA-seeded EMAs with k=0.5 on a linear series:
        //   e1[i]=i (i>=2), e2=i-1 (i>=4), e3=i-2 (i>=6),
        //   e4=i-3 (i>=8), e5=i-4 (i>=10), e6=i-5 (i>=12)
        // T3 = c1*e6 + c2*e5 + c3*e4 + c4*e3 = i + 0.1 after warmup 6*(3-1)=12
        // First-price seeding (the old bug) does not produce these values.
        let prices: Vec<f64> = (1..=16).map(|i| i as f64).collect();
        let period = 3;
        let result = t3(&prices, period, 0.7);
        let warmup = 6 * (period - 1);
        for (i, &v) in result.iter().enumerate().take(warmup) {
            assert!(v.is_nan(), "expected NaN warmup at {i}, got {v}");
        }
        for i in warmup..prices.len() {
            let exp = i as f64 + 0.1;
            assert!(
                result[i].is_finite(),
                "expected finite T3 at {i}, got {}",
                result[i]
            );
            assert!(
                (result[i] - exp).abs() < 1e-10,
                "T3[{i}]: got {} expected {exp}",
                result[i]
            );
        }
    }
}
