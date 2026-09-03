//! Moving-average-difference oscillators: the Absolute and Percentage Price
//! Oscillators (`apo`, `ppo`) and `trix`.

use crate::overlap::{compute_ma_by_type, ma_lookback, MAX_MATYPE};

// ---------------------------------------------------------------------------
// APO / PPO
// ---------------------------------------------------------------------------

/// Absolute Price Oscillator: `fast MA - slow MA`.
///
/// # Arguments
/// * `close` - Price series.
/// * `fastperiod` / `slowperiod` - MA periods; `fastperiod < slowperiod`.
/// * `matype` - MA type applied to *both* legs, `0`–`8` (`8` is an alias of
///   `7` = T3). See the [`overlap::dispatch`](crate::overlap) module docs for
///   the mapping. **`1` (EMA) is this crate's historical behaviour**, not
///   `0`.
///
/// # NaN warm-up
///
/// `ma_lookback(slowperiod, matype)` leading `NaN`s — `slowperiod - 1` for
/// SMA/EMA/WMA/TRIMA, `2 * (slowperiod - 1)` for DEMA, `3 * (…)` for TEMA,
/// `slowperiod` for KAMA, `6 * (…)` for T3. A leading `NaN` prefix in `close`
/// stacks on top of that (the composed-EMA types slip past non-finite input),
/// so every bar is additionally gated on both legs being non-`NaN` rather than
/// trusted from the lookback alone.
///
/// # Edge cases
///
/// All-`NaN` when `fastperiod == 0`, `slowperiod == 0`,
/// `fastperiod >= slowperiod`, or `matype > MAX_MATYPE` (`8`). An
/// out-of-range `matype` is
/// *not* silently treated as SMA — the core has no error type, so an all-`NaN`
/// output is how it reports an unusable argument.
///
/// # TA-Lib compatibility
///
/// `TA_APO(fastperiod, slowperiod, matype)` takes the same three arguments and
/// the same numbering for `0`–`6` and `8`. Two known divergences, both
/// pre-dating this argument: TA-Lib defaults `matype` to `0` (SMA) while
/// ferro-ta's wrappers have always computed the EMA form (`matype = 1`), and
/// **`matype = 7` is T3 here versus MAMA in TA-Lib** — pass `8` for a T3 that
/// means T3 in both libraries.
pub fn apo(close: &[f64], fastperiod: usize, slowperiod: usize, matype: u8) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if fastperiod == 0 || slowperiod == 0 || fastperiod >= slowperiod {
        return result;
    }
    if matype > MAX_MATYPE {
        return result;
    }
    let fast = compute_ma_by_type(close, fastperiod, matype);
    let slow = compute_ma_by_type(close, slowperiod, matype);
    // Lookbacks are monotone in the period, so the slow leg governs.
    let warmup = ma_lookback(slowperiod, matype);
    for i in warmup..n {
        if !fast[i].is_nan() && !slow[i].is_nan() {
            result[i] = fast[i] - slow[i];
        }
    }
    result
}

/// Percentage Price Oscillator: `(fast MA - slow MA) / slow MA * 100`.
/// Returns `(ppo_line, signal_line, histogram)`.
///
/// # Arguments
/// * `close` - Price series.
/// * `fastperiod` / `slowperiod` - MA periods; `fastperiod < slowperiod`.
/// * `signalperiod` - EMA period for the signal line (a ferro-ta extension;
///   `TA_PPO` has no signal output).
/// * `matype` - MA type applied to the fast and slow legs, `0`–`8` (`8` is an
///   alias of `7` = T3). **`1` (EMA) is this crate's historical behaviour**,
///   not `0`. It deliberately
///   does *not* apply to the signal line, which stays an EMA of the PPO line
///   seeded from its first finite window — TA-Lib has no signal here to be
///   compatible with, and re-typing it would change existing output.
///
/// # NaN warm-up
///
/// The PPO line has `ma_lookback(slowperiod, matype)` leading `NaN`s
/// (`slowperiod - 1` for SMA/EMA/WMA/TRIMA, `2 * (slowperiod - 1)` DEMA,
/// `3 * (…)` TEMA, `slowperiod` KAMA, `6 * (…)` T3); the signal and histogram
/// add `signalperiod - 1` on top. A leading `NaN` prefix in `close` stacks
/// further, so bars are gated on the values actually being finite.
///
/// # Edge cases
///
/// All-`NaN` when any period is `0`, `fastperiod >= slowperiod`, or
/// `matype > MAX_MATYPE` (`8`). An out-of-range `matype` is not silently
/// treated as SMA.
///
/// # TA-Lib compatibility
///
/// `TA_PPO(fastperiod, slowperiod, matype)` — same numbering for `0`–`6` and
/// `8`, same two pre-existing divergences as [`apo`] (TA-Lib's `matype`
/// default is `0`, ferro-ta's wrappers compute the `1` form; **`7` is T3 here,
/// MAMA there** — pass `8` for a portable T3). `TA_PPO` returns a single
/// array, so only the PPO line has a TA-Lib counterpart; the signal and
/// histogram are ferro-ta extensions.
pub fn ppo(
    close: &[f64],
    fastperiod: usize,
    slowperiod: usize,
    signalperiod: usize,
    matype: u8,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = close.len();
    let nan3 = || (vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n]);
    if fastperiod == 0 || slowperiod == 0 || signalperiod == 0 || fastperiod >= slowperiod {
        return nan3();
    }
    if matype > MAX_MATYPE {
        return nan3();
    }
    let fast = compute_ma_by_type(close, fastperiod, matype);
    let slow = compute_ma_by_type(close, slowperiod, matype);
    let warmup = ma_lookback(slowperiod, matype);

    let mut ppo_line = vec![f64::NAN; n];
    for i in warmup..n {
        if !fast[i].is_nan() && !slow[i].is_nan() && slow[i] != 0.0 {
            ppo_line[i] = (fast[i] - slow[i]) / slow[i] * 100.0;
        }
    }

    // Signal line = EMA of PPO line, seeded from the first finite PPO window
    // so leading NaNs do not poison the recurrence.
    let signal = crate::overlap::ema_from_first_finite(&ppo_line, signalperiod);
    let mut signal_line = vec![f64::NAN; n];
    let mut hist = vec![f64::NAN; n];
    let sig_warmup = warmup.saturating_add(signalperiod - 1);
    for i in sig_warmup..n {
        if !ppo_line[i].is_nan() && !signal[i].is_nan() {
            signal_line[i] = signal[i];
            hist[i] = ppo_line[i] - signal[i];
        }
    }
    (ppo_line, signal_line, hist)
}

// ---------------------------------------------------------------------------
// TRIX
// ---------------------------------------------------------------------------

/// TRIX: 1-period rate of change of triple-smoothed EMA.
pub fn trix(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod == 0 {
        return result;
    }
    let warmup = 3 * (timeperiod - 1);

    // Triple EMA: each stage seeds from the first finite window of the prior.
    let ema1 = crate::overlap::ema_from_first_finite(close, timeperiod);
    let ema2 = crate::overlap::ema_from_first_finite(&ema1, timeperiod);
    let ema3 = crate::overlap::ema_from_first_finite(&ema2, timeperiod);

    for i in (warmup + 1)..n {
        let prev = ema3[i - 1];
        if !ema3[i].is_nan() && !prev.is_nan() && prev != 0.0 {
            result[i] = (ema3[i] - prev) / prev * 100.0;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{assert_bits, oracle_series};
    use super::*;

    // -----------------------------------------------------------------------
    // Equivalence oracles: verbatim copies of `apo` and `ppo` as they stood
    // before the `matype` argument existed. Both were hard-wired to
    // `crate::overlap::ema`, i.e. TA-Lib's `matype = 1` — *not* `matype = 0`
    // (SMA), which is TA-Lib's default. So the bit-identity contract for
    // these two functions is at `matype = 1`.
    // -----------------------------------------------------------------------
    fn reference_apo(close: &[f64], fastperiod: usize, slowperiod: usize) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];
        if fastperiod == 0 || slowperiod == 0 || fastperiod >= slowperiod {
            return result;
        }
        let fast = crate::overlap::ema(close, fastperiod);
        let slow = crate::overlap::ema(close, slowperiod);
        let warmup = slowperiod - 1;
        for i in warmup..n {
            if !fast[i].is_nan() && !slow[i].is_nan() {
                result[i] = fast[i] - slow[i];
            }
        }
        result
    }

    fn reference_ppo(
        close: &[f64],
        fastperiod: usize,
        slowperiod: usize,
        signalperiod: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let nan3 = || (vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n]);
        if fastperiod == 0 || slowperiod == 0 || signalperiod == 0 || fastperiod >= slowperiod {
            return nan3();
        }
        let fast = crate::overlap::ema(close, fastperiod);
        let slow = crate::overlap::ema(close, slowperiod);
        let warmup = slowperiod - 1;

        let mut ppo_line = vec![f64::NAN; n];
        for i in warmup..n {
            if !fast[i].is_nan() && !slow[i].is_nan() && slow[i] != 0.0 {
                ppo_line[i] = (fast[i] - slow[i]) / slow[i] * 100.0;
            }
        }
        let signal = crate::overlap::ema_from_first_finite(&ppo_line, signalperiod);
        let mut signal_line = vec![f64::NAN; n];
        let mut hist = vec![f64::NAN; n];
        let sig_warmup = warmup.saturating_add(signalperiod - 1);
        for i in sig_warmup..n {
            if !ppo_line[i].is_nan() && !signal[i].is_nan() {
                signal_line[i] = signal[i];
                hist[i] = ppo_line[i] - signal[i];
            }
        }
        (ppo_line, signal_line, hist)
    }

    /// Period pairs plus a few degenerate ones, and series shapes taken from
    /// the module's shared oracle fixtures (ties, monotone, flat, walk) so a
    /// zero MA leg and an all-equal window are both exercised.
    fn period_pairs() -> Vec<(usize, usize)> {
        vec![
            (0, 0),
            (1, 0),
            (0, 5),
            (5, 5),
            (6, 5),
            (1, 2),
            (2, 3),
            (3, 6),
            (5, 14),
            (12, 26),
            (2, 400),
        ]
    }

    #[test]
    fn apo_matype1_is_bit_identical_to_pre_matype_kernel() {
        for (name, series, other) in oracle_series() {
            for close in [&series, &other] {
                for (fast, slow) in period_pairs() {
                    assert_bits(
                        &apo(close, fast, slow, 1),
                        &reference_apo(close, fast, slow),
                        &format!("{name} apo({fast},{slow})"),
                    );
                }
            }
        }
    }

    #[test]
    fn ppo_matype1_is_bit_identical_to_pre_matype_kernel() {
        for (name, series, other) in oracle_series() {
            for close in [&series, &other] {
                for (fast, slow) in period_pairs() {
                    for signal in [0usize, 1, 2, 9] {
                        let ctx = format!("{name} ppo({fast},{slow},{signal})");
                        let (l, s, h) = ppo(close, fast, slow, signal, 1);
                        let (rl, rs, rh) = reference_ppo(close, fast, slow, signal);
                        assert_bits(&l, &rl, &format!("{ctx} line"));
                        assert_bits(&s, &rs, &format!("{ctx} signal"));
                        assert_bits(&h, &rh, &format!("{ctx} hist"));
                    }
                }
            }
        }
    }

    #[test]
    fn apo_ppo_reject_out_of_range_matype() {
        let close: Vec<f64> = (1..=60).map(|i| i as f64).collect();
        for matype in [MAX_MATYPE + 1, 99, u8::MAX] {
            let a = apo(&close, 3, 6, matype);
            assert_eq!(a.len(), close.len());
            assert!(a.iter().all(|v| v.is_nan()), "apo matype={matype}");
            let (l, s, h) = ppo(&close, 3, 6, 2, matype);
            for (label, v) in [("line", l), ("signal", s), ("hist", h)] {
                assert_eq!(v.len(), close.len());
                assert!(v.iter().all(|x| x.is_nan()), "ppo {label} matype={matype}");
            }
        }
    }

    #[test]
    fn apo_matype_selects_the_ma_and_shifts_the_warmup() {
        let close: Vec<f64> = (0..400).map(|i| 100.0 + (i as f64) * 0.25).collect();
        let (fast, slow) = (5usize, 12usize);
        for matype in 0..=7u8 {
            let got = apo(&close, fast, slow, matype);
            let want_fast = crate::overlap::ma(&close, fast, matype);
            let want_slow = crate::overlap::ma(&close, slow, matype);
            let lookback = crate::overlap::ma_lookback(slow, matype);
            for (i, &v) in got.iter().enumerate().take(lookback) {
                assert!(v.is_nan(), "matype={matype}: expected NaN at {i}, got {v}");
            }
            assert!(
                got[lookback].is_finite(),
                "matype={matype}: expected a value at {lookback}"
            );
            for i in lookback..close.len() {
                assert_eq!(
                    got[i].to_bits(),
                    (want_fast[i] - want_slow[i]).to_bits(),
                    "matype={matype} at {i}"
                );
            }
        }
    }

    #[test]
    fn ppo_matype0_is_the_sma_form() {
        // TA-Lib's default. Distinct from `matype = 1`, which is what the
        // ferro-ta wrappers have always produced.
        let close: Vec<f64> = (0..200).map(|i| 100.0 + ((i % 7) as f64)).collect();
        let (line0, _, _) = ppo(&close, 3, 6, 2, 0);
        let (line1, _, _) = ppo(&close, 3, 6, 2, 1);
        let fast = crate::overlap::sma(&close, 3);
        let slow = crate::overlap::sma(&close, 6);
        for i in 5..close.len() {
            let want = (fast[i] - slow[i]) / slow[i] * 100.0;
            assert!((line0[i] - want).abs() < 1e-12, "at {i}");
        }
        assert!(
            (5..close.len()).any(|i| (line0[i] - line1[i]).abs() > 1e-9),
            "matype 0 and 1 must not coincide"
        );
    }

    #[test]
    fn trix_golden_period3() {
        // Hand-computed TRIX(3) on 1..=10.
        // Triple EMA as in tema_golden_period3: EMA3 = [NaN×6, 4, 5, 6, 7]
        // TRIX[i] = (EMA3[i] - EMA3[i-1]) / EMA3[i-1] * 100
        // First value after 3*(3-1)+1 = 7: 25, 20, 100/6
        let prices: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let period = 3;
        let result = trix(&prices, period);
        let first_valid = 3 * (period - 1) + 1;
        for (i, &v) in result.iter().enumerate().take(first_valid) {
            assert!(v.is_nan(), "expected NaN warmup at {i}, got {v}");
        }
        let expected = [25.0, 20.0, 100.0 / 6.0];
        for (offset, &exp) in expected.iter().enumerate() {
            let i = first_valid + offset;
            assert!(
                result[i].is_finite(),
                "expected finite TRIX at {i}, got {}",
                result[i]
            );
            assert!(
                (result[i] - exp).abs() < 1e-10,
                "TRIX[{i}]: got {} expected {exp}",
                result[i]
            );
        }
    }

    #[test]
    fn ppo_signal_from_first_finite() {
        // PPO(2,3,2) on 1..=10. Fast/slow EMAs are SMA-seeded on raw prices
        // (no leading NaN), so the PPO line is already finite after slow-1.
        // The signal must seed from that first finite PPO window, not from
        // leading NaNs (which would poison the entire signal line).
        let prices: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let (ppo_line, signal, hist) = ppo(&prices, 2, 3, 2, 1);
        let line_start = 3 - 1;
        let sig_start = line_start + 2 - 1;
        for (i, &v) in ppo_line.iter().enumerate().take(line_start) {
            assert!(v.is_nan(), "expected NaN PPO line at {i}, got {v}");
        }
        for (i, &v) in signal.iter().enumerate().take(sig_start) {
            assert!(v.is_nan(), "expected NaN PPO signal at {i}, got {v}");
        }
        // PPO[2] = (2.5 - 2) / 2 * 100 = 25
        // PPO[3] = (3.5 - 3) / 3 * 100 = 50/3
        // signal seed = SMA(25, 50/3) = 125/6 at index 3
        assert!((ppo_line[2] - 25.0).abs() < 1e-10);
        assert!((ppo_line[3] - 50.0 / 3.0).abs() < 1e-10);
        assert!(
            signal[sig_start].is_finite(),
            "expected finite PPO signal at {sig_start}, got {}",
            signal[sig_start]
        );
        assert!((signal[3] - 125.0 / 6.0).abs() < 1e-10);
        assert!((hist[3] - (ppo_line[3] - signal[3])).abs() < 1e-10);
        for i in sig_start..prices.len() {
            assert!(signal[i].is_finite(), "PPO signal NaN at {i}");
            assert!((hist[i] - (ppo_line[i] - signal[i])).abs() < 1e-10);
        }
    }
}
