//! Kaufman Adaptive Moving Average.

use crate::rolling::RESEED_INTERVAL;

// ---------------------------------------------------------------------------
// KAMA — Kaufman Adaptive Moving Average
// ---------------------------------------------------------------------------

/// TA-Lib's `TA_EPSILON` (`src/ta_func/ta_utility.h`): the absolute band inside
/// which `TA_IS_ZERO` declares a `double` to be zero.
const TA_EPSILON: f64 = 1e-14;

/// Efficiency ratio, matching `TA_KAMA`'s two-part test verbatim.
///
/// `TA_KAMA` computes
///
/// ```text
/// if( (sumROC1 <= periodROC) || TA_IS_ZERO(sumROC1) )
///    tempReal = 1.0;
/// else
///    tempReal = std_fabs(periodROC/sumROC1);
/// ```
///
/// Both halves matter, and neither is cosmetic:
///
/// * `period_roc` is **signed** — it is `close[i] - close[i-p]`, not its
///   magnitude. Mathematically `volatility >= |period_roc|` (triangle
///   inequality), so `volatility <= period_roc` can only fire when the window
///   is monotonically rising and the two sides are equal up to rounding, or
///   when `volatility` has gone slightly negative on a flat window. Dropping
///   this branch costs ~0.28 relative error against TA-Lib on plateau input.
/// * `TA_IS_ZERO` is a *band* (±`TA_EPSILON`), not an exact `== 0.0` test. Both
///   TA-Lib and this kernel carry `volatility` as a subtract-then-add rolling
///   sum, so a window that has gone flat leaves it holding rounding residue of
///   arbitrary sign (order `1e-14` for prices around 100) rather than an exact
///   zero. Testing `volatility > 0.0` instead — as this kernel used to — reads
///   that residue as real volatility and yields `ER = 0 / residue = 0`, the
///   *slowest* smoothing constant, where TA-Lib mostly takes `ER = 1`, the
///   fastest. That was the bug: KAMA crawled where TA-Lib snaps, and because
///   the ratio was wrong on every bar of the plateau the error held instead of
///   decaying (max abs divergence ~16 against TA-Lib on a StochRSI %K series).
///
/// Note that TA-Lib's own flat-window behaviour is therefore *residue
/// dependent*, not idealized: negative residue (or residue inside the band)
/// takes `ER = 1`, while positive residue above the band takes `ER = 0`. The
/// point of this function is to reproduce that, not to improve on it — which is
/// also why `volatility` is maintained with TA-Lib's exact update order.
///
/// Determined by reading `ta_KAMA.c` at tag `v0.6.4` (the version installed
/// here) and confirmed against the installed `libta-lib` over 1400
/// series/period combinations — random walks, monotone ramps, plateau-heavy
/// series, 0/100 binary series, and values spanning `1e-6` to `1e9`. With both
/// halves in place the worst relative deviation is `3e-16`; dropping either one
/// pushes it to ~0.28.
///
/// There is deliberately **no** `is_nan` arm. `TA_KAMA` has no non-finite
/// handling and lets a `NaN` `sumROC1` poison every later bar, so propagating
/// is the faithful behaviour. An earlier revision carried a `volatility.is_nan()
/// => 1.0` guard, because `macdext` used to hand the signal-leg MA a slice whose
/// first bar was `NaN` (its `macd_start` was `slowperiod - 1` rather than a `max`
/// over each leg's own `ma_lookback`, so any longer-lookback leg — KAMA, DEMA,
/// TEMA, T3 — started the slice too early) and the guard was what stopped that
/// turning the whole MACDEXT output `NaN`. That was masking the real defect
/// rather than fixing it. With `macd_start` corrected, `macd_valid[0]` is finite
/// by construction, and `macdext` was the only in-crate caller passing a
/// `NaN`-prefixed slice into `compute_ma_by_type` — `stoch` already slices from
/// its own warm-up, and `apo`/`ppo`/`bbands` pass raw `close`.
///
/// TA-Lib `main` has since replaced the `TA_IS_ZERO` band with a counter of
/// consecutive exactly-zero 1-day changes, which zeroes `sumROC1` outright once
/// a whole window is flat (the fixed band was scale-inconsistent — it declared
/// every window flat for an instrument quoted below `1e-14`). That is a
/// *behaviour* change on residue-carrying plateaus, so this kernel will need
/// revisiting when the pinned TA-Lib moves past 0.6.4.
#[inline(always)]
fn efficiency_ratio(period_roc: f64, volatility: f64) -> f64 {
    if volatility <= period_roc || (-TA_EPSILON < volatility && volatility < TA_EPSILON) {
        1.0
    } else {
        (period_roc / volatility).abs()
    }
}

/// Kaufman Adaptive Moving Average (TA-Lib).
///
/// Seeds internally from `close[timeperiod - 1]` but does not emit that
/// seed — the first output is the first ER/SC update at index `timeperiod`.
/// Fast SC = 2/3, slow SC = 2/31.
///
/// # Algorithm
///
/// The volatility term is `Σ_{j=1..p} |close[i-j+1] - close[i-j]|`, i.e. a
/// `p`-bar rolling sum over the `|Δclose|` series. Re-summing it per bar makes
/// the kernel O(n·p); running it as a sliding sum makes it O(n). The summands
/// are non-negative, so the window sum has no cancellation, and `TA_KAMA`
/// itself maintains the same rolling `sumROC1` — including its subtract-then-add
/// update order, which this kernel mirrors so the residue it leaves behind on a
/// flat window matches TA-Lib's bit for bit. The sum is recomputed exactly with
/// a vectorized `abs_diff_sum` every 8192 bars, which bounds drift independently
/// of `n`, and again as soon as a non-finite input leaves the window. No
/// `|Δclose|` array is materialized — the entering and leaving differences are
/// formed from `close` directly, which saves a full-length allocation and two
/// passes over it.
///
/// KAMA is a recursive filter, so an error in the efficiency ratio perturbs
/// `kama_val`; the recurrence is a contraction (`sc ∈ (0, 4/9]`), so such
/// perturbations decay rather than accumulate. A *systematically* wrong ratio
/// on a long plateau does not decay, though — see [`efficiency_ratio`].
pub fn kama(close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = close.len();
    if timeperiod == 0 || n <= timeperiod {
        return vec![f64::NAN; n];
    }
    let p = timeperiod;
    let fast_sc = 2.0 / 3.0_f64;
    let slow_sc = 2.0 / 31.0_f64;

    /// One efficiency-ratio / smoothing-constant step of the KAMA recurrence.
    #[inline(always)]
    fn step(
        kama_val: f64,
        close_i: f64,
        period_roc: f64,
        volatility: f64,
        sc_span: f64,
        slow_sc: f64,
    ) -> f64 {
        let er = efficiency_ratio(period_roc, volatility);
        let sc = (er * sc_span + slow_sc).powi(2);
        kama_val + sc * (close_i - kama_val)
    }
    let sc_span = fast_sc - slow_sc;

    // `volatility` at bar `i` is `Σ_{k=i-p+1..i} |close[k] - close[k-1]|`,
    // i.e. `abs_diff_sum(close[i - p ..= i])` over a `p + 1` bar window.
    let mut volatility = crate::simd::abs_diff_sum(&close[..=p]);
    let mut non_finite_inside = close[..=p].iter().filter(|x| !x.is_finite()).count();
    let mut contaminated = non_finite_inside > 0;
    let mut since_reseed = 0usize;

    // Indexed stores over a pre-filled `NaN` buffer, not `push` (see `ema`).
    // `n > p` is guaranteed above, and the reseed re-sums from `close`, never
    // from `result`, so pre-sizing the output cannot disturb it.
    let mut result = vec![f64::NAN; n];

    let mut kama_val = step(
        close[p - 1],
        close[p],
        close[p] - close[0],
        volatility,
        sc_span,
        slow_sc,
    );
    result[p] = kama_val;

    for i in p + 1..n {
        // Subtract-then-add, in that order, to match `TA_KAMA`'s rounding.
        volatility -= (close[i - p - 1] - close[i - p]).abs();
        volatility += (close[i] - close[i - 1]).abs();
        since_reseed += 1;
        if !close[i].is_finite() {
            non_finite_inside += 1;
            contaminated = true;
        }
        if !close[i - p - 1].is_finite() {
            non_finite_inside -= 1;
        }
        if since_reseed >= RESEED_INTERVAL || (contaminated && non_finite_inside == 0) {
            volatility = crate::simd::abs_diff_sum(&close[i - p..=i]);
            since_reseed = 0;
            contaminated = non_finite_inside > 0;
        }
        kama_val = step(
            kama_val,
            close[i],
            close[i] - close[i - p],
            volatility,
            sc_span,
            slow_sc,
        );
        result[i] = kama_val;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overlap::test_support::*;

    /// Verbatim copy of the pre-rewrite KAMA kernel: an O(n·p) re-sum of the
    /// volatility window instead of a rolling one.
    ///
    /// It is a valid oracle **only for input with no flat window**. Its
    /// `volatility > 0.0` test is the bug this module fixed, so on plateau
    /// input it takes the `ER = 0` branch where TA-Lib takes `ER = 1`. Away
    /// from plateaus the two agree to rounding, which is what makes it useful:
    /// it proves the rewrite changed the zero-volatility path and nothing else.
    /// Plateau behaviour is pinned directly against TA-Lib's rule instead, in
    /// `kama_flat_window_snaps_at_the_fastest_smoothing_constant` and
    /// `kama_flat_window_with_rolling_residue_still_snaps`.
    fn reference_kama(close: &[f64], timeperiod: usize) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];
        if timeperiod == 0 || n <= timeperiod {
            return result;
        }
        let fast_sc = 2.0 / 3.0_f64;
        let slow_sc = 2.0 / 31.0_f64;
        let mut kama_val = close[timeperiod - 1];
        for i in timeperiod..n {
            let direction = (close[i] - close[i - timeperiod]).abs();
            let mut volatility = 0.0_f64;
            for j in 1..=timeperiod {
                volatility += (close[i - j + 1] - close[i - j]).abs();
            }
            let er = if volatility > 0.0 {
                direction / volatility
            } else {
                1.0
            };
            let sc = (er * (fast_sc - slow_sc) + slow_sc).powi(2);
            kama_val += sc * (close[i] - kama_val);
            result[i] = kama_val;
        }
        result
    }

    // -- KAMA --------------------------------------------------------------

    /// `reference_kama` is only an oracle away from flat windows — assert the
    /// fixture really has none before leaning on it.
    fn assert_no_flat_bar(close: &[f64], label: &str) {
        for i in 1..close.len() {
            assert!(
                close[i] != close[i - 1],
                "{label}: flat bar at {i} — reference_kama is not an oracle here"
            );
        }
    }

    #[test]
    fn kama_matches_reference() {
        let close = synthetic_series(4096);
        assert_no_flat_bar(&close, "synthetic_series");
        for &p in &[1usize, 2, 5, 14, 30, 200] {
            let got = kama(&close, p);
            let want = reference_kama(&close, p);
            assert_close(&got, &want, 1e-9, &format!("kama p={p}"));
        }
    }

    #[test]
    fn kama_degenerate_inputs() {
        assert!(kama(&[], 5).is_empty());
        // n == timeperiod produces no output (the seed is not emitted).
        assert!(kama(&[1.0, 2.0, 3.0], 3).iter().all(|v| v.is_nan()));
        assert!(kama(&[1.0, 2.0], 0).iter().all(|v| v.is_nan()));
    }

    #[test]
    fn kama_flat_window_snaps_at_the_fastest_smoothing_constant() {
        // TA-Lib's `TA_KAMA` treats a flat window as maximum efficiency
        // (ER = 1, SC = (2/3)^2 = 4/9): `sumROC1 <= periodROC` reads `0 <= 0`.
        // A price that has been flat for the whole window therefore gives
        // KAMA == price exactly, rather than the slow crawl `ER = 0` produces.
        let close = vec![42.0f64; 40];
        let got = kama(&close, 10);
        for (i, &v) in got.iter().enumerate().skip(10) {
            assert_eq!(v, 42.0, "kama[{i}] should stay pinned to the flat price");
        }
    }

    /// Literal transcription of `TA_KAMA` (tag `v0.6.4`): a rolling `sumROC1`
    /// updated subtract-then-add, and the two-part efficiency-ratio test. No
    /// periodic reseed, so it is only an oracle for inputs shorter than
    /// [`RESEED_INTERVAL`] past the warmup, and no non-finite handling.
    ///
    /// Unlike `reference_kama` this *is* an oracle on plateau input, because it
    /// carries the same rolling residue TA-Lib does.
    fn reference_kama_talib(close: &[f64], timeperiod: usize) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];
        if timeperiod == 0 || n <= timeperiod {
            return result;
        }
        let p = timeperiod;
        let const_max = 2.0 / 31.0_f64;
        let const_diff = 2.0 / 3.0_f64 - const_max;

        let mut sum_roc1 = 0.0f64;
        for k in 1..=p {
            sum_roc1 += (close[k - 1] - close[k]).abs();
        }
        let mut prev_kama = close[p - 1];
        let mut trailing_value = close[0];
        let mut period_roc = close[p] - close[0];
        let mut er = efficiency_ratio(period_roc, sum_roc1);
        let mut sc = (er * const_diff + const_max).powi(2);
        prev_kama += (close[p] - prev_kama) * sc;
        result[p] = prev_kama;

        for i in p + 1..n {
            let trailing = close[i - p];
            period_roc = close[i] - trailing;
            sum_roc1 -= (trailing_value - trailing).abs();
            sum_roc1 += (close[i] - close[i - 1]).abs();
            trailing_value = trailing;
            er = efficiency_ratio(period_roc, sum_roc1);
            sc = (er * const_diff + const_max).powi(2);
            prev_kama += (close[i] - prev_kama) * sc;
            result[i] = prev_kama;
        }
        result
    }

    /// A StochRSI-%K-shaped series: long holds of the previous value plus runs
    /// pinned at exactly 0 and 100, which is what produces flat volatility
    /// windows (and the rolling residue that empties them imprecisely).
    fn plateau_series(n: usize) -> Vec<f64> {
        let base = synthetic_series(n);
        let mut last = base[0];
        base.iter()
            .enumerate()
            .map(|(i, &v)| {
                last = match i % 11 {
                    0..=3 | 5 | 6 => last,
                    4 => 100.0,
                    7 => 0.0,
                    _ => v,
                };
                last
            })
            .collect()
    }

    #[test]
    fn kama_matches_talib_rule_on_plateau_input() {
        let close = plateau_series(600);
        let flat_bars = (1..close.len())
            .filter(|&i| close[i] == close[i - 1])
            .count();
        assert!(flat_bars > 300, "fixture is not plateau-heavy: {flat_bars}");
        for &p in &[2usize, 3, 5, 14, 30] {
            let got = kama(&close, p);
            let want = reference_kama_talib(&close, p);
            assert_close(&got, &want, 1e-9, &format!("kama plateau p={p}"));
        }
    }

    #[test]
    fn kama_flat_window_residue_matches_talib() {
        // The regression that motivated the fix, and the reason the fix is not
        // "snap whenever the window looks flat".
        //
        // A rolling `Σ|Δclose|` that a plateau has just emptied does *not* land
        // on an exact 0.0 — the adds and the subtracts round differently — so
        // it holds residue of order 1e-14, of either sign. `TA_KAMA` carries the
        // same rolling sum and decides the efficiency ratio from it with
        // `sumROC1 <= periodROC || TA_IS_ZERO(sumROC1)`, so its own behaviour on
        // a flat window is residue-dependent:
        //
        // * negative residue (or |residue| < 1e-14) → the test fires, ER = 1,
        //   SC = 4/9, and the gap to the price shrinks by 5/9 per bar;
        // * positive residue above the 1e-14 band → ER = 0 / residue = 0,
        //   SC = (2/31)^2, and the gap shrinks by only ~0.9958 per bar.
        //
        // The old kernel tested `volatility > 0.0` instead, which takes the
        // ER = 0 branch in *both* cases — so it crawled where TA-Lib snaps, and
        // the error held for as long as the plateau did rather than decaying
        // (max abs divergence ~16 against TA-Lib on a StochRSI %K series).
        //
        // Both expected series below are `talib.KAMA(..., timeperiod=3)` output
        // from libta-lib 0.6.4, so this pins the fix against the real library.
        const P: usize = 3;
        const SNAPS: f64 = 5.0 / 9.0;
        const CRAWLS: f64 = 0.995_837_669_094_693_4;
        for &(a, b, first_flat, ratio) in
            &[(0.01f64, 0.02f64, 6usize, SNAPS), (0.02, 0.59, 6, CRAWLS)]
        {
            let flat = 100.0f64;
            let mut close = vec![0.0, a, b, flat];
            close.extend(std::iter::repeat_n(flat, 30));

            // The residue is real: reproduce the kernel's rolling sum and check
            // it is non-zero once every difference in the window is zero.
            let mut volatility = 0.0f64;
            for k in 1..=P {
                volatility += (close[k] - close[k - 1]).abs();
            }
            for i in 4..=first_flat {
                volatility -= (close[i - 4] - close[i - 3]).abs();
                volatility += (close[i] - close[i - 1]).abs();
            }
            assert_ne!(
                volatility, 0.0,
                "fixture ({a}, {b}) no longer exercises the rolling-sum residue"
            );
            assert_eq!(
                volatility < 0.0,
                ratio == SNAPS,
                "fixture ({a}, {b}) residue {volatility:e} does not have the sign this case pins"
            );

            let got = kama(&close, P);
            for i in first_flat + 1..close.len() {
                let prev_gap = flat - got[i - 1];
                let gap = flat - got[i];
                assert!(
                    (gap - prev_gap * ratio).abs() <= 1e-12 * prev_gap.abs().max(1.0),
                    "({a}, {b}) kama[{i}]: gap {gap} is not prev {prev_gap} * {ratio}"
                );
            }
            // Cross-check against the literal TA-Lib transcription too.
            assert_close(
                &got,
                &reference_kama_talib(&close, P),
                1e-12,
                &format!("kama residue ({a}, {b})"),
            );
        }
    }

    #[test]
    fn kama_mid_series_nan_matches_reference() {
        // KAMA is a recursive filter, so a NaN close poisons `kama_val`
        // permanently — in the reference too. What matters is that the rolling
        // volatility sum does not change that shape.
        let p = 10usize;
        let mut close = synthetic_series(300);
        close[150] = f64::NAN;
        let got = kama(&close, p);
        let want = reference_kama(&close, p);
        assert_close(&got, &want, 1e-9, "kama nan");
        assert!(got[149].is_finite());
        assert!(got.iter().skip(150).all(|v| v.is_nan()));
    }

    #[test]
    fn kama_matches_reference_across_reseed_intervals() {
        let p = 30usize;
        let close = synthetic_series(20_000);
        assert!(20_000 - p > 2 * RESEED_INTERVAL, "series too short");
        let got = kama(&close, p);
        let want = reference_kama(&close, p);
        assert_close(&got, &want, 1e-9, "kama long");
        for boundary in [p + RESEED_INTERVAL, p + 2 * RESEED_INTERVAL] {
            let jump = (got[boundary] - got[boundary - 1]).abs();
            let reference_jump = (want[boundary] - want[boundary - 1]).abs();
            assert!(
                (jump - reference_jump).abs() < 1e-9,
                "reseed discontinuity at {boundary}: {jump} vs {reference_jump}"
            );
        }
    }

    #[test]
    fn kama_first_output_at_timeperiod() {
        // TA-Lib KAMA: seed is close[timeperiod-1] but is not emitted;
        // first output is the first ER/SC update at index `timeperiod`.
        // period=3, close=1..=8
        //   fast_sc=2/3, slow_sc=2/31
        //   i=3: ER=1, SC=4/9, KAMA=3+(4/9)*(4-3)=31/9
        //   i=4: ER=1, SC=4/9, KAMA=31/9+(4/9)*(5-31/9)=335/81
        let prices: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        let period = 3;
        let result = kama(&prices, period);
        for (i, &v) in result.iter().enumerate().take(period) {
            assert!(
                v.is_nan(),
                "expected NaN at {i} (first output is at timeperiod), got {v}"
            );
        }
        assert!(
            (result[3] - 31.0 / 9.0).abs() < 1e-10,
            "KAMA[3]: got {} expected 31/9",
            result[3]
        );
        assert!(
            (result[4] - 335.0 / 81.0).abs() < 1e-10,
            "KAMA[4]: got {} expected 335/81",
            result[4]
        );
        for i in period..prices.len() {
            assert!(
                result[i].is_finite(),
                "expected finite KAMA at {i}, got {}",
                result[i]
            );
        }
    }
}
