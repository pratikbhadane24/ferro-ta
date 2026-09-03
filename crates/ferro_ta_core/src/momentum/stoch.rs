//! Stochastic oscillators: %K/%D (`stoch`), its fast form (`stochf`) and
//! the RSI-driven variant (`stochrsi`).

use super::rsi::rsi;
use crate::overlap::{compute_ma_by_type, ma_into, ma_lookback, MAX_MATYPE};

/// Compute the Stochastic Oscillator (TA-Lib compatible).
///
/// Returns `(slow_k, slow_d)`, both in the range `[0, 100]`.
///  - Fast %K = 100 * (close - lowest low) / (highest high - lowest low)
///  - Slow %K = MA(fast %K, `slowk_period`, `slowk_matype`)
///  - Slow %D = MA(slow %K, `slowd_period`, `slowd_matype`)
///
/// Uses O(n) sliding max/min. Both outputs are `NaN`-padded until slow %D
/// becomes valid (TA-Lib convention).
///
/// # Arguments
/// * `high` / `low` / `close` - OHLC price series (same length).
/// * `fastk_period` - Lookback for highest high / lowest low.
/// * `slowk_period` - MA period applied to fast %K.
/// * `slowk_matype` - MA type for slow %K, `0`–`8` (`0` = SMA, TA-Lib's
///   default; `8` is an alias of `7` = T3). See the
///   [`overlap`](crate::overlap) module docs for the mapping.
/// * `slowd_period` - MA period applied to slow %K.
/// * `slowd_matype` - MA type for slow %D, `0`–`8`.
///
/// The argument order matches `TA_STOCH(fastk_period, slowk_period,
/// slowk_matype, slowd_period, slowd_matype)` — each matype follows the period
/// it applies to, so a mis-ordered call is a type error rather than a wrong
/// answer.
///
/// # NaN warm-up
///
/// `(fastk_period - 1) + ma_lookback(slowk_period, slowk_matype) +
/// ma_lookback(slowd_period, slowd_matype)` leading `NaN`s on **both** outputs.
/// With the default SMAs that is the familiar
/// `fastk_period + slowk_period + slowd_period - 3`; a non-zero matype extends
/// it (`2 * (p - 1)` DEMA, `3 * (p - 1)` TEMA, `p` KAMA, `6 * (p - 1)` T3).
///
/// # Edge cases
///
/// All-`NaN` when the series is empty, any period is `0`, `n < fastk_period`,
/// or either matype exceeds
/// `MAX_MATYPE` (`8`). An out-of-range matype is
/// not silently treated as SMA.
///
/// # TA-Lib compatibility
///
/// The argument order and every matype except `7` agree with `TA_STOCH`. **`7`
/// is T3 here and MAMA in TA-Lib**, so passing `7` silently computes a
/// different indicator than the same call against TA-Lib; pass `8` for a T3
/// that means T3 in both libraries. See the
/// [`overlap`](crate::overlap) module docs.
#[allow(clippy::too_many_arguments)] // mirrors TA_STOCH's argument list
pub fn stoch(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fastk_period: usize,
    slowk_period: usize,
    slowk_matype: u8,
    slowd_period: usize,
    slowd_matype: u8,
) -> (Vec<f64>, Vec<f64>) {
    let n = high.len();
    let nan_pair = || (vec![f64::NAN; n], vec![f64::NAN; n]);
    if n == 0 || fastk_period < 1 || slowk_period < 1 || slowd_period < 1 {
        return nan_pair();
    }
    if slowk_matype > MAX_MATYPE || slowd_matype > MAX_MATYPE {
        return nan_pair();
    }
    if n < fastk_period {
        return nan_pair();
    }

    let mut slowk = vec![f64::NAN; n];
    let mut slowd = vec![f64::NAN; n];

    // Fused pass: compute fast %K inline with sliding max/min.
    // For typical small windows (5-14), inline scan beats VecDeque overhead.
    let fastk_start = fastk_period - 1;
    let fk_len = n - fastk_start;
    let mut fastk_valid = vec![0.0_f64; fk_len];

    for i in fastk_start..n {
        // Inline sliding max(high) and min(low) over [i - fastk_period + 1 .. i].
        let win_start = i + 1 - fastk_period;
        let mut hh = high[win_start];
        let mut ll = low[win_start];
        for j in (win_start + 1)..=i {
            let h = high[j];
            let l = low[j];
            if h > hh {
                hh = h;
            }
            if l < ll {
                ll = l;
            }
        }
        let range = hh - ll;
        fastk_valid[i - fastk_start] = if range != 0.0 {
            100.0 * (close[i] - ll) / range
        } else {
            0.0
        };
    }

    // Slow %K = MA(fastk_valid, slowk_period, slowk_matype).
    ma_into(
        &fastk_valid,
        slowk_period,
        slowk_matype,
        &mut slowk,
        fastk_start,
    );

    // Slow %D = MA(slowk, slowd_period, slowd_matype). Starts are analytic
    // (TA-Lib's lookback chain), not data-derived, so the default-SMA path is
    // bit-identical to the pre-matype kernel even on NaN-bearing input.
    let slowk_valid_start = fastk_start.saturating_add(ma_lookback(slowk_period, slowk_matype));
    let slowd_valid_start =
        slowk_valid_start.saturating_add(ma_lookback(slowd_period, slowd_matype));

    if slowk_valid_start < n {
        let slowk_valid_slice = &slowk[slowk_valid_start..];
        ma_into(
            slowk_valid_slice,
            slowd_period,
            slowd_matype,
            &mut slowd,
            slowk_valid_start,
        );
    }

    // TA-Lib pads BOTH slowk and slowd with NaNs up to the point where both are valid.
    if slowd_valid_start < n {
        for v in slowk.iter_mut().take(slowd_valid_start) {
            *v = f64::NAN;
        }
    } else {
        for v in slowk.iter_mut().take(n) {
            *v = f64::NAN;
        }
    }

    (slowk, slowd)
}

/// Fast Stochastic. Returns `(fastk, fastd)`.
///
/// Fast %K is the raw stochastic; %D is `MA(fast %K, fastd_period,
/// fastd_matype)`. Both outputs are `NaN`-padded until %D is valid.
///
/// # Arguments
/// * `high` / `low` / `close` - OHLC price series (same length).
/// * `fastk_period` - Lookback for highest high / lowest low.
/// * `fastd_period` - MA period applied to fast %K.
/// * `fastd_matype` - MA type for %D, `0`–`8` (`0` = SMA, TA-Lib's default;
///   `8` is an alias of `7` = T3).
///
/// # NaN warm-up
///
/// `(fastk_period - 1) + ma_lookback(fastd_period, fastd_matype)` on both
/// outputs — `fastk_period + fastd_period - 2` for the default SMA.
///
/// # Edge cases
///
/// All-`NaN` when the series is empty, either period is `0`,
/// `n < fastk_period`, or `fastd_matype` exceeds
/// `MAX_MATYPE` (`8`).
///
/// # TA-Lib compatibility
///
/// `TA_STOCHF(fastk_period, fastd_period, fastd_matype)`; same argument order.
/// Every matype except `7` agrees with TA-Lib: **`7` is T3 here and MAMA in
/// TA-Lib**, so pass `8` for a T3 that means T3 in both libraries. See the
/// [`overlap`](crate::overlap) module docs.
pub fn stochf(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fastk_period: usize,
    fastd_period: usize,
    fastd_matype: u8,
) -> (Vec<f64>, Vec<f64>) {
    // slowk_period=1 with an SMA leaves Fast %K unsmoothed (a 1-bar SMA is the
    // identity, and its lookback is 0); `fastd_matype` then types the %D leg.
    stoch(
        high,
        low,
        close,
        fastk_period,
        1,
        0,
        fastd_period,
        fastd_matype,
    )
}

// ---------------------------------------------------------------------------
// Stochastic RSI
// ---------------------------------------------------------------------------

/// Stochastic RSI. Returns `(fastk, fastd)`.
///
/// %K is the stochastic of the RSI series; %D is
/// `MA(%K, fastd_period, fastd_matype)`.
///
/// # Arguments
/// * `close` - Price series.
/// * `timeperiod` - RSI period.
/// * `fastk_period` - Lookback for the RSI max/min window.
/// * `fastd_period` - MA period applied to %K.
/// * `fastd_matype` - MA type for %D, `0`–`8` (`0` = SMA, TA-Lib's default;
///   `8` is an alias of `7` = T3). See the
///   [`overlap`](crate::overlap) module docs for the mapping.
///
/// # NaN warm-up
///
/// `timeperiod + (fastk_period - 1) + ma_lookback(fastd_period, fastd_matype)`
/// leading `NaN`s on **both** outputs (TA-Lib pads %K to where %D starts).
/// With the default SMA that is `timeperiod + fastk_period + fastd_period - 2`;
/// a non-zero matype extends it (`2 * (p - 1)` DEMA, `3 * (p - 1)` TEMA, `p`
/// KAMA, `6 * (p - 1)` T3). Bars whose %K window contains a `NaN` stay `NaN`,
/// which for `fastd_matype == 0` blanks exactly the affected %D windows.
///
/// # Edge cases
///
/// All-`NaN` when any period is `0` or `fastd_matype` exceeds
/// `MAX_MATYPE` (`8`). An out-of-range matype is
/// not silently treated as SMA.
///
/// # TA-Lib compatibility
///
/// `TA_STOCHRSI(timeperiod, fastk_period, fastd_period, fastd_matype)`; same
/// argument order. The RSI seed differs slightly from TA-Lib's, so ferro-ta can
/// emit values up to two bars sooner (documented in `TA_LIB_COMPATIBILITY.md`).
/// Every matype except `7` agrees with TA-Lib: **`7` is T3 here and MAMA in
/// TA-Lib**, so pass `8` for a T3 that means T3 in both libraries. See the
/// [`overlap`](crate::overlap) module docs.
pub fn stochrsi(
    close: &[f64],
    timeperiod: usize,
    fastk_period: usize,
    fastd_period: usize,
    fastd_matype: u8,
) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    let nan_pair = || (vec![f64::NAN; n], vec![f64::NAN; n]);
    if timeperiod == 0 || fastk_period == 0 || fastd_period == 0 {
        return nan_pair();
    }
    if fastd_matype > MAX_MATYPE {
        return nan_pair();
    }

    let rsi_vals = rsi(close, timeperiod);
    let rsi_warmup = timeperiod;
    let k_warmup = rsi_warmup + fastk_period - 1;
    let d_warmup = k_warmup.saturating_add(ma_lookback(fastd_period, fastd_matype));

    let mut fastk = vec![f64::NAN; n];
    let mut fastd = vec![f64::NAN; n];

    for i in k_warmup..n {
        if rsi_vals[i].is_nan() {
            continue;
        }
        let start = i + 1 - fastk_period;
        if (start..=i).any(|j| rsi_vals[j].is_nan()) {
            continue;
        }
        let mx = rsi_vals[start..=i]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mn = rsi_vals[start..=i]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        fastk[i] = if mx != mn {
            100.0 * (rsi_vals[i] - mn) / (mx - mn)
        } else {
            // TA-Lib STOCHF convention for a flat window.
            0.0
        };
    }

    if fastd_matype == 0 {
        // Kept as a per-window mean rather than routed through `sma_into`:
        // the streaming recurrence would let one interior `NaN` in %K poison
        // every later %D, and this is the shipped (and golden-tested) shape.
        for i in d_warmup..n {
            let start = i + 1 - fastd_period;
            let window = &fastk[start..=i];
            if window.iter().all(|v| !v.is_nan()) {
                fastd[i] = window.iter().sum::<f64>() / fastd_period as f64;
            }
        }
    } else if k_warmup < n {
        // Type the %D leg over the valid %K region only, so the leading NaNs
        // do not poison an EMA-composed seed.
        let typed = compute_ma_by_type(&fastk[k_warmup..], fastd_period, fastd_matype);
        for (j, &v) in typed.iter().enumerate() {
            if !v.is_nan() {
                fastd[k_warmup + j] = v;
            }
        }
    }

    // TA-Lib pads both outputs to the combined lookback, so fastk starts where
    // fastd does rather than `fastd_period - 1` bars earlier.
    for v in fastk[..d_warmup.min(n)].iter_mut() {
        *v = f64::NAN;
    }
    (fastk, fastd)
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{assert_bits, oracle_series};
    use super::*;

    // -----------------------------------------------------------------------
    // Equivalence oracles: verbatim copies of `stoch` and `stochrsi` as they
    // stood before the matype arguments existed (both hard-wired to SMA, i.e.
    // TA-Lib's `matype = 0` default). The new kernels must be bit-identical to
    // these at matype 0 — that is what keeps the TA-Lib conformance suite and
    // the WASM/Flutter goldens in place.
    // -----------------------------------------------------------------------
    fn reference_stoch(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        fastk_period: usize,
        slowk_period: usize,
        slowd_period: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = high.len();
        let nan_pair = || (vec![f64::NAN; n], vec![f64::NAN; n]);
        if n == 0 || fastk_period < 1 || slowk_period < 1 || slowd_period < 1 {
            return nan_pair();
        }
        if n < fastk_period {
            return nan_pair();
        }

        let mut slowk = vec![f64::NAN; n];
        let mut slowd = vec![f64::NAN; n];

        let fastk_start = fastk_period - 1;
        let fk_len = n - fastk_start;
        let mut fastk_valid = vec![0.0_f64; fk_len];

        for i in fastk_start..n {
            let win_start = i + 1 - fastk_period;
            let mut hh = high[win_start];
            let mut ll = low[win_start];
            for j in (win_start + 1)..=i {
                let h = high[j];
                let l = low[j];
                if h > hh {
                    hh = h;
                }
                if l < ll {
                    ll = l;
                }
            }
            let range = hh - ll;
            fastk_valid[i - fastk_start] = if range != 0.0 {
                100.0 * (close[i] - ll) / range
            } else {
                0.0
            };
        }

        crate::overlap::sma_into(&fastk_valid, slowk_period, &mut slowk, fastk_start);

        let slowk_valid_start = fastk_start + slowk_period - 1;
        let slowd_valid_start = slowk_valid_start + slowd_period - 1;

        if slowk_valid_start < n {
            let slowk_valid_slice = &slowk[slowk_valid_start..];
            crate::overlap::sma_into(
                slowk_valid_slice,
                slowd_period,
                &mut slowd,
                slowk_valid_start,
            );
        }

        if slowd_valid_start < n {
            for v in slowk.iter_mut().take(slowd_valid_start) {
                *v = f64::NAN;
            }
        } else {
            for v in slowk.iter_mut().take(n) {
                *v = f64::NAN;
            }
        }

        (slowk, slowd)
    }

    fn reference_stochrsi(
        close: &[f64],
        timeperiod: usize,
        fastk_period: usize,
        fastd_period: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let nan_pair = || (vec![f64::NAN; n], vec![f64::NAN; n]);
        if timeperiod == 0 || fastk_period == 0 || fastd_period == 0 {
            return nan_pair();
        }

        let rsi_vals = rsi(close, timeperiod);
        let rsi_warmup = timeperiod;
        let k_warmup = rsi_warmup + fastk_period - 1;
        let d_warmup = k_warmup + fastd_period - 1;

        let mut fastk = vec![f64::NAN; n];
        let mut fastd = vec![f64::NAN; n];

        for i in k_warmup..n {
            if rsi_vals[i].is_nan() {
                continue;
            }
            let start = i + 1 - fastk_period;
            if (start..=i).any(|j| rsi_vals[j].is_nan()) {
                continue;
            }
            let mx = rsi_vals[start..=i]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let mn = rsi_vals[start..=i]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            fastk[i] = if mx != mn {
                100.0 * (rsi_vals[i] - mn) / (mx - mn)
            } else {
                0.0
            };
        }

        for i in d_warmup..n {
            let start = i + 1 - fastd_period;
            let window = &fastk[start..=i];
            if window.iter().all(|v| !v.is_nan()) {
                fastd[i] = window.iter().sum::<f64>() / fastd_period as f64;
            }
        }

        for v in fastk[..d_warmup.min(n)].iter_mut() {
            *v = f64::NAN;
        }
        (fastk, fastd)
    }

    fn stoch_period_triples() -> Vec<(usize, usize, usize)> {
        vec![
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
            (2, 1, 1),
            (3, 3, 3),
            (5, 3, 3),
            (14, 3, 3),
            (5, 1, 2),
            (7, 5, 4),
            (400, 3, 3),
        ]
    }

    #[test]
    fn stoch_matype0_is_bit_identical_to_pre_matype_kernel() {
        for (name, high, low) in oracle_series() {
            // `oracle_series` yields (high, low) with low <= high; use the
            // midpoint as close so the flat-range branch is still hit by the
            // all-equal and low-cardinality fixtures.
            let close: Vec<f64> = high
                .iter()
                .zip(low.iter())
                .map(|(h, l)| (h + l) * 0.5)
                .collect();
            for (fk, sk, sd) in stoch_period_triples() {
                let ctx = format!("{name} stoch({fk},{sk},{sd})");
                let (k, d) = stoch(&high, &low, &close, fk, sk, 0, sd, 0);
                let (rk, rd) = reference_stoch(&high, &low, &close, fk, sk, sd);
                assert_bits(&k, &rk, &format!("{ctx} slowk"));
                assert_bits(&d, &rd, &format!("{ctx} slowd"));
            }
        }
    }

    #[test]
    fn stochf_matype0_is_bit_identical_to_pre_matype_kernel() {
        for (name, high, low) in oracle_series() {
            let close: Vec<f64> = high
                .iter()
                .zip(low.iter())
                .map(|(h, l)| (h + l) * 0.5)
                .collect();
            for (fk, fd) in [(0usize, 1usize), (1, 0), (1, 1), (3, 2), (5, 3), (14, 3)] {
                let ctx = format!("{name} stochf({fk},{fd})");
                let (k, d) = stochf(&high, &low, &close, fk, fd, 0);
                // The pre-change `stochf` delegated to `stoch(.., 1, fd)`.
                let (rk, rd) = reference_stoch(&high, &low, &close, fk, 1, fd);
                assert_bits(&k, &rk, &format!("{ctx} fastk"));
                assert_bits(&d, &rd, &format!("{ctx} fastd"));
            }
        }
    }

    #[test]
    fn stochrsi_matype0_is_bit_identical_to_pre_matype_kernel() {
        for (name, series, other) in oracle_series() {
            for close in [&series, &other] {
                for (tp, fk, fd) in [
                    (0usize, 1usize, 1usize),
                    (1, 1, 1),
                    (2, 2, 2),
                    (14, 5, 3),
                    (14, 3, 3),
                    (5, 14, 3),
                    (400, 5, 3),
                ] {
                    let ctx = format!("{name} stochrsi({tp},{fk},{fd})");
                    let (k, d) = stochrsi(close, tp, fk, fd, 0);
                    let (rk, rd) = reference_stochrsi(close, tp, fk, fd);
                    assert_bits(&k, &rk, &format!("{ctx} fastk"));
                    assert_bits(&d, &rd, &format!("{ctx} fastd"));
                }
            }
        }
    }

    #[test]
    fn stochastics_reject_out_of_range_matype() {
        let high: Vec<f64> = (1..=80).map(|i| 100.0 + (i % 9) as f64).collect();
        let low: Vec<f64> = high.iter().map(|h| h - 2.0).collect();
        let close: Vec<f64> = high.iter().map(|h| h - 1.0).collect();
        let all_nan = |v: &[f64], ctx: &str| {
            assert_eq!(v.len(), high.len(), "{ctx}: length");
            assert!(v.iter().all(|x| x.is_nan()), "{ctx}: not all NaN");
        };
        for matype in [MAX_MATYPE + 1, 99, u8::MAX] {
            let (k, d) = stoch(&high, &low, &close, 5, 3, matype, 3, 0);
            all_nan(&k, "stoch slowk_matype k");
            all_nan(&d, "stoch slowk_matype d");
            let (k, d) = stoch(&high, &low, &close, 5, 3, 0, 3, matype);
            all_nan(&k, "stoch slowd_matype k");
            all_nan(&d, "stoch slowd_matype d");
            let (k, d) = stochf(&high, &low, &close, 5, 3, matype);
            all_nan(&k, "stochf k");
            all_nan(&d, "stochf d");
            let (k, d) = stochrsi(&close, 14, 5, 3, matype);
            all_nan(&k, "stochrsi k");
            all_nan(&d, "stochrsi d");
        }
    }

    #[test]
    fn stoch_matype_shifts_warmup_per_ma_lookback() {
        let n = 600;
        let high: Vec<f64> = (0..n).map(|i| 100.0 + ((i * 7) % 23) as f64).collect();
        let low: Vec<f64> = high.iter().map(|h| h - 3.0).collect();
        let close: Vec<f64> = high.iter().map(|h| h - 1.5).collect();
        let (fk, sk, sd) = (5usize, 3usize, 3usize);
        for matype in 0..=7u8 {
            let (k, d) = stoch(&high, &low, &close, fk, sk, matype, sd, matype);
            let expected = (fk - 1)
                + crate::overlap::ma_lookback(sk, matype)
                + crate::overlap::ma_lookback(sd, matype);
            for i in 0..expected {
                assert!(k[i].is_nan(), "matype={matype}: slowk at {i} = {}", k[i]);
                assert!(d[i].is_nan(), "matype={matype}: slowd at {i} = {}", d[i]);
            }
            assert!(
                k[expected].is_finite() && d[expected].is_finite(),
                "matype={matype}: expected both outputs at {expected}"
            );
        }
    }

    #[test]
    fn stochf_matype1_smooths_percent_d_with_an_ema() {
        // %K is untouched by `fastd_matype`; only %D changes.
        let high: Vec<f64> = (0..120).map(|i| 100.0 + ((i * 5) % 17) as f64).collect();
        let low: Vec<f64> = high.iter().map(|h| h - 4.0).collect();
        let close: Vec<f64> = high.iter().map(|h| h - 2.0).collect();
        let (k0, d0) = stochf(&high, &low, &close, 5, 3, 0);
        let (k1, d1) = stochf(&high, &low, &close, 5, 3, 1);
        // Same warm-up (EMA and SMA share a `p - 1` lookback), same %K.
        for i in 0..high.len() {
            assert_eq!(
                k0[i].is_nan(),
                k1[i].is_nan(),
                "%K NaN placement differs at {i}"
            );
            if !k0[i].is_nan() {
                assert!((k0[i] - k1[i]).abs() < 1e-12, "%K differs at {i}");
            }
        }
        assert!(
            (0..high.len()).any(|i| !d0[i].is_nan() && (d0[i] - d1[i]).abs() > 1e-9),
            "SMA and EMA %D must not coincide"
        );
    }

    #[test]
    fn stoch_basic() {
        let high = vec![10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5];
        let low = vec![9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5];
        let close = vec![9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0];
        let (slowk, slowd) = stoch(&high, &low, &close, 3, 3, 0, 3, 0);
        // Check that valid values are in [0, 100]
        for v in slowk.iter().filter(|v| !v.is_nan()) {
            assert!(*v >= 0.0 && *v <= 100.0, "slowk out of range: {v}");
        }
        for v in slowd.iter().filter(|v| !v.is_nan()) {
            assert!(*v >= 0.0 && *v <= 100.0, "slowd out of range: {v}");
        }
    }

    #[test]
    fn stochf_golden_sma_d() {
        // Hand-computed STOCHF(fastk=3, fastd=2) — TA-Lib fastd_matype=0 (SMA).
        // Both outputs are NaN-padded until %D is valid (lookback = 3-1+2-1 = 3).
        //
        // Fast %K:
        //   i=2: HH=12, LL=8, K=100*(10-8)/4 = 50
        //   i=3: HH=13, LL=8, K=100*(12-8)/5 = 80
        //   i=4: HH=14, LL=8, K=100*(13-8)/6 = 250/3
        //   i=5: HH=15, LL=10, K=100*(14-10)/5 = 80
        // SMA %D period 2:
        //   i=3: (50+80)/2 = 65
        //   i=4: (80+250/3)/2 = 245/3
        //   i=5: (250/3+80)/2 = 245/3
        let high = [10.0, 12.0, 11.0, 13.0, 14.0, 15.0];
        let low = [8.0, 9.0, 8.0, 10.0, 11.0, 12.0];
        let close = [9.0, 11.0, 10.0, 12.0, 13.0, 14.0];
        let (fastk, fastd) = stochf(&high, &low, &close, 3, 2, 0);
        let first_valid = 3; // fastk_period-1 + fastd_period-1
        for i in 0..first_valid {
            assert!(
                fastk[i].is_nan(),
                "expected NaN fastk at {i}, got {}",
                fastk[i]
            );
            assert!(
                fastd[i].is_nan(),
                "expected NaN fastd at {i}, got {}",
                fastd[i]
            );
        }
        let expected_k = [80.0, 250.0 / 3.0, 80.0];
        let expected_d = [65.0, 245.0 / 3.0, 245.0 / 3.0];
        for (offset, (&exp_k, &exp_d)) in expected_k.iter().zip(expected_d.iter()).enumerate() {
            let i = first_valid + offset;
            assert!(
                (fastk[i] - exp_k).abs() < 1e-10,
                "fastk[{i}]: got {} expected {exp_k}",
                fastk[i]
            );
            assert!(
                (fastd[i] - exp_d).abs() < 1e-10,
                "fastd[{i}]: got {} expected {exp_d} (must be SMA, not EMA)",
                fastd[i]
            );
        }
    }
}
