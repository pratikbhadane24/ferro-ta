//! Volume indicators.

/// Compute On-Balance Volume (OBV).
///
/// OBV is a cumulative indicator that adds volume on up-close bars and
/// subtracts volume on down-close bars. Unchanged closes contribute zero.
/// Bar 0 is seeded with `volume[0]` (TA-Lib; there is no prior close).
/// Returns a `Vec<f64>` of length `n` with no `NaN` values.
///
/// # Arguments
/// * `close` - Price series.
/// * `volume` - Volume series (same length as `close`).
pub fn obv(close: &[f64], volume: &[f64]) -> Vec<f64> {
    let n = close.len();
    if n == 0 || volume.len() != n {
        // OBV documents a NaN-free cumulative series, so an invalid input
        // yields zeros rather than NaN.
        return vec![0.0_f64; n];
    }

    // The direction test is a near coin flip on real (random-walk) price data,
    // so an `if / else if / else` chain mispredicts on roughly half the bars at
    // ~15-20 wasted cycles each - which dominates the single floating-point add
    // that is the actual work. Select the increment with bit arithmetic instead:
    // both comparisons lower to `cmp` + `cset`/`setcc` with no branch at all.
    //
    // The selection is *bit-identical* to the branchy form in every case:
    //   * close[i] > close[i-1]  -> mask = !0, sign = 0    -> delta = volume[i]
    //   * close[i] < close[i-1]  -> mask = !0, sign = 1<<63-> delta = -volume[i]
    //                               (a sign-bit flip is exactly IEEE-754 negation,
    //                               NaN and +-0.0 included)
    //   * equal, or either close NaN -> both compares false, mask = 0, sign = 0
    //                               -> delta = from_bits(0) = +0.0
    // Note the tempting `((d > 0.0) as i32 - (d < 0.0) as i32) as f64 * volume[i]`
    // is NOT equivalent: `0.0 * volume[i]` is NaN for NaN/infinite volume, where
    // the branchy form adds an exact 0.0.
    // Note the Group E `Vec::with_capacity` + `push` prologue trick is *not* used
    // here: OBV seeds with 0.0, so `vec![0.0; n]` lowers to `alloc_zeroed`
    // (lazily-mapped zero pages, essentially free) rather than a real store pass,
    // while `push` would add a capacity check and stack traffic to every bar.
    // Zipped `iter_mut` keeps the writes bounds-check free.
    let mut result = vec![0.0_f64; n];
    let mut acc = volume[0];
    result[0] = acc;
    for ((out, window), &v) in result[1..]
        .iter_mut()
        .zip(close.windows(2))
        .zip(&volume[1..])
    {
        let d = window[1] - window[0];
        let is_down = (d < 0.0) as u64;
        let keep = 0u64.wrapping_sub((d > 0.0) as u64 | is_down);
        let delta = f64::from_bits((v.to_bits() & keep) ^ (is_down << 63));
        acc += delta;
        *out = acc;
    }
    result
}

/// Compute the Money Flow Index (MFI).
///
/// MFI is a volume-weighted RSI, returning values in `[0, 100]`.
/// `typical_price = (H + L + C) / 3`; money flow is positive when
/// typical price rises, negative when it falls. The first `timeperiod`
/// values are `NaN`.
///
/// # Arguments
/// * `high` / `low` / `close` - OHLC price series (same length).
/// * `volume` - Volume series (same length).
/// * `timeperiod` - Lookback window (typically 14).
pub fn mfi(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    timeperiod: usize,
) -> Vec<f64> {
    let n = high.len();
    if timeperiod < 1 || n <= timeperiod || low.len() != n || close.len() != n || volume.len() != n
    {
        return vec![f64::NAN; n];
    }

    let mut pos_flow = vec![0.0_f64; n];
    let mut neg_flow = vec![0.0_f64; n];
    let mut tp_prev = (high[0] + low[0] + close[0]) / 3.0;

    for i in 1..n {
        let tp_cur = (high[i] + low[i] + close[i]) / 3.0;
        let rmf = tp_cur * volume[i];
        if tp_cur > tp_prev {
            pos_flow[i] = rmf;
        } else if tp_cur < tp_prev {
            neg_flow[i] = rmf;
        }
        tp_prev = tp_cur;
    }

    // Sliding window sum over timeperiod bars (indices i+1-timeperiod ..= i).
    // First valid window: indices 1..=timeperiod.
    let mut pos_sum: f64 = pos_flow[1..=timeperiod].iter().sum();
    let mut neg_sum: f64 = neg_flow[1..=timeperiod].iter().sum();
    // TA-Lib convention: zero total money flow (flat window / zero volume)
    // yields 0, not 100. 100*p/(p+n) is algebraically 100 - 100/(1+p/n).
    // Clamp the documented [0, 100] range: pos/(pos+neg) is mathematically in
    // [0, 1], but rounding can exceed 1.0 when neg is denormal.
    let mfi_val = |pos: f64, neg: f64| {
        let denom = pos + neg;
        if denom == 0.0 {
            0.0
        } else {
            (100.0 * pos / denom).clamp(0.0, 100.0)
        }
    };
    // Pre-size and store by index. The `Vec::with_capacity` + `push` shape this
    // replaces did avoid the NaN fill's store pass, but it paid a capacity check
    // and a vec-header reload on *every* bar, which measured slower than the
    // one-time prologue for a kernel this light. The warm-up bars simply keep
    // their initialized `NaN`.
    let mut result = vec![f64::NAN; n];
    result[timeperiod] = mfi_val(pos_sum, neg_sum);

    for i in (timeperiod + 1)..n {
        pos_sum += pos_flow[i] - pos_flow[i - timeperiod];
        neg_sum += neg_flow[i] - neg_flow[i - timeperiod];
        result[i] = mfi_val(pos_sum, neg_sum);
    }
    result
}

/// Chaikin Accumulation/Distribution Line.
///
/// Cumulates `(close - low - (high - close)) / (high - low) * volume`.
pub fn ad(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
    let n = high.len();
    if low.len() != n || close.len() != n || volume.len() != n {
        // AD documents a NaN-free cumulative series, so a mismatch yields zeros.
        return vec![0.0_f64; n];
    }
    // `hl != 0.0` is a value test, not a coin flip: on real OHLC data it is
    // essentially always true, so it predicts near-perfectly and is left branchy.
    // A branchless form would also have to suppress the division's inf/NaN, which
    // costs more than the branch saves here.
    // As in `obv`: the 0.0 seed makes `vec![0.0; n]` an `alloc_zeroed`, which is
    // cheaper than `push`'s per-bar capacity check.
    let mut result = vec![0.0_f64; n];
    let mut ad_val = 0.0_f64;
    for ((((out, &h), &l), &c), &v) in result.iter_mut().zip(high).zip(low).zip(close).zip(volume) {
        let hl = h - l;
        let clv = if hl != 0.0 {
            ((c - l) - (h - c)) / hl
        } else {
            0.0
        };
        ad_val += clv * v;
        *out = ad_val;
    }
    result
}

/// Chaikin A/D Oscillator: fast EMA of AD minus slow EMA of AD.
///
/// Both EMAs are seeded with the *first* A/D value and run from bar 0, with
/// the first output at `slowperiod - 1` (TA-Lib's `ta_ADOSC.c` convention).
/// Note this differs from `overlap::ema`, which uses an SMA seed.
pub fn adosc(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    fastperiod: usize,
    slowperiod: usize,
) -> Vec<f64> {
    let n = high.len();
    if fastperiod < 1
        || slowperiod < 1
        || n == 0
        || low.len() != n
        || close.len() != n
        || volume.len() != n
    {
        return vec![f64::NAN; n];
    }
    let ad_vals = ad(high, low, close, volume);
    let fast_k = 2.0 / (fastperiod as f64 + 1.0);
    let slow_k = 2.0 / (slowperiod as f64 + 1.0);

    let mut fast_ema = ad_vals[0];
    let mut slow_ema = ad_vals[0];
    let warmup = slowperiod - 1;
    // Pre-size and store by index rather than `Vec::with_capacity` + `push`:
    // `push`'s per-bar capacity check and vec-header reload cost more here than
    // the one-time NaN fill it saves. Warm-up bars keep their initialized `NaN`,
    // so only the emitting bars are written.
    let mut result = vec![f64::NAN; n];
    if warmup == 0 {
        // `slowperiod == 1`: bar 0 already emits (both EMAs are seeded from
        // `ad_vals[0]`, so it is exactly 0.0).
        result[0] = fast_ema - slow_ema;
    }
    // `i >= warmup` is monotone, so rather than test it per bar, split the pass
    // at the first emitting bar: the warm-up phase only advances the two EMAs
    // and the emitting phase stores unconditionally. The arithmetic per bar is
    // unchanged, so the output is bit-identical to the single-loop form.
    let store_from = warmup.max(1).min(n);
    for &a in &ad_vals[1..store_from] {
        fast_ema = fast_k * a + (1.0 - fast_k) * fast_ema;
        slow_ema = slow_k * a + (1.0 - slow_k) * slow_ema;
    }
    for (out, &a) in result[store_from..].iter_mut().zip(&ad_vals[store_from..]) {
        fast_ema = fast_k * a + (1.0 - fast_k) * fast_ema;
        slow_ema = slow_k * a + (1.0 - slow_k) * slow_ema;
        *out = fast_ema - slow_ema;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verbatim copy of the pre-optimization branchy OBV, kept as the bit-identity
    /// reference for the branchless rewrite. Do not "modernize" it.
    fn reference_obv(close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0_f64; n];
        if n == 0 || volume.len() != n {
            return result;
        }
        result[0] = volume[0];
        for i in 1..n {
            result[i] = result[i - 1]
                + if close[i] > close[i - 1] {
                    volume[i]
                } else if close[i] < close[i - 1] {
                    -volume[i]
                } else {
                    0.0
                };
        }
        result
    }

    fn assert_obv_bit_identical(close: &[f64], volume: &[f64], label: &str) {
        let got = obv(close, volume);
        let want = reference_obv(close, volume);
        assert_eq!(got.len(), want.len(), "{label}: length");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{label}: OBV[{i}] got {g} ({:#x}) expected {w} ({:#x})",
                g.to_bits(),
                w.to_bits()
            );
        }
    }

    /// Deterministic low-cardinality series: random f64 data essentially never
    /// produces equal adjacent closes, and the unchanged-close case (which must
    /// contribute exactly zero) is the one most likely to break in a rewrite.
    fn low_cardinality_closes(n: usize) -> Vec<f64> {
        // xorshift-ish LCG, then quantize to 5 distinct levels.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                100.0 + ((state >> 33) % 5) as f64
            })
            .collect()
    }

    #[test]
    fn obv_matches_branchy_reference_bit_for_bit() {
        let n = 4096;
        let close = low_cardinality_closes(n);
        // Volumes covering positive, negative, zero and -0.0.
        let volume: Vec<f64> = (0..n)
            .map(|i| match i % 5 {
                0 => 0.0,
                1 => -0.0,
                2 => -1234.5,
                3 => 1e9,
                _ => i as f64,
            })
            .collect();
        assert_obv_bit_identical(&close, &volume, "low-cardinality");

        // Strictly monotone closes: every bar takes the up branch.
        let up: Vec<f64> = (0..n).map(|i| i as f64).collect();
        assert_obv_bit_identical(&up, &volume, "monotone up");
        let down: Vec<f64> = (0..n).map(|i| -(i as f64)).collect();
        assert_obv_bit_identical(&down, &volume, "monotone down");

        // A long run of exactly equal closes must contribute exactly zero.
        let flat = vec![42.0_f64; n];
        assert_obv_bit_identical(&flat, &volume, "flat");
    }

    #[test]
    fn obv_matches_reference_with_non_finite_inputs() {
        // NaN / infinite closes and volumes. This is where the naive
        // `sign * volume[i]` rewrite diverges: 0.0 * NaN is NaN, while the
        // branchy form adds an exact 0.0 on an unchanged close.
        let close = [
            1.0,
            1.0,
            f64::NAN,
            f64::NAN,
            2.0,
            1.0,
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            3.0,
            3.0,
        ];
        let volumes: [[f64; 11]; 3] = [
            [
                10.0,
                f64::NAN,
                20.0,
                f64::NAN,
                30.0,
                40.0,
                f64::NAN,
                50.0,
                60.0,
                f64::NAN,
                f64::NAN,
            ],
            [
                1.0,
                f64::INFINITY,
                2.0,
                f64::NEG_INFINITY,
                3.0,
                4.0,
                f64::INFINITY,
                5.0,
                6.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ],
            [0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0],
        ];
        for (k, v) in volumes.iter().enumerate() {
            assert_obv_bit_identical(&close, v, &format!("non-finite set {k}"));
        }
    }

    #[test]
    fn obv_step_contract_holds() {
        // Independently of the reference copy: each step must move the series by
        // exactly +volume[i], -volume[i], or exactly 0.0.
        let n = 2048;
        let close = low_cardinality_closes(n);
        let volume: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 - 100.0).collect();
        let result = obv(&close, &volume);
        assert_eq!(result[0].to_bits(), volume[0].to_bits());
        for i in 1..n {
            let step = result[i] - result[i - 1];
            let ok = step.to_bits() == volume[i].to_bits()
                || step.to_bits() == (-volume[i]).to_bits()
                || step == 0.0;
            assert!(
                ok,
                "OBV step at {i} was {step}, not +/-{} or 0.0",
                volume[i]
            );
            // And the direction must match the close comparison.
            if close[i] > close[i - 1] {
                assert_eq!(step.to_bits(), volume[i].to_bits(), "up bar {i}");
            } else if close[i] < close[i - 1] {
                assert_eq!(step.to_bits(), (-volume[i]).to_bits(), "down bar {i}");
            } else {
                assert_eq!(step.to_bits(), 0.0_f64.to_bits(), "flat bar {i}");
            }
        }
    }

    #[test]
    fn obv_unchanged_closes_contribute_exactly_zero() {
        let close = [5.0, 5.0, 5.0, 5.0];
        let volume = [7.0, 1e300, -1e300, 3.0];
        let result = obv(&close, &volume);
        // Every increment is exactly +0.0, so the whole series stays at volume[0].
        for (i, v) in result.iter().enumerate() {
            assert_eq!(v.to_bits(), 7.0_f64.to_bits(), "OBV[{i}] = {v}");
        }
    }

    #[test]
    fn obv_empty_and_single_bar() {
        assert!(obv(&[], &[]).is_empty());
        let single = obv(&[10.0], &[123.0]);
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].to_bits(), 123.0_f64.to_bits());
        // Single bar with NaN volume still seeds verbatim.
        let nan_seed = obv(&[10.0], &[f64::NAN]);
        assert!(nan_seed[0].is_nan());
        assert_obv_bit_identical(&[], &[], "empty");
        assert_obv_bit_identical(&[10.0], &[123.0], "single");
    }

    #[test]
    fn obv_negative_and_zero_volumes() {
        let close = [1.0, 2.0, 1.0, 1.0, 3.0];
        let volume = [-100.0, -50.0, 0.0, -0.0, -25.0];
        let expected = [-100.0, -150.0, -150.0, -150.0, -175.0];
        let result = obv(&close, &volume);
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(result[i], exp, "OBV[{i}] = {}", result[i]);
        }
        assert_obv_bit_identical(&close, &volume, "negative/zero volumes");
    }

    #[test]
    fn obv_up_trend() {
        let c = vec![1.0, 2.0, 3.0];
        let v = vec![100.0, 200.0, 300.0];
        let result = obv(&c, &v);
        assert!((result[0] - 100.0).abs() < 1e-10);
        assert!((result[1] - 300.0).abs() < 1e-10);
        assert!((result[2] - 600.0).abs() < 1e-10);
    }

    #[test]
    fn obv_bar0_accumulates_like_talib() {
        // TA-Lib seeds OBV[0] = volume[0] (no prior close to compare),
        // then adds/subtracts subsequent volume. The old bug left bar 0 at 0.
        let c = [1.0, 2.0, 3.0, 2.0, 2.0];
        let v = [100.0, 200.0, 300.0, 400.0, 50.0];
        let result = obv(&c, &v);
        let expected = [100.0, 300.0, 600.0, 200.0, 200.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (result[i] - exp).abs() < 1e-10,
                "OBV[{i}]: got {} expected {exp}",
                result[i]
            );
        }
    }

    #[test]
    fn ad_basic() {
        let h = vec![10.0, 12.0, 11.0];
        let l = vec![8.0, 9.0, 9.0];
        let c = vec![9.0, 11.0, 10.0];
        let v = vec![1000.0, 2000.0, 1500.0];
        let result = ad(&h, &l, &c, &v);
        assert_eq!(result.len(), 3);
        // CLV[0] = ((9-8) - (10-9)) / (10-8) = (1 - 1) / 2 = 0
        assert!((result[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn adosc_basic() {
        let n = 30;
        let h: Vec<f64> = (1..=n).map(|i| i as f64 + 1.0).collect();
        let l: Vec<f64> = (1..=n).map(|i| i as f64 - 1.0).collect();
        let c: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let v: Vec<f64> = vec![1000.0; n];
        let result = adosc(&h, &l, &c, &v, 3, 10);
        assert_eq!(result.len(), n);
        // Warmup period should be NaN
        for i in 0..9 {
            assert!(result[i].is_nan());
        }
    }

    #[test]
    fn mfi_range() {
        let n = 50;
        let high: Vec<f64> = (1..=n).map(|i| i as f64 + 0.5).collect();
        let low: Vec<f64> = (1..=n).map(|i| i as f64 - 0.5).collect();
        let close: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let volume: Vec<f64> = vec![1_000_000.0; n];
        let result = mfi(&high, &low, &close, &volume, 14);
        for v in result.iter().filter(|v| !v.is_nan()) {
            assert!(*v >= 0.0 && *v <= 100.0, "MFI out of range: {v}");
        }
    }

    #[test]
    fn obv_length_mismatch_returns_zeros() {
        let c = [1.0, 2.0, 3.0, 4.0];
        for v in [vec![10.0, 20.0], vec![10.0, 20.0, 30.0, 40.0, 50.0]] {
            let result = obv(&c, &v);
            assert_eq!(result.len(), c.len());
            // OBV documents a NaN-free output, so a mismatch yields zeros.
            assert!(result.iter().all(|x| *x == 0.0));
        }
    }

    #[test]
    fn mfi_length_mismatch_returns_nan() {
        let h = [10.0, 11.0, 12.0, 13.0, 14.0];
        let l = [9.0, 10.0, 11.0, 12.0, 13.0];
        let c = [9.5, 10.5, 11.5, 12.5, 13.5];
        let v = [100.0, 100.0, 100.0, 100.0, 100.0];
        let short = [1.0, 2.0];
        let long = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        for other in [short.as_slice(), long.as_slice()] {
            for result in [
                mfi(&h, other, &c, &v, 2),
                mfi(&h, &l, other, &v, 2),
                mfi(&h, &l, &c, other, 2),
            ] {
                assert_eq!(result.len(), h.len());
                assert!(result.iter().all(|x| x.is_nan()));
            }
        }
    }

    #[test]
    fn ad_length_mismatch_returns_zeros() {
        let h = [10.0, 12.0, 11.0];
        let l = [8.0, 9.0, 9.0];
        let c = [9.0, 11.0, 10.0];
        let v = [1000.0, 2000.0, 1500.0];
        let short = [1.0, 2.0];
        let long = [1.0, 2.0, 3.0, 4.0];
        for other in [short.as_slice(), long.as_slice()] {
            for result in [
                ad(&h, other, &c, &v),
                ad(&h, &l, other, &v),
                ad(&h, &l, &c, other),
            ] {
                assert_eq!(result.len(), h.len());
                // AD documents a NaN-free output, so a mismatch yields zeros.
                assert!(result.iter().all(|x| *x == 0.0));
            }
        }
    }

    #[test]
    fn mfi_warmup_shape_after_push_rewrite() {
        let n = 40;
        let p = 14;
        let high: Vec<f64> = (1..=n).map(|i| i as f64 + 0.5).collect();
        let low: Vec<f64> = (1..=n).map(|i| i as f64 - 0.5).collect();
        let close: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let volume: Vec<f64> = vec![1000.0; n];
        let result = mfi(&high, &low, &close, &volume, p);
        assert_eq!(result.len(), n);
        for i in 0..p {
            assert!(result[i].is_nan(), "MFI[{i}] should be NaN");
        }
        for i in p..n {
            assert!(result[i].is_finite(), "MFI[{i}] should be finite");
        }
    }

    #[test]
    fn adosc_zero_warmup_emits_at_bar_zero() {
        // slowperiod == 1 -> warmup == 0, so bar 0 carries a value (both EMAs are
        // seeded from ad[0], so it is exactly 0.0).
        let h = [10.0, 12.0, 11.0];
        let l = [8.0, 9.0, 9.0];
        let c = [9.0, 11.0, 10.0];
        let v = [1000.0, 2000.0, 1500.0];
        let result = adosc(&h, &l, &c, &v, 1, 1);
        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|x| x.is_finite()));
        assert_eq!(result[0], 0.0);
    }

    #[test]
    fn ad_output_length_matches_input() {
        let h = [10.0, 12.0, 11.0, 11.0];
        let l = [8.0, 9.0, 9.0, 11.0];
        let c = [9.0, 11.0, 10.0, 11.0];
        let v = [1000.0, 2000.0, 1500.0, 500.0];
        let result = ad(&h, &l, &c, &v);
        assert_eq!(result.len(), 4);
        // Bar 3 has high == low, so CLV is forced to 0 and the line is flat.
        assert_eq!(result[3], result[2]);
        assert!(ad(&[], &[], &[], &[]).is_empty());
    }

    #[test]
    fn adosc_length_mismatch_returns_nan() {
        let h = [10.0, 12.0, 11.0];
        let l = [8.0, 9.0, 9.0];
        let c = [9.0, 11.0, 10.0];
        let v = [1000.0, 2000.0, 1500.0];
        let short = [1.0, 2.0];
        let long = [1.0, 2.0, 3.0, 4.0];
        for other in [short.as_slice(), long.as_slice()] {
            for result in [
                adosc(&h, other, &c, &v, 2, 3),
                adosc(&h, &l, other, &v, 2, 3),
                adosc(&h, &l, &c, other, 2, 3),
            ] {
                assert_eq!(result.len(), h.len());
                assert!(result.iter().all(|x| x.is_nan()));
            }
        }
    }
}
