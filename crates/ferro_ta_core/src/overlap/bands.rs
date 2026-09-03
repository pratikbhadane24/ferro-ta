//! Bollinger Bands.

use super::{compute_ma_by_type, ma_lookback, MAX_MATYPE};
use crate::rolling::RollingVariance;

/// Compute Bollinger Bands, returning `(upper, middle, lower)`.
///
/// The middle band is a moving average of type `matype` (`0` = SMA, TA-Lib's
/// default); the outer bands are offset from it by `nbdevup` and `nbdevdn`
/// population standard deviations. O(n) on a single [`RollingVariance`], plus
/// one moving-average pass when `matype != 0`.
///
/// # Arguments
/// * `close` - Price series.
/// * `timeperiod` - Moving-average / standard-deviation window (must be >= 1).
/// * `nbdevup` - Standard deviations above the middle band.
/// * `nbdevdn` - Standard deviations below the middle band.
/// * `matype` - Middle-band moving-average type, `0`-`8`. `0` (SMA) is
///   TA-Lib's default and this function's historical behaviour, and is
///   **bit-identical** to the no-`matype` kernel
///   (`bbands_matype0_is_bit_identical_to_the_sma_kernel`). The numbering is
///   **not** TA-Lib-compatible throughout: `0`-`6` and `8` match `TA_MAType`,
///   but **`7` is T3 here where TA-Lib's `7` is MAMA**, and MAMA is not
///   reachable through any `matype` at all — call [`mama`](super::mama) for
///   it. See the [dispatch module docs](super) for the table. A `matype` above
///   `8` returns all-`NaN` of the input length: no panic, and deliberately no
///   silent SMA fallback, so a typo stays distinguishable from a valid
///   request.
///
/// # Returns
/// `(upper, middle, lower)`, each of length `close.len()`.
///
/// # NaN warm-up
///
/// `ma_lookback(timeperiod, matype)` leading `NaN`s in all three vectors,
/// matching `TA_BBANDS_Lookback` (which is `TA_MA_Lookback`; the standard
/// deviation's own `timeperiod - 1` is never the larger of the two). Writing
/// `p1` for `timeperiod - 1`: `p1` for `matype` 0/1/2/5 (SMA, EMA, WMA,
/// TRIMA), `2 * p1` for 3 (DEMA), `3 * p1` for 4 (TEMA), `timeperiod` for 6
/// (KAMA), `6 * p1` for 7/8 (T3). All three vectors are `NaN` when
/// `timeperiod < 1`, `close.len() < timeperiod`, `matype > 8`, or that warm-up
/// runs past the end of the series.
///
/// # The deviation is always about the *SMA*, whatever `matype` is
///
/// The three bands stay index-aligned — `upper[i]`, `middle[i]` and `lower[i]`
/// all derive from the `timeperiod` bars ending at `i` — and the half-width is
/// `nbdev * sigma` over that window whichever MA centres it. But `sigma` is
/// about the window **mean**, not about the selected MA.
///
/// That is what TA-Lib does, and it is worth stating because it is the
/// surprising choice. `ta_BBANDS.c` computes the middle band with
/// `TA_MA(..., optInMAType, ...)`, then the half-width with a
/// `TA_STDDEV(inReal, ..., 1.0, ...)` call that never receives `optInMAType`;
/// `TA_STDDEV` reduces to `TA_INT_VAR`, whose second moment is about
/// `sum / N`. Its `optInMAType == TA_MAType_SMA` shortcut
/// (`TA_INT_stddev_using_precalc_ma` reuses the middle band as that mean) is
/// valid *only* because the two coincide for SMA, which is itself the evidence
/// that the general path centres the deviation on the SMA. Both branches then
/// finish identically: `upper = middle + nbdevup * sigma`,
/// `lower = middle - nbdevdn * sigma`.
///
/// # Numerical contract
///
/// Mean and second moment both come from [`RollingVariance`], the crate's one
/// vetted rolling-Welford accumulator, driven through
/// [`RollingVariance::advance`] so all three of its reseed triggers apply
/// (elapsed `rolling::RESEED_INTERVAL`, non-finite recovery, and lost
/// precision — `mean`/`m2` non-finite over an all-finite window, `m2`
/// collapsed by more than `rolling::MAX_ACCUMULATOR_RANGE` from its peak, or
/// `m2` subnormal). The reported half-width therefore stays within `2^-26`
/// (~1.5e-8) relative of a two-pass recompute over the same window,
/// independent of series length.
///
/// ## Why this replaced a bare rolling Welford
///
/// This kernel used to carry its own copy of the rolling update whose only
/// defence was an `m2 < 0.0` clamp, with **no** reseed of any kind. That is
/// not an accuracy defence, and it failed from entirely finite input:
///
/// * A single large-but-finite bar destroys `m2`'s low-order bits while in the
///   window, and the damage surfaces once it *leaves*, when the recurrence has
///   nothing left to subtract it from. A `1e8` spike on a price-100 series
///   sufficed: bars 6-8 of
///   `bbands([100, 100.5, 101, 1e8, 100.2, 100.3, 100.4, 100.5, 100.6], 3, 2, 2, 0)`
///   came back with `upper == middle == lower`, the clamp having rounded the
///   wreckage up to a confident hard `0.0`.
/// * The clamp is also inert for a `NaN` `m2` (`NaN < 0.0` is false), which
///   the update reaches from finite input by overflowing `m2` to `+inf` and
///   then adding `-inf`. Without a reseed that `NaN` is permanent.
///
/// Switching to [`RollingVariance`] therefore **changes BBANDS output**,
/// deliberately: on well-conditioned input at or near the last ulp
/// (`bbands_matches_the_unreseeded_reference_on_ordinary_data`), and on the
/// pathological inputs above from wrong to right
/// (`bbands_does_not_collapse_after_a_finite_spike_leaves_the_window`).
///
/// The old inline seed carried a proof that its `m2` could never go negative
/// (`delta2 = x - mean_new` equals `delta * (1 - 1/count)` exactly, so every
/// `delta * delta2` term shares `delta`'s sign). That argument and the
/// `debug_assert!` encoding it moved into [`RollingVariance::seed_from`],
/// which is stronger: a *two-pass* recompute whose `m2` is literally a sum of
/// squares, non-negative by construction rather than by an argument about
/// rounding directions.
pub fn bbands(
    close: &[f64],
    timeperiod: usize,
    nbdevup: f64,
    nbdevdn: f64,
    matype: u8,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = close.len();
    let nan = || vec![f64::NAN; n];
    if timeperiod < 1 || n < timeperiod || matype > MAX_MATYPE {
        return (nan(), nan(), nan());
    }
    // `ma_lookback` is never shorter than the standard deviation's own
    // `timeperiod - 1`, so it alone fixes where output starts.
    let start = ma_lookback(timeperiod, matype);
    debug_assert!(start >= timeperiod - 1);
    if start >= n {
        return (nan(), nan(), nan());
    }
    let mut upper = nan();
    let mut middle = nan();
    let mut lower = nan();

    // `matype == 0` reads the centre straight off the accumulator instead of
    // calling `compute_ma_by_type(.., 0)`. Not an optimization: `sma`'s
    // streaming recurrence is not bit-identical to the accumulator's mean, and
    // the default path has to reproduce the pre-`matype` kernel exactly.
    let ma = (matype != 0).then(|| compute_ma_by_type(close, timeperiod, matype));

    let mut acc = RollingVariance::new(&close[..timeperiod]);
    for i in (timeperiod - 1)..n {
        if i >= timeperiod {
            acc.advance(
                close[i],
                close[i - timeperiod],
                &close[i + 1 - timeperiod..=i],
            );
        }
        if i < start {
            continue;
        }
        let center = match &ma {
            Some(series) => series[i],
            None => acc.mean(),
        };
        write_bands(
            center,
            acc.population_var(),
            nbdevup,
            nbdevdn,
            &mut upper[i],
            &mut middle[i],
            &mut lower[i],
        );
    }

    (upper, middle, lower)
}

/// Write one bar of all three bands from the middle-band value `center` and
/// the window's population variance.
///
/// `var` comes from [`RollingVariance::population_var`], which is
/// non-negative whenever it is a number at all — the rolling update clamps
/// `m2` at zero and every reseed recomputes it as a sum of squares — so `sqrt`
/// never sees a negative argument and a `NaN` half-width is reachable only
/// where a two-pass recompute over the same window would also be `NaN`.
#[inline(always)]
fn write_bands(
    center: f64,
    var: f64,
    nbdevup: f64,
    nbdevdn: f64,
    upper: &mut f64,
    middle: &mut f64,
    lower: &mut f64,
) {
    debug_assert!(
        var >= 0.0 || var.is_nan(),
        "population variance must never be negative, got {var}"
    );
    let std = var.sqrt();
    *middle = center;
    *upper = center + nbdevup * std;
    *lower = center - nbdevdn * std;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(upper, middle, lower)`, as `bbands` returns it.
    type Bands = (Vec<f64>, Vec<f64>, Vec<f64>);

    /// The pre-`RollingVariance`, pre-`matype` kernel — hence four arguments.
    /// No reseed of any kind, an `m2 < 0.0` clamp as its only defence: the
    /// *baseline*, not the truth.
    fn reference_bbands(
        close: &[f64],
        timeperiod: usize,
        nbdevup: f64,
        nbdevdn: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut upper = vec![f64::NAN; n];
        let mut middle = vec![f64::NAN; n];
        let mut lower = vec![f64::NAN; n];
        if timeperiod < 1 || n < timeperiod {
            return (upper, middle, lower);
        }
        let p = timeperiod as f64;
        // Seed: incremental (non-rolling) Welford, deliberately unclamped.
        let mut mean = 0.0_f64;
        let mut m2 = 0.0_f64;
        for (k, &x) in close[..timeperiod].iter().enumerate() {
            let delta = x - mean;
            mean += delta / (k + 1) as f64;
            m2 += delta * (x - mean);
        }
        // Takes `std`, not `m2`: the original applied `.max(0.0)` to the
        // seed's variance and *not* to the rolling phase's, and that asymmetry
        // is load-bearing — `f64::max` returns the non-NaN operand, so hoisting
        // the clamp into the rolling phase would silently turn a `NaN` `m2`
        // into `0.0` and hide the very defect this oracle exists to exhibit.
        let mut emit = |i: usize, mean: f64, std: f64| {
            middle[i] = mean;
            upper[i] = mean + nbdevup * std;
            lower[i] = mean - nbdevdn * std;
        };
        emit(timeperiod - 1, mean, (m2 / p).max(0.0).sqrt());
        for i in timeperiod..n {
            let (x_old, x_new) = (close[i - timeperiod], close[i]);
            let delta = x_new - x_old;
            let old_mean = mean;
            mean += delta / p;
            m2 += delta * ((x_new - mean) + (x_old - old_mean));
            // The entire defence of the pre-fix kernel.
            if m2 < 0.0 {
                m2 = 0.0;
            }
            emit(i, mean, (m2 / p).sqrt());
        }
        (upper, middle, lower)
    }

    /// Exact two-pass population standard deviation over `window`.
    fn exact_std(window: &[f64]) -> f64 {
        let p = window.len() as f64;
        let mean = window.iter().sum::<f64>() / p;
        (window.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / p).sqrt()
    }

    /// Assert both half-widths of every full window equal an exact two-pass
    /// population sigma to `rtol` relative. `nbdev` is the multiplier both
    /// outer bands were computed with.
    fn assert_half_widths_exact(close: &[f64], p: usize, nbdev: f64, rtol: f64, label: &str) {
        let (upper, middle, lower) = bbands(close, p, nbdev, nbdev, 0);
        for i in (p - 1)..close.len() {
            let want = exact_std(&close[i + 1 - p..=i]);
            for (side, got) in [
                ("upper", (upper[i] - middle[i]) / nbdev),
                ("lower", (middle[i] - lower[i]) / nbdev),
            ] {
                if want.is_infinite() {
                    assert!(
                        got.is_infinite(),
                        "{label} {side} i={i}: want inf, got {got}"
                    );
                    continue;
                }
                assert!(
                    (got - want).abs() <= rtol * want.abs().max(1e-12),
                    "{label} {side} i={i}: got={got} want={want}"
                );
            }
        }
    }

    /// Series that stress the accumulator: a `1e8` spike on a price-100
    /// series, an overflow-then-cancel pair, subnormals, a trend, a constant.
    fn adversarial_series() -> Vec<Vec<f64>> {
        vec![
            oracle_close(400),
            vec![100.0, 100.5, 101.0, 1e8, 100.2, 100.3, 100.4, 100.5, 100.6],
            vec![100.0, 1e200, 100.5, 100.6, 100.7, 100.8],
            vec![1e300, -1e300, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![f64::MIN_POSITIVE, 0.0, -f64::MIN_POSITIVE, 0.0, 1e-320],
            (0..64).map(|i| 5e-324 * (i % 3) as f64).collect(),
            (0..300).map(|i| 50.0 + i as f64 * 0.37).collect(),
            vec![7.5; 300],
            vec![42.0; 80],
            {
                let mut v = oracle_close(150);
                v[0] = f64::NAN;
                v[7] = f64::NAN;
                v
            },
            oracle_close(0),
            oracle_close(1),
            oracle_close(5),
        ]
    }

    /// Deterministic pseudo-random price walk around 100.
    fn oracle_close(n: usize) -> Vec<f64> {
        let mut state = 0x1234_5678_9abc_def0_u64;
        let mut price = 100.0_f64;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                price += ((state >> 33) % 21) as f64 * 0.05 - 0.5;
                price
            })
            .collect()
    }

    #[test]
    fn bbands_varying_prices() {
        // A constant window has std = 0, so all three bands coincide.
        let (u, m, l) = bbands(&[2.0; 5], 3, 2.0, 2.0, 0);
        assert!([u[2], m[2], l[2]].iter().all(|b| (b - 2.0).abs() < 1e-10));

        // Hand-computed: every window of three consecutive integers has
        // pop_var = 2/3, and the means are 2, 3, 4.
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (upper, middle, lower) = bbands(&prices, 3, 2.0, 2.0, 0);
        assert!(middle[0].is_nan() && middle[1].is_nan(), "warm-up");
        let std = (2.0_f64 / 3.0).sqrt();
        for (i, mean) in [(2usize, 2.0), (3, 3.0), (4, 4.0)] {
            assert!((middle[i] - mean).abs() < 1e-10, "mean at {i}");
            assert!(
                (upper[i] - (mean + 2.0 * std)).abs() < 1e-10,
                "upper at {i}"
            );
            assert!(
                (lower[i] - (mean - 2.0 * std)).abs() < 1e-10,
                "lower at {i}"
            );
        }
    }

    #[test]
    fn bbands_numerical_stability() {
        // Large offset with tiny variation — where the naive `sum_sq` formula
        // suffers catastrophic cancellation. At scale 1e12 the f64 absolute
        // precision is ~2.2e-4, so 1e-3 is one order of headroom.
        let prices: Vec<f64> = (0..100).map(|i| 1e12 + (i as f64) * 0.01).collect();
        let (upper, middle, lower) = bbands(&prices, 20, 2.0, 2.0, 0);
        for i in 19..100 {
            let want: f64 = prices[i - 19..=i].iter().sum::<f64>() / 20.0;
            assert!(
                (middle[i] - want).abs() < 1e-3,
                "mean at {i}: got {} want {want}",
                middle[i]
            );
            assert!(
                upper[i] >= middle[i] && lower[i] <= middle[i],
                "order at {i}"
            );
        }
    }

    #[test]
    fn bbands_edge_cases() {
        // timeperiod == 1: std = 0 on every bar, so all three bands are price.
        let prices = vec![10.0, 20.0, 30.0];
        let (upper, middle, lower) = bbands(&prices, 1, 2.0, 2.0, 0);
        for i in 0..3 {
            for band in [middle[i], upper[i], lower[i]] {
                assert!((band - prices[i]).abs() < 1e-10, "at {i}");
            }
        }
        // Input shorter than timeperiod, and a zero timeperiod: all NaN.
        for (u, m, l) in [
            bbands(&[1.0, 2.0], 5, 2.0, 2.0, 0),
            bbands(&[1.0, 2.0], 0, 2.0, 2.0, 0),
        ] {
            assert!(u.iter().chain(&m).chain(&l).all(|v| v.is_nan()));
        }
    }

    #[test]
    fn bbands_seed_variance_is_never_negative() {
        // The seed is `RollingVariance::seed_from`, a two-pass recompute whose
        // `m2` is a sum of squares, so the first band value can never be NaN
        // for finite input. Shapes most likely to break it: constant windows,
        // a huge offset with tiny variation, an exactly-symmetric window, and
        // subnormal / near-overflow magnitudes.
        let cases: Vec<Vec<f64>> = vec![
            vec![1.0; 40],
            vec![0.0; 40],
            (0..40).map(|i| 1e12 + (i as f64) * 1e-6).collect(),
            (0..40)
                .map(|i| if i % 2 == 0 { -1.0 } else { 1.0 })
                .collect(),
            (0..40).map(|i| (i as f64) * 5e-324).collect(),
            (0..40).map(|i| 1e300 * (1.0 + i as f64 * 1e-15)).collect(),
        ];
        for (c, close) in cases.iter().enumerate() {
            for &p in &[1usize, 2, 5, 20, 40] {
                let (upper, middle, lower) = bbands(close, p, 2.0, 2.0, 0);
                let seed = p - 1;
                assert!(
                    upper[seed] >= middle[seed] && lower[seed] <= middle[seed],
                    "case {c} p={p}: seed band is NaN or misordered ({}, {}, {})",
                    upper[seed],
                    middle[seed],
                    lower[seed],
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // The `RollingVariance` migration: agreement where the old kernel was
    // right, disagreement where it was wrong.
    // -----------------------------------------------------------------------

    /// The unreseeded reference is the *baseline*, not the truth, so this
    /// bounds the divergence in two places rather than asserting one number.
    ///
    /// * **Gate-shaped configurations** — at most 2000 bars with
    ///   `timeperiod >= 5`, the shape of every
    ///   `tests/integration/test_vs_talib.py` BBANDS case and of
    ///   `benchmarks/test_accuracy.py` — agree to `1e-13` **relative**.
    ///   Measured worst case `2.5e-14`, ~110 ulps of a price-100 band, i.e.
    ///   `2.5e-12` *absolute*: the gate's `atol = 1e-6` has four orders of
    ///   headroom, so this migration cannot move it.
    /// * **The whole matrix**, `timeperiod = 2` over 12000 bars included
    ///   (measured worst case `8.4e-8` relative), is where the divergence
    ///   actually lives — and it lives there because the *reference* drifted.
    ///   The second loop shows that directly: it scores both against an exact
    ///   two-pass recompute of the same window and asserts the new kernel is
    ///   never meaningfully further from it. Both sit at ~2e-5 relative at
    ///   `timeperiod = 2` on `index_level`, the crate's inherent
    ///   rolling-Welford floor at `mean / sigma ~ 2e7`
    ///   (`rolling::RESEED_INTERVAL` bounds drift by advance count, not by
    ///   conditioning), so the comparison there is between the two kernels.
    #[test]
    fn bbands_matches_the_unreseeded_reference_on_ordinary_data() {
        // Gate-shaped, everything else, and the allowed excess distance from
        // an exact recompute, all relative.
        const GATE_RTOL: f64 = 1e-13;
        const WIDE_RTOL: f64 = 1e-7;
        const EXCESS_RTOL: f64 = 1e-8;

        let trending: Vec<f64> = (0..3000).map(|i| 50.0 + i as f64 * 0.37).collect();
        let index_level: Vec<f64> = (0..3000)
            .map(|i| 100_000.0 + ((i % 13) as f64) * 0.01)
            .collect();
        let regimes: Vec<f64> = (0..3000)
            .map(|i| {
                let amp = if (1000..2000).contains(&i) { 2.0 } else { 0.2 };
                100.0 + amp * ((i % 7) as f64 - 3.0)
            })
            .collect();
        let series: Vec<(&str, Vec<f64>)> = vec![
            ("walk_500", oracle_close(500)),
            ("walk_2000", oracle_close(2000)),
            ("walk_12000", oracle_close(12_000)),
            ("trending", trending),
            ("constant", vec![7.5; 500]),
            ("index_level", index_level),
            ("regimes", regimes),
        ];
        for (name, close) in &series {
            for &p in &[1usize, 2, 5, 14, 20, 50, 200] {
                if close.len() < p {
                    continue;
                }
                let (gu, gm, gl) = bbands(close, p, 2.0, 2.5, 0);
                let (wu, wm, wl) = reference_bbands(close, p, 2.0, 2.5);
                let rtol = if close.len() <= 2000 && p >= 5 {
                    GATE_RTOL
                } else {
                    WIDE_RTOL
                };
                for (label, got, want) in [
                    ("upper", &gu, &wu),
                    ("middle", &gm, &wm),
                    ("lower", &gl, &wl),
                ] {
                    for i in 0..close.len() {
                        let ok = if want[i].is_nan() {
                            got[i].is_nan()
                        } else {
                            (got[i] - want[i]).abs() <= rtol * want[i].abs().max(f64::MIN_POSITIVE)
                        };
                        assert!(
                            ok,
                            "{name} p={p} {label} i={i}: got={} want={} (rtol {rtol:e})",
                            got[i], want[i]
                        );
                    }
                }
                // Direction of the divergence: against an exact two-pass
                // recompute the new kernel is never meaningfully worse than the
                // reference, and on the pathological inputs below it is many
                // orders better.
                for i in (p - 1)..close.len() {
                    let exact = exact_std(&close[i + 1 - p..=i]);
                    if !exact.is_finite() || exact <= 0.0 {
                        continue;
                    }
                    let err_new = ((gu[i] - gm[i]) / 2.0 - exact).abs() / exact;
                    let err_ref = ((wu[i] - wm[i]) / 2.0 - exact).abs() / exact;
                    assert!(
                        err_new <= err_ref + EXCESS_RTOL,
                        "{name} p={p} i={i}: new {err_new:e} vs reference {err_ref:e}"
                    );
                }
            }
        }
    }

    /// The reproducer: a `1e8` spike on a price-100 series used to leave bars
    /// 6-8 with `upper == middle == lower` (100.30000000074506 /
    /// 100.40000000074505 / 100.50000000074505), the clamp having converted
    /// total precision loss into a confident hard zero.
    #[test]
    fn bbands_does_not_collapse_after_a_finite_spike_leaves_the_window() {
        let close = [100.0, 100.5, 101.0, 1e8, 100.2, 100.3, 100.4, 100.5, 100.6];
        let p = 3;
        let (upper, middle, lower) = bbands(&close, p, 2.0, 2.0, 0);
        let (ru, rm, rl) = reference_bbands(&close, p, 2.0, 2.0);

        for i in 6..=8 {
            // Premise, not decoration: the old kernel really did collapse all
            // three bands onto each other here.
            assert!(ru[i] == rm[i] && rl[i] == rm[i], "premise broken at {i}");
            assert!(
                upper[i] > middle[i] && middle[i] > lower[i],
                "bands still collapsed at {i}: {} / {} / {}",
                upper[i],
                middle[i],
                lower[i],
            );
        }

        assert_half_widths_exact(&close, p, 2.0, 1e-9, "1e8 spike");
    }

    /// The whole spike-magnitude sweep, not just `1e8`: the pre-fix error was
    /// non-monotone in it — the signature of precision loss, not a logic error.
    #[test]
    fn bbands_half_width_is_exact_across_the_spike_magnitude_sweep() {
        for exp in 2..=150 {
            let spike = 10f64.powi(exp);
            let close = [
                100.0, 100.5, 101.0, spike, 100.2, 100.3, 100.4, 100.5, 100.6,
            ];
            assert_half_widths_exact(&close, 3, 1.0, 1e-9, &format!("spike 1e{exp}"));
        }
    }

    /// `m2` overflows to `+inf`, the next slide adds `-inf` and leaves `NaN`,
    /// and the clamp is inert for it, so the old kernel emitted `NaN` forever.
    #[test]
    fn bbands_recovers_from_an_m2_overflow_to_nan() {
        let mut close = vec![1e300, -1e300];
        close.extend((0..30).map(|k| 100.0 + 0.5 * k as f64));
        let p = 3;
        let (upper, middle, lower) = bbands(&close, p, 2.0, 2.0, 0);
        let (ru, _, _) = reference_bbands(&close, p, 2.0, 2.0);

        // Premise: the reference reports `+inf` on the overflowing seed and is
        // then permanently `NaN` — `m2` took `+inf` then `-inf`, and the clamp
        // is inert for a `NaN`. Its *mean* is wrecked too (`0.0` at bar 4,
        // whose window is `[100.0, 100.5, 101.0]`).
        assert!(ru[p - 1].is_infinite(), "premise: seed is {}", ru[p - 1]);
        assert!(
            ru[p..].iter().all(|v| v.is_nan()),
            "premise: reference did not go permanently NaN"
        );
        // The fixed kernel is finite and correct from the first spike-free
        // window (bar 4: [100.0, 100.5, 101.0], sigma^2 = 1/6).
        for i in 4..close.len() {
            let bands = [upper[i], middle[i], lower[i]];
            assert!(bands.iter().all(|b| b.is_finite()), "i={i}: {bands:?}");
        }
        assert_half_widths_exact(&close[2..], p, 2.0, 1e-12, "post-overflow");
    }

    /// A mid-series `NaN` must corrupt exactly the `timeperiod` windows that
    /// contain it and then recover, as a two-pass kernel would.
    #[test]
    fn bbands_recovers_once_a_nan_leaves_the_window() {
        let mut close: Vec<f64> = (0..40).map(|i| 100.0 + (i % 3) as f64).collect();
        close[15] = f64::NAN;
        let p = 5;
        let (upper, middle, lower) = bbands(&close, p, 2.0, 2.0, 0);
        for i in (p - 1)..close.len() {
            let window = &close[i + 1 - p..=i];
            if window.iter().any(|x| x.is_nan()) {
                assert!(middle[i].is_nan(), "i={i}: expected NaN middle");
                continue;
            }
            let mean = window.iter().sum::<f64>() / p as f64;
            assert!((middle[i] - mean).abs() < 1e-12, "i={i} mean={}", middle[i]);
            let want = exact_std(window);
            for got in [(upper[i] - middle[i]) / 2.0, (middle[i] - lower[i]) / 2.0] {
                assert!(
                    (got - want).abs() <= 1e-12 * want.max(1e-12),
                    "i={i} half-width {got} vs {want}"
                );
            }
        }
    }

    /// Values around `1e-161` are ordinary `f64`s whose squared deviations are
    /// **subnormal**, so `m2` has only a handful of significant bits;
    /// `RollingVariance`'s subnormal trigger forces an exact recompute.
    #[test]
    fn bbands_survives_a_subnormal_second_moment() {
        let base = 1.7e-161;
        let close: Vec<f64> = (0..40)
            .map(|i| base * (1.0 + 0.25 * ((i % 5) as f64)))
            .collect();
        assert!(
            exact_std(&close[..2]) > 0.0,
            "premise broken: sigma is zero"
        );
        for &p in &[2usize, 3, 7] {
            assert_half_widths_exact(&close, p, 1.0, 1e-9, &format!("subnormal p={p}"));
        }
    }

    // -----------------------------------------------------------------------
    // `matype`.
    // -----------------------------------------------------------------------

    /// The Task-1 (`RollingVariance`, no `matype`) kernel, restated. `matype
    /// = 0` must be **bit-identical** to it, hence `to_bits()` below rather
    /// than a tolerance.
    fn reference_bbands_sma(close: &[f64], timeperiod: usize, nbdevup: f64, nbdevdn: f64) -> Bands {
        let n = close.len();
        let mut upper = vec![f64::NAN; n];
        let mut middle = vec![f64::NAN; n];
        let mut lower = vec![f64::NAN; n];
        if timeperiod >= 1 && n >= timeperiod {
            let mut acc = RollingVariance::new(&close[..timeperiod]);
            for i in (timeperiod - 1)..n {
                if i >= timeperiod {
                    acc.advance(
                        close[i],
                        close[i - timeperiod],
                        &close[i + 1 - timeperiod..=i],
                    );
                }
                let mean = acc.mean();
                let std = acc.population_var().sqrt();
                middle[i] = mean;
                upper[i] = mean + nbdevup * std;
                lower[i] = mean - nbdevdn * std;
            }
        }
        (upper, middle, lower)
    }

    /// All three vectors equal, bit for bit.
    fn assert_bands_bits(got: &Bands, want: &Bands, ctx: &str) {
        for (label, g, w) in [
            ("upper", &got.0, &want.0),
            ("middle", &got.1, &want.1),
            ("lower", &got.2, &want.2),
        ] {
            assert_eq!(g.len(), w.len(), "{label} {ctx}: length");
            for (i, (&a, &b)) in g.iter().zip(w.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "{label} {ctx} i={i}: {a} vs {b}");
            }
        }
    }

    /// Two bit-identity contracts in one sweep: `matype = 0` reproduces the
    /// no-`matype` kernel exactly, and `8` is indistinguishable from `7`.
    #[test]
    fn bbands_matype0_is_bit_identical_to_the_sma_kernel() {
        for (si, close) in adversarial_series().iter().enumerate() {
            for p in [0usize, 1, 2, 3, 5, 14, 500] {
                for (up, dn) in [(2.0, 2.0), (1.0, 3.0), (0.0, 0.0)] {
                    let c = format!("series={si} p={p} up={up} dn={dn}");
                    let got = bbands(close, p, up, dn, 0);
                    assert_bands_bits(&got, &reference_bbands_sma(close, p, up, dn), &c);
                    let t3 = bbands(close, p, up, dn, 7);
                    assert_bands_bits(&bbands(close, p, up, dn, 8), &t3, &format!("T3 {c}"));
                }
            }
        }
    }

    #[test]
    fn bbands_out_of_range_matype_returns_all_nan_of_input_length() {
        let close = oracle_close(200);
        for matype in [MAX_MATYPE + 1, 9, 10, 99, u8::MAX] {
            let (u, m, l) = bbands(&close, 20, 2.0, 2.0, matype);
            for (label, v) in [("upper", &u), ("middle", &m), ("lower", &l)] {
                assert_eq!(v.len(), close.len(), "matype={matype} {label}: length");
                assert!(v.iter().all(|x| x.is_nan()), "matype={matype} {label}");
            }
        }
        // No silent SMA fallback: `0` really does produce values.
        let (u0, _, _) = bbands(&close, 20, 2.0, 2.0, 0);
        assert!(u0.iter().any(|x| x.is_finite()));
    }

    /// Warm-up follows `ma_lookback(timeperiod, matype)` in all three vectors,
    /// which also pins index alignment: the first non-`NaN` bar is the same
    /// for upper, middle and lower.
    #[test]
    fn bbands_warmup_follows_ma_lookback_per_matype() {
        let close = oracle_close(600);
        for p in [2usize, 3, 5, 14] {
            for matype in 0..=MAX_MATYPE {
                let (u, m, l) = bbands(&close, p, 2.0, 2.0, matype);
                let lookback = ma_lookback(p, matype);
                for (label, v) in [("upper", &u), ("middle", &m), ("lower", &l)] {
                    let first = v.iter().position(|x| !x.is_nan());
                    assert_eq!(
                        first,
                        Some(lookback),
                        "p={p} matype={matype} {label}: first value at {first:?}"
                    );
                }
            }
        }
        // A warm-up past the end of the series is all-NaN, not a panic:
        // T3 at p = 5 needs 6 * 4 = 24 bars.
        let (u, _, _) = bbands(&oracle_close(10), 5, 2.0, 2.0, 7);
        assert!(u.iter().all(|x| x.is_nan()));
    }

    /// The middle band is exactly `ma(close, timeperiod, matype)` on every
    /// emitted bar, and the half-width is the SMA-centred sigma over the same
    /// window — *not* the deviation about the selected MA, which is the TA-Lib
    /// behaviour documented on `bbands`.
    #[test]
    fn bbands_centres_on_the_selected_ma_but_deviates_about_the_sma() {
        let close = oracle_close(600);
        let p = 14usize;
        for matype in 1..=MAX_MATYPE {
            let (upper, middle, lower) = bbands(&close, p, 2.0, 3.0, matype);
            let ma = compute_ma_by_type(&close, p, matype);
            let start = ma_lookback(p, matype);
            for i in start..close.len() {
                assert_eq!(
                    middle[i].to_bits(),
                    ma[i].to_bits(),
                    "matype={matype} i={i}: middle is not the selected MA"
                );
                // Half-widths are `nbdev * sigma_SMA`, not the deviation about
                // the selected MA.
                let sigma = exact_std(&close[i + 1 - p..=i]);
                for (side, got, nbdev) in [
                    ("upper", upper[i] - middle[i], 2.0),
                    ("lower", middle[i] - lower[i], 3.0),
                ] {
                    assert!(
                        (got - nbdev * sigma).abs() <= 1e-9 * sigma.max(1e-12),
                        "matype={matype} i={i} {side}: {got} vs {}",
                        nbdev * sigma
                    );
                }
            }
        }
    }

    /// `upper >= middle >= lower` wherever all three are finite, for every
    /// `matype` and on adversarial input — the `fuzz_bbands` invariant, pinned
    /// here so a regression shows up in `cargo test` too. It holds because
    /// both half-widths are `nbdev * sigma` with `sigma >= 0`.
    #[test]
    fn bbands_band_ordering_holds_for_every_matype() {
        for (si, close) in adversarial_series().iter().enumerate() {
            for matype in 0..=MAX_MATYPE {
                for &p in &[1usize, 2, 3, 5, 14, 20] {
                    let (u, m, l) = bbands(close, p, 1.3, 3.7, matype);
                    for i in 0..close.len() {
                        let all_finite = u[i].is_finite() && m[i].is_finite() && l[i].is_finite();
                        assert!(
                            !all_finite || (u[i] >= m[i] && m[i] >= l[i]),
                            "series={si} matype={matype} p={p} i={i}: {} / {} / {}",
                            u[i],
                            m[i],
                            l[i]
                        );
                    }
                }
            }
        }
    }
}
