/*!
Fuzz target for `ferro_ta_core::statistic::stddev` and `::var`.

Guards the planned rewrite of both kernels from a two-pass-per-window
computation to a rolling Welford accumulator. The invariants asserted here are
exactly the ones a naive `E[x²] - E[x]²` rewrite breaks:

* output length always equals input length, first `timeperiod - 1` slots `NaN`;
* `stddev >= 0` and `var >= 0` wherever finite — a catastrophic-cancellation
  rewrite yields small *negative* variances, and `sqrt` of those is `NaN`;
* no `NaN` is produced from an all-finite window;
* `stddev² ≈ var` when `nbdev == 1.0`;
* agreement with an inline two-pass population variance.
*/

#![no_main]

use ferro_ta_core::statistic;
use libfuzzer_sys::fuzz_target;

/// Loose relative tolerance, applied to **every** well-formed window.
///
/// Deliberately far above any rounding effect. Its job is gross wrongness —
/// wrong window bounds, an off-by-one warmup, `ddof = 1` instead of `ddof = 0`,
/// a bad divisor, a stale accumulator — all of which land at O(1) relative
/// error. Because it is immune to precision drift it needs no guard, so it
/// covers the ill-conditioned windows that [`TIGHT_REL_TOL`] has to skip.
const LOOSE_REL_TOL: f64 = 1e-3;

/// Tight relative tolerance, applied only to well-conditioned windows.
///
/// Set two orders of magnitude below the ~1e-9 relative error that a naive
/// `Σx² − mean²` rewrite introduces, and far above the few ulps a legitimate
/// reassociation of the window sum costs.
const TIGHT_REL_TOL: f64 = 1e-11;

/// Largest `max|x| over the prefix [0..=i]` to `sigma` ratio at which
/// [`TIGHT_REL_TOL`] is applied.
///
/// A *rolling* second moment — Welford/West included, not only the naive
/// sum-of-squares — carries an absolute error of order `eps * p * D²`, where
/// `D` is the largest magnitude the accumulator has ever absorbed. That is a
/// property of the **prefix**, not of the current window: a single large bar
/// early in the series costs precision in every window after it, until the
/// implementation recomputes exactly. Relative to the window's own variance
/// the error is therefore `eps * (D / sigma)² / p`, and at `D / sigma = 128`
/// that is `2.2e-16 * 16384 ≈ 3.6e-12` — under `TIGHT_REL_TOL` with a few
/// times' margin. Past it, a tight relative comparison measures the
/// accumulator's inherited rounding rather than the kernel's correctness.
///
/// This is not a hole in the target: those windows are still covered by
/// [`LOOSE_REL_TOL`], by the sign assertions and by the no-NaN assertion,
/// none of which have any conditioning caveat.
const MAX_DYNAMIC_RANGE: f64 = 128.0;

/// Minimum variance, relative to the window's squared magnitude, for a
/// *relative* comparison to carry information.
///
/// Below this the reference itself is dominated by cancellation in
/// `x - mean`, so a relative disagreement says nothing about the kernel.
const MIN_REL_VARIANCE: f64 = 1e-8;

/// Two-pass population variance (`ddof = 0`), computed independently of the
/// kernel.
fn reference_population_var(window: &[f64]) -> f64 {
    let p = window.len() as f64;
    let mean = window.iter().sum::<f64>() / p;
    window
        .iter()
        .map(|&x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / p
}

fn run(data: &[u8]) {
    if data.len() < 2 {
        return;
    }

    let timeperiod = ((data[0] as usize) % 64) + 1;
    // Fuzz nbdev over (0, 4] rather than pinning it at 1.0.
    let nbdev = 0.25 + (data[1] as f64 / 255.0) * 3.75;

    let float_bytes = &data[2..];
    let n_floats = float_bytes.len() / 8;
    if n_floats == 0 {
        return;
    }

    let real: Vec<f64> = (0..n_floats)
        .map(|i| {
            let chunk: [u8; 8] = float_bytes[i * 8..(i + 1) * 8].try_into().unwrap();
            f64::from_le_bytes(chunk)
        })
        .collect();
    let n = real.len();

    // --- scaled outputs: shape + sign invariants -----------------------
    let sd_scaled = statistic::stddev(&real, timeperiod, nbdev);
    let var_scaled = statistic::var(&real, timeperiod, nbdev);
    assert_eq!(sd_scaled.len(), n, "STDDEV length mismatch");
    assert_eq!(var_scaled.len(), n, "VAR length mismatch");

    // --- unscaled outputs: reference + relationship --------------------
    let sd = statistic::stddev(&real, timeperiod, 1.0);
    let vr = statistic::var(&real, timeperiod, 1.0);
    assert_eq!(sd.len(), n, "STDDEV(nbdev=1) length mismatch");
    assert_eq!(vr.len(), n, "VAR(nbdev=1) length mismatch");

    let warmup = timeperiod.saturating_sub(1);
    for i in 0..n.min(warmup) {
        assert!(sd[i].is_nan(), "STDDEV warmup[{i}] not NaN: {}", sd[i]);
        assert!(vr[i].is_nan(), "VAR warmup[{i}] not NaN: {}", vr[i]);
        assert!(sd_scaled[i].is_nan(), "scaled STDDEV warmup[{i}] not NaN");
        assert!(var_scaled[i].is_nan(), "scaled VAR warmup[{i}] not NaN");
    }

    if n < timeperiod {
        // Degenerate: everything must be NaN, already covered above for the
        // whole vector length since warmup >= n here.
        return;
    }

    // Largest magnitude the kernel's accumulator can have absorbed by bar `i`.
    let mut prefix_max = real[..warmup.min(n)]
        .iter()
        .fold(0.0_f64, |acc, x| acc.max(x.abs()));

    for i in warmup..n {
        prefix_max = prefix_max.max(real[i].abs());

        // Non-negativity, in both the scaled and unscaled outputs.
        for (name, v) in [
            ("stddev", sd[i]),
            ("var", vr[i]),
            ("stddev*nbdev", sd_scaled[i]),
            ("var*nbdev^2", var_scaled[i]),
        ] {
            if v.is_finite() {
                assert!(v >= 0.0, "{name}[{i}] is negative: {v}");
            }
        }

        let window = &real[i + 1 - timeperiod..=i];
        let all_finite = window.iter().all(|x| x.is_finite());
        if !all_finite {
            continue;
        }

        let mean = window.iter().sum::<f64>() / timeperiod as f64;
        // A finite window can still sum to +/-inf, which makes `x - mean`
        // infinite and every downstream value infinite (never NaN). Only the
        // finite-mean case says anything about the kernel's arithmetic.
        if !mean.is_finite() {
            continue;
        }

        // The headline assertion: finite inputs must never produce NaN. This
        // is precisely the `sqrt(negative)` failure mode of a naive
        // sum-of-squares rewrite.
        assert!(
            !sd[i].is_nan(),
            "STDDEV[{i}] is NaN for an all-finite window (mean={mean})"
        );
        assert!(
            !vr[i].is_nan(),
            "VAR[{i}] is NaN for an all-finite window (mean={mean})"
        );

        // stddev^2 == var when nbdev == 1.0.
        if sd[i].is_finite() && vr[i].is_finite() && vr[i] > 0.0 {
            let sq = sd[i] * sd[i];
            let rel = (sq - vr[i]).abs() / vr[i];
            assert!(
                rel <= 1e-9,
                "STDDEV[{i}]^2 ({sq}) disagrees with VAR[{i}] ({}) rel={rel}",
                vr[i]
            );
        }

        // Reference comparison, guarded to the regime where a relative
        // comparison is meaningful.
        let want = reference_population_var(window);
        let window_scale = window.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()));
        if !want.is_finite() || want <= 0.0 || !window_scale.is_finite() {
            continue;
        }
        if want < MIN_REL_VARIANCE * window_scale * window_scale {
            continue;
        }

        assert!(
            vr[i].is_finite(),
            "VAR[{i}] non-finite ({}) where reference is finite ({want})",
            vr[i]
        );

        let sigma = want.sqrt();
        let rel = (vr[i] - want).abs() / want;

        // Tier 1: unconditional.
        assert!(
            rel <= LOOSE_REL_TOL,
            "VAR[{i}] = {} is grossly wrong vs the two-pass reference {want} \
             (rel={rel}, tp={timeperiod})",
            vr[i]
        );

        // Tier 2: only where the accumulator cannot have inherited a large
        // rounding error from the prefix (see MAX_DYNAMIC_RANGE).
        if !prefix_max.is_finite() || prefix_max > MAX_DYNAMIC_RANGE * sigma {
            continue;
        }
        assert!(
            rel <= TIGHT_REL_TOL,
            "VAR[{i}] = {} disagrees with the two-pass reference {want} \
             (rel={rel}, tp={timeperiod}, max|x| over prefix = {prefix_max})",
            vr[i]
        );

        if sigma > 0.0 && sd[i].is_finite() {
            let rel_sd = (sd[i] - sigma).abs() / sigma;
            assert!(
                rel_sd <= TIGHT_REL_TOL,
                "STDDEV[{i}] = {} disagrees with sqrt(reference) {sigma} (rel={rel_sd})",
                sd[i]
            );
        }
    }
}

fuzz_target!(|data: &[u8]| {
    run(data);
});
