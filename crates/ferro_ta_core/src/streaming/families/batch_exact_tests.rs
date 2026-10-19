//! Batch-side regression tests for the rolling-sum fixes.
//!
//! * Drift: on a 50k-bar random walk near 50,000 every formerly drifting
//!   batch kernel equals an exact two-pass computation on its trailing
//!   window, at every bar, within `1e-9 * max(1, |exact|)`.
//! * Zero volume: after fractional volumes, an all-zero-volume window takes
//!   each kernel's zero-denominator rule (NaN, or MFI's 0) instead of the
//!   rounding residue a running sum leaves behind.
use super::drift_tests::Data;

const EXACT_BARS: usize = 50_000;
const REL_TOL: f64 = 1e-9;

/// Mean taken about `w[0]`, so a flat window's mean is exactly its value
/// (a plain `sum / n` at 50,000 can miss it by an ulp and fake a variance).
fn mean(w: &[f64]) -> f64 {
    let r = w[0];
    r + w.iter().map(|v| v - r).sum::<f64>() / w.len() as f64
}

/// Two-pass population variance.
fn exact_var(w: &[f64]) -> f64 {
    let m = mean(w);
    w.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / w.len() as f64
}

/// Two-pass Pearson correlation; NaN on a zero-variance side.
fn exact_correl(x: &[f64], y: &[f64]) -> f64 {
    let (mx, my) = (mean(x), mean(y));
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        let (dx, dy) = (a - mx, b - my);
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    if sxx > 0.0 && syy > 0.0 {
        sxy / (sxx * syy).sqrt()
    } else {
        f64::NAN
    }
}

/// Two-pass least-squares `(slope, intercept)` of `y` on `0..p`.
fn exact_linreg(y: &[f64]) -> (f64, f64) {
    let x_bar = (y.len() - 1) as f64 / 2.0;
    let y_bar = mean(y);
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (j, v) in y.iter().enumerate() {
        let dx = j as f64 - x_bar;
        sxy += dx * (v - y_bar);
        sxx += dx * dx;
    }
    let slope = sxy / sxx;
    (slope, y_bar - slope * x_bar)
}

/// `mean(fast window) - mean(slow window)` both ending at `src[i]`, taken
/// about `src[i]` so the difference carries no price-level rounding.
fn exact_sma_diff(src: &[f64], i: usize, fast: usize, slow: usize) -> f64 {
    let r = src[i];
    let shifted = |p: usize| src[i + 1 - p..=i].iter().map(|v| v - r).sum::<f64>() / p as f64;
    shifted(fast) - shifted(slow)
}

/// First bar from `first` where `batch` misses `exact(i)` by more than
/// `REL_TOL * max(1, |exact|)` (or disagrees on NaN), plus the worst gap.
fn check_every_bar(
    fails: &mut Vec<String>,
    name: &str,
    batch: &[f64],
    first: usize,
    exact: impl Fn(usize) -> f64,
) {
    let mut worst = (0.0_f64, None);
    for (i, &got) in batch.iter().enumerate().skip(first) {
        let want = exact(i);
        let gap = if got.is_nan() && want.is_nan() {
            0.0
        } else if got.is_nan() || want.is_nan() {
            f64::INFINITY
        } else {
            (got - want).abs() / want.abs().max(1.0)
        };
        if gap > worst.0 {
            worst = (gap, Some((i, got, want)));
        }
    }
    if let (gap, Some((i, got, want))) = worst {
        if gap > REL_TOL {
            fails.push(format!(
                "{name}: worst rel gap {gap:e} at bar {i} (batch {got}, exact {want})"
            ));
        }
    }
}

fn assert_no_failures(fails: &[String]) {
    assert!(fails.is_empty(), "\n{}", fails.join("\n"));
}

/// Every formerly drifting kernel against its two-pass reference on `d`.
fn check_drift_kernels(d: &Data) -> Vec<String> {
    let mut fails = Vec::new();
    let (c, o) = (&d.close, &d.other);
    let win = |s: &'_ [f64], i: usize, p: usize| -> Vec<f64> { s[i + 1 - p..=i].to_vec() };

    for p in [2, 5, 30] {
        check_every_bar(
            &mut fails,
            "correl",
            &crate::statistic::correl(c, o, p),
            p - 1,
            |i| exact_correl(&win(c, i, p), &win(o, i, p)),
        );
        check_every_bar(
            &mut fails,
            "var",
            &crate::statistic::var(c, p, 1.0),
            p - 1,
            |i| exact_var(&win(c, i, p)),
        );
        check_every_bar(
            &mut fails,
            "stddev",
            &crate::statistic::stddev(c, p, 1.0),
            p - 1,
            |i| exact_var(&win(c, i, p)).sqrt(),
        );
        let bbp = crate::extended::bbpercent(c, p, 2.0, 2.0);
        check_every_bar(&mut fails, "bbpercent", &bbp, p - 1, |i| {
            let w = win(c, i, p);
            let sd = exact_var(&w).sqrt();
            // `close - mean` about `close` itself, free of price-level rounding.
            let dev = -w.iter().map(|v| v - c[i]).sum::<f64>() / p as f64;
            (dev + 2.0 * sd) / (4.0 * sd)
        });
    }

    let p = 14;
    let slope = |i: usize| exact_linreg(&win(c, i, p)).0;
    let intercept = |i: usize| exact_linreg(&win(c, i, p)).1;
    let st = crate::statistic::linearreg_slope(c, p);
    check_every_bar(&mut fails, "linearreg_slope", &st, p - 1, slope);
    let ang = crate::statistic::linearreg_angle(c, p);
    check_every_bar(&mut fails, "linearreg_angle", &ang, p - 1, |i| {
        slope(i).atan().to_degrees()
    });
    let ic = crate::statistic::linearreg_intercept(c, p);
    check_every_bar(&mut fails, "linearreg_intercept", &ic, p - 1, intercept);
    let lr = crate::statistic::linearreg(c, p);
    check_every_bar(&mut fails, "linearreg", &lr, p - 1, |i| {
        intercept(i) + slope(i) * (p - 1) as f64
    });
    let tsf = crate::statistic::tsf(c, p);
    check_every_bar(&mut fails, "tsf", &tsf, p - 1, |i| {
        intercept(i) + slope(i) * p as f64
    });

    let median: Vec<f64> = d
        .high
        .iter()
        .zip(&d.low)
        .map(|(h, l)| (h + l) / 2.0)
        .collect();
    let ao = crate::extended::ao(&d.high, &d.low, 5, 34);
    check_every_bar(&mut fails, "ao", &ao, 33, |i| {
        exact_sma_diff(&median, i, 5, 34)
    });
    let po = crate::extended::po(c, 10, 21);
    check_every_bar(&mut fails, "po", &po, 20, |i| exact_sma_diff(c, i, 10, 21));
    let (p, shift) = (20, 20 / 2 + 1);
    check_every_bar(&mut fails, "dpo", &crate::extended::dpo(c, p), p - 1, |i| {
        // exact_sma_diff with a one-bar "fast window" placed `shift` back.
        let r = c[i];
        (c[i - shift] - r) - c[i + 1 - p..=i].iter().map(|v| v - r).sum::<f64>() / p as f64
    });
    fails
}

#[test]
fn drift_kernels_equal_exact_two_pass_on_long_high_priced_walk() {
    assert_no_failures(&check_drift_kernels(&Data::with_len(EXACT_BARS)));
}

/// Bars per cycle of the flat-stretch walk, and the flat run in each.
const FLAT_EVERY: usize = 2_000;
const FLAT_RUN: usize = 200;

/// The walk with every column frozen for `FLAT_RUN` bars out of each
/// `FLAT_EVERY` (longer than every window), so windows go exactly flat
/// (zero variance) and then leave the flat run again.
fn flat_stretch_data() -> Data {
    let mut d = Data::with_len(EXACT_BARS);
    let freeze = |col: &mut Vec<f64>| {
        for i in 1..col.len() {
            if i % FLAT_EVERY > FLAT_EVERY - FLAT_RUN {
                col[i] = col[i - 1];
            }
        }
    };
    for col in [
        &mut d.open,
        &mut d.high,
        &mut d.low,
        &mut d.close,
        &mut d.volume,
        &mut d.other,
    ] {
        freeze(col);
    }
    d
}

#[test]
fn drift_kernels_equal_exact_two_pass_through_flat_stretches() {
    assert_no_failures(&check_drift_kernels(&flat_stretch_data()));
}

// ---------------------------------------------------------------------------
// Zero-volume windows
// ---------------------------------------------------------------------------

const ZERO_BARS: usize = 20_000;
const ZERO_EVERY: usize = 997;
const ZERO_RUN: usize = 80;

/// The high-priced walk with fractional volumes and all-zero-volume runs
/// (`i % 997 < 80`), longer than every window below.
fn zero_volume_data() -> Data {
    let mut d = Data::with_len(ZERO_BARS);
    for (i, v) in d.volume.iter_mut().enumerate() {
        *v = if i % ZERO_EVERY < ZERO_RUN {
            0.0
        } else {
            *v * 0.37
        };
    }
    d
}

fn all_zero(v: &[f64], i: usize, p: usize) -> bool {
    v[i + 1 - p..=i].iter().all(|x| *x == 0.0)
}

#[test]
fn zero_volume_windows_take_the_zero_denominator_rule() {
    let d = zero_volume_data();
    let mut fails = Vec::new();
    let v = &d.volume;
    let (h, l, c) = (&d.high, &d.low, &d.close);
    let mut zero_windows = 0;

    let p = 20;
    let cmf = crate::extended::cmf(h, l, c, v, p);
    check_every_bar(&mut fails, "cmf", &cmf, p - 1, |i| {
        let (mut mfv, mut vol) = (0.0, 0.0);
        for j in i + 1 - p..=i {
            let hl = h[j] - l[j];
            let clv = if hl != 0.0 {
                ((c[j] - l[j]) - (h[j] - c[j])) / hl
            } else {
                0.0
            };
            mfv += clv * v[j];
            vol += v[j];
        }
        if vol != 0.0 {
            mfv / vol
        } else {
            f64::NAN
        }
    });
    for i in p - 1..ZERO_BARS {
        if all_zero(v, i, p) {
            zero_windows += 1;
            if !cmf[i].is_nan() {
                fails.push(format!("cmf bar {i}: zero-volume window gave {}", cmf[i]));
                break;
            }
        }
    }
    assert!(zero_windows > 0);

    let rvol = crate::extended::rvol(v, p);
    check_every_bar(&mut fails, "rvol", &rvol, p - 1, |i| {
        let m = mean(&v[i + 1 - p..=i]);
        if m != 0.0 {
            v[i] / m
        } else {
            f64::NAN
        }
    });

    let (fast, slow) = (5, 10);
    let volosc = crate::extended::volosc(v, fast, slow);
    check_every_bar(&mut fails, "volosc", &volosc, slow - 1, |i| {
        let (f, s) = (mean(&v[i + 1 - fast..=i]), mean(&v[i + 1 - slow..=i]));
        if s != 0.0 {
            100.0 * (f - s) / s
        } else {
            f64::NAN
        }
    });

    let p = 14;
    let mfi = crate::volume::mfi(h, l, c, v, p);
    let tp: Vec<f64> = (0..ZERO_BARS).map(|i| (h[i] + l[i] + c[i]) / 3.0).collect();
    check_every_bar(&mut fails, "mfi", &mfi, p, |i| {
        let (mut pos, mut neg) = (0.0, 0.0);
        for j in i + 1 - p..=i {
            if tp[j] > tp[j - 1] {
                pos += tp[j] * v[j];
            } else if tp[j] < tp[j - 1] {
                neg += tp[j] * v[j];
            }
        }
        if pos + neg == 0.0 {
            0.0
        } else {
            100.0 * pos / (pos + neg)
        }
    });

    // Prefix-sum kernels: a zero window is exactly 0 already (checked so a
    // later rewrite to running sums cannot regress it).
    let p = 20;
    let vwma = crate::extended::vwma(c, v, p);
    let vwap = crate::extended::vwap(h, l, c, v, p);
    for i in p - 1..ZERO_BARS {
        if all_zero(v, i, p) {
            assert!(vwma[i].is_nan(), "vwma bar {i}: {}", vwma[i]);
            assert!(vwap[i].is_nan(), "vwap bar {i}: {}", vwap[i]);
        }
    }
    assert_no_failures(&fails);
}
