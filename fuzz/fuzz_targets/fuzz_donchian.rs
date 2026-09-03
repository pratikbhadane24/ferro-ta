/*!
Fuzz target for `ferro_ta_core::extended::donchian`.

This is the guard for the planned rewrite of the monotonic-deque backend in
`crates/ferro_ta_core/src/rolling.rs` (reached via `math::sliding_max` /
`math::sliding_min`). Getting the deque's *exact* comparison semantics into the
naive reference is the whole deliverable, so they are restated here:

* the max deque pops the back while `vals[back] <= x`, the min deque while
  `vals[back] >= x`. `<=`/`>=` rather than `<`/`>` means an incoming equal
  value pops the whole monotonic run including the front, so among **equal**
  extremes the **most recent** bar wins;
* a `NaN` satisfies neither predicate. It is therefore never popped, and it can
  legitimately surface as the window extreme. The reference below reproduces
  that; it deliberately does **not** use `f64::max` / `f64::min`, which ignore
  `NaN` and would "improve" the semantics into disagreement.

Asserted: three outputs of input length; a consistent `NaN` warmup;
`upper >= middle >= lower` and `middle == (upper + lower) / 2.0` exactly where
finite; and bit-for-bit (`to_bits`) equality of `upper`/`lower` with an
independent O(n·p) reference.
*/

#![no_main]

use ferro_ta_core::extended;
use libfuzzer_sys::fuzz_target;

/// Bit-for-bit equality, with `NaN` compared by *NaN-ness* rather than payload.
///
/// Every finite and infinite value must match to the last bit — that is the
/// property a rewrite has to preserve. `NaN` payloads deliberately are not
/// part of the contract: hardware picks an operand's payload arbitrarily, an
/// arithmetic op quiets a signaling `NaN` from the raw input bytes, and
/// `donchian` substitutes its own canonical `f64::NAN` initializer for a
/// skipped bar rather than forwarding the input's `NaN`.
fn same_value(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        a.is_nan() && b.is_nan()
    } else {
        a.to_bits() == b.to_bits()
    }
}

/// Naive O(n·p) sliding extreme.
///
/// Rather than paraphrase the deque, this replays it from scratch over each
/// window, using the identical pop predicate. `is_max` selects `<=` (maximum)
/// or `>=` (minimum). The surviving front is the window extreme.
fn reference_sliding_extreme(real: &[f64], timeperiod: usize, is_max: bool) -> Vec<f64> {
    let n = real.len();
    let mut out = vec![f64::NAN; n];
    if timeperiod < 1 || n < timeperiod {
        return out;
    }
    let mut stack: Vec<f64> = Vec::with_capacity(timeperiod);
    for i in (timeperiod - 1)..n {
        stack.clear();
        for &x in &real[i + 1 - timeperiod..=i] {
            // Pop the dominated monotonic run. `NaN` on either side makes both
            // comparisons false, so nothing is popped and the NaN survives.
            while let Some(&back) = stack.last() {
                let dominated = if is_max { back <= x } else { back >= x };
                if dominated {
                    stack.pop();
                } else {
                    break;
                }
            }
            stack.push(x);
        }
        out[i] = stack[0];
    }
    out
}

/// Assert the full invariant set for one `(high, low)` pair.
///
/// `check_ordering` says whether `low[i] <= high[i]` holds elementwise. Only
/// then is `upper >= middle >= lower` a property of the kernel: two
/// *independent* random series can perfectly legitimately have
/// `rolling_max(high) < rolling_min(low)`, and asserting the band order on
/// those would be testing the input, not the code.
fn check(high: &[f64], low: &[f64], timeperiod: usize, check_ordering: bool) {
    let n = high.len();
    assert_eq!(low.len(), n, "test harness built ragged inputs");

    let (upper, middle, lower) = extended::donchian(high, low, timeperiod);
    assert_eq!(upper.len(), n, "DONCHIAN upper length mismatch");
    assert_eq!(middle.len(), n, "DONCHIAN middle length mismatch");
    assert_eq!(lower.len(), n, "DONCHIAN lower length mismatch");

    let warmup = timeperiod.saturating_sub(1);
    for i in 0..n.min(warmup) {
        assert!(upper[i].is_nan(), "DONCHIAN upper warmup[{i}] not NaN");
        assert!(middle[i].is_nan(), "DONCHIAN middle warmup[{i}] not NaN");
        assert!(lower[i].is_nan(), "DONCHIAN lower warmup[{i}] not NaN");
    }

    let raw_upper = reference_sliding_extreme(high, timeperiod, true);
    let raw_lower = reference_sliding_extreme(low, timeperiod, false);

    // The kernel emits a bar only when the rolling max of `high` is not NaN,
    // and that single test gates *all three* bands: a NaN anywhere in the
    // `high` window suppresses `lower` too, even where the rolling min of
    // `low` is perfectly finite. Reproduce that coupling rather than
    // second-guessing it.
    let want_upper = &raw_upper;
    let want_lower: Vec<f64> = raw_upper
        .iter()
        .zip(raw_lower.iter())
        .map(|(&u, &l)| if u.is_nan() { f64::NAN } else { l })
        .collect();

    for i in 0..n {
        // Equivalence with the naive reference, bit for bit (see
        // `same_value` for how NaN is handled).
        assert!(
            same_value(upper[i], want_upper[i]),
            "DONCHIAN upper[{i}] = {} != naive rolling max {} (tp={timeperiod})",
            upper[i],
            want_upper[i]
        );
        assert!(
            same_value(lower[i], want_lower[i]),
            "DONCHIAN lower[{i}] = {} != naive rolling min {} (tp={timeperiod})",
            lower[i],
            want_lower[i]
        );

        // `upper` is NaN only when the whole bar was skipped, which leaves all
        // three untouched at their NaN initializer.
        if upper[i].is_nan() {
            assert!(
                middle[i].is_nan() && lower[i].is_nan(),
                "DONCHIAN upper[{i}] is NaN but middle/lower are not ({}, {})",
                middle[i],
                lower[i]
            );
            continue;
        }

        // `middle` is computed as exactly this expression.
        if !lower[i].is_nan() {
            let want_middle = (upper[i] + lower[i]) / 2.0;
            assert!(
                same_value(middle[i], want_middle),
                "DONCHIAN middle[{i}] = {} != (upper + lower) / 2 = {want_middle}",
                middle[i]
            );
        } else {
            assert!(
                middle[i].is_nan(),
                "DONCHIAN middle[{i}] = {} is not NaN despite a NaN lower band",
                middle[i]
            );
        }

        if check_ordering && upper[i].is_finite() && middle[i].is_finite() && lower[i].is_finite() {
            assert!(
                upper[i] >= middle[i],
                "DONCHIAN upper[{i}] ({}) < middle[{i}] ({})",
                upper[i],
                middle[i]
            );
            assert!(
                middle[i] >= lower[i],
                "DONCHIAN middle[{i}] ({}) < lower[{i}] ({})",
                middle[i],
                lower[i]
            );
        }
    }
}

fn run(data: &[u8]) {
    if data.len() < 2 {
        return;
    }

    let timeperiod = ((data[0] as usize) % 64) + 1;

    let float_bytes = &data[1..];
    let n_floats = float_bytes.len() / 8;
    let per = n_floats / 2;
    if per == 0 {
        return;
    }
    let decode = |i: usize| -> f64 {
        let chunk: [u8; 8] = float_bytes[i * 8..(i + 1) * 8].try_into().unwrap();
        f64::from_le_bytes(chunk)
    };
    let high: Vec<f64> = (0..per).map(decode).collect();
    let second: Vec<f64> = (0..per).map(|i| decode(per + i)).collect();

    // Two independent raw-byte series: maximal NaN / +-inf / plateau coverage
    // for the deque equivalence check, but no band ordering to assert.
    check(&high, &second, timeperiod, false);

    // `low[i] <= high[i]` elementwise (NaN still propagates through both the
    // subtraction and `abs`), which makes the band ordering meaningful.
    let low: Vec<f64> = high
        .iter()
        .zip(second.iter())
        .map(|(&h, &d)| h - d.abs())
        .collect();
    check(&high, &low, timeperiod, true);
}

fuzz_target!(|data: &[u8]| {
    run(data);
});
