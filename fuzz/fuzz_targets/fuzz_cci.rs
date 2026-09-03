/*!
Fuzz target for `ferro_ta_core::momentum::cci`.

CCI is the highest-risk kernel in the planned rolling-mean rewrite. It divides
by `0.015 * MAD`, so an error `e` in the rolling mean of the typical price
lands in the output amplified by roughly `1 / (0.015 * MAD)` — of order
`1e4`-`1e5` for a quiet window (`MAD` around `1e-3`-`1e-2` on a series priced
near 1). The Python suite gates CCI against TA-Lib at `atol = 1e-6`; a rolling
mean whose drift is a mere `1e-11` absolute can therefore blow straight through
that gate. Because the tolerance that matters is the one on the *output*, the
reference comparison below is **absolute**, not relative.

Asserted:

* output length equals input length, first `timeperiod - 1` slots `NaN`;
* the documented `mad == 0.0` short circuit returns exactly `0.0` (not `NaN`);
* agreement with an inline two-pass `(tp - mean) / (0.015 * mad)`.
*/

#![no_main]

use ferro_ta_core::momentum;
use libfuzzer_sys::fuzz_target;

/// Absolute tolerance for the reference comparison.
///
/// Matched to the `atol = 1e-6` TA-Lib gate the kernel has to hold, then
/// tightened by two orders of magnitude so the fuzzer fails *before* the
/// conformance suite does. The current kernel is itself two-pass, so today
/// the difference is exactly zero and this passes with full margin.
const ABS_TOL: f64 = 1e-8;

/// The `0.015` Lambert constant baked into CCI.
const CCI_SCALE: f64 = 0.015;

fn decode(bytes: &[u8]) -> Vec<f64> {
    (0..bytes.len() / 8)
        .map(|i| {
            let chunk: [u8; 8] = bytes[i * 8..(i + 1) * 8].try_into().unwrap();
            f64::from_le_bytes(chunk)
        })
        .collect()
}

fn run(data: &[u8]) {
    if data.len() < 2 {
        return;
    }

    let timeperiod = ((data[0] as usize) % 64) + 1;

    let all = decode(&data[1..]);
    // `cci` sizes itself off `high` and indexes `low`/`close` unchecked, so
    // the three series must be exactly equal length.
    let per = all.len() / 3;
    if per == 0 {
        return;
    }
    let high = &all[0..per];
    let low = &all[per..2 * per];
    let close = &all[2 * per..3 * per];
    let n = per;

    let result = momentum::cci(high, low, close, timeperiod);
    assert_eq!(result.len(), n, "CCI length mismatch");

    let warmup = timeperiod.saturating_sub(1);
    for i in 0..n.min(warmup) {
        assert!(result[i].is_nan(), "CCI warmup[{i}] not NaN: {}", result[i]);
    }
    if n < timeperiod {
        return;
    }

    // Independent typical price.
    let tp: Vec<f64> = (0..n)
        .map(|i| (high[i] + low[i] + close[i]) / 3.0)
        .collect();

    for i in warmup..n {
        let window = &tp[i + 1 - timeperiod..=i];
        if !window.iter().all(|x| x.is_finite()) {
            continue;
        }

        let p = timeperiod as f64;
        let mean = window.iter().sum::<f64>() / p;
        if !mean.is_finite() {
            continue;
        }
        let mad = window.iter().map(|&x| (x - mean).abs()).sum::<f64>() / p;
        if !mad.is_finite() {
            continue;
        }

        if mad == 0.0 {
            // Documented contract: an exactly-zero mean absolute deviation
            // yields 0.0, never NaN from a 0/0 division.
            assert_eq!(
                result[i].to_bits(),
                0.0_f64.to_bits(),
                "CCI[{i}] = {} where mad == 0.0; expected exactly +0.0",
                result[i]
            );
            continue;
        }

        let want = (tp[i] - mean) / (CCI_SCALE * mad);
        if !want.is_finite() {
            continue;
        }
        assert!(
            result[i].is_finite(),
            "CCI[{i}] non-finite ({}) where the two-pass reference is finite ({want})",
            result[i]
        );
        let diff = (result[i] - want).abs();
        // Absolute, as argued in the module docs. The `max` is a few-ulp
        // escape hatch that only ever engages for |want| above ~1e12, where
        // an absolute comparison has stopped being expressible in f64 at all;
        // at the CCI magnitudes that matter (|want| of order 1e2-1e5) it is
        // ~1e-11 and ABS_TOL dominates.
        let tol = ABS_TOL.max(8.0 * f64::EPSILON * want.abs());
        assert!(
            diff <= tol,
            "CCI[{i}] = {} disagrees with two-pass reference {want} \
             (abs diff = {diff}, tol = {tol}, tp={timeperiod}, mad={mad})",
            result[i]
        );
    }
}

fuzz_target!(|data: &[u8]| {
    run(data);
});
