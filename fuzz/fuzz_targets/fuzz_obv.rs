/*!
Fuzz target for `ferro_ta_core::volume::obv`.

Documented contract, and the guard for the planned branchless rewrite
(`((d > 0.0) as i32 - (d < 0.0) as i32)`):

* output length equals input length;
* bar 0 is seeded with `volume[0]`;
* every subsequent bar adds exactly `+volume[i]`, `-volume[i]` or `0.0`,
  selected by the sign of `close[i] - close[i - 1]`;
* an *unchanged* close contributes exactly zero — and so does a `NaN`
  difference, where both `>` and `<` are false. That last case is the one a
  branchless rewrite is most likely to get wrong (a `signum` or a
  `partial_cmp().unwrap()` would produce `NaN` or panic instead of `0.0`);
* no `NaN` appears in the output when the reference running sum is finite.

Random `f64` bit patterns essentially never produce equal adjacent closes, so
half of each run is fed a deliberately low-cardinality series built from the
raw bytes, giving long plateaus, zero volumes and negative volumes.
*/

#![no_main]

use ferro_ta_core::volume as vol;
use libfuzzer_sys::fuzz_target;

/// Distinct close levels in the low-cardinality series (small, so that runs of
/// equal closes are frequent).
const CLOSE_LEVELS: u8 = 5;
/// Distinct volume levels; the offset makes zeros and negatives common.
const VOLUME_LEVELS: u8 = 7;
const VOLUME_OFFSET: f64 = 3.0;

/// Bit-for-bit equality, with `NaN` compared by *NaN-ness* rather than payload.
///
/// Every finite and infinite value must match to the last bit — that is the
/// property a rewrite has to preserve. `NaN` payloads deliberately are not
/// part of the contract: hardware picks an operand's payload arbitrarily when
/// adding two NaNs, and any arithmetic op quiets a signaling `NaN` — which the
/// raw input bytes do produce.
fn same_value(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        a.is_nan() && b.is_nan()
    } else {
        a.to_bits() == b.to_bits()
    }
}

/// Recompute OBV independently, branch-per-case, and assert the kernel is
/// bit-identical to it.
fn check(close: &[f64], volume: &[f64]) {
    let n = close.len();
    assert_eq!(volume.len(), n, "test harness built ragged inputs");
    let result = vol::obv(close, volume);
    assert_eq!(result.len(), n, "OBV length mismatch");
    if n == 0 {
        return;
    }

    assert!(
        same_value(result[0], volume[0]),
        "OBV[0] = {} is not seeded with volume[0] = {}",
        result[0],
        volume[0]
    );

    let mut want = volume[0];
    for i in 1..n {
        let d = close[i] - close[i - 1];

        // The three-way contribution, spelled out branchfully.
        let contrib = if d > 0.0 {
            volume[i]
        } else if d < 0.0 {
            -volume[i]
        } else {
            // Equal closes *and* a NaN difference both land here.
            0.0
        };
        assert!(
            contrib.to_bits() == volume[i].to_bits()
                || contrib.to_bits() == (-volume[i]).to_bits()
                || contrib.to_bits() == 0.0_f64.to_bits(),
            "OBV contribution at {i} is none of +volume, -volume, 0.0"
        );

        want += contrib;

        // Bit-exact step contract for every finite and infinite value. Note that comparing
        // `result[i] - result[i - 1]` against the three candidates directly is
        // *unsound* in f64: when the running total dwarfs the volume the
        // addition absorbs it and the observed difference is 0.0 even though
        // the contribution was not. Recomputing `prev + contrib` is the
        // stronger form of the same statement, and it is what makes this a
        // real gate on the branchless rewrite.
        assert!(
            same_value(result[i], want),
            "OBV[{i}] = {} != prev + contrib = {want} (d = {d}, volume = {})",
            result[i],
            volume[i]
        );

        // Explicit no-change cases, stated separately so a failure names the
        // reason rather than just the mismatch.
        // A non-directional bar must not move the line. Two carve-outs, both
        // properties of f64 addition rather than of OBV: a NaN running total
        // can be *quieted* by `+ 0.0` (the raw input bytes do produce
        // signaling NaNs), and `-0.0 + 0.0` is `+0.0`, so this is a value
        // comparison rather than a bit comparison.
        if (d == 0.0 || d.is_nan()) && !result[i - 1].is_nan() {
            assert!(
                result[i] == result[i - 1],
                "OBV[{i}] = {} moved off {} on a non-directional close (d = {d})",
                result[i],
                result[i - 1]
            );
        }

        if want.is_finite() {
            assert!(
                !result[i].is_nan(),
                "OBV[{i}] is NaN where the reference running total is finite ({want})"
            );
        }
    }
}

fn run(data: &[u8]) {
    if data.len() < 16 {
        return;
    }

    // --- run 1: raw f64 bit patterns (NaN, +/-inf, subnormals) ---------
    let n_floats = data.len() / 8;
    let pairs = n_floats / 2;
    if pairs == 0 {
        return;
    }
    let decode = |i: usize| -> f64 {
        let chunk: [u8; 8] = data[i * 8..(i + 1) * 8].try_into().unwrap();
        f64::from_le_bytes(chunk)
    };
    let close: Vec<f64> = (0..pairs).map(decode).collect();
    let volume: Vec<f64> = (0..pairs).map(|i| decode(pairs + i)).collect();
    check(&close, &volume);

    // --- run 2: low-cardinality series with plateaus and zero volume ---
    let half = data.len() / 2;
    let lc_close: Vec<f64> = data[..half]
        .iter()
        .map(|&b| (b % CLOSE_LEVELS) as f64)
        .collect();
    let lc_volume: Vec<f64> = data[half..]
        .iter()
        .map(|&b| (b % VOLUME_LEVELS) as f64 - VOLUME_OFFSET)
        .collect();
    let m = lc_close.len().min(lc_volume.len());
    check(&lc_close[..m], &lc_volume[..m]);
}

fuzz_target!(|data: &[u8]| {
    run(data);
});
