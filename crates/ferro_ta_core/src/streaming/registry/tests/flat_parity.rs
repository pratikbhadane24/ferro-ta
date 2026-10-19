//! The extended streams on bars the random-walk fixture never produces:
//! prices rounded to 5 decimals (like the golden fixture) with flat
//! stretches (open = high = low = close, constant volume). Kernels that carry
//! a running sum can leave a rounding residue where a fresh window sum is
//! exactly 0, which flips a `!= 0` / `sqrt` NaN rule — this pins the NaN mask
//! on exactly that input.
use super::*;

/// Every stream of the `extended_*` families.
const EXTENDED: &[&str] = &[
    "donchian",
    "choppiness_index",
    "bbpercent",
    "bbwidth",
    "historical_volatility",
    "ulcer_index",
    "median",
    "mode",
    "keltner_channels",
    "chandelier_exit",
    "starc",
    "median_bands",
    "chande_kroll_stop",
    "ma_envelopes",
    "chaikin_vol",
    "mass",
    "rwi",
    "ao",
    "ac",
    "po",
    "dpo",
    "rvi",
    "kst",
    "vortex",
    "coppock",
    "dmi",
    "elder_ray",
    "fisher",
    "crsi",
    "tsi",
    "stc",
    "hull_ma",
    "alma",
    "alligator",
    "gator",
    "zlema",
    "frama",
    "mcginley",
    "vidya",
    "cmf",
    "emv",
    "volosc",
    "vroc",
    "rvol",
    "vwma",
    "cho",
    "obv_smoothed",
    "force_index",
    "nvi",
    "pvi",
    "nvi_with_ema",
    "pvi_with_signal",
    "kvo",
    "pvt",
];

const FLAT_PRICE: f64 = 100.0;
const FLAT_VOLUME: f64 = 1000.0;
const DECIMALS: f64 = 1e5;

/// Bar `i` is flat: the first `lead` bars, every 50th bar, and (when
/// `blocks`) the first 30 bars of every 97 — longer than any default window.
fn is_flat(i: usize, lead: usize, blocks: bool) -> bool {
    i < lead || i.is_multiple_of(50) || (blocks && i % 97 < 30)
}

fn flat_columns(case: &Case, lead: usize, blocks: bool) -> Vec<Vec<f64>> {
    let mut cols = columns(case);
    for (col, name) in cols.iter_mut().zip(case.inputs) {
        for (i, v) in col.iter_mut().enumerate() {
            *v = if !is_flat(i, lead, blocks) {
                (*v * DECIMALS).round() / DECIMALS
            } else if *name == "volume" {
                FLAT_VOLUME
            } else {
                FLAT_PRICE
            };
        }
    }
    cols
}

#[test]
fn extended_streams_match_batch_on_flat_rounded_bars() {
    let cases = cases();
    let mut failures = Vec::new();
    for name in EXTENDED {
        let case = cases.iter().find(|c| c.name == *name).expect(name);
        for (lead, blocks) in [(0, false), (3, false), (40, false), (0, true)] {
            let cols = flat_columns(case, lead, blocks);
            let doubled = Variant {
                scale: 2,
                matype: None,
            };
            let mut variants = case.variants();
            variants.insert(1, doubled);
            for variant in variants {
                let want = (case.batch)(&cols, variant);
                let mut stream = (case.stream)(variant);
                let first = want.iter().enumerate().find_map(|(i, w)| {
                    let got = stream.update(&row(&cols, i), true);
                    parity_error(case, i, &got, w)
                });
                failures
                    .extend(first.map(|e| format!("{e} lead={lead} blocks={blocks} {variant:?}")));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "flat-bar parity failures:\n{}",
        failures.join("\n")
    );
}
