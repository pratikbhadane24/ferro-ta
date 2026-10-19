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
/// Scales the fixture volumes to non-dyadic fractions (crypto-like sizes),
/// so running volume sums cannot cancel exactly.
const FRACTIONAL_VOLUME: f64 = 0.37;
/// Zero-volume blocks: bars `i % ZERO_EVERY` in `ZERO_FROM..ZERO_TO`,
/// longer than every doubled default volume window.
const ZERO_EVERY: usize = 200;
const ZERO_FROM: usize = 100;
const ZERO_TO: usize = 160;

/// One input layout of the test.
#[derive(Clone, Copy, Debug)]
enum Layout {
    /// Flat bars: the first `lead`, every 50th, and (when `blocks`) the
    /// first 30 of every 97 — longer than any default window.
    Flat { lead: usize, blocks: bool },
    /// Rounded prices, fractional volumes with all-zero-volume blocks.
    ZeroVolume,
}

const FLAT_LAYOUTS: [Layout; 4] = [
    Layout::Flat {
        lead: 0,
        blocks: false,
    },
    Layout::Flat {
        lead: 3,
        blocks: false,
    },
    Layout::Flat {
        lead: 40,
        blocks: false,
    },
    Layout::Flat {
        lead: 0,
        blocks: true,
    },
];

fn is_flat(i: usize, lead: usize, blocks: bool) -> bool {
    i < lead || i.is_multiple_of(50) || (blocks && i % 97 < 30)
}

fn rounded(v: f64) -> f64 {
    (v * DECIMALS).round() / DECIMALS
}

fn layout_columns(case: &Case, layout: Layout) -> Vec<Vec<f64>> {
    let mut cols = columns(case);
    for (col, name) in cols.iter_mut().zip(case.inputs) {
        let is_volume = *name == "volume";
        for (i, v) in col.iter_mut().enumerate() {
            *v = match layout {
                Layout::Flat { lead, blocks } if is_flat(i, lead, blocks) => {
                    if is_volume {
                        FLAT_VOLUME
                    } else {
                        FLAT_PRICE
                    }
                }
                Layout::Flat { .. } => rounded(*v),
                Layout::ZeroVolume if !is_volume => rounded(*v),
                Layout::ZeroVolume if (ZERO_FROM..ZERO_TO).contains(&(i % ZERO_EVERY)) => 0.0,
                Layout::ZeroVolume => *v * FRACTIONAL_VOLUME,
            };
        }
    }
    cols
}

/// First parity error of `case` on `layout`, for every variant.
fn layout_failures(case: &Case, layout: Layout) -> Vec<String> {
    let cols = layout_columns(case, layout);
    let doubled = Variant {
        scale: 2,
        matype: None,
    };
    let mut variants = case.variants();
    variants.insert(1, doubled);
    variants
        .into_iter()
        .filter_map(|variant| {
            let want = (case.batch)(&cols, variant);
            let mut stream = (case.stream)(variant);
            want.iter()
                .enumerate()
                .find_map(|(i, w)| {
                    let got = stream.update(&row(&cols, i), true);
                    parity_error(case, i, &got, w)
                })
                .map(|e| format!("{e} {layout:?} {variant:?}"))
        })
        .collect()
}

#[test]
fn extended_streams_match_batch_on_flat_rounded_bars() {
    let cases = cases();
    let mut failures = Vec::new();
    for name in EXTENDED {
        let case = cases.iter().find(|c| c.name == *name).expect(name);
        for layout in FLAT_LAYOUTS {
            failures.extend(layout_failures(case, layout));
        }
    }
    assert!(
        failures.is_empty(),
        "flat-bar parity failures:\n{}",
        failures.join("\n")
    );
}

/// Every registry stream with a `volume` input, on fractional volumes with
/// all-zero-volume windows: a kernel whose running volume sum leaves a
/// residue there must be mirrored, not recomputed per window.
#[test]
fn volume_streams_match_batch_on_zero_volume_windows() {
    let cases = cases();
    let volume_cases: Vec<&Case> = cases
        .iter()
        .filter(|c| c.inputs.contains(&"volume"))
        .collect();
    assert!(volume_cases.len() >= 19, "{}", volume_cases.len());
    let failures: Vec<String> = volume_cases
        .iter()
        .flat_map(|case| layout_failures(case, Layout::ZeroVolume))
        .collect();
    assert!(
        failures.is_empty(),
        "zero-volume parity failures:\n{}",
        failures.join("\n")
    );
}
