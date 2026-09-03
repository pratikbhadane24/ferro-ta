//! MA-type dispatchers: MACDFIX, MACDEXT, MA and MAVP.
//!
//! # The `matype` selector
//!
//! Every `matype` argument in this crate resolves through
//! [`compute_ma_by_type`]:
//!
//! | `matype` | MA        | NaN warm-up (bars before the first value) |
//! |----------|-----------|-------------------------------------------|
//! | 0        | SMA       | `p - 1`                                   |
//! | 1        | EMA       | `p - 1`                                   |
//! | 2        | WMA       | `p - 1`                                   |
//! | 3        | DEMA      | `2 * (p - 1)`                             |
//! | 4        | TEMA      | `3 * (p - 1)`                             |
//! | 5        | TRIMA     | `p - 1`                                   |
//! | 6        | KAMA      | `p` (the seed at `p - 1` is not emitted)  |
//! | 7        | T3        | `6 * (p - 1)`                             |
//! | 8        | T3 (alias of 7) | `6 * (p - 1)`                       |
//!
//! Those counts are [`ma_lookback`] and are exact only for an input with no
//! leading `NaN` prefix. The composed-EMA types (3, 4, 7, 8) stack
//! [`ema_from_first_finite`](super::ema_from_first_finite) stages, each of
//! which slips forward past non-finite input, so a leading `NaN` prefix of
//! length `start` — or a gap inside any stage's seed window — pushes the first
//! value later (see the per-kernel "Warm-up" sections). Indicators that derive
//! their output start from `ma_lookback` therefore still gate every bar on the
//! MA actually being non-`NaN`; the lookback only says where output *may*
//! begin, never that it does.
//!
//! # Divergence from TA-Lib's `TA_MAType` (read this before passing `7`)
//!
//! TA-Lib numbers its enum `0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA,
//! 6=KAMA, 7=MAMA, 8=T3`. This crate is **not** enum-compatible with TA-Lib:
//!
//! | `matype` | This crate | TA-Lib `TA_MAType` | Compatible? |
//! |----------|------------|--------------------|-------------|
//! | 0        | SMA        | SMA                | yes         |
//! | 1        | EMA        | EMA                | yes         |
//! | 2        | WMA        | WMA                | yes         |
//! | 3        | DEMA       | DEMA               | yes         |
//! | 4        | TEMA       | TEMA               | yes         |
//! | 5        | TRIMA      | TRIMA              | yes         |
//! | 6        | KAMA       | KAMA               | yes         |
//! | 7        | **T3**     | **MAMA**           | **NO**      |
//! | 8        | T3         | T3                 | yes         |
//! | ≥ 9      | out of range | out of range     | yes         |
//!
//! Two consequences, neither of them softened:
//!
//! 1. **`matype = 7` means T3 here and MAMA in TA-Lib.** Code ported from
//!    TA-Lib that passes `7` expecting MAMA silently gets T3 — no error, no
//!    warning, just a different indicator. `7` keeps its meaning because the
//!    `7 = T3` numbering already ships and is validated across Rust, PyO3,
//!    WASM and Flutter; renumbering it would change the output of existing
//!    callers. Pass `8` for T3 if you want a value that means the same thing
//!    in both libraries.
//! 2. **MAMA is not reachable through any `matype` value.** TA-Lib's `TA_MA`
//!    handles `matype = 7` by calling `TA_MAMA(…, 0.5, 0.05, …)` — the period
//!    is ignored and the FAMA output discarded — and this dispatcher has no
//!    such arm. Callers wanting MAMA must call [`mama`](super::mama) directly;
//!    it takes the two limits explicitly and returns both MAMA and FAMA.
//!
//! This applies to every `matype`-taking surface: [`ma`], [`macdext`],
//! [`mavp`], `apo`, `ppo`, `stoch`, `stochf`, `stochrsi`, and
//! `ma_envelopes` / `obv_smoothed` / `pvi_with_signal`.

use super::{dema, ema, kama, macd, sma, sma_into, t3, tema, trima, wma};

// ---------------------------------------------------------------------------
// MACDFIX / MACDEXT
// ---------------------------------------------------------------------------

/// MACD with fixed 12/26 periods.
pub fn macdfix(close: &[f64], signalperiod: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    macd(close, 12, 26, signalperiod)
}

/// Largest `matype` this crate's dispatcher understands (`8` = T3).
///
/// Callers that accept a `matype` from the outside validate against this and
/// return an all-`NaN` output for anything larger, rather than falling back to
/// SMA: a silent fallback makes a caller's typo indistinguishable from a valid
/// request.
///
/// This matches TA-Lib's upper bound, but the *meaning* of `7` does not — see
/// the module docs: `7` is T3 here and MAMA in TA-Lib, and MAMA has no
/// `matype` at all.
pub(crate) const MAX_MATYPE: u8 = 8;

/// Compute MA by type: 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA,
/// 7=T3, 8=T3 (TA-Lib's number for T3; an exact alias of `7`).
///
/// `pub(crate)` rather than private because the `momentum` module tree needs
/// it (`apo`, `ppo`, `stoch`, `stochf`, `stochrsi`). Deliberately **not**
/// `pub`: the published surface is [`ma`], not the dispatcher.
///
/// An out-of-range `matype` (now anything above `8`) falls back to SMA here.
/// This arm is unreachable for every caller that validates against
/// [`MAX_MATYPE`] first — raising that bound to `8` moved the arm's entry
/// point along with it, so the reasoning is unchanged; it exists only because
/// [`ma`] and [`macdext`] have shipped with that behaviour.
pub(crate) fn compute_ma_by_type(close: &[f64], timeperiod: usize, matype: u8) -> Vec<f64> {
    match matype {
        0 => sma(close, timeperiod),
        1 => ema(close, timeperiod),
        2 => wma(close, timeperiod),
        3 => dema(close, timeperiod),
        4 => tema(close, timeperiod),
        5 => trima(close, timeperiod),
        6 => kama(close, timeperiod),
        // `8` is TA-Lib's T3 slot; identical call so the two are
        // indistinguishable in output (see `matype8_is_bit_identical_to_7`).
        7 | 8 => t3(close, timeperiod, 0.7),
        _ => sma(close, timeperiod),
    }
}

/// Number of leading `NaN` bars [`compute_ma_by_type`] emits for `timeperiod`
/// and `matype` — TA-Lib's `TA_MA_Lookback`.
///
/// Exact for an input with no leading `NaN` prefix; a *lower bound* otherwise
/// (see the module docs). Returns `usize::MAX` for an out-of-range `matype`,
/// which pushes every derived start index past the end of any series.
pub(crate) fn ma_lookback(timeperiod: usize, matype: u8) -> usize {
    let p1 = timeperiod.saturating_sub(1);
    match matype {
        // SMA, EMA, WMA, TRIMA
        0 | 1 | 2 | 5 => p1,
        // Saturating so an absurd `timeperiod` yields a start index past the
        // end of any series instead of overflowing in a debug build.
        3 => p1.saturating_mul(2), // DEMA
        4 => p1.saturating_mul(3), // TEMA
        6 => timeperiod,           // KAMA emits its first ER/SC update at index `p`
        // T3 at both its native `7` and TA-Lib's `8`.
        7 | 8 => p1.saturating_mul(6),
        _ => usize::MAX,
    }
}

/// Write `compute_ma_by_type(src, timeperiod, matype)` into `dest` starting at
/// `dest_offset`, leaving `NaN` slots in `dest` untouched.
///
/// `matype == 0` routes to [`sma_into`], which is what the stochastic kernels
/// have always called — the streaming SMA recurrence is not bit-identical to a
/// freshly summed window, so the default path must keep using it.
pub(crate) fn ma_into(
    src: &[f64],
    timeperiod: usize,
    matype: u8,
    dest: &mut [f64],
    dest_offset: usize,
) {
    if matype == 0 {
        sma_into(src, timeperiod, dest, dest_offset);
        return;
    }
    for (j, &v) in compute_ma_by_type(src, timeperiod, matype)
        .iter()
        .enumerate()
    {
        if !v.is_nan() {
            dest[dest_offset + j] = v;
        }
    }
}

/// MACD with configurable MA types for fast/slow/signal.
///
/// # NaN warm-up
///
/// All three outputs share one start index, as in [`macd`]:
///
/// ```text
/// macd_start = max(ma_lookback(fastperiod, fastmatype),
///                  ma_lookback(slowperiod, slowmatype))
/// first      = macd_start + ma_lookback(signalperiod, signalmatype)
/// ```
///
/// `macd_start` is a `max` over the *two legs' own lookbacks*, not
/// `slowperiod - 1`. Those coincide only when both legs are window MAs
/// (`matype` `0`, `1`, `2`, `5`); a leg typed DEMA / TEMA / T3 / KAMA warms up
/// at `2(p-1)` / `3(p-1)` / `6(p-1)` / `p`, and the fast leg's lookback can
/// exceed the slow leg's (T3 at `fastperiod = 12` needs 66 bars where SMA at
/// `slowperiod = 26` needs 25), so neither leg alone bounds it.
///
/// Getting `macd_start` right is not cosmetic: the signal leg is an MA of
/// `macd_line[macd_start..]`, so a `macd_start` earlier than the true first
/// valid MACD bar hands that MA a slice with a leading `NaN`. Most MA types
/// then emit `NaN` for at least a window, and `kama` propagates it for the
/// whole series — which is how `MACDEXT` at `matype = 6` on all three legs
/// used to return an all-`NaN` result.
///
/// # Invalid arguments
///
/// A zero period, `fastperiod >= slowperiod`, a `matype` above [`MAX_MATYPE`],
/// or a warm-up at or past the end of the series each yield an all-`NaN`
/// output of `close.len()` — the crate-wide convention for a core kernel with
/// no error type (compare [`mavp`]). The PyO3, WASM and Flutter wrappers reject
/// the out-of-range `matype` before it gets here.
pub fn macdext(
    close: &[f64],
    fastperiod: usize,
    fastmatype: u8,
    slowperiod: usize,
    slowmatype: u8,
    signalperiod: usize,
    signalmatype: u8,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = close.len();
    let nan3 = || (vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n]);
    if fastperiod == 0 || slowperiod == 0 || signalperiod == 0 || fastperiod >= slowperiod {
        return nan3();
    }
    if fastmatype > MAX_MATYPE || slowmatype > MAX_MATYPE || signalmatype > MAX_MATYPE {
        return nan3();
    }
    // The first bar at which *both* legs have a value. `ma_lookback` is exact
    // here because `close` is the raw input, so any leading `NaN` prefix in it
    // is the caller's and is handled by the per-bar `is_nan` gate below.
    let macd_start = ma_lookback(fastperiod, fastmatype).max(ma_lookback(slowperiod, slowmatype));
    if macd_start >= n {
        return nan3();
    }
    let fast_ma = compute_ma_by_type(close, fastperiod, fastmatype);
    let slow_ma = compute_ma_by_type(close, slowperiod, slowmatype);
    let mut macd_line = vec![f64::NAN; n];
    for i in macd_start..n {
        if !fast_ma[i].is_nan() && !slow_ma[i].is_nan() {
            macd_line[i] = fast_ma[i] - slow_ma[i];
        }
    }
    let macd_valid: Vec<f64> = macd_line[macd_start..].to_vec();
    let signal_slice = compute_ma_by_type(&macd_valid, signalperiod, signalmatype);
    let mut signal_line = vec![f64::NAN; n];
    // The signal leg's own lookback stacks on top of `macd_start`, and it is
    // `signalperiod - 1` only for the window matypes.
    let warmup = macd_start.saturating_add(ma_lookback(signalperiod, signalmatype));
    for i in warmup..n {
        let j = i - macd_start;
        if j < signal_slice.len() && !signal_slice[j].is_nan() {
            signal_line[i] = signal_slice[j];
        }
    }
    let mut histogram = vec![f64::NAN; n];
    for i in 0..n {
        if !macd_line[i].is_nan() && !signal_line[i].is_nan() {
            histogram[i] = macd_line[i] - signal_line[i];
        }
    }
    // TA-Lib pads all three outputs to the same start (same convention as
    // `macd`). Pad to where the signal line *actually* becomes valid rather
    // than to `warmup`: the two agree for finite input now that `warmup`
    // carries the signal matype's own lookback, but a `NaN` inside the input
    // pushes the first value later still, and leaving macd_line numeric while
    // the other two are NaN would break the shared-start convention.
    let first_signal = signal_line
        .iter()
        .position(|v| !v.is_nan())
        .unwrap_or(macd_line.len());
    for v in macd_line[..first_signal].iter_mut() {
        *v = f64::NAN;
    }
    (macd_line, signal_line, histogram)
}

// ---------------------------------------------------------------------------
// MA (generic dispatcher) / MAVP (variable period)
// ---------------------------------------------------------------------------

/// Generic Moving Average. matype: 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA,
/// 5=TRIMA, 6=KAMA, 7=T3, 8=T3 (TA-Lib's T3 number, an alias of `7`).
///
/// See the [module docs](self) for the per-type warm-up and the divergence
/// from TA-Lib's `TA_MAType` at `7` — `7` is T3 here but MAMA in TA-Lib, and
/// MAMA is not reachable through `matype` at all (call
/// [`mama`](super::mama)). An out-of-range `matype` falls back to SMA
/// (pre-existing behaviour; the PyO3, WASM and Flutter wrappers reject it
/// before it gets here).
pub fn ma(close: &[f64], timeperiod: usize, matype: u8) -> Vec<f64> {
    compute_ma_by_type(close, timeperiod, matype)
}

/// Moving Average with Variable Period per bar.
///
/// At each bar the moving average of type `matype` is taken over the period
/// given by the corresponding element of `periods`, rounded and clamped to
/// `[minperiod, maxperiod]`.
///
/// `periods` must have the same length as `close`; a mismatch returns a
/// `NaN`-filled `Vec` of `close.len()` rather than a partial result. This is
/// the crate-wide convention for a kernel that derives `n` from one slice and
/// then indexes another (compare `utils::crossover` and
/// `extended::trend::alligator`) — the core has no error type, so an unusable
/// argument set is reported as an all-`NaN` output. A `matype` above
/// [`MAX_MATYPE`] is reported the same way.
///
/// # Arguments
/// * `close` - Price series.
/// * `periods` - Per-bar period, same length as `close`.
/// * `minperiod` / `maxperiod` - Clamp bounds (`minperiod >= 1`).
/// * `matype` - MA type, `0`–`8` (`8` is an alias of `7` = T3); see the
///   [module docs](self) for the mapping, the TA-Lib divergence at `7`, and
///   the per-type warm-up. `0` (SMA) is TA-Lib's default.
///
/// # NaN warm-up
///
/// `ma_lookback(maxperiod, matype)` leading `NaN`s — `maxperiod - 1` for the
/// default SMA — regardless of the local per-bar period, matching
/// `TA_MAVP_Lookback`. A non-zero `matype` therefore shifts the first value
/// later (e.g. `6 * (maxperiod - 1)` for T3).
///
/// # TA-Lib compatibility
///
/// `TA_MAVP` computes a full `TA_MA` per distinct period and reads the bar out
/// of it; this does the same, so `matype != 0` bars are exactly
/// `ma(close, p_i, matype)[i]`. The `matype == 0` path keeps the original
/// per-bar window sum, which is bit-identical to the pre-`matype` kernel but
/// *not* to the streaming recurrence in [`sma`].
pub fn mavp(
    close: &[f64],
    periods: &[f64],
    minperiod: usize,
    maxperiod: usize,
    matype: u8,
) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![f64::NAN; n];
    if periods.len() != n {
        return result;
    }
    if minperiod == 0 || maxperiod < minperiod || matype > MAX_MATYPE {
        return result;
    }
    // TA-Lib MAVP outputs NaN until the *maxperiod* MA's lookback regardless
    // of the local per-bar period.
    let start = ma_lookback(maxperiod, matype);
    if start >= n {
        return result;
    }
    if matype == 0 {
        for i in start..n {
            let p = (periods[i].round() as usize).clamp(minperiod, maxperiod);
            if i + 1 >= p {
                let sum: f64 = close[(i + 1 - p)..=i].iter().sum();
                result[i] = sum / p as f64;
            }
        }
        return result;
    }
    // Non-SMA: one full MA pass per *distinct* period actually requested,
    // scattered back to the bars that asked for it. `start` is the maxperiod
    // lookback and lookbacks are monotone in the period, so every selected
    // bar is past its own MA's warm-up.
    let mut clamped = vec![0usize; n];
    let mut requested = vec![false; maxperiod - minperiod + 1];
    for i in start..n {
        let p = (periods[i].round() as usize).clamp(minperiod, maxperiod);
        clamped[i] = p;
        requested[p - minperiod] = true;
    }
    for (offset, &is_requested) in requested.iter().enumerate() {
        if !is_requested {
            continue;
        }
        let p = minperiod + offset;
        let ma = compute_ma_by_type(close, p, matype);
        for i in start..n {
            if clamped[i] == p {
                result[i] = ma[i];
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Equivalence oracle: verbatim copy of `mavp` as it stood before the
    // `matype` argument existed. `matype = 0` must be bit-identical to it.
    // -----------------------------------------------------------------------
    fn reference_mavp(
        close: &[f64],
        periods: &[f64],
        minperiod: usize,
        maxperiod: usize,
    ) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];
        if periods.len() != n {
            return result;
        }
        if minperiod == 0 || maxperiod < minperiod {
            return result;
        }
        for i in (maxperiod - 1)..n {
            let p = (periods[i].round() as usize).clamp(minperiod, maxperiod);
            if i + 1 >= p {
                let sum: f64 = close[(i + 1 - p)..=i].iter().sum();
                result[i] = sum / p as f64;
            }
        }
        result
    }

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

    fn assert_bits(got: &[f64], want: &[f64], ctx: &str) {
        assert_eq!(got.len(), want.len(), "{ctx}: length");
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "{ctx}: index {i}: {g} vs {w}");
        }
    }

    // -----------------------------------------------------------------------
    // MACDEXT warm-up. `macd_start` used to be hard-coded to `slowperiod - 1`,
    // which is only the SMA-family lookback; a leg typed DEMA/TEMA/T3/KAMA
    // warms up later, so the signal leg was handed a slice whose first bar was
    // `NaN`. At `matype = 6` (KAMA) on all three legs that `NaN` propagated to
    // the entire series and MACDEXT returned all `NaN`.
    // -----------------------------------------------------------------------

    /// The start index all three MACDEXT outputs share.
    fn expected_macdext_start(
        fastperiod: usize,
        fastmatype: u8,
        slowperiod: usize,
        slowmatype: u8,
        signalperiod: usize,
        signalmatype: u8,
    ) -> usize {
        ma_lookback(fastperiod, fastmatype).max(ma_lookback(slowperiod, slowmatype))
            + ma_lookback(signalperiod, signalmatype)
    }

    #[test]
    fn macdext_warmup_is_max_leg_lookback_plus_signal_lookback() {
        // 700 bars clears the worst case: T3 legs at 26/9 need 150 + 48 = 198.
        let close = oracle_close(700);
        let (fastperiod, slowperiod, signalperiod) = (12usize, 26usize, 9usize);
        for fastmatype in 0..=MAX_MATYPE {
            for slowmatype in 0..=MAX_MATYPE {
                for signalmatype in 0..=MAX_MATYPE {
                    let (macd, signal, hist) = macdext(
                        &close,
                        fastperiod,
                        fastmatype,
                        slowperiod,
                        slowmatype,
                        signalperiod,
                        signalmatype,
                    );
                    let want = expected_macdext_start(
                        fastperiod,
                        fastmatype,
                        slowperiod,
                        slowmatype,
                        signalperiod,
                        signalmatype,
                    );
                    let ctx = format!("fast={fastmatype} slow={slowmatype} signal={signalmatype}");

                    for (name, out) in [("macd", &macd), ("signal", &signal), ("hist", &hist)] {
                        assert_eq!(out.len(), close.len(), "{ctx}: {name} length");
                        // The NaN prefix is exactly the derived warm-up ...
                        for (i, &v) in out.iter().enumerate().take(want) {
                            assert!(v.is_nan(), "{ctx}: {name} expected NaN at {i}, got {v}");
                        }
                        // ... and the first bar after it is finite, not NaN.
                        assert!(
                            out[want].is_finite(),
                            "{ctx}: {name} expected a finite value at {want}, got {}",
                            out[want]
                        );
                        // No interior NaN: finite input, finite output.
                        assert!(
                            out[want..].iter().all(|v| v.is_finite()),
                            "{ctx}: {name} has an interior NaN"
                        );
                    }
                    // `hist = macd - signal` holds on the shared valid region.
                    for i in want..close.len() {
                        assert!(
                            (hist[i] - (macd[i] - signal[i])).abs() < 1e-10,
                            "{ctx}: hist identity at {i}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn macdext_kama_on_all_three_legs_is_not_all_nan() {
        // The exact regression: `MACDEXT(matype=6, 6, 6)` returned all NaN.
        let close = oracle_close(400);
        let (macd, signal, hist) = macdext(&close, 12, 6, 26, 6, 9, 6);
        // KAMA's lookback is `p`, so: max(12, 26) + 9 = 35.
        assert_eq!(expected_macdext_start(12, 6, 26, 6, 9, 6), 35);
        for (name, out) in [("macd", &macd), ("signal", &signal), ("hist", &hist)] {
            assert!(out.iter().any(|v| v.is_finite()), "{name} is entirely NaN");
            assert!(out[35].is_finite(), "{name} at 35: {}", out[35]);
            assert!(out[34].is_nan(), "{name} at 34: {}", out[34]);
        }
    }

    #[test]
    fn macdext_warmup_is_not_slowperiod_minus_one_for_long_lookback_legs() {
        // Guards against a regression to `macd_start = slowperiod - 1`: the
        // fast leg alone can push the start past the slow leg's lookback.
        let close = oracle_close(700);
        // T3 fast leg (6 * 11 = 66) against an SMA slow leg (25).
        let (macd, _, _) = macdext(&close, 12, 7, 26, 0, 9, 0);
        assert_eq!(expected_macdext_start(12, 7, 26, 0, 9, 0), 66 + 8);
        assert!(macd[66 + 7].is_nan());
        assert!(macd[66 + 8].is_finite());
        // Both legs SMA-like: the old expression was right, and must not move.
        for matype in [0u8, 1, 2, 5] {
            let (macd, _, _) = macdext(&close, 12, matype, 26, matype, 9, matype);
            assert_eq!(
                macd.iter().position(|v| !v.is_nan()),
                Some(26 - 1 + 9 - 1),
                "matype={matype}"
            );
        }
    }

    #[test]
    fn macdext_invalid_arguments_return_all_nan() {
        let close = oracle_close(400);
        let all_nan = |t: (Vec<f64>, Vec<f64>, Vec<f64>), ctx: &str| {
            for (name, out) in [("macd", &t.0), ("signal", &t.1), ("hist", &t.2)] {
                assert_eq!(out.len(), close.len(), "{ctx}: {name} length");
                assert!(out.iter().all(|v| v.is_nan()), "{ctx}: {name} not all NaN");
            }
        };
        // Zero periods and fast >= slow (pre-existing behaviour).
        all_nan(macdext(&close, 0, 1, 26, 1, 9, 1), "fastperiod=0");
        all_nan(macdext(&close, 12, 1, 0, 1, 9, 1), "slowperiod=0");
        all_nan(macdext(&close, 12, 1, 26, 1, 0, 1), "signalperiod=0");
        all_nan(macdext(&close, 26, 1, 12, 1, 9, 1), "fast >= slow");
        // Out-of-range matype on any leg. `ma_lookback` returns `usize::MAX`
        // there, so this also pins the guard that keeps `macd_line[macd_start..]`
        // from panicking.
        for matype in [MAX_MATYPE + 1, 10, 99, u8::MAX] {
            all_nan(macdext(&close, 12, matype, 26, 1, 9, 1), "fastmatype");
            all_nan(macdext(&close, 12, 1, 26, matype, 9, 1), "slowmatype");
            all_nan(macdext(&close, 12, 1, 26, 1, 9, matype), "signalmatype");
        }
        // Warm-up at or past the end of the series: T3 legs need 198 bars.
        let short = oracle_close(150);
        let (m, s, h) = macdext(&short, 12, 7, 26, 7, 9, 7);
        assert_eq!(m.len(), 150);
        assert!(m.iter().chain(s.iter()).chain(h.iter()).all(|v| v.is_nan()));
        // Empty input stays empty.
        assert!(macdext(&[], 12, 1, 26, 1, 9, 1).0.is_empty());
    }

    #[test]
    fn macdext_matype8_is_bit_identical_to_matype7() {
        let close = oracle_close(700);
        for (fast, slow, signal) in [(7u8, 7u8, 7u8), (7, 0, 0), (0, 7, 0), (0, 0, 7)] {
            let alias = |m: u8| if m == 7 { 8 } else { m };
            let a = macdext(&close, 12, fast, 26, slow, 9, signal);
            let b = macdext(&close, 12, alias(fast), 26, alias(slow), 9, alias(signal));
            let ctx = format!("fast={fast} slow={slow} signal={signal}");
            assert_bits(&a.0, &b.0, &format!("macd {ctx}"));
            assert_bits(&a.1, &b.1, &format!("signal {ctx}"));
            assert_bits(&a.2, &b.2, &format!("hist {ctx}"));
        }
    }

    #[test]
    fn macdfix_matches_macdext_with_ema_legs() {
        // `macdfix` is `macd(close, 12, 26, signalperiod)`; the EMA-typed
        // MACDEXT must agree with it, which pins the SMA-family start index
        // against the plain `macd` kernel's own convention.
        let close = oracle_close(400);
        let (fm, fs, fh) = macdfix(&close, 9);
        let (em, es, eh) = macdext(&close, 12, 1, 26, 1, 9, 1);
        for i in 0..close.len() {
            assert_eq!(fm[i].is_nan(), em[i].is_nan(), "macd NaN placement at {i}");
            assert_eq!(
                fs[i].is_nan(),
                es[i].is_nan(),
                "signal NaN placement at {i}"
            );
            assert_eq!(fh[i].is_nan(), eh[i].is_nan(), "hist NaN placement at {i}");
        }
    }

    #[test]
    fn mavp_matype0_is_bit_identical_to_pre_matype_kernel() {
        for n in [0usize, 1, 2, 7, 33, 200] {
            let close = oracle_close(n);
            // Several `periods` shapes: constant, sawtooth, out-of-range,
            // NaN/negative (which the saturating cast folds to `minperiod`).
            let shapes: Vec<Vec<f64>> = vec![
                vec![5.0; n],
                (0..n).map(|i| (i % 9) as f64).collect(),
                (0..n).map(|i| 30.0 - (i % 40) as f64).collect(),
                (0..n)
                    .map(|i| if i % 5 == 0 { f64::NAN } else { -3.0 })
                    .collect(),
            ];
            for (si, periods) in shapes.iter().enumerate() {
                for (minp, maxp) in [(1usize, 1usize), (2, 5), (1, 30), (3, 3), (0, 5), (6, 5)] {
                    let ctx = format!("n={n} shape={si} min={minp} max={maxp}");
                    assert_bits(
                        &mavp(&close, periods, minp, maxp, 0),
                        &reference_mavp(&close, periods, minp, maxp),
                        &ctx,
                    );
                }
            }
        }
    }

    #[test]
    fn mavp_out_of_range_matype_returns_all_nan() {
        let close = oracle_close(60);
        let periods = vec![5.0; close.len()];
        for matype in [MAX_MATYPE + 1, 9, 99, u8::MAX] {
            let got = mavp(&close, &periods, 2, 10, matype);
            assert_eq!(got.len(), close.len());
            assert!(got.iter().all(|v| v.is_nan()), "matype={matype}: {got:?}");
        }
    }

    #[test]
    fn mavp_constant_period_matches_ma_of_that_type() {
        let close = oracle_close(300);
        for matype in 0..=MAX_MATYPE {
            let p = 6usize;
            let periods = vec![p as f64; close.len()];
            let got = mavp(&close, &periods, p, p, matype);
            let want = ma(&close, p, matype);
            // MAVP is NaN until the *maxperiod* lookback, which here is the
            // same period, so the two series must agree slot for slot —
            // except that `matype == 0` uses a fresh window sum rather than
            // the streaming recurrence, so that one gets a tolerance.
            for i in 0..close.len() {
                assert_eq!(
                    got[i].is_nan(),
                    want[i].is_nan(),
                    "matype={matype}: NaN placement differs at {i}"
                );
                if !want[i].is_nan() {
                    assert!(
                        (got[i] - want[i]).abs() <= 1e-9,
                        "matype={matype} at {i}: {} vs {}",
                        got[i],
                        want[i]
                    );
                }
            }
        }
    }

    #[test]
    fn mavp_warmup_follows_ma_lookback_per_matype() {
        let close = oracle_close(400);
        let maxp = 5usize;
        let periods = vec![maxp as f64; close.len()];
        for matype in 0..=MAX_MATYPE {
            let got = mavp(&close, &periods, 2, maxp, matype);
            let lookback = ma_lookback(maxp, matype);
            for (i, &v) in got.iter().enumerate().take(lookback) {
                assert!(v.is_nan(), "matype={matype}: expected NaN at {i}, got {v}");
            }
            assert!(
                got[lookback].is_finite(),
                "matype={matype}: expected a value at {lookback}"
            );
        }
    }

    #[test]
    fn ma_lookback_matches_documented_warmups() {
        // Empirically: where does each MA type first emit a value?
        let close = oracle_close(400);
        for p in [1usize, 2, 3, 5, 14] {
            for matype in 0..=MAX_MATYPE {
                let series = compute_ma_by_type(&close, p, matype);
                let first = series.iter().position(|v| !v.is_nan()).unwrap();
                assert_eq!(
                    first,
                    ma_lookback(p, matype),
                    "matype={matype} p={p}: first value at {first}"
                );
            }
            // TA-Lib's T3 slot must report the same lookback as the crate's.
            assert_eq!(ma_lookback(p, 8), ma_lookback(p, 7), "p={p}");
        }
        assert_eq!(ma_lookback(5, MAX_MATYPE + 1), usize::MAX);
        // Saturating arithmetic keeps the out-of-range sentinel and the huge
        // periods from overflowing in a debug build.
        assert_eq!(ma_lookback(usize::MAX, 8), usize::MAX);
        for matype in [9u8, 10, 99, u8::MAX] {
            assert_eq!(ma_lookback(14, matype), usize::MAX, "matype={matype}");
        }
    }

    // -----------------------------------------------------------------------
    // TA-Lib numbers T3 as 8; this crate numbers it 7 and accepts 8 as an
    // exact alias. The two must be indistinguishable, bit for bit.
    // -----------------------------------------------------------------------
    #[test]
    fn matype8_is_bit_identical_to_matype7() {
        let series: Vec<Vec<f64>> = vec![
            oracle_close(0),
            oracle_close(1),
            oracle_close(5),
            oracle_close(120),
            oracle_close(400),
            (1..=50).map(|i| i as f64).collect(),
            vec![42.0; 80],
            {
                let mut v = oracle_close(150);
                v[0] = f64::NAN;
                v[7] = f64::NAN;
                v
            },
        ];
        // Degenerate periods included: 0, 1, and one longer than the series.
        for (si, close) in series.iter().enumerate() {
            for p in [0usize, 1, 2, 3, 5, 14, 500] {
                let ctx = format!("series={si} p={p}");
                assert_bits(
                    &compute_ma_by_type(close, p, 8),
                    &compute_ma_by_type(close, p, 7),
                    &format!("compute_ma_by_type {ctx}"),
                );
                assert_bits(&ma(close, p, 8), &ma(close, p, 7), &format!("ma {ctx}"));
                assert_bits(
                    &t3(close, p, 0.7),
                    &ma(close, p, 8),
                    &format!("t3 vfactor {ctx}"),
                );
            }
        }
    }

    #[test]
    fn mavp_matype8_is_bit_identical_to_matype7() {
        for n in [0usize, 1, 3, 60, 400] {
            let close = oracle_close(n);
            let periods: Vec<f64> = (0..n).map(|i| 2.0 + (i % 7) as f64).collect();
            for (minp, maxp) in [(1usize, 1usize), (2, 8), (3, 3), (1, 30)] {
                assert_bits(
                    &mavp(&close, &periods, minp, maxp, 8),
                    &mavp(&close, &periods, minp, maxp, 7),
                    &format!("n={n} min={minp} max={maxp}"),
                );
            }
        }
    }

    #[test]
    fn matype_boundary_is_eight() {
        assert_eq!(MAX_MATYPE, 8);
        let close = oracle_close(120);
        let periods = vec![5.0; close.len()];
        // 8 is in range: a real series comes back.
        let accepted = mavp(&close, &periods, 2, 10, 8);
        assert!(
            accepted.iter().any(|v| v.is_finite()),
            "matype=8 must be accepted: {accepted:?}"
        );
        // 9 and up are not.
        for matype in [9u8, 10, 255] {
            let got = mavp(&close, &periods, 2, 10, matype);
            assert_eq!(got.len(), close.len());
            assert!(got.iter().all(|v| v.is_nan()), "matype={matype}: {got:?}");
        }
    }

    #[test]
    fn out_of_range_matype_falls_back_to_sma_in_the_dispatcher() {
        // The unvalidated catch-all arm: unreachable through `ma`'s validating
        // callers, but its documented behaviour must not have moved either.
        let close = oracle_close(60);
        for matype in [9u8, 10, 99, u8::MAX] {
            assert_bits(
                &compute_ma_by_type(&close, 5, matype),
                &sma(&close, 5),
                &format!("matype={matype}"),
            );
        }
    }

    #[test]
    fn mavp_constant_period_matches_sma() {
        let close: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let periods = vec![5.0; close.len()];
        let got = mavp(&close, &periods, 5, 5, 0);
        let want = sma(&close, 5);
        assert_eq!(got.len(), close.len());
        for i in 0..close.len() {
            assert_eq!(
                got[i].is_nan(),
                want[i].is_nan(),
                "NaN placement differs at {i}"
            );
            if !want[i].is_nan() {
                assert!((got[i] - want[i]).abs() < 1e-12, "at {i}");
            }
        }
    }

    #[test]
    fn mavp_length_mismatch_returns_all_nan_of_close_length() {
        let close: Vec<f64> = (1..=20).map(|i| i as f64).collect();

        // `periods` shorter than `close`: previously produced a partial result
        // (values up to `periods.len() - 1`, NaN after); now all NaN.
        let short = vec![5.0; 4];
        let got = mavp(&close, &short, 2, 5, 0);
        assert_eq!(got.len(), close.len());
        assert!(got.iter().all(|v| v.is_nan()), "short periods: {got:?}");

        // `periods` longer than `close` is equally a mismatch.
        let long = vec![5.0; close.len() + 3];
        let got = mavp(&close, &long, 2, 5, 0);
        assert_eq!(got.len(), close.len());
        assert!(got.iter().all(|v| v.is_nan()), "long periods: {got:?}");

        // Empty `periods` against a non-empty `close`.
        let got = mavp(&close, &[], 2, 5, 0);
        assert_eq!(got.len(), close.len());
        assert!(got.iter().all(|v| v.is_nan()));

        // Both empty is *not* a mismatch.
        assert!(mavp(&[], &[], 2, 5, 0).is_empty());
    }

    #[test]
    fn mavp_degenerate_periods_return_all_nan() {
        let close: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let periods = vec![3.0; close.len()];
        for (minp, maxp) in [(0usize, 5usize), (6, 5)] {
            let got = mavp(&close, &periods, minp, maxp, 0);
            assert_eq!(got.len(), close.len());
            assert!(got.iter().all(|v| v.is_nan()), "min={minp} max={maxp}");
        }
    }
}
