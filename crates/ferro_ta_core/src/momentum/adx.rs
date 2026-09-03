//! The ADX family: directional movement (`plus_dm` / `minus_dm`), the
//! directional indicators (`plus_di` / `minus_di`), `dx`, `adx` and `adxr`.
//!
//! Every one of them is derived from the two fused kernels `adx_inner`
//! and `dm_only_inner`.

// ---------------------------------------------------------------------------
// ADX family
// ---------------------------------------------------------------------------

/// Return type for ADX inner (pdm_s, mdm_s, plus_di, minus_di, dx, adx).
type AdxInnerOutput = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

/// Fused inner function for ADX-family indicators.
/// Returns a tuple of (pdm_s, mdm_s, plus_di, minus_di, dx, adx).
fn adx_inner(high: &[f64], low: &[f64], close: &[f64], period: usize) -> AdxInnerOutput {
    let n = high.len();
    // Six full-length outputs, allocated as `vec![f64::NAN; n]` and written by
    // index. The `NaN` prologue is a real store pass over 4.8 MB at
    // n = 100_000 (`NaN` cannot be a lazily-mapped zero page) that the kernel
    // then overwrites almost entirely, and an earlier revision tried to
    // reclaim it with `Vec::with_capacity` + `resize(warmup, NAN)` + `push`.
    // That was **measured as a large regression**: with six output buffers the
    // per-bar capacity check and vec-header reload multiply by six, and the
    // whole ADX family (ADX, ADXR, DX, PLUS_DI, MINUS_DI) went from a
    // comfortable win to a tie or a loss (+37% to +74% wall time). See the
    // note at the top of `momentum/roc.rs`: reclaim a prologue with indexed
    // stores if at all, never with `push`.
    let mut b_pdm = vec![f64::NAN; n];
    let mut b_mdm = vec![f64::NAN; n];
    let mut b_pdi = vec![f64::NAN; n];
    let mut b_mdi = vec![f64::NAN; n];
    let mut b_dx = vec![f64::NAN; n];
    let mut b_adx = vec![f64::NAN; n];

    if n < period || period < 1 || n < 2 {
        return (b_pdm, b_mdm, b_pdi, b_mdi, b_dx, b_adx);
    }

    let m = n - 1;
    let mut tr = vec![0.0_f64; m];
    let mut pdm = vec![0.0_f64; m];
    let mut mdm = vec![0.0_f64; m];

    for i in 0..m {
        let j = i + 1;
        let h_diff = high[j] - high[i];
        let l_diff = low[i] - low[j];
        let hl = high[j] - low[j];
        let hpc = (high[j] - close[i]).abs();
        let lpc = (low[j] - close[i]).abs();
        tr[i] = hl.max(hpc).max(lpc);
        pdm[i] = if h_diff > l_diff && h_diff > 0.0 {
            h_diff
        } else {
            0.0
        };
        mdm[i] = if l_diff > h_diff && l_diff > 0.0 {
            l_diff
        } else {
            0.0
        };
    }

    if m < period {
        return (b_pdm, b_mdm, b_pdi, b_mdi, b_dx, b_adx);
    }

    let mut tr_s = tr[..period].iter().sum::<f64>();
    let mut pdm_s = pdm[..period].iter().sum::<f64>();
    let mut mdm_s = mdm[..period].iter().sum::<f64>();

    // Store one bar of the DI/DX triple at `idx`. `tr_s == 0.0` writes
    // nothing, leaving all three slots at their initialized `NaN`.
    let store_di_dx = |idx: usize,
                       pdm_s: f64,
                       mdm_s: f64,
                       tr_s: f64,
                       b_pdi: &mut [f64],
                       b_mdi: &mut [f64],
                       b_dx: &mut [f64]| {
        if tr_s == 0.0 {
            return;
        }
        let pdi = 100.0 * pdm_s / tr_s;
        let mdi = 100.0 * mdm_s / tr_s;
        let s = pdi + mdi;
        b_pdi[idx] = pdi;
        b_mdi[idx] = mdi;
        b_dx[idx] = if s != 0.0 {
            100.0 * (pdi - mdi).abs() / s
        } else {
            0.0
        };
    };

    // Initial seeded values at index `period`
    b_pdm[period] = pdm_s;
    b_mdm[period] = mdm_s;
    store_di_dx(
        period, pdm_s, mdm_s, tr_s, &mut b_pdi, &mut b_mdi, &mut b_dx,
    );

    let decay = (period - 1) as f64 / period as f64;
    for i in period..m {
        tr_s = tr_s * decay + tr[i];
        pdm_s = pdm_s * decay + pdm[i];
        mdm_s = mdm_s * decay + mdm[i];

        b_pdm[i + 1] = pdm_s;
        b_mdm[i + 1] = mdm_s;
        store_di_dx(i + 1, pdm_s, mdm_s, tr_s, &mut b_pdi, &mut b_mdi, &mut b_dx);
    }

    // Wilder smooth DX to get ADX
    let adx_start = period + period - 1;
    if n > adx_start {
        let mut dx_sum = 0.0;
        let mut valid_dx = true;
        for v in b_dx.iter().skip(period).take(period) {
            if v.is_nan() {
                valid_dx = false;
                break;
            }
            dx_sum += v;
        }
        if valid_dx {
            let mut adx_s = dx_sum / period as f64;
            b_adx[adx_start] = adx_s;
            let alpha = 1.0 / period as f64;
            for i in adx_start + 1..n {
                adx_s += alpha * (b_dx[i] - adx_s);
                b_adx[i] = adx_s;
            }
        }
    }

    (b_pdm, b_mdm, b_pdi, b_mdi, b_dx, b_adx)
}

/// Compute all six ADX-family outputs in a single pass.
///
/// Returns `(plus_dm, minus_dm, plus_di, minus_di, dx, adx)`.
/// Use this when you need multiple ADX-family outputs to avoid redundant
/// computation. All values are in `[0, 100]` except DM which is unbounded.
/// Warmup: DI/DX valid from index `timeperiod`; ADX from `2 * timeperiod - 1`.
///
/// # Arguments
/// * `high` / `low` / `close` - OHLC price series (same length).
/// * `timeperiod` - Wilder smoothing period (typically 14).
pub fn adx_all(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> AdxInnerOutput {
    adx_inner(high, low, close, timeperiod)
}

/// Internal helper for plus_dm and minus_dm that doesn't allocate dummy close prices.
/// Returns (plus_dm, minus_dm) smoothed with Wilder's method.
fn dm_only_inner(high: &[f64], low: &[f64], period: usize) -> (Vec<f64>, Vec<f64>) {
    let n = high.len();
    let mut b_pdm = vec![f64::NAN; n];
    let mut b_mdm = vec![f64::NAN; n];

    if n < period || period < 1 || n < 2 {
        return (b_pdm, b_mdm);
    }

    let m = n - 1;
    let mut pdm = vec![0.0_f64; m];
    let mut mdm = vec![0.0_f64; m];

    for i in 0..m {
        let j = i + 1;
        let h_diff = high[j] - high[i];
        let l_diff = low[i] - low[j];
        pdm[i] = if h_diff > l_diff && h_diff > 0.0 {
            h_diff
        } else {
            0.0
        };
        mdm[i] = if l_diff > h_diff && l_diff > 0.0 {
            l_diff
        } else {
            0.0
        };
    }

    // TA-Lib special-cases period == 1: the raw (unsmoothed) DM is returned,
    // with the first output at index 1.
    if period == 1 {
        b_pdm[1..].copy_from_slice(&pdm);
        b_mdm[1..].copy_from_slice(&mdm);
        return (b_pdm, b_mdm);
    }

    // TA-Lib's PLUS_DM/MINUS_DM lookback is `period - 1`: the seed is the sum
    // of the first `period - 1` DMs, emitted at index `period - 1`.
    let seed_len = period - 1;
    if m < seed_len {
        return (b_pdm, b_mdm);
    }

    let mut pdm_s = pdm[..seed_len].iter().sum::<f64>();
    let mut mdm_s = mdm[..seed_len].iter().sum::<f64>();

    b_pdm[seed_len] = pdm_s;
    b_mdm[seed_len] = mdm_s;

    let decay = (period - 1) as f64 / period as f64;
    for i in seed_len..m {
        pdm_s = pdm_s * decay + pdm[i];
        mdm_s = mdm_s * decay + mdm[i];
        b_pdm[i + 1] = pdm_s;
        b_mdm[i + 1] = mdm_s;
    }

    (b_pdm, b_mdm)
}

/// Compute the Plus Directional Movement (+DM), Wilder smoothed.
///
/// Measures upward price movement. Returns a `Vec<f64>` of length `n`;
/// the first `timeperiod` values are `NaN`.
///
/// # Arguments
/// * `high` / `low` - High and low price series (same length).
/// * `timeperiod` - Wilder smoothing period.
pub fn plus_dm(high: &[f64], low: &[f64], timeperiod: usize) -> Vec<f64> {
    let (pdm, _) = dm_only_inner(high, low, timeperiod);
    pdm
}

/// Compute the Minus Directional Movement (-DM), Wilder smoothed.
///
/// Measures downward price movement. Returns a `Vec<f64>` of length `n`;
/// the first `timeperiod` values are `NaN`.
///
/// # Arguments
/// * `high` / `low` - High and low price series (same length).
/// * `timeperiod` - Wilder smoothing period.
pub fn minus_dm(high: &[f64], low: &[f64], timeperiod: usize) -> Vec<f64> {
    let (_, mdm) = dm_only_inner(high, low, timeperiod);
    mdm
}

/// Compute the Plus Directional Indicator (+DI), Wilder smoothed.
///
/// `+DI = 100 * smoothed(+DM) / smoothed(TR)`. Returns values in `[0, 100]`.
/// The first `timeperiod` values are `NaN`.
///
/// # Arguments
/// * `high` / `low` / `close` - OHLC price series (same length).
/// * `timeperiod` - Wilder smoothing period.
pub fn plus_di(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let (_, _, pdi, _, _, _) = adx_inner(high, low, close, timeperiod);
    pdi
}

/// Compute the Minus Directional Indicator (-DI), Wilder smoothed.
///
/// `-DI = 100 * smoothed(-DM) / smoothed(TR)`. Returns values in `[0, 100]`.
/// The first `timeperiod` values are `NaN`.
///
/// # Arguments
/// * `high` / `low` / `close` - OHLC price series (same length).
/// * `timeperiod` - Wilder smoothing period.
pub fn minus_di(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let (_, _, _, mdi, _, _) = adx_inner(high, low, close, timeperiod);
    mdi
}

/// Compute the Directional Movement Index (DX).
///
/// `DX = 100 * |+DI - -DI| / (+DI + -DI)`. Returns values in `[0, 100]`.
/// The first `timeperiod` values are `NaN`.
///
/// # Arguments
/// * `high` / `low` / `close` - OHLC price series (same length).
/// * `timeperiod` - Wilder smoothing period.
pub fn dx(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let (_, _, _, _, dx_vals, _) = adx_inner(high, low, close, timeperiod);
    dx_vals
}

/// Compute the Average Directional Movement Index (ADX).
///
/// ADX is Wilder's smoothing of DX, measuring trend strength regardless of
/// direction. Returns values in `[0, 100]`. The first `2 * timeperiod - 1`
/// values are `NaN` (DX warmup + ADX smoothing warmup).
///
/// # Arguments
/// * `high` / `low` / `close` - OHLC price series (same length).
/// * `timeperiod` - Wilder smoothing period (typically 14).
pub fn adx(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let (_, _, _, _, _, adx_vals) = adx_inner(high, low, close, timeperiod);
    adx_vals
}

/// Compute the ADX Rating (ADXR).
///
/// `ADXR[i] = (ADX[i] + ADX[i - timeperiod]) / 2`. Smooths ADX further
/// by averaging current ADX with its value `timeperiod` bars ago.
/// Returns values in `[0, 100]`.
///
/// # Arguments
/// * `high` / `low` / `close` - OHLC price series (same length).
/// * `timeperiod` - Wilder smoothing period (typically 14).
pub fn adxr(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> Vec<f64> {
    let n = high.len();
    // Reuse adx_all to compute ADX once, then derive ADXR from it
    let (_, _, _, _, _, adx_vals) = adx_inner(high, low, close, timeperiod);
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 {
        return result;
    }
    // TA-Lib ADXR lag is `timeperiod - 1`, not `timeperiod`.
    let lag = timeperiod - 1;
    for i in lag..n {
        if !adx_vals[i].is_nan() && !adx_vals[i - lag].is_nan() {
            result[i] = (adx_vals[i] + adx_vals[i - lag]) / 2.0;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::momentum::test_support::assert_bits;

    #[test]
    fn adx_nonnegative() {
        let h: Vec<f64> = (1..=50).map(|i| i as f64 + 1.0).collect();
        let l: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let c: Vec<f64> = (1..=50).map(|i| i as f64 + 0.5).collect();
        let result = adx(&h, &l, &c, 14);
        for v in result.iter().filter(|v| !v.is_nan()) {
            assert!(*v >= 0.0);
        }
    }

    // -- Group E on the ADX family ------------------------------------------

    /// The pre-rewrite `adx_inner`, verbatim.
    #[allow(clippy::assign_op_pattern)]
    fn reference_adx_inner(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        period: usize,
    ) -> AdxInnerOutput {
        let n = high.len();
        let mut b_pdm = vec![f64::NAN; n];
        let mut b_mdm = vec![f64::NAN; n];
        let mut b_pdi = vec![f64::NAN; n];
        let mut b_mdi = vec![f64::NAN; n];
        let mut b_dx = vec![f64::NAN; n];
        let mut b_adx = vec![f64::NAN; n];

        if n < period || period < 1 || n < 2 {
            return (b_pdm, b_mdm, b_pdi, b_mdi, b_dx, b_adx);
        }

        let m = n - 1;
        let mut tr = vec![0.0_f64; m];
        let mut pdm = vec![0.0_f64; m];
        let mut mdm = vec![0.0_f64; m];

        for i in 0..m {
            let j = i + 1;
            let h_diff = high[j] - high[i];
            let l_diff = low[i] - low[j];
            let hl = high[j] - low[j];
            let hpc = (high[j] - close[i]).abs();
            let lpc = (low[j] - close[i]).abs();
            tr[i] = hl.max(hpc).max(lpc);
            pdm[i] = if h_diff > l_diff && h_diff > 0.0 {
                h_diff
            } else {
                0.0
            };
            mdm[i] = if l_diff > h_diff && l_diff > 0.0 {
                l_diff
            } else {
                0.0
            };
        }

        if m < period {
            return (b_pdm, b_mdm, b_pdi, b_mdi, b_dx, b_adx);
        }

        let mut tr_s = tr[..period].iter().sum::<f64>();
        let mut pdm_s = pdm[..period].iter().sum::<f64>();
        let mut mdm_s = mdm[..period].iter().sum::<f64>();

        b_pdm[period] = pdm_s;
        b_mdm[period] = mdm_s;
        if tr_s != 0.0 {
            b_pdi[period] = 100.0 * pdm_s / tr_s;
            b_mdi[period] = 100.0 * mdm_s / tr_s;
            let s = b_pdi[period] + b_mdi[period];
            b_dx[period] = if s != 0.0 {
                100.0 * (b_pdi[period] - b_mdi[period]).abs() / s
            } else {
                0.0
            };
        }

        let decay = (period - 1) as f64 / period as f64;
        for i in period..m {
            tr_s = tr_s * decay + tr[i];
            pdm_s = pdm_s * decay + pdm[i];
            mdm_s = mdm_s * decay + mdm[i];

            b_pdm[i + 1] = pdm_s;
            b_mdm[i + 1] = mdm_s;
            if tr_s != 0.0 {
                b_pdi[i + 1] = 100.0 * pdm_s / tr_s;
                b_mdi[i + 1] = 100.0 * mdm_s / tr_s;
                let s = b_pdi[i + 1] + b_mdi[i + 1];
                b_dx[i + 1] = if s != 0.0 {
                    100.0 * (b_pdi[i + 1] - b_mdi[i + 1]).abs() / s
                } else {
                    0.0
                };
            }
        }

        let adx_start = period + period - 1;
        if n > adx_start {
            let mut dx_sum = 0.0;
            let mut valid_dx = true;
            for v in b_dx.iter().skip(period).take(period) {
                if v.is_nan() {
                    valid_dx = false;
                    break;
                }
                dx_sum += v;
            }
            if valid_dx {
                let mut adx_s = dx_sum / period as f64;
                b_adx[adx_start] = adx_s;
                let alpha = 1.0 / period as f64;
                for i in adx_start + 1..n {
                    adx_s = adx_s + alpha * (b_dx[i] - adx_s);
                    b_adx[i] = adx_s;
                }
            }
        }

        (b_pdm, b_mdm, b_pdi, b_mdi, b_dx, b_adx)
    }

    /// Dropping the six `vec![f64::NAN; n]` prologues from `adx_inner` is a
    /// pure allocation change: every store in that kernel was already
    /// sequential. Includes a flat series (which drives `tr_s == 0.0`, the
    /// branch that leaves DI/DX `NaN`) and a mid-series `NaN`.
    #[test]
    fn adx_family_is_bit_identical_after_dropping_the_nan_prologue() {
        let flat = vec![5.0_f64; 60];
        let mut state = 1_357_u64;
        let mut walk = Vec::with_capacity(80);
        let mut price = 20.0_f64;
        for _ in 0..80 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            price += (((state >> 33) % 9) as f64 - 4.0) * 0.25;
            walk.push(price);
        }
        let mut with_nan = walk.clone();
        with_nan[45] = f64::NAN;

        let cases: [(&str, Vec<f64>); 3] = [("flat", flat), ("walk", walk), ("mid_nan", with_nan)];
        for (name, close) in cases {
            let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
            let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
            for period in [
                0,
                1,
                2,
                3,
                14,
                close.len() - 1,
                close.len(),
                close.len() + 1,
            ] {
                let want = reference_adx_inner(&high, &low, &close, period);
                let got = adx_all(&high, &low, &close, period);
                let ctx = |field: &str| format!("adx {name} p={period} {field}");
                assert_bits(&got.0, &want.0, &ctx("plus_dm"));
                assert_bits(&got.1, &want.1, &ctx("minus_dm"));
                assert_bits(&got.2, &want.2, &ctx("plus_di"));
                assert_bits(&got.3, &want.3, &ctx("minus_di"));
                assert_bits(&got.4, &want.4, &ctx("dx"));
                assert_bits(&got.5, &want.5, &ctx("adx"));
                // And the public wrappers stay consistent with the tuple.
                assert_bits(
                    &dx(&high, &low, &close, period),
                    &want.4,
                    &ctx("dx wrapper"),
                );
                assert_bits(
                    &adx(&high, &low, &close, period),
                    &want.5,
                    &ctx("adx wrapper"),
                );
            }
        }
    }
}
