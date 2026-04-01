//! Resampling — OHLCV resampling and multi-timeframe helpers, pure Rust.
//!
//! # Functions
//! - `volume_bars` — Aggregate OHLCV bars into bars of fixed volume size.
//! - `ohlcv_agg`   — Aggregate OHLCV bars given contiguous integer group labels.

/// OHLCV 5-tuple return type alias.
type Ohlcv5 = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

// ---------------------------------------------------------------------------
// volume_bars
// ---------------------------------------------------------------------------

/// Aggregate OHLCV data into volume bars of a fixed volume threshold.
///
/// Each output bar accumulates input bars until `volume_threshold` units of
/// volume have been consumed.  The resulting bar has:
///   - open  = first open of the group
///   - high  = max high of the group
///   - low   = min low of the group
///   - close = last close of the group
///   - volume = sum of volumes (approximately `volume_threshold`)
///
/// Returns `(open, high, low, close, volume)`.
///
/// # Panics
/// Panics if arrays are empty, have unequal lengths, or `volume_threshold <= 0`.
pub fn volume_bars(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    volume_threshold: f64,
) -> Ohlcv5 {
    assert!(volume_threshold > 0.0, "volume_threshold must be > 0");
    let n = open.len();
    assert!(n > 0, "input arrays must be non-empty");
    assert!(
        high.len() == n && low.len() == n && close.len() == n && volume.len() == n,
        "all input arrays must have equal length"
    );

    let mut out_open: Vec<f64> = Vec::new();
    let mut out_high: Vec<f64> = Vec::new();
    let mut out_low: Vec<f64> = Vec::new();
    let mut out_close: Vec<f64> = Vec::new();
    let mut out_vol: Vec<f64> = Vec::new();

    let mut bar_open = open[0];
    let mut bar_high = high[0];
    let mut bar_low = low[0];
    let mut bar_close = close[0];
    let mut bar_vol = volume[0];

    for i in 1..n {
        bar_high = bar_high.max(high[i]);
        bar_low = bar_low.min(low[i]);
        bar_close = close[i];
        bar_vol += volume[i];

        if bar_vol >= volume_threshold {
            out_open.push(bar_open);
            out_high.push(bar_high);
            out_low.push(bar_low);
            out_close.push(bar_close);
            out_vol.push(bar_vol);
            // Start new bar
            if i + 1 < n {
                bar_open = open[i + 1];
                bar_high = high[i + 1];
                bar_low = low[i + 1];
                bar_close = close[i + 1];
                bar_vol = volume[i + 1];
            }
        }
    }
    // Push any remaining partial bar
    if bar_vol > 0.0 && out_vol.last().is_none_or(|&last| last != bar_vol) {
        out_open.push(bar_open);
        out_high.push(bar_high);
        out_low.push(bar_low);
        out_close.push(bar_close);
        out_vol.push(bar_vol);
    }

    (out_open, out_high, out_low, out_close, out_vol)
}

// ---------------------------------------------------------------------------
// ohlcv_agg
// ---------------------------------------------------------------------------

/// Aggregate OHLCV bars by integer group labels.
///
/// Groups consecutive bars with the same label and computes:
///   - open  = first open of the group
///   - high  = max high of the group
///   - low   = min low of the group
///   - close = last close of the group
///   - volume = sum of volumes
///
/// `labels` must be non-decreasing (groups are contiguous).
///
/// Returns `(open, high, low, close, volume)`.
///
/// # Panics
/// Panics if arrays are empty or have unequal lengths.
pub fn ohlcv_agg(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    labels: &[i64],
) -> Ohlcv5 {
    let n = open.len();
    assert!(n > 0, "input arrays must be non-empty");
    assert!(
        high.len() == n
            && low.len() == n
            && close.len() == n
            && volume.len() == n
            && labels.len() == n,
        "all input arrays must have equal length"
    );

    let mut out_open: Vec<f64> = Vec::new();
    let mut out_high: Vec<f64> = Vec::new();
    let mut out_low: Vec<f64> = Vec::new();
    let mut out_close: Vec<f64> = Vec::new();
    let mut out_vol: Vec<f64> = Vec::new();

    let mut cur_label = labels[0];
    let mut bar_open = open[0];
    let mut bar_high = high[0];
    let mut bar_low = low[0];
    let mut bar_close = close[0];
    let mut bar_vol = volume[0];

    for i in 1..n {
        if labels[i] != cur_label {
            out_open.push(bar_open);
            out_high.push(bar_high);
            out_low.push(bar_low);
            out_close.push(bar_close);
            out_vol.push(bar_vol);
            cur_label = labels[i];
            bar_open = open[i];
            bar_high = high[i];
            bar_low = low[i];
            bar_close = close[i];
            bar_vol = volume[i];
        } else {
            bar_high = bar_high.max(high[i]);
            bar_low = bar_low.min(low[i]);
            bar_close = close[i];
            bar_vol += volume[i];
        }
    }
    out_open.push(bar_open);
    out_high.push(bar_high);
    out_low.push(bar_low);
    out_close.push(bar_close);
    out_vol.push(bar_vol);

    (out_open, out_high, out_low, out_close, out_vol)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- volume_bars ---------------------------------------------------------

    #[test]
    fn test_volume_bars_basic() {
        let o = [100.0, 101.0, 102.0, 103.0, 104.0];
        let h = [105.0, 106.0, 107.0, 108.0, 109.0];
        let l = [95.0, 96.0, 97.0, 98.0, 99.0];
        let c = [101.0, 102.0, 103.0, 104.0, 105.0];
        let v = [50.0, 60.0, 40.0, 70.0, 30.0];
        // threshold 100: first bar covers indices 0..2 (vol=110>=100)
        let (ro, rh, rl, rc, rv) = volume_bars(&o, &h, &l, &c, &v, 100.0);
        assert!(rv.len() >= 2);
        // First bar: vol = 50+60 = 110
        assert!((rv[0] - 110.0).abs() < 1e-10);
        assert!((ro[0] - 100.0).abs() < 1e-10);
        assert!((rh[0] - 106.0).abs() < 1e-10);
        assert!((rl[0] - 95.0).abs() < 1e-10);
        assert!((rc[0] - 102.0).abs() < 1e-10);
    }

    #[test]
    fn test_volume_bars_single_element() {
        let (ro, rh, rl, rc, rv) = volume_bars(&[10.0], &[12.0], &[8.0], &[11.0], &[50.0], 100.0);
        assert_eq!(rv.len(), 1);
        assert!((rv[0] - 50.0).abs() < 1e-10);
        assert!((ro[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "volume_threshold must be > 0")]
    fn test_volume_bars_zero_threshold() {
        volume_bars(&[1.0], &[1.0], &[1.0], &[1.0], &[1.0], 0.0);
    }

    #[test]
    #[should_panic(expected = "input arrays must be non-empty")]
    fn test_volume_bars_empty() {
        volume_bars(&[], &[], &[], &[], &[], 100.0);
    }

    // -- ohlcv_agg -----------------------------------------------------------

    #[test]
    fn test_ohlcv_agg_basic() {
        let o = [100.0, 101.0, 102.0, 103.0];
        let h = [105.0, 106.0, 108.0, 109.0];
        let l = [95.0, 96.0, 97.0, 98.0];
        let c = [101.0, 102.0, 103.0, 104.0];
        let v = [10.0, 20.0, 30.0, 40.0];
        let labels: [i64; 4] = [0, 0, 1, 1];
        let (ro, rh, rl, rc, rv) = ohlcv_agg(&o, &h, &l, &c, &v, &labels);
        assert_eq!(ro.len(), 2);
        // Group 0: open=100, high=max(105,106)=106, low=min(95,96)=95, close=102, vol=30
        assert!((ro[0] - 100.0).abs() < 1e-10);
        assert!((rh[0] - 106.0).abs() < 1e-10);
        assert!((rl[0] - 95.0).abs() < 1e-10);
        assert!((rc[0] - 102.0).abs() < 1e-10);
        assert!((rv[0] - 30.0).abs() < 1e-10);
        // Group 1: open=102, high=max(108,109)=109, low=min(97,98)=97, close=104, vol=70
        assert!((ro[1] - 102.0).abs() < 1e-10);
        assert!((rh[1] - 109.0).abs() < 1e-10);
        assert!((rl[1] - 97.0).abs() < 1e-10);
        assert!((rc[1] - 104.0).abs() < 1e-10);
        assert!((rv[1] - 70.0).abs() < 1e-10);
    }

    #[test]
    fn test_ohlcv_agg_single_group() {
        let o = [100.0, 101.0];
        let h = [105.0, 106.0];
        let l = [95.0, 96.0];
        let c = [101.0, 102.0];
        let v = [10.0, 20.0];
        let labels: [i64; 2] = [0, 0];
        let (ro, rh, rl, rc, rv) = ohlcv_agg(&o, &h, &l, &c, &v, &labels);
        assert_eq!(ro.len(), 1);
        assert!((rv[0] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_ohlcv_agg_each_bar_own_group() {
        let o = [100.0, 101.0, 102.0];
        let h = [105.0, 106.0, 107.0];
        let l = [95.0, 96.0, 97.0];
        let c = [101.0, 102.0, 103.0];
        let v = [10.0, 20.0, 30.0];
        let labels: [i64; 3] = [0, 1, 2];
        let (ro, _rh, _rl, _rc, rv) = ohlcv_agg(&o, &h, &l, &c, &v, &labels);
        assert_eq!(ro.len(), 3);
        assert!((rv[0] - 10.0).abs() < 1e-10);
        assert!((rv[1] - 20.0).abs() < 1e-10);
        assert!((rv[2] - 30.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "input arrays must be non-empty")]
    fn test_ohlcv_agg_empty() {
        ohlcv_agg(&[], &[], &[], &[], &[], &[]);
    }
}
