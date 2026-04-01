//! Tick / Trade Aggregation Pipeline — pure Rust, no PyO3.
//!
//! Aggregates raw tick/trade data into OHLCV bars:
//! - **tick bars**   — fixed number of ticks per bar
//! - **volume bars** — fixed volume threshold per bar
//! - **time bars**   — label-based grouping (labels from Python timestamps)

/// OHLCV 5-tuple return type alias.
type Ohlcv5 = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

/// OHLCV 5-tuple plus labels return type alias.
type Ohlcv5AndLabels = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<i64>);

// ---------------------------------------------------------------------------
// aggregate_tick_bars
// ---------------------------------------------------------------------------

/// Aggregate tick/trade data into tick bars (every N ticks become one bar).
///
/// Returns `(open, high, low, close, volume)` where volume = sum of sizes.
///
/// # Panics
/// Panics if `ticks_per_bar == 0`, arrays are empty, or lengths differ.
pub fn aggregate_tick_bars(price: &[f64], size: &[f64], ticks_per_bar: usize) -> Ohlcv5 {
    assert!(ticks_per_bar >= 1, "ticks_per_bar must be >= 1");
    let n = price.len();
    assert!(
        n > 0 && size.len() == n,
        "price and size must be non-empty and equal length"
    );

    let n_bars = n.div_ceil(ticks_per_bar);
    let mut out_open = Vec::with_capacity(n_bars);
    let mut out_high = Vec::with_capacity(n_bars);
    let mut out_low = Vec::with_capacity(n_bars);
    let mut out_close = Vec::with_capacity(n_bars);
    let mut out_vol = Vec::with_capacity(n_bars);

    let mut i = 0;
    while i < n {
        let end = (i + ticks_per_bar).min(n);
        let bar_p = &price[i..end];
        let bar_s = &size[i..end];
        let bar_open = bar_p[0];
        let bar_high = bar_p.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let bar_low = bar_p.iter().cloned().fold(f64::INFINITY, f64::min);
        let bar_close = *bar_p.last().expect("slice cannot be empty");
        let bar_vol: f64 = bar_s.iter().sum();
        out_open.push(bar_open);
        out_high.push(bar_high);
        out_low.push(bar_low);
        out_close.push(bar_close);
        out_vol.push(bar_vol);
        i = end;
    }

    (out_open, out_high, out_low, out_close, out_vol)
}

// ---------------------------------------------------------------------------
// aggregate_volume_bars_ticks
// ---------------------------------------------------------------------------

/// Aggregate tick data into volume bars (fixed volume threshold).
///
/// Accumulates ticks until cumulative size >= `volume_threshold`, then emits
/// a bar. Any remaining partial bar is also emitted.
///
/// Returns `(open, high, low, close, volume)`.
///
/// # Panics
/// Panics if `volume_threshold <= 0`, arrays are empty, or lengths differ.
pub fn aggregate_volume_bars_ticks(price: &[f64], size: &[f64], volume_threshold: f64) -> Ohlcv5 {
    assert!(volume_threshold > 0.0, "volume_threshold must be > 0");
    let n = price.len();
    assert!(
        n > 0 && size.len() == n,
        "price and size must be non-empty and equal length"
    );

    let mut out_open: Vec<f64> = Vec::new();
    let mut out_high: Vec<f64> = Vec::new();
    let mut out_low: Vec<f64> = Vec::new();
    let mut out_close: Vec<f64> = Vec::new();
    let mut out_vol: Vec<f64> = Vec::new();

    let mut bar_open = price[0];
    let mut bar_high = price[0];
    let mut bar_low = price[0];
    let mut bar_close = price[0];
    let mut bar_vol = size[0];

    for i in 1..n {
        bar_high = bar_high.max(price[i]);
        bar_low = bar_low.min(price[i]);
        bar_close = price[i];
        bar_vol += size[i];

        if bar_vol >= volume_threshold {
            out_open.push(bar_open);
            out_high.push(bar_high);
            out_low.push(bar_low);
            out_close.push(bar_close);
            out_vol.push(bar_vol);
            if i + 1 < n {
                bar_open = price[i + 1];
                bar_high = price[i + 1];
                bar_low = price[i + 1];
                bar_close = price[i + 1];
                bar_vol = size[i + 1];
            } else {
                bar_vol = 0.0;
            }
        }
    }
    // Push remaining partial bar
    if bar_vol > 0.0 {
        out_open.push(bar_open);
        out_high.push(bar_high);
        out_low.push(bar_low);
        out_close.push(bar_close);
        out_vol.push(bar_vol);
    }

    (out_open, out_high, out_low, out_close, out_vol)
}

// ---------------------------------------------------------------------------
// aggregate_time_bars
// ---------------------------------------------------------------------------

/// Aggregate tick data into time bars using pre-computed integer bucket labels.
///
/// Each tick is assigned a `label` (e.g. unix_ts // period_secs). Ticks with
/// the same label are accumulated into one bar. Labels must be non-decreasing.
///
/// Returns `(open, high, low, close, volume, unique_labels)`.
///
/// # Panics
/// Panics if arrays are empty or have unequal lengths.
pub fn aggregate_time_bars(price: &[f64], size: &[f64], labels: &[i64]) -> Ohlcv5AndLabels {
    let n = price.len();
    assert!(
        n > 0 && size.len() == n && labels.len() == n,
        "price, size, and labels must be non-empty and equal length"
    );

    let mut out_open: Vec<f64> = Vec::new();
    let mut out_high: Vec<f64> = Vec::new();
    let mut out_low: Vec<f64> = Vec::new();
    let mut out_close: Vec<f64> = Vec::new();
    let mut out_vol: Vec<f64> = Vec::new();
    let mut out_labels: Vec<i64> = Vec::new();

    let mut cur_label = labels[0];
    let mut bar_open = price[0];
    let mut bar_high = price[0];
    let mut bar_low = price[0];
    let mut bar_close = price[0];
    let mut bar_vol = size[0];

    for i in 1..n {
        if labels[i] != cur_label {
            out_open.push(bar_open);
            out_high.push(bar_high);
            out_low.push(bar_low);
            out_close.push(bar_close);
            out_vol.push(bar_vol);
            out_labels.push(cur_label);
            cur_label = labels[i];
            bar_open = price[i];
            bar_high = price[i];
            bar_low = price[i];
            bar_close = price[i];
            bar_vol = size[i];
        } else {
            bar_high = bar_high.max(price[i]);
            bar_low = bar_low.min(price[i]);
            bar_close = price[i];
            bar_vol += size[i];
        }
    }
    out_open.push(bar_open);
    out_high.push(bar_high);
    out_low.push(bar_low);
    out_close.push(bar_close);
    out_vol.push(bar_vol);
    out_labels.push(cur_label);

    (out_open, out_high, out_low, out_close, out_vol, out_labels)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- aggregate_tick_bars -------------------------------------------------

    #[test]
    fn test_tick_bars_exact_division() {
        let price = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        let size = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (o, h, l, c, v) = aggregate_tick_bars(&price, &size, 3);
        assert_eq!(o.len(), 2);
        // Bar 0: ticks 0..3
        assert!((o[0] - 10.0).abs() < 1e-10);
        assert!((h[0] - 12.0).abs() < 1e-10);
        assert!((l[0] - 10.0).abs() < 1e-10);
        assert!((c[0] - 12.0).abs() < 1e-10);
        assert!((v[0] - 6.0).abs() < 1e-10);
        // Bar 1: ticks 3..6
        assert!((o[1] - 13.0).abs() < 1e-10);
        assert!((h[1] - 15.0).abs() < 1e-10);
        assert!((l[1] - 13.0).abs() < 1e-10);
        assert!((c[1] - 15.0).abs() < 1e-10);
        assert!((v[1] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_tick_bars_partial_last_bar() {
        let price = [10.0, 11.0, 12.0, 13.0, 14.0];
        let size = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (o, _h, _l, c, v) = aggregate_tick_bars(&price, &size, 3);
        assert_eq!(o.len(), 2);
        // Partial bar: ticks 3..5
        assert!((o[1] - 13.0).abs() < 1e-10);
        assert!((c[1] - 14.0).abs() < 1e-10);
        assert!((v[1] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_tick_bars_single_tick() {
        let (o, h, l, c, v) = aggregate_tick_bars(&[42.0], &[100.0], 5);
        assert_eq!(o.len(), 1);
        assert!((o[0] - 42.0).abs() < 1e-10);
        assert!((h[0] - 42.0).abs() < 1e-10);
        assert!((l[0] - 42.0).abs() < 1e-10);
        assert!((c[0] - 42.0).abs() < 1e-10);
        assert!((v[0] - 100.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "ticks_per_bar must be >= 1")]
    fn test_tick_bars_zero_ticks() {
        aggregate_tick_bars(&[1.0], &[1.0], 0);
    }

    // -- aggregate_volume_bars_ticks -----------------------------------------

    #[test]
    fn test_volume_bars_ticks_basic() {
        let price = [10.0, 11.0, 12.0, 13.0, 14.0];
        let size = [30.0, 40.0, 50.0, 20.0, 60.0];
        // threshold=70: bar0 = ticks 0+1 (vol=70), bar1 = tick2 (vol=50) + tick3 (vol=70),
        // then tick4 as partial
        let (o, h, l, c, v) = aggregate_volume_bars_ticks(&price, &size, 70.0);
        // First bar: 30+40=70 >= 70
        assert!((o[0] - 10.0).abs() < 1e-10);
        assert!((c[0] - 11.0).abs() < 1e-10);
        assert!((v[0] - 70.0).abs() < 1e-10);
        assert!((h[0] - 11.0).abs() < 1e-10);
        assert!((l[0] - 10.0).abs() < 1e-10);
        assert!(v.len() >= 2);
    }

    #[test]
    fn test_volume_bars_ticks_single() {
        let (o, _h, _l, _c, v) = aggregate_volume_bars_ticks(&[5.0], &[10.0], 100.0);
        assert_eq!(o.len(), 1);
        assert!((v[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "volume_threshold must be > 0")]
    fn test_volume_bars_ticks_zero_threshold() {
        aggregate_volume_bars_ticks(&[1.0], &[1.0], 0.0);
    }

    // -- aggregate_time_bars -------------------------------------------------

    #[test]
    fn test_time_bars_basic() {
        let price = [10.0, 11.0, 12.0, 13.0, 14.0];
        let size = [1.0, 2.0, 3.0, 4.0, 5.0];
        let labels: [i64; 5] = [0, 0, 1, 1, 1];
        let (o, h, l, c, v, out_lbl) = aggregate_time_bars(&price, &size, &labels);
        assert_eq!(o.len(), 2);
        assert_eq!(out_lbl, vec![0, 1]);
        // Group 0: ticks 0,1
        assert!((o[0] - 10.0).abs() < 1e-10);
        assert!((h[0] - 11.0).abs() < 1e-10);
        assert!((l[0] - 10.0).abs() < 1e-10);
        assert!((c[0] - 11.0).abs() < 1e-10);
        assert!((v[0] - 3.0).abs() < 1e-10);
        // Group 1: ticks 2,3,4
        assert!((o[1] - 12.0).abs() < 1e-10);
        assert!((h[1] - 14.0).abs() < 1e-10);
        assert!((l[1] - 12.0).abs() < 1e-10);
        assert!((c[1] - 14.0).abs() < 1e-10);
        assert!((v[1] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_time_bars_all_same_label() {
        let price = [5.0, 6.0, 4.0];
        let size = [10.0, 20.0, 30.0];
        let labels: [i64; 3] = [42, 42, 42];
        let (o, h, l, c, v, out_lbl) = aggregate_time_bars(&price, &size, &labels);
        assert_eq!(o.len(), 1);
        assert_eq!(out_lbl, vec![42]);
        assert!((o[0] - 5.0).abs() < 1e-10);
        assert!((h[0] - 6.0).abs() < 1e-10);
        assert!((l[0] - 4.0).abs() < 1e-10);
        assert!((c[0] - 4.0).abs() < 1e-10);
        assert!((v[0] - 60.0).abs() < 1e-10);
    }

    #[test]
    fn test_time_bars_each_tick_own_label() {
        let price = [10.0, 20.0, 30.0];
        let size = [1.0, 2.0, 3.0];
        let labels: [i64; 3] = [0, 1, 2];
        let (o, _h, _l, _c, v, out_lbl) = aggregate_time_bars(&price, &size, &labels);
        assert_eq!(o.len(), 3);
        assert_eq!(out_lbl, vec![0, 1, 2]);
        assert!((v[0] - 1.0).abs() < 1e-10);
        assert!((v[1] - 2.0).abs() < 1e-10);
        assert!((v[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "price, size, and labels must be non-empty and equal length")]
    fn test_time_bars_empty() {
        aggregate_time_bars(&[], &[], &[]);
    }
}
