//! Crypto and 24/7 market helpers.
//!
//! - `funding_cumulative_pnl` — cumulative PnL from periodic funding rate payments
//! - `continuous_bar_labels`  — assign sequential integer labels based on fixed period size
//! - `mark_session_boundaries` — return indices where a new UTC day begins

/// Compute the cumulative PnL from funding rate payments.
///
/// `position_size` and `funding_rate` must have the same length.
/// PnL at period i = -position_size[i] * funding_rate[i] (longs pay when rate > 0).
pub fn funding_cumulative_pnl(position_size: &[f64], funding_rate: &[f64]) -> Vec<f64> {
    let n = position_size.len();
    let mut out = vec![0.0_f64; n];
    let mut cumulative = 0.0_f64;
    for i in 0..n {
        cumulative += -position_size[i] * funding_rate[i];
        out[i] = cumulative;
    }
    out
}

/// Assign a sequential integer label per bar based on a fixed-size period.
///
/// Bars 0..(period_bars-1) get label 0, bars period_bars..(2*period_bars-1) get label 1, etc.
/// `period_bars` must be >= 1.
pub fn continuous_bar_labels(n_bars: usize, period_bars: usize) -> Vec<i64> {
    (0..n_bars).map(|i| (i / period_bars) as i64).collect()
}

/// Return bar indices where a new UTC day begins (based on nanosecond timestamps).
///
/// Bar 0 is always included as the first boundary.
pub fn mark_session_boundaries(timestamps_ns: &[i64]) -> Vec<i64> {
    let n = timestamps_ns.len();
    if n == 0 {
        return vec![];
    }
    const NS_PER_DAY: i64 = 86_400_000_000_000;
    let mut out = vec![0i64]; // bar 0 is always a boundary
    let mut prev_day = timestamps_ns[0].div_euclid(NS_PER_DAY);
    for (i, &t) in timestamps_ns.iter().enumerate().skip(1) {
        let day = t.div_euclid(NS_PER_DAY);
        if day != prev_day {
            out.push(i as i64);
            prev_day = day;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_funding_cumulative_pnl() {
        let pos = vec![100.0, 100.0, -50.0];
        let rate = vec![0.001, -0.002, 0.001];
        let result = funding_cumulative_pnl(&pos, &rate);
        assert!((result[0] - (-0.1)).abs() < 1e-10);
        assert!((result[1] - 0.1).abs() < 1e-10); // -0.1 + 0.2 = 0.1
        assert!((result[2] - 0.15).abs() < 1e-10); // 0.1 + 0.05 = 0.15
    }

    #[test]
    fn test_continuous_bar_labels() {
        let labels = continuous_bar_labels(7, 3);
        assert_eq!(labels, vec![0, 0, 0, 1, 1, 1, 2]);
    }

    #[test]
    fn test_mark_session_boundaries() {
        let ns_per_day: i64 = 86_400_000_000_000;
        let ts = vec![
            0,                         // day 0
            ns_per_day / 2,            // day 0
            ns_per_day,                // day 1
            ns_per_day + ns_per_day / 2, // day 1
            ns_per_day * 2,            // day 2
        ];
        let result = mark_session_boundaries(&ts);
        assert_eq!(result, vec![0, 2, 4]);
    }

    #[test]
    fn test_empty() {
        assert!(funding_cumulative_pnl(&[], &[]).is_empty());
        assert!(continuous_bar_labels(0, 1).is_empty());
        assert!(mark_session_boundaries(&[]).is_empty());
    }
}
