//! Signal processing helpers.
//!
//! - `rank_values`      — fractional rank of a slice (1-based, ties averaged)
//! - `compose_rank`     — rank-based composite scores for a 2-D signal matrix
//! - `top_n_indices`    — indices of the N largest values
//! - `bottom_n_indices` — indices of the N smallest values

/// Compute fractional rank of each element (1-based, ascending).
/// Ties receive the average of their rank positions.
pub fn rank_values(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let val = x[order[i]];
        let mut j = i + 1;
        while j < n && x[order[j]] == val {
            j += 1;
        }
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[order[k]] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Compute rank-based composite scores for a 2-D signal matrix.
///
/// Each column is ranked independently, and the per-row ranks are summed.
/// `signals` is a slice of columns, each column being a `&[f64]` of the same length.
pub fn compose_rank(signals: &[&[f64]]) -> Vec<f64> {
    if signals.is_empty() {
        return vec![];
    }
    let n_bars = signals[0].len();
    let mut scores = vec![0.0_f64; n_bars];
    for &column in signals {
        let ranks = rank_values(column);
        for (bar_idx, rank) in ranks.into_iter().enumerate() {
            scores[bar_idx] += rank;
        }
    }
    scores
}

/// Return the indices of the N largest values in `x` (descending by value).
pub fn top_n_indices(x: &[f64], n: usize) -> Vec<i64> {
    let len = x.len();
    let k = n.min(len);
    let mut order: Vec<usize> = (0..len).collect();
    order.sort_by(|&a, &b| x[b].partial_cmp(&x[a]).unwrap_or(std::cmp::Ordering::Equal));
    order[..k].iter().map(|&i| i as i64).collect()
}

/// Return the indices of the N smallest values in `x` (ascending by value).
pub fn bottom_n_indices(x: &[f64], n: usize) -> Vec<i64> {
    let len = x.len();
    let k = n.min(len);
    let mut order: Vec<usize> = (0..len).collect();
    order.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    order[..k].iter().map(|&i| i as i64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank_values() {
        let x = vec![3.0, 1.0, 2.0];
        let ranks = rank_values(&x);
        assert!((ranks[0] - 3.0).abs() < 1e-10); // 3.0 is largest → rank 3
        assert!((ranks[1] - 1.0).abs() < 1e-10); // 1.0 is smallest → rank 1
        assert!((ranks[2] - 2.0).abs() < 1e-10); // 2.0 is middle → rank 2
    }

    #[test]
    fn test_rank_values_ties() {
        let x = vec![1.0, 2.0, 2.0, 4.0];
        let ranks = rank_values(&x);
        assert!((ranks[0] - 1.0).abs() < 1e-10);
        assert!((ranks[1] - 2.5).abs() < 1e-10); // tied → average
        assert!((ranks[2] - 2.5).abs() < 1e-10);
        assert!((ranks[3] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_compose_rank() {
        let col1 = vec![3.0, 1.0, 2.0];
        let col2 = vec![1.0, 3.0, 2.0];
        let signals: Vec<&[f64]> = vec![&col1, &col2];
        let scores = compose_rank(&signals);
        // Row 0: rank(3.0)=3 + rank(1.0)=1 = 4
        // Row 1: rank(1.0)=1 + rank(3.0)=3 = 4
        // Row 2: rank(2.0)=2 + rank(2.0)=2 = 4
        assert!((scores[0] - 4.0).abs() < 1e-10);
        assert!((scores[1] - 4.0).abs() < 1e-10);
        assert!((scores[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_top_n_indices() {
        let x = vec![10.0, 50.0, 30.0, 20.0, 40.0];
        let result = top_n_indices(&x, 3);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 1); // 50.0
        assert_eq!(result[1], 4); // 40.0
        assert_eq!(result[2], 2); // 30.0
    }

    #[test]
    fn test_bottom_n_indices() {
        let x = vec![10.0, 50.0, 30.0, 20.0, 40.0];
        let result = bottom_n_indices(&x, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 0); // 10.0
        assert_eq!(result[1], 3); // 20.0
    }

    #[test]
    fn test_top_n_exceeds_len() {
        let x = vec![1.0, 2.0];
        let result = top_n_indices(&x, 5);
        assert_eq!(result.len(), 2);
    }
}
