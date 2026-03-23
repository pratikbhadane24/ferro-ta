//! Continuous futures roll helpers.

/// Weighted stitching using next-contract weights in [0, 1].
pub fn weighted_continuous(front: &[f64], next: &[f64], next_weights: &[f64]) -> Vec<f64> {
    if front.len() != next.len() || front.len() != next_weights.len() {
        return Vec::new();
    }
    front
        .iter()
        .zip(next.iter())
        .zip(next_weights.iter())
        .map(|((&f, &n), &w)| f * (1.0 - w) + n * w)
        .collect()
}

fn roll_index(weights: &[f64]) -> Option<usize> {
    if weights.is_empty() {
        return None;
    }
    weights
        .iter()
        .enumerate()
        .find(|(_, w)| **w >= 0.5)
        .map(|(idx, _)| idx)
        .or_else(|| weights.iter().position(|w| *w > 0.0))
        .or(Some(weights.len() - 1))
}

/// Back-adjusted continuous series using the roll date implied by the weights.
pub fn back_adjusted_continuous(front: &[f64], next: &[f64], next_weights: &[f64]) -> Vec<f64> {
    if front.len() != next.len() || front.len() != next_weights.len() || front.is_empty() {
        return Vec::new();
    }
    let idx = roll_index(next_weights).unwrap_or(front.len() - 1);
    let gap = next[idx] - front[idx];
    front
        .iter()
        .enumerate()
        .map(|(i, &value)| if i < idx { value + gap } else { next[i] })
        .collect()
}

/// Ratio-adjusted continuous series using the roll date implied by the weights.
pub fn ratio_adjusted_continuous(front: &[f64], next: &[f64], next_weights: &[f64]) -> Vec<f64> {
    if front.len() != next.len() || front.len() != next_weights.len() || front.is_empty() {
        return Vec::new();
    }
    let idx = roll_index(next_weights).unwrap_or(front.len() - 1);
    let ratio = if front[idx] == 0.0 {
        1.0
    } else {
        next[idx] / front[idx]
    };
    front
        .iter()
        .enumerate()
        .map(|(i, &value)| if i < idx { value * ratio } else { next[i] })
        .collect()
}

/// Annualized roll yield from front and next prices.
pub fn roll_yield(front_price: f64, next_price: f64, time_to_expiry: f64) -> f64 {
    if !front_price.is_finite()
        || !next_price.is_finite()
        || !time_to_expiry.is_finite()
        || front_price <= 0.0
        || time_to_expiry <= 0.0
    {
        return f64::NAN;
    }
    (next_price / front_price - 1.0) / time_to_expiry
}

#[cfg(test)]
mod tests {
    use super::{
        back_adjusted_continuous, ratio_adjusted_continuous, roll_yield, weighted_continuous,
    };

    #[test]
    fn weighted_roll_blends_contracts() {
        let out = weighted_continuous(&[100.0, 101.0], &[102.0, 103.0], &[0.0, 1.0]);
        assert_eq!(out, vec![100.0, 103.0]);
    }

    #[test]
    fn adjusted_rolls_return_full_series() {
        let weights = [0.0, 0.25, 0.75, 1.0];
        assert_eq!(
            back_adjusted_continuous(
                &[100.0, 101.0, 102.0, 103.0],
                &[101.0, 102.0, 103.0, 104.0],
                &weights
            )
            .len(),
            4
        );
        assert_eq!(
            ratio_adjusted_continuous(
                &[100.0, 101.0, 102.0, 103.0],
                &[101.0, 102.0, 103.0, 104.0],
                &weights
            )
            .len(),
            4
        );
        assert!(roll_yield(100.0, 102.0, 30.0 / 365.0).is_finite());
    }
}
