//! Futures curve and term-structure analytics.

use super::basis;

/// Curve summary metrics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurveSummary {
    pub front_basis: f64,
    pub average_basis: f64,
    pub slope: f64,
    pub is_contango: bool,
}

fn regression_slope(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.len() < 2 {
        return f64::NAN;
    }
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        cov += (x - mean_x) * (y - mean_y);
        var += (x - mean_x) * (x - mean_x);
    }
    if var == 0.0 {
        f64::NAN
    } else {
        cov / var
    }
}

/// Calendar spreads between adjacent contracts.
pub fn calendar_spreads(futures_prices: &[f64]) -> Vec<f64> {
    futures_prices.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Curve slope across tenor buckets.
pub fn curve_slope(tenors: &[f64], futures_prices: &[f64]) -> f64 {
    regression_slope(tenors, futures_prices)
}

/// Summary statistics for a forward curve.
pub fn curve_summary(spot: f64, tenors: &[f64], futures_prices: &[f64]) -> CurveSummary {
    if futures_prices.is_empty() || tenors.len() != futures_prices.len() {
        return CurveSummary {
            front_basis: f64::NAN,
            average_basis: f64::NAN,
            slope: f64::NAN,
            is_contango: false,
        };
    }
    let bases: Vec<f64> = futures_prices
        .iter()
        .map(|&price| basis::basis(spot, price))
        .collect();
    let average_basis = bases.iter().sum::<f64>() / bases.len() as f64;
    let is_contango = futures_prices.windows(2).all(|w| w[1] >= w[0]);
    CurveSummary {
        front_basis: basis::basis(spot, futures_prices[0]),
        average_basis,
        slope: curve_slope(tenors, futures_prices),
        is_contango,
    }
}

#[cfg(test)]
mod tests {
    use super::{calendar_spreads, curve_slope, curve_summary};

    #[test]
    fn calendar_spreads_are_correct() {
        assert_eq!(calendar_spreads(&[100.0, 101.0, 103.0]), vec![1.0, 2.0]);
    }

    #[test]
    fn curve_summary_detects_contango() {
        let summary = curve_summary(100.0, &[0.1, 0.5, 1.0], &[101.0, 102.0, 104.0]);
        assert!(summary.is_contango);
        assert!(curve_slope(&[0.1, 0.5, 1.0], &[101.0, 102.0, 104.0]) > 0.0);
    }
}
