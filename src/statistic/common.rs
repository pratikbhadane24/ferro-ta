/// Rolling linear regression: returns (slope, intercept) for the given window.
pub(super) fn linreg(window: &[f64]) -> (f64, f64) {
    let n = window.len() as f64;
    let sum_x: f64 = (0..window.len()).map(|i| i as f64).sum();
    let sum_y: f64 = window.iter().sum();
    let sum_xy: f64 = window.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
    let sum_x2: f64 = (0..window.len()).map(|i| (i as f64).powi(2)).sum();
    let denom = n * sum_x2 - sum_x * sum_x;
    let slope = if denom != 0.0 {
        (n * sum_xy - sum_x * sum_y) / denom
    } else {
        0.0
    };
    let intercept = (sum_y - slope * sum_x) / n;
    (slope, intercept)
}
