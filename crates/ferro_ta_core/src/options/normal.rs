//! Normal distribution helpers.

const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// Standard normal probability density function.
pub fn pdf(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard normal cumulative distribution function.
///
/// Uses a common Abramowitz-Stegun style approximation that is fast and
/// sufficiently accurate for option pricing work.
pub fn cdf(x: f64) -> f64 {
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.231_641_9 * ax);
    let poly = (((((1.330_274_429 * t - 1.821_255_978) * t) + 1.781_477_937) * t - 0.356_563_782)
        * t
        + 0.319_381_530)
        * t;
    let approx = 1.0 - pdf(ax) * poly;
    if x >= 0.0 {
        approx
    } else {
        1.0 - approx
    }
}

#[cfg(test)]
mod tests {
    use super::{cdf, pdf};

    #[test]
    fn cdf_is_reasonable() {
        assert!((cdf(0.0) - 0.5).abs() < 1e-7);
        assert!((cdf(1.0) - 0.841_344_746).abs() < 5e-5);
        assert!((cdf(-1.0) - 0.158_655_254).abs() < 5e-5);
    }

    #[test]
    fn pdf_is_reasonable() {
        assert!((pdf(0.0) - 0.398_942_280_4).abs() < 1e-10);
    }
}
