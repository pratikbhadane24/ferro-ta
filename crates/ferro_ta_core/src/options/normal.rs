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

/// Lower/upper tail breakpoint for the piecewise inverse-CDF approximation.
const INV_CDF_P_LOW: f64 = 0.024_25;
const INV_CDF_A: [f64; 6] = [
    -3.969_683_028_665_376e1,
    2.209_460_984_245_205e2,
    -2.759_285_104_469_687e2,
    1.383_577_518_672_69e2,
    -3.066_479_806_614_716e1,
    2.506_628_277_459_239,
];
const INV_CDF_B: [f64; 5] = [
    -5.447_609_879_822_406e1,
    1.615_858_368_580_409e2,
    -1.556_989_798_598_866e2,
    6.680_131_188_771_972e1,
    -1.328_068_155_288_572e1,
];
const INV_CDF_C: [f64; 6] = [
    -7.784_894_002_430_293e-3,
    -3.223_964_580_411_365e-1,
    -2.400_758_277_161_838,
    -2.549_732_539_343_734,
    4.374_664_141_464_968,
    2.938_163_982_698_783,
];
const INV_CDF_D: [f64; 4] = [
    7.784_695_709_041_462e-3,
    3.224_671_290_700_398e-1,
    2.445_134_137_142_996,
    3.754_408_661_907_416,
];

fn inv_cdf_tail(q: f64) -> f64 {
    let c = INV_CDF_C;
    let d = INV_CDF_D;
    (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
}

/// Standard normal quantile function (inverse CDF).
///
/// Acklam's rational approximation (relative error < 1.2e-9). Returns
/// `-inf` / `+inf` at 0 / 1 and NaN outside `[0, 1]`.
pub fn inv_cdf(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if p < INV_CDF_P_LOW {
        return inv_cdf_tail((-2.0 * p.ln()).sqrt());
    }
    if p > 1.0 - INV_CDF_P_LOW {
        return -inv_cdf_tail((-2.0 * (1.0 - p).ln()).sqrt());
    }
    let a = INV_CDF_A;
    let b = INV_CDF_B;
    let q = p - 0.5;
    let r = q * q;
    (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
}

#[cfg(test)]
mod tests {
    use super::{cdf, inv_cdf, pdf};

    #[test]
    fn inv_cdf_matches_known_quantiles() {
        assert_eq!(inv_cdf(0.5), 0.0);
        assert!((inv_cdf(0.975) - 1.959_963_984_540_054).abs() < 1e-8);
        assert!((inv_cdf(0.01) + 2.326_347_874_040_841).abs() < 1e-8);
        assert!((inv_cdf(1e-6) + 4.753_424_308_822_899).abs() < 1e-7);
        assert!(inv_cdf(0.0).is_infinite() && inv_cdf(1.0).is_infinite());
        assert!(inv_cdf(-0.1).is_nan() && inv_cdf(1.1).is_nan());
    }

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
