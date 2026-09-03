//! Pivot points (classic, Fibonacci and Camarilla methods).

// ---------------------------------------------------------------------------
// PIVOT_POINTS
// ---------------------------------------------------------------------------

/// Pivot Points — support / resistance levels computed from the previous bar.
///
/// # Arguments
/// * `method` — `"classic"`, `"fibonacci"`, or `"camarilla"`. Returns all-NaN
///   vectors for unknown methods.
///
/// # Returns
/// `(pivot, r1, s1, r2, s2)` arrays. Index 0 is always `NaN` (no previous
/// bar). Mismatched input lengths yield all `NaN`.
#[allow(clippy::type_complexity)]
pub fn pivot_points(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    method: &str,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = high.len();

    let method_lower = method.to_lowercase();
    if !matches!(method_lower.as_str(), "classic" | "fibonacci" | "camarilla")
        || low.len() != n
        || close.len() != n
    {
        // Unknown method or mismatched inputs — return all NaN
        let nan = vec![f64::NAN; n];
        return (nan.clone(), nan.clone(), nan.clone(), nan.clone(), nan);
    }

    // Bar 0 has no previous bar, so it keeps the initialized `NaN`.
    let mut pivot = vec![f64::NAN; n];
    let mut r1 = vec![f64::NAN; n];
    let mut s1 = vec![f64::NAN; n];
    let mut r2 = vec![f64::NAN; n];
    let mut s2 = vec![f64::NAN; n];

    for i in 1..n {
        let ph = high[i - 1];
        let pl = low[i - 1];
        let pc = close[i - 1];
        let hl = ph - pl;
        let p = (ph + pl + pc) / 3.0;
        let (r1v, s1v, r2v, s2v) = match method_lower.as_str() {
            "classic" => (2.0 * p - pl, 2.0 * p - ph, p + hl, p - hl),
            "fibonacci" => (
                p + 0.382 * hl,
                p - 0.382 * hl,
                p + 0.618 * hl,
                p - 0.618 * hl,
            ),
            "camarilla" => (
                pc + 1.1 * hl / 12.0,
                pc - 1.1 * hl / 12.0,
                pc + 1.1 * hl / 6.0,
                pc - 1.1 * hl / 6.0,
            ),
            _ => unreachable!(),
        };
        pivot[i] = p;
        r1[i] = r1v;
        s1[i] = s1v;
        r2[i] = r2v;
        s2[i] = s2v;
    }

    (pivot, r1, s1, r2, s2)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // PIVOT_POINTS tests
    // -----------------------------------------------------------------------

    #[test]
    fn pivot_points_classic() {
        let h = vec![10.0, 12.0, 11.0];
        let l = vec![8.0, 9.0, 8.5];
        let c = vec![9.0, 11.0, 10.0];
        let (pivot, r1, s1, r2, s2) = pivot_points(&h, &l, &c, "classic");
        assert_eq!(pivot.len(), 3);
        // Index 0 is NaN
        assert!(pivot[0].is_nan());
        // Index 1: prev bar H=10, L=8, C=9 => P=(10+8+9)/3=9.0
        assert!((pivot[1] - 9.0).abs() < 1e-10);
        // R1 = 2*P - L = 18 - 8 = 10
        assert!((r1[1] - 10.0).abs() < 1e-10);
        // S1 = 2*P - H = 18 - 10 = 8
        assert!((s1[1] - 8.0).abs() < 1e-10);
        // R2 = P + (H-L) = 9 + 2 = 11
        assert!((r2[1] - 11.0).abs() < 1e-10);
        // S2 = P - (H-L) = 9 - 2 = 7
        assert!((s2[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn pivot_points_fibonacci() {
        let h = vec![10.0, 12.0];
        let l = vec![8.0, 9.0];
        let c = vec![9.0, 11.0];
        let (pivot, r1, s1, _, _) = pivot_points(&h, &l, &c, "fibonacci");
        // Index 1: P = (10+8+9)/3 = 9.0, HL = 2
        assert!((pivot[1] - 9.0).abs() < 1e-10);
        assert!((r1[1] - (9.0 + 0.382 * 2.0)).abs() < 1e-10);
        assert!((s1[1] - (9.0 - 0.382 * 2.0)).abs() < 1e-10);
    }

    #[test]
    fn pivot_points_camarilla() {
        let h = vec![10.0, 12.0];
        let l = vec![8.0, 9.0];
        let c = vec![9.0, 11.0];
        let (pivot, r1, s1, _, _) = pivot_points(&h, &l, &c, "camarilla");
        assert!((pivot[1] - 9.0).abs() < 1e-10);
        // R1 = C + 1.1 * HL / 12 = 9 + 1.1*2/12
        assert!((r1[1] - (9.0 + 1.1 * 2.0 / 12.0)).abs() < 1e-10);
        assert!((s1[1] - (9.0 - 1.1 * 2.0 / 12.0)).abs() < 1e-10);
    }

    #[test]
    fn pivot_points_unknown_method() {
        let h = vec![10.0, 12.0];
        let l = vec![8.0, 9.0];
        let c = vec![9.0, 11.0];
        let (pivot, r1, s1, r2, s2) = pivot_points(&h, &l, &c, "unknown");
        assert!(pivot.iter().all(|v| v.is_nan()));
        assert!(r1.iter().all(|v| v.is_nan()));
        assert!(s1.iter().all(|v| v.is_nan()));
        assert!(r2.iter().all(|v| v.is_nan()));
        assert!(s2.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn pivot_points_empty_input() {
        let (p, r1, s1, r2, s2) = pivot_points(&[], &[], &[], "classic");
        assert!(p.is_empty());
        assert!(r1.is_empty());
        assert!(s1.is_empty());
        assert!(r2.is_empty());
        assert!(s2.is_empty());
    }
}
